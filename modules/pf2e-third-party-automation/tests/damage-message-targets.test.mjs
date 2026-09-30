import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {createCampaignFeats} from '../scripts/campaign-feats.mjs';
import * as api from '../scripts/damage-message-targets.mjs';
const NS='pf2e-third-party-automation',ENEMY='Scene.scene.Token.enemy',RANGED='pf2e-ranged-combat';
const own=d=>d?.flags?.[NS]??{},values=c=>Array.from(c??[]);
test('binding a damage card preserves metadata and leaves the source object unchanged without module APIs',()=>{
 assert.equal(typeof api.withDamageMessageTarget,'function');
 const data={flavor:'original',flags:{pf2e:{context:{target:{token:ENEMY}}},[NS]:{usageGenerated:true},'pf2e-toolbelt':{otherSetting:true,targetHelper:{targets:['Scene.scene.Token.old'],splashTargets:['Scene.scene.Token.splash'],applied:{prior:{0:true}},saveVariants:{null:{saves:{}}}}}}};
 const next=api.withDamageMessageTarget(data,ENEMY);
 assert.deepEqual(next.flags['pf2e-toolbelt'].targetHelper.targets,[ENEMY]);
 assert.deepEqual(data.flags['pf2e-toolbelt'].targetHelper.targets,['Scene.scene.Token.old']);
 assert.deepEqual(next.flags['pf2e-toolbelt'].targetHelper.splashTargets,['Scene.scene.Token.splash']);
 assert.deepEqual(next.flags['pf2e-toolbelt'].targetHelper.applied,{prior:{0:true}});
 assert.deepEqual(next.flags['pf2e-toolbelt'].targetHelper.saveVariants,{null:{saves:{}}});
 assert.equal(next.flags['pf2e-toolbelt'].otherSetting,true);
 assert.equal(next.flags.pf2e.context.target.token,ENEMY);
 assert.equal(next.flavor,'original');
});
test('missing or actor-only recipient cannot create a guessed damage-card target',()=>{
 assert.equal(typeof api.withDamageMessageTarget,'function');
 for(const value of [undefined,null,'','Actor.enemy',{}])assert.throws(()=>api.withDamageMessageTarget({},value),/explicit.*Token/i);
});
function extracted(file,startMarker,endMarker,environment){
 const source=fs.readFileSync(new URL('../scripts/'+file,import.meta.url),'utf8');
 const start=source.indexOf(startMarker),end=source.indexOf(endMarker,start);
 assert(start>=0&&end>start);
 return vm.runInNewContext('('+source.slice(start,end).trim()+')',environment);
}
function rollClass(published){
 return class DamageRoll{constructor(formula){this.formula=formula;}async evaluate(){return this;}async toMessage(data){published.push(data);return data;}};
}
function campaignScope(actor,item,target){const docs=new Map([actor,item,target].map(d=>[d.uuid,d])),messages=new Map([['check',{id:'check'}],['use',{id:'use'}]]),user={id:'gm',active:true};return {game:{user,messages},gm(){if(!user.active)throw Error('主GM已改变')},fromUuid:async uuid=>docs.get(uuid),docs};}
test('campaign custom damage publishes its proven recipient for Toolbelt',async()=>{
 const published=[],DamageRoll=rollClass(published),target={uuid:ENEMY,actor:{uuid:'Actor.enemy'}},actor={id:'pc',uuid:'Actor.pc'},item={uuid:'Actor.pc.Item.ability',type:'action'};
 const postDamage=extracted('campaign-feats.mjs','async function postDamage(','async function maintain(',{
  MODULE_ID:NS,...campaignScope(actor,item,target),manualDamageRoll:async({roll})=>roll.evaluate(),manualDamagePrivacy:()=>null,damageRollClass:()=>DamageRoll,actorTokens:()=>[],withDamageMessageTarget:api.withDamageMessageTarget,
 });
 await postDamage({actor,item,target,formula:'2d6[slashing]',checkId:'check',usageId:'use'});
 assert.deepEqual(Array.from(published[0].flags['pf2e-toolbelt']?.targetHelper?.targets??[]),[ENEMY]);
 assert.equal(published[0].flags.pf2e.context.target.token,ENEMY);
 assert.equal(published[0].flags[NS].campaignAttackMessageId,'check');
});
test('bear support card keeps the hit target rather than its companion speaker',async()=>{
 const published=[],DamageRoll=rollClass(published),supportId='Compendium.pf2e-animal-companions.AC-Support.Item.AvDlo1mgxXd7ZA8W';
 const master={uuid:'Actor.master',flags:{[RANGED]:{animalCompanionId:'bear'}},testUserPermission:()=>true};
 const companion={uuid:'Actor.bear',items:[{sourceId:supportId,uuid:'Actor.bear.Item.support',system:{traits:{}}}]};
 const effect={flags:{[NS]:{masterUuid:master.uuid,processed:[],startedAt:0,companionTokenUuid:'Scene.scene.Token.bear'}},async update(change){this.flags[NS].processed=change[`flags.${NS}.processed`]}};
 const target={uuid:ENEMY,actor:{uuid:'Actor.enemy'}},bear={uuid:'Scene.scene.Token.bear',actor:companion};
 const attack={id:'attack',isCheckRoll:true,timestamp:1,speaker:{actor:'master'},author:{id:'player'},flags:{pf2e:{origin:{type:'weapon',actor:master.uuid},context:{type:'attack-roll',outcome:'success',target:{actor:'Actor.enemy',token:ENEMY},options:[]}}}};
 const handleAttack=extracted('companion-automation.mjs','async function handleAttack(','async function expire(',{
  MODULE_ID:NS,RANGED,SOURCE:{support:supportId},game:{messages:new Map([[attack.id,attack]]),actors:new Map([['master',master],['bear',companion]]),users:new Map([['player',{id:'player'}]])},
  authority:()=>true,ownFlags:own,linkedMaster:()=>master,serial:(_key,callback)=>callback(),supports:()=>[effect],isExpired:()=>false,
  fromUuid:async uuid=>uuid===ENEMY?target:bear,inReach:()=>true,values,sourceOf:i=>i.sourceId,
  CONFIG:{Dice:{rolls:[DamageRoll]}},manualDamageRoll:async({roll})=>roll.evaluate(),manualDamagePrivacy:()=>null,ChatMessage:{getSpeaker:()=>({actor:'bear'})},withDamageMessageTarget:api.withDamageMessageTarget,
 });
 await handleAttack(attack);
 assert.equal(published.length,1);
 assert.deepEqual(Array.from(published[0].flags['pf2e-toolbelt']?.targetHelper?.targets??[]),[ENEMY]);
 assert.equal(published[0].speaker.actor,'bear');
});
test('campaign custom damage waits for the native window and closing it cannot publish damage',async()=>{
 const published=[],DamageRoll=rollClass(published),target={uuid:ENEMY,actor:{uuid:'Actor.enemy'}},actor={id:'pc',uuid:'Actor.pc'},item={uuid:'Actor.pc.Item.ability',type:'action'};let accept;const pending=new Promise(resolve=>accept=resolve);
 const postDamage=extracted('campaign-feats.mjs','async function postDamage(','async function maintain(',{MODULE_ID:NS,...campaignScope(actor,item,target),manualDamageRoll:async()=>{await pending;return null},manualDamagePrivacy:()=>null,damageRollClass:()=>DamageRoll,actorTokens:()=>[],withDamageMessageTarget:api.withDamageMessageTarget});const run=postDamage({actor,item,target,formula:'2d6[slashing]'});await new Promise(resolve=>setImmediate(resolve));assert.equal(published.length,0);accept();await assert.rejects(run,/取消/);assert.equal(published.length,0);
});

for(const changed of ['gm','item','target-actor','source-message'])test(`a campaign manual window cannot publish old damage after its ${changed} changes`,async()=>{
 const published=[],DamageRoll=rollClass(published),target={uuid:ENEMY,actor:{uuid:'Actor.enemy'}},actor={id:'pc',uuid:'Actor.pc'},item={uuid:'Actor.pc.Item.ability',type:'action'},scope=campaignScope(actor,item,target);let accept;const wait=new Promise(resolve=>accept=resolve);
 const postDamage=extracted('campaign-feats.mjs','async function postDamage(','async function maintain(',{MODULE_ID:NS,...scope,manualDamageRoll:async({roll})=>{await wait;return roll.evaluate()},manualDamagePrivacy:()=>null,damageRollClass:()=>DamageRoll,actorTokens:()=>[],withDamageMessageTarget:api.withDamageMessageTarget}),pending=postDamage({actor,item,target,formula:'2d6[slashing]',usageId:'use'});await new Promise(resolve=>setImmediate(resolve));if(changed==='gm')scope.game.user={id:'new'};else if(changed==='item')scope.docs.delete(item.uuid);else if(changed==='target-actor')target.actor={uuid:'Actor.rebound'};else scope.game.messages.delete('use');accept();await assert.rejects(pending,/来源|GM|目标/);assert.equal(published.length,0);
});
function campaignHook({nativeTarget=ENEMY,nativeActor='Actor.enemy',targetTokenUuid=ENEMY,checkId='check',marked=true}={}){
 const state={checkId:'check',itemUuid:'Actor.pc.Item.weapon',actorUuid:'Actor.pc',targetUuid:'Actor.enemy',targetTokenUuid};
 const usage={flags:{[NS]:{campaignStrike:state}}},hooks=new Map(),game={messages:new Map([['use',usage]])},updates=[];
 createCampaignFeats({game}).register({Hooks:{on:(event,callback)=>{hooks.set(event,callback);return event;},off(){}}});
 const message={actor:{uuid:'Actor.pc'},flags:{pf2e:{origin:{uuid:state.itemUuid},context:{type:'damage-roll',target:{actor:nativeActor,token:nativeTarget},options:marked?[NS+':campaign-damage:use:'+checkId]:[]}},'pf2e-toolbelt':{targetHelper:{targets:['Scene.scene.Token.gm-current'],applied:{prior:{0:true}}}}},updateSource(update){
  updates.push(update);
  for(const [path,value]of Object.entries(update)){const keys=path.split('.');let parent=this;for(const key of keys.slice(0,-1))parent=parent[key]??={};parent[keys.at(-1)]=value;}
 }};
 return {message,updates,run:()=>hooks.get('preCreateChatMessage')(message)};
}
test('validated native campaign damage replaces a preexisting wrong Toolbelt target',()=>{
 const f=campaignHook();f.run();
 assert.deepEqual(f.message.flags['pf2e-toolbelt'].targetHelper.targets,[ENEMY]);
 assert.deepEqual(f.message.flags['pf2e-toolbelt'].targetHelper.applied,{prior:{0:true}});
});
for(const options of [{marked:false},{checkId:'stale'},{nativeTarget:'Scene.scene.Token.different'},{nativeActor:'Actor.different'},{targetTokenUuid:null}])test('unrelated or mismatched native campaign damage keeps its own target metadata: '+JSON.stringify(options),()=>{
 const f=campaignHook(options);f.run();
 assert.deepEqual(f.message.flags['pf2e-toolbelt'].targetHelper.targets,['Scene.scene.Token.gm-current']);
 assert.deepEqual(f.updates,[]);
});
