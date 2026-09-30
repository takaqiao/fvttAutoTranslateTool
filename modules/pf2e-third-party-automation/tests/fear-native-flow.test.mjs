import test from 'node:test';
import assert from 'node:assert/strict';
import {createFearAutomation,FEAR_SOURCES} from '../scripts/fear-automation.mjs';
const M='pf2e-third-party-automation';
const update=async function(changes){for(const[path,value]of Object.entries(changes)){let at=this;const keys=path.split('.');for(const k of keys.slice(0,-1))at=at[k]??={};at[keys.at(-1)]=value;}return this;};
function fixture(type='attack-roll'){
 const gm={id:'gm',isGM:true,active:true},player={id:'player',active:true},users=new Map([[gm.id,gm],[player.id,player]]);users.activeGM=gm;
 const actor={id:'pc',uuid:'Actor.pc',type:'character',items:new Map(),flags:{},skills:{intimidation:{rank:4}},testUserPermission:u=>u===gm||u===player,isAllyOf:()=>false,update};
 const feat={id:'feat',actor,type:'feat',sourceId:FEAR_SOURCES.battle};actor.items.set(feat.id,feat);
 const enemy={id:'enemy',uuid:'Actor.enemy',name:'Secret actor',items:new Map(),testUserPermission:u=>u===gm};
 const scene={id:'scene',tokens:new Map()},token=(id,actor,name)=>({id,uuid:`Scene.scene.Token.${id}`,documentName:'Token',parent:scene,actor,name,playersCanSeeName:false,object:{distanceTo(){assert.fail('no distance checks')},checkCollision(){assert.fail('no sight checks')}}});
 const origin=token('pc',actor,'Hero'),target=token('enemy',enemy,'Secret monster');scene.tokens.set(origin.id,origin);scene.tokens.set(target.id,target);
 const message={id:'trigger',actor,author:player,speaker:{actor:actor.id,scene:scene.id,token:origin.id},rolls:[{}],flags:{pf2e:{context:{type,outcome:'criticalSuccess',target:{token:target.uuid}}}},update};
 const requests=[],calls=[],game={user:gm,users,scenes:new Map([[scene.id,scene]]),actors:new Map([[actor.id,actor]]),messages:new Map([[message.id,message]]),time:{worldTime:100},pf2e:{settings:{tokens:{nameVisibility:true}},actions:new Map()}};
 const docs=new Map([actor,origin,target].map(x=>[x.uuid,x]));
 const handlers=new Map(),Hooks={on:()=>1,off(){}},ownerGame={...game,user:player,pf2e:{...game.pf2e,actions:new Map()}},ownerHandlers=new Map();
 const use=(client,variant)=>async args=>{calls.push({client,variant,args});const check={id:'native-check',author:player,speaker:{actor:actor.id},rolls:[{total:25}],flags:{pf2e:{context:{type:'skill-check',options:args.rollOptions,target:{actor:enemy.uuid,token:target.uuid}}}},update};game.messages.set(check.id,check);return [{message:check}]};
 game.pf2e.actions.set('demoralize',{toActionVariant:variant=>({use:use('gm',variant)})});ownerGame.pf2e.actions.set('demoralize',{toActionVariant:variant=>({use:use('player',variant)})});
 const provider=createFearAutomation({game,fromUuid:async uuid=>docs.get(uuid),choose:async request=>{requests.push(request);return target.uuid;}});
 const ownerProvider=createFearAutomation({game:ownerGame,fromUuid:async uuid=>docs.get(uuid),choose:()=>assert.fail('the owner only opens native UI')});
 ownerProvider.register({Hooks,socket:{register:(name,fn)=>ownerHandlers.set(name,fn)}});
 provider.register({Hooks,socket:{register:(name,fn)=>handlers.set(name,fn),executeAsUser:async(name,userId,payload)=>{assert.equal(userId,player.id);return ownerHandlers.get(name).call({socketdata:{userId:gm.id}},payload);}}});
 return {provider,game,player,actor,target,origin,message,requests,calls,ownerHandlers,ownerProvider,ownerGame,docs};
}
for(const type of ['attack-roll','initiative'])test(`Battle Cry ${type} keeps its real rule choice without distance or sight simulation`,async()=>{
 const f=fixture(type);await f.provider.battleCry(f.message,f.player.id);
 assert.equal(f.requests.length,1);assert.equal(f.requests[0].user,f.player);
 assert.doesNotMatch(JSON.stringify(f.requests[0].choices),/Secret/);
 assert.match(f.requests[0].choices[0].label,/目标/);
 assert.equal(f.requests[0].choices.at(-1).value,'decline');
 assert.equal(f.calls.length,1);assert.equal(f.calls[0].client,'player');assert.equal(f.calls[0].args.target,f.target.object);
 assert.equal(f.message.flags[M].fear.battleCry.status,'done');
});

test('an unauthenticated native fear request cannot open a player check',async()=>{
 const f=fixture();const handler=f.ownerHandlers.get('fear:execute');assert.equal(typeof handler,'function');
 const result=await handler.call({socketdata:{userId:f.player.id}},{messageId:f.message.id,kind:'battleCry',nonce:'forged'});assert.equal(result.ok,false);assert.equal(f.calls.length,0);
});
test('Battle Cry does not expose a GM-hidden token as a player target choice',async()=>{
 const f=fixture('initiative');f.target.hidden=true;await f.provider.battleCry(f.message,f.player.id);assert.equal(f.requests.length,0);assert.equal(f.calls.length,0);
});

test('Battle Cry does not require the GM source token to have a canvas object',async()=>{
 const f=fixture();f.origin.object=null;await f.provider.battleCry(f.message,f.player.id);assert.equal(f.calls[0].client,'player');assert.equal(f.message.flags[M].fear.battleCry.status,'done');
});

test('Disturbing Knowledge uses the author native check and keeps its target DC off the public activity',async()=>{
 const f=fixture(),effects=[],rollRequests=[];
 const item={id:'knowledge',uuid:`${f.actor.uuid}.Item.knowledge`,actor:f.actor,type:'feat',sourceId:FEAR_SOURCES.knowledge};f.actor.items.set(item.id,item);
 f.message.flags.pf2e={origin:{uuid:item.uuid},context:{}};f.message.flags[M]={usageInput:{actualUse:true,targetUuids:[f.target.uuid]}};
 f.target.actor.skills={will:{dc:{value:29}}};f.target.actor.createEmbeddedDocuments=async(_type,data)=>{effects.push(...data);return data};
 f.actor.skills.occultism={rank:3,domains:['occultism','skill-check'],roll:async args=>{
  rollRequests.push(args);const check={id:'knowledge-check',author:f.player,speaker:{actor:f.actor.id},rolls:[{total:24,dice:[{faces:20,total:12}]}],flags:{pf2e:{context:{type:'skill-check',dc:args.dc,options:args.extraRollOptions,target:{actor:f.target.actor.uuid,token:f.target.uuid},outcome:'failure',unadjustedOutcome:'failure',dosAdjustments:{}}}},update};
  f.game.messages.set(check.id,check);await args.callback(check.rolls[0],'failure',check);return check.rolls[0];
 }};
 await f.provider.executeUsage({actor:f.actor,item,message:f.message,user:f.player,action:'fear:disturbing-knowledge'});
 assert.equal(rollRequests.length,1);assert.equal(rollRequests[0].skipDialog,false);assert.equal(rollRequests[0].target.uuid,f.target.actor.uuid);assert.equal(rollRequests[0].dc.visible,false);
 assert.equal(effects.length,1);assert.equal(f.game.messages.get('knowledge-check').author,f.player);assert.doesNotMatch(JSON.stringify(f.message.flags[M]),/"dc"\s*:/);
});

test('a legendary Knowledge check cannot apply its effects to a secondary token relinked during the native dialog',async()=>{
 const f=fixture(),effects=[],item={id:'knowledge',uuid:`${f.actor.uuid}.Item.knowledge`,actor:f.actor,type:'feat',sourceId:FEAR_SOURCES.knowledge};f.actor.items.set(item.id,item);
 const victim=uuid=>({uuid,items:new Map(),skills:{will:{dc:{value:29}}},createEmbeddedDocuments:async(_type,data)=>{effects.push(...data);return data}});
 f.target.actor= victim('Actor.enemy');const second={...f.target,id:'secondary',uuid:'Scene.scene.Token.secondary',actor:victim('Actor.secondary')};f.target.parent.tokens.set(second.id,second);f.docs.set(second.uuid,second);
 f.message.flags.pf2e={origin:{uuid:item.uuid},context:{}};f.message.flags[M]={usageInput:{actualUse:true,targetUuids:[f.target.uuid,second.uuid]}};
 f.actor.skills.occultism={rank:4,domains:['occultism'],roll:async args=>{
  const check={id:'knowledge-check',author:f.player,speaker:{actor:f.actor.id},rolls:[{total:24,dice:[{faces:20,total:12}]}],flags:{pf2e:{context:{type:'skill-check',options:args.extraRollOptions,target:{actor:f.target.actor.uuid,token:f.target.uuid},outcome:'failure',unadjustedOutcome:'failure',dosAdjustments:{}}}},update};f.game.messages.set(check.id,check);await args.callback(check.rolls[0],'failure',check);second.actor=victim('Actor.replacement');return check.rolls[0];
 }};
 await assert.rejects(f.provider.executeUsage({actor:f.actor,item,message:f.message,user:f.player,action:'fear:disturbing-knowledge'}),/目标|改变/);assert.equal(effects.length,0);
});

test('closing a native Battle Cry check releases its unused reaction reservation',async()=>{
 const f=fixture();f.ownerGame.pf2e.actions.set('demoralize',{toActionVariant:()=>({use:async()=>[]})});await f.provider.battleCry(f.message,f.player.id);
 assert.equal(f.message.flags[M].fear.battleCry.status,'cancelled');assert.equal(f.actor.flags[M].fear.reactions.length,0);assert.equal(f.calls.length,0);
});

test('relinking the native target during the check cannot settle Battle Cry on the replacement actor',async()=>{
 const f=fixture(),native=f.ownerGame.pf2e.actions.get('demoralize');f.ownerGame.pf2e.actions.set('demoralize',{toActionVariant:variant=>({use:async args=>{const result=await native.toActionVariant(variant).use(args);f.target.actor={uuid:'Actor.replacement',items:new Map(),testUserPermission:()=>false};return result}})});
 await assert.rejects(f.provider.battleCry(f.message,f.player.id),/目标|来源|改变/);assert.equal(f.calls.length,1);assert.notEqual(f.message.flags[M].fear.battleCry.status,'done');
});
