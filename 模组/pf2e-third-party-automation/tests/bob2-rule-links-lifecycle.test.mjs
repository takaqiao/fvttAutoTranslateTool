import test from 'node:test';
import assert from 'node:assert/strict';
let api={};try{api=await import('../scripts/bob2-rule-links.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error}
const ID='pf2e-third-party-automation',S={overture:'Compendium.pf2e.spells-srd.Item.f0Z5mqGA6Yu79B8x',effect:'Compendium.pf2e.spell-effects.Item.ITErgFRfydm1xmnW',aid:'Compendium.pf2e.actionspf2e.Item.HCl3pzVefiv9ZKQW',aidEffect:'Compendium.pf2e.other-effects.Item.AHMUpMbaVkZ5A1KX'};
function patch(object,data){for(const [path,value]of Object.entries(data)){const keys=path.split('.');let node=object;for(const key of keys.slice(0,-1))node=node[key]??={};node[keys.at(-1)]=structuredClone(value)}return object}
function setup(){
 const gm={id:'gm',isGM:true,active:true},user={id:'user',active:true},docs=new Map(),calls={paid:0,rolls:[],creates:[]},answers=[];
 function actor(id){const a={id,type:'character',uuid:'Actor.'+id,name:id,level:5,items:new Map(),flags:{},testUserPermission:()=>true,getStatistic:()=>({rank:2}),async update(data){patch(this,data);return this},async createEmbeddedDocuments(_type,items){return items.map(data=>{const i=structuredClone(data);i.id='effect'+calls.creates.length;i.uuid=a.uuid+'.Item.'+i.id;i.actor=a;i.update=async data=>patch(i,data);a.items.set(i.id,i);docs.set(i.uuid,i);calls.creates.push(i);return i})},async deleteEmbeddedDocuments(_type,ids){for(const id of ids){docs.delete(a.items.get(id)?.uuid);a.items.delete(id)}}};docs.set(a.uuid,a);return a;}
 const bard=actor('bard'),ally=actor('ally'),scene={id:'scene',tokens:new Map()},token={id:'ally',uuid:'Scene.scene.Token.ally',name:'ally',actor:ally,parent:scene,documentName:'Token'},sourceToken={id:'bard',uuid:'Scene.scene.Token.bard',actor:bard,parent:scene,documentName:'Token'};scene.tokens.set('ally',token);scene.tokens.set('bard',sourceToken);docs.set(token.uuid,token);docs.set(sourceToken.uuid,sourceToken);
 const game={world:{id:'bob-2'},system:{id:'pf2e'},user:gm,users:new Map([[gm.id,gm],[user.id,user]]),actors:new Map([['bard',bard],['ally',ally]]),messages:new Map(),scenes:new Map([['scene',scene]]),time:{worldTime:100},modules:new Map([['patreon-v3',{active:true}]])};game.users.activeGM=gm;
 function item(id,type,source){const i={id,uuid:bard.uuid+'.Item.'+id,actor:bard,type,sourceId:source,system:{rules:[]}};bard.items.set(id,i);docs.set(i.uuid,i);return i}
 const spell=item('spell','spell',S.overture),aid=item('aid','action',S.aid);
 function message(id,i,native=true){const m={id,uuid:'ChatMessage.'+id,author:user,speaker:{actor:bard.id,scene:scene.id,token:sourceToken.id},flags:{pf2e:{origin:{uuid:i.uuid,actor:bard.uuid,type:i.type}},[ID]:{usageInput:{actualUse:true,targetUuids:[token.uuid]},...native&&i===spell?{nativeCast:{id:'cast-'+id}}:{}}},async update(data){patch(this,data);return this}};game.messages.set(id,m);docs.set(m.uuid,m);return m;}
 const cast=message('cast',spell),use=message('use',aid),fromUuid=async uuid=>docs.get(uuid)??([S.effect,S.aidEffect].includes(uuid)?{type:'effect',toObject:()=>({type:'effect',name:'Native effect',system:{rules:[],duration:{value:1,unit:'rounds',expiry:'turn-start'}},flags:{},...calls.omitCompendiumSource?{}:{_stats:{compendiumSource:uuid}}})}:null);
 const castEvents={addMatcher:()=>{},ensurePaid:async()=>{calls.paid++;return {id:'cast-cast'}}};
 const runNative=async(ctx,request,_native,beforeRoll)=>{calls.rolls.push(request);if(calls.cancelRoll)return {status:'cancelled'};await beforeRoll?.();return {status:'rolled',check:{rolls:[{options:{degreeOfSuccess:calls.outcome??2}}],flags:{pf2e:{context:{outcome:['criticalFailure','failure','success','criticalSuccess'][calls.outcome??2]}}}}}};
 const choose=async({choices})=>answers.length?answers.shift():choices[0].value;
 const provider=api.createBob2RuleLinks?.({game,fromUuid,choose,runNative,castEvents});
 const context=(i,m,action)=>({actor:bard,item:i,message:m,user,action});
 const prepare=()=>provider?.executeUsage(context(spell,cast,'bob2:overture'));
 const assist=()=>provider?.executeUsage(context(aid,use,'bob2:aid'));
 return {game,bard,ally,spell,aid,cast,use,token,provider,calls,answers,prepare,assist,message,context};
}
test('native Cast prepares one ally skill, then Aid rolls native Performance and grants a single-use native benefit',async()=>{
 const f=setup();await f.prepare();await f.prepare();assert.equal(f.calls.paid,1);assert.equal(f.calls.creates.length,1);
 await f.assist();await f.assist();assert.equal(f.calls.rolls.length,1);assert.equal(f.calls.creates.length,2);
 const request=f.calls.rolls[0];assert.equal(request.nativeAction,'aid');assert.equal(request.statistic,'performance');assert.equal(request.modifiers,undefined);assert.ok(request.options.includes('action:aid'));
 const benefit=[...f.ally.items.values()][0];assert.equal(benefit._stats.compendiumSource,S.aidEffect);assert.equal(benefit.system.rules[0].type,'circumstance');assert.equal(benefit.system.rules[0].value,1);assert.deepEqual(benefit.system.rules[0].removeAfterRoll,['check:statistic:acrobatics']);assert.equal(benefit.system.rules[0].selector,'acrobatics');
 assert.equal([...f.bard.items.values()].filter(i=>i.type==='effect').length,0);
});
test('display-only spell cards cannot create preparation or pay resources',async()=>{
 const f=setup();delete f.cast.flags[ID].nativeCast;
 await assert.rejects(async()=>f.provider.executeUsage(f.context(f.spell,f.cast,'bob2:overture')),/施法/);assert.equal(f.calls.paid,0);assert.equal(f.calls.creates.length,0);
});
test('choice and native roll cancellation preserve preparation and never apply Aid',async()=>{
 for(const stage of ['choice','roll']){
  const f=setup();await f.prepare();if(stage==='choice')f.answers.push(null);else f.calls.cancelRoll=true;
  await f.assist();assert.equal(f.ally.items.size,0);assert.equal([...f.bard.items.values()].filter(i=>i.type==='effect').length,1);
 }
});
test('native Aid critical result uses Performance proficiency, critical failure gives -1, failure gives no effect',async()=>{
 for(const [outcome,rank,bonus]of [[3,2,2],[3,3,3],[3,4,4],[0,2,-1],[1,2,null]]){
  const f=setup();f.calls.outcome=outcome;f.bard.getStatistic=()=>({rank});await f.prepare();await f.assist();
  const result=[...f.ally.items.values()][0];assert.equal(result?.system.rules[0].value??null,bonus);assert.equal([...f.bard.items.values()].filter(i=>i.type==='effect').length,0);
 }
});
test('expired preparation, ordinary Aid and other worlds do not enter Overture',async()=>{
 const f=setup();assert.equal(f.provider?.resolveAction(f.aid),undefined);await f.prepare();const effect=[...f.bard.items.values()].find(i=>i.type==='effect');assert.ok(effect);effect.isExpired=true;assert.equal(f.provider.resolveAction(f.aid),undefined);
 f.game.world.id='other';assert.equal(f.provider.resolveAction(f.spell),undefined);
});
test('Patreon keeps ordinary Aid and excludes only the managed Overture Aid route',()=>{
 const original={'5tMRJEGX3HmeZu3s':{uuid:'5tMRJEGX3HmeZu3s',source:[S.aid],isActive:true,baseRules:[{type:'base',triggerType:'postInfo',target:'TargetEffect',value:S.aidEffect,predicate:['origin:item:aid']}]}};
 const f=setup(),result=api.buildBob2RuleLinkRepairs?.(original,{game:f.game})??{rules:original,changes:[]};assert.equal(result.changes.length,1);assert.deepEqual(result.rules['5tMRJEGX3HmeZu3s'].baseRules[0].predicate,['origin:item:aid',{not:ID+':usage:bob2:aid'}]);assert.equal(result.rules['5tMRJEGX3HmeZu3s'].isActive,true);
});
test('Overture can prepare one of the beneficiary existing Lore statistics',async()=>{
 const f=setup();f.ally.skills={'vampire-lore':{label:'Vampire Lore'}};f.answers.push(f.token.uuid,'vampire-lore');
 await f.prepare();await f.assist();assert.equal([...f.ally.items.values()][0].system.rules[0].selector,'vampire-lore');
});
test('a compendium effect clone retains its exact source when the raw object omits import statistics',async()=>{
 const f=setup();f.calls.omitCompendiumSource=true;await f.prepare();assert.equal(f.provider.resolveAction(f.aid),'bob2:aid');await f.assist();assert.equal([...f.ally.items.values()][0]._stats.compendiumSource,S.aidEffect);
});
