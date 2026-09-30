import test from 'node:test';
import assert from 'node:assert/strict';
import {createEldamonBasicSettlement} from '../scripts/eldamon-basic-settlement.mjs';
import {ELECTRICITY_BASIC_SOURCES as S} from '../scripts/eldamon-electricity.mjs';

function fixture(kind='manipulation'){
 const actor={id:'caster',uuid:'Actor.caster',items:new Map(),testUserPermission:u=>u.id==='player'};
 const item={id:'power',uuid:'Actor.caster.Item.power',actor,sourceId:S[kind]};
 actor.items.set(item.id,item);actor.items.set('element',{sourceId:S.element});
 const token={uuid:'Scene.scene.Token.target',actor:{name:'Target'},name:'T target'};
 const other={uuid:'Scene.scene.Token.other',actor:{name:'Wrong'},name:'Wrong'};
 const user={id:'player',targets:new Set([{document:token}])};
 const message={id:'card',uuid:'ChatMessage.card',timestamp:100,speaker:{actor:actor.id},flags:{pf2e:{origin:{uuid:item.uuid}},'pf2e-third-party-automation':{usageInput:{actualUse:true,targetUuids:[token.uuid]}}}};
 const game={world:{id:'ujx5r8oipw7ercdr'},system:{id:'pf2e',version:'8.5.1'},modules:new Map([['battlezoo-eldamon-pf2e',{active:true}]]),user,actors:new Map([[actor.id,actor]]),messages:new Map([[message.id,message]])};
 const calls=[],prompts=[],hooks=[];
 if(kind==='shield'){actor.items.set('shield-effect',{id:'shield-effect',sourceId:S.shieldEffect,system:{context:{origin:{actor:actor.uuid,item:item.uuid}}}});actor.flags={'pf2e-third-party-automation':{electricity:{activeShield:'use-card',basicActions:{'use-card':{key:'use-card',kind:'shield',status:'armed',messageUuid:message.uuid,itemUuid:item.uuid,effectId:'shield-effect',createdAt:100,expires:null}}}}};}
 const build=(options={})=>createEldamonBasicSettlement({game,fromUuid:async uuid=>[message,token,other,item,actor].find(doc=>doc.uuid===uuid),apply:async payload=>{calls.push(payload);return {kind,manualExpiry:false}},confirm:async context=>{prompts.push(context);return true},notify:()=>{},random:()=> 'basic-confirmation',...options});
 return {game,actor,item,token,other,user,message,calls,prompts,hooks,build};
}

test('basic settlement captures the explicit player T target before its GM request completes',async()=>{
 const f=fixture();let resolve;
 const provider=f.build({apply:payload=>{f.calls.push(payload);return new Promise(r=>{resolve=r})}});
 const pending=provider.settleFromCard(f.message);
 f.user.targets=new Set([{document:f.other}]);await new Promise(r=>setImmediate(r));resolve({});await pending;
 assert.equal(f.calls.length,1);assert.equal(f.calls[0].targetUuid,f.token.uuid);
 assert.equal(f.calls[0].actorUuid,f.actor.uuid);assert.equal(f.calls[0].confirmed,true);
});

test('zero or multiple captured manipulation targets are not replaced by the current selection',async()=>{
 const f=fixture(),provider=f.build();f.user.targets.clear();
 f.message.flags['pf2e-third-party-automation'].usageInput.targetUuids=[];
 await assert.rejects(provider.settleFromCard(f.message),/T/);
 f.user.targets=new Set([{document:f.token},{document:f.other}]);
 f.message.flags['pf2e-third-party-automation'].usageInput.targetUuids=[f.token.uuid,f.other.uuid];
 await assert.rejects(provider.settleFromCard(f.message),/T/);
 assert.equal(f.prompts.length,0);assert.equal(f.calls.length,0);
});

test('the declared card action needs no extra local trigger or adjacency confirmation',async()=>{
 const f=fixture();await f.build({confirm:()=>assert.fail('native card click is the declaration')}).settleFromCard(f.message);
 assert.equal(f.calls.length,1);assert.equal(f.prompts.length,0);
});

test('one pending GM application cannot be double-clicked into two applications',async()=>{
 const f=fixture();let resolve;const provider=f.build({apply:payload=>{f.calls.push(payload);return new Promise(r=>{resolve=r})}});
 const first=provider.settleFromCard(f.message);await provider.settleFromCard(f.message);await new Promise(r=>setImmediate(r));
 resolve({});await first;assert.equal(f.calls.length,1);
});

test('only owned native item cards qualify, including native self-effect shield cards',()=>{
 const f=fixture('shield'),provider=f.build();assert.equal(provider.cardContext(f.message).kind,'shield');
 f.message.flags.pf2e={context:{type:'self-effect',item:f.item.id}};
 assert.equal(provider.cardContext(f.message).kind,'shield');
 f.message.flags.pf2e={context:{type:'damage-roll'},origin:{uuid:f.item.uuid}};
 assert.equal(provider.cardContext(f.message),null);
 f.message.flags.pf2e={origin:{uuid:f.item.uuid}};f.user.id='bystander';
 assert.equal(provider.cardContext(f.message),null);
});

test('absent Eldamon, another world or another system version installs no card hook',()=>{
 for(const change of [f=>f.game.modules.clear(),f=>f.game.world.id='another',f=>f.game.system.version='8.6.0']){
  const f=fixture();change(f);const provider=f.build();
  provider.register({Hooks:{on:(...args)=>f.hooks.push(args)}});
  assert.equal(f.hooks.length,0);assert.equal(provider.cardContext(f.message),null);
 }
});

test('enabled continuation provider indexes relevant cards and actor renders without movement hooks',()=>{
 const f=fixture();f.build().register({Hooks:{on:(name)=>f.hooks.push(name)}});
 assert.ok(f.hooks.includes('renderChatMessageHTML'));assert.ok(f.hooks.includes('createChatMessage'));assert.ok(f.hooks.includes('updateChatMessage'));assert.ok(f.hooks.includes('deleteChatMessage'));assert.ok(f.hooks.includes('renderActorSheetPF2e'));
 assert.equal(f.hooks.some(hook=>/Token|Scene|Move|Sight/i.test(hook)),false);
});

test('manual outside-combat shield expiry is surfaced to the invoking player',async()=>{
 const f=fixture('shield'),notices=[];
 await f.build({apply:async()=>({manualExpiry:true}),notify:message=>notices.push(message)}).settleFromCard(f.message);
 assert.equal(notices.length,1);assert.match(notices[0],/手动/);
});
