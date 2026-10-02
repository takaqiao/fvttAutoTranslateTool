import test from 'node:test';
import assert from 'node:assert/strict';
import {executeActorAction} from '../scripts/runtime.mjs';
import {MODULE_ID,SOURCES,levelDC} from '../scripts/rules.mjs';

function fixture(t,{synthetic=false,linked=false}={}){
 const previous=globalThis.game;t.after(()=>globalThis.game=previous);
 const gm={id:'gm',isGM:true},player={id:'player'},writes=[];
 const actor=uuid=>({id:'base',uuid,type:'character',level:6,items:[{id:'feat',sourceId:SOURCES.circadian}],flags:{},
  system:{abilities:{con:{mod:2}},attributes:{hp:{value:1,max:20}}},testUserPermission:user=>user===player||user===gm,hasCondition:()=>false,
  async update(change){writes.push({actorUUID:this.uuid,change});for(const[path,value]of Object.entries(change)){
   const keys=path.split('.');let target=this;for(const key of keys.slice(0,-1))target=target[key]??={};target[keys.at(-1)]=value;
  }return this;}});
 const base=actor('Actor.base'),first=synthetic?actor('Scene.scene.Token.first.Actor.base'):base,second=synthetic?actor('Scene.scene.Token.second.Actor.base'):base;
 const scene={id:'scene',tokens:new Map([['first',{id:'first',actor:first}],['second',{id:'second',actor:second}]])};
 const game={user:gm,users:{activeGM:gm},time:{worldTime:100},actors:new Map([[base.id,base]]),scenes:new Map([[scene.id,scene]]),messages:new Map()};
 const message={id:'check',author:player,speaker:{actor:base.id,...synthetic||linked?{scene:scene.id,token:'first'}:{}},
  // PF2e 8.5.1 actor -> Foundry14.368 speakerActor/getSpeakerActor: token first, then world Actor.
  get actor(){const {speaker}=this;return (speaker.scene&&speaker.token?game.scenes.get(speaker.scene)?.tokens.get(speaker.token)?.actor:null)??game.actors.get(speaker.actor)??this.author.character??null;},
  flags:{pf2e:{context:{type:'skill-check',options:['action:third-party-circadian'],domains:['survival'],dc:{value:levelDC(6)}}}},rolls:[{options:{degreeOfSuccess:2}}],
  async update(change){writes.push({messageId:this.id,change});for(const[path,value]of Object.entries(change)){
   const keys=path.split('.');let target=this;for(const key of keys.slice(0,-1))target=target[key]??={};target[keys.at(-1)]=value;
  }return this;}};
 game.messages.set(message.id,message);globalThis.game=game;
 return {game,player,base,first,second,scene,message,writes,rest:recipient=>executeActorAction(recipient,'rest',{eligible:true,messageId:message.id},player)};
}

for(const mode of ['world','linked','synthetic'])test(`partial rest accepts the exact original ${mode} Actor and preserves the once guard`,async t=>{
 const f=fixture(t,{synthetic:mode==='synthetic',linked:mode==='linked'}),recipient=mode==='synthetic'?f.first:f.base;
 await f.rest(recipient);assert.equal(recipient.system.attributes.hp.value,7);assert.equal(f.message.flags[MODULE_ID].restApplied,recipient.uuid);
 const before=f.writes.length;await assert.rejects(f.rest(recipient),/24小时|已经结算/);assert.equal(f.writes.length,before);
});

test('partial rest cannot apply one unlinked Token Actor check to another with the same base ID',async t=>{
 const f=fixture(t,{synthetic:true});assert.equal(f.first.id,f.second.id);assert.notEqual(f.first.uuid,f.second.uuid);
 await assert.rejects(f.rest(f.second),/有效.*生存检定/);assert.deepEqual(f.writes,[]);assert.equal(f.second.system.attributes.hp.value,1);assert.equal(f.message.flags[MODULE_ID],undefined);
});

test('partial rest cannot apply a synthetic Actor check to its world base Actor',async t=>{
 const f=fixture(t,{synthetic:true});await assert.rejects(f.rest(f.base),/有效.*生存检定/);assert.deepEqual(f.writes,[]);
});

test('a deleted source Token cannot fall back to its world Actor when settling partial rest',async t=>{
 const f=fixture(t,{synthetic:true});f.scene.tokens.delete('first');assert.equal(f.message.actor,f.base);
 await assert.rejects(f.rest(f.base),/有效.*生存检定/);assert.deepEqual(f.writes,[]);
});

test('a check without an original native Actor cannot settle from speaker ID alone',async t=>{
 const f=fixture(t);f.game.actors.delete(f.base.id);assert.equal(f.message.actor,null);
 await assert.rejects(f.rest(f.base),/有效.*生存检定/);assert.deepEqual(f.writes,[]);
});

test('partial rest keeps the original saved message and does not consume a missing or completed check',async t=>{
 const f=fixture(t);f.game.messages.delete(f.message.id);await assert.rejects(f.rest(f.base),/有效.*生存检定/);assert.deepEqual(f.writes,[]);
 f.game.messages.set(f.message.id,f.message);f.message.flags[MODULE_ID]={restApplied:f.base.uuid};await assert.rejects(f.rest(f.base),/已经结算/);assert.deepEqual(f.writes,[]);
});
