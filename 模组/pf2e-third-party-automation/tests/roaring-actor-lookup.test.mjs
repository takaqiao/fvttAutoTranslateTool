import test from 'node:test';
import assert from 'node:assert/strict';
import {createRoaringApplause} from '../scripts/roaring-applause.mjs';

function fixture(synthetic=false){
 const gm={id:'gm',isGM:true,active:true},users=new Map([[gm.id,gm]]);users.activeGM=gm;
 const scans={world:0,scenes:0,tokens:0},actors=new Map(),scenes=new Map();
 const track=(collection,key)=>{const values=collection.values.bind(collection);collection.values=()=>{scans[key]++;return values()};return collection};
 track(actors,'world');track(scenes,'scenes');
 for(let index=0;index<1000;index++)actors.set(`other${index}`,{id:`other${index}`,uuid:`Actor.other${index}`,flags:{}});
 const scene={id:'scene',tokens:track(new Map(),'tokens')};scenes.set(scene.id,scene);
 for(let index=0;index<200;index++)scene.tokens.set(`other${index}`,{id:`other${index}`,uuid:`Scene.scene.Token.other${index}`,actor:{id:`other${index}`,uuid:`Scene.scene.Token.other${index}.Actor.other${index}`,flags:{}}});
 const actor={id:'target',uuid:synthetic?'Scene.scene.Token.target.Actor.target':'Actor.target',isToken:synthetic,flags:{}};
 const token={id:'target',uuid:'Scene.scene.Token.target',documentName:'Token',parent:scene,actor,actorLink:!synthetic};scene.tokens.set(token.id,token);
 if(synthetic){actor.token=token;token.baseActor={id:'target',uuid:'Actor.target'};token.actorId='target';actors.set('target',token.baseActor)}else actors.set(actor.id,actor);
 const game={world:{id:'ujx5r8oipw7ercdr'},system:{id:'pf2e',version:'8.5.1'},actors,scenes,users,user:gm,combats:new Map(),messages:new Map(),time:{worldTime:10}};
 const provider=createRoaringApplause({game,nativeCasts:{addInvocationAdapter(){}},effects:{list:()=>[]},saveEvidence:{register(){},cleanup(){}}});
 provider.register({Hooks:{on(){return 1},off(){}},socket:{register(){}}});
 for(const key of Object.keys(scans))scans[key]=0;
 return {game,actor,token,scene,scans,provider};
}

for(const synthetic of [false,true])test(`current ${synthetic?'synthetic':'world'} reaction query checks one Actor without enumerating world or scenes`,t=>{
 const f=fixture(synthetic);t.after(()=>f.provider.cleanup());
 for(let query=0;query<8;query++)assert.deepEqual(f.provider.reactionRestriction(f.actor),{status:'clear',sources:[]});
 assert.deepEqual(f.scans,{world:0,scenes:0,tokens:0});
});

for(const [label,change]of [
 ['world Actor replaced',f=>f.game.actors.set(f.actor.id,{...f.actor})],
 ['world Actor deleted',f=>f.game.actors.delete(f.actor.id)],
 ['world UUID no longer matches registry key',f=>f.actor.uuid='Actor.different'],
])test(`${label} cannot receive clear eligibility through a stale linked Token`,t=>{
 const f=fixture();t.after(()=>f.provider.cleanup());change(f);
 assert.equal(f.token.actor,f.actor);
 assert.deepEqual(f.provider.reactionRestriction(f.actor),{status:'manual',sources:[{sourceNonce:null,status:'manual',reason:'actor-not-live'}]});
});

for(const [label,change]of [
 ['synthetic Actor rebuilt',f=>f.token.actor={...f.actor}],
 ['Token deleted',f=>f.scene.tokens.delete(f.token.id)],
 ['Token rebound',f=>f.scene.tokens.set(f.token.id,{...f.token,actor:{...f.actor}})],
 ['Scene deleted',f=>f.game.scenes.delete(f.scene.id)],
 ['synthetic UUID no longer matches current Token',f=>f.actor.uuid='Scene.scene.Token.other.Actor.target'],
])test(`${label} rejects the original synthetic Actor`,t=>{
 const f=fixture(true);t.after(()=>f.provider.cleanup());change(f);
 assert.deepEqual(f.provider.reactionRestriction(f.actor),{status:'manual',sources:[{sourceNonce:null,status:'manual',reason:'actor-not-live'}]});
});
