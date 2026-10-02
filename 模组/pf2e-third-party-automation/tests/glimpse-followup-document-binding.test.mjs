import test from 'node:test';
import assert from 'node:assert/strict';
import {createGlimpseCompat,GLIMPSE_TRIGGER_ID} from '../scripts/glimpse-compat.mjs';
import {GLIMPSE_SOURCES} from '../scripts/glimpse-source.mjs';

function fixture(){
 const callbacks=new Map(),Hooks={once:(name,fn)=>callbacks.set(name,fn),on:()=>1};
 const gm={id:'gm',active:true,isGM:true},users=Object.assign(new Map([[gm.id,gm]]),{activeGM:gm});
 const enemy={id:'enemy',uuid:'Actor.enemy',items:new Map(),async deleteEmbeddedDocuments(_type,ids){for(const id of ids)this.items.delete(id);}};
 const scene={id:'s',tokens:new Map()},token={id:'enemy',uuid:'Scene.s.Token.enemy',documentName:'Token',parent:scene,actor:enemy,actorId:enemy.id,actorLink:true};scene.tokens.set(token.id,token);
 const combat={id:'c',started:true,round:1,turn:0,turns:[{id:'enemy',actor:enemy,token,flags:{}}]};
 let settings={enabled:[GLIMPSE_TRIGGER_ID],disabled:[],sources:[]},node,continuations=0;
 const game={world:{id:'ujx5r8oipw7ercdr'},user:gm,users,actors:new Map([[enemy.id,enemy]]),scenes:new Map([[scene.id,scene]]),combats:new Map([[combat.id,combat]]),system:{id:'pf2e',version:'8.5.1'},modules:new Map([['trigger-engine',{active:true}],['pf2e-trigger-trove',{active:true}]]),pf2e:{ConditionManager:{conditions:new Map([['enfeebled',{uuid:'Condition.enfeebled'}]])}},settings:{get:()=>settings,set:async(_m,_key,next)=>settings=next}};
 class TriggerNode{setOutputValue(key,value){this[key]=value;}async executeNext(){continuations++;const recipient=this.target.actor;const effect={id:'effect',actor:recipient,type:'effect',flags:{},system:{slug:this.slug,context:{origin:{actor:recipient.uuid,token:this.target.token.uuid}},duration:{value:1,unit:'rounds',expiry:'turn-end'},rules:[{key:'GrantItem',uuid:'Condition.enfeebled',inMemoryOnly:true,alterations:[{mode:'override',property:'badge-value',value:2}]}]},async update(changes){for(const [path,value] of Object.entries(changes)){let at=this;const keys=path.split('.');for(const key of keys.slice(0,-1))at=at[key]??={};at[keys.at(-1)]=structuredClone(value);}return this;}};recipient.items.set(effect.id,effect);}}
 const api=createGlimpseCompat({game,api:()=>({TriggerNode}),fromUuid:async uuid=>uuid===GLIMPSE_SOURCES.resistance?{toObject:()=>({type:'effect',system:{rules:[{key:'Resistance',type:'all-damage',value:'@item.origin.level+2'}]}})}:null,query:async payload=>new node()._execute(payload.args)});
 api.register({Hooks});callbacks.get('triggerEngine.registerNodes')((_m,_p,list)=>node=list[0]);callbacks.get('triggerEngine.ready')();
 const expiry={schema:1,combatId:combat.id,combatantId:'enemy',actorUuid:enemy.uuid,tokenUuid:token.uuid,startRound:1,startTurn:0,endRound:2,baselineEnded:null,order:[{id:'enemy',initiative:null}]};
 return {api,game,gm,enemy,scene,token,expiry,get continuations(){return continuations;}};
}
test('the original authorized Glimpse condition continuation remains once only',async()=>{const f=fixture();await f.api.initialize();const result=await f.api.apply({nonce:'legal',enemy:f.token,expiry:f.expiry,authorize:async()=>true});assert.equal(result.effectId,'effect');assert.equal(f.continuations,1);await assert.rejects(f.api.apply({nonce:'legal',enemy:f.token,expiry:f.expiry,authorize:async()=>true}));assert.equal(f.continuations,1);});
test('the private readiness probe and denied source never enter a condition node',async()=>{const f=fixture();await f.api.initialize();assert.equal(f.continuations,0);await assert.rejects(f.api.apply({nonce:'denied',enemy:f.token,expiry:f.expiry,authorize:async()=>false}));assert.equal(f.continuations,0);});
for(const [name,change] of [
 ['GM handoff',f=>{f.game.users.activeGM={id:'next',active:true,isGM:true};}],
 ['current user replacement',f=>{f.game.users.set(f.gm.id,{...f.gm});}],
 ['inactive GM',f=>{f.gm.active=false;}],
 ['enemy relink',f=>{const next={...f.enemy,id:'next',uuid:'Actor.next',items:new Map()};f.game.actors.set(next.id,next);f.token.actor=next;f.token.actorId=next.id;}],
 ['same-UUID enemy replacement',f=>{const next={...f.enemy,items:new Map()};f.game.actors.set(next.id,next);f.token.actor=next;}],
 ['Token replacement',f=>{f.scene.tokens.set(f.token.id,{...f.token});}],
 ['scene deletion',f=>{f.game.scenes.delete(f.scene.id);}]
])test(`Glimpse ${name} during source authorization cannot enter the original condition writer`,async()=>{const f=fixture();await f.api.initialize();await assert.rejects(f.api.apply({nonce:'changed',enemy:f.token,expiry:f.expiry,authorize:async()=>{await Promise.resolve();change(f);return true;}}));assert.equal(f.continuations,0);assert.equal(f.enemy.items.size,0);});
