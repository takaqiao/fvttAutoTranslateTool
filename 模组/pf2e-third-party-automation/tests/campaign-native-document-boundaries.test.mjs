import test from 'node:test';
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {MODULE_ID as ID} from '../scripts/rules.mjs';
import {createThrallAutomation,CONSUME_THRALL_SOURCE} from '../scripts/thrall-automation.mjs';
import {createSocialAutomation} from '../scripts/social-automation.mjs';
import {createCampaignFeats,CAMPAIGN_SOURCES} from '../scripts/campaign-feats.mjs';
const hash=b=>createHash('sha256').update(b).digest('hex');
assert.ok(process.env.FOUNDRY_TOKEN_SOURCE&&process.env.FOUNDRY_ACTOR_SOURCE,'FOUNDRY_TOKEN_SOURCE and FOUNDRY_ACTOR_SOURCE are required pinned Foundry 14.368 source inputs');
const tokenBytes=fs.readFileSync(process.env.FOUNDRY_TOKEN_SOURCE),actorBytes=fs.readFileSync(process.env.FOUNDRY_ACTOR_SOURCE);
assert.equal(hash(tokenBytes),'44b8f03c161d8f077991167d652924228fe2f88b2b47ef4354b98df1a83feba7');assert.equal(hash(actorBytes),'e82580bf9cef39d934c972dee859a3b9ba7ab5f3ebdc7502319dfed1bc214bb3');
function getter(bytes,name){const text=bytes.toString('utf8'),start=text.indexOf('  get '+name+'() {'),end=text.indexOf('\n  }',start)+4;assert.ok(start>=0&&end>start);return text.slice(start,end);}
const tokenActor=getter(tokenBytes,'actor'),baseActor=getter(tokenBytes,'baseActor'),linked=getter(tokenBytes,'isLinked'),isToken=getter(actorBytes,'isToken'),actorToken=getter(actorBytes,'token');
const patch=(doc,changes)=>{for(const[key,value]of Object.entries(changes)){const keys=key.split('.');let at=doc;for(const key of keys.slice(0,-1))at=at[key]??={};at[keys.at(-1)]=structuredClone(value);}};
function fixture(providerName,mode,{synthetic=false}={}){
 const gm={id:'gm',isGM:true,active:true},user={id:'owner',active:true},users=Object.assign(new Map([[gm.id,gm],[user.id,user]]),{activeGM:gm}),game={user:gm,users,actors:new Map(),scenes:new Map(),messages:new Map(),modules:new Map(),time:{worldTime:10}},scene={id:'scene',tokens:new Map(),levels:new Map([['level',{}]])},docs=new Map(),counts={deletes:0,grants:0,effectCreates:0,effectUpdates:0,conditions:0,summaries:0,checks:0,removedReplacementEffects:0},saved=[];
 const Token=Function('game',`return class TokenDocument {static _preventActorDeltaAccess=false;${tokenActor}${baseActor}${linked}}`)(game);
 const Actor=Function('foundry',`return class ActorDocument {${isToken}${actorToken}}`)({documents:{TokenDocument:Token}});
 let fired=false,reconcile=false;
 const mutate=()=>{if(mode==='maintain-base-replaced'&&!reconcile)return;if(fired)return;fired=true;if(['source-base-replaced','maintain-base-replaced'].includes(mode))game.actors.set(sourceBase.id,{...sourceBase});if(mode==='target-base-replaced')game.actors.set(targetBase.id,{...targetBase});if(mode==='source-link-changed')origin.actorLink=true;};
 function actorData(id){return Object.assign(new Actor(),{id,uuid:'Actor.'+id,documentName:'Actor',type:'character',name:id,parent:null,canAct:true,items:new Map(),flags:{},fear:0,system:{resources:{focus:{value:0,max:1}},attributes:{hp:{value:1}},details:{languages:{value:['common']}}},testUserPermission:()=>true,hasCondition:()=>false,isAllyOf:()=>true,getCondition(name){return name==='frightened'&&this.fear>0?{value:this.fear}:null},getStatistic(name){return name==='will'?{dc:{value:15}}:{roll(){},check:{domains:['diplomacy']}}},async update(changes){patch(this,changes);if(changes['system.resources.focus.value']===1)counts.grants++;},async createEmbeddedDocuments(_kind,rows){counts.effectCreates++;if(['empty-create','undefined-create'].includes(mode)&&(providerName==='social'||this===actor))return mode==='empty-create'?[]:undefined;const result=rows.map((row,index)=>({...structuredClone(row),id:'created-'+this.id+'-'+index,uuid:this.uuid+'.Item.created-'+this.id+'-'+index,actor:this,parent:this}));if(mode!=='detached-create')for(const item of result)this.items.set(item.id,item);saved.push(...result);return result;},async deleteEmbeddedDocuments(_kind,ids){for(const id of ids){const current=this.items.get(id);if(current?.replacement)counts.removedReplacementEffects++;this.items.delete(id);}return [];},async decreaseCondition(){counts.conditions++;this.fear--}});}
 const sourceBase=actorData('hero'),targetBase=actorData('target');game.actors.set(sourceBase.id,sourceBase);game.actors.set(targetBase.id,targetBase);
 const origin=Object.assign(new Token(),{id:'origin',uuid:'Scene.scene.Token.origin',documentName:'Token',parent:scene,actorId:sourceBase.id,actorLink:!synthetic,isLazyDelta:false,_source:{level:'level'},object:{center:{x:0,y:0},distanceTo:()=>5,checkCollision:()=>false}}),target=Object.assign(new Token(),{id:'target',uuid:'Scene.scene.Token.target',documentName:'Token',parent:scene,actorId:targetBase.id,actorLink:!synthetic,isLazyDelta:false,object:{center:{x:5,y:0}}});
 const actor=synthetic?actorData(sourceBase.id):sourceBase,recipient=synthetic?actorData(targetBase.id):targetBase;
 if(synthetic){actor.parent=origin;actor.uuid=origin.uuid+'.Actor.'+actor.id;recipient.parent=target;recipient.uuid=target.uuid+'.Actor.'+recipient.id;origin.delta={syntheticActor:actor};target.delta={syntheticActor:recipient};}
 for(const token of [origin,target]){scene.tokens.set(token.id,token);docs.set(token.uuid,token);}game.scenes.set(scene.id,scene);actor.getActiveTokens=()=>[origin];
 if(providerName==='thrall'){recipient.flags={'pf2e-summons-assistant':{summoner:{uuid:actor.uuid}}};recipient.rollOptions={all:{'self:trait:thrall':true}};target.delete=async()=>{counts.deletes++;docs.delete(target.uuid);scene.tokens.delete(target.id);if(mode==='maintain-base-replaced')throw Error('native deletion reply lost');};}else recipient.fear=3;
 const item={id:'feature',uuid:actor.uuid+'.Item.feature',actor,parent:actor,type:providerName==='social'?'feat':'action',sourceId:providerName==='thrall'?CONSUME_THRALL_SOURCE:providerName==='social'?'Compendium.pf2e.feats-srd.Item.6ON8DjFXSMITZleX':CAMPAIGN_SOURCES.walls,system:{frequency:{value:1,max:1,per:'day'}},async update(changes){patch(this,changes)}};actor.items.set(item.id,item);
 const message={id:'use',author:user,speaker:{actor:actor.id,scene:scene.id,token:origin.id},flags:{pf2e:{origin:{actor:actor.uuid,uuid:item.uuid}},[ID]:{usageInput:{actualUse:true,targetUuids:[target.uuid]}}},async update(changes){patch(this,changes)}};game.messages.set(message.id,message);
 const roll={total:25,dice:[{faces:20,total:10}]},card={id:'check',author:user,speaker:{actor:actor.id},rolls:[roll],flags:{pf2e:{context:{options:[],dosAdjustments:{},outcome:'criticalSuccess',unadjustedOutcome:'criticalSuccess'}}}};game.messages.set(card.id,card);
 if(['duplicate-replaced','no-change-update'].includes(mode)){
  const holder=providerName==='social'?recipient:actor,key=providerName==='social'?'social:no-cause-for-alarm:immunity':'walls:'+actor.uuid;
  for(const id of mode==='duplicate-replaced'?['first','duplicate']:['first'])holder.items.set(id,{id,uuid:holder.uuid+'.Item.'+id,type:'effect',actor:holder,parent:holder,flags:{[ID]:providerName==='social'?{nativeEffectKey:key}:{campaignKey:key}},async update(changes){counts.effectUpdates++;patch(this,changes);if(mode==='duplicate-replaced')holder.items.set('duplicate',{id:'duplicate',uuid:holder.uuid+'.Item.duplicate',type:'effect',actor:holder,parent:holder,replacement:true,flags:{[ID]:{otherEffect:true}}});return undefined;}});
 }
 const fromUuid=async uuid=>{if(providerName==='thrall'&&uuid===origin.uuid)mutate();if(uuid.startsWith('Scene.'))return docs.get(uuid);if(providerName==='walls')mutate();return {uuid,toObject:()=>({type:'effect',system:{},flags:{}})}};
 const provider=providerName==='thrall'?createThrallAutomation({game,fromUuid,castEvents:{addActorMatcher(){},addConsumePolicy(){}}}):providerName==='social'?createSocialAutomation({game,runNative:async()=>{counts.checks++;mutate();return {status:'rolled',check:card}}}):createCampaignFeats({game,fromUuid});
 return {game,actor,recipient,origin,target,sourceBase,targetBase,counts,saved,message,provider,beginReconcile(){reconcile=true;},run:()=>provider.executeUsage({actor,item,message,user,action:providerName==='thrall'?'thrall:consume':providerName==='social'?'social:no-cause-for-alarm':'campaign:raise-walls'})};
}

function setup(t,providerName,mode,synthetic=false){
 const old={CONFIG:globalThis.CONFIG,ChatMessage:globalThis.ChatMessage};t.after(()=>Object.assign(globalThis,old));globalThis.CONFIG={Canvas:{polygonBackends:{sound:{testCollision:()=>false}}}};
 const f=fixture(providerName,mode,{synthetic});globalThis.ChatMessage={getSpeaker:()=>({actor:f.actor.id}),create:async()=>{f.counts.summaries++;return {id:'summary'}}};return f;
}
for(const providerName of ['thrall','social','walls'])for(const synthetic of [false,true])test(providerName+' preserves native '+(synthetic?'unlinked synthetic':'linked world')+' actor bindings',async t=>{
 const f=setup(t,providerName,'control',synthetic);await f.run();assert.equal(providerName==='thrall'?f.counts.deletes:f.counts.effectCreates,providerName==='walls'?2:1);if(providerName==='thrall')assert.equal(f.counts.grants,1);
});
for(const providerName of ['thrall','social','walls'])for(const mode of ['source-base-replaced','target-base-replaced','source-link-changed'])test(providerName+' stops '+mode+' across its awaited native boundary',async t=>{
 const f=setup(t,providerName,mode,true);await assert.rejects(f.run());assert.equal(f.counts.deletes,0);assert.equal(f.counts.grants,0);assert.equal(f.counts.effectCreates,0);assert.equal(f.counts.conditions,0);assert.equal(f.counts.summaries,0);
});
for(const providerName of ['social','walls'])for(const mode of ['empty-create','undefined-create','detached-create'])test(providerName+' stops '+mode+' without finishing or replaying partial settlement',async t=>{
 const f=setup(t,providerName,mode);await assert.rejects(f.run());assert.equal(f.counts.effectCreates,1);assert.equal(f.counts.conditions,0);assert.equal(f.counts.summaries,0);
 if(providerName==='walls')assert.equal(f.message.flags[ID].campaignWalls.status,'applying');else assert.deepEqual(f.actor.flags[ID].socialUses,['use']);
 await f.run().catch(()=>{});assert.equal(f.counts.effectCreates,1);assert.equal(f.counts.checks,providerName==='social'?1:0);
});
for(const providerName of ['social','walls'])test(providerName+' stops a duplicate effect document replacement before native deletion',async t=>{
 const f=setup(t,providerName,'duplicate-replaced');await assert.rejects(f.run());assert.equal(f.counts.effectUpdates,1);assert.equal(f.counts.removedReplacementEffects,0);assert.equal(f.counts.effectCreates,0);assert.equal(f.counts.conditions,0);assert.equal(f.counts.summaries,0);
 await f.run().catch(()=>{});assert.equal(f.counts.effectUpdates,1);
});
for(const providerName of ['social','walls'])test(providerName+' accepts an undefined native update result with the current saved effect',async t=>{
 const f=setup(t,providerName,'no-change-update');await f.run();assert.equal(f.counts.effectUpdates,1);assert.equal(f.counts.removedReplacementEffects,0);if(providerName==='social')assert.equal(f.counts.conditions,2);else assert.equal(f.message.flags[ID].campaignWalls.status,'done');
});
test('Thrall reconciliation captures the exact saved synthetic source base before its first lookup',async t=>{
 const f=setup(t,'thrall','maintain-base-replaced',true);await assert.rejects(f.run(),/reply lost/);assert.equal(f.actor.flags[ID].thrallUse.status,'uncertain');f.beginReconcile();await f.provider.maintain(f.actor);assert.equal(f.counts.deletes,1);assert.equal(f.counts.grants,0);assert.equal(f.actor.flags[ID].thrallUse.status,'uncertain');
});
