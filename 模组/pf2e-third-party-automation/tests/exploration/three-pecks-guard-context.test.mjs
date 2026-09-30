import test from 'node:test';import assert from 'node:assert/strict';
import {fixture} from '../salubrious-kiss-fixture.mjs';
import {createSalubriousDamageGuard} from '../../scripts/salubrious-kiss-damage-guard.mjs';
import {createExplorationOwnerOperations} from '../../scripts/exploration/owner-operations.mjs';
import {identityKeys,marker} from '../../scripts/salubrious-kiss-context.mjs';
import {treatmentOutcome} from '../../scripts/salubrious-kiss-rules.mjs';
async function application({automatic=true}={}){
 const f=fixture(),{M}=f;f.claim.userId=f.gm.id;f.claim.privacy={schema:1,userId:f.gm.id,mode:'public',whisper:[],blind:false};f.setClaim(f.claim);f.game.settings={get:(namespace,key)=>namespace==='core'&&key==='messageMode'?'public':false};
 const activity={id:f.claim.nonce,providerId:'refocus',actorUUID:f.actor.uuid,patientUUIDs:[f.patient.uuid],startedAt:100,endsAt:700,state:'planned',source:{type:'coordinator'},options:{threePecks:true}};
 const operations=createExplorationOwnerOperations({game:f.game,fromUuid:f.fromUuid,ledger:{getActivity:async()=>structuredClone(activity)}}),ctx=await operations.createActivityContext(activity);f.game.time.worldTime=700;
 class DamageRoll{constructor(){this._evaluated=true;this.total=38}toJSON(){return {class:'DamageRoll',evaluated:true,total:this.total,formula:treatmentOutcome({degree:2,tier:f.claim.tier}).formula}}}
 const checkRoll={_evaluated:true,total:26,options:{degreeOfSuccess:2},toJSON(){return {class:'CheckRoll',evaluated:true,total:26,options:this.options}}};
 const identity={...Object.fromEntries(identityKeys.map(k=>[k,f.claim[k]])),privacy:structuredClone(f.claim.privacy)},base={author:f.gm,speaker:{actor:f.actor.id,scene:f.scene.id,token:f.token.id},actor:f.actor,blind:false,whisper:[]};
 const context={origin:{actor:f.actor.uuid,token:f.token.uuid},type:'skill-check',action:'treat-wounds',dc:{value:30},domains:['occultism'],options:['action:treat-wounds',marker('check',f.claim)],outcome:'success',messageMode:'public'};
 context.options.push('skip-handling-message');
 const check={...base,id:'C',isCheckRoll:true,rolls:[checkRoll],flags:{pf2e:{origin:{actor:f.actor.uuid,uuid:f.item.uuid,type:'feat'},context,suppressDamageButtons:true},[M]:{salubriousKiss:{kind:'check',...identity}}}};
 const damage={...base,id:'D',isDamageRoll:true,rolls:[new DamageRoll()],flags:{pf2e:{origin:{actor:f.actor.uuid,uuid:f.item.uuid,type:'feat',messageId:check.id},context:{...structuredClone(context),options:[...context.options,marker('damage',f.claim)]},suppressDamageButtons:true},[M]:{salubriousKiss:{kind:'damage',...identity}}}};
 f.game.messages.set('C',check);f.game.messages.set('D',damage);f.claim.state='applying';f.claim.result={checkId:'C',damageId:'D',degree:2};f.claim.application={state:'started',userId:f.gm.id,damageId:'D',targetUuid:f.target.uuid};f.setClaim(f.claim);
 f.actor.flags[M].salubriousKissExecutions=[{nonce:f.claim.nonce,state:'done',result:{checkId:'C',damageId:'D'}}];f.patient.flags[M]={salubriousKiss:{pending:{nonce:f.claim.nonce,actorUuid:f.actor.uuid}}};
 const params={damage:-38,token:f.target,item:f.item,skipIWR:true,final:false,shieldBlockRequest:false,outcome:'success',rollOptions:new Set([...damage.flags.pf2e.context.options.filter(o=>o!=='skip-handling-message'),marker('apply',f.claim),`${M}:source:D:0`])};
 const request={reactor:f.actor,actor:f.patient,item:f.item,token:f.token,target:f.target,check,message:damage,claim:f.claim,params,...automatic?{explorationScope:{activity,ctx}}:{}};
 const guard=createSalubriousDamageGuard({game:f.game,DamageRoll,isExplorationContext:operations.isActivityContext,messagePrivacy:{withNativeApplication:(_request,_claim,fn)=>fn()}});return {...f,guard,operations,activity,ctx,request};
}
for(const change of ['cancelled','time','encounter'])test(`bound Three Pecks final native assertion refuses ${change} after a provider await`,async()=>{
 const f=await application();await f.guard.authorize(f.request);let entered=0;
 await assert.rejects(f.guard.applyDamage(f.patient,f.request.params,async(params,assertNative)=>{
  await Promise.resolve();if(change==='cancelled')f.operations.cancelActivity(f.activity);else if(change==='time')f.game.time.worldTime++;else f.game.combat={started:true};
  assertNative(f.patient,params);entered++;
 }),/探索|上下文/);assert.equal(entered,0);
});
test('copied private context cannot issue a native Three Pecks grant',async()=>{const f=await application();f.request.explorationScope.ctx={...f.ctx};const writes=f.writes.length;await assert.rejects(f.guard.authorize(f.request),/探索|上下文/);assert.equal(f.writes.length,writes)});
test('ordinary Three Pecks native grant preserves its original final application route',async()=>{const f=await application({automatic:false});await f.guard.authorize(f.request);let entered=0;await f.guard.applyDamage(f.patient,f.request.params,async(params,assertNative)=>{assertNative(f.patient,params);entered++});assert.equal(entered,1)});
