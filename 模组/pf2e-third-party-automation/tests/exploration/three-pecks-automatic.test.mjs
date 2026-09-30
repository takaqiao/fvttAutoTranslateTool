import test from 'node:test';
import assert from 'node:assert/strict';
import {fixture} from '../salubrious-kiss-fixture.mjs';
import {createSalubriousExecutor} from '../../scripts/salubrious-kiss-executor.mjs';
import {createSalubriousKiss} from '../../scripts/salubrious-kiss.mjs';
import {createExplorationOwnerOperations} from '../../scripts/exploration/owner-operations.mjs';

async function treatment({mode='public',beforeCallback=()=>{},afterCheck=()=>{},afterDamage=()=>{},authorizeDamage,hpPools}={}){
 const f=fixture(),cards=[],checks=[],evaluations=[];let manualWindows=0;
 f.game.settings={get:(namespace,key)=>namespace==='core'&&key==='messageMode'?mode:false};f.game.scenes.active=f.scene;
 f.claim.userId=f.gm.id;
 f.claim.privacy={schema:1,userId:f.gm.id,mode,whisper:mode==='public'?[]:[f.gm.id],blind:mode==='blind'};
 f.setClaim(f.claim);
 const activity={id:f.claim.nonce,providerId:'refocus',actorUUID:f.actor.uuid,patientUUIDs:[f.patient.uuid],startedAt:100,endsAt:700,state:'planned',source:{type:'coordinator'},options:{threePecks:true,rank:'master'}};
 const operations=createExplorationOwnerOperations({game:f.game,fromUuid:f.fromUuid,ledger:{getActivity:async()=>structuredClone(activity)}});
 const ctx=await operations.createActivityContext(activity);f.game.time.worldTime=700;
 const scope={activity,ctx};
 class DamageRoll {
  constructor(formula){this.formula=formula;this._evaluated=false;this.options={}}
  async evaluate(options){evaluations.push(options);this._evaluated=true;this.total=38;await afterDamage({f,operations,activity});return this}
  toJSON(){return {class:'DamageRoll',formula:this.formula,evaluated:this._evaluated,total:this.total,options:this.options}}
 }
 const check={_evaluated:true,total:26,options:{degreeOfSuccess:2},toJSON(){return {class:'CheckRoll',evaluated:true,total:this.total,options:this.options}}};
 f.actor.skills.occultism.check={async roll(options){
  checks.push(options);await beforeCallback({f,operations,activity});
  const data={author:f.gm.id,speaker:{actor:f.actor.id,scene:f.scene.id,token:f.token.id},blind:mode==='blind',whisper:mode==='public'?[]:[f.gm.id],rolls:[check.toJSON()],flags:{pf2e:{origin:{actor:f.actor.uuid,uuid:f.item.uuid,type:'feat'},context:{origin:{actor:f.actor.uuid,token:f.token.uuid},type:'skill-check',action:'treat-wounds',dc:{value:30},domains:['occultism'],options:options.extraRollOptions,outcome:'success',messageMode:mode}}}};
  await options.callback(check,'success',{toObject:()=>structuredClone(data)});return check;
 }};
 const createMessage=async data=>{
  const json=data.rolls[0],roll=json.class==='DamageRoll'?Object.assign(new DamageRoll(json.formula),{_evaluated:json.evaluated,total:json.total,options:json.options}):check;
  const message={...structuredClone({...data,rolls:[]}),id:`card${cards.length+1}`,rolls:[roll],isCheckRoll:json.class==='CheckRoll',isDamageRoll:json.class==='DamageRoll'};
  cards.push(message);f.game.messages.set(message.id,message);if(message.isCheckRoll)await afterCheck({f,operations,activity});return message;
 };
 const handlers=new Map(),observers=new Map();let hookId=0;
 const Hooks={on:(event,callback)=>{observers.set(++hookId,{event,callback});return hookId},off:(_event,id)=>observers.delete(id),fire:(event,...args)=>{for(const entry of observers.values())if(entry.event===event)entry.callback(...args)}};
 const executor=createSalubriousExecutor({...f,Hooks,checkScope:{run:(_context,fn)=>fn()},createMessage,DamageRoll,isExplorationContext:operations.isActivityContext,authorizeDamage,hpPools,manualDamageRoll:async({roll})=>{manualWindows++;return roll.evaluate()}});
 executor.register({socket:{register:(name,handler)=>handlers.set(name,handler)}});
 return {...f,scope,operations,executor,cards,checks,evaluations,handlers,Hooks,get manualWindows(){return manualWindows}};
}

for(const mode of ['public','blind'])test(`bound automatic Three Pecks rolls once without manual windows and retains ${mode} privacy`,async()=>{
 const f=await treatment({mode}),result=await f.executor.roll(f.claim,f.scope);
 assert.equal(f.checks.length,1);assert.equal(f.checks[0].skipDialog,true);assert.equal(f.checks[0].event,null);
 assert.equal(f.manualWindows,0);assert.deepEqual(f.evaluations,[{allowInteractive:mode!=='blind'}]);
 assert.equal(f.cards.length,2);assert.equal(f.cards[1].flags.pf2e.origin.messageId,f.cards[0].id);
 assert.equal(result.degree,2);assert.equal(f.cards[1].blind,mode==='blind');assert.deepEqual(f.cards[1].whisper,mode==='public'?[]:[f.gm.id]);
 await assert.rejects(f.executor.roll(f.claim,f.scope));assert.equal(f.checks.length,1);assert.equal(f.evaluations.length,1);
});

const invalidScopes=[
 ['copied opaque context',scope=>scope.ctx={...scope.ctx}],
 ['public boolean',(_scope,f)=>{f.scope=true}],
 ['wrong nonce',scope=>scope.activity.id='other'],
 ['wrong actor',scope=>scope.activity.actorUUID='Actor.other'],
 ['wrong patient',scope=>scope.activity.patientUUIDs=['Actor.other']],
 ['group patient claim',scope=>scope.activity.patientUUIDs.push('Actor.other')],
 ['wrong start',scope=>scope.activity.startedAt=99],
 ['wrong provider',scope=>scope.activity.providerId='treat-wounds'],
 ['missing Three Pecks option',scope=>scope.activity.options.threePecks=false],
 ['manual source',scope=>scope.activity.source={type:'user-record',manual:true}],
 ['wrong completion time',(_scope,f)=>f.game.time.worldTime=701],
 ['started encounter',(_scope,f)=>f.game.combat={started:true}],
 ['cancelled activity',(_scope,f)=>f.operations.cancelActivity(f.scope.activity)],
];
for(const [name,change]of invalidScopes)test(`automatic Three Pecks rejects ${name} before any native dice`,async()=>{
 const f=await treatment();change(f.scope,f);await assert.rejects(f.executor.roll(f.claim,f.scope));
 assert.equal(f.checks.length,0);assert.equal(f.evaluations.length,0);assert.equal(f.cards.length,0);assert.equal(f.manualWindows,0);
});

test('ordinary Three Pecks still uses the upstream check and damage confirmations',async()=>{
 const f=await treatment();await f.executor.roll(f.claim);assert.equal(f.checks[0].skipDialog,false);assert.equal(f.checks[0].event,null);assert.equal(f.manualWindows,1);assert.equal(f.cards.length,2);
});
test('socket payload cannot turn an ordinary Three Pecks roll into automatic exploration',async()=>{
 const f=await treatment(),handler=f.handlers.get('salubrious-kiss:roll');
 const result=await handler.call({socketdata:{userId:f.gm.id}},{actorUuid:f.actor.uuid,nonce:f.claim.nonce,automatic:true,exploration:f.scope,activity:f.scope.activity,ctx:f.scope.ctx});
 assert.equal(result.ok,true);assert.equal(f.checks[0].skipDialog,false);assert.equal(f.manualWindows,1);
});

for(const [name,options,expectedCards,expectedEvaluations]of [
 ['cancellation during check',{beforeCallback:({operations,activity})=>operations.cancelActivity(activity)},0,0],
 ['external time after check',{afterCheck:({f})=>{f.game.time.worldTime=701}},1,0],
 ['lost source after check',{afterCheck:({f})=>{f.actor.items.delete(f.item.id)}},1,0],
 ['cancellation during damage',{afterDamage:({operations,activity})=>operations.cancelActivity(activity)},1,1],
 ['changed patient during damage',{afterDamage:({f})=>{f.target.actor=f.actor}},1,1],
])test(`automatic Three Pecks stops on ${name} without publishing stale treatment`,async()=>{
 const f=await treatment(options);await assert.rejects(f.executor.roll(f.claim,f.scope));
 assert.equal(f.cards.length,expectedCards);assert.equal(f.evaluations.length,expectedEvaluations);assert.equal(f.applications.length,0);
 assert.equal(f.actor.flags[f.M].salubriousKissExecutions[0].state,'uncertain');
 await assert.rejects(f.executor.roll(f.claim,f.scope));assert.equal(f.checks.length,1);
});

test('the real Three Pecks subscriber passes its original opaque activity binding to the executor',async()=>{
 const f=fixture(),ctx={validate(){}},seen=[];f.game.scenes.active=f.scene;
 const kiss=createSalubriousKiss({...f,isExplorationContext:context=>context===ctx,validateRefocusNote:()=>true,choose:()=>assert.fail('bound patient and DC must not reopen a choice'),executor:{roll:async(claim,scope)=>{seen.push(scope);return {checkId:'C',damageId:'D',degree:2}},apply:async()=>({messageId:'R'})}});
 const activity={id:'auto1',providerId:'refocus',actorUUID:f.actor.uuid,patientUUIDs:[f.patient.uuid],startedAt:100,endsAt:700,source:{type:'coordinator'},options:{threePecks:true,rank:'trained'}};
 await kiss.claimActivity(activity,ctx);f.game.time.worldTime=700;f.proof.nonce=activity.id;f.proof.userId=f.gm.id;
 f.proof.privacy={schema:1,userId:f.gm.id,mode:'public',whisper:[],blind:false};f.proof.noteId='F';
 f.game.messages.set('F',{id:'F',author:f.gm,speaker:{actor:f.actor.id,scene:f.scene.id,token:f.token.id},blind:false,whisper:[],flags:{[f.M]:{avRefocusNote:{...f.proof,kind:'completion'}}}});
 f.actor.flags[f.M].avRefocusIntent=structuredClone(f.proof);f.actor.flags[f.M].refocusEvents=[{...f.proof,state:'claimed'}];
 await kiss.onRefocus({actor:f.actor,user:f.gm,proof:f.proof});
 assert.equal(seen.length,1);assert.equal(seen[0]?.ctx,ctx);assert.deepEqual(seen[0]?.activity,activity);
 assert.equal((await kiss.completeActivity(activity,ctx)).status,'confirmed');
});

async function subscriber(){
 const f=fixture();f.game.scenes.active=f.scene;
 const activity={id:'auto1',providerId:'refocus',actorUUID:f.actor.uuid,patientUUIDs:[f.patient.uuid],startedAt:100,endsAt:700,state:'planned',source:{type:'coordinator'},options:{threePecks:true,rank:'trained'}};
 const operations=createExplorationOwnerOperations({game:f.game,fromUuid:f.fromUuid,ledger:{getActivity:async()=>structuredClone(activity)}}),ctx=await operations.createActivityContext(activity),scopes=[];
 const executor={roll:async()=>({checkId:'C',damageId:'D',degree:2}),apply:async(_claim,_result,scope)=>{scopes.push(scope);return {messageId:'R'}}};
 const kiss=createSalubriousKiss({...f,executor,isExplorationContext:operations.isActivityContext,validateRefocusNote:()=>true});
 await kiss.claimActivity(activity,ctx);f.game.time.worldTime=700;f.proof.nonce=activity.id;f.proof.userId=f.gm.id;
 f.proof.privacy={schema:1,userId:f.gm.id,mode:'public',whisper:[],blind:false};f.proof.noteId='F';
 f.game.messages.set('F',{id:'F',author:f.gm,speaker:{actor:f.actor.id,scene:f.scene.id,token:f.token.id},blind:false,whisper:[],flags:{[f.M]:{avRefocusNote:{...f.proof,kind:'completion'}}}});
 f.actor.flags[f.M].avRefocusIntent=structuredClone(f.proof);f.actor.flags[f.M].refocusEvents=[{...f.proof,state:'claimed'}];
 return {...f,activity,ctx,operations,executor,kiss,scopes,run:()=>kiss.onRefocus({actor:f.actor,user:f.gm,proof:f.proof})};
}

test('cancellation while saving the choosing claim releases it as a known pre-roll refusal',async()=>{
 const f=await subscriber(),update=f.actor.update.bind(f.actor);let rolls=0;
 f.executor.roll=async()=>{rolls++;assert.fail('cancelled choice must not enter native dice')};
 f.actor.update=async changes=>{const result=await update(changes);if(changes[`flags.${f.M}.salubriousKiss.claims`]?.some(c=>c.state==='choosing'))f.operations.cancelActivity(f.activity);return result};
 const claim=await f.run();assert.equal(claim.state,'declined');assert.equal(f.actor.flags[f.M].salubriousKiss.claims[0].state,'declined');assert.equal(rolls,0);assert.equal(f.effects.length,0);assert.equal(f.patient.flags[f.M]?.salubriousKiss?.pending,undefined);
});

for(const stage of ['applying save','immunity source','immunity creation','completed native application'])test(`automatic subscriber stops after cancellation during ${stage} and preserves completed evidence`,async()=>{
 const f=await subscriber();
 if(stage==='applying save'){
  const update=f.actor.update.bind(f.actor);f.actor.update=async changes=>{const result=await update(changes);if(changes[`flags.${f.M}.salubriousKiss.claims`]?.some(c=>c.state==='applying'))f.operations.cancelActivity(f.activity);return result};
 }else if(stage==='immunity source'){
  const read=f.pack.toObject.bind(f.pack);f.pack.toObject=()=>{f.operations.cancelActivity(f.activity);return read()};
 }else if(stage==='immunity creation'){
  const create=f.patient.createEmbeddedDocuments.bind(f.patient);f.patient.createEmbeddedDocuments=async(...args)=>{const result=await create(...args);f.operations.cancelActivity(f.activity);return result};
 }else{
  const apply=f.executor.apply;f.executor.apply=async(...args)=>{const result=await apply(...args);f.operations.cancelActivity(f.activity);return result};
 }
 await assert.rejects(f.run());
 const claim=f.actor.flags[f.M].salubriousKiss.claims[0];assert.equal(claim.state,'uncertain');assert.deepEqual(claim.result,{checkId:'C',damageId:'D',degree:2});
 assert.equal(f.effects.length,['immunity creation','completed native application'].includes(stage)?1:0);
 assert.equal(f.scopes.length,stage==='completed native application'?1:0);assert.equal(f.patient.removed,undefined);assert.equal(f.patient.flags[f.M].salubriousKiss.pending.nonce,f.activity.id);
 if(stage==='immunity creation')assert.deepEqual(claim.immunityIds,[f.effects[0].uuid]);
 if(stage==='completed native application'){assert.equal(claim.receipt.messageId,'R');assert.equal(f.scopes[0]?.ctx,f.ctx)}
 await assert.rejects(f.run());assert.equal(f.scopes.length,stage==='completed native application'?1:0);
});

async function application(options={}){
 const f=await treatment(options),result=await f.executor.roll(f.claim,f.scope);f.claim.state='applying';f.claim.result=result;f.setClaim(f.claim);
 f.patient.flags[f.M]={salubriousKiss:{pending:{actorUuid:f.actor.uuid,nonce:f.claim.nonce}}};
 f.patient.applyDamage=async params=>{
  f.applications.push(params);
  const receipt={id:'receipt',author:f.gm,speaker:{actor:f.patient.id,scene:f.scene.id,token:f.target.id},blind:false,whisper:[],flags:{pf2e:{context:{type:'damage-taken',options:[...params.rollOptions],messageMode:'public'},appliedDamage:{uuid:f.patient.uuid}},[f.M]:{salubriousKiss:{kind:'receipt',nonce:f.claim.nonce,damageId:result.damageId,targetUuid:f.target.uuid,privacy:f.claim.privacy}}}};
  f.game.messages.set(receipt.id,receipt);f.Hooks.fire('createChatMessage',receipt);return f.patient;
 };
 return {...f,result};
}

for(const stage of ['application save','authorization','HP pool callback'])test(`automatic executor rejects cancellation during ${stage} before native HP application`,async()=>{
 let f,authorized=0,revoked=0,scope;
 const authorizeDamage=async request=>{authorized++;scope=request.explorationScope;if(stage==='authorization')f.operations.cancelActivity(f.scope.activity);return()=>{revoked++}};
 const hpPools=stage==='HP pool callback'?{withNativeApplication:async(_activity,_patient,operation)=>{f.operations.cancelActivity(f.scope.activity);return operation()}}:undefined;
 f=await application({authorizeDamage,hpPools});
 if(stage==='application save'){
  const update=f.actor.update.bind(f.actor);f.actor.update=async changes=>{const result=await update(changes);if(changes[`flags.${f.M}.salubriousKiss.claims`]?.some(c=>c.application))f.operations.cancelActivity(f.scope.activity);return result};
 }
 await assert.rejects(f.executor.apply(f.claim,f.result,f.scope));assert.equal(f.applications.length,0);assert.equal(authorized,stage==='application save'?0:1);assert.equal(revoked,authorized);
 if(authorized)assert.equal(scope?.ctx,f.scope.ctx);
 await assert.rejects(f.executor.apply(f.claim,f.result,f.scope));assert.equal(f.applications.length,0);
});

test('ordinary native application still works without an exploration scope',async()=>{
 const f=await application({authorizeDamage:async()=>()=>{}}),receipt=await f.executor.apply(f.claim,f.result);
 assert.equal(receipt.messageId,'receipt');assert.equal(f.applications.length,1);
});
