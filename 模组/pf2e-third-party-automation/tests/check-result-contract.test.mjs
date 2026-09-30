import {test} from 'node:test';
import assert from 'node:assert/strict';
import {runCheckReactionPipeline} from '../scripts/reaction-checks.mjs';

const ID='pf2e-third-party-automation';
const clone=value=>structuredClone(value);
class Draft {
 constructor(data){this.data=clone(data);this.id=data._id??null;}
 toObject(){return clone(this.data);}
 updateSource(data){this.data={...this.data,...clone(data)};}
 get flags(){return this.data.flags;}
 get rolls(){return this.data.rolls;}
}
function fixture({createMessage=false,reaction=null}={}){
 const event={type:'original-event'},rerollEvent={type:'reroll-event'},calls=[],published=[],callbacks=[];
 const context={actor:{uuid:'Actor.roller'},type:'saving-throw',options:new Set(['secret']),...(createMessage===undefined?{}:{createMessage})};
 const roll=(total,degree)=>({total,options:{degreeOfSuccess:degree},toJSON(){return {total:this.total,options:clone(this.options)}},async render(){return `roll:${this.total}`}});
 const original=roll(8,0),replacement=roll(5,0),drafts=[];
 const makeDraft=(r,reroll)=>new Draft({speaker:{actor:'roller',scene:'s',token:'t'},author:'owner',blind:true,whisper:['gm'],flavor:'native private flavor',rolls:[r.toJSON()],flags:{pf2e:{context:{type:'saving-throw',outcome:'criticalFailure',unadjustedOutcome:'criticalFailure',options:reroll?['secret','fortune','check:reroll']:['secret'],messageMode:'blind',isReroll:reroll,rollTwice:false,substitutions:[]}},other:{preserve:true}}});
 const game={pf2e:{Modifier:class{constructor(data){Object.assign(this,data)}},CheckModifier:class{constructor(slug,base,extra){this.slug=slug;this.modifiers=[...base.modifiers,...extra]}},Check:{renderReroll:async(r,{isOld})=>`${isOld?'old':'new'}:${r.total}`}}};
 const args={game,check:{slug:'will',modifiers:[{slug:'native',modifier:6}]},context,event,
  native:async(check,ctx,_event,callback)=>{const reroll=calls.length>0,r=reroll?replacement:original;calls.push({check,context:ctx,event:_event});assert.equal(ctx.createMessage,false);const draft=makeDraft(r,reroll);drafts.push(draft);await callback(r,'criticalFailure',draft,reroll?rerollEvent:event);return r;},
  decide:async()=>reaction?{reaction,nonce:'use-nonce',actorUuid:'Actor.roller'}:null,
  publish:async data=>{published.push(clone(data));return new Draft({...data,_id:'final-message'})},
  callback:async(...args)=>{callbacks.push(args)},
 };
 return {args,context,event,rerollEvent,calls,published,callbacks,original,replacement,drafts,run:()=>runCheckReactionPipeline(args)};
}

test('createMessage:false returns the native draft to one callback without publishing, preserving private data and event',async()=>{
 const f=fixture(),result=await f.run();
 assert.equal(f.published.length,0);assert.equal(f.calls.length,1);assert.equal(f.callbacks.length,1);
 const [roll,outcome,message,event]=f.callbacks[0];assert.equal(result,f.original);assert.equal(roll,result);assert.equal(outcome,'criticalFailure');assert.equal(message,f.drafts[0]);assert.ok(message instanceof Draft);assert.equal(message.id,null);assert.equal(event,f.event);
 assert.deepEqual(message.data.speaker,{actor:'roller',scene:'s',token:'t'});assert.deepEqual(message.rolls,[result.toJSON()]);assert.deepEqual(message.data.whisper,['gm']);assert.equal(message.data.blind,true);assert.equal(message.data.author,'owner');assert.equal(message.flags.pf2e.context.messageMode,'blind');assert.deepEqual(message.flags.other,{preserve:true});assert.equal(f.context.outcome,outcome);assert.equal(f.context.createMessage,false);
});

test('true and omitted createMessage retain one final publication with the native visibility',async()=>{
 for(const requested of [true,undefined]){
  const f=fixture({createMessage:requested});if(requested===undefined)delete f.context.createMessage;
  const result=await f.run();assert.equal(f.calls.length,1);assert.equal(f.published.length,1);assert.equal(f.callbacks.length,1);assert.equal(f.callbacks[0][2].id,'final-message');assert.notEqual(f.callbacks[0][2],f.drafts[0]);assert.equal(f.callbacks[0][0],result);assert.equal(f.callbacks[0][3],f.event);
  assert.deepEqual(f.published[0].whisper,['gm']);assert.equal(f.published[0].blind,true);assert.equal(f.published[0].flags.pf2e.context.messageMode,'blind');assert.deepEqual(f.published[0].speaker,{actor:'roller',scene:'s',token:'t'});
 }
});

test('draft Clock keeps its native +1 and the worse new roll, delivering only the final draft/outcome/event',async()=>{
 const f=fixture({reaction:'clock'}),result=await f.run();
 assert.equal(f.published.length,0);assert.equal(f.calls.length,2);assert.equal(f.callbacks.length,1);assert.equal(result,f.replacement);
 const [roll,outcome,draft,event]=f.callbacks[0];assert.equal(roll,f.replacement);assert.equal(outcome,'criticalFailure');assert.equal(draft,f.drafts[1]);assert.equal(event,f.rerollEvent);assert.equal(draft.id,null);
 assert.deepEqual(draft.rolls,[f.replacement.toJSON()]);assert.match(draft.data.content,/old:8/);assert.match(draft.data.content,/new:5/);assert.equal(draft.flags[ID].reactionChecks.reaction,'clock');assert.deepEqual(draft.flags[ID].reactionChecks.previousRoll,f.original.toJSON());assert.equal(f.context.isReroll,true);
 assert.deepEqual(f.calls[1].check.modifiers.map(m=>[m.slug,m.modifier]),[['native',6],['turn-back-the-clock',1]]);assert.equal(f.calls[1].context.skipDialog,true);assert.equal(f.calls[1].context.isReroll,true);assert.equal(f.calls[1].context.rollTwice,false);assert.deepEqual(f.calls[1].context.substitutions,[]);assert.equal(f.calls[1].context.options.has('fortune'),true);
});

test('draft Squawk updates the native draft roll and adjusted outcome before callback',async()=>{
 const f=fixture({reaction:'squawk'}),result=await f.run();
 assert.equal(f.published.length,0);assert.equal(f.calls.length,1);assert.equal(f.callbacks.length,1);assert.equal(result.options.degreeOfSuccess,1);assert.equal(f.callbacks[0][1],'failure');assert.equal(f.callbacks[0][2],f.drafts[0]);assert.equal(f.drafts[0].flags.pf2e.context.outcome,'failure');assert.equal(f.drafts[0].rolls[0].options.degreeOfSuccess,1);assert.equal(f.context.outcome,'failure');assert.equal(f.context.unadjustedOutcome,'criticalFailure');
});

test('a disrupted draft Clock preserves the original roll and proof without a second native roll',async()=>{
 const f=fixture({reaction:'clock'});f.args.beforeReroll=async()=>({disrupted:true,nonce:'eat-nonce'});
 assert.equal(await f.run(),f.original);assert.equal(f.calls.length,1);assert.equal(f.published.length,0);assert.equal(f.callbacks.length,1);assert.equal(f.callbacks[0][2],f.drafts[0]);assert.equal(f.drafts[0].flags[ID].reactionChecks.disrupted,true);assert.equal(f.drafts[0].flags.pf2e.context.eatFortune.nonce,'eat-nonce');assert.equal(f.context.options.has('misfortune'),true);
});

test('the original caller publication request is frozen before awaiting native work or a decision',async()=>{
 for(const requested of [false,true]){
  const f=fixture({createMessage:requested});f.args.decide=async()=>{f.context.createMessage=!requested;return null};
  await f.run();assert.equal(f.published.length,requested?1:0);assert.equal(f.callbacks[0][2].id,requested?'final-message':null);
 }
});

test('native cancellation returns unchanged without a decision, publication, or callback',async()=>{
 const f=fixture();let decisions=0,natives=0;f.args.native=async()=>{natives++;return null};f.args.decide=async()=>{decisions++};
 assert.equal(await f.run(),null);assert.equal(natives,1);assert.equal(decisions,0);assert.equal(f.callbacks.length,0);assert.equal(f.published.length,0);
});

test('native and decision errors are propagated without reroll, publication, or callback retries',async()=>{
 for(const phase of ['native-before-callback','native-after-callback','decision']){
  const f=fixture(),native=f.args.native;let calls=0;
  f.args.native=async(...args)=>{calls++;if(phase==='native-before-callback')throw Error(phase);const result=await native(...args);if(phase==='native-after-callback')throw Error(phase);return result};
  if(phase==='decision')f.args.decide=async()=>{throw Error(phase)};
  await assert.rejects(f.run(),new RegExp(phase));assert.equal(calls,1);assert.equal(f.callbacks.length,0);assert.equal(f.published.length,0);
 }
});

test('paid Clock native cancellation or failure does not reroll again or deliver the old result',async()=>{
 for(const throws of [false,true]){
  const f=fixture({reaction:'clock'}),native=f.args.native;let calls=0;
  f.args.native=async(...args)=>{if(++calls===2){if(throws)throw Error('reroll disconnected');return null}return native(...args)};
  await assert.rejects(f.run(),throws?/reroll disconnected/:/重掷未完成/);assert.equal(calls,2);assert.equal(f.published.length,0);assert.equal(f.callbacks.length,0);
 }
});

test('publication cancellation/failure and draft update failure do not invoke callback or retry',async()=>{
 for(const mode of ['cancel-publish','throw-publish','draft-update']){
  const f=fixture({createMessage:mode!=='draft-update'});let attempts=0;
  if(mode==='draft-update'){const native=f.args.native;f.args.native=async(...args)=>{const result=await native(...args);f.drafts[0].updateSource=()=>{attempts++;throw Error('draft validation')};return result};f.args.publish=async()=>{assert.fail('draft must never publish')}}
  else f.args.publish=async()=>{attempts++;if(mode==='throw-publish')throw Error('publish failed');return null};
  await assert.rejects(f.run(),mode==='draft-update'?/draft validation/:mode==='throw-publish'?/publish failed/:/未能发布/);assert.equal(attempts,1);assert.equal(f.calls.length,1);assert.equal(f.callbacks.length,0);
 }
});

test('the original callback is awaited once, including its failure, without republishing or rerolling',async()=>{
 for(const createMessage of [false,true]){
  const f=fixture({createMessage});let release,started,settled=false,calls=0;
  const entered=new Promise(resolve=>{started=resolve}),gate=new Promise((_resolve,reject)=>{release=reject});
  f.args.callback=async()=>{calls++;started();await gate};const task=f.run().finally(()=>{settled=true});await entered;assert.equal(settled,false);release(Error('consumer failed'));
  await assert.rejects(task,/consumer failed/);assert.equal(calls,1);assert.equal(f.calls.length,1);assert.equal(f.published.length,createMessage?1:0);
 }
});
