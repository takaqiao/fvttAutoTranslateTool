import {test} from 'node:test';
import assert from 'node:assert/strict';
import {runCheckReactionPipeline} from '../scripts/reaction-checks.mjs';

function fixture(createMessage=false){
 const calls=[],publications=[],callbacks=[],drafts=[];
 const check={slug:'arcana',modifiers:[{slug:'native-bonus',modifier:7}],totalModifier:7};
 const makeRoll=(total,degree)=>({total,options:{degreeOfSuccess:degree},toJSON(){return {total:this.total,options:{...this.options}}}});
 const original=makeRoll(12,1),replacement=makeRoll(8,0);
 const game={pf2e:{Modifier:class{constructor(){assert.fail('Halfling Luck adds no modifier')}},CheckModifier:class{constructor(){assert.fail('Reuse the exact prepared check')}},Check:{renderReroll:async(r,{isOld})=>`${isOld?'old':'new'}:${r.total}`}}};
 const args={game,check,context:{type:'skill-check',actor:{uuid:'Actor.roller'},options:new Set(['action:recall-knowledge']),createMessage},
  native:async(used,context,event,callback)=>{calls.push({check:used,context,event});const roll=calls.length===1?original:replacement,outcome=roll.options.degreeOfSuccess===1?'failure':'criticalFailure';
   const data={speaker:{actor:'roller'},author:'owner',flags:{pf2e:{context:{type:'skill-check',outcome,unadjustedOutcome:outcome,options:[...context.options],isReroll:!!context.isReroll,substitutions:[],rollTwice:false}}},rolls:[roll.toJSON()]};
   const draft={toObject:()=>structuredClone(data),updateSource:update=>Object.assign(data,structuredClone(update)),get flags(){return data.flags}};drafts.push(draft);await callback(roll,outcome,draft,event);return roll;},
  decide:async()=>({reaction:'halfling-luck',nonce:'paid-invocation',actorUuid:'Actor.roller'}),
  publish:async data=>{publications.push(data);return {id:'published',...data}},
  callback:async(...values)=>callbacks.push(values),
 };
 return {args,check,calls,publications,callbacks,drafts,original,replacement,run:()=>runCheckReactionPipeline(args)};
}

test('paid Halfling Luck keeps a worse native roll with the original check and no Clock bonus',async()=>{
 for(const createMessage of [false,true]){
  const f=fixture(createMessage),result=await f.run();
  assert.equal(result,f.replacement);assert.equal(f.calls.length,2);assert.equal(f.calls[1].check,f.check);
  assert.equal(f.calls[1].context.isReroll,true);assert.equal(f.calls[1].context.skipDialog,true);
  assert.equal(f.calls[1].context.options.has('fortune'),true);assert.equal(f.calls[1].context.options.has('check:reroll'),true);
  assert.equal(f.publications.length,createMessage?1:0);assert.equal(f.callbacks.length,1);assert.equal(f.callbacks[0][0],f.replacement);assert.equal(f.callbacks[0][1],'criticalFailure');
  assert.equal(f.args.context.outcome,'criticalFailure');assert.equal(f.args.context.isReroll,true);
  if(!createMessage)assert.equal(f.callbacks[0][2],f.drafts[1]);
 }
});

test('paid Halfling Luck native cancellation or throw never publishes or returns the old failure',async()=>{
 for(const throws of [false,true]){
  const f=fixture(),native=f.args.native;let attempts=0;
  f.args.native=async(...args)=>{if(++attempts===2){if(throws)throw Error('native disconnected');return null}return native(...args)};
  await assert.rejects(f.run(),throws?/native disconnected/:/半身人幸运.*重掷未完成/);
  assert.equal(attempts,2);assert.equal(f.publications.length,0);assert.equal(f.callbacks.length,0);
 }
});

test('Halfling Luck does not invoke Clock-specific disruption and retains its own paid nonce',async()=>{
 const f=fixture(true);f.args.beforeReroll=()=>assert.fail('Clock-only disruption is not a Halfling Luck receipt');
 await f.run();const result=f.publications[0];
 assert.equal(result.flags['pf2e-third-party-automation'].reactionChecks.reaction,'halfling-luck');
 assert.equal(result.flags['pf2e-third-party-automation'].reactionChecks.nonce,'paid-invocation');
 assert.match(result.content,/old:12/);assert.match(result.content,/new:8/);
});
