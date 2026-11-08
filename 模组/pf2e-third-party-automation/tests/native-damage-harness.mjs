import assert from 'node:assert/strict';
import vm from 'node:vm';

/** Runs applyDamage up to calculateHealthDelta and reports the delta it was
 * about to apply, with every call the IWR callback received. */
export function healthDelta(methodSource,prevent){
 const calls=[],stop=Error('health-delta-captured');
 const context=vm.createContext({game:{modules:new Map([['pf2e-third-party-automation',{api:{async nativeDamageIWR(...args){calls.push(args);return prevent}}}]])},extractNotes:()=>[],applyStackingRules:()=>0,_loc:value=>value});
 const applyDamage=vm.runInContext('({'+methodSource+'}).applyDamage',context);
 return async({damage,hardness,block})=>{
  let delta;
  const params={damage,token:{name:'Target'},shieldBlockRequest:block,rollOptions:new Set(['target:fixture'])};
  const actor={hitPoints:{value:30,max:30,temp:0},heldShield:{id:'shield',_source:{system:{hp:{value:20}}}},hardness,synthetics:{rollNotes:{}},
   attributes:{hp:{},shield:{itemId:'shield',hardness:5,raised:true,hp:{value:20}}},isOfType:type=>type==='character',
   calculateHealthDelta(value){delta=value.delta;throw stop}};
  await assert.rejects(applyDamage.call(actor,params),error=>error===stop);
  return {delta,calls,actor,params};
 };
}
// The shield has hardness 5. In the middle two rows the actor's hardness exceeds
// what is left of the damage, so the applied hardness is smaller than the
// hardness value and the two must not be interchanged.
export const DAMAGE_CASES=[
 ['a blocked hit above both hardness values',{damage:15,hardness:3,block:true},7,10],
 ['hardness 5 against 3 damage left after a block',{damage:8,hardness:5,block:true},0,3],
 ['hardness 5 against an unblocked hit of 3',{damage:3,hardness:5,block:false},0,0],
 ['an unblocked hit above hardness',{damage:20,hardness:5,block:false},15,0],
];
/** The patched method calls back once with the native amounts and, when the
 * callback prevents the damage, applies no health delta. */
export async function assertCallback(originalMethod,patchedMethod,[,input,nativeDelta,shieldDamage],prevent){
 const native=await healthDelta(originalMethod)(input);
 assert.equal(native.delta,nativeDelta);assert.equal(native.calls.length,0);
 const patched=await healthDelta(patchedMethod,prevent)(input),[actor,params,result,rollOptions,amounts]=patched.calls[0];
 assert.equal(patched.calls.length,1);assert.equal(actor,patched.actor);assert.equal(params,patched.params);
 assert.equal(result.finalDamage,input.damage);assert.equal(rollOptions,patched.params.rollOptions);
 assert.deepEqual({...amounts},{actorDamage:native.delta,shieldDamage});
 assert.equal(patched.delta,prevent?0:native.delta);
}
