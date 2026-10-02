import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createNativeTreatment} from '../../scripts/exploration/native-treatment.mjs';
const dice=(formula,total)=>({_evaluated:true,total,toJSON:()=>({formula,total,evaluated:true})});
function fixture({risky=false,actualRisky=false,rank='expert',outcome='success',onNativeBoundary,onCheck}={}){
  const hooks=new Map();let serial=0;
  const Hooks={on:(name,fn)=>{hooks.set(++serial,{name,fn});return serial},off:(_name,id)=>hooks.delete(id)};
  const fire=(name,...args)=>{for(const row of [...hooks.values()])if(row.name===name)row.fn(...args)};
  const messages=new Map(),healer={id:'H',uuid:'Actor.H',items:[],getStatistic:()=>({rank:2})};
  const feat=(slug,extra={})=>{const item={type:'feat',slug,suppressed:false,system:{slug},flags:{},...extra};healer.items.push(item);return item};
  const riskyFeat=feat('risky-surgery'),medic=feat('medic-dedication'),continual=feat('continual-recovery'),ward=feat('ward-medic');
  const assurance=feat('assurance',{flags:{pf2e:{rulesSelections:{assurance:'medicine'}}}});
  let checks=0,cuts=0,applications=0,scopeInput;const effects=[],stages=[];
  const patient={id:'P',uuid:'Actor.P',getSelfRollOptions:()=>[],getContextualClone:(_options,received)=>{effects.push(...received);return {...patient}},
    createEmbeddedDocuments:async()=>[{uuid:'Actor.P.Item.Immunity'}]};
  const persist=(id,roll,checkId,marker)=>{const message={id,actor:healer,author:{id:'G'},rolls:[roll],
    flags:{pf2e:{origin:{messageId:checkId},context:{options:[marker],outcome}}}};
    fire('preCreateChatMessage',message,message);messages.set(id,message);fire('createChatMessage',message);return message;};
  const game={user:{id:'G'},time:{worldTime:600},messages,pf2e:{actions:{get:()=>({use:async options=>{
    onNativeBoundary?.({healer,scopeInput});scopeInput.assertQualification?.();checks++;
    const marker=options.rollOptions[0],check={id:'C',actor:healer,author:{id:'G'},rolls:[{total:22}],
      flags:{pf2e:{context:{action:'treat-wounds',options:[marker],outcome},modifiers:[{slug:'risky-surgery',enabled:actualRisky}]}}};
    messages.set(check.id,check);if(actualRisky)persist('Cut',dice('{1d8[slashing]}',3),check.id,marker);
    if(outcome!=='failure')persist('Heal',dice('{(2d8+15)[healing]}',24),check.id,marker);
    onCheck?.({healer,medic,riskyFeat});return [{actor:healer,message:check,outcome,roll:check.rolls[0]}];
  }})}}};
  game.pf2e.DamageRoll=class{constructor(formula){this.formula=formula}async evaluate(){cuts++;Object.assign(this,dice(this.formula,3));return this}};
  const activity={id:'A',actorUUID:healer.uuid,patientUUIDs:[patient.uuid],startedAt:0,endsAt:600,options:{skill:'medicine',rank,riskySurgery:risky}};
  const native=createNativeTreatment({game,Hooks,fromUuid:async uuid=>uuid.startsWith('Compendium.')?{toObject:()=>({system:{}})}:uuid===healer.uuid?healer:patient,
    checkScope:{runExploration:async(input,op)=>{scopeInput=input;return op()}},ownerOperations:{isActivityContext:c=>c===ctx},
    damageGuard:{authorizeExploration:async()=>()=>{}},hpPools:{withNativeApplication:async(_a,_p,op)=>({result:await op(),poolReceipt:{noChange:true}})},
    createMessage:async data=>persist('CompatCut',dice('{1d8[slashing]}',3),data.flags.pf2e.origin.messageId,data.flags.pf2e.context.options.find(o=>o.startsWith('exploration-activity:'))),
    apply:async request=>{applications++;stages.push(request.stage);const message={id:'R'+applications,author:{id:'G'},speaker:{actor:patient.id},
      flags:{pf2e:{context:{type:'damage-taken',options:[request.source,request.application]},appliedDamage:null}}};
      fire('preCreateChatMessage',message,message);messages.set(message.id,message);fire('createChatMessage',message);return patient},timeoutMs:100});
  const ctx={validate(){}};
  return {native,activity,ctx,healer,feat,riskyFeat,medic,continual,ward,assurance,game,effects,stages,
    get checks(){return checks},get cuts(){return cuts},get applications(){return applications},get scopeInput(){return scopeInput}};
}
for(const change of ['suppressed','isSuppressed','system.suppressed','missing'])test(`declared Risky ${change} rejects before the original check and compatibility cut`,async()=>{
  const f=fixture({risky:true});if(change==='missing')f.healer.items=f.healer.items.filter(item=>item!==f.riskyFeat);
  else if(change==='system.suppressed')f.riskyFeat.system.suppressed=true;else f.riskyFeat[change]=true;
  await assert.rejects(f.native.run(f.activity,f.ctx),/risky-surgery-feat-unavailable/);
  assert.equal(f.checks,0);assert.equal(f.cuts,0);assert.equal(f.applications,0);
});
test('an active Risky feat with disabled competing modifier keeps exactly one compatibility cut',async()=>{
  const f=fixture({risky:true,actualRisky:false});const result=await f.native.run(f.activity,f.ctx);
  assert.equal(result.status,'confirmed');assert.equal(f.checks,1);assert.equal(f.cuts,1);
  assert.deepEqual(f.stages,['surgery','healing']);assert.deepEqual(result.proof.resultIds,['Heal','CompatCut']);
});
test('an enabled native Risky cut is retained without a second compatibility roll',async()=>{
  const f=fixture({risky:true,actualRisky:true});assert.equal((await f.native.run(f.activity,f.ctx)).status,'confirmed');
  assert.equal(f.checks,1);assert.equal(f.cuts,0);assert.deepEqual(f.stages,['surgery','healing']);
});
test('current active Medic supplies the baked typed adjustment once for ordinary TW',async()=>{
  const f=fixture();const result=await f.native.run(f.activity,f.ctx);assert.equal(result.medicBonus,5);
  assert.deepEqual(f.effects.flatMap(effect=>effect.system.rules??[]).map(rule=>[rule.type,rule.value]),[['untyped',-5],['circumstance',5]]);
  assert.equal(f.checks,1);assert.equal(f.applications,1);
});
for(const change of ['suppressed','isSuppressed','system.suppressed'])test(`inactive Medic ${change} at a nonzero native baked rank is rejected before dice`,async()=>{
  const f=fixture();if(change==='system.suppressed')f.medic.system.suppressed=true;else f.medic[change]=true;
  await assert.rejects(f.native.run(f.activity,f.ctx),/native-suppressed-medic-bonus/);
  assert.equal(f.checks,0);assert.equal(f.applications,0);assert.deepEqual(f.effects,[]);
});
for(const change of ['suppressed','isSuppressed','system.suppressed'])test(`inactive Magic Hands ${change} cannot enter the native hard-coded healing formula`,async()=>{
  const f=fixture(),magic=f.feat('magic-hands');if(change==='system.suppressed')magic.system.suppressed=true;else magic[change]=true;
  await assert.rejects(f.native.run(f.activity,f.ctx),/native-suppressed-magic-hands/);
  assert.equal(f.checks,0);assert.equal(f.cuts,0);assert.equal(f.applications,0);
});
test('Magic Hands suppressed at the private native boundary is rejected before dice',async()=>{
  const f=fixture({onNativeBoundary:({healer})=>{healer.items.find(item=>item.slug==='magic-hands').suppressed=true}});f.feat('magic-hands');
  await assert.rejects(f.native.run(f.activity,f.ctx),/native-suppressed-magic-hands/);assert.equal(f.checks,0);assert.equal(f.applications,0);
});

test('trained zero-bonus TW does not manufacture a typed adjustment for suppressed Medic',async()=>{
  const f=fixture({rank:'trained'});f.medic.suppressed=true;
  assert.equal((await f.native.run(f.activity,f.ctx)).medicBonus,0);assert.deepEqual(f.effects,[]);assert.equal(f.checks,1);
});
test('Medic lost after the saved check retains original proof and prevents application',async()=>{
  const f=fixture({onCheck:({medic})=>{medic.suppressed=true}});let error;
  await assert.rejects(f.native.run(f.activity,f.ctx),caught=>{error=caught;return /native-suppressed-medic-bonus/.test(caught.message)});
  assert.deepEqual(error.proof.checkIds,['C']);assert.deepEqual(error.proof.resultIds,['Heal']);assert.equal(f.applications,0);
  assert.equal((await f.native.reconcile({...f.activity,proof:error.proof})).status,'uncertain');assert.equal(f.checks,1);
});
test('Medic removed after the saved check cannot hide its original baked bonus',async()=>{
  const f=fixture({onCheck:({healer,medic})=>{healer.items=healer.items.filter(item=>item!==medic)}});
  await assert.rejects(f.native.run(f.activity,f.ctx),error=>{
    assert.match(error.message,/native-medic-qualification-changed/);assert.deepEqual(error.proof.checkIds,['C']);return true;
  });assert.equal(f.checks,1);assert.equal(f.applications,0);
});
test('Medic gained after the saved check cannot manufacture a baked adjustment',async()=>{
  const f=fixture({onCheck:({healer,medic})=>{healer.items.push(medic)}});f.healer.items=f.healer.items.filter(item=>item!==f.medic);
  await assert.rejects(f.native.run(f.activity,f.ctx),/native-medic-qualification-changed/);assert.equal(f.checks,1);assert.equal(f.applications,0);
});
test('private source callback rejects Risky lost at the actual native boundary',async()=>{
  const f=fixture({risky:true,onNativeBoundary:({healer})=>{healer.items.find(item=>item.slug==='risky-surgery').suppressed=true}});
  await assert.rejects(f.native.run(f.activity,f.ctx),/risky-surgery-feat-unavailable/);
  assert.equal(typeof f.scopeInput.assertQualification,'function');assert.equal(f.checks,0);assert.equal(f.cuts,0);
});
test('declared Continual lost before dice cannot continue shortened immunity',async()=>{
  const f=fixture();f.activity.options.continualRecovery=true;f.continual.suppressed=true;
  await assert.rejects(f.native.run(f.activity,f.ctx),/continual-recovery-feat-unavailable/);assert.equal(f.checks,0);
});
test('an undeclared later Continual feat does not shorten frozen normal immunity',async()=>{
  const f=fixture();const result=await f.native.run(f.activity,f.ctx);assert.equal(result.expiresAt,3600);
});
for(const change of ['feat','rank'])test(`declared Ward multi-patient ${change} loss is rejected before dice`,async()=>{
  const f=fixture();f.activity.patientUUIDs.push('Actor.Q');if(change==='feat')f.ward.suppressed=true;else f.healer.getStatistic=()=>({rank:1});
  await assert.rejects(f.native.run(f.activity,f.ctx),/ward-capacity-unqualified/);assert.equal(f.checks,0);
});
for(const change of ['suppressed','wrong skill'])test(`selected Assurance ${change} is rejected before native substitution`,async()=>{
  const f=fixture();f.activity.options.assurance=true;if(change==='suppressed')f.assurance.suppressed=true;else f.assurance.flags.pf2e.rulesSelections.assurance='occultism';
  await assert.rejects(f.native.run(f.activity,f.ctx),/assurance-feat-unavailable/);assert.equal(f.checks,0);
});
