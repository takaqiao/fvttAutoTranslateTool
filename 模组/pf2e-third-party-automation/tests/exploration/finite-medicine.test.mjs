import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeFiniteMedicine,battleMedicineImmunity,battleMedicineOutcome,battleMedicineDuration,medicAvailability}
  from '../../scripts/exploration/finite-medicine.mjs';

const H='Actor.H',P='Actor.P',I='Actor.H.Item.Medic',BM='Compendium.pf2e.feat-effects.Item.2XEYQNZTCGpdkyR6';
const moduleId='pf2e-third-party-automation';
const protocol={version:1,rootUUID:'JournalEntry.ROOT',epoch:'epoch'};
const baseline=(patch={})=>({id:'review-1',actorUUID:H,itemUUID:I,source:'gm-reviewed',nativeCounter:false,remaining:1,
  period:'daily',checkedAt:-10.5,reviewedBy:'G',previousBaselineId:null,renewal:'initial',...patch});
const session=(review=baseline(),patch={})=>({id:'S',protocol,finiteMedicine:{medicBypass:{baselineByActor:{[H]:review}}},...patch});
const root=(review=baseline())=>({sessions:{S:session(review)},activities:{}});
const request=(patch={})=>({actorUUID:H,itemUUID:I,baselineId:'review-1',now:0,...patch});
const claim=(patch={})=>({id:'medic-bypass:A',baselineId:'review-1',actorUUID:H,itemUUID:I,period:'daily',state:'reserved',
  activityId:'A',permitNonce:null,claimedAt:-4.25,checkId:null,usedAt:null,...patch});
const activity=(proof=claim(),patch={})=>({id:'A',sessionId:'S',providerId:'battle-medicine',actorUUID:H,state:'planned',
  proof:{medicBypass:proof},...patch});
const effect=(patch={})=>({uuid:'Actor.P.Item.E',sourceId:BM,flags:{[moduleId]:{healerUuid:H}},
  system:{start:{value:-10},duration:{value:1,unit:'hours'},context:{origin:{actor:H}}},...patch});
const view=(state,remaining=0,claimIds=[])=>({state,remaining,claimIds});

test('omitted finite inputs disable both adapters and never infer a baseline',()=>{
  assert.deepEqual(normalizeFiniteMedicine(undefined,[H]),{version:1,secondsPerUse:6,
    battleMedicine:{enabled:false,maxUsesByActor:{},rankByActor:{}},
    medicBypass:{enabled:false,maxUsesByActor:{},baselineByActor:{}}});
});
test('finite normalization keeps explicit selected budgets and negative fractional review time',()=>{
  const input={secondsPerUse:0.5,battleMedicine:{enabled:true,maxUsesByActor:{[H]:2},rankByActor:{[H]:'expert'}},
    medicBypass:{enabled:true,maxUsesByActor:{[H]:1},baselineByActor:{[H]:baseline()}}};
  const value=normalizeFiniteMedicine(input,[H]);
  assert.equal(value.secondsPerUse,0.5);assert.equal(value.medicBypass.baselineByActor[H].checkedAt,-10.5);
  assert.deepEqual(value.battleMedicine,{enabled:true,maxUsesByActor:{[H]:2},rankByActor:{[H]:'expert'}});
  input.medicBypass.baselineByActor[H].remaining=0;assert.equal(value.medicBypass.baselineByActor[H].remaining,1);
  assert.equal(normalizeFiniteMedicine({secondsPerUse:0},[H]).secondsPerUse,0);
});
test('finite normalization rejects unsafe data without invoking getters or toJSON',()=>{
  let calls=0;
  for(const input of [Object.create({secondsPerUse:6}),{get secondsPerUse(){calls++;return 6}},
    {battleMedicine:{enabled:true,get maxUsesByActor(){calls++;return {}}}},
    {toJSON(){calls++;return {}}},JSON.parse('{"__proto__":{}}'),{medicBypass:{baselineByActor:{[H]:{...baseline(),get remaining(){calls++;return 1}}}}}]){
    assert.throws(()=>normalizeFiniteMedicine(input,[H]),/invalid-finite-medicine/);
  }
  assert.equal(calls,0);
});
test('finite normalization rejects foreign actors, malformed selections, budgets, ranks and seconds',()=>{
  const inputs=[{version:2},{extra:true},{secondsPerUse:-1},{secondsPerUse:Infinity},{secondsPerUse:Number.MAX_SAFE_INTEGER+1},
    {battleMedicine:{enabled:1}},{battleMedicine:{maxUsesByActor:{[H]:0}}},{battleMedicine:{maxUsesByActor:{[H]:1.5}}},
    {battleMedicine:{maxUsesByActor:{[H]:101}}},{battleMedicine:{rankByActor:{[H]:'untrained'}}},
    {battleMedicine:{rankByActor:{'Actor.Other':'trained'}}},{medicBypass:{maxUsesByActor:{[H]:2}}},
    {medicBypass:{baselineByActor:{'Actor.Other':baseline()}}}];
  for(const input of inputs)assert.throws(()=>normalizeFiniteMedicine(input,[H]),/invalid-finite-medicine/);
  for(const actors of [[H,H],[''],Object.assign([H],{extra:true}),Array(1)]){
    assert.throws(()=>normalizeFiniteMedicine(undefined,actors),/invalid-finite-medicine/);
  }
});
test('review shape cannot pretend to be a native counter or bind another actor or period',()=>{
  const bad=[{source:'native'},{nativeCounter:true},{remaining:2},{checkedAt:NaN},{checkedAt:Number.MAX_SAFE_INTEGER+1},
    {reviewedBy:''},{actorUUID:'Actor.Other'},{itemUUID:'Actor.Other.Item.Medic'},{period:'weekly'},
    {previousBaselineId:'review-0',renewal:'initial'},{previousBaselineId:null,renewal:'hour-window'},
    {period:'daily',previousBaselineId:'review-0',renewal:'hour-window'},{renewal:'rested'},{rootUUID:'JournalEntry.ROOT'}];
  for(const patch of bad)assert.throws(()=>normalizeFiniteMedicine({medicBypass:{baselineByActor:{[H]:baseline(patch)}}},[H]),/invalid-finite-medicine/);
  const missing=baseline();delete missing.source;
  assert.throws(()=>normalizeFiniteMedicine({medicBypass:{baselineByActor:{[H]:missing}}},[H]),/invalid-finite-medicine/);
});
test('explicit daily and hourly renewal markers preserve their predecessor',()=>{
  for(const [period,renewal] of [['daily','new-preparation'],['hourly','hour-window']]){
    const review=baseline({period,renewal,previousBaselineId:'review-0'});
    assert.deepEqual(normalizeFiniteMedicine({medicBypass:{baselineByActor:{[H]:review}}},[H]).medicBypass.baselineByActor[H],review);
  }
});
test('explicit null groups cannot become defaults',()=>{
  for(const input of [{battleMedicine:null},{medicBypass:null}])assert.throws(()=>normalizeFiniteMedicine(input,[H]),/invalid-finite-medicine/);
});
test('an explicit undefined group is invalid rather than omitted configuration',()=>{
  for(const input of [{battleMedicine:undefined},{medicBypass:undefined}])assert.throws(()=>normalizeFiniteMedicine(input,[H]),/invalid-finite-medicine/);
});
test('a malformed owned item cannot establish a reviewed baseline',()=>{
  for(const input of [{medicBypass:{baselineByActor:{[H]:baseline({itemUUID:'Actor.H.Item.Medic.Other'})}}},
    {medicBypass:{baselineByActor:{[H]:baseline({itemUUID:'Actor.H.Item.Bad Item'})}}}]){
    assert.throws(()=>normalizeFiniteMedicine(input,[H]),/invalid-finite-medicine/);
  }
});

test('BM immunity belongs to the patient and healer, independently of shared HP',()=>{
  const effects=[effect(),effect({uuid:'Actor.P.Item.TW',sourceId:'Compendium.pf2e.feat-effects.Item.Lb4q2bBAgxamtix5'})];
  assert.deepEqual(battleMedicineImmunity({effects,patientUUID:P,healerUUID:H,now:0}),{status:'immune',expiresAt:3590,effectIds:['Actor.P.Item.E']});
  assert.deepEqual(battleMedicineImmunity({effects,patientUUID:P,healerUUID:'Actor.Other',now:0}),{status:'clear',expiresAt:null,effectIds:[]});
  assert.deepEqual(battleMedicineImmunity({effects:[effect({patientUUID:'Actor.Other'})],patientUUID:P,healerUUID:H,now:0}),{status:'clear',expiresAt:null,effectIds:[]});
});
test('BM immunity retains negative starts and the greatest valid native deadline',()=>{
  const effects=[effect(),effect({uuid:'Actor.P.Item.E2',flags:{},system:{start:{value:-20.5},duration:{value:2,unit:'hours'},context:{origin:{actor:H}}}})];
  assert.deepEqual(battleMedicineImmunity({effects,patientUUID:P,healerUUID:H,now:-4}),{status:'immune',expiresAt:7179.5,effectIds:['Actor.P.Item.E','Actor.P.Item.E2']});
  assert.deepEqual(battleMedicineImmunity({effects:[effect()],patientUUID:P,healerUUID:H,now:3590}),{status:'clear',expiresAt:null,effectIds:[]});
});
test('native origin and legacy untyped exact source can establish immunity',()=>{
  const e=effect({sourceId:undefined,_stats:{compendiumSource:'Compendium.pf2e.feat-effects.2XEYQNZTCGpdkyR6'},flags:{}});
  assert.equal(battleMedicineImmunity({effects:[e],patientUUID:P,healerUUID:H,now:0}).status,'immune');
});
test('uncertain BM provenance or unbounded time never becomes clear',()=>{
  const unknown=[effect({flags:{},system:{start:{value:0},duration:{value:1,unit:'hours'}}}),
    effect({system:{start:{value:0},duration:{value:1,unit:'hours'},context:{origin:{actor:'Actor.Other'}}}}),
    effect({system:{start:{value:0},duration:{value:-1,unit:'hours'},context:{origin:{actor:H}}}}),
    effect({system:{start:{value:Number.MAX_SAFE_INTEGER},duration:{value:1,unit:'hours'},context:{origin:{actor:H}}}}),
    effect({system:{context:{origin:{actor:H}}}})];
  for(const e of unknown)assert.equal(battleMedicineImmunity({effects:[e],patientUUID:P,healerUUID:H,now:0}).status,'uncertain');
  assert.equal(battleMedicineImmunity({effects:[effect()],patientUUID:P,healerUUID:H,now:Infinity}).status,'uncertain');
});
test('conflicting explicit and native patient identities cannot authorize BM',()=>{
  const e=effect({patientUUID:P,parent:{uuid:'Actor.Other'}});
  assert.deepEqual(battleMedicineImmunity({effects:[e],patientUUID:P,healerUUID:H,now:0}),{status:'uncertain',expiresAt:null,effectIds:[]});
  assert.equal(battleMedicineImmunity({effects:[effect({patientUUID:P,parent:{uuid:P}})],patientUUID:P,healerUUID:H,now:0}).status,'immune');
});
test('a present immunity origin item must belong to its healer',()=>{
  for(const item of ['Actor.Other.Item.BM','bad-item','Actor.H.Item.BM.Other']){
    const e=effect({system:{start:{value:0},duration:{value:1,unit:'hours'},context:{origin:{actor:H,item}}}});
    assert.deepEqual(battleMedicineImmunity({effects:[e],patientUUID:P,healerUUID:H,now:0}),{status:'uncertain',expiresAt:null,effectIds:[]});
  }
  const e=effect({system:{start:{value:0},duration:{value:1,unit:'hours'},context:{origin:{actor:H,item:'Actor.H.Item.BM'}}}});
  assert.equal(battleMedicineImmunity({effects:[e],patientUUID:P,healerUUID:H,now:0}).status,'immune');
});
test('expired or suppressed effects cannot establish a live immunity',()=>{
  for(const patch of [{isExpired:true},{isSuppressed:true},{system:{suppressed:true}}, {remainingDuration:{expired:true}}]){
    assert.deepEqual(battleMedicineImmunity({effects:[effect(patch)],patientUUID:P,healerUUID:H,now:0}),{status:'clear',expiresAt:null,effectIds:[]});
  }
});

test('BM duration respects each exact patient rule independently',()=>{
  for(const [robust,godless,want] of [[false,false,86400],[true,false,3600],[false,true,3600],[true,true,3600]]){
    assert.equal(battleMedicineDuration({robust,godless}),want);
  }
  assert.throws(()=>battleMedicineDuration({robust:'robust',godless:false}),/invalid-battle-medicine/);
});
test('all four BM degrees keep their original roll branch including failed use',()=>{
  const cases=[
    [0,{outcome:'criticalFailure',formula:'1d8',medicBonus:5}],
    [1,{outcome:'failure',formula:null,medicBonus:5}],
    [2,{outcome:'success',formula:'2d8+15',medicBonus:5}],
    [3,{outcome:'criticalSuccess',formula:'4d8+15',medicBonus:5}]];
  for(const [degree,want] of cases)assert.deepEqual(battleMedicineOutcome({degree,rank:'expert',medic:true}),want);
});
test('BM bonuses come from the fixed rank and Medic table without speculative dice',()=>{
  const cases=[['trained',false,'2d8',0],['trained',true,'2d8',0],['expert',false,'2d8+10',0],
    ['master',true,'2d8+40',10],['legendary',true,'2d8+65',15]];
  for(const [rank,medic,formula,medicBonus] of cases)assert.deepEqual(battleMedicineOutcome({degree:2,rank,medic}),{outcome:'success',formula,medicBonus});
  for(const input of [{degree:'success',rank:'trained',medic:false},{degree:4,rank:'trained',medic:false},
    {degree:2,rank:'unknown',medic:false},{degree:2,rank:'expert',medic:1}])assert.throws(()=>battleMedicineOutcome(input),/invalid-battle-medicine/);
});

test('Medic requires an exact reviewed baseline and never supplies an inferred allowance',()=>{
  assert.deepEqual(medicAvailability({sessions:{},activities:{}},request()),view('unreviewed'));
  assert.deepEqual(medicAvailability(root(),request()),view('available',1));
  assert.deepEqual(medicAvailability(root(baseline({remaining:0})),request()),view('spent'));
  assert.deepEqual(medicAvailability(root(),request({itemUUID:'Actor.H.Item.Other'})),view('unreviewed'));
});
test('reserved and claimed allowances remain occupied across separate sessions',()=>{
  for(const state of ['reserved','claimed','uncertain']){
    const r=root();r.sessions.Other=session(baseline(),{id:'Other'});
    r.activities.A=activity(claim({state,permitNonce:state==='reserved'?null:'permit'}),{sessionId:'Other'});
    assert.deepEqual(medicAvailability(r,request()),view('uncertain',0,['medic-bypass:A']));
  }
});
test('a witnessed failed use is spent even while HP or immunity remain uncertain',()=>{
  const r=root();r.activities.A=activity(claim({state:'used',permitNonce:'permit',checkId:'check',usedAt:-4.25}),{state:'uncertain'});
  assert.deepEqual(medicAvailability(r,request()),view('spent',0,['medic-bypass:A']));
  assert.deepEqual(medicAvailability(r,request({now:86400})),view('spent',0,['medic-bypass:A']));
});
test('only a released pre-execution reservation stops occupying the allowance',()=>{
  const r=root();r.activities.A=activity(claim({state:'released'}),{state:'cancelled'});
  assert.deepEqual(medicAvailability(r,request()),view('available',1));
  r.activities.A.executor={permitNonce:'permit'};
  assert.deepEqual(medicAvailability(r,request()),view('uncertain',0,['medic-bypass:A']));
});
test('an external exact-source event after review invalidates any session reference',()=>{
  const r=root();r.sessions.Other=session(baseline(),{id:'Other',finiteMedicineReviewChecks:[{actorUUID:H,checkId:'outside',observedAt:-5.5}]});
  assert.deepEqual(medicAvailability(r,request()),view('uncertain'));
  r.sessions.Other.finiteMedicineReviewChecks[0].observedAt=-10.5;
  assert.deepEqual(medicAvailability(r,request()),view('uncertain'));
  r.sessions.Other.finiteMedicineReviewChecks[0].observedAt=-11;
  assert.deepEqual(medicAvailability(r,request()),view('available',1));
  r.sessions.Other.finiteMedicineReviewChecks[0]={actorUUID:'Actor.Other',checkId:'outside',observedAt:0};
  assert.deepEqual(medicAvailability(r,request()),view('available',1));
});
test('exact same-scope references preserve occupation and conflicting baseline shapes fail closed',()=>{
  const r=root();r.sessions.Other=session(baseline(),{id:'Other'});
  assert.deepEqual(medicAvailability(r,request()),view('available',1));
  r.sessions.Other.finiteMedicine.medicBypass.baselineByActor[H].remaining=0;
  assert.deepEqual(medicAvailability(r,request()),view('uncertain'));
});
test('a guessed baseline id in another protocol cannot reset the current allowance',()=>{
  const r=root();r.sessions.Other=session(baseline(),{id:'Other',protocol:{...protocol,epoch:'other'}});
  assert.deepEqual(medicAvailability(r,request()),view('uncertain'));
});
test('claims from another protocol are not charged against this baseline',()=>{
  const r=root();r.sessions.Other={id:'Other',protocol:{...protocol,epoch:'old'}};
  r.activities.A=activity(claim({state:'used',permitNonce:'permit',checkId:'check',usedAt:-4.25}),{sessionId:'Other'});
  assert.deepEqual(medicAvailability(r,request()),view('available',1));
});
test('a legitimate successor supersedes the old baseline instead of creating two allowances',()=>{
  const r=root();r.sessions.Next=session(baseline({id:'review-2',checkedAt:0,previousBaselineId:'review-1',renewal:'new-preparation'}),{id:'Next'});
  assert.deepEqual(medicAvailability(r,request()),view('spent'));
  assert.deepEqual(medicAvailability(r,request({baselineId:'review-2'})),view('available',1));
  r.activities.A=activity(claim({state:'uncertain',permitNonce:'permit'}));
  assert.deepEqual(medicAvailability(r,request({baselineId:'review-2'})),view('uncertain',0,['medic-bypass:A']));
});
test('missing witnesses and conflicting duplicate claims cannot manufacture availability',()=>{
  const r=root();r.activities.A=activity(claim({state:'used',permitNonce:'permit',usedAt:-4.25}));
  assert.deepEqual(medicAvailability(r,request()),view('uncertain',0,['medic-bypass:A']));
  r.activities.A=activity(claim());r.activities.B=activity(claim({activityId:'B'}),{id:'B'});
  assert.deepEqual(medicAvailability(r,request()),view('uncertain',0,['medic-bypass:A']));
});
test('invalid resource timing fails closed without invoking persisted accessors',()=>{
  assert.deepEqual(medicAvailability(root(),request({now:Infinity})),view('uncertain'));
  const r=root();let calls=0;
  Object.defineProperty(r.sessions.S.finiteMedicine.medicBypass.baselineByActor[H],'remaining',{enumerable:true,get(){calls++;return 1}});
  assert.deepEqual(medicAvailability(r,request()),view('uncertain'));assert.equal(calls,0);
});
