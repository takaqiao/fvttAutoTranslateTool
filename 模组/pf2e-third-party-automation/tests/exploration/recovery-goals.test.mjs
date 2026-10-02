import {test} from 'node:test';
import assert from 'node:assert/strict';

async function helpers(){
  try{return await import('../../scripts/exploration/recovery-goals.mjs')}
  catch(error){if(error.code==='ERR_MODULE_NOT_FOUND')assert.fail('recovery goal contract is not implemented');throw error}
}
const actor=(actorUUID,max,poolUUID=actorUUID)=>({actorUUID,hp:{value:0,max},pool:{poolUUID,ready:true}});
const preferences=targetIntentsByActor=>({version:1,targetIntentsByActor,requireNoWounded:false,failureStop:{enabled:false,limit:3}});

test('preferences default to maximum HP and keep returned flags detached',async()=>{
  const {normalizeRecoveryPreferences}=await helpers(),first=normalizeRecoveryPreferences();
  assert.deepEqual(first,{version:1,targetIntentsByActor:{},requireNoWounded:false,failureStop:{enabled:false,limit:3}});
  first.targetIntentsByActor['Actor.A']={mode:'absolute',value:20};first.failureStop.enabled=true;
  assert.deepEqual(normalizeRecoveryPreferences({}),{version:1,targetIntentsByActor:{},requireNoWounded:false,failureStop:{enabled:false,limit:3}});
});

test('preferences preserve saved patients and normalize optional completion flags',async()=>{
  const {normalizeRecoveryPreferences}=await helpers();
  assert.deepEqual(normalizeRecoveryPreferences({targetIntentsByActor:{'Actor.SAVED':{mode:'percent',value:25}},requireNoWounded:true,failureStop:{enabled:true}}),{version:1,targetIntentsByActor:{'Actor.SAVED':{mode:'percent',value:25}},requireNoWounded:true,failureStop:{enabled:true,limit:3}});
  assert.equal(normalizeRecoveryPreferences({failureStop:{limit:100}}).failureStop.limit,100);
});

test('percent targets use each start prepared maximum with upward rounding',async()=>{
  const {resolveRecoveryGoals}=await helpers(),input=preferences({'Actor.A':{mode:'percent',value:25}}),before=structuredClone(input);
  assert.deepEqual(resolveRecoveryGoals([actor('Actor.A',41)],input),{goalsByPool:[{poolUUID:'Actor.A',targetHP:11}],recoveryGoals:{version:1,patientTargets:[{patientUUID:'Actor.A',poolUUID:'Actor.A',intent:{mode:'percent',value:25},basisMaxHP:41,targetHP:11}],requireNoWounded:false,failureStop:{enabled:false,limit:3}}});
  assert.equal(resolveRecoveryGoals([actor('Actor.A',81)],input).recoveryGoals.patientTargets[0].targetHP,21);
  assert.deepEqual(input,before);
});

test('absolute targets clamp this start while preserving the original intention',async()=>{
  const {resolveRecoveryGoals}=await helpers(),input=preferences({'Actor.A':{mode:'absolute',value:40}});
  assert.deepEqual(resolveRecoveryGoals([actor('Actor.A',30)],input).recoveryGoals.patientTargets,[{patientUUID:'Actor.A',poolUUID:'Actor.A',intent:{mode:'absolute',value:40},basisMaxHP:30,targetHP:30}]);
  assert.equal(input.targetIntentsByActor['Actor.A'].value,40);
});

test('shared pools keep patient intentions and merge the highest resolved target',async()=>{
  const {resolveRecoveryGoals}=await helpers(),input=preferences({'Actor.A':{mode:'percent',value:75},'Actor.B':{mode:'absolute',value:20}});
  assert.deepEqual(resolveRecoveryGoals([actor('Actor.A',80,'Actor.M'),actor('Actor.B',80,'Actor.M')],input),{goalsByPool:[{poolUUID:'Actor.M',targetHP:60}],recoveryGoals:{version:1,patientTargets:[{patientUUID:'Actor.A',poolUUID:'Actor.M',intent:{mode:'percent',value:75},basisMaxHP:80,targetHP:60},{patientUUID:'Actor.B',poolUUID:'Actor.M',intent:{mode:'absolute',value:20},basisMaxHP:80,targetHP:20}],requireNoWounded:false,failureStop:{enabled:false,limit:3}}});
});

test('maximum and zero percent targets support zero prepared HP maxima',async()=>{
  const {resolveRecoveryGoals}=await helpers(),result=resolveRecoveryGoals([actor('Actor.A',42),actor('Actor.B',42),actor('Scene.S.Token.T.Actor.C',0)],preferences({'Actor.B':{mode:'percent',value:0}}));
  assert.deepEqual(result.goalsByPool,[{poolUUID:'Actor.A',targetHP:42},{poolUUID:'Actor.B',targetHP:0},{poolUUID:'Scene.S.Token.T.Actor.C',targetHP:0}]);
  assert.deepEqual(result.recoveryGoals.patientTargets[0].intent,{mode:'max',value:null});
});

test('percent ceiling keeps exact whole targets at safe-integer and decimal boundaries',async()=>{
  const {resolveRecoveryGoals}=await helpers();
  for(const [max,percent,targetHP] of [[9007199254740990,100,9007199254740990],[25,28,7],[80,12.5,10],[80,0.0000001,1]])assert.equal(resolveRecoveryGoals([actor('Actor.A',max)],preferences({'Actor.A':{mode:'percent',value:percent}})).goalsByPool[0].targetHP,targetHP);
});

test('target values reject malformed numeric and mode inputs',async()=>{
  const {normalizeRecoveryPreferences}=await helpers();
  for(const intent of [{mode:'percent',value:NaN},{mode:'percent',value:Infinity},{mode:'percent',value:''},{mode:'percent',value:-1},{mode:'percent',value:101},{mode:'absolute',value:-1},{mode:'absolute',value:2.5},{mode:'absolute',value:Number.MAX_SAFE_INTEGER+1},{mode:'absolute',value:'20'},{mode:'max',value:0},{mode:'unknown',value:null},{mode:'percent'}, {mode:'percent',value:25,other:true}])assert.throws(()=>normalizeRecoveryPreferences(preferences({'Actor.A':intent})),/invalid-recovery-preferences/);
});

test('preference flags reject extra fields, invalid keys, and nonplain records',async()=>{
  const {normalizeRecoveryPreferences}=await helpers();
  for(const input of [null,[],new Date(),{version:2},{requireNoWounded:1},{failureStop:null},{failureStop:{enabled:1}},{failureStop:{limit:0}},{failureStop:{limit:101}},{failureStop:{limit:2.5}},{other:true},{targetIntentsByActor:{A:{mode:'max',value:null}}},{targetIntentsByActor:{'Actor. A':{mode:'max',value:null}}},{targetIntentsByActor:new Map()}])assert.throws(()=>normalizeRecoveryPreferences(input),/invalid-recovery-preferences/);
  const hidden={};Object.defineProperty(hidden,'requireNoWounded',{value:true});assert.throws(()=>normalizeRecoveryPreferences(hidden),/invalid-recovery-preferences/);
  assert.throws(()=>normalizeRecoveryPreferences({[Symbol('extra')]:true}),/invalid-recovery-preferences/);
});

test('preference validation rejects getters without executing them',async()=>{
  const {normalizeRecoveryPreferences}=await helpers();let reads=0;
  const getter=()=>{reads++;return 25},intent={mode:'percent'};Object.defineProperty(intent,'value',{enumerable:true,get:getter});
  assert.throws(()=>normalizeRecoveryPreferences(preferences({'Actor.A':intent})),/invalid-recovery-preferences/);
  const map={};Object.defineProperty(map,'Actor.A',{enumerable:true,get:getter});assert.throws(()=>normalizeRecoveryPreferences(preferences(map)),/invalid-recovery-preferences/);
  const input={};Object.defineProperty(input,'requireNoWounded',{enumerable:true,get:getter});assert.throws(()=>normalizeRecoveryPreferences(input),/invalid-recovery-preferences/);
  assert.equal(reads,0);
});

test('start resolution rejects preferences outside the selected snapshot',async()=>{
  const {normalizeRecoveryPreferences,resolveRecoveryGoals}=await helpers(),input=preferences({'Actor.FOREIGN':{mode:'max',value:null}});
  assert.deepEqual(normalizeRecoveryPreferences(input).targetIntentsByActor,input.targetIntentsByActor);
  assert.throws(()=>resolveRecoveryGoals([actor('Actor.A',40)],input),/invalid-recovery-preferences/);
});

test('start resolution rejects duplicate patients, unavailable pools, and inconsistent prepared maxima',async()=>{
  const {resolveRecoveryGoals}=await helpers();
  const badActors=[[actor('Actor.A',40),actor('Actor.A',40)],[{...actor('Actor.A',40),pool:{poolUUID:'Actor.A',ready:false}}],[{...actor('Actor.A',40),pool:{poolUUID:'Actor.A'}}],[actor('Actor.A',NaN)],[actor('Actor.A',Infinity)],[actor('Actor.A',-1)],[actor('Actor.A',4.5)],[actor('Actor.A',Number.MAX_SAFE_INTEGER+1)],[actor('Actor.A',40,'A')],[actor('Actor.A',40,'Actor.M'),actor('Actor.B',41,'Actor.M')]];
  for(const actors of badActors)assert.throws(()=>resolveRecoveryGoals(actors),/invalid-recovery-actors/);
});

test('normalization and resolution return detached patient intentions and flags',async()=>{
  const {normalizeRecoveryPreferences,resolveRecoveryGoals}=await helpers(),input=preferences({'Actor.A':{mode:'percent',value:25}}),normalized=normalizeRecoveryPreferences(input),result=resolveRecoveryGoals([actor('Actor.A',41)],input);
  normalized.targetIntentsByActor['Actor.A'].value=50;normalized.failureStop.limit=9;
  result.recoveryGoals.patientTargets[0].intent.value=75;result.recoveryGoals.failureStop.enabled=true;
  assert.deepEqual(input,{version:1,targetIntentsByActor:{'Actor.A':{mode:'percent',value:25}},requireNoWounded:false,failureStop:{enabled:false,limit:3}});
});

const saved=()=>({version:1,patientTargets:[{patientUUID:'Actor.A',poolUUID:'Actor.M',intent:{mode:'percent',value:75},basisMaxHP:80,targetHP:60},{patientUUID:'Actor.B',poolUUID:'Actor.M',intent:{mode:'absolute',value:20},basisMaxHP:80,targetHP:20}],requireNoWounded:false,failureStop:{enabled:false,limit:3}});
const poolGoals=()=>[{poolUUID:'Actor.M',targetHP:60}];

test('stored goals validate saved basis without requiring the current prepared maximum',async()=>{
  const {validateRecoveryGoals}=await helpers(),input=saved(),result=validateRecoveryGoals(input,['Actor.A','Actor.B'],poolGoals());
  assert.deepEqual(result,input);result.patientTargets[0].intent.value=50;result.failureStop.limit=8;
  assert.equal(input.patientTargets[0].intent.value,75);assert.equal(input.failureStop.limit,3);
});

test('stored goals reject corrupt targets, pool merges, and patient membership',async()=>{
  const {validateRecoveryGoals}=await helpers();
  for(const change of [v=>v.version=2,v=>v.patientTargets[0].targetHP=59,v=>v.patientTargets[0].basisMaxHP=81,v=>v.patientTargets[1].basisMaxHP=70,v=>v.patientTargets[0].intent.value=101,v=>v.patientTargets[0].patientUUID='Actor.FOREIGN',v=>v.patientTargets[1].patientUUID='Actor.A',v=>v.patientTargets.pop(),v=>v.patientTargets[0].other=true,v=>v.requireNoWounded='false',v=>v.failureStop.limit=0,v=>v.other=true,v=>delete v.failureStop]){const input=saved();change(input);assert.throws(()=>validateRecoveryGoals(input,['Actor.A','Actor.B'],poolGoals()),/invalid-recovery-goals/)}
  for(const goals of [[{poolUUID:'Actor.M',targetHP:59}],[],[{poolUUID:'Actor.M',targetHP:60},{poolUUID:'Actor.M',targetHP:60}],[{poolUUID:'Actor.M',targetHP:60,other:true}],[{poolUUID:'Actor.M',targetHP:60},{poolUUID:'Actor.OTHER',targetHP:0}]])assert.throws(()=>validateRecoveryGoals(saved(),['Actor.A','Actor.B'],goals),/invalid-recovery-goals/);
});

test('stored goals reject getters and custom array properties without reading them',async()=>{
  const {validateRecoveryGoals}=await helpers();let reads=0;const input=saved();
  Object.defineProperty(input.patientTargets[0],'targetHP',{enumerable:true,get(){reads++;return 60}});
  assert.throws(()=>validateRecoveryGoals(input,['Actor.A','Actor.B'],poolGoals()),/invalid-recovery-goals/);assert.equal(reads,0);
  const other=saved();other.patientTargets.extra=true;assert.throws(()=>validateRecoveryGoals(other,['Actor.A','Actor.B'],poolGoals()),/invalid-recovery-goals/);
});
