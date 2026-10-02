import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createCapabilities,refocusUnsupported} from '../../scripts/exploration/capabilities.mjs';
import {createTreatmentProvider} from '../../scripts/exploration/treatment.mjs';

const source=await readFile('C:/Users/Taka/Desktop/fvtt/tmp/fortress-gap-audit-20260925/code/systems/pf2e/pf2e.mjs','utf8');
assert.equal(createHash('sha256').update(source).digest('hex'),'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157');
const nativeDefinition=source.match(/function suppressFeats\(e\) \{[\s\S]*?\n\}/)?.[0];
assert.ok(nativeDefinition);const suppressFeats=Function(`${nativeDefinition};return suppressFeats`)();
function fixture(){
  const healer={uuid:'Actor.H',name:'H',modeOfBeing:'living',getStatistic:()=>({rank:4,mod:30}),
    hasCondition:()=>false,getActiveTokens:()=>[{}],system:{attributes:{hp:{value:20,max:20}},resources:{focus:{value:0,max:1}}}};
  const slugs=['continual-recovery','ward-medic','risky-surgery','assurance','stitch-flesh','link-focus'];
  healer.items=new Map(slugs.map((slug,index)=>{const id='F'+index;return [id,{id,type:'feat',slug,suppressed:false,
    uuid:`Actor.H.Item.${id}`,actor:healer,parent:healer,system:{slug},flags:{pf2e:{itemGrants:{},...slug==='assurance'?{rulesSelections:{assurance:'medicine'}}:{}}}}]}));
  const patient={...healer,uuid:'Actor.P',items:new Map(),getStatistic:()=>({rank:0,mod:0})};
  const actors=new Map([[healer.uuid,healer],[patient.uuid,patient]]);
  const capabilities=createCapabilities({game:{time:{worldTime:0},modules:new Map(),actors:[]},
    fromUuid:async uuid=>actors.get(uuid),hpPools:{discover:actor=>({poolUUID:actor.uuid,ready:true})}});
  const activity={id:'A',actorUUID:healer.uuid,patientUUIDs:[patient.uuid],startedAt:0,endsAt:600,
    options:{skill:'medicine',rank:'trained',riskySurgery:true}};
  let ownerCalls=0;
  const provider=createTreatmentProvider({capabilities,nativeTreatment:{reconcile:async()=>({status:'uncertain'})},
    ownerOperations:{runActivityWithOwner:async()=>{ownerCalls++;return {status:'confirmed'}}}});
  return {healer,capabilities,activity,provider,get ownerCalls(){return ownerCalls}};
}
test('the pinned native suppressFeats field removes all corresponding exploration qualification',async()=>{
  const f=fixture();suppressFeats([...f.healer.items.values()]);const d=await f.capabilities.discover(f.healer.uuid);
  assert.deepEqual(d.slugs,[]);assert.deepEqual(d.items,[]);assert.equal(d.wardCapacity,1);
  assert.equal(d.continualRecovery,false);assert.equal(d.riskySurgery,false);
  assert.deepEqual(d.assuranceSkills,[]);assert.deepEqual(d.refocusUnsupported,[]);
});
for(const [key,set] of [['native',item=>{item.suppressed=true}],['legacy alias',item=>{item.isSuppressed=true}],
  ['legacy system',item=>{item.system.suppressed=true}]])test(`${key} suppression is retained by both discovery and Refocus qualification`,async()=>{
  const f=fixture();for(const item of f.healer.items.values())set(item);
  assert.equal((await f.capabilities.discover(f.healer.uuid)).riskySurgery,false);
  assert.deepEqual(refocusUnsupported(f.healer.items),[]);
});
test('active native feats and older documents without the native field retain qualification',async()=>{
  const f=fixture();for(const item of f.healer.items.values())delete item.suppressed;
  const d=await f.capabilities.discover(f.healer.uuid);assert.equal(d.wardCapacity,8);
  assert.equal(d.continualRecovery,true);assert.equal(d.riskySurgery,true);
  assert.deepEqual(d.assuranceSkills,['medicine']);assert.deepEqual(d.refocusUnsupported,['link-focus']);
});
test('a suppressed duplicate cannot remove the separate active Risky source',async()=>{
  const f=fixture(),active=[...f.healer.items.values()].find(item=>item.slug==='risky-surgery');
  f.healer.items.set('Duplicate',{...active,id:'Duplicate',suppressed:true});
  const d=await f.capabilities.discover(f.healer.uuid);assert.equal(d.riskySurgery,true);
  assert.equal(d.slugs.filter(slug=>slug==='risky-surgery').length,1);
});
test('suppressed Risky is rejected at begin before owner dispatch',async()=>{
  const f=fixture();suppressFeats([...f.healer.items.values()]);
  assert.deepEqual(await f.provider.begin(f.activity),{status:'blocked',reason:'risky-surgery-unqualified'});assert.equal(f.ownerCalls,0);
});
test('Risky lost after begin is reread at complete before owner dispatch',async()=>{
  const f=fixture();assert.equal((await f.provider.begin(f.activity)).status,'started');
  suppressFeats([...f.healer.items.values()]);
  assert.deepEqual(await f.provider.complete(f.activity,{}),{status:'blocked',reason:'risky-surgery-unqualified'});assert.equal(f.ownerCalls,0);
});
