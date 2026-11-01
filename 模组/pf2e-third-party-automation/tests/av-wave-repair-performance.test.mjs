import test from 'node:test';
import assert from 'node:assert/strict';
import {buildAvWaveRepair,WAVE_SOURCE} from '../scripts/av-wave-repair.mjs';
import {createAvAutomation} from '../scripts/av-automation.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';

const originalSlugs=['arctic-rift','breathe-fire','blazing-bolt','falling-star','fireball','frostbite','frozen-fog','ice-storm','ignition','volcanic-eruption'];
const correctedSlugs=['arctic-rift','blazing-bolt','breathe-fire','falling-stars','fireball','frostbite','frozen-fog','howling-blizzard','ice-storm','ignition','volcanic-eruption'];
const spellPredicate=slugs=>[{or:slugs.map(s=>'item:slug:'+s)}];
const correctedPredicate=()=>[...spellPredicate(correctedSlugs),{or:['conservation-of-energy:fire','conservation-of-energy:cold']}];
const energyOptions=()=>[{label:'Fire',value:'fire'},{label:'Cold',value:'cold'}];
const neutralOptions=()=>[{label:'中性（再聚能）',value:'none'},...energyOptions()];
const damageRule=predicate=>({key:'DamageAlteration',mode:'override',property:'damage-type',slug:'base',selectors:['spell-damage'],value:'{item|flags.system.rulesSelections.conservationOfEnergy}',predicate});
const toggleRule=suboptions=>({key:'RollOption',option:'conservation-of-energy',alwaysActive:true,placement:'spellcasting',suboptions});
const unchangedRule=()=>({key:'RollOption',option:'unrelated',predicate:[{and:['feature:wave',{not:'custom:disabled'}]}]});
const repairKey=`flags.${ID}.avWaveRepair`;
const reason='PF2e 8.5 granted spell UUIDs and legal Refocus neutrality';

function fixture(kind){
 const prior={sourceId:WAVE_SOURCE,reason:'Earlier spell list repair',changes:[{index:1,path:'predicate',before:spellPredicate(['legacy-wave']),after:spellPredicate(correctedSlugs)}],metadata:{version:1}};
 const item={sourceId:WAVE_SOURCE,flags:{},system:{rules:[unchangedRule(),damageRule(spellPredicate(originalSlugs)),toggleRule(energyOptions())]}};
 if(['intermediate','repaired','neutrality','multiple'].includes(kind))item.flags[ID]={avWaveRepair:prior};
 if(kind==='intermediate')item.system.rules[1].predicate=spellPredicate(correctedSlugs);
 if(['repaired','neutrality'].includes(kind))item.system.rules[1].predicate=correctedPredicate();
 if(kind==='repaired')item.system.rules[2].suboptions=neutralOptions();
 if(kind==='custom')item.system.rules[1].predicate=spellPredicate([...originalSlugs,'custom-spell']);
 if(kind==='untracked-intermediate')item.system.rules[1].predicate=spellPredicate(correctedSlugs);
 if(kind==='other-source')item.sourceId='Compendium.pf2e.classfeatures.Item.custom';
 if(kind==='multiple')item.system.rules=[toggleRule(energyOptions()),damageRule(spellPredicate(originalSlugs)),unchangedRule(),damageRule(spellPredicate(correctedSlugs)),toggleRule(energyOptions())];
 return item;
}

function expectedRepair(item,kind){
 const rules=structuredClone(item.system.rules),changes=[];
 const damageIndices=kind==='neutrality'?[]:kind==='multiple'?[1,3]:[1];
 const toggleIndices=kind==='multiple'?[0,4]:[2];
 for(const index of damageIndices){
  const before=structuredClone(rules[index].predicate);rules[index].predicate=correctedPredicate();
  changes.push({index,path:'predicate',before,after:correctedPredicate()});
 }
 for(const index of toggleIndices){
  const before=structuredClone(rules[index].suboptions);rules[index].suboptions=neutralOptions();
  changes.push({index,path:'suboptions',before,after:neutralOptions()});
 }
 const prior=structuredClone(item.flags[ID]?.avWaveRepair);
 return {'system.rules':rules,[repairKey]:{...prior,sourceId:WAVE_SOURCE,reason,changes:[...(prior?.changes??[]),...changes]}};
}

function trackClones(t){
 const previous=globalThis.foundry,calls=[];
 globalThis.foundry={utils:{deepClone(value){calls.push({value,bytes:Buffer.byteLength(JSON.stringify(value))});return structuredClone(value)}}};
 t.after(()=>{globalThis.foundry=previous});
 return calls;
}

function recordClones(t,kind,item,calls){
 t.diagnostic(JSON.stringify({kind,inputRulesBytes:Buffer.byteLength(JSON.stringify(item.system.rules)),cloneCalls:calls.length,fullRulesClones:calls.filter(c=>c.value===item.system.rules).length,clonedInputBytes:calls.reduce((n,c)=>n+c.bytes,0)}));
}

function assertIsolated(output,input){
 const inputNodes=new Set();
 const visit=(value,fn)=>{if(!value||typeof value!=='object')return;fn(value);for(const child of Object.values(value))visit(child,fn)};
 visit(input,value=>inputNodes.add(value));
 visit(output,value=>assert.equal(inputNodes.has(value),false,'repair output must not share input rule nodes'));
}

for(const kind of ['repaired','custom','untracked-intermediate','other-source'])test(`${kind} wave rules need no repair or deep clone`,t=>{
 const item=fixture(kind),before=structuredClone(item),calls=trackClones(t);
 const result=buildAvWaveRepair(item);recordClones(t,kind,item,calls);
 assert.equal(result,null);assert.deepEqual(item,before);assert.equal(calls.length,0);
});

for(const kind of ['original','intermediate','multiple','neutrality'])test(`${kind} wave repair clones all rules once and preserves rule and history order`,t=>{
 const item=fixture(kind),before=structuredClone(item),expected=expectedRepair(item,kind),calls=trackClones(t);
 const result=buildAvWaveRepair(item);recordClones(t,kind,item,calls);
 assert.deepEqual(result,expected);assert.deepEqual(item,before);
 assert.equal(calls.filter(c=>c.value===item.system.rules).length,1);
 assertIsolated(result['system.rules'],item);
 const priorCount=item.flags[ID]?.avWaveRepair.changes.length??0,newChanges=result[repairKey].changes.slice(priorCount);
 assertIsolated(newChanges,item);assertIsolated(newChanges,result['system.rules']);
 const unchangedIndex=kind==='multiple'?2:0;
 result['system.rules'][unchangedIndex].predicate[0].and[1].not='changed-output';
 newChanges[0].before[0].label='changed-history';
 newChanges.at(-1).after[0].label='changed-history';
 assert.deepEqual(item,before);
 assert.deepEqual(result[repairKey].changes.slice(0,priorCount),before.flags[ID]?.avWaveRepair.changes??[]);
});

test('wave repairs remain isolated without Foundry deepClone',t=>{
 const previous=globalThis.foundry;t.after(()=>{globalThis.foundry=previous});delete globalThis.foundry;
 const item=fixture('original'),before=structuredClone(item),result=buildAvWaveRepair(item);
 assert.deepEqual(result,expectedRepair(item,'original'));assertIsolated(result['system.rules'],item);
 assertIsolated(result[repairKey].changes,item);assert.deepEqual(item,before);
});

test('AV maintenance writes only the expected repairs and does no work on a second pass',async t=>{
 const gm={id:'gm'},users=new Map([[gm.id,gm]]);users.activeGM=gm;
 const actor={id:'pc',uuid:'Actor.pc',items:new Map()},writes=[],expected=[];
 for(const kind of ['repaired','original','custom','intermediate','other-source','multiple','neutrality']){
  const item=fixture(kind);item.id=kind;item.uuid=actor.uuid+'.Item.'+kind;
  if(['original','intermediate','multiple','neutrality'].includes(kind))expected.push({id:kind,changes:expectedRepair(item,kind)});
  item.update=async changes=>{
   writes.push({id:item.id,changes:structuredClone(changes)});
   item.system.rules=structuredClone(changes['system.rules']);
   item.flags[ID]??={};item.flags[ID].avWaveRepair=structuredClone(changes[repairKey]);
  };
  actor.items.set(item.id,item);
 }
 const game={user:gm,users,time:{worldTime:100}},provider=createAvAutomation({game,castEvents:{addMatcher(){},addCapture(){}}}),calls=trackClones(t);
 await provider.maintain(actor);assert.deepEqual(writes,expected);
 const firstPassClones=calls.length;calls.length=0;
 await provider.maintain(actor);
 t.diagnostic(JSON.stringify({maintenanceWrites:writes.length,firstPassClones,secondPassClones:calls.length}));
 assert.deepEqual(writes,expected);assert.equal(calls.length,0);
});
