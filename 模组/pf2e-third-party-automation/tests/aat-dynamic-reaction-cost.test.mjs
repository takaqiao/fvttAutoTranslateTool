import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {genericReactionAvailable,genericReactionSpent,createReactionBudget} from '../scripts/reaction-budget.mjs';
import {MODULE_ID} from '../scripts/rules.mjs';

const AAT='pf2e-auto-action-tracker';
const fixture=JSON.parse(readFileSync(new URL('./fixtures/aat-reaction-cost-0.19.1.json',import.meta.url),'utf8'));
const {dynamicCosts,entryCost,slotDefinitions,getSlots,allocateSlots}=fixture.regions;
const context=vm.createContext({});
vm.runInContext(`${dynamicCosts}\n${entryCost}\n${slotDefinitions}\nclass Native {${getSlots}\n${allocateSlots}}\nglobalThis.native=Native;`,context);
const barrage=(id,{rank=1,damage=2,type='reaction',category='spell',cost}={})=>({
 type,slug:'force-barrage',msgId:id,category,rank,
 linkedMessages:Array.from({length:damage},(_,i)=>({type:'damage',msgId:`${id}-damage-${i}`})),...(cost===undefined?{}:{cost})
});
function environment(log,max=2,{active=true,version='0.19.1'}={}) {
 const actor={id:'actor',uuid:'Actor.actor',items:[],flags:{},system:{resources:{reactions:{max}}},hasCondition:()=>false};
 const combatant={id:'combatant',actor,flags:{[AAT]:{log}},getFlag(namespace,key){return this.flags[namespace]?.[key];}};
 const game={modules:new Map([[AAT,{active,version}]]),combat:{id:'encounter',started:true,round:2,turn:0,turns:[combatant]},messages:new Map()};
 return {actor,combatant,game};
}
function compare(log,max=2,options) {
 const {actor,combatant,game}=environment(log,max,options),before=JSON.stringify(combatant.flags);
 const expected=context.native.allocateSlots(combatant,log,'reaction').slots.some(slot=>slot.isBase&&!slot.spentBy);
 assert.equal(genericReactionAvailable(actor,game),expected);
 assert.equal(genericReactionSpent(actor,game),!expected);
 assert.equal(JSON.stringify(combatant.flags),before,'budget reads do not mutate the native log');
 return {actor,combatant,game,expected};
}

test('fixture regions match the current upstream source bytes',()=>{
 assert.ok(process.env.AAT_CURRENT_SOURCE,'AAT_CURRENT_SOURCE must point to the original 0.19.1 main.js');
 const source=readFileSync(process.env.AAT_CURRENT_SOURCE,'utf8');
 assert.equal(createHash('sha256').update(source).digest('hex'),fixture.sourceSHA256);
 for(const region of Object.values(fixture.regions))assert.ok(source.includes(region));
});

for(const rank of [1,3,7,undefined])for(const damage of [0,1,2,4])for(const max of [1,2,3])for(const type of ['action','reaction'])
 test(`native dynamic cost: rank=${rank}, damage=${damage}, reactions=${max}, type=${type}`,()=>{
  const entry=barrage('cast',{rank,damage,type});if(rank===undefined)delete entry.rank;
  entry.linkedMessages.push({type:'applied-damage',msgId:'application'});
  compare([entry],max);
 });

for(const [label,log] of [
 ['two dynamic casts',[barrage('first'),barrage('second',{damage:1})]],
 ['ordinary and dynamic',[{type:'reaction',cost:1,msgId:'first'},barrage('second')]],
 ['quickened prefix',[{type:'action',slug:'quickened-casting',msgId:'quickened'},barrage('second')]],
 ['quickened interrupted by action',[{type:'action',slug:'quickened-casting',msgId:'quickened'},{type:'action',cost:1,slug:'stride',msgId:'stride'},barrage('second')]],
 ['quickened already used',[{type:'action',slug:'quickened-casting',msgId:'quickened'},{type:'action',cost:2,category:'spell',msgId:'first-spell'},barrage('second')]],
 ['quickened after cast',[barrage('first'),{type:'action',slug:'quickened-casting',msgId:'quickened'}]],
 ['quickened reaction',[{type:'reaction',slug:'quickened-casting',msgId:'quickened'},barrage('second')]],
 ['explicit numeric override',[barrage('cast',{cost:1,damage:4})]],
 ['zero explicit numeric override',[barrage('cast',{cost:0,damage:4})]],
 ['numeric string dehydrates to registry',[barrage('cast',{cost:'1',damage:4})]],
 ['plain missing reaction cost',[{type:'reaction',slug:'other',msgId:'reaction'}]],
 ['ordinary spell missing cost',[{type:'reaction',slug:'other-spell',category:'spell',msgId:'reaction'}]],
 ['plain null reaction cost',[{type:'reaction',slug:'other',cost:null,msgId:'reaction'}]],
 ['native rank zero default',[barrage('cast',{rank:0})]],
 ['native rank null default',[barrage('cast',{rank:null})]]
 ])test(`native log accounting: ${label}`,()=>compare(log,3));

for(const slug of ['force-barrage','other-spell'])for(const cost of [0,2])for(const consumed of [false,true])
 test(`native Quickened shared adjustment: ${slug}, numeric=${cost}, consumed=${consumed}`,()=>{
  const log=[{type:'action',slug:'quickened-casting',msgId:'quickened'}];
  if(consumed)log.push({type:'action',category:'spell',cost:2,slug:'other-spell',msgId:'earlier-spell'});
  log.push({...barrage('cast',{cost,damage:4}),slug});
  compare(log,2);
 });

for(const mutation of ['no-links','invalid-links','bad-link','rank-string','rank-negative','rank-fraction','rank-infinite','rank-nan','rank-boolean','unknown-function','unknown-object'])
 test(`unproven dynamic ${mutation} cannot advertise a spare reaction`,()=>{
  const entry=barrage('cast');let calls=0;
  if(mutation==='no-links')delete entry.linkedMessages;
  if(mutation==='invalid-links')entry.linkedMessages={};
  if(mutation==='bad-link')entry.linkedMessages=[null];
  if(mutation==='rank-string')entry.rank='3';
  if(mutation==='rank-negative')entry.rank=-1;
  if(mutation==='rank-fraction')entry.rank=1.5;
  if(mutation==='rank-infinite')entry.rank=Infinity;
  if(mutation==='rank-nan')entry.rank=NaN;
  if(mutation==='rank-boolean')entry.rank=false;
  if(mutation==='unknown-function'){entry.slug='new-dynamic';entry.cost=()=>{calls++;return 0;};}
  if(mutation==='unknown-object'){entry.slug='new-dynamic';entry.cost={valueOf(){calls++;return 0;}};}
  const {actor,combatant,game}=environment([entry],3),before=Object.getOwnPropertyDescriptors(entry);
  assert.equal(genericReactionAvailable(actor,game),false);assert.equal(genericReactionSpent(actor,game),true);
  assert.equal(calls,0);assert.deepEqual(Object.getOwnPropertyDescriptors(entry),before);
  assert.strictEqual(combatant.flags[AAT].log[0],entry);
 });

test('ordinary missing costs retain one-slot accounting and native actor identity',()=>{
 const {actor,combatant,game}=environment([{type:'reaction',msgId:'ordinary'}],2);
 assert.equal(genericReactionAvailable(actor,game),true);
 game.combat.turns.unshift({id:'foreign',actor:{uuid:'Actor.foreign'},flags:{[AAT]:{log:[barrage('foreign',{damage:4})]}}});
 assert.equal(genericReactionAvailable(actor,game),true);
 game.combat.turns=game.combat.turns.filter(entry=>entry!==combatant);
 assert.equal(genericReactionAvailable(actor,game),false);
});

test('inactive AAT preserves the existing TPA ledger and its claim exclusions',()=>{
 const {actor,combatant,game}=environment([barrage('cast',{damage:4})],2,{active:false});
 combatant.flags[MODULE_ID]={reactionBudget:{epoch:'encounter:2',entries:[{type:'reaction',claimKey:'own',cost:1}]}};
 assert.equal(genericReactionAvailable(actor,game),true);
 assert.equal(genericReactionSpent(actor,game,{excludeClaimKeys:['own']}),false);
});

test('native AAT ownership still prevents duplicate shield and chat accounting',async()=>{
 const {actor,combatant,game}=compare([barrage('cast')],2,{version:'future-label'});
 const before=JSON.stringify(combatant.flags),budget=createReactionBudget({game});let calls=0;const result={native:true};
 assert.strictEqual(await budget.applyDamage(actor,{damage:4,shieldBlockRequest:true},async()=>{calls++;return result;}),result);
 assert.equal(calls,1);assert.equal(await budget.record({id:'untrusted'},'player'),false);assert.equal(JSON.stringify(combatant.flags),before);
});
