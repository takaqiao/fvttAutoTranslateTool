import {test} from 'node:test';
import assert from 'node:assert/strict';
import {treatmentOutcomeRows} from '../../scripts/exploration/treatment-context.mjs';
import {Predicate,degree} from '../knowledge-pf2e-860-fixture.mjs';

test('twenty deterministic faces preserve native thresholds and natural adjustments',()=>{
 const rows=treatmentOutcomeRows({rank:4,modifier:19,options:new Set()});
 assert.deepEqual(rows.map(r=>[r.rank,r.dc,r.cases.length]),[['trained',15,20],['expert',20,20],['master',30,20],['legendary',40,20]]);
 assert.deepEqual(rows[1].cases.map(c=>c.outcome),[1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3]);
 assert.equal(rows[3].cases[0].outcome,0);
 assert.equal(rows[3].cases[19].outcome,2);
 assert.ok(rows.every(r=>r.cases.every(c=>c.weight===0.05)));
});

test('Assurance has one actual constant result and no natural-die predicate',()=>{
 const predicate={test:options=>options.has('check:total:natural:undefined')&&options.has('check:total:delta:0')};
 const rows=treatmentOutcomeRows({rank:1,modifier:5,assurance:true,options:new Set(['substitute:assurance']),adjustments:[{predicate,adjustments:{success:{label:'source',amount:1}}}]});
 assert.deepEqual(rows[0].cases,[{weight:1,outcome:3}]);
});

test('ordered adjustment overwrite and all priority match native semantics',()=>{
 const rows=treatmentOutcomeRows({rank:1,modifier:10,options:new Set(),adjustments:[
  {adjustments:{success:{label:'first',amount:1}}},
  {adjustments:{success:{label:'second',amount:-1},all:{label:'all',amount:1}}}
 ]});
 assert.equal(rows[0].cases[4].outcome,3,'all wins over the success-specific downgrade');
 const capped=treatmentOutcomeRows({rank:1,modifier:30,options:new Set(),adjustments:[{adjustments:{all:{label:'increase',amount:1},criticalSuccess:{label:'lower',amount:-1}}}]});
 assert.equal(capped[0].cases[1].outcome,2,'a capped all increase is skipped, then the specific adjustment applies');
});

test('per-face predicates receive native facts without changing caller options',()=>{
 const options=new Set(['action:treat-wounds']);let calls=0;
 const rows=treatmentOutcomeRows({rank:1,modifier:5,options,adjustments:[{predicate:{test:facts=>{calls++;return facts.has('check:roll:total:natural:10')&&facts.has('check:total:15')}},adjustments:{success:{label:'source',amount:1}}}]});
 assert.equal(rows[0].cases[9].outcome,3);
 assert.equal(calls,20);
 assert.deepEqual([...options],['action:treat-wounds']);
});

test('unknown degree shapes and invalid context cannot yield verified rows',()=>{
 for(const data of [{rank:0,modifier:1},{rank:5,modifier:1},{rank:1,modifier:Infinity},{rank:1,modifier:1,assurance:'yes'},{rank:1,modifier:1,adjustments:[{adjustments:{success:{label:'x',amount:99}}}]}])assert.throws(()=>treatmentOutcomeRows(data),/invalid-treatment-context/);
});

for(const ownOptions of [undefined,new Set(),new Set(['origin:level:8']),['origin:level:8']])test(`native PF2e 8.6 adjustment options ${ownOptions===undefined?'inherit':'replace'} the check perspective: ${JSON.stringify(ownOptions?[...ownOptions]:null)}`,()=>{
 const options=new Set(['action:treat-wounds','self:level:8']),predicate=['self:level:8','check:total:15'];
 const adjustments=[{predicate:new Predicate(predicate),...(ownOptions===undefined?{}:{options:ownOptions}),adjustments:{success:{label:'context',amount:1}}}];
 const rows=treatmentOutcomeRows({rank:1,modifier:5,options,adjustments});
 assert.equal(rows[0].cases[9].outcome,ownOptions===undefined?3:2);
 for(let die=1;die<=20;die++)assert.equal(rows[0].cases[die-1].outcome,degree({total:die+5,die,dc:15,rollOptions:[...options],dosAdjustments:adjustments}));
 assert.deepEqual([...options],['action:treat-wounds','self:level:8']);
});
test('an adjustment keeps native total predicates in its own option set',()=>{
 const adjustments=[{predicate:new Predicate('origin:level:8','check:total:15'),options:new Set(['origin:level:8']),adjustments:{success:{label:'opposer',amount:1}}}];
 assert.equal(treatmentOutcomeRows({rank:1,modifier:5,adjustments})[0].cases[9].outcome,3);
});
