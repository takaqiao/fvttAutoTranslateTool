import test from 'node:test';
import assert from 'node:assert/strict';
import {createFrequencyTracker} from '../scripts/usage-events.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';

const item=id=>({uuid:`Actor.pc.Item.${id}`,system:{frequency:{value:1,max:1}}});
function observeUse(tracker,entry,id){
 tracker.seed(entry);entry.system.frequency.value=0;
 const proof={id,itemUuid:entry.uuid,userId:'owner',before:1,after:0,createdAt:12.5};
 assert.deepEqual(tracker.observe(entry,{[ID]:{frequencyReceipt:proof}},'owner'),proof);
 return proof;
}
const binding=(entry,messageId='original')=>({itemUuid:entry.uuid,userId:'owner',messageId});

for(const ttl of [undefined,Infinity])test(`${ttl===undefined?'default':'explicit'} infinite TTL does not scan unclaimed receipts on unrelated observations`,()=>{
 let clockReads=0;const tracker=createFrequencyTracker({now:()=>{clockReads++;return 100},matches:()=>true,...ttl===undefined?{}:{ttl}});
 const first=item('first'),second=item('second');observeUse(tracker,first,'first');observeUse(tracker,second,'second');
 clockReads=0;
 for(let index=0;index<20;index++)assert.equal(tracker.observe(second,{},'owner'),null);
 assert.equal(clockReads,0);
 assert.deepEqual(tracker.diagnostic(),{observed:2,unclaimed:2,consumed:0});
});

test('infinite TTL retains valid receipt timestamps and claims only once after a long wait',()=>{
 let time=100,clockReads=0;const tracker=createFrequencyTracker({now:()=>{clockReads++;return time},matches:()=>true});
 observeUse(tracker,item('prior'),'prior');clockReads=0;
 const entry=item('current'),proof=observeUse(tracker,entry,'current');
 assert.equal(clockReads,1);
 time+=1e12;assert.deepEqual(tracker.claim('current',binding(entry)),proof);
 assert.throws(()=>tracker.claim('current',binding(entry,'copy')),/回执/);
 assert.deepEqual(tracker.diagnostic(),{observed:2,unclaimed:1,consumed:1});
 entry.system.frequency.value=1;tracker.seed(entry);entry.system.frequency.value=0;
 assert.equal(tracker.observe(entry,{[ID]:{frequencyReceipt:proof}},'owner'),null);
});

test('infinite receipts keep their item and owner binding before and after claim',()=>{
 const tracker=createFrequencyTracker({now:()=>100,matches:()=>true}),entry=item('bound'),proof=observeUse(tracker,entry,'bound');
 assert.throws(()=>tracker.claim('bound',{...binding(entry),userId:'other'}),/回执/);
 assert.throws(()=>tracker.claim('bound',{...binding(entry),itemUuid:'Actor.other.Item.bound'}),/回执/);
 assert.deepEqual(tracker.claim('bound',binding(entry)),proof);
 assert.throws(()=>tracker.claim('bound',binding(entry)),/回执/);
});

test('remember and document deletion retain the original consumed receipt lifecycle',()=>{
 const tracker=createFrequencyTracker({now:()=>100,matches:()=>true}),first=item('first'),second=item('second');
 const proof=observeUse(tracker,first,'first');observeUse(tracker,second,'second');
 tracker.remember(proof,'original');assert.deepEqual(tracker.diagnostic(),{observed:2,unclaimed:1,consumed:1});
 assert.throws(()=>tracker.claim('first',binding(first)),/回执/);
 tracker.forgetMessage('copy');assert.equal(tracker.diagnostic().consumed,1);
 tracker.forgetMessage('original');assert.equal(tracker.diagnostic().consumed,0);
 tracker.remember(proof,'original');tracker.forget(first.uuid);
 assert.deepEqual(tracker.diagnostic(),{observed:1,unclaimed:1,consumed:0});
 tracker.forget(second.uuid);assert.deepEqual(tracker.diagnostic(),{observed:0,unclaimed:0,consumed:0});
 observeUse(tracker,item('last'),'last');tracker.clear();
 assert.deepEqual(tracker.diagnostic(),{observed:0,unclaimed:0,consumed:0});
});

test('finite TTL keeps the exact expiry boundary and still sweeps expired receipts',()=>{
 let time=100,clockReads=0;const tracker=createFrequencyTracker({now:()=>{clockReads++;return time},ttl:5000,matches:()=>true});
 const entry=item('finite');observeUse(tracker,entry,'finite');clockReads=0;
 time=5100;tracker.observe(entry,{},'owner');assert.equal(clockReads,1);assert.equal(tracker.diagnostic().unclaimed,1);
 time=5101;tracker.observe(entry,{},'owner');assert.equal(clockReads,2);assert.equal(tracker.diagnostic().unclaimed,0);
 assert.throws(()=>tracker.claim('finite',binding(entry)),/回执/);
});

test('finite TTL can reject an expired claim without a later item observation',()=>{
 let time=100;const tracker=createFrequencyTracker({now:()=>time,ttl:10,matches:()=>true}),entry=item('finite-claim');
 observeUse(tracker,entry,'finite-claim');time=111;
 assert.throws(()=>tracker.claim('finite-claim',binding(entry)),/回执/);
});

test('NaN TTL keeps its existing nonexpiring scan and claim behavior',()=>{
 let clockReads=0;const tracker=createFrequencyTracker({now:()=>{clockReads++;return 100},ttl:NaN,matches:()=>true}),entry=item('nan');
 const proof=observeUse(tracker,entry,'nan');clockReads=0;
 tracker.observe(entry,{},'owner');assert.equal(clockReads,1);assert.equal(tracker.diagnostic().unclaimed,1);
 assert.deepEqual(tracker.claim('nan',binding(entry)),proof);
});

test('negative infinite TTL still expires existing receipts at the next observation',()=>{
 const tracker=createFrequencyTracker({now:()=>100,ttl:-Infinity,matches:()=>true}),entry=item('negative-infinite');
 observeUse(tracker,entry,'negative-infinite');assert.equal(tracker.diagnostic().unclaimed,1);
 tracker.observe(entry,{},'owner');assert.equal(tracker.diagnostic().unclaimed,0);
 assert.throws(()=>tracker.claim('negative-infinite',binding(entry)),/回执/);
});
