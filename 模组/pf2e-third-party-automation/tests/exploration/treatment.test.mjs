import {test} from 'node:test';import assert from 'node:assert/strict';
import {extension,completionState} from '../../scripts/exploration/treatment.mjs';
test('early native return cannot complete missing result or application',()=>{
  assert.equal(completionState({outcome:'criticalSuccess',checkId:'C1',expectedResults:2,persistedResultIds:['D1'],applicationReceiptIds:[],expectedApplications:2}),'awaiting-evidence');
  assert.equal(completionState({outcome:'failure',checkId:'C1',expectedResults:1,persistedResultIds:['cut'],applicationReceiptIds:['R1'],expectedApplications:1}),'confirmed');
});
test('extension adds the original rolled healing after fifty further minutes',()=>{
  assert.deepEqual(extension({startedAt:0,checkpointAt:600,outcome:'success',rolledHealing:19}),{endsAt:3600,additionalHealing:19});
  assert.equal(extension({startedAt:0,checkpointAt:600,outcome:'failure',rolledHealing:null}),null);
  assert.throws(()=>extension({startedAt:0,checkpointAt:500,outcome:'success',rolledHealing:19}),/checkpoint/);
});
