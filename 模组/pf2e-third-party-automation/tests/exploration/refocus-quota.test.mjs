import {test} from 'node:test';import assert from 'node:assert/strict';
import {refocusCommitValue} from '../../scripts/exploration/refocus.mjs';
test('one ten-minute baseline Refocus restores one point and full-focus composites restore zero',()=>{assert.equal(refocusCommitValue({before:0,max:3,requested:3,recovery:1}),1);assert.equal(refocusCommitValue({before:2,max:3,requested:3,recovery:1}),3);assert.equal(refocusCommitValue({before:3,max:3,requested:3,recovery:1}),3);assert.throws(()=>refocusCommitValue({before:0,max:3,requested:2,recovery:1}),/source/)});
test('Workbench baseline actually requests one point rather than the entire pool',()=>{assert.equal(refocusCommitValue({before:0,max:2,requested:1,recovery:1}),1)});
