import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {sha256Fallback} from '../scripts/source-hash.mjs';
test('the synchronous digest matches the SHA-256 reference across UTF-8 and block boundaries',()=>{
  for(const value of ['', 'abc', '修复 🌟',...[55,56,63,64,65,1000].map(n=>'a'.repeat(n))]){
    const expected=createHash('sha256').update(value).digest('hex');
    assert.equal(sha256Fallback(value),expected);
  }
});
