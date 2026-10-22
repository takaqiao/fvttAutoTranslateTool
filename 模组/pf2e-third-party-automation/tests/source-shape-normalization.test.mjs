import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeSourceShape} from '../scripts/source-shape-match.mjs';

test('normalized shapes accept consistent short names while retaining binding relationships',()=>{
 const original='function a(b){return b.hp+a(b)}';
 assert.equal(normalizeSourceShape(original),normalizeSourceShape('function x(y){return y.hp+x(y)}'));
 assert.notEqual(normalizeSourceShape(original),normalizeSourceShape('function x(y){return x.hp+x(y)}'));
});

test('normalized shapes preserve properties, literals, statements and fixed host globals',()=>{
 const original='function a(b){ui.show(b.hp,2);return b}';
 for(const changed of ['function a(b){ui.hide(b.hp,2);return b}',
  'function a(b){ui.show(b.hp,3);return b}','function a(b){ui.show(b.hp,2);return !b}',
  'function a(b){xx.show(b.hp,2);return b}']){
  assert.notEqual(normalizeSourceShape(original,{fixed:['ui']}),normalizeSourceShape(changed,{fixed:['ui']}));
 }
 assert.throws(()=>normalizeSourceShape('function a(){return @invalid}'),/source-shape-unsupported/);
});
