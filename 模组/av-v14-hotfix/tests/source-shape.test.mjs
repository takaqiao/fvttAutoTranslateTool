import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeSourceShape} from '../scripts/source-shape.mjs';

test('shapes accept a consistent renaming of short names and private members',()=>{
  const original='async f(e){let t=this.#a(e);return Ds.listen(t,e)}';
  assert.equal(normalizeSourceShape(original),normalizeSourceShape('async f(n){let r=this.#highlight(n);return Os.listen(r,n)}'));
  assert.notEqual(normalizeSourceShape(original),normalizeSourceShape('async f(e){let t=this.#a(e);return Ds.listen(e,e)}'));
});

test('shapes keep properties, literals, long names and fixed host globals verbatim',()=>{
  const original='function a(b){ui.show(b.hp,`x`);return b}';
  for(const changed of [
    'function a(b){ui.hide(b.hp,`x`);return b}',
    'function a(b){ui.show(b.hp,`y`);return b}',
    'function a(b){ui.show(b.hp,`x`);return !b}',
    'function a(b){xx.show(b.hp,`x`);return b}',
    'function a(b){ui.show(b.hpMax,`x`);return b}'
  ])assert.notEqual(normalizeSourceShape(original,{fixed:['ui']}),normalizeSourceShape(changed,{fixed:['ui']}),changed);
  assert.notEqual(normalizeSourceShape('game.x(a)'),normalizeSourceShape('canvas.x(a)'));
});

test('regions the scanner cannot classify are refused',()=>{
  assert.throws(()=>normalizeSourceShape('function a(){return @invalid}'),{message:'source-shape-unsupported'});
});
