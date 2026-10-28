import test from 'node:test';
import assert from 'node:assert/strict';
import { contentHash } from '../../scripts/lib/canon.mjs';

test('key order does not matter', async () => {
  assert.equal(await contentHash({a: 1, b: {c: 2, d: 3}}), await contentHash({b: {d: 3, c: 2}, a: 1}));
});

test('volatile fields are ignored at every level', async () => {
  const a = {_id: 'x', name: 'N', folder: 'f1', sort: 1, ownership: {default: 0}, _stats: {modifiedTime: 1},
             items: [{_id: 'i', name: 'I', sort: 5, _stats: {modifiedTime: 1}}]};
  const b = {...a, folder: 'f2', sort: 9, ownership: {default: 2}, _stats: {modifiedTime: 2},
             items: [{...a.items[0], sort: 7, _stats: {modifiedTime: 9}}]};
  assert.equal(await contentHash(a), await contentHash(b));
});

test('an item field change changes the hash', async () => {
  const a = {items: [{_id: 'i', system: {quantity: 1}}]};
  assert.notEqual(await contentHash(a), await contentHash({items: [{_id: 'i', system: {quantity: 2}}]}));
});

test('hash is 32 lowercase hex chars', async () => {
  assert.match(await contentHash({}), /^[0-9a-f]{32}$/);
});
