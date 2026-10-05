// Exact core Hooks and Array#findSplice fragments, for the Sundry hook-slot tests.
// Usage: node scripts/build-sundry-test-fixtures.mjs /path/to/original/foundry.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
const source = fs.readFileSync(process.argv[2]);
const sha = value => createHash('sha256').update(value).digest('hex');
assert.equal(sha(source),'02248043922e265f368a07922543acb54a7c6402b848ac77e9783093be66545e');
const core = source.toString('utf8');
function slice(name,marker,endMarker) {
  const start = core.indexOf(marker),end = core.indexOf(endMarker,start)+endMarker.length;
  assert.ok(start>=0&&end>start,name);
  const text=core.slice(start,end);
  return {name,text,start,end,startLine:core.slice(0,start).split('\n').length,sha256:sha(text)};
}
const fragments=[slice('Hooks','let Hooks$1 = class Hooks {','\n};'),slice('findSplice','function findSplice(find, replace) {','\n}')];
fs.writeFileSync(new URL('../tests/fixtures/sundry-core-hooks.json',import.meta.url),JSON.stringify(Object.fromEntries(fragments.map(({name,text})=>[name,text])),null,2)+'\n');
fs.writeFileSync(new URL('../tests/fixtures/sundry-core-provenance.json',import.meta.url),JSON.stringify({version:'14.367',sourceSha256:sha(source),extraction:'Exact source slices; JSON only encodes the original text.',fragments:fragments.map(({text,...data})=>data)},null,2)+'\n');
console.log('Extracted exact Hooks class and findSplice function.');
