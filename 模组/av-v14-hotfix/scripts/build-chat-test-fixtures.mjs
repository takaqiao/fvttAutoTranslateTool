// Extract the small, unmodified core fragments used by the standalone Chat tests.
// Usage: node scripts/build-chat-test-fixtures.mjs /absolute/path/to/foundry.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';

const sourcePath = process.argv[2];
if (!sourcePath) throw new Error('Pass the saved original Foundry 14.367 foundry.mjs path.');
const sha256 = data => createHash('sha256').update(data).digest('hex');
const bytes = fs.readFileSync(sourcePath);
const sourceSha256 = sha256(bytes);
assert.equal(sourceSha256, '02248043922e265f368a07922543acb54a7c6402b848ac77e9783093be66545e',
  'Only the audited, unmodified Foundry 14.367 source is accepted.');
const core = bytes.toString('utf8');
const directory = fileURLToPath(new URL('../tests/fixtures/', import.meta.url));
const chatStart = core.indexOf('class ChatLog ');
assert.ok(chatStart >= 0, 'ChatLog class exists');

function extract(name, start, closingPattern) {
  assert.ok(start >= 0, `${name} start exists`);
  const closing = closingPattern.exec(core.slice(start));
  assert.ok(closing, `${name} end exists`);
  const end = start + closing.index + closing[0].length;
  const text = core.slice(start, end);
  return {name, text, sourceStartLine: core.slice(0, start).split('\n').length,
    sourceEndLine: core.slice(0, end).split('\n').length - 1,
    sourceStartCharacter: start, sourceEndCharacter: end, sha256: sha256(text)};
}

const semaphore = extract('Semaphore', core.indexOf('class Semaphore {'), /\r?\n\}\r?\n/);
const methods = ['deleteMessage', '#deleteMessage', '#onScrollLog'].map(name =>
  extract(name, core.indexOf(`\n  ${name}(`, chatStart), /\r?\n  \}\r?\n/));
const files = {
  'chat-semaphore.js': semaphore.text,
  'chat-methods.json': JSON.stringify(Object.fromEntries(methods.map(({name, text}) => [name, text])), null, 2)+'\n'
};
const provenance = {
  foundryVersion: '14.367',
  source: {path: path.resolve(sourcePath), sha256: sourceSha256},
  extraction: 'Exact source slices; no method or class text is rewritten or normalized. JSON escapes only encode the original method strings.',
  fixtureSha256: Object.fromEntries(Object.entries(files).map(([file, content]) => [file, sha256(content)])),
  fragments: [semaphore, ...methods].map(({text, ...record}) => record)
};
fs.mkdirSync(directory, {recursive:true});
for (const [file, content] of Object.entries(files)) fs.writeFileSync(path.join(directory, file), content);
fs.writeFileSync(path.join(directory, 'chat-provenance.json'), JSON.stringify(provenance, null, 2)+'\n');
console.log(`Extracted Semaphore and ${methods.length} ChatLog methods (${Object.values(files).reduce((n,s)=>n+Buffer.byteLength(s),0)} bytes).`);
