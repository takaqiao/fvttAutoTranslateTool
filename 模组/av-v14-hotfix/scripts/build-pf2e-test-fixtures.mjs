import fs from 'node:fs';
import {createHash} from 'node:crypto';

const [oldPath, currentPath] = process.argv.slice(2);
if (!oldPath || !currentPath) throw new Error('Usage: node scripts/build-pf2e-test-fixtures.mjs /8.5.1/pf2e.mjs /8.6.0/pf2e.mjs');
const sha = source => createHash('sha256').update(source).digest('hex');
const directory = new URL('../tests/fixtures/', import.meta.url);
const profiles = [
  ['8.5.1', oldPath, '8fa38a2fcbf848ad967c75876ca33ebf5a46fb7d6a8cc90d8323c0fb5d471bc7'],
  ['8.6.0', currentPath, 'd908c2fa1741d692b00b44e6dacba0bf7e8724d05898d555f9b640b2581d34b4']
];
const portraits = {};
let naming;
for (const [version, file, bundleSha256] of profiles) {
  const source = fs.readFileSync(file, 'utf8');
  if (sha(source) !== bundleSha256) throw new Error(`PF2e ${version} bundle digest mismatch`);
  const release = `https://github.com/foundryvtt/pf2e/releases/tag/pf2e-${version}`;
  const start = source.indexOf('async renderHTML(');
  const end = source.indexOf('\n\t#highlightDoS(', start);
  if (start < 0 || end < start) throw new Error(`PF2e ${version} renderHTML boundaries missing`);
  const renderHTML = source.slice(start, end);
  portraits[version] = {renderHTML, provenance: {release, bundleSha256, start, end, sourceSha256: sha(renderHTML)}};
  if (version === '8.6.0') {
    const start = source.indexOf('function generateItemName(');
    const end = source.indexOf('\n}', start) + 2;
    if (start < 0 || end < start) throw new Error('PF2e 8.6.0 generateItemName boundaries missing');
    const text = source.slice(start, end);
    naming = {text, provenance: {release, bundleSha256, start, end, sourceSha256: sha(text), file: 'pf2e.mjs', systemVersion: version,
      extraction: 'Exact UTF-8 source slice through unindented closing brace; no newline translation.'}};
  }
}
fs.writeFileSync(new URL('tokenizer-chat-native.json', directory), JSON.stringify(portraits, null, 2) + '\n');
fs.writeFileSync(new URL('generate-item-name-8.6.0.js.txt', directory), naming.text);
fs.writeFileSync(new URL('generate-item-name-8.6.0-provenance.json', directory), JSON.stringify(naming.provenance, null, 2) + '\n');
