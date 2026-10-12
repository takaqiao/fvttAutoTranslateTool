import fs from 'node:fs';
import {createHash} from 'node:crypto';

const [oldPath, priorPath, currentPath] = process.argv.slice(2);
if (!oldPath || !priorPath || !currentPath) throw new Error('Usage: node scripts/build-pf2e-test-fixtures.mjs /8.5.1/pf2e.mjs /8.6.0/pf2e.mjs /8.6.1/pf2e.mjs');
const sha = source => createHash('sha256').update(source).digest('hex');
const directory = new URL('../tests/fixtures/', import.meta.url);
const profiles = [
  ['8.5.1', oldPath, '8fa38a2fcbf848ad967c75876ca33ebf5a46fb7d6a8cc90d8323c0fb5d471bc7'],
  ['8.6.0', priorPath, 'd908c2fa1741d692b00b44e6dacba0bf7e8724d05898d555f9b640b2581d34b4'],
  ['8.6.1', currentPath, 'a48f919b8b9a791868ff3eb4eb560b210855932847e8c861ee914e16e3448ed3']
];
const portraits = {}, naming = {};
for (const [version, file, bundleSha256] of profiles) {
  const source = fs.readFileSync(file, 'utf8');
  if (sha(source) !== bundleSha256) throw new Error(`PF2e ${version} bundle digest mismatch`);
  const release = `https://github.com/foundryvtt/pf2e/releases/tag/pf2e-${version}`;
  const start = source.indexOf('async renderHTML(');
  // End at the method's own closing brace: 8.6.1 mangles the private member that follows it.
  const end = source.indexOf('\n\t}', start) + 3;
  if (start < 0 || end < start) throw new Error(`PF2e ${version} renderHTML boundaries missing`);
  const renderHTML = source.slice(start, end);
  portraits[version] = {renderHTML, provenance: {release, bundleSha256, start, end, sourceSha256: sha(renderHTML)}};
  // The 8.5.1 naming fixture predates this generator and keeps its own provenance file.
  if (version !== '8.5.1') {
    const start = source.indexOf('function generateItemName(');
    const end = source.indexOf('\n}', start) + 2;
    if (start < 0 || end < start) throw new Error(`PF2e ${version} generateItemName boundaries missing`);
    const text = source.slice(start, end);
    naming[version] = {text, provenance: {release, bundleSha256, start, end, sourceSha256: sha(text), file: 'pf2e.mjs', systemVersion: version,
      extraction: 'Exact UTF-8 source slice through unindented closing brace; no newline translation.'}};
  }
}
fs.writeFileSync(new URL('tokenizer-chat-native.json', directory), JSON.stringify(portraits, null, 2) + '\n');
for (const [version, {text, provenance}] of Object.entries(naming)) {
  fs.writeFileSync(new URL(`generate-item-name-${version}.js.txt`, directory), text);
  fs.writeFileSync(new URL(`generate-item-name-${version}-provenance.json`, directory), JSON.stringify(provenance, null, 2) + '\n');
}
