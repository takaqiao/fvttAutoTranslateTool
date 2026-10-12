/** Capture the pf2e_compendium_chn 3.2.3 incremental document-index source for regression tests. */
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {sourceTokens, extractDeclaration} from './source-contract.mjs';

const [file] = process.argv.slice(2);
if (!file) throw new Error('Usage: node scripts/build-babele-323-test-fixture.mjs /3.2.3/babele-ondemand-patch.js');
const sha = value => createHash('sha256').update(value).digest('hex');
const bytes = fs.readFileSync(file), source = bytes.toString('utf8');
if (sha(bytes) !== '0a8a7aee3d82f1a9f1a8e9f7cf11a3b21224cc0f1e98bb61e77bfdfc50c9bed7') throw new Error('Expected the pf2e_compendium_chn 3.2.3 babele-ondemand-patch.js');
const target = 'foundry.documents.collections.CompendiumCollection.prototype.indexDocument';
const fragments = {};
const fragment = (name, start, end) => {
  const text = source.slice(start, end);
  fragments[name] = {start, end, sha256: sha(text)};
  return text;
};
const declarations = {};
for (const name of ['documentIndexOperations', 'documentIndexRecoveries', 'scheduleDocumentIndexRebuild', 'clearDocumentIndexData',
  'assertSharedDocumentIndexData', 'prepareDocumentIndexOperation', 'queueDocumentIndexOperation', 'scheduleDocumentIndexEntryReplace',
  'rebuildDocumentIndexCompat', 'normalizePackId', 'translateIndexTitles']) {
  const text = extractDeclaration(source, name), start = text ? source.indexOf(text) : -1;
  if (start < 0 || source.indexOf(text, start + 1) >= 0) throw new Error(`Cannot locate one ${name} declaration`);
  declarations[name] = fragment(name, start, start + text.length);
}
const tokens = sourceTokens(source), registrations = [];
for (const [index, token] of tokens.entries()) {
  if (['libWrapper', '.', 'register', '('].every((value, offset) => tokens[index + offset]?.value === value)
    && tokens[index + 6]?.value === JSON.stringify(target)) registrations.push([token.start, tokens[tokens[index + 3].match + 1].end]);
}
if (registrations.length !== 1) throw new Error('Expected one indexDocument wrapper registration');
const fixture = {version: '3.2.3', declarations, wrapper: fragment('indexDocument wrapper registration', ...registrations[0]),
  provenance: {module: 'pf2e_compendium_chn', file: 'babele-ondemand-patch.js', sha256: sha(bytes), bytes: bytes.length,
    extraction: 'Exact UTF-8 source slices of top-level declarations and the indexDocument libWrapper registration; no newline translation.', fragments}};
fs.writeFileSync(new URL('../tests/fixtures/babele-upstream-3.2.3.json', import.meta.url), JSON.stringify(fixture, null, 2) + '\n');
console.log('Wrote the pf2e_compendium_chn 3.2.3 document-index fixture.');
