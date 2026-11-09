import fs from 'node:fs';
import {createHash} from 'node:crypto';
import vm from 'node:vm';

const [bundlePath] = process.argv.slice(2);
if (!bundlePath) throw new Error('Usage: node scripts/build-pf2e-870-token-test-fixture.mjs /8.7.0/pf2e.mjs');
const sha = text => createHash('sha256').update(text).digest('hex');
const BUNDLE_SHA256 = 'c9a2f176c388b211ef62c4d39fa5061a3525feebad8ca7cf50a82d2e6e0c1da3';
const bundle = fs.readFileSync(bundlePath, 'utf8');
if (sha(bundle) !== BUNDLE_SHA256) throw new Error('PF2e 8.7.0 bundle digest mismatch');

// Exact slice from the first `start` after the unique `anchor`, through `end` (default: the brace
// closing the first block). `wrap` makes the slice a complete expression for a syntax check.
function slice(name, anchor, start, end, wrap = text => `(${text})`) {
  const at = bundle.indexOf(anchor);
  if (at < 0 || bundle.indexOf(anchor, at + 1) >= 0) throw new Error(`${name}: anchor not unique`);
  const from = bundle.indexOf(start, at);
  let to;
  if (end) to = bundle.indexOf(end, from) + end.length;
  else {
    let depth = 0;
    for (to = bundle.indexOf('{', from); to < bundle.length; to++) if (bundle[to] === '{') depth++; else if (bundle[to] === '}' && --depth === 0) break;
    to++;
  }
  const text = bundle.slice(from, to);
  new vm.Script(wrap(text));
  return {text, provenance: {start: from, end: to, sourceSha256: sha(text)}};
}
const fragments = {
  activeEffectApplyChange: slice('ActiveEffectPF2e.applyChange', 'var ActiveEffectPF2e = class extends ActiveEffect {', 'static applyChange(e, t, n) {', undefined, text => `(class extends Object { ${text} })`),
  actorApplyActiveEffects: slice('ActorPF2e.applyActiveEffects', 'this.inventory = new ActorInventory(this, e);', 'applyActiveEffects(e) {', undefined, text => `(class extends Object { ${text} })`),
  actorSize: slice('ActorSizePF2e', 'jn = class ActorSizePF2e {', 'class ActorSizePF2e {'),
  sizes: slice('Dn', '}, Dn = Object.freeze([', 'Object.freeze([', '])')
};
const fixture = {
  provenance: {systemVersion: '8.7.0', release: 'https://github.com/foundryvtt/pf2e/releases/tag/pf2e-8.7.0', bundleSha256: BUNDLE_SHA256,
    extraction: 'Exact UTF-8 source slices; start and end are offsets into the bundle.',
    fragments: Object.fromEntries(Object.entries(fragments).map(([key, {provenance}]) => [key, provenance]))},
  ...Object.fromEntries(Object.entries(fragments).map(([key, {text}]) => [key, text]))
};
fs.writeFileSync(new URL('../tests/fixtures/pf2e-8.7.0-token-size.json', import.meta.url), JSON.stringify(fixture, null, 2) + '\n');
