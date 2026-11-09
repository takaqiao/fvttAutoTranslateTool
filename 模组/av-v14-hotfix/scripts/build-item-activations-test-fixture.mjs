import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';

const [moduleDir, bundlePath, enPath, cnPath, samplesPath] = process.argv.slice(2);
if (!samplesPath) throw new Error('Usage: node scripts/build-item-activations-test-fixture.mjs <pf2e-item-activations dir> <PF2e 8.7.0 pf2e.mjs> <pf2e lang/en.json> <pf2_cn zh_Hans.json> <samples.json>');
const sha = text => createHash('sha256').update(text).digest('hex');
const read = file => fs.readFileSync(file, 'utf8');
const BUNDLE_SHA256 = 'c9a2f176c388b211ef62c4d39fa5061a3525feebad8ca7cf50a82d2e6e0c1da3';

const manifest = JSON.parse(read(path.join(moduleDir, 'module.json')));
const files = {};
const file = rel => { const text = read(path.join(moduleDir, rel)); files[rel] = sha(text); return text; };
const between = (text, start, end) => {
  const from = text.indexOf(start), to = end ? text.indexOf(end, from) : text.length;
  if (from < 0 || to < 0) throw new Error(`Missing ${start}`);
  return text.slice(from, to).trimEnd() + '\n';
};
const misc = file('scripts/helpers/misc.js'), activate = file('scripts/helpers/activate.js'), constants = file('scripts/helpers/const.js');
const sources = {
  generateActivation: file('scripts/helpers/generate-activation.js'),
  onCreate: file('scripts/helpers/on-create.js'),
  activate: between(activate, 'export const marker', 'export async function turnOnOffActivation'),
  misc: between(misc, 'Hooks.on("ready"', 'export const makeFlatModifier'),
  specific: between(constants, 'export const SPECIFIC_EFFECTS', 'export const SPECIFIC_ACTIVATION_NOTES')
};
// PF2e adds these to CONFIG.PF2E.featTraits and, from there, actionTraits.
const homebrew = manifest.flags[manifest.id]['pf2e-homebrew'].featTraits;
const languages = {};
for (const lang of ['cn', 'en']) {
  const data = JSON.parse(file(`languages/${lang}.json`))[manifest.id];
  languages[lang] = {code: data.code, traits: data.traits};
}

// CONFIG.PF2E.actionTraits, resolved from the bundle's object literals and their spreads.
const bundle = read(bundlePath);
if (sha(bundle) !== BUNDLE_SHA256) throw new Error('PF2e 8.7.0 bundle digest mismatch');
const configLine = bundle.split('\n').filter(line => /^\tactionTraits: [\w$]+,$/.test(line));
if (configLine.length !== 1) throw new Error('CONFIG.PF2E.actionTraits not found');
const escape = name => name.replace(/\$/g, '\\$');
const resolved = new Map();
function resolve(name) {
  if (resolved.has(name)) return resolved.get(name);
  let value;
  // `x = B(y, [...])` is the bundle's omit() helper.
  const omit = new RegExp(String.raw`[,\s]${escape(name)} = B\(([\w$]+), \[([^\]]*)\]\)`).exec(bundle);
  if (omit) {
    const drop = new Function(`return [${omit[2]}]`)();
    value = Object.fromEntries(Object.entries(resolve(omit[1])).filter(([key]) => !drop.includes(key)));
  } else {
    const start = new RegExp(String.raw`(?:^|[,\s])${escape(name)} = \{$`, 'm').exec(bundle);
    if (!start) throw new Error(`Object ${name} not found`);
    const open = bundle.indexOf('{', start.index);
    let depth = 0, close = open;
    for (; close < bundle.length; close++) if (bundle[close] === '{') depth++; else if (bundle[close] === '}' && --depth === 0) break;
    const literal = bundle.slice(open, close + 1);
    const spreads = [...new Set([...literal.matchAll(/\.\.\.([\w$]+)/g)].map(match => match[1]))];
    value = new Function(...spreads, `return (${literal});`)(...spreads.map(resolve));
  }
  resolved.set(name, value);
  return value;
}
const actionTraits = resolve(configLine[0].match(/actionTraits: ([\w$]+),/)[1]);
// The module derives each action's slug from its name with PF2e's sluggify.
const sluggify = between(bundle, 'var jt = String.raw', 'function parseHTML');
const flatten = (object, prefix = '', into = {}) => {
  for (const [key, value] of Object.entries(object)) value && typeof value === 'object' ? flatten(value, `${prefix}${key}.`, into) : (into[prefix + key] = value);
  return into;
};
const en = flatten(JSON.parse(read(enPath))), cn = flatten(JSON.parse(read(cnPath)));
const labels = {en: {}, cn: {}};
for (const key of new Set(Object.values(actionTraits))) {
  labels.en[key] = en[key] ?? key;
  labels.cn[key] = cn[key] ?? en[key] ?? key;
}

const fixture = {
  provenance: {
    module: {id: manifest.id, version: manifest.version, files},
    pf2e: {systemVersion: '8.7.0', release: 'https://github.com/foundryvtt/pf2e/releases/tag/pf2e-8.7.0', bundleSha256: BUNDLE_SHA256,
      enSha256: sha(read(enPath)), cnSha256: sha(read(cnPath))},
    samples: 'Activation paragraphs of world item descriptions (Chinese compendium text), one or two per item, nothing else from the items.'
  },
  sources, sluggify, languages, homebrew, actionTraits, labels,
  samples: JSON.parse(read(samplesPath))
};
fs.writeFileSync(new URL('../tests/fixtures/item-activations-14.3.3.json', import.meta.url), JSON.stringify(fixture, null, 1) + '\n');
