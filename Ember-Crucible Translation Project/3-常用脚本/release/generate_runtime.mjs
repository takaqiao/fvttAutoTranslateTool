#!/usr/bin/env node
/**
 * Generate each plugin repo's `babele-mappings.js` from the single mapping
 * definition in `3-常用脚本/extract/mappings.mjs` plus the shared runtime
 * converters.
 *
 * There is deliberately NO second hand-maintained copy of the mapping data:
 * the extractor interprets `mappings.mjs`, and the runtime file is generated
 * from it, so the keys the extractor writes and the keys Babele looks for
 * cannot drift apart.
 *
 * Run this after ANY edit to mappings.mjs or runtime-converters.js.
 *
 * Usage: node generate_runtime.mjs [--project <dir>]
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { CRUCIBLE_LAYER, EMBER_LAYER } from '../extract/mappings.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const PROJECT = arg('--project', path.resolve(__dirname, '..', '..'));

const CONVERTERS_SRC = fs.readFileSync(path.join(__dirname, 'runtime-converters.js'), 'utf8');

const TARGETS = [
  { repo: '2-Crucible汉化插件', layer: CRUCIBLE_LAYER, name: 'CRUCIBLE_LAYER', id: 'crucible-cn' },
  { repo: '1-Ember汉化插件', layer: EMBER_LAYER, name: 'EMBER_LAYER', id: 'ember_cn_unofficial' },
];

const HEADER = (id, name) => `/* eslint-disable */
/**
 * GENERATED FILE — DO NOT EDIT BY HAND.
 *
 * Source of truth:
 *   Ember-Crucible Translation Project/3-常用脚本/extract/mappings.mjs
 *   Ember-Crucible Translation Project/3-常用脚本/release/runtime-converters.js
 * Regenerate with:
 *   node "3-常用脚本/release/generate_runtime.mjs"
 *
 * Module : ${id}
 * Layer  : ${name}
 *
 * The exported mapping is handed to \`babele.registerMapping()\`. It only
 * ENRICHES Babele's built-in defaults — fields it does not mention (notably
 * \`Adventure.actors\` and \`Actor.items\`) keep Babele's own \`document\`
 * converter, which is what provides source-pack fallback: an embedded item
 * carrying \`_stats.compendiumSource\` is translated from its ORIGINAL pack's
 * translation file. That single behaviour covers ~82% of Ember's
 * actor-embedded item text for free. Do not replace those with hand-written
 * traversal converters.
 */
`;

for (const t of TARGETS) {
  const outDir = path.join(PROJECT, t.repo);
  if (!fs.existsSync(outDir)) {
    console.warn(`skip (missing repo): ${outDir}`);
    continue;
  }
  const body = [
    HEADER(t.id, t.name),
    CONVERTERS_SRC.trimStart(),
    '',
    `export const DOCUMENT_MAPPINGS = ${JSON.stringify(t.layer, null, 2)};`,
    '',
  ].join('\n');

  const outFile = path.join(outDir, 'babele-mappings.js');
  fs.writeFileSync(outFile, body, 'utf8');
  const types = Object.keys(t.layer).length;
  console.log(`wrote ${path.relative(PROJECT, outFile)}  (${types} document types, ${body.length} bytes)`);
}
