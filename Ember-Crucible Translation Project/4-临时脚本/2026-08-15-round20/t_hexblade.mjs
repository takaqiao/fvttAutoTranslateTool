/**
 * Round-20 ③ gate: import the REAL ember-hardcoded-cn.mjs and prove
 *   (1) `Warlock Patron: Hexblade` now keeps the English leaf,
 *   (2) the rest of WARLOCK_PATRONS / DIVINE_DOMAINS / SORCEROUS_ORIGINS still translate,
 *   (3) MUTATION BACK-TEST: a copy with `"Hexblade": "咒刃",` put back DOES translate it.
 *
 * (3) is the anti-空转 half: without it, (1) would also "pass" if the whole
 * PREFIXED path were broken and translateText returned everything unchanged.
 * The mutant is written to disk and imported as a real module (no heredoc,
 * no in-memory string patching) so escaping cannot silently defeat it.
 */
import fs from 'fs';
import path from 'path';
import { pathToFileURL, fileURLToPath } from 'url';

const SRC = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs';
const OUT = path.dirname(fileURLToPath(import.meta.url));

// Minimal Foundry surface: the module only registers hooks at load time.
const noop = () => {};
globalThis.Hooks = { once: noop, on: noop, off: noop, callAll: noop, call: noop };
globalThis.game ??= { i18n: { localize: (k) => k, has: () => false }, settings: { register: noop, get: noop }, user: {} };
globalThis.CONFIG ??= {};
globalThis.foundry ??= { utils: {}, applications: {} };

const real = await import(pathToFileURL(SRC).href);
const T = real.translateText;
if (typeof T !== 'function') throw new Error('translateText not exported / not a function');

let fail = 0;
const chk = (label, got, want) => {
  const ok = got === want;
  if (!ok) fail++;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${label}\n     got  ${JSON.stringify(got)}\n     want ${JSON.stringify(want)}`);
};

console.log('=== A. current file ===');
chk('Hexblade leaf stays English',
  T('Warlock Patron: Hexblade'), '邪术师宗主：Hexblade');
chk('regression The Great Old One',
  T('Warlock Patron: The Great Old One'), '邪术师宗主：远古旧日支配者');
chk('regression The Undead',
  T('Warlock Patron: The Undead'), '邪术师宗主：不死');
chk('regression Archfey',
  T('Warlock Patron: The Archfey'), '邪术师宗主：至高妖精');
chk('regression Other',
  T('Warlock Patron: Other'), '邪术师宗主：其他');
chk('regression Divine Domain',
  T('Divine Domain: Twilight'), '神圣领域：暮光');
chk('regression Sorcerous Origin',
  T('Sorcerous Origin: Wild Magic'), '术士起源：狂野魔法');
chk('untouched: bare Hexblade is not in any global table',
  T('Hexblade'), 'Hexblade');
chk('untouched: Spellblade still not hijacked by this file',
  T('Spellblade'), 'Spellblade');

console.log('\n=== B. mutation back-test (put the key back on disk) ===');
const src = fs.readFileSync(SRC, 'utf8');
const NEEDLE = '  // `Hexblade` 有意缺席';
if (!src.includes(NEEDLE)) throw new Error('mutation anchor missing — back-test cannot be trusted');
const mutated = src.replace(NEEDLE, '  "Hexblade": "咒刃",\n' + NEEDLE);
if (mutated === src) throw new Error('mutation did not change the source — back-test is vacuous');
if (!/"Hexblade": "咒刃",/.test(mutated)) throw new Error('mutant does not contain the re-added key');
const MUT = path.join(OUT, '_mutant_hexblade.mjs');
fs.writeFileSync(MUT, mutated, 'utf8');
console.log(`mutant written: ${MUT} (${mutated.length - src.length} bytes added)`);
const mut = await import(pathToFileURL(MUT).href);
const got = mut.translateText('Warlock Patron: Hexblade');
console.log(`mutant output: ${JSON.stringify(got)}`);
if (got === '邪术师宗主：咒刃') console.log('PASS back-test: the assertion in A really is sensitive to this key');
else { fail++; console.log('FAIL back-test: re-adding the key changed nothing -> test A proves nothing'); }
fs.unlinkSync(MUT);

console.log(`\nTOTAL FAIL=${fail}`);
process.exit(fail ? 1 : 0);
