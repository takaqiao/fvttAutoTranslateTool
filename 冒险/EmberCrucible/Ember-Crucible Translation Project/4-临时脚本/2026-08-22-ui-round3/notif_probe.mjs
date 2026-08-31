/**
 * Offline proof for the ESCALATION in unit ③: the macro literal
 *   "You must select a single Grayling token."
 * flows through the ember_cn_unofficial notification channel, and TODAY that
 * channel returns it unchanged.
 *
 * Method: take a READ-ONLY COPY of scripts/ember-hardcoded-cn.mjs (that file
 * belongs to another unit and is not touched), append an export of
 * `translateNotification` + the two tables, and call it.
 *
 * SELF-PROOF (constraint 6), two separate assertions:
 *   A. COUNT   : the copy must expose exactly the table sizes the main gate
 *                reports for this channel -> NOTIFICATION_PATTERNS === 27.
 *                (Proves we sliced/loaded the right module, right revision.)
 *   B. IDENTITY: a known-present entry must round-trip to its known Chinese,
 *                and a known-absent near-miss must round-trip UNCHANGED.
 *                (Proves we are calling the real lookup, not a stub that
 *                 happens to return the right number of things.)
 */
import fs from 'fs';

const P = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project';
const SRC = `${P}/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs`;
const OUT = `${P}/4-临时脚本/2026-08-22-ui-round3/_notif_copy.mjs`;

globalThis.Hooks = { once() {}, on() {} };

let src = fs.readFileSync(SRC, 'utf8');
// strip the static import of the self-check panel (it pulls in Foundry globals)
src = src.replace(/^import[^\n]*ember-cn-selfcheck[^\n]*$/m, '');
src += `

export { translateNotification, NOTIFICATIONS, NOTIFICATION_PATTERNS };
`;
fs.writeFileSync(OUT, src, 'utf8');

const M = await import(`file:///${OUT}`);
const { translateNotification: tn, NOTIFICATIONS, NOTIFICATION_PATTERNS } = M;

const results = [];
const rec = (id, ok, detail) => { results.push({ id, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'} ${id}  ${detail}`); };

/* A - count */
rec('A-table-sizes', NOTIFICATION_PATTERNS.length === 27,
  `NOTIFICATIONS=${Object.keys(NOTIFICATIONS).length} entries, NOTIFICATION_PATTERNS=${NOTIFICATION_PATTERNS.length} (gate records 27)`);

/* B - identity: present entry translates, near-miss does not */
const present = 'Ember game state successfully reset!';
const nearMiss = 'Ember game state successfully resets!';
rec('B1-known-entry-translates', tn(present) === '余烬战役状态已重置！', `${JSON.stringify(present)} -> ${JSON.stringify(tn(present))}`);
rec('B2-near-miss-passthrough', tn(nearMiss) === nearMiss, `${JSON.stringify(nearMiss)} -> ${JSON.stringify(tn(nearMiss))}`);

/* THE MEASUREMENT */
const TARGET = 'You must select a single Grayling token.';
const got = tn(TARGET);
rec('M-grayling-currently-untranslated', got === TARGET, `translateNotification(${JSON.stringify(TARGET)}) -> ${JSON.stringify(got)}`);

/* Would the proposed entry work?  Simulate the exact table addition. */
const PROPOSED_CN = '只有在选中单个灰蛾灵 Grayling 指示物时才能使用该宏。';
NOTIFICATIONS[TARGET] = PROPOSED_CN;
rec('M2-proposed-entry-would-hit', tn(TARGET) === PROPOSED_CN, `after adding the entry -> ${JSON.stringify(tn(TARGET))}`);
const negative = 'You must select a single Grayling token'; // no full stop -> must NOT hit
rec('M3-proposed-negative-passthrough', tn(negative) === negative,
  `near-miss without the full stop -> ${JSON.stringify(tn(negative))} (unchanged = correct; the entry is EXACT, not a pattern)`);

// NOTE: on Windows the imported ESM file stays locked for the life of this
// process, so this rm is a no-op here; the copy is deleted by the caller after
// the run. Verified: after `node notif_probe.mjs` the file is still on disk.
try { fs.rmSync(OUT, { force: true }); } catch { /* locked while imported */ }
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
fs.writeFileSync(`${P}/4-临时脚本/2026-08-22-ui-round3/notif_probe.json`, JSON.stringify(results, null, 2), 'utf8');
if (failed.length) process.exit(1);
