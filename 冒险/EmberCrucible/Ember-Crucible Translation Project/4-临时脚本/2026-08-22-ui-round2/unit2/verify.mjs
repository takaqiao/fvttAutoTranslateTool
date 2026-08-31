/**
 * 本轮改动的验收：拿**改前 / 改后**两份 ember-hardcoded-cn.mjs 各造一个 harness，
 * 用**同一个** translateText 实现跑同一批真实输入，逐条比对。
 *
 * 用法：node verify.mjs <改前.mjs> <改后.mjs> <newkeys.json> <out.json>
 *
 * 前置自证（三件，任一不成立当场 exit 2）：
 *   A 切对条数 —— 两份 harness 都 import 成功，且「改后表键数 − 改前表键数」恰好等于
 *                 newkeys.json 的条数（不多不少，证明我数的就是我加的那批）；
 *   B 切对对象 —— **负控制**：改前那份对这批新键必须**一条都不翻**（若已经翻了，
 *                 说明我把已覆盖的当成缺口了，整轮计数作废）；
 *   C 判定式不空转 —— 拿一个**上游根本没有**的假串跑作用域表，必须不翻
 *                 （若连假串都被翻，说明查表逻辑恒真，后面所有绿都是假的）。
 *
 * 三段验收：
 *   ① 新增每条：作用域表下必须翻成登记的中文；
 *   ② 已有条目零行为变化：改前表里的**每一个**键，两份实现在
 *      「全局通道」与「作用域通道」下的输出必须逐字节相同；
 *   ③ 作用域实测：新增的每一个键在**全局通道**（不传 extra）下必须**一条都不命中**。
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const STUB_IMPORT = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const NAMES = ["TOKEN_MAKER_UI", "TOKEN_MAKER_WINDOW_UI", "EMBER_WINDOW_UI", "EXACT",
               "DIALOG_UI", "EMBER_DIALOG_UI", "MOOD_PANEL", "CHAT_UI", "WEATHER"];
const die = (m) => { console.error("verify: " + m); process.exit(2); };

async function load(src, tag) {
  const s = fs.readFileSync(src, "utf8");
  if (!s.includes(STUB_IMPORT)) die(`${tag}: 找不到要打桩的 import 行`);
  for (const n of NAMES) if (!new RegExp("\\nconst " + n + " = ").test(s)) die(`${tag}: 找不到 ${n}`);
  const h = path.join(process.cwd(), `_v_${tag}.mjs`);
  fs.writeFileSync(h,
    "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
    + s.replace(STUB_IMPORT, "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
    + `\nexport { ${NAMES.join(", ")} };\n`, "utf8");
  return import(pathToFileURL(h).href);
}

const [, , F_OLD, F_NEW, F_KEYS, F_OUT] = process.argv;
const OLD = await load(F_OLD, "old");
const NEW = await load(F_NEW, "new");
const newKeys = JSON.parse(fs.readFileSync(F_KEYS, "utf8"));

/* ── 前置自证 A ── */
const dOld = Object.keys(OLD.TOKEN_MAKER_UI).length;
const dNew = Object.keys(NEW.TOKEN_MAKER_UI).length;
if (dNew - dOld !== newKeys.length) {
  die(`PRECHECK-A FAIL 表键数 ${dOld} → ${dNew}（差 ${dNew - dOld}），但 newkeys 有 ${newKeys.length} 条`);
}
console.log(`PRECHECK-A OK  TOKEN_MAKER_UI ${dOld} → ${dNew}，恰好 +${newKeys.length}（与 newkeys 逐数对上）`);

/* ── 前置自证 B（负控制）── */
const preTranslated = newKeys.filter(k => OLD.translateText(k, OLD.TOKEN_MAKER_WINDOW_UI) !== k);
if (preTranslated.length) die(`PRECHECK-B FAIL 改前就已经翻了 ${preTranslated.length} 条：${preTranslated.slice(0, 5)}`);
console.log(`PRECHECK-B OK  改前对这 ${newKeys.length} 条新键一条都不翻（负控制成立）`);

/* ── 前置自证 C（判定式不空转）── */
const FAKE = "Zzq Frobnicated Widget Not Upstream";
if (NEW.translateText(FAKE, NEW.TOKEN_MAKER_WINDOW_UI) !== FAKE) die("PRECHECK-C FAIL 假串竟被翻了 —— 查表恒真");
console.log("PRECHECK-C OK  假串不被翻（查表不是恒真）");

/* ── ① 新增每条：真实输入验 ── */
const bad1 = [];
for (const k of newKeys) {
  const got = NEW.translateText(k, NEW.TOKEN_MAKER_WINDOW_UI);
  const want = NEW.TOKEN_MAKER_UI[k];
  if (got !== want) bad1.push([k, got, want]);
}
console.log(`① 新增 ${newKeys.length} 条：作用域下译文与登记值一致 ${newKeys.length - bad1.length} / ${newKeys.length}`);
if (bad1.length) console.log("   不一致：", bad1.slice(0, 8));

/* ── ② 已有条目零行为变化：改前**所有**表的每一个键 ── */
const probe = new Set();
for (const n of NAMES) {
  const t = OLD[n];
  if (t && !Array.isArray(t)) for (const k of Object.keys(t)) probe.add(k);
}
const bad2 = [];
for (const k of probe) {
  const a1 = OLD.translateText(k), b1 = NEW.translateText(k);
  const a2 = OLD.translateText(k, OLD.TOKEN_MAKER_WINDOW_UI), b2 = NEW.translateText(k, NEW.TOKEN_MAKER_WINDOW_UI);
  const a3 = OLD.translateText(k, OLD.EMBER_DIALOG_UI), b3 = NEW.translateText(k, NEW.EMBER_DIALOG_UI);
  const a4 = OLD.translateText(k, OLD.MOOD_PANEL), b4 = NEW.translateText(k, NEW.MOOD_PANEL);
  if (a1 !== b1 || a2 !== b2 || a3 !== b3 || a4 !== b4) bad2.push([k, [a1, b1], [a2, b2], [a3, b3], [a4, b4]]);
}
console.log(`② 已有条目零行为变化：抽 ${probe.size} 个既有键 × 4 条通道，行为变了的 ${bad2.length} 条`);
if (bad2.length) console.log("   变了的：", bad2.slice(0, 8));

/* ── ③ 作用域实测：新键在全局通道一条都不许命中 ── */
const bad3 = newKeys.filter(k => NEW.translateText(k) !== k);
console.log(`③ 作用域实测：${newKeys.length} 条新键走全局通道（不传 extra），命中的 ${bad3.length} 条`);
if (bad3.length) console.log("   漏到全局的：", bad3.slice(0, 8));

/* ── ③b 作用域实测（另三张作用域表也不许吃到）── */
const bad3b = [];
for (const tbl of ["EMBER_DIALOG_UI", "MOOD_PANEL", "CHAT_UI", "WEATHER", "EMBER_WINDOW_UI"]) {
  for (const k of newKeys) if (NEW.translateText(k, NEW[tbl]) !== k) bad3b.push([tbl, k]);
}
console.log(`③b 新键在另 5 张作用域表下命中的 ${bad3b.length} 条`);
if (bad3b.length) console.log("   ", bad3b.slice(0, 8));

const pass = !bad1.length && !bad2.length && !bad3.length && !bad3b.length;
fs.writeFileSync(F_OUT, JSON.stringify({ dOld, dNew, n: newKeys.length, bad1, bad2, bad3, bad3b, pass }, null, 1), "utf8");
console.log(pass ? "\n验收 PASS（①②③③b 四段全绿）" : "\n验收 FAIL");
process.exit(pass ? 0 : 1);
