/**
 * 拿**发布中的** ember-hardcoded-cn.mjs 本体去判每条候选串「已覆盖 / 仍缺口」。
 *
 * 不复制第二份表：造一份 harness 副本，只追加 `export`，函数体一字不改
 * （与 3-常用脚本/qa/translate_cases_runner.mjs 同法）。
 *
 * 判「已覆盖」的三条通道：
 *   ① 上游 lang/en.json 的值（i18n 通道，我们的 lang/cn.json 已 1:1 覆盖）
 *   ② translateText(s) 或 translateText(s, 全部作用域表合并) 改动了它
 *   ③ translateNotification(s) 改动了它
 *
 * 用法：node coverage.mjs <candidates.json> <out.json>
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const SRC = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
const EN = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/lang/en.json";
const STUB_IMPORT = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";

const die = (m) => { console.error("coverage: " + m); process.exit(2); };

let src = fs.readFileSync(SRC, "utf8");
if (!src.includes(STUB_IMPORT)) die("找不到要打桩的 import 行");

// 需要导出的作用域表 —— 每个名字都先确认顶层声明存在，找不到当场失败。
const SCOPED = ["EMBER_WINDOW_UI", "DIALOG_UI", "EMBER_DIALOG_UI", "CHAT_UI", "ATTUNEMENT_TAB",
                "MOOD_PANEL", "SETTINGS_UI", "SCENE_CONTROL_UI", "NOTE_TYPES", "WEATHER",
                "LANGUAGES", "LANGUAGE_CATEGORIES", "SOUNDSCAPE_GROUPS", "ARRANGEMENT_LEAVES",
                "MOODS", "ATTUNEMENTS", "MOON_NAMES", "ATTUNEMENT_ITEM_NAMES", "SCROLLING_TEXT",
                "DIALOG_TITLES", "PROSEMIRROR_BLOCK_TITLES", "VISTA_PLACEMENT_EN", "EXACT"];
for (const n of SCOPED) {
  if (!new RegExp("\\nconst " + n + " = ").test(src)) die(`找不到 ${n} 的顶层声明`);
}
if (!/\nfunction translateNotification\(/.test(src)) die("找不到 translateNotification");

const harness = path.join(process.cwd(), "_harness.mjs");
fs.writeFileSync(harness,
  "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
  + src.replace(STUB_IMPORT,
      "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
  + `\nexport { translateNotification, ${SCOPED.join(", ")} };\n`,
  "utf8");

const M = await import(pathToFileURL(harness).href);
const { translateText, translateNotification } = M;
if (typeof translateText !== "function") die("没导出 translateText");

// 合并全部作用域表 —— 判「覆盖」时口径要最宽：只要**任何一条通道**能翻它，就不算缺口。
const ALL = {};
for (const n of SCOPED) Object.assign(ALL, M[n] ?? {});
if (Object.keys(ALL).length < 300) die(`合并作用域表只有 ${Object.keys(ALL).length} 条，明显不对`);

// i18n 通道
const enJson = JSON.parse(fs.readFileSync(EN, "utf8"));
const enVals = new Set();
(function walk(o) {
  if (typeof o === "string") { enVals.add(o.trim()); return; }
  if (o && typeof o === "object") for (const v of Object.values(o)) walk(v);
})(enJson);
if (enVals.size < 400) die(`lang/en.json 只抽到 ${enVals.size} 个值，明显不对`);

/* --------------------------------------------- 前置自证：切对条数 + 改对地方 ---- */
// (1) 已知**已覆盖**的串必须判成 covered，并且要点得出是哪条通道
const POS = [
  ["Ember Token Maker", "scoped"], ["Reset All", "scoped"], ["Show Tracks", "scoped"],
  ["Mine Cart Destination", "exact"], ["Select Destination", "exact"],
  ["Elevator Destination", "exact"], ["Ember: Teleport Destination", "exact"],
  ["Ember Ancestry", "i18n"], ["Attunement: Aura", "prefixed"],
];
// (2) 已知**未覆盖**的串必须判成 gap
const NEG = ["Hair Roots", "Loading Zone", "Ooze Farm", "Clear Custom Offsets", "Allow Restricted Parts"];

function classify(s) {
  if (enVals.has(s)) return "i18n";
  if (translateText(s) !== s) return "global";
  if (translateText(s, ALL) !== s) return "scoped";
  if (translateNotification(s) !== s) return "notify";
  return null;
}
const bad = [];
for (const [s] of POS) if (classify(s) === null) bad.push(`阳性对照判成缺口了：${JSON.stringify(s)}`);
for (const s of NEG) { const c = classify(s); if (c !== null) bad.push(`阴性对照判成已覆盖(${c})：${JSON.stringify(s)}`); }
if (bad.length) { for (const b of bad) console.error("  " + b); die("覆盖判据前置自证失败"); }
console.log(`覆盖判据前置自证通过：阳性 ${POS.length}/${POS.length}，阴性 ${NEG.length}/${NEG.length}`);

/* --------------------------------------------------------------------- 主跑 */
const rows = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const gaps = [], covered = [];
for (const r of rows) {
  const c = classify(r.s);
  if (c === null) gaps.push(r); else { r.by = c; covered.push(r); }
}
fs.writeFileSync(process.argv[3], JSON.stringify(gaps, null, 1), "utf8");
const byCh = {};
for (const r of covered) byCh[r.by] = (byCh[r.by] || 0) + 1;
console.log(`候选 ${rows.length} → 已覆盖 ${covered.length}（${JSON.stringify(byCh)}）／缺口 ${gaps.length}`);
