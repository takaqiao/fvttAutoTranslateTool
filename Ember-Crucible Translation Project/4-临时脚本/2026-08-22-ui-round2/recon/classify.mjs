/**
 * 拿**发布中的** ember-hardcoded-cn.mjs 本体判每条候选串走哪条通道 / 是不是缺口。
 * 造 harness 副本，只追加 export，函数体一字不改（同 2026-08-18-ui-gaps/coverage.mjs）。
 *
 * 用法：node classify.mjs <candidates.json(对象)> <out.json>
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const SRC = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
const EN = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/lang/en.json";
const STUB_IMPORT = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const die = (m) => { console.error("classify: " + m); process.exit(2); };

let src = fs.readFileSync(SRC, "utf8");
if (!src.includes(STUB_IMPORT)) die("找不到要打桩的 import 行");

const SCOPED = ["EMBER_WINDOW_UI", "DIALOG_UI", "EMBER_DIALOG_UI", "CHAT_UI", "ATTUNEMENT_TAB",
                "MOOD_PANEL", "SETTINGS_UI", "SCENE_CONTROL_UI", "NOTE_TYPES", "WEATHER",
                "LANGUAGES", "LANGUAGE_CATEGORIES", "SOUNDSCAPE_GROUPS", "ARRANGEMENT_LEAVES",
                "MOODS", "ATTUNEMENTS", "MOON_NAMES", "ATTUNEMENT_ITEM_NAMES", "SCROLLING_TEXT",
                "DIALOG_TITLES", "PROSEMIRROR_BLOCK_TITLES", "VISTA_PLACEMENT_EN", "EXACT",
                "TOKEN_MAKER_UI", "TOKEN_MAKER_WINDOW_UI"];
const EXTRA = ["DIALOG_TITLE_PATTERNS", "DIALOG_TITLE_I18N", "PREFIXED", "PATTERNS", "NOTIFICATION_PATTERNS", "NOTIFICATIONS"];
for (const n of [...SCOPED, ...EXTRA]) {
  if (!new RegExp("\\nconst " + n + " = ").test(src)) die(`找不到 ${n} 的顶层声明`);
}
if (!/\nfunction translateNotification\(/.test(src)) die("找不到 translateNotification");

const harness = path.join(process.cwd(), "_harness.mjs");
fs.writeFileSync(harness,
  "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
  + src.replace(STUB_IMPORT,
      "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
  + `\nexport { translateNotification, ${[...SCOPED, ...EXTRA].join(", ")} };\n`,
  "utf8");

const M = await import(pathToFileURL(harness).href);
const { translateText, translateNotification } = M;
if (typeof translateText !== "function") die("没导出 translateText");

const ALL = {};
for (const n of SCOPED) Object.assign(ALL, M[n] ?? {});
if (Object.keys(ALL).length < 300) die(`合并作用域表只有 ${Object.keys(ALL).length} 条，明显不对`);

const enJson = JSON.parse(fs.readFileSync(EN, "utf8"));
const enVals = new Set();
(function walk(o) {
  if (typeof o === "string") { enVals.add(o.trim()); return; }
  if (o && typeof o === "object") for (const v of Object.values(o)) walk(v);
})(enJson);
if (enVals.size < 400) die(`lang/en.json 只抽到 ${enVals.size} 个值，明显不对`);

function classify(s) {
  if (enVals.has(s)) return "i18n";
  if (translateText(s) !== s) return "global";
  if (translateText(s, ALL) !== s) return "scoped";
  if (translateNotification(s) !== s) return "notify";
  return null;
}

/* ---- 前置自证 A：切对条数（阳性/阴性对照全数命中）---- */
const POS = [
  ["Mine Cart Destination", "global"], ["Forwards", "scoped"], ["Backwards", "scoped"],
  ["Activate this mine cart with no passenger?", "global"],
  ["No destinations are currently reachable. Adjust the track levers and try again.", "global"],
  ["Interact With Objects", "i18n"], ["Tar Pit", "global"],
];
const NEG = ["Zzzz Not A Real String", "Loading Zone Xyzzy"];
const bad = [];
for (const [s, want] of POS) { const c = classify(s); if (c === null) bad.push(`阳性对照判成缺口：${JSON.stringify(s)}`); }
for (const s of NEG) { const c = classify(s); if (c !== null) bad.push(`阴性对照判成已覆盖(${c})：${JSON.stringify(s)}`); }
if (bad.length) { for (const b of bad) console.error("  " + b); die("前置自证失败"); }
console.log(`前置自证通过：阳性 ${POS.length}/${POS.length}，阴性 ${NEG.length}/${NEG.length}`);

/* ---- 主跑 ---- */
const rows = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const res = {};
const byCh = {};
for (const [s, meta] of Object.entries(rows)) {
  const probe = s.startsWith("TPL:") ? s.slice(4) : s;
  const c = classify(probe);
  res[s] = { ...meta, channel: c ?? "GAP", cn: c ? (translateText(probe, ALL) !== probe ? translateText(probe, ALL) : (enVals.has(probe) ? "(i18n)" : translateText(probe))) : null };
  byCh[c ?? "GAP"] = (byCh[c ?? "GAP"] || 0) + 1;
}
fs.writeFileSync(process.argv[3], JSON.stringify(res, null, 1), "utf8");
console.log(`候选 ${Object.keys(rows).length} →`, JSON.stringify(byCh));
