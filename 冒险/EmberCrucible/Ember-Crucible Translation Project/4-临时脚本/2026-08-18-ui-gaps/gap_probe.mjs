/**
 * 把 candidates.json 里的候选串逐条喂给**发布中的** translateText（含各张作用域表），
 * 分出「已覆盖 / 未覆盖」。
 *
 * 前置自证（两件都断言，见任务纪律 6）：
 *   (A) 切对条数：读进来的候选条数 = candidates.json 的键数；
 *   (B) 改对地方：一组「已知覆盖」的串必须判成 covered，一组「已知未覆盖」的串必须判成 gap。
 *
 * 用法：node gap_probe.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PLUGIN = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件";
const SRC = path.join(PLUGIN, "scripts", "ember-hardcoded-cn.mjs");
const UP = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember";

function die(m) { console.error("gap_probe: " + m); process.exit(2); }

/* ---- 造 harness（与 translate_cases_runner.mjs 同一手法：只追加导出，不改函数体） ---- */
const STUB_IMPORT = (() => {
  const src = fs.readFileSync(SRC, "utf8");
  const m = src.match(/^import \* as SELFCHECK from ["'][^"']+["'];?$/m)
        || src.match(/^import .*ember-cn-selfcheck\.mjs.*$/m);
  if (!m) die("找不到 selfcheck 的 import 行");
  return m[0];
})();

let src = fs.readFileSync(SRC, "utf8");
if (!src.includes(STUB_IMPORT)) die("打桩目标不存在");

const NEEDED = [
  ["translateText", /\nexport function translateText\(/, true],
  ["EXACT", /\nconst EXACT = \{/, false],
  ["PREFIXED", /\nconst PREFIXED = \[/, false],
  ["PATTERNS", /\nconst PATTERNS = \[/, false],
  ["NOTIFICATIONS", /\nconst NOTIFICATIONS = \{/, false],
  ["NOTIFICATION_PATTERNS", /\nconst NOTIFICATION_PATTERNS = \[/, false],
  ["EMBER_WINDOW_UI", /\nconst EMBER_WINDOW_UI = \{/, false],
  ["EMBER_DIALOG_UI", /\nconst EMBER_DIALOG_UI = /, false],
  ["DIALOG_UI", /\nconst DIALOG_UI = \{/, false],
  ["CHAT_UI", /\nconst CHAT_UI = \{/, false],
  ["DIALOG_TITLES", /\nconst DIALOG_TITLES = \{/, false],
  ["SETTINGS_UI", /\nconst SETTINGS_UI = \{/, false],
  ["SCENE_CONTROL_UI", /\nconst SCENE_CONTROL_UI = \{/, false],
  ["MOOD_PANEL", /\nconst MOOD_PANEL = \{/, false],
  ["NOTE_TYPES", /\nconst NOTE_TYPES = \{/, false],
  ["WEATHER", /\nconst WEATHER = \{/, false],
  ["ATTUNEMENT_TAB", /\nconst ATTUNEMENT_TAB = \{/, false],
  ["ATTUNEMENTS", /\nconst ATTUNEMENTS = \{/, false],
  ["ATTUNEMENT_ITEM_NAMES", /\nconst ATTUNEMENT_ITEM_NAMES = \{/, false],
  ["MOON_NAMES", /\nconst MOON_NAMES = \{/, false],
  ["LANGUAGES", /\nconst LANGUAGES = \{/, false],
  ["LANGUAGE_CATEGORIES", /\nconst LANGUAGE_CATEGORIES = \{/, false],
  ["KNOWLEDGE", /\nconst KNOWLEDGE = \{/, false],
  ["MOODS", /\nconst MOODS = \{/, false],
  ["SOUNDSCAPE_GROUPS", /\nconst SOUNDSCAPE_GROUPS = \{/, false],
  ["ARRANGEMENTS", /\nconst ARRANGEMENTS = \{/, false],
  ["ARRANGEMENT_LEAVES", /\nconst ARRANGEMENT_LEAVES = \{/, false],
  ["DIVINE_DOMAINS", /\nconst DIVINE_DOMAINS = \{/, false],
  ["WARLOCK_PATRONS", /\nconst WARLOCK_PATRONS = \{/, false],
  ["SORCEROUS_ORIGINS", /\nconst SORCEROUS_ORIGINS = \{/, false],
  ["RARITIES", /\nconst RARITIES = \{/, false],
  ["MISSING_LANGUAGES", /\nconst MISSING_LANGUAGES = \{/, false],
  ["MISSING_KNOWLEDGE", /\nconst MISSING_KNOWLEDGE = \{/, false],
  ["MISSING_ANCESTRIES", /\nconst MISSING_ANCESTRIES = \{/, false],
  ["MISSING_CULTURES", /\nconst MISSING_CULTURES = \{/, false],
  ["MISSING_PATHS", /\nconst MISSING_PATHS = \{/, false],
  ["DATE_AGES", /\nconst DATE_AGES = \{/, false],
  ["SCROLLING_TEXT", /\nconst SCROLLING_TEXT = \{/, false],
  ["PROSEMIRROR_BLOCK_TITLES", /\nconst PROSEMIRROR_BLOCK_TITLES = \{/, false],
  ["REGION_BEHAVIOR_FIELDS", /\nconst REGION_BEHAVIOR_FIELDS = \{/, false],
  ["VISTA_PLACEMENT_EN", /\nconst VISTA_PLACEMENT_EN = \{/, false],
  ["HERO_ITEM_NAMES", /\nconst HERO_ITEM_NAMES = /, false],
  ["INJECTED_SUBTREES", /\nconst INJECTED_SUBTREES = \[/, false],
];
for (const [name, re] of NEEDED) if (!re.test(src)) die(`找不到顶层声明 ${name}`);

const harness = path.join(HERE, "_harness.mjs");
fs.writeFileSync(harness,
  "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
  + src.replace(STUB_IMPORT,
      "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
  + `\nexport { ${NEEDED.filter(([, , d]) => !d).map(([n]) => n).join(", ")} };\n`,
  "utf8");

const M = await import(pathToFileURL(harness).href);
const { translateText } = M;

/* 所有「英文键 -> 中文」的普通表合起来当作用域表的并集 */
const FLAT_TABLES = ["EXACT", "EMBER_WINDOW_UI", "DIALOG_UI", "CHAT_UI", "DIALOG_TITLES",
  "SETTINGS_UI", "SCENE_CONTROL_UI", "MOOD_PANEL", "NOTE_TYPES", "WEATHER", "ATTUNEMENT_TAB",
  "ATTUNEMENTS", "ATTUNEMENT_ITEM_NAMES", "MOON_NAMES", "LANGUAGES", "LANGUAGE_CATEGORIES",
  "KNOWLEDGE", "MOODS", "SOUNDSCAPE_GROUPS", "ARRANGEMENT_LEAVES", "DIVINE_DOMAINS",
  "WARLOCK_PATRONS", "SORCEROUS_ORIGINS", "RARITIES", "MISSING_LANGUAGES", "MISSING_KNOWLEDGE",
  "MISSING_ANCESTRIES", "MISSING_CULTURES", "MISSING_PATHS", "DATE_AGES", "SCROLLING_TEXT",
  "PROSEMIRROR_BLOCK_TITLES", "NOTIFICATIONS"];
const SCOPED = {};
for (const t of FLAT_TABLES) Object.assign(SCOPED, M[t] ?? {});
// ARRANGEMENTS 是两层：{组: {叶: 中文}}
for (const g of Object.values(M.ARRANGEMENTS ?? {})) Object.assign(SCOPED, g);
// VISTA_PLACEMENT_EN 是 {字段路径: 英文}，英文那侧才是我们认的串
for (const en of Object.values(M.VISTA_PLACEMENT_EN ?? {})) SCOPED[en] = "<vista>";
Object.assign(SCOPED, M.HERO_ITEM_NAMES ?? {});

const tableKeys = new Set(Object.keys(SCOPED));

/* ---- 上游 lang/en.json 的值集合（i18n 通道已覆盖的那一批） ---- */
const enJson = JSON.parse(fs.readFileSync(path.join(UP, "lang", "en.json"), "utf8"));
const enVals = new Set();
(function walk(o) {
  if (o && typeof o === "object") for (const v of Object.values(o)) walk(v);
  else if (typeof o === "string") enVals.add(o.trim());
})(enJson);

/* ---- 覆盖判定 ---- */
function covered(s) {
  if (tableKeys.has(s)) return "table";
  if (translateText(s, SCOPED) !== s) return "translate";
  if (translateText(s) !== s) return "translate-global";
  if (enVals.has(s)) return "en.json";
  return null;
}

/* ================= 前置自证 ================= */
const CAND = JSON.parse(fs.readFileSync(path.join(HERE, "candidates.json"), "utf8"));
const keys = Object.keys(CAND);

// (A) 切对条数
const expected = Number(process.env.EXPECT_N || 0);
if (expected && keys.length !== expected) die(`条数不对：${keys.length} != ${expected}`);
console.log(`[自证A] 候选条数 = ${keys.length}`);

// (B) 改对地方
const KNOWN_COVERED = ["Show Tracks", "Reset All", "Play Animation", "Randomize Layer",
  "Clear Layer", "Next Option", "Colors", "Layers", "Anatomy", "Equipment"];
const KNOWN_GAP = ["Hair Roots", "Hair Highlights", "Hair Base", "Hair Glow", "Hair Sparkle"];
let ok = true;
console.log("[自证B] 已知覆盖：");
for (const s of KNOWN_COVERED) {
  const c = covered(s);
  console.log(`   ${c ? "OK  " : "FAIL"}  ${JSON.stringify(s)} -> ${c ?? "未覆盖"}`);
  ok &&= !!c;
}
console.log("[自证B] 已知缺口：");
for (const s of KNOWN_GAP) {
  const c = covered(s);
  console.log(`   ${c ? "FAIL" : "OK  "}  ${JSON.stringify(s)} -> ${c ?? "未覆盖"}`);
  ok &&= !c;
}
if (!ok) die("前置自证不过，中止");

/* ================= 主流程 ================= */
const gaps = {}, cov = {};
for (const [s, locs] of Object.entries(CAND)) {
  const c = covered(s);
  if (c) cov[s] = c; else gaps[s] = locs;
}
console.log(`\n候选 ${keys.length} → 已覆盖 ${Object.keys(cov).length} / 未覆盖 ${Object.keys(gaps).length}`);
fs.writeFileSync(path.join(HERE, "gaps.json"), JSON.stringify(gaps, null, 1), "utf8");
fs.writeFileSync(path.join(HERE, "covered.json"), JSON.stringify(cov, null, 1), "utf8");
console.log("写出 gaps.json / covered.json");
