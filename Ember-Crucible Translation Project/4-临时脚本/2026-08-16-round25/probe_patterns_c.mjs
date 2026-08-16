/**
 * 第二十五轮 ③ 号工作面的 PATTERNS 双向验探针。
 *
 * ⚠ 本文件是**手写落盘**的，不是 bash heredoc / python 改写脚本生成的 ——
 *   正则里的 \b \s 一旦经改写脚本转手就会失效（本项目登记的空转形态 (f)）。
 *
 * 做三件事：
 *   ① 正例：上游**真会产出**的串仍然翻得动。`Result of` 那条按 enrichCriticalResult
 *      （ember.mjs:22905-22913）的定义域穷举；音景那条拿**从 ember.mjs 里现抠出来的**
 *      全部编排名（不是抄我们自己的表）逐条过一遍 —— 这才叫双向验。
 *   ② 反例：复核那 13 条一条都不许被误翻。
 *   ③ 回归：本轮没碰的那些 PATTERNS 分支各测一条，证明只收紧了该收紧的两条。
 *
 * 用法：node probe_patterns_c.mjs
 * 退出码 0 = 全过。
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const DATA = "C:/Users/Taka/AppData/Local/FoundryVTT/Data";
const PROJ = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const HERE = path.join(PROJ, "4-临时脚本/2026-08-16-round25");
const SRC = path.join(PROJ, "1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs");

/* ── 造 harness：把 SELFCHECK / Hooks 桩掉，只留 translateText ── */
const harness = path.join(HERE, "_hc_patterns_c.mjs");
{
  const src = fs.readFileSync(SRC, "utf8");
  const IMPORT_LINE = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
  if (!src.includes(IMPORT_LINE)) throw new Error("找不到 SELFCHECK 的 import 行，转换中止");
  const stub = "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n";
  fs.writeFileSync(harness,
    stub + src.replace(IMPORT_LINE,
      "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };"),
    "utf8");
}
const { translateText } = await import(pathToFileURL(harness).href);

let pass = 0, fail = 0;
const eq = (name, got, want) => {
  const ok = got === want;
  console.log(`${ok ? "  ok  " : "  FAIL"} ${name}${ok ? "" : `\n         得到 ${JSON.stringify(got)}\n         应为 ${JSON.stringify(want)}`}`);
  ok ? pass++ : fail++;
};
/** 反例：整串**必须原样返回**（一个字都不许动） */
const untouched = (s) => eq(`反例不动：${JSON.stringify(s)}`, translateText(s), s);

/* ================================================================== */
/*  ① 正例：Result of —— 按上游定义域穷举                              */
/* ================================================================== */
console.log("\n── ① 正例 · Result of（ember.mjs:22909/22912，dc 为整数，只拼 `${dc-5}-` / `${dc+5}+`）──");
// dc 取 0 / 5 / 8 / 18 / 100 / -3（Number.isInteger 允许负数），两支各算一遍
for (const dc of [0, 5, 8, 18, 100, -3]) {
  eq(`Result of ${dc - 5}-`, translateText(`Result of ${dc - 5}-`), `结果：${dc - 5}-`);
  eq(`Result of ${dc + 5}+`, translateText(`Result of ${dc + 5}+`), `结果：${dc + 5}+`);
}

/* ================================================================== */
/*  ② 正例：音景 —— 编排名从 ember.mjs 现抠，不抄我们自己的表          */
/* ================================================================== */
console.log("\n── ② 正例 · 音景（编排名取自上游 ember.mjs，逐条过 translateText）──");

function blockAt(s, openIdx) {
  let d = 0, q = null;
  for (let i = openIdx; i < s.length; i++) {
    const c = s[i];
    if (q) { if (c === "\\") { i++; continue; } if (c === q) q = null; continue; }
    if (c === '"' || c === "'" || c === "`") { q = c; continue; }
    if (c === "/" && s[i + 1] === "/") { i = s.indexOf("\n", i); if (i < 0) break; continue; }
    if (c === "/" && s[i + 1] === "*") { i = s.indexOf("*/", i) + 1; continue; }
    if (c === "{") d++;
    else if (c === "}") { d--; if (d === 0) return s.slice(openIdx, i + 1); }
  }
  return null;
}

const emberSrc = fs.readFileSync(path.join(DATA, "modules/ember/scripts/ember.mjs"), "utf8");
const reg = emberSrc.match(/var soundscapes=\/\*#__PURE__\*\/Object\.freeze\(\{__proto__:null,([^}]*)\}\)/);
if (!reg) { console.log("  FAIL 抠不到 soundscapes 注册表"); fail++; }
const pairs = reg[1].split(",").map(s => s.trim()).filter(Boolean)
  .map(s => { const [k, v] = s.split(":"); return { key: k, varName: v }; });
console.log(`  注册表 ${pairs.length} 个音景`);

const upstreamArr = new Set();
let unresolved = 0;
for (const { varName } of pairs) {
  const re = new RegExp(`(?:^|[;}])var ${varName.replace(/\$/g, "\\$")} = \\{`, "m");
  const m = re.exec(emberSrc);
  if (!m) { unresolved++; continue; }
  const body = blockAt(emberSrc, emberSrc.indexOf("{", m.index));
  if (!body) { unresolved++; continue; }
  const aIdx = body.search(/\n  arrangements: \{/);
  if (aIdx < 0) continue;
  const aBody = blockAt(body, body.indexOf("{", aIdx));
  for (const mm of aBody.matchAll(/\n      label: "([^"]*)"/g)) upstreamArr.add(mm[1]);
}
eq("所有音景都解析出来了（unresolved 必须为 0）", unresolved, 0);
console.log(`  上游唯一编排名 ${upstreamArr.size} 条`);
if (upstreamArr.size < 100) { console.log("  FAIL 抠出来的编排名太少，探针多半空转了"); fail++; }

/**
 * 收紧之后**预期翻不动**的编排名，共 8 条，逐条有据（不是「测不过就加进来」）：
 *   · `Seven Sails` —— 第二十二轮**裁定故意留英**（两仓库 43115 条英文叶 en-hits=0，
 *     连 `Sails` 都 0 命中，查不到这个名字指什么，宁可露英文也不猜）。见 ARRANGEMENTS 表注释。
 *   · `Events` —— 属音景 `events`（label "Ember Events", type "events"）；
 *   · `Clear` / `Drizzle` / `Rain` / `Thunderstorm` / `Arcane Fog` / `Mayis Storm`
 *     —— 属音景 `weather`（label "Ember Weather", type "weather"）。
 *     这两个音景的 type 不是 music/environment，**不进播放列表侧栏那两个下拉**
 *     （ember.mjs:15916-15922 只取 music(41)+environment(1) 共 42 个），
 *     所以第二十一/二十二轮建 ARRANGEMENTS 表时本来就不在全集里（212 vs 本探针的 219）。
 * 断言写成**恰好等于这 8 条**：多一条少一条都失败，防止「反正有豁免」把新缺口盖过去。
 */
const EXPECT_UNCOVERED = new Set([
  "Seven Sails", "Events",
  "Clear", "Drizzle", "Rain", "Thunderstorm", "Arcane Fog", "Mayis Storm"
]);
let arrOk = 0; const arrBad = [];
for (const label of upstreamArr) {
  for (const ch of ["Music", "Environment"]) {
    const got = translateText(`${ch}: ${label}`);
    if (got !== `${ch}: ${label}` && got.startsWith(ch === "Music" ? "音乐：" : "环境音：")) arrOk++;
    else arrBad.push(label);
  }
}
const bad = [...new Set(arrBad)].sort();
eq(`上游 ${upstreamArr.size} 条编排名里翻不动的**恰好**是登记的那 8 条`,
   JSON.stringify(bad), JSON.stringify([...EXPECT_UNCOVERED].sort()));
if (JSON.stringify(bad) !== JSON.stringify([...EXPECT_UNCOVERED].sort())) {
  console.log(`         多出来的：${JSON.stringify(bad.filter(x => !EXPECT_UNCOVERED.has(x)))}`);
  console.log(`         少掉的：  ${JSON.stringify([...EXPECT_UNCOVERED].filter(x => !bad.includes(x)))}`);
}
console.log(`  实际翻动 ${arrOk} 条 / ${(upstreamArr.size - bad.length) * 2} 条应翻动`);

// Reset 那一支（ember.mjs:16255）与 mood 尾巴那一支（:16267-16268）
eq("Music: Reset", translateText("Music: Reset"), "音乐：重置");
eq("Environment: Reset", translateText("Environment: Reset"), "环境音：重置");
eq("Music: Ancient Ruins (Tension)", translateText("Music: Ancient Ruins (Tension)"), "音乐：远古遗迹（紧张）");
eq("Music: Ankarist Theme (Calm)", translateText("Music: Ankarist Theme (Calm)"), "音乐：安卡里斯特的主题（平静）");
// 第三支 `Music Mood: X` 走 PREFIXED，不该被本条正则吃掉
eq("Music Mood: Calm 仍走 PREFIXED", translateText("Music Mood: Calm"), "音乐氛围：平静");

/* ================================================================== */
/*  ③ 反例：复核造的 13 条，一条都不许被误翻                            */
/* ================================================================== */
console.log("\n── ③ 反例 · 13 条（含复核点名的 3 条误翻）──");
[
  "Result of the investigation was inconclusive",   // 复核点名 ①
  "Music: my custom playlist",                      // 复核点名 ②
  "Environment: Rain",                              // 复核点名 ③（Rain 只在 WEATHER 里，不是编排名）
  "Result of the roll",
  "Result of a long and fruitless search",
  "Result of 13 plus",
  "Music: Session 3 Playlist",
  "Environment: Forest ambience",
  "Music: The Beatles",
  "Environment: Dust Storm",
  "Environment: Tempest",
  "Music: Rain (Calm)",
  "Result of two checks"
].forEach(untouched);

/* ================================================================== */
/*  ④ 回归：本轮没碰的 PATTERNS 分支各测一条                            */
/* ================================================================== */
console.log("\n── ④ 回归 · 未改动的分支 ──");
eq("+2 Boons", translateText("+2 Boons"), "+2 恩惠骰");
eq("-6 Banes", translateText("-6 Banes"), "-6 祸骰");
eq("Age of Beasts - 12 Years Ago", translateText("Age of Beasts - 12 Years Ago"), "野兽时代 · 12 年前");
eq("Age of the Tower - Current Year", translateText("Age of the Tower - Current Year"), "高塔时代 · 本年");
eq("Day 43 - 12:00", translateText("Day 43 - 12:00"), "第 43 天 - 12:00");
eq("Day 43", translateText("Day 43"), "第 43 天");
eq("Outcome 3", translateText("Outcome 3"), "结果 3");
eq("5 Others", translateText("5 Others"), "其他 5 项");
eq("Threat 12.5", translateText("Threat 12.5"), "威胁等级 12.5");
eq("Vantage Point: X", translateText("Vantage Point: Somewhere"), "制高点：Somewhere");
eq("Interactable: X", translateText("Interactable: Lever"), "可交互物：Lever");
eq("Breeze (4.5 mph)", translateText("Breeze (4.5 mph)"), "Breeze（4.5 英里/时）");
eq("Activate Attunement: Abyss", translateText("Activate Attunement: Abyss"), "激活同调：深渊");
eq("[[/language borel]]", translateText("[[/language borel]]").startsWith("语言："), true);
eq("Day, Generic 不许被吃（远景资源名）", translateText("Day, Generic"), "Day, Generic");
// EXACT / PREFIXED 通道抽样，证明 harness 是活的
eq("EXACT: Critical Success", translateText("Critical Success"), "大成功");
eq("PREFIXED: Attunement: Abyss", translateText("Attunement: Abyss"), "同调：深渊");

/* ================================================================== */
/*  ⑤ 旧实现回归对照：证明这组用例**不是空转**                          */
/* ================================================================== */
console.log("\n── ⑤ 旧实现回归对照（把收紧前的两条正则装回来跑同一批反例）──");
// 收紧前的两条（第二十四轮结束时的原样），手写、不经改写脚本
const OLD = [
  { re: /^Result of (.+)$/, cn: (m) => `结果：${m[1]}` },
  { re: /^(Music|Environment): (.+?)(?: \((Calm|Tension)\))?$/,
    cn: (m) => `${m[1] === "Music" ? "音乐" : "环境音"}：${m[2]}` }
];
const oldTranslate = (s) => {
  for (const { re, cn } of OLD) { const m = s.match(re); if (m) return cn(m); }
  return s;
};
const REVIEW_3 = [
  "Result of the investigation was inconclusive",
  "Music: my custom playlist",
  "Environment: Rain"
];
let oldBit = 0;
for (const s of REVIEW_3) {
  const o = oldTranslate(s), n = translateText(s);
  console.log(`  · ${JSON.stringify(s)}\n      旧 → ${JSON.stringify(o)}\n      新 → ${JSON.stringify(n)}`);
  if (o !== s) oldBit++;
}
eq("旧实现确实会咬这 3 条（否则本组用例是空转）", oldBit, 3);
eq("新实现一条都不咬", REVIEW_3.filter(s => translateText(s) !== s).length, 0);

// 代价也如实报：那 8 条上游编排名旧实现会把**前缀**译成中文，新实现整串留英
console.log("  收紧的代价（如实报，不藏）：");
for (const s of ["Music: Seven Sails", "Environment: Rain", "Music: Events"]) {
  console.log(`  · ${JSON.stringify(s)}  旧 → ${JSON.stringify(oldTranslate(s))}   新 → ${JSON.stringify(translateText(s))}`);
}

console.log(`\n合计：通过 ${pass} / 失败 ${fail}`);
process.exit(fail ? 1 : 0);
