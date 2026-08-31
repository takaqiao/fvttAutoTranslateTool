/**
 * 用例：B 档「已认到译文的合集」这条聚合数**必须点名到具体项**，并分「良性 / 可疑」。
 * ==============================================================================
 *
 * 为什么要有它（2026-08-17 第二次真实世界冒烟验证的跟进）
 * ------------------------------------------------------
 * 真实报告里这一行是：`✅ 已认到译文的合集（22 项）— 21/22 个合集有对应译文文件`。
 * **它没说那个「1」是谁。** 今天它是良性的（差的是 `crucible.crafting`，上游连目录都没有），
 * **但明天某个真包丢了译文，报出来长得一模一样。**
 *
 * ⚠ 三条纪律（本项目血泪，逐条对照）
 * ---------------------------------
 * ① **驱动的是面板真身**（`runSelfCheck` → `checkBabele`），判据一行都不抄 ——
 *    抄一份就等于测了个副本（登记的空转形态 (c)/(h)）。
 * ② **前置自证两件都做**：
 *    · 条数 ＝ 已知真值：夹具**不是手打的**，是从**真实的** `modules/ember/module.json`
 *      ＋ `systems/crucible/system.json` ＋ 两个插件仓 `compendium/cn/` 的实际文件列表现算出来的；
 *      跑用例之前先断言「该世界加载 22 个合集」「没有译文的恰好是 `crucible.crafting` 一个」，
 *      对不上**当场硬失败**，不往下跑。
 *    · 改对了地方：反向用例的每一处文本改动都断言**命中条数 ＝ 预期**，
 *      **且改动的偏移落在 `checkBabele()` 的源码区间内**（只数对条数不等于改对了地方）。
 * ③ **模拟输入的字段形状指得出上游契约出处**（形态 (h)）：
 *    · `babele.isTranslated(pack)` —— `modules/babele/script/babele.js:557`
 *      `@param {string} pack compendium name (ex. dnd5e.classes)`；
 *      `translation-session.js:111 isTranslated(collection)` → `translatedCompendiumFor(collection)`
 *      → `mapped-compendiums.js` 里以**字符串**为键的 Map。⇒ 桩**只认字符串**，
 *        面板要是又把 `CompendiumCollection` 对象喂进来（v1.1.23 修的正是这个），这里当场红。
 *    · `pack.collection` / `pack.metadata.{packageName,label,name,type}` —— Foundry
 *      `CompendiumCollection`，元数据逐字来自包清单里的 `packs[]` 条目（本探针就是读那两份清单）。
 *    · `pack.index` 的条目只给 `_id` / `name` —— `name` 在每种文档的 `compendiumIndexFields`
 *      里都有（面板 D 档那段注释已核过）；**不给 `pages` 之类运行时拿不到的字段**。
 *
 * 跑法：node probe_pack_naming.mjs   （非 0 退出 = 有用例没过）
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

/** 变异体（掏空判据的面板副本）落在**系统临时目录**，跑完整目录删掉。
 *  ⚠ 别落在产物目录里：那是 98 KB 一份的**面板副本**，留在仓里迟早被人（或某条按文件名扫的判据）
 *    当成真身读走。实测 node 在 Windows 上 `rmSync` 刚 import 过的 .mjs 并不总能真的删掉。 */
const MUT = fs.mkdtempSync(path.join(os.tmpdir(), "ec-mutant-"));

const PROJ = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const DATA = "C:/Users/Taka/AppData/Local/FoundryVTT/Data";
const PANEL = path.join(PROJ, "1-Ember汉化插件/scripts/ember-cn-selfcheck.mjs");
const OUT = path.join(PROJ, "4-临时脚本/2026-08-17-smoke2");

const log = [];
let failures = 0;
function say(s) { log.push(s); process.stdout.write(s + "\n"); }
function ok(name, cond, detail) {
  if (!cond) failures++;
  say(`${cond ? "PASS" : "FAIL"}  ${name}${detail ? " — " + detail : ""}`);
  return cond;
}
function hard(msg) { say("HARD-FAIL  " + msg); fs.writeFileSync(path.join(OUT, "probe_pack_naming.log"), log.join("\n"), "utf8"); process.exit(2); }

/* ────────────────────────────────────────────────────────────────
   0 · 夹具：**从真实数据现算**，不手打
   ──────────────────────────────────────────────────────────────── */
const emberManifest = JSON.parse(fs.readFileSync(path.join(DATA, "modules/ember/module.json"), "utf8"));
const crucibleManifest = JSON.parse(fs.readFileSync(path.join(DATA, "systems/crucible/system.json"), "utf8"));

/** 该世界系统是 crucible ⇒ 上游 ember 标了 `system: dnd5e` 的包不加载（Foundry 的包过滤）。 */
const WORLD_SYSTEM = "crucible";
const declared = [
  ...emberManifest.packs.map(p => ({ src: "ember", ...p })),
  ...crucibleManifest.packs.map(p => ({ src: "crucible", ...p })),
];
const loaded = declared.filter(p => !p.system || p.system === WORLD_SYSTEM
  || (Array.isArray(p.system) && p.system.includes(WORLD_SYSTEM)));

/** 我们仓里实际存在的译文文件（babele 认不认到译文，实质就是有没有这个文件 + mapping）。 */
const cnFiles = new Set([
  ...fs.readdirSync(path.join(PROJ, "1-Ember汉化插件/compendium/cn")),
  ...fs.readdirSync(path.join(PROJ, "2-Crucible汉化插件/compendium/cn")),
].filter(f => f.endsWith(".json")).map(f => f.slice(0, -5)));

const fixture = loaded.map(p => ({
  collection: `${p.src}.${p.name}`,
  src: p.src,
  label: p.label ?? p.name,
  type: p.type,
  /** 上游这个包在磁盘上到底有没有内容（目录不存在 = 空包）。 */
  diskDir: path.join(DATA, p.src === "ember" ? "modules/ember" : "systems/crucible", p.path ?? `packs/${p.name}`),
}));
for (const f of fixture) {
  f.hasTranslation = cnFiles.has(f.collection);
  f.upstreamExists = fs.existsSync(f.diskDir);
}

const untranslated = fixture.filter(f => !f.hasTranslation).map(f => f.collection);

/* ── 前置自证 ①：条数 ＝ 已知真值 ───────────────────────────── */
say("== 前置自证（对不上就不往下跑）==");
if (!ok("上游声明的合集共 26 个（ember 10 + crucible 16）",
  declared.length === 26 && emberManifest.packs.length === 10 && crucibleManifest.packs.length === 16,
  `实得 ${declared.length}（ember ${emberManifest.packs.length} + crucible ${crucibleManifest.packs.length}）`)) hard("上游包数与已知真值不符");
if (!ok("crucible 世界实际加载 22 个（ember 那 4 个 dnd5e 包不加载）",
  loaded.length === 22,
  `实得 ${loaded.length}；被过滤掉的：${declared.filter(p => !loaded.includes(p)).map(p => p.name).join(" · ")}`)) hard("加载包数与已知真值不符");
/* ── 前置自证 ②：内容对 —— 差的那个必须真的是 `crucible.crafting` ── */
if (!ok("这 22 个里**没有译文的恰好 1 个**，且就是 `crucible.crafting`",
  untranslated.length === 1 && untranslated[0] === "crucible.crafting",
  `实得 ${untranslated.length} 个：${untranslated.join(" · ") || "（无）"}`)) hard("没有译文的包与已知真值不符");
if (!ok("`crucible.crafting` 在上游**连目录都不存在**（⇒ 空包，index 恒 0）",
  fixture.find(f => f.collection === "crucible.crafting")?.upstreamExists === false,
  `packs/ 下实际有 ${fs.readdirSync(path.join(DATA, "systems/crucible/packs")).length} 个目录，声明 16 个`)) hard("crafting 目录状态与已知真值不符");
ok("另有 `ember.dnd5e-items` 没有译文，但它**不在这 22 个里**（所以那个「1」不是它）",
  !cnFiles.has("ember.dnd5e-items") && !fixture.some(f => f.collection === "ember.dnd5e-items"),
  "");
say("");

/* ────────────────────────────────────────────────────────────────
   1 · 桩：只桩 game / Hooks / babele，判据一行不抄
   ──────────────────────────────────────────────────────────────── */
/** @param {{collection:string,label:string}[]} specs @param {Set<string>} translated */
function makeGame(specs, translated, argTypes) {
  const packs = specs.map(s => ({
    collection: s.collection,
    documentName: s.type ?? "Item",
    metadata: {
      packageName: s.collection.split(".")[0],
      name: s.collection.split(".").slice(1).join("."),
      label: s.label,
      type: s.type ?? "Item",
    },
    // index：Foundry 里是 Collection（有 .size）。这里用数组 —— 面板的 `packIndexSize`
    // 两种都接；条目只给 `_id` / `name`（`compendiumIndexFields` 里确实有的字段）。
    index: (s.entries ?? []).map((n, i) => ({ _id: `id${i}`, name: n })),
  }));
  return {
    version: "14.366",
    world: { title: "探针世界", id: "probe" },
    system: { id: WORLD_SYSTEM, version: "0.10.1" },
    modules: {
      get(id) {
        if (id === "babele") return { active: true, version: "2.9.1" };
        if (id === "ember_cn_unofficial") return { active: true, version: "1.1.23" };
        if (id === "crucible-cn") return { active: true, version: "0.9.13" };
        if (id === "foundry_chn") return { active: true, version: "1.0.0" };
        return undefined;                     // 上游 ember 不装 ⇒ E/F 档报 skip，不碰 DOM
      },
    },
    packs,
    i18n: { translations: { PROBE: 1 }, localize: (k) => k },
    babele: {
      // ⚠ 逐字照 `babele.js:557` 的 `@param {string} pack`：**只认字符串**。
      isTranslated(pack) {
        argTypes.push(typeof pack);
        if (typeof pack !== "string") return false;   // 真库就是这个行为（Map 以字符串为键）
        return translated.has(pack);
      },
    },
  };
}

const CJK_NAMES = ["深渊 The Abyss", "余烬之心", "凯西安沙刀", "空洞之月", "指示物"];

/** 跑一次面板真身，返回「已认到译文的合集」那条 Check。 */
async function runCase(panelPath, specs, translated) {
  const argTypes = [];
  globalThis.Hooks = { once() {}, on() {} };
  globalThis.CONFIG = {};
  globalThis.game = makeGame(specs, translated, argTypes);
  const SC = await import(pathToFileURL(panelPath).href + `?v=${Math.random()}`);
  const checks = await SC.runSelfCheck({});
  const row = checks.find(c => c.name === "已认到译文的合集");
  if (!row) hard("面板没吐出「已认到译文的合集」这一行 —— 判据的锚点变了，本探针必须重锚");
  return { row, argTypes, checks };
}

/** 三个用例共用的夹具形状：22 个包，除 `withoutTranslation` 外都有译文。 */
function specsFor({ craftingEntries }) {
  return fixture.map(f => ({
    collection: f.collection,
    label: f.label,
    type: f.type,
    entries: f.collection === "crucible.crafting" ? craftingEntries : CJK_NAMES,
  }));
}
const translatedSet = new Set(fixture.filter(f => f.hasTranslation).map(f => f.collection));

/* ────────────────────────────────────────────────────────────────
   2 · 三个正向用例
   ──────────────────────────────────────────────────────────────── */
async function runAllCases(panelPath, tag) {
  const res = {};
  say(`== ${tag} ==`);

  // ① 只有空包没译文 → 报「预期」，且**仍然是 21/22，分母不许动**
  {
    const { row, argTypes } = await runCase(panelPath, specsFor({ craftingEntries: [] }), translatedSet);
    const d = row.detail ?? "";
    const items = (row.items ?? []).join("\n");
    res.case1 = {
      status: row.status, checked: row.checked,
      ratio2122: d.includes("21/22"), noFakeFull: !d.includes("22/22"),
      named: items.includes("crucible.crafting"),
      expected: items.includes("预期"),
      argsAllString: argTypes.length === 22 && argTypes.every(t => t === "string"),
    };
    ok("①-a 状态是 ok（空包没译文是良性）", row.status === "ok", `实得 ${row.status}`);
    ok("①-b 报文里仍是 **21/22**（分母一条没动）", res.case1.ratio2122, d.slice(0, 60));
    ok("①-c 报文里**没有**假的 22/22", res.case1.noFakeFull, "");
    ok("①-d 分母 checked 仍是 22", row.checked === 22, `实得 ${row.checked}`);
    ok("①-e **点名**到 `crucible.crafting`", res.case1.named, items.slice(0, 120));
    ok("①-f 点名之后标明是「预期」（上游空包）", res.case1.expected, "");
    ok("①-g 传给 `isTranslated` 的 22 次全是**字符串**（v1.1.23 那个假警报的回归闸）",
      res.case1.argsAllString, `实得 ${[...new Set(argTypes)].join("/")} × ${argTypes.length}`);
  }

  // ② 一个**有内容**的包没译文 → 必须点名并提醒
  {
    const { row } = await runCase(panelPath, specsFor({ craftingEntries: CJK_NAMES }), translatedSet);
    const d = row.detail ?? "";
    const items = (row.items ?? []).join("\n");
    res.case2 = {
      status: row.status, ratio2122: d.includes("21/22"),
      named: items.includes("crucible.crafting"), warnsContent: items.includes("5 条内容"),
      saysLook: items.includes("这一条要看"),
    };
    ok("②-a 状态升成 warn（有内容却没译文 = 要提醒）", row.status === "warn", `实得 ${row.status}`);
    ok("②-b 报文里仍是 **21/22**", res.case2.ratio2122, "");
    ok("②-c **点名**到 `crucible.crafting`", res.case2.named, items.slice(0, 120));
    ok("②-d 报出它有 5 条内容、并明说要看", res.case2.warnsContent && res.case2.saysLook, items.slice(0, 200));
    ok("②-e **没有**被误报成「预期」", !items.includes("预期"), "");
  }

  // ③ 全部有译文 → 22/22
  {
    const all = new Set(fixture.map(f => f.collection));
    const { row } = await runCase(panelPath, specsFor({ craftingEntries: CJK_NAMES }), all);
    const d = row.detail ?? "";
    res.case3 = { status: row.status, full: d.includes("22/22"), noItems: (row.items ?? []).length === 0 };
    ok("③-a 状态 ok", row.status === "ok", `实得 ${row.status}`);
    ok("③-b 报文是 **22/22**", res.case3.full, d.slice(0, 60));
    ok("③-c 没有差额，也就没有点名清单", res.case3.noItems, "");
  }
  say("");
  return res;
}

const forward = await runAllCases(PANEL, "正向用例（面板真身）");

/* ────────────────────────────────────────────────────────────────
   3 · 反向用例：把「点名」那段掏空，用例**必须变红**
   ──────────────────────────────────────────────────────────────── */
const src = fs.readFileSync(PANEL, "utf8");
/* ⚠ 前置自证「改对了地方」：先算出 `checkBabele()` 的源码区间，
      每一处改动的偏移都必须落在里面 —— 只数对条数不等于改对了地方。 */
const fnStart = src.indexOf("\nfunction checkBabele() {");
const fnEnd = src.indexOf("\nfunction checkI18n() {");
if (fnStart < 0 || fnEnd < 0 || fnEnd <= fnStart) hard("抠不到 `checkBabele()` 的源码区间 —— 面板结构变了，反向用例必须重锚");
say(`== 反向用例（掏空点名逻辑，必须变红）==`);
ok("前置自证 · `checkBabele()` 源码区间抠得到", fnEnd > fnStart, `[${fnStart}, ${fnEnd}) 共 ${fnEnd - fnStart} 字节`);

/** 做一次文本变异：断言命中条数 ＝ 预期，且每一处都落在 checkBabele 区间内。 */
function mutate(from, to, expectHits) {
  let hits = 0, i = 0, bad = 0;
  while ((i = src.indexOf(from, i)) !== -1) { hits++; if (i < fnStart || i >= fnEnd) bad++; i += from.length; }
  const okHits = hits === expectHits, okWhere = bad === 0;
  ok(`前置自证 · 变异「${from.slice(0, 34)}…」命中 ${expectHits} 处`, okHits, `实得 ${hits}`);
  ok(`前置自证 · 这 ${hits} 处**全部落在 checkBabele() 区间内**`, okWhere, `越界 ${bad} 处`);
  if (!okHits || !okWhere) hard("变异的前置自证没过 —— 反向用例这次没跑成，不是通过");
  return src.split(from).join(to);
}

async function runMutant(name, text, check) {
  const p = path.join(OUT, `_mutant_${name}.mjs`);
  fs.writeFileSync(p, text, "utf8");
  const before = failures;
  const silent = [];
  const realSay = log.push.bind(log);
  try { await check(p); } catch (e) { say(`（变异体跑出异常：${e?.message ?? e}）`); }
  void realSay; void silent;
  return before;
}

/* M1 · 掏空「点名」：items 恒为空数组 */
{
  const text = mutate("      const items = [", "      const items = []; const __gutted = [", 1);
  const p = path.join(MUT, "_mutant_M1_no_naming.mjs");
  fs.writeFileSync(p, text, "utf8");
  const { row } = await runCase(p, specsFor({ craftingEntries: [] }), translatedSet);
  const named = (row.items ?? []).join("\n").includes("crucible.crafting");
  ok("M1 掏空点名后，用例 ①-e（点名）**当场变红**", named === false,
    named ? "⚠⚠ 掏空了还能过 —— 说明那条断言是恒真的，白写了" : "变异体里确实点不出名 ⇒ 断言有效");
}

/* M2 · 掏空「良性 / 可疑」分类：一律当成空包 */
{
  const text = mutate("((n === 0 && sampled > 0) ? emptyOnes : suspectOnes)", "((true) ? emptyOnes : suspectOnes)", 1);
  const p = path.join(MUT, "_mutant_M2_all_benign.mjs");
  fs.writeFileSync(p, text, "utf8");
  const { row } = await runCase(p, specsFor({ craftingEntries: CJK_NAMES }), translatedSet);
  ok("M2 把「有内容却没译文」也当良性后，用例 ②-a（warn）**当场变红**", row.status !== "warn",
    row.status === "warn" ? "⚠⚠ 掏空了还是 warn —— 那条断言是恒真的" : `变异体报 ${row.status} ⇒ 断言有效`);
}

/* M3 · 摘分母：把分母换成分子（第二十四轮那个假 0 的同型动作） */
{
  const text = mutate("${yes.length}/${packs.length} 个合集有对应译文文件。**差的",
    "${yes.length}/${yes.length} 个合集有对应译文文件。**差的", 2);
  const p = path.join(MUT, "_mutant_M3_shrink_denominator.mjs");
  fs.writeFileSync(p, text, "utf8");
  const { row } = await runCase(p, specsFor({ craftingEntries: [] }), translatedSet);
  const d = row.detail ?? "";
  ok("M3 把分母摘成分子后，用例 ①-b/①-c（21/22 且无假 22/22）**当场变红**",
    !(d.includes("21/22") && !d.includes("22/22")),
    `变异体报文：${d.slice(0, 40)} ⇒ 断言有效`);
}
say("");

/* ────────────────────────────────────────────────────────────────
   4 · 收尾
   ──────────────────────────────────────────────────────────────── */
fs.rmSync(MUT, { recursive: true, force: true });
say(`== 合计：失败 ${failures} 条 ==`);
fs.writeFileSync(path.join(OUT, "probe_pack_naming.result.json"), JSON.stringify({
  fixture: { declared: declared.length, loaded: loaded.length, untranslated },
  forward, failures,
}, null, 1), "utf8");
fs.writeFileSync(path.join(OUT, "probe_pack_naming.log"), log.join("\n"), "utf8");
process.exit(failures ? 1 : 0);
