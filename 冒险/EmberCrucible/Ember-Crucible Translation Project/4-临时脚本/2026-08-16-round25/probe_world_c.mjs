/**
 * 「真实 Foundry 世界」那一侧的模拟：给判据一份**真的**合集索引，看两件事
 *   ① `The Abyss` 会不会被「合集索引条目名」那一路语料接住（我在表注释里写了会，这里验）；
 *   ② 三张 MISSING_* 的 `kind:'pack-identifier'`：
 *      · index 里没有 `system.identifier` 时，必须报「无从查起」而不是假绿；
 *      · index 里有 identifier 时，才真的去比。
 *
 * ⚠ 页名不是编的：从仓内**英文基准** `1-Ember汉化插件/compendium/en/ember.crucible-adventure.json`
 *   里读出来的（babele 基准＝上游英文原文）。identifier 那一路offline 拿不到真数据，
 *   所以只做「没有 identifier → 必须报无从查起」这一侧的断言，**不编造 identifier 冒充真数据**。
 *
 * 手写落盘，不经改写脚本。用法：node probe_world_c.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const DATA = "C:/Users/Taka/AppData/Local/FoundryVTT/Data";
const PROJ = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const HERE = path.join(PROJ, "4-临时脚本/2026-08-16-round25");
const SELFCHECK = pathToFileURL(path.join(PROJ, "1-Ember汉化插件/scripts/ember-cn-selfcheck.mjs")).href;

const fetched = { ok: 0, fail: 0 };
globalThis.fetch = async (url) => {
  const p = path.join(DATA, url);
  if (!fs.existsSync(p) || !fs.statSync(p).isFile()) { fetched.fail++; return { ok: false, text: async () => "" }; }
  fetched.ok++;
  const t = fs.readFileSync(p, "utf8");
  return { ok: true, text: async () => t };
};

/* ── 从英文基准里抠出 JournalEntry 的条目名与页名 ── */
function adventureIndex(file) {
  const d = JSON.parse(fs.readFileSync(file, "utf8"));
  const out = [];
  const walk = (node) => {
    if (!node || typeof node !== "object") return;
    if (Array.isArray(node)) { node.forEach(walk); return; }
    if (typeof node.name === "string") out.push(node.name);
    for (const v of Object.values(node)) walk(v);
  };
  walk(d.entries ?? d);
  return out;
}
const advNames = adventureIndex(path.join(PROJ, "1-Ember汉化插件/compendium/en/ember.crucible-adventure.json"));
console.log(`英文基准里抠到 ${advNames.length} 个 name`);
const hasAbyss = advNames.includes("The Abyss");
const hasHeart = advNames.includes("Heart of Ember");
console.log(`  其中 "The Abyss" ${hasAbyss ? "在" : "不在"} / "Heart of Ember" ${hasHeart ? "在" : "不在"}`);

globalThis.game = {
  system: { id: "crucible" },
  packs: [
    { collection: "ember.crucible-adventure",
      metadata: { packageName: "ember", name: "crucible-adventure" },
      index: advNames.map(n => ({ name: n })) },
    // 角色选项包：index 里**没有** system.identifier（很多世界就是这样，index 默认字段有限）
    { collection: "ember.crucible-character",
      metadata: { packageName: "ember", name: "crucible-character" },
      index: [{ name: "Oaken" }, { name: "Keth" }, { name: "Anchorite Marine" }] }
  ]
};

const SC = await import(SELFCHECK);

/* ── 拿当前仓库里的表跑 ── */
const harness = path.join(HERE, "_hc_world_c.mjs");
{
  const src = fs.readFileSync(path.join(PROJ, "1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs"), "utf8");
  const IMPORT_LINE = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
  if (!src.includes(IMPORT_LINE)) throw new Error("找不到 SELFCHECK 的 import 行，转换中止");
  fs.writeFileSync(harness,
    "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n" +
    src.replace(IMPORT_LINE, "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };") +
    "\nexport { SELFCHECK_TABLES as __SELFCHECK_TABLES };\n", "utf8");
}
const mod = await import(pathToFileURL(harness).href);
const checks = await SC.keyLiveness(mod.__SELFCHECK_TABLES);

for (const c of checks) {
  const cnt = c.status === "skip" ? "—" : String(c.checked);
  console.log(`${c.status.padEnd(5)} ${cnt.padStart(5)}  ${c.name}`);
  if (c.status === "warn") for (const i of (c.items ?? [])) console.log(`        · ${i}`);
  if (/^MISSING_(ANCESTRIES|CULTURES|PATHS)$/.test(c.name)) console.log(`        → ${c.detail ?? c.note ?? c.message ?? ""}`);
}
const st = checks.find(c => c.name === "合计")?.stats;
console.log("\n──────── 合计 ────────");
console.log(JSON.stringify({ ...st, miss: undefined }, null, 2));
console.log(`面板口径「上游查无此串」：去重 ${st.missDistinct} 条（报文条数 ${st.rawMiss}）`);
console.log("剩下的 miss 明细：");
for (const k of st.miss) console.log(`  · ${JSON.stringify(k)}`);
console.log(`fetch：成功 ${fetched.ok} / 失败 ${fetched.fail}`);

/* ================================================================== */
/*  第二场：index 里**有** system.identifier 时，pack-identifier 真的在比 */
/*  ⚠ 这里的 identifier 是**构造**的，用途只有一个：证明「有数据时判据会去比、   */
/*     命中会报 warn、没命中会报 ok」，**不冒充上游真数据**。               */
/* ================================================================== */
console.log("\n════ 第二场：给角色选项包补上 system.identifier ════");
const charPack = game.packs.find(p => p.collection === "ember.crucible-character");

// (a) 全部对不上 → 三张表都该报 ok（「上游仍然没有」＝预期状态）
charPack.index = [{ name: "Nobody", system: { identifier: "nobody" } },
                  { name: "Nothing", system: { identifier: "nothing" } }];
const cA = await SC.keyLiveness(mod.__SELFCHECK_TABLES);
for (const n of ["MISSING_ANCESTRIES", "MISSING_CULTURES", "MISSING_PATHS"]) {
  const r = cA.find(c => c.name === n);
  console.log(`  (a) ${n}: status=${r?.status} checked=${r?.checked}`);
}

// (b) 上游「补上了」两条 → 该报 warn 并点名
charPack.index = [{ name: "Oaken", system: { identifier: "Oaken" } },
                  { name: "Anchorite Marine", system: { identifier: "AnchoriteMarine" } }];
const cB = await SC.keyLiveness(mod.__SELFCHECK_TABLES);
for (const n of ["MISSING_ANCESTRIES", "MISSING_CULTURES", "MISSING_PATHS"]) {
  const r = cB.find(c => c.name === n);
  console.log(`  (b) ${n}: status=${r?.status} checked=${r?.checked} items=${JSON.stringify(r?.items ?? [])}`);
}
const stB = cB.find(c => c.name === "合计")?.stats;
console.log(`  (b) 合计 rawChecked=${stB.rawChecked}（比第一场多的 19 条正是三张 MISSING_* 的键）`);
