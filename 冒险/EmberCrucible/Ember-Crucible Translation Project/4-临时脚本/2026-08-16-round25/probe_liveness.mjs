/**
 * 离线驱动**真身** `keyLiveness`（ember-cn-selfcheck.mjs），不抄一份副本。
 *
 * ⚠ 为什么不像上一轮那样自己复刻一遍判据：探针抄一份就等于测了个副本 ——
 *   真身改了而副本没改（或反过来）时探针照样绿，那正是本项目登记的空转形态。
 *   所以这里只做三件事：桩掉 `fetch` / `game`，把表喂进去，把它**自己算的** `.stats` 读出来。
 *
 * ⚠ 本文件是**手写**的，不是脚本生成的 —— 正则里的 \b \s 一旦经 python 字符串转手就会失效（形态 f）。
 *
 * 用法：
 *   node probe_liveness.mjs before   # 拿第二十四轮**加白名单之前**的原表跑（77 那份基线）
 *   node probe_liveness.mjs after    # 拿当前仓库里的表跑
 *   node probe_liveness.mjs unit     # 词边界 / 空白折叠 / 假阴性对照的单元用例
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const DATA = "C:/Users/Taka/AppData/Local/FoundryVTT/Data";
const PROJ = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const SELFCHECK = pathToFileURL(path.join(PROJ, "1-Ember汉化插件/scripts/ember-cn-selfcheck.mjs")).href;
const ROUND24 = path.join(PROJ, "4-临时脚本/2026-08-16-round24");
const HERE = path.join(PROJ, "4-临时脚本/2026-08-16-round25");

/* ── 桩：Foundry 的 fetch 是相对 Data 根的路径 ── */
const fetched = { ok: 0, fail: 0, paths: [] };
globalThis.fetch = async (url) => {
  const p = path.join(DATA, url);
  fetched.paths.push(url);
  if (!fs.existsSync(p) || !fs.statSync(p).isFile()) { fetched.fail++; return { ok: false, text: async () => "" }; }
  fetched.ok++;
  const t = fs.readFileSync(p, "utf8");
  return { ok: true, text: async () => t };
};
/* 世界侧：跑 crucible 世界这一侧（onlyOn:'dnd5e' 的表本来就 skip）。
   ⚠ `game.packs` 给空数组 —— 离线拿不到合集索引，于是「合集索引条目名」那一路
   必须如实报成 0 条。探针**不许**假装拿到了。 */
globalThis.game = { system: { id: "crucible" }, packs: [] };

const SC = await import(SELFCHECK);

/* -------------------------------------------------- */
/*  unit：词边界 / 空白折叠 / 假阴性对照                */
/* -------------------------------------------------- */
async function unit() {
  const { makeCorpus, makeCorpusSet, foldWS } = SC.__corpusInternals;
  let pass = 0, fail = 0;
  const t = (name, got, want) => {
    const ok = JSON.stringify(got) === JSON.stringify(want);
    console.log(`${ok ? "  ok  " : "  FAIL"} ${name}${ok ? "" : `  得到 ${JSON.stringify(got)}，应为 ${JSON.stringify(want)}`}`);
    ok ? pass++ : fail++;
  };

  // ── 词边界的专门用例：语料里**只放 `Abyssal`、不放 `Abyss`** ──
  const c1 = makeCorpus("const x = 'Abyssal Plane'; // Oakenshield Bejakov Waerdling", "词边界用例");
  t("语料只有 Abyssal 时，Abyss 必须判为不存在", c1.has("Abyss"), false);
  t("Abyssal 自己仍然找得到", c1.has("Abyssal"), true);
  t("Oaken 嵌在 Oakenshield 里 → 不算命中", c1.has("Oaken"), false);
  t("Bejak 嵌在 Bejakov 里 → 不算命中", c1.has("Bejak"), false);
  t("Waerd 嵌在 Waerdling 里 → 不算命中", c1.has("Waerd"), false);
  t("独立成词时照旧命中", makeCorpus("the Abyss awaits").has("Abyss"), true);
  t("句末标点后仍算独立成词", makeCorpus("into the Abyss.").has("Abyss"), true);
  t("以非词字符收尾的键不要求右边界", makeCorpus("<p>Allow Retry?</p>").has("Allow Retry?"), true);

  // ── 空白折叠 ──
  const c2 = makeCorpus("<span>\n      Create a new custom vista composition,\n      starting from your currently viewed one.\n  </span>", "折叠用例");
  t("模板里的换行+缩进折叠后能命中",
    c2.has("Create a new custom vista composition, starting from your currently viewed one."), true);
  t("键自己带换行也折叠", c2.has("Create a new custom\n vista composition,"), true);
  t("foldWS 把连续空白压成一个空格", foldWS("a \n\t b"), "a b");

  // ── 非 ASCII 键不加词边界（\w 式边界对 CJK 没有意义）──
  t("CJK 键照旧走裸子串", makeCorpus("这是空洞之月的说明").has("空洞之月"), true);

  // ── 语料集：find 返回**命中在哪一份** ──
  const set = makeCorpusSet([makeCorpus("alpha", "甲"), makeCorpus("beta", "乙")]);
  t("find 报出命中来源", set.find("beta"), "乙");
  t("都没有时返回 null", set.find("gamma"), null);

  // ── 假阴性对照：5 个确实不存在的串，必须 5/5 报出来 ──
  const fakes = {
    "Zzq Frobnicated Widget": "假串一",
    "This String Does Not Exist Upstream At All": "假串二",
    "Quaffle Marmalade Dispenser": "假串三",
    "Vorpal Blancmange Protocol": "假串四",
    "Xylophone Requisition Form": "假串五"
  };
  const checks = await SC.keyLiveness({ FAKE: fakes });
  const st = checks.find(c => c.name === "合计")?.stats;
  t("5 个构造的不存在串全部报出（假阴性对照）", st?.missDistinct, 5);
  t("对照表确实核了 5 个键", st?.checkedDistinct, 5);

  // ── D5 新增 kind：field-path / pack-identifier / 键级 keyKinds ──
  const byName = (cs, n) => cs.find(c => c.name === n);

  const c3 = await SC.keyLiveness({
    VISTA_PLACEMENT_FIELDS: { table: { "illumination.blurStrength": "x" }, kind: "field-path" }
  });
  t("field-path 报 skip（不是 warn）", byName(c3, "VISTA_PLACEMENT_FIELDS")?.status, "skip");

  // pack-identifier：合集索引里**真的**能取到 identifier 时，才算查过
  globalThis.game.packs = [{
    collection: "ember.character",
    metadata: { packageName: "ember", name: "character" },
    index: [{ name: "Oaken", system: { identifier: "oaken" } }, { name: "Keth", system: { identifier: "keth" } }]
  }];
  const c4 = await SC.keyLiveness({ MISSING_ANCESTRIES: { table: { keth: "凯斯", zzz: "无" }, kind: "pack-identifier" } });
  const r4 = byName(c4, "MISSING_ANCESTRIES");
  t("pack-identifier 查 index，命中的报 warn", r4?.status, "warn");
  t("pack-identifier 只认 identifier（keth 命中、zzz 不命中）", r4?.items?.length, 1);

  globalThis.game.packs = [];
  const c5 = await SC.keyLiveness({ MISSING_ANCESTRIES: { table: { keth: "凯斯" }, kind: "pack-identifier" } });
  t("合集取不到 identifier 时报「无从查起」而不是绿", byName(c5, "MISSING_ANCESTRIES")?.status, "skip");

  const c6 = await SC.keyLiveness({
    ATTUNEMENTS: { table: { "The Abyss": "深渊", "Zzq Frobnicated Widget": "假串" }, keyKinds: { "The Abyss": "data" } }
  });
  t("keyKinds 把单个键分流出去（另起一行 skip）", byName(c6, "ATTUNEMENTS · 键级分流")?.status, "skip");
  t("分流之后本表只核剩下的键", byName(c6, "ATTUNEMENTS")?.checked, 1);
  t("分流不影响假阴性：剩下那个假串仍然报出", byName(c6, "合计")?.stats?.missDistinct, 1);

  // 全表都被分流走 → 这一档**一个键都没核**，必须 skip 而不是报一条「0 键全部找得到」的绿
  const c7 = await SC.keyLiveness({ ONLY_DIVERTED: { table: { "The Abyss": "深渊" }, keyKinds: { "The Abyss": "data" } } });
  t("整表被分流光时报 skip（不许 0 项报绿）", byName(c7, "ONLY_DIVERTED")?.status, "skip");
  t("0 项报绿的反证：本表没有 ok 行", byName(c7, "ONLY_DIVERTED")?.checked, 0);

  console.log(`\nunit：通过 ${pass} / 失败 ${fail}`);
  return fail;
}

/* -------------------------------------------------- */
/*  before / after：拿真表跑                            */
/* -------------------------------------------------- */
async function real(which) {
  const harness = which === "before"
    ? path.join(ROUND24, "_hc_before.mjs")
    : path.join(HERE, "_hc_now.mjs");
  if (which !== "before") {
    // 现表的 harness 现做：与 round24 的 fix_mkharness.mjs 同一套桩
    const src = fs.readFileSync(path.join(PROJ, "1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs"), "utf8");
    const IMPORT_LINE = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
    if (!src.includes(IMPORT_LINE)) throw new Error("找不到 SELFCHECK 的 import 行，转换中止");
    const stub = "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n";
    fs.writeFileSync(harness,
      stub + src.replace(IMPORT_LINE, "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };") +
      "\nexport { SELFCHECK_TABLES as __SELFCHECK_TABLES };\n", "utf8");
  }
  const mod = await import(pathToFileURL(harness).href);
  const checks = await SC.keyLiveness(mod.__SELFCHECK_TABLES);

  for (const c of checks) {
    const cnt = c.status === "skip" ? "—" : String(c.checked);
    console.log(`${c.status.padEnd(5)} ${cnt.padStart(5)}  ${c.name}`);
    if (c.status === "warn") for (const i of (c.items ?? [])) console.log(`        · ${i}`);
  }
  const st = checks.find(c => c.name === "合计")?.stats;
  console.log("\n──────── 合计 ────────");
  console.log(JSON.stringify({ ...st, miss: undefined }, null, 2));
  console.log(`\n面板口径「上游查无此串」：去重 ${st.missDistinct} 条（报文条数 ${st.rawMiss}）`);
  console.log(`fetch：成功 ${fetched.ok} / 失败 ${fetched.fail}`);
  console.log("剩下的 miss 明细：");
  for (const k of st.miss) console.log(`  · ${JSON.stringify(k)}`);
  return 0;
}

const mode = process.argv[2] ?? "unit";
process.exit(mode === "unit" ? await unit() : await real(mode));
