/**
 * 自检面板 **D 档的常设执行体**：离线驱动真身 `keyLiveness`，把面板自己算的数吐成 JSON。
 *
 * 由 `assert_resolutions.py` 的 `panel_liveness` 断言（R-selfcheck-d-liveness）以
 * `node selfcheck_panel_runner.mjs <spec.json>` 的形式调用；spec / 结果的字段见下面 SPEC 一节。
 *
 * ⚠ 本文件的由来（第二十八轮 · 必读）：
 *   它的前身是 `5-临时脚本/2026-08-16-round26/probe_panel_report.mjs` —— 一次性探针，
 *   **git 未跟踪**。而「719 distinct / 1273 raw · miss 4 键 / 7 报文」这组数被 PROJECT.md、
 *   面板 why、复核报告三路引用，**可再现性却挂在一台机器上的一份未入库脚本上**。
 *   那是本项目登记的空转形态 (c) 在更高一层复发 —— 不是判据读了旧快照，是**「结论」本身就是快照**。
 *   ⇒ 提升进 `4-常用脚本/qa/` 并入库，配一条常设断言**每次现跑**，记录值只当「不得低于 / 不得高于」用。
 *
 * ⚠ 三条纪律（照抄前身，本项目登记的空转形态换来的）：
 *   · 只桩 `fetch` / `game` / `Hooks`，**判据一行都不抄** —— 抄一份就等于测了个副本（形态 c/h）；
 *   · 数一律读面板挂在「合计」行 `stats` 上的**它自己算的数**，本执行体不另算一份；
 *   · `game.packs` 老实给空数组 —— 离线拿不到合集索引，**不许捏一份 index 喂进去**
 *     （形态 (h)：真实 Foundry 的 `pack.index` 上有哪些字段由 `compendiumIndexFields` 决定，
 *      数据文件里有 `journal[].pages[].name` 推不出运行时拿得到它）。
 *
 * ⚠ 模拟输入的上游契约出处（形态 (h) 的要求：探针的输入必须指得出处）：
 *   面板用的是 `fetch(url)`（:98 `fetchText`），url 形如 `modules/ember/scripts/ember.mjs`
 *   （:698 `const EMBER_ROOT = "modules/ember";`）—— Foundry 把 **Data 根目录**挂在站点根上，
 *   所以离线桩就是 `path.join(dataRoot, url)`。本执行体**不信任** spec 里的 dataRoot：
 *   它从面板源码里现抠 `EMBER_ROOT`，确认 `dataRoot/<EMBER_ROOT>/scripts/ember.mjs` 真的存在，
 *   对不上当场硬失败 —— 免得「桩指到了空目录，语料一份没抓到，而判据照跑」。
 *
 * ⚠ **变异开关**（`drop_corpus` / `keep_table_fraction` / `count_no_unwrap` /
 *   `ledger_offset` / `fake_ledger_flags`）只给 `--selftest` 的回测用。
 *   每个开关的方向都是**只会让结果更差**（少抓语料 ⇒ miss 涨；摘掉表项 ⇒ 覆盖掉；
 *   表侧计数退回不解包 ⇒ 对账恒等式断；台账偏移 ⇒ 账不平且没核数涨；旗标作假 ⇒ 直接红），
 *   所以它们进不了「调一下开关就变绿」的作弊路径 —— 这是有意选的方向。
 *   ⚠⚠ 往这里加开关时**必须守住这个方向**：任何能把某个数往好里推的开关，
 *   都会立刻变成「跑闸时顺手带一个 spec 字段」的单点绕法。
 *
 * SPEC（JSON 文件，路径作为 argv[2] 传入）：
 *   panel               面板 .mjs 的绝对路径（ember 侧那一份）
 *   tables_src          `ember-hardcoded-cn.mjs` 的绝对路径
 *   stub_import         被判文件里那一行 SELFCHECK import（逐字符，找不到当场失败）
 *   data_root           Foundry Data 目录（fetch 桩的根）
 *   out                 结果 JSON 的写出路径
 *   fakes               假阴性对照用的自造串（必须 100% 报出）
 *   substr_probe        子串型改词的边界复现：一个「被截短的」键，预期报出 0 条
 *   drop_corpus         [变异] url 里含这些子串就让 fetch 报失败
 *   keep_table_fraction [变异] 每张表只保留前这么大比例的条目
 *   count_no_unwrap     [变异] 表侧独立计数退回 V18 之前那个**不解包**的写法
 *   ledger_offset       [变异] 给报出去的「按设计没核」加一个正偏移（把账弄不平）
 *   fake_ledger_flags   [变异] 谎报 ledgerBalanced=false / ledgerNoReason=1
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

function die(msg) {
  process.stderr.write(msg + "\n");
  process.exit(2);
}

const specPath = process.argv[2];
if (!specPath) die("用法：node selfcheck_panel_runner.mjs <spec.json>");
const spec = JSON.parse(fs.readFileSync(specPath, "utf8"));

for (const k of ["panel", "tables_src", "stub_import", "data_root", "out"]) {
  if (!spec[k]) die(`spec 缺字段 ${k} —— 这一条没跑成，不是通过`);
}
for (const k of ["panel", "tables_src"]) {
  if (!fs.existsSync(spec[k])) die(`${k} 指向的文件不存在：${spec[k]}`);
}

/* ── 面板源码：现抠 EMBER_ROOT，用来验 dataRoot 桩指对了地方（形态 (h)） ── */
const panelSrc = fs.readFileSync(spec.panel, "utf8");
const mRoot = panelSrc.match(/const EMBER_ROOT = "([^"]+)";/);
if (!mRoot) {
  die("面板源码里抠不到 `const EMBER_ROOT = \"…\";` —— 面板换写法了，"
    + "fetch 桩的上游契约出处失效，本执行体**必须重新锚定**，不许照跑");
}
const EMBER_ROOT = mRoot[1];
const anchorFile = path.join(spec.data_root, EMBER_ROOT, "scripts", "ember.mjs");
if (!fs.existsSync(anchorFile)) {
  die(`fetch 桩指到的上游不存在：${anchorFile}\n`
    + `（data_root=${spec.data_root} · 面板的 EMBER_ROOT=${EMBER_ROOT}）——`
    + `语料一份都抓不到而判据照跑，正是空转形态 (h)`);
}

/* ── 桩 ── */
let fOk = 0, fFail = 0;
const dropped = [];
const drop = spec.drop_corpus || [];
globalThis.fetch = async (url) => {
  const u = String(url);
  if (drop.some((d) => u.includes(d))) { dropped.push(u); fFail++; return { ok: false, text: async () => "" }; }
  const p = path.join(spec.data_root, u);
  if (!fs.existsSync(p) || !fs.statSync(p).isFile()) { fFail++; return { ok: false, text: async () => "" }; }
  fOk++;
  return { ok: true, text: async () => fs.readFileSync(p, "utf8") };
};
globalThis.Hooks = { once() {}, on() {} };
globalThis.game = { system: { id: "crucible" }, packs: [] };

const SC = await import(pathToFileURL(spec.panel).href);
if (typeof SC.keyLiveness !== "function") {
  die("面板没导出 `keyLiveness` —— D 档的执行体不在了，这一条没跑成");
}

/* ── 现表 harness：把被判文件的 SELFCHECK_TABLES 导出来（每次现生成，跑完即删） ── */
let TABLES;
{
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ec-panel-"));
  try {
    const harness = path.join(tmpDir, "_hc_now.mjs");
    const src = fs.readFileSync(spec.tables_src, "utf8");
    if (!src.includes(spec.stub_import)) {
      die(`被判文件里找不到 SELFCHECK 的 import 行：${spec.stub_import} —— `
        + "转换中止（不许猜一个写法凑合，那就是测了个副本）");
    }
    fs.writeFileSync(harness,
      "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
      + src.replace(spec.stub_import,
        "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
      + "\nexport { SELFCHECK_TABLES as __SELFCHECK_TABLES };\n", "utf8");
    const mod = await import(pathToFileURL(harness).href);
    TABLES = mod.__SELFCHECK_TABLES;
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}
if (!TABLES || !Object.keys(TABLES).length) die("SELFCHECK_TABLES 是空的 —— 没跑成");

/* 表侧的**独立计数**：不经面板，自己数一遍「表里一共有多少个不同的键」。
 * 面板自报的 registeredRaw / registeredDistinct 是另一条汇总路径 —— 两个数摆在一起，
 * 「面板悄悄少核了一批键」这件事才有得对（光看面板自报的数，它少核了也自洽）。
 *
 * ⚠⚠ **第三十二轮 V18：这个函数从前是错的，而且它的注释宣称的用途与实现不符** ——
 *   它直接 `Object.keys(spec)`，**不解包** `{ table, kind, onlyOn, packs, corpus }`
 *   这 13 张包装表：对包装对象数出来的是那 5 个**元数据属性名**（伪键），
 *   同时**漏掉包装里 189 raw / 171 distinct 个真实键**。
 *   于是它吐的是 721 distinct / 1304 raw，**真值是 887 / 1459**，两个数都错。
 *   而 V14 那条「面板 719/1273 vs 表侧 721/1304」挂了三轮，成因正是这一处
 *   —— 差在**数表这一侧**，面板一个键都没漏核（见面板 `keyLiveness` 的「记账口径」段）。
 *   ⇒ 用与面板 `:1116` `spec?.table ?? spec` **同一个口径**解包；
 *     并且解包完的数**必须真的有人判**（见 `a_panel_liveness` 的恒等式
 *     `tableKeysRaw == registeredRaw` / `tableKeysDistinct == registeredDistinct`）——
 *     「算了、印了、没人判」正是本项目登记的空转形态，V18 的另一半就是它。
 *
 * ⚠ **只数字面量键的表**（`{英文串: 中文}`）—— 数组形态的表（PREFIXED / PATTERNS /
 *   NOTIFICATION_PATTERNS 那三张「命中即整串替换」的）键**不是字面量**，D 档按设计核不了它们
 *   （对它们直接报 ⛔），所以它们不进这个数，另计一份 `regexEntries`。 */
function unwrapTable(x) {
  // 与面板 keyLiveness 的 `const table = spec?.table ?? spec;` 逐字同口径。
  return Array.isArray(x) ? x : (x?.table ?? x);
}

function countTableKeys(tabs) {
  const seen = new Set();
  let raw = 0, regexEntries = 0, objTables = 0, arrTables = 0, wrapped = 0;
  for (const x of Object.values(tabs)) {
    if (x && !Array.isArray(x) && typeof x === "object"
        && Object.prototype.hasOwnProperty.call(x, "table")) wrapped++;
    // [变异] 退回 V18 之前那个不解包的写法 —— 只会让对账恒等式断，不会让谁变绿
    const t = spec.count_no_unwrap ? x : unwrapTable(x);
    if (Array.isArray(t)) { arrTables++; regexEntries += t.length; continue; }
    if (!t || typeof t !== "object") continue;
    objTables++;
    for (const k of Object.keys(t)) { raw++; seen.add(k); }
  }
  return { distinct: seen.size, raw, regexEntries, objTables, arrTables, wrapped };
}

/* [变异开关] 摘掉每张表的一部分条目 —— 只会让覆盖掉，不会让谁变绿。
 * ⚠ 同一个不解包的坑在这里也有过：直接切包装对象的 `Object.entries()` 切掉的是
 *   **元数据属性**（`corpus`/`packs`/…），13 张包装表里的真实键**一条都没被摘掉**，
 *   变异开关对它们等于没开。这里按包装形态原样保留外壳、只摘里面那张表。 */
const frac = spec.keep_table_fraction;
let tablesIn = TABLES;
if (frac != null) {
  if (!(frac > 0 && frac <= 1)) die(`keep_table_fraction 必须落在 (0,1]，实得 ${frac}`);
  const trim = (t) => {
    if (Array.isArray(t)) return t.slice(0, Math.ceil(t.length * frac));
    if (t && typeof t === "object") {
      const e = Object.entries(t);
      return Object.fromEntries(e.slice(0, Math.ceil(e.length * frac)));
    }
    return t;
  };
  tablesIn = {};
  for (const [name, x] of Object.entries(TABLES)) {
    if (x && !Array.isArray(x) && typeof x === "object"
        && Object.prototype.hasOwnProperty.call(x, "table")) {
      tablesIn[name] = { ...x, table: trim(x.table) };
    } else tablesIn[name] = trim(x);
  }
}

/* ── 现跑 ── */
const checks = await SC.keyLiveness(tablesIn);
const byName = (n) => checks.find((c) => c.name === n);
const total = byName("合计");
if (!total || !total.stats) {
  die("面板没吐出「合计」行（或它身上没有 stats）—— D 档的记账口子变了，本执行体必须重新锚定");
}
const st = total.stats;

/* 逐表行的交叉记账：把「合计」自报的 rawChecked 与**逐行 checked 之和**对一遍。
 * 两个数走的是面板里两条不同的汇总路径，对不上就是面板自己的记账坏了。 */
const SUMMARY_ROWS = new Set(["合计", "上游语料", "Adventure 内层页名"]);
const tableRows = checks.filter((c) => !SUMMARY_ROWS.has(c.name));
const rowCheckedSum = tableRows.reduce((a, c) => a + (Number(c.checked) || 0), 0);

/* 「每张表都得有一行」：喂进去几张表，报文里就得有几张表的行（核不了的表报 ⛔/skip，
 * 那是**如实标死**，也算有行）。少一行 = 有一张表被静默吞了，光看合计的数看不出来。 */
const rowNames = new Set(tableRows.map((c) => c.name));
const tablesWithoutRow = Object.keys(tablesIn).filter((n) => !rowNames.has(n));

/* ── 假阴性对照：自造的不存在串必须 100% 报出（面板的匹配器要是瞎了，这里当场露馅） ── */
const fakes = spec.fakes || {};
let fakeChecked = 0, fakeMiss = 0;
if (Object.keys(fakes).length) {
  const fk = (await SC.keyLiveness({ FAKE: fakes })).find((c) => c.name === "合计");
  if (!fk || !fk.stats) die("假阴性对照跑不出「合计」行 —— 没跑成");
  fakeChecked = fk.stats.checkedDistinct;
  fakeMiss = fk.stats.missDistinct;
}

/* ── 边界复现：一串是另一串的子串时，本档**看不见**（这是写进报文的边界，不是缺陷） ── */
let substrMiss = null;
if (spec.substr_probe) {
  const sub = await SC.keyLiveness({ SUBSTR: { [spec.substr_probe]: "x" } });
  const s = sub.find((c) => c.name === "合计");
  if (!s || !s.stats) die("子串边界复现跑不出「合计」行 —— 没跑成");
  substrMiss = s.stats.missDistinct;
}

const tk = countTableKeys(tablesIn);

/* [变异] 台账那一侧：偏移只许为正（把「按设计没核」做大 ⇒ 账不平 + 上限被顶破），
 * 旗标作假只许往坏里说。两个开关都进不了「调一下就变绿」那条路。 */
const off = spec.ledger_offset ?? 0;
if (!(Number.isInteger(off) && off >= 0)) die(`ledger_offset 必须是非负整数，实得 ${off}`);
const ledgerNoReasonNames = Array.isArray(st.ledgerNoReason) ? st.ledgerNoReason : [];

const result = {
  section: total.section ?? null,
  counts: {
    tableKeysDistinct: tk.distinct,
    tableKeysRaw: tk.raw,
    tableRegexEntries: tk.regexEntries,
    tableObjTables: tk.objTables,
    tableArrTables: tk.arrTables,
    tableWrapped: tk.wrapped,
    checkedDistinct: st.checkedDistinct,
    rawChecked: st.rawChecked,
    missDistinct: st.missDistinct,
    rawMiss: st.rawMiss,
    /* ── 面板自己算的**分母**（第三十一轮 V14 归因造出来的八个数）。
     *   ⚠ 第三十二轮 V18 的另一半：它们**造出来了却一个都没转发**，
     *     runner 不转发 ⇒ 规则里的 min/max 够不着 ⇒ 全 `4-常用脚本/qa/` 零引用
     *     ⇒ 这批分母本身就是新一份「算了、印了、没人判」。转发在这里，判在
     *     `a_panel_liveness`（registered… 与 wrapped… 与 regex… 走 min，unchecked… 走 max，
     *     ledgerBalanced / ledgerNoReason 是**不含阈值**的恒等式，调不松）。 */
    registeredRaw: st.registeredRaw,
    registeredDistinct: st.registeredDistinct,
    uncheckedRaw: st.uncheckedRaw + off,
    uncheckedDistinct: st.uncheckedDistinct + off,
    ledgerBalanced: spec.fake_ledger_flags ? false : st.ledgerBalanced,
    ledgerNoReason: spec.fake_ledger_flags ? 1 : ledgerNoReasonNames.length,
    wrappedTables: st.wrappedTables,
    regexTables: st.regexTables,
    fetchOk: fOk,
    fetchFail: fFail,
    tableRows: tableRows.length,
    tablesFedIn: Object.keys(tablesIn).length,
    tablesWithoutRow: tablesWithoutRow.length,
    rowCheckedSum,
    tplFiles: st.tpl?.files ?? 0,
    tplParts: (st.tpl?.litOk ?? 0) + (st.tpl?.dynOk ?? 0) + (st.tpl?.unrefOk ?? 0),
    tierDirect: st.tierCounts?.[0] ?? 0,
    derivedKeys: Array.isArray(st.derivedKeys) ? st.derivedKeys.length : (st.derivedKeys ?? 0),
    weakShort: Array.isArray(st.weakShort) ? st.weakShort.length : (st.weakShort ?? 0),
    fakeChecked,
    fakeMiss,
    fakeWanted: Object.keys(fakes).length,
    substrMiss
  },
  miss: st.miss ?? [],
  ledgerNoReasonNames: spec.fake_ledger_flags
    ? ["（变异开关 fake_ledger_flags 谎报的一张表）"] : ledgerNoReasonNames,
  tablesWithoutRowNames: tablesWithoutRow,
  droppedUrls: dropped
};
fs.writeFileSync(spec.out, JSON.stringify(result, null, 1), "utf8");
process.stdout.write(JSON.stringify(result.counts) + "\n");
