/**
 * V14 归因探针：面板自报 719/1273 vs 表侧 721/1304，差 2 键 / 31 条 —— 差在哪、为什么。
 *
 * ⚠ 硬约束 4：前置自证。往下走之前先断言「我切出来的数 ＝ 已知真值」：
 *     39 张表（字面量 37 + 数组 2）· 表侧 raw 1304 / distinct 721 · ARRANGEMENT_LEAVES 212
 *     · 面板自报 719 / 1273。任一条对不上 —— 当场硬失败，不出结论。
 * ⚠ 纪律：判据（keyLiveness）一行都不抄，数一律读面板挂在「合计」行 stats 上的它自己算的数。
 *     `countTableKeys` 是**逐字符照抄 3-常用脚本/qa/selfcheck_panel_runner.mjs:129-139**（被测对象本身），
 *     不是另写一份等价实现 —— 抄的是「表侧计数器」这个被怀疑的对象，不是判据。
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = "C:\\Users\\Taka\\Desktop\\fvtt\\Ember-Crucible Translation Project";
const PANEL = path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-cn-selfcheck.mjs");
const TABLES_SRC = path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs");
const STUB_IMPORT = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const DATA_ROOT = "C:\\Users\\Taka\\AppData\\Local\\FoundryVTT\\Data";
const OUT = path.join(ROOT, "4-临时脚本", "2026-08-16-round31", "probe_v14_attrib.out.json");

const log = [];
function say(s) { log.push(s); process.stdout.write(s + "\n"); }
function die(msg) { process.stderr.write("HARD-FAIL: " + msg + "\n"); process.exit(2); }
function must(cond, msg) { if (!cond) die(msg); }

/* ── 桩（照抄 runner 的形状：只桩 fetch / game / Hooks） ── */
let fOk = 0, fFail = 0;
globalThis.fetch = async (url) => {
  const p = path.join(DATA_ROOT, String(url));
  if (!fs.existsSync(p) || !fs.statSync(p).isFile()) { fFail++; return { ok: false, text: async () => "" }; }
  fOk++;
  return { ok: true, text: async () => fs.readFileSync(p, "utf8") };
};
globalThis.Hooks = { once() {}, on() {} };
globalThis.game = { system: { id: "crucible" }, packs: [] };

/* ── 现表 harness（照抄 runner:99-120） ── */
let TABLES;
{
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ec-v14-"));
  try {
    const harness = path.join(tmpDir, "_hc_now.mjs");
    const src = fs.readFileSync(TABLES_SRC, "utf8");
    must(src.includes(STUB_IMPORT), "被判文件里找不到 SELFCHECK 的 import 行");
    fs.writeFileSync(harness,
      "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
      + src.replace(STUB_IMPORT,
        "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
      + "\nexport { SELFCHECK_TABLES as __SELFCHECK_TABLES };\n", "utf8");
    TABLES = (await import(pathToFileURL(harness).href)).__SELFCHECK_TABLES;
  } finally { fs.rmSync(tmpDir, { recursive: true, force: true }); }
}
must(TABLES && Object.keys(TABLES).length, "SELFCHECK_TABLES 是空的");

/* ── 被测对象①：表侧计数器，逐字符照抄 runner:129-139 ── */
function countTableKeys(tabs) {
  const seen = new Set();
  let raw = 0, regexEntries = 0, objTables = 0, arrTables = 0;
  for (const t of Object.values(tabs)) {
    if (Array.isArray(t)) { arrTables++; regexEntries += t.length; continue; }
    if (!t || typeof t !== "object") continue;
    objTables++;
    for (const k of Object.keys(t)) { raw++; seen.add(k); }
  }
  return { distinct: seen.size, raw, regexEntries, objTables, arrTables, seen };
}

/* ══ 前置自证 ══ */
const tk = countTableKeys(TABLES);
must(Object.keys(TABLES).length === 39, `表数 ${Object.keys(TABLES).length} ≠ 已知真值 39`);
must(tk.objTables === 37, `字面量表 ${tk.objTables} ≠ 37`);
must(tk.arrTables === 2, `数组表 ${tk.arrTables} ≠ 2`);
must(tk.raw === 1304, `表侧 raw ${tk.raw} ≠ 已知真值 1304`);
must(tk.distinct === 721, `表侧 distinct ${tk.distinct} ≠ 已知真值 721`);
const leaves = TABLES.ARRANGEMENT_LEAVES;
must(leaves && Object.keys(leaves).length === 212, `ARRANGEMENT_LEAVES ${Object.keys(leaves ?? {}).length} ≠ 212`);
say(`[自证 1/2] 表数 39（字面量 37 + 数组 2）· raw 1304 / distinct 721 · ARRANGEMENT_LEAVES 212 —— 全部命中已知真值`);

const SC = await import(pathToFileURL(PANEL).href);
must(typeof SC.keyLiveness === "function", "面板没导出 keyLiveness");
const checks = await SC.keyLiveness(TABLES);
const total = checks.find((c) => c.name === "合计");
must(total && total.stats, "面板没吐出「合计」行 / stats");
const st = total.stats;
must(st.checkedDistinct === 719, `面板 checkedDistinct ${st.checkedDistinct} ≠ 已知真值 719`);
must(st.rawChecked === 1273, `面板 rawChecked ${st.rawChecked} ≠ 已知真值 1273`);
must(st.missDistinct === 4 && st.rawMiss === 7, `面板 miss ${st.missDistinct}/${st.rawMiss} ≠ 4/7`);
say(`[自证 2/2] 面板自报 719 distinct / 1273 raw · miss 4 键 / 7 报文 —— 命中已知真值`);
say(`⇒ 待归因的差：distinct 721-719=2 · raw 1304-1273=31`);

/* ══ 归因 ══ */
// 表侧计数器不区分「表本身」和「{table, kind, onlyOn, …} 包装对象」。
// 包装对象走 `for (const k of Object.keys(t))` 时，数出来的是**元数据属性名**，不是翻译键。
const SPEC_PROPS = ["table", "kind", "onlyOn", "packs", "corpus"];
const wrapped = [], plain = [];
for (const [name, spec] of Object.entries(TABLES)) {
  if (Array.isArray(spec)) continue;
  if (!spec || typeof spec !== "object") continue;
  const isWrap = Object.prototype.hasOwnProperty.call(spec, "table");
  (isWrap ? wrapped : plain).push({ name, spec });
}
say("");
say(`包装型表（含 \`table\` 字段）${wrapped.length} 张 · 裸表 ${plain.length} 张`);

let wrapPseudoRaw = 0;
const wrapPseudoDistinct = new Set();
const wrapRows = [];
for (const { name, spec } of wrapped) {
  const props = Object.keys(spec);
  wrapPseudoRaw += props.length;
  props.forEach((p) => wrapPseudoDistinct.add(p));
  const realKeys = Object.keys(spec.table ?? {});
  wrapRows.push({ name, kind: spec.kind ?? "literal", onlyOn: spec.onlyOn ?? null,
                  pseudoProps: props, pseudoRaw: props.length, realKeys: realKeys.length });
}
must(props_are_known(wrapPseudoDistinct), `包装对象出现了未登记的属性名：${[...wrapPseudoDistinct]}`);
function props_are_known(s) { return [...s].every((p) => SPEC_PROPS.includes(p)); }

let plainRaw = 0;
const plainDistinct = new Set();
for (const { spec } of plain) for (const k of Object.keys(spec)) { plainRaw++; plainDistinct.add(k); }

say(`裸表键：raw ${plainRaw} / distinct ${plainDistinct.size}`);
say(`包装对象贡献给表侧计数器的**伪键**：raw ${wrapPseudoRaw} / distinct ${wrapPseudoDistinct.size}` +
    `（${[...wrapPseudoDistinct].sort().join(" · ")}）`);
must(plainRaw + wrapPseudoRaw === tk.raw, `裸表 ${plainRaw} + 伪键 ${wrapPseudoRaw} ≠ 表侧 raw ${tk.raw}`);
say(`  ✔ 拆分闭合：${plainRaw} + ${wrapPseudoRaw} = ${tk.raw}`);

// 面板这一侧：包装表里只有 MISSING_LANGUAGES 这一张真被计了数（absent-by-design，语料是 .mjs，离线也跑得动）
const panelCounted = st.rawChecked - plainRaw;
say(`面板 rawChecked ${st.rawChecked} − 裸表 ${plainRaw} = ${panelCounted} —— 来自包装表的真实键`);
const ml = Object.keys(TABLES.MISSING_LANGUAGES?.table ?? {});
say(`  MISSING_LANGUAGES 真实键 ${ml.length} 个：${ml.map((k) => `\`${k}\``).join(" · ")}`);
must(panelCounted === ml.length, `包装表被计入的条数 ${panelCounted} ≠ MISSING_LANGUAGES 键数 ${ml.length}`);

// distinct 侧
const mlNew = ml.filter((k) => !plainDistinct.has(k));
say(`  其中不与裸表键重名的 ${mlNew.length} 个 ⇒ 面板 distinct = ${plainDistinct.size} + ${mlNew.length} = ${plainDistinct.size + mlNew.length}`);
must(plainDistinct.size + mlNew.length === st.checkedDistinct,
  `裸表 distinct ${plainDistinct.size} + ${mlNew.length} ≠ 面板 ${st.checkedDistinct}`);

const pseudoNew = [...wrapPseudoDistinct].filter((p) => !plainDistinct.has(p) && !ml.includes(p));
say(`  表侧 distinct = ${plainDistinct.size} + 伪键 ${wrapPseudoDistinct.size} = ${tk.distinct}` +
    `（伪键与裸表键无重名：${pseudoNew.length}/${wrapPseudoDistinct.size}）`);
must(plainDistinct.size + wrapPseudoDistinct.size === tk.distinct, "表侧 distinct 拆分不闭合");

say("");
say(`═══ 归因结论 ═══`);
say(`raw  差 31 = 伪键 ${wrapPseudoRaw} − 面板真计的 MISSING_LANGUAGES ${ml.length}`);
must(wrapPseudoRaw - ml.length === tk.raw - st.rawChecked, "raw 归因不闭合");
say(`distinct 差 2 = 伪键 distinct ${wrapPseudoDistinct.size} − ${mlNew.length}`);
must(wrapPseudoDistinct.size - mlNew.length === tk.distinct - st.checkedDistinct, "distinct 归因不闭合");
say(`⇒ 那「2 个 distinct 键 / 31 条 raw」**根本不是键** —— 是 \`{table, kind, onlyOn, packs, corpus}\``);
say(`  这几个元数据属性名，被表侧计数器 countTableKeys 当成翻译键数了（它不 unwrap \`spec.table\`）。`);
say(`  ⇒ 差在**基准那一侧**，不是面板漏核。`);

/* ── 顺带：面板真正**没核**的键（口径差的实体），按 kind 分桶 —— 这才是该写进报文的东西 ── */
say("");
say(`═══ 面板按设计不计入 checkedDistinct 的真实键（这是真·口径差，与上面的伪键无关）═══`);
const buckets = [];
let uncheckedRaw = 0;
const uncheckedDistinct = new Set();
for (const r of wrapRows) {
  const counted = r.name === "MISSING_LANGUAGES";
  if (counted) continue;
  uncheckedRaw += r.realKeys;
  for (const k of Object.keys(TABLES[r.name].table)) uncheckedDistinct.add(k);
  buckets.push(r);
}
for (const r of buckets) say(`  ${r.name.padEnd(24)} kind=${(r.kind).padEnd(16)} onlyOn=${String(r.onlyOn).padEnd(8)} 真实键 ${r.realKeys}`);
say(`  合计：未计入 raw ${uncheckedRaw} / distinct ${uncheckedDistinct.size}`);
say(`  ⇒ 表里**登记**的真实键总数 = 裸表 ${plainRaw} + 包装表真实键 ${uncheckedRaw + ml.length} = ${plainRaw + uncheckedRaw + ml.length}（raw）`);

const result = {
  selfProof: { tables: 39, objTables: tk.objTables, arrTables: tk.arrTables, tableRaw: tk.raw,
               tableDistinct: tk.distinct, arrangementLeaves: 212,
               panelChecked: st.checkedDistinct, panelRaw: st.rawChecked,
               panelMissDistinct: st.missDistinct, panelRawMiss: st.rawMiss },
  attribution: {
    plainRaw, plainDistinct: plainDistinct.size,
    wrapPseudoRaw, wrapPseudoDistinct: [...wrapPseudoDistinct].sort(),
    missingLanguagesKeys: ml,
    rawGap: tk.raw - st.rawChecked, distinctGap: tk.distinct - st.checkedDistinct,
    verdict: "基准侧缺陷：countTableKeys 不 unwrap spec.table，把 {table,kind,onlyOn,packs,corpus} 元数据属性名当键数了"
  },
  uncheckedByDesign: { rawKeys: uncheckedRaw, distinctKeys: uncheckedDistinct.size, rows: buckets },
  registeredRealKeysRaw: plainRaw + uncheckedRaw + ml.length,
  fetch: { ok: fOk, fail: fFail },
  log
};
fs.writeFileSync(OUT, JSON.stringify(result, null, 1), "utf8");
say(`\n写出 ${OUT}`);
