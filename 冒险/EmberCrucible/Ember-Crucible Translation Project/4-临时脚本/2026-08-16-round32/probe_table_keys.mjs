/**
 * 前置自证型探针（纪律 ⚠4）：先断言「我切出来的表数 = 已知真值」，再往下数键。
 *
 * 已知真值来自 R-selfcheck-d-liveness 的 min.tablesFedIn = 39（主闸每轮现跑，不是快照），
 * 以及面板 stats.wrappedTables（面板自己算的，第三十一轮 V14 归因的产物）。
 * 对不上就当场退出 —— 不许「切错了还照数」。
 *
 * 用法：node probe_table_keys.mjs <tables_src> <stub_import>
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [, , tablesSrc, stubImport] = process.argv;
if (!tablesSrc || !stubImport) {
  process.stderr.write("用法：node probe_table_keys.mjs <tables_src> <stub_import>\n");
  process.exit(2);
}

globalThis.Hooks = { once() {}, on() {} };
globalThis.game = { system: { id: "crucible" }, packs: [] };

let TABLES;
{
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ec-probe-"));
  try {
    const harness = path.join(tmpDir, "_hc_now.mjs");
    const src = fs.readFileSync(tablesSrc, "utf8");
    if (!src.includes(stubImport)) {
      process.stderr.write("被判文件里找不到 SELFCHECK import 行 —— 中止\n");
      process.exit(2);
    }
    fs.writeFileSync(harness,
      "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
      + src.replace(stubImport,
        "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
      + "\nexport { SELFCHECK_TABLES as __SELFCHECK_TABLES };\n", "utf8");
    TABLES = (await import(pathToFileURL(harness).href)).__SELFCHECK_TABLES;
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

/* ── 前置自证：切出来的条数必须等于已知真值 ── */
const names = Object.keys(TABLES);
const WANT_TABLES = 39;
if (names.length !== WANT_TABLES) {
  process.stderr.write(`前置自证失败：切出 ${names.length} 张表，已知真值 ${WANT_TABLES}\n`);
  process.exit(3);
}
const wrapped = names.filter((n) => {
  const s = TABLES[n];
  return s && !Array.isArray(s) && typeof s === "object"
    && Object.prototype.hasOwnProperty.call(s, "table");
});
process.stdout.write(`前置自证 ok：${names.length} 张表（已知真值 ${WANT_TABLES}）；包装形态 ${wrapped.length} 张\n`);

/* ── 旧算法（不解包）── */
function oldCount(tabs) {
  const seen = new Set();
  let raw = 0, regexEntries = 0, objTables = 0, arrTables = 0;
  for (const t of Object.values(tabs)) {
    if (Array.isArray(t)) { arrTables++; regexEntries += t.length; continue; }
    if (!t || typeof t !== "object") continue;
    objTables++;
    for (const k of Object.keys(t)) { raw++; seen.add(k); }
  }
  return { distinct: seen.size, raw, regexEntries, objTables, arrTables };
}

/* ── 新算法（穿过 `{table,…}` 包装，与面板 :1116 `spec?.table ?? spec` 同口径）── */
function newCount(tabs) {
  const seen = new Set();
  let raw = 0, regexEntries = 0, objTables = 0, arrTables = 0;
  for (const x of Object.values(tabs)) {
    const t = Array.isArray(x) ? x : (x?.table ?? x);
    if (Array.isArray(t)) { arrTables++; regexEntries += t.length; continue; }
    if (!t || typeof t !== "object") continue;
    objTables++;
    for (const k of Object.keys(t)) { raw++; seen.add(k); }
  }
  return { distinct: seen.size, raw, regexEntries, objTables, arrTables };
}

const o = oldCount(TABLES), n = newCount(TABLES);
process.stdout.write("旧（不解包）：" + JSON.stringify(o) + "\n");
process.stdout.write("新（解包）　：" + JSON.stringify(n) + "\n");

/* 元数据属性名被当成键的证据 */
const META = new Set(["table", "kind", "onlyOn", "packs", "corpus", "keyKinds"]);
const oldSeen = new Set();
for (const t of Object.values(TABLES)) {
  if (Array.isArray(t) || !t || typeof t !== "object") continue;
  for (const k of Object.keys(t)) oldSeen.add(k);
}
process.stdout.write("旧算法里落在集合中的元数据属性名："
  + [...oldSeen].filter((k) => META.has(k)).join(" / ") + "\n");
process.stdout.write(`包装表贡献的真实键：raw ${n.raw - (o.raw - wrapped.reduce((a, w) => a + Object.keys(TABLES[w]).length, 0))} （核对用）\n`);
