/* 文档这一路（第三十一轮）：把「零常设判据」那个减法按**订正后的分母**重算一遍。
 *
 * 背景：第二十九／三十轮写的是 `721 − 212 = 509`，而 721 这个分母已被 ② 号路归因证明是错的
 * （countTableKeys 不 unwrap `spec.table`：多算 5 个元数据属性名、漏掉包装里 189 个真键）。
 * 真分母是**穿过包装数出来的** distinct 887 / raw 1459。
 *
 * 本脚本只做两件事：① 导出订正后的 distinct 真实键全集；② 顺带把 ARRANGEMENT_LEAVES 交给 python 那一半。
 * 前置自证：不 unwrap 必须复现 721/1304，unwrap 必须得 887/1459 —— 对不上当场退出。
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const SRC = path.join(ROOT, "1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs");
const OUT = path.join(ROOT, "4-临时脚本/2026-08-16-round31/doc_probe_zero_judge.keys.json");
const STUB = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";

globalThis.Hooks = { once() {}, on() {} };
let TABLES;
{
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ec-zj-"));
  try {
    const h = path.join(tmpDir, "_hc_now.mjs");
    const src = fs.readFileSync(SRC, "utf8");
    if (!src.includes(STUB)) { console.error("找不到 import 行"); process.exit(2); }
    fs.writeFileSync(h,
      "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
      + src.replace(STUB, "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
      + "\nexport { SELFCHECK_TABLES as __SELFCHECK_TABLES, ARRANGEMENT_LEAVES as __LEAVES };\n", "utf8");
    const m = await import(pathToFileURL(h).href);
    TABLES = m.__SELFCHECK_TABLES;
    var LEAVES = Object.keys(m.__LEAVES);
  } finally { fs.rmSync(tmpDir, { recursive: true, force: true }); }
}

function count(tabs, unwrap) {
  const seen = new Set();
  let raw = 0;
  for (const x of Object.values(tabs)) {
    const t = unwrap ? (Array.isArray(x) ? x : (x?.table ?? x)) : x;
    if (Array.isArray(t) || !t || typeof t !== "object") continue;
    for (const k of Object.keys(t)) { raw++; seen.add(k); }
  }
  return { seen, raw };
}

const bad = count(TABLES, false), good = count(TABLES, true);
const want = [["不 unwrap raw", bad.raw, 1304], ["不 unwrap distinct", bad.seen.size, 721],
              ["unwrap raw", good.raw, 1459], ["unwrap distinct", good.seen.size, 887],
              ["ARRANGEMENT_LEAVES distinct", new Set(LEAVES).size, 212]];
for (const [n, got, exp] of want) {
  console.log(`[前置自证] ${n} = ${got}（已知真值 ${exp}）`);
  if (got !== exp) { console.error(`  ✗ 对不上，退出`); process.exit(3); }
}

const leaves = new Set(LEAVES);
const reg = [...good.seen];
const notLeaf = reg.filter(k => !leaves.has(k));
const leafNotReg = [...leaves].filter(k => !good.seen.has(k));
const oldReg = [...bad.seen];
const oldNotLeaf = oldReg.filter(k => !leaves.has(k));

console.log("");
console.log(`旧分母（错的）：721 − 212 = ${721 - 212}（纯减法）· 真集合差 = ${oldNotLeaf.length}`);
console.log(`新分母（订正）：887 − 212 = ${887 - 212}（纯减法）· **真集合差 = ${notLeaf.length}**`);
console.log(`ARRANGEMENT_LEAVES 里不在登记真实键集合中的：${leafNotReg.length} 个`);
fs.writeFileSync(OUT, JSON.stringify({
  registeredDistinct: reg.length, leaves: leaves.size,
  notLeaf: notLeaf.length, leafNotReg, oldSetDiff: oldNotLeaf.length,
}, null, 1), "utf8");
console.log("写出 " + OUT);
