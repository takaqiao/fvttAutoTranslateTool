/**
 * 拆词等价探针：我们的 `splitPartId()` 与**上游那一行原样的正则**必须逐字符相同。
 *
 * ⚠ 上游那条正则**从 ember.mjs 里现抠**，不是抄一份 —— 抄一份就等于测了个副本。
 *   抠不到（上游改了写法 / 换了打包格式）**当场失败**，不许静默跳过。
 *
 * 前置自证（两件）：
 *   P1 切对条数 —— 抠出来的正则源码必须**恰好一处**命中；样本条数 == 声明的条数。
 *   P2 切对对象 —— 样本必须覆盖五种形态各至少一条：带数字的 / 连续大写的 / 单个单词的 /
 *      腿姿拼串的 / 含 0 的（`[A-Z1-9]` 不含 0 的那条怪癖），逐条断言形态判据成立。
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const EMBER = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs";
const TABLES = process.argv[2];
const OUT = process.argv[3];
const die = (m) => { console.error("split_equiv: " + m); process.exit(2); };

/* ── 从上游现抠那一行 ── */
const src = fs.readFileSync(EMBER, "utf8");
const hits = [...src.matchAll(/choice\.label = partId\.split\("\/"\)\.at\(-1\)\.replace\((\/.+?\/g), " \$1"\);/g)];
if (hits.length !== 1) die(`P1 FAIL 上游 getLayerChoicesV2 的拆词行命中 ${hits.length} 处，期望恰好 1 处`);
const reSrc = hits[0][1];
console.log(`P1 OK  上游拆词行恰好命中 1 处，正则源码 = ${reSrc}`);
// eslint-disable-next-line no-eval
const upstreamRe = eval(reSrc);
const upstream = (id) => id.split("/").at(-1).replace(upstreamRe, " $1");

/* ── 载入被判文件（只加导出，函数体一个字节不改）── */
const STUB = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const NAMES = ["TOKEN_MAKER_PART_IDS", "TOKEN_MAKER_PARTS", "LEG_POSE_BASES",
               "LEG_POSE_SUFFIXES", "MARBLED_HAND_BASES", "splitPartId"];
const s = fs.readFileSync(TABLES, "utf8");
if (!s.includes(STUB)) die("找不到要打桩的 import 行");
for (const n of NAMES) if (!new RegExp("\\n(?:const|function) " + n + "[ (=]").test(s)) die(`找不到顶层声明 ${n}`);
const h = path.join(path.dirname(OUT), "_split_harness.mjs");
fs.writeFileSync(h, "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
  + s.replace(STUB, "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
  + `\nexport { ${NAMES.join(", ")} };\n`, "utf8");
const M = await import(pathToFileURL(h).href);

/* ── P2 样本：五种形态各至少一条 ── */
const SAMPLE = [
  // 带数字
  "Beard1", "Smooth5", "Bare1a", "Bare2Pattern3", "Plated4", "Scaled1",
  // 连续大写
  "BeardWizard", "BrokenAedirHeavy", "SoddenDecayedPalmDown", "ChiseledMasculineMarbled",
  "WhiskersDown2", "SpikesLarge1",
  // 单个单词
  "Cheeks", "Woody", "Hoof", "Claw",
  // 腿姿拼串
  "Smooth1Backward", "BrokenArmoredNeutral", "AedirHeavySitting", "MarbledForward",
  // 含 0（`[A-Z1-9]` 不含 0）
  "Beard10", "Beard11"
];
const hasDigit = SAMPLE.filter(x => /\d/.test(x)).length;
const hasCaps = SAMPLE.filter(x => /[a-z][A-Z]/.test(x)).length;
const hasWord = SAMPLE.filter(x => /^[A-Z][a-z]+$/.test(x)).length;
const hasPose = SAMPLE.filter(x => /(Backward|Forward|Neutral|Sitting)$/.test(x)).length;
const hasZero = SAMPLE.filter(x => /0/.test(x)).length;
if (SAMPLE.length < 20) die(`P2 FAIL 样本只有 ${SAMPLE.length} 条，任务书要求 ≥20`);
if (!(hasDigit && hasCaps && hasWord && hasPose && hasZero))
  die(`P2 FAIL 五种形态覆盖不全：数字${hasDigit} 大写${hasCaps} 单词${hasWord} 腿姿${hasPose} 含0${hasZero}`);
console.log(`P2 OK  样本 ${SAMPLE.length} 条，五种形态齐备（带数字 ${hasDigit} · 连续大写 ${hasCaps} · `
  + `单词 ${hasWord} · 腿姿 ${hasPose} · 含 0 ${hasZero}）`);

const bad = [];
for (const id of SAMPLE) {
  const a = upstream(id), b = M.splitPartId(id);
  if (a !== b) bad.push([id, a, b]);
}
console.log(`① 样本 ${SAMPLE.length} 条：上游原样正则 vs 本模块 splitPartId —— 不一致 ${bad.length} 条`);
if (bad.length) console.log("   ", bad.slice(0, 8));

/* ── ② 全量：表里 671 个 id 逐条对 ── */
const ids = Object.keys(M.TOKEN_MAKER_PART_IDS);
const bad2 = ids.filter(id => upstream(id) !== M.splitPartId(id));
console.log(`② 全表 ${ids.length} 个 id 逐条对：不一致 ${bad2.length} 条`);

/* ── ③ 拼串族：显示名必须等于「上游对拼出来的 id 现算的结果」 ── */
const bad3 = [];
let n3 = 0;
for (const base of M.LEG_POSE_BASES) for (const pose of M.LEG_POSE_SUFFIXES) {
  const id = base + pose; n3++;
  const want = upstream(id);
  if (!(want in M.TOKEN_MAKER_PARTS)) bad3.push([id, want, "(不在显示名表里)"]);
}
for (const base of M.MARBLED_HAND_BASES) {
  const id = "Marbled" + base; n3++;
  const want = upstream(id);
  if (!(want in M.TOKEN_MAKER_PARTS)) bad3.push([id, want, "(不在显示名表里)"]);
}
console.log(`③ 拼串族 ${n3} 条：拼出来的 id 用上游正则算出的显示名，表里查不到的 ${bad3.length} 条`);
if (bad3.length) console.log("   ", bad3.slice(0, 8));

/* ── ④ 显示名表条数 ── */
const nDisp = Object.keys(M.TOKEN_MAKER_PARTS).length;
console.log(`④ 显示名表共 ${nDisp} 条（id 表 ${ids.length} + 拼串 ${nDisp - ids.length}）`);

const pass = !bad.length && !bad2.length && !bad3.length && nDisp === ids.length + 107;
fs.writeFileSync(OUT, JSON.stringify({ reSrc, sample: SAMPLE.length, bad, bad2, bad3, nDisp, ids: ids.length, pass }, null, 1), "utf8");
console.log(pass ? "\n拆词等价 PASS" : "\n拆词等价 FAIL");
process.exit(pass ? 0 : 1);
