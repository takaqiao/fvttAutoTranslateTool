/**
 * 拿**发布中的** ember-hardcoded-cn.mjs 本体量指示物制作器那两个窗口的覆盖率。
 * 造 harness 副本，只追加 export，函数体一字不改（同 recon/classify.mjs 的做法）。
 *
 * 用法：node coverage.mjs <in.json:{layer_labels:{},part_display:{}}> <out.json>
 *
 * 前置自证（两件都断言，任一不成立当场 exit 2）：
 *   ① **切对条数**：读进来的条数 == 输入 JSON 里的条数（不静默丢键）；
 *   ② **切对对象**：**负控制** —— 不传作用域表时，已知在 TOKEN_MAKER_UI 里的
 *      `Alloy` / `Pauldron Left` 必须**一条都不翻**（证明量的是作用域表，不是全局 EXACT）；
 *      **正控制** —— 传作用域表时这两条必须翻。两道都过才继续。
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const SRC = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
const STUB_IMPORT = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const die = (m) => { console.error("coverage: " + m); process.exit(2); };

let src = fs.readFileSync(SRC, "utf8");
if (!src.includes(STUB_IMPORT)) die("找不到要打桩的 import 行");

const NAMES = ["TOKEN_MAKER_UI", "TOKEN_MAKER_WINDOW_UI", "EMBER_WINDOW_UI", "EXACT", "DIALOG_UI"];
for (const n of NAMES) {
  if (!new RegExp("\\nconst " + n + " = ").test(src)) die(`找不到 ${n} 的顶层声明`);
}

const harness = path.join(process.cwd(), "_cov_harness.mjs");
fs.writeFileSync(harness,
  "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
  + src.replace(STUB_IMPORT,
      "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
  + `\nexport { ${NAMES.join(", ")} };\n`,
  "utf8");

const M = await import(pathToFileURL(harness).href);
const { translateText } = M;
const { TOKEN_MAKER_UI, TOKEN_MAKER_WINDOW_UI, EMBER_WINDOW_UI, EXACT } = M;

/* ── 前置自证 ② 负控制 / 正控制 ── */
for (const probe of ["Alloy", "Pauldron Left"]) {
  if (!(probe in TOKEN_MAKER_UI)) die(`前置自证失败：${probe} 不在 TOKEN_MAKER_UI 里`);
  if (translateText(probe) !== probe) die(`负控制失败：不传作用域表时 ${probe} 竟被翻了`);
  if (translateText(probe, TOKEN_MAKER_WINDOW_UI) === probe) die(`正控制失败：传作用域表时 ${probe} 没被翻`);
}
console.log("PRECHECK-负控制/正控制 OK（Alloy · Pauldron Left：全局不翻、作用域翻）");

const input = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = {};
for (const [group, items] of Object.entries(input)) {
  const keys = Array.isArray(items) ? items : Object.keys(items);
  const res = { n: keys.length, covered: [], gapScoped: [], globalHit: [] };
  for (const k of keys) {
    const scoped = translateText(k, TOKEN_MAKER_WINDOW_UI);
    const global = translateText(k);
    if (scoped !== k) res.covered.push(k); else res.gapScoped.push(k);
    if (global !== k) res.globalHit.push([k, global]);
  }
  /* ── 前置自证 ① 切对条数 ── */
  if (res.covered.length + res.gapScoped.length !== keys.length) die(`条数对不上：${group}`);
  out[group] = res;
  console.log(`${group}: 共 ${keys.length} → 作用域内已覆盖 ${res.covered.length} / 缺 ${res.gapScoped.length}`
    + `（其中全局通道也命中的 ${res.globalHit.length} 条）`);
}
fs.writeFileSync(process.argv[3], JSON.stringify(out, null, 1), "utf8");
console.log("WROTE " + process.argv[3]);
