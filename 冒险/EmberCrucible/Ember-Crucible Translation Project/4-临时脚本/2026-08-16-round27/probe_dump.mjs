/**
 * 一次性探针：把候选串喂给**发布中的** translateText / translateNotification，打印实得输出。
 *
 * 只用来**读**当前行为以便把正例的期望值逐字符固化进断言（人工逐条核对译文对不对），
 * 不承担判据职责 —— 判据是 `R-patterns-translate-cases` 那条常设闸。
 *
 * ⚠ 打桩方式与 translate_cases_runner.mjs 完全一致（同一行 import 替换 + Hooks 桩），
 *   并且**额外把未导出的 translateNotification 导出**（追加一行 export，不改函数体）。
 *
 * 用法：node probe_dump.mjs <in.json>   in.json = {"text": [...], "notify": [...]}
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const SRC = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
const STUB = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const OUT = path.join(path.dirname(process.argv[1]), "_probe_harness.mjs");

let src = fs.readFileSync(SRC, "utf8");
if (!src.includes(STUB)) { console.error("找不到要打桩的 import 行"); process.exit(2); }
if (!/\nfunction translateNotification\(/.test(src)) {
  console.error("找不到 translateNotification 的定义"); process.exit(2);
}
fs.writeFileSync(OUT,
  "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
  + src.replace(STUB, "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
  + "\nexport { translateNotification, PREFIXED, PATTERNS, NOTIFICATION_PATTERNS };\n",
  "utf8");

const mod = await import(pathToFileURL(OUT).href);
const spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const rows = [];
for (const s of spec.text || []) rows.push(["T", s, mod.translateText(s)]);
for (const s of spec.notify || []) rows.push(["N", s, mod.translateNotification(s)]);
console.log(JSON.stringify({
  counts: {
    prefixed: mod.PREFIXED.length,
    patterns: mod.PATTERNS.length,
    notification_patterns: mod.NOTIFICATION_PATTERNS.length
  },
  rows
}, null, 1));
