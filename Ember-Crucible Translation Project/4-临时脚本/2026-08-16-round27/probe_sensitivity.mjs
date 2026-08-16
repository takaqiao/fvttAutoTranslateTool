/**
 * 灵敏度回测（第二十七轮验收用，一次性）：
 * **逐条**改坏 PREFIXED 19 / PATTERNS 28 / NOTIFICATION_PATTERNS 31，
 * 每一条都必须让 `R-patterns-translate-cases` 的用例集**至少出现一处违规**。
 *
 * 为什么可以用「运行时改对象」而不是「改源码再跑主闸」
 * ----------------------------------------------------
 * 三张表是 `const 数组`，但**元素是可变对象**，而 `translateText` / `translateNotification`
 * 每次调用都现读 `PREFIXED[i].cn` / `PATTERNS[i].cn` / `NOTIFICATION_PATTERNS[i].cn`。
 * 所以「把第 i 条的译文换掉」在语义上与「在源码里把第 i 条的译文改坏」等价，
 * 而闸的判决本来就只有一个口径：**用例集里有没有出现违规**。
 * ⚠ 用例集**直接从发布中的 `RESOLUTIONS.assertions.json` 读**，不在这里另抄一份
 *   （抄一份就是拿自检验一份和产线不同的配置）。
 * ⚠ 主闸那一侧的源码级灵敏度回测另有一份、固化在 `--selftest` 里（第 9–13 条）。
 *
 * 用法：node probe_sensitivity.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const P = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const SRC = `${P}/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs`;
const RULES = `${P}/5-其他内容/RESOLUTIONS.assertions.json`;
const STUB = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const OUT = path.join(path.dirname(process.argv[1]), "_sens_harness.mjs");

const src = fs.readFileSync(SRC, "utf8");
if (!src.includes(STUB)) { console.error("找不到要打桩的 import 行"); process.exit(2); }
fs.writeFileSync(OUT,
  "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
  + src.replace(STUB, "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
  + "\nexport { translateNotification, PREFIXED, PATTERNS, NOTIFICATION_PATTERNS };\n",
  "utf8");

const M = await import(pathToFileURL(OUT).href);
const rule = JSON.parse(fs.readFileSync(RULES, "utf8")).assertions
  .find((r) => r.kind === "translate_cases");
if (!rule) { console.error("规则文件里没有 translate_cases 规则"); process.exit(2); }

/** 跑一遍用例集，返回违规条数（口径与 translate_cases_runner 的 ②③⑤ 段一致）。 */
function violations() {
  let n = 0;
  for (const s of rule.negative) if (M.translateText(s) !== s) n++;
  for (const [i, w] of rule.positive) if (M.translateText(i) !== w) n++;
  for (const s of rule.notify_negative) if (M.translateNotification(s) !== s) n++;
  for (const [i, w] of rule.notify_positive) if (M.translateNotification(i) !== w) n++;
  return n;
}

const base = violations();
const report = { baseline_violations: base, groups: {} };

function sweep(name, table, mutate, restore) {
  const rows = [];
  for (let i = 0; i < table.length; i++) {
    const saved = restore(table[i]);
    mutate(table[i]);
    const n = violations();
    Object.assign(table[i], saved);
    rows.push({ i, label: String(table[i].en ?? table[i].re), violations: n, red: n > base });
  }
  const red = rows.filter((r) => r.red).length;
  report.groups[name] = { size: table.length, red, silent: rows.filter((r) => !r.red) };
  return red === table.length;
}

const okP = sweep("PREFIXED", M.PREFIXED,
  (e) => { e.cn = "改坏了"; }, (e) => ({ cn: e.cn }));
const okT = sweep("PATTERNS", M.PATTERNS,
  (e) => { e.cn = () => "改坏了"; }, (e) => ({ cn: e.cn }));
const okN = sweep("NOTIFICATION_PATTERNS", M.NOTIFICATION_PATTERNS,
  (e) => { e.cn = () => "改坏了"; }, (e) => ({ cn: e.cn }));

report.all_green_after_restore = violations() === base;
report.verdict = { PREFIXED: okP, PATTERNS: okT, NOTIFICATION_PATTERNS: okN };
console.log(JSON.stringify(report, null, 1));
process.exit(okP && okT && okN && report.all_green_after_restore ? 0 : 1);
