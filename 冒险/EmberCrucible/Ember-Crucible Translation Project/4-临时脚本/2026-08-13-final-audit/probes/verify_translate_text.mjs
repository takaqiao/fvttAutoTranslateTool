/**
 * verify_translate_text.mjs —— 复核：把上游增强器**实际产出的字符串**
 * 灌进 ember-hardcoded-cn.mjs 的 translateText，看哪些原样返回。
 *
 * 直接 import 插件里导出的 translateText，避免我口算 PREFIXED/EXACT 匹配顺序出错。
 * 只读，不写库。
 *
 * ⚠ 2026-08-16 第二十八轮改：原来 import 的是同目录的 `_shim_hardcoded.mjs` —— 那是一份
 *   2026-08-13 的**打桩副本**（PREFIXED 只有 4 条，真身当时已 19 条），抬头注释还逐字照搬真身。
 *   于是这支探针量的是一份过期副本，真身怎么改它都不知道 —— 本项目登记的空转形态 (h)
 *   「空转的是喂输入的那个探针」。诱饵已删；这里改成**现 import 真身**，
 *   只补一个 `globalThis.Hooks` 桩（真身 import 时会注册渲染钩子，Node 里没有）。
 */
globalThis.Hooks = globalThis.Hooks ?? {once() {}, on() {}};
const {translateText} = await import("../../../1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs");

const cases = [
  // ember 增强器实际输出（name/label 已是 babele 译好的中文并列名）
  ["ember enrichAncestry",       "Ancestry: 赫尔加伦 Hulgrun"],
  ["ember enrichCulture",        "Culture: 玛兹兰 Maziran"],
  ["ember enrichPath",           "Path: 光耀者 Lightsworn"],
  ["ember enrichAttunement",     "Attunement: 玛伊斯 Mayis"],
  ["ember enrichAttunement+奖励", "Attunement: 玛伊斯 Mayis (+1)"],
  ["ember enrichLanguage",       "Language: 卢玛语"],
  ["ember soundscape music",     "Music: Lyla Theme"],
  ["ember soundscape reset",     "Music: Reset"],
  ["ember soundscape mood",      "Music Mood: Calm"],
  ["crucible enrichTalent",      "Talent: 识别法术"],
  ["crucible enrichSpell tip",   "Spell tooltips are still TO-DO."],
  ["ember hero sheet h3",        "Cosmological Attunements"],
  ["ember hero sheet tag",       "Active"],
  ["ember hero sheet tooltip",   "Make Active"],
  ["ember hero sheet 月名",       "Abyss"],
  ["ember codex bestiary meta",  "Threat minion"],
  ["ember codex empty",          "Select a discovered creature from the left menu."],
  ["ember codex exit",           "Exit"],
  ["ember creation path hint",   "Spend 9 points across 6 ability scores, allocating up to 3 points per ability."],
  ["crucible lang category",     "Ancient Languages"],
  // 对照组：已知被覆盖的
  ["对照 事件状态",               "Event Completed"],
  ["对照 日历",                   "Day 1 - 12:00"],
  ["对照 恩惠骰",                 "+2 Boons"]
];

for ( const [what, s] of cases ) {
  const out = translateText(s);
  console.log(`${out === s ? "未翻译" : "已翻译"}  ${what.padEnd(26)} ${JSON.stringify(s)} -> ${JSON.stringify(out)}`);
}
