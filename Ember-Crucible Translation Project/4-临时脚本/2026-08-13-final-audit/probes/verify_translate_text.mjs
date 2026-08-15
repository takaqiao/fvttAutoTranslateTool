/**
 * verify_translate_text.mjs —— 复核：把上游增强器**实际产出的字符串**
 * 灌进 ember-hardcoded-cn.mjs 的 translateText，看哪些原样返回。
 *
 * 直接 import 插件里导出的 translateText，避免我口算 PREFIXED/EXACT 匹配顺序出错。
 * 只读，不写库。
 */
import {translateText} from "./_shim_hardcoded.mjs";

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
