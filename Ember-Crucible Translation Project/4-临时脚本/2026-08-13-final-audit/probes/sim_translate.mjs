// 直接 import 汉化模块的 translateText 做端到端模拟（只读）
import { readFileSync } from "node:fs";
const p = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
let src = readFileSync(p, "utf8");
// 剥掉需要 Foundry 全局的部分：只留到「翻译引擎」结束
src = src.slice(0, src.indexOf("/*  3. 补丁"));
const mod = await import("data:text/javascript;base64," + Buffer.from(src).toString("base64"));
const T = mod.translateText;
const cases = [
  "Ancestry: 凯斯 Keth", "Culture: 奥尔达尼 Ordani", "Path: 学院辍学者 Academy Dropout",
  "Attunement: Heart of Ember", "Language: Luma", "Knowledge: Alchemy",
  "Music Mood: Calm", "Music: Reset", "Music: Lyla Theme", "Environment: Shent Ruins",
  "Ancestry", "Culture", "Path", "Attunement", "Token", "Class",
  "Ring", "Enable Flow", "Close Valve", "Interact", "Observe",
  "Ring Alarm Bell?", "Modify Flow Control Valve?", "Steam Cleansing Cutoff",
  "Ancient Languages", "Obscure Languages", "Standard Languages",
  "Knowledge: Stars", "Award Attunement: Aura (+2)", "Activate Attunement: Aura",
  "Revoke Attunement: Aura", "Interactable: AlarmBell1", "Vantage Point: Sunset Overlook"
];
for (const c of cases) {
  const out = T(c);
  console.log(`${out === c ? "✘未翻" : "✔已翻"}  ${JSON.stringify(c)}  ->  ${JSON.stringify(out)}`);
}
