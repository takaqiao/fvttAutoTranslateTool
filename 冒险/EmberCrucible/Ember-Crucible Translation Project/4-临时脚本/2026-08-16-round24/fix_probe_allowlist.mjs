/**
 * 核两件事（都在动文件之前）：
 *   ① EMBER_WINDOW_UI 里**注释写着 ember.mjs:NNNN 出处**的那 11 条，是不是真的能在
 *      判据语料（ember.mjs + crucible-compiled.mjs）里找到 —— 只有全找得到，
 *      「白名单 11 条继续核、其余按 .hbs 来源跳过」这个改法才不会把绿的变成红的。
 *   ② VISTA_PLACEMENT_EN 的 14 个**英文显示串**（不是字段路径）是不是都找得到 ——
 *      报告③建议改核这一批，得先证明它比核路径强。
 * 顺带把 EMBER_WINDOW_UI 当前「找得到」的 28 条全列出来，看白名单之外那 17 条是不是
 * 都属于「.hbs 来源但巧合命中」（那正是判据无因果性的证据）。
 */
import fs from "node:fs";

const EMBER = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs";
const CRUCIBLE = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/crucible-compiled.mjs";
const corpus = fs.readFileSync(EMBER, "utf8") + "\n" + fs.readFileSync(CRUCIBLE, "utf8");
const has = (s) => corpus.includes(s);

const mod = await import(process.argv[2]);

const MJS_SOURCED = [
  "Topic Overview", "Overview", "Secret Information",
  "Ancestry", "Class",
  "Ember Token Maker", "Randomization Rules", "Build", "Stance",
  "Play Animation", "Stop Animation"
];

console.log("① EMBER_WINDOW_UI 白名单（注释出处写的是 ember.mjs:NNNN）");
for (const k of MJS_SOURCED) {
  const inTable = k in mod.__EMBER_WINDOW_UI;
  console.log(`   ${has(k) ? "找得到" : "查无此串"}  在表里=${inTable}  ${JSON.stringify(k)}`);
}
const found = Object.keys(mod.__EMBER_WINDOW_UI).filter(has);
console.log(`\n   当前找得到的共 ${found.length} 条；白名单之外的 ${found.length - MJS_SOURCED.length} 条（.hbs 来源，纯巧合命中）：`);
for (const k of found) if (!MJS_SOURCED.includes(k)) console.log(`   · ${JSON.stringify(k)}`);

console.log("\n② VISTA_PLACEMENT_EN 的 14 个英文显示串");
for (const [p, en] of Object.entries(mod.__VISTA_PLACEMENT_EN)) {
  console.log(`   ${has(en) ? "找得到" : "查无此串"}  ${JSON.stringify(en).padEnd(26)} ← 路径 ${p}（路径本身：${has(p) ? "找得到" : "查无此串"}）`);
}
