import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { fileURLToPath } from "node:url";
const HERE = path.dirname(fileURLToPath(import.meta.url));
const M = await import(pathToFileURL(path.join(HERE, "_harness.mjs")).href);
const snippet = fs.readFileSync(path.join(HERE, "token_maker_ui.snippet.js"), "utf8");
const NEW = eval("({" + snippet.replace(/^\s*\/\/.*$/gm, "") + "})");
console.log("新表键数", Object.keys(NEW).length);
for (const [tname, t] of [["EXACT", M.EXACT], ["EMBER_WINDOW_UI", M.EMBER_WINDOW_UI],
                          ["DIALOG_UI", M.DIALOG_UI], ["CHAT_UI", M.CHAT_UI],
                          ["SETTINGS_UI", M.SETTINGS_UI], ["SCENE_CONTROL_UI", M.SCENE_CONTROL_UI],
                          ["MOOD_PANEL", M.MOOD_PANEL], ["ATTUNEMENTS", M.ATTUNEMENTS]]) {
  const same = [], diff = [];
  for (const k of Object.keys(NEW)) {
    if (!(k in t)) continue;
    (t[k] === NEW[k] ? same : diff).push([k, t[k], NEW[k]]);
  }
  if (same.length || diff.length)
    console.log(`  ${tname}: 同键同译 ${same.length}${same.length ? " " + JSON.stringify(same) : ""} | 同键异译 ${diff.length}${diff.length ? " " + JSON.stringify(diff) : ""}`);
}
// 用真 translateText 看看这些键在**不带作用域表**时会不会被别的表吃掉
let eaten = [];
for (const k of Object.keys(NEW)) { const t = M.translateText(k); if (t !== k) eaten.push([k, t]); }
console.log("不带作用域表时已被全局通道翻动的键：", eaten.length, JSON.stringify(eaten));
