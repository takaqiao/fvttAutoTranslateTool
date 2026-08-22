/** 新增键与现有各表的碰撞体检。用法：node collide.mjs <keys.json:[...]> */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const SRC = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
const STUB_IMPORT = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
let src = fs.readFileSync(SRC, "utf8");
const NAMES = ["TOKEN_MAKER_UI", "TOKEN_MAKER_WINDOW_UI", "EMBER_WINDOW_UI", "EXACT", "DIALOG_UI",
               "EMBER_DIALOG_UI", "MOOD_PANEL", "CHAT_UI", "SETTINGS_UI", "SCENE_CONTROL_UI",
               "NOTE_TYPES", "WEATHER", "ATTUNEMENT_TAB", "DIALOG_TITLES"];
const harness = path.join(process.cwd(), "_col_harness.mjs");
fs.writeFileSync(harness,
  "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
  + src.replace(STUB_IMPORT, "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
  + `\nexport { ${NAMES.join(", ")} };\n`, "utf8");
const M = await import(pathToFileURL(harness).href);

const keys = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
console.log(`体检 ${keys.length} 个新键`);
let n = 0;
for (const name of NAMES) {
  const t = M[name];
  if (!t || Array.isArray(t)) continue;
  const hit = keys.filter(k => k in t);
  if (hit.length) { n += hit.length; console.log(`  ${name}: ${hit.length} 撞 → ${hit.map(k => `${k}=${t[k]}`).join(" · ")}`); }
}
if (!n) console.log("  与以上 14 张表零碰撞");
