import fs from "node:fs"; import path from "node:path"; import {pathToFileURL} from "node:url";
const SRC = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
const s = fs.readFileSync(SRC, "utf8");
const h = path.join(process.cwd(), "_bt.mjs");
const N = ["translateNotification","DIALOG_UI","EMBER_WINDOW_UI","TOKEN_MAKER_UI","TOKEN_MAKER_PARTS","MOOD_PANEL","ATTUNEMENTS","DIALOG_TITLES"];
fs.writeFileSync(h, "globalThis.Hooks=globalThis.Hooks??{once(){},on(){}};\n"
  + s.replace("import * as SELFCHECK from './ember-cn-selfcheck.mjs';","const SELFCHECK={SUBTREE_SELECTORS:[],registerSelfCheck(){},keyLiveness(){}};")
  + "\nexport {" + N.join(",") + "};\n","utf8");
const M = await import(pathToFileURL(h).href); fs.unlinkSync(h);
const SC = {...M.DIALOG_UI,...M.EMBER_WINDOW_UI,...M.TOKEN_MAKER_UI,...M.TOKEN_MAKER_PARTS,...M.MOOD_PANEL,...M.ATTUNEMENTS,...M.DIALOG_TITLES};
const L = JSON.parse(fs.readFileSync("backtick_html.json","utf8"));
const miss = L.filter(x => M.translateText(x, SC) === x && M.translateNotification(x) === x);
console.log(`反引号 HTML 文本 ${L.length} 条 · 已盖 ${L.length-miss.length} · 仍缺 ${miss.length}`);
console.log(miss.slice(0,30).map(x=>"   "+x.slice(0,72)).join("\n"));
