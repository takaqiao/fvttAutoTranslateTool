import fs from "node:fs"; import path from "node:path"; import {pathToFileURL} from "node:url";
const SRC="C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
const s=fs.readFileSync(SRC,"utf8"); const h=path.join(process.cwd(),"_w.mjs");
const N=["translateNotification","DIALOG_UI","EMBER_WINDOW_UI","TOKEN_MAKER_UI","TOKEN_MAKER_PARTS","MOOD_PANEL","ATTUNEMENTS","DIALOG_TITLES","EXACT"];
fs.writeFileSync(h,"globalThis.Hooks=globalThis.Hooks??{once(){},on(){}};\n"
 + s.replace("import * as SELFCHECK from './ember-cn-selfcheck.mjs';","const SELFCHECK={SUBTREE_SELECTORS:[],registerSelfCheck(){},keyLiveness(){}};")
 + "\nexport {"+N.join(",")+"};\n","utf8");
const M=await import(pathToFileURL(h).href); fs.unlinkSync(h);
const SC={...M.DIALOG_UI,...M.EMBER_WINDOW_UI,...M.TOKEN_MAKER_UI,...M.TOKEN_MAKER_PARTS,...M.MOOD_PANEL,...M.ATTUNEMENTS,...M.DIALOG_TITLES};
const L=JSON.parse(fs.readFileSync("weather.json","utf8"));
for (const [type,kind,v] of L) {
  const g = M.translateText(v, null);              // 全局 EXACT/PATTERNS
  const sc = M.translateText(v, SC);               // 并表作用域
  const win = M.translateText(v, M.EMBER_WINDOW_UI); // 日历真正用的那张表
  console.log(`${(type??"").padEnd(12)}${kind}  ${v.padEnd(16)} 全局=${g===v?"—":g}  并表=${sc===v?"—":sc}  窗口表=${win===v?"—":win}`);
}
