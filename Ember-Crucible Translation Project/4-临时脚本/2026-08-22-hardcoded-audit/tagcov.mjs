import fs from "node:fs"; import path from "node:path"; import {pathToFileURL} from "node:url";
const SRC = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
const s = fs.readFileSync(SRC,"utf8"); const h = path.join(process.cwd(),"_tc.mjs");
const N=["translateNotification","DIALOG_UI","EMBER_WINDOW_UI","TOKEN_MAKER_UI","TOKEN_MAKER_PARTS","MOOD_PANEL","ATTUNEMENTS","DIALOG_TITLES"];
fs.writeFileSync(h,"globalThis.Hooks=globalThis.Hooks??{once(){},on(){}};\n"
 + s.replace("import * as SELFCHECK from './ember-cn-selfcheck.mjs';","const SELFCHECK={SUBTREE_SELECTORS:[],registerSelfCheck(){},keyLiveness(){}};")
 + "\nexport {"+N.join(",")+"};\n","utf8");
const M = await import(pathToFileURL(h).href); fs.unlinkSync(h);
const SC = {...M.DIALOG_UI,...M.EMBER_WINDOW_UI,...M.TOKEN_MAKER_UI,...M.TOKEN_MAKER_PARTS,...M.MOOD_PANEL,...M.ATTUNEMENTS,...M.DIALOG_TITLES};
const L = JSON.parse(fs.readFileSync("tagged.json","utf8"));
let ok=0; const miss=[];
for (const raw of L) {
  // 与运行时同口径：DOM 会把 `<p>X</p>` 拆成文本节点 X
  const parts = raw.split(/<[^>]+>/).map(x=>x.replace(/\s+/g," ").trim()).filter(Boolean);
  const allCovered = parts.length>0 && parts.every(p => M.translateText(p,SC)!==p || M.translateNotification(p)!==p);
  if (allCovered) ok++; else miss.push(raw.slice(0,80));
}
console.log(`拆成文本节点后：已盖 ${ok} / ${L.length}，仍缺 ${miss.length}`);
miss.forEach(x=>console.log("   "+x));
