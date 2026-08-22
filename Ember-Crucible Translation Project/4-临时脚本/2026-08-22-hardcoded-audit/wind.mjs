import fs from "node:fs"; import path from "node:path"; import {pathToFileURL} from "node:url";
const SRC="C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
const s=fs.readFileSync(SRC,"utf8"); const h=path.join(process.cwd(),"_wd.mjs");
fs.writeFileSync(h,"globalThis.Hooks=globalThis.Hooks??{once(){},on(){}};\n"
 + s.replace("import * as SELFCHECK from './ember-cn-selfcheck.mjs';","const SELFCHECK={SUBTREE_SELECTORS:[],registerSelfCheck(){},keyLiveness(){}};")
 + "\nexport {EMBER_WINDOW_UI};\n","utf8");
const M=await import(pathToFileURL(h).href); fs.unlinkSync(h);
for (const t of ["Breeze (12 mph)","Calm (0 mph)","Gale (48 mph)","Squall (60 mph)","Windy (24 mph)"])
  console.log(t.padEnd(20), "→", M.translateText(t, M.EMBER_WINDOW_UI));
