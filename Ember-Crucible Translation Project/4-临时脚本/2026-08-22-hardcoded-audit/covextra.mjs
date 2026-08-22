import fs from "node:fs"; import path from "node:path"; import {pathToFileURL} from "node:url";
const SRC="C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
const raw=fs.readFileSync(SRC,"utf8");
const names=[...raw.matchAll(/^const ([A-Z][A-Z_0-9]*) = \{/gm)].map(m=>m[1]);
const ex=new Set([...raw.matchAll(/^export (?:const|function) ([A-Za-z_$][\w$]*)/gm)].map(m=>m[1]));
const need=names.filter(n=>!ex.has(n));
const h=path.join(process.cwd(),"_ce.mjs");
fs.writeFileSync(h,"globalThis.Hooks=globalThis.Hooks??{once(){},on(){}};\n"
 + raw.replace("import * as SELFCHECK from './ember-cn-selfcheck.mjs';","const SELFCHECK={SUBTREE_SELECTORS:[],registerSelfCheck(){},keyLiveness(){}};")
 + "\nexport {"+need.join(",")+", translateNotification};\n","utf8");
const M=await import(pathToFileURL(h).href); fs.unlinkSync(h);
const U={}; for(const n of names){const t=M[n]; if(t&&typeof t==="object"&&!Array.isArray(t)) for(const[k,v] of Object.entries(t)) if(typeof v==="string") U[k]??=v;}
const E=JSON.parse(fs.readFileSync("extra_fields.json","utf8"));
const all=[...new Set(Object.values(E).flat())];
const miss=all.filter(s=>!(s in U) && M.translateText(s,U)===s);
console.log(`补扫 ${all.length} 条 · 已覆盖 ${all.length-miss.length} · 仍缺 ${miss.length}`);
miss.forEach(s=>console.log("   ",s));
