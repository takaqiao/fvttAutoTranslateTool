import fs from "node:fs"; import path from "node:path"; import {pathToFileURL} from "node:url";
const SRC="C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
const raw=fs.readFileSync(SRC,"utf8");
const names=[...raw.matchAll(/^const ([A-Z][A-Z_0-9]*) = \{/gm)].map(m=>m[1]);
const exported=new Set([...raw.matchAll(/^export (?:const|function) ([A-Za-z_$][\w$]*)/gm)].map(m=>m[1]));
const need=names.filter(n=>!exported.has(n));
const h=path.join(process.cwd(),"_vn.mjs");
fs.writeFileSync(h,"globalThis.Hooks=globalThis.Hooks??{once(){},on(){}};\n"
 + raw.replace("import * as SELFCHECK from './ember-cn-selfcheck.mjs';","const SELFCHECK={SUBTREE_SELECTORS:[],registerSelfCheck(){},keyLiveness(){}};")
 + "\nexport {"+need.join(",")+", translateNotification};\n","utf8");
const M=await import(pathToFileURL(h).href); fs.unlinkSync(h);

// ① 新加的词在几张表里出现？（同一个词在两张表里给了不同中文＝分裂风险）
const NEW=["Vista","Event","Discovery","Area Map","Unique","Repeating","Playable","Barrier",
           "Surface of Ember","The Pathways","Chaotic Refraction","Drakonbane","Spark of Ember"];
console.log("=== 新词在各表的分布（同词不同译要盯）===");
for(const w of NEW){
  const hits=names.filter(n=>M[n]&&typeof M[n]==="object"&&!Array.isArray(M[n])&&(w in M[n])).map(n=>`${n}="${M[n][w]}"`);
  const vals=new Set(hits.map(h=>h.split('="')[1]));
  console.log(`  ${w.padEnd(20)} ${hits.join(" | ")||"(仅新表)"}${vals.size>1?"   ⚠ 同词不同译":""}`);
}
// ② 这 40 条现在覆盖了没有
const UNION={};
for(const n of names){const t=M[n]; if(t&&typeof t==="object"&&!Array.isArray(t)) for(const[k,v] of Object.entries(t)) if(typeof v==="string") UNION[k]??=v;}
const W=JSON.parse(fs.readFileSync("worklist.json","utf8"));
const GM=new Set(["EmberRandomTokenSchema","EmberSoundOrchestration","EmberCanvasCommonHelpers","registerKeybindings","EmberTerrainLayer","EmberRegionMapFogManager","EmberVista"]);
const todo=Object.entries(W).filter(([k])=>!GM.has(k)).flatMap(([,v])=>v);
const covered=(s)=>(s in UNION)||M.translateText(s,UNION)!==s||M.translateNotification(s)!==s
  || s.split(/<[^>]+>/).map(x=>x.trim()).filter(Boolean).every(p=>M.translateText(p,UNION)!==p);
const miss=todo.filter(x=>!covered(x));
console.log(`\n=== 待做 ${todo.length} 条 → 仍缺 ${miss.length} ===`);
miss.forEach(x=>console.log("   ",x.slice(0,80)));
