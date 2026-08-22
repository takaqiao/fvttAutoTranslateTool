/**
 * 覆盖率**重测**：把插件里**所有**「英文→中文」的表都并进来，不只是 translateText 那条通道。
 *
 * ⚠ 第一版只喂了作用域表 + 全局 EXACT/PATTERNS，于是把 12 个 `patch*` 函数在**数据侧**
 *   改掉的那批全判成「未覆盖」—— 天气 26 条就是这么被误报的（它由 patchWeatherLabels
 *   改 `slices[*].config.weather[*].label`，压根不经过 DOM 查表）。
 *   ⇒ 判据边界写死：本项目有**两条**汉化通道，量覆盖率必须两条都算。
 */
import fs from "node:fs"; import path from "node:path"; import {pathToFileURL} from "node:url";
const SRC="C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
const raw=fs.readFileSync(SRC,"utf8");
// 顶层 `const XXX = {` 的表名全抓（含已 export 的要排除，免得重复导出）
const names=[...raw.matchAll(/^const ([A-Z][A-Z_0-9]*) = \{/gm)].map(m=>m[1]);
const exported=new Set([...raw.matchAll(/^export (?:const|function) ([A-Za-z_$][\w$]*)/gm)].map(m=>m[1]));
const need=names.filter(n=>!exported.has(n));
const h=path.join(process.cwd(),"_c2.mjs");
fs.writeFileSync(h,"globalThis.Hooks=globalThis.Hooks??{once(){},on(){}};\n"
 + raw.replace("import * as SELFCHECK from './ember-cn-selfcheck.mjs';","const SELFCHECK={SUBTREE_SELECTORS:[],registerSelfCheck(){},keyLiveness(){}};")
 + "\nexport {"+need.join(",")+", translateNotification};\n","utf8");
const M=await import(pathToFileURL(h).href); fs.unlinkSync(h);
console.log(`表 ${names.length} 张：${names.join(" · ")}`);

const UNION={};
let n=0;
for (const nm of names){
  const t=M[nm];
  if(!t||typeof t!=="object"||Array.isArray(t)) continue;
  for(const [k,v] of Object.entries(t)) if(typeof k==="string"&&typeof v==="string"){UNION[k]??=v;n++;}
}
console.log(`并成一张：${Object.keys(UNION).length} 键（累计 ${n} 条）`);

const covered=(s)=> (s in UNION) || M.translateText(s,UNION)!==s || M.translateNotification(s)!==s;
const cov=JSON.parse(fs.readFileSync("coverage.json","utf8"));
console.log("");
let g=0,gm=0;
for(const [name,d] of Object.entries(cov)){
  if(!name.startsWith("ember")) continue;
  const still=d.missing.filter(x=>!covered(x));
  g+=d.missing.length; gm+=still.length;
  console.log(`${name.padEnd(22)} 原报未覆盖 ${String(d.missing.length).padStart(5)} → 并入数据侧表后 ${String(still.length).padStart(5)}（少报 ${d.missing.length-still.length}）`);
  fs.writeFileSync(`still.${name.replace(/[^\w]/g,"_")}.json`, JSON.stringify(still,null,1),"utf8");
}
console.log(`\nember 合计：${g} → ${gm}`);
