import fs from "node:fs"; import path from "node:path"; import {pathToFileURL} from "node:url";
const P="C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const C=await import(pathToFileURL(`${P}/2-Crucible汉化插件/crucible-hardcoded-cn.mjs`).href);
const src=fs.readFileSync(`${P}/2-Crucible汉化插件/crucible-hardcoded-cn.mjs`,"utf8");
const grab=(n)=>Object.fromEntries([...src.matchAll(new RegExp(`const ${n} = \{([\s\S]*?)\n\};`))]
  .flatMap(m=>[...m[1].matchAll(/"([^"]+)":\s*"([^"]+)"/g)].map(x=>[x[1],x[2]])));
console.log("### crucible-cn 0.9.18（18 条）");
for(const [t,n] of [["掷骰卡·加值/减值来源","BOON_BANE_LABELS"],["动作卡·上下文标签 tooltip","CONTEXT_TOOLTIPS"],["各类卡·输入框 placeholder","PLACEHOLDERS"]]){
  const o=grab(n); console.log(`\n[${t}] ${Object.keys(o).length} 条`);
  for(const [k,v] of Object.entries(o)) console.log(`   ${k.padEnd(22)} → ${v}`);
}
const ap=[...src.matchAll(/\{ en: "([^"]+)", cn: "([^"]+)" \}/g)].map(m=>[m[1],m[2]]);
console.log(`\n[创建页·加减按钮 aria-label] ${ap.length} 条（只改前缀，物品名归 Babele）`);
for(const [k,v] of ap) console.log(`   ${(k+"{物品名}").padEnd(22)} → ${v}{物品名}`);

// ember 本次新增
const esrc=fs.readFileSync(`${P}/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs`,"utf8");
const eb=Object.fromEntries([...esrc.matchAll(/const BOON_BANE_LABELS = \{([\s\S]*?)\n\};/)]
  .flatMap(m=>[...m[1].matchAll(/"([^"]+)":\s*"([^"]+)"/g)].map(x=>[x[1],x[2]])));
console.log(`\n### ember v1.1.31\n\n[掷骰卡·加值/减值来源（ember 加的）] ${Object.keys(eb).length} 条`);
for(const [k,v] of Object.entries(eb)) console.log(`   ${k.padEnd(22)} → ${v}`);
