// S2 回归：旧表里的每一条 EXACT 键在新文件里必须仍然能翻出**同样**的结果（除有意订正的）。
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const dir = path.dirname(fileURLToPath(import.meta.url));
const ORIG = "C:\\Users\\Taka\\Desktop\\fvtt\\Ember-Crucible Translation Project\\1-Ember汉化插件\\scripts\\ember-hardcoded-cn.mjs";

async function load(file, tag) {
  let src = fs.readFileSync(file, "utf8").replace(/Hooks\.once\("ready"[\s\S]*$/, "");
  src += `\nexport const __E = EXACT; export const __P = PREFIXED;\n`;
  const tmp = path.join(dir, `_s2_reg_${tag}.mjs`);
  fs.writeFileSync(tmp, src, "utf8");
  const m = await import("file://" + tmp.replace(/\\/g, "/"));
  fs.unlinkSync(tmp);
  return m;
}

const oldM = await load(ORIG, "old");
const newM = await load(path.join(dir, "s2_patched.mjs"), "new");

let fail = 0;
const 有意订正 = { "Install Junction Wheel": "安装枢纽轮盘" };

for (const [k, v] of Object.entries(oldM.__E)) {
  const got = newM.translateText(k);
  const want = 有意订正[k] ?? v;
  if (got !== want) { console.log("REGRESS", JSON.stringify(k), "old:", v, "new:", got); fail++; }
}

// 新 EXACT 不允许出现重复键覆盖（spread 之后被后面的字面量顶掉）
const raw = fs.readFileSync(path.join(dir, "s2_patched.mjs"), "utf8");
const body = raw.slice(raw.indexOf("const EXACT = {"), raw.indexOf("/** 掷骰结果档位"));
const keys = [...body.matchAll(/^\s*"((?:[^"\\]|\\.)*)":/gm)].map(m => m[1]);
const dup = keys.filter((k, i) => keys.indexOf(k) !== i);
if (dup.length) { console.log("EXACT 里有重复字面量键:", dup); fail += dup.length; }

// DIALOG_TITLES 的键不得与 EXACT 自己的字面量键相撞
const dt = raw.slice(raw.indexOf("const DIALOG_TITLES = {"), raw.indexOf("/** 动态拼出来的 Ember 对话框标题"));
const dtKeys = [...dt.matchAll(/^\s*"((?:[^"\\]|\\.)*)":/gm)].map(m => m[1]);
const clash = dtKeys.filter(k => keys.includes(k));
if (clash.length) { console.log("DIALOG_TITLES 与 EXACT 字面量键相撞:", clash); fail += clash.length; }

console.log(`EXACT 旧键 ${Object.keys(oldM.__E).length} 条，新 EXACT 字面量 ${keys.length} 条，DIALOG_TITLES ${dtKeys.length} 条`);
console.log(fail ? `FAILURES: ${fail}` : "REGRESS ALL PASS");
process.exit(fail ? 1 : 0);
