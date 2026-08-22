/**
 * 上表前的三道核：
 *   ① 新键的**显示名**（splitPartId 之后）彼此不许撞
 *   ② 不许撞发布中的 TOKEN_MAKER_PARTS（778 键）
 *   ③ 同一图层里两个不同部件不许拿到同一个中文（玩家会看到两行一模一样的选项）
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
const SRC = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
const s = fs.readFileSync(SRC, "utf8");
const h = path.join(process.cwd(), "recon", "_col.mjs");
fs.writeFileSync(h, "globalThis.Hooks = globalThis.Hooks ?? { once(){}, on(){} };\n"
  + s.replace("import * as SELFCHECK from './ember-cn-selfcheck.mjs';",
    "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck(){}, keyLiveness(){} };")
  + "\nexport { TOKEN_MAKER_PARTS, TOKEN_MAKER_PART_IDS, splitPartId };\n", "utf8");
const M = await import(pathToFileURL(h).href);
fs.unlinkSync(h);

const rows = fs.readFileSync("review.tsv", "utf8").trim().split("\n").map(l => l.split("\t"));
console.log(`新增 ${rows.length} 条`);

const disp = new Map();
const dupDisp = [];
for (const [fam, seg, cn] of rows) {
  const d = M.splitPartId(seg);
  if (disp.has(d)) dupDisp.push([d, disp.get(d), seg]);
  else disp.set(d, seg);
}
console.log(`① 新键之间显示名相撞：${dupDisp.length}` + (dupDisp.length ? " ⇒ " + JSON.stringify(dupDisp.slice(0, 6)) : ""));

const hitExisting = rows.filter(([, seg]) => M.splitPartId(seg) in M.TOKEN_MAKER_PARTS);
console.log(`② 撞发布中的表：${hitExisting.length}` + (hitExisting.length ? " ⇒ " + hitExisting.slice(0, 6).map(r => r[1]).join(" · ") : ""));

const byFam = {};
for (const [fam, seg, cn] of rows) ((byFam[fam] ??= {})[cn] ??= []).push(seg);
const same = [];
for (const [fam, m] of Object.entries(byFam))
  for (const [cn, segs] of Object.entries(m)) if (segs.length > 1) same.push([fam, cn, segs]);
console.log(`③ 同图层里两个部件拿到同一个中文：${same.length}`);
for (const x of same) console.log("    ", x[0], x[1], "⇐", x[2].join(" · "));

// 跨图层同名（不算错，但列出来看一眼）
const all = {};
for (const [fam, seg, cn] of rows) (all[cn] ??= []).push(`${fam}/${seg}`);
const cross = Object.entries(all).filter(([, v]) => v.length > 1);
console.log(`\n（参考）跨图层同中文：${cross.length}`);
for (const [cn, v] of cross.slice(0, 12)) console.log("    ", cn, "⇐", v.join(" · "));
