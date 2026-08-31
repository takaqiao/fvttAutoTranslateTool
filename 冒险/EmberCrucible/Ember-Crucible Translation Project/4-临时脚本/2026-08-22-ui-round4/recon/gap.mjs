/**
 * 覆盖账：1458 条真显示名里，发布中的表盖住多少、缺哪些。
 * 用**将要发出去的那个文件**里的 `TOKEN_MAKER_PARTS`（含 107 条拼串），不另仿一份查表逻辑。
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const SRC = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
const STUB = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const s = fs.readFileSync(SRC, "utf8");
const h = path.join(process.cwd(), "recon", "_gap.mjs");
fs.writeFileSync(h, "globalThis.Hooks = globalThis.Hooks ?? { once(){}, on(){} };\n"
  + s.replace(STUB, "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck(){}, keyLiveness(){} };")
  + "\nexport { TOKEN_MAKER_PARTS, TOKEN_MAKER_PART_IDS, splitPartId };\n", "utf8");
const M = await import(pathToFileURL(h).href);
fs.unlinkSync(h);

const U = JSON.parse(fs.readFileSync("recon/universe.json", "utf8"));
const segs = Object.keys(U.segs);
if (segs.length !== U.displayNames) throw new Error("universe.json 自相矛盾");
console.log(`全集 ${segs.length} 条显示名（末段去重）`);
console.log(`发布中的表：TOKEN_MAKER_PART_IDS ${Object.keys(M.TOKEN_MAKER_PART_IDS).length} 键 → 展开后 TOKEN_MAKER_PARTS ${Object.keys(M.TOKEN_MAKER_PARTS).length} 键`);

const hit = [], miss = [];
for (const seg of segs) {
  const disp = M.splitPartId(seg);
  (M.TOKEN_MAKER_PARTS[disp] ? hit : miss).push({ seg, disp, layers: U.segs[seg] });
}
console.log(`\n盖住 ${hit.length} / ${segs.length}（${Math.round(hit.length / segs.length * 100)}%）  仍缺 ${miss.length}`);

// 缺口按「主图层」归组：L/R、Lower 归到同一族，便于分工
const fam = (ls) => {
  const a = ls.map(l => l.replace(/(L|R)$/, "").replace(/Lower$/, ""));
  return [...new Set(a)].sort().join("+");
};
const groups = {};
for (const m of miss) (groups[fam(m.layers)] ??= []).push(m);
console.log("\n缺口按族：");
for (const [k, v] of Object.entries(groups).sort((a, b) => b[1].length - a[1].length)) {
  console.log(`  ${k.padEnd(28)} ${String(v.length).padStart(4)}`);
}
fs.writeFileSync("recon/gap.json", JSON.stringify({
  total: segs.length, covered: hit.length, missing: miss.length,
  groups: Object.fromEntries(Object.entries(groups).map(([k, v]) => [k, v.map(x => x.seg).sort()])),
}, null, 1), "utf8");
console.log("\n→ recon/gap.json");
