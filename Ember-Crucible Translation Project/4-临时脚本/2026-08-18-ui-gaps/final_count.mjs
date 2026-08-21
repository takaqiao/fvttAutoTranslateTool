import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
const HERE = path.dirname(fileURLToPath(import.meta.url));
const M = await import(pathToFileURL(path.join(HERE, "_h_new.mjs")).href);
const OLD = await import(pathToFileURL(path.join(HERE, "_h_old.mjs")).href);
const gaps = JSON.parse(fs.readFileSync(path.join(HERE, "gaps.json"), "utf8"));
const tiered = JSON.parse(fs.readFileSync(path.join(HERE, "gaps_tiered.json"), "utf8"));
const added = new Set([...Object.keys(M.TOKEN_MAKER_UI),
                       ...Object.keys(M.DIALOG_UI).filter(k => !(k in OLD.DIALOG_UI))]);
const inGap = [...added].filter(k => k in gaps);
const notInGap = [...added].filter(k => !(k in gaps));
console.log(`缺口全集 N = ${Object.keys(gaps).length}`);
console.log(`本轮新增键 M = ${added.size}（其中落在 N 里的 ${inGap.length}，`
  + `不在 N 里的 ${notInGap.length}：${JSON.stringify(notInGap)}）`);
console.log(`   ↑ 不在 N 里的成因：探针的 covered() 用的是**所有普通表的并集**，`
  + `而作用域表只在特定窗口生效 —— 「并集里有」≠「那个窗口里生效」，这是探针已知的偏乐观口径。`);
console.log(`仍缺 K = ${Object.keys(gaps).length - inGap.length}`);
const byTier = {};
for (const [t, m] of Object.entries(tiered)) {
  byTier[t] = Object.keys(m).filter(k => !added.has(k)).length;
}
console.log("K 的分档：", JSON.stringify(byTier));
