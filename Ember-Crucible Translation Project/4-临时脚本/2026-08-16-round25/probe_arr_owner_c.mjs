/**
 * 那 8 条「表里没有」的上游编排名，各自属于哪个音景、那个音景的 type 是什么。
 * 手写落盘，不经改写脚本。
 */
import fs from "node:fs";

const EMBER = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs";
const src = fs.readFileSync(EMBER, "utf8");

function blockAt(s, openIdx) {
  let d = 0, q = null;
  for (let i = openIdx; i < s.length; i++) {
    const c = s[i];
    if (q) { if (c === "\\") { i++; continue; } if (c === q) q = null; continue; }
    if (c === '"' || c === "'" || c === "`") { q = c; continue; }
    if (c === "/" && s[i + 1] === "/") { i = s.indexOf("\n", i); if (i < 0) break; continue; }
    if (c === "/" && s[i + 1] === "*") { i = s.indexOf("*/", i) + 1; continue; }
    if (c === "{") d++;
    else if (c === "}") { d--; if (d === 0) return s.slice(openIdx, i + 1); }
  }
  return null;
}

const reg = src.match(/var soundscapes=\/\*#__PURE__\*\/Object\.freeze\(\{__proto__:null,([^}]*)\}\)/);
const pairs = reg[1].split(",").map(s => s.trim()).filter(Boolean)
  .map(s => { const [k, v] = s.split(":"); return { key: k, varName: v }; });

const want = new Set(["Events", "Seven Sails", "Clear", "Drizzle", "Rain",
                      "Thunderstorm", "Arcane Fog", "Mayis Storm"]);
const byType = {};
for (const { key, varName } of pairs) {
  const re = new RegExp(`(?:^|[;}])var ${varName.replace(/\$/g, "\\$")} = \\{`, "m");
  const m = re.exec(src);
  if (!m) continue;
  const body = blockAt(src, src.indexOf("{", m.index));
  const label = body.match(/\n  label: "([^"]*)"/)?.[1] ?? null;
  const type = body.match(/\n  type: "([^"]*)"/)?.[1] ?? null;
  const id = body.match(/\n  id: "([^"]*)"/)?.[1] ?? null;
  byType[type] = (byType[type] ?? 0) + 1;
  const aIdx = body.search(/\n  arrangements: \{/);
  if (aIdx < 0) continue;
  const aBody = blockAt(body, body.indexOf("{", aIdx));
  const arr = [...aBody.matchAll(/\n      label: "([^"]*)"/g)].map(x => x[1]);
  const hit = arr.filter(a => want.has(a));
  if (hit.length) console.log(`音景 key=${key} id=${id} label=${JSON.stringify(label)} type=${JSON.stringify(type)}  ->  ${JSON.stringify(hit)}`);
}
console.log("音景 type 分布：", byType);
