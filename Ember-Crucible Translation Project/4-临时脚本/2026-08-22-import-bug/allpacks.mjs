/**
 * 把 crucibleDescription 拿**全部上游包 × 全部 cn 译文**跑一遍，确认现在这版
 * 不会再把对象形态的 description 压成字符串（上一轮只验了 crucible-adventure 一个包）。
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
const F = "C:/Program Files/Foundry Virtual Tabletop/resources/app";
globalThis.foundry = { utils: await import(pathToFileURL(`${F}/common/utils/helpers.mjs`).href) };
const { ClassicLevel } = await import(pathToFileURL(`${F}/node_modules/classic-level/index.js`).href);
const PROJ = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const MOD = await import(pathToFileURL(`${PROJ}/1-Ember汉化插件/babele-mappings.js`).href);

const SOURCES = [
  ["ember", "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs", `${PROJ}/1-Ember汉化插件/compendium/cn`],
  ["crucible", "C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/packs", `${PROJ}/2-Crucible汉化插件/compendium/cn`],
];
const shape = (v) => v === undefined ? "undefined" : v === null ? "null"
  : typeof v === "string" ? "string" : "object{" + Object.keys(v).join(",") + "}";

let tested = 0, broken = 0, changed = 0;
const bad = [];
for (const [tag, packRoot, cnRoot] of SOURCES) {
  if (!fs.existsSync(packRoot)) { console.log(`跳过 ${tag}：找不到 ${packRoot}`); continue; }
  const cnFiles = fs.existsSync(cnRoot) ? fs.readdirSync(cnRoot).filter(f => f.endsWith(".json")) : [];
  const TRANS = {};
  for (const f of cnFiles) {
    try { TRANS[f.replace(/\.json$/, "")] = JSON.parse(fs.readFileSync(path.join(cnRoot, f), "utf8")); }
    catch (e) { console.log(`  ⚠ 读不动 ${f}: ${e.message}`); }
  }
  for (const p of fs.readdirSync(packRoot)) {
    const dir = path.join(packRoot, p);
    if (!fs.statSync(dir).isDirectory()) continue;
    const db = new ClassicLevel(dir, { valueEncoding: "json" });
    try { await db.open(); } catch { continue; }
    // 找对应的 cn 文件（名字里含 pack 名）
    const key = Object.keys(TRANS).find(k => k.endsWith("." + p));
    const T = key ? TRANS[key] : null;
    const byName = {};
    const collect = (o) => {
      if (!o || typeof o !== "object") return;
      if (typeof o.name === "string" && o.system && "description" in (o.system ?? {})) byName[o.name] ??= o;
      for (const v of Object.values(o)) {
        if (Array.isArray(v)) v.forEach(collect); else if (v && typeof v === "object") collect(v);
      }
    };
    for await (const [, v] of db.iterator()) collect(v);
    await db.close();
    // cn 侧：抓所有含 description 的条目（entries.*.items.* 或 顶层 name→{description}）
    const cnByName = {};
    const walkT = (o) => {
      if (!o || typeof o !== "object") return;
      for (const [k, v] of Object.entries(o)) {
        if (v && typeof v === "object") {
          if ("description" in v) cnByName[k] ??= v;
          walkT(v);
        }
      }
    };
    if (T) walkT(T);
    let n = 0;
    for (const [name, item] of Object.entries(byName)) {
      const t = cnByName[name];
      if (!t) continue;
      const src = item.system.description;
      const out = MOD.crucibleDescription(src, t.description);
      tested++; n++;
      if (shape(src).startsWith("object") && shape(out) === "string") {
        broken++; bad.push([tag, p, name, item.type, shape(src), shape(t.description), shape(out)]);
      } else if (shape(src) !== shape(out)) {
        changed++; bad.push([tag, p, name, item.type, shape(src), shape(t.description), shape(out) + " ⚠形状变了"]);
      }
    }
    if (n) console.log(`  ${tag}/${p}  比对 ${n} 条`);
  }
}
console.log(`\n合计比对 ${tested} 条 · **对象被压成字符串 ${broken}** · 其他形状变化 ${changed}`);
for (const r of bad.slice(0, 20)) console.log("   ", r.join(" | "));
