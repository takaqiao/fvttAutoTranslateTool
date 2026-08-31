/** 拿**真上游值** + **真 cn 译文**跑发布中的 crucibleDescription，看它吐出什么形状。 */
import fs from "node:fs";
import { pathToFileURL } from "node:url";
const F = "C:/Program Files/Foundry Virtual Tabletop/resources/app";
globalThis.foundry = { utils: await import(pathToFileURL(`${F}/common/utils/helpers.mjs`).href) };
const { ClassicLevel } = await import(pathToFileURL(`${F}/node_modules/classic-level/index.js`).href);
const MOD = await import(pathToFileURL("C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/babele-mappings.js").href);

const db = new ClassicLevel("C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs/crucible-adventure", { valueEncoding: "json" });
await db.open();
let items = [];
for await (const [, v] of db.iterator()) if (Array.isArray(v.items)) items = items.concat(v.items);
await db.close();

const cn = JSON.parse(fs.readFileSync("C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/compendium/cn/ember.crucible-adventure.json", "utf8"));
const T = cn.entries["Ember Early Access"].items;

let bad = 0, ok = 0;
const rows = [];
for (const it of items) {
  const t = T[it.name];
  if (!t) continue;
  const src = it.system?.description;
  const out = MOD.crucibleDescription(src, t.description);
  const srcShape = src === undefined ? "undefined" : typeof src === "string" ? "string" : "object{" + Object.keys(src).join(",") + "}";
  const outShape = out === undefined ? "undefined" : typeof out === "string" ? "string" : "object{" + Object.keys(out).join(",") + "}";
  const tShape = t.description === undefined ? "undefined" : typeof t.description === "string" ? "string" : "object{" + Object.keys(t.description).join(",") + "}";
  const shapeBroken = (srcShape.startsWith("object") && outShape === "string");
  if (shapeBroken) { bad++; rows.push([it.name, it.type, srcShape, tShape, outShape]); }
  else ok++;
}
console.log(`比对 ${ok + bad} 条 item：形状保持 ${ok} · **被压成字符串 ${bad}**`);
for (const r of rows.slice(0, 20)) console.log("   ", r.join("  |  "));
