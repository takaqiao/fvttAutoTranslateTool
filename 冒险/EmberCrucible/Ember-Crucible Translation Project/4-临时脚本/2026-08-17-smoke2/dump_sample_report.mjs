/** 只做一件事：拿真实世界形状的夹具跑一次面板真身，把 B / C 两档的报文原样落盘，供人眼复核。 */
import fs from "node:fs"; import path from "node:path"; import { pathToFileURL } from "node:url";
const PROJ = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const DATA = "C:/Users/Taka/AppData/Local/FoundryVTT/Data";
const em = JSON.parse(fs.readFileSync(path.join(DATA, "modules/ember/module.json"), "utf8"));
const cr = JSON.parse(fs.readFileSync(path.join(DATA, "systems/crucible/system.json"), "utf8"));
const loaded = [...em.packs.map(p => ({ src: "ember", ...p })), ...cr.packs.map(p => ({ src: "crucible", ...p }))]
  .filter(p => !p.system || p.system === "crucible");
const cn = new Set([...fs.readdirSync(path.join(PROJ, "1-Ember汉化插件/compendium/cn")),
                    ...fs.readdirSync(path.join(PROJ, "2-Crucible汉化插件/compendium/cn"))]
                    .filter(f => f.endsWith(".json")).map(f => f.slice(0, -5)));
const names = ["深渊 The Abyss", "余烬之心", "Prestidigitation", "空洞之月", "指示物"];
globalThis.Hooks = { once() {}, on() {} }; globalThis.CONFIG = {};
globalThis.game = {
  version: "14.366", world: { title: "冒烟世界" }, system: { id: "crucible", version: "0.10.1" },
  modules: { get: (id) => ({ babele: { active: true, version: "2.9.1" }, ember_cn_unofficial: { active: true, version: "1.1.23" },
    "crucible-cn": { active: true, version: "0.9.13" }, foundry_chn: { active: true, version: "1.0.0" } })[id] },
  packs: loaded.map(p => ({ collection: `${p.src}.${p.name}`, documentName: p.type,
    metadata: { packageName: p.src, name: p.name, label: p.label, type: p.type },
    index: (p.name === "crafting" ? [] : names).map((n, i) => ({ _id: `i${i}`, name: n })) })),
  i18n: { translations: { X: 1 }, localize: (k) => k === "EMBER.CALENDAR.REGION" ? "地区地图" : k },
  babele: { isTranslated: (c) => typeof c === "string" && cn.has(c) },
};
const SC = await import(pathToFileURL(path.join(PROJ, "1-Ember汉化插件/scripts/ember-cn-selfcheck.mjs")).href);
const checks = await SC.runSelfCheck({});
const lines = [];
for (const c of checks) {
  if (!["B Babele 通道", "C i18n 通道"].includes(c.section)) continue;
  lines.push(`- [${c.status}] **${c.name}**（查了 ${c.checked} 项） — ${c.detail}`);
  for (const i of (c.items ?? [])) lines.push(`    - ${i}`);
}
fs.writeFileSync("sample_report_BC.md", lines.join("\n"), "utf8");
process.stdout.write(lines.join("\n") + "\n");
