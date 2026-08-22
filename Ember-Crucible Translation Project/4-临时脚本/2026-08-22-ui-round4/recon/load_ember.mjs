/**
 * 直接把 ember.mjs 装进 Node，读出 `EmberDynamicToken.TEMPLATES` —— 部件全集的**真身来源**。
 * 图集只是贴图，`templateLayer.parts` 才是「这个部件会不会出现在选择器里」的判据。
 * 上一轮用图集当分母，于是把 54 条 `…Lower`（texture 覆盖帧、永不上屏）算了进去。
 *
 * 装不进去也没关系：本文件只负责报「装到哪一步炸的」，好判断这条路走不走得通。
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const EMBER = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs";
const FOUNDRY = "C:/Program Files/Foundry Virtual Tabletop/resources/app";
const U = await import(pathToFileURL(`${FOUNDRY}/common/utils/helpers.mjs`).href);
// Foundry 往 Math / Array / String / Number 上挂的扩展（Math.toRadians 之类）。
// 这些是**真身**，不是仿写 —— ember 顶层就在用它们。
await import(pathToFileURL(`${FOUNDRY}/common/primitives/_module.mjs`).href);

const noop = () => {};
// 计算属性名（`[CONST.REGION_EVENTS.TOKEN_MOVE_IN]: …`）会把这个哨兵当键用，
// 于是必须能转成原始值 —— 否则 static initializer 直接抛「Cannot convert object to primitive」。
// 每次访问返回**同一个** Any，但字符串化时给一个稳定的独立名字，免得所有计算键塌成一个。
let _anySeq = 0;
const _anyName = new WeakMap();
const mkAny = () => new Proxy(function () {}, {
  get(t, k) {
    if (k === "then") return undefined;
    if (k === Symbol.toPrimitive) return () => nameOf(t);
    if (k === "toString") return () => nameOf(t);
    if (k === Symbol.toStringTag) return nameOf(t);
    if (!_kids.has(t)) _kids.set(t, new Map());
    const m = _kids.get(t);
    if (!m.has(k)) m.set(k, mkAny());
    return m.get(k);
  },
  apply: () => mkAny(),
  construct: () => mkAny(),
});
const _kids = new WeakMap();
function nameOf(t) {
  if (!_anyName.has(t)) _anyName.set(t, "__ANY_" + (_anySeq += 1) + "__");
  return _anyName.get(t);
}
const Any = mkAny();
globalThis.Hooks = { once: noop, on: noop, callAll: noop, call: noop };
globalThis.window = globalThis;
// 自定义元素：ember 在模块顶层就 define 了一批 <ember-…> 元素。Node 里没有 CE 注册表，
// 而 HTMLElement 又必须是个**真类**（那些元素 extends 它），Proxy 冒充不了。
globalThis.HTMLElement = class {};
globalThis.HTMLDivElement = class extends globalThis.HTMLElement {};
globalThis.HTMLInputElement = class extends globalThis.HTMLElement {};
globalThis.customElements = { define: noop, get: () => undefined, whenDefined: async () => {} };
globalThis.document = new Proxy({}, { get: () => mkAny() });
globalThis.location = { href: "http://localhost:30000/", origin: "http://localhost:30000" };
globalThis.CONFIG = new Proxy({}, { get: () => mkAny() });
globalThis.CONST = new Proxy({}, { get: () => mkAny() });
globalThis.game = new Proxy({}, { get: () => mkAny() });
globalThis.ui = new Proxy({}, { get: () => mkAny() });
globalThis.canvas = new Proxy({}, { get: () => mkAny() });
globalThis.Color = (await import(pathToFileURL(`${FOUNDRY}/common/utils/color.mjs`).href)).default
  ?? (await import(pathToFileURL(`${FOUNDRY}/common/utils/color.mjs`).href)).Color;
globalThis.PIXI = new Proxy({}, { get: () => mkAny() });
globalThis.foundry = {
  utils: U,
  applications: new Proxy({}, { get: () => new Proxy({}, { get: () => mkAny() }) }),
  documents: new Proxy({}, { get: () => mkAny() }),
  canvas: new Proxy({}, { get: () => mkAny() }),
  data: new Proxy({}, { get: () => mkAny() }),
  abstract: new Proxy({}, { get: () => mkAny() }),
  helpers: new Proxy({}, { get: () => mkAny() }),
  audio: new Proxy({}, { get: () => mkAny() }),
  grid: new Proxy({}, { get: () => mkAny() }),
};
for (const n of ["Actor", "Item", "Token", "TokenDocument", "Scene", "JournalEntry", "Macro",
  "RegionBehavior", "ChatMessage", "Combat", "Combatant", "User", "Folder", "Application",
  "FormApplication", "Dialog", "DocumentSheet", "ActorSheet", "ItemSheet", "Roll", "Ray",
  "PIXI", "SearchFilter", "ContextMenu", "TextEditor", "Handlebars", "loadTemplates",
  "renderTemplate", "fromUuid", "fromUuidSync", "getDocumentClass", "Hooks"]) {
  if (!(n in globalThis)) globalThis[n] = Any;
}

let M = null, err = null;
try { M = await import(pathToFileURL(EMBER).href); }
catch (e) { err = e; }

if (err) {
  console.log("装不进去：" + (err?.message ?? err));
  console.log((err?.stack ?? "").split("\n").slice(0, 6).join("\n"));
  process.exit(3);
}
console.log("装进去了。导出名 " + Object.keys(M).length + " 个");
const cls = Object.values(M).find(v => v && v.TEMPLATES && typeof v.TEMPLATES === "object");
if (!cls) { console.log("⚠ 导出里找不到带 TEMPLATES 的类：" + Object.keys(M).slice(0, 40).join(", ")); process.exit(4); }
const T = cls.TEMPLATES;
console.log("模板 " + Object.keys(T).length + " 个：" + Object.keys(T).join(", "));
const out = {};
for (const [tid, tpl] of Object.entries(T)) {
  out[tid] = {};
  for (const [lid, layer] of Object.entries(tpl.layers ?? {})) {
    const parts = Array.isArray(layer.parts)
      ? Object.fromEntries(layer.parts.map(p => [p.id ?? p, p]))
      : (layer.parts ?? {});
    out[tid][lid] = { label: layer.set?.label ?? layer.label ?? null, parts: Object.keys(parts) };
  }
}
fs.writeFileSync("recon/templates.json", JSON.stringify(out, null, 1), "utf8");
const ids = new Set();
for (const t of Object.values(out)) for (const l of Object.values(t)) for (const p of l.parts) ids.add(p);
console.log("templateLayer.parts 里的唯一部件 id：" + ids.size);
