/**
 * 部件**显示名全集**的权威口径：`templateLayer.parts` 的 id 末段。
 *
 * 上屏路径（ember.mjs:64457 `getLayerChoicesV2`，labels:true 时）：
 *     partId.split("/").at(-1).replace(/(?<!^)([A-Z1-9])/g, " $1")
 * ⇒ 分母是**进得了 templateLayer.parts 的 id**，不是图集里的贴图。
 *   上一轮拿图集当分母，把 54 条 `…Lower`（texture 覆盖帧、永不进 parts）算了进去。
 *
 * 前置自证（跨两个**互相独立**的上游来源，不是自己验自己）：
 *   C1 形状：每个 id 必须是 `<ns>/<layer>/<Part>` 三段
 *   C2 跨源：每个 id 必须在随包图集（4 份 .json / 5649 帧，与 ember.mjs 是两份文件）里
 *      查得到同名 frame —— 对不上的**逐条列出来**，不许静默丢
 *   C3 反向：图集里有、而 parts 里没有的末段，必须**正好**是那批 texture 覆盖帧
 *      （下面按 `…Lower` 之外的特征也一并统计，不预设结论）
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const ATLAS_DIR = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/assets/tokens/maker";
const ATLAS_TRUTH = { "Character0.json": 1631, "Character1.json": 3013, "Monster0.json": 783, "Party0.json": 222 };

const { TEMPLATES } = await import(pathToFileURL("recon/_slice.mjs").href);
const partsOf = (l) => Array.isArray(l.parts)
  ? l.parts.map((p) => (typeof p === "string" ? p : p.id))
  : Object.keys(l.parts ?? {});

const idInfo = new Map();              // 完整 id → {layers:Set, seg}
for (const [tid, tpl] of Object.entries(TEMPLATES)) {
  for (const [lid, layer] of Object.entries(tpl.layers ?? {})) {
    for (const p of partsOf(layer)) {
      if (!p) continue;
      if (!idInfo.has(p)) idInfo.set(p, { layers: new Set(), seg: p.split("/").at(-1) });
      idInfo.get(p).layers.add(lid);
    }
  }
}
const ids = [...idInfo.keys()];
console.log(`templateLayer.parts 完整 id：${ids.length}`);

const badShape = ids.filter((i) => i.split("/").length !== 3);
if (badShape.length) throw new Error(`自证 C1 挂：${badShape.length} 个 id 不是三段：${badShape.slice(0, 6).join(" · ")}`);
console.log(`自证 C1 OK  ${ids.length} 个 id 全是 <ns>/<layer>/<Part> 三段`);

const frames = new Set();
for (const [f, n] of Object.entries(ATLAS_TRUTH)) {
  const d = JSON.parse(fs.readFileSync(path.join(ATLAS_DIR, f), "utf8"));
  const ks = Object.keys(d.frames);
  if (ks.length !== n) throw new Error(`自证 C2 挂：${f} 读到 ${ks.length} 帧，已知真值 ${n}`);
  for (const k of ks) frames.add(k);
}
console.log(`自证 C2 前半 OK  图集 ${frames.size} 帧，逐份对上已知真值`);
const notInAtlas = ids.filter((i) => !frames.has(i));
console.log(`自证 C2 后半  parts 里有、图集里没有的 id：${notInAtlas.length}`
  + (notInAtlas.length ? "  ⇒ " + notInAtlas.slice(0, 12).join(" · ") : ""));

// 显示名全集 = 末段去重
const segs = new Map();                // 末段 → {ids:[], layers:Set}
for (const [id, info] of idInfo) {
  if (!segs.has(info.seg)) segs.set(info.seg, { ids: [], layers: new Set() });
  const e = segs.get(info.seg);
  e.ids.push(id);
  for (const l of info.layers) e.layers.add(l);
}
console.log(`\n⇒ 显示名全集（末段去重）：${segs.size}`);

// C3 反向：图集独有的末段
const atlasSegs = new Set([...frames].map((f) => f.split("/").at(-1)));
const atlasOnly = [...atlasSegs].filter((s) => !segs.has(s));
const lowerish = atlasOnly.filter((s) => /Lower$/.test(s));
console.log(`自证 C3  图集有、parts 没有的末段：${atlasOnly.length}（其中 …Lower ${lowerish.length}）`);

// 按图层归组，给下一步分工用
const byLayer = {};
for (const [seg, e] of segs) {
  for (const l of e.layers) (byLayer[l] ??= []).push(seg);
}
fs.writeFileSync("recon/universe.json", JSON.stringify({
  fullIds: ids.length,
  displayNames: segs.size,
  notInAtlas,
  atlasOnlySegs: atlasOnly,
  byLayer: Object.fromEntries(Object.entries(byLayer).map(([k, v]) => [k, v.sort()])),
  segs: Object.fromEntries([...segs].map(([k, v]) => [k, [...v.layers].sort()])),
}, null, 1), "utf8");
console.log("\n各图层的显示名数（前 20）：");
for (const [k, v] of Object.entries(byLayer).sort((a, b) => b[1].length - a[1].length).slice(0, 20)) {
  console.log(`  ${k.padEnd(14)} ${v.length}`);
}
