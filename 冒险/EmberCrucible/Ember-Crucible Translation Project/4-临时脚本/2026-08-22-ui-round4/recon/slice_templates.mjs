/**
 * 只把 ember.mjs 的**模板那一段**切出来求值，读出 23 个模板的 `layers[*].parts`。
 *
 * 为什么不整份装：试过，六层 stub 之后卡在 `HEXES["3196.2895"].terrain`（地区数据，
 * 与部件八竿子打不着）。整份装是兔子洞；切片只要**切得能自证**就够。
 *
 * 切片边界（回上游源码逐个确认，不是猜的）：
 *   53648 `const CHARACTER_ATLAS = [...]`      ← 模板要的图集清单
 *   53660 `function cloneLayer(...)`           ← 每个模板的 layers 都由它克隆
 *   54114 `const LAYERS$3 = {...}`             ← 图层原型（parts 就住在这儿）
 *   55230 `const COLORS$k = {...}`
 *   55678 `const TEMPLATE$7 = {...}`           ← 基模板，其余 22 个由它 deepClone
 *   59435 `function makeLegPoseParts(...)`     ← 运行时拼出来的那一族腿部姿态
 *   61260 `var templates = Object.freeze({...})` ← 23 个模板的汇总，切片到此为止
 *
 * 前置自证三件（任一不成立当场炸停）：
 *   A 切片首尾行必须**逐字符**等于上面写死的那两行 —— 上游一改行号就当场报错，不会静默切错段
 *   B 模板数必须是 23，且 id 集合等于写死的那一份
 *   C 手工已知真值：human 模板的 `helm` 图层里必须有 `helmBarbuteMetal`，
 *     `hair` 图层里必须有 `hairAfro` —— 只对条数不对对象，正是本项目栽过的坑
 */
import fs from "node:fs";
import { pathToFileURL } from "node:url";

const EMBER = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs";
const FOUNDRY = "C:/Program Files/Foundry Virtual Tabletop/resources/app";
const U = await import(pathToFileURL(`${FOUNDRY}/common/utils/helpers.mjs`).href);
await import(pathToFileURL(`${FOUNDRY}/common/primitives/_module.mjs`).href);
const ColorMod = await import(pathToFileURL(`${FOUNDRY}/common/utils/color.mjs`).href);

const FIRST = 'const CHARACTER_ATLAS = ["modules/ember/assets/tokens/maker/Character0.json"];';
const LAST_PREFIX = "var templates=/*#__PURE__*/Object.freeze({__proto__:null,altyra:ALTYRA,";

const lines = fs.readFileSync(EMBER, "utf8").split("\n");
const i0 = lines.findIndex((l) => l.trim() === FIRST);
const i1 = lines.findIndex((l) => l.startsWith(LAST_PREFIX));
if (i0 < 0) throw new Error("自证 A 挂：切片起始行找不到（上游改了 CHARACTER_ATLAS 的写法？）");
if (i1 < 0 || i1 <= i0) throw new Error("自证 A 挂：切片结束行找不到或在起始行之前");
console.log(`自证 A OK  切片 = 源码第 ${i0 + 1} ~ ${i1 + 1} 行（共 ${i1 - i0 + 1} 行）`);

// 结束行同一行里还跟着后续代码（打包器把多份文件挤在一行），只取到 `});` 为止
const lastLine = lines[i1];
const cut = lastLine.indexOf("});") + 3;
const slice = lines.slice(i0, i1).join("\n") + "\n" + lastLine.slice(0, cut);

const harness = `
globalThis.foundry = { utils: U };
globalThis.Color = ColorMod.default ?? ColorMod.Color;
${slice}
export const TEMPLATES = { ...templates };
`;
fs.mkdirSync("recon", { recursive: true });
fs.writeFileSync("recon/_slice.mjs",
  `import * as U from ${JSON.stringify(pathToFileURL(`${FOUNDRY}/common/utils/helpers.mjs`).href)};\n` +
  `import * as ColorMod from ${JSON.stringify(pathToFileURL(`${FOUNDRY}/common/utils/color.mjs`).href)};\n` +
  `import ${JSON.stringify(pathToFileURL(`${FOUNDRY}/common/primitives/_module.mjs`).href)};\n` +
  harness, "utf8");

const { TEMPLATES } = await import(pathToFileURL("recon/_slice.mjs").href + "?t=" + i1);

const WANT = ["altyra","ashka","construct","corak","drakon","fej","hulgrun","human","humanAbyssal",
  "humanUndead","jurtak","keth","kiska","kivahr","nirae","partyBanner","signborn","thornling",
  "undeadMonster","vrjnhar","wirrun","zeph"];
const got = Object.keys(TEMPLATES).sort();
if (got.length !== WANT.length || WANT.sort().some((w, k) => w !== got[k])) {
  throw new Error(`自证 B 挂：模板 id 集合对不上。拿到 ${got.length} 个：${got.join(",")}`);
}
console.log(`自证 B OK  模板 ${got.length} 个，id 集合与写死的一份逐个相等`);

const partsOf = (layer) => Array.isArray(layer.parts)
  ? layer.parts.map((p) => (typeof p === "string" ? p : p.id))
  : Object.keys(layer.parts ?? {});

const humanHelm = partsOf(TEMPLATES.human.layers.helm);
const humanHair = partsOf(TEMPLATES.human.layers.hair);
for (const [arr, want, where] of [[humanHelm, "helmBarbuteMetal", "human.helm"],
                                  [humanHair, "hairAfro", "human.hair"]]) {
  if (!arr.includes(want)) throw new Error(`自证 C 挂：${where} 里没有已知真值 ${want}（拿到 ${arr.length} 条）`);
}
console.log(`自证 C OK  human.helm ${humanHelm.length} 条含 helmBarbuteMetal · human.hair ${humanHair.length} 条含 hairAfro`);

// ── 产出：id → 出现在哪些 (模板, 图层) ──────────────────────────────────
const byId = new Map();
const byLayer = {};
for (const [tid, tpl] of Object.entries(TEMPLATES)) {
  for (const [lid, layer] of Object.entries(tpl.layers ?? {})) {
    const ps = partsOf(layer);
    byLayer[lid] = (byLayer[lid] ?? new Set());
    for (const p of ps) {
      if (!p) continue;
      byLayer[lid].add(p);
      if (!byId.has(p)) byId.set(p, new Set());
      byId.get(p).add(`${tid}/${lid}`);
    }
  }
}
const out = {
  sliceLines: [i0 + 1, i1 + 1],
  templates: Object.keys(TEMPLATES).length,
  uniqueParts: byId.size,
  byLayerCount: Object.fromEntries(Object.entries(byLayer).map(([k, v]) => [k, v.size])),
  parts: Object.fromEntries([...byId].map(([k, v]) => [k, [...v]])),
};
fs.writeFileSync("recon/templates_parts.json", JSON.stringify(out, null, 1), "utf8");
console.log(`\ntemplateLayer.parts 唯一 id：${byId.size}`);
console.log(`图层 ${Object.keys(byLayer).length} 个，前 12 大：`);
for (const [k, v] of Object.entries(out.byLayerCount).sort((a, b) => b[1] - a[1]).slice(0, 12)) {
  console.log(`  ${k.padEnd(16)} ${v}`);
}
