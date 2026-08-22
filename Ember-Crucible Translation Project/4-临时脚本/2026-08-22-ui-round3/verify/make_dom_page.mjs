/**
 * 造一张**真浏览器**里跑的验收页：真 DOM、真 CSS 选择器引擎、真 `translateTokenMakerParts()`。
 *
 * ⚠ 被测代码是从被判文件里**按锚点整段切出来的原文**，一个字节不改（切不到 / 切到多处当场失败）。
 *   在 node 里跑不了这一段：它要 `querySelectorAll`，而 node 没有 DOM ——
 *   自己写一个假的 `querySelectorAll` 等于把「选择器选得对不对」这件事测成同义反复。
 *
 * 页面结构照 `templates/applications/token-maker/layers.hbs` 逐行搭：
 *   `.choice.build` / `.choice.stance` 各一行（体格 / 姿态，**不该**被部件表碰），
 *   `.choice.layer` 若干行（部件名 + `N of M` 计数行）。
 * 另外在窗口内、窗口外各摆一批**别的模块 / 核心**的近似串，一条都不许被吃。
 */
import fs from "node:fs";
import path from "node:path";

const [, , TABLES, OUT] = process.argv;
const die = (m) => { console.error("make_dom_page: " + m); process.exit(2); };
// 被判文件是 CRLF；切片前统一成 LF，免得锚点里的换行对不上（只影响切片，不改语义）
const src = fs.readFileSync(TABLES, "utf8").replace(/\r\n/g, "\n");

function slice(startAnchor, endAnchor) {
  const i = src.indexOf(startAnchor);
  if (i < 0) die(`切不到锚点：${startAnchor}`);
  if (src.indexOf(startAnchor, i + 1) >= 0) die(`锚点命中多处：${startAnchor}`);
  const j = src.indexOf(endAnchor, i);
  if (j < 0) die(`切不到结束锚点：${endAnchor}`);
  return src.slice(i, j + endAnchor.length);
}

const parts = [
  slice("const TOKEN_MAKER_PART_IDS = {", "\n};"),
  slice("const LEG_POSE_BASES = [", '"Straw", "Swirled", "Wood", "Wood1", "Woody"];'),
  slice('const LEG_POSE_SUFFIXES = [', '"Sitting"];'),
  slice("const MARBLED_HAND_BASES = [", '"Weapon"];'),
  slice("const splitPartId = (id)", '" $1");'),
  slice("const joinCn = (a, b)", "`${a}${b}`);"),
  slice("const TOKEN_MAKER_PARTS = (() => {", "\n})();"),
  slice("function translateTokenMakerParts(root) {", "\n  return n;\n}")
].join("\n\n");
if (!/TOKEN_MAKER_PARTS\[raw\]/.test(parts)) die("切出来的 translateTokenMakerParts 里找不到查表那一行 —— 切歪了");

/* 别的模块 / 核心真会产出的 20 条近似串：一条都不许被吃。 */
const FOREIGN = [
  "Heavy", "Lithe", "Light", "Medium", "Large", "Small", "Simple", "Open", "Down", "Up",
  "Casual Friday", "Hold Item Please", "3 of 12 items", "12 of", "of 12",
  "Bare Necessities", "Claw Attack", "Sword of Truth", "Metal 1 Ingot", "Beard Wizard Hat"
];

const layerRows = [
  ["Head", "Heavy", "3 of 12"],          // ← 本轮修的错译：部件 Heavy 该是「粗壮」不是体格「壮硕」
  ["Torso", "Lithe", "5 of 40"],         // ← 同上：部件 Lithe 该是「柔韧」不是体格「纤瘦」
  ["Face", "Beard Wizard", "7 of 62"],
  ["Leg Left", "Smooth 1 Backward", "12 of 128"],   // 拼串族
  ["Hand Left", "Marbled Casual", "2 of 79"],       // 大理石手部拼串族
  ["Hair", "None", "1 of 97"],           // 兜底值：部件表不该动它（归 TOKEN_MAKER_UI 的「无」）
  ["Helm", "Helmet Metal Horned", "4 of 86"]        // 装备族：本轮**仍缺**，必须原样留英
];

const html = `<!doctype html><meta charset="utf-8"><title>token-maker part scope</title>
<div id="outside">${FOREIGN.map(t => `<span class="part">${t}</span><span class="count">${t}</span>`).join("")}</div>
<div id="ember-token-maker" class="application ember">
  <div class="token-maker-layers flexcol scrollable">
    <h2 class="ember-header column-header">Layers</h2>
    <div class="choice build flexrow">
      <div class="title flexcol"><span class="layer">Build</span><span class="part">Heavy</span></div>
    </div>
    <div class="choice stance flexrow">
      <div class="title flexcol"><span class="layer">Stance</span><span class="part">Sitting</span></div>
    </div>
    ${layerRows.map(([l, p, c], i) => `<div class="choice layer flexrow" data-layer-id="l${i}">
      <div class="title flexcol"><span class="layer">${l}</span><span class="part">${p}</span><span class="count">${c}</span></div>
    </div>`).join("\n    ")}
  </div>
  <div id="foreign-inside">${FOREIGN.map(t => `<span>${t}</span>`).join("")}</div>
</div>
<pre id="result">running…</pre>
<script type="module">
const warn = (...a) => console.warn(a);
${parts}

const root = document.getElementById("ember-token-maker");
const before = [...root.querySelectorAll(".token-maker-layers .choice .part, .token-maker-layers .choice .count")]
  .map(e => e.textContent);
const n = translateTokenMakerParts(root);
const read = (sel) => [...document.querySelectorAll(sel)].map(e => e.textContent);
const out = {
  changed: n,
  build: read(".choice.build .part"),
  stance: read(".choice.stance .part"),
  layerParts: read(".choice.layer .part"),
  layerCounts: read(".choice.layer .count"),
  outsideParts: read("#outside .part"),
  outsideCounts: read("#outside .count"),
  insideForeign: read("#foreign-inside span"),
  tableSize: Object.keys(TOKEN_MAKER_PARTS).length,
  idSize: Object.keys(TOKEN_MAKER_PART_IDS).length,
  heavyPart: TOKEN_MAKER_PARTS["Heavy"], lithePart: TOKEN_MAKER_PARTS["Lithe"],
  before
};
document.getElementById("result").textContent = JSON.stringify(out, null, 1);
globalThis.__RESULT__ = out;
</script>`;
fs.writeFileSync(OUT, html, "utf8");
console.log(`已写 ${OUT}（切出 ${parts.length} 字节被测原文，外部近似串 ${FOREIGN.length} 条，图层行 ${layerRows.length} 行）`);
