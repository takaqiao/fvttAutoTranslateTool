/**
 * 第二十八轮 · NOTIFICATION_PATTERNS「定义域过松 + 整句重构」分诊
 *
 * 这张表挂在全局 `ui.notifications.notify` 上，**别的模块发的每一条提示都要过一遍**。
 * 一条正则一旦命中，整句被重写成中文 —— 所以判据不是「译得好不好」，而是
 * **别的模块有没有可能产出一条落进这个定义域的句子**。
 *
 * 分诊三个可复算的量（不掺我的主观排序）：
 *   · open_groups   —— 正则里 `(.+)` 这类**开放**捕获组的个数（`\d+` / 枚举不算）；
 *   · anchor_hits   —— 正则的**字面骨架**里出现的 Ember 专有词（下表 VOCAB，逐个可查出处）；
 *   · longest_lit   —— 最长的一段连续字面量（越短越容易撞）。
 * 判级：anchor_hits = 0 且 open_groups ≥ 1 ⇒ HIGH（无厂商锚点 + 开放定义域）；
 *       anchor_hits = 0 且 open_groups = 0 ⇒ MED（无锚点但定义域封闭）；
 *       anchor_hits ≥ 1 ⇒ LOW（句子里带 Ember 专有词，别的模块撞不上）。
 *
 * ⚠ 正则**从真身现读**（同 harness 打桩），不在这里抄一份。
 * 用法：node probe_overreach_triage.mjs <输出 json>
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const PROJ = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const SRC = path.join(PROJ, "1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs");
const OUT = process.argv[2] ?? path.join(path.dirname(new URL(import.meta.url).pathname), "overreach_triage.json");

/**
 * Ember / Crucible 专有词。每个词都能在上游指出出处，别的模块的提示里几乎不会出现。
 * 「Ember |」前缀本身是最强的锚点（上游自己加的模块前缀）。
 */
const VOCAB = [
  "Ember", "Vista", "composition", "compositions", "attunement", "Attunement",
  "Soulbound", "Soulmark", "vantage point", "interactable", "Region Map",
  "Token Maker", "gameplay event", "Corpuleth", "caravan", "Bronze Rask",
];

const IMPORT_LINE = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const srcText = fs.readFileSync(SRC, "utf8");
globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ec-r28-triage-"));
let rows;
try {
  const harness = path.join(tmp, "_h.mjs");
  fs.writeFileSync(harness,
    "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
    + srcText.replace(IMPORT_LINE,
        "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
    + "\nexport { NOTIFICATION_PATTERNS as __NP };\n", "utf8");
  const mod = await import(pathToFileURL(harness).href);

  rows = mod.__NP.map(({ re }, i) => {
    const s = re.source;
    // 骨架 = 去掉所有捕获组之后剩下的字面量段
    const parts = s.replace(/^\^/, "").replace(/\$$/, "").split(/\([^)]*\)/g);
    const lits = parts.map((x) => x.replace(/\\(.)/g, "$1")).filter((x) => x.trim() !== "");
    const skeleton = lits.join(" ␣ ");
    const openGroups = (s.match(/\(\.\+\)|\(\.\*\)|\(\[\^"\]\+\)/g) || []).length;
    const anchors = VOCAB.filter((v) => skeleton.includes(v));
    const longest = lits.reduce((m, x) => Math.max(m, x.trim().length), 0);
    const grade = anchors.length ? "LOW" : (openGroups ? "HIGH" : "MED");
    return { i, source: s, skeleton, open_groups: openGroups, anchor_hits: anchors, longest_lit: longest, grade };
  });
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}

const by = (g) => rows.filter((r) => r.grade === g);
for (const g of ["HIGH", "MED", "LOW"]) {
  console.log(`\n=== ${g} (${by(g).length}) ===`);
  for (const r of by(g)) {
    console.log(`  [${String(r.i).padStart(2)}] open=${r.open_groups} lit=${r.longest_lit} anchors=${r.anchor_hits.join(",") || "—"}`);
    console.log(`       /${r.source}/`);
  }
}
console.log(`\n合计 ${rows.length} 条 · HIGH ${by("HIGH").length} · MED ${by("MED").length} · LOW ${by("LOW").length}`);
fs.writeFileSync(OUT, JSON.stringify({ src: SRC, vocab: VOCAB, total: rows.length, rows }, null, 1), "utf8");
console.log(`→ ${OUT}`);
