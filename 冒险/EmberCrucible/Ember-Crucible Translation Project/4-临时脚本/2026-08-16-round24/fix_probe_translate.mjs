/**
 * 改动前后 translateText 的行为必须**逐字节相同**（本轮改键 0 条、删键 0 条，
 * 只动了注释与「交给自检面板什么」）。
 *
 * 三段：
 *   A 全量回归：拿改动前 EXACT / DIALOG_UI / EMBER_WINDOW_UI / NOTIFICATIONS 的**每一个键**
 *     当输入，比改前改后的 translateText 输出；再加 PREFIXED / PATTERNS 的真实形状若干。
 *   B 本轮定性过的每一条键，喂**上游运行时真会产出的那个串**（含模板串拼出来的、
 *     带 <strong> 切碎的碎片、带换行缩进的原始文本节点），看是不是真的翻得动。
 *   C 空白折叠那一路（translateNode 的回退）单独模拟一次 —— 判据说这几条靠折叠命中，
 *     那就把带真实换行 + 缩进的原文喂进去验。
 */
const before = await import(process.argv[2]);
const after = await import(process.argv[3]);

let fail = 0;
const eq = (label, a, b) => {
  if (a !== b) { fail++; console.log(`  ✗ ${label}\n     改前 ${JSON.stringify(a)}\n     改后 ${JSON.stringify(b)}`); }
};

console.log("A 全量回归：改前每张表的每个键 → translateText");
let n = 0;
for (const t of ["__EXACT", "__DIALOG_UI", "__EMBER_WINDOW_UI", "__NOTIFICATIONS",
                 "__ATTUNEMENTS", "__ATTUNEMENT_TAB", "__MOON_NAMES"]) {
  for (const k of Object.keys(before[t])) {
    eq(`${t}[${JSON.stringify(k)}]`, before.translateText(k), after.translateText(k));
    // 作用域表要带 extra 再跑一遍（DIALOG_UI / EMBER_WINDOW_UI 是作用域表，全局那一路查不到）
    eq(`${t}[${JSON.stringify(k)}] (extra)`,
       before.translateText(k, before[t]), after.translateText(k, after[t]));
    n += 2;
  }
}
// PREFIXED / PATTERNS 的真实形状
const SHAPES = [
  "Attunement: Aura", "Attunement: The Abyss", "Language: Common",
  "+2 Boons", "-6 Banes", "[[/language moiré]]",
  "You will gain", "Music Mood: Battle"
];
for (const s of SHAPES) { eq(`形状 ${JSON.stringify(s)}`, before.translateText(s), after.translateText(s)); n++; }
console.log(`   比过 ${n} 条输入，不一致 ${fail} 条`);

console.log("\nB 本轮定性过的键 —— 喂上游运行时真会产出的串");
const RUNTIME = [
  // ① EXACT：模板串三元 ember.mjs:23042/23047 的四种取值（data-tooltip-text 上的实际值）
  ["Event Completed", null],
  ["Event Not Completed", null],
  ["Event Outcome Completed", null],
  ["Event Outcome Not Completed", null],
  // ② EXACT：codex / creation 的 .hbs 裸串（文本节点原样）
  ["Entry Date", null],
  ["Select a quest from the left menu.", null],
  ["Select a discovered creature from the left menu.", null],
  ["Select a character from the left menu.", null],
  ["Select a biome or location from the left menu.", null],
  ["Increase Ability Score", null],
  ["Decrease Ability Score", null],
  ["Spend 9 points across 6 ability scores, allocating up to 3 points per ability.", null],
  // ③ DIALOG_UI：<strong> 切出来的碎片（作用域表，要带 extra）
  ["? Security doors lock, the alarm sounds, and both construct elevators descend to the Construct Assembly.", "dialog"],
  ["? Security doors unlock, the alarm silences, and both construct elevators return to the Tradeway.", "dialog"],
  ["at rank 1 (Lesser Soulmark)?", "dialog"],
  ["to rank 2 (Greater Soulmark)?", "dialog"],
  ["to rank 3 (Deathly Soulmark)?", "dialog"],
  ["and gain", "dialog"],
  ["The Abyss", "dialog"],
  // ④ NOTIFICATIONS 的三条跨行相加（相加在运行时已经是一整行）
  ["When you are ready to begin the Ember game, activate this Scene which will automatically begin the first quest event, \"The Sheltered Campsite\".", "notif"],
  ["The Soulbound Progression macro can only be used by a Gamemaster user.", "notif"],
  ["The Soulbound Progression macro must be used while viewing a specific character sheet for a Hero or Adversary.", "notif"],
  // ⑤ ATTUNEMENT_TAB 的两条 .hbs 串
  ["Make Active", "tab"],
  ["Cosmological Attunements", "tab"],
  // ⑥ EMBER_WINDOW_UI 里被报过的代表串
  ["Event Flowchart", "window"],
  ["Aster Progression", "window"],
  ["Wind Direction", "window"],
  ["Precolorization", "window"]
];
const scope = (tag) => ({
  dialog: after.__DIALOG_UI, notif: after.__NOTIFICATIONS,
  tab: after.__ATTUNEMENT_TAB, window: after.__EMBER_WINDOW_UI
}[tag] ?? null);
for (const [input, tag] of RUNTIME) {
  const out = after.translateText(input, scope(tag));
  const ok = out !== input;
  if (!ok) fail++;
  console.log(`  ${ok ? "翻得动" : "✗ 没翻"}  ${JSON.stringify(input).slice(0, 72)} → ${JSON.stringify(out).slice(0, 40)}`);
}

console.log("\nC 空白折叠那一路（模拟 translateNode 的回退）");
// 上游模板串带真实换行 + 缩进，文本节点拿到的就是这个原样
const RAW = [
  ["\n  Resetting the event step for this event may introduce critical errors into your Ember game state. \n            Are you sure you wish to proceed?\n", "dialog"],
  ["\n  Beginning this event may introduce critical errors into your Ember game state. \n            Are you sure you wish to proceed?\n", "dialog"],
  [" at \n                   rank 1 (Lesser Soulmark)?", "dialog"],
  ["Create a new custom vista composition, starting from your currently viewed one, that will\n      be used as the starting point from which you can make further changes.", "window"]
];
for (const [raw, tag] of RAW) {
  const extra = scope(tag);
  let out = after.translateText(raw, extra);
  if (out === raw) {                       // 复刻 translateNode:2103-2105 的折叠回退
    const flat = raw.trim().replace(/\s+/g, " ");
    const c = after.translateText(flat, extra);
    if (c !== flat) out = raw.replace(raw.trim(), c);
  }
  const ok = out !== raw;
  if (!ok) fail++;
  console.log(`  ${ok ? "折叠后翻得动" : "✗ 折叠也没翻"}  ${JSON.stringify(raw).slice(0, 64)}… → ${JSON.stringify(out.trim()).slice(0, 40)}`);
}

console.log(`\n总不一致 / 没翻动 ${fail} 条`);
process.exit(fail ? 1 : 0);
