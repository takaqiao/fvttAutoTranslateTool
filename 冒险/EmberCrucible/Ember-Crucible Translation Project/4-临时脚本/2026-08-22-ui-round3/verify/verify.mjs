/**
 * 本轮改动的验收：拿**改前（插件仓 HEAD = v1.1.28）/ 改后**两份 ember-hardcoded-cn.mjs 各造一个
 * harness（只在副本上追加一行 export，函数体一个字节不改），用**同一批真实输入**逐条比对。
 *
 * 用法：node verify.mjs <改前.mjs> <改后.mjs> <out.json>
 *
 * 前置自证（四件，任一不成立当场 exit 2）：
 *   A 切对条数 —— 三张表的键数差必须**恰好**等于本轮登记的新增条数（DIALOG_UI +6 ·
 *                 EMBER_WINDOW_UI +4 · PATTERNS +1 · TOKEN_MAKER_PART_IDS 新表 671）；
 *   B 切对对象 —— **负控制**：改前那份对这批新键/新句一条都不许翻得动
 *                 （若改前就翻了，说明我把已覆盖的当成缺口，整轮计数作废）；
 *   C 判定式不空转 —— 上游根本没有的假串走同样的通道必须不翻；
 *   D 表对象同一 —— 两份 harness 拿到的是**各自文件里那张表**，不是同一个对象
 *                 （否则「零行为变化」是同义反复）。
 *
 * 五段验收：
 *   ① DIALOG_UI 新 6 条：作用域下译成登记值；全局通道**一条都不许命中**；
 *   ② EMBER_WINDOW_UI 新 4 条：Ember 窗口通道下译成登记值；全局通道不命中；
 *   ③ PATTERNS 新 1 条：正例逐字符相符 + 4 条近似反例一个字都不许动；
 *   ④ 部件显示名 778 条：查表值 = 登记值；且**全局通道 / 四张既有作用域表**一条都不许吃到；
 *   ⑤ 已有条目零行为变化：改前**所有**表的每一个键 × 5 条通道逐字节相同。
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const STUB = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const COMMON = ["EMBER_WINDOW_UI", "EXACT", "DIALOG_UI", "EMBER_DIALOG_UI", "TOKEN_MAKER_UI",
                "TOKEN_MAKER_WINDOW_UI", "MOOD_PANEL", "CHAT_UI", "WEATHER", "PATTERNS",
                "NOTIFICATIONS", "SCROLLING_TEXT", "SETTINGS_UI", "translateText"];
const NEWONLY = ["TOKEN_MAKER_PART_IDS", "TOKEN_MAKER_PARTS"];
const die = (m) => { console.error("verify: " + m); process.exit(2); };

async function load(src, tag, names) {
  const s = fs.readFileSync(src, "utf8");
  if (!s.includes(STUB)) die(`${tag}: 找不到要打桩的 import 行`);
  // 已经被上游文件自己 `export` 的名字（translateText）不能再追加一次，否则重复导出。
  const add = [];
  for (const n of names) {
    const own = new RegExp("\\n(?:const|function) " + n + "[ (=]").test(s);
    const exp = new RegExp("\\nexport (?:const|function) " + n + "[ (=]").test(s);
    if (!own && !exp) die(`${tag}: 找不到顶层声明 ${n}`);
    if (!exp) add.push(n);
  }
  const h = path.join(path.dirname(process.argv[4]), `_v3_${tag}.mjs`);
  fs.writeFileSync(h, "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
    + s.replace(STUB, "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
    + `\nexport { ${add.join(", ")} };\n`, "utf8");
  return import(pathToFileURL(h).href);
}

const [, , F_OLD, F_NEW, F_OUT] = process.argv;
const OLD = await load(F_OLD, "old", COMMON);
const NEW = await load(F_NEW, "new", [...COMMON, ...NEWONLY]);

const DIALOG_NEW = ["Location", "Hex", "Teleport to a specific location?",
                    "Teleport to a destination hex?", "This lever has already been activated.", "Transport"];
const WINDOW_NEW = ["Customize World Details",
  "Add basic adventure information and stylized Ember background artwork to the World join screen.",
  "Preserve World State",
  "Retain current game state data that would otherwise be overwritten when importing the adventure."];

/* ── A 切对条数 ── */
const dD = Object.keys(NEW.DIALOG_UI).length - Object.keys(OLD.DIALOG_UI).length;
const dW = Object.keys(NEW.EMBER_WINDOW_UI).length - Object.keys(OLD.EMBER_WINDOW_UI).length;
const dP = NEW.PATTERNS.length - OLD.PATTERNS.length;
const nIds = Object.keys(NEW.TOKEN_MAKER_PART_IDS).length;
const nDisp = Object.keys(NEW.TOKEN_MAKER_PARTS).length;
if (dD !== 6 || dW !== 4 || dP !== 1) die(`A FAIL 键数差 DIALOG_UI ${dD}（期望 6）/ EMBER_WINDOW_UI ${dW}（期望 4）/ PATTERNS ${dP}（期望 1）`);
if (nIds !== 671 || nDisp !== 778) die(`A FAIL 部件表 id ${nIds}（期望 671）/ 显示名 ${nDisp}（期望 778）`);
if (Object.keys(OLD).includes("TOKEN_MAKER_PART_IDS")) die("A FAIL 改前那份竟然已经有部件表");
console.log(`A OK  DIALOG_UI +${dD} · EMBER_WINDOW_UI +${dW} · PATTERNS +${dP} · 部件 id 表 ${nIds} 键 → 显示名 ${nDisp} 条`);

/* ── D 表对象同一性（防同义反复）── */
if (OLD.DIALOG_UI === NEW.DIALOG_UI || OLD.TOKEN_MAKER_UI === NEW.TOKEN_MAKER_UI)
  die("D FAIL 两份 harness 拿到的是同一个表对象 —— 比对没有意义");
console.log("D OK  两份 harness 的表是各自文件里的独立对象");

/* ── B 负控制：改前一条都不许翻得动 ── */
const preD = DIALOG_NEW.filter(k => OLD.translateText(k, OLD.EMBER_DIALOG_UI) !== k);
const preW = WINDOW_NEW.filter(k => OLD.translateText(k, OLD.EMBER_WINDOW_UI) !== k);
const POS = "Transport everyone inside the tower to Golden Flats?";
const preP = OLD.translateText(POS) !== POS;
const preParts = Object.keys(NEW.TOKEN_MAKER_PARTS)
  .filter(k => OLD.translateText(k, OLD.TOKEN_MAKER_WINDOW_UI) !== k);
if (preD.length || preW.length || preP) die(`B FAIL 改前就翻得动：DIALOG ${preD} / WINDOW ${preW} / PATTERN ${preP}`);
console.log(`B OK  改前对 6+4+1 条新增一条都不翻；778 条部件显示名里改前已被顺带覆盖的 ${preParts.length} 条`);
console.log(`      （顺带覆盖的那批：${preParts.slice(0, 12).join(" · ")}${preParts.length > 12 ? " …" : ""}）`);

/* ── C 判定式不空转 ── */
const FAKE = "Zzq Frobnicated Widget Not Upstream";
if (NEW.translateText(FAKE, NEW.TOKEN_MAKER_WINDOW_UI) !== FAKE) die("C FAIL 假串竟被翻了 —— 查表恒真");
if (NEW.TOKEN_MAKER_PARTS[FAKE] !== undefined) die("C FAIL 部件表对假串有值");
console.log("C OK  假串在两条通道下都不翻");

/* ── ① DIALOG_UI ── */
const bad1 = [], bad1g = [];
for (const k of DIALOG_NEW) {
  const got = NEW.translateText(k, NEW.EMBER_DIALOG_UI), want = NEW.DIALOG_UI[k];
  if (got !== want) bad1.push([k, got, want]);
  if (NEW.translateText(k) !== k) bad1g.push([k, NEW.translateText(k)]);
}
console.log(`① DIALOG_UI 新 6 条：作用域下与登记值一致 ${6 - bad1.length}/6；漏到全局通道 ${bad1g.length} 条`);
if (bad1.length || bad1g.length) console.log("   ", bad1, bad1g);

/* ── ② EMBER_WINDOW_UI ── */
const bad2 = [], bad2g = [];
for (const k of WINDOW_NEW) {
  const got = NEW.translateText(k, NEW.EMBER_WINDOW_UI), want = NEW.EMBER_WINDOW_UI[k];
  if (got !== want) bad2.push([k, got, want]);
  if (NEW.translateText(k) !== k) bad2g.push([k, NEW.translateText(k)]);
}
console.log(`② EMBER_WINDOW_UI 新 4 条：窗口通道下一致 ${4 - bad2.length}/4；漏到全局通道 ${bad2g.length} 条`);
if (bad2.length || bad2g.length) console.log("   ", bad2, bad2g);

/* ── ③ PATTERNS：正例 + 近似反例 ── */
const posCases = [
  ["Transport everyone inside the tower to Golden Flats?", "将塔内所有人传送至 Golden Flats？"],
  ["Transport everyone inside the tower to 西格纳拉 Signara?", "将塔内所有人传送至 西格纳拉 Signara？"]
];
const negCases = [
  "Please Transport everyone inside the tower to the docks?",   // 不在串首
  "Transport everyone inside the tower to safety.",             // 句末不是问号
  "Zzq Transport everyone inside the tower to  Zzq",            // 机械近似探针形态
  "Transport everyone outside the tower to Golden Flats?"       // 骨架被改了一个词
];
const bad3 = posCases.filter(([i, o]) => NEW.translateText(i) !== o);
const bad3n = negCases.filter(s => NEW.translateText(s) !== s);
console.log(`③ PATTERNS 新 1 条：正例 ${posCases.length - bad3.length}/${posCases.length} 逐字符相符；`
  + `近似反例被吃掉 ${bad3n.length}/${negCases.length} 条`);
if (bad3.length) console.log("   正例不符：", bad3.map(([i]) => [i, NEW.translateText(i)]));
if (bad3n.length) console.log("   反例被吃：", bad3n.map(s => [s, NEW.translateText(s)]));

/* ── ④ 部件显示名：查表值正确 + 不漏到任何既有通道 ── */
const dispKeys = Object.keys(NEW.TOKEN_MAKER_PARTS);
const bad4 = dispKeys.filter(k => typeof NEW.TOKEN_MAKER_PARTS[k] !== "string" || !NEW.TOKEN_MAKER_PARTS[k]);
// 与既有作用域表同串的那批：本表的值可以与既有表不同（Heavy / Lithe 正是要不同），
// 但**既有通道的输出必须与改前逐字节相同** —— 那一条由 ⑤ 统一判。这里只判「没有新增泄漏」。
const leak = [];
for (const k of dispKeys) {
  const before = OLD.translateText(k), after = NEW.translateText(k);
  if (before !== after) leak.push([k, before, after]);
  for (const t of ["TOKEN_MAKER_WINDOW_UI", "EMBER_DIALOG_UI", "MOOD_PANEL", "WEATHER", "CHAT_UI"]) {
    const b = OLD.translateText(k, OLD[t]), a = NEW.translateText(k, NEW[t]);
    if (b !== a) leak.push([t, k, b, a]);
  }
}
console.log(`④ 部件显示名 ${dispKeys.length} 条：值非法 ${bad4.length} 条；`
  + `在全局 + 5 张既有通道上相对改前的行为变化 ${leak.length} 处（必须为 0：本表不并进任何既有通道）`);
if (leak.length) console.log("   ", leak.slice(0, 8));

/* ── ⑤ 已有条目零行为变化 ── */
const probe = new Set();
for (const n of COMMON) {
  const t = OLD[n];
  if (t && !Array.isArray(t) && typeof t === "object") for (const k of Object.keys(t)) probe.add(k);
}
const bad5 = [];
for (const k of probe) {
  const rows = [
    [OLD.translateText(k), NEW.translateText(k)],
    [OLD.translateText(k, OLD.TOKEN_MAKER_WINDOW_UI), NEW.translateText(k, NEW.TOKEN_MAKER_WINDOW_UI)],
    [OLD.translateText(k, OLD.EMBER_DIALOG_UI), NEW.translateText(k, NEW.EMBER_DIALOG_UI)],
    [OLD.translateText(k, OLD.MOOD_PANEL), NEW.translateText(k, NEW.MOOD_PANEL)],
    [OLD.translateText(k, OLD.WEATHER), NEW.translateText(k, NEW.WEATHER)]
  ];
  // 本轮**有意**新增的 10 条整串键当然会变，单列出来，不算漂移
  const intended = DIALOG_NEW.includes(k) || WINDOW_NEW.includes(k);
  if (!intended && rows.some(([a, b]) => a !== b)) bad5.push([k, rows]);
}
console.log(`⑤ 已有条目零行为变化：${probe.size} 个既有键 × 5 条通道，非有意变化 ${bad5.length} 条`);
if (bad5.length) console.log("   ", bad5.slice(0, 6));

/* ── ⑥ 本轮点名的两处错译：定值断言 ── */
const fix = {
  "部件 Heavy": [NEW.TOKEN_MAKER_PARTS["Heavy"], "粗壮"],
  "部件 Lithe": [NEW.TOKEN_MAKER_PARTS["Lithe"], "柔韧"],
  "体格 Heavy（不许被改）": [NEW.translateText("Heavy", NEW.TOKEN_MAKER_WINDOW_UI), "壮硕"],
  "体格 Lithe（不许被改）": [NEW.translateText("Lithe", NEW.TOKEN_MAKER_WINDOW_UI), "纤瘦"]
};
const bad6 = Object.entries(fix).filter(([, [g, w]]) => g !== w);
console.log(`⑥ Heavy / Lithe 同串异义：4 条定值断言，不符 ${bad6.length} 条`);
if (bad6.length) console.log("   ", bad6);

const pass = !bad1.length && !bad1g.length && !bad2.length && !bad2g.length
  && !bad3.length && !bad3n.length && !bad4.length && !leak.length && !bad5.length && !bad6.length;
fs.writeFileSync(F_OUT, JSON.stringify({
  dD, dW, dP, nIds, nDisp, preCovered: preParts, bad1, bad1g, bad2, bad2g,
  bad3, bad3n, bad4, leak, bad5, bad6, probe: probe.size, pass
}, null, 1), "utf8");
console.log(pass ? "\n验收 PASS（①②③④⑤⑥ 六段全绿）" : "\n验收 FAIL");
process.exit(pass ? 0 : 1);
