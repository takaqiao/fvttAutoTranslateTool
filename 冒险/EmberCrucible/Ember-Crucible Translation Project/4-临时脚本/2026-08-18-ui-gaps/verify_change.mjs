/**
 * 本轮改动的**行为验收**：
 *   (1) 已有条目**零行为变化** —— 把 HEAD 那份与工作区这份都 import 进来，
 *       在同一个语料上逐条比 `translateText(s)` 与 `translateText(s, extra)`，差异必须**只出现在新增键上**；
 *   (2) 新增的每一条都用**真实输入**验：在指示物制作器的作用域表下必须翻成预期译文，
 *       在**不带**该作用域表时必须一个字都不动（证明它没泄漏成全局）；
 *   (3) 矿车目的地那十条 + 两个表单那五条在 DIALOG_UI 下必须翻得动。
 *
 * 前置自证（两件都断言）：
 *   (A) 切对条数 —— 语料条数、两份表的键数差、都当场印出来并断言与预期相等；
 *   (B) 改对地方 —— 差异集合必须**恰好等于**本轮新增的键集（多一个少一个都 FAIL）。
 */
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PLUGIN = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件";
const REL = "scripts/ember-hardcoded-cn.mjs";
const STUB_IMPORT = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const STUB = "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };";

const EXPORTS = ["EXACT", "PREFIXED", "PATTERNS", "DIALOG_UI", "EMBER_WINDOW_UI",
                 "EMBER_DIALOG_UI", "CHAT_UI", "NOTIFICATIONS", "NOTIFICATION_PATTERNS"];

function build(name, src, extraExports = []) {
  if (!src.includes(STUB_IMPORT)) throw new Error(`${name}: 打桩目标不存在`);
  const names = [...EXPORTS, ...extraExports];
  for (const n of names) {
    if (!new RegExp(`\\nconst ${n} = `).test(src)) throw new Error(`${name}: 找不到 ${n} 的顶层声明`);
  }
  const f = path.join(HERE, `_h_${name}.mjs`);
  fs.writeFileSync(f,
    "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
    + src.replace(STUB_IMPORT, STUB)
    + `\nexport { ${names.join(", ")} };\n`, "utf8");
  return import(pathToFileURL(f).href);
}

const headSrc = execFileSync("git", ["show", `HEAD:${REL}`], { cwd: PLUGIN, encoding: "utf8" });
const workSrc = fs.readFileSync(path.join(PLUGIN, REL), "utf8");

const OLD = await build("old", headSrc);
const NEW = await build("new", workSrc, ["TOKEN_MAKER_UI", "TOKEN_MAKER_WINDOW_UI"]);

let fail = 0;
const chk = (ok, msg) => { console.log((ok ? "  OK   " : "  FAIL ") + msg); if (!ok) fail++; };

/* ---------------- 自证 A：条数 ---------------- */
console.log("== 自证 A：条数 ==");
const addedTM = Object.keys(NEW.TOKEN_MAKER_UI);
const addedDlg = Object.keys(NEW.DIALOG_UI).filter(k => !(k in OLD.DIALOG_UI));
console.log(`  TOKEN_MAKER_UI 键数 ${addedTM.length}`);
console.log(`  DIALOG_UI ${Object.keys(OLD.DIALOG_UI).length} -> ${Object.keys(NEW.DIALOG_UI).length}（新增 ${addedDlg.length}）`);
chk(addedTM.length === 149, "TOKEN_MAKER_UI 恰好 149 键");
chk(addedDlg.length === 15, "DIALOG_UI 恰好新增 15 键");
for (const t of ["EXACT", "PREFIXED", "PATTERNS", "EMBER_WINDOW_UI", "CHAT_UI",
                 "NOTIFICATIONS", "NOTIFICATION_PATTERNS"]) {
  const a = Array.isArray(OLD[t]) ? OLD[t].length : Object.keys(OLD[t]).length;
  const b = Array.isArray(NEW[t]) ? NEW[t].length : Object.keys(NEW[t]).length;
  chk(a === b, `${t} 条数未变（${a}）`);
}

/* ---------------- 自证 B + (1) 零行为变化 ---------------- */
console.log("\n== (1) 已有条目零行为变化 ==");
const corpus = Object.keys(JSON.parse(fs.readFileSync(path.join(HERE, "candidates.json"), "utf8")));
// 语料再并上两份表的全部键，保证每条既有条目都被打到
for (const t of ["EXACT", "DIALOG_UI", "EMBER_WINDOW_UI", "CHAT_UI", "NOTIFICATIONS"])
  corpus.push(...Object.keys(OLD[t]));
const uniq = [...new Set(corpus)];
console.log(`  语料 ${uniq.length} 条（候选集 + 旧表全部键，去重）`);

const added = new Set([...addedTM, ...addedDlg]);
const diffs = { global: [], dialog: [], window: [] };
for (const s of uniq) {
  if (OLD.EXACT !== undefined) {
    const a = OLD.translateText(s), b = NEW.translateText(s);
    if (a !== b) diffs.global.push([s, a, b]);
  }
  {
    const a = OLD.translateText(s, OLD.EMBER_DIALOG_UI), b = NEW.translateText(s, NEW.EMBER_DIALOG_UI);
    if (a !== b) diffs.dialog.push([s, a, b]);
  }
  {
    const a = OLD.translateText(s, OLD.EMBER_WINDOW_UI), b = NEW.translateText(s, NEW.EMBER_WINDOW_UI);
    if (a !== b) diffs.window.push([s, a, b]);
  }
}
chk(diffs.global.length === 0, `全局通道（不带作用域表）差异 ${diffs.global.length} 条`
  + (diffs.global.length ? " " + JSON.stringify(diffs.global.slice(0, 5)) : ""));
chk(diffs.window.length === 0, `EMBER_WINDOW_UI 通道差异 ${diffs.window.length} 条`
  + (diffs.window.length ? " " + JSON.stringify(diffs.window.slice(0, 5)) : ""));
const dlgKeys = new Set(diffs.dialog.map(d => d[0]));
const extra = [...dlgKeys].filter(k => !added.has(k));
const missed = addedDlg.filter(k => !dlgKeys.has(k));
chk(extra.length === 0, `EMBER_DIALOG_UI 通道的差异**只**出现在本轮新增键上（越界 ${extra.length}）`
  + (extra.length ? " " + JSON.stringify(extra.slice(0, 8)) : ""));
chk(missed.length === 0, `新增的 15 条 DIALOG_UI 键在该通道上**都**生效（未生效 ${missed.length}）`
  + (missed.length ? " " + JSON.stringify(missed) : ""));

/* ---------------- (2) 新增键逐条真实输入验 ---------------- */
console.log("\n== (2) TOKEN_MAKER_UI 149 键逐条验 ==");
let hit = 0, leak = 0, wrong = [];
for (const k of addedTM) {
  const got = NEW.translateText(k, NEW.TOKEN_MAKER_WINDOW_UI);
  if (got === NEW.TOKEN_MAKER_UI[k]) hit++; else wrong.push([k, got, NEW.TOKEN_MAKER_UI[k]]);
  // 泄漏检查：不带作用域表 / 只带 EMBER_WINDOW_UI 时都必须原样不动
  if (NEW.translateText(k) !== k || NEW.translateText(k, NEW.EMBER_WINDOW_UI) !== k) leak++;
}
chk(hit === addedTM.length, `作用域表下逐字符翻对 ${hit}/${addedTM.length}`
  + (wrong.length ? " 错：" + JSON.stringify(wrong.slice(0, 5)) : ""));
chk(leak === 0, `不带该作用域表时一个字都不动（泄漏 ${leak} 条）`);

/* ---------------- (3) DIALOG_UI 新增键逐条验 ---------------- */
console.log("\n== (3) DIALOG_UI 新增 15 键逐条验 ==");
for (const k of addedDlg) {
  const got = NEW.translateText(k, NEW.DIALOG_UI);
  const want = NEW.DIALOG_UI[k];
  chk(got === want, `${JSON.stringify(k)} -> ${JSON.stringify(got)}（期望 ${JSON.stringify(want)}）`);
}

/* ---------------- (4) 反向：新增键不许被全局通道吃掉别的模块的串 ---------------- */
console.log("\n== (4) 反例：形状相近但上游产不出的串，一个字都不许动 ==");
const NEG = ["Loading Zone A", "The Loading Zone", "Junction Box", "Remove All",
             "Add Actor Folder", "Human Being", "Head of the Table", "Water Elemental",
             "Standard Action", "Base Attack Bonus"];
for (const s of NEG) {
  const g1 = NEW.translateText(s);
  const g2 = NEW.translateText(s, NEW.TOKEN_MAKER_WINDOW_UI);
  const g3 = NEW.translateText(s, NEW.DIALOG_UI);
  chk(g1 === s && g2 === s && g3 === s, `${JSON.stringify(s)} 三条通道均原样返回`);
}

console.log(fail === 0 ? "\n全部通过。" : `\n${fail} 项未通过。`);
process.exit(fail === 0 ? 0 : 1);
