/**
 * 记账口径落地验证：把面板「合计」行新增的 ledger 统计与报文逐行打出来。
 * ⚠ 前置自证：被核键数 / miss 必须与改前一模一样（719 / 1273 / 4 / 7），
 *   且 ledger 必须自己对平（已核 + 没核 = 登记）—— 对不上当场硬失败。
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = "C:\\Users\\Taka\\Desktop\\fvtt\\Ember-Crucible Translation Project";
const PANEL = path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-cn-selfcheck.mjs");
const TABLES_SRC = path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs");
const STUB = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const DATA_ROOT = "C:\\Users\\Taka\\AppData\\Local\\FoundryVTT\\Data";

const die = (m) => { process.stderr.write("HARD-FAIL: " + m + "\n"); process.exit(2); };
const must = (c, m) => { if (!c) die(m); };

globalThis.fetch = async (u) => {
  const p = path.join(DATA_ROOT, String(u));
  if (!fs.existsSync(p) || !fs.statSync(p).isFile()) return { ok: false, text: async () => "" };
  return { ok: true, text: async () => fs.readFileSync(p, "utf8") };
};
globalThis.Hooks = { once() {}, on() {} };
globalThis.game = { system: { id: "crucible" }, packs: [] };

let TABLES;
{
  const d = fs.mkdtempSync(path.join(os.tmpdir(), "ec-led-"));
  try {
    const h = path.join(d, "_hc.mjs");
    const src = fs.readFileSync(TABLES_SRC, "utf8");
    must(src.includes(STUB), "找不到 SELFCHECK import 行");
    fs.writeFileSync(h, "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
      + src.replace(STUB, "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
      + "\nexport { SELFCHECK_TABLES as __SELFCHECK_TABLES };\n", "utf8");
    TABLES = (await import(pathToFileURL(h).href)).__SELFCHECK_TABLES;
  } finally { fs.rmSync(d, { recursive: true, force: true }); }
}

const SC = await import(pathToFileURL(PANEL).href);
const checks = await SC.keyLiveness(TABLES);
const total = checks.find((c) => c.name === "合计");
must(total?.stats, "没有合计/stats");
const s = total.stats;

must(s.checkedDistinct === 719, `被核键数 ${s.checkedDistinct} ≠ 719（不许下降/漂移）`);
must(s.rawChecked === 1273, `rawChecked ${s.rawChecked} ≠ 1273`);
must(s.missDistinct === 4 && s.rawMiss === 7, `miss ${s.missDistinct}/${s.rawMiss} ≠ 4/7`);
must(s.ledgerBalanced === true, "ledger 没对平");
must(s.rawChecked + s.uncheckedRaw === s.registeredRaw,
  `账没平：${s.rawChecked} + ${s.uncheckedRaw} ≠ ${s.registeredRaw}`);
must(Array.isArray(s.ledgerNoReason) && s.ledgerNoReason.length === 0,
  `有没核却说不出原因的表：${s.ledgerNoReason}`);

const o = [];
const say = (x) => { o.push(x); process.stdout.write(x + "\n"); };
say(`[自证] 719 / 1273 · miss 4/7 —— 与改前一致，被核键数没有下降`);
say(`registered raw ${s.registeredRaw} / distinct ${s.registeredDistinct}` +
    ` · unchecked raw ${s.uncheckedRaw} / distinct ${s.uncheckedDistinct}` +
    ` · 包装表 ${s.wrappedTables} · 正则表 ${s.regexTables}/${s.regexEntries} 条` +
    ` · balanced=${s.ledgerBalanced}`);
say(`ledger: ${JSON.stringify(s.ledger)}`);
say("");
say("── 合计行 why ──");
say(total.why);
say("");
say("── 合计行 details（末尾记账口径部分）──");
const det = total.details ?? total.items ?? [];
for (const d of det) say("  " + d);

fs.writeFileSync(path.join(ROOT, "4-临时脚本", "2026-08-16-round31", "probe_ledger_render.out.txt"),
  o.join("\n") + "\n", "utf8");
