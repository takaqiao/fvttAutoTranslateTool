/**
 * 第二十六轮 · 探针②：**驱动真身** `keyLiveness`，把面板自己算的报文原样打出来。
 *
 * ⚠ 三条纪律（本项目登记的空转形态换来的）：
 *   · 只桩 `fetch` / `game`，**判据一行都不抄** —— 抄一份就等于测了个副本（形态 c/h）；
 *   · 数一律读面板挂在 `合计.stats` 上的**它自己算的数**，探针不另算一份；
 *   · `game.packs` 老实给空数组 —— 离线拿不到合集索引，**不许捏一份 index 喂进去**
 *     （形态 (h)：真实 Foundry 的 `pack.index` 上有哪些字段由 `compendiumIndexFields` 决定，
 *      数据文件里有 `journal[].pages[].name` 推不出运行时拿得到它）。
 *
 * 用法：node probe_panel_report.mjs
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const DATA = "C:/Users/Taka/AppData/Local/FoundryVTT/Data";
const PROJ = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const HERE = path.join(PROJ, "4-临时脚本/2026-08-16-round26");

let fOk = 0, fFail = 0;
globalThis.fetch = async (url) => {
  const p = path.join(DATA, url);
  if (!fs.existsSync(p) || !fs.statSync(p).isFile()) { fFail++; return { ok: false, text: async () => "" }; }
  fOk++;
  return { ok: true, text: async () => fs.readFileSync(p, "utf8") };
};
globalThis.Hooks = { once() {}, on() {} };
globalThis.game = { system: { id: "crucible" }, packs: [] };

const SC = await import(pathToFileURL(path.join(PROJ, "1-Ember汉化插件/scripts/ember-cn-selfcheck.mjs")).href);

/* 现表的 harness：把 ember-hardcoded-cn.mjs 的 SELFCHECK_TABLES 导出来（与上一轮同一套桩）
 * ⚠ 第二十七轮改：harness 写进 `mkdtemp`、import 完 `finally` 删。
 *   原来写在 HERE（版本化目录）里，跑完留下一份 3233 行的 `_hc_now.mjs` —— 空转形态 (c) 的诱饵：
 *   今天与真身同源，上游一改就是过期快照，而它长得像「现表」，会被人 import 去当现表读。
 *   harness 每次都从真身现生成，留着零收益。 */
let mod;
{
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ec-r26-panel-"));
  try {
    const harness = path.join(tmpDir, "_hc_now.mjs");
    const src = fs.readFileSync(path.join(PROJ, "1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs"), "utf8");
    const IMPORT_LINE = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
    if (!src.includes(IMPORT_LINE)) throw new Error("找不到 SELFCHECK 的 import 行，转换中止");
    fs.writeFileSync(harness,
      "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n" +
      src.replace(IMPORT_LINE, "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };") +
      "\nexport { SELFCHECK_TABLES as __SELFCHECK_TABLES };\n", "utf8");
    mod = await import(pathToFileURL(harness).href);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

const checks = await SC.keyLiveness(mod.__SELFCHECK_TABLES);
const byName = (n) => checks.find(c => c.name === n);

console.log("══════ 上游语料（面板原文） ══════");
for (const i of byName("上游语料").items) console.log("  · " + i);

const adv = byName("Adventure 内层页名");
console.log("\n══════ Adventure 内层页名 ══════");
console.log(adv ? `  ${adv.status}  ${adv.detail}` : "  （离线 game.packs 为空，认不出 Adventure 合集 —— 本条按设计不出，不是漏报）");

console.log("\n══════ 逐表报文里**贴着那条绿**的弱证据标记 ══════");
{
  const rows = checks.filter(c => /短词|拼接展开|合集索引条目名/.test(c.detail ?? "") && c.name !== "上游语料" && c.name !== "合计");
  console.log(`  带弱证据标记的表：${rows.length} 张`);
  for (const c of rows) console.log(`  · [${c.status} ${c.checked}] ${c.name}\n      ${c.detail}`);
}

console.log("\n══════ 合计（面板原文 + 证据力分档） ══════");
const total = byName("合计");
console.log("  " + total.detail);
for (const i of total.items) console.log("  · " + i);

const st = total.stats;
console.log("\n══════ 面板自己算的数 ══════");
console.log(JSON.stringify({
  checkedDistinct: st.checkedDistinct, rawChecked: st.rawChecked,
  missDistinct: st.missDistinct, rawMiss: st.rawMiss,
  tpl: st.tpl, tierCounts: st.tierCounts,
  derivedKeys: st.derivedKeys.length, weakShort: st.weakShort.length,
  miss: st.miss
}, null, 1));
console.log(`fetch：成功 ${fOk} / 失败 ${fFail}`);

/* ── 假阴性对照：6 个自造的不存在串，必须 6/6 报出 ── */
const FAKES = {
  "Zzq Frobnicated Widget": "假串一",
  "This String Does Not Exist Upstream At All": "假串二",
  "Quaffle Marmalade Dispenser": "假串三",
  "Vorpal Blancmange Protocol": "假串四",
  "Xylophone Requisition Form": "假串五",
  "Gyroscopic Pemmican Requisition": "假串六"
};
const fk = (await SC.keyLiveness({ FAKE: FAKES })).find(c => c.name === "合计").stats;
console.log("\n══════ 假阴性对照 ══════");
console.log(`  自造 ${Object.keys(FAKES).length} 个不存在的串 → 报出 ${fk.missDistinct} 个（核了 ${fk.checkedDistinct} 键）`);

/* ── D 档边界：一串是另一串子串的那类改词，本档看不见（报文里必须写着） ── */
const sub = await SC.keyLiveness({ SUBSTR: { "Increase Ability": "x" } });
const subMiss = sub.find(c => c.name === "合计").stats.missDistinct;
console.log("\n══════ 子串型改词的边界（复现） ══════");
console.log(`  上游现有 \`Increase Ability Score\`，喂进「截短后」的 \`Increase Ability\` → 报出 ${subMiss} 条（预期 0，即**看不见**）`);
console.log(`  合计报文里有没有写明这条边界：${total.items.some(i => i.includes("边界之二")) ? "有" : "**没有**"}`);
console.log(`  合计报文里有没有写明 Adventure 内层页名查不了：${total.items.some(i => i.includes("边界之三")) ? "有" : "**没有**"}`);

const okFake = fk.missDistinct === Object.keys(FAKES).length;
const okSub = subMiss === 0;
const okTpl = st.tpl.files === st.tpl.litOk + st.tpl.dynOk + st.tpl.unrefOk;
console.log(`\n${okFake ? "  ok  " : "  FAIL"} 假阴性对照 6/6`);
console.log(`${okSub ? "  ok  " : "  FAIL"} 子串型改词确实看不见（这是被写进报文的**边界**，不是被修好的缺陷）`);
console.log(`${okTpl ? "  ok  " : "  FAIL"} 模板三路记账加起来等于抓到的份数`);
process.exit(okFake && okSub && okTpl ? 0 : 1);
