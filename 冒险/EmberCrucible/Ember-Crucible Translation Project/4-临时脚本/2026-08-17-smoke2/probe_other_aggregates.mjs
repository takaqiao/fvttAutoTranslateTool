/**
 * 用例：同一类缺陷的另外两条聚合数（B 档「合集索引抽样」· C 档「裁决键抽验」）。
 * ==================================================================================
 *
 * · B 档「合集索引抽样」的 ok 分支从前只报 `抽 137 条，中文 130 条（95%）` ——
 *   **那 7 条是谁，一个字都没有**。「按约定保留英文原名」与「某个包真的漏译了」
 *   在这条报文上**长得一模一样**。
 * · C 档「裁决键抽验」更隐蔽：取不到的探针键从前是**静静 `continue`** 掉的，
 *   报文写「查 1 条，全部符合既定裁决」，**分母自己缩了水而报文不留痕迹** ——
 *   那是本项目第二十四轮那个假 0 的形态（把项挪出视野换一个好看的数）。
 *
 * ⚠ 纪律与 `probe_pack_naming.mjs` 同：驱动面板真身、前置自证两件（条数 ＋ 地方）、
 *   桩的字段形状指得出上游契约、每条正向断言都配一条**掏空即变红**的反向用例。
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

/** 变异体落系统临时目录（理由同 `probe_pack_naming.mjs`：别在仓里留 98 KB 的面板副本）。 */
const MUT = fs.mkdtempSync(path.join(os.tmpdir(), "ec-mutant-"));

const PROJ = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const PANEL = path.join(PROJ, "1-Ember汉化插件/scripts/ember-cn-selfcheck.mjs");
const OUT = path.join(PROJ, "4-临时脚本/2026-08-17-smoke2");

const log = [];
let failures = 0;
function say(s) { log.push(s); process.stdout.write(s + "\n"); }
function ok(name, cond, detail) {
  if (!cond) failures++;
  say(`${cond ? "PASS" : "FAIL"}  ${name}${detail ? " — " + detail : ""}`);
  return cond;
}
function hard(msg) { say("HARD-FAIL  " + msg); fs.writeFileSync(path.join(OUT, "probe_other_aggregates.log"), log.join("\n"), "utf8"); process.exit(2); }

const src = fs.readFileSync(PANEL, "utf8");

/* ── 前置自证 · C 档探针表的**真值条数**从源码现抠，不手打 ── */
const probeBlock = src.match(/const I18N_PROBES = \[([\s\S]*?)\n\];/);
if (!probeBlock) hard("抠不到 `I18N_PROBES` —— C 档结构变了，本探针必须重锚");
const probeKeys = [...probeBlock[1].matchAll(/\["([^"]+)",/g)].map(m => m[1]);
ok("前置自证 · 从面板源码抠出的探针键 = 2 条", probeKeys.length === 2, probeKeys.join(" · "));
if (probeKeys.length !== 2) hard("探针键条数与已知真值不符");

/* ── 桩 ─────────────────────────────────────────────────────────
   契约出处：`game.i18n.localize(key)` —— Foundry `Localization#localize`，
   **键取不到时原样吐回键名**（面板 `if (got === key)` 正是照这个契约写的）。
   `pack.index` 条目只给 `_id`/`name`（`compendiumIndexFields` 里确有的字段）。 */
function makeGame({ packEntries, i18nMap }) {
  const asked = [];
  return {
    version: "14.366",
    world: { title: "探针世界", id: "probe" },
    system: { id: "crucible", version: "0.10.1" },
    modules: { get(id) { return id === "babele" ? { active: true, version: "2.9.1" } : undefined; } },
    packs: Object.entries(packEntries).map(([col, names]) => ({
      collection: col,
      documentName: "Item",
      metadata: { packageName: col.split(".")[0], name: col.split(".")[1], label: col.split(".")[1], type: "Item" },
      index: names.map((n, i) => ({ _id: `id${i}`, name: n })),
    })),
    i18n: {
      translations: { PROBE: 1 },
      localize(k) { asked.push(k); return Object.prototype.hasOwnProperty.call(i18nMap, k) ? i18nMap[k] : k; },
    },
    babele: { isTranslated(p) { return typeof p === "string"; } },
    __asked: asked,
  };
}

async function run(panelPath, opts) {
  globalThis.Hooks = { once() {}, on() {} };
  globalThis.CONFIG = {};
  const g = makeGame(opts);
  globalThis.game = g;
  const SC = await import(pathToFileURL(panelPath).href + `?v=${Math.random()}`);
  const checks = await SC.runSelfCheck({});
  return { checks, row: (n) => checks.find(c => c.name === n), g };
}

const MIXED = {
  "crucible.spell": ["火球术 Fireball", "闪电箭", "Prestidigitation", "空洞之月"],
  "crucible.talent": ["坚毅", "Hexblade", "余烬之心"],
};
const ALL_CN = { "crucible.spell": ["火球术", "闪电箭"], "crucible.talent": ["坚毅"] };

/* ────────────────────────────────────────────────────────────────
   1 · B 档「合集索引抽样」
   ──────────────────────────────────────────────────────────────── */
say("== B 档 · 合集索引抽样 ==");
{
  const { row } = await run(PANEL, { packEntries: MIXED, i18nMap: {} });
  const r = row("合集索引抽样");
  if (!r) hard("面板没吐出「合集索引抽样」—— 锚点变了");
  const d = r.detail ?? "", items = (r.items ?? []).join("\n");
  ok("B-a 分母仍是抽样总数 7", r.checked === 7, `实得 ${r.checked}`);
  ok("B-b 百分比照报（5/7）", d.includes("中文 5 条") && d.includes("7 条"), d.slice(0, 60));
  ok("B-c **逐条点名**那 2 条没有中文的条目",
    items.includes("Prestidigitation") && items.includes("Hexblade"), items.replace(/\n/g, " | "));
  ok("B-d 点名里带上是哪个包", items.includes("crucible.spell :: Prestidigitation"), "");
  ok("B-e 报文明说「真的漏译时长得一模一样」，不许只报百分比",
    d.includes("长得一模一样"), "");
}
{
  const { row } = await run(PANEL, { packEntries: ALL_CN, i18nMap: {} });
  const r = row("合集索引抽样");
  ok("B-f 全中文时报 100%、没有点名清单",
    r.detail.includes("全部是中文") && (r.items ?? []).length === 0, r.detail.slice(0, 50));
}

/* ────────────────────────────────────────────────────────────────
   2 · C 档「裁决键抽验」：分母不许自己缩水
   ──────────────────────────────────────────────────────────────── */
say("");
say("== C 档 · 裁决键抽验 ==");
{
  // 只给第一个探针键一个正确的值，第二个键**世界里取不到** ⇒ 从前会静静缩成「查 1 条」
  const { row, g } = await run(PANEL, { packEntries: ALL_CN, i18nMap: { [probeKeys[0]]: "地区地图" } });
  const r = row("裁决键抽验");
  if (!r) hard("面板没吐出「裁决键抽验」—— 锚点变了");
  const d = r.detail ?? "", items = (r.items ?? []).join("\n");
  ok("前置自证 · 面板确实把 2 个探针键都问了一遍",
    probeKeys.every(k => g.__asked.includes(k)), g.__asked.filter(k => probeKeys.includes(k)).join(" · "));
  ok("C-a 状态 ok（查到的那条符合裁决）", r.status === "ok", `实得 ${r.status}`);
  ok("C-b 报文带**分母**：查 1/2 条", d.includes("1/2"), d.slice(0, 50));
  ok("C-c **点名**那条没查的键，并明说「不是通过」",
    items.includes(probeKeys[1]) && (items.includes("本次没查") || d.includes("根本没查")),
    items.replace(/\n/g, " | ").slice(0, 160));
  ok("C-d 报文明说另有 1 条没查", d.includes("另有 1 条"), "");
}
{
  const { row } = await run(PANEL, { packEntries: ALL_CN, i18nMap: { [probeKeys[0]]: "地区地图", [probeKeys[1]]: "奥拉" } });
  const r = row("裁决键抽验");
  ok("C-e 两条都取到时报 2/2 且明说没有静默少查",
    r.detail.includes("2/2") && r.detail.includes("全部"), r.detail.slice(0, 60));
}
{
  const { row } = await run(PANEL, { packEntries: ALL_CN, i18nMap: { [probeKeys[0]]: "区域地图" } });
  const r = row("裁决键抽验");
  ok("C-f 值与裁决不符时仍是 fail，且分母与点名都在",
    r.status === "fail" && r.detail.includes("1/2") && (r.items ?? []).join("\n").includes(probeKeys[1]),
    `${r.status} · ${r.detail.slice(0, 40)}`);
}

/* ────────────────────────────────────────────────────────────────
   3 · 反向用例：掏空即变红
   ──────────────────────────────────────────────────────────────── */
say("");
say("== 反向用例 ==");
function region(startMark, endMark) {
  const a = src.indexOf(startMark), b = src.indexOf(endMark);
  if (a < 0 || b < 0 || b <= a) hard(`抠不到区间 ${startMark} → ${endMark}`);
  return [a, b];
}
function mutate(from, to, expectHits, [lo, hi], where) {
  let hits = 0, i = 0, bad = 0;
  while ((i = src.indexOf(from, i)) !== -1) { hits++; if (i < lo || i >= hi) bad++; i += from.length; }
  ok(`前置自证 · 变异「${from.slice(0, 30)}…」命中 ${expectHits} 处`, hits === expectHits, `实得 ${hits}`);
  ok(`前置自证 · 这 ${hits} 处全部落在 ${where} 区间内`, bad === 0, `越界 ${bad} 处`);
  if (hits !== expectHits || bad) hard("变异前置自证没过 —— 反向用例没跑成，不是通过");
  return src.split(from).join(to);
}

const B_RANGE = region("\nfunction checkBabele() {", "\nfunction checkI18n() {");
const C_RANGE = region("\nfunction checkI18n() {", "\n/*  E · 运行时补丁 ");

/* N1 · B 档：把点名清单掏空 */
{
  const text = mutate("      cappedList(enNames)));", "      []));", 1, B_RANGE, "checkBabele()");
  const p = path.join(MUT, "_mutant_N1.mjs");
  fs.writeFileSync(p, text, "utf8");
  const { row } = await run(p, { packEntries: MIXED, i18nMap: {} });
  const items = (row("合集索引抽样").items ?? []).join("\n");
  ok("N1 掏空 B 档点名后，B-c **当场变红**", !items.includes("Prestidigitation"),
    items ? "⚠⚠ 掏空了还点得出名" : "变异体点不出名 ⇒ 断言有效");
}
/* N2 · C 档：把取不到的键退回静静 continue */
{
  const text = mutate("      absent.push(", "      [].push(", 1, C_RANGE, "checkI18n()");
  const p = path.join(MUT, "_mutant_N2.mjs");
  fs.writeFileSync(p, text, "utf8");
  const { row } = await run(p, { packEntries: ALL_CN, i18nMap: { [probeKeys[0]]: "地区地图" } });
  const r = row("裁决键抽验");
  const items = (r.items ?? []).join("\n");
  ok("N2 退回静默 continue 后，C-c（点名没查的键）**当场变红**", !items.includes(probeKeys[1]),
    items ? "⚠⚠ 掏空了还点得出名" : "变异体点不出名 ⇒ 断言有效");
}
/* N3 · C 档：把分母缩回「查 N 条」（去掉 /总数） */
{
  const text = mutate("`查 ${checked}/${I18N_PROBES.length} 条，全部符合既定裁决。`",
    "`查 ${checked} 条，全部符合既定裁决。`", 1, C_RANGE, "checkI18n()");
  const p = path.join(MUT, "_mutant_N3.mjs");
  fs.writeFileSync(p, text, "utf8");
  const { row } = await run(p, { packEntries: ALL_CN, i18nMap: { [probeKeys[0]]: "地区地图" } });
  ok("N3 把分母缩回去后，C-b（1/2）**当场变红**", !row("裁决键抽验").detail.includes("1/2"),
    "变异体报文没有分母 ⇒ 断言有效");
}

fs.rmSync(MUT, { recursive: true, force: true });
say("");
say(`== 合计：失败 ${failures} 条 ==`);
fs.writeFileSync(path.join(OUT, "probe_other_aggregates.log"), log.join("\n"), "utf8");
process.exit(failures ? 1 : 0);
