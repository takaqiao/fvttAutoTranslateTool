/**
 * 2026-08-17 · 驱动**发布中的面板真身**验证 B 档两处修复。
 *
 * 与 probe_babele_arg.mjs 的区别：那份验的是「桩逻辑对不对」，
 * 这份 import 真实的 ember-cn-selfcheck.mjs、桩掉 Foundry 全局，验**真身的行为**。
 *
 * 两处修复：
 *   ① 根因：`isTranslated()` 要 collection **字符串**，此前传的是 CompendiumCollection 对象
 *   ② 交叉印证：本行与「索引抽样」互相矛盾时不许再报自信的 ❌
 *
 * ⚠ 前置自证：先断言本探针**真的驱动到了** B 档（能取到那两行），再往下判。
 */
const PANEL = "../../1-Ember汉化插件/scripts/ember-cn-selfcheck.mjs";

let fail = 0;
const ok = (c, m) => { console.log(`  ${c ? "ok  " : "FAIL"}  ${m}`); if (!c) fail++; };

/** 照 babele 2.9.1 真身语义：packs 是**以字符串为键**的 Map */
function babeleStub(translatedIds) {
  const packs = new Map(translatedIds.map(id => [id, { translated: true }]));
  return { active: true, version: "2.9.1", isTranslated: p => !!(packs.get(p)?.translated) };
}

function packStub(collection, names) {
  const [pkg, name] = collection.split(".");
  return {
    collection,
    metadata: { packageName: pkg, id: name, label: collection },
    index: names.map(n => ({ name: n })),
  };
}

/** 装一套最小的 Foundry 全局，跑 runSelfCheck，取回 B 档的行 */
async function runB({ translatedIds, packs }) {
  globalThis.game = {
    modules: new Map([
      ["babele", { active: true, version: "2.9.1" }],
      ["ember_cn_unofficial", { active: true, version: "1.1.22" }],
      ["crucible-cn", { active: true, version: "0.9.12" }],
      ["ember", { active: true, version: "0.6.0" }],
      ["foundry_chn", { active: false }],
    ]),
    system: { id: "crucible", version: "0.10.1" },
    packs,
    babele: babeleStub(translatedIds),
    i18n: { translations: {} },
    settings: { register: () => {}, registerMenu: () => {}, get: () => false },
  };
  globalThis.foundry = { applications: { api: {} } };
  globalThis.CONFIG = {};
  globalThis.document = undefined;

  const mod = await import(PANEL + `?v=${Math.random()}`);
  const checks = await mod.runSelfCheck({});
  return checks.filter(c => String(c.section).startsWith("B"));
}

const NAMES_CN = ["深渊 The Abyss", "余烬之心", "阿克图瑞尔 Arcturel", "同调", "地区地图", "任务", "指示物", "阶位"];
const NAMES_EN = ["The Abyss", "Heart of Ember", "Arcturel", "Attunement", "Region Map", "Quest", "Token", "Rank"];

console.log("前置自证：探针真的驱动到了 B 档");
{
  const rows = await runB({
    translatedIds: ["ember.crucible-adventure"],
    packs: [packStub("ember.crucible-adventure", NAMES_CN)],
  });
  ok(rows.length >= 2, `B 档取回 ${rows.length} 行（应 ≥2：已认到译文 + 索引抽样）`);
  ok(rows.some(r => r.name === "已认到译文的合集"), "取到了「已认到译文的合集」这一行");
  ok(rows.some(r => r.name === "合集索引抽样"), "取到了「合集索引抽样」这一行");
}

console.log("\n① 根因：译文文件真的在时，必须认到（此前传对象→永远 0）");
{
  const rows = await runB({
    translatedIds: ["ember.crucible-adventure", "ember.adventure"],
    packs: [packStub("ember.crucible-adventure", NAMES_CN), packStub("ember.adventure", NAMES_CN)],
  });
  const r = rows.find(x => x.name === "已认到译文的合集");
  console.log(`    实得：[${r.status}] ${r.detail}`);
  ok(r.status === "ok", "状态是 ok（修复前这里必然是 fail —— 真实世界报的就是 22 个一个都没认到）");
  ok(/2\/2/.test(r.detail), "认到 2/2");
}

console.log("\n② 交叉印证：认不到、但屏上是中文 → 不许再报自信的 ❌");
{
  const rows = await runB({
    translatedIds: [],                                   // 模拟判据又错了
    packs: [packStub("ember.crucible-adventure", NAMES_CN)],
  });
  const r = rows.find(x => x.name === "已认到译文的合集");
  console.log(`    实得：[${r.status}] ${String(r.detail).slice(0, 76)}…`);
  ok(r.status === "warn", "降级为 warn，不是 fail");
  ok(/自相矛盾/.test(r.detail), "报文明说两行自相矛盾");
  ok(/别去核译文|先去核这行代码/.test(r.detail), "把矛头指向判据本身，而不是译文");
}

console.log("\n③ 防恒真：认不到、屏上也确实是英文 → 仍然必须报 fail");
{
  const rows = await runB({
    translatedIds: [],
    packs: [packStub("ember.crucible-adventure", NAMES_EN)],
  });
  const r = rows.find(x => x.name === "已认到译文的合集");
  console.log(`    实得：[${r.status}] ${String(r.detail).slice(0, 76)}…`);
  ok(r.status === "fail", "真的没生效时仍报 fail —— 不是把检查改成永不失败换来的绿");
  ok(/互相印证/.test(r.detail), "说明两条互相印证，这次多半是真的");
}

console.log(`\n合计：${fail === 0 ? "全部通过" : fail + " 条 FAIL"}`);
process.exit(fail ? 1 : 0);
