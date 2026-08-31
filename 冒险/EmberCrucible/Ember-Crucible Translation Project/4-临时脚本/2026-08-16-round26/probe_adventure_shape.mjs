/**
 * 第二十六轮 · 复核探针 ①：Adventure 合集的真实文档形状
 *
 * 要回答的只有三个问题（都必须**读真库**，不许拿数据文件形状推运行时形状 —— 形态 (h)）：
 *   ① `ember.crucible-adventure` 里到底有几个文档？（决定 (ii) 路线的成本）
 *   ② `The Abyss` / `Heart of Ember` 到底躺在哪一层？
 *   ③ 顶层文档有没有 `pages` 字段？（面板里 `e.pages ?? []` 那一路的死活）
 *
 * ⚠ 本探针**不**回答「运行时 index 里有没有 pages」—— 那由
 *   `Adventure.metadata.compendiumIndexFields`（common/documents/adventure.mjs:26）决定，
 *   与库里存了什么无关。两件事分开验，别混。
 */
import { ClassicLevel } from "file:///C:/Program Files/Foundry Virtual Tabletop/resources/app/node_modules/classic-level/index.js";

const PACKS = ["crucible-adventure", "adventure"];
const NEEDLES = ["The Abyss", "Heart of Ember"];

for (const packName of PACKS) {
  const dir = `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs/${packName}`;
  const db = new ClassicLevel(dir, { valueEncoding: "json" });
  await db.open();
  const docs = [];
  for await (const [k, v] of db.iterator()) docs.push([k, v]);
  console.log(`\n===== ${packName} =====`);
  console.log(`文档数：${docs.length}`);
  for (const [k, v] of docs) {
    const topKeys = Object.keys(v ?? {});
    console.log(`  ${k} :: name=${JSON.stringify(v?.name)}`);
    console.log(`    顶层字段：${topKeys.join(", ")}`);
    console.log(`    顶层有 pages? ${Object.hasOwn(v ?? {}, "pages")}`);
    console.log(`    journal 条目数：${Array.isArray(v?.journal) ? v.journal.length : "(无 journal 数组)"}`);
    if (Array.isArray(v?.journal)) {
      let pageCount = 0;
      for (const j of v.journal) pageCount += Array.isArray(j?.pages) ? j.pages.length : 0;
      console.log(`    journal[].pages 总页数：${pageCount}`);
    }
  }

  // 逐针定位：在文档树里找出 needle 出现在哪条路径的 `name` 字段上
  for (const needle of NEEDLES) {
    const hits = [];
    const walk = (node, path) => {
      if (Array.isArray(node)) { node.forEach((x, i) => walk(x, `${path}[${i}]`)); return; }
      if (node && typeof node === "object") {
        for (const [kk, vv] of Object.entries(node)) {
          if (kk === "name" && vv === needle) hits.push(`${path}.name`);
          walk(vv, `${path}.${kk}`);
        }
      }
    };
    for (const [k, v] of docs) walk(v, `${k}`);
    console.log(`  「${needle}」作为 name 出现在 ${hits.length} 处：`);
    for (const h of hits.slice(0, 8)) console.log(`      ${h}`);
  }
  await db.close();
}
