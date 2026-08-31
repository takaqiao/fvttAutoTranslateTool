/**
 * 2026-08-17 冒烟验证暴露的缺陷 · 复现测试
 *
 * 真实世界自检报告里出现自相矛盾的两行：
 *     ❌ 已认到译文的合集（22 项）—— 22 个合集一个都没认到译文
 *     ✅ 合集索引抽样（137 项）—— 抽 137 条，中文 137 条（100%）
 *
 * 根因（读 babele 2.9.1 真身确认，不是推测）：
 *   babele.js:557        isTranslated(pack)            @param {string} pack  ex. "dnd5e.classes"
 *   translation-session.js:111  isTranslated(collection) → !!translatedCompendiumFor(collection)
 *   mapped-compendiums.js:60    translated(pack)      → this.get(pack)
 *   mapped-compendiums.js:49    get(pack)             → this.packs.get(pack)   ← 以**字符串**为键的 Map
 * ⇒ 传 CompendiumCollection **对象**进去必然 miss，返回 false，与库里有没有译文无关。
 *
 * 本探针用一个**照着真身语义写的桩**（只认字符串键）来复现：
 *   传对象 → false（旧写法，制造假警报）
 *   传 p.collection 字符串 → true（正确）
 *
 * ⚠ 前置自证：先断言桩自己的行为与真身一致，再往下判 —— 否则这个探针自己就是空转。
 */

const TRANSLATED = new Set([
  "ember.crucible-adventure", "ember.crucible-character", "ember.adventure",
]);

/** 照着 mapped-compendiums.js 的 Map 语义写：**只认字符串键** */
function makeBabeleStub() {
  const packs = new Map();
  for (const id of TRANSLATED) packs.set(id, { translated: true });
  return {
    active: true,
    version: "2.9.1",
    isTranslated(pack) {           // babele.js:557 的语义
      return !!(packs.get(pack)?.translated);
    },
  };
}

function makePacks() {
  return [
    { collection: "ember.crucible-adventure", metadata: { packageName: "ember", id: "crucible-adventure" } },
    { collection: "ember.crucible-character", metadata: { packageName: "ember", id: "crucible-character" } },
    { collection: "ember.adventure", metadata: { packageName: "ember", id: "adventure" } },
    { collection: "crucible.rules", metadata: { packageName: "crucible", id: "rules" } },
  ];
}

let fail = 0;
const ok = (cond, msg) => { console.log(`  ${cond ? "ok  " : "FAIL"}  ${msg}`); if (!cond) fail++; };

// ── 前置自证：桩的行为必须与真身一致，否则本探针无意义 ──
console.log("前置自证（桩 vs babele 2.9.1 真身语义）");
{
  const b = makeBabeleStub();
  const p = makePacks()[0];
  ok(b.isTranslated("ember.crucible-adventure") === true,
     "字符串键命中 → true（mapped-compendiums.js:49 `this.packs.get(pack)`）");
  ok(b.isTranslated(p) === false,
     "对象键 miss → false（Map 以字符串为键，传对象必然 miss）");
  ok(b.isTranslated("ember.no-such-pack") === false, "未登记的字符串 → false");
}

// ── 复现：旧写法 vs 新写法 ──
console.log("\n复现（4 个合集，其中 3 个真的有译文）");
{
  const b = makeBabeleStub();
  const packs = makePacks();

  const oldWay = packs.filter(p => { try { return b.isTranslated(p); } catch { return false; } });
  const newWay = packs.filter(p => {
    const id = p.collection ?? `${p.metadata?.packageName}.${p.metadata?.id}`;
    try { return b.isTranslated(id); } catch { return false; }
  });

  console.log(`    旧写法（传对象）  认到 ${oldWay.length}/${packs.length}`);
  console.log(`    新写法（传字符串）认到 ${newWay.length}/${packs.length}`);

  ok(oldWay.length === 0,
     "旧写法：即使 3 个合集真的有译文，也报 0 —— 这就是真实世界那条 ❌ 的来源");
  ok(newWay.length === 3,
     "新写法：认到 3 个，与真实情况一致");
  ok(newWay.length !== packs.length,
     "反向不误伤：crucible.rules 没有译文文件，不该被算进去（防「改成恒真」）");
}

// ── 防恒真：一份译文都没有时，新写法必须仍然报 0 ──
console.log("\n防恒真（真的一份译文都没有）");
{
  const empty = { active: true, version: "2.9.1", isTranslated: () => false };
  const packs = makePacks();
  const n = packs.filter(p => empty.isTranslated(p.collection)).length;
  ok(n === 0, "库里确实没译文时，新写法仍报 0 —— 不是把检查改成恒真换来的绿");
}

console.log(`\n合计：${fail === 0 ? "全部通过" : fail + " 条 FAIL"}`);
process.exit(fail ? 1 : 0);
