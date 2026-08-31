#!/usr/bin/env node
/**
 * probe3 —— 直接回到 LevelDB 真身，逐字确认 probe2 报出的 5 处「仓里英文基线 ≠ 上游」
 * 到底哪一边是上游。
 *
 * 不经过 extract_en.mjs：只开库、取出 `!adventures!` 的原始 JSON 字符串，
 * 在**未解析的字节串**上找特征子串。这样连「抽取器改了字符」这条可能性也排除掉。
 */
import path from 'path';
import { createRequire } from 'module';

const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const DATA = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data';

const NEEDLES = [
  // [仓里 compendium/en 的写法, 上游新抽出来的写法]
  ['Silver Beam Consortium are', 'Silver Beam C0nsortium are'],
  ['[[/skillCheck athletics 15]] check', '[[/skillCheck athletics 15 check'],
  ['@Condition[exhaustion]</sub>', '@Condition[exhaustion</sub>'],
  ['Charisma (Persuasion) check', 'Charisma {Persuasion} check'],
  ['&amp;reference[Paralyzed] for 1 minute', 'reference[Paralyzed] for 1 minute'],
];

/**
 * 判决部分：不靠子串计数，而是把 gap_detail.json 里记下的 5 处漂移，
 * 用**文档内的具体路径**从 LevelDB 原文档取值，与两边逐字比对。
 * 子串计数只作旁证（同一个字符串可能在别的页面里也出现，所以计数本身判不了案 ——
 * 上一版正是因此吐了三条「不确定」）。
 */
import fs from 'fs';
import { fileURLToPath } from 'url';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const drifts = JSON.parse(fs.readFileSync(path.join(__dirname, 'gap_detail.json'), 'utf8')).drifts;

/** 在 adventure blob 里按 `journals.<期刊名>.pages.<页名>.text` / `items.<名>.effects.<名>.description` 解析。 */
// 叶路径里的段是 **mapping 的字段名**，不一定等于文档里的属性名。
// Babele 的 Adventure 映射写的是 `journals: {path:'journal', …}`，
// 所以 `journals.…` 在文档里要走 `journal`。上一版没做这层换算，8 条判不出来。
const SEG_ALIAS = { journals: 'journal' };
function resolveLeaf(adv, leaf) {
  const segs = leaf.split('.');
  let node = adv;
  for (let i = 0; i < segs.length; i += 1) {
    const s0 = segs[i];
    const s = (node && typeof node === 'object' && !Array.isArray(node) && !(s0 in node) && SEG_ALIAS[s0]) ? SEG_ALIAS[s0] : s0;
    if (Array.isArray(node)) {
      const hit = node.find((x) => x?.name === s);
      if (!hit) return { ok: false, at: leaf, why: `数组里找不到 name=${s}` };
      node = hit;
      continue;
    }
    if (node && typeof node === 'object' && s in node) { node = node[s]; continue; }
    // `text` 这一段在文档里是 `text.content`
    if (s === 'text' && node?.text?.content !== undefined) { node = node.text.content; continue; }
    return { ok: false, at: leaf, why: `到 ${s} 断了` };
  }
  return { ok: true, value: node };
}

console.log('=== 判决：按文档路径取值逐字比对 ===');
let decided = 0; let freshWins = 0;
for (const d of drifts) {
  const packName = d.pack.split('.').slice(1).join('.');
  const dir = path.join(DATA, 'modules/ember/packs', packName);
  const db = new ClassicLevel(dir, { createIfMissing: false });
  let adv = null;
  for await (const [k, v] of db.iterator()) {
    if (k.toString().startsWith('!adventures!')) adv = JSON.parse(v.toString());
  }
  await db.close();
  // 叶路径末段是 `text`，但文档里是 `text.content`；先试原样，再试 text.content
  let r = resolveLeaf(adv, d.leaf);
  if (r.ok && r.value && typeof r.value === 'object' && typeof r.value.content === 'string') r = { ok: true, value: r.value.content };
  if (!r.ok) { console.log(`  [??] ${d.pack} ${d.leaf} -> 解析失败：${r.why}`); continue; }
  decided += 1;
  const eqFresh = r.value === d.freshEn;
  const eqStored = r.value === d.storedEn;
  if (eqFresh && !eqStored) freshWins += 1;
  console.log(`  [${eqFresh && !eqStored ? '判定：上游＝新抽，仓里 en 被改过' : (eqStored && !eqFresh ? '判定：上游＝仓里 en，新抽有问题' : '判定：两边都不等，异常')}] `
    + `${d.pack} :: ${d.leaf}`);
}
console.log(`  可判决 ${decided} / ${drifts.length}，其中「上游＝新抽」${freshWins} 条\n`);

console.log('=== 旁证：原始字节里的子串计数（同串可能在别处也出现，仅供参考）===');
for (const pack of ['adventure', 'crucible-adventure']) {
  const dir = path.join(DATA, 'modules/ember/packs', pack);
  const db = new ClassicLevel(dir, { createIfMissing: false });
  let raw = '';
  for await (const [k, v] of db.iterator()) {
    if (k.toString().startsWith('!adventures!')) raw += v.toString();
  }
  await db.close();
  console.log(`\n--- ember.${pack}  (原始 JSON 字节数 ${raw.length}) ---`);
  for (const [stored, fresh] of NEEDLES) {
    // JSON 里的字符串是转义过的：`"` 之类要还原。这里的特征串不含需要转义的字符，
    // 但 `&amp;` 与方括号原样出现，可直接找。
    const a = raw.split(stored).length - 1;
    const b = raw.split(fresh).length - 1;
    console.log(`  仓里写法 "${stored.slice(0, 45)}" 出现 ${a} 次 | 上游新抽写法 "${fresh.slice(0, 45)}" 出现 ${b} 次`
      + `  => 真身是：${b > 0 && a === 0 ? '新抽的那个（仓里 en 被改过）' : (a > 0 && b === 0 ? '仓里的那个（新抽有问题）' : '不确定')}`);
  }
}
