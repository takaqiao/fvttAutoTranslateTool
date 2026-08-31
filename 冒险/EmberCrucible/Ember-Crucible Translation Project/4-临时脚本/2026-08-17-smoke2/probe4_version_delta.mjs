#!/usr/bin/env node
/**
 * probe4 —— crucible 0.10.1 → 0.10.2 的版本增量。
 *
 * 本轮对账进行到一半时，**上游在这台机器上自己动了**：
 * `Data/systems/crucible/` 整目录于 2026-08-17 03:41 被替换（system.json 由
 * 0.10.1 变 0.10.2，`download` 指向 release-0.10.2/system.zip）。
 * 本探针拿本轮**先后两次从 LevelDB 真身抽出来的英文基线**逐叶比：
 *   snapshot_crucible_0.10.1/en_fresh_crucible/  （03:25 抽，system.json=0.10.1）
 *   en_fresh/crucible/                            （03:45 抽，system.json=0.10.2）
 * 两次用的是同一个抽取器、同一份 mapping，所以差额只可能来自上游内容本身。
 *
 * 这同时是项目登记的未验证事项「上游一动就会有缺口」的一次**真实自然实验**。
 *
 * 用法： node probe4_version_delta.mjs   产物： version_delta.md / version_delta.json
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OLD = path.join(__dirname, 'snapshot_crucible_0.10.1/en_fresh_crucible');
const NEW = path.join(__dirname, 'en_fresh/crucible');
const CN = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/2-Crucible汉化插件/compendium/cn';

const readJSON = (p) => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : null);
function leaves(node, prefix = '', out = {}) {
  if (typeof node === 'string') { if (node.trim()) out[prefix] = node; return out; }
  if (Array.isArray(node)) { node.forEach((v, i) => leaves(v, `${prefix}[${i}]`, out)); return out; }
  if (node && typeof node === 'object') for (const [k, v] of Object.entries(node)) leaves(v, prefix ? `${prefix}.${k}` : k, out);
  return out;
}

const oldSrc = readJSON(path.join(OLD, '_source.json'));
const newSrc = readJSON(path.join(NEW, '_source.json'));

const rows = [];
const added = []; const removed = []; const changed = [];
for (const p of newSrc.packs) {
  const file = `crucible.${p.pack}.json`;
  const a = readJSON(path.join(OLD, file)); const b = readJSON(path.join(NEW, file));
  const cn = readJSON(path.join(CN, file));
  const A = {}; const B = {};
  for (const [k, v] of Object.entries(a?.entries ?? {})) A[k] = leaves(v);
  for (const [k, v] of Object.entries(b?.entries ?? {})) B[k] = leaves(v);
  let nAdd = 0; let nRem = 0; let nChg = 0; let chgChars = 0; let addChars = 0;
  for (const k of Object.keys(B)) {
    if (!(k in A)) { added.push({ pack: p.pack, entry: k, kind: 'entry' }); nAdd += 1; continue; }
    for (const [lp, s] of Object.entries(B[k])) {
      if (!(lp in A[k])) {
        nAdd += 1; addChars += s.length;
        added.push({ pack: p.pack, entry: k, leaf: lp, chars: s.length,
          cnHasIt: !!(cn && leaves(cn.entries?.[k] ?? {})[lp]) });
      } else if (A[k][lp] !== s) {
        nChg += 1; chgChars += Math.abs(s.length - A[k][lp].length);
        changed.push({ pack: p.pack, entry: k, leaf: lp,
          oldChars: A[k][lp].length, newChars: s.length,
          cnHasIt: !!(cn && leaves(cn.entries?.[k] ?? {})[lp]),
          old: A[k][lp], new: s });
      }
    }
    for (const lp of Object.keys(A[k])) if (!(lp in B[k])) { nRem += 1; removed.push({ pack: p.pack, entry: k, leaf: lp }); }
  }
  for (const k of Object.keys(A)) if (!(k in B)) { removed.push({ pack: p.pack, entry: k, kind: 'entry' }); nRem += 1; }
  if (nAdd || nRem || nChg) rows.push({ pack: p.pack, added: nAdd, addedChars: addChars, removed: nRem, changed: nChg, changedCharDelta: chgChars });
}

const md = [];
md.push(`# crucible ${oldSrc.packageVersion} → ${newSrc.packageVersion} 上游增量（同一抽取器、同一 mapping，两次直读 LevelDB）`);
md.push('');
md.push('| pack | 新增叶 | 新增字符 | 消失叶 | 文本变化叶 |');
md.push('|---|--:|--:|--:|--:|');
for (const r of rows) md.push(`| ${r.pack} | ${r.added} | ${r.addedChars} | ${r.removed} | ${r.changed} |`);
md.push('');
md.push(`合计：新增 ${added.length} 叶 / ${added.reduce((a, x) => a + (x.chars ?? 0), 0)} 字符 · `
  + `消失 ${removed.length} 叶 · 文本变化 ${changed.length} 叶`);
md.push('');
md.push('## 新增（上游 0.10.2 才有的可译文本）');
md.push('');
md.push('| pack | 条目 | 叶 | 字符 | cn 侧已有? |');
md.push('|---|---|---|--:|---|');
for (const x of added) md.push(`| ${x.pack} | \`${x.entry}\` | \`${x.leaf ?? '(整条)'}\` | ${x.chars ?? ''} | ${x.cnHasIt ? '有' : '**无**'} |`);
md.push('');
md.push('## 文本变化（键没变、英文原文变了 ⇒ 已有译文对应的是旧英文）');
md.push('');
md.push('| pack | 条目 | 叶 | 旧字符 | 新字符 | cn 侧有译文? |');
md.push('|---|---|---|--:|--:|---|');
for (const x of changed) md.push(`| ${x.pack} | \`${x.entry}\` | \`${x.leaf}\` | ${x.oldChars} | ${x.newChars} | ${x.cnHasIt ? '有（已过时）' : '无'} |`);

fs.writeFileSync(path.join(__dirname, 'version_delta.json'),
  `${JSON.stringify({ from: oldSrc.packageVersion, to: newSrc.packageVersion, rows, added, removed, changed }, null, 2)}\n`, 'utf8');
fs.writeFileSync(path.join(__dirname, 'version_delta.md'), `${md.join('\n')}\n`, 'utf8');
console.log(md.join('\n'));
