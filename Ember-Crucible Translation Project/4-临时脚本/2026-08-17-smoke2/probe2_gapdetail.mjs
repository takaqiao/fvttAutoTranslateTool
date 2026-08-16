#!/usr/bin/env node
/**
 * probe2 —— 把 probe1 量出来的叶级缺口逐条摊开，并解释「上游漂移」的 10 处叶变化。
 *
 * probe1 的结论是：条目级缺口 0，叶级缺口只落在 ember 的两个冒险包
 * （adventure 25 叶 / crucible-adventure 88 叶，合计 113 叶 / 9476 字符）。
 * 本探针给出这 113 叶的**完整名单**（不是抽样）、按叶路径末段分类、并附英文原文，
 * 好让项目所有者裁「译什么、怎么译」。
 *
 * 用法： node probe2_gapdetail.mjs
 * 产物： gap_detail.json / gap_detail.md
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project';
const readJSON = (p) => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : null);

function leaves(node, prefix = '', out = {}) {
  if (typeof node === 'string') { if (node.trim()) out[prefix] = node; return out; }
  if (Array.isArray(node)) { node.forEach((v, i) => leaves(v, `${prefix}[${i}]`, out)); return out; }
  if (node && typeof node === 'object') {
    for (const [k, v] of Object.entries(node)) leaves(v, prefix ? `${prefix}.${k}` : k, out);
  }
  return out;
}

const TARGETS = [
  { pkg: 'crucible', repo: path.join(ROOT, '2-Crucible汉化插件'), fresh: path.join(__dirname, 'en_fresh/crucible') },
  { pkg: 'ember', repo: path.join(ROOT, '1-Ember汉化插件'), fresh: path.join(__dirname, 'en_fresh/ember') },
];

const gaps = [];
const drifts = [];

for (const t of TARGETS) {
  const src = readJSON(path.join(t.fresh, '_source.json'));
  for (const p of src.packs) {
    const file = `${src.packageId}.${p.pack}.json`;
    const upRaw = readJSON(path.join(t.fresh, file));
    const cnRaw = readJSON(path.join(t.repo, 'compendium/cn', file));
    const stRaw = readJSON(path.join(t.repo, 'compendium/en', file));
    if (!upRaw) continue;

    for (const [key, val] of Object.entries(upRaw.entries ?? {})) {
      const u = leaves(val);
      const c = cnRaw ? leaves(cnRaw.entries?.[key] ?? {}) : {};
      const s = stRaw ? leaves(stRaw.entries?.[key] ?? {}) : {};
      for (const [lp, str] of Object.entries(u)) {
        if (!(typeof c[lp] === 'string' && c[lp].trim())) {
          gaps.push({ pack: `${src.packageId}.${p.pack}`, entry: key, leaf: lp, chars: str.length, en: str });
        }
        if (stRaw && s[lp] !== str) {
          drifts.push({ pack: `${src.packageId}.${p.pack}`, entry: key, leaf: lp,
            storedEn: s[lp] ?? null, freshEn: str });
        }
      }
    }
  }
}

/** 叶路径的「类别」：取最后一个非下标段，再带上倒数第二个容器段。 */
function category(leaf) {
  const segs = leaf.replace(/\[\d+\]/g, '[]').split('.');
  return segs.slice(-3).join('.');
}
const byCat = {};
for (const g of gaps) {
  const c = category(g.leaf);
  (byCat[c] ||= { n: 0, chars: 0, samples: [] });
  byCat[c].n += 1; byCat[c].chars += g.chars;
  if (byCat[c].samples.length < 5) byCat[c].samples.push(g);
}

const md = [];
md.push('# 叶级缺口完整名单（上游有文本、cn 侧该叶无译文）');
md.push('');
md.push(`总计 **${gaps.length} 叶 / ${gaps.reduce((a, g) => a + g.chars, 0)} 字符**。`);
md.push('');
md.push('## 按叶路径分类');
md.push('');
md.push('| 叶路径类别 | 叶数 | 字符 | 例 |');
md.push('|---|--:|--:|---|');
for (const [c, v] of Object.entries(byCat).sort((a, b) => b[1].n - a[1].n)) {
  md.push(`| \`${c}\` | ${v.n} | ${v.chars} | ${v.samples.slice(0, 2).map((s) => `\`${s.en.slice(0, 40).replace(/\|/g, '\\|')}\``).join(' · ')} |`);
}
md.push('');
md.push('## 逐条（全量，非抽样）');
md.push('');
md.push('| # | pack | 条目 | 叶路径 | 字符 | 英文原文 |');
md.push('|--:|---|---|---|--:|---|');
gaps.forEach((g, i) => {
  md.push(`| ${i + 1} | ${g.pack} | \`${g.entry}\` | \`${g.leaf}\` | ${g.chars} | ${g.en.replace(/\|/g, '\\|').replace(/\n/g, ' ').slice(0, 200)} |`);
});

md.push('');
md.push('# 上游漂移逐条（新抽英文基线 vs 仓里 compendium/en 快照）');
md.push('');
md.push(`共 ${drifts.length} 处。`);
md.push('');
md.push('| # | pack | 条目 | 叶路径 | 仓里快照 | 新抽真身 |');
md.push('|--:|---|---|---|---|---|');
drifts.forEach((d, i) => {
  md.push(`| ${i + 1} | ${d.pack} | \`${d.entry}\` | \`${d.leaf}\` | ${String(d.storedEn).replace(/\|/g, '\\|').replace(/\n/g, ' ').slice(0, 120)} | ${String(d.freshEn).replace(/\|/g, '\\|').replace(/\n/g, ' ').slice(0, 120)} |`);
});

fs.writeFileSync(path.join(__dirname, 'gap_detail.json'),
  `${JSON.stringify({ gaps, drifts, byCategory: byCat }, null, 2)}\n`, 'utf8');
fs.writeFileSync(path.join(__dirname, 'gap_detail.md'), `${md.join('\n')}\n`, 'utf8');
console.log(`gaps=${gaps.length} chars=${gaps.reduce((a, g) => a + g.chars, 0)}  drifts=${drifts.length}`);
console.log(md.slice(0, 40).join('\n'));
