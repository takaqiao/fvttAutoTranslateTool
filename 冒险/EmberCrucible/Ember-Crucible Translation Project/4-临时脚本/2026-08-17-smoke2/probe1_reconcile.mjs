#!/usr/bin/env node
/**
 * probe1 —— 上游真实包内容 vs 我们译文的逐包逐键对账。
 *
 * 三方对账：
 *   UPSTREAM  = 本轮**从 LevelDB 真身**新抽的英文基线（en_fresh/，由
 *               3-常用脚本/extract/extract_en.mjs 解释 mappings.mjs 生成）
 *   STORED_EN = 仓里既有的英文基线快照（compendium/en/），只用来量**上游漂移**
 *   CN        = 我们的译文（compendium/cn/）
 *
 * ⚠ 对账的「可译字段」不是拍脑袋列的：UPSTREAM 的键集就是
 *   `effectiveMappings()` 的产物 —— 与 Babele 运行时查找的键集同源
 *   （见 mappings.mjs 文件头与 extract_en.mjs 的注释）。
 *
 * ⚠ 报「缺口 0」的前提是 probe0_selfattest.mjs 先跑绿（前置自证）。
 *
 * 用法： node probe1_reconcile.mjs
 * 产物： reconcile.json / reconcile.md
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project';

const TARGETS = [
  { pkg: 'crucible', repo: path.join(ROOT, '2-Crucible汉化插件'), fresh: path.join(__dirname, 'en_fresh/crucible') },
  { pkg: 'ember', repo: path.join(ROOT, '1-Ember汉化插件'), fresh: path.join(__dirname, 'en_fresh/ember') },
];

const readJSON = (p) => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : null);

/** 把一个 entry 摊平成 {leafPath: string}。数组下标也进路径。 */
function leaves(node, prefix = '', out = {}) {
  if (typeof node === 'string') { if (node.trim()) out[prefix] = node; return out; }
  if (Array.isArray(node)) { node.forEach((v, i) => leaves(v, `${prefix}[${i}]`, out)); return out; }
  if (node && typeof node === 'object') {
    for (const [k, v] of Object.entries(node)) leaves(v, prefix ? `${prefix}.${k}` : k, out);
  }
  return out;
}

const chars = (obj) => Object.values(obj).reduce((a, s) => a + s.length, 0);

function flattenPack(pack) {
  // 返回 {entryKey: {leafPath: str}}，外加 folders 单独一份
  const entries = {};
  for (const [k, v] of Object.entries(pack?.entries ?? {})) entries[k] = leaves(v);
  const folders = {};
  for (const [k, v] of Object.entries(pack?.folders ?? {})) if (typeof v === 'string' && v.trim()) folders[k] = v;
  return { entries, folders };
}

function diffKeys(a, b) {
  const onlyA = Object.keys(a).filter((k) => !(k in b));
  const onlyB = Object.keys(b).filter((k) => !(k in a));
  const both = Object.keys(a).filter((k) => k in b);
  return { onlyA, onlyB, both };
}

const report = { generatedAt: new Date().toISOString(), packages: [] };
const md = [];

for (const t of TARGETS) {
  const src = readJSON(path.join(t.fresh, '_source.json'));
  const cnDir = path.join(t.repo, 'compendium/cn');
  const enDir = path.join(t.repo, 'compendium/en');
  const storedSrc = readJSON(path.join(enDir, '_source.json'));

  const pkgRec = {
    packageId: src.packageId,
    upstreamVersion: src.packageVersion,
    storedBaselineVersion: storedSrc?.packageVersion ?? null,
    storedBaselineExtractedAt: storedSrc?.extractedAt ?? null,
    declaredPacks: src.declaredPacks,
    extractedPacks: src.extractedPacks,
    skippedPacks: src.skippedPacks,
    packs: [],
    totals: {},
  };

  for (const p of src.packs) {
    const file = `${src.packageId}.${p.pack}.json`;
    const up = flattenPack(readJSON(path.join(t.fresh, file)));
    const cnRaw = readJSON(path.join(cnDir, file));
    const stRaw = readJSON(path.join(enDir, file));
    const cn = flattenPack(cnRaw);
    const st = flattenPack(stRaw);

    const eKeys = diffKeys(up.entries, cn.entries);
    const fKeys = diffKeys(up.folders, cn.folders);

    // 叶级：只在共有条目内比，缺口条目的叶单独统计
    let leafUp = 0; let leafUpChars = 0;
    let leafCovered = 0; let leafMissingInCn = 0; let leafMissingChars = 0;
    let leafSameAsEnglish = 0;
    const missingLeafSamples = [];
    for (const k of Object.keys(up.entries)) {
      const u = up.entries[k]; const c = cn.entries[k] ?? {};
      for (const [lp, s] of Object.entries(u)) {
        leafUp += 1; leafUpChars += s.length;
        if (typeof c[lp] === 'string' && c[lp].trim()) {
          leafCovered += 1;
          if (c[lp] === s) leafSameAsEnglish += 1;
        } else {
          leafMissingInCn += 1; leafMissingChars += s.length;
          if (missingLeafSamples.length < 20) missingLeafSamples.push({ entry: k, leaf: lp, chars: s.length });
        }
      }
    }
    // 死叶：cn 有、上游没有
    let leafDead = 0;
    const deadLeafSamples = [];
    for (const k of Object.keys(cn.entries)) {
      const u = up.entries[k] ?? {};
      for (const lp of Object.keys(cn.entries[k])) {
        if (!(lp in u)) { leafDead += 1; if (deadLeafSamples.length < 20) deadLeafSamples.push({ entry: k, leaf: lp }); }
      }
    }

    // 上游漂移：新抽 vs 仓里既有英文基线
    const drift = (() => {
      if (!stRaw) return { storedFilePresent: false };
      const d = diffKeys(up.entries, st.entries);
      let changed = 0; const samples = [];
      for (const k of d.both) {
        const a = up.entries[k]; const b = st.entries[k];
        for (const [lp, s] of Object.entries(a)) {
          if (b[lp] !== s) { changed += 1; if (samples.length < 10) samples.push({ entry: k, leaf: lp }); }
        }
      }
      return {
        storedFilePresent: true,
        entriesAddedUpstream: d.onlyA, entriesRemovedUpstream: d.onlyB,
        leavesChanged: changed, leafChangeSamples: samples,
      };
    })();

    const rec = {
      pack: p.pack,
      documentType: p.documentType,
      upstreamDocuments: p.documents,
      upstreamEntries: Object.keys(up.entries).length,
      upstreamMergedDuplicates: p.mergedDuplicates,
      upstreamFolders: Object.keys(up.folders).length,
      upstreamLeaves: leafUp,
      upstreamLeafChars: leafUpChars,
      cnFilePresent: !!cnRaw,
      cnEntries: Object.keys(cn.entries).length,
      cnFolders: Object.keys(cn.folders).length,
      entriesCovered: eKeys.both.length,
      entriesMissingInCn: eKeys.onlyA,       // 真缺口
      entriesOrphanInCn: eKeys.onlyB,        // 孤儿
      foldersMissingInCn: fKeys.onlyA,
      foldersOrphanInCn: fKeys.onlyB,
      leafCovered,
      leafMissingInCn,
      leafMissingChars,
      leafSameAsEnglish,
      missingLeafSamples,
      leafDead,
      deadLeafSamples,
      drift,
    };
    pkgRec.packs.push(rec);
  }

  // 仓里有 cn 文件、但上游根本没有对应包的 —— 整包孤儿
  const cnFiles = fs.existsSync(cnDir) ? fs.readdirSync(cnDir).filter((f) => f.endsWith('.json')) : [];
  const known = new Set(src.packs.map((p) => `${src.packageId}.${p.pack}.json`));
  pkgRec.cnFilesWithoutUpstreamPack = cnFiles.filter((f) => !known.has(f));
  // 上游声明/抽出了包、cn 侧却没有文件的
  pkgRec.upstreamPacksWithoutCnFile = pkgRec.packs.filter((r) => !r.cnFilePresent).map((r) => r.pack);

  const sum = (f) => pkgRec.packs.reduce((a, r) => a + (typeof r[f] === 'number' ? r[f] : 0), 0);
  const sumLen = (f) => pkgRec.packs.reduce((a, r) => a + r[f].length, 0);
  pkgRec.totals = {
    upstreamDocuments: sum('upstreamDocuments'),
    upstreamEntries: sum('upstreamEntries'),
    upstreamLeaves: sum('upstreamLeaves'),
    upstreamLeafChars: sum('upstreamLeafChars'),
    cnEntries: sum('cnEntries'),
    entriesCovered: sum('entriesCovered'),
    entriesMissingInCn: sumLen('entriesMissingInCn'),
    entriesOrphanInCn: sumLen('entriesOrphanInCn'),
    leafCovered: sum('leafCovered'),
    leafMissingInCn: sum('leafMissingInCn'),
    leafMissingChars: sum('leafMissingChars'),
    leafSameAsEnglish: sum('leafSameAsEnglish'),
    leafDead: sum('leafDead'),
    foldersMissingInCn: sumLen('foldersMissingInCn'),
    foldersOrphanInCn: sumLen('foldersOrphanInCn'),
    driftEntriesAdded: pkgRec.packs.reduce((a, r) => a + (r.drift.entriesAddedUpstream?.length ?? 0), 0),
    driftEntriesRemoved: pkgRec.packs.reduce((a, r) => a + (r.drift.entriesRemovedUpstream?.length ?? 0), 0),
    driftLeavesChanged: pkgRec.packs.reduce((a, r) => a + (r.drift.leavesChanged ?? 0), 0),
  };
  report.packages.push(pkgRec);

  /* ---- markdown ---- */
  md.push(`\n## ${pkgRec.packageId} v${pkgRec.upstreamVersion}`
    + ` — 声明 ${pkgRec.declaredPacks} 包 / 抽出 ${pkgRec.extractedPacks} 包`
    + (pkgRec.skippedPacks?.length ? ` / 跳过 ${pkgRec.skippedPacks.length}` : ''));
  md.push('');
  md.push('| pack | type | 上游文档 | 上游条目 | 上游可译叶 | 上游字符 | cn 条目 | 覆盖条目 | 上游有·cn 无 | cn 有·上游无 | 缺叶 | 缺字符 | 死叶 |');
  md.push('|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|');
  for (const r of pkgRec.packs) {
    md.push(`| ${r.pack} | ${r.documentType} | ${r.upstreamDocuments} | ${r.upstreamEntries} | ${r.upstreamLeaves} | `
      + `${r.upstreamLeafChars} | ${r.cnFilePresent ? r.cnEntries : '—(无文件)'} | ${r.entriesCovered} | `
      + `${r.entriesMissingInCn.length} | ${r.entriesOrphanInCn.length} | ${r.leafMissingInCn} | ${r.leafMissingChars} | ${r.leafDead} |`);
  }
  const T = pkgRec.totals;
  md.push(`| **合计** |  | **${T.upstreamDocuments}** | **${T.upstreamEntries}** | **${T.upstreamLeaves}** | `
    + `**${T.upstreamLeafChars}** | **${T.cnEntries}** | **${T.entriesCovered}** | **${T.entriesMissingInCn}** | `
    + `**${T.entriesOrphanInCn}** | **${T.leafMissingInCn}** | **${T.leafMissingChars}** | **${T.leafDead}** |`);
  md.push('');
  md.push(`上游漂移（新抽基线 vs 仓里 compendium/en 快照 ${pkgRec.storedBaselineVersion} @ ${pkgRec.storedBaselineExtractedAt}）：`
    + `新增条目 ${T.driftEntriesAdded} · 消失条目 ${T.driftEntriesRemoved} · 叶文本变化 ${T.driftLeavesChanged}`);
  if (pkgRec.upstreamPacksWithoutCnFile.length) {
    md.push(`\n⚠ 上游有包、cn 侧无译文文件：${pkgRec.upstreamPacksWithoutCnFile.join(', ')}`);
  }
  if (pkgRec.cnFilesWithoutUpstreamPack.length) {
    md.push(`\n⚠ cn 侧有文件、上游无同名包：${pkgRec.cnFilesWithoutUpstreamPack.join(', ')}`);
  }
  for (const r of pkgRec.packs) {
    if (r.entriesMissingInCn.length) {
      md.push(`\n### 缺口 · ${pkgRec.packageId}.${r.pack} —— 上游有、cn 无，共 ${r.entriesMissingInCn.length} 条`);
      md.push(r.entriesMissingInCn.slice(0, 20).map((k) => `- \`${k}\``).join('\n'));
    }
    if (r.entriesOrphanInCn.length) {
      md.push(`\n### 孤儿 · ${pkgRec.packageId}.${r.pack} —— cn 有、上游无，共 ${r.entriesOrphanInCn.length} 条`);
      md.push(r.entriesOrphanInCn.slice(0, 20).map((k) => `- \`${k}\``).join('\n'));
    }
  }
}

fs.writeFileSync(path.join(__dirname, 'reconcile.json'), `${JSON.stringify(report, null, 2)}\n`, 'utf8');
fs.writeFileSync(path.join(__dirname, 'reconcile.md'), `${md.join('\n')}\n`, 'utf8');
console.log(md.join('\n'));
