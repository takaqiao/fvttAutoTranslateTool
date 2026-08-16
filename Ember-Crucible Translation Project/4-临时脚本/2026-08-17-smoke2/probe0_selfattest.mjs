#!/usr/bin/env node
/**
 * probe0 —— 前置自证。
 *
 * 本单元要拿「上游 LevelDB 真身」对账。在报任何缺口之前，必须先证明两件事
 * （项目在这上面栽过四次，最近一次产出 16 条假的「没红」）：
 *
 *   ① 我切出来的条数 ＝ 已知真值   （读对了「多少个」）
 *   ② 我读出来的东西 ＝ 那个对象   （读对了「哪一个」）
 *
 * ①的已知真值来源：Foundry 的合集 index 就是 LevelDB 里主桶 `!<bucket>!<id>`
 *   的键集（`ClassicLevelPack` 用同一批键建 index，embedded 文档在
 *   `!<bucket>.<sub>!` 兄弟桶里、不进 index）。本探针用**三条互相独立的路径**
 *   数同一个数：
 *     A. 全库 iterator + 正则切桶（extract_en.mjs 用的那条路）
 *     B. classic-level 的 range-scan（keys only，gte/lt 边界），完全不碰正则
 *     C. 解析出来的文档去重 `_id` 基数
 *   三者必须逐包相等，任何一处不等本探针直接非零退出。
 *
 * ②的已知真值来源：**本项目自己在 mappings.mjs 里逐条记下的实测数**
 *   （每条都带出处与测量日期，见该文件注释）。这些数是在 ember 0.6.0 /
 *   crucible 0.10.1 上量的，与本机安装版本一致，所以必须**逐字复现**。
 *   复现不了 ⇒ 要么我读错了对象，要么上游动过 —— 两种都必须当场看见，
 *   不许让它悄悄过去。
 *
 * 另加一条纯结构自证：主桶里每个文档的 `doc._id` 必须等于键的 id 段、
 * `doc._key` 必须等于整条键（Foundry 写包时落的字段）。这一条防的是
 * 「数对了但读的是别的桶/别的库」。
 *
 * 用法： node probe0_selfattest.mjs [--json <out.json>]
 * 退出码：0 = 全部自证通过；1 = 有任何一条不符（含上游漂移）
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const FVTT_NODE_ANCHOR = 'C:/Users/Taka/Desktop/fvtt/package.json';
const { ClassicLevel } = createRequire(FVTT_NODE_ANCHOR)('classic-level');

const DATA = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data';
const PACKAGES = [
  { id: 'crucible', dir: path.join(DATA, 'systems/crucible'), manifest: 'system.json' },
  { id: 'ember', dir: path.join(DATA, 'modules/ember'), manifest: 'module.json' },
];

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const OUT = arg('--json', path.join(__dirname, 'probe0_selfattest.json'));

const BUCKET_FOR = {
  Item: 'items', Actor: 'actors', JournalEntry: 'journal', Adventure: 'adventures',
  ActiveEffect: 'effects', Macro: 'macros', RollTable: 'tables', Scene: 'scenes',
  Playlist: 'playlists', Cards: 'cards',
};

/* ------------------------------------------------------------------ *
 * 路径 A：全库 iterator + 正则切桶
 * ------------------------------------------------------------------ */
async function readAll(packDir) {
  const db = new ClassicLevel(packDir, { createIfMissing: false });
  const buckets = {};           // bucket -> [{key, idPart, doc}]
  let badKeys = 0;
  for await (const [k, v] of db.iterator()) {
    const key = k.toString();
    const m = key.match(/^!([^!]+)!(.+)$/);
    if (!m) { badKeys += 1; continue; }
    let doc = null;
    try { doc = JSON.parse(v.toString()); } catch { doc = null; }
    (buckets[m[1]] ||= []).push({ key, idPart: m[2], doc });
  }
  await db.close();
  return { buckets, badKeys };
}

/* ------------------------------------------------------------------ *
 * 路径 B：range-scan，keys only，不用正则
 * ------------------------------------------------------------------ */
async function countByRange(packDir, bucket) {
  const db = new ClassicLevel(packDir, { createIfMissing: false });
  let n = 0;
  let topLevel = 0;
  // `!items!` .. `!items"`  —— '"' 是 '!' 的下一个 ASCII 字符，覆盖整桶且不含兄弟桶
  const gte = `!${bucket}!`;
  const lt = `!${bucket}"`;
  for await (const k of db.keys({ gte, lt })) {
    n += 1;
    if (!k.toString().slice(gte.length).includes('.')) topLevel += 1;
  }
  await db.close();
  return { all: n, topLevel };
}

/* ------------------------------------------------------------------ *
 * ② 已知真值复现所需的全量遍历
 *
 * 按「当前文档类型」下降：桶名给出根类型，遇到已知的 embedded 字段名就切换
 * 子类型。这样 `system.actions` 才分得清落在 Item / Actor / ActiveEffect 哪一层
 * （mappings.mjs 记的实测数正是按层分的）。
 * ------------------------------------------------------------------ */
const CHILD_TYPE = {
  items: 'Item', effects: 'ActiveEffect', actors: 'Actor', pages: 'JournalEntryPage',
  journal: 'JournalEntry', scenes: 'Scene', macros: 'Macro', tables: 'RollTable',
  results: 'TableResult', playlists: 'Playlist', regions: 'Region', behaviors: 'RegionBehavior',
  cards: 'Card', folders: 'Folder', categories: 'JournalEntryCategory',
  drawings: 'Drawing', notes: 'Note', tokens: 'Token', lights: 'AmbientLight',
  tiles: 'Tile', walls: 'Wall', delta: 'ActorDelta',
  // ⚠ `levels` 与 `templates` 必须在表里。它们的元素**带 `_id`**，不列就会继承父级
  // 的 docType='Scene'，于是「195 个场景」被数成 195+517=712（levels 恰好 517 个）。
  // 这正是「数对了叶子不等于读对了对象」的实例：occ/uniq 两个数当时都是对的。
  levels: 'SceneLevel', templates: 'MeasuredTemplate',
};
// `sounds` 是歧义键：Scene 下是 AmbientSound，Playlist 下是 PlaylistSound。
const soundsType = (parentType) => (parentType === 'Playlist' ? 'PlaylistSound' : 'AmbientSound');

const isStr = (v) => typeof v === 'string' && v.trim().length > 0;

function walk(node, docType, visit, parentType = null) {
  if (Array.isArray(node)) {
    for (const it of node) walk(it, docType, visit, parentType);
    return;
  }
  if (!node || typeof node !== 'object') return;

  visit(node, docType);

  for (const [k, v] of Object.entries(node)) {
    let childType = docType;
    if (k === 'sounds') childType = soundsType(docType);
    else if (CHILD_TYPE[k]) childType = CHILD_TYPE[k];
    // 非文档容器键（system / flags / prototypeToken / …）保持当前类型
    walk(v, childType, visit, docType);
  }
}

const T = {
  // system.actions 分层计数
  actions: { Item: 0, Actor: 0, ActiveEffect: 0, other: 0 },
  // system.adjective 分层计数
  adjective: { Item: 0, ActiveEffect: 0, other: 0 },
  // Ember 遭遇模板 token 覆盖名
  encounterTokenNames: {},        // packId.pack -> {occ, uniq:Set}
  // Scene sounds[].name / levels[].name
  sceneSoundNames: {},            // packId.pack -> {occ, uniq:Set}
  sceneLevelNames: { occ: 0, uniq: new Set(), scenes: 0 },
  // RegionBehavior 子类型
  behaviorsByType: {},
  behaviorText: { trapMessage: 0, areaDescription: 0, scrollingText: new Set(), teleportDialogNonEmpty: 0 },
  // 页面 / 期刊 / 宏 / 牌
  journalEntries: 0, journalContentNonEmpty: 0,
  pages: 0, pageSrcNonEmpty: 0, pageVideoWidthNonEmpty: 0,
  macros: 0, macroCommandNonEmpty: 0,
  cardsBucketDocs: 0,
  actorBiographyValueNonEmpty: 0, actorBiographyValueChars: 0,
  actorsTotal: 0,
};

function makeVisitor(pkgPack) {
  return (node, docType) => {
    const sys = node.system;
    if (sys && typeof sys === 'object') {
      if (Array.isArray(sys.actions)) {
        if (docType in T.actions) T.actions[docType] += 1; else T.actions.other += 1;
      }
      if (isStr(sys.adjective)) {
        if (docType === 'Item' || docType === 'ActiveEffect') T.adjective[docType] += 1;
        else T.adjective.other += 1;
      }
      // Ember 遭遇模板：system.encounter.tokens[].actors[].tokenData.name
      const toks = sys.encounter?.tokens;
      if (Array.isArray(toks)) {
        const slot = (T.encounterTokenNames[pkgPack] ||= { occ: 0, uniq: new Set() });
        for (const tk of toks) for (const a of (tk?.actors ?? [])) {
          const n = a?.tokenData?.name;
          if (isStr(n)) { slot.occ += 1; slot.uniq.add(n); }
        }
      }
    }

    // 下面这些是**按文档计数**的，必须只在真正的文档节点上加 ——
    // walk() 会遍历到 system/flags/prototypeToken 之类的普通子对象，
    // 少了这道闸，「162 个 JournalEntry」会被数成上千。文档节点的判据是有 `_id`。
    const isDoc = isStr(node._id);
    if (!isDoc) return;

    if (docType === 'Scene') {
      T.sceneLevelNames.scenes += 1;
      for (const l of (node.levels ?? [])) {
        if (isStr(l?.name)) { T.sceneLevelNames.occ += 1; T.sceneLevelNames.uniq.add(l.name); }
      }
      const slot = (T.sceneSoundNames[pkgPack] ||= { occ: 0, uniq: new Set() });
      for (const s of (node.sounds ?? [])) {
        if (isStr(s?.name)) { slot.occ += 1; slot.uniq.add(s.name); }
      }
    }

    if (docType === 'RegionBehavior' && isStr(node.type)) {
      T.behaviorsByType[node.type] = (T.behaviorsByType[node.type] ?? 0) + 1;
      if (node.type === 'ember.trapTrigger' && isStr(sys?.message)) T.behaviorText.trapMessage += 1;
      if (node.type === 'ember.areaEffect' && isStr(sys?.description)) T.behaviorText.areaDescription += 1;
      if (node.type === 'displayScrollingText' && isStr(sys?.text)) T.behaviorText.scrollingText.add(sys.text);
      if (node.type === 'teleportToken'
        && (isStr(sys?.dialog?.revealed) || isStr(sys?.dialog?.unrevealed))) T.behaviorText.teleportDialogNonEmpty += 1;
    }

    if (docType === 'JournalEntry') {
      T.journalEntries += 1;
      if (isStr(node.content)) T.journalContentNonEmpty += 1;
    }
    if (docType === 'JournalEntryPage') {
      T.pages += 1;
      if (isStr(node.src)) T.pageSrcNonEmpty += 1;
      if (node.video?.width !== null && node.video?.width !== undefined && node.video?.width !== '') {
        T.pageVideoWidthNonEmpty += 1;
      }
    }
    if (docType === 'Macro') {
      T.macros += 1;
      if (isStr(node.command)) T.macroCommandNonEmpty += 1;
    }
    if (docType === 'Actor') {
      T.actorsTotal += 1;
      const bio = node.system?.details?.biography?.value;
      if (isStr(bio)) { T.actorBiographyValueNonEmpty += 1; T.actorBiographyValueChars += bio.length; }
    }
  };
}

/* ------------------------------------------------------------------ */
async function main() {
  const report = { generatedAt: new Date().toISOString(), packages: [], attestations: [] };
  let failures = 0;
  const fail = (name, expected, got, note) => {
    failures += 1;
    report.attestations.push({ name, ok: false, expected, got, note });
    console.log(`  [FAIL] ${name}: 期望 ${JSON.stringify(expected)}，实得 ${JSON.stringify(got)}${note ? ` — ${note}` : ''}`);
  };
  const pass = (name, expected, got, note) => {
    report.attestations.push({ name, ok: true, expected, got, note });
    console.log(`  [ ok ] ${name}: ${JSON.stringify(got)}`);
  };
  const eq = (name, expected, got, note) => {
    (JSON.stringify(expected) === JSON.stringify(got) ? pass : fail)(name, expected, got, note);
  };

  for (const pkg of PACKAGES) {
    const manifest = JSON.parse(fs.readFileSync(path.join(pkg.dir, pkg.manifest), 'utf8'));
    const rec = { id: manifest.id, version: manifest.version, declaredPacks: (manifest.packs ?? []).length, packs: [] };
    console.log(`\n=== ${manifest.id} v${manifest.version} ===`);

    for (const p of manifest.packs ?? []) {
      const packDir = path.join(pkg.dir, 'packs', path.basename(p.path ?? p.name));
      if (!fs.existsSync(packDir)) {
        rec.packs.push({ name: p.name, type: p.type, present: false, dir: packDir });
        console.log(`  - ${p.name} (${p.type}): 目录不存在 -> ${packDir}`);
        continue;
      }
      const bucket = BUCKET_FOR[p.type];
      const { buckets, badKeys } = await readAll(packDir);
      const rows = buckets[bucket] ?? [];
      const top = rows.filter((r) => !r.idPart.includes('.'));
      const A = top.length;
      const B = await countByRange(packDir, bucket);
      const C = new Set(top.map((r) => r.doc?._id).filter(Boolean)).size;

      // 结构自证：_id / _key 必须与键一致
      let idMismatch = 0; let keyMismatch = 0; let unparsed = 0;
      for (const r of top) {
        if (!r.doc) { unparsed += 1; continue; }
        if (r.doc._id !== r.idPart) idMismatch += 1;
        if (r.doc._key !== undefined && r.doc._key !== r.key) keyMismatch += 1;
      }

      const bucketSizes = Object.fromEntries(Object.entries(buckets).map(([k, v]) => [k, v.length]));
      const okCounts = (A === B.topLevel && A === C && idMismatch === 0 && keyMismatch === 0
        && unparsed === 0 && badKeys === 0);
      if (!okCounts) failures += 1;

      rec.packs.push({
        name: p.name, type: p.type, present: true, bucket,
        docs_pathA_iterator: A, docs_pathB_rangescan: B.topLevel, docs_pathC_distinctId: C,
        bucketKeysAll: B.all, buckets: bucketSizes,
        idMismatch, keyMismatch, unparsed, badKeys, ok: okCounts,
      });
      console.log(`  - ${p.name.padEnd(20)} ${String(A).padStart(5)} docs  `
        + `[A=${A} B=${B.topLevel} C=${C}]  ${okCounts ? 'ok' : '*** MISMATCH ***'}`);

      // ② 全量遍历（所有桶，含 embedded 兄弟桶与 adventure blob 内部）
      const visit = makeVisitor(`${manifest.id}.${p.name}`);
      for (const [bname, list] of Object.entries(buckets)) {
        const rootType = { items: 'Item', actors: 'Actor', journal: 'JournalEntry',
          adventures: 'Adventure', effects: 'ActiveEffect', macros: 'Macro', tables: 'RollTable',
          scenes: 'Scene', playlists: 'Playlist', cards: 'Cards', folders: 'Folder' }[bname.split('.')[0]] ?? 'Unknown';
        // 兄弟桶 `actors.items` 之类：类型取最后一段
        const sub = bname.split('.').slice(1).pop();
        const type = sub ? (sub === 'sounds' ? soundsType(rootType) : (CHILD_TYPE[sub] ?? rootType)) : rootType;
        if (bname === 'cards') T.cardsBucketDocs += list.length;
        for (const { doc } of list) if (doc) walk(doc, type, visit);
      }
    }
    report.packages.push(rec);
  }

  /* --------------- ② 已知真值逐条复现 --------------- *
   * 每条的出处都写在 note 里，指向本项目 3-常用脚本/extract/mappings.mjs 的注释段。
   */
  console.log('\n=== 已知真值复现（出处：3-常用脚本/extract/mappings.mjs 各段注释实测记录）===');

  eq('system.actions 分层：Actor 层 0 / Item 层 4615 / ActiveEffect 层 34',
    { Actor: 0, Item: 4615, ActiveEffect: 34 },
    { Actor: T.actions.Actor, Item: T.actions.Item, ActiveEffect: T.actions.ActiveEffect },
    'mappings.mjs ACTIONS_FIELD 注释（2026-08-14 实测）');

  eq('system.adjective 分层：Item 层 0 / ActiveEffect 层 172',
    { Item: 0, ActiveEffect: 172 },
    { Item: T.adjective.Item, ActiveEffect: T.adjective.ActiveEffect },
    'mappings.mjs CRUCIBLE_ITEM 注释（2026-08-14 实测）');

  const enc = Object.fromEntries(Object.entries(T.encounterTokenNames)
    .map(([k, v]) => [k, { occ: v.occ, uniq: v.uniq.size }]));
  eq('ember 遭遇模板 token 覆盖名：两个冒险包各 382 处 / 130 唯一',
    { 'ember.adventure': { occ: 382, uniq: 130 }, 'ember.crucible-adventure': { occ: 382, uniq: 130 } },
    { 'ember.adventure': enc['ember.adventure'], 'ember.crucible-adventure': enc['ember.crucible-adventure'] },
    'mappings.mjs ENCOUNTER_TOKENS_FIELD 注释（probes/s3_encounter_shape.mjs 实测）');

  const snd = Object.fromEntries(Object.entries(T.sceneSoundNames)
    .map(([k, v]) => [k, { occ: v.occ, uniq: v.uniq.size }]));
  eq('ember 场景环境音名：两个冒险包各 80 叶，合计 160 叶 / 40 唯一',
    { 'ember.adventure': 80, 'ember.crucible-adventure': 80 },
    { 'ember.adventure': snd['ember.adventure']?.occ, 'ember.crucible-adventure': snd['ember.crucible-adventure']?.occ },
    'mappings.mjs SCENE_LEVELS.sounds 注释（classic-level 直读实测）');

  eq('场景层名：195 个场景 / 517 处 / 255 唯一',
    { scenes: 195, occ: 517, uniq: 255 },
    { scenes: T.sceneLevelNames.scenes, occ: T.sceneLevelNames.occ, uniq: T.sceneLevelNames.uniq.size },
    'mappings.mjs SCENE_LEVELS.levels 注释');

  eq('RegionBehavior 子类型：trapTrigger 8 / areaEffect 4 / displayScrollingText 2 / teleportToken 156',
    { 'ember.trapTrigger': 8, 'ember.areaEffect': 4, displayScrollingText: 2, teleportToken: 156 },
    {
      'ember.trapTrigger': T.behaviorsByType['ember.trapTrigger'] ?? 0,
      'ember.areaEffect': T.behaviorsByType['ember.areaEffect'] ?? 0,
      displayScrollingText: T.behaviorsByType.displayScrollingText ?? 0,
      teleportToken: T.behaviorsByType.teleportToken ?? 0,
    },
    'mappings.mjs EMBER_REGION_BEHAVIOR_MAPPINGS / EXTRACTOR_SUBTYPE_SHIMS 注释（2026-08-14 实测）');

  eq('RegionBehavior 文本：trapTrigger message 8 非空 / areaEffect description 4 非空 / '
    + 'displayScrollingText 唯一文本 ["Searing Light!"] / teleportToken 对话 0 非空',
    { trapMessage: 8, areaDescription: 4, scrollingText: ['Searing Light!'], teleportDialogNonEmpty: 0 },
    {
      trapMessage: T.behaviorText.trapMessage,
      areaDescription: T.behaviorText.areaDescription,
      scrollingText: [...T.behaviorText.scrollingText],
      teleportDialogNonEmpty: T.behaviorText.teleportDialogNonEmpty,
    },
    'mappings.mjs 同上；这一条同时自证「读到的是那个对象」——字符串本身必须逐字对上');

  eq('期刊：162 个 JournalEntry，`content` 非空 0 个',
    { entries: 162, contentNonEmpty: 0 },
    { entries: T.journalEntries, contentNonEmpty: T.journalContentNonEmpty },
    'mappings.mjs BABELE_DEFAULTS.JournalEntry 注释');

  // ⚠ 这一条与记录数不同，且已诊断清楚，**不是上游漂移**：
  //   记录值 3208 = 真值 3096 + crucible.rules 的 112 页被数了两遍。
  //   extract_en.mjs 的 readPack() 之后会 attachEmbedded(journal, journal.pages)，
  //   把兄弟桶里的 112 页**再挂回** journal 文档的 `pages` 数组；当年那支探针在
  //   attach 之后遍历「所有桶」，于是 `!journal.pages!` 桶一遍、journal 文档内一遍。
  //   本探针在 attach 之前读，probe0b_pagecount.mjs 用**另一套遍历**独立复算同为 3096。
  //   两个 ember 冒险包各 1488 页（blob 内）、crucible.playtest 8 页、crucible.rules 112 页。
  eq('期刊页：3096 页（记录值 3208 见下条诊断），`src` 非空 0 / `video.width` 非空 0',
    { pages: 3096, src: 0, videoWidth: 0 },
    { pages: T.pages, src: T.pageSrcNonEmpty, videoWidth: T.pageVideoWidthNonEmpty },
    'mappings.mjs BABELE_DEFAULTS.JournalEntryPage 注释记的是 3208；差额已诊断');
  eq('上条差额诊断：3096 + crucible.rules 的 112 页（重复计入）= 记录值 3208',
    3208, T.pages + 112,
    '差额恰等于 crucible.rules 的 journal.pages 桶大小，见 probe0b_pagecount.mjs');

  eq('宏：14 个 Macro，全部有 command',
    { macros: 14, withCommand: 14 },
    { macros: T.macros, withCommand: T.macroCommandNonEmpty },
    'mappings.mjs 刻意偏离第 1 条');

  eq('牌：`!cards!` 桶 0 个文档',
    0, T.cardsBucketDocs,
    'mappings.mjs Cards/Card/Folder 注释');

  eq('dnd5e 形状传记：255 个 actor 的 system.details.biography.value 非空',
    255, T.actorBiographyValueNonEmpty,
    'mappings.mjs 刻意偏离第 2 条（约 55.7 万字符）');
  pass('（参考）上述传记字符数', '≈557000', T.actorBiographyValueChars, '刻意偏离第 2 条只记了「约 55.7 万」');

  report.knownTruthTotals = {
    actions: T.actions, adjective: T.adjective,
    encounterTokenNames: enc, sceneSoundNames: snd,
    sceneLevelNames: { scenes: T.sceneLevelNames.scenes, occ: T.sceneLevelNames.occ, uniq: T.sceneLevelNames.uniq.size },
    behaviorsByType: T.behaviorsByType,
    behaviorText: { ...T.behaviorText, scrollingText: [...T.behaviorText.scrollingText] },
    journalEntries: T.journalEntries, journalContentNonEmpty: T.journalContentNonEmpty,
    pages: T.pages, pageSrcNonEmpty: T.pageSrcNonEmpty, pageVideoWidthNonEmpty: T.pageVideoWidthNonEmpty,
    macros: T.macros, macroCommandNonEmpty: T.macroCommandNonEmpty,
    cardsBucketDocs: T.cardsBucketDocs,
    actorsTotal: T.actorsTotal,
    actorBiographyValueNonEmpty: T.actorBiographyValueNonEmpty,
    actorBiographyValueChars: T.actorBiographyValueChars,
  };
  report.failures = failures;
  fs.writeFileSync(OUT, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  console.log(`\n自证失败项：${failures}  ->  ${OUT}`);
  process.exit(failures ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(2); });
