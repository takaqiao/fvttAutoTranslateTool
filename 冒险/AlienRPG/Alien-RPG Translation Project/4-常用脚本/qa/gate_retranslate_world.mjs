/**
 * gate_retranslate_world.mjs —— 「重新汉化本世界」修复工具的闸门。
 *
 * 用法：node "4-常用脚本/qa/gate_retranslate_world.mjs"
 *
 * 这个工具改的是**世界文档**，没有撤销键。所以它的闸门必须回答四个问题，
 * 而且每一档都要报「本次查了多少项」——「0 个问题」和「没查到东西」不是一回事。
 *
 *   G  守卫编译     DO-NOT-TRANSLATE.json 编成判据后，条目数对不对？
 *                   缺节 / 空表时会不会**拒绝运行**（而不是静默放行）？
 *   C  判据行为     正例放行、反例拦下；两个已知的**误报陷阱**必须放行。
 *   T  真实译文     用 Babele 2.9.1 的**真converter**把 raw dump 翻一遍，
 *                   拿到的就是「一次干净的重新导入会写进世界的样子」。不模拟。
 *   W  端到端       拿真实译文 + 真实英文世界跑规划 -> 应用 -> 再规划。
 *                   **幂等性就在这里证**：第二遍必须是 0 条。
 *   S  灵敏度       故意把一个冻结名译成中文，守卫**必须**拦下。
 *                   这一档是防「判据空转」的 —— 全绿而判据其实没跑，是本项目最贵的一类翻车。
 *   P  出货         包内登记表与项目登记表逐字节相同；module.json 声明了这个 esmodule。
 */

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';

const PROJ = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const HUB = path.join(PROJ, '1-系统汉化插件');
const STARTER = path.join(PROJ, '2-新手包汉化插件');
const DATA = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data';
const CORE = 'C:/Program Files/Foundry Virtual Tabletop/resources/app';
const BABELE = path.join(DATA, 'modules/babele/script');

const readJSON = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
const sha = (b) => crypto.createHash('sha256').update(b).digest('hex');

let pass = 0;
const failures = [];
const counts = {};
function check(group, name, actual, expected) {
  counts[group] = (counts[group] ?? 0) + 1;
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) { pass++; console.log(`  PASS [${group}] ${name}`); }
  else { failures.push(`[${group}] ${name}\n      expected ${e}\n      actual   ${a}`); console.log(`  FAIL [${group}] ${name}`); }
}
function checkThrows(group, name, fn, needle) {
  counts[group] = (counts[group] ?? 0) + 1;
  try {
    fn();
    failures.push(`[${group}] ${name}\n      expected throw containing ${JSON.stringify(needle)}, got none`);
    console.log(`  FAIL [${group}] ${name} (没抛)`);
  } catch (err) {
    if (String(err?.message ?? err).includes(needle)) { pass++; console.log(`  PASS [${group}] ${name}`); }
    else {
      failures.push(`[${group}] ${name}\n      expected throw containing ${JSON.stringify(needle)}\n      actual   ${String(err?.message ?? err)}`);
      console.log(`  FAIL [${group}] ${name} (抛的不是这条)`);
    }
  }
}

/* ══════════════════════════════════════════════════════════════════════
 * 被测对象
 * ══════════════════════════════════════════════════════════════════════ */
const TOOL = await import(pathToFileURL(path.join(HUB, 'scripts/retranslate-world.mjs')).href);
const { buildGuard, checkRename, indexTranslationFile, planRetranslation, hasCJK, reportToMarkdown, docTypeFromRole, COVERED_TYPES } = TOOL;

const REGISTER_SHIPPED = readJSON(path.join(HUB, 'data/DO-NOT-TRANSLATE.json'));
const REGISTER_PROJECT_BYTES = fs.readFileSync(path.join(PROJ, '7-其他内容/DO-NOT-TRANSLATE.json'));
const REGISTER_SHIPPED_BYTES = fs.readFileSync(path.join(HUB, 'data/DO-NOT-TRANSLATE.json'));

const CN_SYSTEM = readJSON(path.join(HUB, 'compendium/cn/alienrpg.alien-rpg-system.json'));
const CN_STARTER = readJSON(path.join(STARTER, 'compendium/cn/alien-evolved-starterset.alien-evolved-starter-set.json'));

const RAW_SYSTEM = readJSON(path.join(PROJ, '6-工作区/raw-dumps/system.json'));
const RAW_STARTER = readJSON(path.join(PROJ, '6-工作区/raw-dumps/starterset.json'));
const firstAdventure = (dump) => dump[Object.keys(dump)[0]];

/* ══════════════════════════════════════════════════════════════════════
 * G —— 守卫编译
 * ══════════════════════════════════════════════════════════════════════ */
console.log('\n=== G 守卫编译 ===');
const guard = buildGuard(REGISTER_SHIPPED);

check('G', 'role -> 文档类型：Adventure', docTypeFromRole('Adventure document name'), 'Adventure');
check('G', 'role -> 文档类型：JournalEntry', docTypeFromRole('JournalEntry name shown after import'), 'JournalEntry');
check('G', 'role -> 文档类型：Scene', docTypeFromRole('Scene name activated after import'), 'Scene');
check('G', 'role 认不出来时返回 null（调用方会转成「全类型冻结」+告警）', docTypeFromRole('some future thing'), null);

// name_lookups：9 条里 3 条带 unfrozen_by_shim（MU/TH/ER ×2 + STARTER SET），必须被排除。
check('G', 'T-SHIMMED 条目被排除的条数', guard.stats.shimmedSkipped, 3);
check('G', '被排除的 shim 名不在 JournalEntry 冻结集里',
  guard.frozenByType.get('JournalEntry')?.has('MU/TH/ER Instructions.') ?? false, false);
check('G', '被排除的 shim 名不在 JournalEntry 冻结集里（新手包）',
  guard.frozenByType.get('JournalEntry')?.has('STARTER SET - HOW TO USE THIS MODULE') ?? false, false);
// 但**没带** shim 的核心书那条必须还冻着（NBSP 版本）
check('G', 'corerules 的欢迎日志（NBSP 串）仍冻结',
  guard.frozenByType.get('JournalEntry')?.has('CORE RULES\u00a0-\u00a0HOW\u00a0TO\u00a0USE\u00a0THIS\u00a0MODULE') ?? false, true);

check('G', '3 个硬编码 Folder 名全部冻结',
  ['Alien Tables', 'Alien Creature Tables', 'Alien Mother Tables'].every((n) => guard.frozenByType.get('Folder')?.has(n)), true);
check('G', 'Folder 冻结集恰好 3 条', guard.frozenByType.get('Folder')?.size ?? 0, 3);
check('G', '11 个硬编码 RollTable 名全部冻结',
  (REGISTER_SHIPPED.sections.rolltable_names.entries).every((e) => guard.frozenByType.get('RollTable')?.has(e.string)), true);
check('G', 'RollTable 冻结集条数（11 条表名，无重复）', guard.frozenByType.get('RollTable')?.size ?? 0, 11);
check('G', '6 个 .toUpperCase() 天赋名全部冻结',
  ['PACK MULE', 'TAKE CONTROL', 'NERVES OF STEEL', 'TOUGH', 'HARDENED', 'STOIC'].every((n) => guard.frozenUpperByType.get('Item')?.has(n)), true);
check('G', '3 个 Adventure 名冻结',
  ['Alien RPG System', 'Alien Evolved Core Rules', 'Alien Evolved Starter Set'].every((n) => guard.frozenByType.get('Adventure')?.has(n)), true);
check('G', '2 个开场 Scene 名冻结',
  ['Alien Evolved Core Rules', 'Alien Evolved Starter Set'].every((n) => guard.frozenByType.get('Scene')?.has(n)), true);
check('G', 'RollTable 前缀过滤器（Critical Injuries）已装载',
  guard.prefixRules.filter((r) => r.docType === 'RollTable' && r.prefix === 'Critical Injuries').length, 1);
check('G', 'Item 名字子串判据（RPG）已装载', guard.substringRules.length, 1);
check('G', '子串判据的三条算子', guard.substringRules[0].tests.map((t) => `${t.op}:${t.literal}`), ['includes: RPG ', 'startsWith:RPG', 'endsWith:RPG']);
check('G', '哨兵值从登记表读出（不写死）', guard.noneSentinel, 'None');
check('G', '没有意外告警', guard.warnings, []);

// —— 守卫不完整时必须**拒绝运行** ——
for (const missing of ['name_lookups', 'rolltable_names', 'folder_names', 'item_names', 'name_substring_tests']) {
  const broken = structuredClone(REGISTER_SHIPPED);
  delete broken.sections[missing];
  checkThrows('G', `缺 sections.${missing} -> 拒绝运行`, () => buildGuard(broken), '拒绝运行');
}
checkThrows('G', '没有 sections -> 拒绝运行', () => buildGuard({}), '拒绝运行');
{
  const emptied = structuredClone(REGISTER_SHIPPED);
  for (const k of ['name_lookups', 'rolltable_names', 'folder_names']) emptied.sections[k].entries = [];
  checkThrows('G', '冻结项编译出 0 条 -> 拒绝运行（防判据空转）', () => buildGuard(emptied), '判据空转');
}

/* ══════════════════════════════════════════════════════════════════════
 * C —— 判据行为（含两个已知误报陷阱）
 * ══════════════════════════════════════════════════════════════════════ */
console.log('\n=== C 判据行为 ===');
const blockedBy = (t, f, to) => checkRename(t, f, to, guard)?.rule ?? null;

check('C', '冻结 Folder「Alien Tables」不许改', blockedBy('Folder', 'Alien Tables', '异形表格'), 'T-FROZEN/name');
check('C', '冻结 Folder「Alien Creature Tables」不许改', blockedBy('Folder', 'Alien Creature Tables', '异形生物表'), 'T-FROZEN/name');
check('C', '冻结 RollTable「Panic Table」不许改', blockedBy('RollTable', 'Panic Table', '恐慌表'), 'T-FROZEN/name');
check('C', '未冻结 Folder「Careers」可以改', blockedBy('Folder', 'Careers', '职业'), null);
check('C', '未冻结 Folder「Skill-Stunts」可以改', blockedBy('Folder', 'Skill-Stunts', '技能特技'), null);
check('C', 'Macro 名可以改', blockedBy('Macro', 'Alien -  GM Dice Roller', '异形 -  GM 投骰器'), null);

// —— 陷阱 1：同一个字符串，冻结范围只在 Adventure/Scene，**不含 Folder** ——
// 新手包里真有一个叫「Alien Evolved Starter Set」的 Folder，译文文件把它译成
// 「异形进化版新手包」。若判据不分文档类型，这次合法改名会被误挡。
check('C', '陷阱1：同名 Scene 冻结', blockedBy('Scene', 'Alien Evolved Starter Set', '异形进化版新手包'), 'T-FROZEN/name');
check('C', '陷阱1：同名 Folder **不**冻结（真实译文文件就是这么译的）',
  blockedBy('Folder', 'Alien Evolved Starter Set', '异形进化版新手包'), null);

// —— 陷阱 2：RPG 子串判据只管 Item，不管 Folder ——
// 新手包有个 Folder 叫「Alien RPG Mother Aids」，译成「异形RPG老妈辅助工具」后
// 不再满足 ' RPG ' / startsWith / endsWith 任何一条。但负重代码遍历的是**物品**，
// 文件夹名从不进那段代码。判据若不分型，这次合法改名会被误挡。
check('C', '陷阱2：Item 名丢掉 RPG 子串 -> 拦下',
  blockedBy('Item', 'M5A3 RPG Launcher', 'M5A3 火箭发射器'), 'T-FROZEN-SUBSTRING');
check('C', '陷阱2：Item 名保住 RPG 子串（双语尾）-> 放行',
  blockedBy('Item', 'M5A3 RPG Launcher', 'M5A3 火箭筒 RPG Launcher'), null);
check('C', '陷阱2：同形 Folder 名 -> 放行（真实译文文件就是这么译的）',
  blockedBy('Folder', 'Alien RPG Mother Aids', '异形RPG老妈辅助工具'), null);

check('C', '天赋名大写比较：纯中文 -> 拦下', blockedBy('Item', 'Pack Mule', '驮马'), 'T-FROZEN/uppercase');
check('C', '天赋名大写比较：双语尾也拦（大写形态变了）', blockedBy('Item', 'Pack Mule', '驮马 Pack Mule'), 'T-FROZEN/uppercase');
check('C', '天赋名大写比较：只改大小写 -> 放行（大写形态没变）', blockedBy('Item', 'Pack Mule', 'PACK MULE'), null);
check('C', '前缀过滤器：丢掉 Critical Injuries 前缀 -> 拦下',
  blockedBy('RollTable', 'Critical Injuries on Xenomorphs', '异形重伤'), 'T-FROZEN/prefix');
check('C', '前缀过滤器：保留前缀 -> 放行',
  blockedBy('RollTable', 'Critical Injuries on Xenomorphs', 'Critical Injuries 异形'), null);
check('C', '目标名为空 -> 拦下', blockedBy('Folder', 'Careers', ''), 'sanity');
check('C', 'from === to 时不判据（无变化）', checkRename('Folder', 'Alien Tables', 'Alien Tables', guard), null);

check('C', 'hasCJK：英文 false', hasCJK('<b>KEEPING IT TOGETHER:</b> you barely'), false);
check('C', 'hasCJK：中文 true', hasCJK('<b>保持镇定：</b> 你勉强'), true);
check('C', 'hasCJK：只有中文标点也算（全角冒号在 CJK 标点区外，靠汉字判）', hasCJK('ToM - 会议室'), true);
check('C', 'hasCJK：纯 ASCII 名 false', hasCJK('Alien Evolved Starter Set'), false);

/* ══════════════════════════════════════════════════════════════════════
 * T —— 用 Babele 2.9.1 的真 converter 产出「一次干净重导会写进世界的样子」
 * ══════════════════════════════════════════════════════════════════════ */
console.log('\n=== T 真实 Babele 译文 ===');

const H = await import(pathToFileURL(path.join(CORE, 'common/utils/helpers.mjs')).href);
const CollectionMod = await import(pathToFileURL(path.join(CORE, 'common/utils/collection.mjs')).href);
const CONSTS = await import(pathToFileURL(path.join(CORE, 'common/constants.mjs')).href);
const Collection = CollectionMod.default ?? CollectionMod;

globalThis.foundry = { utils: { ...H, Collection } };
globalThis.CONST = CONSTS;
globalThis.CONFIG = { debug: {} };
globalThis.Hooks = { callAll: () => true, call: () => true, once: () => {}, on: () => {} };
globalThis.fromUuidSync = () => null;

const B = pathToFileURL(BABELE).href + '/';
const { ConverterRegistry } = await import(B + 'converter/converter-registry.js');
const { IdentityExtractorRegistry } = await import(B + 'identity/identity-extractor-registry.js');
const { DocumentMappings } = await import(B + 'mapping/document-mappings.js');
const { MappedCompendiums } = await import(B + 'compendium/mapped-compendiums.js');
const { Converters } = await import(B + 'converter/converters.js');
const { DocumentConverter } = await import(B + 'converter/document-converter.js');
const { StructuredDataConverter } = await import(B + 'converter/structured-data-converter.js');
const { LookupConverter } = await import(B + 'converter/lookup-converter.js');
const { TokenizedLookupConverter } = await import(B + 'converter/tokenized-lookup-converter.js');
const { ReferencedDocumentFieldConverter } = await import(B + 'converter/referenced-document-field-converter.js');
const { DOCUMENT_MAPPINGS } = await import(pathToFileURL(path.join(HUB, 'babele-mappings.js')).href);

const PACK_META = [
  { name: 'alien-rpg-system', type: 'Adventure', label: 'Alien RPG System', packageName: 'alienrpg', packageType: 'system', id: 'alienrpg.alien-rpg-system' },
  { name: 'alien-evolved-starter-set', type: 'Adventure', label: 'Alien Evolved Starter Set', packageName: 'alien-evolved-starterset', packageType: 'module', id: 'alien-evolved-starterset.alien-evolved-starter-set' },
];

async function babeleTranslate(translationsByCollection) {
  const converterRegistry = new ConverterRegistry({
    ...Converters.legacyRegistrations(),
    document: new DocumentConverter(),
    structured: new StructuredDataConverter(),
    lookup: new LookupConverter(),
    tokenizedLookup: new TokenizedLookupConverter(),
    referencedDocumentField: new ReferencedDocumentFieldConverter(),
  });
  const identityExtractors = new IdentityExtractorRegistry(IdentityExtractorRegistry.defaultExtractors());
  identityExtractors.registerAll?.({
    range: (d) => { const [s, e] = d?.range ?? []; return Number.isInteger(s) && Number.isInteger(e) ? `${s}-${e}` : null; },
  });
  const documentMappings = new DocumentMappings(undefined, {
    registeredMappings: [DOCUMENT_MAPPINGS], identityExtractors, converterRegistry,
  });
  globalThis.game = { data: { packs: PACK_META }, packs: new Collection() };
  return new MappedCompendiums({
    documentMappings,
    translations: { for: (c) => translationsByCollection[c] ?? null },
    translationStrategies: [],
    language: 'cn',
  }).load();
}

const TRANSLATIONS = {
  'alienrpg.alien-rpg-system': CN_SYSTEM,
  'alien-evolved-starterset.alien-evolved-starter-set': CN_STARTER,
};
const mc = await babeleTranslate(TRANSLATIONS);
const TRANSLATED = {
  'alienrpg.alien-rpg-system': mc.translate('alienrpg.alien-rpg-system', structuredClone(firstAdventure(RAW_SYSTEM))),
  'alien-evolved-starterset.alien-evolved-starter-set': mc.translate('alien-evolved-starterset.alien-evolved-starter-set', structuredClone(firstAdventure(RAW_STARTER))),
};

check('T', '系统包：文件夹 Careers 被译出', (TRANSLATED['alienrpg.alien-rpg-system'].folders.find((f) => f._id === firstAdventure(RAW_SYSTEM).folders.find((x) => x.name === 'Careers')._id) ?? {}).name, '职业');
check('T', '系统包：4 个宏全部中文',
  TRANSLATED['alienrpg.alien-rpg-system'].macros.every((m) => hasCJK(m.name)), true);
check('T', '系统包：3 张表的名字仍是英文（译文文件本身就是英文，冻结项）',
  TRANSLATED['alienrpg.alien-rpg-system'].tables.map((t) => t.name).sort(),
  ['Panic Response Table', 'Panic Table', 'Stress Response Table']);
check('T', '系统包：表结果正文已中文化',
  TRANSLATED['alienrpg.alien-rpg-system'].tables.every((t) => t.results.some((r) => hasCJK(r.description))), true);
check('T', '新手包：场景「Alien Evolved Starter Set」名字未动（冻结项）',
  TRANSLATED['alien-evolved-starterset.alien-evolved-starter-set'].scenes.find((s) => s.name === 'Alien Evolved Starter Set') !== undefined, true);

/* ══════════════════════════════════════════════════════════════════════
 * W —— 端到端：英文世界 -> 规划 -> 应用 -> 再规划（幂等性证明）
 * ══════════════════════════════════════════════════════════════════════ */
console.log('\n=== W 端到端 + 幂等性 ===');

/**
 * 把 raw dump 的 Adventure 铺成「刚被非 Babele 导入」的世界。
 *
 * ⚠ **按 _id 去重，后导入的覆盖先导入的** —— 这不是图省事，是真实语义：
 *   三个包**故意共用 _id**（实测 system∩starterset 文件夹 4 / 表 2，
 *   starterset∩corerules 文件夹 11 / 表 9 / 物品 18 / 角色 4 / 场景 1），
 *   而 `Adventure#import` 对已存在的 _id 走的是 update 分支
 *   （client/documents/adventure.mjs:141-142 + :188-197），
 *   世界集合又是按 id 索引的 —— 同一个 _id 不可能有两份文档。
 *   早先的版本没去重，于是模拟世界里同 id 出现两份，「应用」时 find() 只改到第一份，
 *   第二遍规划就冒出 22 条，看起来像幂等性失败。那是**模型不对**，不是工具的错。
 */
function worldFromRaw(...advs) {
  const byType = Object.fromEntries(COVERED_TYPES.map((t) => [t, new Map()]));
  for (const adv of advs) {
    for (const f of adv.folders ?? []) byType.Folder.set(f._id, { _id: f._id, name: f.name });
    for (const m of adv.macros ?? []) byType.Macro.set(m._id, { _id: m._id, name: m.name });
    for (const t of adv.tables ?? []) byType.RollTable.set(t._id, { _id: t._id, name: t.name, results: (t.results ?? []).map((r) => ({ _id: r._id, range: r.range, name: r.name, description: r.description, documentUuid: r.documentUuid ?? null })) });
    for (const i of adv.items ?? []) byType.Item.set(i._id, { _id: i._id, name: i.name });
    for (const a of adv.actors ?? []) byType.Actor.set(a._id, { _id: a._id, name: a.name, system: { rTables: a.system?.rTables, cTables: a.system?.cTables } });
    for (const j of adv.journal ?? []) byType.JournalEntry.set(j._id, { _id: j._id, name: j.name, pages: (j.pages ?? []).map((p) => ({ _id: p._id, name: p.name, text: { content: p.text?.content } })) });
    for (const s of adv.scenes ?? []) byType.Scene.set(s._id, { _id: s._id, name: s.name });
  }
  return Object.fromEntries(Object.entries(byType).map(([k, m]) => [k, [...m.values()]]));
}

/** 造规划器要的 sources。byId 来自真实译文，byName 来自真实译文文件。 */
function sourcesFor(translations = TRANSLATIONS, translated = TRANSLATED) {
  const FIELD_TO_TYPE = { folders: 'Folder', macros: 'Macro', tables: 'RollTable', items: 'Item', actors: 'Actor', journal: 'JournalEntry', scenes: 'Scene' };
  return Object.entries(translated).map(([collection, adv]) => {
    const byNameByType = indexTranslationFile(translations[collection]).byType;
    const byType = {};
    for (const [field, docType] of Object.entries(FIELD_TO_TYPE)) {
      const byId = new Map();
      for (const d of adv[field] ?? []) if (d?._id) byId.set(d._id, d);
      byType[docType] = { byId, byName: byNameByType.get(docType) ?? new Map() };
    }
    return { key: collection, label: adv.name, packCollection: collection, translationUrl: `modules/x/compendium/cn/${collection}.json`, byType };
  });
}

/** 把待办真的写进模拟世界（对应 applyPlan 的语义）。 */
function applyToFakeWorld(world, report) {
  const find = (type, id) => (world[type] ?? []).find((d) => d._id === id);
  for (const op of report.ops) {
    const doc = find(op.docType, op.docId);
    if (!doc) throw new Error(`应用失败：${op.docType} ${op.docId} 不在模拟世界里`);
    if (op.kind === 'name') doc.name = op.to;
    else if (op.kind === 'tableRef') {
      const [, field] = op.path.split('.');
      doc.system[field] = op.to;
    } else if (op.kind === 'result') {
      const r = doc.results.find((x) => x._id === op.embeddedId);
      // ⚠ 必须按 op.path 写：TableResult 现在既修 description 也修 name
      // （带 documentUuid 的除外）。写死 description 会把 name 的待办灌进正文，
      // 第二遍立刻看起来像幂等性失败。
      if (op.path === 'name') r.name = op.to;
      else r.description = op.to;
    } else if (op.kind === 'page') {
      const p = doc.pages.find((x) => x._id === op.embeddedId);
      if (op.path === 'name') p.name = op.to;
      else p.text.content = op.to;
    } else throw new Error(`未知 op.kind: ${op.kind}`);
  }
}

const SOURCES = sourcesFor();
const world1 = worldFromRaw(firstAdventure(RAW_SYSTEM), firstAdventure(RAW_STARTER));
const scanned = Object.fromEntries(Object.entries(world1).map(([k, v]) => [k, v.length]));
console.log('  模拟世界规模：', JSON.stringify(scanned));

const r1 = planRetranslation({ sources: SOURCES, world: world1, guard });
check('W', '第一遍：有活可干（改名 > 0）', r1.stats.nameOps > 0, true);
check('W', '第一遍：有正文可修（> 0）', r1.stats.contentOps > 0, true);
check('W', '第一遍：全部按 _id 匹配上（keepId 导入的世界不该有漏网）', r1.stats.unmatched, 0);
check('W', '第一遍：按名字兜底一次也没用上（_id 全中）', r1.stats.matchedByName, 0);
check('W', '第一遍：没有被冻结判据挡住的（译文文件本身就是合规的）', r1.blocked.length, 0);
check('W', '第一遍：每条待办都带原值（可据此回退）', r1.ops.every((o) => typeof o.from === 'string'), true);
check('W', '第一遍：没有意外告警', r1.warnings, []);
console.log(`  第一遍：改名 ${r1.stats.nameOps} · 正文 ${r1.stats.contentOps} · 表名引用 ${r1.stats.refOps} · 跳过 ${r1.skipped.length}`);

applyToFakeWorld(world1, r1);

const r2 = planRetranslation({ sources: SOURCES, world: world1, guard });
check('W', '⭐ 第二遍：0 条待办（幂等）', r2.ops.length, 0);
check('W', '⭐ 第二遍：0 条被挡', r2.blocked.length, 0);
check('W', '⭐ 第二遍：0 条跳过', r2.skipped.length, 0);
applyToFakeWorld(world1, r2);
const r3 = planRetranslation({ sources: SOURCES, world: world1, guard });
check('W', '⭐ 第三遍：仍是 0 条', r3.ops.length, 0);

// 冻结名在真实数据上确实一个都没被改
const touchedNames = new Set(r1.ops.filter((o) => o.kind === 'name').map((o) => `${o.docType}|${o.from}`));
const frozenTouched = [];
for (const [type, set] of guard.frozenByType) for (const n of set) if (touchedNames.has(`${type}|${n}`)) frozenTouched.push(`${type}|${n}`);
check('W', '冻结名一个都没被改', frozenTouched, []);
check('W', '三个硬编码 Folder 名在应用后仍是英文',
  ['Alien Tables', 'Alien Creature Tables', 'Alien Mother Tables'].filter((n) => world1.Folder.some((f) => f.name === n)).length, 3);
check('W', '11 张硬编码表名里，包内存在的那几张应用后仍是英文',
  world1.RollTable.filter((t) => guard.frozenByType.get('RollTable').has(t.name)).length,
  worldFromRaw(firstAdventure(RAW_SYSTEM), firstAdventure(RAW_STARTER)).RollTable.filter((t) => guard.frozenByType.get('RollTable').has(t.name)).length);
check('W', 'Actor 的 system.rTables/cTables 一条都没动（表名全是英文，无需同步）', r1.stats.refOps, 0);
check('W', '哨兵 None 仍在 Actor 上原样保留',
  world1.Actor.filter((a) => a.system.rTables === 'None' || a.system.cTables === 'None').length > 0, true);

/* —— 场景 2：车主世界的真实混合态（日志已是中文，其余英文） ———————— */
const world2 = worldFromRaw(firstAdventure(RAW_SYSTEM));
{
  // 复刻 alienrpg.mjs showReleaseNotes() 干的事：只把那一篇 JournalEntry 换成译文版。
  const tj = TRANSLATED['alienrpg.alien-rpg-system'].journal[0];
  const wj = world2.JournalEntry.find((j) => j._id === tj._id);
  wj.name = tj.name;
  wj.pages = tj.pages.map((p) => ({ _id: p._id, name: p.name, text: { content: p.text?.content } }));
}
const r4 = planRetranslation({ sources: SOURCES, world: world2, guard });
check('W', '混合态：日志一条待办都没有（它已经是对的）',
  r4.ops.filter((o) => o.docType === 'JournalEntry').length, 0);
check('W', '混合态：文件夹 / 宏仍然有活干',
  r4.ops.filter((o) => o.kind === 'name' && ['Folder', 'Macro'].includes(o.docType)).length > 0, true);
check('W', '混合态：表结果正文仍然有活干', r4.stats.contentOps > 0, true);

/* —— 场景 3：车主自己改过的内容不许被覆盖 ————————————————————— */
const world3 = worldFromRaw(firstAdventure(RAW_SYSTEM));
{
  const t = world3.RollTable[0];
  t.results[0].description = '这是我自己写的表结果，不许动';
  const f = world3.Folder.find((x) => x.name === 'Careers');
  f.name = '我自己起的名字';
}
const r5 = planRetranslation({ sources: SOURCES, world: world3, guard });
check('W', '车主改过的表结果被跳过、不被覆盖',
  r5.ops.some((o) => o.kind === 'result' && o.embeddedId === world3.RollTable[0].results[0]._id), false);
check('W', '车主改过的表结果出现在「主动跳过」里（不是静默吞掉）',
  r5.skipped.some((s) => s.embeddedId === world3.RollTable[0].results[0]._id), true);
check('W', '车主改过的文件夹名被跳过',
  r5.ops.some((o) => o.kind === 'name' && o.from === '我自己起的名字'), false);
check('W', 'force=true 时才覆盖车主改过的名字',
  planRetranslation({ sources: SOURCES, world: world3, guard, options: { force: true } })
    .ops.some((o) => o.kind === 'name' && o.from === '我自己起的名字' && o.to === '职业'), true);

/* —— 场景 4：手工重建过的文档（_id 丢了）靠英文名兜底 ——————————— */
const world4 = worldFromRaw(firstAdventure(RAW_SYSTEM));
{
  const f = world4.Folder.find((x) => x.name === 'Skill-Stunts');
  f._id = 'RebuiltByHand0001'; // 世界里被删了重建，id 不再与包一致
}
const r6 = planRetranslation({ sources: SOURCES, world: world4, guard });
check('W', '按名字兜底能修到手工重建的文件夹',
  r6.ops.some((o) => o.kind === 'name' && o.docId === 'RebuiltByHand0001' && o.to === '技能特技' && o.match === 'name'), true);
check('W', '关掉兜底就修不到（选项确实生效，不是摆设）',
  planRetranslation({ sources: SOURCES, world: world4, guard, options: { nameFallback: false } })
    .ops.some((o) => o.docId === 'RebuiltByHand0001'), false);
check('W', 'content=false 时不产生任何正文待办',
  planRetranslation({ sources: SOURCES, world: world1, guard, options: { content: false } }).stats.contentOps, 0);

/* ══════════════════════════════════════════════════════════════════════
 * S —— 灵敏度：判据必须真的会拦
 * ══════════════════════════════════════════════════════════════════════ */
console.log('\n=== S 灵敏度（判据不是空转） ===');

/**
 * 造一份「译文文件被人翻错了」的变体，再走一遍真 Babele。
 *
 * ⚠ 两个包都要改：`Alien Tables` 这类文件夹的 _id 是**跨包共用**的，只改一个包
 *   会先撞上 ambiguous-id-match（两个包给的译名不一致 -> 不猜），冻结判据根本轮不到
 *   出场 —— 那样这一档测的就不是冻结判据了。改一致才是对冻结判据的真实施压。
 */
async function mutantRun(mutateSys, mutateStarter) {
  const cnS = structuredClone(CN_SYSTEM);
  const cnT = structuredClone(CN_STARTER);
  mutateSys?.(cnS.entries['Alien RPG System']);
  mutateStarter?.(cnT.entries['Alien Evolved Starter Set']);
  const t = { 'alienrpg.alien-rpg-system': cnS, 'alien-evolved-starterset.alien-evolved-starter-set': cnT };
  const m = await babeleTranslate(t);
  const tr = {
    'alienrpg.alien-rpg-system': m.translate('alienrpg.alien-rpg-system', structuredClone(firstAdventure(RAW_SYSTEM))),
    'alien-evolved-starterset.alien-evolved-starter-set': m.translate('alien-evolved-starterset.alien-evolved-starter-set', structuredClone(firstAdventure(RAW_STARTER))),
  };
  const w = worldFromRaw(firstAdventure(RAW_SYSTEM), firstAdventure(RAW_STARTER));
  const rep = planRetranslation({ sources: sourcesFor(t, tr), world: w, guard });
  return { rep };
}

{
  const { rep } = await mutantRun(
    (adv) => { adv.folders['Alien Tables'] = '异形表格'; },
    (adv) => { adv.folders['Alien Tables'] = '异形表格'; },
  );
  check('S', '译文把冻结 Folder 名翻了 -> 被拦下（不是被应用）',
    rep.blocked.some((b) => b.docType === 'Folder' && b.from === 'Alien Tables' && b.rule === 'T-FROZEN/name'), true);
  check('S', '……而且确实没有对应的待办', rep.ops.some((o) => o.from === 'Alien Tables'), false);
}
{
  // 只改一个包：跨包共用 _id 的文件夹会得到两个不同答案 -> 必须记 ambiguous-id-match，不猜。
  // （`Alien Tables` 的 _id Btepu5tifRV0Pj7w 在 system / starterset / corerules 三个包里都是同一个。）
  const { rep } = await mutantRun((adv) => { adv.folders['Alien Tables'] = '异形表格'; }, null);
  check('S', '跨包共用 _id 但译名分歧 -> ambiguous-id-match（不猜）',
    rep.blocked.some((b) => b.docType === 'Folder' && b.from === 'Alien Tables' && b.rule === 'ambiguous-id-match'), true);
  check('S', '……分歧时确实没有对应的待办', rep.ops.some((o) => o.from === 'Alien Tables'), false);
}
{
  // Panic Table 只在系统包里（新手包没有这张表），所以不存在跨包分歧，冻结判据直接生效。
  const { rep } = await mutantRun((adv) => { adv.tables['Panic Table'].name = '恐慌表'; }, null);
  check('S', '译文把冻结 RollTable 名翻了 -> 被拦下',
    rep.blocked.some((b) => b.docType === 'RollTable' && b.from === 'Panic Table' && b.rule === 'T-FROZEN/name'), true);
  check('S', '……而且确实没有对应的待办', rep.ops.some((o) => o.from === 'Panic Table'), false);
}
{
  // 系统包里没有天赋物品，用「按名字兜底」这条路来验大写判据的活性：
  // 世界里放一个叫 Pack Mule 的 Item，译文文件里给它一个中文名。
  const cn = structuredClone(CN_SYSTEM);
  cn.entries['Alien RPG System'].items['Pack Mule'] = { name: '驮马' };
  const t = { ...TRANSLATIONS, 'alienrpg.alien-rpg-system': cn };
  const w = worldFromRaw(firstAdventure(RAW_SYSTEM));
  w.Item.push({ _id: 'FakePackMule0001', name: 'Pack Mule' });
  const rep = planRetranslation({ sources: sourcesFor(t, TRANSLATED), world: w, guard });
  check('S', '译文把 .toUpperCase() 天赋名翻了 -> 被拦下',
    rep.blocked.some((b) => b.docType === 'Item' && b.from === 'Pack Mule' && b.rule === 'T-FROZEN/uppercase'), true);
}
{
  // 表名改了但没管 Actor 引用 —— 引用同步必须自己冒出来。
  const cn = structuredClone(CN_STARTER);
  cn.entries['Alien Evolved Starter Set'].tables['EV - Chestburster Attacks'].name = '破胸者攻击';
  const t = { ...TRANSLATIONS, 'alien-evolved-starterset.alien-evolved-starter-set': cn };
  const m = await babeleTranslate(t);
  const tr = { ...TRANSLATED, 'alien-evolved-starterset.alien-evolved-starter-set': m.translate('alien-evolved-starterset.alien-evolved-starter-set', structuredClone(firstAdventure(RAW_STARTER))) };
  const w = worldFromRaw(firstAdventure(RAW_STARTER));
  const rep = planRetranslation({ sources: sourcesFor(t, tr), world: w, guard });
  check('S', '表名一动，指向它的 Actor.system.rTables 同步跟上',
    rep.ops.filter((o) => o.kind === 'tableRef' && o.from === 'EV - Chestburster Attacks' && o.to === '破胸者攻击').length > 0, true);
  check('S', '哨兵 None 绝不跟着动', rep.ops.some((o) => o.kind === 'tableRef' && o.from === 'None'), false);
  // 同步之后再跑一遍应该收敛
  applyToFakeWorld(w, rep);
  check('S', '同步后再规划 -> 0 条（引用修复也是幂等的）',
    planRetranslation({ sources: sourcesFor(t, tr), world: w, guard }).ops.length, 0);
}
{
  // 同一个英文名在两个包里译成不同中文，且世界里那份文档没有 _id 对应 -> 必须拒绝猜。
  const cnA = structuredClone(CN_SYSTEM);
  const cnB = structuredClone(CN_STARTER);
  cnA.entries['Alien RPG System'].folders['Shared Folder Name'] = '甲译法';
  cnB.entries['Alien Evolved Starter Set'].folders['Shared Folder Name'] = '乙译法';
  const t = { 'alienrpg.alien-rpg-system': cnA, 'alien-evolved-starterset.alien-evolved-starter-set': cnB };
  const w = worldFromRaw(firstAdventure(RAW_SYSTEM));
  w.Folder.push({ _id: 'AmbiguousFolder01', name: 'Shared Folder Name' });
  const rep = planRetranslation({ sources: sourcesFor(t, TRANSLATED), world: w, guard });
  check('S', '译名歧义时拒绝猜，改记进 blocked',
    rep.blocked.some((b) => b.docId === 'AmbiguousFolder01' && b.rule === 'ambiguous-name-match'), true);
  check('S', '……而且没有对应待办', rep.ops.some((o) => o.docId === 'AmbiguousFolder01'), false);
}
{
  // 改名后撞名必须留告警（Folder 尤其要紧：getName 只返回第一个）
  const cn = structuredClone(CN_SYSTEM);
  cn.entries['Alien RPG System'].folders['Careers'] = '重名';
  cn.entries['Alien RPG System'].folders['Skill-Stunts'] = '重名';
  const t = { ...TRANSLATIONS, 'alienrpg.alien-rpg-system': cn };
  const m = await babeleTranslate(t);
  const tr = { ...TRANSLATED, 'alienrpg.alien-rpg-system': m.translate('alienrpg.alien-rpg-system', structuredClone(firstAdventure(RAW_SYSTEM))) };
  const rep = planRetranslation({ sources: sourcesFor(t, tr), world: worldFromRaw(firstAdventure(RAW_SYSTEM)), guard });
  check('S', '改名后撞名会留告警', rep.warnings.some((w) => w.includes('同叫')), true);
}
{
  // 译文文件缺失时必须明说「按名字兜底不可用」，不能假装没事
  const srcs = sourcesFor().map((s) => ({ ...s, translationUrl: null }));
  const rep = planRetranslation({ sources: srcs, world: worldFromRaw(firstAdventure(RAW_SYSTEM)), guard });
  check('S', '译文文件缺失 -> 明确告警（不装作没事）', rep.warnings.some((w) => w.includes('没找到对应的 compendium/cn 译文文件')), true);
}

/* ══════════════════════════════════════════════════════════════════════
 * R —— 真实三包环境的回归（2026-08-30 对抗复核新增）
 *
 * 前面的 W 档只喂了 system + starterset 两个包，于是它测出来的是
 * 「301 条待办 / 0 条被挡 / 0 条跳过」。但真实世界里**装着三个** Adventure 包：
 * alien-evolved-corerules 也在（modules/alien-evolved-corerules/module.json
 * 声明 pack `alien-evolved-core-rules` type Adventure），而
 * 3-核心书汉化插件/compendium/cn/ 目前**是空的** —— 它没有译文。
 *
 * 三个包**故意共用 _id**，所以一个没有译文的包会对每一个共用 _id 投出
 * 「英文原名」这一票，`agreeOn` 把它当成分歧就整类拒绝改名。
 * 实测（未修前）：172 条待办 / 32 条被挡 / 95 条跳过 —— 比应有的少修 129 条。
 *
 * 本档钉死修复后的语义，并把「没有译文的包不投票」这条判据做**灵敏度**验证：
 * 一旦有人把 voting 过滤去掉，R 档立刻变红。
 * ══════════════════════════════════════════════════════════════════════ */
console.log('\n=== R 三包真实环境 ===');

const RAW_CORE_PATH = path.join(PROJ, '6-工作区/raw-dumps/corerules.json');
if (!fs.existsSync(RAW_CORE_PATH)) {
  check('R', `raw dump 存在：${RAW_CORE_PATH}`, false, true);
} else {
  const RAW_CORE = readJSON(RAW_CORE_PATH);
  const CORE_COLL = 'alien-evolved-corerules.alien-evolved-core-rules';
  // 核心书没有译文文件 —— 这不是疏漏，是当前的真实状态，闸门要照它测。
  const CORE_CN_DIR = path.join(PROJ, '3-核心书汉化插件/compendium/cn');
  const coreHasCn = fs.existsSync(CORE_CN_DIR)
    && fs.readdirSync(CORE_CN_DIR).some((f) => f.endsWith('.json'));
  check('R', '核心书目前没有译文文件（本档的前提；有了就该把它接进 TRANSLATIONS）', coreHasCn, false);

  // 没有译文 -> mc.translate 原样返回，正是 Babele 的真实行为。
  const TRANSLATED_CORE = mc.translate(CORE_COLL, structuredClone(firstAdventure(RAW_CORE)));
  check('R', '未翻译的包读出来仍是英文（Babele 真实行为）',
    TRANSLATED_CORE.folders.some((f) => f.name === 'Alien Sub-Tables'), true);

  const FIELD_TO_TYPE_R = { folders: 'Folder', macros: 'Macro', tables: 'RollTable', items: 'Item', actors: 'Actor', journal: 'JournalEntry', scenes: 'Scene' };
  const sourceOf = (collection, adv, cnJson, babeleTranslated) => {
    const byNameByType = cnJson ? indexTranslationFile(cnJson).byType : new Map();
    const byType = {};
    for (const [field, docType] of Object.entries(FIELD_TO_TYPE_R)) {
      const byId = new Map();
      for (const d of adv[field] ?? []) if (d?._id) byId.set(d._id, d);
      byType[docType] = { byId, byName: byNameByType.get(docType) ?? new Map() };
    }
    return { key: collection, label: adv.name, packCollection: collection,
      translationUrl: cnJson ? `modules/x/compendium/cn/${collection}.json` : null,
      babeleTranslated, byType };
  };

  const S_SYS = () => sourceOf('alienrpg.alien-rpg-system', TRANSLATED['alienrpg.alien-rpg-system'], CN_SYSTEM, true);
  const S_STA = () => sourceOf('alien-evolved-starterset.alien-evolved-starter-set', TRANSLATED['alien-evolved-starterset.alien-evolved-starter-set'], CN_STARTER, true);
  const S_COR = (flag) => sourceOf(CORE_COLL, TRANSLATED_CORE, null, flag);

  const world3 = () => worldFromRaw(
    firstAdventure(RAW_SYSTEM), firstAdventure(RAW_STARTER), firstAdventure(RAW_CORE));

  // ── 灵敏度：没有译文的包若拿到投票权，会把正确译名整片挡掉 ──────────────
  const wVote = world3();
  const rVote = planRetranslation({ sources: [S_SYS(), S_STA(), S_COR(null)], world: wVote, guard });
  check('R', '⭐ 灵敏度：让无译文的包参与判定 -> 大量待办被「分歧」挡掉', rVote.blocked.length > 0, true);
  check('R', '⭐ 灵敏度：被挡的判据就是 ambiguous-id-match',
    rVote.blocked.every((b) => b.rule === 'ambiguous-id-match'), true);
  check('R', '⭐ 灵敏度：被挡的「分歧」里有一半是英文原名（说明那不是分歧，是弃权）',
    rVote.blocked.some((b) => String(b.to).includes('Alien Sub-Tables')), true);

  const wMute = world3();
  const SRC3 = [S_SYS(), S_STA(), S_COR(false)];
  const rMute = planRetranslation({ sources: SRC3, world: wMute, guard });
  check('R', '⭐ 无译文的包不投票 -> 0 条被挡', rMute.blocked.length, 0);
  check('R', '⭐ 无译文的包不投票 -> 待办数不少于两包环境', rMute.ops.length >= r1.ops.length, true);
  check('R', '被静音的包在报告里明说了', rMute.warnings.some((w) => w.includes(CORE_COLL)), true);
  check('R', 'votingPacks 统计正确', rMute.stats.votingPacks, 2);
  console.log(`  三包环境：改名 ${rMute.stats.nameOps} · 正文 ${rMute.stats.contentOps} · 被挡 ${rMute.blocked.length} · 跳过 ${rMute.skipped.length}`);

  // ── 三包环境下的幂等性（两包环境证不出来）──────────────────────────────
  applyToFakeWorld(wMute, rMute);
  const rMute2 = planRetranslation({ sources: SRC3, world: wMute, guard });
  check('R', '⭐ 三包环境第二遍：0 条待办', rMute2.ops.length, 0);
  applyToFakeWorld(wMute, rMute2);
  const rMute3 = planRetranslation({ sources: SRC3, world: wMute, guard });
  check('R', '⭐ 三包环境第三遍：仍是 0 条', rMute3.ops.length, 0);

  // ── 车主的症状：文件夹与宏修好，3 张表修不了（译文文件里就是英文）────────
  const nameOpFor = (t, n) => rMute.ops.find((o) => o.kind === 'name' && o.docType === t && o.from === n);
  check('R', '车主症状：Careers -> 职业', nameOpFor('Folder', 'Careers')?.to, '职业');
  check('R', '车主症状：Skill-Stunts -> 技能特技', nameOpFor('Folder', 'Skill-Stunts')?.to, '技能特技');
  check('R', '车主症状：4 个宏全部改名',
    ['Alien - Player Ad-hoc YZE Dice Roller', 'Alien -  GM Dice Roller',
      'Alien - Roll on selected Creature table V10', 'Alien - Roll on selected Mother table V10']
      .filter((n) => nameOpFor('Macro', n)).length, 4);
  check('R', '⭐ 车主症状：3 张表**修不了**（译文文件里的 name 本身就是英文，且是冻结项）',
    ['Panic Table', 'Panic Response Table', 'Stress Response Table'].map((n) => !!nameOpFor('RollTable', n)),
    [false, false, false]);

  // ── 血溅半径：别的模块 / 车主自建的同名文档不能被改 ──────────────────────
  const wForeign = world3();
  const FOREIGN = [
    ['Folder', 'fgnAAAAAAAAAAA1', 'Careers'], ['Folder', 'fgnAAAAAAAAAAA2', 'Weapons'],
    ['Folder', 'fgnAAAAAAAAAAA3', 'Creatures'], ['Item', 'fgnAAAAAAAAAAA4', 'Combat Knife'],
    ['JournalEntry', 'fgnAAAAAAAAAAA5', 'MU/TH/ER Instructions.'],
  ];
  for (const [t, id, name] of FOREIGN) {
    const row = { _id: id, name };
    if (t === 'RollTable') row.results = [];
    if (t === 'JournalEntry') row.pages = [];
    if (t === 'Actor') row.system = {};
    wForeign[t].push(row);
  }
  const rForeign = planRetranslation({ sources: SRC3, world: wForeign, guard });
  check('R', '⭐ 别处来的同名文档一个都没被改名',
    rForeign.ops.filter((o) => String(o.docId).startsWith('fgn')).length, 0);
  check('R', '而且逐条写进了「主动跳过」，不是静默无视',
    rForeign.skipped.filter((s) => String(s.docId).startsWith('fgn')).length, FOREIGN.length);

  // ── 但按名字兜底本身不能因此变成摆设 ────────────────────────────────────
  const wRebuilt = world3();
  wRebuilt.Folder = wRebuilt.Folder.filter((d) => d.name !== 'Careers');   // 包里那个被删了
  wRebuilt.Folder.push({ _id: 'rebuiltAAAAAAA1', name: 'Careers' });        // 车主手工重建
  const rRebuilt = planRetranslation({ sources: SRC3, world: wRebuilt, guard });
  check('R', '⭐ 手工重建（无 _id 对应）的文档仍能按名字修好 —— 兜底没被改成摆设',
    rRebuilt.ops.find((o) => o.docId === 'rebuiltAAAAAAA1')?.to, '职业');

  // ── 唯一的破坏性路径：Babele 没生效时 force 会把世界刷回英文 ──────────────
  // 造一个「已经汉化好」的世界，再喂**英文**包（= Babele 关着时读到的东西）。
  const wCn = worldFromRaw(
    TRANSLATED['alienrpg.alien-rpg-system'],
    TRANSLATED['alien-evolved-starterset.alien-evolved-starter-set']);
  const enSources = [
    sourceOf('alienrpg.alien-rpg-system', firstAdventure(RAW_SYSTEM), null, false),
    sourceOf('alien-evolved-starterset.alien-evolved-starter-set', firstAdventure(RAW_STARTER), null, false),
  ];
  const rRevert = planRetranslation({ sources: enSources, world: wCn, guard, options: { force: true } });
  check('R', '⭐ 全部包都没译文生效时，即使勾了 force 也产不出任何待办', rRevert.ops.length, 0);
  check('R', '而且报告里明说了「无从查起」', rRevert.warnings.some((w) => w.includes('无从查起')), true);
  // 灵敏度：把 babeleTranslated 抹掉（回到修复前的「不知道就放行」），必须能造出回退
  const rRevertOld = planRetranslation({
    sources: enSources.map((s) => ({ ...s, babeleTranslated: null })), world: wCn, guard, options: { force: true } });
  check('R', '⭐ 灵敏度：若不静音则确实会把中文名刷回英文（证明上一条不是空转）',
    rRevertOld.ops.filter((o) => o.kind === 'name' && hasCJK(o.from) && !hasCJK(o.to)).length > 0, true);
}

// ── 源码钉：isTranslated 收的是合集 id 字符串，不是 pack 对象 ────────────────
// babele/script/babele.js:552-559 `@param {string} pack compendium name (ex. dnd5e.classes)`
// -> mapped-compendiums.js:49-51 `get(pack) { return this.packs.get(pack) ?? null; }`（Collection 即 Map）
// 传对象进去必然 undefined，于是**每个包都报 false**：修复前那句「没有译文生效」
// 在完全健康的世界里每次都弹；修复后本工具拿它当拒绝运行的依据，传错即全禁。
{
  // 只看**代码行**：注释里逐字引用了 Babele 的原文 `isTranslated?.(pack)`，
  // 不剥注释的话这条钉会被自己的解释文字咬到。
  const codeLines = fs.readFileSync(path.join(HUB, 'scripts/retranslate-world.mjs'), 'utf8')
    .split('\n').filter((l) => !/^\s*(\/\/|\*|\/\*)/.test(l));
  const src = codeLines.join('\n');
  check('R', '⭐ isTranslated 传的是合集 id 字符串', /isTranslated\?\.\(collection\)/.test(src), true);
  check('R', '⭐ 没有再把 pack 对象传给 isTranslated', /isTranslated\?\.\(pack\)/.test(src), false);
  const babeleSrc = fs.readFileSync(path.join(BABELE, 'babele.js'), 'utf8');
  check('R', 'Babele 侧的契约仍是「compendium name」（上游一改就该重看这条）',
    /@param \{string\} pack compendium name/.test(babeleSrc), true);
}

// ── TableResult.name：本地译文优先，只有带 documentUuid 的才是动态值 ──────────
// referenced-document-field-converter.js:15-17 第一优先级就是本地译文：
//   `if (typeof context.translation !== "undefined" && context.translation !== null)
//      { return context.translation; }`
// 所以「name 是现取的、写死会钉成快照」只对带 documentUuid 的结果成立。
// 一刀切不修 name 会白白漏掉一批已经译好的结果名。
{
  const conv = fs.readFileSync(path.join(BABELE, 'converter/referenced-document-field-converter.js'), 'utf8');
  check('R', '⭐ referencedDocumentField 的第一优先级是本地译文（改了就得重看 name 判据）',
    /if \(typeof context\.translation !== "undefined" && context\.translation !== null\)/.test(conv), true);

  const wRes = worldFromRaw(firstAdventure(RAW_SYSTEM), firstAdventure(RAW_STARTER));
  const rRes = planRetranslation({ sources: SOURCES, world: wRes, guard });
  const nameOps = rRes.ops.filter((o) => o.kind === 'result' && o.path === 'name');
  check('R', '⭐ 不带 documentUuid 的结果名会被修（不是一刀切放弃）', nameOps.length > 0, true);
  check('R', '⭐ 修的每一条结果名都不带 documentUuid', nameOps.every((o) => {
    const t = wRes.RollTable.find((x) => x._id === o.docId);
    const r = t?.results?.find((x) => x._id === o.embeddedId);
    return !r?.documentUuid;
  }), true);
  // 灵敏度：给一条结果安上 documentUuid，它的 name 就必须退出待办
  const victim = nameOps[0];
  const wDyn = worldFromRaw(firstAdventure(RAW_SYSTEM), firstAdventure(RAW_STARTER));
  const vt = wDyn.RollTable.find((x) => x._id === victim.docId);
  vt.results.find((x) => x._id === victim.embeddedId).documentUuid = 'Compendium.foo.bar.RollTable.abcdefghijklmnop';
  const rDyn = planRetranslation({ sources: SOURCES, world: wDyn, guard });
  check('R', '⭐ 灵敏度：结果一旦带上 documentUuid，它的 name 立刻退出待办（动态值不钉快照）',
    rDyn.ops.some((o) => o.kind === 'result' && o.path === 'name' && o.embeddedId === victim.embeddedId), false);
  check('R', '而同一条结果的 description 不受影响',
    rDyn.ops.some((o) => o.kind === 'result' && o.path === 'description' && o.embeddedId === victim.embeddedId),
    rRes.ops.some((o) => o.kind === 'result' && o.path === 'description' && o.embeddedId === victim.embeddedId));
}

/* ══════════════════════════════════════════════════════════════════════
 * P —— 出货
 * ══════════════════════════════════════════════════════════════════════ */
console.log('\n=== P 出货 ===');
check('P', '包内登记表与项目登记表逐字节相同', sha(REGISTER_SHIPPED_BYTES), sha(REGISTER_PROJECT_BYTES));
const MODULE_JSON = readJSON(path.join(HUB, 'module.json'));
check('P', 'module.json 声明了 scripts/retranslate-world.mjs',
  (MODULE_JSON.esmodules ?? []).includes('scripts/retranslate-world.mjs'), true);
check('P', '声明的 esmodule 都存在',
  (MODULE_JSON.esmodules ?? []).filter((f) => !fs.existsSync(path.join(HUB, f))), []);
check('P', 'data/DO-NOT-TRANSLATE.json 存在（fetch 不到就拒绝运行，等于工具废掉）',
  fs.existsSync(path.join(HUB, 'data/DO-NOT-TRANSLATE.json')), true);
check('P', '工具里 REGISTER_URL 指向包内那份', TOOL.REGISTER_URL, 'modules/alienrpg-cn/data/DO-NOT-TRANSLATE.json');
check('P', '覆盖的七种文档类型', [...COVERED_TYPES], ['Folder', 'Macro', 'RollTable', 'Item', 'Actor', 'JournalEntry', 'Scene']);
{
  const md = reportToMarkdown(r1);
  check('P', 'Markdown 报告含变更清单表头', md.includes('| 类型 | 文档 | 位置 | 原 | 新 | 匹配 |'), true);
  check('P', 'Markdown 报告行数 >= 待办数', md.split('\n').length > r1.ops.length, true);
}
{
  // 执行失败必须出现在报告里 —— 半写成功（表名改了、引用没跟上）是最危险的状态。
  const withFailure = { ...r1, failed: [{ docType: 'RollTable', docName: 'X', path: 'name', error: 'boom' }] };
  const md = reportToMarkdown(withFailure);
  check('P', 'Markdown 报告会列出执行失败', md.includes('## 执行失败') && md.includes('boom'), true);
}
{
  // 拒绝运行的两条硬闸，端到端验一次（不是只验 buildGuard）：
  //   1) 非 GM -> 抛
  //   2) 登记表 fetch 失败 -> 抛，且**没有**降级成无守卫地跑
  const savedGame = globalThis.game;
  const savedFetch = globalThis.foundry.utils.fetchJsonWithTimeout;
  const savedUi = globalThis.ui;
  globalThis.ui = { notifications: { warn() {}, info() {}, error() {} } };

  globalThis.game = { user: { isGM: false }, modules: { get: () => null }, packs: [], collections: { get: () => null } };
  let threw = null;
  try { await TOOL.retranslateWorld(); } catch (e) { threw = String(e.message ?? e); }
  check('P', '非 GM -> 拒绝运行', threw !== null && threw.includes('只有 GM'), true);

  globalThis.game = { user: { isGM: true }, modules: { get: () => null }, packs: [], collections: { get: () => null } };
  globalThis.foundry.utils.fetchJsonWithTimeout = async () => { throw new Error('404 Not Found'); };
  threw = null;
  try { await TOOL.retranslateWorld(); } catch (e) { threw = String(e.message ?? e); }
  check('P', '登记表读不到 -> 拒绝运行（不降级为无守卫）', threw !== null && threw.includes('404'), true);

  globalThis.foundry.utils.fetchJsonWithTimeout = savedFetch;
  globalThis.game = savedGame;
  globalThis.ui = savedUi;
}

/* ══════════════════════════════════════════════════════════════════════ */
console.log('\n────────────────────────────────────────────');
console.log('每档检查数：', JSON.stringify(counts));
console.log(`通过 ${pass} / ${pass + failures.length}`);
if (failures.length) {
  console.log('\n失败明细：');
  for (const f of failures) console.log('  ' + f);
  process.exit(1);
}
console.log('全部通过。');
