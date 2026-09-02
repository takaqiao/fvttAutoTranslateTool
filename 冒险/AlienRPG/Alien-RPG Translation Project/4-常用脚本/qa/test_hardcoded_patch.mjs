#!/usr/bin/env node
/**
 * test_hardcoded_patch.mjs —— 运行时补丁（第三条汉化通道）的正反例测试。
 *
 *   node "4-常用脚本/qa/test_hardcoded_patch.mjs"
 *
 * ══════════════════════════════════════════════════════════════════════
 * 为什么这个测试**必须**直接 import 出货文件
 * ══════════════════════════════════════════════════════════════════════
 * 上一个项目发过一次这样的事故：一条为自己写的**子串**正则挂在全局
 * `ui.notifications.notify` 上，把别的模块的 'Mirror Image does not exist!'
 * 改成了 '镜子 Image 不存在！'。这类事故的判据只有一条 ——
 * **别人的字符串必须原样穿过我们的函数** —— 而这条判据只有拿真正出货的那份
 * 代码去跑才算数。所以本文件 `import` 的是
 *   1-系统汉化插件/scripts/alienrpg-hardcoded-cn.mjs
 * 本体（它顶层的 `HOOKS` 垫片让裸 node 也能 import），不抄任何副本。
 *
 * 三组断言：
 *   A. 出货函数的正例 —— 该翻的都翻了。
 *   B. 出货函数的反例 —— **不该动的一个字都没动**（别的模块的通知、含相同
 *      单词的普通散文、结构相似但锚点不符的文档、非字符串）。
 *   C. 与源码的一致性 —— 表里每条 `from` / `expect` 都必须能在
 *      alienrpg 4.1.13 的源码或模板里逐字节找到；顶层 i18n 键的全局冲突面
 *      必须为 0；译文模板必须存在且不含残留英文。
 *      这一组是防「上游改了而我们的表静默过期」的唯一手段。
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const PROJ = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const HUB = path.join(PROJ, '1-系统汉化插件');
const SYS = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg';
const DATA = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data';
const CORE_LANG = 'C:/Program Files/Foundry Virtual Tabletop/resources/app/public/lang';
const PATCHER = path.join(HUB, 'scripts/alienrpg-hardcoded-cn.mjs');

const mod = await import(pathToFileURL(PATCHER).href);
const { translateNotification, applyDocRules, __TEST__ } = mod;
const { CHAT_RULES, ITEM_RULES, LITERAL_LABELS, SETTINGS_MENU_RETARGET, TEMPLATE_OVERRIDES, DOM_TEXT_REPLACEMENTS, DOM_ATTR_REPLACEMENTS, DRAW_TOOLTIP_RULE, rewriteCritGlue } = __TEST__;

let pass = 0;
const failures = [];

function check(group, name, actual, expected) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) {
    pass += 1;
    console.log(`  PASS [${group}] ${name}`);
  } else {
    failures.push(`[${group}] ${name}\n        expected ${e}\n        actual   ${a}`);
    console.log(`  FAIL [${group}] ${name}\n        expected ${e}\n        actual   ${a}`);
  }
}

function read(p) {
  return fs.readFileSync(p, 'utf8');
}

/* ══════════════════════════════════════════════════════════════════════
 * A. 正例 —— 出货函数确实翻译了它该翻的
 * ══════════════════════════════════════════════════════════════════════ */
console.log('\n=== A. POSITIVE — the shipping functions translate what they must ===');

// A1-A3 通知垫片（三条都是模板字符串，通道 A 结构上够不到）
check('A', 'notify: rollItemMacro 的缺失物品提示 (alienrpg.mjs:559)',
  translateNotification('Could not find item Pulse Rifle. You may need to delete and recreate this macro.'),
  '未找到物品 Pulse Rifle。可能需要删除并重新创建这个宏。');

check('A', 'notify: ReImport 完成提示 (init.mjs:170)',
  translateNotification('Re-Import Completed Created 42 Assets'),
  '重新导入完成，共创建 42 项资源');

check('A', 'notify: logger.error 的系统标题前缀 (logger.mjs:19)',
  translateNotification('Alien RPG - Core System | ERROR | something exploded'),
  '异形 RPG - 核心系统 | 错误 | something exploded');

// A4 devmsg 的聊天署名（alias 不过 localize，只能在落库前改）
check('A', 'chat: devmsg 署名 (devmsg.js:24)',
  applyDocRules(CHAT_RULES, { speaker: { alias: 'Alien RPG News' }, content: '<p>news</p>' }),
  { 'speaker.alias': '异形 RPG 快讯' });

// A5 恐慌结束
check('A', 'chat: Panic is over (actor.mjs:1589)',
  applyDocRules(CHAT_RULES, { content: 'Panic is over', speaker: { actor: 'abc' } }),
  { content: '恐慌结束' });

// A6 先攻 flavor（双锚点：flags.core.initiativeRoll + ' <br> '）
check('A', 'chat: 先攻 flavor 带卡图 (combat.mjs:91)',
  applyDocRules(CHAT_RULES, {
    flags: { core: { initiativeRoll: true } },
    flavor: 'Ripley rolls for Initiative! <br> <div style="text-align: center;"><img src="x.png"></div>',
  }),
  { flavor: 'Ripley 掷先攻！<br> <div style="text-align: center;"><img src="x.png"></div>' });

check('A', 'chat: 先攻 flavor 无卡图 (combat.mjs:130)',
  applyDocRules(CHAT_RULES, { flags: { core: { initiativeRoll: true } }, flavor: 'Ash rolls for Initiative! <br> ' }),
  { flavor: 'Ash 掷先攻！<br> ' });

// A7 追骰按钮 title
check('A', 'chat: PUSH Roll? 按钮 title (YZEDiceRoller.mjs:389)',
  applyDocRules(CHAT_RULES, {
    content: '<button class="alien-Push-button" title="PUSH Roll?">追骰</button>',
  }),
  { content: '<button class="alien-Push-button" title="要追骰吗？">追骰</button>' });

// A8 重伤卡的中英夹缝（聊天侧）
check('A', 'chat: 重伤卡 -1 医疗夹缝 (actor.mjs:1909 -> crit-roll-character.hbs:29)',
  applyDocRules(CHAT_RULES, { content: 'BLEEDING<br> -1 to <strong>医疗</strong> roll' }),
  { content: 'BLEEDING<br> <strong>医疗</strong>检定 -1' });

// A9 重伤卡的中英夹缝（物品侧，同一个 speanex 的另一条出口）
check('A', 'item: critical-injury 的 effects 夹缝 (actor.mjs:2058)',
  applyDocRules(ITEM_RULES, {
    type: 'critical-injury',
    system: { attributes: { effects: 'CRUSHED<br> -2 to <strong>医疗</strong> roll' } },
  }),
  { 'system.attributes.effects': 'CRUSHED<br> <strong>医疗</strong>检定 -2' });

// A10 同一条消息里两处夹缝（replaceAll 语义）
check('A', 'crit glue: 一条串里两处都要改',
  rewriteCritGlue('a<br> -1 to <strong>医疗</strong> roll b<br> -2 to <strong>医疗</strong> roll'),
  'a<br> <strong>医疗</strong>检定 -1 b<br> <strong>医疗</strong>检定 -2');

/* ══════════════════════════════════════════════════════════════════════
 * B. 反例 —— 不该动的一个字都不能动
 * ══════════════════════════════════════════════════════════════════════ */
console.log('\n=== B. NEGATIVE — everything else must pass through byte-identical ===');

const MUST_NOT_MATCH = [
  // ── 别的模块 / 别的系统发的通知 ──────────────────────────────────────
  ['另一个模块：Mirror Image（上个项目的原始事故串）', 'Mirror Image does not exist!'],
  ['另一个模块：含 item 与 macro 两个词但不同句', 'Could not find item macro settings for this actor'],
  ['另一个模块：Sequencer 风格报错', 'Sequencer | Could not find item in database'],
  ['另一个模块：以 ERROR 结尾的日志式通知', 'Some Other Module | ERROR | boom'],
  ['另一个系统：dnd5e 的同名 boilerplate 之外的句子', 'Could not find item. You may need to reload.'],
  // ── 普通散文里出现相同单词 ───────────────────────────────────────────
  ['散文：讲 Re-Import 的一句话', 'The Re-Import Completed Created 42 Assets successfully, see the log.'],
  ['散文：句首多一个词', 'Warning: Could not find item X. You may need to delete and recreate this macro.'],
  ['散文：句尾少一个句号', 'Could not find item X. You may need to delete and recreate this macro'],
  ['散文：Assets 写成 Asset（单数）', 'Re-Import Completed Created 1 Asset'],
  ['散文：数量不是数字', 'Re-Import Completed Created many Assets'],
  ['散文：系统标题相近但不相同', 'Alien RPG Core System | ERROR | x'],
  ['散文：ERROR 前后空格不同', 'Alien RPG - Core System |ERROR| x'],
  // ── 上游自己已经本地化的通知（通道 A 负责，垫片不许碰） ──────────────
  ['上游已本地化：ALIENRPG.NoPanicTable 的中文值', '未找到恐慌表'],
  ['通道 A 负责的静态串必须原样穿过垫片', 'Import Complete'],
];
for (const [name, input] of MUST_NOT_MATCH) {
  check('B', `notify 原样透传 — ${name}`, translateNotification(input), input);
}

// 非字符串必须原样透传（notifications.mjs:109-110 允许传 Error 对象）
const err = new Error('boom');
check('B', 'notify: Error 对象原样透传（同一个引用）', translateNotification(err) === err, true);
check('B', 'notify: undefined 原样透传', translateNotification(undefined), undefined);
check('B', 'notify: null 原样透传', translateNotification(null), null);
check('B', 'notify: 数字原样透传', translateNotification(42), 42);

// ── 聊天规则的反例：结构锚点缺一不可 ──────────────────────────────────
const CHAT_MUST_NOT_MATCH = [
  ['别的模块的先攻消息（无 initiativeRoll flag）', { flavor: 'Ripley rolls for Initiative! <br> x' }],
  ['core 自己的先攻消息（有 flag，但没有 " <br> " 那一段）', { flags: { core: { initiativeRoll: true } }, flavor: 'Ripley rolls for Initiative!' }],
  ['别人的署名恰好含 Alien RPG 但不相等', { speaker: { alias: 'Alien RPG News Feed' }, content: 'x' }],
  ['content 含 Panic is over 但不是整串', { content: '<p>Panic is over now, breathe.</p>' }],
  ['别的模块的 push 按钮（类名不同）', { content: '<button class="push-button" title="PUSH Roll?">x</button>' }],
  ['类名对但 title 文案不同', { content: '<button class="alien-Push-button" title="Push?">x</button>' }],
  ['夹缝形状相近但数字不是 1/2', { content: 'x<br> -3 to <strong>医疗</strong> roll' }],
  ['夹缝形状相近但缺 <strong>', { content: 'x<br> -1 to 医疗 roll' }],
  ['已经翻过一遍的内容（幂等：不再命中）', { content: 'BLEEDING<br> <strong>医疗</strong>检定 -1' }],
  ['完全无关的聊天卡', { content: '<div class="dnd5e chat-card">Fireball</div>' }],
  ['空对象', {}],
];
for (const [name, data] of CHAT_MUST_NOT_MATCH) {
  check('B', `chat 不命中 — ${name}`, applyDocRules(CHAT_RULES, data), null);
}

const ITEM_MUST_NOT_MATCH = [
  ['同样的夹缝但物品类型不是 critical-injury', { type: 'talent', system: { attributes: { effects: 'x<br> -1 to <strong>医疗</strong> roll' } } }],
  ['类型对但 effects 里没有夹缝', { type: 'critical-injury', system: { attributes: { effects: '<p>流血不止</p>' } } }],
  ['类型对但 effects 不存在', { type: 'critical-injury', system: { attributes: {} } }],
  ['别的系统的物品', { type: 'weapon', name: 'Mirror Image' }],
];
for (const [name, data] of ITEM_MUST_NOT_MATCH) {
  check('B', `item 不命中 — ${name}`, applyDocRules(ITEM_RULES, data), null);
}

// @DRAW tooltip 规则的正反例（正则锚点）
const drawRe = DRAW_TOOLTIP_RULE.pattern;
check('B', 'draw tooltip: 正例命中',
  'Draw from Panic Table. <br> 按住 Shift 可加修正'.match(drawRe) !== null, true);
check('B', 'draw tooltip: 反例 — 别的模块的 tooltip 不含 ". <br> "',
  'Draw from the deck'.match(drawRe), null);
check('B', 'draw tooltip: 反例 — 句中出现 Draw from 而非句首',
  'You may Draw from Panic Table. <br> x'.match(drawRe), null);

/* ══════════════════════════════════════════════════════════════════════
 * C. 与 alienrpg 4.1.13 源码的一致性
 * ══════════════════════════════════════════════════════════════════════ */
console.log('\n=== C. SOURCE CONSISTENCY — every table entry must still exist upstream ===');

// C1 通道 A：每个英文键都必须在活跃源码里逐字节出现
const LITERAL_SITES = {
  'Import Complete': 'module/apps/init.mjs',
  'There was a problem with the Import': 'module/apps/init.mjs',
  'You can only create macro buttons for owned Items': 'module/alienrpg.mjs',
  'No submenu found for the provided key': 'module/helpers/alienRPGConfig.mjs',
  Yellow: 'module/alienrpg.mjs',
  AlienBlack: 'module/alienrpg.mjs',
  Colors: 'module/alienrpg.mjs',
  'Alien RPG - Blank': 'module/alienrpg.mjs',
  'Alien RPG - Full Dice': 'module/alienrpg.mjs',
};
check('C', 'LITERAL_LABELS 的条目数与站点表一致',
  Object.keys(LITERAL_LABELS).length, Object.keys(LITERAL_SITES).length);
for (const [literal, rel] of Object.entries(LITERAL_SITES)) {
  const src = read(path.join(SYS, rel));
  check('C', `源码里仍有字面量 ${JSON.stringify(literal)} (${rel})`, src.includes(`"${literal}"`), true);
  check('C', `LITERAL_LABELS 收了 ${JSON.stringify(literal)}`, literal in LITERAL_LABELS, true);
}

// C2 通道 A 的全局冲突面必须为 0（顶层键是跨包共享的）
function flat(obj, prefix, out) {
  for (const [k, v] of Object.entries(obj)) {
    const kk = prefix + k;
    if (v && typeof v === 'object' && !Array.isArray(v)) flat(v, kk + '.', out);
    else out[kk] = v;
  }
  return out;
}
const langFiles = [];
if (fs.existsSync(CORE_LANG)) {
  for (const f of fs.readdirSync(CORE_LANG)) if (f.endsWith('.json')) langFiles.push(path.join(CORE_LANG, f));
}
for (const kind of ['systems', 'modules']) {
  const base = path.join(DATA, kind);
  if (!fs.existsSync(base)) continue;
  for (const pkg of fs.readdirSync(base)) {
    for (const sub of ['lang', 'languages']) {
      const dir = path.join(base, pkg, sub);
      if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) continue;
      const stack = [dir];
      while (stack.length) {
        const d = stack.pop();
        for (const e of fs.readdirSync(d, { withFileTypes: true })) {
          const p = path.join(d, e.name);
          if (e.isDirectory()) stack.push(p);
          else if (e.name.endsWith('.json')) langFiles.push(p);
        }
      }
    }
  }
}
const conflicts = [];
for (const f of langFiles) {
  // 本模块自己的 lang 文件不算冲突（它归我们所有）
  if (f.replace(/\\/g, '/').includes('/modules/alienrpg-cn/')) continue;
  let j;
  try {
    j = JSON.parse(fs.readFileSync(f, 'utf8'));
  } catch {
    continue;
  }
  if (!j || typeof j !== 'object' || Array.isArray(j)) continue;
  const fl = flat(j, '', {});
  for (const key of Object.keys(LITERAL_LABELS)) {
    if (typeof j[key] === 'string' || typeof fl[key] === 'string') conflicts.push(`${key} @ ${f}`);
  }
}
console.log(`  (扫了 ${langFiles.length} 份语言 JSON)`);
check('C', '顶层 i18n 键在本机所有包里的冲突面为 0', conflicts, []);

// C3 通道 A′：三个 expect 必须与 init.mjs 里的字面量逐字节相同
const initSrc = read(path.join(SYS, 'module/apps/init.mjs'));
const retarget = SETTINGS_MENU_RETARGET['alienrpg.import'];
check('C', "registerMenu name 仍是 'Import Adventure'", initSrc.includes(`name: "${retarget.name.expect}"`), true);
check('C', "registerMenu label 仍是 'Re-Import'", initSrc.includes(`label: "${retarget.label.expect}"`), true);
// hint 是模板字符串：把 ${adventurePackName} 展开后比对
const packName = initSrc.match(/export const adventurePackName = "([^"]+)"/)?.[1];
check('C', 'adventurePackName 仍是 Alien RPG System (T-FROZEN)', packName, 'Alien RPG System');
const hintLiteral = initSrc.match(/hint: `([^`]*)`/)?.[1];
check('C', 'registerMenu hint 展开后与 expect 逐字节相同',
  hintLiteral?.replace('${adventurePackName}', packName), retarget.hint.expect);
// 三个目标键必须在系统 en.json 与 cn.json 里都存在
const sysEn = JSON.parse(read(path.join(SYS, 'lang/en.json')));
const sysCn = JSON.parse(read(path.join(SYS, 'lang/cn.json')));
for (const field of ['name', 'label', 'hint']) {
  const key = retarget[field].key.replace(/^ALIENRPG\./, '');
  check('C', `目标键 ALIENRPG.${key} 在系统 en.json 里存在`, typeof sysEn.ALIENRPG?.[key] === 'string', true);
  check('C', `目标键 ALIENRPG.${key} 在系统 cn.json 里有中文`, typeof sysCn.ALIENRPG?.[key] === 'string', true);
}
// 而且上游确实一处都没引用它们 —— 引用了就该用上游的，不该我们来接
const moduleSrc = [];
(function walk(d) {
  for (const e of fs.readdirSync(d, { withFileTypes: true })) {
    const p = path.join(d, e.name);
    if (e.isDirectory()) walk(p);
    else if (e.name.endsWith('.mjs') || e.name.endsWith('.js') || e.name.endsWith('.hbs')) moduleSrc.push(read(p));
  }
})(path.join(SYS, 'module'));
for (const f of fs.readdirSync(path.join(SYS, 'templates'), { recursive: true })) {
  const p = path.join(SYS, 'templates', String(f));
  if (fs.statSync(p).isFile()) moduleSrc.push(read(p));
}
const allSrc = moduleSrc.join('\n');
check('C', 'forceImportName/Label/Hint 在上游仍然 0 引用（所以由我们接管）',
  /forceImport(Name|Label|Hint)/.test(allSrc), false);

// C4 通道 B：抢注目标必须仍在上游存在，译文文件必须在仓里存在
for (const [upstream, ours] of Object.entries(TEMPLATE_OVERRIDES)) {
  const upstreamAbs = path.join(SYS, upstream.replace('systems/alienrpg/', ''));
  check('C', `上游模板仍存在：${upstream}`, fs.existsSync(upstreamAbs), true);
  const oursAbs = path.join(HUB, ours.replace('modules/alienrpg-cn/', ''));
  check('C', `译文模板已入库：${ours}`, fs.existsSync(oursAbs), true);
  // 上游模板仍被某个 sheet 引用（否则我们在抢注一份死模板）
  check('C', `上游仍引用 ${upstream}`, allSrc.includes(upstream), true);
}
// 译文模板不得残留 ALL-CAPS 英文标签，且必须保住上游那唯一一个 {{localize}}
const planetCn = read(path.join(HUB, 'templates/alienrpg/actor/planet-general.hbs'));
check('C', 'planet-general 译文里没有残留的全大写英文标签',
  planetCn.match(/>\s*[A-Z][A-Z /]{2,}\s*</g), null);
check('C', 'planet-general 译文保留了上游唯一的 {{localize}}',
  planetCn.includes('{{localize "ALIENRPG.Name"}}'), true);
const creatureCn = read(path.join(HUB, 'templates/alienrpg/actor/creature-header.hbs'));
for (const [lit, key] of [['Speed', 'ALIENRPG.Speed'], ['Mobility', 'ALIENRPG.Skillmobility'], ['Observation', 'ALIENRPG.Skillobservation'], ['Acid Splash', 'ALIENRPG.SkillAcidSplash']]) {
  check('C', `creature-header: data-label='${lit}' 已换成 {{localize "${key}"}}`,
    creatureCn.includes(`data-label='{{localize "${key}"}}'`) && !creatureCn.includes(`data-label='${lit}'`), true);
}
// 上游那份仍然是写死的（换句话说：这个补丁还有存在的必要）
const creatureEn = read(path.join(SYS, 'templates/actor/creature-header.hbs'));
check('C', '上游 creature-header 的 4 个 data-label 仍是写死英文',
  ["data-label='Speed'", "data-label='Mobility'", "data-label='Observation'", "data-label='Acid Splash'"].every((s) => creatureEn.includes(s)), true);

// C5 通道 C：每条 DOM 规则的 from + selector 都必须能在活跃模板里找到
const LIVE_TEMPLATES = {};
(function walkT(d) {
  for (const e of fs.readdirSync(d, { withFileTypes: true })) {
    const p = path.join(d, e.name);
    if (e.isDirectory()) walkT(p);
    else if (e.name.endsWith('.hbs') || e.name.endsWith('.html')) LIVE_TEMPLATES[p.replace(/\\/g, '/')] = read(p);
  }
})(path.join(SYS, 'templates'));
LIVE_TEMPLATES[path.join(SYS, 'module/helpers/alienprgSettings.hbs').replace(/\\/g, '/')] =
  read(path.join(SYS, 'module/helpers/alienprgSettings.hbs'));

function countInTemplates(needle) {
  let n = 0;
  for (const src of Object.values(LIVE_TEMPLATES)) {
    n += src.split(needle).length - 1;
  }
  return n;
}
// 精确计数：18 个加减档位按钮 = 9 个 Minus + 9 个 Plus（2026-08-29 逐行重新推导）
check('C', "模板里仍有 9 处 title='Minus'", countInTemplates("title='Minus'"), 9);
check('C', "模板里仍有 9 处 title='Plus'", countInTemplates("title='Plus'"), 9);
check('C', "模板里仍有 2 处 title='Stunts'", countInTemplates("title='Stunts'"), 2);
check('C', '模板里仍有 4 处 >NPC?<', countInTemplates('>NPC?<'), 4);
check('C', '模板里仍有 2 处 >SPECIALTY<', countInTemplates('>SPECIALTY<'), 2);
check('C', '模板里仍有 1 处活跃的 title="Create item"（territory-systems.hbs）',
  (LIVE_TEMPLATES[path.join(SYS, 'templates/actor/territory-systems.hbs').replace(/\\/g, '/')] || '').includes('title="Create item"'), true);
check('C', '活跃的设置模板里仍有 CRT UI Sheets',
  (LIVE_TEMPLATES[path.join(SYS, 'module/helpers/alienprgSettings.hbs').replace(/\\/g, '/')] || '').includes('CRT UI Sheets'), true);

// {MISSING_CREW} 与 No Stunts Entered 是 JS 侧字面量
check('C', '{MISSING_CREW} 仍在 spacecraft-sheet.mjs 与 vehicle-sheet.mjs 里',
  read(path.join(SYS, 'module/sheets/spacecraft-sheet.mjs')).includes('"{MISSING_CREW}"') &&
  read(path.join(SYS, 'module/sheets/vehicle-sheet.mjs')).includes('"{MISSING_CREW}"'), true);
check('C', '<h2>No Stunts Entered</h2> 的 4 个写入点仍在',
  ['character-sheet.mjs', 'synthetic-sheet.mjs', 'spacecraft-sheet.mjs', 'vehicle-sheet.mjs']
    .every((f) => read(path.join(SYS, 'module/sheets', f)).includes('chatData = "<h2>No Stunts Entered</h2>"')), true);

// C6 DOM 规则表的自洽：selector/from 齐全，attr 规则的 from 必须是字符串或带锚点的正则
for (const r of [...DOM_TEXT_REPLACEMENTS, ...DOM_ATTR_REPLACEMENTS]) {
  check('C', `DOM 规则有 selector 与 from：${r.selector} / ${r.from}`,
    typeof r.selector === 'string' && typeof r.from === 'string' && r.from.length > 0, true);
}
check('C', 'DRAW tooltip 正则两端都有锚点',
  DRAW_TOOLTIP_RULE.pattern.source.startsWith('^') && DRAW_TOOLTIP_RULE.pattern.source.endsWith('$'), true);
for (const { pattern } of __TEST__.NOTIFICATION_PATTERNS) {
  check('C', `通知正则两端都有锚点：${pattern.source.slice(0, 40)}…`,
    pattern.source.startsWith('^') && pattern.source.endsWith('$'), true);
}

// C7 不许与 lang 通道重复：LITERAL_LABELS 的中文值不得等于系统 cn.json 已有的任何译文键的值
//    （那意味着这条本该走 lang/cn.json）
const sysCnFlat = flat(sysCn, '', {});
const dupes = [];
for (const [en, raw] of Object.entries(LITERAL_LABELS)) {
  if (typeof raw !== 'string') continue;
  for (const [k, v] of Object.entries(sysCnFlat)) {
    if (v === raw && (sysEn.ALIENRPG?.[k.replace(/^ALIENRPG\./, '')] === en)) dupes.push(`${en} == ${k}`);
  }
}
check('C', '没有一条通道 A 的键其实能走 lang/cn.json', dupes, []);

/* ══════════════════════════════════════════════════════════════════════ */
console.log(`\n${'='.repeat(70)}`);
console.log(`PASS ${pass}   FAIL ${failures.length}`);
if (failures.length) {
  console.log('\nFAILURES:');
  for (const f of failures) console.log('  - ' + f);
  process.exit(1);
}
console.log('ALL GREEN');
