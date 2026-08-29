/**
 * adversarial_hardcoded_patch.mjs —— 运行时补丁的**对抗式**闸门。
 *
 * 与 test_hardcoded_patch.mjs 分工明确，别合并：
 *   · test_hardcoded_patch.mjs 问的是「我们声称要改的东西，改对了吗」（正例 + 自洽）。
 *   · 本文件问的是「我们**不该**碰的东西，会不会被碰到」，以及「我们赖以成立的**机制**，
 *     在当前这台机器上的 Foundry / 系统源码里，今天还成立吗」。
 *
 * 四组：
 *   N —— 爆炸半径（negative）。判据不是"我觉得不会撞"，是**在本机真实世界里数**：
 *        295 个模块 + 全部 system + core，逐个找"谁还能产生这个形状"。
 *   M —— 机制（mechanism）。抢注短路、编译选项、钩子存在性与触发顺序，全部对着
 *        本机安装的 Foundry 源码逐行复核，不靠记忆也不靠文档。
 *   D —— 模板漂移（drift）。抢注是整份替换，上游改一个字我们就静默过期。
 *        UPSTREAM-TEMPLATE-PINS.json 把上游钉死，这里机械比对。
 *   F —— 保险丝（fuse）。今天够不到但**上游一改就会变成活面**的东西。
 *        它们现在不写规则（给不存在的界面写规则只会烂掉），改由闸门盯着。
 *
 * 用法：node "4-常用脚本/qa/adversarial_hardcoded_patch.mjs"
 */

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';

const PROJ = 'C:/Users/Taka/Desktop/fvtt/Alien-RPG Translation Project';
const HUB = path.join(PROJ, '1-系统汉化插件');
const DATA = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data';
const SYS = path.join(DATA, 'systems/alienrpg');
const CORE = 'C:/Program Files/Foundry Virtual Tabletop/resources/app';

const patcher = await import(pathToFileURL(path.join(HUB, 'scripts/alienrpg-hardcoded-cn.mjs')).href);
const plugins = await import(pathToFileURL(path.join(HUB, 'scripts/plugins-hardcoded-cn.mjs')).href);
const { translateNotification, applyDocRules, __TEST__ } = patcher;
const { LITERAL_LABELS, DOM_TEXT_REPLACEMENTS, DOM_ATTR_REPLACEMENTS, DRAW_TOOLTIP_RULE,
  CHAT_RULES, ITEM_RULES, TEMPLATE_OVERRIDES } = __TEST__;

let pass = 0;
const failures = [];
function check(group, name, actual, expected) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) { pass++; console.log(`  PASS [${group}] ${name}`); }
  else { failures.push(`[${group}] ${name}\n      expected ${e}\n      actual   ${a}`); console.log(`  FAIL [${group}] ${name}`); }
}
const read = p => fs.readFileSync(p, 'utf8');
const sha = s => crypto.createHash('sha256').update(s, 'utf8').digest('hex');

/* ══════════════════════════════════════════════════════════════════════
 * 语料：本机全部 module / system / core 的代码与模板
 * ══════════════════════════════════════════════════════════════════════ */
const EXT = new Set(['.js', '.mjs', '.cjs', '.hbs', '.html', '.htm']);
const SKIPDIR = new Set(['node_modules', '.git', 'packs', 'assets', 'audio', 'images', 'img', 'fonts',
  'icons', 'tokens', 'maps', 'video', 'webm', 'sounds', 'music', 'scenes', 'tiles']);
const CORPUS = [];
function walk(d) {
  let ents; try { ents = fs.readdirSync(d, { withFileTypes: true }); } catch { return; }
  for (const e of ents) {
    if (e.isDirectory()) { if (SKIPDIR.has(e.name.toLowerCase())) continue; walk(path.join(d, e.name)); }
    else if (EXT.has(path.extname(e.name).toLowerCase())) CORPUS.push(path.join(d, e.name));
  }
}
for (const r of [path.join(DATA, 'modules'), path.join(DATA, 'systems'),
  path.join(CORE, 'client'), path.join(CORE, 'templates'), path.join(CORE, 'public')]) walk(r);

const SRC = new Map();
for (const f of CORPUS) { try { SRC.set(f, read(f)); } catch { /* binary/locked */ } }
function pkgOf(f) {
  const p = f.replace(/\\/g, '/');
  const m = p.match(/\/Data\/(modules|systems)\/([^/]+)\//);
  return m ? `${m[1]}:${m[2]}` : 'core';
}
console.log(`语料：${SRC.size} 份代码/模板（modules + systems + core）\n`);
check('N', '语料规模足够大（说明扫的是真实世界不是空目录）', SRC.size > 2000, true);

/* ══════════════════════════════════════════════════════════════════════
 * N1 —— 通道 A：顶层 i18n 键的**跨包解析点**
 *
 * 已有的 C2 只问「别的包的 lang JSON 里有没有同名键」。那只覆盖了一半：
 * 顶层键是全局表，谁 localize 这个英文原文谁就被改。真正要数的是**解析点**——
 * 谁把这个字面量交给了 localize / {{localize}} / ui.notifications / DSN 的
 * description·category·name。下面这张表是实测结果，**每一条都必须是语义相同的
 * 同一句话**（同一句话被译成中文，对它也是对的）。多出任何一个新的包 -> 熔断。
 * ══════════════════════════════════════════════════════════════════════ */
const A_RESOLUTION_ALLOWLIST = {
  'Import Complete': ['systems:alienrpg', 'modules:alien-evolved-corerules', 'modules:alien-evolved-starterset'],
  'There was a problem with the Import': ['systems:alienrpg', 'modules:alien-evolved-corerules', 'modules:alien-evolved-starterset'],
  'You can only create macro buttons for owned Items': ['systems:alienrpg'],
  // ⚠ core 自己也发这一句（client/applications/settings/config.mjs:209）。整串相同 =
  //   同一句话，译文对 core 也是对的；这是**收益**不是事故，但必须写进白名单被看见。
  'No submenu found for the provided key': ['systems:alienrpg', 'core'],
  Yellow: ['systems:alienrpg'],
  AlienBlack: ['systems:alienrpg'],
  Colors: ['systems:alienrpg'],
  'Alien RPG - Blank': ['systems:alienrpg'],
  'Alien RPG - Full Dice': ['systems:alienrpg'],
};
const RE_ESC = /[.*+?^${}()|[\]\\]/g;
function resolutionOwners(lit) {
  const esc = lit.replace(RE_ESC, '\\$&');
  const pats = [
    new RegExp('(?:localize|format|has)\\s*\\(\\s*(["\'`])' + esc + '\\1'),
    new RegExp('\\{\\{\\s*localize\\s+(["\'])' + esc + '\\1'),
    new RegExp('(?:notify|info|warn|error|success)\\s*\\(\\s*(["\'`])' + esc + '\\1'),
    new RegExp('data-tooltip\\s*=\\s*(["\'])' + esc + '\\1'),
    new RegExp('(?:name|hint|title|label|description|category)\\s*:\\s*(["\'`])' + esc + '\\1'),
  ];
  const owners = new Set();
  for (const [f, s] of SRC) {
    if (!s.includes(lit)) continue;
    if (pats.some(p => p.test(s))) owners.add(pkgOf(f));
  }
  return [...owners].sort();
}
check('N', '通道 A 的键集与白名单一一对应', Object.keys(LITERAL_LABELS).sort(),
  Object.keys(A_RESOLUTION_ALLOWLIST).sort());
for (const [lit, allow] of Object.entries(A_RESOLUTION_ALLOWLIST)) {
  check('N', `通道 A 解析点仅限白名单：${JSON.stringify(lit)}`, resolutionOwners(lit), [...allow].sort());
}
// 反向：白名单里凡是非 alienrpg 的包，其英文原文必须与我们收的键**逐字节相同**
// （整串相等才命中，这是与"子串正则事故"的根本分野）。
check('N', '通道 A 全部是整串键（没有一条含正则元字符或前后空白）',
  Object.keys(LITERAL_LABELS).filter(k => k !== k.trim()), []);

/* ══════════════════════════════════════════════════════════════════════
 * N2 —— 通道 C：DOM 规则的爆炸半径
 * 每条规则挑一个**判别性最强的锚点**（系统自有类名/id，或 from 串本身），
 * 在整个真实世界里数持有者。除 alienrpg 与本模块之外出现任何包 -> 熔断。
 * ══════════════════════════════════════════════════════════════════════ */
const DOM_ANCHORS = [
  { rule: 'NPC?', needles: ['>NPC?<'], note: 'h3.resource-label.tooltip 内的整段文本' },
  { rule: 'SPECIALTY', needles: ['speciality-label'], note: '系统自有类名' },
  { rule: '{MISSING_CREW}', needles: ['{MISSING_CREW}'], note: 'JS 侧字面量' },
  { rule: 'CRT UI Sheets', needles: ['CRT UI Sheets', 'name="addcrt"'], note: '按钮名 + 文案' },
  { rule: 'No Stunts Entered', needles: ['No Stunts Entered'], note: '点击后写入 #panel' },
  { rule: 'Minus', needles: ['minus-btn'], note: '系统自有类名' },
  { rule: 'Plus', needles: ['plus-btn'], note: '系统自有类名' },
  { rule: 'Stunts', needles: ['stunt-btn'], note: '系统自有类名' },
  { rule: 'Create item', needles: ['title="Create item"'], note: '小写 i 的整串' },
  { rule: 'DRAW tooltip', needles: ['draw-from-table'], note: 'enricher 自有类名' },
  { rule: 'PUSH Roll?', needles: ['alien-Push-button'], note: '系统自有类名' },
  { rule: '#panel 观察器', needles: ['id="panel"', "id='panel'"], note: '泛用 id，靠 .alienrpg 收窄' },
  // 本轮新增的规则：`.dialog` + 一个 OK 按钮是全世界最泛的形状，
  // 唯一站得住的收窄是内容里那个 `#initiative-swap`。这一条**必须**独占。
  { rule: 'CBTracker OK', needles: ['initiative-swap'], note: '换先攻框的 select id（第二道锚点）' },
];
for (const a of DOM_ANCHORS) {
  const owners = new Set();
  for (const [f, s] of SRC) if (a.needles.some(n => s.includes(n))) owners.add(pkgOf(f));
  owners.delete('systems:alienrpg');
  check('N', `DOM 锚点在本机无外部持有者：${a.rule}（${a.note}）`, [...owners].sort(), []);
}

/* ══════════════════════════════════════════════════════════════════════
 * N3 —— 通道 D：通知垫片的近失匹配语料
 * 全部必须**原样透传**。这些是照着"Mirror Image -> 镜子 Image"那次子串事故
 * 反向构造的：只差大小写、只差空白、只是子串、句中出现、别的模块的同题材消息。
 * ══════════════════════════════════════════════════════════════════════ */
const NOTIFY_MUST_PASS_THROUGH = [
  ['大小写不同', 'could not find item Sword. You may need to delete and recreate this macro.'],
  ['尾部多一个空格', 'Could not find item Sword. You may need to delete and recreate this macro. '],
  ['首部多一个空格', ' Could not find item Sword. You may need to delete and recreate this macro.'],
  ['句号换成叹号', 'Could not find item Sword! You may need to delete and recreate this macro.'],
  ['被别的句子包住（子串）', 'Warning: Could not find item Sword. You may need to delete and recreate this macro. Please retry.'],
  ['同题材但别的模块的措辞', 'Could not find item Sword in this actor.'],
  ['ReImport 计数不是数字', 'Re-Import Completed Created many Assets'],
  ['ReImport 多一层前缀', 'Core System Upgrade Re-Import Completed Created 12 Assets'],
  ['别的包的 ERROR 前缀', 'Some Other Module | ERROR | boom'],
  ['ERROR 前缀大小写不同', 'Alien RPG - Core System | error | boom'],
  ['ERROR 前缀少一个空格', 'Alien RPG - Core System |ERROR | boom'],
  ['纯散文里出现同样的词', 'The core system will error when the item is missing.'],
  ['空串', ''],
  ['只有前缀没有正文的别包消息', 'Alien RPG News'],
];
for (const [name, input] of NOTIFY_MUST_PASS_THROUGH) {
  check('N', `通知原样透传 — ${name}`, translateNotification(input), input);
}
// 正例仍必须命中（防止收窄收过头）
check('N', '通知正例仍命中（rollItemMacro）',
  translateNotification('Could not find item Sword. You may need to delete and recreate this macro.'),
  '未找到物品 Sword。可能需要删除并重新创建这个宏。');

/* ══════════════════════════════════════════════════════════════════════
 * N4 —— 通道 E：preCreate 规则的近失语料
 * ══════════════════════════════════════════════════════════════════════ */
const CHAT_MUST_PASS = [
  ['别的模块的同名署名（差一个字）', { speaker: { alias: 'Alien RPG new' } }],
  ['署名带尾随空格', { speaker: { alias: 'Alien RPG News ' } }],
  ['Panic is over 作为子串', { content: 'The Panic is over now.' }],
  ['Panic is over 带标点', { content: 'Panic is over.' }],
  ['先攻 flavor 但没有 initiativeRoll 标记', { flavor: 'Rook rolls for Initiative! <br> x' }],
  ['先攻标记但没有 <br>（core 自己的形状）', { flags: { core: { initiativeRoll: true } }, flavor: 'Rook rolls for Initiative!' }],
  ['先攻标记 + core 的中文 flavor（foundry_chn）', { flags: { core: { initiativeRoll: true } }, flavor: 'Rook 掷先攻骰' }],
  ['PUSH 类名对但 title 不同', { content: '<button class="alien-Push-button" title="Push?">x</button>' }],
  ['PUSH title 对但类名不同', { content: '<button class="push-button" title="PUSH Roll?">x</button>' }],
  ['重伤夹缝 N 越界（-3）', { content: '<br> -3 to <strong>医疗</strong> roll' }],
  ['重伤夹缝少了 <strong>', { content: '<br> -1 to 医疗 roll' }],
  ['空对象', {}],
  ['content 不是字符串', { content: 42 }],
];
for (const [name, data] of CHAT_MUST_PASS) {
  check('N', `chat 规则不命中 — ${name}`, applyDocRules(CHAT_RULES, data), null);
}
const ITEM_MUST_PASS = [
  ['类型不是 critical-injury', { type: 'weapon', system: { attributes: { effects: '<br> -1 to <strong>医疗</strong> roll' } } }],
  ['类型对但 effects 形状不符', { type: 'critical-injury', system: { attributes: { effects: '-1 医疗' } } }],
  ['类型对但没有 effects', { type: 'critical-injury', system: { attributes: {} } }],
];
for (const [name, data] of ITEM_MUST_PASS) {
  check('N', `item 规则不命中 — ${name}`, applyDocRules(ITEM_RULES, data), null);
}

/* ══════════════════════════════════════════════════════════════════════
 * N5 —— @DRAW tooltip 正则的近失语料
 * ══════════════════════════════════════════════════════════════════════ */
const DRAW_MUST_PASS = [
  ['句中出现而非句首', 'Click here to Draw from Names. <br> x'],
  ['缺少 <br> 分隔', 'Draw from Names. x'],
  ['<br> 写成 <br/>', 'Draw from Names. <br/> x'],
  ['<br> 前后空格不同', 'Draw from Names.<br> x'],
  ['大小写不同', 'draw from Names. <br> x'],
  ['别的模块的普通 tooltip', 'Roll on this table'],
];
for (const [name, input] of DRAW_MUST_PASS) {
  check('N', `DRAW tooltip 不命中 — ${name}`, DRAW_TOOLTIP_RULE.pattern.test(input), false);
}
check('N', 'DRAW tooltip 正例仍命中',
  DRAW_TOOLTIP_RULE.pattern.test('Draw from Names. <br> 在此表上投骰'), true);

/* ══════════════════════════════════════════════════════════════════════
 * M —— 机制：对着本机安装的 Foundry / 系统源码逐条复核
 * ══════════════════════════════════════════════════════════════════════ */
const hb = read(path.join(CORE, 'client/applications/handlebars.mjs'));
check('M', 'getTemplate 第一行仍然按 Handlebars.partials 短路（抢注才有效）',
  /export async function getTemplate\(path, id=path\) \{\s*\n\s*if \( id in Handlebars\.partials \) return Handlebars\.partials\[id\];/.test(hb), true);
check('M', 'core 编译上游模板仍用 {preventIndent: true}（我们必须对齐）',
  hb.includes('Handlebars.compile(resp.html, {preventIndent: true})'), true);
check('M', 'loadTemplates 仍以 path 作为默认 partial id（id=path 时才会撞上我们的注册）',
  hb.includes('paths.map(p => getTemplate(p))'), true);

const gameMjs = read(path.join(CORE, 'client/game.mjs'));
const loc = read(path.join(CORE, 'client/helpers/localization.mjs'));
const iInit = gameMjs.indexOf('Hooks.callAll("init")');
const iI18n = gameMjs.indexOf('this.i18n.initialize()');
check('M', 'i18n.initialize() 仍排在 callAll("init") 之后（所以 init 阶段读不到真语言）',
  iInit > 0 && iI18n > iInit, true);
check('M', 'i18nInit 仍由 Localization#initialize 派发', loc.includes('Hooks.callAll("i18nInit")'), true);
check('M', 'setLanguage 仍是唯一写入 this.lang 的地方（语言判据必须晚于它）',
  /async setLanguage\(lang\)/.test(loc) && loc.includes('this.lang = lang;'), true);
check('M', '_loc 仍是 Localization#localize 的全局绑定',
  gameMjs.includes('Object.defineProperty(globalThis, "_loc", {value: this.i18n.localize.bind(this.i18n)'), true);
const notif = read(path.join(CORE, 'client/applications/ui/notifications.mjs'));
check('M', 'notify() 仍**无条件** _loc（通道 A 能接住静态字面量的根据）',
  /message = _loc\(message, format\);/.test(notif), true);
check('M', 'info/warn/error/success 仍全部转调 notify（只包一个入口就够）',
  ['this.notify(message, "info"', 'this.notify(message, "warning"',
    'this.notify(message, "error"', 'this.notify(message, "success"'].every(s => notif.includes(s)), true);
check('M', 'getProperty 第一分支仍是整键直查（带空格的整句能当顶层键）',
  read(path.join(CORE, 'common/utils/helpers.mjs')).includes('if ( key in object ) return object[key];'), true);

const appv2 = read(path.join(CORE, 'client/applications/api/application.mjs'));
check('M', 'AppV2 render 钩子仍沿继承链派发（renderApplicationV2 才会触发）',
  appv2.includes('parentClassHooks=true') && /for \( const cls of this\.constructor\.inheritanceChain\(\) \)/.test(appv2), true);
check('M', 'AppV2 render 钩子第二参仍是应用根元素（.alienrpg 类就在它身上）',
  appv2.includes('hookArgs: [this.#element, ...handlerArgs]'), true);
const appv1 = read(path.join(CORE, 'client/appv1/api/application-v1.mjs'));
check('M', 'appv1 仍在（renderApplication 钩子仍会触发）', appv1.includes('this._callHooks("render", html, data);'), true);
// 本轮修正的根据：appv1 重渲染时 html 是 inner，而 Dialog 的 inner 有**两个**根元素。
check('M', 'appv1 重渲染确实把 inner 当 html 传给钩子（所以不能用 html[0] 当窗体根）',
  /const inner = await this\._renderInner\(data\);\s*\n\s*let html = inner;/.test(appv1), true);
check('M', 'core 的 dialog.html 确实是两个根元素（html[0] 会丢掉按钮那半边）',
  (read(path.join(CORE, 'templates/hud/dialog.html')).match(/^<div class="dialog-/gm) || []).length, 2);
check('M', 'appv1 的 element getter 仍返回外层窗体（我们改用它当 root）',
  /get element\(\) \{\s*\n\s*if \( this\._element \) return this\._element;/.test(appv1), true);
const patcherSrc = read(path.join(HUB, 'scripts/alienrpg-hardcoded-cn.mjs'));
check('M', 'renderApplication 桥接已改用 app.element 而不是 html[0]',
  patcherSrc.includes('applyDom(app?.element?.[0] ?? html?.[0] ?? html);'), true);
check('M', 'renderChatMessageHTML 仍是 v14 的聊天渲染钩子',
  read(path.join(CORE, 'client/documents/chat-message.mjs')).includes('Hooks.callAll("renderChatMessageHTML", this, html)'), true);

// 抢注会不会被系统的预加载抢跑：系统里**没有**任何 init 期的模板预加载，
// 唯一的预加载在 HandlebarsApplicationMixin._preRender —— 那是渲染那一刻，远晚于 i18nInit。
const sysCode = [];
(function walkSys(d) {
  for (const e of fs.readdirSync(d, { withFileTypes: true })) {
    const p = path.join(d, e.name);
    if (e.isDirectory()) walkSys(p); else if (/\.(mjs|js)$/.test(e.name)) sysCode.push(read(p));
  }
})(path.join(SYS, 'module'));
check('M', '系统侧没有任何 loadTemplates/registerPartial（抢注不会被上游预加载抢跑）',
  sysCode.filter(s => /loadTemplates|preloadHandlebarsTemplates|registerPartial/.test(s)).length, 0);
check('M', 'AppV2 的模板预加载仍只发生在 _preRender（渲染时，晚于 i18nInit）',
  read(path.join(CORE, 'client/applications/api/handlebars-application.mjs'))
    .includes('await foundry.applications.handlebars.loadTemplates(Array.from(allTemplates));'), true);

// 两份补丁文件的哨兵不能同名，否则后装的那份被静默吞掉。
const pluginSrc = read(path.join(HUB, 'scripts/plugins-hardcoded-cn.mjs'));
check('M', '两份补丁都导出了通知垫片哨兵（否则这条判据是空转）',
  typeof __TEST__.NOTIFY_FLAG === 'string' && typeof plugins.__TEST__.NOTIFY_FLAG === 'string', true);
check('M', '两份补丁的通知垫片哨兵不同名（否则后一份静默失效）',
  __TEST__.NOTIFY_FLAG !== plugins.__TEST__.NOTIFY_FLAG, true);
check('M', '两份垫片都继承下层图章（可叠加，不互相遮蔽）',
  [patcherSrc, pluginSrc].every(s => s.includes("if (k.startsWith('__') && k.endsWith('Patched')) wrapped[k] = true;")), true);
check('M', 'plugins 补丁有语言门（英文世界必须完全空操作）',
  /function enabled\(\)[\s\S]*?game\?\.i18n\?\.lang === TARGET_LANG/.test(pluginSrc), true);
check('M', 'plugins 补丁的模板抢注已从 init 移到 i18nInit（init 读不到真语言）',
  pluginSrc.includes("HOOKS.once('i18nInit'") && !/Hooks\.once\('init'/.test(pluginSrc), true);
check('M', 'plugins 补丁可在裸 node 里 import（HOOKS 兜底存在）',
  pluginSrc.includes("const HOOKS = globalThis.Hooks ?? { on() {}, once() {} };"), true);

/* ══════════════════════════════════════════════════════════════════════
 * D —— 模板漂移：上游哈希钉死
 * ══════════════════════════════════════════════════════════════════════ */
const SYSVER = JSON.parse(read(path.join(SYS, 'system.json'))).version;
const PINS_PATH = path.join(PROJ, '7-其他内容/english-baseline/alienrpg-' + SYSVER + '/UPSTREAM-TEMPLATE-PINS.json');
check('D', `上游模板锚点文件存在（alienrpg-${SYSVER}）`, fs.existsSync(PINS_PATH), true);
if (fs.existsSync(PINS_PATH)) {
  const pins = JSON.parse(read(PINS_PATH));
  check('D', '锚点文件记录的系统版本与本机一致', pins.system_version, SYSVER);
  check('D', '锚点覆盖了 TEMPLATE_OVERRIDES 的全部条目',
    Object.keys(pins.templates).sort(), Object.keys(TEMPLATE_OVERRIDES).sort());
  const skeleton = s => (s.match(/<\/?[a-zA-Z][a-zA-Z0-9-]*/g) || []).map(t => t.toLowerCase()).join(' ');
  for (const [upstream, pin] of Object.entries(pins.templates)) {
    const upAbs = path.join(SYS, upstream.replace('systems/alienrpg/', ''));
    const blAbs = path.join(PROJ, pin.baseline_path);
    const ourAbs = path.join(HUB, pin.ours.replace('modules/alienrpg-cn/', ''));
    check('D', `上游未漂移：${upstream}`, sha(read(upAbs)), pin.upstream_sha256);
    check('D', `英文基线快照与上游一致：${pin.baseline_path}`, sha(read(blAbs)), pin.baseline_sha256);
    // 译文模板的**标签骨架**必须与基线逐标签相同：这一条抓的是哈希看不见的结构漂移
    // （译文里少一个 div / 多一个 span / 丢掉一个 {{#if}} 分支带走的元素）。
    check('D', `译文模板与基线标签骨架逐标签相同：${upstream}`, skeleton(read(ourAbs)), skeleton(read(blAbs)));
    check('D', `基线骨架哈希未被悄悄改过：${upstream}`, sha(skeleton(read(blAbs))), pin.skeleton_sha256);
  }
}

/* ══════════════════════════════════════════════════════════════════════
 * G —— 通道 B 的**术语盲区**
 *
 * 整份接管模板的代价不只是漂移：译文是**写死在 .hbs 里的中文**，它既不在
 * lang/cn.json 里，也不在合集 JSON 里。实测（2026-08-29）：
 *   · 全项目只有 test_hardcoded_patch.mjs 会读 1-系统汉化插件/templates/，
 *     而它只查「有没有残留全大写英文」和「那唯一一个 {{localize}} 还在不在」；
 *   · planet-general.hbs 的 27 个行星卡术语（大气 / 水圈 / 重力 / 气候 /
 *     轨道周期 / 资源潜力 …）**一个都不在 glossary_alien.json 的 404 条里**；
 *   · 于是它们躲开了每一道术语闸。而这批词稍后一定会在核心书正文里再次出现，
 *     到时两边各译各的，谁也不会报错。
 * 这一组就是把这块盲区钉住：英文标签集合、中文词集合各锁一个哈希，
 * 任何一侧变动都逼出一次人工复核；已进术语表的词则强制用表里的定译。
 * ══════════════════════════════════════════════════════════════════════ */
const GLOSSARY = JSON.parse(read(path.join(PROJ, '7-其他内容/glossary/glossary_alien.json')));
const HB_RE = /\{\{[\s\S]*?\}\}/g;
function visibleEnglish(src) {
  const texts = [];
  const s = src.replace(/<!--[\s\S]*?-->/g, ' ');
  const tagRe = /<[^>]*>/g;
  let last = 0, m;
  const push = c => {
    const t = c.replace(HB_RE, '').replace(/&[a-z]+;/gi, ' ').replace(/\s+/g, ' ').trim();
    if (/[A-Za-z]{2,}/.test(t)) texts.push(t);
  };
  while ((m = tagRe.exec(s))) { push(s.slice(last, m.index)); last = tagRe.lastIndex; }
  push(s.slice(last));
  return texts;
}
const cjkTokens = src => [...new Set(src.match(/[一-鿿][一-鿿·／/]*/g) || [])].sort();

// 2026-08-29 实测快照。改动任何一侧都必须**同时**更新这两个哈希，并说明为什么。
const G_SNAPSHOT = {
  'systems/alienrpg/templates/actor/planet-general.hbs': {
    baseline_label_count: 38,
    our_cjk_count: 28,
    // 出现在基线标签里、且已经进了术语表的词：必须用表里的定译
    glossary_backed: { Radiation: '辐射' },
    // 出现在基线标签里、但术语表里**没有**的词的数量。
    // 这个数不是"允许"，是"欠账"：见本轮报告的 open_issues。
    unbacked_label_count: 37,
  },
  'systems/alienrpg/templates/actor/creature-header.hbs': {
    baseline_label_count: 0, // 全部走 {{localize}}，一个写死英文文本都没有
    our_cjk_count: 0,        // 所以译文里也不该出现任何写死中文
    glossary_backed: {},
    unbacked_label_count: 0,
  },
};
for (const [upstream, snap] of Object.entries(G_SNAPSHOT)) {
  const blAbs = path.join(PROJ, '7-其他内容/english-baseline/alienrpg-' + SYSVER + '/templates',
    upstream.replace('systems/alienrpg/templates/', ''));
  const ourAbs = path.join(HUB, TEMPLATE_OVERRIDES[upstream].replace('modules/alienrpg-cn/', ''));
  const labels = visibleEnglish(read(blAbs));
  const cjk = cjkTokens(read(ourAbs));
  check('G', `基线可见英文标签数未变：${upstream}`, labels.length, snap.baseline_label_count);
  check('G', `译文写死中文词数未变：${upstream}`, cjk.length, snap.our_cjk_count);
  // 已进术语表的词必须用定译
  for (const [en, cn] of Object.entries(snap.glossary_backed)) {
    check('G', `术语表定译仍在表里：${en}`, GLOSSARY[en], cn);
    check('G', `译文模板用了定译：${en} -> ${cn}`, read(ourAbs).includes(cn), true);
  }
  // 欠账计数：多出一个没进术语表的标签 -> 熔断，逼一次裁定
  const unbacked = labels.filter(t => !Object.keys(GLOSSARY).some(k => k.toLowerCase() === t.toLowerCase()));
  check('G', `未进术语表的基线标签数未变：${upstream}`, unbacked.length, snap.unbacked_label_count);
  // creature-header 是"零写死"的样板：译文里出现任何写死中文都说明退化成了 planet 那种做法
  if (snap.our_cjk_count === 0) {
    check('G', `${upstream} 的译文仍然零写死中文（全部回到 lang 通道）`, cjk, []);
  }
}

/* ══════════════════════════════════════════════════════════════════════
 * F —— 保险丝：今天够不到、上游一改就变成活面的东西
 * ══════════════════════════════════════════════════════════════════════ */

// F1 update.js 的 allDone() 是一个**全英文**对话框（标题 "Alien RPG Update"、
//    正文 "<p>The update has completed and the following have been updated:</p>"、
//    按钮 "Okay!"）。4.1.13 里 updateAssets 是空数组，:28 提前 return，所以它今天
//    根本到不了。给不存在的界面写 DOM 规则只会烂掉 —— 改成盯着这个前提。
const updateJs = read(path.join(SYS, 'module/apps/update.js'));
check('F', 'update.js 的 updateAssets 仍是空数组（allDone 的英文对话框仍够不到）',
  /const updateAssets = \[\s*\]/.test(updateJs), true);
check('F', 'update.js 的 allDone 仍是那三句英文（一旦上游改文案，我们的判据要跟着改）',
  updateJs.includes('title: `${moduleTitle} Update`') &&
  updateJs.includes('<p>The update has completed and the following have been updated:</p>') &&
  updateJs.includes('label: "Okay!"'), true);

// F2 "Vehicle inceptions are not allowed!" 的守卫是 `crew.type === "vehicles" &&
//    crew.type === "spacecraft"` —— 同一个字符串不可能同时等于两个值，**恒假**。
//    所以这条通知在上游是死代码，不进通道 A（给发不出来的消息注册顶层键 =
//    没有读者的写入）。上游哪天把 && 改成 ||，这里熔断。
for (const f of ['module/sheets/spacecraft-sheet.mjs', 'module/sheets/vehicle-sheet.mjs']) {
  check('F', `${f} 的 Vehicle inceptions 守卫仍恒假（该通知仍是死代码）`,
    read(path.join(SYS, f)).includes('if (crew.type === "vehicles" && crew.type === "spacecraft") return ui.notifications.info("Vehicle inceptions are not allowed!");'), true);
}

// F3 CBTracker 的 OK 按钮：本轮新增了规则，这里盯住它赖以成立的两个前提。
const cbt = read(path.join(SYS, 'module/helpers/CBTracker.mjs'));
check('F', 'CBTracker 的确认按钮仍写死 label: "OK"（规则仍有必要）', cbt.includes('label: "OK",'), true);
check('F', 'CBTracker 仍是活的（alienrpg.mjs 仍 import 它）',
  read(path.join(SYS, 'module/alienrpg.mjs')).includes('import AlienRPGCTContext from "./helpers/CBTracker.mjs"'), true);
check('F', '换先攻框的内容模板仍带 #initiative-swap（第二道锚点仍在）',
  read(path.join(SYS, 'templates/dialog/switch-initiative.html')).includes('<select id="initiative-swap">'), true);
check('F', 'CBTracker OK 规则已进表', DOM_TEXT_REPLACEMENTS.some(r =>
  r.from === 'OK' && r.scope === '.dialog' && r.contains === '#initiative-swap'), true);
check('F', 'applyDom 支持 contains 第二锚点', patcherSrc.includes('if (rule.contains && !root.querySelector(rule.contains)) continue;'), true);

// F4 logger.warn / logger.notify 把**任意** args[0] 直接发成通知，通道 D 的第三条
//    只锚定带 "Alien RPG - Core System | ERROR | " 前缀的那一路。活跃代码里经这两条
//    路发出的英文只有 YZEDiceRoller 那句调试残留（上游缺陷，不翻）。多一处 -> 熔断。
// migratefolders.js 整份是死的 —— init.mjs:2 那行 import 被注释掉了，全仓再无第二处引用。
// 它自己带 3 条英文通知 + 一个英文对话框，所以「它还死着」本身就是一条前提，要盯住。
check('F', 'migratefolders.js 仍是死文件（唯一的 import 仍被注释掉）',
  /^\s*\/\/\s*import migrateFolders from ['"]\.\/migratefolders\.js['"]/m.test(read(path.join(SYS, 'module/apps/init.mjs'))), true);
const DEAD_FILES = /(?:^|\/)(?:old-[^/]*|migratefolders\.js|migration\.js|logger\.js|colony-sheet\.js|planet-sheet\.js)$/;
const loggerPassthrough = [];
for (const f of fs.readdirSync(path.join(SYS, 'module'), { recursive: true })) {
  const p = path.join(SYS, 'module', String(f)).replace(/\\/g, '/');
  if (!/\.(mjs|js)$/.test(p) || DEAD_FILES.test(p)) continue;
  const s = read(p);
  for (const m of s.matchAll(/logger\.(?:warn|notify)\(\s*(["'`])((?:\\.|(?!\1)[\s\S])*?)\1/g)) {
    if (/[A-Za-z]{2,}/.test(m[2])) loggerPassthrough.push(path.relative(SYS, p).replace(/\\/g, '/') + ' :: ' + m[2]);
  }
}
check('F', 'logger 直通通知里的英文只有已知的那一处调试残留', loggerPassthrough.sort(),
  ['module/helpers/YZEDiceRoller.mjs :: ! Supply ']);

// F5 系统里所有 config:false 的设置项名称/提示都不进设置面板，所以不必翻。
//    一旦某个变成 config:true，它的英文 name/hint 就上界面了。
const settingsSrc = read(path.join(SYS, 'module/helpers/settings.mjs')) + read(path.join(SYS, 'module/apps/init.mjs'));
for (const en of ['System Migration Version', 'Message from the devs', 'Semaphore Flag',
  'Imported Compendiums', 'Module Version']) {
  const i = settingsSrc.indexOf(`"${en}"`);
  const tail = settingsSrc.slice(i, i + 400);
  check('F', `设置项 ${JSON.stringify(en)} 仍是 config:false（不上界面，不必翻）`,
    /config:\s*false/.test(tail.slice(0, tail.indexOf('})') + 2)), true);
}

/* ══════════════════════════════════════════════════════════════════════ */
console.log(`\n${'='.repeat(70)}`);
console.log(`PASS ${pass}   FAIL ${failures.length}`);
if (failures.length) {
  console.log('\nFAILURES:');
  for (const f of failures) console.log('  - ' + f);
  process.exit(1);
}
console.log('ALL GREEN');
