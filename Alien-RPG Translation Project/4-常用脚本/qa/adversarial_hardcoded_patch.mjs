/**
 * adversarial_hardcoded_patch.mjs —— 运行时补丁的**对抗式**闸门。
 *
 * 与 test_hardcoded_patch.mjs 分工明确，别合并：
 *   · test_hardcoded_patch.mjs 问的是「我们声称要改的东西，改对了吗」（正例 + 自洽）。
 *   · 本文件问的是「我们**不该**碰的东西，会不会被碰到」，以及「我们赖以成立的**机制**，
 *     在当前这台机器上的 Foundry / 系统源码里，今天还成立吗」。
 *
 * 五组：
 *   N —— 爆炸半径（negative）。判据不是"我觉得不会撞"，是**在本机真实世界里数**：
 *        295 个模块 + 全部 system + core，逐个找"谁还能产生这个形状"。
 *   M —— 机制（mechanism）。抢注短路、编译选项、钩子存在性与触发顺序，全部对着
 *        本机安装的 Foundry 源码逐行复核，不靠记忆也不靠文档。
 *   D —— 模板漂移（drift）。抢注是整份替换，上游改一个字我们就静默过期。
 *        UPSTREAM-TEMPLATE-PINS.json 把上游钉死，这里机械比对。
 *   F —— 保险丝（fuse）。今天够不到但**上游一改就会变成活面**的东西。
 *        它们现在不写规则（给不存在的界面写规则只会烂掉），改由闸门盯着。
 *
 *   S —— 卡尺（shim）。唯一一条被主动**解冻**的 T-FROZEN 串：`MU/TH/ER Instructions.`。
 *        它不再冻着，是因为通道 F 的译名回退垫片接管了 `game.journal.getName`。
 *        垫片一旦被删，译名就必须同时撤掉 —— 这一组就是盯那条**双向依赖**的：
 *        机制（core / 系统源码事实）、联动（垫片 / 出货包 / 切片 / 登记表 file:line）、
 *        行为（直接跑出货的 installNameFallback，正例 + 一整排反例）。
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

/* ══════════════════════════════════════════════════════════════════════
 * ══ 插件侧（terminal 4.0.11 / motion-tracker-multideck 1.0.2）══
 *
 * 这两个插件**没有任何 i18n 管道**（module.json 无 languages、代码里 game.i18n
 * 命中 0、模板里 {{localize 命中 0}}），全部译文由 scripts/plugins-hardcoded-cn.mjs
 * 的五条通道承担。通道越靠近 DOM，爆炸半径越大，所以下面每一条通道都要在**本机
 * 真实世界**里数一遍持有者，再用近失语料证明收窄没有收过头也没有漏。
 * ══════════════════════════════════════════════════════════════════════ */
const P = plugins.__TEST__;
const MD_DIR = path.join(DATA, 'modules/motion-tracker-multideck');
const TERM_DIR = path.join(DATA, 'modules/terminal');
const mdJs = read(path.join(MD_DIR, 'scripts/multideck.js'));
const termHooks = read(path.join(TERM_DIR, 'scripts/hooks.js'));
const termJs = read(path.join(TERM_DIR, 'scripts/terminal.js'));
const termPresetsSrc = read(path.join(TERM_DIR, 'scripts/presets.js'));
const termHbs = read(path.join(TERM_DIR, 'templates/terminal.hbs'));
const MD_VER = JSON.parse(read(path.join(MD_DIR, 'module.json'))).version;
const TERM_VER = JSON.parse(read(path.join(TERM_DIR, 'module.json'))).version;
check('N', '插件侧判据是对着已知版本推的：multideck', MD_VER, '1.0.2');
check('N', '插件侧判据是对着已知版本推的：terminal', TERM_VER, '4.0.11');

/* ══════════════════════════════════════════════════════════════════════
 * N6 —— 插件通道 A：顶层 i18n 键的跨包解析点
 * 与 N1 同一把尺子。每一条键的解析点都必须**只有它自己的那个插件**。
 * ══════════════════════════════════════════════════════════════════════ */
const PLUGIN_A_ALLOWLIST = {
  // motion-tracker-multideck：窗口标题 :623 / 设置菜单 :850-852 / 4 条静态通知
  'Alien RPG - Motion Tracker Multideck Companion — Scene Links': ['modules:motion-tracker-multideck'],
  'Configure Scene Links': ['modules:motion-tracker-multideck'],
  'Choose which Scenes belong to the same location so the Motion Tracker can detect contacts across them.':
    ['modules:motion-tracker-multideck'],
  'That point could not be saved. You can type the X and Y coordinates manually instead.':
    ['modules:motion-tracker-multideck'],
  'Motion Tracker Multideck Companion Scene links saved.': ['modules:motion-tracker-multideck'],
  'Alien RPG - Motion Tracker Multideck Companion requires Alien RPG - Motion Tracker to be active.':
    ['modules:motion-tracker-multideck'],
  'The Multideck Companion could not connect to the Motion Tracker. Check the browser console for details.':
    ['modules:motion-tracker-multideck'],
  // terminal：3 个 config:true 设置项的 name/hint
  '🔆 Screensaver': ['modules:terminal'],
  'Skill check required Terminals display an interactive visual. This might not aesthetically mesh well with some game systems.':
    ['modules:terminal'],
  '🔔 Extra Notifications': ['modules:terminal'],
  'The GM gets notifications about even minor actions within Terminal (e.g. when someone is given observer permission to a Journal, or a macro script is ran)':
    ['modules:terminal'],
  '⚡ Keep Cache Warm': ['modules:terminal'],
  '(Designed for Forge but could be used for any sever behind CloudFlare DNS): This will keep a cache warm in an edge network. This significantly improves Terminal load times. At the cost of more overall network calls.':
    ['modules:terminal'],
};
check('N', '插件通道 A 的键集与白名单一一对应', Object.keys(P.LITERAL_LABELS).sort(),
  Object.keys(PLUGIN_A_ALLOWLIST).sort());
for (const [lit, allow] of Object.entries(PLUGIN_A_ALLOWLIST)) {
  check('N', `插件通道 A 解析点仅限白名单：${JSON.stringify(lit.slice(0, 46))}`, resolutionOwners(lit), [...allow].sort());
}
check('N', '插件通道 A 全部是整串键（没有一条含前后空白）',
  Object.keys(P.LITERAL_LABELS).filter(k => k !== k.trim()), []);
check('N', '插件通道 A 每条都声明了 requires（没装那个插件就不写全局表）',
  Object.entries(P.LITERAL_LABELS).filter(([, v]) => !v.requires || typeof v.cn !== 'string').map(([k]) => k), []);
// 键必须与上游源码里的英文原文**逐字节相同**：抄错一个字节 = 一条永不命中的死键。
const PLUGIN_A_SRC = mdJs + ' ' + termHooks;
check('N', '插件通道 A 的每条键都在上游源码里逐字节存在',
  Object.keys(P.LITERAL_LABELS).filter(k => !PLUGIN_A_SRC.includes(k)), []);
// requires 与实际持有者对得上（terminal 的键不能挂 multideck 的 requires）
check('N', '插件通道 A 的 requires 与解析点持有者一致',
  Object.entries(P.LITERAL_LABELS)
    .filter(([k, v]) => resolutionOwners(k)[0] !== `modules:${v.requires}`)
    .map(([k]) => k), []);

/* ══════════════════════════════════════════════════════════════════════
 * N7 —— 插件通道 A：本机 lang JSON 的顶层键冲突面必须是 0
 * 顶层键是全局表，别的包若把同一串定义成顶层键，就是我们和它抢同一个格子。
 * （代码里写 `if (typeof translations[key] === 'string') 让出` 只是运行期兜底，
 *  这里要的是「本机根本不存在这个竞争」这条更强的事实。）
 * ══════════════════════════════════════════════════════════════════════ */
const LANGFILES = [];
(function walkLang(d) {
  let ents; try { ents = fs.readdirSync(d, { withFileTypes: true }); } catch { return; }
  for (const e of ents) {
    const p = path.join(d, e.name);
    if (e.isDirectory()) { if (SKIPDIR.has(e.name.toLowerCase())) continue; walkLang(p); }
    else if (e.name.endsWith('.json') && /[\\/](lang|languages|i18n)[\\/]/.test(p)) LANGFILES.push(p);
  }
})(path.join(DATA, 'modules'));
(function walkLang2(d) {
  let ents; try { ents = fs.readdirSync(d, { withFileTypes: true }); } catch { return; }
  for (const e of ents) {
    const p = path.join(d, e.name);
    if (e.isDirectory()) { if (SKIPDIR.has(e.name.toLowerCase())) continue; walkLang2(p); }
    else if (e.name.endsWith('.json') && /[\\/](lang|languages|i18n)[\\/]/.test(p)) LANGFILES.push(p);
  }
})(path.join(DATA, 'systems'));
check('N', 'lang JSON 语料规模足够大（说明扫的是真实世界）', LANGFILES.length > 150, true);
const langCollisions = [];
for (const f of LANGFILES) {
  let j; try { j = JSON.parse(read(f)); } catch { continue; }
  for (const k of Object.keys(P.LITERAL_LABELS)) {
    if (Object.prototype.hasOwnProperty.call(j, k)) langCollisions.push(path.basename(path.dirname(f)) + '/' + path.basename(f) + ' :: ' + k);
  }
}
check('N', '插件通道 A 的键在本机所有 lang JSON 里都不是顶层键', langCollisions.sort(), []);

/* ══════════════════════════════════════════════════════════════════════
 * N8 —— 插件通道 C：4 条带 ${} 的通知，近失语料必须原样透传
 * 照着「Mirror Image -> 镜子 Image」那次子串事故反向构造：
 * 只差大小写 / 只差标点 / 只是子串 / 插值位越界 / 别的模块的同题材消息。
 * ══════════════════════════════════════════════════════════════════════ */
const PLUGIN_NOTIFY_MUST_PASS = [
  ['side 不是 A/B（小写）', 'Select Scene a before marking a matching point.'],
  ['side 是别的字母', 'Select Scene C before marking a matching point.'],
  ['句号换成叹号', 'Select Scene A before marking a matching point!'],
  ['被别的句子包住（子串）', 'Warning: Select Scene A before marking a matching point. Retry.'],
  ['尾部多一个空格', 'Select Scene A before marking a matching point. '],
  ['Opening 少了序号', 'Opening Deck 3. Click matching point for Scene A.'],
  ['Opening 的序号不是数字', 'Opening Deck 3. Click matching point two for Scene A.'],
  ['Opening 的 side 越界', 'Opening Deck 3. Click matching point 2 for Scene C.'],
  ['坐标不是数字', 'Saved matching point on Deck 3: (x, y).'],
  ['坐标少一个括号', 'Saved matching point on Deck 3: (1, 2.'],
  ['坐标中间没有空格', 'Saved matching point on Deck 3: (1,2).'],
  ['Scene link 的引号是单引号', "Scene link 'A/B' uses the same Scene for both Scene A and Scene B."],
  ['Scene link 少了句号', 'Scene link "A/B" uses the same Scene for both Scene A and Scene B'],
  ['别的模块的同题材消息', 'Select Scene A before continuing.'],
  ['纯散文里出现同样的词', 'You should select scene A before marking a matching point in the other module.'],
  ['空串', ''],
  ['已经是中文（幂等，不能二次改写）', '请先选择场景 A，再标记对位点。'],
];
for (const [name, input] of PLUGIN_NOTIFY_MUST_PASS) {
  check('N', `插件通知原样透传 — ${name}`, plugins.translatePluginNotification(input), input);
}
check('N', '插件通知对非字符串原样透传（Error 对象不被拍平）',
  plugins.translatePluginNotification(42), 42);
// 正例仍必须命中（防止收窄收过头）—— 四条各一个
const PLUGIN_NOTIFY_MUST_HIT = [
  ['Select Scene B before marking a matching point.', '请先选择场景 B，再标记对位点。'],
  ['Opening Deck 3. Click matching point 2 for Scene A.', '正在打开 Deck 3。请为场景 A 点击第 2 个对位点。'],
  ['Saved matching point on Deck 3: (1024.5, -12).', '已在 Deck 3 上保存对位点：(1024.5, -12)。'],
  ['Scene link "A/B" uses the same Scene for both Scene A and Scene B.', '场景链接“A/B”把同一个场景同时用作场景 A 和场景 B。'],
];
for (const [input, want] of PLUGIN_NOTIFY_MUST_HIT) {
  check('N', `插件通知正例仍命中 — ${JSON.stringify(input.slice(0, 40))}`, plugins.translatePluginNotification(input), want);
}
// 上游那 4 条模板串确实长这个样子（否则我们的正则是对着幻觉写的）
for (const needle of [
  'ui.notifications.warn(`Select Scene ${side.toUpperCase()} before marking a matching point.`)',
  'ui.notifications.info(`Opening ${scene.name}. Click matching point ${ai + 1} for Scene ${side.toUpperCase()}.`)',
  'ui.notifications.info(`Saved matching point on ${scene.name}: (${x}, ${y}).`)',
  'ui.notifications.error(`Scene link "${link.name || link.id}" uses the same Scene for both Scene A and Scene B.`)',
]) {
  check('N', `上游模板串未漂移：${JSON.stringify(needle.slice(22, 62))}`, mdJs.includes(needle), true);
}

/* ══════════════════════════════════════════════════════════════════════
 * N9 —— 插件通道 D：terminal 4 个按钮标签的 DOM 爆炸半径
 * ══════════════════════════════════════════════════════════════════════ */
const TERMINAL_DOM_ANCHORS = [
  { a: 'terminal-folders', note: '观察器挂载点，也是第二道锚点' },
  { a: 'terminal-charge-btn', note: '按钮自有类名' },
  { a: 'terminal-ping-btn', note: '按钮自有类名' },
  { a: 'terminal-lights-btn', note: '按钮自有类名' },
  { a: 'terminal-explore-btn', note: '按钮自有类名' },
];
for (const { a, note } of TERMINAL_DOM_ANCHORS) {
  const owners = new Set();
  for (const [f, s] of SRC) if (s.includes(a)) owners.add(pkgOf(f));
  owners.delete('modules:terminal');
  check('N', `terminal DOM 锚点在本机无外部持有者：${a}（${note}）`, [...owners].sort(), []);
}
// ⚠ 反向判据：`.terminal-window` **有**外部持有者，所以它不能当锚点。
//   这一条不是「担心」，是实测：alien-mu-th-ur 也用这个类名。
const windowOwners = new Set();
for (const [f, s] of SRC) if (s.includes('terminal-window')) windowOwners.add(pkgOf(f));
check('N', '`terminal-window` 确实是多包共用的类名（所以不可用作锚点）',
  [...windowOwners].sort(), ['modules:alien-mu-th-ur', 'modules:terminal']);
check('N', '我们的补丁没有把 terminal-window 当锚点用',
  read(path.join(HUB, 'scripts/plugins-hardcoded-cn.mjs')).includes(".querySelector?.('.terminal-window')"), false);
// 钩子名 renderTerminal 由 constructor.name 派发：本机只有一个 class Terminal
const terminalClassOwners = new Set();
for (const [f, s] of SRC) if (/(?:^|\s)class\s+Terminal\b/.test(s)) terminalClassOwners.add(pkgOf(f));
check('N', 'renderTerminal 钩子名在本机只有一个持有者', [...terminalClassOwners].sort(), ['modules:terminal']);

// 换标签函数本身的近失语料：整串相等才改，差一个字节都放行。
function fakeRoot(nodes) {
  return {
    querySelectorAll(sel) {
      const cls = sel.slice(1);
      return nodes.filter(n => n.cls.split(' ').includes(cls));
    },
  };
}
const RELABEL_CASES = [
  ['整串相等 -> 改', 'terminal-button terminal-ping-btn', 'Detect Motion', '侦测运动'],
  ['尾部多一个空格 -> 放行', 'terminal-button terminal-ping-btn', 'Detect Motion ', 'Detect Motion '],
  ['大小写不同 -> 放行', 'terminal-button terminal-ping-btn', 'detect motion', 'detect motion'],
  ['被包在更长的串里 -> 放行', 'terminal-button terminal-ping-btn', 'Detect Motion Now', 'Detect Motion Now'],
  ['文本对但类名不对 -> 放行', 'terminal-button terminal-macro-btn', 'Detect Motion', 'Detect Motion'],
  ['类名对但已是中文 -> 放行（幂等）', 'terminal-button terminal-ping-btn', '侦测运动', '侦测运动'],
  ['GM 自定义的按钮名 -> 放行', 'terminal-button terminal-lights-btn', 'Kill the lights', 'Kill the lights'],
];
for (const [name, cls, before, after] of RELABEL_CASES) {
  const node = { cls, textContent: before };
  P.relabelTerminalButtons(fakeRoot([node]));
  check('N', `terminal 按钮改名 — ${name}`, node.textContent, after);
}
check('N', 'terminal 按钮改名对空 root 不抛', P.relabelTerminalButtons(null), 0);
// 4 条译文都得有活的上游落点
for (const [cls, { en }] of Object.entries(P.TERMINAL_BUTTON_LABELS)) {
  const re = new RegExp('className = "terminal-button ' + cls + '"\\s*\\n\\s*div\\.textContent = "'
    + en.replace(RE_ESC, '\\$&') + '"');
  check('N', `terminal 按钮上游落点未漂移：${cls}`, re.test(termJs), true);
}

/* ══════════════════════════════════════════════════════════════════════
 * N10 —— 插件通道 E：9 个 ASCII 大标题只动 <h1>，点阵艺术一个字节不碰
 * 直接 import 上游 presets.js（纯数据模块，裸 node 可载），拿**真实**的 ASCII
 * 对象跑我们的出货函数，而不是在测试里抄一份等价实现。
 * ══════════════════════════════════════════════════════════════════════ */
const upstreamPresets = await import(pathToFileURL(path.join(TERM_DIR, 'scripts/presets.js')).href);
const UP_ASCII = upstreamPresets.ASCII;
check('N', '上游 presets.js 仍导出 ASCII 对象', typeof UP_ASCII === 'object' && UP_ASCII !== null, true);
let artBytes = 0;
for (const [key, { en, cn }] of Object.entries(P.TERMINAL_ASCII_HEADINGS)) {
  const before = UP_ASCII[key];
  const after = P.translateAsciiHeading(before, en, cn);
  check('N', `通道 E 命中：${key}`, typeof after === 'string', true);
  if (typeof after !== 'string') continue;
  // ① <pre> 起点之后的每一个字节都必须原样保留（点阵艺术零改动）
  const artBefore = before.slice(before.indexOf('<pre'));
  const artAfter = after.slice(after.indexOf('<pre'));
  check('N', `通道 E 未碰点阵艺术：${key}`, artAfter === artBefore, true);
  artBytes += artBefore.length;
  // ② 改动只发生在 h1 内，且长度差恰好等于「中文 + <span></span> - 英文」
  const delta = after.length - before.length;
  check('N', `通道 E 的字节差恰好只来自 h1 文本：${key}`, delta, cn.length + 13 - en.length);
  // ③ 幂等：对已改过的串再跑一次不再命中（needle 已不存在）
  check('N', `通道 E 幂等：${key}`, P.translateAsciiHeading(after, en, cn), null);
}
// 2026-08-29 实测：这 9 条里 <pre> 之后的点阵艺术共 2207 字符（整个 ASCII 对象
// 18 条、7082 字符，其中非标题部分 6603）。阈值只是防「上面那些比对是空跑」。
check('N', '通道 E 保护住的点阵艺术字符数（有实际体量，不是空跑）', artBytes > 2000, true);
const E_MUST_PASS = [
  ['上游改了标题文字 -> 不动', '<h1 style="font-family: inherit">Map Downloaded!</h1><pre>art</pre>'],
  ['同样的串只出现在 <pre> 里 -> 不动', '<h1 style="font-family: inherit">Other</h1><pre>>Map Downloaded</h1></pre>'],
  ['整条不是以 <h1 开头 -> 不动', '<pre>art</pre><h1 style="font-family: inherit">Map Downloaded</h1>'],
  ['大小写不同 -> 不动', '<h1 style="font-family: inherit">MAP DOWNLOADED</h1><pre>art</pre>'],
  ['多一个空格 -> 不动', '<h1 style="font-family: inherit">Map Downloaded </h1><pre>art</pre>'],
];
for (const [name, src] of E_MUST_PASS) {
  check('N', `通道 E 不命中 — ${name}`, P.translateAsciiHeading(src, 'Map Downloaded', '地图已下载'), null);
}
check('N', '通道 E 对非字符串不抛', P.translateAsciiHeading(undefined, 'x', 'y'), null);
// 9 句英文原文在本机只有 terminal 一个持有者
for (const { en } of Object.values(P.TERMINAL_ASCII_HEADINGS)) {
  const owners = new Set();
  for (const [f, s] of SRC) if (s.includes(en)) owners.add(pkgOf(f));
  check('N', `ASCII 大标题在本机无外部持有者：${en}`, [...owners].sort(), ['modules:terminal']);
}

/* ══════════════════════════════════════════════════════════════════════
 * M2 —— 插件侧机制：对着本机 Foundry / 插件源码逐条复核
 * ══════════════════════════════════════════════════════════════════════ */
check('M', 'appv1 的 title getter 仍 _loc（multideck 窗口标题走通道 A 的根据）',
  /get title\(\) \{\s*\n\s*return _loc\(this\.options\.title\);/.test(appv1), true);
check('M', 'appv1 的 _renderInner 仍走 renderTemplate（抢注 partial 才会被命中）',
  appv1.includes('await foundry.applications.handlebars.renderTemplate(this.template, data)'), true);
check('M', 'renderTemplate 仍第一句就 getTemplate（抢注短路点）',
  /export async function renderTemplate\(path, data\) \{\s*\n\s*const template = await getTemplate\(path\);/.test(hb), true);
const cfgMjs = read(path.join(CORE, 'client/applications/settings/config.mjs'));
check('M', 'settings 面板仍把 menu 的 name/hint/label 摘成 label/hint/buttonText',
  cfgMjs.includes('label: menu.name,') && cfgMjs.includes('hint: menu.hint,') && cfgMjs.includes('buttonText: menu.label'), true);
check('M', 'settings 面板仍对 setting.name / setting.hint 跑 _loc',
  cfgMjs.includes('data.field.label ||= _loc(setting.name ?? "");') &&
  cfgMjs.includes('data.field.hint ||= _loc(setting.hint ?? "");'), true);
const cfgCat = read(path.join(CORE, 'templates/settings/config-category.hbs'));
check('M', '设置面板模板仍对 menu 的三个字段 {{localize}}',
  ['{{localize entry.label}}', '{{localize entry.buttonText}}', '{{localize entry.hint}}']
    .every(s => cfgCat.includes(s)), true);
check('M', 'registerMenu 仍不做任何本地化（所以必须在渲染前写好 i18n 表）',
  /registerMenu\(namespace, key, data\) \{[\s\S]*?this\.menus\.set\(data\.key, data\);/.test(
    read(path.join(CORE, 'client/helpers/client-settings.mjs'))), true);
check('M', 'AppV2 的 render 钩子仍在 _onRender 的 Promise resolve **之后**才派发',
  appv2.includes('if ( async && (response instanceof Promise) ) return response.then(r => {'), true);

// multideck 侧
check('M', 'multideck 的 SceneLinkManager 仍是 appv1 FormApplication',
  mdJs.includes('class SceneLinkManager extends foundry.appv1.api.FormApplication'), true);
check('M', 'multideck 的 template 路径仍与我们抢注的 key 逐字节相同',
  mdJs.includes('template: `modules/${MODULE_ID}/templates/link-manager.hbs`') &&
  mdJs.includes('const MODULE_ID = "motion-tracker-multideck"'), true);
check('M', 'multideck 的窗口标题仍是我们收的那一串',
  mdJs.includes('title: "Alien RPG - Motion Tracker Multideck Companion — Scene Links"'), true);
check('M', 'multideck 仍没有任何 game.i18n / {{localize}}（lang 文件仍是没有读者的写入）',
  /game\.i18n/.test(mdJs) || /\{\{\s*localize/.test(read(path.join(MD_DIR, 'templates/link-manager.hbs'))), false);

// terminal 侧
check('M', 'terminal 的 4 个按钮仍 append 到 .terminal-folders（观察器挂对了地方）',
  /const sibling = lain\.querySelector\(`\.terminal-folders`\)/.test(termJs) &&
  (termJs.match(/sibling\.appendChild\(div\)/g) || []).length >= 4, true);
check('M', 'terminal.hbs 里仍有 .terminal-folders（渲染时就存在，可以立刻挂观察器）',
  termHbs.includes('<div class="terminal-folders"></div>'), true);
check('M', 'terminal 的按钮 onclick 仍在点击那一刻读 div.textContent（所以改 DOM 会连带翻内容页）',
  termJs.includes('data.content = div.textContent') && termJs.includes('showContent(newPointer, ASCII.MOTION, div.textContent)'), true);
check('M', 'terminal 的密码流仍在渲染之后才 createHTML（所以必须挂观察器而不是扫一遍）',
  /static password\(\)[\s\S]*?a\.createHTML\(\)/.test(termJs), true);
check('M', 'terminal.js 仍从 ./presets.js 具名导入 ASCII（改属性才会被它看见）',
  termJs.includes('import { ASCII } from "./presets.js"'), true);
check('M', 'terminal 仍没有任何 game.i18n（3 个设置项只能走通道 A）',
  /game\.i18n/.test(termJs + termHooks + termPresetsSrc), false);

/* ══════════════════════════════════════════════════════════════════════
 * D2 —— 插件模板漂移
 * ══════════════════════════════════════════════════════════════════════ */
const PLUGIN_PINS_PATH = path.join(PROJ, '7-其他内容/english-baseline/motion-tracker-multideck-'
  + MD_VER + '/UPSTREAM-TEMPLATE-PINS.json');
check('D', `插件上游模板锚点文件存在（motion-tracker-multideck-${MD_VER}）`, fs.existsSync(PLUGIN_PINS_PATH), true);
if (fs.existsSync(PLUGIN_PINS_PATH)) {
  const pins = JSON.parse(read(PLUGIN_PINS_PATH));
  check('D', '插件锚点文件记录的版本与本机一致', pins.module_version, MD_VER);
  check('D', '插件锚点覆盖了 TEMPLATE_OVERRIDES 的全部条目',
    Object.keys(pins.templates).sort(), Object.keys(P.TEMPLATE_OVERRIDES).sort());
  const skeleton = s => (s.match(/<\/?[a-zA-Z][a-zA-Z0-9-]*/g) || []).map(t => t.toLowerCase()).join(' ');
  // handlebars 表达式序列（去掉 {{!-- 注释 --}}）：骨架看不见 {{#if}} 被改坏，这条专抓它
  const hbsExprs = s => (s.replace(/\{\{!--[\s\S]*?--\}\}/g, '').match(/\{\{[^}]*\}\}/g) || []).join('');
  for (const [upstream, pin] of Object.entries(pins.templates)) {
    const upAbs = path.join(DATA, upstream);
    const blAbs = path.join(PROJ, pin.baseline_path);
    const ourAbs = path.join(HUB, pin.ours.replace('modules/alienrpg-cn/', ''));
    check('D', `插件上游未漂移：${upstream}`, sha(read(upAbs)), pin.upstream_sha256);
    check('D', `插件英文基线快照与上游一致：${pin.baseline_path}`, sha(read(blAbs)), pin.baseline_sha256);
    check('D', `插件译文模板与基线标签骨架逐标签相同：${upstream}`, skeleton(read(ourAbs)), skeleton(read(blAbs)));
    check('D', `插件基线骨架哈希未被悄悄改过：${upstream}`, sha(skeleton(read(blAbs))), pin.skeleton_sha256);
    check('D', `插件译文模板的 handlebars 表达式与基线逐条相同：${upstream}`,
      hbsExprs(read(ourAbs)), hbsExprs(read(blAbs)));
    check('D', `插件译文模板里没有残留可见英文：${upstream}`, visibleEnglish(
      read(ourAbs).replace(/\{\{!--[\s\S]*?--\}\}/g, ' ')), []);
    check('D', `插件基线可见英文标签数未变：${upstream}`,
      visibleEnglish(read(blAbs).replace(/\{\{!--[\s\S]*?--\}\}/g, ' ')).length, pin.baseline_visible_english_count);
    // 只看**值**里有实义英文的那几条：placeholder="X" / "Y" 是坐标轴名，两边都不翻。
    const attrs = s => (s.match(/(?:title|placeholder)="[^"]*"/g) || [])
      .filter(a => /[A-Za-z]{3,}/.test(a.slice(a.indexOf('"') + 1, -1)));
    check('D', `插件基线的可见属性串未变：${upstream}`, attrs(read(blAbs)), pin.baseline_attr_strings);
  }
}

/* ══════════════════════════════════════════════════════════════════════
 * F6-F10 —— 插件侧保险丝：今天成立、上游一改就要重新裁定的前提
 * ══════════════════════════════════════════════════════════════════════ */

// F6 multideck 唯一那个 game.settings.register 仍是 config:false。
//    它的 name "Scene Links" / hint 因此**不进设置面板**，按姊妹文件 F5 的判据不翻。
//    一旦变成 config:true，这两句英文就上界面 —— 熔断，逼一次补译。
{
  const i = mdJs.indexOf('name: "Scene Links"');
  const tail = mdJs.slice(i, i + 400);
  check('F', 'multideck 的 LINKS_SETTING 仍是 config:false（"Scene Links" 不上界面，不必翻）',
    i > 0 && /config:\s*false/.test(tail.slice(0, tail.indexOf('})') + 2)), true);
  check('F', 'multideck 仍只有 1 个 settings.register 与 1 个 registerMenu（多出来的要重新盘点）',
    [(mdJs.match(/game\.settings\.register\(/g) || []).length,
      (mdJs.match(/game\.settings\.registerMenu\(/g) || []).length], [1, 1]);
  check('F', 'multideck 的 ui.notifications 仍是 8 条（多一条就有漏译）',
    (mdJs.match(/ui\.notifications\./g) || []).length, 8);
}

// F7 terminal 的 3 个设置项仍是 config:true（我们翻了它们，前提是它们真的上界面）。
for (const en of ['🔆 Screensaver', '🔔 Extra Notifications', '⚡ Keep Cache Warm']) {
  const i = termHooks.indexOf(`name: "${en}"`);
  const tail = termHooks.slice(i, i + 500);
  check('F', `terminal 设置项 ${JSON.stringify(en)} 仍是 config:true（翻它是有读者的）`,
    i > 0 && /config:\s*true/.test(tail.slice(0, tail.indexOf('})') + 2)), true);
}
// F7b 本轮**有意**未翻的 styleMenu：它是 GM 专属的样式编辑器入口，不在「玩家可见面」
//     这一刀的范围内。盯住它还在，将来补译时不至于忘了它。
check('F', 'terminal 的 styleMenu 仍在、且仍是本轮范围外的英文（记账用）',
  termHooks.includes('name: "🎨 Styles"') && termHooks.includes('label: "Edit Styles"'), true);
check('F', 'terminal 的 styleMenu 三串确实**没有**被我们收进通道 A',
  ['🎨 Styles', 'Edit Styles', 'Create custom styling for your terminals']
    .filter(k => k in P.LITERAL_LABELS), []);

// F8 CJK 缺陷之一：generateTable 的 padEnd。我们决定**不修**，判据有二，都要盯住。
check('F', 'generateTable 仍是 initCLI 内的局部函数声明（既不导出也不挂对象，够不到）',
  /\n      function generateTable\(headers, rows\) \{/.test(termJs)
  && !/export\s+(?:default\s+)?function generateTable/.test(termJs)
  && !/\bgenerateTable\s*[:=]/.test(termJs), true);
check('F', 'generateTable 仍用 String.length + padEnd 排版（CJK 会顶歪边框的根源仍在）',
  termJs.includes("cell.toString().padEnd(columnWidths[index], padChar)"), true);
// 我们本轮译出的每一个中文串，都不能出现在 generateTable 的任何调用点上。
{
  const ourCn = [
    ...Object.values(P.TERMINAL_BUTTON_LABELS).map(v => v.cn),
    ...Object.values(P.TERMINAL_ASCII_HEADINGS).map(v => v.cn),
  ];
  const callSites = (termJs.match(/generateTable\([\s\S]{0,200}?\)/g) || []).join('\n');
  check('F', 'generateTable 的调用点里没有任何本轮译文（所以 padEnd 缺陷够不到我们）',
    ourCn.filter(cn => callSites.includes(cn)), []);
  check('F', 'generateTable 确实有调用点（这条判据不是空转）',
    (termJs.match(/generateTable\(/g) || []).length >= 6, true);
}

// F8b terminal 的 CLI 有三处把按钮的 textContent 当**标识符**用（不是当显示文本）：
//     改中文若落在它们身上，会静默改变命令行为（cat 找不到页 / sh 的路径变成空串 /
//     ls 的文件名列变中文再撞上 padEnd）。实测三处读的都**不是**我们改文本的那 4 个类
//     —— 它们读的是 .terminal-journal-page（世界数据）与 .terminal-macro-btn。
{
  const idReadSites = [
    { note: 'cat 按名字找页 (:1255/:1264)', cls: 'terminal-journal-page',
      needle: "const buttons = Array.from(lain.querySelectorAll('.terminal-journal-page'));" },
    { note: 'sh 用按钮名拼 /usr/local/bin 路径 (:1298/:1305)', cls: 'terminal-macro-btn',
      needle: "const btn = lain.querySelector('.terminal-macro-btn');" },
    { note: 'ls 的 FILENAME 列 (:1471/:1478)', cls: 'terminal-journal-page',
      needle: "const btns = lain.querySelectorAll('.terminal-journal-page');" },
  ];
  for (const s of idReadSites) {
    check('F', `CLI 标识符读取点未漂移：${s.note}`, termJs.includes(s.needle), true);
    check('F', `CLI 标识符读取点不在我们改文本的类里：${s.cls}`, s.cls in P.TERMINAL_BUTTON_LABELS, false);
  }
  check('F', 'CLI 里再没有第四处把 textContent 当标识符用（有就要重新盘点）',
    (termJs.match(/norm\(b\.textContent\)|btn\?\.textContent\.replace|el\.textContent\.trim\(\)/g) || []).length, 3);
}

// F9 CJK 缺陷之二：scrambleText 的拉丁字符池。我们用上游自己的判据退出（给 h1 加 span）。
check('F', 'scrambleText 的字符池仍是纯拉丁（中文进去必然抖动，退出策略仍有必要）',
  termJs.includes('scrambleText: { text: original, chars: "abcdefghijklmnopqrstuvwxyz ", ease: "none" }'), true);
check('F', '打 .terminal-typewriter 的判据仍是 children.length === 0（我们的 <span> 退出策略仍成立）',
  termJs.includes('if ((tag === "H1" || tag === "H2" || tag === "H3") && node.children.length === 0)'), true);
check('F', '.terminal-typewriter 仍只被 scramble 那一处消费（加 span 不会带走别的行为）',
  (termJs.match(/terminal-typewriter/g) || []).length, 2);
check('F', 'terminal 的 28 个内置样式仍全部 effectScramble: true（所以这条缺陷是默认开启的）',
  [(termPresetsSrc.match(/effectScramble: true/g) || []).length,
    (termPresetsSrc.match(/effectScramble: false/g) || []).length], [28, 0]);
check('F', '我们的译文 h1 确实带 span（退出策略真的写进了出货代码）',
  P.translateAsciiHeading('<h1 x>Map Downloaded</h1><pre>a</pre>', 'Map Downloaded', '地图已下载')
    .includes('><span>地图已下载</span></h1>'), true);

// F10 本轮**有意**未翻的 config.hbs（9287 字符的 GM 配置说明）。盯住体量，别默默长大。
check('F', 'terminal 的 config.hbs 仍在本轮范围外（体量未变，记账用）',
  read(path.join(TERM_DIR, 'templates/config.hbs')).length, 26354);

/* ══════════════════════════════════════════════════════════════════════
 * S —— 通道 F：getName 译名回退垫片
 *
 * 这一组守的是**唯一一条被主动解冻的 T-FROZEN 串**：
 * `"MU/TH/ER Instructions."`。它冻着的原因从来不是「名字不能变」，而是
 * alienrpg 4.1.13 有 6 处 `game.journal.getName(<英文名>)`，其中 4 处**裸解引用**
 * 返回值。垫片给这条查找加了一层译名回退，于是名字可以译了 —— 代价是多了一条
 * **双向依赖**：垫片没了，译名就必须同时撤掉，否则首次开世界当场抛。
 *
 * 所以这一组分三层：
 *   S-M 机制：垫片赖以成立的 Foundry / 系统源码事实，逐行对本机安装的源码复核。
 *   S-L 联动：垫片里的译名、出货包里的译名、登记表里记的 file:line，三者机械比对。
 *   S-B 行为：**跑真的出货函数** installNameFallback，正例 + 一整排反例。
 * ══════════════════════════════════════════════════════════════════════ */

const { installNameFallback } = patcher;
const { NAME_FALLBACKS, GETNAME_FLAG } = __TEST__;
const SHIM_EN = NAME_FALLBACKS.journal[0].en;
const SHIM_CN = NAME_FALLBACKS.journal[0].cn;

/* ---------------- S-M 机制 ---------------- */

// S-M1 原方法的形状。垫片假设 getName 住在 Collection.prototype 上、按 `e.name === name`
//      线性查找、并接受 {strict} 选项（strict 未命中时**抛异常**）。上游改任何一条，
//      「先 soft 查一遍、失败再把 strict 交回去」这套就不成立了。
const COLLECTION_SRC = read(path.join(CORE, 'common/utils/collection.mjs'));
check('S', 'core 的 Collection#getName 仍是 (name, {strict=false}={}) 签名',
  COLLECTION_SRC.includes('getName(name, {strict=false}={}) {'), true);
check('S', 'core 的 Collection#getName 仍按 e.name === name 线性查找',
  COLLECTION_SRC.includes('const entry = this.find(e => e.name === name);'), true);
check('S', 'core 的 Collection#getName 在 strict 未命中时仍抛异常（我们把 strict 原样交回去才有意义）',
  /if \( strict && \(entry === undefined\) \) \{\s*\n\s*throw new Error/.test(COLLECTION_SRC), true);

// S-M2 getName 只有这**一个**实现。DocumentCollection / WorldCollection / Journal 都没有
//      自己的覆盖 —— 所以在实例上挂一个自有属性，遮蔽的就是全部实现，不会漏掉一层。
for (const f of ['client/documents/abstract/document-collection.mjs',
  'client/documents/abstract/world-collection.mjs',
  'client/documents/collections/journal.mjs']) {
  check('S', `${f} 仍未自己实现 getName（实例遮蔽即可覆盖全部实现）`,
    /(^|\n)\s{2}(?:async\s+)?getName\s*\(/.test(read(path.join(CORE, f))), false);
}

// S-M3 装在 setup 的前提：game.journal 必须**先于** callAll("setup") 被造出来，
//      而系统那几处活的调用都在 ready 里。三行的相对顺序是承重的。
const GAME_SRC = read(path.join(CORE, 'client/game.mjs'));
const gameLines = GAME_SRC.split('\n');
const lineOf = (needle, from = 0) => {
  for (let i = from; i < gameLines.length; i++) if (gameLines[i].includes(needle)) return i + 1;
  return -1;
};
const L_INITDOCS = lineOf('this.initializeDocuments();         // Initialize world documents');
const L_SETUP = lineOf('Hooks.callAll("setup");');
const L_READY = lineOf('Hooks.callAll("ready");');
check('S', 'client/game.mjs 里 initializeDocuments() 仍在 callAll("setup") 之前（setup 时拿得到 game.journal）',
  L_INITDOCS > 0 && L_SETUP > 0 && L_INITDOCS < L_SETUP, true);
check('S', 'client/game.mjs 里 callAll("setup") 仍在 callAll("ready") 之前（我们一定早于系统的 ready 监听器）',
  L_SETUP > 0 && L_READY > 0 && L_SETUP < L_READY, true);
check('S', 'game.journal 的集合名仍由 initializeDocuments 的 initOrder 造出（JournalEntry 仍在表内）',
  /const initOrder = \[[^\]]*"JournalEntry"/.test(GAME_SRC), true);

// S-M4 哨兵唯一。两份补丁文件曾共用 `__alienCnPatched`，后装的那份静默 return，
//      一条译文都不生效。这条判据不许它再发生一次。
const patcherSrcS = read(path.join(HUB, 'scripts/alienrpg-hardcoded-cn.mjs'));
const pluginsSrcS = read(path.join(HUB, 'scripts/plugins-hardcoded-cn.mjs'));
check('S', '哨兵 GETNAME_FLAG 与 NOTIFY_FLAG 不同名', GETNAME_FLAG !== __TEST__.NOTIFY_FLAG, true);
check('S', '哨兵只在本文件出现，plugins-hardcoded-cn.mjs 没有同名图章',
  pluginsSrcS.includes(GETNAME_FLAG), false);
{
  // 语料是本机 Data/modules + Data/systems + core（本汉化中枢自己不在里面 —— 它跑在 VPS 上，
  // 本地没装）。所以这条问的正是：**别人**有没有用同一个图章名。
  const owners = new Set();
  for (const [f, src] of SRC) if (src.includes(GETNAME_FLAG)) owners.add(pkgOf(f));
  check('S', `全机语料里没有第二个包用同名哨兵 ${GETNAME_FLAG}`, [...owners].sort(), []);
}

// S-M5 只装一处，且只装在 game.journal 上。多一个安装点就要重新盘爆炸半径。
{
  const calls = [...patcherSrcS.matchAll(/^\s*installNameFallback\(([^)]*)\)/gm)].map(m => m[1].trim());
  check('S', 'installNameFallback 在出货代码里只有一个调用点', calls.length, 1);
  check('S', '唯一的调用点就是 game.journal', calls[0], 'globalThis.game?.journal, NAME_FALLBACKS.journal');
}
check('S', '垫片没有去动 Collection.prototype（爆炸半径限定在 game.journal 这一个实例）',
  /Collection\.prototype\.getName\s*=/.test(patcherSrcS), false);
check('S', '安装点挂在 setup 钩子上',
  /HOOKS\.once\('setup', \(\) => \{\s*\n\s*installNameFallback\(globalThis\.game\?\.journal/.test(patcherSrcS), true);

// S-M6 上游那 6 处调用点仍在。垫片是替它们兜底的；它们要是没了，垫片就该跟着撤。
{
  const initSrc = read(path.join(SYS, 'module/apps/init.mjs'));
  const mainSrc = read(path.join(SYS, 'module/alienrpg.mjs'));
  const migSrc = read(path.join(SYS, 'module/apps/migratefolders.js'));
  const count = (s, n) => s.split(n).length - 1;
  check('S', 'init.mjs 仍声明 welcomeJournalEntry 为那一串',
    initSrc.includes(`export const welcomeJournalEntry = "${SHIM_EN}"`), true);
  check('S', 'init.mjs 仍有 2 处 getName(welcomeJournalEntry).show() 裸解引用',
    count(initSrc, 'game.journal.getName(welcomeJournalEntry).show()'), 2);
  check('S', 'alienrpg.mjs 仍声明 releaseNoteName 为那一串',
    mainSrc.includes(`const releaseNoteName = "${SHIM_EN}";`), true);
  check('S', 'alienrpg.mjs 仍有 3 处 getName(releaseNoteName)',
    count(mainSrc, 'game.journal.getName(releaseNoteName)'), 3);
  check('S', 'alienrpg.mjs:592 的 .id 仍是裸解引用（垫片正是替它兜底）',
    mainSrc.includes('const selected = game.journal.getName(releaseNoteName).id;'), true);
  check('S', 'showReleaseNotes 的 catch 仍是空的（不兜底就静默失败，所以必须兜）',
    /\} catch \(error\) \{\s*\n\s*\/\/ logger\.debug\('Error', error\);\s*\n\s*\}/.test(mainSrc), true);
  check('S', 'migratefolders.js:120 仍是 getName(<英文名>).show() 裸解引用',
    migSrc.includes(`game.journal.getName("${SHIM_EN}").show()`), true);
}

/* ---------------- S-L 联动 ---------------- */

// S-L1 垫片里的英文 = 上游源码里的字面量。抄错一个字节，回退就永远不触发。
check('S', '垫片的 en 与上游字面量逐字节相等', SHIM_EN, 'MU/TH/ER Instructions.');

// S-L2 垫片里的 cn = 出货包里的日志名 = 出货包里的页名。这三处任一漂移，
//      回退就会去查一个不存在的名字，等于没装垫片。
const CNPACK = JSON.parse(read(path.join(HUB, 'compendium/cn/alienrpg.alien-rpg-system.json')));
const cnJournal = CNPACK.entries['Alien RPG System'].journals[SHIM_EN];
check('S', '出货包的日志名 === 垫片的 cn', cnJournal.name, SHIM_CN);
check('S', '出货包的页名 === 垫片的 cn', cnJournal.pages[SHIM_EN].name, SHIM_CN);
check('S', 'Babele 的键仍是英文原串（键是查找键，永远不译）',
  Object.keys(CNPACK.entries['Alien RPG System'].journals), [SHIM_EN]);

// S-L3 MU/TH/ER 这个词元保持 ASCII —— 船载电脑的名字，MU-TH-UR 插件里也这么写。
check('S', '译名里的 MU/TH/ER 词元仍是 ASCII', SHIM_CN.startsWith('MU/TH/ER '), true);
check('S', '译名里除 MU/TH/ER 词元外全是中文',
  /^MU\/TH\/ER [一-鿿]+$/.test(SHIM_CN), true);

// S-L4 工作区切片是包里那两个名字的**唯一**上游。改包不改切片，下一次
//      prep_sys_units.py --collect 就会把译名洗回英文。
{
  const slicePath = path.join(PROJ, '6-工作区/phase2/SYS-J-name.cn.json');
  check('S', '工作区切片 SYS-J-name.cn.json 存在', fs.existsSync(slicePath), true);
  const slice = JSON.parse(read(slicePath));
  check('S', '切片的 journal_name === 垫片的 cn', slice.journal_name, SHIM_CN);
  check('S', '切片的 page_name === 垫片的 cn', slice.page_name, SHIM_CN);
  const prep = read(path.join(PROJ, '4-常用脚本/parallel/prep_sys_units.py'));
  check('S', 'prep_sys_units.py --collect 仍从切片取 journal_name / page_name',
    prep.includes('journal_name = jname.get("journal_name"') && prep.includes('page_name = jname.get("page_name"'), true);
}

// S-L5 登记表把**这条依赖**写死了：垫片的 file:line、哨兵名，以及「垫片没了怎么办」。
//      登记表是 build_register.py 从源码 re-derive 的，所以这里只需验证它没说谎。
{
  const REG = JSON.parse(read(path.join(PROJ, '7-其他内容/DO-NOT-TRANSLATE.json')));
  const shimmed = REG.sections.name_lookups.entries.filter(e => e.unfrozen_by_shim);
  check('S', '登记表里恰好 2 条被垫片解冻（welcomeJournalEntry + releaseNoteName）',
    shimmed.map(e => e.id).sort(), ['system.releaseNoteName', 'system.welcomeJournalEntry']);
  check('S', '登记表声明了 T-SHIMMED 这一层', typeof REG.tiers['T-SHIMMED'], 'string');
  check('S', '登记表的 shimmed_count 与实际条数一致',
    REG.sections.name_lookups.shimmed_count, shimmed.length);
  const shimLines = read(path.join(HUB, 'scripts/alienrpg-hardcoded-cn.mjs')).split('\n');
  for (const e of shimmed) {
    const sh = e.unfrozen_by_shim;
    check('S', `${e.id}：登记表记的译名与垫片一致`, sh.translated_to, SHIM_CN);
    check('S', `${e.id}：登记表记的哨兵与出货代码一致`, sh.shim.sentinel, GETNAME_FLAG);
    check('S', `${e.id}：登记表记的垫片文件路径正确`,
      sh.shim.file, '1-系统汉化插件/scripts/alienrpg-hardcoded-cn.mjs');
    // file:line 逐条回解 —— 引用漂了就是红，不是注释烂掉那么轻
    for (const [field, needle] of [
      ['pairs_declared_at', 'const NAME_FALLBACKS = {'],
      ['installer_at', 'export function installNameFallback'],
      ['installed_at', 'installNameFallback(globalThis.game?.journal'],
    ]) {
      const n = Number(String(sh.shim[field]).split(':').pop());
      check('S', `${e.id}：${field} 的行号仍指向 ${JSON.stringify(needle)}`,
        Number.isInteger(n) && n > 0 && (shimLines[n - 1] || '').includes(needle), true);
    }
    check('S', `${e.id}：登记表写明了「垫片一旦移除就退回 T-FROZEN」`,
      /REVERTS TO T-FROZEN/.test(sh.if_shim_removed), true);
    check('S', `${e.id}：登记表点名了要一起撤掉的切片文件`,
      sh.if_shim_removed.includes('SYS-J-name.cn.json'), true);
    check('S', `${e.id}：登记表自检 verified_shim_present / verified_lockstep 均为真`,
      [sh.verified_shim_present, sh.verified_lockstep], [true, true]);
  }
}

/* ---------------- S-B 行为：跑真的出货函数 ---------------- */

/** 自有属性版假集合（对应 ownGetName 分支）。 */
function fakeOwn(names) {
  const docs = names.map((n, i) => ({ name: n, id: 'id' + i }));
  const col = {
    docs,
    calls: [],
    getName(name, { strict = false } = {}) {
      col.calls.push(name);
      const e = docs.find(d => d.name === name);
      if (strict && e === undefined) throw new Error(`An entry with name ${name} does not exist in the collection`);
      return e ?? undefined;
    },
  };
  return col;
}

/** 原型方法版假集合（对应 proto 分支 —— 这才是 Foundry 的真实形状）。 */
class FakeProtoCollection {
  constructor(names) {
    this.docs = names.map((n, i) => ({ name: n, id: 'id' + i }));
    this.calls = [];
  }
  getName(name, { strict = false } = {}) {
    this.calls.push(name);
    const e = this.docs.find(d => d.name === name);
    if (strict && e === undefined) throw new Error(`An entry with name ${name} does not exist in the collection`);
    return e ?? undefined;
  }
}

const savedGame = globalThis.game;
function withGame(g, fn) {
  globalThis.game = g;
  try { return fn(); } finally { globalThis.game = savedGame; }
}
const CN_WORLD = { system: { id: 'alienrpg' }, i18n: { lang: 'cn' } };

// S-B1 正例：那一串**恰好**解析到译名文档。
withGame(CN_WORLD, () => {
  const col = fakeOwn([SHIM_CN, '另一本日志']);
  check('S', '装载成功（中文世界 + alienrpg 系统）', installNameFallback(col, NAME_FALLBACKS.journal), true);
  const got = col.getName(SHIM_EN);
  check('S', '★正例：getName(英文原串) 解析到译名文档', got && got.name, SHIM_CN);
  check('S', '★正例：先按原名查、落空后才用译名查（顺序正确，只多一次查找）',
    col.calls, [SHIM_EN, SHIM_CN]);
});

// S-B2 反例：别的日志名一律不碰。
withGame(CN_WORLD, () => {
  const col = fakeOwn([SHIM_CN, '另一本日志']);
  installNameFallback(col, NAME_FALLBACKS.journal);
  check('S', '反例：不存在的别的名字仍返回 undefined', col.getName('Ship Manifest'), undefined);
  check('S', '反例：查别的名字时不会多查一次（垫片完全不介入）', col.calls, ['Ship Manifest']);
  const other = col.getName('另一本日志');
  check('S', '反例：存在的别的名字原样返回', other && other.name, '另一本日志');
});

// S-B3 反例：原查找命中时垫片不介入 —— 英文世界、以及已经按英文名导入过的旧世界，
//      行为与没装垫片时**逐位相同**。
withGame(CN_WORLD, () => {
  const col = fakeOwn([SHIM_EN, SHIM_CN]);  // 两本都在：必须拿到英文那本
  installNameFallback(col, NAME_FALLBACKS.journal);
  const got = col.getName(SHIM_EN);
  check('S', '★反例：原查找命中时返回原文档（不是译名文档）', got && got.name, SHIM_EN);
  check('S', '★反例：原查找命中时不发生第二次查找', col.calls, [SHIM_EN]);
});

// S-B4 反例：别的模块拿 game.journal.getName 干别的事，结果必须与没装垫片时逐个相等。
withGame(CN_WORLD, () => {
  const names = [SHIM_CN, 'Ship Manifest', '另一本日志', ''];
  const wrapped = fakeOwn(names);
  const bare = fakeOwn(names);
  installNameFallback(wrapped, NAME_FALLBACKS.journal);
  const probes = ['Ship Manifest', '另一本日志', SHIM_CN, '', 'nope', 'MU/TH/ER Instructions',
    'mu/th/er instructions.', ' MU/TH/ER Instructions.'];
  const a = probes.map(n => { const d = wrapped.getName(n); return d ? d.name : String(d); });
  const b = probes.map(n => { const d = bare.getName(n); return d ? d.name : String(d); });
  check('S', '★反例：8 个旁路名字的结果与未打补丁时逐个相同（含大小写/空格近似串）', a, b);
});

// S-B5 反例：非中文世界 / 非 alienrpg 系统 —— 垫片必须是**完全的空操作**：
//      不装、不留自有属性、不盖图章。
for (const [label, g] of [
  ['英文世界（lang=en）', { system: { id: 'alienrpg' }, i18n: { lang: 'en' } }],
  ['繁中世界（lang=zh-tw）', { system: { id: 'alienrpg' }, i18n: { lang: 'zh-tw' } }],
  ['别的系统（system=pf2e）', { system: { id: 'pf2e' }, i18n: { lang: 'cn' } }],
  ['game 还没装配好', undefined],
]) {
  withGame(g, () => {
    const col = fakeOwn([SHIM_CN]);
    const before = col.getName;
    check('S', `★${label}：installNameFallback 返回 false（没装）`,
      installNameFallback(col, NAME_FALLBACKS.journal), false);
    check('S', `${label}：getName 仍是原来那个函数（没被替换）`, col.getName === before, true);
    check('S', `${label}：没有盖上哨兵图章`, Boolean(col.getName[GETNAME_FLAG]), false);
    check('S', `${label}：英文原串仍查不到（垫片真的没生效）`, col.getName(SHIM_EN), undefined);
  });
}

// S-B6 防重复包装：热重载 / 双重加载时第二次必须是空操作，而不是包第二层。
withGame(CN_WORLD, () => {
  const col = fakeOwn([SHIM_CN]);
  check('S', '首次装载返回 true', installNameFallback(col, NAME_FALLBACKS.journal), true);
  const first = col.getName;
  check('S', '★二次装载返回 false（不叠第二层）', installNameFallback(col, NAME_FALLBACKS.journal), false);
  check('S', '二次装载后 getName 仍是第一层那个函数', col.getName === first, true);
  col.calls.length = 0;
  col.getName(SHIM_EN);
  check('S', '二次装载后仍只多查一次（没被包两层）', col.calls, [SHIM_EN, SHIM_CN]);
});

// S-B7 strict 语义原样保留。core 的 getName 在 strict 未命中时抛异常；
//      我们先 soft 查，回退也落空时再把**调用方自己的 options** 交回去，让上游抛它自己的错。
withGame(CN_WORLD, () => {
  const col = fakeOwn([SHIM_CN]);
  installNameFallback(col, NAME_FALLBACKS.journal);
  let threw = null;
  try { col.getName('Ship Manifest', { strict: true }); } catch (e) { threw = e.message; }
  check('S', '★strict：不相干的名字未命中时仍然抛（错误信息来自上游）',
    threw, 'An entry with name Ship Manifest does not exist in the collection');
  let got, err = null;
  try { got = col.getName(SHIM_EN, { strict: true }); } catch (e) { err = e.message; }
  check('S', '★strict：英文原串走回退命中时不抛，返回译名文档', [err, got && got.name], [null, SHIM_CN]);
});
withGame(CN_WORLD, () => {
  const col = fakeOwn([]);  // 两本都没有
  installNameFallback(col, NAME_FALLBACKS.journal);
  let threw = null;
  try { col.getName(SHIM_EN, { strict: true }); } catch (e) { threw = e.message; }
  check('S', '★strict：英文原串与译名都不存在时仍然抛（不吞异常）',
    threw, `An entry with name ${SHIM_EN} does not exist in the collection`);
  const soft = col.getName(SHIM_EN);
  check('S', 'strict:false 且两本都没有时返回 undefined（不是 null，与上游一致）', soft, undefined);
});

// S-B8 原型分支：Foundry 的真实形状是 getName 住在原型上。装完之后
//      **原型不能被动过** —— 动了就等于把 game.actors / 每个合集包一起改了。
withGame(CN_WORLD, () => {
  const protoBefore = FakeProtoCollection.prototype.getName;
  const col = new FakeProtoCollection([SHIM_CN]);
  check('S', '原型分支：装载成功', installNameFallback(col, NAME_FALLBACKS.journal), true);
  const got = col.getName(SHIM_EN);
  check('S', '★原型分支：回退命中，且 this 绑定正确（拿得到 this.docs）', got && got.name, SHIM_CN);
  check('S', '★原型分支：Collection 原型上的 getName 一个字节都没动',
    FakeProtoCollection.prototype.getName === protoBefore, true);
  const sibling = new FakeProtoCollection([SHIM_CN]);
  check('S', '★原型分支：同类的**别的实例**完全没被波及（这就是不碰原型的意义）',
    sibling.getName(SHIM_EN), undefined);
});

// S-B9 挂上去的自有属性必须**不可枚举**：集合会被遍历/序列化，多一个可枚举方法
//      就可能被别人当数据处理。
withGame(CN_WORLD, () => {
  const col = new FakeProtoCollection([SHIM_CN]);
  const keysBefore = Object.keys(col).sort();
  installNameFallback(col, NAME_FALLBACKS.journal);
  check('S', '★装载后 Object.keys 不变（自有属性是不可枚举的）', Object.keys(col).sort(), keysBefore);
  const d = Object.getOwnPropertyDescriptor(col, 'getName');
  check('S', '自有属性描述符：不可枚举、可写、可配置（能被干净地撤掉）',
    [d.enumerable, d.writable, d.configurable], [false, true, true]);
});

// S-B10 脏输入不许炸。上游哪天传个 undefined 进来，垫片不能是那个抛异常的人。
withGame(CN_WORLD, () => {
  const col = fakeOwn([SHIM_CN]);
  installNameFallback(col, NAME_FALLBACKS.journal);
  const out = [];
  for (const bad of [undefined, null, 0, 42, {}, []]) {
    try { out.push(col.getName(bad) === undefined ? 'undefined' : 'doc'); } catch (e) { out.push('THREW'); }
  }
  check('S', '★脏输入（undefined/null/数字/对象/数组）一律安全返回 undefined',
    out, ['undefined', 'undefined', 'undefined', 'undefined', 'undefined', 'undefined']);
});

// S-B11 空表 / 坏参数时装载器自己不装，也不炸。
withGame(CN_WORLD, () => {
  check('S', '没有集合时返回 false', installNameFallback(null, NAME_FALLBACKS.journal), false);
  check('S', '集合没有 getName 时返回 false', installNameFallback({}, NAME_FALLBACKS.journal), false);
  check('S', '空 pairs 表时返回 false（不装一个什么都不做的壳）',
    installNameFallback(fakeOwn([]), []), false);
});

/* ══════════════════════════════════════════════════════════════════════ */
console.log(`\n${'='.repeat(70)}`);
console.log(`PASS ${pass}   FAIL ${failures.length}`);
if (failures.length) {
  console.log('\nFAILURES:');
  for (const f of failures) console.log('  - ' + f);
  process.exit(1);
}
console.log('ALL GREEN');
