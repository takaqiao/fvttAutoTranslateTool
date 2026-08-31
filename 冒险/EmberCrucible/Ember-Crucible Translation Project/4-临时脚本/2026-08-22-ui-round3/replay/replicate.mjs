/**
 * 离线复刻器：不开 Foundry，也把「42 条 crucible 译文被 foundry_chn 顶掉」这件事复现出来，
 * 再把 `2-Crucible汉化插件/lang-reclaim.js` 的回写逻辑接上去，证明它抢得回来、且不误伤。
 *
 * ⚠ 关键：`expandObject` / `mergeObject` / `getProperty` **不是仿写的**，是直接 import
 *   本机 Foundry v14 的 `common/utils/helpers.mjs`。仿写等于把「我以为 Foundry 是这么做的」
 *   当成证据 —— 本项目要的是真身。回写逻辑同理，import 的是**将要发出去的那个文件**。
 *
 * 前置自证分两件（本项目血泪：只断言条数会让「切对条数、读错对象」整轮蒙混过去）：
 *   PRECHECK-A  切对条数：各输入文件的键数 / 裸串数 / 目标键数 = 已知真值；
 *   PRECHECK-B  切对对象：点名的**具体键**必须是点名的**具体值 / 具体类型**，
 *               并且各带一条**已知反例**（例：foundry_chn 的 `SETTINGS` 是对象、不是裸串）。
 *   两组都是硬停（throw），不进通过/失败计数 —— 前置自证挂了，后面的数一个都不能信。
 *
 * 跑法：node replicate.mjs [输出.json]
 */

import fs from 'node:fs';
import { pathToFileURL } from 'node:url';

const FOUNDRY = 'C:/Program Files/Foundry Virtual Tabletop/resources/app';
const DATA = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data';
const PROJ = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project';
const PLUGIN = `${PROJ}/2-Crucible汉化插件`;

const U = await import(pathToFileURL(`${FOUNDRY}/common/utils/helpers.mjs`).href);
const RECLAIM = await import(pathToFileURL(`${PLUGIN}/lang-reclaim.js`).href);
const { expandObject, mergeObject, getProperty } = U;
const { reclaimTranslations } = RECLAIM;

/* ───────────────────────── 记账 ───────────────────────── */

const results = [];
let passed = 0;
let failed = 0;

function check(cond, label, detail) {
  if (cond) { passed += 1; results.push({ ok: true, label }); console.log(`  PASS  ${label}`); }
  else {
    failed += 1;
    results.push({ ok: false, label, detail: detail ?? null });
    console.log(`  FAIL  ${label}${detail === undefined ? '' : `\n        ${JSON.stringify(detail)}`}`);
  }
}

/** 前置自证专用：不通过就当场炸停，不进通过/失败计数。 */
function must(cond, label, detail) {
  if (!cond) throw new Error(`前置自证失败：${label}${detail === undefined ? '' : ` — ${JSON.stringify(detail)}`}`);
  console.log(`  自证 OK  ${label}`);
}

const readJson = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
const sameSet = (a, b) => a.length === b.length && new Set(a).size === new Set([...a, ...b]).size;

/* ───────────────── PRECHECK-A：切对条数 ───────────────── */

console.log('\n══════ PRECHECK-A 切对条数 ══════');

const ourCn = readJson(`${PLUGIN}/lang/cn.json`);
const ourKeys = Object.keys(ourCn);
must(ourKeys.length === 1845, 'crucible-cn lang/cn.json 顶层 1845 键', ourKeys.length);
must(ourKeys.every((k) => typeof ourCn[k] === 'string'),
  'crucible-cn lang/cn.json 顶层值全为字符串',
  ourKeys.filter((k) => typeof ourCn[k] !== 'string'));

const tokenKeys = ourKeys.filter((k) => k === 'TOKEN' || k.startsWith('TOKEN.'));
const warnKeys = ourKeys.filter((k) => k === 'WARNING' || k.startsWith('WARNING.'));
const VICTIMS_EXPECTED = [...tokenKeys, ...warnKeys];
must(tokenKeys.length === 32, 'TOKEN.* 32 条', tokenKeys.length);
must(warnKeys.length === 10, 'WARNING.* 10 条', warnKeys.length);
must(VICTIMS_EXPECTED.length === 42, '目标受害面 42 条', VICTIMS_EXPECTED.length);
must(tokenKeys.filter((k) => k.startsWith('TOKEN.LABELS.')).length === 5, '其中 TOKEN.LABELS.* 5 条');
must(tokenKeys.filter((k) => k.startsWith('TOKEN.MOVEMENT.')).length === 27, '其中 TOKEN.MOVEMENT.* 27 条');

const chn = readJson(`${DATA}/modules/foundry_chn/cn.json`);
const chnBare = Object.keys(chn).filter((k) => typeof chn[k] === 'string');
must(Object.keys(chn).length === 177, 'foundry_chn/cn.json 顶层 177 键', Object.keys(chn).length);
must(chnBare.length === 98, 'foundry_chn/cn.json 顶层裸字符串 98 条', chnBare.length);

const cruEn = readJson(`${DATA}/systems/crucible/lang/en.json`);
must(Object.keys(cruEn).length === 51, 'crucible 系统 lang/en.json 顶层 51 键', Object.keys(cruEn).length);

/* ───────────────── PRECHECK-B：切对对象（各带已知反例） ───────────────── */

console.log('\n══════ PRECHECK-B 切对对象（不是只切对条数） ══════');

must(chn.TOKEN === '指示物', '已知真值：foundry_chn.TOKEN 是裸串「指示物」', chn.TOKEN);
must(chn.WARNING === '警告', '已知真值：foundry_chn.WARNING 是裸串「警告」', chn.WARNING);
must(typeof chn.CONTROLS === 'object' && Object.keys(chn.CONTROLS).length === 248,
  '已知反例①：foundry_chn.CONTROLS 是 248 键的嵌套对象、不是裸串',
  typeof chn.CONTROLS === 'object' ? Object.keys(chn.CONTROLS).length : typeof chn.CONTROLS);
must(!chnBare.includes('SETTINGS') && typeof chn.SETTINGS === 'object',
  '已知反例②：foundry_chn.SETTINGS 是对象、没被误判成裸串');
must(Object.keys(chn).filter((k) => k.startsWith('TOKEN.') || k.startsWith('WARNING.')).length === 0,
  '肇事文件里 TOKEN. / WARNING. 点号键 0 条（所以它顶掉之后什么都补不回来）');

must(ourCn['TOKEN.MOVEMENT.ACTIONS.walk.label'] === '行走',
  '已知真值：我们译的 TOKEN.MOVEMENT.ACTIONS.walk.label = 行走', ourCn['TOKEN.MOVEMENT.ACTIONS.walk.label']);
must(ourCn['WARNING.NoParty'] === '你必须先为此配置一个主要队伍。',
  '已知真值：我们译的 WARNING.NoParty', ourCn['WARNING.NoParty']);
must(ourCn['ACTION.TAGS.Context'] !== undefined,
  '已知反例③：同文件里一条与本案无关的键确实存在（说明读的是整份表，不是切片）');

must(typeof cruEn.TOKEN === 'object' && typeof cruEn.WARNING === 'object',
  '英文侧 crucible/en.json 里 TOKEN / WARNING 本就是命名空间（_fallback 靠它出英文）');

// 设计前提本身也要对着真身验，不能靠记忆
must(getProperty({ 'a.b': 'FLAT', a: { b: 'NESTED' } }, 'a.b') === 'FLAT',
  '真身 getProperty：顶层整串点号键走 `key in object` 快路径，优先于嵌套路径');
const probeNE = {};
Object.defineProperty(probeNE, 'a.b', { value: 'FLAT', enumerable: false, writable: true, configurable: true });
must(getProperty(probeNE, 'a.b') === 'FLAT', '真身 getProperty：非枚举自有属性照样命中');
must(Object.keys(probeNE).length === 0, '非枚举属性不进 Object.keys（这正是 enumerable:false 想要的）');

/* ───────────────── 载入顺序：复刻 #getTranslations ───────────────── */

console.log('\n══════ 载入顺序 ══════');

// core lang/cn.json：`CONST.CORE_SUPPORTED_LANGUAGES = ["en"]` ⇒ cn 走不到这条，核对一遍
const constsSrc = fs.readFileSync(`${FOUNDRY}/common/constants.mjs`, 'utf8');
must(/CORE_SUPPORTED_LANGUAGES\s*=\s*Object\.freeze\(\["en"\]\)/.test(constsSrc),
  'CORE_SUPPORTED_LANGUAGES = ["en"] ⇒ 核心自己的 cn.json 根本不参与合并');

// 系统 crucible 没有 cn 语言声明 ⇒ 系统这一档 0 份文件
const cruSystem = readJson(`${DATA}/systems/crucible/system.json`);
must((cruSystem.languages ?? []).filter((l) => l.lang === 'cn').length === 0,
  'crucible 系统没有 cn 语言声明（系统这一档 0 份）');

/** 按目录名排序枚举模块，取其 cn 语言文件 —— 复刻 `game.modules.values()` 的字母序。 */
function discoverModuleCnFiles() {
  const out = [];
  for (const dir of fs.readdirSync(`${DATA}/modules`).sort()) {
    const mj = `${DATA}/modules/${dir}/module.json`;
    if (!fs.existsSync(mj)) continue;
    let j;
    try { j = readJson(mj); } catch { continue; }
    for (const l of j.languages ?? []) {
      if (l.lang !== 'cn') continue;
      // `#filterLanguagePaths` 的两个条件：l.system 必须等于当前系统、l.module 必须已启用。
      if (l.system && l.system !== 'crucible') continue;
      const abs = `${DATA}/modules/${dir}/${String(l.path).replace(/^\.\//, '')}`;
      if (!fs.existsSync(abs)) continue;
      out.push({ id: j.id ?? dir, dir, abs });
    }
  }
  return out;
}

const moduleFiles = discoverModuleCnFiles();
// crucible-cn 那一份换成**项目里将要发出去的那一份**（装在 Data 里的是已发布的 0.9.14）
const iCru = moduleFiles.findIndex((m) => m.dir === 'crucible-cn');
must(iCru >= 0, '载入清单里有 crucible-cn');
const installedCru = readJson(moduleFiles[iCru].abs);
must(JSON.stringify(Object.entries(installedCru).sort()) === JSON.stringify(Object.entries(ourCn).sort()),
  '装在 Data 里的 crucible-cn/lang/cn.json 与项目工作树逐键相同（换用工作树那份不改变结论）');
moduleFiles[iCru] = { ...moduleFiles[iCru], abs: `${PLUGIN}/lang/cn.json`, from: 'worktree' };

const iChn = moduleFiles.findIndex((m) => m.dir === 'foundry_chn');
must(iChn >= 0, '载入清单里有 foundry_chn');
must(iCru < iChn, `字母序：crucible-cn(#${iCru}) 排在 foundry_chn(#${iChn}) 之前 —— 后来者才盖得住先来者`);
console.log(`  载入顺序：${moduleFiles.map((m, i) => `${i}:${m.dir}`).join('  ')}`);

/* ───────────────── 合并：真身 expandObject + mergeObject ───────────────── */

function build(files) {
  const t = {};
  for (const f of files) {
    // 复刻 #loadTranslationFile：先 expandObject，再按顺序 mergeObject
    mergeObject(t, expandObject(readJson(f.abs)), { inplace: true });
  }
  return t;
}

const SCEN = {
  full: moduleFiles,                                                  // 本机全部 cn 语言包（最吵的一档）
  min: [moduleFiles[iCru], moduleFiles[iChn]],                        // 只留 crucible-cn + 肇事者
  noOffender: moduleFiles.filter((m) => m.dir !== 'foundry_chn'),     // 拿掉肇事者
};

/**
 * 我们的哪些键在这份 translations 里**根本查不到字符串** —— 这才是「被顶掉」的定义，
 * 也正是屏幕上出英文的那一批（`localize()` 拿不到字符串就走 `_fallback`）。
 * 扫**全部 1845 条**，不只扫那 42 条：这样「恰好 42」才是结论，不是假设。
 */
function deadOf(t) {
  return ourKeys.filter((k) => typeof getProperty(t, k) !== 'string');
}

/**
 * 查得到字符串、但不是我们的译文 —— 别的语言包**合法地**改写了这条键。
 * 这一档必须与「被顶掉」分开数：回写逻辑对它是**故意不动**的。
 */
function overriddenOf(t) {
  return ourKeys.filter((k) => typeof getProperty(t, k) === 'string' && getProperty(t, k) !== ourCn[k]);
}

console.log('\n══════ 复现「被顶掉」 ══════');

const tFull0 = build(SCEN.full);
const tMin0 = build(SCEN.min);
const tNo0 = build(SCEN.noOffender);

// 前置自证：不注入肇事者时必须是 0 —— 否则「42」只说明这个探测器恒返回 42
must(deadOf(tNo0).length === 0,
  '前置自证：拿掉 foundry_chn ⇒ 被顶掉 0 条（探测器不是恒真）', deadOf(tNo0).slice(0, 10));

check(deadOf(tMin0).length === 42 && sameSet(deadOf(tMin0), VICTIMS_EXPECTED),
  '场景 min（crucible-cn + foundry_chn）：被顶掉的恰好是那 42 条',
  { n: deadOf(tMin0).length, diff: deadOf(tMin0).filter((k) => !VICTIMS_EXPECTED.includes(k)) });
// ⚠ full 场景里那 42 条**不是全灭**：pf2_cn（字母序在 foundry_chn 之后）自己也译了核心的
//   `TOKEN.MOVEMENT.ACTIONS.*`，它那个 TOKEN 对象把裸串又覆盖回去，顺带救活我们 3 条。
//   其中 jump / walk 与我们译文碰巧同字，blink 它译「传送」、我们译「闪现」。
//   ⇒ 这 3 条在 full 场景里**不该由我们抢**（查得到字符串就不动），下面逐条验。
//   真实的 Crucible 世界不会开 pf2e 的汉化包，所以 min 场景（42 条全灭）才是用户现场的形状；
//   full 场景是刻意做吵，用来验「不误伤别人合法翻译」。
const SURVIVE_IN_FULL = [
  'TOKEN.MOVEMENT.ACTIONS.jump.label',
  'TOKEN.MOVEMENT.ACTIONS.walk.label',
  'TOKEN.MOVEMENT.ACTIONS.blink.label',
];
check(deadOf(tFull0).length === 39 && sameSet([...deadOf(tFull0), ...SURVIVE_IN_FULL], VICTIMS_EXPECTED),
  '场景 full（本机全部 cn 语言包）：那 42 条里 39 条被顶掉，另 3 条被 pf2_cn 的 TOKEN.MOVEMENT.* 覆盖回来了',
  { n: deadOf(tFull0).length, diff: deadOf(tFull0).filter((k) => !VICTIMS_EXPECTED.includes(k)) });
check(deadOf(tNo0).length === 0, '场景 noOffender（无 foundry_chn）：被顶掉 0 条');

// ⚠ 另一档，必须与「被顶掉」分开：本机把**全部** cn 语言包都算进来时，
//   pf2_cn / 5e_chn 会**合法地**改写我们 3 条键（它们查得到字符串，只是不是我们的译文）。
//   真实的 Crucible 世界不会开 pf2e 的汉化包，这一档纯粹是把场景做吵一点 ——
//   它的价值在于证明回写逻辑对「别人合法翻译的键」是**故意不动**的。
const OVERRIDDEN_FULL = overriddenOf(tNo0);
check(sameSet(OVERRIDDEN_FULL, ['TOKEN.MOVEMENT.ACTIONS.blink.label', 'TYPES.Item.ancestry', 'TYPES.Item.spell']),
  '场景 full/noOffender：被别的语言包合法改写的恰好 3 条（pf2_cn 2 条 + 与 5e_chn 共 1 条）',
  OVERRIDDEN_FULL);
check(overriddenOf(tMin0).length === 0, '场景 min：没有第三方合法改写（干净对照组）', overriddenOf(tMin0));

// 机理本身，不只是症状
check(tMin0.TOKEN === '指示物' && tMin0.WARNING === '警告',
  '机理：min 场景下 TOKEN / WARNING 两个命名空间被整块换成裸串',
  { TOKEN: tMin0.TOKEN, WARNING: tMin0.WARNING });
check(typeof tFull0.TOKEN === 'object' && tFull0.WARNING === '警告',
  '机理：full 场景下 TOKEN 又被字母序更靠后的 pf2_cn / zzz_mod_chn 的对象覆盖回对象（里面装的是他们的键，我们的只剩同名那 3 条），WARNING 仍是裸串',
  { TOKEN: typeof tFull0.TOKEN, tokenKeys: typeof tFull0.TOKEN === 'object' ? Object.keys(tFull0.TOKEN) : null });
check(typeof tNo0.TOKEN === 'object' && typeof tNo0.WARNING === 'object',
  '机理：拿掉肇事者后两个命名空间都还是对象');

// 症状：查不到就走英文 fallback，屏幕上看到的是英文原文、不是裸键
const fallbackEn = expandObject(readJson(`${DATA}/systems/crucible/lang/en.json`));
check(getProperty(fallbackEn, 'TOKEN.MOVEMENT.ACTIONS.walk.label') === 'Walk',
  '症状：英文 fallback 里同一条键是 "Walk" ⇒ 玩家看到的是英文原文',
  getProperty(fallbackEn, 'TOKEN.MOVEMENT.ACTIONS.walk.label'));

/* ───────────────── 接上回写逻辑 ───────────────── */

console.log('\n══════ 回写：抢回来 ══════');

/** 深拷贝出「可枚举投影」的规范化快照：JSON.stringify 天然跳过非枚举属性。 */
function enumSnapshot(value) {
  if (Array.isArray(value)) return value.map(enumSnapshot);
  if (value && typeof value === 'object') {
    const out = {};
    for (const k of Object.keys(value).sort()) out[k] = enumSnapshot(value[k]);
    return out;
  }
  return value;
}
const enumJson = (v) => JSON.stringify(enumSnapshot(v));

/** 顶层自有属性清单（含非枚举），带描述符 —— 用来证明「新增了什么、没改什么」。 */
function ownInventory(obj) {
  const out = {};
  for (const k of Object.getOwnPropertyNames(obj).sort()) {
    const d = Object.getOwnPropertyDescriptor(obj, k);
    out[k] = { enumerable: d.enumerable, type: typeof d.value };
  }
  return out;
}

/** 递归收集全部可枚举叶路径 → 值。用于「逐键比对」而不是只比一个 JSON 串。 */
function leafMap(obj, prefix = '', out = new Map()) {
  for (const k of Object.keys(obj)) {
    const v = obj[k];
    const p = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === 'object' && !Array.isArray(v)) leafMap(v, p, out);
    // ⚠ 数组必须序列化再比：两次 build 出来的是**不同的数组实例**，
    //   用 Object.is 比引用会把一堆根本没动过的键判成「变了」（第一版就是这么假红的）。
    else out.set(p, Array.isArray(v) ? `[]${JSON.stringify(v)}` : v);
  }
  return out;
}

function runScenario(name, files, expectN) {
  console.log(`\n── 场景 ${name} ──`);
  const pristine = build(files);        // 只读参照，不动它
  const t = build(files);               // 被回写的那一份
  // 期望抢回哪些键**现算**，不写死 —— 写死等于把结论当输入。
  // 调用方只给「应当是几条」，两边对不上就红。
  const expectReclaimed = deadOf(pristine);
  check(expectReclaimed.length === expectN,
    `[${name}] 回写前「查不到字符串」的恰好 ${expectN} 条`, expectReclaimed.length);
  check(expectReclaimed.every((k) => VICTIMS_EXPECTED.includes(k)),
    `[${name}] 这些键全部落在 TOKEN.* / WARNING.* 这两个被顶掉的命名空间里`,
    expectReclaimed.filter((k) => !VICTIMS_EXPECTED.includes(k)));
  const beforeEnum = enumJson(t);
  const beforeInv = ownInventory(t);
  const beforeLeaves = leafMap(pristine);

  const report = reclaimTranslations(t, ourCn, getProperty);

  // ① 抢回的条数与集合
  check(report.reclaimed.length === expectReclaimed.length && sameSet(report.reclaimed, expectReclaimed),
    `[${name}] 抢回 ${expectReclaimed.length} 条，且恰好是预期那批`,
    { got: report.reclaimed.length, diff: report.reclaimed.filter((k) => !expectReclaimed.includes(k)) });
  check(report.skippedObject.length === 0 && report.skippedBadValue.length === 0,
    `[${name}] 没有「位置被命名空间占用」/「值不是字符串」的跳过项`,
    { skippedObject: report.skippedObject, skippedBadValue: report.skippedBadValue });
  check(report.alreadyString.length === ourKeys.length - expectReclaimed.length,
    `[${name}] 其余 ${ourKeys.length - expectReclaimed.length} 条本来就查得到 ⇒ 一次都没写`,
    report.alreadyString.length);

  // ② 每一条被写的键，写之前确实是 undefined（不是去遮别人已有的东西）
  const notUndefined = report.reclaimed.filter((k) => getProperty(pristine, k) !== undefined);
  check(notUndefined.length === 0, `[${name}] 被写的键在写之前全部是 undefined（没遮任何既有可解析值）`, notUndefined);

  // ③ 回写后我们的键全部查得到字符串；仍与我们译文不同的，恰好是**回写前就被别人合法改写**的那些
  const stillDead = deadOf(t);
  check(stillDead.length === 0, `[${name}] 回写后我们的 ${ourKeys.length} 条键全部查得到字符串（不再走英文 fallback）`, stillDead.slice(0, 10));
  const overriddenBefore = overriddenOf(pristine);
  check(sameSet(overriddenOf(t), overriddenBefore),
    `[${name}] 回写后「不是我们译文」的那批，与回写前逐条相同（${overriddenBefore.length} 条，一条都没抢）`,
    { before: overriddenBefore, after: overriddenOf(t) });

  // ④ 反向不误伤（其一）：可枚举投影逐字节不变 ⇒ 一个既有键都没动
  check(enumJson(t) === beforeEnum, `[${name}] 回写前后「可枚举投影」完全相等（既有键一个没动）`);

  // ④' 反向不误伤（其二）：逐键比对，不靠一个 JSON 串
  const afterLeaves = leafMap(t);
  const changed = [];
  for (const [p, v] of beforeLeaves) if (!Object.is(afterLeaves.get(p), v)) changed.push(p);
  for (const p of afterLeaves.keys()) if (!beforeLeaves.has(p)) changed.push(`+${p}`);
  check(changed.length === 0, `[${name}] 逐键比对：${beforeLeaves.size} 条既有叶路径无一变化、无一新增`, changed.slice(0, 10));

  // ⑤ 新增的自有属性 = 恰好那批键，且全部非枚举、全部属于我们的 cn.json
  const afterInv = ownInventory(t);
  const added = Object.keys(afterInv).filter((k) => !(k in beforeInv));
  const mutated = Object.keys(beforeInv).filter((k) => JSON.stringify(beforeInv[k]) !== JSON.stringify(afterInv[k]));
  check(sameSet(added, expectReclaimed), `[${name}] 新增自有属性恰好是那批键`, { added: added.length });
  check(mutated.length === 0, `[${name}] 既有自有属性的描述符一个都没变`, mutated);
  check(added.every((k) => afterInv[k].enumerable === false), `[${name}] 新增的键全部是非枚举`);
  check(added.every((k) => Object.hasOwn(ourCn, k)), `[${name}] 新增的键全部出自我们自己的 lang/cn.json（反向不误伤）`,
    added.filter((k) => !Object.hasOwn(ourCn, k)));

  // ⑥ 幂等：再跑一次，写入 0 次，且对象逐属性不变
  const invBeforeSecond = ownInventory(t);
  const enumBeforeSecond = enumJson(t);
  const report2 = reclaimTranslations(t, ourCn, getProperty);
  check(report2.reclaimed.length === 0 && report2.alreadyString.length === ourKeys.length,
    `[${name}] 幂等：第二次跑写入 0 次、${ourKeys.length} 条全部落在「已经查得到」`,
    { reclaimed: report2.reclaimed.length, alreadyString: report2.alreadyString.length });
  check(JSON.stringify(ownInventory(t)) === JSON.stringify(invBeforeSecond) && enumJson(t) === enumBeforeSecond,
    `[${name}] 幂等：第二次跑前后对象完全相同`);

  return { name, files: files.map((f) => f.dir), report: { reclaimed: report.reclaimed.length, alreadyString: report.alreadyString.length, skippedObject: report.skippedObject, skippedBadValue: report.skippedBadValue } };
}

const scenarioOut = [
  runScenario('① full（有 foundry_chn）', SCEN.full, 39),        // 42 减去 pf2_cn 覆盖回来的 3 条
  runScenario('① min（有 foundry_chn）', SCEN.min, 42),          // 干净的两方对照：42 条全灭
  runScenario('② noOffender（无 foundry_chn）', SCEN.noOffender, 0), // 期望抢回 0 条 ⇒ 纯 no-op
];

/* ───────────────── enumerable:false 的实测理由 ───────────────── */

console.log('\n══════ 为什么必须 enumerable:false（热重载通道实测） ══════');

// client/game.mjs:#hotReloadJSON 会 mergeObject(game.i18n.translations, translations)，
// 而 mergeObject 在 _d===0 时若发现 original 有点号键就整份 expandObject。
const tEnum = build(SCEN.min);
tEnum['TOKEN.MOVEMENT.ACTIONS.walk.label'] = '行走';   // 可枚举写法
let enumErr = null;
try { mergeObject(tEnum, {}, { inplace: true }); } catch (err) { enumErr = err; }
check(enumErr instanceof TypeError,
  '可枚举的扁平键 ⇒ 真身 mergeObject 走 expandObject 分支，在 `TOKEN` 已是字符串处抛 TypeError',
  enumErr ? `${enumErr.name}: ${enumErr.message}` : '没抛');

const tNonEnum = build(SCEN.min);
reclaimTranslations(tNonEnum, ourCn, getProperty);
let nonEnumErr = null;
try { mergeObject(tNonEnum, {}, { inplace: true }); } catch (err) { nonEnumErr = err; }
check(nonEnumErr === null, '非枚举的扁平键 ⇒ 同一次 mergeObject 不抛',
  nonEnumErr ? `${nonEnumErr.name}: ${nonEnumErr.message}` : null);
check(getProperty(tNonEnum, 'TOKEN.MOVEMENT.ACTIONS.walk.label') === '行走',
  '非枚举的扁平键 ⇒ 过完 mergeObject 之后照样查得到');

/* ───────────────── lang/cn.json 结构闸（本轮没改它，复核一遍） ───────────────── */

console.log('\n══════ lang/cn.json 结构 ══════');

check(ourKeys.filter((k) => typeof ourCn[k] !== 'string').length === 0, '顶层非字符串值 0 条');
const prefixClash = ourKeys.filter((k) => ourKeys.some((j) => j !== k && j.startsWith(`${k}.`)));
check(prefixClash.length === 0, 'raw 键点号前缀相撞 0 条', prefixClash.slice(0, 10));

/* ───────────────── 汇总 ───────────────── */

const summary = {
  generated: new Date().toISOString(),
  foundry: FOUNDRY,
  loadOrder: moduleFiles.map((m) => m.dir),
  victimsExpected: VICTIMS_EXPECTED,
  scenarios: scenarioOut,
  passed,
  failed,
  results,
};
const outPath = process.argv[2] ?? 'replicate_out.json';
fs.writeFileSync(outPath, JSON.stringify(summary, null, 1), 'utf8');

console.log(`\n══════ 合计：${passed} 通过 / ${failed} 失败 ══════`);
console.log(`写出 ${outPath}`);
process.exit(failed === 0 ? 0 : 1);
