/**
 * 离线验「面板新加的『命名空间被顶 · 抢回』这一节，在六种情形下报的话对不对」。
 *
 * ⚠ 三样东西都用真身，不仿写：
 *   · `expandObject / mergeObject / getProperty` —— 本机 Foundry v14 的 common/utils/helpers.mjs
 *   · `reclaimTranslations` —— 将要发出去的 2-Crucible汉化插件/lang-reclaim.js
 *   · `checkLangSquat` —— 将要发出去的自检文件本身（临时补一行 export 再 import，原文件不动）
 *   仿写等于把「我以为它是这么写的」当证据。本项目栽过。
 *
 * 前置自证：真实语料跑出来的被顶条数必须正好 42，且根正好是 TOKEN / WARNING —— 不对就炸停。
 */
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const FOUNDRY = 'C:/Program Files/Foundry Virtual Tabletop/resources/app';
const DATA = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data';
const PROJ = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project';
const PLUGIN = `${PROJ}/2-Crucible汉化插件`;
const SELFCHECK = `${PLUGIN}/selfcheck/cn-selfcheck.mjs`;

const U = await import(pathToFileURL(`${FOUNDRY}/common/utils/helpers.mjs`).href);
const { expandObject, mergeObject, getProperty } = U;
const { reclaimTranslations } = await import(pathToFileURL(`${PLUGIN}/lang-reclaim.js`).href);

// 自检文件里 checkLangSquat 是内部函数：复制一份、尾部补 export 再 import，真文件一个字节不动。
const scSrc = fs.readFileSync(SELFCHECK, 'utf8');
const scTmp = path.join(process.cwd(), '_sc.mjs');
fs.writeFileSync(scTmp, scSrc + '\nexport { checkLangSquat, I18N_PROBES, Check };\n', 'utf8');
const SC = await import(pathToFileURL(scTmp).href);
fs.unlinkSync(scTmp);

let pass = 0, fail = 0;
const bad = [];
const ck = (cond, label, detail) => {
  if (cond) { pass++; console.log('  PASS  ' + label); }
  else { fail++; bad.push({ label, detail }); console.log('  FAIL  ' + label + (detail === undefined ? '' : '  ' + JSON.stringify(detail))); }
};
const must = (cond, label, detail) => {
  if (!cond) throw new Error('前置自证失败：' + label + (detail === undefined ? '' : ' — ' + JSON.stringify(detail)));
  console.log('  自证 OK  ' + label);
};

/* ── 造真实的「被顶」现场 ─────────────────────────────────────────────── */
const ourCn = JSON.parse(fs.readFileSync(`${PLUGIN}/lang/cn.json`, 'utf8'));
const chnPath = `${DATA}/modules/foundry_chn/cn.json`;
const haveChn = fs.existsSync(chnPath);
must(haveChn, 'foundry_chn 语料在本机（否则这一验没有真实肇事者可用）', chnPath);
const chnCn = JSON.parse(fs.readFileSync(chnPath, 'utf8'));

function buildTranslations({ withSquatter }) {
  const t = {};
  mergeObject(t, expandObject(ourCn), { inplace: true });          // crucible-cn 字母序在前
  if (withSquatter) mergeObject(t, expandObject(chnCn), { inplace: true });
  return t;
}

const squatted = buildTranslations({ withSquatter: true });
const report = reclaimTranslations(squatted, ourCn, getProperty);
must(report.reclaimed.length === 42, '真实语料下被顶条数 = 42', report.reclaimed.length);
must(JSON.stringify([...new Set(report.reclaimed.map(k => k.split('.')[0]))].sort()) === '["TOKEN","WARNING"]',
  '被顶的根正好是 TOKEN / WARNING', [...new Set(report.reclaimed.map(k => k.split('.')[0]))]);

/* ── 造 game 假身：localize 走真 getProperty，取不到就吐回键名（与 core 同形） ── */
function makeGame(translations, pkg) {
  return {
    modules: { get: (id) => (id === 'crucible-cn' ? pkg : undefined) },
    i18n: {
      translations,
      localize(key) {
        const v = getProperty(translations, key);
        return typeof v === 'string' ? v : key;
      },
    },
  };
}
const mkPkg = (state, version = '0.9.16') => ({
  active: true, version,
  api: state === null ? {} : { getReclaimState: () => state },
});
const run = (translations, pkg) => {
  globalThis.game = makeGame(translations, pkg);
  return SC.checkLangSquat('C i18n 通道', translations);
};
const joined = (r) => r.map(c => c.detail + ' || ' + (c.items ?? []).join(' || ')).join(' ~~ ');

console.log('\n══════ 六种情形 ══════');

// ① 没装 crucible-cn ⇒ skip（不是通过）
{
  globalThis.game = { modules: { get: () => undefined }, i18n: { translations: {}, localize: k => k } };
  const r = SC.checkLangSquat('C', {});
  ck(r.length === 1 && r[0].status === 'skip', '① 没装 crucible-cn ⇒ skip', r.map(x => x.status));
  ck(/没装或没启用/.test(r[0].detail), '① 说清了为什么无从查起');
}

// ② 装了但没挂账本 ⇒ fail（旧版本 / i18nInit 没跑）
{
  const r = run(squatted, mkPkg(null, '0.9.15'));
  ck(r[0].status === 'fail', '② 装了但没挂账本 ⇒ fail', r.map(x => x.status));
  ck(/0\.9\.15/.test(r[0].detail) && /不是「通过」/.test(r[0].detail), '② 点名了版本、并说明这不是通过');
}

// ③ 抢回器抛了 ⇒ fail 且带上原始消息
{
  const r = run(squatted, mkPkg({ phase: 'error', error: 'HTTP 404 for …/lang/cn.json', report: null, tableSize: 0 }));
  ck(r[0].status === 'fail' && /HTTP 404/.test(r[0].detail), '③ phase=error ⇒ fail 且带原始消息', r[0].detail?.slice(0, 60));
}

// ④ 真实「被顶 + 抢回成功」⇒ ok，条数 42，两个根都点名
{
  const st = { phase: 'done', error: null, report, tableSize: Object.keys(ourCn).length };
  const r = run(squatted, mkPkg(st));
  const main = r[0];
  ck(main.status === 'ok', '④ 被顶 + 抢回成功 ⇒ ok', main.status);
  ck(main.checked === 42, '④ checked = 42（点的是抢回条数，不是笼统的"若干"）', main.checked);
  const j = joined([main]);
  ck(/`TOKEN`/.test(j) && /`WARNING`/.test(j), '④ 两个被顶的根都逐个点名');
  ck(/32 条/.test(j) && /10 条/.test(j), '④ 每个根下的条数分开报（TOKEN 32 / WARNING 10）', j.match(/\*\*\d+ 条\*\*/g));
  ck(/指示物/.test(j) && /警告/.test(j), '④ 把顶层现在被换成的那个裸串原样显示出来');
  ck(new RegExp('42/42').test(j), '④ 报了「复验 42/42 取得到中文」');
}

// ⑤ 没人顶 ⇒ ok，但必须说清「不等于抢回器没用」
{
  const clean = buildTranslations({ withSquatter: false });
  const rep2 = reclaimTranslations(clean, ourCn, getProperty);
  must(rep2.reclaimed.length === 0, '⑤ 无肇事者时写入 0 次', rep2.reclaimed.length);
  const r = run(clean, mkPkg({ phase: 'done', error: null, report: rep2, tableSize: Object.keys(ourCn).length }));
  ck(r[0].status === 'ok' && /一个都没被顶掉/.test(r[0].detail), '⑤ 没人顶 ⇒ ok');
  ck(/不等于/.test(r[0].detail) && new RegExp(String(Object.keys(ourCn).length)).test(r[0].detail),
    '⑤ 报了扫描分母、并声明「0 条 ≠ 抢回器没用」');
}

// ⑥ 账本说抢回了、复验却仍是英文 ⇒ 必须 fail（判据空转第 (h) 形态）
{
  const tampered = buildTranslations({ withSquatter: true });
  // 只跑抢回、再把其中 3 条悄悄改回英文，模拟「写入动作做了但玩家通道上没生效」
  const rep3 = reclaimTranslations(tampered, ourCn, getProperty);
  const victims = rep3.reclaimed.slice(0, 3);
  for (const k of victims) Object.defineProperty(tampered, k, { value: 'Walk', writable: true, enumerable: false, configurable: true });
  const r = run(tampered, mkPkg({ phase: 'done', error: null, report: rep3, tableSize: Object.keys(ourCn).length }));
  ck(r[0].status === 'fail', '⑥ 账本说抢回了但复验是英文 ⇒ fail', r[0].status);
  ck(/仍有 3 条取不到中文/.test(r[0].detail), '⑥ 点了具体条数', r[0].detail?.slice(0, 80));
  ck(victims.every(v => joined([r[0]]).includes(v)), '⑥ 三条都逐条点名');
}

// ⑦ 探针表里确实加了那两条被顶过的键
{
  const keys = SC.I18N_PROBES.map(p => p[0]);
  ck(keys.includes('TOKEN.MOVEMENT.ACTIONS.walk.label') && keys.includes('WARNING.NoParty'),
    '⑦ I18N_PROBES 补了两条被顶过的键', keys);
  for (const [k, want] of SC.I18N_PROBES) {
    if (!(k in ourCn)) continue;
    ck(ourCn[k].includes(want), `⑦ 探针期望值与 cn.json 对得上：${k} 含「${want}」`, ourCn[k]);
  }
}

console.log('\n══════ 合计 ' + pass + ' 通过 / ' + fail + ' 失败 ══════');
if (fail) { console.log(JSON.stringify(bad, null, 1)); process.exit(1); }
