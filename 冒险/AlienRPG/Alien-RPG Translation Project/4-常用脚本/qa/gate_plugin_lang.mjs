/**
 * gate_plugin_lang.mjs —— `lang/plugins/*.json` 六份插件语言文件、七个条件入口的闸门。
 *
 * 为什么要有这个文件：v0.1.0 把四份 `{}` 空壳一路发了出去。发布闸只看 `lang/cn.json`，
 * QA 目录里 **没有任何一个脚本** 碰过 `lang/plugins/`（2026-08-29 实测：
 * `grep -l "lang/plugins" 4-常用脚本/qa/*` 命中 0）。CI 那道闸后来补上了「非空」，
 * 但「非空」离「对」还很远 —— 下面九组判据是把「对」写死。
 *
 * 最要命的一条是 P8。MU/TH/UR 的 `MOTHER.Keywords.*` **不是给人看的文案，是命令词表**：
 *   main.js:1148/1152/1153/1156 —— 全模块仅有的四个读取点，全部形如
 *       orderWords = [ localize('MOTHER.Keywords.Ordre').toUpperCase(), 'ORDER' ]
 *   它们只喂给 isSpecialOrder()/isCerberus() 做 `includes` 判定，**永不渲染**。
 * 而真正解析编号的 handleSpecialOrder()（main.js:4567-4579）用的是 12 条写死的
 * ASCII/法语正则去剥前缀，没有中文分支。于是把 Keywords 译成中文的后果是：
 *   `指令 937` 被 isSpecialOrder() 放行 → 走进特殊指令分支 → orderKey 仍是 `指令 937`
 *   → `orders[orderKey]` 落空 → MOTHER.commandNotFound。
 * 对没 hack 的玩家还会更糟：放行后先撞 MOTHER.AccessDenied，并给 GM 发一条
 * `MUTHUR.SpecialOrderAttempt` 入侵告警 —— 一个中文误输入变成一次假警报。
 * 结论：这四个键必须与上游 en.json 逐字节相同。本闸门盯死它。
 *
 * 用法：node "4-常用脚本/qa/gate_plugin_lang.mjs"
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const PROJ = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const WORKSPACE = path.resolve(PROJ, '..', '..', '..');
const HUB = path.join(PROJ, '1-系统汉化插件');
const MODULES = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules';
const LOCAL_YZE = path.join(WORKSPACE, '模组', 'yearzero-combat-fvtt', 'dist');

const EVOLVED_STUNT_ALIASES = [
  'ALIENRPG.机械', 'ALIENRPG.近战', 'ALIENRPG.耐力', 'ALIENRPG.射击',
  'ALIENRPG.机动', 'ALIENRPG.驾驶', 'ALIENRPG.指挥', 'ALIENRPG.操控',
  'ALIENRPG.医疗', 'ALIENRPG.侦察', 'ALIENRPG.生存', 'ALIENRPG.通信科技',
];

/** 每份插件语言文件的判据表。upstreamLang 为 null 表示上游没有语言文件。 */
const SPEC = {
  'alien-mu-th-ur': {
    upstreamLang: 'lang/en.json',
    srcGlob: ['scripts'],
    namespaces: ['MUTHUR', 'MOTHER'],
    /** 命令词表：必须与上游逐字节相同（见文件头 P8）。 */
    frozenKeys: ['MOTHER.Keywords.Ordre', 'MOTHER.Keywords.Special',
      'MOTHER.Keywords.Special2', 'MOTHER.Keywords.Protocol'],
  },
  motion_tracker: {
    upstreamLang: 'lang/en.json',
    srcGlob: ['scripts', 'module', 'src'],
    namespaces: ['MOTIONTRACKER', 'MotionTracker'],
    frozenKeys: [],
  },
  babele: {
    upstreamLang: 'lang/en.json',
    srcGlob: ['scripts', 'src', 'templates', 'modules'],
    namespaces: ['BABELE'],
    frozenKeys: [],
  },
  'token-action-hud-alien': {
    upstreamLang: 'languages/en.json',
    srcGlob: ['scripts'],
    namespaces: ['tokenActionHud'],
    frozenKeys: [],
    /** 上游 en.json 是 `{"tokenActionHud":{}}`，标签全是 ALIENRPG.* —— 这份必须保持空。 */
    mustBeEmpty: true,
  },
  'alien-evolved-starterset': {
    upstreamLang: 'lang/en.json',
    srcGlob: ['module'],
    namespaces: ['ALIENRPG'],
    frozenKeys: [],
    allowSystemNamespace: true,
    allowedExtraKeys: EVOLVED_STUNT_ALIASES,
  },
  'alien-evolved-corerules': {
    upstreamLang: 'lang/en.json',
    srcGlob: ['module'],
    namespaces: ['ALIENRPG'],
    frozenKeys: [],
    allowSystemNamespace: true,
    allowedExtraKeys: EVOLVED_STUNT_ALIASES,
  },
  'yze-combat': {
    upstreamLang: 'lang/en.json',
    srcGlob: [],
    namespaces: ['COMBAT', 'SETTINGS', 'YZEC'],
    frozenKeys: [],
    moduleDir: LOCAL_YZE,
  },
};

let pass = 0;
const failures = [];
function check(name, actual, expected) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) { pass += 1; console.log(`  PASS ${name}`); }
  else { failures.push(`${name}\n      expected ${e}\n      actual   ${a}`); console.log(`  FAIL ${name}`); }
}
const read = p => fs.readFileSync(p, 'utf8');
const flat = (d, p = '') => Object.entries(d).reduce((o, [k, v]) => {
  const kk = p ? `${p}.${k}` : k;
  if (v && typeof v === 'object' && !Array.isArray(v)) Object.assign(o, flat(v, kk)); else o[kk] = v;
  return o;
}, {});
const placeholders = s => (typeof s === 'string'
  ? [...new Set([...s.matchAll(/\{([A-Za-z0-9_.]+)\}/g)].map(m => m[1]))].sort() : []);

/** 原文里重复的键：JSON.parse 静默保留最后一个，只能扫原文。 */
function duplicateKeys(raw) {
  const dups = []; const scopes = [new Set()]; let i = 0;
  while (i < raw.length) {
    const c = raw[i];
    if (c === '"') {
      let j = i + 1, s = '';
      while (j < raw.length) {
        if (raw[j] === '\\') { s += raw[j + 1]; j += 2; continue; }
        if (raw[j] === '"') break;
        s += raw[j]; j += 1;
      }
      let k = j + 1; while (k < raw.length && /\s/.test(raw[k])) k += 1;
      if (raw[k] === ':') {
        const cur = scopes[scopes.length - 1];
        if (cur.has(s)) dups.push(s); else cur.add(s);
      }
      i = j + 1; continue;
    }
    if (c === '{') { scopes.push(new Set()); i += 1; continue; }
    if (c === '}') { scopes.pop(); i += 1; continue; }
    i += 1;
  }
  return dups;
}

/** 插件自己的源码（不含 lang 目录）—— 用来判定「这个键有没有读者」。 */
function pluginSource(modDir, globs) {
  const out = [];
  const walk = d => {
    let ents; try { ents = fs.readdirSync(d, { withFileTypes: true }); } catch { return; }
    for (const e of ents) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) {
        if (/^(node_modules|\.git|lang|languages|i18n)$/i.test(e.name)) continue;
        walk(p);
      } else if (/\.(js|mjs|cjs|hbs|html|htm)$/i.test(e.name)) {
        try { out.push(read(p)); } catch { /* locked */ }
      }
    }
  };
  for (const g of globs) walk(path.join(modDir, g));
  walk(modDir);
  return out.join('\n');
}

console.log('gate_plugin_lang —— lang/plugins/*.json 闸门\n');

const manifest = JSON.parse(read(path.join(HUB, 'module.json')));
const declared = manifest.languages.filter(l => l.path.startsWith('lang/plugins/'));

check('P0 module.json 声明了七个条件式插件语言入口', declared.length, 7);
check('P0 每一条都带 module 门（少一个门 = 没装插件也写全局表）',
  declared.filter(l => !l.module).map(l => l.path), []);
check('P0 每一条的 lang 都是 cn', [...new Set(declared.map(l => l.lang))], ['cn']);

const allFlat = {};   // 跨文件撞键用
const ownerOf = {};

for (const l of declared) {
  const spec = SPEC[l.module];
  console.log(`\n── ${l.path}   (module 门: ${l.module}) ──`);
  if (!spec) { failures.push(`未知的 module 门 ${l.module}，判据表里没有它`); console.log('  FAIL 判据表缺项'); continue; }

  /* P1 —— module 门必须与已装模块的 id 逐字节相同。抄错一个字节 = 文件永不装载。 */
  const modDir = spec.moduleDir ?? path.join(MODULES, l.module);
  const modManifest = path.join(modDir, 'module.json');
  if (!fs.existsSync(modManifest)) {
    failures.push(`P1 ${l.module} 源码/安装目录不存在，无法核对 id：${modDir}`);
    console.log('  FAIL P1 插件源码或安装目录不存在，无法核对'); continue;
  }
  const installedId = JSON.parse(read(modManifest)).id;
  check(`P1 module 门与已装 id 逐字节相同（${l.module}）`, installedId, l.module);

  /* P2 —— 文件存在、能解析、顶层是对象、没有 BOM、没有重复键。 */
  const abs = path.join(HUB, l.path);
  check(`P2 文件存在（${path.basename(l.path)}）`, fs.existsSync(abs), true);
  if (!fs.existsSync(abs)) continue;
  const buf = fs.readFileSync(abs);
  check('P2 没有 UTF-8 BOM', buf[0] === 0xEF && buf[1] === 0xBB && buf[2] === 0xBF, false);
  const raw = buf.toString('utf8');
  let obj = null;
  try { obj = JSON.parse(raw); } catch (err) { failures.push(`P2 ${l.path} 解析失败：${err.message}`); }
  check('P2 能解析且顶层是对象', obj !== null && typeof obj === 'object' && !Array.isArray(obj), true);
  if (!obj) continue;
  check('P2 原文里没有重复键（JSON.parse 会静默吃掉前一个）', duplicateKeys(raw), []);

  const cn = flat(obj);
  const cnKeys = Object.keys(cn);

  /* P3 —— 故意为空的那一份必须保持为空。 */
  if (spec.mustBeEmpty) {
    check('P3 tah-alien-cn.json 仍然是空对象', raw.trim(), '{}');
    check('P3 tah-alien-cn.json 一个键都没有', cnKeys.length, 0);
    const upstream = path.join(modDir, spec.upstreamLang);
    const up = flat(JSON.parse(read(upstream)));
    check('P3 上游 languages/en.json 本身也是 0 键（所以确实无处可译）', Object.keys(up).length, 0);
    const src = pluginSource(modDir, spec.srcGlob);
    const own = [...new Set([...src.matchAll(/localize\(\s*['"]([^'"]+)['"]/g)].map(m => m[1]))].sort();
    check('P3 它读的键全部落在 ALIENRPG.* / tokenActionHud.*（都由别处覆盖）',
      own.filter(k => !/^(ALIENRPG|tokenActionHud)\./.test(k)), []);
    // 它读的 ALIENRPG.* 必须已被系统覆盖文件 lang/cn.json 译到
    const hubCn = flat(JSON.parse(read(path.join(HUB, 'lang/cn.json'))));
    check('P3 它读的每个 ALIENRPG.* 键都已在 lang/cn.json 里有译文',
      own.filter(k => k.startsWith('ALIENRPG.') && !(k in hubCn)), []);
    for (const k of cnKeys) { ownerOf[k] = l.path; allFlat[k] = cn[k]; }
    continue;
  }

  /* P4 —— 覆盖率：上游 en.json 的每一个键都必须在。 */
  const upstreamPath = path.join(modDir, spec.upstreamLang);
  check(`P4 上游 ${spec.upstreamLang} 存在`, fs.existsSync(upstreamPath), true);
  if (!fs.existsSync(upstreamPath)) continue;
  const en = flat(JSON.parse(read(upstreamPath)));
  check('P4 上游每一个键都被覆盖', Object.keys(en).filter(k => !(k in cn)), []);

  /* P5 —— 插值占位符 1:1。多一个少一个都会在界面上露出 {xxx}。 */
  check('P5 占位符与上游逐条一致',
    cnKeys.filter(k => (k in en) && JSON.stringify(placeholders(en[k])) !== JSON.stringify(placeholders(cn[k]))), []);

  /* P6 —— 命名空间：不许有 ALIENRPG.* 泄漏，不许跑到插件命名空间之外。 */
  if (!spec.allowSystemNamespace) {
    check('P6 没有 ALIENRPG.* 泄漏（那是系统文件的地盘）',
      cnKeys.filter(k => k.startsWith('ALIENRPG.')), []);
  }
  check(`P6 全部落在 ${spec.namespaces.join(' / ')} 之内`,
    cnKeys.filter(k => !spec.namespaces.some(n => k === n || k.startsWith(`${n}.`))), []);

  /* P7 —— 不许有死键：既不在上游 en.json、插件源码里也没人读。 */
  const src = pluginSource(modDir, spec.srcGlob);
  const allowedExtraKeys = spec.allowedExtraKeys ?? [];
  check('P7 上游没有的键，必须是源码读取键或登记过的动态别名',
    cnKeys.filter(k => !(k in en) && !src.includes(k) && !allowedExtraKeys.includes(k)), []);
  check('P7 登记的动态别名全部存在',
    allowedExtraKeys.filter(k => !(k in cn)), []);

  /* P8 —— 命令词表冻结（见文件头）。 */
  for (const k of spec.frozenKeys) {
    check(`P8 命令词表与上游逐字节相同：${k}`, cn[k], en[k]);
  }
  if (spec.frozenKeys.length) {
    // 判据本身还活着吗：读取点还在、剥前缀的正则还是 ASCII/法语。
    const mainJs = read(path.join(modDir, 'scripts/main.js'));
    const reads = [...mainJs.matchAll(/localize\(\s*['"]MOTHER\.Keywords\.(\w+)['"]/g)].map(m => m[1]).sort();
    check('P8 上游读取点仍是这四个（变了就重新裁定）', reads, ['Ordre', 'Protocol', 'Special', 'Special2']);
    check('P8 上游剥前缀链里仍然没有非 ASCII 分支',
      /\.replace\(\/\^[^/]*[\u4e00-\u9fff][^/]*\//.test(mainJs), false);
    check('P8 Keywords 只用于解析、从不渲染（读取点数 == includes 判定处数）',
      (mainJs.match(/MOTHER\.Keywords\./g) || []).length, 4);
  }

  /* P9 —— 展开成嵌套时不能出现「叶子同时又是枝」。 */
  check('P9 没有既是叶子又是枝的键（expandObject 会静默丢一个）',
    cnKeys.filter(a => cnKeys.some(b => b !== a && b.startsWith(`${a}.`))), []);

  for (const k of cnKeys) {
    if (ownerOf[k] && ownerOf[k] !== l.path) {
      failures.push(`P10 跨文件撞键：${k}（${ownerOf[k]} 与 ${l.path}）`);
    }
    ownerOf[k] = l.path; allFlat[k] = cn[k];
  }
}

/* P10 —— 五份文件合并成一张全局表，键不能互撞（含系统覆盖文件）。 */
const hubCn = flat(JSON.parse(read(path.join(HUB, 'lang/cn.json'))));
check('P10 插件文件与 lang/cn.json 没有撞键',
  Object.keys(ownerOf).filter(k => k in hubCn), []);

console.log(`\n${'='.repeat(70)}\nPASS ${pass}   FAIL ${failures.length}`);
if (failures.length) {
  console.log('\n失败明细：');
  for (const f of failures) console.log(`  · ${f}`);
  process.exitCode = 1;
} else {
  console.log('ALL GREEN');
}
