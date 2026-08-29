#!/usr/bin/env node
/**
 * 从**唯一一份** mapping 定义生成 alienrpg-cn 的 `babele-mappings.js`。
 *
 * 来源（两份，缺一不可）：
 *   4-常用脚本/extract/mappings.mjs          —— 声明式 mapping（抽取器也读这一份）
 *   4-常用脚本/release/runtime-converters.js —— 运行时 converter（translate 方向）
 *
 * 之所以生成而不是手写：抽取器按 mappings.mjs 决定写出哪些 key，Babele 按
 * babele-mappings.js 决定查哪些 key。两边若各留一份手抄件，迟早对不上，而且
 * 对不上的表现是「某些字段静默不翻」，没有报错。
 *
 * 用法：
 *   node "4-常用脚本/release/generate_runtime.mjs"
 *   node "4-常用脚本/release/generate_runtime.mjs" --project <项目根目录>
 *   node "4-常用脚本/release/generate_runtime.mjs" --check   # 只校验，不写盘
 *
 * ─────────────────────────────────────────────────────────────────────────
 * 与 Ember/Crucible 版的差异（本文件是从
 * `Ember-Crucible Translation Project/3-常用脚本/release/generate_runtime.mjs`
 * 移植来的，原件仍在那里，可对照）：
 *
 *  1. 目标只有一个。EC 有 ember / crucible 两个仓各拿一份 layer；Alien 侧
 *     `registerMapping` 是**全局层**，按项目决议只能由中枢模块 alienrpg-cn 注册
 *     一次，新手包 / 核心书两个模块不生成也不注册 mapping。
 *
 *  2. 静态 import 改成动态 import，并且缺 export 不算失败。
 *     EC 版顶部写 `import { CRUCIBLE_LAYER, EMBER_LAYER } from '../extract/mappings.mjs'`。
 *     静态 import 一个不存在的具名导出是**链接期 SyntaxError**，脚本连启动都做不到，
 *     而骨架阶段（mappings.mjs 还是 EC 逐字副本、没有 `ALIEN_LAYER` 的那几个小时）
 *     必须能跑。改成动态 import 后：缺 layer 时生成一份合法空壳，layer 落地后
 *     同一条命令原地把空壳换成真货。
 *     状态（2026-08-29 11:11 复核）：`ALIEN_LAYER` 已存在，产物已是真 mapping，
 *     `--check` 通过。这一条现在只是「为什么保留动态 import」的存档理由。
 *
 *  3. converter 源文件可缺席。EC 版无条件读 release/runtime-converters.js 并把
 *     整份文本贴进产物。那份文件现在装的是 crucible 专用 converter
 *     （crucibleDescription / crucibleActions / crucibleTokenName …），贴进
 *     alienrpg-cn 等于让本模块的全局层带上一批**对 alienrpg 文档毫无依据**的改写。
 *     所以这里只认 Alien 自己的 converter 文件，找不到就产出空 PROJECT_CONVERTERS。
 *
 * ⚠ 未闭合缺口（2026-08-29 11:11 实测，属于 mapping/release 任务，不属于抽取任务）：
 *   `ALIEN_LAYER` 里 `Actor.rollTable` / `Actor.critTable` 引用了 converter
 *   `alienRollTableRef`，但 `release/runtime-converters.alien.js` **不存在**，
 *   产物里 `PROJECT_CONVERTERS = {}`。抽取方向有实现
 *   （mappings.mjs 的 `EXTRACT_CONVERTERS.alienRollTableRef`），translate 方向没有。
 *   这正是 PROJECT.md §3.1 点名的「converter 只改一个方向」的形状。
 * ─────────────────────────────────────────────────────────────────────────
 */
import fs from 'fs';
import path from 'path';
import { pathToFileURL, fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const arg = (n, d) => {
  const i = argv.indexOf(n);
  return i >= 0 ? argv[i + 1] : d;
};
const CHECK_ONLY = argv.includes('--check');

/** 项目根 = 本文件的上上级（4-常用脚本/release/ -> 4-常用脚本/ -> 项目根） */
const PROJECT = path.resolve(arg('--project', path.resolve(__dirname, '..', '..')));

const MAPPINGS_SRC = path.join(__dirname, '..', 'extract', 'mappings.mjs');

/**
 * Alien 专用 converter。**故意不叫** runtime-converters.js：那个名字现在被 EC 的
 * crucible converter 占着（逐字副本，尚未改指向）。等 mapping 任务把 Alien 的
 * converter 写出来时，落到这个文件名上，本脚本自动开始把它贴进产物。
 */
const CONVERTERS_SRC = path.join(__dirname, 'runtime-converters.alien.js');

/** 只有中枢模块拿 mapping。见文件头第 1 条。 */
const TARGET = {
  repo: '1-系统汉化插件',
  id: 'alienrpg-cn',
  layerExport: 'ALIEN_LAYER',
};

const EMPTY_CONVERTERS = `/* ------------------------------------------------------------------ *
 * Runtime converters (translate direction).
 *
 * 尚未编写。这些 converter 的 EXTRACT 方向住在
 * 4-常用脚本/extract/extract_en.mjs —— 改动一侧产出的形状时，**同一个 commit**
 * 里改另一侧。
 *
 * Babele 的函数式 converter 签名（babele 2.9.1）：
 *   fn(value, translation, source, contextCompendium, allTranslations, runtime, params)
 * ------------------------------------------------------------------ */

export const PROJECT_CONVERTERS = {};
`;

const HEADER = (id, layerExport, converterOrigin) => `/* eslint-disable */
/**
 * GENERATED FILE — DO NOT EDIT BY HAND.
 * 生成文件，禁止手改。手改会在下一次跑生成器时被无声覆盖。
 *
 * Source of truth:
 *   Alien-RPG Translation Project/4-常用脚本/extract/mappings.mjs        (${layerExport})
 *   Alien-RPG Translation Project/4-常用脚本/release/${converterOrigin}
 * Regenerate with:
 *   node "4-常用脚本/release/generate_runtime.mjs"
 *
 * Module : ${id}
 * Layer  : ${layerExport}
 *
 * 导出的 mapping 交给 \`babele.registerMapping()\`。它是**全局层**，只 ENRICH
 * Babele 的内置默认值：没提到的字段（尤其 \`Adventure.actors\` 与 \`Actor.items\`）
 * 保留 Babele 自己的 \`document\` converter —— 那条路径才提供 source-pack 回退：
 * 带 \`_stats.compendiumSource\` 的内嵌文档会从它**原本所属**的包的译文里取翻译。
 * 不要用手写的遍历 converter 去替换它们。
 *
 * ⚠ 全局层对**所有系统的所有文档**生效，不只是 alienrpg。凡是可能撞上别人字段名
 * 的 variant，都要带 \`_when\` 守卫（mapping-block.js:176-206 支持
 * all / any / equals / in / exists）。
 */
`;

async function loadLayer() {
  if (!fs.existsSync(MAPPINGS_SRC)) {
    console.warn(`! 找不到 ${path.relative(PROJECT, MAPPINGS_SRC)}，产出空 mapping`);
    return { layer: {}, ok: false, reason: 'mappings.mjs 不存在' };
  }
  let mod;
  try {
    mod = await import(pathToFileURL(MAPPINGS_SRC).href);
  } catch (err) {
    console.warn(`! 载入 mappings.mjs 失败（${err.message}），产出空 mapping`);
    return { layer: {}, ok: false, reason: `import 失败: ${err.message}` };
  }
  const layer = mod[TARGET.layerExport];
  if (!layer || typeof layer !== 'object') {
    console.warn(
      `! mappings.mjs 未导出 ${TARGET.layerExport}（现有导出：${Object.keys(mod).join(', ') || '无'}），产出空 mapping`
    );
    return { layer: {}, ok: false, reason: `缺少 ${TARGET.layerExport} 导出` };
  }
  return { layer, ok: true, reason: '' };
}

function loadConverters() {
  if (fs.existsSync(CONVERTERS_SRC)) {
    return { text: fs.readFileSync(CONVERTERS_SRC, 'utf8').trimStart(), origin: path.basename(CONVERTERS_SRC) };
  }
  console.warn(`! 找不到 ${path.basename(CONVERTERS_SRC)}，产出空 PROJECT_CONVERTERS`);
  return { text: EMPTY_CONVERTERS, origin: `${path.basename(CONVERTERS_SRC)} (缺席，已用空壳)` };
}

const { layer, ok, reason } = await loadLayer();
const converters = loadConverters();

const body = [
  HEADER(TARGET.id, TARGET.layerExport, converters.origin),
  converters.text,
  '',
  `export const DOCUMENT_MAPPINGS = ${JSON.stringify(layer, null, 2)};`,
  '',
].join('\n');

const outDir = path.join(PROJECT, TARGET.repo);
if (!fs.existsSync(outDir)) {
  console.error(`x 目标仓不存在：${outDir}`);
  process.exit(1);
}
const outFile = path.join(outDir, 'babele-mappings.js');

if (CHECK_ONLY) {
  const current = fs.existsSync(outFile) ? fs.readFileSync(outFile, 'utf8') : null;
  if (current === body) {
    console.log(`ok  ${path.relative(PROJECT, outFile)} 与源一致`);
    process.exit(0);
  }
  console.error(`x  ${path.relative(PROJECT, outFile)} 与源**不一致** —— 重新跑生成器`);
  process.exit(1);
}

fs.writeFileSync(outFile, body, 'utf8');
const types = Object.keys(layer).length;
console.log(
  `wrote ${path.relative(PROJECT, outFile)}  (${types} 个文档类型, ${Buffer.byteLength(body, 'utf8')} 字节)` +
    (ok ? '' : `  [空壳：${reason}]`)
);
