/**
 * 把 ember-hardcoded-cn.mjs 变成一个可以在 Node 里 import 的副本。
 *
 * 只做三件事，一个字节的表内容都不改：
 *   ① 把 `import * as SELFCHECK from './ember-cn-selfcheck.mjs'` 换成一个不做事的桩
 *      （面板本体要 Foundry 全局，Node 里跑不起来，也不是本探针要测的东西）；
 *   ② 补一个 `Hooks` 桩（模块体里有两处 `Hooks.once`）；
 *   ③ 在末尾追加 export，把要核的表暴露出来。
 *
 * ⚠ 这个转换写在 .mjs 里而不是用 python 生成，正是为了避开本项目登记的空转形态 (f)：
 *   python 字符串会把正则里的 \b \s 二次转义掉，探针当场变成假绿。
 */
import fs from "node:fs";
import path from "node:path";

const SRC = process.argv[2];
const OUT = process.argv[3];

let s = fs.readFileSync(SRC, "utf8");

const IMPORT_LINE = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
if (!s.includes(IMPORT_LINE)) throw new Error("找不到 SELFCHECK 的 import 行，转换中止（宁可报错也不要静默产出一个测不着的副本）");
s = s.replace(IMPORT_LINE,
  "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };");

const HOOK_STUB = `globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n`;

const TAIL = `

export {
  SELFCHECK_TABLES as __SELFCHECK_TABLES,
  EXACT as __EXACT,
  DIALOG_UI as __DIALOG_UI,
  EMBER_WINDOW_UI as __EMBER_WINDOW_UI,
  NOTIFICATIONS as __NOTIFICATIONS,
  ATTUNEMENTS as __ATTUNEMENTS,
  ATTUNEMENT_TAB as __ATTUNEMENT_TAB,
  MOON_NAMES as __MOON_NAMES,
  VISTA_PLACEMENT_EN as __VISTA_PLACEMENT_EN,
  VISTA_PLACEMENT_FIELDS as __VISTA_PLACEMENT_FIELDS
};
`;

fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, HOOK_STUB + s + TAIL, "utf8");
console.log(`已生成 ${OUT}（${s.length} 字节源 + 桩 + export）`);
