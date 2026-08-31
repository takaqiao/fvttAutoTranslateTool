/**
 * 第二十九轮 V9 验收探针：裁掉「现网越界的通知正则」之后的**双向**实测。
 *
 * 判的是**发布中的那份** 1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs：
 * 在临时 harness 副本上只追加一行 `export {…}`（函数体一个字节不改），与
 * 3-常用脚本/qa/translate_cases_runner.mjs 同一套做法。
 *
 * ⚠ 硬约束 4：探针的**模拟输入必须指得出上游契约出处**。下面每一条输入都在注释里
 *   写了 ember.mjs 的行号与取值链；没有出处的输入 = 空转形态 (h)。
 *
 * 三段：
 *   (P) 正例：**保留下来的**条目仍翻得动，且译文逐字符相符（用上游真会产出的形状）。
 *   (N) 反例：别的模块 / Foundry 核心发的近似通知，一条都不许被吃
 *             —— 含本轮已坐实的两条现网越界现场。
 *   (Z) 零变化：其余未动条目逐条对比「改动前 vs 改动后」的输出，任何差异都是 bug。
 *       改动前那一份从 git HEAD 现取（`git show HEAD:<path>`），不是手抄的快照。
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const ROOT = path.resolve(process.argv[2]
  ?? "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project");
// ⚠ `1-Ember汉化插件` 自己是一个 git 仓（外仓之外的独立仓），(Z) 段的对照物要在**那个仓**里取。
const EMBER_REPO = path.join(ROOT, "1-Ember汉化插件");
const REL = "scripts/ember-hardcoded-cn.mjs";
const SRC = path.join(EMBER_REPO, REL);
const STUB_IMPORT = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";

function die(msg) {
  console.error(`探针**没跑成**（不是通过）：${msg}`);
  process.exit(2);
}

/** 把一份源码做成可 import 的 harness 副本，返回模块命名空间。 */
async function load(src, tag) {
  if (!src.includes(STUB_IMPORT)) die(`${tag}：找不到要打桩的 import 行`);
  for (const [name, re] of [
    ["translateNotification", /\nfunction translateNotification\(/],
    ["NOTIFICATION_PATTERNS", /\nconst NOTIFICATION_PATTERNS = \[/],
    ["NOTIFICATIONS", /\nconst NOTIFICATIONS = \{/]
  ]) if (!re.test(src)) die(`${tag}：找不到 ${name} 的顶层声明`);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "v9probe-"));
  const p = path.join(dir, `${tag}.mjs`);
  fs.writeFileSync(p,
    "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
    + src.replace(STUB_IMPORT,
      "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
    + "\nexport { translateNotification, NOTIFICATION_PATTERNS, NOTIFICATIONS };\n",
    "utf8");
  try { return await import(pathToFileURL(p).href); }
  catch (e) { die(`${tag}：import 失败 ${e.message}`); }
}

const NOW = await load(fs.readFileSync(SRC, "utf8"), "now");
let BEFORE = null;
try {
  const old = execFileSync("git", ["show", `HEAD:${REL}`],
    { cwd: EMBER_REPO, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  BEFORE = await load(old, "before");
} catch (e) {
  die(`取 git HEAD 版被判文件失败：${e.message} —— (Z) 段没有对照物就不能报绿`);
}

const T = NOW.translateNotification, T0 = BEFORE.translateNotification;
let fail = 0;
const say = (ok, seg, msg) => {
  if (!ok) fail++;
  console.log(`${ok ? "PASS" : "FAIL"} [${seg}] ${msg}`);
};

/* ═══════════════ (P) 正例：保留下来的条目仍翻得动，译文逐字符相符 ═══════════════ */
// 每条输入的上游出处写在第三列。
const POS = [
  // 与 :51766（本轮删掉的那条）同出一个函数 `#importValue`（ember.mjs:51757-51771）的
  // 另一支：句内带 "Token Maker"，是厂商锚点，本轮保留。kind 只可能是 layer / color
  // （:51760 的模板串里 `${kind}`，调用点 :51903 / :51938 两处写死）。
  ["This layer is not available in the Token Maker's current template preview.",
   "该层在指示物制作器当前的模板预览中不可用。", "ember.mjs:51760 (kind=layer)"],
  ["This color is not available in the Token Maker's current template preview.",
   "该颜色在指示物制作器当前的模板预览中不可用。", "ember.mjs:51760 (kind=color)"],
  // 本轮**收紧**的那条。上游 :61049 两个位是 `h1.key` / `h0.key`；`get key()`（:738）
  // → `getKey()`（:908）恒返回 `${prefix}.${i}.${j}`，i/j 是整数（:955-956 的
  // Number.isInteger 校验），prefix 现有 "s"（:119780）与 "p"（:120120）两个。
  ["Adjacent hex s.12.4 is not directly reachable from current hex s.12.3.",
   "相邻六边格 s.12.4 无法从当前六边格 s.12.3 直达。", "ember.mjs:61049 · key=s.i.j"],
  ["Adjacent hex p.0.0 is not directly reachable from current hex s.-3.7.",
   "相邻六边格 p.0.0 无法从当前六边格 s.-3.7 直达。", "ember.mjs:61049 · 另一个 slice 前缀 + 负偏移"],
  // MED 三条（本轮不动），插值位是 `\d+`：ember.mjs:37966 / 120582 / 120627。
  ["Completed resting for 8 hours without incident!", "顺利完成了 8 小时的休息！",
   "ember.mjs:37966 (remainingHours)"],
  ["Your short rest has been interrupted after 10 minutes!", "短休在 10 分钟后被打断！",
   "ember.mjs:120582"],
  ["Your long rest has been interrupted after 6 hours!", "长休在 6 小时后被打断！",
   "ember.mjs:120627"],
  // 第二十八轮收紧过的镜子那条，本轮不动：ember.mjs:98432，targetId ∈ a1…a23 / b1…b23。
  ["Mirror a1 does not exist!", "镜子 a1 不存在！", "ember.mjs:98432 · #MIRRORS 键"]
];
for (const [inp, want, src] of POS) {
  const got = T(inp);
  say(got === want, "P", `${JSON.stringify(inp)} → ${JSON.stringify(got)}`
    + (got === want ? `  ⟨${src}⟩` : `  期望 ${JSON.stringify(want)}  ⟨${src}⟩`));
}

/* ═══════════════ (N) 反例：别人的通知一条都不许被吃 ═══════════════ */
const NEG = [
  // ★ 本轮已坐实的两条**现网越界现场**：改动前会被整句重写，改动后必须原样返回。
  ['"myLayer" is already registered for this layer.', "现网越界现场①（:51766 旧正则）"],
  ['"crimson" is already registered for this color.', "现网越界现场②（:51766 旧正则）"],
  // 同一条删除带出的其它第三方形状
  ['"tokenHUD" is already registered for this layer.', "第三方注册去重的通用措辞"],
  // 被删的休息族：dnd5e 系休息模块会发的近似句子
  ["Your rest was interrupted by the Ambush event!", "被删 :38009 的原形状"],
  ["Your rest was interrupted after 3 hours by the Ambush event!", "被删 :37970 的原形状"],
  ["Your rest was interrupted by the Random Encounter event!", "第三方休息模块"],
  // 被删的 group actor：dnd5e 核心就有 type:"group" 的角色
  ['You cannot create multiple tokens for the "Bandits" group actor.', "被删 :60902 的原形状"],
  ['You cannot create multiple tokens for the "The Party" group actor.', "dnd5e group actor"],
  // hex 收紧之后必须挡住的形状（上游 getKey 恒带前缀，这些都是别人才会发的）
  ["Adjacent hex A is not directly reachable from current hex B.", "hex 收紧：非坐标"],
  ["Adjacent hex 12.4 is not directly reachable from current hex 12.3.", "hex 收紧：缺前缀（上游产不出）"],
  ["Adjacent hex north is not directly reachable from current hex south.", "hex 收紧：方位词"],
  // 老的几条对照（第二十八轮已建，确认没被本轮改动破坏）
  ["Mirror Image does not exist!", "第二十八轮现场（镜影术）"],
  ["You do not have permission to update this Document.", "Foundry 核心"],
  ["Token Mold | Settings saved.", "别的模块"]
];
for (const [inp, why] of NEG) {
  const got = T(inp);
  say(got === inp, "N", `${JSON.stringify(inp)} → ${got === inp ? "原样不动" : JSON.stringify(got)}  ⟨${why}⟩`);
}

/* ═══════════════ (Z) 其余未动条目：逐条对比改动前/后，任何差异都是 bug ═══════════════ */
// 输入集合 = 主闸规则文件里那 36 条通知正例（上游真会产出的形状，出处已在规则里逐条登记）
// ∪ 上面 (P)(N) 两段。对照物是 git HEAD 版的同一份实现。
// 允许出现差异的输入，**只有本轮明确改到的那 5 条正则所覆盖的串**，逐条列在 EXPECT 里。
const rules = JSON.parse(fs.readFileSync(
  path.join(ROOT, "5-其他内容", "RESOLUTIONS.assertions.json"), "utf8"));
const tc = rules.assertions.find(a => a.kind === "translate_cases");
if (!tc) die("规则文件里找不到 translate_cases 规则 —— (Z) 段的输入集合无从取起");
const inputs = [...new Set([
  ...tc.notify_positive.map(x => x[0]),
  ...tc.notify_negative,
  ...POS.map(x => x[0]),
  ...NEG.map(x => x[0])
])];
// 本轮**故意**改变行为的输入（删 4 条 + 收紧 1 条）。每条都要求「改动前翻了、改动后不翻」，
// 或收紧那条的「改动前后都翻且译文相同」。不在这张表里的输入，前后必须逐字符一致。
const EXPECT_CHANGED = new Set([
  '"horns" is already registered for this layer.',
  '"red" is already registered for this color.',
  '"myLayer" is already registered for this layer.',
  '"crimson" is already registered for this color.',
  '"tokenHUD" is already registered for this layer.',
  "Your rest was interrupted after 3 hours by the Ambush event!",
  "Your rest was interrupted by the Ambush event!",
  "Your rest was interrupted by the Random Encounter event!",
  'You cannot create multiple tokens for the "Bandits" group actor.',
  'You cannot create multiple tokens for the "The Party" group actor.',
  // hex 收紧带来的三条：旧的 `(.+)` 会把任意串塞进中文框（「相邻六边格 A 无法从当前六边格 B 直达。」），
  // 收紧后一律原样放行。这三条正是「收紧」这个动作的**收益面**，不是回归。
  "Adjacent hex 12.4 is not directly reachable from current hex 12.3.",
  "Adjacent hex A is not directly reachable from current hex B.",
  "Adjacent hex north is not directly reachable from current hex south."
]);
let z_same = 0;
for (const s of inputs) {
  const a = T0(s), b = T(s);
  if (EXPECT_CHANGED.has(s)) {
    const ok = (a !== s) && (b === s);   // 改动前会翻，改动后原样放行
    say(ok, "Z", `**故意改变** ${JSON.stringify(s)}：前=${JSON.stringify(a)} 后=${JSON.stringify(b)}`
      + (ok ? "" : "  ← 期望「前翻后不翻」"));
  } else {
    if (a === b) { z_same++; continue; }
    say(false, "Z", `**未预期的行为变化** ${JSON.stringify(s)}：前=${JSON.stringify(a)} 后=${JSON.stringify(b)}`);
  }
}
say(true, "Z", `其余 ${z_same} 条输入前后逐字符一致（对照物 = git HEAD 版实现）`);

/* ═══════════════ 表长 ═══════════════ */
console.log(`\nNOTIFICATION_PATTERNS: ${BEFORE.NOTIFICATION_PATTERNS.length} → `
  + `${NOW.NOTIFICATION_PATTERNS.length}`);
console.log(fail === 0 ? "\n全部通过。" : `\n共 ${fail} 条不通过。`);
process.exit(fail === 0 ? 0 : 1);
