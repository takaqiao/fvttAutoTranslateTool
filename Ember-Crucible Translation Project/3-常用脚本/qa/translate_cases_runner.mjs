/**
 * `assert_resolutions.py` 的 `translate_cases` 断言的 **node 侧执行体**。
 *
 * 为什么是一个独立的 .mjs 而不是在 Python 里重写一遍正则
 * ----------------------------------------------------
 * `PATTERNS` / `PREFIXED` / `NOTIFICATION_PATTERNS` 是**能主动改坏译文**的三张表
 * （命中就替换整串），而第二十六轮以前全项目一道常设闸都没有：自检面板的 D 档按设计核不了
 * 正则表（对 PREFIXED 19 + PATTERNS 28 共 47 键直接报 ⛔），61 条断言里 `PATTERNS` 出现 0 次，
 * 上一轮验它的探针是一次性的、跑完没人再跑。第二十五轮真正咬人的缺陷
 * （`^Result of (.+)$` 会把任意英文句子吃掉 →「结果：…」）就出在这里。
 *
 * ⇒ 本文件**真的 import 发布中的 `ember-hardcoded-cn.mjs`** 跑用例。
 *   在 Python 侧照抄一份正则表来判，等于又造一个会漂移的副本 —— 那是本项目反复吃亏的形态
 *   （见 `_run_same_en_split` 的注释：判据不许复制第二份实现）。
 *
 * ⚠ 本文件是**手写落盘**的，不是 bash heredoc / Python 改写脚本生成的 ——
 *   正则里的 `\b` `\s` `\d` 一旦经改写脚本转手就会失效而主闸照样全绿（空转形态 (f)）。
 *
 * 第二十七轮补的三件事（上一轮复核挑出来的没兑现处）
 * ------------------------------------------------
 * (1) **覆盖归因**（§④）。上一轮闸名写着 `PATTERNS / PREFIXED`，实测 79 条用例把 PATTERNS
 *     打满 28/28，`PREFIXED` 却只触到 **2/19**（`Attunement`、`Music Mood`）、**一条反例都没有**；
 *     把 `Knowledge`/`Location`/`Quest`/`Talent`/`Rarity` 五条译名逐个改坏重跑，闸 5/5 保持 0 违规。
 *     ⇒ 本轮把用例补满，并加一道**机械的覆盖归因**：三张表的**每一条**都必须被至少一条正例触到，
 *     漏一条就报违规。这样「闸名写着它、其实没盖住」不可能再靠人眼漏过去。
 * (2) **通知闸**（§⑤）。`NOTIFICATION_PATTERNS` 31 条正则命中即整串替换，而
 *     `translateNotification` 包的是 `ui.notifications.notify` —— **所有模块共用的全局钩子**，
 *     爆炸半径比 `PATTERNS`（只在 Ember 自己认出的子树里跑）**更大**，却一道常设闸都没有。
 *     本轮用同一个 runner 盖上：31 条逐条正例 + 反例（重点是「别的模块发的通知不许被吃掉」）
 *     + 空白折叠回退支。
 * (3) **抽取器的 (h) 残留**（§⑥）。复核把上游 40 条 `label:` 行各加一个空格（模拟上游换打包
 *     格式），编排名 219 → 196 **静默下降而闸仍 0 违规** —— 因为 `min_labels` 只卡 150、比现值低 69，
 *     而且找不到 `arrangements` 的那一支 `continue` 掉了、**不计 unresolved**。两处都已改。
 *
 * 空转形态 (h)：**空转的是给判据喂输入的那个探针**（第二十六轮登记）
 * -----------------------------------------------------------------
 * 上一轮的 `probe_world_c.mjs` 把英文基准里所有层级的 name 拍平成顶层 index 条目喂给判据，
 * 而真实 Foundry 从不产出那种 index —— 判据本身是好的，它验的输入是自己捏的。
 * ⇒ **本文件喂给被判函数的每一类模拟输入，其形状都必须指得出上游契约的出处：**
 *   · `Result of ${dc-5}-` / `Result of ${dc+5}+`
 *       ← `enrichCriticalResult`，ember.mjs:22905-22913（先 `dc = Number(dc)` 再
 *         `if (!Number.isInteger(dc)) return match`，之后**只可能**拼出这两种形状）；
 *   · `${channel.capitalize()}: ${arrangement.label}`
 *       ← `EmberSoundscape.enricherHTML`，ember.mjs:16266；
 *         `label += \` (${mood.titleCase()})\`` ← :16267-16268；`…: Reset` ← :16255；
 *   · PREFIXED 19 条前缀的出处逐条可查（`Ancestry:` ember.mjs:22934 · `Culture:` :22954 /
 *     :123464 · `Path:` :22986 / :123467 · `Knowledge:` :122900 / :123707 · `Language:` :123688 /
 *     :126547 · `Attunement:` :23008 / :23181 · `Quest:` :25236 · `Identifier:` :35842 ·
 *     `Divine Domain:`/`Warlock Patron:`/`Sorcerous Origin:` :36354/:36359/:36364 ·
 *     `Lifespan:` :122595 与 crucible-async.mjs:189 · `Rarity:` crucible-async.mjs:188 ·
 *     `Talent:` crucible/module/enrichers.mjs:739 · `Area Map:`/`Location:`/`Biome:`/`Terrain:`
 *     templates/applications/hex-hud.hbs:13/47/50/52 · `Music Mood:` ember.mjs:16271）；
 *   · 通知语 31 条正例的插值位逐条对应 `ui.notifications.*` 的调用点（行号写在被判文件的表里），
 *     其中两条**故意带上游源码里的换行 + 缩进**（ember.mjs:36922 / 126652 是跨行模板串，
 *     运行时消息真的带那段空白），用来钉住 `translateNotification` 的空白折叠回退支；
 *   · 全量编排名 —— **不是抄我们自己的 ARRANGEMENTS 表**（那是被判方），
 *     而是从上游 `ember.mjs` 的 `soundscapes` 注册表里**现抠** `arrangements[].label`，
 *     那正是 :16266 那个 `arrangement.label` 的来源。抠不出来 ⇒ 当场失败，不许跑成空集。
 *
 * 用法：node translate_cases_runner.mjs <spec.json>
 *   spec.json 由 Python 侧写出，字段见下面的读取处。
 *   结果 JSON 写到 spec.out；退出码 0 = 跑成了（**不代表用例全过**，过没过看 out 里的
 *   `violations`），非 0 = **根本没跑成**（node 找不到文件 / import 炸了 / 抠不到编排名）——
 *   Python 侧必须把「没跑成」判成失败而不是静默跳过（空转形态 (e)）。
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

function die(msg) {
  console.error(`translate_cases_runner: ${msg}`);
  process.exit(2);
}

/* ------------------------------------------------------------------ spec */

const specPath = process.argv[2];
if (!specPath) die("用法：node translate_cases_runner.mjs <spec.json>");
let spec;
try {
  spec = JSON.parse(fs.readFileSync(specPath, "utf8"));
} catch (e) {
  die(`spec 读不动 ${specPath}：${e.message}`);
}
for (const k of ["src", "harness", "out", "ember_mjs", "stub_import"]) {
  if (!spec[k]) die(`spec 缺字段 ${k}`);
}

/* ------------------------------------------------ ① 造 harness 并 import */

// `ember-hardcoded-cn.mjs` 顶上 import 的 `./ember-cn-selfcheck.mjs` 在 node 下会去碰
// Foundry 的全局；这里把它换成一个桩，并补一个 Hooks 桩。**替换目标必须存在**，
// 不存在说明上游那行改了 —— 当场失败，不许「没替换成也照跑」。
let src;
try {
  src = fs.readFileSync(spec.src, "utf8");
} catch (e) {
  die(`被判文件读不动 ${spec.src}：${e.message}`);
}
if (!src.includes(spec.stub_import)) {
  die(`被判文件里找不到要打桩的 import 行 ${JSON.stringify(spec.stub_import)} —— `
    + `上游那行改了？转换中止（不许跳过：跳过就是形态 (e)）`);
}

// 被判文件只 `export` 了 `translateText`。`translateNotification` 与三张表都是模块私有的，
// 而本闸要判的正是它们 —— 所以在 **harness 副本**上追加一行 `export {…}`。
// ⚠ 只追加导出、**一个字节的函数体都不改**：被判的仍然是发布中的那份实现。
// ⚠ 每个要导出的名字都先在源码里确认声明存在，找不到就**当场失败** ——
//   少一个名字就静默少导一个，等于这一段悄悄不判了（形态 (e)）。
// 第三列 `alreadyExported`：`translateText` 被判文件自己就 `export` 了，再追加一次会
// 「Duplicate export」当场炸；其余七个都是模块私有的，必须补导出。
const NEEDED = [
  ["translateText", /\nexport function translateText\(/, true],
  ["translateLeaf", /\nfunction translateLeaf\(/, false],
  ["translateNotification", /\nfunction translateNotification\(/, false],
  ["EXACT", /\nconst EXACT = \{/, false],
  ["PREFIXED", /\nconst PREFIXED = \[/, false],
  ["PATTERNS", /\nconst PATTERNS = \[/, false],
  ["NOTIFICATIONS", /\nconst NOTIFICATIONS = \{/, false],
  ["NOTIFICATION_PATTERNS", /\nconst NOTIFICATION_PATTERNS = \[/, false],
  // 第二十八轮加：⑥ 段要拿它与**上游现抠的编排名**做集合包含检查（见那一段的说明）。
  ["ARRANGEMENT_LEAVES", /\nconst ARRANGEMENT_LEAVES = \{/, false]
];
for (const [name, re] of NEEDED) {
  if (!re.test(src)) {
    die(`被判文件里找不到 ${name} 的顶层声明（改名了？被删了？被改成非 export 了？）—— `
      + `本闸判的就是它，找不到必须当场失败，不许少判一段还报绿`);
  }
}
const exportLine =
  `\nexport { ${NEEDED.filter(([, , done]) => !done).map(([n]) => n).join(", ")} };\n`;

const stub = "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n";
fs.mkdirSync(path.dirname(spec.harness), { recursive: true });
fs.writeFileSync(spec.harness,
  stub + src.replace(spec.stub_import,
    "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
  + exportLine,
  "utf8");

let M;
try {
  M = await import(pathToFileURL(spec.harness).href);
} catch (e) {
  die(`import 被判文件失败：${e.message}`);
}
const { translateText, translateNotification, translateLeaf,
        EXACT, PREFIXED, PATTERNS, NOTIFICATIONS, NOTIFICATION_PATTERNS,
        ARRANGEMENT_LEAVES } = M;
if (typeof translateText !== "function" || typeof translateNotification !== "function") {
  die("被判文件没有导出 translateText / translateNotification —— 用例无从跑起");
}
for (const [name, v] of [["PREFIXED", PREFIXED], ["PATTERNS", PATTERNS],
                         ["NOTIFICATION_PATTERNS", NOTIFICATION_PATTERNS]]) {
  if (!Array.isArray(v) || v.length === 0) die(`${name} 不是非空数组 —— 表被清空了？`);
}

const violations = [];
const counts = {};
const V = (group, input, got, want, note) =>
  violations.push({ group, input, got, want, note: note || "" });

/* ------------------------------------------------------------ ② 反例 (A) */

// 「上游产不出、但形状相近」的英文串：一个字都不许被 translateText 改动。
// 其中 3 条是第二十五轮复核造出来的真缺陷现场（旧实现会把它们译成中文）；
// 第二十七轮补的一批钉的是 **PREFIXED 那一层**：`startsWith(en + ": ")` 只锚在**串首**，
// 所以「别的模块产出的 `标签: 值` 形状」与「近似但上游产不出的写法」都必须原样不动。
const negative = spec.negative || [];
counts.negative = negative.length;
for (const s of negative) {
  const got = translateText(s);
  if (got !== s) V("negative", s, got, s, "上游产不出这种串，判据把它吃掉了 = 误翻散文");
}

/* ------------------------------------------------------------ ③ 正例 (B) */

// 上游**真会产出**的串：必须仍然翻得动，且译文**逐字符**相符。
// 只断言「翻动了」不够 —— 收紧过头会把译文改成别的东西而断言照样绿。
const positive = spec.positive || [];
counts.positive = positive.length;
for (const row of positive) {
  const [input, want] = row;
  const got = translateText(input);
  if (got !== want) V("positive", input, got, want, "上游真会产出这一串，译文必须逐字符相符");
}

/* -------------------------------------------------- ④ 覆盖归因 (B2)：三张表逐条 */
//
// 为什么非要有这一段（第二十七轮）
// --------------------------------
// 上一轮的用例数看着很足（79 条），实际 `PREFIXED` 只触到 19 条里的 2 条 ——
// **用例数不等于覆盖**，而闸名里写着 PREFIXED。人眼数不出来，就让判据自己数。
//
// ⚠ 这里的 `attribute()` 是 `translateText` **派发顺序**的镜像（EXACT → PREFIXED → PATTERNS，
//   见被判文件 translateText 的函数体）。镜像有漂移风险 —— 所以每条用例都做
//   **预测输出 vs 实跑输出**的自洽校验：派发顺序、`translateLeaf` 的兜底写法、
//   `text.replace(raw, …)` 的收尾方式，任一处被改动而镜像没跟上，这里当场报违规。
//   （镜像用的是**真表对象本身**，不是抄一份表 —— 抄表才是本项目禁止的那种第二实现。）

function attribute(raw) {
  if (raw in EXACT) return { ch: "exact" };
  for (let i = 0; i < PREFIXED.length; i++) {
    if (raw.startsWith(`${PREFIXED[i].en}: `)) return { ch: "prefixed", i };
  }
  for (let i = 0; i < PATTERNS.length; i++) {
    const m = raw.match(PATTERNS[i].re);
    if (m) return { ch: "patterns", i, m };
  }
  return { ch: "none" };
}

function predict(text, a) {
  const raw = text.trim();
  if (a.ch === "exact") return text.replace(raw, EXACT[raw]);
  if (a.ch === "prefixed") {
    const { en, cn, table } = PREFIXED[a.i];
    return text.replace(raw, `${cn}：${translateLeaf(raw.slice(en.length + 2), table)}`);
  }
  if (a.ch === "patterns") return text.replace(raw, PATTERNS[a.i].cn(a.m));
  return text;
}

const covPrefixed = new Set();
const covPatterns = new Set();
let absorbed = 0;
for (const [input] of positive) {
  const a = attribute(input.trim());
  if (a.ch === "prefixed") covPrefixed.add(a.i);
  if (a.ch === "patterns") covPatterns.add(a.i);
  const pre = predict(input, a);
  const got = translateText(input);
  if (pre !== got) {
    V("attribution", input, got, pre,
      `覆盖归因的派发镜像与实跑对不上（归因=${a.ch}${a.i ?? ""}）——`
      + `translateText 的派发顺序 / translateLeaf 的兜底 / 收尾 replace 被改过而镜像没跟上`);
  }
}
for (const s of negative) {
  const a = attribute(s.trim());
  if (a.ch !== "none") absorbed++;           // 命中了某条但按设计原样返回（如 Rank 的查表兜底）
}
counts.prefixed_size = PREFIXED.length;
counts.prefixed_covered = covPrefixed.size;
counts.patterns_size = PATTERNS.length;
counts.patterns_covered = covPatterns.size;
counts.negative_absorbed = absorbed;

// ⚠ 第二十八轮把**表长下限**从这里搬走了（搬进 Python 侧的 `_two_layer_floor`）。
//   原因见 `assert_resolutions.py` 里那个函数的文档注释：下限是个**可调的数**，
//   而可调的数必须两层（现算 + 历史记录）一起判，两层的判法只该有一处实现。
//   留在这里的全是**不含阈值**的恒等式 —— 调不松，也就不必两层。
function reportCoverage(group, table, covered, label) {
  const miss = [];
  for (let i = 0; i < table.length; i++) if (!covered.has(i)) miss.push(i);
  if (miss.length) {
    V(group, "未被任何正例触到的条目下标", JSON.stringify(miss), "[]",
      `${label} ${table.length} 条里有 ${miss.length} 条没有正例 ——`
      + `闸名写着它、实际没盖住，正是第二十六轮 PREFIXED 2/19 那个洞`);
  }
}
reportCoverage("coverage", PREFIXED, covPrefixed, "PREFIXED");
reportCoverage("coverage", PATTERNS, covPatterns, "PATTERNS");

/* --------------------------------------------- ⑤ 通知闸 (D)：NOTIFICATION_PATTERNS */
//
// `translateNotification`（被判文件 :2284 起）包的是 `ui.notifications.notify` ——
// **所有模块共用的全局钩子**。它刻意不走 `translateText`（那道口子有全局 EXACT），
// 只查 `NOTIFICATIONS` + `NOTIFICATION_PATTERNS` 两张 Ember 自己的表。
// 但那 31 条正则同样是「命中即整串替换」，爆炸半径比 PATTERNS 还大 ——
// PATTERNS 只在 Ember 自己认出的子树里跑，这里是别的模块发的每一条提示都要过一遍。
// ⇒ 反例的重点就是**别的模块发的通知一个字都不许被吃掉**。

function attributeNotify(raw) {
  const one = (s) => {
    if (s in NOTIFICATIONS) return { ch: "exact", s };
    for (let i = 0; i < NOTIFICATION_PATTERNS.length; i++) {
      const m = s.match(NOTIFICATION_PATTERNS[i].re);
      if (m) return { ch: "np", i, m, s };
    }
    return null;
  };
  const a = one(raw);
  if (a) return a;
  const flat = raw.replace(/\s+/g, " ");
  if (flat !== raw) {
    const b = one(flat);
    if (b) { b.flat = true; return b; }
  }
  return { ch: "none" };
}

function predictNotify(text, a) {
  const raw = text.trim();
  if (a.ch === "exact") return text.replace(raw, NOTIFICATIONS[a.s]);
  if (a.ch === "np") return text.replace(raw, NOTIFICATION_PATTERNS[a.i].cn(a.m));
  return text;
}

const nNeg = spec.notify_negative || [];
const nPos = spec.notify_positive || [];
counts.notify_negative = nNeg.length;
counts.notify_positive = nPos.length;

for (const s of nNeg) {
  const got = translateNotification(s);
  if (got !== s) {
    V("notify_negative", s, got, s,
      "这条提示不是 Ember 发的（或上游产不出这种写法），而我们包的是全局 "
      + "ui.notifications.notify —— 把它吃掉就是改别的模块的界面");
  }
}

const covNP = new Set();
let flatHits = 0;
for (const row of nPos) {
  const [input, want] = row;
  const got = translateNotification(input);
  if (got !== want) {
    V("notify_positive", input, got, want, "上游真会发这一条，译文必须逐字符相符");
  }
  const a = attributeNotify(input.trim());
  if (a.ch === "np") covNP.add(a.i);
  if (a.flat) flatHits++;
  const pre = predictNotify(input, a);
  if (pre !== got) {
    V("attribution", input, got, pre,
      `通知派发镜像与实跑对不上（归因=${a.ch}${a.i ?? ""}）—— translateNotification 的查表顺序`
      + `或空白折叠回退被改过而镜像没跟上`);
  }
}
counts.np_size = NOTIFICATION_PATTERNS.length;
counts.np_covered = covNP.size;
counts.np_flat_fallback = flatHits;
reportCoverage("coverage", NOTIFICATION_PATTERNS, covNP, "NOTIFICATION_PATTERNS");

// 空白折叠回退支（被判文件 :2299-2302）：上游 ember.mjs:36922 / 126652 是**跨行**模板串，
// 源码里的换行 + 缩进原样进了消息文本。没有一条带真实空白的用例，这一支等于没判。
// ⚠ 这里**故意不做成可配的阈值**：「至少有一条正例是靠回退支命中的」是**存在性**，
//   不是能松能紧的门槛。可配的数就多一条作弊路径（第二十八轮堵的正是那条）。
if (flatHits < 1) {
  V("coverage", "空白折叠回退支", String(flatHits), ">=1",
    "没有一条正例是靠 translateNotification 的空白折叠回退命中的 —— 那一支无人看守");
}

/* --------------------------------------- ⑥ 全量护栏 (C)：编排名从上游现抠 */

const arr = spec.arrangements;
if (arr) {
  const emberSrc = (() => {
    try {
      return fs.readFileSync(spec.ember_mjs, "utf8");
    } catch (e) {
      die(`上游 ember.mjs 读不动 ${spec.ember_mjs}：${e.message} —— `
        + `编排名无从抠起，这一段必须当场失败而不是跑成空集`);
    }
  })();

  /** 从 `openIdx` 处的 `{` 起取一个配平的花括号块（跳过字符串与注释）。 */
  function blockAt(s, openIdx) {
    let d = 0, q = null;
    for (let i = openIdx; i < s.length; i++) {
      const c = s[i];
      if (q) {
        if (c === "\\") { i++; continue; }
        if (c === q) q = null;
        continue;
      }
      if (c === '"' || c === "'" || c === "`") { q = c; continue; }
      if (c === "/" && s[i + 1] === "/") { i = s.indexOf("\n", i); if (i < 0) break; continue; }
      if (c === "/" && s[i + 1] === "*") { i = s.indexOf("*/", i) + 1; continue; }
      if (c === "{") d++;
      else if (c === "}") { d--; if (d === 0) return s.slice(openIdx, i + 1); }
    }
    return null;
  }

  // `soundscapes` 注册表 —— :16266 的 `arrangement.label` 就是从这里来的。
  const reg = emberSrc.match(
    /var soundscapes=\/\*#__PURE__\*\/Object\.freeze\(\{__proto__:null,([^}]*)\}\)/);
  if (!reg) {
    die("上游 ember.mjs 里抠不到 soundscapes 注册表 —— 上游打包形状变了？"
      + "这一段必须当场失败：抠不出编排名却照跑，就是拿一个空集换全绿");
  }
  const entries = reg[1].split(",").map((s) => s.trim()).filter(Boolean)
    .map((s) => { const [k, v] = s.split(":"); return { key: k, varName: v }; });

  const labels = new Set();
  let unresolved = 0;
  const unresolvedKeys = [];
  for (const { key, varName } of entries) {
    const re = new RegExp(`(?:^|[;}])var ${varName.replace(/\$/g, "\\$")} = \\{`, "m");
    const m = re.exec(emberSrc);
    if (!m) { unresolved++; unresolvedKeys.push(`${key}(找不到定义)`); continue; }
    const body = blockAt(emberSrc, emberSrc.indexOf("{", m.index));
    if (!body) { unresolved++; unresolvedKeys.push(`${key}(花括号不配平)`); continue; }
    const aIdx = body.search(/\n  arrangements: \{/);
    // ⚠ 第二十七轮修 (h) 残留：这一支原先是 `continue`（**不计 unresolved**）。
    //   复核把上游 40 条 `label:` 行各加 1 个空格模拟换打包格式，编排名 219 → 196
    //   **静默下降而闸仍 0 违规** —— 抽取器锚在精确空白上，锚空了却一声不响。
    //   现在「这个音景压根没有 arrangements 块」与「有但抠不出来」一样计入 unresolved：
    //   全 44 个音景实测都带 arrangements 块，一旦有音景抠不出来，就是抽取器锚失效。
    if (aIdx < 0) { unresolved++; unresolvedKeys.push(`${key}(没有 arrangements 块)`); continue; }
    const aBody = blockAt(body, body.indexOf("{", aIdx));
    if (!aBody) { unresolved++; unresolvedKeys.push(`${key}(arrangements 花括号不配平)`); continue; }
    for (const mm of aBody.matchAll(/\n      label: "([^"]*)"/g)) labels.add(mm[1]);
  }
  counts.soundscapes = entries.length;
  counts.unresolved = unresolved;
  counts.labels = labels.size;
  if (unresolved) {
    V("arrangements", unresolvedKeys.join(" / ").slice(0, 200), String(unresolved), "0",
      `有 ${unresolved} 个音景的编排名解析不出来 —— 抠出来的编排名不全，这一段的「全量」名不副实`);
  }
  // ⚠ 第二十八轮：编排名的**两道数量下限**（`min_labels` / `labels_recorded`）搬去 Python 侧，
  //   与另外几个可调的数一起走同一套「现算 + 历史记录」两层判法（见 `_two_layer_floor`）。
  //   这里换上一道**不含阈值**的：
  //
  //   **被判文件里 `ARRANGEMENT_LEAVES` 的每一个键，都必须能在「上游现抠的编排名」里找到。**
  //
  //   为什么这一道比数量门槛硬：`ARRANGEMENT_LEAVES` 那 212 个键，当初就是从这份上游注册表
  //   抄下来的，两边是**同源**的。抽取器的锚（缩进 / 引号 / 字段名）一旦失效，抠出来的那份
  //   会**缺**一批名字，缺的那批立刻在这里点名报出来 —— 不需要任何人去猜「门槛该定多少才够紧」。
  //   第二十七轮的教训正是「门槛松到够不着现值」（150 vs 现值 219，掉 23 条照样在门槛之上）；
  //   门槛定得准是运气，集合包含关系不靠运气。
  //   ⚠ 例外集合 `leaves_not_upstream` **写死当护栏**（现为 `Reset` 一条：它来自
  //     ember.mjs:16255 的 `${channel.capitalize()}: Reset`，不是 arrangements[].label）——
  //     以后谁往表里加一个上游注册表里没有的叶子，这里会响，那是**要的**。
  const leafKeys = Object.keys(ARRANGEMENT_LEAVES || {});
  if (!leafKeys.length) {
    V("arrangements", "ARRANGEMENT_LEAVES", "0", ">0",
      "被判文件里的叶子表是空的 —— 这一道包含检查会退化成恒真，当场判失败");
  }
  const exempt = new Set(arr.leaves_not_upstream || []);
  const orphanLeaves = leafKeys.filter((k) => !labels.has(k) && !exempt.has(k));
  const staleExempt = [...exempt].filter((k) => labels.has(k) || !leafKeys.includes(k));
  if (orphanLeaves.length) {
    V("arrangements", "ARRANGEMENT_LEAVES 里上游查无此名的叶子",
      JSON.stringify(orphanLeaves.slice(0, 12)), "[]",
      `${orphanLeaves.length} 个叶子在「上游现抠的 ${labels.size} 个编排名」里找不到。`
      + `两种可能：① 抽取器的锚失效了，抠出来的那份缺了一批（形态 (h)，第二十七轮实测过`
      + `219→196 静默下降）；② 上游真删了这些编排名。**先分清是哪一种**，别直接改判据`);
  }
  if (staleExempt.length) {
    V("arrangements", "例外集合过期", JSON.stringify(staleExempt), "[]",
      "登记在 `leaves_not_upstream` 里的名字，如今要么上游有了、要么表里没了 —— "
      + "护栏集合必须贴着现状，留着过期条目等于给自己开了个不会响的口子");
  }

  const channels = arr.channels || [];
  const prefixes = arr.prefixes || {};
  const stuckLabels = new Set();
  let translated = 0, pairs = 0;
  for (const label of labels) {
    for (const ch of channels) {
      const input = `${ch}: ${label}`;
      pairs++;
      const got = translateText(input);
      if (got !== input && got.startsWith(prefixes[ch])) translated++;
      else stuckLabels.add(label);
    }
  }
  counts.pairs = pairs;
  counts.translated = translated;
  counts.stuck_labels = stuckLabels.size;

  // 「整串留英的**恰好**是登记的那几个」—— 这个集合**写死**当护栏：
  // 上游新增编排名时它会响，那是**要的**（人来看一眼这条新名字该不该进表）。
  const want = [...(arr.expect_untranslated || [])].sort();
  const got = [...stuckLabels].sort();
  if (JSON.stringify(got) !== JSON.stringify(want)) {
    const extra = got.filter((x) => !want.includes(x));
    const missing = want.filter((x) => !got.includes(x));
    V("arrangements", "整串留英的编排名集合", JSON.stringify(got), JSON.stringify(want),
      `多出来的 ${JSON.stringify(extra)}（上游新增了编排名？还是判据被收紧过头了）`
      + `／少掉的 ${JSON.stringify(missing)}（判据被放宽了？还是上游删了这条编排名）`);
  }
  const wantTranslated = (labels.size - want.length) * channels.length;
  if (translated !== wantTranslated) {
    V("arrangements", "翻得动的条数", String(translated), String(wantTranslated),
      "全量护栏对不上：留英集合与翻得动的条数必须互补");
  }
  counts.expect_translated = wantTranslated;
}

/* ------------------------------------------------------------------ 出参 */

const result = { ok: violations.length === 0, counts, violations };
fs.mkdirSync(path.dirname(spec.out), { recursive: true });
fs.writeFileSync(spec.out, JSON.stringify(result, null, 1), "utf8");
process.exit(0);
