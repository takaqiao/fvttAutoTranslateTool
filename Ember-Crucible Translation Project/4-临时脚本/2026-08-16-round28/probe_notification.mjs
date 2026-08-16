/**
 * 第二十八轮 · NOTIFICATION_PATTERNS 越界探针（②号工作面）
 *
 * 为什么要它：`^Mirror (.+) does not exist!$` 挂在 `translateNotification` 上，而后者包的是
 * **全局** `ui.notifications.notify`。`(.+)` 把 `Mirror Image does not exist!`（镜影术，任何
 * 法术模块都可能发）也吃掉，重写成一句完全错误的中文 —— 这是「定义域过松 + 整句重构」型越界。
 *
 * 三条纪律（对着 §3.7 登记的空转形态逐条）：
 *   · 被测函数**从真身现 import**（打桩只替掉 SELFCHECK 与 Hooks），不抄表 —— 形态 (b)/(h)；
 *   · 每条用例都带 `line`，指向 `modules/ember/scripts/ember.mjs` 的**上游调用点**，
 *     探针会去那一行把模板字面量的**静态片段**抠出来，逐段核对用例里确实按序出现；
 *     核不上就整条 FAIL。这样「模拟输入」不是我编的，是上游契约推出来的 —— 形态 (h)；
 *   · 上游文件读不到 / 行号对不上 → 直接 throw，不静默跳过 —— 形态 (e)。
 *
 * 产物里同时记 **被判源文件自身的 sha256**（`src_sha256`），
 * 这样 before/after 两份产物能区分「改动无行为影响」与「两次跑的是同一份文件」。
 *
 * 用法：node probe_notification.mjs <输出 json> [被测源码路径]
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { pathToFileURL } from "node:url";

const PROJ = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const UPSTREAM = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs";
const OUT = process.argv[2];
const SRC = process.argv[3] ?? path.join(PROJ, "1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs");
if (!OUT) throw new Error("用法：node probe_notification.mjs <输出 json> [源码路径]");

/* ============================================================ */
/*  语料：31 条 NOTIFICATION_PATTERNS 的正例 + 越界负例          */
/*  line = ember.mjs 上游调用点行号（1-based）                   */
/* ============================================================ */

/** 正例：形状由上游模板字面量决定，探针会回源核对。 */
const POSITIVE = [
  { line: 3149,   in: 'Attunement activation is not yet implemented for system "pf2e".' },
  { line: 33205,  in: "The Ankarist Overlook Vista is not yet configured to support the placement of Tokens." },
  { line: 34509,  in: 'Deleted saved Vista composition "Dusk".' },
  { line: 34557,  in: 'Updated composition for Level "Upper".' },
  { line: 34570,  in: 'Saved composition to Scene with identifier "bleak-archive".' },
  { line: 34779,  in: 'Imported composition to Level "Lower".' },
  { line: 36885,  in: "Awarded attunement progression points to Actors: Lyla, Thren" },
  { line: 37966,  in: "Completed resting for 8 hours without incident!" },
  { line: 37970,  in: "Your rest was interrupted after 3 hours by the Night Ambush event!" },
  { line: 38009,  in: "Your rest was interrupted by the Night Ambush event!" },
  { line: 120582, in: "Your short rest has been interrupted after 20 minutes!" },
  { line: 120627, in: "Your long rest has been interrupted after 5 hours!" },
  { line: 51386,  in: "Saved Ember Dynamic Token configuration for Lyla" },
  { line: 51839,  in: "Saved Ember Dynamic Token randomization parameters to Actor Lyla" },
  { line: 61882,  in: "Applied fog exploration from the Broken Spire vantage point!" },
  { line: 67418,  in: "Discovered vantage point, Broken Spire!" },
  { line: 126644, in: "Added the Soulbound talent to Lyla at rank 1." },
  { line: 126665, in: "Upgraded Soulbound talent on Lyla to rank 2." },
  { line: 36922,  in: "Event nosuch-event is not a recognized gameplay event. This likely indicates some issue with the Ember module installation." },
  { line: 49480,  in: 'Ember | "horns-01" is not a known Token Maker part in any template layer.' },
  // kind 的定义域是 "layer" / "color"（ember.mjs:51903 / 51938 两个调用点写死），两个值都跑
  { line: 51760,  in: "This layer is not available in the Token Maker's current template preview." },
  { line: 51760,  in: "This color is not available in the Token Maker's current template preview." },
  { line: 51766,  in: '"horns-01" is already registered for this layer.' },
  { line: 51766,  in: '"#ff0000" is already registered for this color.' },
  { line: 60902,  in: 'You cannot create multiple tokens for the "Wandren Pack" group actor.' },
  { line: 61049,  in: "Adjacent hex 12.7 is not directly reachable from current hex 12.6." },
  { line: 61232,  in: "You are not allowed to delete the Caravan Token from the Region Map." },
  { line: 63556,  in: 'The Scene "Bleak Archive" does not have compositions defined.' },
  { line: 63865,  in: 'User "Taka" wants to modify interactable "a5" in Scene Bleak Archive. You must be present in this Scene to acknowledge this operation.' },
  { line: 96361,  in: 'Ember | Transit destination scene "nosuch" was not found.' },
  // ⚠ 本轮收紧的那条。targetId 的定义域是 BleakArchiveAreaMap.#MIRRORS 的键
  //   （ember.mjs:97859 静态表 = a1…a23 / b1…b23；98130 用同一张表的键填 #mirrors；
  //    98430 的 targetId 来自 mirror.config.path 的键，同样只可能是这批 id）。
  { line: 98432,  in: "Mirror a5 does not exist!" },
  { line: 98432,  in: "Mirror b23 does not exist!" },
  { line: 98432,  in: "Mirror a1 does not exist!" },
  { line: 121323, in: "Attunement feat for Abyss rank 2 could not be resolved." },
  { line: 126652, in: "Lyla cannot progress their Soulbound rank further as they already bear a Deathly Soulmark." },
];

/**
 * 越界负例：**不是** Ember 发的、但落在旧正则定义域里的串。一条都不许被改写。
 * `why` 写清楚这串现实里从哪来。
 */
const NEGATIVE = [
  { in: "Mirror Image does not exist!",
    why: "Mirror Image = 镜影术，标准法术名；任何法术/效果类模块都可能这么报" },
  { in: "Mirror Universe does not exist!",
    why: "泛化：Mirror + 任意英文名" },
  { in: "Mirror does not exist!",
    why: "空插值退化形（旧正则 (.+) 要求非空，这条本来就不该命中，作对照）" },
  { in: "Mirror 2 does not exist!",
    why: "纯数字索引 —— 上游 #MIRRORS 的键从来不是纯数字，属别的模块的形状" },
  { in: "Mirror mirror on the wall does not exist!",
    why: "泛化：带空格的任意短语" },
  { in: "Token Mold | Settings saved.",
    why: "对照组：完全无关的第三方模块提示" },
  { in: "Mirror a5 does not exist.",
    why: "句末是句点不是叹号 —— 上游是叹号，这条不该命中" },
  { in: "Prefix Mirror a5 does not exist!",
    why: "带前缀 —— ^ 锚点应当拦住" },
];

/* ============================================================ */
/*  上游契约核对：把用例的形状钉回 ember.mjs 的模板字面量        */
/* ============================================================ */

const up = fs.readFileSync(UPSTREAM, "utf8");
// ⚠ ember.mjs 是**混合行尾**（大段 LF，局部 CRLF）。绝对不能用
//   `split(/\r?\n/).slice(0, n).join("\n").length` 反推字节偏移 —— 每条 CRLF 行少算一字节，
//   到十万行时偏出几百 KB，探针会去核一段风马牛不相及的模板（第一版就是这么自己空转的）。
//   所以只在**行数组**上取窗口，不做偏移换算。
const upLines = up.split(/\r?\n/);
if (upLines.length < 130000) throw new Error("ember.mjs 行数异常，上游版本对不上");

/** 取第 line 行（1-based）起 15 行窗口里的第一段反引号模板，返回其**静态片段**数组。 */
function upstreamChunks(line) {
  const seg = upLines.slice(line - 1, line + 14).join("\n");
  const i = seg.indexOf("`");
  if (i < 0) throw new Error(`ember.mjs:${line} 起 15 行内找不到反引号模板`);
  let j = i + 1, depth = 0, buf = "", closed = false;
  const chunks = [];
  for (; j < seg.length; j++) {
    const c = seg[j];
    if (depth === 0 && c === "\\") { buf += seg[j + 1]; j++; continue; }
    if (depth === 0 && c === "`") { closed = true; break; }
    if (depth === 0 && c === "$" && seg[j + 1] === "{") { chunks.push(buf); buf = ""; depth = 1; j++; continue; }
    if (depth > 0) {
      if (c === "{") depth++;
      else if (c === "}") depth--;
      continue;
    }
    buf += c;
  }
  if (!closed) throw new Error(`ember.mjs:${line} 的模板在 15 行窗口内没有闭合`);
  chunks.push(buf);
  return chunks.map((s) => s.replace(/\s+/g, " ")).filter((s) => s.trim() !== "");
}

/** 用例串里必须按序出现上游的全部静态片段，否则这条用例是我编的，不是上游契约推出来的。 */
function checkContract(line, input) {
  const chunks = upstreamChunks(line);
  const flat = input.replace(/\s+/g, " ");
  let pos = 0;
  for (const ch of chunks) {
    const k = flat.indexOf(ch.trim(), pos);
    if (k < 0) return { ok: false, chunks, missing: ch.trim() };
    pos = k + ch.trim().length;
  }
  return { ok: true, chunks };
}

/* ============================================================ */
/*  跑                                                          */
/* ============================================================ */

const IMPORT_LINE = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const srcText = fs.readFileSync(SRC, "utf8");
if (!srcText.includes(IMPORT_LINE)) throw new Error("找不到 SELFCHECK 的 import 行");

globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ec-r28-notif-"));
let result;
try {
  const harness = path.join(tmp, "_h.mjs");
  fs.writeFileSync(harness,
    "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
    + srcText.replace(IMPORT_LINE,
        "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
    + "\nexport { translateNotification as __translateNotification };\n",
    "utf8");
  const mod = await import(pathToFileURL(harness).href);
  const tn = mod.__translateNotification;

  const contract_fail = [];
  const pos = POSITIVE.map((c) => {
    const k = checkContract(c.line, c.in);
    if (!k.ok) contract_fail.push({ line: c.line, in: c.in, missing: k.missing, chunks: k.chunks });
    const out = tn(c.in);
    return { line: c.line, in: c.in, out, translated: out !== c.in, contract_ok: k.ok };
  });
  const neg = NEGATIVE.map((c) => {
    const out = tn(c.in);
    return { in: c.in, out, untouched: out === c.in, why: c.why };
  });

  const rows = [...pos.map((r) => [r.in, r.out]), ...neg.map((r) => [r.in, r.out])];
  result = {
    src: SRC,
    src_sha256: crypto.createHash("sha256").update(fs.readFileSync(SRC)).digest("hex"),
    src_bytes: fs.statSync(SRC).size,
    upstream: UPSTREAM,
    upstream_sha256: crypto.createHash("sha256").update(fs.readFileSync(UPSTREAM)).digest("hex"),
    rows_sha256: crypto.createHash("sha256").update(JSON.stringify(rows)).digest("hex"),
    positive_total: pos.length,
    positive_translated: pos.filter((r) => r.translated).length,
    positive_untranslated: pos.filter((r) => !r.translated).map((r) => r.in),
    negative_total: neg.length,
    negative_eaten: neg.filter((r) => !r.untouched),
    contract_fail,
    positive: pos,
    negative: neg,
  };
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}

fs.writeFileSync(OUT, JSON.stringify(result, null, 1), "utf8");

const bad = [];
if (result.contract_fail.length) bad.push(`契约核对失败 ${result.contract_fail.length} 条`);
if (result.positive_untranslated.length) bad.push(`正例未翻 ${result.positive_untranslated.length} 条`);
if (result.negative_eaten.length) bad.push(`负例被吃 ${result.negative_eaten.length} 条`);
console.log(`src_sha256=${result.src_sha256.slice(0, 16)}…  src_bytes=${result.src_bytes}`);
console.log(`正例 ${result.positive_translated}/${result.positive_total} 翻出  ·  负例 ${result.negative_total - result.negative_eaten.length}/${result.negative_total} 未被碰  ·  契约核对 ${result.positive_total - result.contract_fail.length}/${result.positive_total}`);
for (const r of result.positive_untranslated) console.log(`  正例未翻: ${JSON.stringify(r)}`);
for (const r of result.negative_eaten) console.log(`  负例被吃: ${JSON.stringify(r.in)} -> ${JSON.stringify(r.out)}`);
for (const r of result.contract_fail) console.log(`  契约失败: ember.mjs:${r.line} 缺片段 ${JSON.stringify(r.missing)}`);
console.log(`→ ${OUT}`);
console.log(bad.length ? `RESULT=FAIL (${bad.join("; ")})` : "RESULT=PASS");
