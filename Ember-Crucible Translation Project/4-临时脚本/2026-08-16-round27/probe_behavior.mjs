/**
 * 第二十七轮 · 行为快照探针（②号工作面用）
 *
 * 目的：本轮**只动注释**，任何行为差异都是 bug。所以在改之前 / 改之后各跑一次，
 *      比 `translateText` 与 `translateNotification` 在同一份语料上的**逐条输出**。
 *
 * 三条纪律：
 *   · harness 写在 `mkdtemp` 里、`finally` 删掉 —— **不在版本化目录里留副本**
 *     （本项目登记的空转形态 (c)：留在轮次目录里的判据/表源码副本，上游一改就成过期快照）；
 *   · 语料**从被测文件自己的 `SELFCHECK_TABLES` 现抠**，不在探针里另抄一份键表
 *     （抄一份 = 测了个副本，形态 (h)）；
 *   · 只桩 `Hooks` / `SELFCHECK`，被测函数一行都不改写。
 *
 * 用法：node probe_behavior.mjs <输出 json> [被测源码路径]
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { pathToFileURL } from "node:url";

const PROJ = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const OUT = process.argv[2];
const SRC = process.argv[3] ?? path.join(PROJ, "1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs");
if (!OUT) throw new Error("用法：node probe_behavior.mjs <输出 json> [源码路径]");

const IMPORT_LINE = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const src = fs.readFileSync(SRC, "utf8");
if (!src.includes(IMPORT_LINE)) throw new Error("找不到 SELFCHECK 的 import 行");

globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ec-r27-behav-"));
let result;
try {
  const harness = path.join(tmp, "_h.mjs");
  fs.writeFileSync(harness,
    "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
    + src.replace(IMPORT_LINE,
        "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
    + "\nexport { SELFCHECK_TABLES as __TABLES, translateNotification as __translateNotification };\n",
    "utf8");
  const mod = await import(pathToFileURL(harness).href);

  /* ── 语料①：把 SELFCHECK_TABLES 里登记的每一张表的**键**全抠出来 ── */
  const keys = new Set();
  const eat = (t) => {
    if (!t || typeof t !== "object") return;
    for (const k of Object.keys(t)) keys.add(k);
  };
  for (const v of Object.values(mod.__TABLES)) {
    if (v && typeof v === "object" && "table" in v) eat(v.table);
    else eat(v);
  }

  /* ── 语料②：正则路径的构造用例（PATTERNS / 通知插值 / 富文本增强器）──
   *   出处：本轮之前已经落地的 `4-临时脚本/2026-08-16-round26/probe_expected.mjs` 的 CASES，
   *   那批是对着上游出处逐条核过才收的；这里只当**行为指纹**用，不判对错。 */
  const CASES = [
    "Result of -5-", "Result of 5+", "Result of 0-", "Result of 10+",
    "Result of 3-", "Result of 13+", "Result of 13-", "Result of 23+",
    "Result of 95-", "Result of 105+", "Result of -8-", "Result of 2+",
    "Music: Reset", "Environment: Reset",
    "Music: Ankarist Theme", "Environment: Ankarist Theme",
    "Music: Ancient Ruins (Tension)", "Music: Ankarist Theme (Calm)",
    "Music: Ancient Giants Tension", "Music Mood: Calm",
    "+2 Boons", "-6 Banes", "+1 Boons", "-1 Banes",
    "Age of Beasts - 12 Years Ago", "Age of the Tower - Current Year",
    "Age of Creation - -300 Years Ago", "After Shattering - 5 Years From Now",
    "Day 43 - 12:00", "Day 43", "Outcome 3", "5 Others", "Threat 12.5", "Threat -2",
    "Vantage Point: Somewhere", "Interactable: Lever", "Breeze (4.5 mph)",
    "Activate Attunement: Abyss", "Award Attunement: Abyss", "Revoke Attunement: Abyss",
    "Token Maker Part Usage: Horns", "Attunement: Abyss", "Critical Success",
    "[[/language borel]]", "[[/language kost]]", "[[/language moiré]]",
    "[[/knowledge Shent]]", "[[/knowledge nosuchid]]",
    "Abyss Rank 3", "Nosuch Attunement Rank 3",
    "An example Kessian character.",
    "The Abyss", "Heart of Ember", "Attunement: The Abyss", "Attunement: Heart of Ember",
    "and gain", "to rank 2 (Greater Soulmark)?", "to rank 3 (Deathly Soulmark)?",
    /* 负例：不该被碰的 */
    "Result of the investigation was inconclusive", "Music: my custom playlist",
    "Environment: Rain", "", "   ", "Some\nMulti\nLine", "已经是中文",
    /* 空白折叠回退：通知那一路的既定口径 */
    "You  must   select  a  token.", "\n  You must select a token.  \n",
  ];
  for (const c of CASES) keys.add(c);

  const corpus = [...keys].sort();
  const rows = [];
  for (const s of corpus) {
    let tt, tn;
    try { tt = mod.translateText(s); } catch (e) { tt = "‹THROW› " + e.message; }
    try { tn = mod.__translateNotification(s); } catch (e) { tn = "‹THROW› " + e.message; }
    rows.push([s, tt === undefined ? "‹undefined›" : String(tt),
                  tn === undefined ? "‹undefined›" : String(tn)]);
  }
  const blob = JSON.stringify(rows);
  result = {
    src: SRC,
    // ⚠ 2026-08-16 第二十八轮补：原来只记 `sha256`（= 输出 rows 的哈希），于是
    //   before/after 两份产物里**没有任何字段**能区分「改动确实无行为影响」与
    //   「两次跑的根本是同一份文件」—— 「零行为变化」这类结论就此不可证。
    //   现在把**被判源文件自身**的哈希与字节数也记进来：
    //   结论成立的形状是 `src_sha256` 不同 + `rows_sha256` 相同；两个都相同只说明白跑了一趟。
    //   字段名同时从 `sha256` 改为 `rows_sha256`，因为「sha256」不点名它哈希的是什么，
    //   正是上面那个洞的由来。（第二十七轮及更早的产物里仍叫 `sha256`，含义等同 `rows_sha256`。）
    src_sha256: crypto.createHash("sha256").update(fs.readFileSync(SRC)).digest("hex"),
    src_bytes: fs.statSync(SRC).size,
    corpus_size: corpus.length,
    rows_sha256: crypto.createHash("sha256").update(blob).digest("hex"),
    rows,
  };
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}

fs.writeFileSync(OUT, JSON.stringify(result, null, 1), "utf8");
console.log(`src_sha256=${result.src_sha256}  src_bytes=${result.src_bytes}`);
console.log(`corpus=${result.corpus_size}  rows_sha256=${result.rows_sha256}`);
console.log(`→ ${OUT}`);
