/**
 * 第二十八轮 · 快照诱饵扫描（判据换代）
 *
 * 旧判据是「体积 > 40k」。它漏掉了两份：
 *   · `2026-08-13-final-audit/probes/_shim_hardcoded.mjs` 22136 B —— 体积不够门槛；
 *   · `2026-08-16-round22/ember-hardcoded-cn.mjs.bak` 182808 B —— 体积够，但后缀不是 `.mjs`。
 * 两条漏法说明**体积和后缀都不是判据**。真正的判据是：
 *   这份文件里有没有**真值表 / 判据自身的定义**，从而可能被人当成现表读。
 *
 * 新判据（DEFS）：文件正文里出现任意一条表/判据的**定义式**。
 * 命中即要求它带死戳（第一行 `// ⚠ …不要当现表读…`），否则报出来。
 *
 * 会豁免两类（每类都必须给得出理由，不是「看着像探针就放过」）：
 *   · READS_TRUTH —— 文件里出现真身路径且把定义式当**切片锚点**用（它读的是真值，不是副本）；
 *   · DELTA       —— 定义式出现在 `old` / `new` / diff 上下文里，自我标注了是「差量」而非现表。
 *
 * 用法：node scan_decoys.mjs [根目录]        默认扫 `4-临时脚本/`
 * 退出码 1 = 有未处置的诱饵。
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const PROJ = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const ROOT = process.argv[2] ?? path.join(PROJ, "4-临时脚本");
const TRUTH_REL = "1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";

/** 表/判据的定义式。⚠ 这里是**定义**（后面跟 `[` / `{`），不是引用。 */
const DEFS = [
  "const PREFIXED = [",
  "const PATTERNS = [",
  "const NOTIFICATION_PATTERNS = [",
  "const EXACT = {",
  "const SELFCHECK_TABLES = {",
  "const KINDS = {",           // 断言那侧的 kind 分派表
];

const DEAD_MARK = "不要当现表读";

function walk(dir, out = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) { if (e.name !== "node_modules" && e.name !== ".git") walk(p, out); }
    else out.push(p);
  }
  return out;
}

// ⚠ 扫描器自己**必须排除**：它正文里就写着 DEFS 与 DEAD_MARK 两组字面量，
//   不排除的话它会把自己判成「已死戳」—— 判据豁免了判据自己，就是形态 (d) 的另一种长相。
const SELF = fileURLToPath(import.meta.url);
const OUT = path.join(path.dirname(SELF), "decoy_scan.json");

/** 定义式名字，供差量上下文匹配用 */
const DEF_NAMES = DEFS.map((d) => d.replace(/^const\s+/, "").replace(/\s*=\s*[[{]\s*$/, ""));
const DELTA_LINE = new RegExp(`^\\s*[-+]\\s*const (${DEF_NAMES.join("|")})\\b`, "m");

const rows = [];
for (const p of walk(ROOT)) {
  // 排除扫描器自己与它自己的产物（产物里回显了 DEFS 清单）
  if (path.resolve(p) === path.resolve(SELF) || path.resolve(p) === path.resolve(OUT)) continue;
  let txt;
  try { txt = fs.readFileSync(p, "utf8"); } catch { continue; }
  const hits = DEFS.filter((d) => txt.includes(d));
  if (!hits.length) continue;

  const head = txt.slice(0, 2000);
  // 死戳必须在**抬头**（前 1200 字节）。写在文件中段的「不要当现表读」不算数 ——
  // 谁会读到一半才发现自己读的是快照。
  const marked = txt.slice(0, 1200).includes(DEAD_MARK);
  const ext = path.extname(p);
  // 豁免①：差量 —— 定义式落在 diff 的 -/+ 行、`"old":` 字段、或 add(old, new) 调用里。
  //   这类文件**自我标注**了「我是从 X 到 Y 的一步」，没人会拿它当现表。先判它，
  //   因为差量文件往往也提到真身路径，会被豁免②误收。
  const isDelta = ext === ".diff" || DELTA_LINE.test(txt)
    || /"old"\s*:\s*"/.test(txt) || /^\s*add\(\s*$/m.test(txt);
  // 豁免②：读真值 —— 必须是**可执行源码**，且定义式紧挨着切片调用当锚点用
  //   （`…indexOf("const EXACT = {")` 这种）。JSON 报告里提一嘴真身路径不算。
  const isSource = [".mjs", ".js", ".py", ".cjs"].includes(ext);
  const anchorUse = hits.some((d) => new RegExp(
    `(indexOf|index|split|slice|find|partition)\\s*\\(\\s*["'\`]${d.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`
  ).test(txt));
  const readsTruth = isSource && txt.includes("ember-hardcoded-cn.mjs") && anchorUse;

  let verdict, why;
  if (marked) { verdict = "OK/已死戳"; why = "抬头 1200 字节内有死戳"; }
  else if (isDelta) { verdict = "OK/差量"; why = "定义式在 diff 行 / old 字段 / add() 调用里，自我标注为差量"; }
  else if (readsTruth) { verdict = "OK/读真值"; why = "可执行源码，定义式只作切片锚点，实际读的是真身"; }
  else { verdict = "诱饵"; why = "含表定义、抬头无死戳、既不是差量也不读真值"; }

  rows.push({
    file: path.relative(PROJ, p).replace(/\\/g, "/"),
    bytes: fs.statSync(p).size,
    defs: hits,
    head_copies_truth: head.includes("翻译 Ember 里 **babele 够不到** 的硬编码字符串"),
    verdict, why,
  });
}

rows.sort((a, b) => (a.verdict === "诱饵" ? 0 : 1) - (b.verdict === "诱饵" ? 0 : 1) || a.file.localeCompare(b.file));
const bad = rows.filter((r) => r.verdict === "诱饵");
for (const r of rows) {
  console.log(`${r.verdict.padEnd(10)} ${String(r.bytes).padStart(7)}  ${r.file}`);
  console.log(`${" ".repeat(11)}defs=${r.defs.join(" ")}${r.head_copies_truth ? "  ⚠抬头照搬真身" : ""}`);
}
console.log(`\n合计 ${rows.length} 份含表定义 · 诱饵 ${bad.length} 份`);
fs.writeFileSync(OUT, JSON.stringify({ root: ROOT, defs: DEFS, total: rows.length, decoys: bad.length, rows }, null, 1), "utf8");
process.exit(bad.length ? 1 : 0);
