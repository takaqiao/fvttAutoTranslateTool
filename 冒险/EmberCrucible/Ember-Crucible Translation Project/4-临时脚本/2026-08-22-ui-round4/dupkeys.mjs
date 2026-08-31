/**
 * 逐表查重键 + 查「新加的词是否已被别的表占掉」。直接 import 真身文件（第三轮那套 stub 手法）。
 * 前置自证：源码键数与运行时键数必须相等 —— 不等就说明有重复键被对象字面量静默吃掉。
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
const SRC = process.argv[2];
const STUB = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const NAMES = ["EXACT", "PREFIXED", "NOTIFICATION_PATTERNS", "DIALOG_UI",
  "DIALOG_TITLES", "ATTUNEMENTS", "EMBER_WINDOW_UI", "TOKEN_MAKER_UI", "TOKEN_MAKER_PART_IDS", "ARRANGEMENTS",
  "SOUNDSCAPE_GROUPS"];
const s = fs.readFileSync(SRC, "utf8");
const decl = (n) => new RegExp("\n(?:export )?const " + n + " = \{");
const present = NAMES.filter(n => decl(n).test(s));
const missing = NAMES.filter(n => !present.includes(n));
if (missing.length) console.log("（本版无此表，跳过）:", missing.join(" · "));
const h = path.join(process.cwd(), "_dup.mjs");
fs.writeFileSync(h, "globalThis.Hooks = globalThis.Hooks ?? { once(){}, on(){} };\n"
  + s.replace(STUB, "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck(){}, keyLiveness(){} };")
  + "\nexport { " + present.join(", ") + " };\n", "utf8");
const M = await import(pathToFileURL(h).href);
fs.unlinkSync(h);

// 键的判据：前面是 `{` / `,` / 行首（值那一侧前面必是 `: `，天然排除）。
// 一行多个键的写法在本文件里很常见，行首锚定会漏数 —— 上一版就是这么把 671 数成 160 的。
const KEY_RE = new RegExp('(?:^|[{,])[ \t\r\n]*"([^"]*)"[ \t]*:', "gm");
let bad = 0;
for (const n of present) {
  const t = M[n];
  if (!t || typeof t !== "object") continue;
  const body = s.split(decl(n))[1];
  const end = body.indexOf("\n};");
  const seg = end < 0 ? body : body.slice(0, end);
  const keys = [];
  KEY_RE.lastIndex = 0;
  let m;
  while ((m = KEY_RE.exec(seg))) keys.push(m[1]);
  const seen = new Set(), dups = [];
  for (const k of keys) { if (seen.has(k)) dups.push(k); seen.add(k); }
  // 本文件里有表用 `...OTHER` 把别的表整张展进来（EXACT 展 DIALOG_TITLES、
  // DIALOG_UI 展 ATTUNEMENTS）。不把展开的那部分算进来，源码键永远少于运行时键，
  // 自证就永远是黄的 —— 黄灯看久了等于没灯。
  // 本文件里有表用 `...OTHER` 把别的表整张展进来（EXACT 展 DIALOG_TITLES、
  // DIALOG_UI 展 ATTUNEMENTS）。不把展开的那一部分算进来，源码键永远少于运行时键，
  // 自证就永远是黄的——黄灯看久了等于没灯。
  const spreads = seg.split("...").slice(1)
    .map(x => (x.match(/^[A-Z_0-9]+/) || [])[0]).filter(Boolean)
    .filter(nm => M[nm] && typeof M[nm] === "object");
  const spreadKeys = spreads.reduce((a, nm) => a + Object.keys(M[nm]).length, 0);
  const runtime = Object.keys(t).length;
  const accounted = keys.length + spreadKeys;
  const flag = dups.length ? " ❌"
    : (accounted === runtime
      ? " ok" + (spreads.length ? "（含展开 " + spreads.join("+") + "）" : "")
      : " ⚠ 源码 " + keys.length + " + 展开 " + spreadKeys + " ≠ 运行时 " + runtime);
  console.log(n.padEnd(22) + " 源码键 " + String(keys.length).padStart(5)
    + "  运行时键 " + String(runtime).padStart(5) + "  重复 " + dups.length
    + (dups.length ? " ⇒ " + dups.slice(0, 8).join(" · ") : "") + flag);
  if (dups.length) bad += dups.length;
}
console.log("");
for (const w of ["Surface", "Pathways", "Location", "Hex", "Heavy", "Lithe"]) {
  const hits = present.filter(n => M[n] && typeof M[n] === "object" && (w in M[n]))
    .map(n => n + '="' + M[n][w] + '"');
  console.log('  "' + w + '"'.padEnd(12) + " → " + (hits.length ? hits.join("  |  ") : "（无表登记）"));
}
console.log("");
console.log(bad ? "❌ 有 " + bad + " 个重复键" : "✅ 无重复键");
