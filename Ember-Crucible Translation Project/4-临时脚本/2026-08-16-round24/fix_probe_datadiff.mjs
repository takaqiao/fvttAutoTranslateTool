/**
 * 证明「本轮一个键都没动」：把改动前后两份源码**去掉注释**后逐行比，
 * 差异必须全部落在文件末尾自检接线那一段（SELFCHECK_TABLES / DETECTOR_BLIND / 三个 helper）。
 *
 * 去注释用的是一个走字符串状态的小扫描器（认单引号 / 双引号 / 反引号 / 正则字面量），
 * 不是拿正则去啃 —— 本文件里带 `//` 的字符串（`@UUID[...]`、URL 之类）会把纯正则打穿。
 */
import fs from "node:fs";

function stripComments(src) {
  let out = "";
  let i = 0;
  const n = src.length;
  let prevSig = "";           // 上一个有意义的字符，用来区分「除号」与「正则开头」
  while (i < n) {
    const c = src[i], d = src[i + 1];
    if (c === "/" && d === "/") { while (i < n && src[i] !== "\n") i++; continue; }
    if (c === "/" && d === "*") { i += 2; while (i < n && !(src[i] === "*" && src[i + 1] === "/")) i++; i += 2; continue; }
    if (c === '"' || c === "'" || c === "`") {
      const q = c; out += c; i++;
      while (i < n) {
        if (src[i] === "\\") { out += src.slice(i, i + 2); i += 2; continue; }
        out += src[i];
        if (src[i] === q) { i++; break; }
        i++;
      }
      prevSig = q; continue;
    }
    if (c === "/" && /[=(,:[!&|?{};\n]/.test(prevSig || "\n")) {   // 正则字面量
      out += c; i++;
      while (i < n) {
        if (src[i] === "\\") { out += src.slice(i, i + 2); i += 2; continue; }
        if (src[i] === "[") { while (i < n && src[i] !== "]") { out += src[i]; i++; } }
        out += src[i];
        if (src[i] === "/") { i++; break; }
        i++;
      }
      prevSig = "/"; continue;
    }
    out += c;
    if (!/\s/.test(c)) prevSig = c;
    i++;
  }
  return out.split("\n").map(l => l.trimEnd()).filter(l => l.trim()).join("\n");
}

const a = stripComments(fs.readFileSync(process.argv[2], "utf8")).split("\n");
const b = stripComments(fs.readFileSync(process.argv[3], "utf8")).split("\n");

// 公共前缀 / 公共后缀，中间就是真正改的那一段
let head = 0;
while (head < a.length && head < b.length && a[head] === b[head]) head++;
let tail = 0;
while (tail < a.length - head && tail < b.length - head && a[a.length - 1 - tail] === b[b.length - 1 - tail]) tail++;

console.log(`去注释后：改前 ${a.length} 行 / 改后 ${b.length} 行`);
console.log(`公共前缀 ${head} 行 · 公共后缀 ${tail} 行`);
console.log(`改前独有 ${a.length - head - tail} 行 · 改后独有 ${b.length - head - tail} 行\n`);
console.log("---- 改前独有 ----");
for (const l of a.slice(head, a.length - tail)) console.log("  - " + l);
console.log("---- 改后独有 ----");
for (const l of b.slice(head, b.length - tail)) console.log("  + " + l);
