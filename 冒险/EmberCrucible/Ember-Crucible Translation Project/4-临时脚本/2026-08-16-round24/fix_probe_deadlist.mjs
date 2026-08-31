/**
 * 死名单告警的**灵敏度回测** —— 不做这一步，「没告警」等于没证据（本项目登记过的空转形态）。
 *
 * 做法：把改后的源码复制一份，只往 DETECTOR_BLIND.EXACT 里注入一条**表里不存在**的键，
 * import 这份副本，看 checkNames 是不是真的 warn 了。
 * 顺带反向证一次：不注入的那份必须**一条都不 warn**。
 */
import fs from "node:fs";
import path from "node:path";

const SRC = process.argv[2];
const dir = path.dirname(SRC);
const src = fs.readFileSync(SRC, "utf8");

const ANCHOR = '  EXACT: [\n';
if (!src.includes(ANCHOR)) throw new Error("找不到注入锚点，回测中止（宁可报错也不要假绿）");
const injected = src.replace(ANCHOR, ANCHOR + '    "__这条键在 EXACT 里根本不存在__",\n');
if (injected === src) throw new Error("注入没生效，回测中止");
const OUT = path.join(dir, "_hc_deadlist.mjs");
fs.writeFileSync(OUT, injected, "utf8");

const warns = [];
const realWarn = console.warn;
console.warn = (...a) => warns.push(a.map(String).join(" "));

await import("./" + path.basename(SRC));           // 干净的那份
const cleanWarns = warns.filter(w => w.includes("自检分流名单")).length;
warns.length = 0;
await import("./" + path.basename(OUT));           // 注了毒的那份
const dirtyWarns = warns.filter(w => w.includes("自检分流名单")).length;
console.warn = realWarn;

console.log(`干净副本：死名单告警 ${cleanWarns} 条（应为 0）`);
console.log(`注毒副本：死名单告警 ${dirtyWarns} 条（应 ≥1）`);
for (const w of warns.filter(w => w.includes("自检分流名单"))) console.log("   " + w);
const ok = cleanWarns === 0 && dirtyWarns >= 1;
console.log(ok ? "\n灵敏度回测通过：注了毒会响、不注毒不响" : "\n✗ 灵敏度回测失败 —— 这道告警是空转的");
process.exit(ok ? 0 : 1);
