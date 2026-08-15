// S2 DOM 冒烟：用最小 Node 桩验证 translateNode 的作用域表、属性白名单与折叠空白重试。
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const dir = path.dirname(fileURLToPath(import.meta.url));

globalThis.Node = { TEXT_NODE: 3, ELEMENT_NODE: 1 };

class T { constructor(v) { this.nodeType = 3; this.nodeValue = v; this.childNodes = []; } }
class E {
  constructor(attrs = {}, children = []) {
    this.nodeType = 1; this.attrs = attrs; this.childNodes = children;
  }
  getAttribute(k) { return this.attrs[k] ?? null; }
  setAttribute(k, v) { this.attrs[k] = v; }
}

let src = fs.readFileSync(path.join(dir, "s2_patched.mjs"), "utf8");
src = src.replace(/Hooks\.once\("ready"[\s\S]*$/, "");
src += `\nexport const __T = {translateNode, DIALOG_UI, ATTUNEMENT_TAB, DIALOG_TITLES, DIALOG_TITLE_PATTERNS};\n`;
const tmp = path.join(dir, "_s2_dom_mod.mjs");
fs.writeFileSync(tmp, src, "utf8");
const { __T } = await import("file://" + tmp.replace(/\\/g, "/"));
const { translateNode, DIALOG_UI, ATTUNEMENT_TAB, DIALOG_TITLES, DIALOG_TITLE_PATTERNS } = __T;

let fail = 0;
const eq = (got, want, label) => { if (got !== want) { console.log("FAIL", label, JSON.stringify(got), "want", JSON.stringify(want)); fail++; } };

// 1. 对话框按钮：不带作用域表不动，带上就翻
const btn = new T("Ring");
translateNode(btn);
eq(btn.nodeValue, "Ring", "btn-noscope");
translateNode(btn, DIALOG_UI);
eq(btn.nodeValue, "敲响", "btn-scoped");

// 2. 折叠空白：模板串跨行的整段正文
const p = new T("Resetting the event step for this event may introduce critical errors into your Ember game state. \n            Are you sure you wish to proceed?");
translateNode(p, DIALOG_UI);
eq(p.nodeValue, "重置该事件的步骤可能给你的余烬战役状态引入严重错误。确定要继续吗？", "flatten");

// 3. 折叠空白 + PATTERNS（带插值的跨行句）
const p2 = new T("The 迷炉商队 event is not currently available because its prerequisites are not \n            satisfied.");
translateNode(p2, DIALOG_UI);
eq(p2.nodeValue, "迷炉商队 事件当前不可用，其前置条件尚未满足。", "flatten-pattern");

// 4. 首尾空白保留
const p3 = new T("  Activate this elevator?  ");
translateNode(p3, DIALOG_UI);
eq(p3.nodeValue, "  启动这台升降机？  ", "keep-outer-ws");

// 5. 同调页签：aria-label + 11 个短名 + Active 标签
const tab = new E({}, [
  new E({}, [new T("Cosmological Attunements")]),
  new E({}, [new T("Abyss")]),
  new E({}, [new T("Active")]),
  new E({ "aria-label": "Make Active", "data-tooltip": "" })
]);
translateNode(tab, ATTUNEMENT_TAB);
eq(tab.childNodes[0].childNodes[0].nodeValue, "寰宇同调", "attn-h3");
eq(tab.childNodes[1].childNodes[0].nodeValue, "深渊", "attn-label");
eq(tab.childNodes[2].childNodes[0].nodeValue, "激活中", "attn-active");
eq(tab.childNodes[3].getAttribute("aria-label"), "设为激活", "attn-aria");

// 6. 幂等：再跑一遍不应变化
const before = JSON.stringify([tab.childNodes[0].childNodes[0].nodeValue, tab.childNodes[3].attrs]);
translateNode(tab, ATTUNEMENT_TAB);
eq(JSON.stringify([tab.childNodes[0].childNodes[0].nodeValue, tab.childNodes[3].attrs]), before, "idempotent");

// 7. 认框判据：ember.mjs 里抠出来的静态标题应当全部可认
const dump = JSON.parse(fs.readFileSync(
  "C:\\Users\\Taka\\Desktop\\fvtt\\Ember-Crucible Translation Project\\4-临时脚本\\2026-08-14-fix\\probes\\s2_dialog_dump.json", "utf8"));
const unknown = [];
for (const t of Object.keys(dump.titles)) {
  if (t.includes("${")) continue;                       // 动态标题，交给 DIALOG_TITLE_PATTERNS
  if (t in DIALOG_TITLES) continue;
  if (DIALOG_TITLE_PATTERNS.some(re => re.test(t))) continue;
  // ember.mjs:95615 那个 dialog:{} 少写了 window 层，DialogV2 读不到，
  // 实际标题落到基类兜底的 `Interactable: ${id}`，由 DIALOG_TITLE_PATTERNS 认。
  if (t === "Aedir Signalpost Stealth Field Generator") continue;
  unknown.push(t);
}
if (unknown.length) { console.log("认不出的静态标题:", unknown); fail += unknown.length; }

console.log(fail ? `FAILURES: ${fail}` : "DOM ALL PASS");
fs.unlinkSync(tmp);
process.exit(fail ? 1 : 0);
