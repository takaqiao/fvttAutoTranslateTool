// S2 冒烟：把打过补丁的 mjs 里的 Hooks 入口剥掉，跑 translateText / translateNode 的实际输出。
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const dir = path.dirname(fileURLToPath(import.meta.url));
let src = fs.readFileSync(path.join(dir, "s2_patched.mjs"), "utf8");
src = src.replace(/Hooks\.once\("ready"[\s\S]*$/, "");
// 暴露内部表供断言
src += `\nexport const __T = {translateNode, DIALOG_TITLES, DIALOG_UI, DIALOG_TITLE_PATTERNS, DIALOG_TITLE_I18N, ATTUNEMENT_TAB, INJECTED_SUBTREES, EXACT, PREFIXED, PATTERNS, MISSING_LANGUAGES};\n`;

const tmp = path.join(dir, "_s2_smoke_mod.mjs");
fs.writeFileSync(tmp, src, "utf8");

const mod = await import("file://" + tmp.replace(/\\/g, "/"));
const { translateText, __T } = mod;

const cases = [
  ["Ancestry: 人类 Human", "血统：人类 Human"],
  ["Culture: 阿克图里安 Arcturian", "文化：阿克图里安 Arcturian"],
  ["Path: 辛达里克入门者 Cindaric Initiate", "道途：辛达里克入门者 Cindaric Initiate"],
  ["Talent: 符文：控制 Rune: Control", "天赋：符文：控制 Rune: Control"],
  ["Attunement: Aura", "同调：奥拉"],
  ["Language: Pathward", "语言：径道语"],
  ["Ancestry", "血统"],
  ["Attunement", "同调"],
  ["Token", "令牌"],
  ["Day 43", "第 43 天"],
  ["Day 43 - 12:00", "第 43 天 - 12:00"],
  ["[[/language borel]]", "语言：博雷尔语"],
  ["[[/language kost]]", "语言：科斯特语"],
  ["Spell tooltips are still TO-DO.", "法术悬浮提示尚未实现。"],
  ["Ring Alarm Bell?", "敲响警钟？"],
  ["Install Junction Wheel", "安装枢纽轮盘"],
  ["Reset Event", "重置事件"],
  ["Temple Lunarium", "神殿月辉宫 Temple Lunarium"],
  ["Interactable: alarmBell1", "可交互物：alarmBell1"],
  ["Token Maker Part Usage: kiska/eyes/Fluffy2", "令牌制作器部件用量：kiska/eyes/Fluffy2"],
  ['Are you sure you wish to proceed and delete the "夜色" composition? This cannot be undone.',
    "确定要删除「夜色」构图吗？此操作无法撤销。"],
  ["Activate this mine cart with 阿卡里斯 Ankarist as its passenger?", "以 阿卡里斯 Ankarist 为乘客启动这辆矿车？"],
  ["There are downstream events of 迷炉商队 which have been started or completed.",
    "迷炉商队 存在已开始或已完成的下游事件。"],
  ["The 迷炉商队 event is not currently available because its prerequisites are not satisfied.",
    "迷炉商队 事件当前不可用，其前置条件尚未满足。"],
  ["Do you want to complete this event and transition the Party to the Pathways section of the Region map?",
    "是否完成此事件并将队伍转移到区域地图的通路 Pathways 区段？"],
  ["Do you want to transition the Party to the Pathways section of the Region map?",
    "是否将队伍转移到区域地图的通路 Pathways 区段？"],
  ["Award Attunement: 奥拉 Aura (+3)", "授予同调：奥拉 Aura (+3)"],
  ["Activate Attunement: Abyss", "激活同调：深渊"],
  // 作用域表：不带 extra 时必须原样返回
  ["Ring", "Ring"],
  ["Close", "Close"],
  ["Active", "Active"],
  ["Enable Flow", "Enable Flow"]
];

let fail = 0;
for (const [inp, want] of cases) {
  const got = translateText(inp);
  if (got !== want) { console.log("FAIL", JSON.stringify(inp), "->", JSON.stringify(got), "want", JSON.stringify(want)); fail++; }
}

// 作用域表命中
const scoped = [
  ["Ring", "敲响"], ["Close", "关闭"], ["Enable Flow", "开启流量"], ["Change", "更改"],
  ["Composition", "构图"], ["Forwards", "前进方向"], ["Interact", "交互"]
];
for (const [inp, want] of scoped) {
  const got = translateText(inp, __T.DIALOG_UI);
  if (got !== want) { console.log("FAIL(scoped)", inp, "->", got); fail++; }
}
const attn = [["Cosmological Attunements", "寰宇同调"], ["Make Active", "设为激活"], ["Active", "激活中"], ["Abyss", "深渊"], ["Signara", "西格纳拉"]];
for (const [inp, want] of attn) {
  const got = translateText(inp, __T.ATTUNEMENT_TAB);
  if (got !== want) { console.log("FAIL(attn)", inp, "->", got); fail++; }
}

// 折叠空白：模板串跨行的整句
const long = "Resetting the event step for this event may introduce critical errors into your Ember game state. \n            Are you sure you wish to proceed?";
const flat = long.trim().replace(/\s+/g, " ");
if (!(flat in __T.DIALOG_UI)) { console.log("FAIL flat key missing:", JSON.stringify(flat)); fail++; }

console.log(fail ? `FAILURES: ${fail}` : "ALL PASS");
fs.unlinkSync(tmp);
process.exit(fail ? 1 : 0);
