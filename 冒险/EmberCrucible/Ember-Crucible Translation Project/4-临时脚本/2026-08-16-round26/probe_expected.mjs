/**
 * 只做一件事：把候选正例逐条过一遍**发布中的** translateText，把实得译文打出来，
 * 供我逐条**人工核对**后再写进 RESOLUTIONS.assertions.json。
 *
 * ⚠ 不许反过来用 —— 「跑出什么就写什么」等于把断言的期望值定义成实现本身，
 *   那样的闸永远绿。下面每一条我都对着表／上游出处核过才收。
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const PROJ = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";
const HERE = path.join(PROJ, "4-临时脚本/2026-08-16-round26");
const SRC = path.join(PROJ, "1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs");
const IMPORT_LINE = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";

const src = fs.readFileSync(SRC, "utf8");
if (!src.includes(IMPORT_LINE)) throw new Error("找不到 SELFCHECK 的 import 行");
/* ⚠ 第二十七轮改：harness 写进 `mkdtemp`、import 完立刻删。
 *   原来写在 HERE（版本化目录）里，跑完就在轮次目录留下一份 3231 行的 `_h_expected.mjs` ——
 *   那是本项目登记的空转形态 (c) 的典型诱饵：今天与真身同源，上游一改就成过期快照，
 *   而它长得就像一份「现表」，会被人 import 去当现表读。harness 每次从 SRC 现生成，留着零收益。 */
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ec-r26-expected-"));
let translateText;
try {
  const harness = path.join(tmpDir, "_h_expected.mjs");
  fs.writeFileSync(harness,
    "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
    + src.replace(IMPORT_LINE,
      "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };"),
    "utf8");
  ({ translateText } = await import(pathToFileURL(harness).href));
} finally {
  fs.rmSync(tmpDir, { recursive: true, force: true });
}

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
  "You have 3 unspent ability points to allocate as part of your",
  "You have 1 unspent ability increases to allocate as part of your",
  "Are you sure you wish to proceed and delete the \"Foo\" composition? This cannot be undone.",
  "Activate this mine cart with Bob as its passenger?",
  "There are downstream events of Foo which have been started or completed.",
  "The Foo event is not currently available because its prerequisites are not satisfied.",
  "Do you want to transition the Party to the Pathways section of the Region map?",
  "Do you want to complete this event and transition the Party to the Pathways section of the Region map?",
  "the active attunement for Bob.",
  "the active attunement for Bob. You will lose",
  "the active attunement for Bob. You will gain",
  "\"foo\" - 3 matches in 2 locations",
  "\"foo\" - 1 match in 1 location",
  "No results found for \u201cfoo\u201d.",
  "Day, Generic",
];

const out = [];
for (const c of CASES) out.push([c, translateText(c)]);
fs.writeFileSync(path.join(HERE, "expected_probe.json"), JSON.stringify(out, null, 1), "utf8");
for (const [a, b] of out) console.log(`${JSON.stringify(a)}\n   -> ${JSON.stringify(b)}${a === b ? "   [未翻动]" : ""}`);
