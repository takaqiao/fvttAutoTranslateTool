> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 14 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 14: crit-result-parse-hardening —— 重伤结果不再靠切碎富文本按下标取值

**Files:**
- Create: `scripts/features/crit-result-parse-hardening.pure.mjs`
- Create: `scripts/features/crit-result-parse-hardening.mjs`
- Test: `test/crit-result-parse-hardening.test.mjs`
- Modify: `lang/en.json`、`lang/cn.json`
- Modify: `scripts/main.mjs`（只加两行：`/* AEA-ANCHOR: imports */` 后一条 import、`/* AEA-ANCHOR: features */` 后一个数组成员；不碰任何生命周期锚点、不碰 `api`）

**Interfaces:**

- Consumes:
  - `scripts/const.mjs` 的 `MID`（模组 id 字符串 `"alien-evolved-automation"`）与 `SYSTEM_ID`（`"alienrpg"`）
  - `kernel/features.mjs` 的 `features.register(def)`（`def = {id, default, gmOnly, requires, hint}`）与 `features.enabled(id) -> boolean`
  - `kernel/patches.mjs` 的 `patches.register(def)` / `patches.status()`；`def = {id, type, target, minSystem, fixedIn, probe(), apply()}`，`apply()` 是无参安装器、自己负责装包裹；`status()` 返回 `[{id, type, target, applied, reason, fixedIn}]`
  - `kernel/registry.mjs` 的 `registry.table(key) -> RollTable|null`，本任务只用两个键：`"critInjuryEvolved"`、`"critInjury1e"`
  - `kernel/selftest.mjs` 的 `selftest.register({id, label, run})`；`label` 传的是 **i18n 键**（登记发生在 `init`，那时语言包还没加载，`localize()` 只会回声键名），`run()` 返回 `{ok:boolean, detail:string}`
  - `scripts/main.mjs` 里已存在的 `/* AEA-ANCHOR: imports */` 与 `/* AEA-ANCHOR: features */` 两个锚点注释，以及 `const FEATURES = [...]` 数组；`init` 钩子里有 `for (const f of FEATURES) safely(..., () => f.register())`，`ready` 钩子里有 `for (const f of FEATURES) await safely(..., () => f.install())`

- Produces:
  - `crit-result-parse-hardening.pure.mjs`（纯层，**不得**引用任何 Foundry 全局，只导出 `pure*`）：
    `pureTimeLimitCode(name)->number`、`pureCritRowFields(html)->{injury,fatal,timeLimit,effects,healing}`、
    `pureFatalOf(text, labels)->{fatal,fatalMod}`、`pureTimeLimitOf(text, labels)->number`、
    `pureHealingOf(text, labels)->{kind,formula,label}`、`pureMarkerSet(english, localized)->string[]`、
    `pureParseCritSpec(spec)->Map`、`pureCritCatalog(variant)->Map|null`、
    `pureCritOutcome({variant,total,rowHtml,catalog,labels})->outcome`、
    `pureCritSkillMods(variant, total)->{mobility,rangedCbt,observation,manipulation,closeCbt,stamina,comtech,command}`、
    `pureHealingFix(planned, storedText, labels)->{action:"keep"|"permanent"|"none"|"roll", formula?}`、
    `pureCritParseIsBuggy(source)->boolean`
  - `crit-result-parse-hardening.mjs`（副作用层）：`critParseWrapper(wrapped, actor, type, dataset, manCrit)`、
    `probeCritResultParse()`、`export const critResultParseHardeningFeature = { id, register(), install() }`
  - `test/crit-result-parse-hardening.test.mjs`：**53 条脱桩真单测**（不装 Foundry 桩、不造任何全局），夹具逐字取自出货合集里的真实行
  - 三条自检条目：`crit-result-parse-hardening.probe`、`.tableRows`、`.legacyLookup`

**契约对表（先看这五条，免得验收时被误判成违规）**

1. **本任务用 `libWrapper.register` 是合法的。** 契约 §4/K1 判给 `rollBus` 独占的四个目标是
   `yzeRoll` / `abilityRoll` / `itemRoll` / `pushRoll`；本任务包的是
   `CONFIG.Actor.documentClass.prototype.rollCrit`，**不在那四个之内**，rollBus 也不提供
   `addStage("rollCrit")`。**不要**把它改写成 `rollBus.addStage(...)` —— 那个 target 不存在。
2. **开关与自检齐全。** `register()` 里 `features.register({id:"crit-result-parse-hardening", default:"full"})`，
   包装器**执行时**第一件事就是 `features.enabled(FEATURE_ID)`，关掉即原样放行、不需要重载世界；
   另有三条 `selftest.register(...)`。
3. **不装测试桩。** 全部 53 条测试都是纯函数真单测，只用对象字面量，**不 import `test/stubs/foundry.mjs`、
   不写 `globalThis.game`**。契约 §0.3 的桩行为契约对本任务不适用。
4. **不使用 `cards`。** 本特性不往聊天卡上塞面板，只在系统自己抛异常时按系统原样重建它自己那张卡，
   所以 §4/K8 那条「每条特性必须先在 mount 下建 `aea-<feature-id>` 子元素再 render」对本任务不适用。
5. **init 顺序。** 本特性的 `register()` 由 `main.mjs` 的 `for (const f of FEATURES)` 循环调用，
   而该循环排在 `features.registerSettings()` **之前**，因此 `register()` 里登记的 def 一定在设置生成时已在册。
   本任务不需要、也不得自己去调 `features.registerSettings()`。

**断言归属对照表**（哪条保障住在哪里；少一行就是掉了一条覆盖）

| 要保障的事 | 住在哪里 |
|---|---|
| 七段行文本（六段出货真实行 + 一段中文行）都能切出五格，缺格不抛 | vitest，22 条 |
| 两张 36 行表的行号目录逐行正确 | vitest，9 条 **加** 自检 `.tableRows`（拿目录去比对世界里真实绑定的那张表，逐行三项对照） |
| probe 双向可判（4.1.13 为真、修好后为假） | vitest，5 条 |
| 技能修正表、治疗时间改写决策、标签集拼装 | vitest，17 条 |
| libWrapper MIXED 真的装上了 | 自检 `.probe` + 手工验证 Step 21 |
| draw 观察器真的触发、且 `finally` 真的还原 | 手工验证 Step 21、Step 22（真实 libWrapper 链与 `RollTable#draw` 没有可忠实模拟的桩，不编造单测） |
| 系统自己的按名查表是否还能找到同一张表 | 自检 `.legacyLookup` + 手工验证 Step 24 |
| 系统抛异常后由本模组补出物品、状态与聊天卡 | 手工验证 Step 22 |
| 中文世界文案仍中文、机械值仍正确 | 手工验证 Step 23 |

---

**读者背景（不熟 Foundry / Alien RPG 的人先看这段）**

Foundry VTT 是一个浏览器里跑的桌游平台。「系统（system）」提供规则实现，「模组（module）」在其上打补丁。本模组用 `libWrapper` 这个第三方库去包裹系统的方法：`libWrapper.register(模组id, "点分路径", 包装函数, "MIXED")` 之后，调用那个方法会先进你的包装函数，第一个参数 `wrapped` 是原函数，其余参数是原本的入参；`MIXED` 表示你既可以调 `wrapped` 也可以不调。

`RollTable` 是 Foundry 的随机表文档，`await table.draw({displayChat:false})` 掷一次并返回 `{roll, results}`：`roll.total` 是骰值，`results[0].description` 是抽中那行的富文本。Alien RPG 的重伤（Critical Injury）表是 D66（两颗 d6 读成十位+个位，共 36 行），行号形如 11、12…66。

**本任务要修的缺陷**

每行富文本长这样：

```html
<b>INJURY: </b>Arm artery cut <br /><b>FATAL: </b>Yes, –1 <br /><b>TIME LIMIT: </b>One Turn <br /><b>EFFECTS: </b>Can’t use arm. <br /><b>HEALING TIME: </b>[[1d6]] days
```

系统在 `module/documents/actor.mjs:1782` 的 `rollCrit(actor, type, dataset, manCrit)` 里这样读（下面每行我都重新打开核对过）：

```js
// :1869 / :1873 —— manCrit 为空时直接掷；给了 manCrit 就先 new Roll(manCrit).evaluate() 再把 roll 传给 draw
test1 = await atable.draw({ displayChat: false });
test1 = await atable.draw({ roll: roll, displayChat: false });
// :1875
const messG = test1.results[0].description;
// :1880-1885
cleanText = messG.replace(/(<b>)|(<p>|)(<strong>)|(<\/b>)|(<\/p>)|(<\/strong>)/gi, "");
factorFour = cleanText.replace(/<br \/>/gi, "<br>");
testArray = factorFour.split(/[:] |<br>/gi);      // 按「冒号空格」或 <br> 切碎
// :1887-1889
let speanex = testArray[7];                        // 第 7 段当 EFFECTS
if (testArray[9] !== game.i18n.localize("ALIENRPG.Permanent")) {
  if (testArray[9].length > 0) { ... }             // 第 9 段当 HEALING TIME
// :1902-1921
switch (testArray[3]) {                            // 第 3 段当 FATAL
  case game.i18n.localize("ALIENRPG.Yes") + " ":              // 带尾随空格
  case game.i18n.localize("ALIENRPG.Yes") + ", –1 ":          // EN DASH U+2013，:1909 追加医疗急救罚值备注
  case game.i18n.localize("ALIENRPG.Yes") + ", –2 ":          // :1915 同上
  default: cFatal = false;
}
// :1923-1942
switch (testArray[5]) {                            // 第 5 段当 TIME LIMIT
  case game.i18n.localize("ALIENRPG.OneShift") + " ": healTime = 3; break;
  default: healTime = 0; break;
}
```

结果写进角色身上的物品（`:2050-2069`）：`system.attributes.fatal` = `cFatal`，
`system.attributes.timelimit.value` = `healTime`（0–4 的码，`module/helpers/config.mjs:526-532`），
`system.attributes.healingtime.value` = `testArray[9]`，`system.attributes.effects` = `speanex`；
聊天卡数据在 `:2074-2085`（三格一律印**原始单元格文字**），卡在 `:2173` 渲染、`:2210-2211` 发出，
状态图标在 `:2199` 切换。技能修正两张表在 `:1944-1998`（Evolved）与 `:2000-2045`（一版）。

**三条已实证的失效**（下面每条都在出货合集里逐行核对过，复核命令见 Step 8）

1. **`XmFybNUYJ4C2OQyE`（1e 版 "Critical injuries"）第 11、12 行会崩。** 这两行的 FATAL / TIME LIMIT / HEALING TIME 全为空，且结尾写作 `<strong>HEALING TIME:</strong></p>` —— 冒号后**没有空格**，`/[:] /` 切不出分隔符。整行只切出 9 段（下标 0–8），`testArray[9]` 是 `undefined`，`:1889` 的 `.length` 抛 TypeError。玩家点 Roll Crit：不加物品、不发卡、不弹警告。
2. **`PqqpXQ1aPnzmb6RP`（Evolved 版 "EV - Critical Injuries"）的 TIME LIMIT 全部失效。** 这张表写的是 `<strong>TIME LIMIT: </strong>Shift<br />` —— 值是裸的 `Shift`，**没有尾随空格**、没有 "One " 前缀。系统拿它去比 `"One Shift "`，永不匹配，落 `default: healTime = 0`。于是 Punctured Lung(44)、Bleeding Gut(45)、Infected Wound(46)、Cracked Spine(52)、Ruptured Jugular(61) 这些致命伤，装到角色卡上的物品一律写 **TIME LIMIT: None**。聊天卡是对的（`:2076` 印的是原始单元格文字），所以 GM 盯聊天卡看不出问题。53–56 行的时限是 `Stretch`，系统压根没有对应码（`config.mjs:526-532` 只到第 4 档）。
3. **中文世界（本项目的实际部署形态）里静默失灵。** 已核对 `systems/alienrpg/lang/cn.json`：`ALIENRPG.Yes` 仍是 `"Yes"`、`ALIENRPG.Permanent` 仍是 `"Permanent"`（两个键都没翻）。表被 Babele 译成中文后 FATAL 那格是「是」，去比 `localize("ALIENRPG.Yes") + " "` 永不相等 → **`cFatal` 恒 `false`**；TIME LIMIT 同理恒落 `default` → **`healTime` 恒 `0`**；`:1909`/`:1915` 那两句医疗急救罚值备注也永远追加不上。若翻译改用了全角冒号 `：`，`/[:] /` 连切都切不动，`testArray[9]` 变 `undefined`，退化成第 1 条那样整次调用抛异常。两种形态本任务都覆盖（前者走「落地后改写」，后者走「接住异常自行重建」）。

**修法**：把机械数值（致命与否、致命修正、时限档、治疗时间骰式、技能修正）从「切文本」换成「按 D66 行号查表」—— 这些数值是**行号**的函数，不是文案的函数，翻译改不了行号。文案（伤名、效果描述）仍从表里读，中文世界照样显示中文。碰到 GM 自制的家规表（行号对不上目录），退回一个比系统宽容得多的解析器：接受 `<br>` / `<br />` / `</p>` 三种换行、半角与全角冒号、各种连字符（U+2010–U+2015、U+2212 统一归一成 `-`）、有无 "One " 前缀与尾随空格；标签集由调用方注入（英文默认 + 世界语言 + 模组自带的中文标记），任何一格缺失都返回空值而**绝不抛异常**。

**接缝**：`libWrapper` 以 `MIXED` 包 `CONFIG.Actor.documentClass.prototype.rollCrit`，只处理 `type === "character"` 分支 —— 这不是偷懒：`synthetic` / `creature` 分支（`:2092-2124`）只把行文本切成「名称 + 效果」两段写进物品，**根本没有 fatal / timelimit / healingtime 三个字段**，没有可硬化的机械数值；`spacecraft` 分支（`:2126-2160`）同理。契约 §4 K2 声明的 `critInjurySynthetic` / `critInjuryXeno` 两个键是给别的消费者用的，本任务只消费 `critInjuryEvolved`（`PqqpXQ1aPnzmb6RP`）与 `critInjury1e`（`XmFybNUYJ4C2OQyE`）。

为了知道系统抽到哪一行，包装器在调用原函数期间给那张 RollTable 文档挂一个一次性的 `draw` 观察器（`finally` 必还原）—— 只观察不改动，所以掷骰分布、Dice So Nice 动画、聊天卡措辞都不变。原函数抛异常时（上面第 1、3 条），包装器接住并自己补上物品、状态与聊天卡。

**一条如实记录的边界**：走「落地后改写」那条路径时，本特性只改 `fatal` / `timelimit.value` / `healingtime.value` 三个字段，**不重写 `effects`** —— 那一格是系统从行里切出来的原文，重写有清掉 GM 手改内容的风险。因此中文世界里若系统没抛异常，`:1909`/`:1915` 那句「-1 到医疗急救」备注仍会缺失（致命标记与时限已被修正）。只有走「接住异常自行重建」那条路径时，本特性才会按行号目录把这句备注补回去。

---

- [ ] **Step 1: 写会失败的解析器测试（用真实行文本作夹具）**

新建 `test/crit-result-parse-hardening.test.mjs`。下面 7 段 `FIXTURE_*` 里，前 6 段是从出货合集逐字 dump 出来的真实行（Step 8 给了复核命令），**不要改动一个字符**，尤其是 U+2013 EN DASH（`–`）与 U+2019 右单引号（`’`）：

```js
import { describe, expect, it } from "vitest";
import {
  pureCritRowFields,
  pureFatalOf,
  pureHealingOf,
  pureTimeLimitCode,
  pureTimeLimitOf,
} from "../scripts/features/crit-result-parse-hardening.pure.mjs";

// RollTable XmFybNUYJ4C2OQyE "Critical injuries" (1e), row [11,11] — crashes the system today.
const FIXTURE_1E_11 =
  "<p><strong>INJURY: </strong>Winded <br /><strong>FATAL: </strong><br /><strong>TIME LIMIT: </strong><br /><strong>EFFECTS: </strong><br /><strong>HEALING TIME:</strong></p>";
// row [52,52] — the "Yes, –1" shape with an EN DASH.
const FIXTURE_1E_52 =
  "<b>INJURY: </b>Arm artery cut <br /><b>FATAL: </b>Yes, –1 <br /><b>TIME LIMIT: </b>One Turn <br /><b>EFFECTS: </b>Can’t use arm. <br /><b>HEALING TIME: </b>[[1d6]] days";
// row [54,54] — healing time "Permanent".
const FIXTURE_1E_54 =
  "<b>INJURY: </b>Severed arm <br /><b>FATAL: </b>Yes, –1 <br /><b>TIME LIMIT: </b>One Shift <br /><b>EFFECTS: </b>Can’t use arm. <br /><b>HEALING TIME: </b>Permanent";
// RollTable PqqpXQ1aPnzmb6RP "EV - Critical Injuries", row [44,44] — "Shift", no trailing space.
const FIXTURE_EV_44 =
  "<p><strong>INJURY: </strong>Punctured Lung <br /><strong>FATAL: </strong>Yes <br /><strong>TIME LIMIT: </strong>Shift<br /><strong>EFFECTS: </strong>–2 dice on MOBILITY and CLOSE COMBAT. Needs surgery.<br /><strong>HEALING TIME: </strong>[[1d6]] days</p>";
// row [53,53] — "Stretch", a unit the system has no code for.
const FIXTURE_EV_53 =
  "<p><strong>INJURY: </strong>Arm Arterial Bleeding <br /><strong>FATAL: </strong>Yes <br /><strong>TIME LIMIT: </strong>Stretch<br /><strong>EFFECTS: </strong>–1 die on all rolls that normally require two arms.<br /><strong>HEALING TIME: </strong>[[1d6]] days</p>";
// row [66,66] — instant death, empty healing cell, a <span> the system never strips.
const FIXTURE_EV_66 =
  "<p><strong>INJURY: </strong>Impaled Heart <br /><strong>FATAL: </strong>Yes <br /><strong>TIME LIMIT: </strong>–<br /><strong>EFFECTS: </strong><span style=\"color:red;font-size:larger;font-weight:bold\"><strong>Your heart beats for the last time.</strong></span><br /><strong>HEALING TIME:</strong><br /></p>";

// What a Babele-translated world can hand the parser: every English marker is gone,
// and the colon is the full-width one.
const FIXTURE_CN_44 =
  "<p><strong>伤势：</strong>肺部穿孔<br /><strong>致命：</strong>是<br /><strong>时限：</strong>一轮班<br /><strong>效果：</strong>机动与近战 −2 骰<br /><strong>治疗时间：</strong>[[1d6]] 天</p>";

// A hand-built marker set. `round` is deliberately "一轮", a PREFIX of the shift marker
// "一轮班", so that the longest-match rule is actually exercised; the system's own
// cn.json has OneRound = "一回合", which would never collide, so only a hand-built set
// can cover this. The effect layer builds the real one by localizing a fixed key list.
const CN_LABELS = {
  yes: ["yes", "是"],
  permanent: ["permanent", "永久"],
  none: ["none", "无"],
  units: {
    round: ["round", "一轮"],
    turn: ["turn", "一回合"],
    shift: ["shift", "一轮班"],
    day: ["day", "一天"],
    stretch: ["stretch", "一段时间"],
  },
};

describe("pureCritRowFields", () => {
  it("reads all five cells out of a 1e row", () => {
    const f = pureCritRowFields(FIXTURE_1E_52);
    expect(f.injury).toBe("Arm artery cut");
    expect(f.fatal).toBe("Yes, –1");
    expect(f.timeLimit).toBe("One Turn");
    expect(f.effects).toBe("Can’t use arm.");
    expect(f.healing).toBe("[[1d6]] days");
  });
  it("returns empty cells instead of throwing on the row that crashes the system", () => {
    const f = pureCritRowFields(FIXTURE_1E_11);
    expect(f.injury).toBe("Winded");
    expect(f.fatal).toBe("");
    expect(f.timeLimit).toBe("");
    expect(f.healing).toBe("");
  });
  it("reads an EV row whose cells carry no trailing space", () => {
    const f = pureCritRowFields(FIXTURE_EV_44);
    expect(f.fatal).toBe("Yes");
    expect(f.timeLimit).toBe("Shift");
    expect(f.healing).toBe("[[1d6]] days");
  });
  it("strips the <span> the system leaves in the effects cell", () => {
    expect(pureCritRowFields(FIXTURE_EV_66).effects).toBe("Your heart beats for the last time.");
  });
  it("splits on the full-width colon a translated row uses", () => {
    const f = pureCritRowFields(FIXTURE_CN_44);
    expect(f.injury).toBe("肺部穿孔");
    expect(f.fatal).toBe("是");
    expect(f.timeLimit).toBe("一轮班");
  });
  it("survives a row with no separators at all", () => {
    expect(pureCritRowFields("")).toEqual({ injury: "", fatal: "", timeLimit: "", effects: "", healing: "" });
  });
});

describe("pureFatalOf", () => {
  it("reads a plain Yes", () => {
    expect(pureFatalOf(pureCritRowFields(FIXTURE_EV_44).fatal)).toEqual({ fatal: true, fatalMod: 0 });
  });
  it("reads the medical-aid penalty behind an EN DASH", () => {
    expect(pureFatalOf(pureCritRowFields(FIXTURE_1E_52).fatal)).toEqual({ fatal: true, fatalMod: -1 });
  });
  it("reads an ASCII hyphen just the same", () => {
    expect(pureFatalOf("Yes, -2")).toEqual({ fatal: true, fatalMod: -2 });
  });
  it("is not fatal when the cell is empty", () => {
    expect(pureFatalOf("")).toEqual({ fatal: false, fatalMod: 0 });
  });
  it("is not fatal on No", () => {
    expect(pureFatalOf("No")).toEqual({ fatal: false, fatalMod: 0 });
  });
  it("reads a translated affirmative when the caller injects the world's labels", () => {
    expect(pureFatalOf(pureCritRowFields(FIXTURE_CN_44).fatal, CN_LABELS)).toEqual({ fatal: true, fatalMod: 0 });
  });
});

describe("pureTimeLimitOf", () => {
  it("reads the 1e 'One Turn' wording", () => {
    expect(pureTimeLimitOf("One Turn")).toBe(pureTimeLimitCode("turn"));
  });
  it("reads the EV 'Shift' wording that the system misses entirely", () => {
    expect(pureTimeLimitOf(pureCritRowFields(FIXTURE_EV_44).timeLimit)).toBe(pureTimeLimitCode("shift"));
  });
  it("reads Stretch, a unit the system has no code for", () => {
    expect(pureTimeLimitOf(pureCritRowFields(FIXTURE_EV_53).timeLimit)).toBe(pureTimeLimitCode("stretch"));
  });
  it("treats an EN DASH placeholder as no time limit", () => {
    expect(pureTimeLimitOf(pureCritRowFields(FIXTURE_EV_66).timeLimit)).toBe(pureTimeLimitCode("none"));
  });
  it("treats an empty cell as no time limit", () => {
    expect(pureTimeLimitOf("")).toBe(pureTimeLimitCode("none"));
  });
  it("prefers the longest matching translated marker", () => {
    // "一轮班" (shift) must win over "一轮" (round), which is a prefix of it.
    expect(pureTimeLimitOf(pureCritRowFields(FIXTURE_CN_44).timeLimit, CN_LABELS)).toBe(pureTimeLimitCode("shift"));
  });
});

describe("pureHealingOf", () => {
  it("extracts the dice formula out of an inline roll", () => {
    expect(pureHealingOf(pureCritRowFields(FIXTURE_EV_44).healing))
      .toEqual({ kind: "formula", formula: "1d6", label: "[[1d6]] days" });
  });
  it("recognises Permanent", () => {
    expect(pureHealingOf(pureCritRowFields(FIXTURE_1E_54).healing))
      .toEqual({ kind: "permanent", formula: null, label: "Permanent" });
  });
  it("recognises an empty healing cell", () => {
    expect(pureHealingOf(pureCritRowFields(FIXTURE_EV_66).healing))
      .toEqual({ kind: "none", formula: null, label: "" });
  });
  it("still finds the formula in a translated cell", () => {
    expect(pureHealingOf(pureCritRowFields(FIXTURE_CN_44).healing, CN_LABELS).formula).toBe("1d6");
  });
});
```

- [ ] **Step 2: 跑它，看它失败**

Run: `npx vitest run test/crit-result-parse-hardening.test.mjs`

Expected: FAIL —— `Failed to resolve import "../scripts/features/crit-result-parse-hardening.pure.mjs" from "test/crit-result-parse-hardening.test.mjs"`，22 条用例一条都没执行。

- [ ] **Step 3: 写宽容解析器**

新建 `scripts/features/crit-result-parse-hardening.pure.mjs`：

```js
// Pure layer. Must not reference any Foundry global (game / ui / canvas / CONFIG /
// Hooks / foundry / ChatMessage / Roll / libWrapper). Only pure* symbols are exported.

/**
 * Time-limit codes. 0-4 are the system's own (module/helpers/config.mjs:526-532,
 * CONFIG.ALIENRPG.crit_timelimit_list). 5 is new: the Evolved table uses "Stretch",
 * which the system has no code for at all.
 */
const TIME_LIMIT_CODE = Object.freeze({ none: 0, round: 1, turn: 2, shift: 3, day: 4, stretch: 5 });

/** Default (English) marker set, used when the caller injects nothing. */
const DEFAULT_LABELS = Object.freeze({
  yes: ["yes"],
  permanent: ["permanent"],
  none: ["none"],
  units: { round: ["round"], turn: ["turn"], shift: ["shift"], day: ["day"], stretch: ["stretch"] },
});

export function pureTimeLimitCode(name) {
  return TIME_LIMIT_CODE[name] ?? TIME_LIMIT_CODE.none;
}

/** U+2010..U+2015 (all the dashes) and U+2212 (minus sign) become an ASCII hyphen. */
function normalize(value) {
  return String(value ?? "").replace(/[\u2010-\u2015\u2212]/g, "-").trim();
}

function startsWithAny(value, markers) {
  return (markers ?? []).some((m) => m && value.startsWith(String(m).toLowerCase()));
}

/**
 * Split one RollTable row into its five labelled cells. Tolerates <br>, <br/>,
 * <br />, </p>, stray <span>, &nbsp;, the full-width colon, missing cells and
 * missing trailing spaces. Never throws.
 * The system instead splits on /[:] |<br>/ (actor.mjs:1885) and then indexes
 * testArray[3]/[5]/[9], which is undefined on several shipped rows.
 */
export function pureCritRowFields(html) {
  const text = String(html ?? "")
    .replace(/<br[^>]*>/gi, "\n")
    .replace(/<\/p[^>]*>/gi, "\n")
    .replace(/<[^>]*>/g, "")
    .replace(/&nbsp;/gi, " ");

  const cells = [];
  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (!line) continue;
    // Both the ASCII colon and the full-width colon a translation will use.
    let at = line.indexOf(":");
    if (at < 0) at = line.indexOf("\uFF1A");
    if (at < 0) {
      // No label on this line: it is a continuation of the previous cell.
      if (cells.length) cells[cells.length - 1] = `${cells[cells.length - 1]} ${line}`.trim();
      continue;
    }
    cells.push(line.slice(at + 1).trim());
  }

  const at = (index) => cells[index] ?? "";
  return { injury: at(0), fatal: at(1), timeLimit: at(2), effects: at(3), healing: at(4) };
}

/** "Yes", "Yes, –1", "Yes, -2", "No", "是", "" -> {fatal, fatalMod}. */
export function pureFatalOf(text, labels = DEFAULT_LABELS) {
  const value = normalize(text).toLowerCase();
  if (!startsWithAny(value, labels.yes)) return { fatal: false, fatalMod: 0 };
  const penalty = value.match(/[-][ ]*([0-9]+)/);
  return { fatal: true, fatalMod: penalty ? -Number(penalty[1]) : 0 };
}

/** "One Shift", "Shift", "Stretch", "一轮班", "–", "" -> a time-limit code. */
export function pureTimeLimitOf(text, labels = DEFAULT_LABELS) {
  const value = normalize(text).toLowerCase().replace(/^one[ ]+/, "");
  if (!value || value === "-") return TIME_LIMIT_CODE.none;
  if (startsWithAny(value, labels.none)) return TIME_LIMIT_CODE.none;

  const pairs = [];
  for (const [unit, markers] of Object.entries(labels.units ?? {})) {
    for (const marker of markers ?? []) pairs.push([unit, String(marker).toLowerCase()]);
  }
  // Longest marker first, so "一轮班" (shift) beats "一轮" (round), which is its prefix.
  pairs.sort((a, b) => b[1].length - a[1].length);
  for (const [unit, marker] of pairs) {
    if (marker && value.startsWith(marker)) return pureTimeLimitCode(unit);
  }
  return TIME_LIMIT_CODE.none;
}

/** "[[1d6]] days", "Permanent", "永久", "Shift", "" -> {kind, formula, label}. */
export function pureHealingOf(text, labels = DEFAULT_LABELS) {
  const label = normalize(text);
  if (!label || label === "-") return { kind: "none", formula: null, label: "" };
  if (startsWithAny(label.toLowerCase(), labels.permanent)) return { kind: "permanent", formula: null, label };

  const open = label.indexOf("[[");
  const close = label.indexOf("]]");
  const inner = open >= 0 && close > open ? label.slice(open + 2, close).trim() : label;
  if (/^[0-9]+d[0-9]+$/.test(inner)) return { kind: "formula", formula: inner, label };
  return { kind: "text", formula: null, label };
}
```

- [ ] **Step 4: 跑它，看它通过**

Run: `npx vitest run test/crit-result-parse-hardening.test.mjs`

Expected: PASS —— `Tests  22 passed (22)`。

- [ ] **Step 5: 提交**

```bash
git add scripts/features/crit-result-parse-hardening.pure.mjs test/crit-result-parse-hardening.test.mjs && git commit -m "feat(crit-result-parse-hardening): 宽容的重伤行解析器，夹具取自出货表真实行" -m "系统 actor.mjs:1885 按 /[:] |<br>/ 切碎后取 testArray[3]/[5]/[9]，
比对目标带尾随空格、其中两个用 EN DASH U+2013，还有一个是裸字面量 Shift。
夹具是从 alien-evolved-corerules 合集 dump 出来的真实行：
- XmFybNUYJ4C2OQyE 的 11 行只切得出 9 段，testArray[9] 为 undefined，:1889 抛 TypeError；
- PqqpXQ1aPnzmb6RP 的 44 行时限写作 Shift（无尾随空格、无 One 前缀），比不上 One Shift。
新解析器接受三种换行、半角与全角冒号、各种连字符、有无 One 前缀与尾随空格，
标签集由调用方注入，中英两套标签跑同一批夹具，任何一格缺失都返回空值不抛。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: 写会失败的「按行号查表」测试**

追加到测试文件末尾，并把 import 列表补成 `pureCritCatalog, pureCritOutcome, pureCritRowFields, pureFatalOf, pureHealingOf, pureParseCritSpec, pureTimeLimitCode, pureTimeLimitOf`：

```js
describe("pureParseCritSpec", () => {
  it("turns the compact spec into a lookup keyed by D66 total", () => {
    const map = pureParseCritSpec("44 1 0 S 1d6|56 0 0 - 3d6|54 1 -1 S P|14 0 0 - T");
    expect(map.get(44)).toEqual({ fatal: true, fatalMod: 0, timeLimit: pureTimeLimitCode("shift"), healing: { kind: "formula", formula: "1d6" } });
    expect(map.get(56)).toEqual({ fatal: false, fatalMod: 0, timeLimit: pureTimeLimitCode("none"), healing: { kind: "formula", formula: "3d6" } });
    expect(map.get(54).healing).toEqual({ kind: "permanent", formula: null });
    expect(map.get(14).healing).toEqual({ kind: "text", formula: null });
  });
});

describe("pureCritCatalog", () => {
  it("carries all 36 rows of both shipped tables", () => {
    expect(pureCritCatalog("evolved").size).toBe(36);
    expect(pureCritCatalog("first").size).toBe(36);
  });
  it("returns null for a variant it does not know", () => {
    expect(pureCritCatalog("spacecraft")).toBe(null);
  });
  it("knows the Evolved time limits the system reads as None", () => {
    const ev = pureCritCatalog("evolved");
    expect(ev.get(44).timeLimit).toBe(pureTimeLimitCode("shift"));
    expect(ev.get(45).timeLimit).toBe(pureTimeLimitCode("shift"));
    expect(ev.get(52).timeLimit).toBe(pureTimeLimitCode("shift"));
    expect(ev.get(53).timeLimit).toBe(pureTimeLimitCode("stretch"));
    expect(ev.get(61).timeLimit).toBe(pureTimeLimitCode("round"));
  });
  it("knows the Evolved rows whose healing cell is free text, not a formula", () => {
    // Rows 14 and 15 say "HEALING TIME: Shift"; the system special-cases exactly this
    // string at actor.mjs:1890-1891 and we must not re-roll it into a number.
    expect(pureCritCatalog("evolved").get(14).healing).toEqual({ kind: "text", formula: null });
    expect(pureCritCatalog("evolved").get(15).healing).toEqual({ kind: "text", formula: null });
  });
  it("knows the 1e rows that crash the parser today", () => {
    const first = pureCritCatalog("first");
    expect(first.get(11)).toEqual({ fatal: false, fatalMod: 0, timeLimit: pureTimeLimitCode("none"), healing: { kind: "none", formula: null } });
    expect(first.get(12).fatal).toBe(false);
    expect(first.get(62)).toEqual({ fatal: true, fatalMod: -2, timeLimit: pureTimeLimitCode("round"), healing: { kind: "formula", formula: "3d6" } });
  });
  it("knows the 1e row 44 is One Day, not the Shift the Evolved table uses", () => {
    expect(pureCritCatalog("first").get(44).timeLimit).toBe(pureTimeLimitCode("day"));
    expect(pureCritCatalog("evolved").get(44).timeLimit).toBe(pureTimeLimitCode("shift"));
  });
});

describe("pureCritOutcome", () => {
  it("prefers the catalog and still takes the prose from the row", () => {
    const out = pureCritOutcome({ variant: "evolved", total: 44, rowHtml: FIXTURE_EV_44, catalog: pureCritCatalog("evolved") });
    expect(out.source).toBe("catalog");
    expect(out.fatal).toBe(true);
    expect(out.timeLimit).toBe(pureTimeLimitCode("shift"));
    expect(out.healing).toEqual({ kind: "formula", formula: "1d6" });
    expect(out.healingRaw).toBe("[[1d6]] days");
    expect(out.injury).toBe("Punctured Lung");
  });
  it("still gets the mechanics right when the row has been translated", () => {
    const parsedOnly = pureCritOutcome({ variant: "evolved", total: 44, rowHtml: FIXTURE_CN_44, catalog: null });
    expect(parsedOnly.fatal).toBe(false);                            // exactly what the system produces today
    expect(parsedOnly.timeLimit).toBe(pureTimeLimitCode("none"));
    const out = pureCritOutcome({ variant: "evolved", total: 44, rowHtml: FIXTURE_CN_44, catalog: pureCritCatalog("evolved") });
    expect(out.source).toBe("catalog");
    expect(out.fatal).toBe(true);
    expect(out.timeLimit).toBe(pureTimeLimitCode("shift"));
    expect(out.injury).toBe("肺部穿孔");
  });
  it("falls back to the parser for a house-ruled row the catalog does not know", () => {
    const out = pureCritOutcome({ variant: "evolved", total: 99, rowHtml: FIXTURE_1E_52, catalog: pureCritCatalog("evolved") });
    expect(out.source).toBe("parsed");
    expect(out.fatal).toBe(true);
    expect(out.fatalMod).toBe(-1);
    expect(out.timeLimit).toBe(pureTimeLimitCode("turn"));
  });
  it("never throws on the 1e row that kills the system", () => {
    const out = pureCritOutcome({ variant: "first", total: 11, rowHtml: FIXTURE_1E_11, catalog: pureCritCatalog("first") });
    expect(out.fatal).toBe(false);
    expect(out.timeLimit).toBe(pureTimeLimitCode("none"));
    expect(out.injury).toBe("Winded");
  });
});
```

- [ ] **Step 7: 跑它，看它失败**

Run: `npx vitest run test/crit-result-parse-hardening.test.mjs -t "pureCritOutcome"`

Expected: FAIL —— 4 条 `pureCritOutcome` 用例全红，报 `TypeError: pureCritOutcome is not a function`（`pureCritCatalog` / `pureParseCritSpec` 同样是 `undefined`，因为 `.pure.mjs` 还没导出它们）。

- [ ] **Step 8: 写行号目录**

追加到 `scripts/features/crit-result-parse-hardening.pure.mjs`。

两张 36 行的表是逐行 dump 后誊录、并用脚本把每一行的解析结果与誊录值逐格比对过的（72 行零分歧）。要自己复核，在一个装了 `classic-level` 的目录下跑（合集里装的是 Adventure 文档，表在 `adventure.tables` 数组里，所以要多剥一层）：

```bash
node --input-type=module -e '
import { ClassicLevel } from "classic-level";
const db = new ClassicLevel("C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-corerules/packs/alien-evolved-core-rules", { valueEncoding: "json" });
const want = new Set(["PqqpXQ1aPnzmb6RP", "XmFybNUYJ4C2OQyE"]);
for await (const [, v] of db.iterator()) {
  for (const t of (Array.isArray(v?.tables) ? v.tables : [])) {
    if (!want.has(t?._id)) continue;
    console.log("=== ", t._id, t.name, t.results.length);
    for (const r of t.results) console.log(JSON.stringify({ range: r.range, d: r.description }));
  }
}
await db.close();'
```

```js
/**
 * Compact row spec: "<d66> <fatal 0|1> <fatalMod> <timeLimit> <healing>", rows joined by "|".
 *   timeLimit: - none, R round, T turn, S shift, D day, X stretch
 *   healing:   - none, P permanent, T free text taken verbatim from the row, or a dice formula (1d6)
 */
export function pureParseCritSpec(spec) {
  const TIME = { "-": "none", R: "round", T: "turn", S: "shift", D: "day", X: "stretch" };
  const map = new Map();
  for (const row of String(spec ?? "").split("|")) {
    const parts = row.trim().split(" ").filter(Boolean);
    if (parts.length !== 5) continue;
    const [total, fatal, fatalMod, timeLimit, healing] = parts;
    map.set(Number(total), {
      fatal: fatal === "1",
      fatalMod: Number(fatalMod),
      timeLimit: pureTimeLimitCode(TIME[timeLimit] ?? "none"),
      healing:
        healing === "-" ? { kind: "none", formula: null }
        : healing === "P" ? { kind: "permanent", formula: null }
        : healing === "T" ? { kind: "text", formula: null }
        : { kind: "formula", formula: healing },
    });
  }
  return map;
}

/** RollTable PqqpXQ1aPnzmb6RP "EV - Critical Injuries", transcribed row by row. */
const EVOLVED_SPEC = [
  "11 0 0 - -", "12 0 0 - -", "13 0 0 - -", "14 0 0 - T", "15 0 0 - T", "16 0 0 - 1d6",
  "21 0 0 - 1d6", "22 0 0 - 1d6", "23 0 0 - 1d6", "24 0 0 - 1d6", "25 0 0 - 2d6", "26 0 0 - 1d6",
  "31 0 0 - 1d6", "32 0 0 - 2d6", "33 0 0 - 2d6", "34 0 0 - 2d6", "35 0 0 - 3d6", "36 0 0 - 3d6",
  "41 0 0 - 3d6", "42 0 0 - 2d6", "43 0 0 - -", "44 1 0 S 1d6", "45 1 0 S 1d6", "46 1 0 S -",
  "51 1 0 S 3d6", "52 1 0 S 4d6", "53 1 0 X 1d6", "54 1 0 X 1d6", "55 1 0 X 4d6", "56 1 0 X 4d6",
  "61 1 0 R 2d6", "62 1 0 R 3d6", "63 1 0 R 4d6", "64 1 0 - -", "65 1 0 - -", "66 1 0 - -",
].join("|");

/** RollTable XmFybNUYJ4C2OQyE "Critical injuries" (first edition), transcribed row by row. */
const FIRST_SPEC = [
  "11 0 0 - -", "12 0 0 - -", "13 0 0 - -", "14 0 0 - -", "15 0 0 - -", "16 0 0 - 1d6",
  "21 0 0 - 1d6", "22 0 0 - 1d6", "23 0 0 - 1d6", "24 0 0 - 1d6", "25 0 0 - 2d6", "26 0 0 - 2d6",
  "31 0 0 - 1d6", "32 0 0 - 1d6", "33 0 0 - 2d6", "34 0 0 - 2d6", "35 0 0 - 2d6", "36 0 0 - 2d6",
  "41 0 0 - 2d6", "42 0 0 - 3d6", "43 0 0 - 3d6", "44 1 0 D 1d6", "45 1 0 S 1d6", "46 1 0 S 2d6",
  "51 1 0 D 2d6", "52 1 -1 T 1d6", "53 1 -1 T 1d6", "54 1 -1 S P", "55 1 -1 S P", "56 0 0 - 3d6",
  "61 1 -1 R 2d6", "62 1 -2 R 3d6", "63 1 0 - -", "64 1 0 - -", "65 1 0 - -", "66 1 0 - -",
].join("|");

const CATALOGS = Object.freeze({ evolved: pureParseCritSpec(EVOLVED_SPEC), first: pureParseCritSpec(FIRST_SPEC) });

export function pureCritCatalog(variant) {
  return CATALOGS[variant] ?? null;
}

/**
 * Mechanics come from the catalog, keyed on the D66 row number, because fatal /
 * time limit / healing are a function of the row, not of the wording — that is what
 * survives translation. Prose still comes from the row so a translated world still
 * reads in its own language. Rows the catalog does not know (a house-ruled table)
 * fall back to the tolerant parser.
 */
export function pureCritOutcome({ variant, total, rowHtml, catalog, labels }) {
  const cells = pureCritRowFields(rowHtml);
  const spec = catalog && typeof catalog.get === "function" ? catalog.get(Number(total)) : null;

  if (spec) {
    return {
      source: "catalog", variant, total: Number(total),
      fatal: spec.fatal, fatalMod: spec.fatalMod, timeLimit: spec.timeLimit,
      healing: spec.healing, healingRaw: cells.healing,
      injury: cells.injury, effects: cells.effects,
    };
  }

  const fatal = pureFatalOf(cells.fatal, labels);
  const healing = pureHealingOf(cells.healing, labels);
  return {
    source: "parsed", variant, total: Number(total),
    fatal: fatal.fatal, fatalMod: fatal.fatalMod,
    timeLimit: pureTimeLimitOf(cells.timeLimit, labels),
    healing: { kind: healing.kind, formula: healing.formula },
    healingRaw: cells.healing,
    injury: cells.injury, effects: cells.effects,
  };
}
```

注意：`pureCritOutcome` 在没有 `labels` 时把 `undefined` 传下去，三个解析函数各自落回英文默认集 —— 这正是「翻译过的行 + 没有目录」那条用例期望复现的今日症状。

- [ ] **Step 9: 跑它，看它通过**

Run: `npx vitest run test/crit-result-parse-hardening.test.mjs`

Expected: PASS —— `Tests  31 passed (31)`。

- [ ] **Step 10: 提交**

```bash
git add scripts/features/crit-result-parse-hardening.pure.mjs test/crit-result-parse-hardening.test.mjs && git commit -m "feat(crit-result-parse-hardening): 机械数值改按 D66 行号查表，文案仍从行里读" -m "致命/致命修正/时限档/治疗骰式是行号的函数，不是文案的函数，翻译改不了它们。
两张出货表 PqqpXQ1aPnzmb6RP 与 XmFybNUYJ4C2OQyE 各 36 行逐行誊录，
再用脚本把每行的解析结果与誊录值逐格比对，72 行零分歧。
EV 的 14/15 两行治疗时间是文本 Shift 而非骰式（系统在 actor.mjs:1890 也特判了这个字面量），
单列一个 T 记号；1e 的 44 行是 One Day 而 EV 的 44 行是 Shift，两版不能互抄。
中文世界的用例直接把今天的症状钉死：只靠解析时 cFatal 恒 false、时限恒 0，
查表后同一行拿到 致命=true、时限=一轮班。家规表（行号对不上）退回解析器。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 11: 写会失败的 probe 判定 / 技能修正 / 治疗时间改写 / 标签集测试**

这四样都是副作用层马上要用的纯逻辑，先把它们的测试写出来。追加到测试文件末尾，import 列表再补 `pureCritParseIsBuggy, pureCritSkillMods, pureHealingFix, pureMarkerSet`：

```js
// The verbatim 4.1.13 body fragments the probe keys on:
// actor.mjs:1885-1889 (positional read) and :1923-1935 (trailing-space compare).
const SOURCE_4_1_13 = `
          testArray = factorFour.split(/[:] |<br>/gi);

          let speanex = testArray[7];
          if (testArray[9] !== game.i18n.localize("ALIENRPG.Permanent")) {
            if (testArray[9].length > 0) {
          switch (testArray[5]) {
            case game.i18n.localize("ALIENRPG.None") + " ":
              healTime = 0;
              break;
            case game.i18n.localize("ALIENRPG.OneShift") + " ":
              healTime = 3;
              break;
`;
// Both halves repaired upstream: cells read by label, units matched after trimming.
const SOURCE_FIXED = `
          const cells = readLabelledCells(messG);
          const healingCell = cells.healing.trim();
          healTime = TIME_LIMIT_BY_UNIT[cells.timeLimit.trim().toLowerCase()] ?? 0;
`;
// Only the positional read repaired; the trailing-space compare still eats "Shift".
const SOURCE_HALF_A = `
          const cells = readLabelledCells(messG);
          switch (testArray[5]) {
            case game.i18n.localize("ALIENRPG.OneShift") + " ":
              healTime = 3;
              break;
`;
// Only the compare repaired; testArray[9] still explodes on the 1e rows 11 and 12.
const SOURCE_HALF_B = `
          testArray = factorFour.split(/[:] |<br>/gi);
          if (testArray[9].length > 0) {
          healTime = TIME_LIMIT_BY_UNIT[cells.timeLimit.trim().toLowerCase()] ?? 0;
`;

describe("pureCritParseIsBuggy", () => {
  it("says the defect is present in the shipped 4.1.13 body", () => {
    expect(pureCritParseIsBuggy(SOURCE_4_1_13)).toBe(true);
  });
  it("says the defect is gone once both halves are repaired", () => {
    expect(pureCritParseIsBuggy(SOURCE_FIXED)).toBe(false);
  });
  it("still says buggy when only the positional read was repaired", () => {
    expect(pureCritParseIsBuggy(SOURCE_HALF_A)).toBe(true);
  });
  it("still says buggy when only the label compare was repaired", () => {
    expect(pureCritParseIsBuggy(SOURCE_HALF_B)).toBe(true);
  });
  it("says not-buggy when there is no source to judge, so we never patch blind", () => {
    expect(pureCritParseIsBuggy("")).toBe(false);
    expect(pureCritParseIsBuggy(null)).toBe(false);
  });
});

const ZERO_MODS = {
  mobility: 0, rangedCbt: 0, observation: 0, manipulation: 0,
  closeCbt: 0, stamina: 0, comtech: 0, command: 0,
};

describe("pureCritSkillMods", () => {
  it("carries the Evolved modifiers for row 15", () => {
    expect(pureCritSkillMods("evolved", 15)).toEqual({ ...ZERO_MODS, rangedCbt: -1, observation: -1 });
  });
  it("uses the reachable value for the row the system lists twice", () => {
    // actor.mjs:1989 sets stamina = -2; the duplicate case at :1992 (stamina = -3) is dead code.
    expect(pureCritSkillMods("evolved", 62)).toEqual({ ...ZERO_MODS, stamina: -2 });
  });
  it("carries the first-edition modifiers for row 44", () => {
    expect(pureCritSkillMods("first", 44)).toEqual({ ...ZERO_MODS, mobility: -2, stamina: -2 });
  });
  it("is all zeroes for a row or a variant with no modifiers", () => {
    expect(pureCritSkillMods("evolved", 11)).toEqual(ZERO_MODS);
    expect(pureCritSkillMods("nosuchvariant", 44)).toEqual(ZERO_MODS);
  });
});

describe("pureHealingFix", () => {
  it("keeps a stored Permanent", () => {
    expect(pureHealingFix({ kind: "permanent", formula: null }, "Permanent")).toEqual({ action: "keep" });
  });
  it("keeps a translated Permanent when the caller injects the labels", () => {
    expect(pureHealingFix({ kind: "permanent", formula: null }, "永久", CN_LABELS)).toEqual({ action: "keep" });
  });
  it("replaces a wrong stored value with Permanent", () => {
    expect(pureHealingFix({ kind: "permanent", formula: null }, "5 days")).toEqual({ action: "permanent" });
  });
  it("keeps a number the system already rolled, so the item and the card agree", () => {
    expect(pureHealingFix({ kind: "formula", formula: "1d6" }, "3 days")).toEqual({ action: "keep" });
  });
  it("rolls when the stored value is still the unrolled inline formula", () => {
    expect(pureHealingFix({ kind: "formula", formula: "1d6" }, "[[1d6]] days")).toEqual({ action: "roll", formula: "1d6" });
  });
  it("rolls when nothing was stored at all", () => {
    expect(pureHealingFix({ kind: "formula", formula: "2d6" }, "")).toEqual({ action: "roll", formula: "2d6" });
  });
  it("keeps an empty cell that is supposed to be empty", () => {
    expect(pureHealingFix({ kind: "none", formula: null }, "")).toEqual({ action: "keep" });
    expect(pureHealingFix({ kind: "none", formula: null }, "None")).toEqual({ action: "keep" });
  });
  it("clears a stored value that should have been empty", () => {
    expect(pureHealingFix({ kind: "none", formula: null }, "17 days")).toEqual({ action: "none" });
  });
  it("never touches a free-text healing cell such as the Evolved rows 14 and 15", () => {
    expect(pureHealingFix({ kind: "text", formula: null }, "Shift")).toEqual({ action: "keep" });
  });
});

describe("pureMarkerSet", () => {
  it("strips the One prefix so a localized 'One Shift' collapses onto 'shift'", () => {
    expect(pureMarkerSet(["shift"], ["One Shift"])).toEqual(["shift"]);
  });
  it("keeps a genuinely different translated marker and drops the duplicate", () => {
    expect(pureMarkerSet(["yes"], ["是", "Yes"])).toEqual(["yes", "是"]);
  });
  it("drops empty, blank and nullish entries", () => {
    expect(pureMarkerSet(["none"], [null, "", "   ", undefined])).toEqual(["none"]);
  });
  it("returns an empty array when given nothing", () => {
    expect(pureMarkerSet(undefined, undefined)).toEqual([]);
  });
});
```

- [ ] **Step 12: 跑它，看它失败**

Run: `npx vitest run test/crit-result-parse-hardening.test.mjs -t "pureCritParseIsBuggy"`

Expected: FAIL —— 5 条全红，报 `TypeError: pureCritParseIsBuggy is not a function`。

- [ ] **Step 13: 写这四个纯函数**

追加到 `scripts/features/crit-result-parse-hardening.pure.mjs`：

```js
/**
 * Injectable probe predicate: given the source text of the system's rollCrit,
 * decide whether the defect is STILL present. Two independent halves, either of
 * which alone is enough to keep the patch:
 *   - "testArray[9]": the positional read that is undefined on 1e rows 11 and 12
 *     and throws at actor.mjs:1889;
 *   - the OneShift compare with an appended trailing space at actor.mjs:1933,
 *     which the Evolved table's bare "Shift" can never match.
 * Quotes are normalized first because the shipped body may use either quote style.
 * An empty source means the method is not there to judge, so we report not-buggy
 * rather than patch blind.
 */
export function pureCritParseIsBuggy(source) {
  const src = String(source ?? "").replace(/['"`]/g, '"').replace(/\s+/g, " ").trim();
  if (!src) return false;
  const positionalHealingRead = src.includes("testArray[9]");
  const trailingSpaceCompare = src.includes('localize("ALIENRPG.OneShift") + " "');
  return positionalHealingRead || trailingSpaceCompare;
}

/**
 * The per-row skill modifiers the system writes onto the injury item, transcribed
 * from actor.mjs:1944-1998 (evolved) and :2000-2045 (first edition). These eight keys
 * are the ones the system actually writes at :2058-2066 — a subset of the twelve in
 * CONFIG.ALIENRPG.skills (module/helpers/config.mjs:52-65), which is what the item's
 * skill-modifier schema is generated from (module/data/base-item.mjs:23-31).
 */
const SKILL_KEYS = Object.freeze([
  "mobility", "rangedCbt", "observation", "manipulation", "closeCbt", "stamina", "comtech", "command",
]);

const SKILL_MOD_SPEC = Object.freeze({
  evolved: {
    15: { rangedCbt: -1, observation: -1 },
    16: { observation: -1, comtech: -1 },
    21: { observation: -2 },
    24: { manipulation: -2 },
    31: { observation: -1, manipulation: -1 },
    32: { mobility: -2, closeCbt: -2 },
    33: { rangedCbt: -2, observation: -2 },
    34: { manipulation: -2, command: -2 },
    42: { manipulation: -2 },
    44: { mobility: -2, closeCbt: -2 },
    51: { observation: -2, comtech: -2 },
    61: { stamina: -1 },
    // actor.mjs lists case 62 twice (:1989 stamina = -2, :1992 stamina = -3);
    // the second is unreachable, so -2 is what the system actually produces.
    62: { stamina: -2 },
  },
  first: {
    14: { mobility: -2 },
    15: { rangedCbt: -2, observation: -2 },
    16: { mobility: -2 },
    21: { observation: -2 },
    24: { manipulation: -2 },
    31: { observation: -1, manipulation: -1 },
    33: { mobility: -2, closeCbt: -2 },
    34: { rangedCbt: -2, observation: -2 },
    44: { mobility: -2, stamina: -2 },
    51: { mobility: -2 },
    61: { stamina: -1 },
    62: { stamina: -2 },
  },
});

export function pureCritSkillMods(variant, total) {
  const row = SKILL_MOD_SPEC[variant]?.[Number(total)] ?? {};
  const out = {};
  for (const key of SKILL_KEYS) out[key] = row[key] ?? 0;
  return out;
}

/**
 * Decide what to do with the healing-time string the system already stored.
 * "keep" matters: when the system rolled [[1d6]] correctly it printed that same
 * number on the chat card, so re-rolling would make the item and the card disagree.
 */
export function pureHealingFix(planned, storedText, labels = DEFAULT_LABELS) {
  const kind = planned?.kind ?? "text";
  const stored = normalize(storedText).toLowerCase();
  if (kind === "permanent") {
    return startsWithAny(stored, labels.permanent) ? { action: "keep" } : { action: "permanent" };
  }
  if (kind === "none") {
    return !stored || stored === "-" || startsWithAny(stored, labels.none) ? { action: "keep" } : { action: "none" };
  }
  if (kind === "formula") {
    return !stored || stored.includes("[[") ? { action: "roll", formula: planned.formula } : { action: "keep" };
  }
  return { action: "keep" };
}

/**
 * Build one marker list for the tolerant parser: lowercase everything, drop the
 * leading "One " the 1e table uses, drop blanks, de-duplicate keeping first order.
 * The effect layer feeds this the English default plus whatever the world's language
 * pack says, which is why it has to be a pure, testable function rather than inline
 * string juggling next to game.i18n.
 */
export function pureMarkerSet(english, localized) {
  const out = [];
  for (const raw of [...(english ?? []), ...(localized ?? [])]) {
    const value = String(raw ?? "").toLowerCase().replace(/^one[ ]+/, "").trim();
    if (value && !out.includes(value)) out.push(value);
  }
  return out;
}
```

- [ ] **Step 14: 跑它，看它通过**

Run: `npx vitest run test/crit-result-parse-hardening.test.mjs`

Expected: PASS —— `Tests  53 passed (53)`。

- [ ] **Step 15: 提交**

```bash
git add scripts/features/crit-result-parse-hardening.pure.mjs test/crit-result-parse-hardening.test.mjs && git commit -m "feat(crit-result-parse-hardening): probe 判定抽成可注入纯函数并双向单测" -m "终检要求每个 probe 是一层薄壳，判定逻辑必须是可注入、能双向测的纯谓词，
否则一条恒真的子串匹配会在上游修好后继续重复打补丁。
pureCritParseIsBuggy 按两个独立半边判定：positional testArray[9] 与 OneShift 的尾随空格比较，
任一半边还在就该装；四段夹具覆盖 4.1.13 原文、两半各修一半、以及全修好，
外加取不到源码时返回 false（不盲打补丁）。
另补三个纯函数：
- pureCritSkillMods 誊录 actor.mjs:1944-2045 的两张修正表，含 :1992 那条不可达的重复 case 62；
- pureHealingFix 让系统已正确掷出的天数一律保留，避免物品与聊天卡数字对不上；
- pureMarkerSet 把「英文默认 + 世界语言」拼成标签集，从副作用层挪进纯层才测得动。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 16: 写副作用层**

新建 `scripts/features/crit-result-parse-hardening.mjs`：

```js
import { MID, SYSTEM_ID } from "../const.mjs";
import { features } from "../kernel/features.mjs";
import { patches } from "../kernel/patches.mjs";
import { registry } from "../kernel/registry.mjs";
import { selftest } from "../kernel/selftest.mjs";
import {
  pureCritCatalog,
  pureCritOutcome,
  pureCritParseIsBuggy,
  pureCritRowFields,
  pureCritSkillMods,
  pureFatalOf,
  pureHealingFix,
  pureHealingOf,
  pureMarkerSet,
  pureTimeLimitCode,
  pureTimeLimitOf,
} from "./crit-result-parse-hardening.pure.mjs";

const FEATURE_ID = "crit-result-parse-hardening";
const TARGET = "CONFIG.Actor.documentClass.prototype.rollCrit";
const VARIANT_KEY = { evolved: "critInjuryEvolved", first: "critInjury1e" };
const CHAT_TEMPLATE = "systems/alienrpg/templates/chat/crit-roll-character.hbs";

/**
 * Source text of the untouched system method, snapshotted at init BEFORE anything
 * wraps it. It must be taken this early: after patches.applyAll() runs,
 * CONFIG.Actor.documentClass.prototype.rollCrit is libWrapper's dispatcher and its
 * toString() is libWrapper's own source, not the system's. main.mjs runs
 * `for (const f of FEATURES) f.register()` as the very first thing in the init hook,
 * which is why this lands before any patch.
 */
let sourceSnapshot = "";

/** One line over the injectable predicate. true means the defect is STILL present. */
export function probeCritResultParse() {
  return pureCritParseIsBuggy(sourceSnapshot);
}

/** The system keeps two rule variants behind one world setting (helpers/settings.mjs:14). */
function activeVariant() {
  return game.settings.get(SYSTEM_ID, "evolved") ? "evolved" : "first";
}

/**
 * Localize only the keys that actually exist. game.i18n.localize echoes the key back
 * when it is missing, and an echoed "ALIENRPG.OneRound" would then be registered as a
 * marker — harmless noise at best, a false match at worst. game.i18n.has() screens it.
 */
function localizedMarkers(...keys) {
  const found = [];
  for (const key of keys) {
    if (key && game.i18n.has(key)) found.push(game.i18n.localize(key));
  }
  return found;
}

/**
 * Marker set for the fallback parser, built from the world's current language.
 * The module supplies its own AEA.crit.* markers alongside the system's keys because
 * systems/alienrpg/lang/cn.json leaves ALIENRPG.Yes as "Yes" and ALIENRPG.Permanent as
 * "Permanent" — without our own "是" / "永久" the Chinese fallback parser reads nothing.
 */
function worldLabels() {
  return {
    yes: pureMarkerSet(["yes"], localizedMarkers("ALIENRPG.Yes", "AEA.crit.yes")),
    permanent: pureMarkerSet(["permanent"], localizedMarkers("ALIENRPG.Permanent", "AEA.crit.permanent")),
    none: pureMarkerSet(["none"], localizedMarkers("ALIENRPG.None", "AEA.crit.none")),
    units: {
      round: pureMarkerSet(["round"], localizedMarkers("ALIENRPG.OneRound")),
      turn: pureMarkerSet(["turn"], localizedMarkers("ALIENRPG.OneTurn")),
      shift: pureMarkerSet(["shift"], localizedMarkers("ALIENRPG.OneShift", "ALIENRPG.Shift")),
      day: pureMarkerSet(["day"], localizedMarkers("ALIENRPG.OneDay")),
      stretch: pureMarkerSet(["stretch"], localizedMarkers("AEA.crit.timeLimitStretch")),
    },
  };
}

function timeLimitLabel(code) {
  const entry = CONFIG.ALIENRPG?.crit_timelimit_list?.[code];
  return game.i18n.localize(entry?.label ?? "ALIENRPG.None");
}

/**
 * The healing-time string to store. `stored` is what the system already wrote (or ""
 * when we are rebuilding from scratch); we only overwrite it when it is demonstrably
 * wrong, so a number the system rolled stays equal to the number on its chat card.
 * The system itself uses Roll#result at actor.mjs:1895; Roll#total is the same value
 * for a plain NdM and is a number rather than a string.
 */
async function healingText(outcome, stored) {
  const fix = pureHealingFix(outcome.healing, stored, worldLabels());
  if (fix.action === "keep") return String(stored ?? "");
  if (fix.action === "permanent") return game.i18n.localize("ALIENRPG.Permanent");
  if (fix.action === "none") return game.i18n.localize("ALIENRPG.None");

  const rolled = await new Roll(fix.formula).evaluate();
  const raw = String(outcome.healingRaw ?? "");
  const open = raw.indexOf("[[");
  const close = raw.indexOf("]]");
  const tail = open >= 0 && close > open ? `${raw.slice(0, open)}${raw.slice(close + 2)}`.trim() : "";
  return tail ? `${rolled.total} ${tail}` : String(rolled.total);
}

/**
 * Recreate what the system failed to create when its own parser threw, mirroring
 * actor.mjs:2050-2069 (item), :2199 (status effect) and :2173-2211 (chat card).
 */
async function rebuildInjury(doc, actor, outcome, row, manCrit) {
  const cells = pureCritRowFields(row?.description ?? "");
  const healingtime = await healingText(outcome, "");
  const mods = pureCritSkillMods(outcome.variant, outcome.total);
  const name = `#${outcome.total} ${outcome.injury}`.trim();
  const img = row?.img || "icons/svg/blood.svg";

  // The system appends this same note at actor.mjs:1909 / :1915 whenever the FATAL cell
  // carried a medical-aid penalty. We rebuild it from the catalog's fatalMod, so a
  // translated table gets the note that the system's literal string compare would miss.
  let effects = outcome.effects;
  if (outcome.fatal && outcome.fatalMod !== 0) {
    effects += `<br> ${outcome.fatalMod} to <strong>${game.i18n.localize("ALIENRPG.SkillmedicalAid")}</strong> roll`;
  }

  await doc.createEmbeddedDocuments("Item", [{
    type: "critical-injury",
    img,
    name,
    "system.header.active": true,
    "system.attributes.fatal": outcome.fatal,
    "system.attributes.timelimit.value": outcome.timeLimit,
    "system.attributes.healingtime.value": healingtime,
    "system.attributes.effects": effects,
    "system.modifiers.skills.mobility.value": mods.mobility,
    "system.modifiers.skills.rangedCbt.value": mods.rangedCbt,
    "system.modifiers.skills.observation.value": mods.observation,
    "system.modifiers.skills.manipulation.value": mods.manipulation,
    "system.modifiers.skills.closeCbt.value": mods.closeCbt,
    "system.modifiers.skills.stamina.value": mods.stamina,
    "system.modifiers.skills.comtech.value": mods.comtech,
    "system.modifiers.skills.command.value": mods.command,
  }]);

  if (actor.system?.general?.critInj?.value === 1) await doc.toggleStatusEffect("criticalinj");

  // The card prints the row's own cells, exactly as the system does at :2074-2076,
  // so a translated world keeps reading in its own language.
  const html = await foundry.applications.handlebars.renderTemplate(CHAT_TEMPLATE, {
    actorname: actor.name,
    img,
    name,
    fatal: cells.fatal || game.i18n.localize("ALIENRPG.None"),
    timelimit: cells.timeLimit || timeLimitLabel(outcome.timeLimit),
    healingtime: healingtime || game.i18n.localize("ALIENRPG.None"),
    effects,
    manCrit,
  });
  const chatData = {
    user: game.user.id,
    speaker: { actor: actor.id },
    content: html,
    other: game.users.contents.filter((u) => u.isGM).map((u) => u.id),
    sound: CONFIG.sounds.dice,
  };
  ChatMessage.applyRollMode(chatData, game.settings.get("core", "rollMode"));
  return ChatMessage.create(chatData);
}

/**
 * libWrapper MIXED wrapper for rollCrit(actor, type, dataset, manCrit).
 * Declared as a function (not an arrow) so `this` stays the Actor document, which is
 * what the system's own character branch uses at actor.mjs:2069 and :2199.
 * NOTE ON CONTRACT: rollCrit is NOT one of the four targets the roll bus owns
 * (yzeRoll / abilityRoll / itemRoll / pushRoll), so registering it here is legal and
 * must not be converted into rollBus.addStage() — there is no such stage target.
 */
export async function critParseWrapper(wrapped, actor, type, dataset, manCrit) {
  const doc = this ?? actor;

  // The synthetic / creature branches (actor.mjs:2092-2124) and the spacecraft branch
  // (:2126-2160) store only a name and an effects string — they have no fatal /
  // timelimit / healingtime fields to harden, so we never touch them.
  if (!features.enabled(FEATURE_ID) || type !== "character") {
    return wrapped(actor, type, dataset, manCrit);
  }

  const variant = activeVariant();
  const table = registry.table(VARIANT_KEY[variant]);
  if (!table) {
    console.warn(`${MID} | registry key "${VARIANT_KEY[variant]}" is unbound; leaving rollCrit alone`);
    ui.notifications.warn(game.i18n.localize("AEA.crit.unboundTable"));
    return wrapped(actor, type, dataset, manCrit);
  }

  // Observe — never replace — the draw, so we learn which D66 row came up even when
  // the system's own parser throws at :1889 before it creates anything. The dice, the
  // Dice So Nice animation and the chat card wording are all unchanged. The system
  // reaches the same document object via game.tables.getName (:1811-1817), so patching
  // our own reference is enough as long as both resolve to the same table — which is
  // exactly what the .legacyLookup self-test measures.
  // Scope: one awaited call. rollCrit is a single user action, so nested or concurrent
  // draws on the same table are not a case we serialize; the restore is still correct
  // if it happens (each frame restores what it found).
  let observed = null;
  const hadOwnDraw = Object.hasOwn(table, "draw");
  const ownDraw = hadOwnDraw ? table.draw : undefined;
  const innerDraw = table.draw;
  table.draw = async function aeaObservingDraw(...drawArgs) {
    const result = await innerDraw.apply(this, drawArgs);
    observed = result;
    return result;
  };

  const before = new Set(doc.items.filter((i) => i.type === "critical-injury").map((i) => i.id));
  let systemResult;
  let systemThrew = null;
  try {
    systemResult = await wrapped(actor, type, dataset, manCrit);
  } catch (error) {
    systemThrew = error;
  } finally {
    if (hadOwnDraw) table.draw = ownDraw;
    else delete table.draw;
  }

  if (!observed) {
    // Either rollCrit bailed out early (its own name lookup missed and it returned at
    // :1821-1822) or it drew from a different table than the one we are bound to.
    // Nothing to correct.
    if (systemThrew) throw systemThrew;
    console.warn(`${MID} | rollCrit drew nothing from the bound ${variant} table; run the ${FEATURE_ID}.legacyLookup self-test`);
    return systemResult;
  }

  const row = observed.results?.[0] ?? null;
  const outcome = pureCritOutcome({
    variant,
    total: observed.roll?.total ?? 0,
    rowHtml: row?.description ?? "",
    catalog: pureCritCatalog(variant),
    labels: worldLabels(),
  });

  if (systemThrew) {
    console.warn(`${MID} | system rollCrit threw on D66 ${outcome.total}; rebuilding the injury and the card`, systemThrew);
    return rebuildInjury(doc, actor, outcome, row, manCrit);
  }

  const created = doc.items.find((i) => i.type === "critical-injury" && !before.has(i.id));
  if (!created) return systemResult;
  // Three fields only. `effects` is left alone on purpose: it is the row's own prose as
  // the system sliced it, and a GM may have edited it; rewriting it would risk clobbering.
  await created.update({
    "system.attributes.fatal": outcome.fatal,
    "system.attributes.timelimit.value": outcome.timeLimit,
    "system.attributes.healingtime.value": await healingText(outcome, created.system?.attributes?.healingtime?.value),
  });
  return systemResult;
}

export const critResultParseHardeningFeature = {
  id: FEATURE_ID,

  register() {
    sourceSnapshot = String(CONFIG.Actor?.documentClass?.prototype?.rollCrit ?? "");

    // gmOnly stays false on purpose: a player clicking Roll Crit on their own character
    // must get the hardened path too, and this switch is what gates execution.
    features.register({ id: FEATURE_ID, default: "full", gmOnly: false, requires: [], hint: "" });

    patches.register({
      id: FEATURE_ID,
      type: "MIXED",
      target: TARGET, // metadata for patches.status(); apply() installs the wrapper itself
      minSystem: "4.1.13",
      fixedIn: null,
      probe: probeCritResultParse,
      apply: () => {
        // The system's own dropdown only offers codes 0-4 (helpers/config.mjs:526-532)
        // and the item sheet renders it with selectOptions ... localize=true
        // (templates/item/item-header.hbs:74), so adding a fifth entry is enough for
        // the Evolved "Stretch" unit to show up. timelimit.value is a NumberField with
        // min 0 (data/item-crit-inj.mjs:22-28), so 5 validates.
        // Added unconditionally, even when the feature switch is off, so that an item
        // written earlier with timelimit 5 still renders a label instead of a blank.
        CONFIG.ALIENRPG.crit_timelimit_list[pureTimeLimitCode("stretch")] = {
          id: pureTimeLimitCode("stretch"),
          label: "AEA.crit.timeLimitStretch",
        };
        libWrapper.register(MID, TARGET, critParseWrapper, "MIXED");
      },
    });

    selftest.register({
      id: `${FEATURE_ID}.probe`,
      label: `AEA.selftest.${FEATURE_ID}.probe`,
      run() {
        const defect = probeCritResultParse();
        const entry = patches.status().find((p) => p.id === FEATURE_ID) ?? null;
        const applied = Boolean(entry?.applied);
        return {
          ok: defect === applied,
          detail:
            `snapshot=${sourceSnapshot.length} chars, defect=${defect}, applied=${applied}, ` +
            `reason=${entry?.reason ?? "not-registered"}, type=${entry?.type ?? "-"}, ` +
            `target=${entry?.target ?? "-"}, fixedIn=${entry?.fixedIn ?? "none"}`,
        };
      },
    });

    selftest.register({
      id: `${FEATURE_ID}.tableRows`,
      label: `AEA.selftest.${FEATURE_ID}.tableRows`,
      run() {
        // Two checks in one: every row of the live table is in the catalog, and where
        // the row's own cells are legible they agree with the catalog. A hand
        // transcription error therefore shows up here instead of silently writing
        // wrong data onto a character.
        // Evidence rule: a parser result equal to its DEFAULT is not evidence of
        // anything (not-fatal, time limit none, free text are all what an unreadable
        // cell yields), so those rows are counted as "unreadable" and reported without
        // failing. That keeps a translated world from going red over wording drift
        // while still catching a real transcription error.
        const variant = activeVariant();
        const table = registry.table(VARIANT_KEY[variant]);
        if (!table) return { ok: false, detail: `registry key "${VARIANT_KEY[variant]}" unbound` };

        const catalog = pureCritCatalog(variant);
        const labels = worldLabels();
        const missing = [];
        const disagree = [];
        let unreadable = 0;

        for (const result of table.results.contents) {
          const total = Number(result.range?.[0] ?? 0);
          const spec = catalog.get(total);
          if (!spec) { missing.push(total); continue; }
          const cells = pureCritRowFields(result.description ?? "");

          const fatal = pureFatalOf(cells.fatal, labels);
          if (cells.fatal && fatal.fatal) {
            if (fatal.fatal !== spec.fatal || fatal.fatalMod !== spec.fatalMod) {
              disagree.push(`${total} FATAL row=${fatal.fatal}/${fatal.fatalMod} catalog=${spec.fatal}/${spec.fatalMod}`);
            }
          } else if (cells.fatal) unreadable += 1;

          const time = pureTimeLimitOf(cells.timeLimit, labels);
          if (cells.timeLimit && time !== pureTimeLimitCode("none")) {
            if (time !== spec.timeLimit) disagree.push(`${total} TIME LIMIT row=${time} catalog=${spec.timeLimit}`);
          } else if (cells.timeLimit) unreadable += 1;

          const heal = pureHealingOf(cells.healing, labels);
          if (cells.healing && (heal.kind === "formula" || heal.kind === "permanent")) {
            if (heal.kind !== spec.healing.kind || (heal.formula ?? null) !== (spec.healing.formula ?? null)) {
              disagree.push(`${total} HEALING row=${heal.kind}/${heal.formula} catalog=${spec.healing.kind}/${spec.healing.formula}`);
            }
          } else if (cells.healing) unreadable += 1;
        }

        const parts = [];
        if (missing.length) parts.push(`rows the catalog does not know, parser fallback will be used: ${missing.join(", ")}`);
        if (disagree.length) parts.push(`cells that contradict the catalog: ${disagree.join("; ")}`);
        return {
          ok: missing.length === 0 && disagree.length === 0,
          detail: parts.length
            ? `${parts.join(" | ")} (${unreadable} further cells were not legible in this world's language and were not compared)`
            : `all ${table.results.contents.length} rows of the ${variant} table are in the catalog and agree with it (${unreadable} cells not legible in this world's language, not compared)`,
        };
      },
    });

    selftest.register({
      id: `${FEATURE_ID}.legacyLookup`,
      label: `AEA.selftest.${FEATURE_ID}.legacyLookup`,
      run() {
        // Diagnostic only. The system finds its own table by display name
        // (actor.mjs:1811-1817); if that has drifted it bails out at :1821-1822 and our
        // draw observer never fires. getName appears here as a MEASUREMENT of the
        // system's own path, never to resolve a document for module logic — module
        // logic always goes through registry.table().
        const variant = activeVariant();
        const bound = registry.table(VARIANT_KEY[variant]);
        const names = variant === "evolved"
          ? [game.i18n.localize("ALIENRPG.EVCriticalInjuries"), "EV - Critical Injuries"]
          : [game.i18n.localize("ALIENRPG.CriticalInjuries"), "Critical Injuries", "Critical injuries"];
        const found = names.map((n) => game.tables.getName(n)).find(Boolean);
        const ok = Boolean(bound) && found?.id === bound.id;
        return {
          ok,
          detail: ok
            ? `the system's own name lookup still reaches the bound ${variant} table`
            : `bound=${bound?.id ?? "none"}, system lookup=${found?.id ?? "none"}; rollCrit will warn NoCharCrit at actor.mjs:1821 and return without drawing`,
        };
      },
    });
  },

  // ready stage: nothing to install. The wrapper goes on in the patch's apply(),
  // which patches.applyAll() runs earlier in the same ready sequence (the ready.patches
  // anchor is above the FEATURES install loop in main.mjs).
  install() {},
};
```

- [ ] **Step 17: 跑全量测试**

Run: `npm test`

Expected: PASS —— `test/crit-result-parse-hardening.test.mjs` 53 passed，其余测试文件计数与本步之前完全一致（本步没动任何别的文件）。

- [ ] **Step 18: 加 i18n 键**

合并进 `lang/en.json` 已有的顶层 `AEA` 对象（语言包是嵌套结构，顶层只允许一个 `AEA` 键）：

```json
{
  "AEA": {
    "feature": {
      "crit-result-parse-hardening": {
        "name": "Critical injury parsing",
        "hint": "Read a drawn critical injury's fatal flag, time limit and healing time from the row number instead of scraping the row's rich text, so a translated or house-edited table no longer silently reports 'not fatal, heals in 0'."
      }
    },
    "selftest": {
      "crit-result-parse-hardening": {
        "probe": "Critical injury parsing: wrapper installed and defect still present",
        "tableRows": "Critical injury parsing: row catalog matches the bound table",
        "legacyLookup": "Critical injury parsing: the system's own table name lookup still resolves"
      }
    },
    "crit": {
      "yes": "Yes",
      "permanent": "Permanent",
      "none": "None",
      "timeLimitStretch": "Stretch",
      "unboundTable": "No critical-injury table is bound, so critical injuries are still read the system's way."
    }
  }
}
```

合并进 `lang/cn.json`（`AEA.crit.yes` / `.permanent` / `.none` 三个键存在的唯一理由是：系统自己的 `cn.json` 把 `ALIENRPG.Yes` 留成 `"Yes"`、`ALIENRPG.Permanent` 留成 `"Permanent"`，没有这三个键，中文世界的兜底解析器一个字也读不出来）：

```json
{
  "AEA": {
    "feature": {
      "crit-result-parse-hardening": {
        "name": "重伤结果解析",
        "hint": "重伤的致命标记、时限与治疗时间改按表格行号读取，不再切碎富文本按下标取值；表格被翻译或被改过之后，不会再静默变成「不致命、恢复 0」。"
      }
    },
    "selftest": {
      "crit-result-parse-hardening": {
        "probe": "重伤结果解析：包裹已装上，且缺陷确实还在",
        "tableRows": "重伤结果解析：行号目录与已绑定的表逐行一致",
        "legacyLookup": "重伤结果解析：系统自己的按名查表仍能找到同一张表"
      }
    },
    "crit": {
      "yes": "是",
      "permanent": "永久",
      "none": "无",
      "timeLimitStretch": "一段时间",
      "unboundTable": "尚未绑定重伤表，重伤结果仍按系统原有方式读取。"
    }
  }
}
```

- [ ] **Step 19: 在 main.mjs 登记本特性（只碰两个锚点，不碰生命周期）**

`scripts/main.mjs` 已经由建立它的那个任务逐字写出了十个锚点注释、`api` 的八个 `null` 槽、以及两个显式数组。
`init` 钩子里有 `for (const f of FEATURES) safely(..., () => f.register());`，
`ready` 钩子里有 `for (const f of FEATURES) await safely(..., () => f.install());`。
**本任务一行生命周期代码都不插、`api` 一个键都不碰**，只做两处按锚点定位的声明式改动：

1. 搜到这一行（逐字）：`/* AEA-ANCHOR: imports */`，在**它的下一行**插入：

   ```js
   import { critResultParseHardeningFeature } from "./features/crit-result-parse-hardening.mjs";
   ```

2. 搜到这一行（逐字）：`/* AEA-ANCHOR: features */`，在**它的下一行**插入（数组成员，注意保留结尾逗号）：

   ```js
     critResultParseHardeningFeature,
   ```

插完之后那两处应当长这样：

```js
import { MID } from "./const.mjs";
/* AEA-ANCHOR: imports */
import { critResultParseHardeningFeature } from "./features/crit-result-parse-hardening.mjs";
...
const FEATURES = [
  /* AEA-ANCHOR: features */
  critResultParseHardeningFeature,
];
```

不要新增 `critResultParseHardeningFeature.register()` / `.install()` 之类的直接调用 —— 那两个 `for..of` 循环已经是唯一调用点，重复调会让 `features.register` 收到同一个 id 两次，`patches.register` 也会重复登记，`apply()` 里的 `libWrapper.register` 第二次注册会直接抛错。

- [ ] **Step 20: 提交**

```bash
git add scripts/features/crit-result-parse-hardening.mjs scripts/main.mjs lang/en.json lang/cn.json && git commit -m "feat(crit-result-parse-hardening): MIXED 包裹 rollCrit，落地后按行号校正三项数值" -m "包装器在调用原函数期间给绑定的重伤表挂一次性 draw 观察器（finally 必还原），
只观察不改动，掷骰分布、DsN 动画与聊天卡措辞都不变。
拿到行号后用行号目录改写 system.attributes.fatal / timelimit.value / healingtime.value；
effects 一格刻意不动（那是行里的原文，GM 可能改过）。
治疗时间只在系统那一格确实写错时才改写，系统已正确掷出的天数一律保留，
免得物品上的数字和它自己那张聊天卡对不上。
系统自己抛异常时（1e 表 11、12 行 testArray[9] 为 undefined；中文全角冒号世界同理），
接住并按 actor.mjs:2050-2069 / :2199 / :2173-2211 的形状自行补出物品、状态与聊天卡，
含 :1944-2045 那两张技能修正表，以及 :1909/:1915 那句医疗急救罚值备注。
另给 CONFIG.ALIENRPG.crit_timelimit_list 补第 5 档 Stretch —— 系统只到第 4 档，
Evolved 表 53-56 行用的就是这个单位；timelimit.value 是 min 0 的 NumberField，5 能过校验。
worldLabels 用 game.i18n.has 挡住缺键回声，并补 AEA.crit.yes/permanent/none 三个模组自有标记：
系统 cn.json 把 ALIENRPG.Yes 与 ALIENRPG.Permanent 留成英文，没有它们中文兜底解析读不出东西。
表按 evolved 世界设置从 registry 取 critInjuryEvolved 或 critInjury1e，模组自己永不 getName。
只处理 character 分支：synthetic/creature/spacecraft 分支压根没有这三个字段。
rollCrit 不属于 rollBus 独占的四个目标，故此处直接 libWrapper.register 合规。
main.mjs 只在 imports 与 features 两个锚点后各加一行，不碰任何生命周期锚点、不碰 api。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 21: MANUAL VERIFICATION —— Evolved 表时限不再恒为 None**

1. 启动 Foundry，进入装了 `alienrpg` 系统与 `alien-evolved-corerules` 合集的世界，确认本模组已启用。
2. 世界设置 → Alien RPG，确认 **Evolved** 开着（控制台 `game.settings.get("alienrpg","evolved")` 为 `true`）。
3. 打开本模组的 GM 重绑面板，把 `critInjuryEvolved` 绑到 **EV - Critical Injuries**（`_id` 为 `PqqpXQ1aPnzmb6RP`）。
4. F12 控制台先确认补丁真的装上了：

   ```js
   game.modules.get("alien-evolved-automation").api.patches.status()
     .find(p => p.id === "crit-result-parse-hardening")
   ```
   **期望**：`applied: true`、`reason: "ok"`、`type: "MIXED"`、`target: "CONFIG.Actor.documentClass.prototype.rollCrit"`。
5. 随便挑一个 `character` 类型的角色，控制台执行（第四个参数 `"44"` 是手动指定 D66 结果 = Punctured Lung；系统在 `actor.mjs:1871-1873` 会把它 `new Roll("44").evaluate()` 后交给 `draw({roll})`，所以观察器照常触发）：

   ```js
   const a = game.actors.getName("<角色名>");
   await a.rollCrit(a, "character", {}, "44");
   ```
6. **期望**：角色卡的物品列表里多出 `#44 Punctured Lung`；打开它 —— **FATAL 打勾**、**TIME LIMIT 下拉选中 One Shift**、HEALING TIME 是「N days」，且这个 N 与刚发出的那张重伤聊天卡上的数字**一致**。
7. **今天的行为（把本特性在模组设置里设为 off、刷新后可复现）**：同一物品的 TIME LIMIT 显示 **None**。对 `"45"`、`"46"`、`"52"`、`"61"` 重复，症状相同。

- [ ] **Step 22: MANUAL VERIFICATION —— 1e 表 11/12 行不再吞掉整次重伤**

1. 世界设置里关掉 **Evolved**（系统会自动重载世界）。
2. 在重绑面板把 `critInjury1e` 绑到 **Critical injuries**（`_id` 为 `XmFybNUYJ4C2OQyE`）。
3. 控制台执行两次：

   ```js
   const a = game.actors.getName("<角色名>");
   await a.rollCrit(a, "character", {}, "11");
   await a.rollCrit(a, "character", {}, "12");
   ```
4. **期望**：角色卡上加出两个物品（`#11 Winded`、`#12 Stunned`），各发一张重伤聊天卡；控制台各有一条
   `alien-evolved-automation | system rollCrit threw on D66 11; rebuilding the injury and the card`。
   两个物品的 FATAL 不打勾、TIME LIMIT 为 None、HEALING TIME 为 None —— 这正是这两行该有的样子。
5. 再执行一次 `await a.rollCrit(a, "character", {}, "52")`，**期望**正常走系统路径（控制台无 rebuilding 日志），
   物品 FATAL 打勾、TIME LIMIT 为 One Turn、EFFECTS 末尾带系统追加的「-1 to Medical Aid roll」。
   这一步同时在验证观察器的 `finally` 确实把 `table.draw` 还原了：若没还原，第二次调用会叠一层观察器。
   控制台执行 `Object.hasOwn(game.tables.get("XmFybNUYJ4C2OQyE"), "draw")`，
   **期望 `false`**（掷骰结束后表上不该留下任何自有 `draw` 属性）。
6. **今天的行为**：点 Roll Crit 完全没反应 —— 不加物品、不发卡、不弹警告，只有 F12 里一条
   `TypeError: Cannot read properties of undefined (reading 'length')`。

- [ ] **Step 23: MANUAL VERIFICATION —— 中文世界**

1. 启用 Babele 与 `alienrpg` 的中文翻译，让 EV - Critical Injuries 的行文本变成中文；重启世界。
2. 重复 Step 21 的第 5 步（D66 = 44）。
3. **期望（无论走哪条路径）**：物品名与 EFFECTS 显示**中文**，同时 **FATAL 打勾、TIME LIMIT 显示 One Shift 的中文标签**；聊天卡照旧是中文。
4. **今天的行为分两种，取决于译文里的冒号**，两种本特性都覆盖，请照实记录你这次遇到的是哪一种：
   - 译文保留了半角「冒号 + 空格」→ 系统的位置切分仍能切出 10 段，不报错，但 `ALIENRPG.Yes` 在 `cn.json` 里没翻（仍是 `"Yes"`），
     所以 `cFatal` 恒 `false`、`healTime` 恒 `0`：**物品是中文，FATAL 不打勾、TIME LIMIT 写 None，没有任何报错或提示**。
     本特性走「落地后按行号改写」路径修正（控制台**没有** rebuilding 日志）。
   - 译文改用了全角「：」→ `/[:] /` 切不动，`testArray[9]` 为 `undefined`，整次调用在 `:1889` 抛 TypeError，
     **点 Roll Crit 完全没反应**。本特性走「接住异常自行重建」路径修正（控制台**有** rebuilding 日志，
     且重建出来的物品 EFFECTS 末尾会带上那句医疗急救罚值备注，如果该行有的话）。
5. 这一条是本特性对本项目最重要的收益：死亡骰链条不再静默失效。

- [ ] **Step 24: 跑模组自检并提交记录**

F12 控制台执行：

```js
await game.modules.get("alien-evolved-automation").api.selftest.runAll();
```

Expected：

- `crit-result-parse-hardening.probe` → `ok: true`，detail 里 `snapshot` 是个几千的字符数（不是 0）、`defect=true`、`applied=true`、`reason=ok`。
  若 `snapshot=0`，说明快照取晚了（`register()` 必须由 `init` 里那个 `for (const f of FEATURES)` 循环最先调到，
  在 `patches.applyAll()` 之前），回 Step 19 确认没有多余的直调。
- `crit-result-parse-hardening.tableRows` → `ok: true`，detail 为
  `all 36 rows of the evolved table are in the catalog and agree with it (N cells not legible ...)`。
  若它报出 `cells that contradict the catalog`，说明**行号目录誊录错了或世界里的表被改过**，
  按 Step 8 的命令重新 dump 那一行核对后再改目录，不要跳过。
  末尾那个「not legible」计数在中文世界会大于 0，属正常（那些格子按「默认值不算证据」的规则未参与比对）。
- `crit-result-parse-hardening.legacyLookup` 是**诊断项**：英文世界应为 `true`；中文／Babele 世界若为 `false`，
  说明系统自己的按名查表已经找不到那张表（`actor.mjs:1811-1817`；`cn.json` 里 `ALIENRPG.EVCriticalInjuries` 这个键根本不存在，
  `localize` 只会回声键名），此时 `rollCrit` 会在 `:1821-1822` 弹 NoCharCrit 直接返回，
  本特性的观察器根本不会被触发。把 detail 原样记进提交信息，并回重绑面板确认绑定无误。

把三条结果写进提交：

```bash
git commit --allow-empty -m "test(crit-result-parse-hardening): 本机冒烟三项 + 自检三条通过" -m "Evolved 表 44/45/46/52/61 各行的时限不再是 None；
1e 表 11、12 两行不再吞掉整次重伤，改由模组补出物品、状态与聊天卡；
掷骰结束后表上没有残留的自有 draw 属性，观察器还原正常；
中文世界下文案仍是中文，而致命标记与时限按行号取到正确值（本次遇到的是哪条路径已记录）。
tableRows 自检把行号目录与世界里真实那张表逐行三项对过，无缺行、无分歧。
legacyLookup 诊断项结果一并记录。
按项目纪律，本机只是冒烟机，发布权威在 VPS。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```
