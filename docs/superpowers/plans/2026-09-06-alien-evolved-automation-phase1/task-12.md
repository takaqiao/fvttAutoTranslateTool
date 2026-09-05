> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 12 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 12: roll-pool-integrity —— 基础骰下限、压力骰不受修正影响、GM 掷骰不再凭空消失

**Files:**
- Create: `scripts/features/roll-pool-integrity.pure.mjs`
- Create: `scripts/features/roll-pool-integrity.mjs`
- Create: `test/roll-pool-integrity.test.mjs`
- Modify: `lang/en.json`、`lang/cn.json`（往已有的顶层 `AEA` 对象里合并键，嵌套结构）
- Modify: `scripts/main.mjs`（**只加两行**：`/* AEA-ANCHOR: imports */` 之后一行 import、`/* AEA-ANCHOR: features */` 之后一个数组元素。其余八个锚点与四个生命周期钩子的函数体**一行都不许动**）
- **不得修改** `test/stubs/foundry.mjs`（该文件属主是内核第一号任务，本任务只读）

**Interfaces:**

- **Consumes:**
  - `scripts/const.mjs` 的 `MID`（模组 id 字符串 `"alien-evolved-automation"`）与 `SYSTEM_ID`（`"alienrpg"`）
  - `scripts/kernel/features.mjs` 的 `features.register(def)`、`features.enabled(id)`
  - `scripts/kernel/patches.mjs` 的 `patches.register(def)`、`patches.status()`（返回 `[{id, type, target, applied, reason, fixedIn}]`）
  - `scripts/kernel/rollbus.mjs` 的 **`rollBus.addStage(target, {id, order, around})`** —— 介入系统掷骰的**唯一**合法通路，见下方「为什么不许自己调 libWrapper」
  - `scripts/kernel/resolver.mjs` 的 `resolver.actorById(id, {warn})`（遗留裸 actor id → Actor|null）
  - `scripts/kernel/selftest.mjs` 的 `selftest.register({id, label, run})`，其中 **`label` 是 i18n 键、不是已本地化文本**
  - `test/stubs/foundry.mjs` 的 `installFoundryStub()` / `uninstallFoundryStub()`（**只读使用**，不得改那个文件、不得就地造 `globalThis.game`）
  - 运行时全局：`game.alienrpg.yze.yzeRoll`（被修的系统函数，探针读它的源码文本）；`libWrapper` 仅出现在一条**反向断言**里（断言我们绝不注册它）
- **Produces:**
  - `scripts/features/roll-pool-integrity.pure.mjs` 的七个纯函数：
    `pureIsResourceRoll({actortype, isRadiationReducedLabel}) -> boolean`、
    `pureIsPushedRoll(reRoll) -> boolean`、
    `pureFloorApplies({actortype, isRadiationReducedLabel, reRoll}) -> boolean`、
    `pureResolveRollPools({base, stress, floorApplies}) -> {base, stress, change:"none"|"floored"|"clamped"}`、
    `pureNeedsAutoPanicSuppression({stress, autoPanic, resourceRoll, actorResolved}) -> boolean`、
    `pureSourceIsRecognizable(source) -> boolean`、
    `pureRollPoolIsBuggy(source) -> boolean`
  - `scripts/features/roll-pool-integrity.mjs` 的 `probeRollPoolIntegrity() -> boolean`、
    **`rollPoolIntegrityStage(next, args)`**（rollBus 阶段函数，签名见 Step 13）、
    `export const rollPoolIntegrityFeature = { id, register(), install() }`
  - `test/roll-pool-integrity.test.mjs`：52 条真测试（26 条纯逻辑 + 7 条探针判据 + 19 条桩驱动）
  - i18n 键：`AEA.feature.roll-pool-integrity.name` / `.hint`、`AEA.rollPool.floored`、
    `AEA.selftest.roll-pool-integrity.probe` / `.exemptions`
  - 两条自检条目，与手工验收一一对应（自检条目被删时这张表就会出现孤行）：

    | 自检 id | 它断言什么 | 对应手工步骤 |
    |---|---|---|
    | `roll-pool-integrity.probe` | 真实系统源码里的缺陷是否仍在，与 `patches.status()` 报告的 `applied` 是否一致 | Step 25 |
    | `roll-pool-integrity.exemptions` | 六种池子组合在**当前语言环境**下的钳制结果（vitest 只能跑英文键，跑不了 Babele 改过名的世界） | Step 21 / 22 / 23 |

---

**背景（不需要 Foundry 或 Alien RPG 前置知识）**

Foundry VTT 是网页版桌面 RPG 平台：「系统」提供规则实现，「模组」在其上打补丁。我们写的是模组，装在 `Data/modules/alien-evolved-automation/`；被修的系统是 `Data/systems/alienrpg/`（4.1.13）。系统把全部掷骰集中在一个静态函数：

```js
// systems/alienrpg/module/helpers/YZEDiceRoller.mjs:31
static async yzeRoll(actortype, blind, reRoll, label,
                     r1Dice, col1, r2Dice, col2,
                     actorid, itemid, tactorid, moddata)
```

`r1Dice` 是黑色「基础骰」数，`r2Dice` 是黄色「压力骰」数。22 个调用点全写作 `yze.yzeRoll(...)`，而 `game.alienrpg.yze` 就是该类对象本身（`module/alienrpg.mjs:72-81` 在系统的 `init` 钩子里把类挂上去），所以介入这一个静态属性对全部调用点生效。

**为什么不许自己调 libWrapper（这一轮最重要的一条）**

**libWrapper** 是一个已安装的模组，提供全局 `libWrapper`，用来安全地包裹别人的函数。它有一条硬规则：**同一个模组对同一个目标只能注册一次**，第二次注册直接抛
`A wrapper for '<target>' (ID=<n>) has already been registered by <module>.`。

本模组的内核里有一层「掷骰记录层」（`scripts/kernel/rollbus.mjs`，下称 **rollBus**），它已经把 `game.alienrpg.yze.yzeRoll` 用 libWrapper 包住了 —— 它要在每次掷骰前后记录成功数、骰池、推骰父子关系，写进聊天卡的模组标记里。所以 `yzeRoll` 这个目标**已经被本模组占用**。如果本任务再去 `libWrapper.register(MID, "game.alienrpg.yze.yzeRoll", ...)`，结果不是报错就是**静默变成空操作**（上一轮的计划审计正是在这里抓到两条修复全程没生效）。

因此 rollBus 提供了排队接口，本任务**必须**用它：

```js
rollBus.addStage(target, { id, order = 0, around })
// target 取值："yzeRoll" | "abilityRoll" | "itemRoll" | "pushRoll"
// around(next, args, thisArg) -> any
//   · 必须恰好调用一次 next(args)；args 是一个**数组**，就是那 12 个位置参数
//   · 可以改 args（换一个新数组传给 next）、可以改返回值
// order 小的先执行（更靠外）；同 order 按注册顺序
```

`addStage` 只是**排队**：它把阶段登记进 rollBus 的阶段表，由 rollBus 自己那一次 libWrapper 注册在运行时按序调用。本任务的 `apply()` 在 `ready` 阶段的 `patches.applyAll()` 里跑，**早于** `rollBus.install()`，这是契约钉死的顺序（`ready.patches` 子锚点排在 `ready.rollbus` 之前），所以「先排队、后安装」是正常路径，不需要等待任何东西。

**顺带一条本模组内部的约定**（不用你做，只是让你知道为什么必须改 `args` 而不是改别处）：rollBus 写进聊天卡的 `pools.base` / `pools.stress` 一律读系统的全局 `game.alienrpg.rollArr`（**真正掷出去的骰子**），而不是读它自己看到的入参 —— 因为它看到的正是你钳制**之前**的值。所以你在阶段里改的池子会被如实记录，你不需要再通知任何人。

**要修的两个缺陷（已逐行核对 4.1.13 源码）**

缺陷一，`YZEDiceRoller.mjs:137-145`：

```js
} else {
    if (r1Dice < 0) {
        r2Dice = r2Dice + r1Dice          // :139 基础骰的负缺口被塞进压力骰
        if (r2Dice < 1) {
            return ui.notifications.warn(game.i18n.localize("ALIENRPG.NoDice"))  // :141 整次掷骰被拒，不建卡
        }
    }
    roll1 = 0 + "db"
}
```

规则书（`alien-evolved-corerules` 合集，日志「Alien Evolved Player Guide」的「3. SKILLS AND TALENTS」页，逐字）：

> The rules or the GM can modify your chances, adding or removing base dice for you to roll. … **You can never go below one base die. Modifiers never affect stress dice.**

所以 −4 修正应把基础池压到 1 颗、压力池原封不动；今天却是基础池 0 颗、压力池被扣 4 颗，甚至整次被拒。

缺陷二，出货的 GM 掷骰宏 `macros/gmRollYZEDiceMacro.js:34` 只传 8 个参数：

```js
await game.alienrpg.yze.yzeRoll(hostile, blind, reRoll, label, r1Data, 'Black', r2Data, 'Stress');
```

于是 `actorid` 是 `undefined`。压力骰出 1 时进自动恐慌分支（`:186-243`），`:198` 先查世界设置 `game.settings.get("alienrpg", "autopanic")`（默认开，注册在 `module/helpers/settings.mjs:26`，`scope: "world"`、`default: true`、无条件注册），然后 `:201`（以及 `:210`/`:222`/`:231`）`myActor = game.actors.get(actorid)` 得到 `undefined`，紧接着的 `:204`（同形还有 `:213`/`:225`/`:234`）`panicroll: myActor.getRollData().header.stress.value` 抛 TypeError。抛出点在 `:201-205`，而 `ChatMessage.create(chatData)` 在 `:416` —— 异常一抛整张卡就没了：GM 看到骰子动画播完却没有任何聊天卡，且只在压力骰出 1 时发生，像随机故障。

**必须避开的三个陷阱：不能在这个接缝上无差别加下限。**

*陷阱一，资源掷骰的黄骰根本不是压力骰。* `YZEDiceRoller.mjs:154`：

```js
if (actortype === "supply" || label === game.i18n.localize("ALIENRPG.RadiationReduced")) {
    if (r2Dice > 6) { r2Dice = 6; com = `${r2Dice}` + "ds" } else { com = `${roll2}` }
} else { com = `${roll1}` + "+" + `${roll2}` }
```

补给（`module/documents/actor.mjs:1633` 传 `"supply"`）、弹药（`module/documents/item.mjs:642` 同样传 `"supply"`，`:626` 的 `r1Data = 0`）、飞船与载具补给（`spacecraft-sheet.mjs`、`vehicle-sheet.mjs` 同形）、减辐射（`module/documents/actor.mjs:1524-1533` 的 1e 分支传 `label = localize("ALIENRPG.RadiationReduced")`、`:1516` 的 `r1Data = 0`，而 `actortype` 传的是 `effectiveActorType = actor.type`，所以这一路**只能**靠标签认出来）这几类的「基础池为 0」是系统故意的，绝不能钳到 1。

注意这里的 `label ===` 比较**不是**契约 §7 禁止的「按显示名查文档」：它是把同一个 i18n 键正向本地化后自比，调用方（`actor.mjs:1515`）与系统判据（`:154`）用的是同一句 `localize("ALIENRPG.RadiationReduced")`，任何语言下都逐字相等；我们必须与系统的分支判据**完全一致**，否则钳制会和系统实际掷的池子对不上。禁止反向解析译文。

*陷阱二，推骰（push）重掷的是「上一次没出 6 的那些骰子」，0 颗是合法结果。* `module/documents/actor.mjs:1302` 的 `pushRoll` 这样算重掷池：

```js
// :1313-1314
const reRoll1 = game.alienrpg.rollArr.r1Dice - game.alienrpg.rollArr.r1Six;
const reRoll2 = game.alienrpg.rollArr.r2Dice + 1 - (game.alienrpg.rollArr.r2One + game.alienrpg.rollArr.r2Six);
```

上一轮的黑骰若全是 6，`reRoll1` 就是 0 —— 这是「没有可重掷的基础骰」，不是「修正把池子压没了」。此时补一颗黑骰等于凭空多给玩家一次成功机会。推骰路径靠第三参 `reRoll` 辨认：聊天卡上的推骰按钮（`module/alienrpg.mjs:474` 写 `let reRoll = "push";`，勾了多重推骰复选框时 `:477` 改成 `"mPush"`）；**其它所有调用点传的是布尔 `true`/`false`**（`YZEDiceRoller.mjs:341-342`、`:378` 都按这两个字符串分支）。所以判据是 `reRoll === "push" || reRoll === "mPush"`，不是「reRoll 为真」。

*陷阱三，两个出货的掷骰宏是「裸骰工具」，不是角色检定。* `macros/gmRollYZEDiceMacro.js:2` 与 `macros/playerRollYZEDiceMacro.js:2` 都写死 `let hostile = false;` 并把它当第一参 `actortype` 传进去。GM 在对话框里填「基础 0、压力 6」是明确的意图，不该被我们改成 1 颗。而其余全部调用点传的 `actortype` 都是非空字符串：`actor.type`（`"character"` / `"synthetic"` / `"creature"` / `"vehicles"` / `"spacecraft"`）、字面量 `"supply"`、或 `module/documents/actor.mjs:2562` 的 `const hostile = "creature"`。所以下限规则只在 `typeof actortype === "string" && actortype.length > 0` 时生效。

但「负基础骰不许去偷压力骰」这条**对裸骰宏也成立**：GM 若填了 −3，今天照样会去扣黄骰。对这类掷骰我们把基础池夹到 0（而不是 1），偷窃分支就进不去了。

*第四个要保住的行为*：`YZEDiceRoller.mjs:120-122` 的

```js
if (!r1Dice && !r2Dice) {
    return ui.notifications.warn(game.i18n.localize("ALIENRPG.NoAttribute"))
}
```

两池皆空时该弹「无属性」，不能被下限规则变成一次 1 颗骰的掷骰。

**探针（probe）为什么是两个纯谓词而不是一句 `includes`**

契约 §4 K7 要求每个补丁带一个可运行的 `probe()`：返回 `true` 表示「缺陷仍在，该装」，返回 `false` 表示「上游已修，退休」。判据只能来自系统函数的源码文本（`Function.prototype.toString()`）。本任务把判据拆成两个可单测的纯谓词，各自要有正反两个方向的夹具，避免写成一个恒真的 `includes` —— 那正是「上游修好后我们重复修一遍」这号风险的来源。

三个记号（都已在 4.1.13 里逐字核对过，且确认没有任何守卫写法出现在同一份源码里）：

- 偷压力骰：源码含 `r2Dice = r2Dice + r1Dice`（`:139`）。这句话本身就是缺陷，修好后不可能还留着。
- 恐慌裸解引用：源码含 `panicroll: myActor.getRollData()`（`:204`/`:213`/`:225`/`:234`）。上游若加可选链就变成 `myActor?.getRollData()`，记号消失。若上游改成提前返回而保留裸解引用，我们会继续报「缺陷在」并保持抑制 —— 这个方向是无害的（见下），且诚实：裸解引用确实还在。
- 确认「我们读到的确实是系统那个函数」：`ui.notifications.warn(game.i18n.localize("ALIENRPG.NoAttribute"))`（`:121`）。读不到（例如 rollBus 已经把它包成了 libWrapper 的分发壳）时 `probe()` 取**保守方向**返回 `true`，理由要写进注释：本补丁的两半对一个已经修好的系统都是空操作 —— 池子已 ≥1 时钳制不改任何值；掷骰本来就没有角色时，修好的系统也只会跳过恐慌。所以「读不到就装」不会造成重复修复。

---

- [ ] **Step 1: 写下会失败的纯函数测试（骰池整形）**

新建 `test/roll-pool-integrity.test.mjs`：

```js
import { describe, expect, it } from "vitest";
import {
  pureFloorApplies,
  pureIsPushedRoll,
  pureIsResourceRoll,
  pureNeedsAutoPanicSuppression,
  pureResolveRollPools,
} from "../scripts/features/roll-pool-integrity.pure.mjs";

describe("pureIsResourceRoll", () => {
  it("treats the supply actortype as a resource roll", () => {
    expect(pureIsResourceRoll({ actortype: "supply", isRadiationReducedLabel: false })).toBe(true);
  });
  it("treats the radiation-reduced label as a resource roll", () => {
    expect(pureIsResourceRoll({ actortype: "character", isRadiationReducedLabel: true })).toBe(true);
  });
  it("treats an ordinary character skill roll as a test, not a resource roll", () => {
    expect(pureIsResourceRoll({ actortype: "character", isRadiationReducedLabel: false })).toBe(false);
  });
});

describe("pureIsPushedRoll", () => {
  it("recognises the push button's reRoll value", () => {
    expect(pureIsPushedRoll("push")).toBe(true);
  });
  it("recognises the multi-push reRoll value", () => {
    expect(pureIsPushedRoll("mPush")).toBe(true);
  });
  it("does not mistake the boolean reRoll flag for a push", () => {
    expect(pureIsPushedRoll(true)).toBe(false);
  });
  it("treats a first roll as not pushed", () => {
    expect(pureIsPushedRoll(false)).toBe(false);
  });
});

describe("pureFloorApplies", () => {
  it("applies to a character skill roll", () => {
    expect(pureFloorApplies({ actortype: "character", isRadiationReducedLabel: false, reRoll: false })).toBe(true);
  });
  it("applies to a creature attack roll", () => {
    expect(pureFloorApplies({ actortype: "creature", isRadiationReducedLabel: false, reRoll: false })).toBe(true);
  });
  it("does not apply to a supply roll", () => {
    expect(pureFloorApplies({ actortype: "supply", isRadiationReducedLabel: false, reRoll: false })).toBe(false);
  });
  it("does not apply to a radiation-reduction roll", () => {
    expect(pureFloorApplies({ actortype: "character", isRadiationReducedLabel: true, reRoll: false })).toBe(false);
  });
  it("does not apply to a pushed roll, whose zero base pool means no dice are left to reroll", () => {
    expect(pureFloorApplies({ actortype: "character", isRadiationReducedLabel: false, reRoll: "push" })).toBe(false);
  });
  it("does not apply to the raw dice-roller macros, which pass false as the actortype", () => {
    expect(pureFloorApplies({ actortype: false, isRadiationReducedLabel: false, reRoll: true })).toBe(false);
  });
});

describe("pureResolveRollPools", () => {
  it("floors the base pool at one die and leaves stress untouched", () => {
    expect(pureResolveRollPools({ base: -4, stress: 3, floorApplies: true }))
      .toEqual({ base: 1, stress: 3, change: "floored" });
  });
  it("floors a base pool that a penalty reduced to exactly zero", () => {
    expect(pureResolveRollPools({ base: 0, stress: 3, floorApplies: true }))
      .toEqual({ base: 1, stress: 3, change: "floored" });
  });
  it("floors a negative base pool even when there are no stress dice", () => {
    expect(pureResolveRollPools({ base: -2, stress: 0, floorApplies: true }))
      .toEqual({ base: 1, stress: 0, change: "floored" });
  });
  it("leaves an empty roll empty so the system can still warn NoAttribute", () => {
    expect(pureResolveRollPools({ base: 0, stress: 0, floorApplies: true }))
      .toEqual({ base: 0, stress: 0, change: "none" });
  });
  it("never floors a resource roll, whose base pool is deliberately unrolled", () => {
    expect(pureResolveRollPools({ base: 0, stress: 6, floorApplies: false }))
      .toEqual({ base: 0, stress: 6, change: "none" });
  });
  it("never floors a push whose base dice all showed a six", () => {
    expect(pureResolveRollPools({ base: 0, stress: 4, floorApplies: false }))
      .toEqual({ base: 0, stress: 4, change: "none" });
  });
  it("clamps a negative base pool to zero when the floor does not apply, so stress stays whole", () => {
    expect(pureResolveRollPools({ base: -3, stress: 5, floorApplies: false }))
      .toEqual({ base: 0, stress: 5, change: "clamped" });
  });
  it("passes a healthy pool through unchanged", () => {
    expect(pureResolveRollPools({ base: 5, stress: 2, floorApplies: true }))
      .toEqual({ base: 5, stress: 2, change: "none" });
  });
});

describe("pureNeedsAutoPanicSuppression", () => {
  it("suppresses auto-panic for an actor-less roll that carries stress dice", () => {
    expect(pureNeedsAutoPanicSuppression({ stress: 6, autoPanic: true, resourceRoll: false, actorResolved: false })).toBe(true);
  });
  it("leaves auto-panic alone when the roll has a real actor", () => {
    expect(pureNeedsAutoPanicSuppression({ stress: 6, autoPanic: true, resourceRoll: false, actorResolved: true })).toBe(false);
  });
  it("leaves auto-panic alone when there are no stress dice at all", () => {
    expect(pureNeedsAutoPanicSuppression({ stress: 0, autoPanic: true, resourceRoll: false, actorResolved: false })).toBe(false);
  });
  it("leaves auto-panic alone on a resource roll", () => {
    expect(pureNeedsAutoPanicSuppression({ stress: 6, autoPanic: true, resourceRoll: true, actorResolved: false })).toBe(false);
  });
  it("does nothing when the world has auto-panic switched off", () => {
    expect(pureNeedsAutoPanicSuppression({ stress: 6, autoPanic: false, resourceRoll: false, actorResolved: false })).toBe(false);
  });
});
```

- [ ] **Step 2: 跑它，看它失败**

Run: `npx vitest run test/roll-pool-integrity.test.mjs`

Expected: FAIL —— `Failed to resolve import "../scripts/features/roll-pool-integrity.pure.mjs" from "test/roll-pool-integrity.test.mjs"`。26 条用例一条都没执行。

- [ ] **Step 3: 写纯函数层（骰池整形）**

新建 `scripts/features/roll-pool-integrity.pure.mjs`。契约 §0.1 分层铁律：本文件不得引用任何 Foundry 全局（`game` / `ui` / `canvas` / `CONFIG` / `Hooks` / `foundry` / `ChatMessage` / `Roll` / `libWrapper`）；契约 §6 要求 `.pure.mjs` 只导出 `pure*` 函数，所以记号常量留作模块私有，由下面的纯函数代为暴露。

```js
// Pure layer. Must not touch game / ui / canvas / CONFIG / Hooks / foundry /
// ChatMessage / Roll / libWrapper. Only pure* functions are exported.

/** The actortype the system passes for supply, ammo, vehicle and spacecraft resource rolls. */
const RESOURCE_ACTOR_TYPE = "supply";

/** The two reRoll values the chat card's push button sends (alienrpg.mjs:474, :477). */
const PUSH_REROLL_VALUES = ["push", "mPush"];

/**
 * A "resource roll" is one where the yellow pool is NOT stress (supply, ammo,
 * radiation reduction). YZEDiceRoller.mjs:154 builds the formula from the yellow
 * pool alone and never rolls the base pool, so a base pool of 0 is correct there
 * and must never be floored. The caller decides isRadiationReducedLabel by
 * forward-localizing the same key the system compares against.
 */
export function pureIsResourceRoll({ actortype, isRadiationReducedLabel }) {
  return actortype === RESOURCE_ACTOR_TYPE || isRadiationReducedLabel === true;
}

/**
 * A pushed roll rerolls the dice that did not show a six (actor.mjs:1313-1314).
 * A base pool of 0 there means "every base die was a six", not "a penalty ate
 * the pool", so flooring it would hand the player a die they did not earn.
 */
export function pureIsPushedRoll(reRoll) {
  return PUSH_REROLL_VALUES.includes(reRoll);
}

/**
 * The one-base-die floor is a rule about character tests. It does not apply to
 * resource rolls, to pushes, or to the two raw dice-roller macros, which pass
 * the boolean false as their actortype (gmRollYZEDiceMacro.js:2,
 * playerRollYZEDiceMacro.js:2) while every other call site passes a non-empty
 * actor type string.
 */
export function pureFloorApplies({ actortype, isRadiationReducedLabel, reRoll }) {
  if (pureIsResourceRoll({ actortype, isRadiationReducedLabel })) return false;
  if (pureIsPushedRoll(reRoll)) return false;
  return typeof actortype === "string" && actortype.length > 0;
}

/**
 * Rulebook, Player Guide "3. SKILLS AND TALENTS":
 *   "You can never go below one base die. Modifiers never affect stress dice."
 *
 * stress is returned exactly as it came in, always: that is the whole point.
 * When the floor does not apply we still lift a negative base pool to 0, because
 * YZEDiceRoller.mjs:139 subtracts a negative base pool from the stress pool and
 * :141 then refuses the whole roll. Both pools empty is left alone so the system
 * still reaches its own NoAttribute warning at :121.
 */
export function pureResolveRollPools({ base, stress, floorApplies }) {
  const b = Number(base);
  const s = Number(stress);
  const unchanged = { base, stress, change: "none" };
  if (!Number.isFinite(b)) return unchanged;

  const lower = floorApplies === true ? 1 : 0;
  const hasStress = Number.isFinite(s) && s >= 1;
  if (b === 0 && !hasStress) return unchanged;
  if (b >= lower) return unchanged;

  return { base: lower, stress, change: lower === 1 ? "floored" : "clamped" };
}

/**
 * The system's auto-panic branch does game.actors.get(actorid) at :201 and
 * dereferences the result at :204 with no guard. With no actorid the whole roll
 * throws before its chat card is created at :416. Detect that shape.
 *
 * resourceRoll is belt and braces: :186-190 already excludes supply and
 * radiation rolls from the branch, and we do not want to depend on that staying
 * true across system versions.
 */
export function pureNeedsAutoPanicSuppression({ stress, autoPanic, resourceRoll, actorResolved }) {
  if (autoPanic !== true) return false;
  if (resourceRoll) return false;
  if (!(Number(stress) >= 1)) return false;
  return !actorResolved;
}
```

- [ ] **Step 4: 跑它，看它通过**

Run: `npx vitest run test/roll-pool-integrity.test.mjs`

Expected: PASS — `Tests  26 passed (26)`。

- [ ] **Step 5: 提交纯函数层**

```bash
git add scripts/features/roll-pool-integrity.pure.mjs test/roll-pool-integrity.test.mjs
git commit -m "feat(roll-pool-integrity): 基础骰下限与三类豁免的纯函数层" -m "规则原文：永远不能低于一颗基础骰，修正永不影响压力骰。
三类豁免逐条对应源码：补给/弹药/减辐射的黄骰不是压力骰（YZEDiceRoller.mjs:154）；
推骰重掷的是没出 6 的骰子，0 颗是合法结果（documents/actor.mjs:1313）；
两个裸骰宏把 actortype 传成布尔 false（gmRollYZEDiceMacro.js:2），GM 填 0 就是 0。
豁免时仍把负基础池夹到 0，堵住 :139 的偷窃分支；两池皆空保留，让系统照旧弹 NoAttribute。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: 为探针判据写会失败的测试**

把 Step 1 写下的 import 语句改成下面这行（新增两个名字）：

```js
import {
  pureFloorApplies,
  pureIsPushedRoll,
  pureIsResourceRoll,
  pureNeedsAutoPanicSuppression,
  pureResolveRollPools,
  pureRollPoolIsBuggy,
  pureSourceIsRecognizable,
} from "../scripts/features/roll-pool-integrity.pure.mjs";
```

在文件末尾追加下面整段。五个夹具函数**永远不会被调用**，只会被 `String()` 取源码 —— 探针在真实系统上做的就是这件事，所以夹具用真函数而不是字符串常量，能顺带保证被测文本经过了和生产代码同一套解析。夹具不是整段源码的复制品：它逐字包含探针要找的三个记号（各自的源码行写在注释里），其余部分用注释省略。

```js
// --- probe fixtures -------------------------------------------------------
// Marker A (YZEDiceRoller.mjs:139): r2Dice = r2Dice + r1Dice
// Marker B (YZEDiceRoller.mjs:204): panicroll: myActor.getRollData()
// Marker C (YZEDiceRoller.mjs:121): the NoAttribute warning, proof that we are
//          looking at the real yzeRoll body and not at a libWrapper shell.

function fixtureSystem4113(r1Dice, r2Dice, actorid, tactorid) {
  if (!r1Dice && !r2Dice) {
    return ui.notifications.warn(game.i18n.localize("ALIENRPG.NoAttribute"));
  }
  if (r1Dice >= 1) {
    // ... :123-136 omitted
  } else {
    if (r1Dice < 0) {
      r2Dice = r2Dice + r1Dice;
      if (r2Dice < 1) {
        return ui.notifications.warn(game.i18n.localize("ALIENRPG.NoDice"));
      }
    }
  }
  // ... :146-197 omitted
  if (game.settings.get("alienrpg", "autopanic")) {
    if (tactorid !== "spacecraft") {
      myActor = game.actors.get(actorid);
      dataset = {
        panicroll: myActor.getRollData().header.stress.value,
      };
      myActor.rollPanic(myActor, dataset);
    }
  }
}

function fixturePoolFixedOnly(r1Dice, r2Dice, actorid, tactorid) {
  if (!r1Dice && !r2Dice) {
    return ui.notifications.warn(game.i18n.localize("ALIENRPG.NoAttribute"));
  }
  if (r1Dice < 1) {
    r1Dice = 1;
  }
  if (game.settings.get("alienrpg", "autopanic")) {
    if (tactorid !== "spacecraft") {
      myActor = game.actors.get(actorid);
      dataset = {
        panicroll: myActor.getRollData().header.stress.value,
      };
      myActor.rollPanic(myActor, dataset);
    }
  }
}

function fixturePanicFixedOnly(r1Dice, r2Dice, actorid, tactorid) {
  if (!r1Dice && !r2Dice) {
    return ui.notifications.warn(game.i18n.localize("ALIENRPG.NoAttribute"));
  }
  if (r1Dice < 0) {
    r2Dice = r2Dice + r1Dice;
  }
  if (game.settings.get("alienrpg", "autopanic")) {
    if (tactorid !== "spacecraft") {
      myActor = game.actors.get(actorid);
      dataset = {
        panicroll: myActor?.getRollData().header.stress.value,
      };
      myActor?.rollPanic(myActor, dataset);
    }
  }
}

function fixtureBothFixed(r1Dice, r2Dice, actorid, tactorid) {
  if (!r1Dice && !r2Dice) {
    return ui.notifications.warn(game.i18n.localize("ALIENRPG.NoAttribute"));
  }
  if (r1Dice < 1) {
    r1Dice = 1;
  }
  if (game.settings.get("alienrpg", "autopanic")) {
    if (tactorid !== "spacecraft") {
      myActor = game.actors.get(actorid);
      dataset = {
        panicroll: myActor?.getRollData().header.stress.value,
      };
      myActor?.rollPanic(myActor, dataset);
    }
  }
}

function fixtureWrapperShell(...args) {
  return libWrapper._dispatch(this, args);
}

describe("pureSourceIsRecognizable", () => {
  it("recognises the real yzeRoll body", () => {
    expect(pureSourceIsRecognizable(String(fixtureSystem4113))).toBe(true);
  });
  it("does not recognise a libWrapper dispatch shell", () => {
    expect(pureSourceIsRecognizable(String(fixtureWrapperShell))).toBe(false);
  });
});

describe("pureRollPoolIsBuggy", () => {
  it("reports the shipped 4.1.13 body as still defective", () => {
    expect(pureRollPoolIsBuggy(String(fixtureSystem4113))).toBe(true);
  });
  it("still reports a body where only the pool theft was fixed", () => {
    expect(pureRollPoolIsBuggy(String(fixturePoolFixedOnly))).toBe(true);
  });
  it("still reports a body where only the panic dereference was guarded", () => {
    expect(pureRollPoolIsBuggy(String(fixturePanicFixedOnly))).toBe(true);
  });
  it("retires itself once upstream fixes both halves", () => {
    expect(pureRollPoolIsBuggy(String(fixtureBothFixed))).toBe(false);
  });
  it("reports nothing for a non-string source", () => {
    expect(pureRollPoolIsBuggy(undefined)).toBe(false);
  });
});
```

- [ ] **Step 7: 跑它，看它失败**

Run: `npx vitest run test/roll-pool-integrity.test.mjs`

Expected: FAIL —— 整个文件在收集阶段就报 `SyntaxError: The requested module '/scripts/features/roll-pool-integrity.pure.mjs' does not provide an export named 'pureRollPoolIsBuggy'`，33 条用例一条都没执行（Step 4 已通过的 26 条也一起变红，这是正常的：ESM 具名导入在模块求值前就失败）。

- [ ] **Step 8: 写探针判据**

把下面这段追加到 `scripts/features/roll-pool-integrity.pure.mjs` 末尾：

```js
/** YZEDiceRoller.mjs:139 — the base-pool deficit being subtracted from stress. */
const MARK_BASE_POOL_THEFT = "r2Dice = r2Dice + r1Dice";

/** YZEDiceRoller.mjs:204/213/225/234 — myActor dereferenced with no guard. */
const MARK_PANIC_BARE_DEREF = "panicroll: myActor.getRollData()";

/** YZEDiceRoller.mjs:121 — proof that this text really is the system's yzeRoll. */
const MARK_YZE_BODY = 'ui.notifications.warn(game.i18n.localize("ALIENRPG.NoAttribute"))';

/**
 * True when the text really is the system's yzeRoll body. A libWrapper dispatch
 * shell, an empty string or anything else answers false, and the effect layer
 * then decides what to do about not knowing.
 */
export function pureSourceIsRecognizable(source) {
  return typeof source === "string" && source.includes(MARK_YZE_BODY);
}

/**
 * True means "the defect is still present, keep the patch". Either half alone is
 * enough: they are two independent repairs riding one stage.
 */
export function pureRollPoolIsBuggy(source) {
  if (typeof source !== "string") return false;
  return source.includes(MARK_BASE_POOL_THEFT) || source.includes(MARK_PANIC_BARE_DEREF);
}
```

- [ ] **Step 9: 跑它，看它通过**

Run: `npx vitest run test/roll-pool-integrity.test.mjs`

Expected: PASS — `Tests  33 passed (33)`。

- [ ] **Step 10: 提交探针判据**

```bash
git add scripts/features/roll-pool-integrity.pure.mjs test/roll-pool-integrity.test.mjs
git commit -m "feat(roll-pool-integrity): 探针判据拆成两个可双向单测的纯谓词" -m "两个缺陷记号各自逐字取自 4.1.13（:139 的偷窃语句、:204 的裸解引用），
第三个记号（:121 的 NoAttribute）用来确认读到的确实是系统函数而不是 libWrapper 分发壳。
四个夹具覆盖四种上游状态：原样、只修池子、只修解引用、两个都修 —— 最后一个必须让探针退休。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 11: 为副作用层写会失败的测试**

契约 §0.3 规定：测试里的 Foundry 全局**只能**来自共享桩 `test/stubs/foundry.mjs`，**不得**在测试里就地造 `globalThis.game`，也**不得修改那个桩文件**（它由内核第一号任务独占实现，并有 `test/stub-fidelity.test.mjs` 逐条守卫）。本测试用到的都是桩的行为契约里已经承诺过的东西，直接用即可：

- `game.settings.register(ns, key, def)` 把 `def.default` 播种进后备存储，`get` 读得到、`set` 可写并返回 Promise，`get` 对**未注册**的键抛 `Error`；
- `game.i18n.localize` / `format`：默认词典为空，`localize` 对未知键**原样回声键名**（所以下面的减辐射用例拿 `game.i18n.localize("ALIENRPG.RadiationReduced")` 当 label，两边一致）；
- `ui.notifications.info`（记进 `ctx.notifications`）；
- `libWrapper.register`（我们只用来做一条**反向断言**：确认本特性从不注册它）。

若跑起来发现桩缺了其中任何一样，**不要在本文件里就地糊**，也不要去改桩 —— 那是内核第一号任务的缺陷，报给主控。

`game.alienrpg` 是系统自己的命名空间，桩不提供 —— 我们在每个用例里把它当**夹具数据**赋到桩的 `game` 上，这和往桩里塞世界文档是同一性质，不是另造一个假的子系统。

在文件顶部现有 import 之后插入这段（`vi.mock` 会被 vitest 提升到所有 import 之前，写在这里只是为了读起来顺）：

```js
import { afterEach, beforeEach, vi } from "vitest";
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";
import { MID, SYSTEM_ID } from "../scripts/const.mjs";

// Our own kernel modules are mocked: they are our seams, not Foundry's.
vi.mock("../scripts/kernel/features.mjs", () => ({
  features: { register: vi.fn(), enabled: vi.fn(() => true) },
}));
vi.mock("../scripts/kernel/patches.mjs", () => ({
  patches: { register: vi.fn(), status: vi.fn(() => []) },
}));
vi.mock("../scripts/kernel/selftest.mjs", () => ({ selftest: { register: vi.fn() } }));
vi.mock("../scripts/kernel/resolver.mjs", () => ({ resolver: { actorById: vi.fn() } }));
vi.mock("../scripts/kernel/rollbus.mjs", () => ({ rollBus: { addStage: vi.fn() } }));

import { features } from "../scripts/kernel/features.mjs";
import { patches } from "../scripts/kernel/patches.mjs";
import { resolver } from "../scripts/kernel/resolver.mjs";
import { rollBus } from "../scripts/kernel/rollbus.mjs";
import { selftest } from "../scripts/kernel/selftest.mjs";
import {
  probeRollPoolIntegrity,
  rollPoolIntegrityFeature,
  rollPoolIntegrityStage,
} from "../scripts/features/roll-pool-integrity.mjs";
```

再把下面整段追加到文件末尾。`rollArgs` 帮手把 §2 那 12 个参数按名字组装成**一个数组**（rollBus 的阶段函数第二参就是这个数组），免得每条用例数逗号：

```js
// The 12 positional arguments of YZEDiceRoller.mjs:31, by name, as one array.
function rollArgs({ actortype = "character", blind = false, reRoll = false, label = "Heavy Machinery",
                    r1Dice = 4, col1 = "Black", r2Dice = 0, col2 = "Yellow",
                    actorid = "abc", itemid, tactorid, moddata } = {}) {
  return [actortype, blind, reRoll, label, r1Dice, col1, r2Dice, col2, actorid, itemid, tactorid, moddata];
}

describe("rollPoolIntegrityStage", () => {
  let infoSpy;

  beforeEach(() => {
    installFoundryStub();
    game.alienrpg = { yze: { yzeRoll: fixtureSystem4113 } };
    game.settings.register(SYSTEM_ID, "autopanic", { scope: "world", config: true, type: Boolean, default: true });
    features.enabled.mockReturnValue(true);
    resolver.actorById.mockImplementation((id) => (id === "abc" ? { id: "abc", name: "Ripley" } : null));
    infoSpy = vi.spyOn(ui.notifications, "info");
  });
  afterEach(() => {
    vi.restoreAllMocks();
    uninstallFoundryStub();
  });

  it("hands the system a floored base pool, an untouched stress pool and one notice", async () => {
    const next = vi.fn(async () => "card");
    await rollPoolIntegrityStage(next, rollArgs({ r1Dice: -4, r2Dice: 3 }));
    expect(next).toHaveBeenCalledTimes(1);
    expect(next.mock.calls[0][0][4]).toBe(1);
    expect(next.mock.calls[0][0][6]).toBe(3);
    expect(infoSpy).toHaveBeenCalledTimes(1);
  });

  it("leaves a supply roll's zero base pool alone and says nothing", async () => {
    const next = vi.fn(async () => "card");
    await rollPoolIntegrityStage(next, rollArgs({ actortype: "supply", label: "Air", r1Dice: 0, r2Dice: 5 }));
    expect(next.mock.calls[0][0][4]).toBe(0);
    expect(next.mock.calls[0][0][6]).toBe(5);
    expect(infoSpy).not.toHaveBeenCalled();
  });

  it("leaves a radiation-reduced roll's zero base pool alone", async () => {
    const next = vi.fn(async () => "card");
    const label = game.i18n.localize("ALIENRPG.RadiationReduced");
    await rollPoolIntegrityStage(next, rollArgs({ label, r1Dice: 0, r2Dice: 3 }));
    expect(next.mock.calls[0][0][4]).toBe(0);
    expect(infoSpy).not.toHaveBeenCalled();
  });

  it("leaves a push alone when every base die showed a six", async () => {
    const next = vi.fn(async () => "card");
    await rollPoolIntegrityStage(next, rollArgs({ reRoll: "push", r1Dice: 0, r2Dice: 4 }));
    expect(next.mock.calls[0][0][4]).toBe(0);
    expect(next.mock.calls[0][0][6]).toBe(4);
    expect(infoSpy).not.toHaveBeenCalled();
  });

  it("clamps the raw GM macro's negative base pool to zero instead of flooring it", async () => {
    const next = vi.fn(async () => "card");
    await rollPoolIntegrityStage(next, rollArgs({ actortype: false, label: "GM", reRoll: true, r1Dice: -3, r2Dice: 5, actorid: undefined }));
    expect(next.mock.calls[0][0][4]).toBe(0);
    expect(next.mock.calls[0][0][6]).toBe(5);
    expect(infoSpy).not.toHaveBeenCalled();
  });

  it("hides autopanic from the system while an actor-less roll runs, then restores it", async () => {
    const seen = [];
    const next = vi.fn(async () => {
      seen.push(game.settings.get(SYSTEM_ID, "autopanic"));
      return "card";
    });
    await rollPoolIntegrityStage(next, rollArgs({ actortype: false, label: "GM", reRoll: true, r1Dice: 1, r2Dice: 6, actorid: undefined }));
    expect(seen).toEqual([false]);
    expect(game.settings.get(SYSTEM_ID, "autopanic")).toBe(true);
  });

  it("restores the setting even when the system throws", async () => {
    const next = vi.fn(async () => { throw new Error("boom"); });
    await expect(rollPoolIntegrityStage(next, rollArgs({ actortype: false, label: "GM", reRoll: true, r1Dice: 1, r2Dice: 6, actorid: undefined })))
      .rejects.toThrow("boom");
    expect(game.settings.get(SYSTEM_ID, "autopanic")).toBe(true);
  });

  it("keeps hiding autopanic while a second actor-less roll is still in flight", async () => {
    const seen = [];
    let releaseA;
    let releaseB;
    const nextA = vi.fn(() => new Promise((resolve) => { releaseA = () => resolve("A"); }));
    const nextB = vi.fn(() => new Promise((resolve) => {
      releaseB = () => { seen.push(game.settings.get(SYSTEM_ID, "autopanic")); resolve("B"); };
    }));
    const args = rollArgs({ actortype: false, label: "GM", reRoll: true, r1Dice: 1, r2Dice: 6, actorid: undefined });
    const pA = rollPoolIntegrityStage(nextA, args);
    const pB = rollPoolIntegrityStage(nextB, args);
    releaseA();
    await pA;
    releaseB();
    await pB;
    expect(seen).toEqual([false]);
    expect(game.settings.get(SYSTEM_ID, "autopanic")).toBe(true);
  });

  it("does not touch autopanic when the roll has a real actor", async () => {
    const seen = [];
    const next = vi.fn(async () => {
      seen.push(game.settings.get(SYSTEM_ID, "autopanic"));
      return "card";
    });
    await rollPoolIntegrityStage(next, rollArgs({ label: "Ranged Combat", r1Dice: 4, r2Dice: 6 }));
    expect(seen).toEqual([true]);
  });

  it("passes everything through untouched when the GM switched the feature off", async () => {
    features.enabled.mockReturnValue(false);
    const next = vi.fn(async () => "card");
    await rollPoolIntegrityStage(next, rollArgs({ r1Dice: -4, r2Dice: 3 }));
    expect(next).toHaveBeenCalledTimes(1);
    expect(next.mock.calls[0][0][4]).toBe(-4);
    expect(next.mock.calls[0][0][6]).toBe(3);
    expect(infoSpy).not.toHaveBeenCalled();
  });
});

describe("probeRollPoolIntegrity", () => {
  beforeEach(() => {
    installFoundryStub();
    game.alienrpg = { yze: { yzeRoll: fixtureSystem4113 } };
  });
  afterEach(() => {
    vi.restoreAllMocks();
    uninstallFoundryStub();
  });

  it("reports the defect as present on a shipped 4.1.13 system", () => {
    expect(probeRollPoolIntegrity()).toBe(true);
  });

  it("retires itself when the running system has both halves fixed", () => {
    game.alienrpg.yze.yzeRoll = fixtureBothFixed;
    expect(probeRollPoolIntegrity()).toBe(false);
  });

  it("falls back to the snapshot taken at init once rollBus has wrapped the target", () => {
    rollPoolIntegrityFeature.register();
    game.alienrpg.yze.yzeRoll = fixtureWrapperShell;
    expect(probeRollPoolIntegrity()).toBe(true);
  });

  it("assumes the defect is present when the source cannot be read at all", async () => {
    vi.resetModules();
    const fresh = await import("../scripts/features/roll-pool-integrity.mjs");
    delete game.alienrpg;
    expect(fresh.probeRollPoolIntegrity()).toBe(true);
  });
});

describe("rollPoolIntegrityFeature.register", () => {
  beforeEach(() => {
    installFoundryStub();
    game.alienrpg = { yze: { yzeRoll: fixtureSystem4113 } };
    vi.clearAllMocks();
  });
  afterEach(() => {
    vi.restoreAllMocks();
    uninstallFoundryStub();
  });

  it("registers the feature so the GM can switch it off", () => {
    rollPoolIntegrityFeature.register();
    expect(features.register).toHaveBeenCalledWith({
      id: "roll-pool-integrity", default: "full", gmOnly: false, requires: [], hint: "",
    });
  });

  it("registers the patch as MIXED with its version metadata", () => {
    rollPoolIntegrityFeature.register();
    const def = patches.register.mock.calls[0][0];
    expect(def.id).toBe("roll-pool-integrity");
    expect(def.type).toBe("MIXED");
    expect(def.target).toBe("game.alienrpg.yze.yzeRoll");
    expect(def.minSystem).toBe("4.1.13");
    expect(def.fixedIn).toBe(null);
    expect(def.probe()).toBe(true);
  });

  it("installs itself as a rollBus stage on yzeRoll when the kernel calls apply()", () => {
    rollPoolIntegrityFeature.register();
    patches.register.mock.calls[0][0].apply();
    expect(rollBus.addStage).toHaveBeenCalledTimes(1);
    const [target, stage] = rollBus.addStage.mock.calls[0];
    expect(target).toBe("yzeRoll");
    expect(stage.id).toBe("roll-pool-integrity");
    expect(stage.order).toBe(10);
    expect(stage.around).toBe(rollPoolIntegrityStage);
  });

  it("never registers a libWrapper wrapper of its own, because rollBus owns that target", () => {
    const registerSpy = vi.spyOn(libWrapper, "register");
    rollPoolIntegrityFeature.register();
    patches.register.mock.calls[0][0].apply();
    expect(registerSpy).not.toHaveBeenCalled();
  });

  it("registers both selftest entries with i18n keys as their labels", () => {
    rollPoolIntegrityFeature.register();
    const defs = selftest.register.mock.calls.map((c) => c[0]);
    expect(defs.map((d) => d.id)).toEqual(["roll-pool-integrity.probe", "roll-pool-integrity.exemptions"]);
    expect(defs.map((d) => d.label)).toEqual([
      "AEA.selftest.roll-pool-integrity.probe",
      "AEA.selftest.roll-pool-integrity.exemptions",
    ]);
  });
});
```

- [ ] **Step 12: 跑它，看它失败**

Run: `npx vitest run test/roll-pool-integrity.test.mjs`

Expected: FAIL —— `Failed to resolve import "../scripts/features/roll-pool-integrity.mjs" from "test/roll-pool-integrity.test.mjs"`；整个文件收集失败，52 条用例一条都没执行。

- [ ] **Step 13: 写副作用层**

新建 `scripts/features/roll-pool-integrity.mjs`。五点契约要求，逐条对着写：

1. **不许自己调 `libWrapper.register`。** `game.alienrpg.yze.yzeRoll` 这个目标由 rollBus 独占（同一模组对同一目标二次注册会抛错或静默失效）。我们经 `rollBus.addStage("yzeRoll", {id, order, around})` 排队。
2. §4 K7 的补丁 def 里 `target` / `type` **只是元数据**（供 `patches.status()` 展示），`apply()` 是**无参自装器**，自己负责装 —— 对本任务来说「装」就是排一个阶段。
3. §7 禁止 `game.actors.get()`；遗留裸 actor id 一律走 `resolver.actorById`。
4. §6 的特性模块形状固定为 `export const <camelId>Feature = { id, register(), install() }`。本特性 ready 阶段无事可做（阶段由 `patches.applyAll()` 排上），`install()` 是空实现但必须存在，因为 `main.mjs` 会对 `FEATURES` 数组里的每一项调它。
5. §7 要求执行前查 `features.enabled(id)`。查在阶段函数里而不是安装时，所以 GM 改档位立刻生效、不用重载世界。

```js
import { MID, SYSTEM_ID } from "../const.mjs";
import { features } from "../kernel/features.mjs";
import { patches } from "../kernel/patches.mjs";
import { resolver } from "../kernel/resolver.mjs";
import { rollBus } from "../kernel/rollbus.mjs";
import { selftest } from "../kernel/selftest.mjs";
import {
  pureFloorApplies,
  pureIsResourceRoll,
  pureNeedsAutoPanicSuppression,
  pureResolveRollPools,
  pureRollPoolIsBuggy,
  pureSourceIsRecognizable,
} from "./roll-pool-integrity.pure.mjs";

const FEATURE_ID = "roll-pool-integrity";
const TARGET = "game.alienrpg.yze.yzeRoll";
const STAGE_TARGET = "yzeRoll";
const STAGE_ORDER = 10;
const MIN_SYSTEM = "4.1.13";

/** Source text of the untouched system function; refreshed on every readable read. */
let systemSource = "";

function readSystemSource() {
  const live = String(game.alienrpg?.yze?.yzeRoll ?? "");
  if (!pureSourceIsRecognizable(live)) return systemSource;
  systemSource = live;
  return live;
}

/**
 * true means the defect is STILL present in the running system.
 *
 * Once rollBus has installed its libWrapper wrapper on this target, reading the
 * live property gives libWrapper's dispatch shell instead of the system body, so
 * we fall back to the snapshot taken at init. If we have no readable text at all
 * we answer true on purpose: both halves of this patch are no-ops against an
 * already correct system (a pool of 1 or more is never touched; a roll that has
 * no actor has nothing to panic), so failing towards "install" cannot double-fix.
 */
export function probeRollPoolIntegrity() {
  const source = readSystemSource();
  if (!pureSourceIsRecognizable(source)) {
    console.warn(`${MID} | ${FEATURE_ID}: cannot read ${TARGET} source, assuming the defect is present`);
    return true;
  }
  return pureRollPoolIsBuggy(source);
}

// --- auto-panic suppression ------------------------------------------------
// The system reads game.settings.get("alienrpg", "autopanic") at :198 and then
// dereferences game.actors.get(actorid) at :201-204 with no guard. We cannot
// give it an actor that does not exist, so we make that one setting read as
// false for the duration of the call. The shim is an own property on the single
// client-side settings object; nothing is written to the world. A depth counter
// (not a save/restore pair) keeps two overlapping rolls from tearing each
// other's shim down.

let suppressionDepth = 0;
let originalSettingsGet = null;
let settingsGetWasOwn = false;

function pushAutoPanicSuppression() {
  if (suppressionDepth++ > 0) return;
  settingsGetWasOwn = Object.hasOwn(game.settings, "get");
  originalSettingsGet = game.settings.get;
  game.settings.get = function (namespace, key) {
    if (namespace === SYSTEM_ID && key === "autopanic") return false;
    return originalSettingsGet.call(this, namespace, key);
  };
}

function popAutoPanicSuppression() {
  if (suppressionDepth > 0) suppressionDepth--;
  if (suppressionDepth > 0) return;
  if (settingsGetWasOwn) game.settings.get = originalSettingsGet;
  else delete game.settings.get;
  originalSettingsGet = null;
}

/** Read the real world value even while our own shim is installed. */
function readAutoPanic() {
  const get = originalSettingsGet ?? game.settings.get;
  return get.call(game.settings, SYSTEM_ID, "autopanic") === true;
}

/**
 * rollBus stage, not a libWrapper wrapper: rollBus owns this target and hands
 * every queued stage the same shape.
 *
 *   around(next, args, thisArg)
 *     · args is the 12-element positional argument array of YZEDiceRoller.mjs:31
 *     · next(args) must be called exactly once, with whatever array we want the
 *       system to actually receive
 *
 * The third parameter (thisArg) is unused here: yzeRoll is a static method and
 * reads nothing off `this`; rollBus applies the binding for us.
 */
export async function rollPoolIntegrityStage(next, args) {
  if (!features.enabled(FEATURE_ID)) return next(args);

  const [actortype, blind, reRoll, label, r1Dice, col1, r2Dice, col2, actorid, itemid, tactorid, moddata] = args;

  // Forward-localize the very key the system compares against at :154. This is a
  // branch predicate, not a document lookup by display name.
  const radiationLabel = game.i18n.localize("ALIENRPG.RadiationReduced");
  const isRadiationReducedLabel =
    typeof radiationLabel === "string" && radiationLabel.length > 0 && label === radiationLabel;

  const resourceRoll = pureIsResourceRoll({ actortype, isRadiationReducedLabel });
  const floorApplies = pureFloorApplies({ actortype, isRadiationReducedLabel, reRoll });
  const pools = pureResolveRollPools({ base: r1Dice, stress: r2Dice, floorApplies });

  if (pools.change === "floored") {
    ui.notifications.info(game.i18n.format("AEA.rollPool.floored", { from: Number(r1Dice) }));
  } else if (pools.change === "clamped") {
    console.warn(`${MID} | ${FEATURE_ID}: base pool ${r1Dice} clamped to 0 so it cannot eat stress dice`);
  }

  const nextArgs = [actortype, blind, reRoll, label, pools.base, col1, pools.stress, col2, actorid, itemid, tactorid, moddata];

  const suppress = pureNeedsAutoPanicSuppression({
    stress: pools.stress,
    autoPanic: readAutoPanic(),
    resourceRoll,
    actorResolved: Boolean(actorid && resolver.actorById(actorid, { warn: false })),
  });
  if (!suppress) return next(nextArgs);

  console.warn(`${MID} | ${FEATURE_ID}: ${TARGET} called without a resolvable actor, auto-panic suppressed for this call`);
  pushAutoPanicSuppression();
  try {
    return await next(nextArgs);
  } finally {
    popAutoPanicSuppression();
  }
}

export const rollPoolIntegrityFeature = {
  id: FEATURE_ID,

  register() {
    // Snapshot the system body now, at init, while nothing has wrapped it yet.
    readSystemSource();

    // The visible name and hint come from AEA.feature.<id>.name / .hint.
    features.register({ id: FEATURE_ID, default: "full", gmOnly: false, requires: [], hint: "" });

    patches.register({
      id: FEATURE_ID,
      type: "MIXED",
      target: TARGET, // metadata for patches.status() only
      minSystem: MIN_SYSTEM,
      fixedIn: null,
      probe: probeRollPoolIntegrity,
      // rollBus owns the libWrapper registration for this target; we queue a
      // stage instead. applyAll() runs at the ready.patches anchor, before
      // rollBus.install() at ready.rollbus, and addStage only enqueues, so the
      // order is correct.
      apply: () => rollBus.addStage(STAGE_TARGET, {
        id: FEATURE_ID,
        order: STAGE_ORDER,
        around: rollPoolIntegrityStage,
      }),
    });

    // Both selftest labels are i18n KEYS, not display text: register() runs at
    // init, which is earlier than i18nInit, so no language file is loaded yet and
    // localize() would only echo the key back. The runner localizes them.
    selftest.register({
      id: `${FEATURE_ID}.probe`,
      label: `AEA.selftest.${FEATURE_ID}.probe`,
      run() {
        const defect = probeRollPoolIntegrity();
        const row = patches.status().find((p) => p.id === FEATURE_ID) ?? null;
        const applied = Boolean(row?.applied);
        return {
          ok: defect === applied,
          detail: `system ${game.system.version} · defect=${defect} · applied=${applied}`
            + ` · reason=${row?.reason ?? "not-registered"} · fixedIn=${row?.fixedIn ?? "none"}`,
        };
      },
    });

    selftest.register({
      id: `${FEATURE_ID}.exemptions`,
      label: `AEA.selftest.${FEATURE_ID}.exemptions`,
      // vitest can only exercise the English keys. This entry runs the same six
      // cases against the live locale, so a Babele-renamed world shows up here.
      run() {
        const radiationLabel = game.i18n.localize("ALIENRPG.RadiationReduced");
        const rows = [
          ["character", "Heavy Machinery", false, -4, 3, 1, 3],
          ["character", "Heavy Machinery", false, 0, 0, 0, 0],
          ["supply", "Air", false, 0, 5, 0, 5],
          ["character", radiationLabel, false, 0, 3, 0, 3],
          ["character", "Ranged Combat", "push", 0, 4, 0, 4],
          [false, "GM", true, -3, 5, 0, 5],
        ];
        const bad = [];
        for (const [actortype, label, reRoll, base, stress, wantBase, wantStress] of rows) {
          const isRadiationReducedLabel = label === radiationLabel;
          const got = pureResolveRollPools({
            base,
            stress,
            floorApplies: pureFloorApplies({ actortype, isRadiationReducedLabel, reRoll }),
          });
          if (got.base !== wantBase || got.stress !== wantStress) {
            bad.push(`${actortype}/${label}: ${base}b${stress}s -> ${got.base}b${got.stress}s`);
          }
        }
        return {
          ok: bad.length === 0,
          detail: bad.length ? bad.join("; ") : `6 rows ok in the live locale (radiation label = "${radiationLabel}")`,
        };
      },
    });
  },

  // ready stage: nothing to do. patches.applyAll() queues the stage, and this
  // feature subscribes to no hook and registers no card action.
  install() {},
};
```

- [ ] **Step 14: 跑它，看它通过**

Run: `npx vitest run test/roll-pool-integrity.test.mjs`

Expected: PASS — `Tests  52 passed (52)`。

- [ ] **Step 15: 加 i18n 键**

契约 §7：面向用户的字符串一律 `game.i18n.localize()`；模组自有键前缀 `AEA.`、**嵌套结构**、顶层只有 `AEA` 一个键。§4 K5：特性显示名与说明固定为 `AEA.feature.<exact feature id>.name` / `.hint`，`<exact feature id>` 逐字等于 `features.register({id})` 的 id（本特性是 `roll-pool-integrity`，带连字符，作为 JSON 键要加引号，不要改写成驼峰）。两条自检的 `label` 同理，是键不是文本。

把下面的片段**合并进** `lang/en.json` 已有的 `AEA` 对象（不要替换整个对象，也不要新增第二个顶层键）：

```json
{
  "AEA": {
    "feature": {
      "roll-pool-integrity": {
        "name": "Dice pool integrity",
        "hint": "Keep the base dice pool at one die minimum and never let a base-dice penalty eat stress dice. Also stops an actor-less GM roll from vanishing when a stress die shows a 1. Supply, ammo, radiation and pushed rolls are exempt."
      }
    },
    "selftest": {
      "roll-pool-integrity": {
        "probe": "Dice pool integrity: is the system defect still there?",
        "exemptions": "Dice pool integrity: exemptions in the live locale"
      }
    },
    "rollPool": {
      "floored": "Base dice raised to 1 (modifiers had taken it to {from}). Stress dice unchanged."
    }
  }
}
```

同样合并进 `lang/cn.json`：

```json
{
  "AEA": {
    "feature": {
      "roll-pool-integrity": {
        "name": "骰池完整性",
        "hint": "基础骰池永远不低于一颗，基础骰的负修正不再从压力骰里扣。同时让没有角色的 GM 掷骰在压力骰出 1 时不再整张卡消失。补给、弹药、减辐射与推骰不受影响。"
      }
    },
    "selftest": {
      "roll-pool-integrity": {
        "probe": "骰池完整性：系统缺陷是否仍在",
        "exemptions": "骰池完整性：当前语言环境下的豁免判据"
      }
    },
    "rollPool": {
      "floored": "基础骰已提升到 1 颗（修正原本把它压到 {from}）。压力骰不受影响。"
    }
  }
}
```

- [ ] **Step 16: 跑全套，确认语言包护栏没被打红**

Run: `npm test`

Expected: PASS — 全仓库测试全绿，其中包括语言包结构护栏（断言 `lang/*.json` 顶层只有 `AEA` 一个键、且 en 与 cn 的键集合一致）。若护栏红了，是上一步的 JSON 合并把片段贴成了第二个顶层对象，或只往其中一个语言文件里加了 `selftest` 段，回去改成两边都合并进同一个 `AEA` 里。

- [ ] **Step 17: 在 main.mjs 的两个锚点上接线**

`scripts/main.mjs` 里有十个 `/* AEA-ANCHOR: … */` 注释锚点，本任务**只碰其中两个**，一共加两行、删零行。四个生命周期钩子（`init` / `i18nInit` / `diceSoNiceReady` / `ready`）的函数体**一行都不许动**：`init` 里已经有 `for (const f of FEATURES) safely(..., () => f.register())`，`ready` 里已经有 `for (const f of FEATURES) await safely(..., () => f.install())`，本特性的 `register()` 与 `install()` **由这两个循环调用**。**严禁**在任何生命周期锚点里再直调一次 `rollPoolIntegrityFeature.register()` 或 `.install()` —— 那会让 `features.register` / `patches.register` / `selftest.register` 重复登记，`apply()` 也会被排两遍阶段。

顺带一条你可以放心依赖的顺序事实：`init` 里那两个 `for..of` 循环排在 `features.registerSettings()` **之前**，所以我们在 `register()` 里调的 `features.register(def)` 一定赶得上设置项与配置菜单的生成。

第一处，找到文件顶部 import 区里的这一行：

```js
/* AEA-ANCHOR: imports */
```

在**它的下一行**加：

```js
import { rollPoolIntegrityFeature } from "./features/roll-pool-integrity.mjs";
```

第二处，找到 `FEATURES` 数组里的这一行：

```js
  /* AEA-ANCHOR: features */
```

在**它的下一行**加：

```js
  rollPoolIntegrityFeature,
```

Run: `grep -n "rollPoolIntegrityFeature" scripts/main.mjs; git diff --numstat scripts/main.mjs`

Expected: grep 正好两行命中 —— 一行 import（在 `/* AEA-ANCHOR: imports */` 之后）、一行数组元素（在 `/* AEA-ANCHOR: features */` 之后，且位于 `const FEATURES = [` 与 `];` 之间）；`git diff --numstat` 输出 `2	0	scripts/main.mjs`（两行新增、零行删除）。若删除数不是 0，说明你改到了不该改的行，`git checkout -- scripts/main.mjs` 重来。

- [ ] **Step 18: 再跑全套**

Run: `npm test`

Expected: PASS — 全绿。若 `main.mjs` 的生命周期用例报「注册顺序」或「FEATURES 成员」相关的失败，检查那一行是否确实在 `[` 与 `]` 之间、且没有多余的直调行。

- [ ] **Step 19: 提交副作用层与接线**

```bash
git add scripts/features/roll-pool-integrity.mjs scripts/main.mjs lang/en.json lang/cn.json test/roll-pool-integrity.test.mjs
git commit -m "feat(roll-pool-integrity): 以 rollBus 阶段钳制基础池并救回无角色的 GM 掷骰" -m "两处缺陷：
1) YZEDiceRoller.mjs:139 把基础骰的负缺口塞进压力池，压到 <1 时 :141 直接拒绝掷骰；
2) macros/gmRollYZEDiceMacro.js:34 只传 8 个参数，actorid 为 undefined，
   自动恐慌分支 :201 取到 undefined，:204 裸解引用抛 TypeError，
   早于 :416 的 ChatMessage.create，整张卡消失。
介入方式改为 rollBus.addStage(\"yzeRoll\", ...)：yzeRoll 这个 libWrapper 目标由 rollBus 独占，
本特性若自行 register 会与之撞车而静默失效。
第二处用带深度计数的 game.settings.get 遮蔽（finally 必还原、两次并发不会互相拆台）
把 autopanic 对这一次调用隐去，不改世界设置。
下限只对角色检定生效：补给/弹药/减辐射按 actortype 与正向本地化的 RadiationReduced 标签豁免，
推骰按 reRoll === push/mPush 豁免（全 6 时重掷 0 颗是合法的），
两个裸骰宏按 actortype 是布尔 false 豁免，但负基础池仍夹到 0 以堵住偷窃分支。
probe 源码快照在 init 取，rollBus 装壳后回落到快照；读不到时保守返回 true（两半对已修系统都是空操作）。
两条自检的 label 存 i18n 键而非文本：登记发生在 init，早于 i18nInit。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 20: MANUAL VERIFICATION —— 阶段真的排上了**

桩测得到的只是「`apply()` 调了 `rollBus.addStage`」；真实的 libWrapper 链、rollBus 的阶段顺序、以及「排队早于安装」这件事只有在 Foundry 里才验得了。

在本机 Foundry 打开装有 `alienrpg` 4.1.13 的世界，启用 `alien-evolved-automation` 与 `lib-wrapper`。F12 控制台执行：

```js
game.modules.get("alien-evolved-automation").api.patches.status().find((p) => p.id === "roll-pool-integrity");
```

Expected: 返回 `{id: "roll-pool-integrity", type: "MIXED", target: "game.alienrpg.yze.yzeRoll", applied: true, reason: "ok", fixedIn: null}`。

再执行：

```js
libWrapper.get_module_wrappers?.("alien-evolved-automation")?.map((w) => w.name);
```

Expected: 列表里 `game.alienrpg.yze.yzeRoll` **只出现一次**（那一次是 rollBus 的）。若出现两次，或控制台里有 `has already been registered by alien-evolved-automation`，说明本特性错误地自行注册了 libWrapper，回到 Step 13 把 `apply()` 改回 `rollBus.addStage`。（`get_module_wrappers` 是 lib-wrapper 的调试接口；若你的 lib-wrapper 版本没有它，改用「设置 → 模组设置 → lib-wrapper → 显示活动包裹」面板看同一份清单。）

- [ ] **Step 21: MANUAL VERIFICATION —— 基础骰下限**

1. 打开任意 character 角色卡，把 Stress Level 调到 3。
2. 右键点一个技能（例如 Heavy Machinery），在弹出的修正对话框里填 `-9`，确认。
3. **期望**：出现聊天卡；黑色骰 **1 颗**、黄色骰仍 **3 颗**；右下角出现提示条「基础骰已提升到 1 颗（修正原本把它压到 -9）。压力骰不受影响。」
4. **今天的行为（对照组）**：到「配置设置 → 模组设置」里，把本模组特性面板中的「骰池完整性」（英文界面是 Dice pool integrity）切到 Off，重做第 2 步 —— 会看到黑 0 黄 0、或者根本没有卡只弹一句 "No Dice to Roll"。看完把它切回 Full（切换立即生效，不用重载世界）。

- [ ] **Step 22: MANUAL VERIFICATION —— 补给掷骰与推骰未被误伤**

1. 同一角色卡切到 Consumables，点 Air 的补给掷骰按钮。
   **期望**：卡上只有黄色骰，颗数等于剩余值（>6 截到 6）；**没有**黑色骰、**没有**「基础骰已提升」提示。若出现黑色骰，立刻停下来查传进来的 `actortype`。
2. 把角色的 Stress Level 留在 3，右键一个技能填 `-9`（即上一步那种被钳到 1 颗黑骰的掷骰），重复掷，直到那颗黑骰**掷出 6**（一颗骰约六次一次）。
3. 在这张卡上点 **Push**。
   **期望**：推骰卡上**没有黑色骰**（上一轮唯一的基础骰已经是 6，没有可重掷的），黄色骰是 3+1=4 颗；**没有**「基础骰已提升」提示。
4. 若推骰卡上冒出 1 颗黑骰，说明推骰豁免没生效 —— 去查 `pureIsPushedRoll` 拿到的 `reRoll` 到底是不是字符串 `"push"`。

- [ ] **Step 23: MANUAL VERIFICATION —— GM 掷骰不再消失**

1. 世界设置里确认 `Alien RPG` 的 "Auto Panic" 开着（默认开）。
2. 打开宏栏里的世界宏 **"Alien - GM Dice Roller"**（系统冒险包导入的 `macros/gmRollYZEDiceMacro.js` 冻结副本）。
3. 基础骰填 `1`、压力骰填 `6`，点 Roll，重复 10 次。
4. **期望**：10 次都出卡；出现黄色 1 的那几次卡上有红色闪烁的 "roll stress" 提示，**但没有恐慌卡、没有报错**；F12 控制台里有 `alien-evolved-automation | roll-pool-integrity: game.alienrpg.yze.yzeRoll called without a resolvable actor, auto-panic suppressed for this call`。
5. 基础骰填 `0`、压力骰填 `6` 再掷一次：**期望**卡上就是 0 黑 6 黄（裸骰工具按 GM 填的数掷，不该被提到 1 颗），且**没有**提示条。
6. 十次掷完后回到「配置设置 → 模组设置 → Alien RPG → Auto Panic」，**期望**它仍然是**开着**的（我们只在调用期间遮蔽了这一次读取，从不写世界设置）。
7. **今天的行为**：只要一颗黄色骰出 1，那一次完全没有聊天卡，F12 里是 `TypeError: Cannot read properties of undefined (reading 'getRollData')`。

- [ ] **Step 24: MANUAL VERIFICATION —— 记录层看到的是钳制后的池子**

在 Step 21 那张「黑 1 黄 3」的聊天卡上右键 → 检查元素拿到消息 id（或直接用最后一条消息），F12 执行：

```js
const m = game.messages.contents.at(-1);
game.modules.get("alien-evolved-automation").api.rollBus.recordOf(m)?.pools;
```

Expected: `{base: 1, stress: 3}` —— 内核记录的是**真正掷出去**的池子（读系统的 `game.alienrpg.rollArr`），而不是钳制前的 `-9`。若这里显示 `{base: -9, ...}`，说明阶段顺序反了（我们的阶段跑到了记录层外面），把 `STAGE_ORDER` 报给主控，不要自行改内核。

- [ ] **Step 25: 跑模组自检**

在 F12 控制台执行：

```js
await game.modules.get("alien-evolved-automation").api.selftest.runAll();
```

Expected: 返回的数组里 `roll-pool-integrity.probe` 与 `roll-pool-integrity.exemptions` 两条都是 `ok: true`；前者 detail 形如 `system 4.1.13 · defect=true · applied=true · reason=ok · fixedIn=none`，后者形如 `6 rows ok in the live locale (radiation label = "…")` —— 中文世界里括号里应当是中文的「辐射降低」类译名，若那里显示的是英文原文或键名本身，说明系统语言包没加载，先解决那个再继续。两条的 `label` 字段应当显示为**已本地化的中文**（如「骰池完整性：系统缺陷是否仍在」）；若显示的是 `AEA.selftest.roll-pool-integrity.probe` 这样的裸键，说明 Step 15 的语言包键没合并进去。

- [ ] **Step 26: 记录本机冒烟结果**

```bash
git commit --allow-empty -m "test(roll-pool-integrity): 本机冒烟六项 + 自检两条通过" -m "libWrapper 清单里 yzeRoll 只有 rollBus 一处包裹，本特性以阶段排队接入；
技能 -9 修正 → 1 黑 3 黄并提示，recordOf().pools 记的是钳制后的 {base:1,stress:3}；
补给掷骰仍是纯黄色池；上一轮黑骰出 6 后推骰不再凭空多一颗黑骰；
GM 掷骰宏 1 基础/6 压力连开十次都出卡，压力出 1 时只提示不恐慌，0 基础照旧掷 0 黑，
掷完 Auto Panic 世界设置仍为开；
patches.status() 报 applied=true/reason=ok，两条自检 ok 且 label 显示为中文。
按项目纪律，本机只是冒烟机，发布权威在 VPS。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```
