> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 7 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 7: K1 副作用层 —— RollBus 四入口包裹（`kernel/rollbus.mjs`）与 main.mjs 五处接线

**Files:**
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/kernel/rollbus.mjs`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/test/rollbus.test.mjs`
- Modify: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/lang/en.json`、`.../lang/cn.json` —— 各加三个自检条目名（合并进已有的 `AEA` 对象，不覆盖同级已有键）
- Modify: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/main.mjs` —— 该文件已由更早的任务建好，里面有十个逐字锚点注释。本任务只碰其中三个锚点与 `api` 字面量里属于自己的那一个槽：`/* AEA-ANCHOR: imports */`、`api` 里的 `rollBus: null`、`/* AEA-ANCHOR: i18nInit */`、`/* AEA-ANCHOR: ready.rollbus */`。**一律按锚点文本定位，不用行号；不碰任何别的锚点、不改别人的行、不增删 `api` 的键**
- **不得修改** `test/stubs/foundry.mjs` —— 共享测试桩的行为由契约 §0.3 定死并**由建桩的那个任务独占**，本任务只读不改（下面 Step 7 给出桩不达标时的处理方式）

**Interfaces:**

- Consumes:
  - `scripts/const.mjs` → `MID = "alien-evolved-automation"`、`FLAG_ROLL = "roll"`、`HOOK_ROLL_RESOLVED = "aea.rollResolved"`
  - `scripts/kernel/record.mjs`（纯层，不碰任何 Foundry 全局）→
    - `buildRollRecord(input)`，`input = {args, rollArr, refs, ctx, userId, worldTime, now, id, labelIndex}` → `RollRecord`
    - `buildLabelIndex(localize, keys)` → `{已本地化文本: i18n 键}`，同一段文本被两个键共用时该文本映射到 `null`
    - `pureReverseLabelKey(label, index)` → `string|null`
    - `LABEL_KEYS` —— **反查标签用的 i18n 键表，属主是 `record.mjs`，全仓只此一份**（契约 §4 K1 [v3.1]）。本任务 **import 它，绝不定义、绝不导出同名符号**
    - 本任务对 `buildRollRecord` 的六条具体期望（跑红时先看是不是这里对不上，**不要改 rollbus 去迁就**）：`args.actortype === "supply"` → `kind === "supply"`；`ctx.attr` 非空 → `"attribute"`；`ctx.itemUuid` 非空 → `"weapon"`；两者都空 → `"skill"`；`successes` = `rollArr.r1Six + rollArr.r2Six`；`banes` = `rollArr.r2One`（压力骰的 1；基础骰的 1 不是 bane）。`pools.base`/`pools.stress` 取 `rollArr.r1Dice`/`rollArr.r2Dice`，**不取 `args`**
  - `scripts/kernel/resolver.mjs` → `resolver.fromSpeaker(speaker) -> Actor|null`、`resolver.refs(actor, token) -> {actorUuid, tokenUuid}`、`resolver.soleToken(actor) -> Token|null`（`getActiveTokens()` 恰好一个时返回它，否则 `null`，不猜）
  - `scripts/kernel/selftest.mjs` → `selftest.register({id, label, run})`，`label` 存 **i18n 键**、不存译文；`run()` 返回 `{ok:boolean, detail:string}`；`selftest.runAll()` 返回 `[{id, label, ok, detail}]`
  - `test/stubs/foundry.mjs` → `installFoundryStub(options) -> ctx`、`uninstallFoundryStub()`（契约 §0.3 定死的共享桩）。本任务依赖它的四条行为：`Hooks.callAll` 真派发给 `Hooks.on` 的回调；`ChatMessage.create` 把文档推进 `ctx.messages` 并**先同步派发 `preCreateChatMessage` 再派发 `createChatMessage`**，文档实例带可用的 `updateSource()`；`libWrapper.register` 记进 `ctx.wrappers`；`ui.notifications.*` 记进 `ctx.notifications`。用到的 ctx 成员：`ctx.messages`、`ctx.wrappers`、`ctx.notifications`、`ctx.userId`
  - 全局 `libWrapper`（module.json 里的硬依赖，装在 `Data/modules/lib-wrapper`）
- Produces:
  - `export const rollBus = { install(), recordOf(message), depth(), context(), setLabelIndex(index), addStage(target, {id, order, around}) }` —— 契约 §4 K1 定死的**六个**成员，一个不多一个不少
  - `export function pureRollMarker(actorid, itemid) -> string`
  - `export function pureClaimIndex(stack, content) -> number`
  - `export function pureBindContext(ctxStack, actorid) -> number`
  - **本模组四个 libWrapper 目标的独占属主**：`game.alienrpg.yze.yzeRoll`、`CONFIG.Actor.documentClass.prototype.abilityRoll`、`CONFIG.Item.documentClass.prototype.roll`、`CONFIG.Actor.documentClass.prototype.pushRoll`。别的任何特性／修复要介入这四个目标，一律经 `rollBus.addStage()` 排队 —— lib-wrapper 1.13.5 对同一目标的重复注册会抛 `A wrapper for '<target>' (ID=<n>) has already been registered by <module>.`，自己去 register 的那一方会**静默变成空操作**
  - 对外钩子 `aea.rollResolved`，以 `Hooks.callAll(HOOK_ROLL_RESOLVED, record, message)` 在每个客户端的 `createChatMessage` 里发出
  - 落盘位置 `message.flags["alien-evolved-automation"].roll`
  - 三条 `selftest` 条目：`rollbus.wrappers`、`rollbus.labelIndex`、`rollbus.attribution`
  - main.mjs 的两条 import、`api.rollBus` 槽、i18nInit 一行、ready 一行
  - 「同一基础 actor 的两个非链接 token 各掷一次武器 → 两个不同的 `tokenUuid`」这条硬断言在 Step 9 的 vitest 里（**不要**再登记名为 `resolver-token-identity` 的自检条目，那个 id 归 K4 resolver 那个任务，重复 id 会撞车）
  - **本任务不产出 `LABEL_KEYS`**（属主是 `record.mjs`），**不产出 `cards.init()` / `registry.resolveAll()` / `patches.applyAll()` 的接线**（各有自己的属主任务与自己的 ready 子锚点）

**背景（Foundry 概念 + 六条载重约束；写代码前把下面引用的源码行全部打开看过，行号已按 alienrpg 4.1.13 逐条复核）**

*Foundry 名词，三十秒版：*
- **Hook**：Foundry 的全局同步事件总线。`Hooks.on("x", fn)` 订阅，`Hooks.callAll("x", …)` 广播。两个关键钩子：`preCreateChatMessage(doc, data, options, userId)` 在聊天消息**写库之前**、只在**发起创建的那个客户端**上同步触发，此刻 `doc.updateSource({...})` 写进去的东西会被算进即将落库的数据，因而对所有客户端可见；`createChatMessage(message, options, userId)` 在**每个**客户端上、消息已存在之后触发。
- **libWrapper**：第三方模组，用来安全地包裹别人的函数。`libWrapper.register(moduleId, "game.a.b.c", fn, "WRAPPER")` 把 `globalThis.game.a.b.c` 换掉，调用时变成 `fn(wrapped, ...args)` —— 第一个参数是原函数（已绑好 `this`），`this` 保持不变。**WRAPPER** 的语义承诺是「我一定调用 wrapped 并原样返回它的结果」，所以多个模组能叠加；**MIXED/OVERRIDE** 不作此承诺。libWrapper 按类型排序：同一目标上，所有 WRAPPER 一律跑在 MIXED/OVERRIDE 的**外层**，**与注册先后无关**。目标路径可以是任意从 `globalThis` 出发的点号路径，包括 `CONFIG.Actor.documentClass.prototype.<method>`。同一目标被同一模组注册第二次会抛错。
- **uuid**：Foundry 文档的全局唯一地址串，例如 `Actor.abc123`。如果角色是场景上某个**非链接 token**（unlinked token —— 那个 token 持有自己的一份角色副本，而不是共用世界里的基础角色），地址是 `Scene.s1.Token.t1.Actor.abc123`；注意这个合成 actor 的 `.id` **和基础 actor 的 id 一模一样**，只有 uuid 不同。系统到处存 `actor.id` 再 `game.actors.get()`，那会把 token 信息整个丢掉 —— 场上三只同名 Drone 会被当成一只。所以记录里存 uuid，不存 id。
- **钩子归属**（契约 §0.2）：生命周期钩子（`init`/`i18nInit`/`setup`/`ready`/`diceSoNiceReady`）只在 `main.mjs` 挂；**领域钩子由拥有它的内核模块自己挂**，`preCreateChatMessage` 与 `createChatMessage` 属于本模块，只在 `install()` 里各挂一次。特性不许自己挂钩子。

*为什么要包四个入口（契约 §4 K1 的表）：* 系统全程不发任何掷骰钩子；`game.alienrpg.yze.yzeRoll` 是汇点，22 个调用点全部零解构地写作 `yze.yzeRoll(...)`，且各处 `import { yze }` 拿到的是同一个类对象（`alienrpg.mjs:74-79` 把它挂进 `game.alienrpg`），所以换掉这一个静态属性就全覆盖。但汇点**丢掉了下游必需的三样东西**，它们只在上游函数的入参里存在：

| 包裹目标（libWrapper 路径逐字） | 真实签名（源码行） | 采集什么 |
|---|---|---|
| `game.alienrpg.yze.yzeRoll` | `YZEDiceRoller.mjs:31` 12 个位置参数 | 汇点：压栈、armed 标志、绑定上下文帧、内层落定后同步复查 `rollArr` |
| `CONFIG.Actor.documentClass.prototype.abilityRoll` | `actor.mjs:187` `abilityRoll(actor, dataset, rollMod)` | `ctx.attr = args[1].attr`、`ctx.dataset`、`ctx.actor = args[0]`、`ctx.token` |
| `CONFIG.Item.documentClass.prototype.roll` | `item.mjs:39` `roll(right, dataset)` | `ctx.itemUuid = this.uuid`、`ctx.dataset`、`ctx.actor = this.actor`、`ctx.token` |
| `CONFIG.Actor.documentClass.prototype.pushRoll` | `actor.mjs:1302` `pushRoll(actor, reRoll, hostile, blind, message)` | `ctx.parentRollId`、`ctx.pushCount`、`ctx.inheritedRefs`（actor/token 从父记录的 `actorUuid`/`tokenUuid` 继承） |

四个方法都确实存在（`CONFIG.Actor.documentClass = alienrpgActor` 在 `alienrpg.mjs:123`，`CONFIG.Item.documentClass = alienrpgItem` 在 `:138`），四个都是 `async`，所以包装器返回 Promise 是忠实的。

*六条载重约束：*

1. **不能「await 原函数之后再盖 flag」。** `yzeRoll` 在 `:416` 自己 `await ChatMessage.create(chatData)`，`:417` 直接 `return`（返回 `undefined`）。包装器恢复执行时消息早就写完了。→ 唯一正确的时机是 `preCreateChatMessage`；那一刻内嵌的 `buildChat` 已经跑完，`game.alienrpg.rollArr` 数据完整。写入**只能**用 `document.updateSource({[\`flags.${MID}.${FLAG_ROLL}\`]: record})`（契约 §3 [v3.1]）：钩子第一个参数是尚未落库的文档实例，直接给 `data` 赋值不会进库，`document.flags[...] = x` 也不会。**禁止**用 `message.update()` 回填（多一次写库、卡片会闪一下）。
2. **上游三个入口都不 await 汇点。** `actor.mjs:305-315`（abilityRoll → 只传 9 个参数）、`item.mjs:114`／`:130`／`:175`… 全是裸调 `yze.yzeRoll(...)`，**没有 await**；只有 `actor.mjs:1315`（pushRoll）是 `await yze.yzeRoll(...)`。后果：abilityRoll／Item#roll 的上下文帧会在卡片建出来**之前**就出栈。所以绑定必须在 `yzeRoll` 被调用的**那一瞬间同步做完**，并且汇点帧要**持有那个 ctx 对象的引用**；等到 `preCreateChatMessage` 再去查 `ctxStack` 一定是空的。
3. **两个提前返回不建任何消息。** `:120` `if (!r1Dice && !r2Dice)` → `:121` `return ui.notifications.warn(game.i18n.localize("ALIENRPG.NoAttribute"))`；`:138` 的 `else` 分支里 `if (r1Dice < 0) { r2Dice = r2Dice + r1Dice; if (r2Dice < 1)` → `:141` `return ui.notifications.warn(game.i18n.localize("ALIENRPG.NoDice"))`。包装器必须容忍「这次调用没产生消息」，把栈帧干净地弹掉。
4. **重入：恐慌卡先于触发卡创建。** `:198-242`，压力骰出 1 且开了 `autopanic` 时，`:208` 执行 `myActor.rollResolve(myActor, dataset)` —— **没有 await**；`:229` 的 `rollPanic`、`:218`/`:239` 的 `rollAbility` 同理。这些函数自己 `new Roll()` 后 `ChatMessage.create` 出一张恐慌卡，而这一切发生在外层 `:416` 之前。所以「最近一条消息就是我的」必然错。归属只能靠调用栈 + 内容指纹。指纹是现成的，`:62` 那行：
   ```js
   let chatMessage = `<div class="chatBG" data-item-id="` + itemid + `" data-actor-id="` + actorid + `">`
   ```
   全系统只有这一处产出 `data-item-id="…" data-actor-id="…"` 这个形状，而恐慌卡起头是 `<h2 class="alienchatred ctooltip">`。于是「内容前缀是否等于我这一帧的 marker」就是精确归属。
5. **快照必须同步。** `game.alienrpg.rollArr`（`alienrpg.mjs:95-105`）是唯一的全局可变对象，`:107-114` 在每次掷骰开头逐字段清零。`preCreateChatMessage` 的处理函数里从读 `rollArr` 到用完为止**不许出现任何 `await`**。契约还要求「内层返回后同步快照」：本任务的落地是认领时取一份、内层落定后再同步取一份比对，两份对不上就说明**同一个客户端**上两次掷骰交错着把 `rollArr` 清了（`rollArr` 是客户端本地全局，两个玩家各自有一份，不会互相干扰），计数并告警，`rollbus.attribution` 自检读这个计数。
6. **`preCreateChatMessage` 里抛异常会打断系统自己的聊天卡。** 整个处理函数体（**含第一句的 `rollArr` 快照**）必须包在一个 try/catch 里，出错只记日志。

*上下文与汇点的绑定规则：* 先按 actor id 匹配、匹配不上取栈顶。理由是两处「帧里的 actor 不是卡上的 actor」的真实路径：载具／飞船的 `abilityRoll` 帧里是载具，而 `yzeRoll` 的第 9 参是驾驶员（`actor.mjs:235` `actorId = dataset.actorid`）；飞船炮位的 `Item#roll` 帧里是飞船，第 9 参是开火的船员（`item.mjs:399` `actorid = fCrew[shooter].firerID`，而 `:55` 的默认值是 `this.actor.id`）。这两种情况下按 id 匹配会落空，取栈顶正好是那一帧；而记录里的 `actorUuid` 会走 speaker 回退路径解析出**船员**而不是飞船 —— 这正是我们要的。

*`tokenUuid` 的诚实边界（契约 §4 K1）：* 卡是 `ChatMessage.getSpeaker({actor: actorid})` 建的（`YZEDiceRoller.mjs:398-401`），token 在消息存在之前就被丢掉了。所以 `tokenUuid` 只在三种情况下非空：掷骰源自 token 的合成 actor（`actor.isToken && actor.token`）、actor 在当前场景恰好只有一个活动 token（`resolver.soleToken`）、推骰从父记录继承。**其余情况一律如实记 `null`，绝不用 `game.actors.get()` 兜底伪造。**

*为什么还要 `addStage`（契约 §4 K1 [v3.1]）：* 一期 b 里有修复需要改 `yzeRoll` 的入参（例如把补给／辐射的压力骰钳到 6）。如果那条修复自己去 `libWrapper.register("…yzeRoll", …)`，lib-wrapper 会因为同一目标已被本模组注册而抛错，那条修复**静默变成空操作**。所以本模块开一个排队口：`rollBus.addStage(target, {id, order, around})`，`around(next, args, thisArg)` 恰好调用一次 `next(args)`，可以改 `args`、可以改返回值；`order` 小的先执行（更靠外），同 `order` 按注册顺序。阶段队列在**调用时**才读，所以在 `install()` 之前登记（修复的 `apply()` 跑在 ready 段更靠前的位置）完全有效。

---

- [ ] **Step 1: 写下三个纯函数的失败测试**

新建 `test/rollbus.test.mjs`，写入：

```js
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { pureRollMarker, pureClaimIndex, pureBindContext } from "../scripts/kernel/rollbus.mjs";

const frame = (actorid, itemid, claimed = false) => ({
  marker: pureRollMarker(actorid, itemid),
  claimed,
});

describe("pureRollMarker", () => {
  it("reproduces YZEDiceRoller.mjs:62 byte for byte", () => {
    expect(pureRollMarker("actor00000000001", "item00000000001")).toBe(
      '<div class="chatBG" data-item-id="item00000000001" data-actor-id="actor00000000001">',
    );
  });

  it("reproduces the string interpolation of a missing item id too", () => {
    // The system concatenates the raw value: `... data-item-id="` + undefined + `" ...`
    expect(pureRollMarker("actor00000000001", undefined)).toBe(
      '<div class="chatBG" data-item-id="undefined" data-actor-id="actor00000000001">',
    );
    // actor.mjs:1315-1326 (pushRoll) passes the number 0 in the itemid slot.
    expect(pureRollMarker("actor00000000001", 0)).toBe(
      '<div class="chatBG" data-item-id="0" data-actor-id="actor00000000001">',
    );
  });
});

describe("pureClaimIndex", () => {
  const A = frame("actorA", undefined);
  const B = frame("actorB", undefined);
  const contentA = `${A.marker}<h2>Rolling Heavy Machinery</h2>`;
  const contentB = `${B.marker}<h2>Rolling Observation</h2>`;

  it("refuses anything that is not a yzeRoll card", () => {
    // The auto-panic card created while our frame is still armed (YZEDiceRoller.mjs:208).
    const panic = '<h2 class="alienchatred ctooltip">PANIC CONDITION +7</h2>';
    expect(pureClaimIndex([A], panic)).toBe(-1);
    expect(pureClaimIndex([A], "")).toBe(-1);
    expect(pureClaimIndex([A], undefined)).toBe(-1);
  });

  it("returns -1 when nothing is armed", () => {
    expect(pureClaimIndex([], contentA)).toBe(-1);
  });

  it("attributes by marker, not by stack position", () => {
    // Two overlapping rolls: A started first, B is on top, but A's card arrives first.
    expect(pureClaimIndex([A, B], contentA)).toBe(0);
    expect(pureClaimIndex([A, B], contentB)).toBe(1);
  });

  it("skips a frame that already claimed a message", () => {
    const claimedA = frame("actorA", undefined, true);
    const freshA = frame("actorA", undefined);
    // Same actor rolling twice: the deepest UNCLAIMED matching frame wins.
    expect(pureClaimIndex([claimedA, freshA], contentA)).toBe(1);
    expect(pureClaimIndex([claimedA], contentA)).toBe(-1);
  });

  it("prefers the deepest unclaimed matching frame", () => {
    const outer = frame("actorA", undefined);
    const inner = frame("actorA", undefined);
    expect(pureClaimIndex([outer, inner], contentA)).toBe(1);
  });

  it("falls back to the deepest unclaimed frame when no marker matches", () => {
    // Safety net for a future system version that reformats line 62: the card is still
    // recognisably a yzeRoll card, so an armed frame should claim it rather than lose it.
    const odd = '<div class="chatBG" data-item-id="x" data-actor-id="who?">body';
    expect(pureClaimIndex([A, B], odd)).toBe(1);
    expect(pureClaimIndex([frame("actorA", undefined, true)], odd)).toBe(-1);
  });
});

describe("pureBindContext", () => {
  const ctxA = { actorId: "actorA" };
  const ctxB = { actorId: "actorB" };

  it("returns -1 when no context frame is open", () => {
    expect(pureBindContext([], "actorA")).toBe(-1);
    expect(pureBindContext(undefined, "actorA")).toBe(-1);
  });

  it("binds the deepest frame whose actor id matches the roll's 9th argument", () => {
    expect(pureBindContext([ctxA, ctxB], "actorA")).toBe(0);
    expect(pureBindContext([ctxA, ctxB], "actorB")).toBe(1);
    expect(pureBindContext([ctxA, ctxA, ctxB], "actorA")).toBe(1);
  });

  it("falls back to the innermost frame when nothing matches", () => {
    // A vehicle roll: abilityRoll's actor is the vehicle, but actor.mjs:235 puts the PILOT's
    // id into the actorid slot, so the ids never match and the top frame is still the right one.
    expect(pureBindContext([ctxA], "pilot001")).toBe(0);
    expect(pureBindContext([ctxA, ctxB], "pilot001")).toBe(1);
    expect(pureBindContext([ctxA, ctxB], undefined)).toBe(1);
  });
});
```

- [ ] **Step 2: 跑它，看它失败**

Run: `npx vitest run test/rollbus.test.mjs`
Expected: FAIL —— `Error: Failed to resolve import "../scripts/kernel/rollbus.mjs" from "test/rollbus.test.mjs". Does the file exist?`

- [ ] **Step 3: 建 `scripts/kernel/rollbus.mjs`，只写文件头与三个纯函数**

新建 `scripts/kernel/rollbus.mjs`：

```js
import { MID, FLAG_ROLL, HOOK_ROLL_RESOLVED } from "../const.mjs";
import { buildRollRecord, buildLabelIndex, pureReverseLabelKey, LABEL_KEYS } from "./record.mjs";
import { resolver } from "./resolver.mjs";
import { selftest } from "./selftest.mjs";

/**
 * K1 effect layer (CONTRACT §4 K1). Everything that DECIDES something lives in the three pure*
 * functions below: they touch no Foundry global and are unit-tested directly. The imports above
 * are evaluated at load time but none of them reads a Foundry global at module scope.
 *
 * LABEL_KEYS belongs to record.mjs and there is exactly ONE copy of it in the whole module
 * (CONTRACT §4 K1 [v3.1]). This file imports it and MUST NOT export a symbol of that name.
 */

/** Every yzeRoll chat card starts with this, and nothing else in the system does. */
const ROLL_CONTENT_PREFIX = '<div class="chatBG" data-item-id="';

/**
 * Rebuild the opening string of the chat card that yzeRoll is about to create.
 * Verbatim from YZEDiceRoller.mjs:62:
 *   `<div class="chatBG" data-item-id="` + itemid + `" data-actor-id="` + actorid + `">`
 * String concatenation, so undefined becomes the text "undefined" — reproduce that exactly.
 * @param {string|number|undefined} actorid yzeRoll's 9th parameter
 * @param {string|number|undefined} itemid yzeRoll's 10th parameter
 * @returns {string}
 */
export function pureRollMarker(actorid, itemid) {
  return `<div class="chatBG" data-item-id="${itemid}" data-actor-id="${actorid}">`;
}

/**
 * Decide which armed yzeRoll call owns an incoming chat message.
 * Attribution is by call stack + content fingerprint, never by "the most recent message":
 * auto-panic (YZEDiceRoller.mjs:198-242) creates its card BEFORE the card that triggered it.
 * @param {Array<{marker:string, claimed:boolean}>} stack innermost call last
 * @param {string} content the message content about to be created
 * @returns {number} index into stack, or -1 for "not ours"
 */
export function pureClaimIndex(stack, content) {
  if (typeof content !== "string" || !content.startsWith(ROLL_CONTENT_PREFIX)) return -1;
  const frames = Array.isArray(stack) ? stack : [];
  for (let i = frames.length - 1; i >= 0; i--) {
    const frame = frames[i];
    if (frame && !frame.claimed && content.startsWith(frame.marker)) return i;
  }
  // The card is a yzeRoll card but no marker matched (a future system version reformatted
  // line 62). Better to attribute it to the innermost armed call than to drop it.
  for (let i = frames.length - 1; i >= 0; i--) {
    if (frames[i] && !frames[i].claimed) return i;
  }
  return -1;
}

/**
 * Decide which open upstream context frame a yzeRoll call belongs to.
 * Match on the actor id first; fall back to the innermost frame, which is the right answer for
 * vehicle/spacecraft rolls where actor.mjs:235 substitutes the PILOT's id and item.mjs:399
 * substitutes the firing CREW MEMBER's id into the actorid slot.
 * @param {Array<{actorId:string|null}>} ctxStack innermost last
 * @param {string|number|undefined} actorid yzeRoll's 9th parameter
 * @returns {number} index into ctxStack, or -1 when nothing is open
 */
export function pureBindContext(ctxStack, actorid) {
  const frames = Array.isArray(ctxStack) ? ctxStack : [];
  if (frames.length === 0) return -1;
  const id = actorid === undefined || actorid === null ? "" : String(actorid);
  if (id) {
    for (let i = frames.length - 1; i >= 0; i--) {
      if (frames[i] && String(frames[i].actorId ?? "") === id) return i;
    }
  }
  return frames.length - 1;
}
```

- [ ] **Step 4: 跑它，看它通过**

Run: `npx vitest run test/rollbus.test.mjs`
Expected: PASS，11 个用例全绿。

- [ ] **Step 5: 提交纯层**

```bash
git add scripts/kernel/rollbus.mjs test/rollbus.test.mjs && git commit -m "feat(kernel): K1 副作用层的三个纯判定函数

掷骰归属的三个决策全部下沉成纯函数，脱开 Foundry 直接单测：

- pureRollMarker：逐字复刻 YZEDiceRoller.mjs:62 拼出来的卡片开头。系统用的是
  字符串加号拼接，itemid 缺失时会拼出字面量 undefined，照抄。
- pureClaimIndex：按调用栈 + 内容指纹认领消息。自动恐慌在 :198-242 不 await
  地调 rollResolve/rollPanic，恐慌卡先于触发卡创建，所以'最近一条消息就是我的'
  必然错。取最深一个未认领且指纹相符的帧；指纹全不匹配但确实是掷骰卡时，退回
  最深的未认领帧，以防上游改版重排 :62 那行导致整条记录丢失。
- pureBindContext：按 actor id 绑上游上下文帧，匹配不上取栈顶。载具/飞船的
  abilityRoll 帧里是载具而第 9 参是驾驶员（actor.mjs:235），飞船炮位的
  Item#roll 帧里是飞船而第 9 参是开火船员（item.mjs:399），两种情况栈顶都对。

反查标签用的 i18n 键表 LABEL_KEYS 归 record.mjs 所有，本文件只 import，全仓不
存在第二份键表。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: 写下共享测试桩的三条前置断言（只读桩，绝不改桩）**

后面所有集成测试都压在共享桩的三条行为上（契约 §0.3）。先把它们钉成显式断言，这样将来桩变了会在这里报，而不是在十几个包裹用例里报出莫名其妙的错。**把下面的两条 import 加到文件顶部的 import 区**（跟 Step 1 那条并列），再把 `describe` 追加到文件末尾：

```js
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";
import { MID as MID_ID, FLAG_ROLL as FLAG_KEY, HOOK_ROLL_RESOLVED } from "../scripts/const.mjs";
```

```js
describe("shared stub preconditions", () => {
  let ctx;

  beforeEach(() => {
    ctx = installFoundryStub();
  });

  afterEach(() => {
    uninstallFoundryStub();
  });

  it("dispatches Hooks.callAll to handlers registered with Hooks.on", () => {
    const seen = [];
    globalThis.Hooks.on("aea.probe", (x) => seen.push(x));
    globalThis.Hooks.callAll("aea.probe", 1);
    expect(seen).toEqual([1]);
  });

  it("fires preCreateChatMessage before createChatMessage and honours updateSource", async () => {
    const order = [];
    globalThis.Hooks.on("preCreateChatMessage", (doc) => {
      order.push("pre");
      doc.updateSource({ "flags.probe.value": 7 });
    });
    globalThis.Hooks.on("createChatMessage", (message) => order.push(`create:${message.flags?.probe?.value}`));
    await globalThis.ChatMessage.create({ content: "hello" });
    expect(order).toEqual(["pre", "create:7"]);
    expect(ctx.messages).toHaveLength(1);
  });

  it("books libWrapper registrations without replacing the target", () => {
    // The stub is a LEDGER, not a real libWrapper: it records {module, target, fn, type} and
    // leaves the target alone. That is why the integration tests below apply the booked
    // wrappers to their own fixtures themselves.
    const before = globalThis.game.i18n.localize;
    const fn = function () {};
    globalThis.libWrapper.register("probe", "game.i18n.localize", fn, "WRAPPER");
    expect(ctx.wrappers).toHaveLength(1);
    expect(ctx.wrappers[0]).toMatchObject({ module: "probe", target: "game.i18n.localize", type: "WRAPPER" });
    expect(ctx.wrappers[0].fn).toBe(fn);
    expect(globalThis.game.i18n.localize).toBe(before);
  });
});
```

- [ ] **Step 7: 跑它，看它通过**

Run: `npx vitest run test/rollbus.test.mjs -t "shared stub preconditions"`
Expected: PASS，3 个用例。

若任何一条 FAIL：那是共享桩与契约 §0.3 的偏差，**不要改 `test/stubs/foundry.mjs`**（该文件由建桩的任务独占，并且有一份 `test/stub-fidelity.test.mjs` 守着它），也**不要**在本文件里自造 `globalThis.Hooks` / `globalThis.ChatMessage` 顶替它。把失败用例名与报错原文原样记下来交给桩的属主，本任务先停在这一步。唯一的例外是第三条：如果桩**真的替换了目标**（`expect(globalThis.game.i18n.localize).toBe(before)` 红了），那说明桩比契约多做了一步，此时把 Step 8 的 `applyRecordedWrappers()` 整个删掉并把 `beforeEach` 里那次调用一并删掉即可，其余不动。

- [ ] **Step 8: 写下集成测试的夹具**

追加到 `test/rollbus.test.mjs`。**本文件的夹具规矩，只此一条**：共享桩负责一切 Foundry 通用的东西；测试只**追加**桩不可能知道的 alienrpg 专有成员（`game.alienrpg`、`CONFIG.Actor/Item.documentClass`），通用但桩没给的用 `??=` 补（`game.time`、`game.messages`），**绝不替换桩已经装好的成员，绝不改桩文件**。

```js
import { buildLabelIndex, LABEL_KEYS } from "../scripts/kernel/record.mjs";

let stub;      // ctx from installFoundryStub()
let rollBus;   // re-imported per test so module state cannot leak between tests
let resolver;  // MUST come from the same fresh graph as rollBus (a test monkey-patches it)
let selftest;  // ditto, or selftest.runAll() would query a different instance
let resolved;  // every (record, message) pair emitted on aea.rollResolved
let scripted;  // what each queued roll writes into rollArr, in call-RESUME order
let inFlight;  // fire-and-forget yzeRoll promises, because the system never awaits them
let undoWrappers = [];

/** Whatever the stub's dictionary says. Rolling with L(key) as the label means the reverse
 *  index must map it back to `key`, no matter which dictionary the stub ships. */
const L = (key) => globalThis.game.i18n.localize(key);
const sent = () => stub.messages;
const recordAt = (i) => stub.messages[i]?.flags?.[MID_ID]?.[FLAG_KEY] ?? null;

const trackRoll = (promise) => {
  inFlight.push(promise);
  return promise;
};
const settleRolls = async () => {
  while (inFlight.length) await Promise.all(inFlight.splice(0));
};

/**
 * The stub books libWrapper.register into ctx.wrappers but does not touch the target (asserted
 * in "shared stub preconditions"), so the test applies the booked wrappers itself — to its OWN
 * fixture objects — with real libWrapper WRAPPER semantics: fn(wrapped, ...args), `this` kept.
 * Every application records how to undo itself: FakeActor.prototype is shared between tests.
 */
function applyRecordedWrappers(ctx) {
  for (const entry of ctx.wrappers) {
    expect(entry.type).toBe("WRAPPER");
    const parts = entry.target.split(".");
    const prop = parts.pop();
    let owner = globalThis;
    for (const part of parts) owner = owner[part];
    const original = owner[prop];
    if (typeof original !== "function") throw new Error(`no such wrap target: ${entry.target}`);
    const had = Object.prototype.hasOwnProperty.call(owner, prop);
    undoWrappers.push(() => {
      if (had) owner[prop] = original;
      else delete owner[prop];
    });
    owner[prop] = function (...args) {
      return entry.fn.call(this, original.bind(this), ...args);
    };
  }
}

/**
 * Faithful replay of YZEDiceRoller.yzeRoll's OBSERVABLE behaviour, line refs inline.
 * `scripted` supplies what buildChat would have written into rollArr on top of the dice the
 * call actually asked for.
 */
async function fakeYzeRoll(actortype, blind, reRoll, label, r1Dice, col1, r2Dice, col2, actorid, itemid, tactorid, moddata) {
  const arr = globalThis.game.alienrpg.rollArr;
  arr.r1Dice = 0; arr.r1One = 0; arr.r1Six = 0;    // :107-109
  arr.r2Dice = 0; arr.r2One = 0; arr.r2Six = 0;    // :110-112
  arr.sCount = 0; arr.tLabel = "";                 // :113-114
  if (!r1Dice && !r2Dice) {                        // :120-121 NoAttribute — no message at all
    return globalThis.ui.notifications.warn("ALIENRPG.NoAttribute");
  }
  if (r1Dice < 0) {                                // :137-142 the else branch
    r2Dice = r2Dice + r1Dice;                      // :139
    if (r2Dice < 1) {                              // :140-141 NoDice — no message at all
      return globalThis.ui.notifications.warn("ALIENRPG.NoDice");
    }
  }
  if (actortype === "supply" && r2Dice > 6) r2Dice = 6;  // :154-157 the system's own clamp
  await Promise.resolve();                         // stands in for the awaited Roll#evaluate at :130 / :165
  const plan = scripted.shift() ?? {};
  arr.r1Dice = Math.max(0, r1Dice);                // what buildChat writes back
  arr.r2Dice = Math.max(0, r2Dice);
  arr.tLabel = label;                              // buildChat closes over yzeRoll's label
  Object.assign(arr, plan.rollArr ?? {});
  if (plan.panicCard) {
    // :198-242 auto-panic: rollResolve is called WITHOUT await and its card lands first.
    await globalThis.ChatMessage.create({
      content: '<h2 class="alienchatred ctooltip">PANIC CONDITION +7</h2>',
      speaker: { actor: actorid, token: null },
    });
  }
  await globalThis.ChatMessage.create({
    user: stub.userId,                                                                            // :398
    speaker: { actor: actorid, token: null },                                                     // :399-401
    content: `${pureRollMarker(actorid, itemid)}<h2 class="alienchatwhite">Roll ${label} </h2>`,   // :62 + :126
    flags: { tactorid },                                                                          // :406 — flat, not namespaced
  });
  return undefined;                                                                               // :416-417
}

/** In a live world actor.token is a TokenDocument while getActiveTokens() hands back placeables
 *  whose .document is the TokenDocument. The fixture answers to both so the test does not pin
 *  down which of the two K4's resolver reaches for. */
const tokenDoc = (id) => {
  const doc = { id, uuid: `Scene.s1.Token.${id}` };
  doc.document = doc;
  return doc;
};

/** alienrpgActor stand-in: only the members rollbus wraps or reads. */
class FakeActor {
  constructor(id, type = "character") {
    this.id = id;
    this.type = type;
    this.uuid = `Actor.${id}`;
    this.isToken = false;
    this.token = null;
    this._tokens = [];
  }
  getActiveTokens() {
    return this._tokens;
  }
  async abilityRoll(actor, dataset) {                    // actor.mjs:187
    await Promise.resolve();                             // the awaited DialogV2 before the roll
    // actor.mjs:305-315 — NOT awaited, and only 9 arguments.
    trackRoll(
      globalThis.game.alienrpg.yze.yzeRoll(
        actor.type, false, dataset.reRoll ?? false, dataset.label,
        dataset.roll ?? 5, "Black", dataset.stress ?? 2, "Yellow",
        dataset.actorid ?? actor.id,
      ),
    );
    return undefined;
  }
  async pushRoll(actor, reRoll, hostile, blind, message) { // actor.mjs:1302
    await Promise.resolve();                               // the actor.update() await at :1303-1310
    return globalThis.game.alienrpg.yze.yzeRoll(           // :1315 — this one IS awaited
      hostile, blind, reRoll, globalThis.game.alienrpg.rollArr.tLabel,
      3, "Black", 3, "Yellow", actor.id, 0, message.flags?.tactorid,
    );
  }
}

/** alienrpgItem stand-in: only Item#roll, which is what rollbus wraps. */
class FakeItem {
  constructor(id, actor) {
    this.id = id;
    this.actor = actor;
    this.uuid = `${actor.uuid}.Item.${id}`;
  }
  async roll(right, dataset) {                            // item.mjs:39
    await Promise.resolve();
    // item.mjs:114-125 — NOT awaited, 10 arguments, actorid defaults to this.actor.id (:55).
    trackRoll(
      globalThis.game.alienrpg.yze.yzeRoll(
        this.actor.type, false, false, dataset.label,
        4, "Black", 2, "Yellow", dataset.actorid ?? this.actor.id, this.id,
      ),
    );
    return undefined;
  }
}

const SKILL_RESULT = { rollArr: { r1One: 1, r1Six: 2, r2One: 0, r2Six: 1 } };
```

- [ ] **Step 9: 写下四入口包裹的失败测试**

继续追加到 `test/rollbus.test.mjs`：

```js
describe("rollBus.install", () => {
  beforeEach(async () => {
    scripted = [];
    inFlight = [];
    resolved = [];
    stub = installFoundryStub();

    // alienrpg's own globals — the shared stub cannot know these.
    globalThis.game.alienrpg = {
      yze: { yzeRoll: fakeYzeRoll },
      rollArr: { r1Dice: 0, r1One: 0, r1Six: 0, r2Dice: 0, r2One: 0, r2Six: 0, tLabel: "", sCount: 0, multiPush: 0 },
    };
    globalThis.CONFIG.Actor = { documentClass: FakeActor };
    globalThis.CONFIG.Item = { documentClass: FakeItem };
    // Generic Foundry members the stub may or may not ship: fill in only if missing.
    globalThis.game.time ??= { worldTime: 1200 };
    globalThis.game.messages ??= { contents: stub.messages };

    // Fresh module state per test: install() is one-shot by design, and rollbus/resolver/selftest
    // must all come from the SAME post-reset graph or they cannot see each other.
    vi.resetModules();
    ({ rollBus } = await import("../scripts/kernel/rollbus.mjs"));
    ({ resolver } = await import("../scripts/kernel/resolver.mjs"));
    ({ selftest } = await import("../scripts/kernel/selftest.mjs"));

    // Exactly what main.mjs does at i18nInit. buildLabelIndex is pure, so the statically imported
    // copy is interchangeable with the one inside the fresh graph.
    rollBus.setLabelIndex(buildLabelIndex(globalThis.game.i18n.localize.bind(globalThis.game.i18n), LABEL_KEYS));
    rollBus.install();
    applyRecordedWrappers(stub);
    globalThis.Hooks.on(HOOK_ROLL_RESOLVED, (record, message) => resolved.push({ record, message }));
  });

  afterEach(() => {
    undoWrappers.splice(0).reverse().forEach((undo) => undo());
    uninstallFoundryStub();
  });

  const rawRoll = (over = {}) =>
    globalThis.game.alienrpg.yze.yzeRoll(
      over.actortype ?? "character", false, over.reRoll ?? false, over.label ?? L("ALIENRPG.SkillheavyMach"),
      over.r1Dice ?? 5, "Black", over.r2Dice ?? 2, "Yellow",
      over.actorid ?? "actorA", over.itemid, over.tactorid, over.moddata,
    );

  it("stamps the record onto the card and emits aea.rollResolved once", async () => {
    // The GM-macro path: no upstream frame, so actorUuid depends on K4's speaker fallback and is
    // asserted in the three context-driven cases below instead (and in MANUAL VERIFICATION 9).
    scripted.push(SKILL_RESULT);
    await rawRoll();
    expect(sent()).toHaveLength(1);
    const record = recordAt(0);
    expect(record.v).toBe(1);
    expect(record.kind).toBe("skill");
    expect(record.label).toBe(L("ALIENRPG.SkillheavyMach"));
    expect(record.labelKey).toBe("ALIENRPG.SkillheavyMach");
    expect(record.pools).toEqual({ base: 5, stress: 2 });
    expect(record.successes).toBe(3);
    expect(record.banes).toBe(0);
    expect(record.push).toEqual({ count: 0, pushable: true, parentRollId: null });
    expect(record.userId).toBe(stub.userId);
    expect(record.at.worldTime).toBe(globalThis.game.time.worldTime);
    expect(typeof record.id).toBe("string");
    expect(record.id.length).toBeGreaterThan(0);
    expect(resolved).toHaveLength(1);
    expect(resolved[0].record).toEqual(record);
    expect(resolved[0].message).toBe(sent()[0]);
    expect(rollBus.depth()).toBe(0);
    expect(rollBus.context()).toBeNull();
  });

  it("registers exactly the four wrap targets, all WRAPPER, all owned by this module", () => {
    expect(stub.wrappers).toHaveLength(4);
    expect(stub.wrappers.map((w) => w.target)).toEqual([
      "game.alienrpg.yze.yzeRoll",
      "CONFIG.Actor.documentClass.prototype.abilityRoll",
      "CONFIG.Item.documentClass.prototype.roll",
      "CONFIG.Actor.documentClass.prototype.pushRoll",
    ]);
    for (const entry of stub.wrappers) {
      expect(entry.module).toBe(MID_ID);
      expect(entry.type).toBe("WRAPPER");
    }
  });

  it("is one-shot: a second install() registers nothing more", () => {
    // lib-wrapper throws on a second registration of the same target, so the guard is load-bearing.
    rollBus.install();
    expect(stub.wrappers).toHaveLength(4);
  });

  it("threads dataset.attr from abilityRoll even though the system never awaits the roll", async () => {
    const actor = new FakeActor("actorA");
    scripted.push(SKILL_RESULT);
    await actor.abilityRoll(actor, { label: L("ALIENRPG.AbilityStr"), attr: "str", roll: 5, stress: 2 });
    // abilityRoll has already returned and its context frame is gone (actor.mjs:305 fires the
    // roll without await). The sink frame must still hold the captured frame by reference.
    expect(rollBus.context()).toBeNull();
    await settleRolls();
    const record = recordAt(0);
    expect(record.kind).toBe("attribute");
    expect(record.attr).toBe("str");
    expect(record.labelKey).toBe("ALIENRPG.AbilityStr");
    expect(record.actorUuid).toBe("Actor.actorA");
    expect(record.tokenUuid).toBeNull();   // no active token: the honest answer, never a guess
    expect(rollBus.depth()).toBe(0);
  });

  it("threads the item uuid from Item#roll into the record", async () => {
    const actor = new FakeActor("actorA");
    const item = new FakeItem("item00000000001", actor);
    scripted.push({ rollArr: { r1Six: 2, r2One: 1 } });
    await item.roll(false, { label: "M41A Pulse Rifle" });
    await settleRolls();
    const record = recordAt(0);
    expect(record.kind).toBe("weapon");
    expect(record.itemUuid).toBe("Actor.actorA.Item.item00000000001");
    expect(record.actorUuid).toBe("Actor.actorA");
    expect(record.labelKey).toBeNull(); // an item name is not an i18n key
  });

  it("keeps two unlinked tokens of one base actor apart on the weapon path", async () => {
    // The three-Drone case. An unlinked token's synthetic actor keeps the BASE actor's id and
    // only its uuid differs, so anything that resolves by id collapses all three onto one.
    const drone = (tokenId) => {
      const a = new FakeActor("droneBase", "creature");
      a.isToken = true;
      a.token = tokenDoc(tokenId);
      a.uuid = `Scene.s1.Token.${tokenId}.Actor.droneBase`;
      return a;
    };
    scripted.push(SKILL_RESULT, SKILL_RESULT);
    await new FakeItem("claw", drone("t1")).roll(false, { label: "Claw" });
    await settleRolls();
    await new FakeItem("claw", drone("t2")).roll(false, { label: "Claw" });
    await settleRolls();
    expect(recordAt(0).tokenUuid).toBe("Scene.s1.Token.t1");
    expect(recordAt(1).tokenUuid).toBe("Scene.s1.Token.t2");
    expect(recordAt(0).actorUuid).not.toBe(recordAt(1).actorUuid);
  });

  it("threads the parent record id and the push count from pushRoll", async () => {
    const actor = new FakeActor("actorA");
    scripted.push(SKILL_RESULT);
    await actor.abilityRoll(actor, { label: L("ALIENRPG.SkillheavyMach"), roll: 5, stress: 2 });
    await settleRolls();
    const first = recordAt(0);
    expect(first.push).toEqual({ count: 0, pushable: true, parentRollId: null });

    scripted.push({ rollArr: { r1Six: 1, r2One: 1 } });
    await actor.pushRoll(actor, "push", "character", false, sent()[0]);
    const pushed = recordAt(1);
    expect(pushed.push.count).toBe(1);
    expect(pushed.push.pushable).toBe(false);
    expect(pushed.push.parentRollId).toBe(first.id);
    expect(pushed.id).not.toBe(first.id);
  });

  it("inherits the parent card's refs on a push instead of collapsing onto the base actor", async () => {
    // actor.mjs:1315 hands yzeRoll the BASE actor id, and the base actor here has two active
    // tokens, so re-resolving would honestly yield tokenUuid null and lose the identity.
    const base = new FakeActor("actorA");
    base._tokens = [tokenDoc("t1"), tokenDoc("t2")];
    const parent = {
      flags: {
        [MID_ID]: {
          [FLAG_KEY]: {
            v: 1,
            id: "parent0000000001",
            actorUuid: "Scene.s1.Token.t1.Actor.actorA",
            tokenUuid: "Scene.s1.Token.t1",
            push: { count: 0, pushable: true, parentRollId: null },
          },
        },
      },
    };
    scripted.push({ rollArr: { r1Six: 1 } });
    await base.pushRoll(base, "push", "character", false, parent);
    const pushed = recordAt(0);
    expect(pushed.actorUuid).toBe("Scene.s1.Token.t1.Actor.actorA");
    expect(pushed.tokenUuid).toBe("Scene.s1.Token.t1");
    expect(pushed.push.parentRollId).toBe("parent0000000001");
  });

  it("reports the dice actually rolled, not the dice requested", async () => {
    // YZEDiceRoller.mjs:154-157 clamps a supply roll to 6 stress dice. args say 9; only rollArr
    // reflects what was thrown, which is why CONTRACT §3 pins pools to rollArr.
    scripted.push({ rollArr: { r2One: 1, r2Six: 2 } });
    await rawRoll({ actortype: "supply", reRoll: true, label: "Rounds Supply", r1Dice: 0, r2Dice: 9 });
    expect(recordAt(0).pools).toEqual({ base: 0, stress: 6 });
  });

  it("disarms cleanly when yzeRoll returns without creating a message", async () => {
    await rawRoll({ r1Dice: 0, r2Dice: 0 });  // :120-121 NoAttribute
    await rawRoll({ r1Dice: -4, r2Dice: 2 }); // :137-141 NoDice
    expect(sent()).toHaveLength(0);
    expect(resolved).toHaveLength(0);
    expect(stub.notifications).toHaveLength(2);
    expect(rollBus.depth()).toBe(0);
  });

  it("does not stamp the auto-panic card that is created before ours", async () => {
    scripted.push({ panicCard: true, rollArr: { r1Six: 1, r2One: 1 } });
    await rawRoll();
    expect(sent()).toHaveLength(2);
    expect(recordAt(0)).toBeNull();       // the panic card
    expect(recordAt(1).banes).toBe(1);    // the roll card
    expect(resolved).toHaveLength(1);
  });

  it("attributes two overlapping rolls by their own frame, not by recency", async () => {
    scripted.push(
      { rollArr: { r1Six: 3 } },            // resumes first: actorA
      { rollArr: { r1Six: 0, r2One: 1 } },  // resumes second: actorB
    );
    const a = rawRoll({ actorid: "actorA", label: "A" });
    const b = rawRoll({ actorid: "actorB", label: "B" });
    expect(rollBus.depth()).toBe(2);
    await Promise.all([a, b]);
    const byActor = Object.fromEntries(sent().map((d, i) => [d.speaker.actor, recordAt(i)]));
    expect(byActor.actorA.successes).toBe(3);
    expect(byActor.actorA.label).toBe("A");
    expect(byActor.actorB.successes).toBe(0);
    expect(byActor.actorB.banes).toBe(1);
    expect(rollBus.depth()).toBe(0);
  });

  it("records a supply check from its arguments", async () => {
    scripted.push({ rollArr: { r2One: 2, r2Six: 1 } });
    await rawRoll({ actortype: "supply", reRoll: true, label: "Rounds Supply", r1Dice: 0, r2Dice: 6 });
    const supply = recordAt(0);
    expect(supply.kind).toBe("supply");
    expect(supply.banes).toBe(2);
    expect(supply.push.pushable).toBe(false);
    expect(supply.consumed).toEqual({ ammo: null }); // phase 1: ammo is never observable here
    expect(supply.targets).toEqual([]);              // phase 1: filled by the phase-2 targeting feature
  });

  it("uses the label index injected at i18nInit rather than rebuilding one", async () => {
    rollBus.setLabelIndex({ [L("ALIENRPG.SkillheavyMach")]: "AEA.test.injected" });
    scripted.push(SKILL_RESULT);
    await rawRoll();
    expect(recordAt(0).labelKey).toBe("AEA.test.injected");
  });

  it("ignores messages that carry no record", () => {
    expect(rollBus.recordOf(null)).toBeNull();
    expect(rollBus.recordOf({ flags: {} })).toBeNull();
    expect(rollBus.recordOf({ flags: { [MID_ID]: { [FLAG_KEY]: "nope" } } })).toBeNull();
  });

  it("falls back to empty refs when the resolver throws", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    resolver.fromSpeaker = () => {
      throw new Error("boom");
    };
    scripted.push(SKILL_RESULT);
    await rawRoll();
    expect(recordAt(0).actorUuid).toBeNull();
    expect(recordAt(0).successes).toBe(3);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it("never lets a broken record abort the system's own chat card", async () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    // Something in the environment blowing up mid-stamp. game.time is a fixture of this file,
    // so making it throw touches nothing the shared stub owns.
    Object.defineProperty(globalThis.game, "time", {
      configurable: true,
      get() {
        throw new Error("boom");
      },
    });
    scripted.push(SKILL_RESULT);
    await expect(rawRoll()).resolves.toBeUndefined();
    expect(sent()).toHaveLength(1);
    expect(recordAt(0)).toBeNull();
    expect(resolved).toHaveLength(0);
    expect(error).toHaveBeenCalled();
    expect(rollBus.depth()).toBe(0);
    error.mockRestore();
  });
});
```

- [ ] **Step 10: 写下 `addStage` 的失败测试**

追加到 `describe("rollBus.install")` 的**内部末尾**（在它的最后一个 `it` 之后、右花括号之前），这样复用它的 `beforeEach`：

```js
  describe("addStage", () => {
    it("runs a queued stage around yzeRoll and lets it rewrite the arguments", async () => {
      rollBus.addStage("yzeRoll", {
        id: "test.clamp",
        around(next, args) {
          const patched = [...args];
          patched[6] = 6; // r2Dice, the 7th positional parameter (YZEDiceRoller.mjs:31-44)
          return next(patched);
        },
      });
      scripted.push({ rollArr: { r1Six: 1 } });
      await rawRoll({ r1Dice: 5, r2Dice: 9 });
      expect(recordAt(0).pools).toEqual({ base: 5, stress: 6 });
    });

    it("orders stages by order first and by registration second", async () => {
      const seen = [];
      const probe = (id, order) =>
        rollBus.addStage("yzeRoll", {
          id,
          order,
          around: (next, args) => {
            seen.push(id);
            return next(args);
          },
        });
      probe("b", 10);
      probe("a", -10);
      probe("c", -10);
      scripted.push(SKILL_RESULT);
      await rawRoll();
      expect(seen).toEqual(["a", "c", "b"]);
    });

    it("still claims the card when a stage rewrote the ids the marker is built from", async () => {
      rollBus.addStage("yzeRoll", {
        id: "test.itemid",
        around(next, args) {
          const patched = [...args];
          patched[9] = "item00000000009"; // itemid, the 10th positional parameter
          return next(patched);
        },
      });
      scripted.push(SKILL_RESULT);
      await rawRoll();
      expect(sent()[0].content).toContain('data-item-id="item00000000009"');
      expect(recordAt(0).successes).toBe(3);
    });

    it("hands the wrapped object to the stage as thisArg", async () => {
      const seen = [];
      rollBus.addStage("itemRoll", {
        id: "test.this",
        around(next, args, thisArg) {
          seen.push(thisArg?.uuid);
          return next(args);
        },
      });
      const actor = new FakeActor("actorA");
      scripted.push(SKILL_RESULT);
      await new FakeItem("item00000000001", actor).roll(false, { label: "Knife" });
      await settleRolls();
      expect(seen).toEqual(["Actor.actorA.Item.item00000000001"]);
    });

    it("ignores a second stage registered under the same id", async () => {
      let calls = 0;
      const stage = {
        id: "test.once",
        around: (next, args) => {
          calls += 1;
          return next(args);
        },
      };
      rollBus.addStage("yzeRoll", stage);
      rollBus.addStage("yzeRoll", stage);
      scripted.push(SKILL_RESULT);
      await rawRoll();
      expect(calls).toBe(1);
    });

    it("rolls anyway when a stage forgets to call next()", async () => {
      const error = vi.spyOn(console, "error").mockImplementation(() => {});
      rollBus.addStage("yzeRoll", { id: "test.forgot", around: () => undefined });
      scripted.push(SKILL_RESULT);
      await rawRoll();
      expect(sent()).toHaveLength(1);
      expect(recordAt(0).successes).toBe(3);
      expect(error).toHaveBeenCalled();
      error.mockRestore();
    });

    it("rolls anyway when a stage throws before calling next()", async () => {
      const error = vi.spyOn(console, "error").mockImplementation(() => {});
      rollBus.addStage("yzeRoll", {
        id: "test.throws",
        around() {
          throw new Error("boom");
        },
      });
      scripted.push(SKILL_RESULT);
      await rawRoll();
      expect(sent()).toHaveLength(1);
      expect(recordAt(0).successes).toBe(3);
      expect(error).toHaveBeenCalled();
      error.mockRestore();
    });

    it("refuses an unknown target and a stage with no around()", () => {
      expect(() => rollBus.addStage("nope", { id: "x", around: (next, args) => next(args) })).toThrow();
      expect(() => rollBus.addStage("yzeRoll", { id: "x" })).toThrow();
      expect(() => rollBus.addStage("yzeRoll", { around: (next, args) => next(args) })).toThrow();
    });
  });
```

- [ ] **Step 11: 跑它们，看它们失败**

Run: `npx vitest run test/rollbus.test.mjs -t "rollBus.install"`
Expected: FAIL —— 25 个用例全在 `beforeEach` 里挂掉，报 `TypeError: Cannot read properties of undefined (reading 'setLabelIndex')`（`rollbus.mjs` 目前没有名为 `rollBus` 的导出，解构得到 `undefined`）。

- [ ] **Step 12: 写模块状态与几个小工具**

追加到 `scripts/kernel/rollbus.mjs` 末尾：

```js
/** yzeRoll's 12 positional parameters, in source order (YZEDiceRoller.mjs:31-44). */
const ARG_NAMES = [
  "actortype", "blind", "reRoll", "label",
  "r1Dice", "col1", "r2Dice", "col2",
  "actorid", "itemid", "tactorid", "moddata",
];

/** The six rollArr fields that describe the dice; used to spot a concurrent re-zero. */
const DICE_FIELDS = ["r1Dice", "r1One", "r1Six", "r2Dice", "r2One", "r2Six"];

const state = {
  installed: false,
  stack: [],            // armed yzeRoll frames, innermost last
  ctxStack: [],         // open upstream context frames, innermost last
  stages: { yzeRoll: [], abilityRoll: [], itemRoll: [], pushRoll: [] },
  stageSeq: 0,
  labelIndex: null,
  labelIndexSource: "none",  // "i18nInit" once main.mjs injects one, "lazy" if we had to build it
  idSeq: 0,
  wrapFailures: [],
  raceWarnings: 0,
};

function argsToObject(args) {
  const out = {};
  for (let i = 0; i < ARG_NAMES.length; i++) out[ARG_NAMES[i]] = args[i];
  return out;
}

function nonEmptyString(value) {
  return typeof value === "string" && value.length > 0;
}

/**
 * The reverse label table. main.mjs builds it at i18nInit and injects it through
 * setLabelIndex(); this lazy branch is the SAME builder over the SAME key list (record.mjs owns
 * both), so a GM macro that fires before i18nInit still gets an identical table, never a second one.
 */
function labelIndex() {
  if (state.labelIndex) return state.labelIndex;
  let localize = (key) => key;
  try {
    if (typeof game?.i18n?.localize === "function") localize = game.i18n.localize.bind(game.i18n);
  } catch {
    /* i18n not up yet: every label reverses to null, which is the honest answer */
  }
  state.labelIndex = buildLabelIndex(localize, LABEL_KEYS);
  state.labelIndexSource = "lazy";
  return state.labelIndex;
}

function newId() {
  const randomID = globalThis.foundry?.utils?.randomID;
  if (typeof randomID === "function") return randomID();
  return `aea${Date.now().toString(36)}${(++state.idSeq).toString(36)}`;
}

function popFrom(list, item) {
  const i = list.indexOf(item);
  if (i >= 0) list.splice(i, 1);
}

/** Run `done` once the wrapped async call settles, whichever way it settles. */
function settle(out, done) {
  return Promise.resolve(out).then(
    (value) => {
      done();
      return value;
    },
    (err) => {
      done();
      throw err;
    },
  );
}

/**
 * CONTRACT §4 K1: an unlinked token's synthetic actor names its own token; otherwise the only
 * honest answer is K4's "the one active token, or nothing". Never game.actors.get() as a fallback.
 */
function tokenOf(actor) {
  if (!actor) return null;
  if (actor.isToken && actor.token) return actor.token;
  try {
    return resolver.soleToken(actor) ?? null;
  } catch {
    return null;
  }
}

function pushCtx(patch) {
  const ctx = {
    attr: null, itemUuid: null, dataset: null,
    actorId: null, actor: null, token: null,
    inheritedRefs: null, parentRollId: null, pushCount: 0,
    ...patch,
  };
  state.ctxStack.push(ctx);
  return ctx;
}

/** A dataset is a DOMStringMap; copy it into a plain object before it crosses into the pure layer. */
function plainDataset(dataset) {
  if (!dataset || typeof dataset !== "object") return null;
  const out = {};
  for (const key of Object.keys(dataset)) out[key] = dataset[key];
  return out;
}

/** Strip the Documents out of a context frame: the pure layer only takes plain data. */
function plainCtx(ctx, args, refs) {
  const derived =
    nonEmptyString(args?.itemid) && refs.actorUuid ? `${refs.actorUuid}.Item.${args.itemid}` : null;
  return {
    attr: nonEmptyString(ctx?.attr) ? ctx.attr : null,
    itemUuid: nonEmptyString(ctx?.itemUuid) ? ctx.itemUuid : derived,
    dataset: ctx?.dataset ?? null,
    parentRollId: ctx?.parentRollId ?? null,
    pushCount: Number(ctx?.pushCount) || 0,
  };
}

function sameDice(a, b) {
  return DICE_FIELDS.every((f) => Number(a?.[f] ?? 0) === Number(b?.[f] ?? 0));
}
```

- [ ] **Step 13: 写 refs 解析与两个钩子处理函数**

继续追加到 `scripts/kernel/rollbus.mjs` 末尾：

```js
/**
 * uuid-first identity (K4). NEVER game.actors.get(speaker.actor): that loses the token, so five
 * identical marine tokens off one base actor would all share one record.
 * Three sources, in this order:
 *  1. a push inherits the parent card's already-resolved refs verbatim — it is by definition the
 *     same combatant, and actor.mjs:1315 hands yzeRoll the BASE actor id, so re-resolving would
 *     collapse three unlinked Drones back onto one actor;
 *  2. the upstream context frame, but only when it is about the same actor the card names —
 *     a vehicle roll puts the PILOT's id in the actorid slot (actor.mjs:235) and a spacecraft
 *     weapon puts the firing CREW MEMBER's id there (item.mjs:399) while the frame holds the ship;
 *  3. the message speaker, through the resolver.
 */
function refsFor(frame, doc) {
  const empty = { actorUuid: null, tokenUuid: null };
  try {
    const ctx = frame.ctx;
    const sameSubject = !!ctx && String(ctx.actorId ?? "") === String(frame.args.actorid ?? "");
    if (sameSubject && ctx.inheritedRefs?.actorUuid) return { ...ctx.inheritedRefs };
    if (sameSubject && ctx.actor) return resolver.refs(ctx.actor, ctx.token ?? null) ?? empty;
    const speaker = doc?.speaker ?? {};
    const actor = resolver.fromSpeaker(speaker);
    const token = speaker.token ? (globalThis.canvas?.tokens?.get(speaker.token)?.document ?? null) : null;
    return resolver.refs(actor, token) ?? empty;
  } catch (err) {
    console.warn(`${MID} | resolver failed on a roll message`, err);
    return empty;
  }
}

/**
 * Fires synchronously on the creating client, before the message is written. This is the only
 * moment that works: yzeRoll creates the message itself at YZEDiceRoller.mjs:416 and returns
 * undefined at :417, so there is nothing left to hook afterwards, and message.update() backfill
 * is banned (an extra write plus a visible flicker).
 * THERE MUST BE NO await IN THIS FUNCTION: game.alienrpg.rollArr is a single mutable global that
 * the next yzeRoll clears field by field at YZEDiceRoller.mjs:107-114.
 * The whole body sits in one try/catch — throwing here would abort the system's own chat card.
 */
function onPreCreateChatMessage(doc, data, options, userId) {
  try {
    const rollArr = { ...(game?.alienrpg?.rollArr ?? {}) }; // synchronous snapshot, first statement
    const content =
      typeof doc?.content === "string" ? doc.content : typeof data?.content === "string" ? data.content : "";
    const index = pureClaimIndex(state.stack, content);
    if (index < 0) return; // not ours: the system creates plenty of other messages
    const frame = state.stack[index];
    frame.claimed = true;
    frame.snapshot = rollArr;
    const refs = refsFor(frame, doc);
    const record = buildRollRecord({
      args: frame.args,
      rollArr, // CONTRACT §3: pools come from here, never from args
      refs,
      ctx: plainCtx(frame.ctx, frame.args, refs),
      userId: userId ?? doc?.author?.id ?? game?.user?.id ?? "",
      worldTime: game?.time?.worldTime ?? 0,
      now: Date.now(),
      id: newId(),
      labelIndex: labelIndex(),
    });
    // CONTRACT §3 [v3.1]: the hook's first argument is the not-yet-stored document. Assigning to
    // `data`, or to doc.flags, would not reach the database — only updateSource does.
    doc.updateSource({ [`flags.${MID}.${FLAG_ROLL}`]: record });
  } catch (err) {
    console.error(`${MID} | failed to stamp a roll record`, err);
  }
}

/** Fires on every client once the message exists — that is where features want the hook. */
function onCreateChatMessage(message) {
  const record = rollBus.recordOf(message);
  if (!record) return;
  Hooks.callAll(HOOK_ROLL_RESOLVED, record, message);
}
```

- [ ] **Step 14: 写阶段链与四个包装器**

继续追加到 `scripts/kernel/rollbus.mjs` 末尾：

```js
/**
 * Run one target's stage queue and then the real function.
 * CONTRACT §4 K1 [v3.1]: rollBus owns these four libWrapper targets outright, so anything else
 * that needs to intervene queues here. around(next, args, thisArg) must call next(args) exactly
 * once. A misbehaving stage is logged and stepped over: it must never cost the player a roll.
 */
function runStages(target, args, thisArg, inner) {
  const stages = state.stages[target];
  if (!stages.length) return inner(args);
  const chain = (i, currentArgs) => {
    if (i >= stages.length) return inner(currentArgs);
    const stage = stages[i];
    let calls = 0;
    let downstream;
    const next = (nextArgs) => {
      calls += 1;
      if (calls > 1) {
        console.error(`${MID} | stage ${stage.id} called next() twice on ${target}; ignored the extra call`);
        return downstream;
      }
      downstream = chain(i + 1, Array.isArray(nextArgs) ? nextArgs : currentArgs);
      return downstream;
    };
    let out;
    try {
      out = stage.around(next, currentArgs, thisArg);
    } catch (err) {
      console.error(`${MID} | stage ${stage.id} threw on ${target}`, err);
      // If it blew up before calling next, the roll has not happened yet: run it. If it blew up
      // afterwards, hand back the promise the real call is already running on, or the frame
      // would be popped before the card is created.
      return calls === 0 ? chain(i + 1, currentArgs) : downstream;
    }
    if (calls === 0) {
      console.error(`${MID} | stage ${stage.id} never called next() on ${target}; ran the roll anyway`);
      return chain(i + 1, currentArgs);
    }
    return out;
  };
  return chain(0, args);
}

/**
 * The sink. Arms a frame, binds it to the innermost matching upstream context AT CALL TIME, runs
 * the stage queue, hands control to the real yzeRoll, and pops the frame once it settles.
 * Binding must happen here and the frame must hold the ctx object by reference: three of the four
 * upstream call sites fire yzeRoll WITHOUT await (actor.mjs:305, item.mjs:114/130/175…), so their
 * context frames are already off the stack by the time the card is created.
 */
function wrapYzeRoll(wrapped, ...args) {
  const frame = { args: argsToObject(args), ctx: null, marker: "", claimed: false, snapshot: null };
  const bound = pureBindContext(state.ctxStack, frame.args.actorid);
  frame.ctx = bound >= 0 ? state.ctxStack[bound] : null;
  frame.marker = pureRollMarker(frame.args.actorid, frame.args.itemid);
  state.stack.push(frame);
  // Re-read the arguments at the innermost point: a stage may have rewritten them, and the card
  // the system builds at YZEDiceRoller.mjs:62 will carry the NEW ids.
  const invoke = (finalArgs) => {
    frame.args = argsToObject(finalArgs);
    frame.marker = pureRollMarker(frame.args.actorid, frame.args.itemid);
    return wrapped(...finalArgs);
  };
  let out;
  try {
    out = runStages("yzeRoll", args, this, invoke);
  } catch (err) {
    popFrom(state.stack, frame);
    throw err;
  }
  return settle(out, () => {
    try {
      // CONTRACT §4 K1: a synchronous snapshot at return time. Compared against the one taken at
      // claim time, it detects THIS client having re-zeroed rollArr in between (two interleaved
      // rolls from one client; rollArr is client-local, so other players cannot cause it).
      const tail = { ...(game?.alienrpg?.rollArr ?? {}) };
      if (frame.claimed && frame.snapshot && !sameDice(frame.snapshot, tail)) {
        state.raceWarnings += 1;
        console.warn(`${MID} | rollArr changed between claim and return`, frame.snapshot, tail);
      }
    } catch (err) {
      console.warn(`${MID} | could not re-read rollArr after the roll`, err);
    } finally {
      popFrom(state.stack, frame);
    }
  });
}

/** Push a context frame, run the stage queue, run the real method, pop the frame. */
function withCtx(target, ctx, wrapped, args, thisArg) {
  let out;
  try {
    out = runStages(target, args, thisArg, (finalArgs) => wrapped(...finalArgs));
  } catch (err) {
    popFrom(state.ctxStack, ctx);
    throw err;
  }
  return settle(out, () => popFrom(state.ctxStack, ctx));
}

/** actor.mjs:187 abilityRoll(actor, dataset, rollMod) — the only source of dataset.attr. */
function wrapAbilityRoll(wrapped, ...args) {
  const subject = args[0] ?? this; // the actor is the FIRST ARGUMENT here, not `this`
  const dataset = args[1];
  const ctx = pushCtx({
    attr: nonEmptyString(dataset?.attr) ? dataset.attr : null, // read at :194 and then dropped
    dataset: plainDataset(dataset),
    actorId: subject?.id ?? null,
    actor: subject ?? null,
    token: tokenOf(subject),
  });
  return withCtx("abilityRoll", ctx, wrapped, args, this);
}

/** item.mjs:39 roll(right, dataset) — the only source of the rolled item's uuid. */
function wrapItemRoll(wrapped, ...args) {
  const ctx = pushCtx({
    itemUuid: nonEmptyString(this?.uuid) ? this.uuid : null,
    dataset: plainDataset(args[1]),
    actorId: this?.actor?.id ?? null,   // item.mjs:55 — the default actorid handed to yzeRoll
    actor: this?.actor ?? null,
    token: tokenOf(this?.actor),
  });
  return withCtx("itemRoll", ctx, wrapped, args, this);
}

/** actor.mjs:1302 pushRoll(actor, reRoll, hostile, blind, message) — the only source of the parent card. */
function wrapPushRoll(wrapped, ...args) {
  const actor = args[0] ?? null;
  const parent = rollBus.recordOf(args[4]); // the 5th argument is the message being pushed
  const ctx = pushCtx({
    actorId: actor?.id ?? null,
    actor,
    token: tokenOf(actor),
    // Identity is inherited from the parent record, not re-derived from the base actor id.
    inheritedRefs: nonEmptyString(parent?.actorUuid)
      ? { actorUuid: parent.actorUuid, tokenUuid: parent.tokenUuid ?? null }
      : null,
    parentRollId: parent?.id ?? null,
    pushCount: (Number(parent?.push?.count) || 0) + 1,
  });
  return withCtx("pushRoll", ctx, wrapped, args, this);
}

/** [stage target, libWrapper path, wrapper]. The order here is the registration order. */
const WRAP_TARGETS = [
  ["yzeRoll", "game.alienrpg.yze.yzeRoll", wrapYzeRoll],
  ["abilityRoll", "CONFIG.Actor.documentClass.prototype.abilityRoll", wrapAbilityRoll],
  ["itemRoll", "CONFIG.Item.documentClass.prototype.roll", wrapItemRoll],
  ["pushRoll", "CONFIG.Actor.documentClass.prototype.pushRoll", wrapPushRoll],
];
```

- [ ] **Step 15: 写 `rollBus` 导出（六个成员）**

继续追加到 `scripts/kernel/rollbus.mjs` 末尾：

```js
export const rollBus = {
  /** Call once, from the ready lifecycle hook in main.mjs. */
  install() {
    if (state.installed) return;
    state.installed = true;
    const lw = globalThis.libWrapper;
    if (typeof lw?.register !== "function") {
      // lib-wrapper is a hard dependency, but a GM can still disable it. Fail loudly into the
      // self-test rather than throwing a ReferenceError that takes the whole module down.
      for (const [, target] of WRAP_TARGETS) state.wrapFailures.push(target);
      console.error(`${MID} | libWrapper is not available; no roll wrapper was installed`);
    } else {
      for (const [, target, fn] of WRAP_TARGETS) {
        // All 22 yzeRoll call sites write `yze.yzeRoll(...)` with no destructuring and every
        // `import { yze }` is the same class object, so replacing that one static property covers
        // the whole system. WRAPPER (not OVERRIDE) everywhere so other modules can stack.
        try {
          lw.register(MID, target, fn, "WRAPPER");
        } catch (err) {
          state.wrapFailures.push(target);
          console.error(`${MID} | could not wrap ${target}`, err);
        }
      }
    }
    Hooks.on("preCreateChatMessage", onPreCreateChatMessage);
    Hooks.on("createChatMessage", onCreateChatMessage);
  },

  /** @returns {object|null} the RollRecord stamped on a message, or null */
  recordOf(message) {
    const record = message?.flags?.[MID]?.[FLAG_ROLL];
    if (!record || typeof record !== "object") return null;
    if (typeof record.v !== "number") return null;
    return record;
  },

  /** @returns {number} how many yzeRoll calls are currently in flight */
  depth() {
    return state.stack.length;
  },

  /** @returns {object|null} the innermost open upstream context frame */
  context() {
    return state.ctxStack.length ? state.ctxStack[state.ctxStack.length - 1] : null;
  },

  /**
   * CONTRACT §5: main.mjs injects the reverse label table at i18nInit, built by
   * record.buildLabelIndex() over record.LABEL_KEYS. There is exactly one such table.
   * @param {Record<string, string|null>|null} index
   */
  setLabelIndex(index) {
    state.labelIndex = index && typeof index === "object" ? index : null;
    state.labelIndexSource = state.labelIndex ? "i18nInit" : "none";
  },

  /**
   * CONTRACT §4 K1 [v3.1]: the ONLY way for anything else in this module to intervene on one of
   * K1's four wrap targets. lib-wrapper refuses a second registration of the same target
   * ("A wrapper for '<target>' (ID=<n>) has already been registered by <module>."), so a repair
   * that called libWrapper.register itself would silently become a no-op.
   * Stages may be queued before OR after install(): the wrappers read the queue at call time,
   * which is what lets a repair queue one from patches.applyAll(), i.e. before rollBus.install().
   * @param {"yzeRoll"|"abilityRoll"|"itemRoll"|"pushRoll"} target
   * @param {{id:string, order?:number, around:(next:Function, args:any[], thisArg:any)=>any}} stage
   *        around must call next(args) exactly once; it may rewrite args and the return value.
   *        Lower order runs first (further out); ties break on registration order.
   */
  addStage(target, { id, order = 0, around } = {}) {
    const queue = state.stages[target];
    if (!queue) throw new Error(`${MID} | unknown stage target "${target}"`);
    if (!nonEmptyString(id)) throw new Error(`${MID} | a stage on "${target}" needs an id`);
    if (typeof around !== "function") {
      throw new Error(`${MID} | stage "${id}" needs an around(next, args, thisArg) function`);
    }
    if (queue.some((s) => s.id === id)) return; // idempotent: applyAll may run more than once
    queue.push({ id, order: Number(order) || 0, around, seq: ++state.stageSeq });
    queue.sort((a, b) => a.order - b.order || a.seq - b.seq);
  },
};
```

- [ ] **Step 16: 跑整份测试，看它通过**

Run: `npx vitest run test/rollbus.test.mjs`
Expected: PASS，39 个用例全绿（纯函数 11 + 桩前置 3 + 包裹 17 + 阶段 8）。
再跑一次 `npm test` 确认没把别的测试带崩。

若 `threads dataset.attr…` / `threads the item uuid…` / `keeps two unlinked tokens…` 这三条挂在 `record.actorUuid` 或 `record.tokenUuid` 上，说明 K4 的 `resolver.refs` / `resolver.soleToken` 需要共享桩没提供的全局。**不要改 `test/stubs/foundry.mjs`，不要改 rollbus 去迁就，也不要 `vi.mock` 掉 resolver**：先试 `installFoundryStub({ world: { [actor.uuid]: actor } })` 这个桩自己的选项（契约 §0.3：`ctx.documents` 支撑 `fromUuidSync`），仍然失败就把报错原文交给桩的属主。
若 `rollbus.attribution` 相关断言（Step 18 之后）报 `stamped=0`，说明桩的 `game.messages` 不是由 `ctx.messages` 支撑的：把 Step 9 里那行 `globalThis.game.messages ??= …` 改成显式 `globalThis.game.messages = { contents: stub.messages };`，并在旁边写明「这是给桩补 alienrpg 侧的读法，不是改桩文件」。

- [ ] **Step 17: 提交副作用层**

```bash
git add scripts/kernel/rollbus.mjs test/rollbus.test.mjs && git commit -m "feat(kernel): 包裹四个掷骰入口并在 preCreateChatMessage 盖上掷骰记录（K1）

系统全程不发任何掷骰钩子。汇点是 game.alienrpg.yze.yzeRoll —— 22 个调用点全部
零解构地写作 yze.yzeRoll(...)，且各处 import 拿到的是同一个类对象，所以 libWrapper
用 WRAPPER 换掉这一个静态属性即可全覆盖。但汇点丢掉了下游必需的三样东西，它们
只在上游函数的入参里存在，所以按契约 §4 K1 一共包四处：

- game.alienrpg.yze.yzeRoll（YZEDiceRoller.mjs:31）：压栈、armed 标志、绑定上下文帧。
- CONFIG.Actor.documentClass.prototype.abilityRoll（actor.mjs:187）：采第 0 参的
  actor 与 dataset.attr。该值在 :194 读进局部变量后即丢，技能路径只给 yzeRoll
  传 9 个参数（:305-315）。
- CONFIG.Item.documentClass.prototype.roll（item.mjs:39）：采 this.uuid 与 this.actor。
- CONFIG.Actor.documentClass.prototype.pushRoll（actor.mjs:1302）：采被推那条消息的
  记录 id 与推骰次数；施动身份直接继承父记录的 actorUuid/tokenUuid —— :1315 交给
  yzeRoll 的是基础 actor id，重新解析会把三只非链接 Drone 塌回同一个基础 actor。

这四个 libWrapper 目标由本模块独占：lib-wrapper 对同一目标的重复注册会抛错，
第二家会静默变成空操作，所以另开 addStage(target, {id, order, around}) 让修复与
特性排队介入。队列在调用时才读，因此在 install() 之前登记同样有效。

上下文帧是一个栈，且绑定必须在 yzeRoll 被调用的那一瞬间同步完成、汇点帧持有 ctx
对象的引用：上游三个入口里有三处（actor.mjs:305、item.mjs:114/130/175…）根本不
await 汇点，卡片建出来时它们的帧早就出栈了。绑定按 actor id 匹配、匹配不上取栈顶，
正好覆盖载具的驾驶员（actor.mjs:235）与飞船炮位的开火船员（item.mjs:399）。

六条载重约束按源码逐条处理：

- 不能 await 原函数后再盖 flag：它在 :416 自建 ChatMessage、:417 返回 undefined。
  改为在 preCreateChatMessage 里用 document.updateSource 盖，此时 buildChat 已跑完，
  rollArr 数据完整；禁止 message.update() 回填。
- 两个不建消息的提前返回（:120-121 NoAttribute、:137-141 NoDice）：栈帧在内层
  Promise 落定后无条件弹出，没有消息也不留残帧。
- 重入：自动恐慌在 :198-242 不 await 地调 rollResolve/rollPanic，恐慌卡先于触发卡
  创建。归属改用调用栈 + :62 那行的内容指纹（data-item-id/data-actor-id 全系统
  仅此一处），取最深一个未认领且指纹相符的帧。
- 快照同步：preCreateChatMessage 处理函数内全程无 await，第一句就把 rollArr 浅拷
  出来；内层落定后再同步取一份比对，不一致即计一次竞态并告警。
- pools 一律取 rollArr 而不是 args：补给骰在 :154-157 会被系统钳到 6。
- 处理函数整体包 try/catch —— 在 preCreateChatMessage 里抛异常会打断系统自己的
  聊天卡。

label 的 i18n 键靠 record.mjs 的 LABEL_KEYS 反查还原，绝不反解译文；键表全仓只有
record.mjs 一份，本模块只 import。actor 一律经 resolver 解析成 uuid，绝不
game.actors.get(speaker.actor)；无法诚实归属 token 时 tokenUuid 如实记 null。

对外发 aea.rollResolved(record, message)，后续特性一律订阅这一个接缝。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 18: 写下三条自检条目的失败测试**

Foundry 里真正跑起来的正确性（真实 libWrapper 链、真实 Babele 译名、真实并发）桩模拟不了，按契约 §0.4 登记进 `selftest`。追加到 `describe("rollBus.install")` 的**内部末尾**（跟 `describe("addStage")` 并列），复用同一套 `beforeEach`：

```js
  describe("self-test entries", () => {
    it("registers three runnable entries and they pass in a healthy world", async () => {
      scripted.push(SKILL_RESULT);
      await rawRoll();
      const byId = Object.fromEntries((await selftest.runAll()).map((r) => [r.id, r]));
      expect(byId["rollbus.wrappers"]?.ok).toBe(true);
      expect(byId["rollbus.wrappers"].detail).toContain("failed=[]");
      expect(byId["rollbus.wrappers"].detail).toContain("openFrames=0");
      expect(byId["rollbus.labelIndex"]?.ok).toBe(true);
      expect(byId["rollbus.labelIndex"].detail).toContain("source=i18nInit");
      expect(byId["rollbus.attribution"]?.ok).toBe(true);
      expect(byId["rollbus.attribution"].detail).toContain("stamped=1");
      expect(byId["rollbus.attribution"].detail).toContain("snapshotRaces=0");
    });

    it("says so when nothing was injected and it had to build the index itself", async () => {
      rollBus.setLabelIndex(null);
      const entry = (await selftest.runAll()).find((r) => r.id === "rollbus.labelIndex");
      expect(entry.detail).toContain("source=lazy");
    });

    it("reports a label that no longer reverses instead of silently mis-classifying", async () => {
      // What a Babele layer that translates two keys to one string looks like from here: the
      // colliding text maps to null in the index, so the key cannot be recovered.
      rollBus.setLabelIndex({ "unrelated text": "AEA.nope" });
      const entry = (await selftest.runAll()).find((r) => r.id === "rollbus.labelIndex");
      expect(entry.ok).toBe(false);
      expect(entry.detail).toContain("ALIENRPG.Radiation");
    });
  });
```

- [ ] **Step 19: 跑它，看它失败**

Run: `npx vitest run test/rollbus.test.mjs -t "self-test entries"`
Expected: FAIL —— 三个用例全挂，第一个报 `TypeError: Cannot read properties of undefined (reading 'ok')`（`byId["rollbus.wrappers"]` 不存在，还没有登记任何条目）。

- [ ] **Step 20: 写出自检注册并在 `install()` 里调用**

在 `scripts/kernel/rollbus.mjs` 里，`WRAP_TARGETS` 之后、`export const rollBus` 之前插入：

```js
/**
 * The Foundry-dependent half of K1's verification. CONTRACT §0.4: what the test stub cannot model
 * faithfully — the real libWrapper chain, real translations, real concurrency — gets a self-test
 * entry a GM can run on demand in a live world. `label` is an i18n KEY, never localized text:
 * the runner localizes it when it reports.
 */
function registerSelfTests() {
  selftest.register({
    id: "rollbus.wrappers",
    label: "AEA.selftest.rollbus.wrappers",
    run() {
      const leaked = state.stack.length + state.ctxStack.length;
      return {
        ok: state.wrapFailures.length === 0 && leaked === 0,
        detail: `failed=[${state.wrapFailures.join(", ")}] openFrames=${leaked}`,
      };
    },
  });

  selftest.register({
    id: "rollbus.labelIndex",
    label: "AEA.selftest.rollbus.labelIndex",
    run() {
      // The Babele guard: if a translation layer collapses two keys onto one string, the reverse
      // map reports the ambiguity here instead of silently mis-classifying radiation cards.
      const index = labelIndex();
      const probes = ["ALIENRPG.Armor", "ALIENRPG.Radiation", "ALIENRPG.RadiationReduced"];
      const bad = probes.filter((key) => {
        try {
          return pureReverseLabelKey(game.i18n.localize(key), index) !== key;
        } catch {
          return true;
        }
      });
      const size = Object.keys(index).length;
      return {
        ok: bad.length === 0,
        detail: bad.length
          ? `ambiguous or missing: ${bad.join(", ")} (source=${state.labelIndexSource})`
          : `${size} labels indexed (source=${state.labelIndexSource})`,
      };
    },
  });

  selftest.register({
    id: "rollbus.attribution",
    label: "AEA.selftest.rollbus.attribution",
    run() {
      const messages = game.messages?.contents ?? [];
      const ids = new Set();
      let stamped = 0;
      let dupes = 0;
      let foreign = 0;
      for (const message of messages) {
        const record = rollBus.recordOf(message);
        if (!record) continue;
        stamped += 1;
        if (ids.has(record.id)) dupes += 1;
        else ids.add(record.id);
        if (typeof message.content !== "string" || !message.content.startsWith(ROLL_CONTENT_PREFIX)) foreign += 1;
      }
      return {
        ok: dupes === 0 && foreign === 0 && state.raceWarnings === 0,
        detail: `stamped=${stamped} duplicateIds=${dupes} nonRollCards=${foreign} snapshotRaces=${state.raceWarnings}`,
      };
    },
  });
}
```

并在 `rollBus.install()` 里、两个 `Hooks.on(...)` 之后追加一行：

```js
    registerSelfTests();
```

（自检条目登记在 `install()` 而不是 init 阶段：契约 §4 K1 给 `rollBus` 的导出面只有六个成员、没有 `init()`，而模块顶层不做安装；`runAll()` 一律在 ready 之后按需触发，时机上没有差别。`install()` 有 `state.installed` 一次性守卫，所以不会重复登记。）

- [ ] **Step 21: 把三个自检条目名补进语言文件**

`lang/en.json` 与 `lang/cn.json` 是嵌套结构，模组自有键一律 `AEA.*` 前缀。往**已有的** `AEA` 对象里合并下面这一支；如果 `AEA.selftest` 已经被别的任务建过，就只往里加 `rollbus` 这一个子对象，**不要覆盖同级已有的键**。

`lang/en.json`：
```json
    "selftest": {
      "rollbus": {
        "wrappers": "RollBus: four wrap targets installed, no leaked frames",
        "labelIndex": "RollBus: localized labels reverse-map to their i18n keys",
        "attribution": "RollBus: every stamped card is a roll card and every record id is unique"
      }
    }
```

`lang/cn.json`：
```json
    "selftest": {
      "rollbus": {
        "wrappers": "掷骰总线：四个包裹目标已装，没有泄漏的调用帧",
        "labelIndex": "掷骰总线：译名能反查回 i18n 键",
        "attribution": "掷骰总线：每张盖章的卡都是掷骰卡，记录 id 唯一"
      }
    }
```

- [ ] **Step 22: 跑它，看它通过**

Run: `npx vitest run test/rollbus.test.mjs`
Expected: PASS，42 个用例全绿。
Run: `npm test`
Expected: 整套绿。

- [ ] **Step 23: 提交自检与语言键**

```bash
git add scripts/kernel/rollbus.mjs test/rollbus.test.mjs lang/en.json lang/cn.json && git commit -m "feat(kernel): RollBus 登记三条自检条目

桩模拟不了的三件事各配一条可在真实世界里按需跑的条目（契约 §0.4）：

- rollbus.wrappers：四个包裹目标是不是真的装上了（lib-wrapper 被 GM 关掉时
  install() 不抛 ReferenceError，而是把四个目标全记进 failed 列表），以及有没有
  泄漏的栈帧。
- rollbus.labelIndex：译名反查有没有撞车。Babele 之类的翻译层若把两个键翻成同
  一段文本，反查表会把该文本标成歧义，这里报出来，而不是默默把辐射卡认成护甲卡。
  detail 里带 source=i18nInit/lazy，可据此判断 main.mjs 的 i18nInit 接线跑没跑。
- rollbus.attribution：每张盖了 flag 的卡是不是真的掷骰卡、记录 id 是否唯一、
  有没有出现快照竞态。

label 一律存 i18n 键（AEA.selftest.rollbus.*）而不是译文，本地化由 runAll() 负责，
三个键同时补进 lang/en.json 与 lang/cn.json。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 24: 接进 `main.mjs` —— import 锚点两行**

`scripts/main.mjs` 已由更早的任务建好，顶部有这一行逐字锚点：

```js
/* AEA-ANCHOR: imports */
```

在它的**下一行**插入两行：

```js
import { rollBus } from "./kernel/rollbus.mjs";
import { buildLabelIndex, LABEL_KEYS } from "./kernel/record.mjs";
```

（`buildLabelIndex` 与 `LABEL_KEYS` 都出自 `record.mjs` —— 契约 §4 K1 [v3.1] 把这张键表判给 record.mjs，全仓只此一份。若 `./kernel/record.mjs` 那行 import 已经在，就不要写第二遍，只把缺的符号补进已有的花括号里。）

- [ ] **Step 25: 接进 `main.mjs` —— 把 `api` 里属于本任务的那个 `null` 换掉**

文件里有这个字面量（契约 §5 [v3.1] 定死八个键、初始全为 `null`）：

```js
export const api = {
  features: null, patches: null, resolver: null, registry: null,
  rollBus: null, diceBarrier: null, cards: null, selftest: null,
};
```

把第二行里的 `rollBus: null,` 改成 `rollBus,`（ES 简写，值就是上一步 import 进来的对象）。改完那一行长这样：

```js
  rollBus, diceBarrier: null, cards: null, selftest: null,
```

**只动 `rollBus` 这一个槽**：不得替换 `api` 这个对象本身，不得增删键，不得碰另外七个槽（它们各有属主任务，你看到的 `null` 是别人还没做，不是缺陷）。

- [ ] **Step 26: 接进 `main.mjs` —— i18nInit 一行**

文件里这个钩子目前是一行：

```js
Hooks.once("i18nInit",        () => { /* AEA-ANCHOR: i18nInit */ });
```

把这一行整体展开成四行，**锚点注释逐字保留**：

```js
Hooks.once("i18nInit", () => {
  /* AEA-ANCHOR: i18nInit */
  rollBus.setLabelIndex(buildLabelIndex(game.i18n.localize.bind(game.i18n), LABEL_KEYS));
});
```

`i18nInit` 是 Foundry 在语言文件装载完、`ready` 之前触发的一次性钩子，是唯一既能拿到译文又早于任何掷骰的时机。契约 §5 的 i18nInit 段就这一行，本任务是它唯一的属主 —— 别的任务往这个锚点插东西属于越界，看到了也不要替他改。

- [ ] **Step 27: 接进 `main.mjs` —— ready 一行**

`ready` 段有四个**有序**子锚点，各有各的属主。找到属于本任务的这一行逐字文本：

```js
  /* AEA-ANCHOR: ready.rollbus */
```

在它的**下一行**插入：

```js
  rollBus.install();
```

**不要**碰 `ready.registry` / `ready.patches` / `ready.cards` 这三个锚点，也不要在这里补它们缺的调用 —— 顺序已经由四个子锚点在骨架里排死（registry → patches → rollbus → cards），你只填自己这一格。

（顺带记一笔，免得日后有人想「挪一下顺序」：libWrapper 按类型排序，WRAPPER 恒在 MIXED/OVERRIDE 外层，**与注册先后无关**；而且钳制骰池那类修复在 v3.1 之后走的是 `rollBus.addStage()`，队列在调用时才读。所以 `rollBus.install()` 排在 `patches.applyAll()` 之后不是为了包裹层次，而是为了让 `patches.status()` 的报告与实际链路一致。K1 报的骰数本来也只从 `rollArr` 取。）

- [ ] **Step 28: 确认 main.mjs 没写坏**

Run: `node --check scripts/main.mjs && npm test`
Expected: `node --check` 无输出（语法通过），`npm test` 整套绿。

- [ ] **Step 29: MANUAL VERIFICATION —— 在 Foundry 里逐条验**

自动化到此为止。自检条目能告诉你「装上了没有、译名有没有撞车、有没有张冠李戴」，但**真实的 libWrapper 链、真实的角色卡点击路径、真实的自动恐慌时序、真实的非链接 token**只有在真 Foundry 里才成立。启动 Foundry，进一个装了 `alienrpg` 4.1.13 系统、启用了 `lib-wrapper` 与本模组的世界，按 F12 开控制台，先贴这四行：

```js
const api = game.modules.get("alien-evolved-automation").api;
console.log("api.rollBus =", api.rollBus);   // 必须是对象，不能是 null
const last = (n = 1) => game.messages.contents.slice(-n).map(m => m.flags["alien-evolved-automation"]?.roll ?? null);
Hooks.on("aea.rollResolved", (r, m) => console.log("aea.rollResolved", r.kind, r.labelKey, r.successes, r.banes, r.tokenUuid, m.id));
```

第二行打出 `null` 就说明 Step 25 的 `api` 槽没接上，先回去补。然后依次做这九件事，每件事后在控制台核对：

1. **掷一次技能。** 打开一个压力值 ≥1 的 PC 角色卡，点某个技能（例如 HEAVY MACHINERY），对话框里直接确认。
   期望：`last()[0]` 不是 null；`v === 1`；`kind === "skill"`；`labelKey === "ALIENRPG.SkillheavyMach"`（**中文世界里同样应该是这个键，不是译文**）；`pools.base` 等于卡片上黑骰的数量、`pools.stress` 等于黄骰数量；`successes` 等于卡片上两处 "Sixes" 数字之和；`banes` 等于压力骰的 "Ones"；`push` 为 `{count:0, pushable:true, parentRollId:null}`；`actorUuid` 非 null；控制台恰好打印一条 `aea.rollResolved`。
2. **掷一次属性。** 点角色卡上的 STRENGTH／AGILITY 之类的属性按钮。
   期望：`kind === "attribute"`；`attr` 是 `"str"`／`"agl"`／`"emp"`／`"wit"` 之一。这一条专验 `abilityRoll` 包装器 —— 这个值在系统里 `actor.mjs:194` 读完就丢，汇点根本看不见，而且 `actor.mjs:305` 调汇点时**不 await**，所以它同时验证了「帧按引用捕获」这条约束。
3. **掷一次护甲。** 点角色卡上的 ARMOR 按钮。
   期望：`kind === "armor"`；`push.pushable === false`（`actor.mjs:254-256` 强制 `reRoll = true`）；`pools.stress === 0`。
4. **用一件武器攻击。** 点物品栏里一把武器的掷骰按钮。
   期望：`kind === "weapon"`；`itemUuid` 形如 `Actor.….Item.…`；`consumed.ammo === null`（一期已定：弹药子掷骰在 `YZEDiceRoller.mjs:571-611` 直接 `weapon.update()` 扣弹，从不经过 `rollArr`，二期随 ammo-economy 补）。
5. **非链接 token 的身份隔离。** 在场景里放**两个**同一基础 actor 的非链接 token（拖两次同一个 creature，token 配置里 Actor Link 关掉），分别双击打开各自的角色卡，各用同一件武器掷一次。
   期望：两条记录的 `tokenUuid` 不同，形如 `Scene.….Token.t1` / `Scene.….Token.t2`；两条的 `actorUuid` 也不同。若两条的 `tokenUuid` 都是 `null` 或相同，说明武器路径的 token 采集没生效 —— 回到 Step 9 的 `keeps two unlinked tokens…` 用例复现。
6. **掷出一颗压力 1，触发自动恐慌。** 先在 Settings → Configure Settings → System Settings 里确认 "Automatic Panic Roll" 开着。反复掷同一个技能直到压力骰出 1，此时聊天区会多出两张卡。
   期望：`last(2)` 返回 `[null, {…}]` —— **先创建的恐慌卡没有 flag，后创建的掷骰卡有**（直接验证第 4 条载重约束：归属靠调用栈 + 内容指纹，不是「最近一条消息」）。掷骰卡的 `banes >= 1`。`aea.rollResolved` 仍然只打印一条。
7. **推骰。** 在一张还带 PUSH 按钮的卡上点 PUSH。
   期望：新卡带 flag；`push.count === 1`、`push.pushable === false`、**`push.parentRollId` 等于上一张卡的 `record.id`**；若父卡是第 5 步那种 token 卡，新卡的 `tokenUuid` 应当**等于父卡的 tokenUuid**（继承生效）。注意 `successes` 只是这次新掷出的 6 —— 保留的 6 在父记录里，累计总数由消费者顺 `parentRollId` 求和，这是契约 §3 定死的语义，不是缺陷。
8. **补给检定（顺带验 pools 取的是实掷值）。** 在 PC 角色卡上点一个消耗品（AIR / POWER / FOOD / WATER）的补给按钮；挑一个当前值**大于 6** 的消耗品。
   期望：`kind === "supply"`、`pools.base === 0`、`pools.stress === 6`（系统在 `YZEDiceRoller.mjs:154-157` 把补给骰钳到 6，记录必须报实掷的 6 而不是消耗品面板上的数字）、`banes` 等于这次消耗掉的点数、`push.pushable === false`。
9. **GM 宏（证明包的是函数本身，不只是角色卡上的点击路径）。** 新建一个 script 宏（名字换成世界里真实的 PC）并运行：
   ```js
   const actor = game.actors.getName("Ripley");
   game.alienrpg.yze.yzeRoll("character", false, false, "Macro Test", 4, "Black", 0, "Yellow", actor.id);
   ```
   期望：卡片同样带 flag，`kind === "skill"`、`label === "Macro Test"`、`labelKey === null`（不在 LABEL_KEYS 里）、`pools.stress === 0`、`actorUuid` 非 null（这一条走的是 speaker → resolver 回退路径，单测覆盖不到它，只有这里能验）。

**再做两个反例**（都必须**没有** flag）：在聊天框敲 `/r 1d20`；随便右键一个 RollTable 点 Roll。若这两条也被盖上 flag，说明 `pureClaimIndex` 的前缀判定被绕过了，回到 Step 9 补用例。

**再验一次阶段口**（证明 `addStage` 在真实 libWrapper 链上也走得通，一期 b 的钳制修复要靠它）：

```js
api.rollBus.addStage("yzeRoll", { id: "manual.probe", around: (next, args) => { console.log("stage saw", args[3], args[4], args[6]); return next(args); } });
```

再掷一次技能。期望：控制台先打出一行 `stage saw <标签> <黑骰数> <黄骰数>`，然后卡片照常出现并照常带 flag。刷新页面即可移除这个临时阶段。

**最后跑自检并核对**：

```js
await api.selftest.runAll();
```

期望三条全 `ok:true`：
- `rollbus.wrappers` 的 detail 里 `failed=[]` 且 `openFrames=0`；
- `rollbus.labelIndex` 的 detail 里已索引条目数 ≥ 20，且**必须写着 `source=i18nInit`** —— 写着 `source=lazy` 就说明 Step 26 的 i18nInit 接线没跑到（检查那一行是不是插错了钩子）；
- `rollbus.attribution` 的 `stamped` 等于你本轮掷出的 yzeRoll 卡片数，且 `duplicateIds=0 nonRollCards=0 snapshotRaces=0`。

`snapshotRaces` 不为 0 说明**同一个客户端**上两次掷骰交错着把 `rollArr` 清了（`rollArr` 是客户端本地全局，别的玩家掷骰不会影响你这一份）—— 记下复现步骤，这是二期把 `rollArr` 换成按调用隔离的容器的依据，**不要**靠加 `await` 去「修」它。

- [ ] **Step 30: 提交接线**

```bash
git add scripts/main.mjs && git commit -m "feat(main): 在 i18nInit 注入 labelIndex、在 ready.rollbus 安装 RollBus

按契约 §5 的锚点接四处，全部用锚点注释文本定位而不是行号：

- AEA-ANCHOR: imports 后加两条 import：rollbus.mjs 的 rollBus，以及 record.mjs 的
  buildLabelIndex 与 LABEL_KEYS。反查键表归 record.mjs 所有，全模组只此一份。
- api 字面量里把 rollBus 那一个 null 换成导入的对象，八个键一个不增不减。
- AEA-ANCHOR: i18nInit 下一行注入反查表。表在译文装载后、任何掷骰之前构建一次。
- AEA-ANCHOR: ready.rollbus 下一行安装四个包装器。ready 段的四个子锚点已按
  registry → patches → rollbus → cards 排死，本次只填 rollbus 这一格。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```
