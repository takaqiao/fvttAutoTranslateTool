> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 11 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 11: 推骰绑回它自己那张卡（`push-correctness`）

**背景（实现者必读）**

你不需要事先了解 Foundry VTT 或 Alien RPG。下面把这条缺陷需要的全部前提讲清楚。

- **Foundry VTT** 是一个网页版桌面角色扮演平台。**system**（这里是 `alienrpg` 4.1.13）提供规则实现；**module**（我们写的 `alien-evolved-automation`）在运行时给系统打补丁。掷骰结果以**聊天卡**（chat card，一条 `ChatMessage` 文档）的形式出现在右侧聊天栏里，卡的 HTML 存在 `message.content` 里，每次界面重绘都按这份 HTML 重新渲染。
- **Alien RPG 的「推骰」（Push）规则**：多承受 1 点**压力**（Stress），多掷一颗压力骰，把**所有没出 6 的骰子**重掷；已经出 6 的保留计数。核心规则里一次掷骰只能推一次。
- 系统把「掷骰」这件事全部汇集到一个静态方法：`systems/alienrpg/module/helpers/YZEDiceRoller.mjs:31` 的 `yzeRoll(actortype, blind, reRoll, label, r1Dice, col1, r2Dice, col2, actorid, itemid, tactorid, moddata)`（12 个参数，已逐字核对）。它自己在 `:416` 调 `ChatMessage.create(chatData)` 建卡、`:417` 直接 `return`。
- 本模组的内核 K1（RollBus）在卡落库之前，把这一次掷骰的**结构化记录**（`RollRecord`）写进 `message.flags["alien-evolved-automation"].roll`。本任务全部依赖它，**永不**去解析渲染出来的文字。

**系统今天错在哪（两处，都已逐行核对 4.1.13 源码）**

1. `systems/alienrpg/module/documents/actor.mjs:1302-1328` 的 `pushRoll(actor, reRoll, hostile, blind, message)`：
   - `:1303-1312` **先**扣压力（世界设置 `game.settings.get("alienrpg", "evolved")` 为真时加 `actor.getRollData().general.addpanic.value`，否则加 1）；
   - `:1313-1314` 重掷池子却是从**当前的全局** `game.alienrpg.rollArr` 算的：
     ```js
     const reRoll1 = game.alienrpg.rollArr.r1Dice - game.alienrpg.rollArr.r1Six;
     const reRoll2 = game.alienrpg.rollArr.r2Dice + 1 - (game.alienrpg.rollArr.r2One + game.alienrpg.rollArr.r2Six);
     ```
   - `:1319` 标题取 `game.alienrpg.rollArr.tLabel`，而不是传进来的那条 `message`。
   - 这个全局在 `alienrpg.mjs:96-106` 建出，在**每次**掷骰开头被 `YZEDiceRoller.mjs:107-114` 逐字段清零，随后 `:434` 把 `tLabel` 填成**最新那次**掷骰的 label、`:443-445` 填上最新那次的骰数与 6/1 计数。所以只要你推的不是世界里最新的那张卡，就会**按最新那次掷骰的剩余骰数、顶着最新那次掷骰的标题**重掷 —— 而压力**已经先扣掉了**。
2. `systems/alienrpg/module/alienrpg.mjs:464-497` 的点击处理器：
   - `:472` 用 `game.actors.get(message.speaker.actor)` 拿角色。这是**基础 actor**。Foundry 里一个 actor 可以被拖到场景上生成多个 **token**（棋子）；若 token 是「非链接」（unlinked，`actorLink = false`），每个 token 各自持有一份 actor 数据副本。系统在 `alienrpg.mjs:366-376` 的 `preCreateToken` 里对勾了 `system.header.npc` 的角色**强制**非链接。写到基础 actor 上，五个同源 token 的压力会一起涨。
   - `:487-493` 是 `switch (actor.type)`，只有 `case "character"` 会调 `pushRoll`，`default: return`。而合成人（`synthetic`）勾了 `system.header.synthstress`（角色卡上写作 "Human Panic, Push, etc."）之后，`actor.mjs:225-226` 会把 `effectiveActorType` 改成 `"character"` 让它按人类掷骰，于是 `YZEDiceRoller.mjs:378` 的 `if (!reRoll || reRoll === "mPush")` 分支**照样画出 Push 按钮**，点下去却什么都不发生 —— 没有卡、没有报错、没有提示。

**三条必须先记住的源码事实（否则会做出错误的“修复”）**

- `RollRecord.pools.base` / `pools.stress` 是 K1 从 `game.alienrpg.rollArr.r1Dice` / `r2Dice` 取的**实掷骰数**，不是掷骰请求里的参数值（骰池钳制类补丁跑在更内层，请求值可能大于实掷值）。所以拿记录算重掷池，连被钳制过的那次掷骰也能算对。
- **系统的 MultiPush 是一个真功能，不能被本任务拒掉**：勾了卡上的 "Allow multi-push" 复选框后，`reRoll` 变成 `"mPush"`，`YZEDiceRoller.mjs:378` 对 mPush 卡**仍然画推骰按钮**（只是 `:379-385` 跳过复选框本身），`:344-357` 还会累计打印成功数总和。也就是说「已经推过一次」并不等于「不能再推」。本特性因此**只认 `record.push.pushable` 这一个字段**作为「还能不能推」的权威判据，**不**自作主张用 `push.count > 0` 去拦。`pushable` 由 K1 RollBus 在建记录时产出（`mPush` 卡恒为 `true`，普通 `push` 产物为 `false`），本任务**只消费、不重算**。
- `yzeRoll` 在 `:100`（清零之前）读一次 `const oldRoll = rollArr.r1Six + rollArr.r2Six`，用来在推骰卡上打印累计成功数。也就是说这个全局有**两个**读者：`pushRoll:1313-1319` 与 `yzeRoll:100`。

**为什么修法是「把系统要读的那个全局按本卡记录填对」，而不是「整个替换掉 pushRoll」**

本模组的内核 K1（RollBus）**独占**四个 libWrapper 目标（`yzeRoll`、`abilityRoll`、`itemRoll`、`pushRoll`），任何别的代码再去 `libWrapper.register` 同一个目标，lib-wrapper 1.13.5 会抛 `A wrapper for '<target>' (ID=<n>) has already been registered by <module>.` —— 修复**静默变成空操作**。所以介入 `pushRoll` 的唯一合法通路是 `rollBus.addStage("pushRoll", {id, order, around})`，它的 `around(next, args, thisArg)` 语义是「改入参 → 恰好调用一次 `next(args)` → 可改返回值」。

这个语义正好配得上本缺陷的形状：**缺陷不在算法，在输入**。`:1313-1314` 的算术本身是对的（当前骰数减去已出的 6、压力骰 +1 再减去 1 和 6），错的是它从一个描述「世界里最新那次掷骰」的全局里取值。于是本阶段做的事是：委托之前把 `game.alienrpg.rollArr` 的七个字段按**这张卡的记录**填好，再把 `actor` / `reRoll` / `actortype` / `blind` 四个入参改对，然后原样交给 `next(args)`。好处有四：

1. 压力的扣法（`evolved` 与 `addpanic`）仍然只有系统一份实现，模组不复制；
2. `yzeRoll:100` 的 `oldRoll` 读的也是我们填的这份种子，累计成功数因此有了正确的来源（**注意**：那行文字的显示归另一条特性管，本任务只是不再喂给它错数据）；
3. 补丁的退休判据（`probe()`）与修法是同一个前提 —— 只有当原实现「还在读全局」时种子才有意义，上游一旦改成从 message 读，`probe()` 返回 `false`，补丁自己退休；
4. 阶段只写 `r1Dice / r1One / r1Six / r2Dice / r2One / r2Six / tLabel` 七个键，**不碰** `sCount` 与 `multiPush`（`:357` 的 mPush 累计位），不越界到成功数显示的地盘。

唯一不调用 `next` 的分支是**否决**：记录缺失、记录说不可推、或普通合成人。理由是系统 `:1303-1312` **先扣压力再算池子**，一旦委托进去，压力已经扣掉，之后再发现没有可用的池子就晚了 —— 而「扣了压力却什么也没重掷」比原缺陷更糟。否决时给出一条黄色提示并返回 `null`。

**本任务的范围边界**

- 推骰卡上那行「Following the Push, You have a Total of N successes」（`YZEDiceRoller.mjs:344-372`）的**显示**由另一条特性负责，**本任务不动它**，也不写 `rollArr.multiPush`。手工验收时看到那行数字不对，不是本任务的缺陷。
- `RollRecord.push.parentRollId` 在一期**只产出、不消费**：系统已经在卡上打印累计计数，本特性不再重复一份推骰谱系。不要因为“没人用”就把它当死字段删掉。
- 特性开关（`off`）只收回**模组自己画的控件**；规则纠正（阶段）不随开关退出，它由补丁内核按版本区间与 `probe()` 管理。理由：关掉开关会让系统那颗按钮重新可见，若那时阶段也退出，点它就又回到「按最新那次掷骰重掷」的错误行为 —— 开关是 UI 的开关，不是规则损坏的开关。（契约 §7 那条「关掉即原样放行」约束的是**一期 b 的修复包**，本任务是一期特性，其开关语义在此显式定死。）

**名词解释**

- **libWrapper**：Foundry 生态的函数包裹库，全局名 `libWrapper`。同一个 `模组 + 目标` 只能注册一次。本任务**一行 `libWrapper` 都不写**，全部经 `rollBus.addStage` 排队。
- **stage（阶段）**：`rollBus` 在它独占的那一层包裹里维护的有序链。`around(next, args, thisArg)` 里 `args` 是**原参数组成的数组**，`next(args)` 把（可能被你改过的）数组交给链上的下一层，最终到达系统原方法；`order` 小的更靠外，同 `order` 按注册顺序。
- **probe()**：本模组补丁的自退休机制。返回 `true` 表示「缺陷仍在，该装补丁」，返回 `false` 表示上游已修，补丁退休并提示 GM 一次。
- **unlinked token**：见上。判断「该给谁加压力」必须走 `resolver`，绝不用 `game.actors.get(speaker.actor)`。

**修复策略（三条，缺一不可）**

1. **补丁**：`patches.register` 一条 `type: "WRAPPER"` 的补丁，它的 `apply()` 是无参自装器，里面**只**调 `rollBus.addStage("pushRoll", {id, order: 0, around: pushStage})`。`pushStage` 先按本卡记录算出计划，否决时不委托；通过时填种子、改四个入参、`next(args)` 恰好一次。
2. **卡片**：在模组自有挂点下**本特性自己的分区元素**里画自己的 Push 按钮（`data-action="aea-push"`），并把系统那颗按钮和它的 multiPush 复选框 `display:none`。这样合成人也能推，且**不会双绑** —— 系统的监听器还挂在一颗看不见的按钮上，点不到（若只抑制监听器不隐藏按钮，一次点击会推两次、扣两份压力）。**只在系统本来就画了 Push 按钮的卡上替换**，所以绝不会在不该推的卡上凭空多出按钮。
3. **点击一律经 `actor.pushRoll(...)` 进入，绝不直接调阶段函数**。内核 K1 在 `pushRoll` 上的那层包裹是 `push.parentRollId` 与 `push.count` 的**唯一采集点**，阶段链跑在它的里面；绕过 `pushRoll` 直接调阶段，新卡的这两个字段全是空的。

**Files:**
- Create: `scripts/features/push-correctness.pure.mjs`
- Create: `scripts/features/push-correctness.mjs`
- Create: `templates/push-controls.hbs`
- Modify: `lang/en.json`（并入顶层 `AEA` 对象）
- Modify: `lang/cn.json`（并入顶层 `AEA` 对象）
- Modify: `scripts/main.mjs`（**只加两行**：`/* AEA-ANCHOR: imports */` 之后一行 `import`，`/* AEA-ANCHOR: features */` 之后一行数组成员；**不加任何生命周期调用，不加任何 `Hooks.on`，不碰其余八个锚点**）
- Test: `test/push-correctness.pure.test.mjs`
- Test: `test/push-correctness.wiring.test.mjs`
- Test: `test/push-correctness.main-wiring.test.mjs`
- 只读参照（**不要改动系统文件**）：`C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg/module/documents/actor.mjs:1302-1328`、`:225-226`、`.../module/data/actor-synthetic.mjs:32`、`:61`、`:238`、`.../module/alienrpg.mjs:96-106`、`:366-376`、`:464-497`、`.../module/helpers/YZEDiceRoller.mjs:31`、`:76-90`、`:100`、`:107-114`、`:126`、`:344-357`、`:378-391`、`:398-401`、`:416-417`、`:434`、`:443-445`

**Interfaces:**

- Consumes:
  - `scripts/const.mjs` → `MID = "alien-evolved-automation"`
  - `scripts/kernel/features.mjs` → `features.register(def)`、`features.enabled(id) -> boolean`、`features.all() -> def[]`
  - `scripts/kernel/patches.mjs` → `patches.register(def)`、`patches.status() -> [{id, type, target, applied, reason, fixedIn}]`
    def 语义：`{id, type:"WRAPPER"|"MIXED"|"OVERRIDE"|"DATA"|"HOOK", target:string|null, minSystem, fixedIn, probe(), apply()}`；
    **`target` 与 `type` 只是给 `status()` 展示用的元数据，`applyAll()` 不据此注册**；`apply()` 是**无参自装器**。
  - `scripts/kernel/rollbus.mjs` →
    - `rollBus.recordOf(message) -> RollRecord|null`
    - `rollBus.addStage(target, {id, order = 0, around})`：`target` 取 `"yzeRoll" | "abilityRoll" | "itemRoll" | "pushRoll"`；`around(next, args, thisArg) -> any`，`args` 是原参数数组，**必须恰好调用一次 `next(args)`**（本任务的否决分支是唯一例外，见上文「为什么修法是…」）；`order` 小的先执行（更靠外）。**这四个目标由 rollBus 独占 libWrapper 注册权，任何别处再 `libWrapper.register` 会被抛错拒绝并静默失效。**
  - `scripts/kernel/resolver.mjs` → `resolver.actorOf(record) -> Actor|null`、`resolver.fromSpeaker(speaker) -> Actor|null`、`resolver.soleToken(actor) -> Token|null`（场景上恰好只有一个活动 token 时返回那个**放置 token**，其 `.document.disposition` 才是本场景的实际阵营；否则返回 `null`，不猜）
  - `scripts/kernel/cards.mjs` →
    - `cards.onRender(name, handler)`：**特性拿到挂点的唯一通路**。`handler({message, element, mount, record})`，其中 `record` 是 `rollBus.recordOf(message)` 的结果（可能为 `null`）；cards 在自己独占的渲染钩子里按注册顺序**同步**调用各 handler，异常被 cards 捕获后继续调下一个；`mount` 由 cards **惰性创建** —— 只有 handler 第一次**读取这个属性**时才插入 DOM。
    - `cards.render(target, templatePath, data) -> Promise`：**只替换 `target` 自身的 `innerHTML`**，绝不触碰父节点、兄弟节点或挂点上的其它内容；内部 `await renderTemplate` 后写入。**必须传本特性自建的分区子元素，禁止直接传 `mount`** —— 一期有两条特性共用同一个挂点，直接对 `mount` 渲染会清掉另一条特性的内容。分区类名按契约取 `aea-<feature-id>`，本特性即 `aea-push-correctness`。
    - `cards.registerAction(name, handler)`，`handler(event, {action, message, messageId, element, mount})`。
  - `scripts/kernel/selftest.mjs` → `selftest.register({id, label, run})`，`run() -> {ok:boolean, detail:string}`
  - RollRecord v1 字段：`v`、`id`、`label`、`pools.{base,stress}`、`results.{baseSixes,baseOnes,stressSixes,stressOnes}`、`push.{count,pushable,parentRollId}`、`actorUuid`、`tokenUuid`
  - 系统全局：`game.alienrpg.rollArr`（`alienrpg.mjs:96-106` 建出的可变全局）
  - 系统 i18n 键（沿用系统原文，不重造）：`ALIENRPG.NoToken`
  - `test/stubs/foundry.mjs` → `installFoundryStub(options) -> ctx`、`uninstallFoundryStub()`、`foundryStubContext() -> ctx|null`（`ctx.wrappers` 记录桩上发生过的 `libWrapper.register`）。**只读不改，禁止在自己的测试里就地造 `globalThis.game` 之类的全局。**
- Produces:
  - `purePushPlan({record, actorType, synthstress, multiPush}) -> {ok, reason, reRoll?, label?, rollId?, seed?}`，其中 `seed = {r1Dice, r1One, r1Six, r2Dice, r2One, r2Six, tLabel}`
  - `purePushRollIsBuggy(source) -> boolean`
  - `hideSystemPushControls(root) -> {had:boolean, hadMulti:boolean}`（副作用层，碰 DOM，故意不带 `pure` 前缀）
  - `PUSH_FEATURE_ID = "push-correctness"`、`PUSH_PATCH_ID = "push-roll-binds-to-its-own-card"`、`PUSH_TARGET`、`PUSH_STAGE`、`PUSH_I18N_KEYS`
  - `pushFeature = { id, register(), install() }`
  - `templates/push-controls.hbs`
  - 三条 selftest 条目：`push-correctness.stage-live`、`push-correctness.controls-hidden`、`push-correctness.i18n`
  - `test/push-correctness.pure.test.mjs`（15 例）、`test/push-correctness.wiring.test.mjs`（7 例）、`test/push-correctness.main-wiring.test.mjs`（4 例）
  - 阶段函数 `pushStage(next, args, thisArg)` 是**模块内私有**，不导出：它唯一的合法入口是 `rollBus` 的阶段链，导出会诱使别处直接调用而绕过 K1 的采集点。

**DOM 断言的去向**（本仓库不装 jsdom，vitest 跑不了真 DOM；下表让每一条丢掉的断言都有明确接管人，删掉任何一条 selftest 条目都会在这张表上留下孤行）

| 在 vitest 里跑不了的断言 | 接管人 |
|---|---|
| 系统那颗推骰按钮被隐藏后不可点（不双绑、不双扣压力） | selftest `push-correctness.controls-hidden` + 手工 D.4 |
| 补丁登记了、apply 过了、阶段真的排进了 rollBus 的 pushRoll 链 | selftest `push-correctness.stage-live` + 手工 A.1 / A.3 |
| 种子真的生效（推旧卡按旧卡的池子与标题重掷） | 手工 B.3 |
| 运行时 i18n 键真的存在（不会在按钮上印出键名） | selftest `push-correctness.i18n` + `test/push-correctness.wiring.test.mjs` 读 `lang/*.json` |
| 挂点里只出现一份推骰控件（重复渲染幂等） | 手工 A.4 |
| mPush 卡上按钮仍在、复选框不在 | 手工 B.5 |
| 特性关掉后系统控件重新可见、规则纠正仍在 | 手工 D.5 |

---

- [ ] **Step 1: 写下纯函数层的失败测试**

新建 `test/push-correctness.pure.test.mjs`。这一层完全不碰 Foundry 全局，是真单测。两处刻意的设计：`seed` 的断言之外再单独写一条「用系统 `:1313-1314` 的公式算一遍」，让种子与系统算术的配合有据可查；`purePushRollIsBuggy` 的**两个方向都要测**，只测「返回 true」的探针等于永远装补丁，上游修好后会双重修复。

```js
import { describe, it, expect } from "vitest";
import { purePushPlan, purePushRollIsBuggy } from "../scripts/features/push-correctness.pure.mjs";

/** 一条完整的 v1 RollRecord，按 CONTRACT §3 逐字段写全 */
function makeRecord(over = {}) {
  return {
    v: 1,
    id: "roll-old",
    actorUuid: "Actor.abc",
    tokenUuid: "Scene.s1.Token.t1",
    userId: "user-1",
    kind: "skill",
    label: "Heavy Machinery",
    labelKey: null,
    attr: null,
    itemUuid: null,
    pools: { base: 5, stress: 2 },
    results: { baseSixes: 1, baseOnes: 0, stressSixes: 1, stressOnes: 1 },
    successes: 2,
    banes: 1,
    push: { count: 0, pushable: true, parentRollId: null },
    targets: [],
    consumed: { ammo: null },
    at: { worldTime: 0, real: 0 },
    ...over,
  };
}

describe("purePushPlan", () => {
  it("种子的七个字段全部来自这张卡的记录", () => {
    const plan = purePushPlan({ record: makeRecord(), actorType: "character" });
    expect(plan.ok).toBe(true);
    expect(plan.seed).toEqual({
      r1Dice: 5,
      r1One: 0,
      r1Six: 1,
      r2Dice: 2,
      r2One: 1,
      r2Six: 1,
      tLabel: "Heavy Machinery",
    });
  });

  it("按系统 actor.mjs:1313-1314 的公式，种子算出的正是这张卡的剩余骰", () => {
    const { seed } = purePushPlan({ record: makeRecord(), actorType: "character" });
    // Verbatim from systems/alienrpg/module/documents/actor.mjs:1313-1314.
    const reRoll1 = seed.r1Dice - seed.r1Six;
    const reRoll2 = seed.r2Dice + 1 - (seed.r2One + seed.r2Six);
    expect(reRoll1).toBe(4); // 5 - 1
    expect(reRoll2).toBe(1); // 2 + 1 - (1 + 1)
  });

  it("标题取这张卡记录里的 label，不来自任何全局", () => {
    const plan = purePushPlan({ record: makeRecord({ label: "Observation" }), actorType: "character" });
    expect(plan.label).toBe("Observation");
    expect(plan.seed.tLabel).toBe("Observation");
    expect(plan.rollId).toBe("roll-old");
  });

  it("记录自相矛盾时把种子钳到算不出负骰数", () => {
    const { seed } = purePushPlan({
      record: makeRecord({
        pools: { base: 2, stress: 1 },
        results: { baseSixes: 3, baseOnes: 0, stressSixes: 1, stressOnes: 2 },
      }),
      actorType: "character",
    });
    expect(seed.r1Dice - seed.r1Six).toBe(0);
    expect(seed.r2Dice + 1 - (seed.r2One + seed.r2Six)).toBe(0);
  });

  it("勾了多推时 reRoll 变成 mPush", () => {
    expect(purePushPlan({ record: makeRecord(), actorType: "character" }).reRoll).toBe("push");
    expect(purePushPlan({ record: makeRecord(), actorType: "character", multiPush: true }).reRoll).toBe("mPush");
  });

  it("没有记录的旧卡拒绝推骰，且不会给出种子", () => {
    const plan = purePushPlan({ record: null, actorType: "character" });
    expect(plan.ok).toBe(false);
    expect(plan.reason).toBe("no-record");
    expect(plan.seed).toBeUndefined();
  });

  it("将来版本的记录（v 不是 1）按无记录处理，不拿新 schema 硬算旧公式", () => {
    const plan = purePushPlan({ record: makeRecord({ v: 2 }), actorType: "character" });
    expect(plan.ok).toBe(false);
    expect(plan.reason).toBe("no-record");
  });

  it("pushable 为 false 的卡拒绝再推", () => {
    const plan = purePushPlan({
      record: makeRecord({ push: { count: 1, pushable: false, parentRollId: "roll-0" } }),
      actorType: "character",
    });
    expect(plan.ok).toBe(false);
    expect(plan.reason).toBe("not-pushable");
  });

  it("已经推过但 pushable 仍为 true 的 mPush 卡可以再推（系统 :378 对 mPush 卡照样画按钮）", () => {
    const plan = purePushPlan({
      record: makeRecord({ push: { count: 1, pushable: true, parentRollId: "roll-0" } }),
      actorType: "character",
    });
    expect(plan.ok).toBe(true);
    expect(plan.seed.r1Dice - plan.seed.r1Six).toBe(4);
  });

  it("普通合成人不推骰", () => {
    const plan = purePushPlan({ record: makeRecord(), actorType: "synthetic", synthstress: false });
    expect(plan.ok).toBe(false);
    expect(plan.reason).toBe("synthetic");
  });

  it("勾了 synthstress 的合成人可以推骰（系统今天这里是死按钮）", () => {
    const plan = purePushPlan({ record: makeRecord(), actorType: "synthetic", synthstress: true });
    expect(plan.ok).toBe(true);
    expect(plan.seed.r1Dice - plan.seed.r1Six).toBe(4);
  });
});

describe("purePushRollIsBuggy", () => {
  // Verbatim shape of alienrpg 4.1.13, module/documents/actor.mjs:1313-1319.
  const BUGGY = `async pushRoll(actor, reRoll, hostile, blind, message) {
    const reRoll1 = game.alienrpg.rollArr.r1Dice - game.alienrpg.rollArr.r1Six;
    const reRoll2 = game.alienrpg.rollArr.r2Dice + 1 - (game.alienrpg.rollArr.r2One + game.alienrpg.rollArr.r2Six);
    await yze.yzeRoll(hostile, blind, reRoll, game.alienrpg.rollArr.tLabel, reRoll1);
  }`;

  // The shape an upstream fix would take: pools and title come from the message.
  const FIXED = `async pushRoll(actor, reRoll, hostile, blind, message) {
    const rec = message.getFlag("alienrpg", "roll");
    const reRoll1 = rec.pools.base - rec.results.baseSixes;
    await yze.yzeRoll(hostile, blind, reRoll, rec.label, reRoll1);
  }`;

  it("4.1.13 的实现被判为「缺陷仍在」", () => {
    expect(purePushRollIsBuggy(BUGGY)).toBe(true);
  });

  it("从 message 取池子与标题的实现被判为「上游已修」，补丁应退休", () => {
    expect(purePushRollIsBuggy(FIXED)).toBe(false);
  });

  it("只剩一半特征（改了标题没改池子）仍判为已修，宁可漏装不要双修", () => {
    expect(purePushRollIsBuggy(BUGGY.replace("game.alienrpg.rollArr.tLabel", "rec.label"))).toBe(false);
  });

  it("拿不到源码时返回 false", () => {
    expect(purePushRollIsBuggy(null)).toBe(false);
    expect(purePushRollIsBuggy(undefined)).toBe(false);
  });
});
```

- [ ] **Step 2: 跑它，看它失败**

Run: `npx vitest run test/push-correctness.pure.test.mjs`
Expected: FAIL —— `Error: Failed to load url ../scripts/features/push-correctness.pure.mjs (resolved id: ...). Does the file exist?`，`Test Files  1 failed`，15 个用例一个都没执行。

- [ ] **Step 3: 实现纯函数层**

新建 `scripts/features/push-correctness.pure.mjs`。此文件**只导出 `pure*` 函数**，**不 import 任何东西**，不引用任何 Foundry 全局。拒绝原因用字面量字符串，测试里逐字断言 —— 不导出常量枚举，副作用层自己做「原因 → i18n 键」的映射。

```js
/**
 * Pure layer for `push-correctness`.
 * No imports, no Foundry globals: vitest imports this file as-is.
 */

/**
 * Decide everything about a push from the record on the card being pushed.
 *
 * The repair works by REPAIRING THE INPUT the system reads, not by replacing
 * the system's method. alienrpg's actor.mjs:1313-1319 derives the re-roll pools
 * and the card title from the mutable global `game.alienrpg.rollArr`, which by
 * then describes the newest roll in the world; `seed` is exactly what that
 * global must contain for the system's own (correct) arithmetic to describe
 * THIS card instead.
 *
 * record.pools holds the dice actually rolled — the roll bus snapshots
 * game.alienrpg.rollArr rather than the requested pool — so a clamped roll
 * re-rolls the right number of dice.
 *
 * @param {object} input
 * @param {object|null} input.record    v1 RollRecord of the card being pushed
 * @param {string|null} input.actorType "character" | "synthetic" | ...
 * @param {boolean} [input.synthstress] actor.system.header.synthstress
 * @param {boolean} [input.multiPush]   the extra-push checkbox was ticked
 * @returns {{ok:boolean, reason:string|null, reRoll?:"push"|"mPush", label?:string,
 *            rollId?:string, seed?:{r1Dice:number, r1One:number, r1Six:number,
 *            r2Dice:number, r2One:number, r2Six:number, tLabel:string}}}
 */
export function purePushPlan({ record, actorType = null, synthstress = false, multiPush = false }) {
  if (!record || record.v !== 1) return { ok: false, reason: "no-record" };

  // `pushable` is the record's single authority on "can this still be pushed".
  // Deliberately NOT also refusing on push.count > 0: the system's MultiPush
  // option re-draws the push button on an mPush card (YZEDiceRoller.mjs:378) and
  // keeps a running success total (:344-357), so a second push is legitimate.
  if (record.push?.pushable === false) return { ok: false, reason: "not-pushable" };

  if (actorType === "synthetic" && !synthstress) return { ok: false, reason: "synthetic" };

  const pools = record.pools || {};
  const res = record.results || {};
  const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);

  // Clamp so the system's own subtractions can never go negative on a damaged
  // record: yzeRoll would end up building `new Roll("-1db")` out of it.
  const r1Dice = Math.max(0, num(pools.base));
  const r1Six = Math.min(Math.max(0, num(res.baseSixes)), r1Dice);
  const r2Dice = Math.max(0, num(pools.stress));
  const r2Six = Math.min(Math.max(0, num(res.stressSixes)), r2Dice);
  // actor.mjs:1314 computes r2Dice + 1 - (r2One + r2Six); keep that >= 0.
  const r2One = Math.min(Math.max(0, num(res.stressOnes)), r2Dice + 1 - r2Six);

  return {
    ok: true,
    reason: null,
    reRoll: multiPush ? "mPush" : "push",
    label: record.label,
    rollId: record.id,
    seed: {
      r1Dice,
      r1One: Math.max(0, num(res.baseOnes)),
      r1Six,
      r2Dice,
      r2One,
      r2Six,
      tLabel: record.label,
    },
  };
}

/**
 * The retirement predicate for the pushRoll patch, split out so both branches
 * are unit-testable: an effect-layer probe that inspects the live installed
 * system can only ever observe the buggy 4.1.13 and would never prove that a
 * fixed upstream retires the patch.
 *
 * The defect IS "it reads the mutable global instead of the message", so both
 * the pool arithmetic and the title lookup must still be global-sourced for the
 * seed-the-global repair to be worth installing at all.
 *
 * @param {string|null|undefined} source Function.prototype.toString of pushRoll
 */
export function purePushRollIsBuggy(source) {
  if (typeof source !== "string") return false;
  return source.includes("rollArr.r1Dice") && source.includes("rollArr.tLabel");
}
```

- [ ] **Step 4: 跑它，看它通过**

Run: `npx vitest run test/push-correctness.pure.test.mjs`
Expected: PASS —— `Test Files  1 passed`、`Tests  15 passed`。

- [ ] **Step 5: 提交纯函数层**

```bash
git add scripts/features/push-correctness.pure.mjs test/push-correctness.pure.test.mjs && git commit -m "$(cat <<'EOF'
feat(push): 从卡片自己的掷骰记录算出重掷种子

系统 actor.mjs:1313-1314 从全局 game.alienrpg.rollArr 取池子、:1319 用
rollArr.tLabel 当标题，而该全局每次掷骰都被 YZEDiceRoller.mjs:107-114
清零、:434/:443-445 重填，于是推旧卡等于按最新那次掷骰重掷。

算术本身没错，错的是输入。因此纯函数层产出的不是「重掷几颗」，而是一份
seed —— 委托给系统之前该把那个全局填成什么样，让 :1313-1314 与 :100 的
oldRoll 都读到这张卡自己的数。种子按记录钳到算不出负骰数。

「还能不能推」只认 record.push.pushable 一个字段：系统的 MultiPush 会在
mPush 卡上重画按钮（:378）并累计成功数（:344-357），按 push.count 拦会把
它一起拒掉。

补丁退休判据拆成 purePushRollIsBuggy(source) 这个可注入的纯谓词，
正反两个方向都有单测 —— 只测「返回 true」的探针等于永远不退休。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: 写下会失败的接线测试**

新建 `test/push-correctness.wiring.test.mjs`。它验六件肉眼极易漏过的事：id 一致性（症状是设置面板显示一串裸 kebab-case）、补丁元数据、**`apply()` 真的走 `rollBus.addStage` 而不是 `libWrapper.register`**（症状最隐蔽：libWrapper 会因目标已被 rollBus 占用而抛错，补丁静默变空操作）、`install()` 只经 cards 的两个注册口接线、本特性没有自挂钩子也没碰 libWrapper、以及语言键完整性（症状是按钮上直接印出键名）。Foundry 全局一律用共享桩，**不得**自己造 `globalThis.game`。

```js
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { readFileSync } from "node:fs";
import { installFoundryStub, uninstallFoundryStub, foundryStubContext } from "./stubs/foundry.mjs";

// The stub must be on globalThis before the feature module (and the kernels it
// imports) are evaluated, hence top-level install + dynamic import.
installFoundryStub();
const { pushFeature, PUSH_FEATURE_ID, PUSH_PATCH_ID, PUSH_TARGET, PUSH_STAGE, PUSH_I18N_KEYS } = await import(
  "../scripts/features/push-correctness.mjs"
);
const { features } = await import("../scripts/kernel/features.mjs");
const { patches } = await import("../scripts/kernel/patches.mjs");
const { cards } = await import("../scripts/kernel/cards.mjs");
const { rollBus } = await import("../scripts/kernel/rollbus.mjs");

// Capture the patch def as it is registered: patches.status() reports metadata
// only, and this test needs to call the real apply() in isolation.
const patchDefs = [];
const realPatchRegister = patches.register;
patches.register = (def) => {
  patchDefs.push(def);
  return realPatchRegister.call(patches, def);
};

function loadLang(file) {
  return JSON.parse(readFileSync(new URL(`../lang/${file}`, import.meta.url), "utf8"));
}

function hasKey(root, dotted) {
  return dotted.split(".").reduce((node, seg) => (node == null ? undefined : node[seg]), root) !== undefined;
}

beforeAll(() => pushFeature.register());
afterAll(() => {
  patches.register = realPatchRegister;
  uninstallFoundryStub();
});

describe("pushFeature wiring", () => {
  it("特性 id 与导出的常量、注册进 features 的 def 三者一致", () => {
    expect(pushFeature.id).toBe(PUSH_FEATURE_ID);
    expect(PUSH_FEATURE_ID).toBe("push-correctness");
    const def = features.all().find((d) => d.id === PUSH_FEATURE_ID);
    expect(def).toBeDefined();
    expect(def.default).toBe("full");
    expect(def.requires).toEqual([]);
  });

  it("补丁 id 与特性 id 是两个不同的标识；type/target 只是 status() 的展示元数据", () => {
    expect(PUSH_PATCH_ID).toBe("push-roll-binds-to-its-own-card");
    expect(PUSH_PATCH_ID).not.toBe(PUSH_FEATURE_ID);
    const def = patchDefs.find((d) => d.id === PUSH_PATCH_ID);
    expect(def, "pushFeature.register() 没有登记这条补丁").toBeDefined();
    expect(def.type).toBe("WRAPPER");
    expect(def.target).toBe(PUSH_TARGET);
    expect(PUSH_TARGET).toBe("CONFIG.Actor.documentClass.prototype.pushRoll");
  });

  it("apply() 经 rollBus.addStage 排队，绝不自己调 libWrapper.register", () => {
    const def = patchDefs.find((d) => d.id === PUSH_PATCH_ID);
    const ctx = foundryStubContext();
    const wrappersBefore = ctx.wrappers.length;
    const staged = [];
    const realAddStage = rollBus.addStage;
    rollBus.addStage = (target, opts) => staged.push([target, opts]);
    try {
      def.apply();
    } finally {
      rollBus.addStage = realAddStage;
    }
    expect(staged).toHaveLength(1);
    expect(PUSH_STAGE).toBe("pushRoll");
    expect(staged[0][0]).toBe(PUSH_STAGE);
    expect(staged[0][1].id).toBe(PUSH_PATCH_ID);
    expect(typeof staged[0][1].around).toBe("function");
    // rollBus owns the libWrapper registration on this target; a second one
    // would be refused by libWrapper and the patch would silently do nothing.
    expect(ctx.wrappers.length).toBe(wrappersBefore);
  });

  it("install() 只经 cards.onRender / cards.registerAction 接线，且用约定的名字", () => {
    const renderers = [];
    const actions = [];
    const realOnRender = cards.onRender;
    const realRegisterAction = cards.registerAction;
    cards.onRender = (name) => renderers.push(name);
    cards.registerAction = (name) => actions.push(name);
    try {
      pushFeature.install();
    } finally {
      cards.onRender = realOnRender;
      cards.registerAction = realRegisterAction;
    }
    expect(renderers).toEqual(["push-controls"]);
    expect(actions).toEqual(["aea-push"]);
  });

  it("特性模块既不自挂钩子，也不碰 libWrapper（渲染钩子归 cards、四个掷骰目标归 rollBus 独占）", () => {
    const src = readFileSync(new URL("../scripts/features/push-correctness.mjs", import.meta.url), "utf8");
    expect(src).not.toMatch(/\bHooks\s*\./);
    expect(src).not.toMatch(/\blibWrapper\b/);
  });

  it.each(["en.json", "cn.json"])("%s 含有本特性用到的全部键", (file) => {
    const lang = loadLang(file);
    const missing = PUSH_I18N_KEYS.filter((k) => !hasKey(lang, k));
    expect(missing).toEqual([]);
  });
});
```

- [ ] **Step 7: 跑它，看它失败**

Run: `npx vitest run test/push-correctness.wiring.test.mjs`
Expected: FAIL —— `Error: Failed to load url ../scripts/features/push-correctness.mjs (resolved id: ...). Does the file exist?`，`Test Files  1 failed`，7 个用例未执行。

- [ ] **Step 8: 加 i18n 键**

模组自有字符串一律 `AEA.` 前缀、嵌套结构；特性的显示名与说明固定走 `AEA.feature.<id>.name` / `AEA.feature.<id>.hint`，`<id>` 逐字等于 `features.register({id})` 的 id。

在 `lang/en.json` 顶层的 `"AEA"` 对象里并入（若已有 `"feature"` 子对象则**只添子键，不要覆盖兄弟键**）：

```json
"push": {
  "button": "Push",
  "multi": "Extra push",
  "noRecord": "This card has no roll record — it was rolled before the automation module was enabled, so there is nothing to re-roll. Roll again instead.",
  "notPushable": "This roll has already been pushed and cannot be pushed again.",
  "synthetic": "Synthetics do not suffer Stress and do not push."
},
"feature": {
  "push-correctness": {
    "name": "Push binds to its own card",
    "hint": "Bind the Push button to the record on its own card instead of the last roll made in the world."
  }
}
```

在 `lang/cn.json` 顶层的 `"AEA"` 对象里并入：

```json
"push": {
  "button": "推骰",
  "multi": "额外推骰",
  "noRecord": "这张卡没有掷骰记录 —— 它是在本模组启用之前掷的，没有可重掷的池子。请重新掷一次。",
  "notPushable": "这次掷骰已经推过，不能再推。",
  "synthetic": "合成人不承受压力，也不推骰。"
},
"feature": {
  "push-correctness": {
    "name": "推骰绑回自己那张卡",
    "hint": "把推骰按钮绑回它自己那张卡的记录，而不是世界里最新的那次掷骰。"
  }
}
```

- [ ] **Step 9: 建模板**

新建 `templates/push-controls.hbs`（Handlebars 模板，Foundry 的 `renderTemplate` 会编译它）。两个字符串都不需要参数替换，所以直接用 Foundry 内置的 `{{localize}}` 助手，不必在 JS 里预先格式化。按钮上的 `data-action="aea-push"` 是 cards 的委托监听器认动作用的：

```hbs
<span class="aea-push-inner">
  {{#if showMulti}}
  <label class="aea-multipush">
    <input type="checkbox" class="aea-multipush-box" /> {{localize "AEA.push.multi"}}
  </label>
  {{/if}}
  <button type="button" class="aea-push-button" data-action="aea-push">{{localize "AEA.push.button"}}</button>
</span>
```

- [ ] **Step 10: 写副作用层的前半 —— 常量、隐藏系统控件、唯一的推骰阶段**

新建 `scripts/features/push-correctness.mjs`，一次写全 import 区，写到 `pushStage` 为止：

```js
import { MID } from "../const.mjs";
import { features } from "../kernel/features.mjs";
import { patches } from "../kernel/patches.mjs";
import { selftest } from "../kernel/selftest.mjs";
import { cards } from "../kernel/cards.mjs";
import { rollBus } from "../kernel/rollbus.mjs";
import { resolver } from "../kernel/resolver.mjs";
import { purePushPlan, purePushRollIsBuggy } from "./push-correctness.pure.mjs";

export const PUSH_FEATURE_ID = "push-correctness";
export const PUSH_PATCH_ID = "push-roll-binds-to-its-own-card";

/** Metadata for patches.status() — the method this patch intervenes on. */
export const PUSH_TARGET = "CONFIG.Actor.documentClass.prototype.pushRoll";

/**
 * The roll bus stage token. The bus owns the single libWrapper registration on
 * that method; every other consumer queues a stage instead, because libWrapper
 * refuses a second registration on the same target and the loser of that race
 * becomes a silent no-op.
 */
export const PUSH_STAGE = "pushRoll";

/** Single source of truth for the keys this feature localizes at runtime. */
export const PUSH_I18N_KEYS = [
  "AEA.push.button",
  "AEA.push.multi",
  "AEA.push.noRecord",
  "AEA.push.notPushable",
  "AEA.push.synthetic",
  `AEA.feature.${PUSH_FEATURE_ID}.name`,
  `AEA.feature.${PUSH_FEATURE_ID}.hint`,
];

const TEMPLATE = `modules/${MID}/templates/push-controls.hbs`;

/** This feature's own section inside the shared mount (contract: aea-<feature-id>). */
const SECTION = `aea-${PUSH_FEATURE_ID}`;

/** purePushPlan's refusal reasons -> the message shown to the clicker. */
const BLOCK_MESSAGE = {
  "no-record": "AEA.push.noRecord",
  "not-pushable": "AEA.push.notPushable",
  synthetic: "AEA.push.synthetic",
};

/**
 * Hide the system's own push controls on a rendered card so its listener
 * (module/alienrpg.mjs:464-497) can no longer be clicked. Suppressing the
 * listener without hiding the button is what would otherwise double-bind the
 * click and charge two Stress for one push.
 *
 * The controls are emitted together at YZEDiceRoller.mjs:381-389 as
 *   <span>Allow multi-push </span> <input class="multiPush"> <button class="alien-Push-button">
 * and the checkbox is skipped entirely on an mPush card (:379), so the walk is
 * by DOM structure — never by matching rendered text, which Babele rewrites.
 *
 * @param {ParentNode|null} root the rendered message root
 * @returns {{had:boolean, hadMulti:boolean}} had=this card really had a system
 *          push button; hadMulti=it also had the multi-push checkbox
 */
export function hideSystemPushControls(root) {
  const btn = root?.querySelector?.("button.alien-Push-button");
  if (!btn) return { had: false, hadMulti: false };
  btn.style.display = "none";

  let hadMulti = false;
  const box = btn.previousElementSibling;
  if (box?.matches?.("input.multiPush")) {
    hadMulti = true;
    box.style.display = "none";
    const label = box.previousElementSibling;
    if (label?.tagName === "SPAN") label.style.display = "none";
  }
  return { had: true, hadMulti };
}

/**
 * The disposition that decides a blind (GM-whispered) push. -1 is HOSTILE.
 * The system reads actor.prototypeToken (alienrpg.mjs:483), which on an
 * unlinked token can disagree with the token actually on the scene.
 * resolver.soleToken returns the placeable Token only when the actor has
 * exactly one active token; its TokenDocument carries `disposition`.
 */
function dispositionOf(actor) {
  const doc = actor.token ?? resolver.soleToken(actor)?.document ?? actor.prototypeToken;
  return doc?.disposition;
}

/**
 * The one push implementation, queued on the roll bus as a stage over
 * alienrpgActor#pushRoll. NOT exported: its only legal entry is the bus's
 * stage chain, because the bus's own wrapper — which sits outside every stage
 * — is the only place push.parentRollId and push.count are captured.
 *
 * Contract shape: around(next, args, thisArg); `args` is the original argument
 * array [actor, reRoll, hostile, blind, message] and next(args) hands a
 * (possibly rewritten) array to the next layer, ending at the system method.
 *
 * The repair is to the INPUT, not the algorithm: actor.mjs:1313-1314 subtracts
 * correctly, it just subtracts from a global that describes the newest roll in
 * the world. Seeding that global from this card's record makes the system's own
 * arithmetic — and the carried-success total yzeRoll reads at :100 — describe
 * this card. Only the seven fields pushRoll and :100 read are written; sCount
 * and multiPush (the mPush carry at :357) are left alone.
 *
 * @param {(args:any[]) => any} next
 * @param {any[]} args
 * @param {Actor} thisArg
 */
function pushStage(next, args, thisArg) {
  const [callerActor, reRoll, , , message] = args;

  const rollArr = game.alienrpg?.rollArr;
  // Nothing to repair if the system global is gone: delegate untouched rather
  // than invent behaviour.
  if (!rollArr) return next(args);

  const record = rollBus.recordOf(message);

  // Resolve through the record's uuids first: an unlinked token owns its own
  // Stress track, and game.actors.get(speaker.actor) — what the system does at
  // alienrpg.mjs:472 — would charge the shared base actor instead.
  const actor =
    (record ? resolver.actorOf(record) : null) ??
    resolver.fromSpeaker(message?.speaker) ??
    callerActor ??
    thisArg ??
    null;
  if (!actor) {
    // Reusing the system's own string for the system's own situation.
    ui.notifications.warn(game.i18n.localize("ALIENRPG.NoToken"));
    return null;
  }

  const synthstress = !!actor.system?.header?.synthstress;
  const plan = purePushPlan({
    record,
    actorType: actor.type,
    synthstress,
    multiPush: reRoll === "mPush",
  });

  // Refuse BEFORE delegating. The system charges Stress at actor.mjs:1303-1312
  // and only afterwards looks at the pool, so "delegate and hope" would charge
  // for a push that has nothing to re-roll. This is the one branch that does
  // not call next(); every other path calls it exactly once.
  if (!plan.ok) {
    const key = BLOCK_MESSAGE[plan.reason];
    if (key) ui.notifications.warn(game.i18n.localize(key));
    return null;
  }

  Object.assign(rollArr, plan.seed);

  // A synthstress synthetic must roll as "character" or yzeRoll's push branches
  // (:76-90 for the "Push" title, :344 for the running total) skip it. This
  // mirrors effectiveActorType at actor.mjs:225-226. `hostile` is the third
  // positional of pushRoll and becomes yzeRoll's `actortype`.
  const actortype = actor.type === "synthetic" && synthstress ? "character" : actor.type;

  return next([actor, plan.reRoll, actortype, dispositionOf(actor) === -1, message]);
}
```

- [ ] **Step 11: 写副作用层的后半 —— 探针、渲染、动作、自检、特性对象**

追加到 `scripts/features/push-correctness.mjs` 末尾（import 区在 Step 10 已经写全，这一步不动它）：

```js
/**
 * Cached because the probe must read the ORIGINAL method body. patches.applyAll()
 * runs at the ready.patches anchor, before rollBus.install() at ready.rollbus,
 * so the first read sees the system's own function; after the bus installs, the
 * prototype property is libWrapper's wrapper and a second read would find no
 * `rollArr` in it and wrongly conclude "upstream fixed it".
 */
let probeResult = null;

/** Set by apply(); the only in-process evidence that the stage was queued. */
let stageQueued = false;

function probePushRoll() {
  if (probeResult !== null) return probeResult;
  const fn = CONFIG?.Actor?.documentClass?.prototype?.pushRoll;
  probeResult = typeof fn === "function" ? purePushRollIsBuggy(Function.prototype.toString.call(fn)) : false;
  return probeResult;
}

/**
 * Get (or create) this feature's own container inside the shared mount.
 * cards.render() replaces the innerHTML of whatever element it is given, so
 * rendering straight into the mount would wipe the other phase-1 feature that
 * shares it.
 */
function sectionOf(mount) {
  let section = mount.querySelector(`:scope > .${SECTION}`);
  if (!section) {
    section = mount.ownerDocument.createElement("div");
    section.className = SECTION;
    mount.appendChild(section);
  }
  return section;
}

/**
 * Called by cards' single render listener, in registration order, synchronously.
 * Deliberately does NOT wait for the dice animation barrier: these are controls,
 * not a result, and holding them back would leave the card with no push button
 * at all for the length of the animation.
 */
function renderPushControls(ctx) {
  // Do NOT destructure `mount` in the parameter list: cards creates the mount
  // element lazily on first property access, so touching it before we know we
  // will draw would leave an empty <div class="aea-mount"> on every card.
  const { element, record } = ctx;
  if (!features.enabled(PUSH_FEATURE_ID)) return;

  // Only take over controls the system actually drew, so no button appears on a
  // card that was never pushable in the first place (YZEDiceRoller.mjs:378).
  const sys = hideSystemPushControls(element);
  if (!sys.had) return;

  // A record that says "done" means no controls at all — not even ours.
  // A missing record still gets our button: clicking it explains why it cannot
  // be pushed, instead of silently leaving the card with no button.
  if (record?.push?.pushable === false) return;

  const mount = ctx.mount; // first touch: this is what inserts the mount node
  if (!mount) return;
  // cards.render returns a Promise (it awaits renderTemplate); the handler
  // itself is called synchronously, so attach the failure path rather than
  // await it.
  cards.render(sectionOf(mount), TEMPLATE, { showMulti: sys.hadMulti }).catch((err) =>
    console.error(`${MID} | ${PUSH_FEATURE_ID}: rendering push controls failed`, err),
  );
}

/** Self-test: the patch is registered, applied, and its stage really got queued. */
function selftestStageLive() {
  const entry = patches.status().find((p) => p.id === PUSH_PATCH_ID);
  if (!entry) return { ok: false, detail: `patch ${PUSH_PATCH_ID} was never registered` };
  if (!entry.applied) {
    return {
      ok: false,
      detail: `patch not applied (type=${entry.type}, reason=${entry.reason}, fixedIn=${entry.fixedIn})`,
    };
  }
  if (!stageQueued) return { ok: false, detail: "apply() ran but rollBus.addStage was never reached" };
  if (typeof CONFIG?.Actor?.documentClass?.prototype?.pushRoll !== "function") {
    return { ok: false, detail: `${entry.target} is not a function` };
  }
  return { ok: true, detail: `stage "${PUSH_STAGE}" queued on ${entry.target} (reason=${entry.reason})` };
}

/** Self-test: no system push button anywhere in the log is still clickable. */
function selftestControlsHidden() {
  const leaked = [...document.querySelectorAll("button.alien-Push-button")].filter(
    (b) => b.style.display !== "none",
  );
  if (!features.enabled(PUSH_FEATURE_ID)) {
    return { ok: true, detail: `feature off: ${leaked.length} system buttons left visible on purpose` };
  }
  const ours = document.querySelectorAll("button.aea-push-button").length;
  if (leaked.length) {
    return { ok: false, detail: `${leaked.length} system push buttons still clickable — double-bind risk` };
  }
  return { ok: true, detail: `no clickable system push button; ${ours} module buttons in the log` };
}

/** Self-test: every key this feature localizes exists in the active language. */
function selftestI18n() {
  const missing = PUSH_I18N_KEYS.filter((k) => !game.i18n.has(k));
  return missing.length
    ? { ok: false, detail: `missing i18n keys: ${missing.join(", ")}` }
    : { ok: true, detail: `${PUSH_I18N_KEYS.length} keys present` };
}

export const pushFeature = {
  id: PUSH_FEATURE_ID,

  /** init phase: main.mjs runs `for (const f of FEATURES) f.register()`. */
  register() {
    features.register({
      id: PUSH_FEATURE_ID,
      default: "full",
      gmOnly: false,
      requires: [],
      // Display name and hint come from AEA.feature.<id>.name / .hint.
      hint: "",
    });

    // The patch is governed by the patch kernel (version range + probe), not by
    // the feature toggle: turning the feature off only takes back the module's
    // own controls, which makes the system's button visible again — and that
    // button must still land on the right card's dice pool.
    patches.register({
      id: PUSH_PATCH_ID,
      type: "WRAPPER",
      target: PUSH_TARGET, // metadata for status(); applyAll() does not register from it
      minSystem: "4.1.13",
      fixedIn: null,
      probe: probePushRoll,
      apply() {
        // No libWrapper here: rollBus owns the registration on this target and a
        // second one would be refused, leaving this patch a silent no-op.
        // order 0 — nothing else in phase 1 stages pushRoll.
        rollBus.addStage(PUSH_STAGE, { id: PUSH_PATCH_ID, order: 0, around: pushStage });
        stageQueued = true;
      },
    });

    selftest.register({
      id: `${PUSH_FEATURE_ID}.stage-live`,
      label: "Push stage is registered, applied and queued on the roll bus",
      run: selftestStageLive,
    });
    selftest.register({
      id: `${PUSH_FEATURE_ID}.controls-hidden`,
      label: "No system push button is clickable on any rendered card",
      run: selftestControlsHidden,
    });
    selftest.register({
      id: `${PUSH_FEATURE_ID}.i18n`,
      label: "Push feature i18n keys resolve in the active language",
      run: selftestI18n,
    });
  },

  /** ready phase: main.mjs runs `for (const f of FEATURES) f.install()` after cards.init(). */
  install() {
    cards.onRender("push-controls", renderPushControls);

    cards.registerAction("aea-push", async (event, { message, element, mount }) => {
      if (!features.enabled(PUSH_FEATURE_ID)) return;

      const record = rollBus.recordOf(message);
      const actor = (record ? resolver.actorOf(record) : null) ?? resolver.fromSpeaker(message?.speaker);
      if (!actor) {
        ui.notifications.warn(game.i18n.localize("ALIENRPG.NoToken"));
        return;
      }

      const box = (mount ?? element)?.querySelector?.(".aea-multipush-box");
      const reRoll = box?.checked ? "mPush" : "push";

      // The delegated listener sits on `document`, so event.currentTarget is the
      // document — find the real button to lock out an impatient double click.
      const button = event.target?.closest?.("button.aea-push-button") ?? null;
      if (button) button.disabled = true;
      try {
        // Enter through the document method, never straight into the stage: the
        // roll bus wraps pushRoll from the outside and that wrapper is the only
        // place push.parentRollId and push.count get captured. The stage
        // normalizes actor / reRoll / actortype / blind, so passing what we know
        // here is enough.
        await actor.pushRoll(actor, reRoll, actor.type, false, message);
      } finally {
        if (button) button.disabled = false;
      }
    });
  },
};
```

- [ ] **Step 12: 跑接线测试，看它通过**

Run: `npx vitest run test/push-correctness.wiring.test.mjs`
Expected: PASS —— `Tests  7 passed`（id 一致 1 条、补丁元数据 1 条、addStage 排队且未碰 libWrapper 1 条、install 接线 1 条、无自挂钩子无 libWrapper 1 条、两份语言文件各 1 条）。

- [ ] **Step 13: 写下 main.mjs 接线的失败测试**

新建 `test/push-correctness.main-wiring.test.mjs`。它把「这一行接线有没有主人」变成可断言的事实：`main.mjs` 里有两个循环 —— `init` 阶段 `for (const f of FEATURES) f.register()`、`ready` 阶段 `for (const f of FEATURES) f.install()` —— 所以本特性**唯一**要做的接线就是把自己列进 `FEATURES`。后两条用例钉死「十个锚点一个不少」「生命周期钩子只有四个、只在 main.mjs」，也就是本任务没有顺手动别人的插入点。

```js
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";

const src = readFileSync(new URL("../scripts/main.mjs", import.meta.url), "utf8");

/** The ten anchors main.mjs must carry, verbatim. */
const ANCHORS = [
  "imports",
  "features",
  "repairs",
  "init",
  "i18nInit",
  "diceSoNiceReady",
  "ready.registry",
  "ready.patches",
  "ready.rollbus",
  "ready.cards",
];

describe("main.mjs 接线：push-correctness", () => {
  it("在 imports 锚点之后导入了 pushFeature", () => {
    expect(src).toMatch(
      /import\s*\{[^}]*\bpushFeature\b[^}]*\}\s*from\s*["']\.\/features\/push-correctness\.mjs["'];/,
    );
    const anchorAt = src.indexOf("/* AEA-ANCHOR: imports */");
    const importAt = src.indexOf('from "./features/push-correctness.mjs"');
    expect(anchorAt).toBeGreaterThan(-1);
    expect(importAt).toBeGreaterThan(anchorAt);
  });

  it("pushFeature 列进了 FEATURES 数组（register/install 由 main.mjs 的两个循环统一调用）", () => {
    const arr = src.match(/const\s+FEATURES\s*=\s*\[([\s\S]*?)\]/);
    expect(arr, "main.mjs 里找不到 const FEATURES = [...]").not.toBeNull();
    expect(arr[1]).toMatch(/\bpushFeature\b/);
    // The anchor stays inside the array for the tasks that insert after us.
    expect(arr[1]).toContain("/* AEA-ANCHOR: features */");
  });

  it("十个锚点一个不少（本任务只在 imports 与 features 两处插行）", () => {
    const missing = ANCHORS.filter((a) => !src.includes(`/* AEA-ANCHOR: ${a} */`));
    expect(missing).toEqual([]);
  });

  it("本任务没有往 main.mjs 里添任何钩子：仍然只有四个生命周期钩子", () => {
    const hooks = [...src.matchAll(/Hooks\.\w+\(\s*["']([^"']+)["']/g)].map((m) => m[1]).sort();
    expect(hooks).toEqual(["diceSoNiceReady", "i18nInit", "init", "ready"]);
  });
});
```

- [ ] **Step 14: 跑它，看它失败**

Run: `npx vitest run test/push-correctness.main-wiring.test.mjs`
Expected: FAIL —— 前两条红：`expected '…main.mjs 全文…' to match /import\s*\{[^}]*\bpushFeature\b…/` 与 `expected '…' to match /\bpushFeature\b/`；后两条（十个锚点、四个生命周期钩子）绿。`Tests  2 failed | 2 passed`。

- [ ] **Step 15: 在 main.mjs 接线（只加两行，按锚点定位）**

在 `scripts/main.mjs` 里**按锚点注释文本定位，不要用行号**。骨架长这样（由建立 main.mjs 的那个任务逐字写出，你只读不改）：

```js
import { MID } from "./const.mjs";
/* AEA-ANCHOR: imports */

export const api = { /* 八个键，值可能已被别的任务填上 */ };

const FEATURES = [
  /* AEA-ANCHOR: features */
];
```

1. 在 `/* AEA-ANCHOR: imports */` 的**下一行**插入：

```js
import { pushFeature } from "./features/push-correctness.mjs";
```

2. 在 `/* AEA-ANCHOR: features */` 的**下一行**插入（注意行尾逗号，缩进两格）：

```js
  pushFeature,
```

**这两行就是全部。**其余八个锚点 —— `repairs`、`init`、`i18nInit`、`diceSoNiceReady`、`ready.registry`、`ready.patches`、`ready.rollbus`、`ready.cards` —— **一个字都不要动**，`api` 那个对象**不要替换、不要增删键**。原因：`init` 段已经有 `for (const f of FEATURES) f.register()`，`ready` 段在四个子锚点之后已经有 `for (const f of FEATURES) f.install()`。所以本任务**一行生命周期代码都不加**：`register()`（注册特性 + 注册补丁 + 登记三条自检）与 `install()`（`cards.onRender` + `cards.registerAction`）都由那两个循环调到，顺序天然正确 —— `ready.patches` 里 `patches.applyAll()` 把阶段排进队，`ready.rollbus` 里 `rollBus.install()` 把包裹装上，`ready.cards` 里 `cards.init()` 让挂点与委托监听就位，之后才轮到 `install()`。**禁止**再自己直调一次 `register()` / `install()`：那会重复登记特性与补丁。

同样**不要**在 `main.mjs` 里单独调 `patches.register` 或 `selftest.register`：它们已经在 `pushFeature.register()` 内部完成。

- [ ] **Step 16: 跑全部测试，看它通过**

Run: `npx vitest run test/push-correctness.main-wiring.test.mjs`
Expected: PASS —— `Tests  4 passed`。

Run: `npm test`
Expected: PASS —— 本任务新增的三个文件共 26 个用例全绿（15 + 7 + 4），且**不得**有任何既有测试变红。

- [ ] **Step 17: MANUAL VERIFICATION —— 在 Foundry 里验四组**

桩模拟不了真实的 libWrapper 链、阶段链与聊天卡渲染时序，下面每一条都要在真世界里做一遍。

前置：世界启用 `alien-evolved-automation` 与 `lib-wrapper`，系统 `alienrpg` 4.1.13，聊天卡上已经能看到本模组写的 flag（掷一次骰后在控制台执行 `game.messages.contents.at(-1).flags["alien-evolved-automation"].roll`，应返回一个对象）。

**A. 补丁装上了**
1. F12 打开控制台，执行 `game.modules.get("alien-evolved-automation").api.patches.status()`。
   - **期望**：数组里有一项 `{id: "push-roll-binds-to-its-own-card", type: "WRAPPER", target: "CONFIG.Actor.documentClass.prototype.pushRoll", applied: true, reason: "ok", fixedIn: null}`。
2. 控制台里没有 `libWrapper: Conflict detected`，也**没有** `A wrapper for 'CONFIG.Actor.documentClass.prototype.pushRoll' ... has already been registered by alien-evolved-automation` 这类报错。若出现后者，说明有人绕过 `rollBus.addStage` 自己去 `libWrapper.register` 了同一个目标。
3. 执行 `game.modules.get("alien-evolved-automation").api.selftest.runAll()`。
   - **期望**：`push-correctness.stage-live`、`push-correctness.controls-hidden`、`push-correctness.i18n` 三条全部 `ok: true`。`stage-live` 的 `detail` 形如 `stage "pushRoll" queued on CONFIG.Actor.documentClass.prototype.pushRoll (reason=ok)`；`i18n` 的 `detail` 是 `7 keys present`。
4. 让一个角色掷一次技能，在那张卡上右键 → 检查元素。
   - **期望**：卡里有且只有一个 `div.aea-mount`，它下面有且只有一个 `div.aea-push-correctness`，其中有且只有一个 `button.aea-push-button`。把聊天面板弹出成独立窗口（聊天栏标题栏的弹出按钮）再看同一张卡，仍然是各一个 —— 重复渲染不会叠出第二份控件，也不会把挂点里别的特性的内容清掉。

**B. 推旧卡（这条就是缺陷本体）**
1. 角色 A 掷一次 **HEAVY MACHINERY**，记下卡上的黑骰数、`Sixes` 数、黄骰数、`Ones` 数，以及角色当前压力值。
2. 让**同一个或另一个**角色再掷一次别的技能，例如 **OBSERVATION**，且骰数明显不同。此时在控制台执行 `game.alienrpg.rollArr.tLabel`。
   - **期望**：返回 `"Observation"` —— 这正是系统 `:1319` 会拿去当标题的那个全局，它现在描述的是**最新**那次掷骰。这一步不是验收，是让你亲眼看到缺陷的成因。
3. 滚回第 1 张卡，点模组画的 **推骰** 按钮。
   - **修复前的症状**（用来对照）：新卡标题写着 Observation，骰数是第 2 次掷骰的剩余骰数，而压力已经先扣掉了。
   - **期望（修复后）**：新卡的 `<h2>` 标题是 **`Push` + 第 1 张卡的技能名**（`YZEDiceRoller.mjs:76-90` 给 rType 填 `ALIENRPG.Push`，`:126` 拼成标题），**不是** Observation。黑骰数 = 第 1 张卡的黑骰数 − 它的 Sixes 数；黄骰数 = 第 1 张卡的黄骰数 + 1 − (它的 Ones + Sixes)。压力恰好 +1（世界设置 `evolved` 开启时为角色的 addpanic 值）。
   - 新卡上那行「Following the Push, You have a Total of N」的**显示**归另一条特性管，不在本任务范围内。
4. 在这张**新的推骰卡**上找推骰按钮。
   - **期望**：**一颗按钮都没有**（系统与模组都不画）。系统 `YZEDiceRoller.mjs:378` 的条件是 `if (!reRoll || reRoll === "mPush")`，普通推骰卡的 `reRoll` 是 `"push"`，整段控件都不发出；模组只在系统画了按钮的卡上替换，所以也不画。
5. 回到一张还能推的卡，**先勾上「额外推骰」再点推骰**。
   - **期望**：新卡上**又出现推骰按钮**，但**没有**「额外推骰」复选框（`:379-385` 只在 `reRoll !== "mPush"` 时画复选框）。再点一次能再推一次，每次压力都 +1 —— 这是系统自带的 MultiPush 功能，模组不得把它拒掉。
   - **若这里弹出「这次掷骰已经推过，不能再推。」**：缺陷不在本特性，而在记录产出方。在控制台执行 `game.messages.contents.at(-1).flags["alien-evolved-automation"].roll.push`，`mPush` 卡的 `pushable` 必须是 `true`（哪怕 `count > 0`）。把这条现象报给记录产出方，**不要**在本特性里改成按 `push.count` 判断。

**C. 合成人不再是死按钮**
1. 建一个 **synthetic**（合成人）角色，在角色卡上勾 "Human Panic, Push, etc."（`system.header.synthstress`）。
2. 掷一次技能，卡上会出现推骰按钮，点它。
   - **修复前的症状**：什么都不发生 —— 没有卡、没有报错、没有提示（`alienrpg.mjs:487-493` 的 `switch` 直接 `return`）。
   - **期望（修复后）**：出现一张推骰卡，压力 +1，标题带 `Push` 前缀（合成人被阶段改成按 `"character"` 掷，见 `actor.mjs:225-226`）。合成人的压力骰基数为 0（synthetic 分支只给 `stressMod`），所以推骰后黄骰通常是 1 颗。合成人的 `system.header.stress` 与 `general.addpanic` 在数据模型里都存在（`module/data/actor-synthetic.mjs:32`、`:238`），所以系统那段扣压力的代码对它同样跑得通。
3. **不要重掷**，回到角色卡把 "Human Panic, Push, etc." 取消勾选，再回到刚才那张**还带按钮的旧卡**点推骰（卡的 HTML 已经存进消息里，取消勾选不会让按钮消失）。
   - **期望**：弹出黄色提示「合成人不承受压力，也不推骰。」，**压力没有任何变化**，也没有新卡 —— 否决发生在委托之前，系统那段先扣压力的代码根本没跑到。

**D. 非链接 token 与双绑**
1. 把同一个 `character` 拖到场景上生成两个 token，在每个 token 的配置里确认 "Link Actor Data" **未勾选**（系统在 `alienrpg.mjs:366-376` 会对勾了 NPC 的角色自动这么做）。
2. **双击场景上的 token 1**（打开的是这个 token 自己的角色卡，不是侧栏里的基础角色），从这张卡掷一次技能。在控制台执行 `game.messages.contents.at(-1).flags["alien-evolved-automation"].roll.tokenUuid`。
   - **期望**：返回形如 `"Scene.xxx.Token.yyy.Actor.zzz"` 的字符串，不是 `null` —— 从 token 表单发起的掷骰，施动 token 是可归属的。
   - 点推骰后：**只有 token 1** 的压力上升；token 2 与侧栏里的基础角色卡上的压力**不变**。推出来的新卡上再查 `tokenUuid`，应与父卡相同（推骰帧从父记录继承施动者）。
3. 若第 2 步的 `tokenUuid` 是 `null`（例如从侧栏基础角色卡掷、或用 GM 宏掷，且当前场景上不止一个该角色的活动 token）：**这是如实记录，不是缺陷**。系统在 `YZEDiceRoller.mjs:398-401` 用 `ChatMessage.getSpeaker({actor: actorid})` 建卡，token 信息在消息存在之前就被丢掉了。此时压力会落到 `speaker` 能解析到的角色上。**不要**在本特性里用 `canvas.tokens.controlled` 之类的猜测去兜底伪造一个 token。
4. 回到一张还能推的卡，**快速连点两次**推骰按钮。
   - **期望**：压力总共只涨一次的量，只出现一张新卡。若涨了两倍，说明系统那颗按钮没被隐藏、两个监听器同时生效 —— 在控制台执行 `[...document.querySelectorAll("button.alien-Push-button")].map(b => b.style.display)`，每一项都应是 `"none"`；或直接重跑 A.3 的 `push-correctness.controls-hidden`。
5. 打开「配置设置 → 模组设置」，把「推骰绑回自己那张卡」改成 `off`，重新掷一次骰。
   - **期望**：系统原来的推骰按钮与 "Allow multi-push" 复选框重新可见，模组的按钮消失。此时点系统那颗按钮，走的**仍然**是排在 rollBus 上的这个阶段，所以 B 组的行为依旧正确（标题与骰数来自被推的那张卡）—— 规则纠正归补丁内核管，不随特性开关退出；开关只收回模组自己画的控件。

- [ ] **Step 18: 提交特性与补丁**

```bash
git add scripts/features/push-correctness.mjs templates/push-controls.hbs test/push-correctness.wiring.test.mjs test/push-correctness.main-wiring.test.mjs scripts/main.mjs lang/en.json lang/cn.json && git commit -m "$(cat <<'EOF'
feat(push): 推骰绑回它自己那张卡

补丁半：patches.register 的 apply() 只调 rollBus.addStage("pushRoll", ...)
排一个 around 阶段 —— 那四个掷骰目标的 libWrapper 注册权归 rollBus 独占，
再注册一次会被 libWrapper 拒绝、修复静默变空操作。阶段在委托给下一层之前，
把系统 :1313-1319 与 :100 要读的那个全局按本卡记录填成七个字段的种子，并把
actor / reRoll / actortype / blind 四个入参改对，然后 next(args) 恰好一次；
记录缺失或不可推时先否决再返回，绝不让系统 :1303-1312 先把压力扣掉。
probe 走可单测的纯谓词并缓存首次结果 —— rollBus 装上包裹之后原方法源码不
可达，再探会误判成「上游已修」。

卡片半：经 cards.onRender 拿到挂点，在本特性自己的 aea-push-correctness
分区里画推骰按钮（cards.render 只替换传入元素的 innerHTML，直接渲染挂点会
清掉同挂点上另一条特性的内容），并按 DOM 结构隐藏系统那颗按钮与 multiPush
复选框（不解绑，因而不会双绑双扣压力）；只在系统本来就画了按钮的卡上替换。
点击一律经 actor.pushRoll 进入，绝不直接调阶段函数 —— 包裹 pushRoll 是
push.parentRollId / push.count 的唯一采集点。

角色一律经 resolver 从记录的 uuid 解析，非链接 token 各扣各的压力；阵营取
本场景实际那个 token（resolver.soleToken）；勾了 synthstress 的合成人不再
是点了没反应的死按钮。三条断言登记进 selftest，main.mjs 只多两行：imports
锚点后一行 import、features 锚点后一行数组成员，十个锚点一个未动。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```
