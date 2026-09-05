> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 16 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 16: 五条表单类修复 —— 飞船阶段提交、异形酸血、主机技能行、改名 token、负重

> **分组理由**：这五条修的都是「表单／文档创建这一侧挂着的东西」——`alienrpgSpacecraftSheet.DEFAULT_OPTIONS.actions` 里的一个动作、`alienrpgCreatureSheet` 转给 actor 的酸血调用、`item-sheet.hbs` 少写的一个属性、`preCreateToken` 的按名查 actor、以及三张角色卡原型上同名的 `_computeEncumbrance`。它们不碰掷骰记录（`RollRecord`）、不碰聊天卡挂点、不碰抽表。

> **本任务与内核的关系**：消费四个内核模块 —— `patches`（缺陷补丁登记）、`selftest`（自检登记）、`features`（每条修复各一个开关）、`resolver`（actor/token 解析）。**不**消费 `rollBus` / `cards` / `registry`。
>
> **本任务不碰 `rollBus` 的四个包裹目标**（`yzeRoll` / `abilityRoll` / `itemRoll` / `pushRoll`）。lib-wrapper 对同一目标重复注册会抛错，任何介入这四个目标的代码必须走 `rollBus.addStage()`；本任务包裹的是 `creatureAcidRoll`、`_prepareContext`、`_computeEncumbrance`，三者都不在那四个之内，所以直接用 `libWrapper.register` 是合法的。

---

**Files:**
- Create: `scripts/repairs/repair-support.mjs`（两个共用小工具：开关状态、源码读取）
- Create: `scripts/repairs/spacecraft-phase.pure.mjs`
- Create: `scripts/repairs/spacecraft-phase.mjs`
- Create: `scripts/repairs/creature-acid.pure.mjs`
- Create: `scripts/repairs/creature-acid.mjs`
- Create: `scripts/repairs/mainframe-skill-row.pure.mjs`
- Create: `scripts/repairs/mainframe-skill-row.mjs`
- Create: `scripts/repairs/token-defaults.pure.mjs`
- Create: `scripts/repairs/token-defaults.mjs`
- Create: `scripts/repairs/encumbrance.pure.mjs`
- Create: `scripts/repairs/encumbrance.mjs`
- Create: `test/repair-sheets.test.mjs`
- Create: `test/repair-registration.test.mjs`
- Create: `test/repair-gate.test.mjs`
- Modify: `scripts/main.mjs`（只在两个锚点注释后各插五行，见 Step 41）
- Modify: `lang/en.json`, `lang/cn.json`（各加 16 个键，并入已有的顶层 `AEA` 对象内）

**Interfaces:**

*Consumes*（全部由别的任务产出，本任务只 import，不得重建）：
- `import { MID, SYSTEM_ID } from "../const.mjs";` —— `MID === "alien-evolved-automation"`，`SYSTEM_ID === "alienrpg"`。
- `import { patches } from "../kernel/patches.mjs";` —— `patches.register({id, type, target, minSystem, fixedIn, probe, apply})`。语义定死：`type` 是 `"WRAPPER"|"MIXED"|"OVERRIDE"|"DATA"|"HOOK"` 之一；`target` **仅是给 `status()` 展示的元数据**，`applyAll()` 不据它注册任何东西；`probe()` 返回 `true` 表示「缺陷仍在，该装」、`false` 表示「上游已修，退休」；`apply()` 是**无参自装器**，自己负责调 `libWrapper.register(...)`、自己挂钩子、或自己改数据。
  `patches.status()` 返回 `[{id, type, target, applied, reason, fixedIn}]` —— **六个字段一个不多一个不少**；`reason` 只有七个取值：`"ok"`（已装）/ `"disabled"` / `"version-not-applicable"` / `"probe-false"` / `"libwrapper-missing"` / `"error"` / `"pending"`。**`status()` 列出全部已登记补丁**，包括 `applyAll()` 还没跑到的（那时是 `applied:false, reason:"pending"`）——本任务的登记测试正是断言这个状态。
- `import { selftest } from "../kernel/selftest.mjs";` —— `selftest.register({id, label, run})`；`label` 是 **i18n 键**（登记发生在 `init`，早于 `i18nInit`，那时语言包还没加载，`localize()` 只会回声键名，所以只能存键，由 `runAll()` 负责本地化）；`run()` 返回 `{ok: boolean, detail: string}`，可 async；`selftest.runAll()` 返回 `[{id, label, ok, detail}]`。**`patches.register()` 不会自动登记任何自检条目**，自检一律显式登记。
- `import { features } from "../kernel/features.mjs";` —— `features.register({id, default, gmOnly, requires})` 在 `init` 登记开关；`features.enabled(id)` 在**执行时**查这个开关，`features.mode(id)` 取 `"full"|"prompt"|"off"`。显示名走 i18n 键 `AEA.feature.<id>.name` / `AEA.feature.<id>.hint`。开关值存在**唯一一个**世界设置 `SETTING_FEATURES` 里（`const.mjs` 只导出这一个特性设置键），值形如 `{[id]: "full"|"prompt"|"off"}`。
- `import { resolver } from "../kernel/resolver.mjs";` —— `resolver.soleToken(actor)`：actor 在当前场景**恰好只有一个**活动 token 时返回它，否则返回 `null`（不猜）；`resolver.refs(actor, token)` 返回 `{actorUuid, tokenUuid}`。
- `import { installFoundryStub, uninstallFoundryStub, foundryStubContext } from "./stubs/foundry.mjs";` —— 测试用的共享 Foundry 全局桩。它是**真实现**：`game.settings` 由 `ctx.settings` 支撑（`get` 未注册键会抛错）、`Hooks.callAll` 真的派发给 `Hooks.on` 注册的回调、`libWrapper.register` 记进 `ctx.wrappers`、`ChatMessage.create` 推进 `ctx.messages`、`game.system.version` 默认 `"4.1.13"`。`ctx.hooks.on` 的条目形状是 `{name, fn, once}`。桩由别的任务独占实现，本任务**只读不改**：禁止在测试里自己造 `globalThis.game`、也禁止用私有 Map 顶替 `game.settings`。
- `scripts/main.mjs` 里已有的 `/* AEA-ANCHOR: imports */` 与 `/* AEA-ANCHOR: repairs */` 两个锚点注释，以及两条由 main.mjs 自己拥有的循环行：`init` 阶段的 `for (const r of REPAIRS) r.register()` 与 `ready` 阶段的 `for (const r of REPAIRS) r.install?.()`。本任务**不新增任何生命周期行**，只在这两个锚点后各插五行。

*Produces*：
- 两个共用小工具（`scripts/repairs/repair-support.mjs`）：`switchState(id)`、`readSourceText(path, assign)`。
- 五个修复对象，形状统一为 `{ id, register(), install() }`：
  - `export const spacecraftPhaseRepair`（`scripts/repairs/spacecraft-phase.mjs`）
  - `export const creatureAcidRepair`（`scripts/repairs/creature-acid.mjs`）
  - `export const mainframeSkillRowRepair`（`scripts/repairs/mainframe-skill-row.mjs`）
  - `export const tokenDefaultsRepair`（`scripts/repairs/token-defaults.mjs`）
  - `export const encumbranceRepair`（`scripts/repairs/encumbrance.mjs`）
- 八个补丁（`patches.status()` 里可见）：

  | 补丁 id | type | target |
  |---|---|---|
  | `spacecraft-phase-submit` | OVERRIDE | `game.alienrpg.ActorSheets.alienrpgSpacecraftSheet.DEFAULT_OPTIONS.actions.ShipPhaseSubmit` |
  | `spacecraft-phase-persist` | WRAPPER | `game.alienrpg.ActorSheets.alienrpgSpacecraftSheet.prototype._prepareContext` |
  | `creature-acid-args` | MIXED | `game.alienrpg.alienrpgActor.prototype.creatureAcidRoll` |
  | `mainframe-skill-row` | HOOK | `renderalienrpgItemSheet` |
  | `token-defaults-by-reference` | HOOK | `preCreateToken` |
  | `encumbrance-capacity:alienrpgCharacterSheet` | MIXED | `game.alienrpg.ActorSheets.alienrpgCharacterSheet.prototype._computeEncumbrance` |
  | `encumbrance-capacity:alienrpgSyntheticSheet` | MIXED | `game.alienrpg.ActorSheets.alienrpgSyntheticSheet.prototype._computeEncumbrance` |
  | `encumbrance-capacity:alienrpgColonySheet` | MIXED | `game.alienrpg.ActorSheets.alienrpgColonySheet.prototype._computeEncumbrance` |

- 五个特性开关 id（每条修复一个，`default:"full"`、`gmOnly:true`）：`spacecraft-phase`、`creature-acid`、`mainframe-skill-row`、`token-defaults`、`encumbrance`。
- 五个自检 id：`repair.spacecraft-phase`、`repair.creature-acid`、`repair.mainframe-skill-row`、`repair.token-defaults`、`repair.encumbrance`。
- 纯函数（各自在同名 `.pure.mjs` 里，**只**导出 `pure` 开头的符号，一律不引用任何 Foundry 全局）：
  - `pureShipPhaseChatData(input)`、`pureMergePhaseSelection(stored, name, value)`、`pureRestorePlan(stored)`、`purePhaseSelectNames()`、`pureShipPhaseIsBuggy(source)`、`pureShipPhaseSelectsAreVolatile(templateSource)`
  - `pureAcidArgs(a, b)`、`pureAcidPool(rawRating)`、`pureAcidPlan(dataset)`、`pureAcidNoBloodContent(localize)`、`pureAcidRollIsBuggy(source)`
  - `pureRangedRowSelector()`、`pureRangedRowIsBuggy(templateSource)`
  - `pureTokenDefaults(input)`、`pureTokenLookupIsBuggy(source)`
  - `pureCapacityStrength(strAttr)`、`pureAcceptedNames(rawList)`、`pureTalentNames(talent)`、`purePackMule(talents, acceptedNames)`、`pureEncumbrance(input)`、`pureEncumbranceIsBuggy(source)`
- 16 个 i18n 键：`AEA.feature.<五个 id>.name` / `.hint`（10 个）、`AEA.selftest.repair.<五个 id>`（5 个）、`AEA.repair.packMuleAliases`（1 个）。

*断言归属对照表*（凡是 vitest 跑不动的断言，都必须在这张表里有去处；某条自检被人删掉时，这张表会留下一行孤儿）：

| 跑不动的断言 | 去处 |
|---|---|
| 四个 Submit 按钮真的能建卡 | 自检 `repair.spacecraft-phase` + 手工步骤 3 |
| 宣告过的阶段活过一次重渲染 | 自检 `repair.spacecraft-phase`（报 `persist=ours`）+ 手工步骤 4 |
| 酸血包装真的装在原型上 | 自检 `repair.creature-acid`（复用同一个 `pureAcidRollIsBuggy`）+ 手工步骤 5/6/7 |
| 同一基础卡的两只 token 各自点酸血，两张「没有酸血」卡的 `speaker.token` 不同 | 手工步骤 8（`ChatMessage.getSpeaker` 的分支依赖 `canvas.tokens.controlled` 与真实 TokenDocument，桩无法忠实模拟，见契约 §0.4） |
| RANGED COMBAT 那一行可点 | 自检 `repair.mainframe-skill-row`（只报安装状态与模板判定）+ 手工步骤 9 |
| 改名 token 仍被设成敌对非链接 | 自检 `repair.token-defaults` + 手工步骤 14 |
| 三张卡的 `_computeEncumbrance` 真的被替换 | 自检 `repair.encumbrance`（复用同一个 `pureEncumbranceIsBuggy`）+ 手工步骤 10/11/12/13 |
| 五个开关在设置界面里可见、关掉即时生效 | 手工步骤 15（`features.enabled` 的**逻辑**由 `test/repair-gate.test.mjs` 在桩上真测，见 Step 26） |

---

**Foundry 概念（本任务用得到的，逐条解释）**

- **ApplicationV2 的 `DEFAULT_OPTIONS.actions`**：一张 `{动作名: 处理函数}` 表。模板里写 `data-action="X"` 的元素被点击时，Foundry 在应用根元素上做事件委托，查这张表并以 `handler.call(应用实例, event, target)` 调用——所以处理函数的 `this` 是**表单实例**，两个入参是 `(PointerEvent, HTMLElement)`。**关键**：这张表在类体求值那一刻就把 `this._onShipPhase` 的**引用**抄了进去，事后替换静态方法毫无作用，必须直接改写 `DEFAULT_OPTIONS.actions.ShipPhaseSubmit` 这一项本身。每次构造表单时才读这张表，所以在 ready 阶段改，之后打开的每张表单都生效。
- **`_prepareContext(options)`**：ApplicationV2 每次渲染前调用的一个 async 方法，返回丢给 Handlebars 的上下文对象。模板里的 `{{selectOptions 列表 selected=sensorPhase}}` 读的就是上下文根上的 `sensorPhase` 键——所以往上下文里补一个键，就等于让下拉在服务端就选中正确项，不用碰 DOM。
- **`HTMLElement.dataset`**：DOM 的 `data-*` 属性集合，类型是 `DOMStringMap`——**每一个值都是字符串**，`data-roll="0"` 读出来是 `"0"` 而不是数字 `0`。
- **libWrapper**：一个社区库，用 `libWrapper.register(模块id, "点分路径", fn, 类型)` 包裹别的包上的函数，路径从 `globalThis` 解析。`"MIXED"` 类型的 `fn` 第一个参数是 `wrapped`（下一层函数），其余是原始入参，**可以选择不调 `wrapped`**；`"WRAPPER"` 类型同形但语义上要求必须调；`"OVERRIDE"` 没有 `wrapped`。三种情况下 `this` 都保持不变。本任务一律用 MIXED / WRAPPER，**不用 libWrapper 的 OVERRIDE**——因为开关关掉时要能把控制权原样交回系统，OVERRIDE 拿不到 `wrapped` 就做不到。
- **flag**：Foundry 允许任何文档挂一份模组私有数据：`actor.setFlag(模组id, 键, 值)` / `actor.getFlag(...)`。它随文档存库，是模组给系统文档加字段的正规途径。
- **render 钩子**：ApplicationV2 每次渲染后触发 `render<类名>`，回调签名 `(application, element, context, options)`，`element` 是这张表单的根 `HTMLElement`。
- **`preCreateToken` 钩子**：token 落库**之前**触发，回调可以用 `document.updateSource(变更)` 就地改将要写入的数据。`document.actor` 在这一刻已经指向真实的 Actor 文档。
- **非链接 token（`actorLink: false`）**：token 复制一份 actor 数据自己用，三只同名 Drone 各有各的血量；`actorLink: true` 则三只共享一份，打死一只三只全倒。
- **`ChatMessage.getSpeaker({actor, token})`**：决定聊天卡署名。我重新读了 `resources/app/client/documents/chat-message.mjs:231-272`：CASE 1 要求 `token` 是真的 `Token` 或 `TokenDocument`；CASE 2 要求 `actor instanceof Actor`；**传字符串 id 一个分支都不匹配**，会掉到 CASE 4（用当前选中的 token）或 CASE 6（署名成当前用户，即 GM）。系统所有出卡点传的都是字符串 id，这就是「署名成 GM」和「三只 Drone 署名乱跳」的根因。
- **`author` 而不是 `user`**：`common/documents/chat-message.mjs:49` 的 schema 字段叫 `author`；`user` 早已不在 schema 里，系统写的 `user: game.user.id` 是被丢弃的死键。我们写 `author`。
- **Babele**：社区汉化框架。它翻译文档时会把原名写进 `flags.babele.originalName`（`modules/babele/script/translation/document-translation.js:73-93`，v2.9.1 实测）。这是「中文世界里认出英文原名」的唯一可靠钥匙。

---

**你要修的源码事实（每一行我都重新打开确认过；行号对应系统 4.1.13 与 Foundry V14）**

1. `module/sheets/spacecraft-sheet.mjs:487` —— `type: CONST.CHAT_MESSAGE_TYPES.OTHER,`。我在 `resources/app/common/constants.mjs` 里逐行查过：只有 `CHAT_MESSAGE_STYLES`（:242，`OTHER: 0`），**全文没有 `CHAT_MESSAGE_TYPES`**。所以 `CONST.CHAT_MESSAGE_TYPES` 是 `undefined`，读 `.OTHER` 抛 TypeError。这个处理函数在 `:34` 注册为 `ShipPhaseSubmit: this._onShipPhase`（裸函数形式，不是 `{handler, buttons}` 形式），函数体在 `:461`；模板 `templates/actor/spacecraft-combat-phases.hbs` 的四个 Submit 按钮（:9/:16/:23/:30）全走它，**四个按钮全是哑的**。
2. 同一函数 `:477` 用 `foundry.applications.handlebars.renderTemplate("systems/alienrpg/templates/chat/ship-combat.hbs", htmlData)` 出卡，`htmlData` 的三个键是 `phaseName` / `actorname` / `action`（模板 `templates/chat/ship-combat.hbs` 的 :3/:4/:9 逐字读这三个名字，且是 `{{ }}` 双花括号——Handlebars 默认转义，注入安全）。**我们复用这张系统模板，不自己拼 HTML**。它 `:482-484` 的 `speaker: { actor: actorID }` 是裸字符串 id（见上面 getSpeaker 那条），`:485` 的 `other:` 键不是 ChatMessage schema 字段，`:479` 的 `user:` 也不是——三处我们都改掉。
3. 同模板的四个 `<select>` 名为 `sensorPhase` / `pilotPhase` / `gunnerPhase` / `engineerPhase`（:6/:13/:20/:27），不是 `system.*` 路径。表单开着 `submitOnChange`（`:45-49`），提交时这四个键不在 schema 里被丢弃。而模板写的是 `{{selectOptions config.sensor_list selected=sensorPhase ...}}`——`_prepareContext`（`:100-153`）从头到尾没有设过 `sensorPhase` 这个键（只设了 `:142-145` 四个 `*_list`），所以 `selected` 永远是 `undefined`，任何一次重渲染都把下拉打回第一项。
4. `module/sheets/creature-sheet.mjs:567-569`：`static async _onCreatureAcidRoll(actor, dataset) { this.actor.creatureAcidRoll(actor, dataset); }`——但 ApplicationV2 传进来的是 `(event, target)`。所以 `actor` 位上是 `PointerEvent`，`dataset` 位上是 `HTMLElement`。注册在 `:35`（`creatureAcidRoll: this._onCreatureAcidRoll`），模板按钮在 `templates/actor/creature-header.hbs:54`，带 `data-roll='{{system.general.acidSplash.value}}'` 与 `data-label='Acid Splash'`，**没有 `data-spbutt`**。
5. `module/documents/actor.mjs:2557` 的 `creatureAcidRoll(actor, dataset)`：`:2558` 读 `dataset.dataset.label`，`:2559` 读 `Number(dataset.dataset.roll || 0)`——靠 `dataset` 恰好是 HTMLElement 才碰巧能读到；`:2563` 的 `if (dataset.dataset.roll !== 0)` 拿 **DOMStringMap 的字符串**去和数字 `0` 严格比较，**永远为真**，于是 `:2599-2617` 那张已经写好本地化文案的卡是**死代码**（文案在 `lang/en.json:39-40`：`AcidAttack` = "Acid Blood"、`AcidBlood` = "This Creature does not have Acid Blood"）。`:2564` 还有一条 `if (dataset.dataset.spbutt === "armor" && r1Data < 1) return;` 的静默早退（`spbutt` 只从 `RollAbility` 那两行传来，见 `creature-header.hbs:19/:35`，酸血按钮不带它——但我们照样保留这条语义）。`:2597` 的 `yze.yzeRoll(..., actor.id)` 里 `actor` 是 Event，`.id` 是 `undefined`。
6. `module/data/actor-creature.mjs:66`：`acidSplash.value` 是 `StringField({ initial: "-" })`。没有酸血的怪，字段值是字符串 `"-"`，`Number("-")` 是 `NaN`。
7. `modules/token-action-hud-alien/scripts/roll-handler.js:246-253` 用**真 actor** 和一个**普通对象** `{roll, label}` 调 `actor.creatureAcidRoll(actor, rData)`，`rData` 没有 `.dataset`——今天直接 TypeError，按钮完全无反应。所以归一化必须按 `instanceof Event` 嗅探，**绝不能按参数位置**。
8. `templates/item/item-sheet.hbs:64` —— RANGED COMBAT 那一行的 `<label>` 缺 `data-action='RollComputer'`，它的三个兄弟（:52 Comtech、:56 Piloting、:60 Observation）都有（我逐行比对过四行的属性）。`RollComputer` 动作本身在 `module/sheets/item-sheet.mjs:30` 注册、处理函数在 `:597`，都好好的，只差这一个属性。四个 label 都带 `for='system.modifiers.skills.<技能>.value'`，这是稳定选择器。
9. `module/alienrpg.mjs:366-376` —— `Hooks.on("preCreateToken", async (document, tokenData, options, userID) => { const aTarget = game.actors.find((i) => i.name === tokenData.name); if (aTarget.type !== "spacecraft" && aTarget.system.header.npc) {...} })`：按 token 名字反查 actor，token 一改名就 `aTarget` 为 `undefined`，下一行读 `.type` 抛 TypeError，敌对/非链接两项默认值都设不上。
10. `module/sheets/character-sheet.mjs:483-506` 的 `_computeEncumbrance(totalWeight, actorData)`（`synthetic-sheet.mjs:470`、`colony-sheet.mjs:331` 是逐字孪生，只有缩进不同）：

```js
let enc = {
  max: actorData.actor.system.attributes.str.value * 4,
  value: Math.round(totalWeight * 100) / 100,
  value: totalWeight,                     // ← 重复键，未取整的浮点覆盖了取整结果
};
for (const i of actorData.talents) {
  if (i.name.toUpperCase() === "PACK MULE") { ... }   // ← 按显示名比对，汉化后必死
}
enc.pct = Math.min((enc.value * 100) / enc.max, 99);  // ← max 为 0 时 Infinity → 99
enc.encumbered = enc.pct > 50;                         // ← 于是新角色恒定「已负重」
```

  `str.value` 是基础力量；`str.mod` 才是装备/天赋加成后的力量——`module/data/actor-character.mjs:457-460` 明写 `this.attributes[abl].mod = Number(value) + Number(attrMod[abl])`。方法末尾还有 `this.actor.addCondition("encumbered")` / `removeCondition`（`addCondition` 在 `module/documents/actor.mjs:1368`，是 async，但系统在这里没 await——我们保持同样的调用形态，不改变时序）。
11. **孪生体的两处差异**（我逐个 grep 过）：`character-sheet.mjs:462` 与 `synthetic-sheet.mjs:449` 都设了 `context.talents` 并在 `:467` / `:454` 调用本方法，而 **`colony-sheet.mjs` 全文只有 `:331` 的定义、没有任何调用点，也从没设过 `context.talents`**——它那份是死代码，一旦被接上就会在 `for (const i of actorData.talents)` 上抛「undefined is not iterable」。`module/data/actor-colony.mjs:20` 的 `attributes` 里也没有 `str`。我们照样接管这一张，因为我们的实现对两者都免疫（`talents ?? []`、力量取不到就当 0）；这属于顺手把一颗哑弹拆了。

---

- [ ] **Step 1: 写第一批失败测试 —— 飞船阶段卡的模板数据**

新建 `test/repair-sheets.test.mjs`：

```js
import { describe, expect, it } from "vitest";
import { pureShipPhaseChatData } from "../scripts/repairs/spacecraft-phase.pure.mjs";

describe("pureShipPhaseChatData", () => {
  it("fills exactly the three placeholders ship-combat.hbs reads", () => {
    const d = pureShipPhaseChatData({
      shipName: "USCSS Montero",
      phaseLabel: "1. Sensor Phase",
      actionLabel: "Scan Enemy Ship",
      style: 0,
    });
    // systems/alienrpg/templates/chat/ship-combat.hbs reads {{actorname}} (:3),
    // {{phaseName}} (:4) and {{action}} (:9) and nothing else. The speaker is NOT
    // built here: it needs a real Actor/TokenDocument and therefore belongs to the
    // side-effect layer.
    expect(Object.keys(d.templateData).sort()).toEqual(["action", "actorname", "phaseName"]);
    expect(d.templateData.actorname).toBe("USCSS Montero");
    expect(d.templateData.phaseName).toBe("1. Sensor Phase");
    expect(d.templateData.action).toBe("Scan Enemy Ship");
  });

  it("falls back to style 0 when the caller could not resolve one", () => {
    expect(pureShipPhaseChatData({ shipName: "S", phaseLabel: "P", actionLabel: "A" }).style).toBe(0);
    expect(pureShipPhaseChatData({ shipName: "S", phaseLabel: "P", actionLabel: "A", style: undefined }).style).toBe(0);
    expect(pureShipPhaseChatData({ shipName: "S", phaseLabel: "P", actionLabel: "A", style: 2 }).style).toBe(2);
  });
});
```

- [ ] **Step 2: 跑一遍，看它失败**

Run: `npx vitest run test/repair-sheets.test.mjs`
Expected: FAIL —— `Error: Failed to load url ../scripts/repairs/spacecraft-phase.pure.mjs`（文件还不存在）。

- [ ] **Step 3: 写出 `pureShipPhaseChatData`**

新建 `scripts/repairs/spacecraft-phase.pure.mjs`：

```js
/**
 * Assemble the template payload for one Space Combat Phase declaration.
 *
 * The system builds the same payload at module/sheets/spacecraft-sheet.mjs:479-488
 * but stamps `type: CONST.CHAT_MESSAGE_TYPES.OTHER` (:487); that constant was
 * deleted from Foundry (common/constants.mjs:242 keeps only CHAT_MESSAGE_STYLES,
 * whose OTHER is 0), so reading `.OTHER` off `undefined` throws and all four
 * Submit buttons are dead.
 *
 * We reuse the system's own chat template, so this function returns the template
 * DATA rather than HTML. The three key names are fixed by
 * systems/alienrpg/templates/chat/ship-combat.hbs — {{actorname}}, {{phaseName}},
 * {{action}} — and Handlebars escapes all three, so no escaping is done here.
 *
 * @param {object} input
 * @param {string} input.shipName
 * @param {string} input.phaseLabel
 * @param {string} input.actionLabel
 * @param {number} [input.style]  numeric CHAT_MESSAGE_STYLES value resolved by the caller
 * @returns {{templateData: {actorname: string, phaseName: string, action: string}, style: number}}
 */
export function pureShipPhaseChatData({ shipName, phaseLabel, actionLabel, style }) {
  return {
    templateData: {
      actorname: String(shipName ?? ""),
      phaseName: String(phaseLabel ?? ""),
      action: String(actionLabel ?? ""),
    },
    style: Number.isInteger(style) ? style : 0,
  };
}
```

- [ ] **Step 4: 跑一遍，看它通过**

Run: `npx vitest run test/repair-sheets.test.mjs`
Expected: 2 passed。

- [ ] **Step 5: 追加失败测试 —— 下拉持久化的纯逻辑与两个探针谓词**

追加到 `test/repair-sheets.test.mjs`（顶部 import 行改成把新符号一并引进来）：

```js
import {
  pureMergePhaseSelection,
  purePhaseSelectNames,
  pureRestorePlan,
  pureShipPhaseIsBuggy,
  pureShipPhaseSelectsAreVolatile,
} from "../scripts/repairs/spacecraft-phase.pure.mjs";

describe("phase selection persistence", () => {
  it("lists the four select names the template ships", () => {
    expect(purePhaseSelectNames()).toEqual(["sensorPhase", "pilotPhase", "gunnerPhase", "engineerPhase"]);
  });

  it("merges one selection without mutating the stored object", () => {
    const stored = { sensorPhase: "scan" };
    const next = pureMergePhaseSelection(stored, "pilotPhase", "evade");
    expect(next).toEqual({ sensorPhase: "scan", pilotPhase: "evade" });
    expect(stored).toEqual({ sensorPhase: "scan" });
  });

  it("hands _prepareContext only the stored, known, non-empty select names", () => {
    // The template does {{selectOptions config.sensor_list selected=sensorPhase}},
    // so these keys go on the CONTEXT ROOT, not under system.*.
    expect(pureRestorePlan({ sensorPhase: "scan", bogus: "x", pilotPhase: "" })).toEqual({ sensorPhase: "scan" });
    expect(pureRestorePlan(null)).toEqual({});
  });
});

describe("spacecraft phase probes", () => {
  const buggy = `static async _onShipPhase(event, target) {
    let chatData = { sound: CONFIG.sounds.lock, type: CONST.CHAT_MESSAGE_TYPES.OTHER };
    return ChatMessage.create(chatData);
  }`;
  const fixed = `static async _onShipPhase(event, target) {
    let chatData = { sound: CONFIG.sounds.lock, style: CONST.CHAT_MESSAGE_STYLES.OTHER };
    return ChatMessage.create(chatData);
  }`;

  it("reports the 4.1.13 handler as buggy and a corrected one as fixed", () => {
    expect(pureShipPhaseIsBuggy(buggy)).toBe(true);
    expect(pureShipPhaseIsBuggy(fixed)).toBe(false);
  });

  it("reports an unreadable handler as not buggy so we never patch a symbol we cannot see", () => {
    expect(pureShipPhaseIsBuggy("")).toBe(false);
    expect(pureShipPhaseIsBuggy(null)).toBe(false);
  });

  it("reports the shipped select names as volatile and schema paths as safe", () => {
    expect(pureShipPhaseSelectsAreVolatile(`<select id="sensorPhase" name="sensorPhase">`)).toBe(true);
    expect(pureShipPhaseSelectsAreVolatile(`<select name="system.general.sensorPhase.value">`)).toBe(false);
    expect(pureShipPhaseSelectsAreVolatile("")).toBe(false);
  });
});
```

- [ ] **Step 6: 跑一遍，看它失败**

Run: `npx vitest run test/repair-sheets.test.mjs`
Expected: FAIL —— `SyntaxError: The requested module '../scripts/repairs/spacecraft-phase.pure.mjs' does not provide an export named 'pureMergePhaseSelection'`。

- [ ] **Step 7: 补齐 `spacecraft-phase.pure.mjs` 的其余四个纯函数**

追加到 `scripts/repairs/spacecraft-phase.pure.mjs`：

```js
const PHASE_SELECT_NAMES = ["sensorPhase", "pilotPhase", "gunnerPhase", "engineerPhase"];

/** The four <select name> values in templates/actor/spacecraft-combat-phases.hbs:6/13/20/27. */
export function purePhaseSelectNames() {
  return [...PHASE_SELECT_NAMES];
}

/** @returns {object} a new stored-selections object; never mutates the input. */
export function pureMergePhaseSelection(stored, name, value) {
  return { ...(stored ?? {}), [name]: String(value ?? "") };
}

/**
 * Turn the stored flag into the extra context keys _prepareContext must expose.
 *
 * templates/actor/spacecraft-combat-phases.hbs asks Handlebars for
 * `{{selectOptions config.sensor_list selected=sensorPhase ...}}`, i.e. a key
 * named exactly like the select, sitting on the context ROOT. The system's
 * _prepareContext (module/sheets/spacecraft-sheet.mjs:100-153) never sets them.
 *
 * @param {object|null|undefined} stored
 * @returns {object} keys to merge into the render context; {} when nothing is stored
 */
export function pureRestorePlan(stored) {
  if (!stored || typeof stored !== "object") return {};
  const out = {};
  for (const name of PHASE_SELECT_NAMES) {
    if (stored[name]) out[name] = String(stored[name]);
  }
  return out;
}

function squash(source) {
  return String(source ?? "").replace(/\s+/g, " ");
}

/**
 * @param {string} source Function.prototype.toString() of the registered ShipPhaseSubmit handler
 * @returns {boolean} true while the handler still reads the deleted CONST.CHAT_MESSAGE_TYPES
 */
export function pureShipPhaseIsBuggy(source) {
  const s = squash(source);
  if (!s) return false; // nothing readable -> nothing to patch
  return s.includes("CHAT_MESSAGE_TYPES");
}

/**
 * @param {string} templateSource text of templates/actor/spacecraft-combat-phases.hbs
 * @returns {boolean} true while the phase selects are named off-schema and get dropped on submit
 */
export function pureShipPhaseSelectsAreVolatile(templateSource) {
  const s = squash(templateSource);
  if (!s) return false;
  return s.includes('name="sensorPhase"');
}
```

- [ ] **Step 8: 跑一遍，看它通过**

Run: `npx vitest run test/repair-sheets.test.mjs`
Expected: 8 passed。

- [ ] **Step 9: 提交**

```bash
git add scripts/repairs/spacecraft-phase.pure.mjs test/repair-sheets.test.mjs && git commit -m "$(cat <<'EOF'
fix(repairs): 飞船战斗阶段的纯逻辑与两个探针谓词

spacecraft-sheet.mjs:487 用的 CONST.CHAT_MESSAGE_TYPES 已从 Foundry 移除
（common/constants.mjs:242 只剩 CHAT_MESSAGE_STYLES，OTHER 是 0），读 .OTHER
抛 TypeError，四个 Submit 按钮全是哑的。聊天模板数据抽成不碰 Foundry 全局的
纯函数，并复用系统自己的 templates/chat/ship-combat.hbs，键名与它的三个占位符
对齐；署名不在纯层拼，它需要真的 Actor/TokenDocument。

另加下拉持久化：pureRestorePlan 产出的是要并进 _prepareContext 的上下文键
（模板写的是 selectOptions ... selected=sensorPhase，系统从没设过这个键），
以及两个可双向单测的探针谓词——上游改用 CHAT_MESSAGE_STYLES 后
pureShipPhaseIsBuggy 立刻返回 false，补丁自动退休，不会双重修复。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 10: 追加失败测试 —— 酸血的参数归一化、骰池、分支决策与探针**

追加到 `test/repair-sheets.test.mjs`：

```js
import {
  pureAcidArgs,
  pureAcidNoBloodContent,
  pureAcidPlan,
  pureAcidPool,
  pureAcidRollIsBuggy,
} from "../scripts/repairs/creature-acid.pure.mjs";

describe("pureAcidArgs", () => {
  it("normalises the creature sheet's (event, target) call shape", () => {
    const target = { dataset: { roll: "6", label: "Acid Splash", spbutt: "" } };
    const r = pureAcidArgs(new Event("click"), target);
    expect(r.fromEvent).toBe(true);
    expect(r.dataset).toEqual({ roll: "6", label: "Acid Splash", spbutt: "" });
  });

  it("normalises token-action-hud-alien's (actor, {roll,label}) call shape", () => {
    const r = pureAcidArgs({ id: "xeno1", name: "Drone" }, { roll: "6", label: "Acid Splash" });
    expect(r.fromEvent).toBe(false);
    expect(r.dataset).toEqual({ roll: "6", label: "Acid Splash" });
  });

  it("sniffs on instanceof Event, never on argument position", () => {
    expect(pureAcidArgs({ id: "a", type: "creature" }, { roll: "0" }).fromEvent).toBe(false);
  });

  it("hands back the slot-0 actor only when slot 0 is not an Event", () => {
    expect(pureAcidArgs({ id: "xeno1" }, {}).actorArg).toEqual({ id: "xeno1" });
    expect(pureAcidArgs(new Event("click"), { dataset: {} }).actorArg).toBe(null);
  });

  it("survives a caller that passes nothing usable", () => {
    expect(pureAcidArgs(undefined, undefined).dataset).toEqual({});
  });
});

describe("pureAcidPool", () => {
  it("treats the schema initial '-' as zero", () => {
    expect(pureAcidPool("-")).toBe(0);
  });

  it("treats empty, null and undefined as zero", () => {
    expect(pureAcidPool("")).toBe(0);
    expect(pureAcidPool(null)).toBe(0);
    expect(pureAcidPool(undefined)).toBe(0);
  });

  it("accepts both the DOMStringMap string and a real number", () => {
    expect(pureAcidPool("6")).toBe(6);
    expect(pureAcidPool(6)).toBe(6);
  });

  it("clamps nonsense and negatives to zero", () => {
    expect(pureAcidPool("abc")).toBe(0);
    expect(pureAcidPool(-3)).toBe(0);
  });
});

describe("pureAcidPlan", () => {
  it("rolls when the creature really has acid blood", () => {
    expect(pureAcidPlan({ roll: "6", label: "Acid Splash" })).toEqual({ action: "roll", pool: 6 });
  });

  it("announces 'no acid blood' for the schema initial '-'", () => {
    expect(pureAcidPlan({ roll: "-", label: "Acid Splash" })).toEqual({ action: "announce", pool: 0 });
  });

  it("stays silent on a zero-rated armour click, exactly like actor.mjs:2564", () => {
    expect(pureAcidPlan({ roll: "0", spbutt: "armor" })).toEqual({ action: "skip", pool: 0 });
  });

  it("still rolls armour with a real rating", () => {
    expect(pureAcidPlan({ roll: "3", spbutt: "armor" })).toEqual({ action: "roll", pool: 3 });
  });
});

describe("pureAcidNoBloodContent", () => {
  it("reuses the system's own two localization keys, inventing no text", () => {
    // systems/alienrpg/module/documents/actor.mjs:2600-2601 builds exactly this,
    // and lang/en.json:39-40 already carries both keys.
    expect(pureAcidNoBloodContent((k) => k)).toBe(
      "<h2>ALIENRPG.AcidAttack</h2><h4><i>ALIENRPG.AcidBlood</i></h4>"
    );
  });
});

describe("pureAcidRollIsBuggy", () => {
  it("reports the 4.1.13 body as buggy", () => {
    expect(pureAcidRollIsBuggy(`async creatureAcidRoll(actor, dataset) { if (dataset.dataset.roll !== 0) {} }`)).toBe(true);
  });

  it("reports a numeric-guard rewrite as fixed", () => {
    expect(pureAcidRollIsBuggy(`async creatureAcidRoll(actor, dataset) { const r = Number(dataset?.dataset?.roll); if (r > 0) {} }`)).toBe(false);
  });

  it("reports an unreadable body as not buggy", () => {
    expect(pureAcidRollIsBuggy("")).toBe(false);
  });
});
```

- [ ] **Step 11: 跑一遍，看它失败**

Run: `npx vitest run test/repair-sheets.test.mjs -t "pureAcidArgs"`
Expected: FAIL —— `Error: Failed to load url ../scripts/repairs/creature-acid.pure.mjs`。

- [ ] **Step 12: 写出 `creature-acid.pure.mjs`**

新建 `scripts/repairs/creature-acid.pure.mjs`：

```js
/**
 * `creatureAcidRoll` (module/documents/actor.mjs:2557) is reached from two callers
 * with two incompatible shapes:
 *
 *  - the creature sheet (module/sheets/creature-sheet.mjs:567) forwards
 *    ApplicationV2's (event, target) verbatim, so slot 0 is a PointerEvent and
 *    slot 1 is an HTMLElement whose .dataset is a DOMStringMap (every value is a
 *    STRING);
 *  - token-action-hud-alien (scripts/roll-handler.js:246-253) calls it with a real
 *    Actor and a plain object {roll, label} that has no .dataset at all, which is
 *    why that button throws today.
 *
 * Sniff on `instanceof Event` — never on argument position — because upstream may
 * re-register the action in the {handler, buttons} form and reshape the arguments
 * again without warning.
 *
 * @returns {{dataset: object, fromEvent: boolean, actorArg: object|null}}
 */
export function pureAcidArgs(a, b) {
  const fromEvent = typeof Event !== "undefined" && a instanceof Event;
  const actorArg = !fromEvent && a && typeof a === "object" ? a : null;
  const raw = b && typeof b === "object" ? b : {};
  const inner = raw.dataset && typeof raw.dataset === "object" ? raw.dataset : raw;
  return { dataset: { ...inner }, fromEvent, actorArg };
}

/**
 * A creature's acid splash rating lives in a StringField whose schema initial is
 * the literal "-" (module/data/actor-creature.mjs:66). Number("-") is NaN, so a
 * plain coercion would hand the system a NaN dice pool.
 *
 * @param {string|number|null|undefined} rawRating
 * @returns {number} dice pool, 0 meaning "this creature has no acid blood"
 */
export function pureAcidPool(rawRating) {
  const n = Number(rawRating);
  if (!Number.isFinite(n) || n <= 0) return 0;
  return n;
}

/**
 * Decide which of the three things an acid click means. Mirrors the system's own
 * three outcomes (module/documents/actor.mjs:2563-2618) but on a real number:
 *
 *  - "skip":     an armour click with no rating — actor.mjs:2564 returns silently;
 *  - "announce": no acid blood — the card the system can never reach, because
 *                `dataset.dataset.roll !== 0` compares a DOMStringMap STRING to
 *                the NUMBER 0 and is therefore always true;
 *  - "roll":     hand the pool to the system and let it run its dialog + yzeRoll.
 *
 * @param {object} dataset already normalised by pureAcidArgs
 * @returns {{action: "skip"|"announce"|"roll", pool: number}}
 */
export function pureAcidPlan(dataset) {
  const pool = pureAcidPool(dataset?.roll);
  if (String(dataset?.spbutt ?? "") === "armor" && pool < 1) return { action: "skip", pool: 0 };
  if (pool === 0) return { action: "announce", pool: 0 };
  return { action: "roll", pool };
}

/**
 * The "this creature has no acid blood" card body, byte-for-byte the markup the
 * system builds at module/documents/actor.mjs:2600-2601, using the system's own
 * two keys (lang/en.json:39-40) so translated worlds keep their existing text.
 *
 * We build this card ourselves instead of letting the system's else-branch do it,
 * because that branch stamps `speaker: ChatMessage.getSpeaker({actor: actor.id})`
 * — a STRING id, which matches none of getSpeaker's actor/token branches
 * (client/documents/chat-message.mjs:231-247) and so bylines the card as whatever
 * token happens to be selected, or as the GM.
 *
 * @param {(key: string) => string} localize injected so this stays Foundry-free
 * @returns {string}
 */
export function pureAcidNoBloodContent(localize) {
  return `<h2>${localize("ALIENRPG.AcidAttack")}</h2><h4><i>${localize("ALIENRPG.AcidBlood")}</i></h4>`;
}

/**
 * @param {string} source Function.prototype.toString() of creatureAcidRoll
 * @returns {boolean} true while the string-vs-number guard is still in place
 */
export function pureAcidRollIsBuggy(source) {
  const s = String(source ?? "").replace(/\s+/g, " ");
  if (!s) return false;
  return s.includes("dataset.dataset.roll !== 0");
}
```

- [ ] **Step 13: 跑一遍，看它通过**

Run: `npx vitest run test/repair-sheets.test.mjs`
Expected: 25 passed。

- [ ] **Step 14: 提交**

```bash
git add scripts/repairs/creature-acid.pure.mjs test/repair-sheets.test.mjs && git commit -m "$(cat <<'EOF'
fix(repairs): 酸血调用的参数归一化、骰池与三分支决策（纯函数）

creature-sheet.mjs:567 把 ApplicationV2 的 (event, target) 原样转给
actor.mjs:2557 的 creatureAcidRoll，于是 actor 位上是 PointerEvent；
token-action-hud-alien 那边用真 actor + 普通对象 {roll,label} 调它，
没有 .dataset，今天直接 TypeError。归一化按 instanceof Event 嗅探，
不按参数位置，两种调用形态收敛成同一个形状，并额外交回槽 0 的真 actor。

acidSplash.value 的 schema 初始值是字符串 "-"，Number("-") 是 NaN，
pureAcidPool 把它归成真正的数字 0；pureAcidPlan 把系统那三个出口
（armor 静默早退 / 没有酸血 / 正常掷骰）变成可单测的决策。
「没有酸血」那张卡的正文用系统自己的 AcidAttack + AcidBlood 两个键拼，
不新造文案，汉化世界照旧显示译文。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 15: 追加失败测试 —— 主机技能行与改名 token 两条的纯逻辑**

追加到 `test/repair-sheets.test.mjs`：

```js
import { pureRangedRowIsBuggy, pureRangedRowSelector } from "../scripts/repairs/mainframe-skill-row.pure.mjs";
import { pureTokenDefaults, pureTokenLookupIsBuggy } from "../scripts/repairs/token-defaults.pure.mjs";

describe("mainframe Ranged Combat row", () => {
  // verbatim from systems/alienrpg/templates/item/item-sheet.hbs:64
  const shipped = `<label for='system.modifiers.skills.rangedCbt.value' class='resource-label rollcomputer' data-roll='{{system.modifiers.skills.rangedCbt.value}}' data-label='{{localize "ALIENRPG.SkillrangedCbt"}}'>X</label>`;
  const patched = shipped.replace("class='resource-label rollcomputer'", "class='resource-label rollcomputer' data-action='RollComputer'");

  it("reports the shipped row as missing its action and a patched row as fixed", () => {
    expect(pureRangedRowIsBuggy(`line1\n${shipped}\nline3`)).toBe(true);
    expect(pureRangedRowIsBuggy(`line1\n${patched}\nline3`)).toBe(false);
  });

  it("reports a template without that row, or an unreadable one, as not buggy", () => {
    expect(pureRangedRowIsBuggy("<p>nothing here</p>")).toBe(false);
    expect(pureRangedRowIsBuggy("")).toBe(false);
  });

  it("targets the row by its stable for-attribute", () => {
    expect(pureRangedRowSelector()).toBe('label[for="system.modifiers.skills.rangedCbt.value"]');
  });
});

describe("pureTokenDefaults", () => {
  const HOSTILE = -1;

  it("makes an NPC token hostile and unlinked", () => {
    expect(pureTokenDefaults({ actorType: "creature", isNpc: true, disposition: 0, actorLink: true, hostile: HOSTILE }))
      .toEqual({ disposition: -1, actorLink: false });
  });

  it("changes nothing when the token is already hostile and unlinked", () => {
    expect(pureTokenDefaults({ actorType: "creature", isNpc: true, disposition: -1, actorLink: false, hostile: HOSTILE }))
      .toEqual({});
  });

  it("leaves spacecraft alone", () => {
    expect(pureTokenDefaults({ actorType: "spacecraft", isNpc: true, disposition: 0, actorLink: true, hostile: HOSTILE }))
      .toEqual({});
  });

  it("leaves player characters alone", () => {
    expect(pureTokenDefaults({ actorType: "character", isNpc: false, disposition: 0, actorLink: true, hostile: HOSTILE }))
      .toEqual({});
  });
});

describe("pureTokenLookupIsBuggy", () => {
  it("reports the 4.1.13 name lookup as buggy and a document.actor rewrite as fixed", () => {
    expect(pureTokenLookupIsBuggy(`const aTarget = game.actors.find((i) => i.name === tokenData.name);`)).toBe(true);
    expect(pureTokenLookupIsBuggy(`const aTarget = document.actor;`)).toBe(false);
    expect(pureTokenLookupIsBuggy("")).toBe(false);
  });
});
```

- [ ] **Step 16: 跑一遍，看它失败**

Run: `npx vitest run test/repair-sheets.test.mjs -t "mainframe Ranged Combat row"`
Expected: FAIL —— `Error: Failed to load url ../scripts/repairs/mainframe-skill-row.pure.mjs`。

- [ ] **Step 17: 写出 `mainframe-skill-row.pure.mjs`**

新建 `scripts/repairs/mainframe-skill-row.pure.mjs`：

```js
const SELECTOR = 'label[for="system.modifiers.skills.rangedCbt.value"]';

/** The one row in templates/item/item-sheet.hbs:64 that lost its data-action. */
export function pureRangedRowSelector() {
  return SELECTOR;
}

/**
 * templates/item/item-sheet.hbs ships four skill rows for a Mainframe item
 * (header.type.value === "9"). Three of them (:52 Comtech, :56 Piloting,
 * :60 Observation) carry data-action='RollComputer'; the fourth (:64 Ranged
 * Combat) does not, so clicking it does nothing. The action itself is registered
 * (module/sheets/item-sheet.mjs:30) and its handler exists (:597) — only the
 * attribute is missing.
 *
 * @param {string} templateSource text of templates/item/item-sheet.hbs
 * @returns {boolean} true while that row still lacks the attribute
 */
export function pureRangedRowIsBuggy(templateSource) {
  const line = String(templateSource ?? "")
    .split("\n")
    .find((l) => l.includes("<label") && l.includes("system.modifiers.skills.rangedCbt.value"));
  if (!line) return false;
  return !line.includes("RollComputer");
}
```

- [ ] **Step 18: 写出 `token-defaults.pure.mjs`**

新建 `scripts/repairs/token-defaults.pure.mjs`：

```js
/**
 * Decide what a freshly created token needs changed.
 *
 * The system does this at module/alienrpg.mjs:366-376 inside an anonymous
 * `Hooks.on("preCreateToken", ...)` callback, and resolves the actor by
 * `game.actors.find((i) => i.name === tokenData.name)` — rename the token and
 * `aTarget` is undefined, so the very next line (`aTarget.type`) throws and
 * neither default is set. The two conditions below are the system's own
 * (`aTarget.type !== "spacecraft" && aTarget.system.header.npc`).
 *
 * The numeric hostile value is injected rather than read from CONST so this stays
 * free of Foundry globals; the caller passes CONST.TOKEN_DISPOSITIONS.HOSTILE.
 *
 * @param {object} input
 * @param {string} input.actorType
 * @param {boolean} input.isNpc
 * @param {number} input.disposition
 * @param {boolean} input.actorLink
 * @param {number} input.hostile
 * @returns {{disposition?: number, actorLink?: boolean}} empty object = leave it alone
 */
export function pureTokenDefaults({ actorType, isNpc, disposition, actorLink, hostile }) {
  if (actorType === "spacecraft") return {};
  if (!isNpc) return {};
  const changes = {};
  if (disposition !== hostile) changes.disposition = hostile;
  if (actorLink !== false) changes.actorLink = false;
  return changes;
}

/**
 * @param {string} source text of module/alienrpg.mjs
 * @returns {boolean} true while the hook still resolves the actor by token name
 */
export function pureTokenLookupIsBuggy(source) {
  const s = String(source ?? "").replace(/\s+/g, " ");
  if (!s) return false;
  return s.includes("i.name === tokenData.name");
}
```

- [ ] **Step 19: 跑一遍，看它通过**

Run: `npx vitest run test/repair-sheets.test.mjs`
Expected: 33 passed。

- [ ] **Step 20: 提交**

```bash
git add scripts/repairs/mainframe-skill-row.pure.mjs scripts/repairs/token-defaults.pure.mjs test/repair-sheets.test.mjs && git commit -m "$(cat <<'EOF'
fix(repairs): 主机技能行选择器与 token 默认值判定（纯函数）

item-sheet.hbs:64 的 RANGED COMBAT 一行漏了 data-action='RollComputer'，
它的三个兄弟 :52/:56/:60 都有；动作注册（item-sheet.mjs:30）与处理函数
（:597）都在，只差这一个属性。谓词按行定位，上游补上属性后立刻转 false。

alienrpg.mjs:366 的 preCreateToken 按 token 名字反查 actor，token 一改名
aTarget 就是 undefined，下一行读 .type 抛错，敌对与非链接两项都设不上。
把「该改成什么」抽成不碰 Foundry 全局的纯判定，敌对常量由调用方注入，
两个条件与系统原判据逐字一致。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 21: 追加失败测试 —— 负重**

追加到 `test/repair-sheets.test.mjs`：

```js
import {
  pureAcceptedNames,
  pureCapacityStrength,
  pureEncumbrance,
  pureEncumbranceIsBuggy,
  purePackMule,
  pureTalentNames,
} from "../scripts/repairs/encumbrance.pure.mjs";

describe("pureCapacityStrength", () => {
  it("prefers the gear-modified Strength over the base value", () => {
    // module/data/actor-character.mjs:457-460 stores base + item/talent modifiers in .mod
    expect(pureCapacityStrength({ value: 4, mod: 5 })).toBe(5);
  });

  it("falls back to the base value, then to zero", () => {
    expect(pureCapacityStrength({ value: 4 })).toBe(4);
    expect(pureCapacityStrength(undefined)).toBe(0);
    expect(pureCapacityStrength({ value: "x", mod: null })).toBe(0);
  });
});

describe("Pack Mule identification", () => {
  it("splits and normalises the alias list", () => {
    expect([...pureAcceptedNames(" pack mule | Lastdragger ")]).toEqual(["PACK MULE", "LASTDRAGGER"]);
    expect([...pureAcceptedNames("")]).toEqual([]);
  });

  it("offers both the display name and Babele's preserved original name", () => {
    expect(pureTalentNames({ name: "背包骡子", flags: { babele: { originalName: "Pack Mule" } } }))
      .toEqual(["背包骡子", "PACK MULE"]);
  });

  it("matches an English world by display name", () => {
    expect(purePackMule([{ name: " pack mule " }], pureAcceptedNames("PACK MULE"))).toBe(true);
  });

  it("matches a Babele-translated world by flags.babele.originalName", () => {
    const talents = [{ name: "背包骡子", flags: { babele: { originalName: "Pack Mule" } } }];
    expect(purePackMule(talents, pureAcceptedNames("PACK MULE"))).toBe(true);
  });

  it("is false for unrelated, empty and missing talent lists", () => {
    expect(purePackMule([{ name: "Nerves of Steel" }], pureAcceptedNames("PACK MULE"))).toBe(false);
    expect(purePackMule([], pureAcceptedNames("PACK MULE"))).toBe(false);
    expect(purePackMule(undefined, pureAcceptedNames("PACK MULE"))).toBe(false);
  });
});

describe("pureEncumbrance", () => {
  it("is Strength x4 and rounds the displayed weight to two decimals", () => {
    // the system's duplicate `value` key lets the raw float through: 0.1*3 = 0.30000000000000004
    const e = pureEncumbrance({ strength: 4, totalWeight: 0.1 * 3, packMule: false });
    expect(e.max).toBe(16);
    expect(e.value).toBe(0.3);
  });

  it("doubles capacity for Pack Mule", () => {
    expect(pureEncumbrance({ strength: 4, totalWeight: 0, packMule: true }).max).toBe(32);
  });

  it("uses the gear-modified Strength, so an exosuit gives 20kg not 16kg", () => {
    expect(pureEncumbrance({ strength: 5, totalWeight: 0, packMule: false }).max).toBe(20);
  });

  it("never reports a fresh zero-Strength character as encumbered", () => {
    const e = pureEncumbrance({ strength: 0, totalWeight: 1, packMule: false });
    expect(e.max).toBe(0);
    expect(e.pct).toBe(0);
    expect(e.encumbered).toBe(false);
  });

  it("is encumbered strictly above half capacity, with an integer pct capped at 99", () => {
    expect(pureEncumbrance({ strength: 4, totalWeight: 8, packMule: false }).encumbered).toBe(false);
    expect(pureEncumbrance({ strength: 4, totalWeight: 8.5, packMule: false }).encumbered).toBe(true);
    const over = pureEncumbrance({ strength: 4, totalWeight: 100, packMule: false });
    expect(over.pct).toBe(99);
    expect(Number.isInteger(over.pct)).toBe(true);
  });
});

describe("pureEncumbranceIsBuggy", () => {
  it("reports the 4.1.13 body as buggy, tabs and all", () => {
    expect(pureEncumbranceIsBuggy(`\t\t\tmax: actorData.actor.system.attributes.str.value * 4,`)).toBe(true);
  });

  it("reports a str.mod rewrite as fixed", () => {
    expect(pureEncumbranceIsBuggy(`      max: actorData.actor.system.attributes.str.mod * 4,`)).toBe(false);
    expect(pureEncumbranceIsBuggy("")).toBe(false);
  });
});
```

- [ ] **Step 22: 跑一遍，看它失败**

Run: `npx vitest run test/repair-sheets.test.mjs -t "pureCapacityStrength"`
Expected: FAIL —— `Error: Failed to load url ../scripts/repairs/encumbrance.pure.mjs`。

- [ ] **Step 23: 写出 `encumbrance.pure.mjs`**

新建 `scripts/repairs/encumbrance.pure.mjs`。注意 `purePackMule` 顶上那段 JSDoc 是**必须逐字写上**的：它记录了本模组对模块契约 §7「永不比对显示名」的一处有据可查的偏离。

```js
/**
 * Carrying capacity is driven by Strength AFTER gear and talent modifiers.
 * module/data/actor-character.mjs:457-460 writes that into `attributes.str.mod`;
 * the sheets read `attributes.str.value` (the base score) instead. Colony actors
 * have no `str` at all (module/data/actor-colony.mjs:20), hence the double
 * fallback.
 *
 * @param {{value?: number, mod?: number}|undefined} strAttr
 * @returns {number}
 */
export function pureCapacityStrength(strAttr) {
  for (const candidate of [strAttr?.mod, strAttr?.value]) {
    const n = Number(candidate);
    if (Number.isFinite(n)) return Math.max(0, n);
  }
  return 0;
}

/**
 * Build the set of accepted Pack Mule names from a "|"-separated alias list.
 * @param {string} rawList
 * @returns {Set<string>}
 */
export function pureAcceptedNames(rawList) {
  const parts = String(rawList ?? "")
    .split("|")
    .map((s) => s.trim().toUpperCase())
    .filter(Boolean);
  return new Set(parts);
}

/**
 * Every name one talent can be recognised by, normalised.
 *
 * Babele (modules/babele, script/translation/document-translation.js:73-93,
 * v2.9.1) preserves the untranslated name in `flags.babele.originalName`, which
 * is what makes a Chinese world's talent still recognisable as Pack Mule.
 *
 * @param {object} talent
 * @returns {string[]}
 */
export function pureTalentNames(talent) {
  const raw = [talent?.name, talent?.flags?.babele?.originalName, talent?.originalName];
  const out = [];
  for (const n of raw) {
    if (typeof n !== "string") continue;
    const norm = n.trim().toUpperCase();
    if (norm && !out.includes(norm)) out.push(norm);
  }
  return out;
}

/**
 * The system does `i.name.toUpperCase() === "PACK MULE"`
 * (module/sheets/character-sheet.mjs:491 and its two twins), which silently
 * halves a translated world's carrying capacity.
 *
 * ---------------------------------------------------------------------------
 * DOCUMENTED DEVIATION from module contract §7 rule 2 ("never compare display
 * names"). The rule's escape hatch for factory-pack documents is to address them
 * by immutable `_id` / UUID, and that is what we would do — but the Pack Mule
 * talent ships inside alien-evolved-corerules as a single Snappy-compressed
 * Adventure document in a LevelDB pack, and its per-item `_id` is not readable
 * without importing the Adventure first. There is therefore no verifiable id to
 * bind to at the moment this function runs. We fall back to
 * `flags.babele.originalName` first (an exact, translation-proof key) and to an
 * explicit GM-editable alias list (`AEA.repair.packMuleAliases`) second. This
 * deviation is retired in phase 2, when `registry.declare` gains `kind:"item"`.
 * ---------------------------------------------------------------------------
 *
 * @param {Array<object>|undefined} talents
 * @param {Set<string>} acceptedNames
 * @returns {boolean}
 */
export function purePackMule(talents, acceptedNames) {
  for (const t of talents ?? []) {
    for (const n of pureTalentNames(t)) {
      if (acceptedNames.has(n)) return true;
    }
  }
  return false;
}

/**
 * Capacity is Strength x4 weight units, doubled by Pack Mule; strictly over half
 * of it the character counts as encumbered.
 *
 * Three defects are repaired relative to module/sheets/character-sheet.mjs:483-506:
 *  1. the object literal writes `value` twice, so the rounded figure is
 *     overwritten by the raw float ("0.30000000000000004kg" in the header);
 *  2. capacity comes from `str.value` (base Strength) instead of `str.mod`;
 *  3. `pct` divides by a zero capacity, producing Infinity, which
 *     Math.min(..., 99) turns into 99 and `pct > 50` turns into a permanent
 *     Encumbered condition on every brand-new character.
 *
 * @param {object} input
 * @param {number} input.strength
 * @param {number} input.totalWeight
 * @param {boolean} input.packMule
 * @returns {{max: number, value: number, pct: number, encumbered: boolean}}
 */
export function pureEncumbrance({ strength, totalWeight, packMule }) {
  const str = Number.isFinite(Number(strength)) ? Math.max(0, Number(strength)) : 0;
  const max = str * (packMule ? 8 : 4);
  const raw = Number(totalWeight);
  const value = Math.round((Number.isFinite(raw) ? raw : 0) * 100) / 100;
  if (max <= 0) return { max: 0, value, pct: 0, encumbered: false };
  return {
    max,
    value,
    pct: Math.min(Math.round((value * 100) / max), 99),
    encumbered: value > max / 2,
  };
}

/**
 * @param {string} source Function.prototype.toString() of _computeEncumbrance
 * @returns {boolean} true while capacity is still computed from base Strength
 */
export function pureEncumbranceIsBuggy(source) {
  const s = String(source ?? "").replace(/\s+/g, " ");
  if (!s) return false;
  return s.includes("str.value * 4");
}
```

- [ ] **Step 24: 跑一遍，看它通过**

Run: `npx vitest run test/repair-sheets.test.mjs`
Expected: 47 passed。

- [ ] **Step 25: 提交**

```bash
git add scripts/repairs/encumbrance.pure.mjs test/repair-sheets.test.mjs && git commit -m "$(cat <<'EOF'
fix(repairs): 负重的容量、取整与背包骡子识别（纯函数）

character-sheet.mjs:483-506（synthetic-sheet.mjs:470 与 colony-sheet.mjs:331
是逐字孪生）里三处缺陷：对象字面量把 value 写了两遍，取整结果被原始浮点覆盖；
容量按 str.value 而非装备加成后的 str.mod 算（actor-character.mjs:457-460
才是加成后的值）；max 为 0 时 pct 是 Infinity，被 Math.min(...,99) 变成 99，
再被 pct > 50 判成「已负重」，新建角色一开卡就顶着状态图标。

背包骡子不再只比对显示名：先看 Babele 保留的 flags.babele.originalName
（babele 2.9.1 在 document-translation.js:73-93 写入），再看别名表。
purePackMule 的 JSDoc 里逐字记下了这是对契约 §7「永不比对显示名」的一处
有据可查的偏离，以及为什么拿不到可核实的 _id（corerules 那颗
Snappy 压缩的 Adventure 单文档），二期给 registry 加 kind:"item" 后收回。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 26: 写两个桩测试（先失败）—— 登记与开关**

两个文件都用契约 §0.3 的共享 Foundry 桩，**不要**自己造全局、也不要改桩。

新建 `test/repair-registration.test.mjs`：

```js
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";
import { features } from "../scripts/kernel/features.mjs";
import { patches } from "../scripts/kernel/patches.mjs";
import { selftest } from "../scripts/kernel/selftest.mjs";
import { spacecraftPhaseRepair } from "../scripts/repairs/spacecraft-phase.mjs";
import { creatureAcidRepair } from "../scripts/repairs/creature-acid.mjs";
import { mainframeSkillRowRepair } from "../scripts/repairs/mainframe-skill-row.mjs";
import { tokenDefaultsRepair } from "../scripts/repairs/token-defaults.mjs";
import { encumbranceRepair } from "../scripts/repairs/encumbrance.mjs";

beforeEach(() => installFoundryStub());
afterEach(() => uninstallFoundryStub());

/** patches.status() -> [{id, type, target, applied, reason, fixedIn}], six fields exactly. */
function rowOf(id) {
  return patches.status().find((r) => r.id === id);
}

/** Every repair owns one GM-only switch, registered under the repair's own id. */
function switchOf(id) {
  return features.all().find((d) => d.id === id);
}

async function selftestIds() {
  return (await selftest.runAll()).map((r) => r.id);
}

describe("repair registration", () => {
  it("registers the spacecraft phase repair", async () => {
    expect(spacecraftPhaseRepair.id).toBe("spacecraft-phase");
    expect(typeof spacecraftPhaseRepair.install).toBe("function");
    spacecraftPhaseRepair.register();
    // Registered but not applied yet: status() must still list the row.
    expect(rowOf("spacecraft-phase-submit")).toMatchObject({
      type: "OVERRIDE",
      applied: false,
      reason: "pending",
      fixedIn: null,
    });
    expect(rowOf("spacecraft-phase-persist")).toMatchObject({
      type: "WRAPPER",
      target: "game.alienrpg.ActorSheets.alienrpgSpacecraftSheet.prototype._prepareContext",
      applied: false,
      reason: "pending",
    });
    expect(switchOf("spacecraft-phase")).toMatchObject({ default: "full", gmOnly: true });
    expect(await selftestIds()).toContain("repair.spacecraft-phase");
  });

  it("registers the creature acid repair", async () => {
    creatureAcidRepair.register();
    expect(rowOf("creature-acid-args")).toMatchObject({
      type: "MIXED",
      target: "game.alienrpg.alienrpgActor.prototype.creatureAcidRoll",
      applied: false,
      reason: "pending",
    });
    expect(switchOf("creature-acid")).toMatchObject({ default: "full", gmOnly: true });
    expect(await selftestIds()).toContain("repair.creature-acid");
  });

  it("registers the mainframe skill row repair", async () => {
    mainframeSkillRowRepair.register();
    expect(rowOf("mainframe-skill-row")).toMatchObject({ type: "HOOK", target: "renderalienrpgItemSheet" });
    expect(switchOf("mainframe-skill-row")).toMatchObject({ default: "full", gmOnly: true });
    expect(await selftestIds()).toContain("repair.mainframe-skill-row");
  });

  it("registers the token defaults repair", async () => {
    tokenDefaultsRepair.register();
    expect(rowOf("token-defaults-by-reference")).toMatchObject({ type: "HOOK", target: "preCreateToken" });
    expect(switchOf("token-defaults")).toMatchObject({ default: "full", gmOnly: true });
    expect(await selftestIds()).toContain("repair.token-defaults");
  });

  it("registers one encumbrance patch per sheet class", async () => {
    encumbranceRepair.register();
    for (const cls of ["alienrpgCharacterSheet", "alienrpgSyntheticSheet", "alienrpgColonySheet"]) {
      expect(rowOf(`encumbrance-capacity:${cls}`)).toMatchObject({
        type: "MIXED",
        target: `game.alienrpg.ActorSheets.${cls}.prototype._computeEncumbrance`,
        reason: "pending",
      });
    }
    expect(switchOf("encumbrance")).toMatchObject({ default: "full", gmOnly: true });
    expect(await selftestIds()).toContain("repair.encumbrance");
  });
});
```

新建 `test/repair-gate.test.mjs`：

```js
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { foundryStubContext, installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";
import { MID, SETTING_FEATURES } from "../scripts/const.mjs";
import { features } from "../scripts/kernel/features.mjs";
import { patches } from "../scripts/kernel/patches.mjs";
import { switchState } from "../scripts/repairs/repair-support.mjs";
import { tokenDefaultsRepair } from "../scripts/repairs/token-defaults.mjs";

beforeEach(() => installFoundryStub());
afterEach(() => uninstallFoundryStub());

let registered = false;

/** Reproduce main.mjs's init + ready order: register defs, then settings, then apply. */
async function bootRepair() {
  if (!registered) {
    tokenDefaultsRepair.register();
    registered = true;
  }
  features.registerSettings();
  await patches.applyAll();
}

/** A TokenDocument stand-in for a renamed, unlinked NPC token. */
function fakeTokenDocument() {
  const doc = {
    disposition: 0,
    actorLink: true,
    actor: { type: "creature", system: { header: { npc: true } } },
    changes: null,
  };
  doc.updateSource = (c) => { doc.changes = c; };
  return doc;
}

describe("repair switches", () => {
  it("reports 'unregistered' rather than throwing before settings exist", () => {
    // selftest.runAll() may be invoked before features.registerSettings(); reading an
    // unregistered world setting throws, so switchState swallows that one case.
    expect(switchState("token-defaults")).toBe("unregistered");
  });

  it("applies hostile + unlinked defaults to a renamed NPC token", async () => {
    await bootRepair();
    expect(switchState("token-defaults")).toBe("on");
    const doc = fakeTokenDocument();
    Hooks.callAll("preCreateToken", doc, {}, {}, "user1");
    expect(doc.changes).toEqual({ disposition: -1, actorLink: false });
  });

  it("does nothing once the GM turns that one switch off, and never double-hooks", async () => {
    await bootRepair();
    await game.settings.set(MID, SETTING_FEATURES, { "token-defaults": "off" });
    expect(features.mode("token-defaults")).toBe("off");

    const doc = fakeTokenDocument();
    Hooks.callAll("preCreateToken", doc, {}, {}, "user1");
    expect(doc.changes).toBe(null);

    // bootRepair() ran applyAll() twice across these two tests; apply() must be
    // idempotent, so exactly one preCreateToken handler is registered.
    const hooks = foundryStubContext().hooks.on.filter((h) => h.name === "preCreateToken");
    expect(hooks).toHaveLength(1);
  });
});
```

- [ ] **Step 27: 跑一遍，看它们失败**

Run: `npx vitest run test/repair-registration.test.mjs test/repair-gate.test.mjs`
Expected: FAIL —— `Error: Failed to load url ../scripts/repairs/spacecraft-phase.mjs`（副作用层六个文件都还不存在）。

- [ ] **Step 28: 写两个共用小工具**

新建 `scripts/repairs/repair-support.mjs`：

```js
import { MID } from "../const.mjs";
import { features } from "../kernel/features.mjs";

/**
 * The switch state of one repair, for selftest detail strings.
 *
 * `features.enabled()` reads a world setting, and reading an unregistered setting
 * throws by design. selftest.runAll() can legitimately be called before
 * features.registerSettings() has run (in tests, or from the console during a
 * broken init), so this one call site converts that specific failure into a word.
 * Execution-time gates must NOT use this — they call features.enabled() directly
 * so a genuinely broken setup fails loudly.
 *
 * @param {string} id
 * @returns {"on"|"off"|"unregistered"}
 */
export function switchState(id) {
  try {
    return features.enabled(id) ? "on" : "off";
  } catch {
    return "unregistered";
  }
}

/**
 * Start reading one system file so a probe() can inspect its text later.
 *
 * Foundry serves system sources and templates over HTTP from the world, so a
 * probe can literally read the shipped code and decide whether the defect is
 * still there. Under vitest there is no such server and no `window`, so we read
 * nothing and every probe falls back to "assume the defect is present".
 *
 * @param {string} path e.g. "systems/alienrpg/module/alienrpg.mjs"
 * @param {(text: string) => void} assign
 */
export function readSourceText(path, assign) {
  if (typeof window === "undefined" || typeof fetch !== "function") return;
  fetch(path)
    .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then(assign)
    .catch((err) => console.warn(`${MID} | could not read ${path}`, err));
}
```

- [ ] **Step 29: 写飞船阶段的副作用层**

新建 `scripts/repairs/spacecraft-phase.mjs`：

```js
import { MID, SYSTEM_ID } from "../const.mjs";
import { features } from "../kernel/features.mjs";
import { patches } from "../kernel/patches.mjs";
import { resolver } from "../kernel/resolver.mjs";
import { selftest } from "../kernel/selftest.mjs";
import { readSourceText, switchState } from "./repair-support.mjs";
import {
  pureMergePhaseSelection,
  purePhaseSelectNames,
  pureRestorePlan,
  pureShipPhaseChatData,
  pureShipPhaseIsBuggy,
  pureShipPhaseSelectsAreVolatile,
} from "./spacecraft-phase.pure.mjs";

const ID = "spacecraft-phase";
const SHEET_PATH = "game.alienrpg.ActorSheets.alienrpgSpacecraftSheet";
const CONTEXT_TARGET = `${SHEET_PATH}.prototype._prepareContext`;
const CHAT_TEMPLATE = `systems/${SYSTEM_ID}/templates/chat/ship-combat.hbs`;
const PHASE_TEMPLATE = `systems/${SYSTEM_ID}/templates/actor/spacecraft-combat-phases.hbs`;
const FLAG_PHASE = "shipPhase";

let phaseTemplateSource = null;
let originalHandler = null;
let submitApplied = false;
let persistApplied = false;

function sheetClass() {
  return game.alienrpg?.ActorSheets?.alienrpgSpacecraftSheet;
}

/** The registered action entry may be a bare function or a {handler, buttons} object. */
function registeredHandler() {
  const entry = sheetClass()?.DEFAULT_OPTIONS?.actions?.ShipPhaseSubmit;
  return typeof entry === "object" && entry !== null ? entry.handler : entry;
}

function chatStyleOther() {
  // CHAT_MESSAGE_TYPES was deleted from Foundry; CHAT_MESSAGE_STYLES replaced it
  // (common/constants.mjs:242, OTHER: 0). Read both so an older world still works.
  return CONST.CHAT_MESSAGE_STYLES?.OTHER ?? CONST.CHAT_MESSAGE_TYPES?.OTHER ?? 0;
}

/**
 * ApplicationV2 calls an action handler as handler.call(sheet, event, target).
 *
 * With the switch off we hand control straight back to whatever was in the
 * actions map before us, so a GM can disable this repair without reloading.
 */
async function onShipPhaseSubmit(event, target) {
  if (!features.enabled(ID)) return originalHandler?.call(this, event, target);
  event.preventDefault();
  const select = target.previousElementSibling;
  if (!select || select.tagName !== "SELECT") return;

  const actor = this.actor;
  const token = actor.token ?? resolver.soleToken(actor);
  const data = pureShipPhaseChatData({
    shipName: actor.name,
    phaseLabel: game.i18n.localize(`ALIENRPG.${select.name}`),
    // textContent, not innerText: we want the option's own label, and innerText
    // would force a layout pass on every click.
    actionLabel: select.selectedOptions[0]?.textContent?.trim() ?? "",
    style: chatStyleOther(),
  });

  // The four selects are named sensorPhase..engineerPhase rather than system.*
  // paths, so submitOnChange drops them and the declaration is lost on the next
  // render. Park it in our own flag; _prepareContext feeds it back in.
  await actor.setFlag(MID, FLAG_PHASE, pureMergePhaseSelection(actor.getFlag(MID, FLAG_PHASE), select.name, select.value));

  const chatData = {
    // `author` is the schema field (common/documents/chat-message.mjs:49);
    // the system's `user:` key has not existed since v12 and is dropped.
    author: game.user.id,
    // A real Actor + TokenDocument, not the system's bare string id: getSpeaker
    // matches none of its actor/token branches on a string
    // (client/documents/chat-message.mjs:231-247).
    speaker: ChatMessage.getSpeaker({ actor, token }),
    content: await foundry.applications.handlebars.renderTemplate(CHAT_TEMPLATE, data.templateData),
    style: data.style,
    sound: CONFIG.sounds.lock,
  };
  ChatMessage.applyRollMode(chatData, game.settings.get("core", "rollMode"));
  return ChatMessage.create(chatData);
}

/**
 * libWrapper WRAPPER on the sheet's own _prepareContext (module/sheets/
 * spacecraft-sheet.mjs:100). The template asks for `selected=sensorPhase` on the
 * context root and the system never puts it there, so we merge the stored
 * declaration back in — no DOM poking, no post-render flicker.
 */
async function prepareContextWithPhases(wrapped, options) {
  const context = await wrapped(options);
  if (!features.enabled(ID)) return context;
  return Object.assign(context, pureRestorePlan(this.document?.getFlag(MID, FLAG_PHASE)));
}

export const spacecraftPhaseRepair = {
  id: ID,

  register() {
    features.register({ id: ID, default: "full", gmOnly: true, requires: [] });
    readSourceText(PHASE_TEMPLATE, (t) => { phaseTemplateSource = t; });

    patches.register({
      id: "spacecraft-phase-submit",
      type: "OVERRIDE",
      target: `${SHEET_PATH}.DEFAULT_OPTIONS.actions.ShipPhaseSubmit`,
      minSystem: "4.1.13",
      fixedIn: null,
      probe: () => pureShipPhaseIsBuggy(String(registeredHandler() ?? "")),
      apply: () => {
        // The class body copied a REFERENCE to _onShipPhase into this map when it
        // was evaluated, so replacing the static method would change nothing —
        // the map entry itself has to be overwritten. Idempotent: once ours is in
        // place we never capture it as the "original" again.
        if (registeredHandler() === onShipPhaseSubmit) return;
        originalHandler = registeredHandler() ?? null;
        sheetClass().DEFAULT_OPTIONS.actions.ShipPhaseSubmit = onShipPhaseSubmit;
        submitApplied = true;
      },
    });

    patches.register({
      id: "spacecraft-phase-persist",
      type: "WRAPPER",
      target: CONTEXT_TARGET,
      minSystem: "4.1.13",
      fixedIn: null,
      // Unread template = assume the defect is present; merging keys that the
      // template no longer reads is harmless, so an early apply costs nothing.
      probe: () => (phaseTemplateSource === null ? true : pureShipPhaseSelectsAreVolatile(phaseTemplateSource)),
      apply: () => {
        if (persistApplied) return;
        libWrapper.register(MID, CONTEXT_TARGET, prepareContextWithPhases, "WRAPPER");
        persistApplied = true;
      },
    });

    selftest.register({
      id: "repair.spacecraft-phase",
      label: "AEA.selftest.repair.spacecraft-phase",
      run: () => {
        const stillBuggy = pureShipPhaseIsBuggy(String(registeredHandler() ?? ""));
        const ours = registeredHandler() === onShipPhaseSubmit;
        return {
          ok: (ours || !stillBuggy) && persistApplied,
          detail:
            `handler=${ours ? "ours" : stillBuggy ? "system-buggy" : "system-fixed"}` +
            ` submit=${submitApplied} persist=${persistApplied ? "ours" : "off"}` +
            ` selects=${phaseTemplateSource === null ? "unread" : pureShipPhaseSelectsAreVolatile(phaseTemplateSource) ? "volatile" : "schema-backed"}` +
            ` names=${purePhaseSelectNames().join(",")} switch=${switchState(ID)}`,
        };
      },
    });
  },

  /**
   * Nothing to do at ready: every effect of this repair lands in the two
   * patches' apply(), which patches.applyAll() has already run earlier in the
   * same ready phase. Present because every repair module exposes the same shape.
   */
  install() {},
};
```

- [ ] **Step 30: 跑这一条，看它通过**

Run: `npx vitest run test/repair-registration.test.mjs -t "registers the spacecraft phase repair"`
Expected: 这条 `it` 不再因 `spacecraft-phase.mjs` 缺失而失败。整个文件仍会报 `Failed to load url ../scripts/repairs/creature-acid.mjs`——属正常，继续下一步。

- [ ] **Step 31: 写酸血的副作用层**

新建 `scripts/repairs/creature-acid.mjs`：

```js
import { MID } from "../const.mjs";
import { features } from "../kernel/features.mjs";
import { patches } from "../kernel/patches.mjs";
import { resolver } from "../kernel/resolver.mjs";
import { selftest } from "../kernel/selftest.mjs";
import { switchState } from "./repair-support.mjs";
import {
  pureAcidArgs,
  pureAcidNoBloodContent,
  pureAcidPlan,
  pureAcidRollIsBuggy,
} from "./creature-acid.pure.mjs";

const ID = "creature-acid";
const TARGET = "game.alienrpg.alienrpgActor.prototype.creatureAcidRoll";

let applied = false;

/**
 * The "this creature has no acid blood" card, spoken by the creature itself.
 *
 * The system builds the same card at module/documents/actor.mjs:2599-2617 but
 * stamps `ChatMessage.getSpeaker({actor: actor.id})` — a STRING. getSpeaker
 * (client/documents/chat-message.mjs:231-247) matches a string against neither
 * its token branch nor its actor branch, so it falls through to "whatever token
 * is currently selected", and failing that to the GM. Passing the real Actor and
 * its TokenDocument takes CASE 1, which is what makes three Drones dragged off
 * one base sheet produce three distinguishable bylines.
 */
async function announceNoAcidBlood(actor) {
  const token = actor?.token ?? resolver.soleToken(actor);
  const chatData = {
    author: game.user.id,
    speaker: ChatMessage.getSpeaker({ actor, token }),
    content: pureAcidNoBloodContent(game.i18n.localize.bind(game.i18n)),
    // Same {actorUuid, tokenUuid} shape the roll bus stores on roll cards, so a
    // GM can still tell three identically-named Drones apart from the card.
    flags: { [MID]: { acid: resolver.refs(actor, token) } },
  };
  ChatMessage.applyRollMode(chatData, game.settings.get("core", "rollMode"));
  return ChatMessage.create(chatData);
}

/**
 * libWrapper MIXED wrapper: (wrapped, ...originalArgs), `this` unchanged.
 *
 * `this` is the Actor the method was invoked on — the synthetic token actor for
 * an unlinked token — and both real callers already invoke it on the right actor
 * (the sheet does `this.actor.creatureAcidRoll(...)`, the HUD does
 * `actor.creatureAcidRoll(actor, rData)`), so `this` is always the right answer
 * even though the sheet mistakenly passes a PointerEvent in slot 0.
 *
 * KNOWN BOUNDARY: for a creature that DOES have acid blood we delegate to the
 * system, which rolls through `yze.yzeRoll(..., actor.id)` and builds the card's
 * speaker inside YZEDiceRoller.mjs:398-401 from that string id. That byline
 * cannot be corrected from here: the card is created deep inside yzeRoll, and
 * yzeRoll is the roll bus's exclusive libWrapper target, while substituting an
 * Actor for the id would break `game.actors.get(actorid)` (YZEDiceRoller.mjs:201,
 * 210, 222, 231, 572) and the `data-actor-id` attribute at :62.
 */
function aeaAcidRoll(wrapped, a, b) {
  if (!features.enabled(ID)) return wrapped(a, b);
  const norm = pureAcidArgs(a, b);
  const actor = this ?? norm.actorArg;
  const rating = norm.dataset.roll ?? actor?.system?.general?.acidSplash?.value;
  const plan = pureAcidPlan({ ...norm.dataset, roll: rating });
  if (plan.action === "skip") return;
  if (plan.action === "announce") return announceNoAcidBlood(actor);
  return wrapped(actor, { dataset: { ...norm.dataset, roll: plan.pool } });
}

export const creatureAcidRepair = {
  id: ID,

  register() {
    features.register({ id: ID, default: "full", gmOnly: true, requires: [] });

    patches.register({
      id: "creature-acid-args",
      type: "MIXED",
      target: TARGET,
      minSystem: "4.1.13",
      fixedIn: null,
      probe: () => pureAcidRollIsBuggy(String(game.alienrpg?.alienrpgActor?.prototype?.creatureAcidRoll ?? "")),
      apply: () => {
        if (applied) return;
        libWrapper.register(MID, TARGET, aeaAcidRoll, "MIXED");
        applied = true;
      },
    });

    selftest.register({
      id: "repair.creature-acid",
      label: "AEA.selftest.repair.creature-acid",
      run: () => {
        // Once libWrapper is in front of it, the prototype property is the
        // dispatcher, whose source no longer carries the buggy guard. Same
        // predicate as the probe, used as a post-condition.
        const stillBuggy = pureAcidRollIsBuggy(String(game.alienrpg?.alienrpgActor?.prototype?.creatureAcidRoll ?? ""));
        return {
          ok: applied && !stillBuggy,
          detail: `applied=${applied} guardStillVisible=${stillBuggy} switch=${switchState(ID)}`,
        };
      },
    });
  },

  /** Nothing to do at ready: the wrapper is installed by the patch's apply(). */
  install() {},
};
```

- [ ] **Step 32: 跑这一条，看它通过**

Run: `npx vitest run test/repair-registration.test.mjs -t "registers the creature acid repair"`
Expected: 这条 `it` 通过。整个文件仍会报 `Failed to load url ../scripts/repairs/mainframe-skill-row.mjs`——继续下一步。

- [ ] **Step 33: 写主机技能行的副作用层**

新建 `scripts/repairs/mainframe-skill-row.mjs`。注意 `apply()` 里挂钩子这件事：模块契约 §0.2 禁止特性与修复自挂钩子，**唯一例外**是 `type: "HOOK"` 的已登记补丁可以在自己的 `apply()` 里挂领域钩子（内核独占的 `preCreateChatMessage` / `createChatMessage` / `diceSoNiceRollComplete` / `renderChatMessageHTML` / `renderChatMessage` 与五个生命周期钩子除外）。用这条例外必须同时满足：只在 `apply()` 里挂一次、钩子名写进 `target` 供 `status()` 展示、重复 `apply()` 不重复挂、回调第一行查 `features.enabled(id)`。下面这两条 HOOK 补丁四条全占。

```js
import { SYSTEM_ID } from "../const.mjs";
import { features } from "../kernel/features.mjs";
import { patches } from "../kernel/patches.mjs";
import { selftest } from "../kernel/selftest.mjs";
import { readSourceText, switchState } from "./repair-support.mjs";
import { pureRangedRowIsBuggy, pureRangedRowSelector } from "./mainframe-skill-row.pure.mjs";

const ID = "mainframe-skill-row";
const HOOK = "renderalienrpgItemSheet";
const ITEM_TEMPLATE = `systems/${SYSTEM_ID}/templates/item/item-sheet.hbs`;

let templateSource = null;
let hookId = null;

/**
 * ApplicationV2 render hook: (application, element, context, options).
 * Setting `dataset.action` is exactly the attribute the template forgot; the
 * sheet's own delegated listener (which reads `closest("[data-action]")`) then
 * routes the click to the already-registered RollComputer handler.
 * Idempotent: a row that upstream has fixed already carries the attribute.
 */
function injectRangedCombatAction(_app, element) {
  if (!features.enabled(ID)) return;
  const label = element?.querySelector?.(pureRangedRowSelector());
  if (!label || label.dataset.action) return;
  label.dataset.action = "RollComputer";
}

export const mainframeSkillRowRepair = {
  id: ID,

  register() {
    features.register({ id: ID, default: "full", gmOnly: true, requires: [] });
    readSourceText(ITEM_TEMPLATE, (t) => { templateSource = t; });

    patches.register({
      id: "mainframe-skill-row",
      type: "HOOK",
      target: HOOK,
      minSystem: "4.1.13",
      fixedIn: null,
      // Unread template = assume the attribute is missing; the injector already
      // returns early when it is present, so an early apply is a no-op.
      probe: () => (templateSource === null ? true : pureRangedRowIsBuggy(templateSource)),
      apply: () => {
        if (hookId !== null) return; // idempotent: never hook twice
        hookId = Hooks.on(HOOK, injectRangedCombatAction);
      },
    });

    selftest.register({
      id: "repair.mainframe-skill-row",
      label: "AEA.selftest.repair.mainframe-skill-row",
      run: () => ({
        ok: hookId !== null,
        detail: `hook=${hookId ?? "none"} template=${
          templateSource === null ? "unread" : pureRangedRowIsBuggy(templateSource) ? "missing-action" : "already-fixed"
        } selector=${pureRangedRowSelector()} switch=${switchState(ID)}`,
      }),
    });
  },

  /** Nothing to do at ready: the render hook is installed by the patch's apply(). */
  install() {},
};
```

- [ ] **Step 34: 跑这一条，看它通过**

Run: `npx vitest run test/repair-registration.test.mjs -t "registers the mainframe skill row repair"`
Expected: 这条 `it` 通过。整个文件仍会报 `Failed to load url ../scripts/repairs/token-defaults.mjs`——继续下一步。

- [ ] **Step 35: 写改名 token 的副作用层**

**先读这一段再动手**：原始清单要求「包裹系统那处按名查找」。那个接缝**不存在**——缺陷在 `module/alienrpg.mjs:366` 一个**直接传给 `Hooks.on("preCreateToken", ...)` 的匿名箭头函数**里，它不是任何对象的属性，没有可寻址路径，libWrapper 只能包裹「某对象/原型上的一个属性」，因此无法以它为 target。我们能做的是**注册自己的 `preCreateToken`**（走的正是上一步说的 `type:"HOOK"` 例外），用 `document.actor`（这一刻 Foundry 已经把 TokenDocument 关联到真实 Actor）而不是按名字扫 `game.actors`，把结果做对。系统那次抛错压不掉，但 Foundry 的 `Hooks` 分发本来就 `try/catch` 每个回调，所以它只在控制台留一条错误，不会中断 token 创建。

新建 `scripts/repairs/token-defaults.mjs`：

```js
import { SYSTEM_ID } from "../const.mjs";
import { features } from "../kernel/features.mjs";
import { patches } from "../kernel/patches.mjs";
import { selftest } from "../kernel/selftest.mjs";
import { readSourceText, switchState } from "./repair-support.mjs";
import { pureTokenDefaults, pureTokenLookupIsBuggy } from "./token-defaults.pure.mjs";

const ID = "token-defaults";
const HOOK = "preCreateToken";
const SYSTEM_SOURCE = `systems/${SYSTEM_ID}/module/alienrpg.mjs`;

let systemSource = null;
let hookId = null;

/**
 * preCreateToken hook: (document, data, options, userId).
 * `document.actor` is already resolved here, so a renamed token resolves exactly
 * like an unrenamed one. HOSTILE has been -1 since v0.7 and is spelled out as a
 * fallback so this stays runnable when CONST is shaped differently.
 */
function onPreCreateToken(document) {
  if (!features.enabled(ID)) return;
  const actor = document.actor;
  if (!actor) return;
  const changes = pureTokenDefaults({
    actorType: actor.type,
    isNpc: !!actor.system?.header?.npc,
    disposition: document.disposition,
    actorLink: document.actorLink,
    hostile: CONST.TOKEN_DISPOSITIONS?.HOSTILE ?? -1,
  });
  if (Object.keys(changes).length) document.updateSource(changes);
}

export const tokenDefaultsRepair = {
  id: ID,

  register() {
    features.register({ id: ID, default: "full", gmOnly: true, requires: [] });
    readSourceText(SYSTEM_SOURCE, (t) => { systemSource = t; });

    patches.register({
      id: "token-defaults-by-reference",
      type: "HOOK",
      target: HOOK,
      minSystem: "4.1.13",
      fixedIn: null,
      // Unread source = assume the name lookup is still there; our hook writes the
      // same two values the system intends, so an early apply is harmless.
      probe: () => (systemSource === null ? true : pureTokenLookupIsBuggy(systemSource)),
      apply: () => {
        if (hookId !== null) return; // idempotent: never hook twice
        hookId = Hooks.on(HOOK, onPreCreateToken);
      },
    });

    selftest.register({
      id: "repair.token-defaults",
      label: "AEA.selftest.repair.token-defaults",
      run: () => ({
        ok: hookId !== null,
        detail: `hook=${hookId ?? "none"} systemLookup=${
          systemSource === null
            ? "unread"
            : pureTokenLookupIsBuggy(systemSource)
              ? "by-name (upstream defect present)"
              : "fixed upstream — this repair is now redundant"
        } switch=${switchState(ID)}`,
      }),
    });
  },

  /** Nothing to do at ready: the hook is installed by the patch's apply(). */
  install() {},
};
```

- [ ] **Step 36: 跑开关闸门测试，看它通过**

Run: `npx vitest run test/repair-gate.test.mjs`
Expected: 3 passed —— 开关未注册时 `switchState` 报 `"unregistered"`；开关为默认 `full` 时改名 NPC token 拿到 `{disposition:-1, actorLink:false}`；把 `token-defaults` 设成 `"off"` 后同一个钩子什么都不做，且 `preCreateToken` 上只有一个处理函数（证明 `apply()` 幂等）。

- [ ] **Step 37: 写负重的副作用层**

新建 `scripts/repairs/encumbrance.mjs`：

```js
import { MID } from "../const.mjs";
import { features } from "../kernel/features.mjs";
import { patches } from "../kernel/patches.mjs";
import { selftest } from "../kernel/selftest.mjs";
import { switchState } from "./repair-support.mjs";
import {
  pureAcceptedNames,
  pureCapacityStrength,
  pureEncumbrance,
  pureEncumbranceIsBuggy,
  purePackMule,
} from "./encumbrance.pure.mjs";

const ID = "encumbrance";
const SHEET_CLASSES = ["alienrpgCharacterSheet", "alienrpgSyntheticSheet", "alienrpgColonySheet"];
const applied = new Set();

function protoOf(cls) {
  return game.alienrpg?.ActorSheets?.[cls]?.prototype;
}

function targetOf(cls) {
  return `game.alienrpg.ActorSheets.${cls}.prototype._computeEncumbrance`;
}

/**
 * libWrapper MIXED: (wrapped, ...originalArgs), `this` is the sheet.
 *
 * MIXED rather than OVERRIDE so that turning the switch off hands the original
 * method back without a world reload.
 *
 * colony-sheet.mjs never sets context.talents and colony actors have no `str`
 * (module/data/actor-colony.mjs:20); its copy of this method has no caller at
 * all today, but `talents ?? []` inside purePackMule plus the Strength fallback
 * mean ours is safe if upstream ever wires it up.
 */
function computeEncumbrance(wrapped, totalWeight, context) {
  if (!features.enabled(ID)) return wrapped(totalWeight, context);
  const enc = pureEncumbrance({
    strength: pureCapacityStrength(context?.actor?.system?.attributes?.str),
    totalWeight,
    packMule: purePackMule(context?.talents, pureAcceptedNames(game.i18n.localize("AEA.repair.packMuleAliases"))),
  });
  // Keep the system's side effect, driven off the corrected number. addCondition
  // is async and the system does not await it here either (character-sheet.mjs:503);
  // _computeEncumbrance is synchronous, so we keep the same fire-and-forget shape.
  if (enc.encumbered) this.actor.addCondition("encumbered");
  else this.actor.removeCondition("encumbered");
  return enc;
}

export const encumbranceRepair = {
  id: ID,

  register() {
    features.register({ id: ID, default: "full", gmOnly: true, requires: [] });

    for (const cls of SHEET_CLASSES) {
      patches.register({
        id: `encumbrance-capacity:${cls}`,
        type: "MIXED",
        target: targetOf(cls),
        minSystem: "4.1.13",
        fixedIn: null,
        probe: () => pureEncumbranceIsBuggy(String(protoOf(cls)?._computeEncumbrance ?? "")),
        apply: () => {
          if (applied.has(cls)) return;
          libWrapper.register(MID, targetOf(cls), computeEncumbrance, "MIXED");
          applied.add(cls);
        },
      });
    }

    selftest.register({
      id: "repair.encumbrance",
      label: "AEA.selftest.repair.encumbrance",
      run: () => {
        const state = (cls) =>
          applied.has(cls)
            ? "ours"
            : pureEncumbranceIsBuggy(String(protoOf(cls)?._computeEncumbrance ?? ""))
              ? "system-buggy"
              : "system-fixed-or-missing";
        const ok = SHEET_CLASSES.every((cls) => state(cls) !== "system-buggy");
        return {
          ok,
          detail: `${SHEET_CLASSES.map((cls) => `${cls}=${state(cls)}`).join(" ")} switch=${switchState(ID)}`,
        };
      },
    });
  },

  /** Nothing to do at ready: the three wrappers are installed by their apply(). */
  install() {},
};
```

- [ ] **Step 38: 跑两个桩测试，看它们全绿**

Run: `npx vitest run test/repair-registration.test.mjs test/repair-gate.test.mjs`
Expected: 8 passed（登记 5 条 + 开关 3 条）。

- [ ] **Step 39: 加 16 个 i18n 键**

`lang/en.json` —— 把下面三段并进**已有的**顶层 `AEA` 对象里（语言包是嵌套结构，顶层只允许一个 `AEA` 键；不要另起一个顶层键）：

```json
{
  "AEA": {
    "feature": {
      "spacecraft-phase": {
        "name": "Repair: Space Combat Phase submit",
        "hint": "Makes the four Submit buttons on a spacecraft's Space Combat Phases tab post their chat card again, and remembers the declared phase across re-renders. Turn off to restore the system's own (broken) handler."
      },
      "creature-acid": {
        "name": "Repair: Acid Splash button",
        "hint": "Accepts both the creature sheet's and Token Action HUD's call shapes, and posts the system's own 'no acid blood' card for creatures whose rating is '-'."
      },
      "mainframe-skill-row": {
        "name": "Repair: Mainframe Ranged Combat row",
        "hint": "Adds the missing click action to the RANGED COMBAT row of a Mainframe item sheet, so it rolls like the three rows above it."
      },
      "token-defaults": {
        "name": "Repair: NPC token defaults",
        "hint": "Sets hostile disposition and unlinked actor data on new NPC tokens by document reference, so a renamed token still gets them."
      },
      "encumbrance": {
        "name": "Repair: carrying capacity",
        "hint": "Computes capacity from gear-modified Strength, rounds the carried weight, stops brand-new characters from being permanently Encumbered, and recognises Pack Mule in translated worlds."
      }
    },
    "selftest": {
      "repair": {
        "spacecraft-phase": "Spacecraft phase Submit posts a card and the declaration survives a re-render",
        "creature-acid": "creatureAcidRoll accepts both call shapes and speaks as the creature",
        "mainframe-skill-row": "Mainframe item sheet: the RANGED COMBAT row carries its click action",
        "token-defaults": "New NPC tokens get hostile disposition and actorLink:false by document reference",
        "encumbrance": "All three character sheets compute capacity from modified Strength"
      }
    },
    "repair": {
      "packMuleAliases": "PACK MULE"
    }
  }
}
```

`lang/cn.json` 同样并入，`feature.*` 与 `selftest.*` 用中文，**`packMuleAliases` 的值保持英文原名不动**：

```json
{
  "AEA": {
    "feature": {
      "spacecraft-phase": {
        "name": "修复：太空战阶段提交",
        "hint": "让飞船「太空战阶段」页签上的四个 Submit 按钮重新能出聊天卡，并让宣告过的阶段活过重渲染。关掉即交还系统自己那个（会抛错的）处理函数。"
      },
      "creature-acid": {
        "name": "修复：酸血按钮",
        "hint": "同时接受怪物卡与 Token Action HUD 两种调用形态；评级为「-」的怪改出系统自带的「没有酸血」提示卡。"
      },
      "mainframe-skill-row": {
        "name": "修复：主机的远程战斗行",
        "hint": "给主机物品卡上 RANGED COMBAT 那一行补上缺失的点击动作，让它像上面三行一样能掷骰。"
      },
      "token-defaults": {
        "name": "修复：NPC token 默认值",
        "hint": "新建 NPC token 时按文档引用设敌对阵营与非链接数据，token 改过名也照样生效。"
      },
      "encumbrance": {
        "name": "修复：负重上限",
        "hint": "容量按装备加成后的力量算、载重取整、新建角色不再恒定「已负重」，汉化世界里也能认出背包骡子。"
      }
    },
    "selftest": {
      "repair": {
        "spacecraft-phase": "飞船阶段 Submit 能出卡，且宣告活过一次重渲染",
        "creature-acid": "creatureAcidRoll 接受两种调用形态，并以该怪物署名",
        "mainframe-skill-row": "主机物品卡：RANGED COMBAT 那一行带上了点击动作",
        "token-defaults": "新建 NPC token 按文档引用拿到敌对阵营与 actorLink:false",
        "encumbrance": "三张角色卡都按加成后的力量算容量"
      }
    },
    "repair": {
      "packMuleAliases": "PACK MULE"
    }
  }
}
```

为什么中文包不填中文译名：本模组不猜测别人的译名。中文世界里天赋叫什么由汉化包决定，而 Babele 会把英文原名保留在 `flags.babele.originalName` 里，`purePackMule` 优先看它，所以**Babele 汉化世界不需要改这个键**。只有「GM 手动改了天赋名、又没走 Babele」这一种情况需要把本世界的名字用 `|` 追加进来（例如 `"PACK MULE|背包骡子"`）。

- [ ] **Step 40: 提交六个副作用层文件与语言键**

```bash
git add scripts/repairs/repair-support.mjs scripts/repairs/spacecraft-phase.mjs scripts/repairs/creature-acid.mjs scripts/repairs/mainframe-skill-row.mjs scripts/repairs/token-defaults.mjs scripts/repairs/encumbrance.mjs test/repair-registration.test.mjs test/repair-gate.test.mjs lang/en.json lang/cn.json && git commit -m "$(cat <<'EOF'
fix(repairs): 五条表单类修复的副作用层、开关与自检登记

八个补丁分五个模块登记，每个模块形状统一为 {id, register, install}，
并各带一个 gmOnly 的特性开关（默认 full）：包裹与钩子在**执行时**查
features.enabled(id)，关掉即把控制权原样交回系统，不需要重载世界。
因此三条 libWrapper 补丁一律用 MIXED/WRAPPER 而非 OVERRIDE——OVERRIDE
拿不到 wrapped，做不到「关掉即放行」。

1. spacecraft-phase-submit（OVERRIDE）直接改写
   alienrpgSpacecraftSheet.DEFAULT_OPTIONS.actions.ShipPhaseSubmit
   ——类体求值时已把 _onShipPhase 的引用抄进这张表，改静态方法无效；
   style 读 CHAT_MESSAGE_STYLES?.OTHER ?? CHAT_MESSAGE_TYPES?.OTHER ?? 0，
   卡片复用系统自带的 templates/chat/ship-combat.hbs，署名改走
   ChatMessage.getSpeaker({actor, token})（系统传的是裸字符串 id，
   chat-message.mjs:231-247 一个分支都不匹配，最后署名成 GM）。
   spacecraft-phase-persist 改为包裹 _prepareContext：模板写的是
   selectOptions ... selected=sensorPhase，把宣告并进上下文即可，
   不必在渲染后改 DOM。
2. creature-acid-args（MIXED）归一化两种调用形态；没有酸血时不再指望
   系统那段够不着的 else，改由我们用它自己的两个 i18n 键出卡，
   署名带上真 TokenDocument，同一基础卡的两只 token 不再混为一谈。
   有酸血时仍交回系统掷骰——那张卡的署名在 yzeRoll 内部生成，
   属于 rollBus 独占的包裹目标，本任务改不了，已在 JSDoc 里写明边界。
3. mainframe-skill-row（HOOK）给 item-sheet.hbs:64 缺失的 RollComputer
   动作做渲染期注入，注入前先看属性在不在，上游补上后自动变成空操作。
4. token-defaults-by-reference（HOOK）改用 document.actor 设敌对与非链接。
   原方案「包裹 alienrpg.mjs:366 的按名查找」作废：那是传给 Hooks.on 的
   匿名箭头函数，没有可寻址路径，libWrapper 无法以它为 target。
5. encumbrance-capacity:<三张卡>（MIXED）三张卡同名的 _computeEncumbrance。

两条 HOOK 补丁按契约 §0.2 的唯一例外办：只在 apply() 里挂一次、钩子名写进
target、重复 apply() 不重复挂、回调第一行查开关。五条自检的 label 存的是
i18n 键而非已本地化文本——登记发生在 init，那时语言包还没加载。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 41: 把五个修复接进 `scripts/main.mjs`**

`main.mjs` 已经存在，里面有一串 `/* AEA-ANCHOR: xxx */` 锚点注释。**按锚点文本定位，不用行号**；本步只碰两个锚点，**不新增任何生命周期行**（`init` 的 `for (const r of REPAIRS) r.register()` 与 `ready` 的 `r.install?.()` 都是 main.mjs 自己的既有循环），也**不碰** `api` 对象、不碰四个 `ready.*` 子锚点。

其一，在 `/* AEA-ANCHOR: imports */` 这一行的**下方**插入这五行：

```js
import { spacecraftPhaseRepair } from "./repairs/spacecraft-phase.mjs";
import { creatureAcidRepair } from "./repairs/creature-acid.mjs";
import { mainframeSkillRowRepair } from "./repairs/mainframe-skill-row.mjs";
import { tokenDefaultsRepair } from "./repairs/token-defaults.mjs";
import { encumbranceRepair } from "./repairs/encumbrance.mjs";
```

其二，在 `/* AEA-ANCHOR: repairs */` 这一行的**下方**插入这五行（保留数组里可能已有的其它成员，不要删）：

```js
  spacecraftPhaseRepair,
  creatureAcidRepair,
  mainframeSkillRowRepair,
  tokenDefaultsRepair,
  encumbranceRepair,
```

- [ ] **Step 42: 跑全套**

Run: `npm test`
Expected: 全绿；本任务三个测试文件贡献 55 条（`test/repair-sheets.test.mjs` 47 条 + `test/repair-registration.test.mjs` 5 条 + `test/repair-gate.test.mjs` 3 条）。若报错指向别的任务的文件，先确认不是本任务引起的再继续。

- [ ] **Step 43: 手工验证（在本机冒烟世界里逐条做完，逐条记下观察结果）**

**MANUAL VERIFICATION**

前置：世界里启用本模组与 `alienrpg` 4.1.13，装好 libWrapper。以 GM 身份进入。

1. F12 控制台执行
   `game.modules.get("alien-evolved-automation").api.patches.status().filter(r => ["spacecraft-phase-submit","spacecraft-phase-persist","creature-acid-args","mainframe-skill-row","token-defaults-by-reference"].includes(r.id) || r.id.startsWith("encumbrance-capacity"))`。
   **期待**：8 行；`type` 依次是 OVERRIDE / WRAPPER / MIXED / HOOK / HOOK / MIXED / MIXED / MIXED；每行 `applied: true` 且 `reason: "ok"`；每行恰好六个字段 `{id, type, target, applied, reason, fixedIn}`。
2. 执行 `await game.modules.get("alien-evolved-automation").api.selftest.runAll()`，看 id 以 `repair.` 开头的五条。**期待**：五条全部 `ok: true`；`label` 是中文/英文句子而不是 `AEA.selftest.*` 字样（说明运行器做了本地化）；`repair.spacecraft-phase` 的 detail 含 `handler=ours persist=ours`，`repair.creature-acid` 含 `guardStillVisible=false`，`repair.encumbrance` 三张卡都是 `=ours`，五条的 detail 末尾都是 `switch=on`。
3. 打开一个 `spacecraft` 类型 actor → Space Combat Phases 页签 → Sensor Phase 下拉选一项（例如 Scan Enemy Ship）→ 点它右边的 Submit。**期待**：聊天区出现一张卡，标题是 Ship Combat，写着飞船名、"1. Sensor Phase"、所选行动，并播放锁扣音效；卡的署名是这艘飞船（不是 GM）。**修之前点这个按钮什么都不会发生**（控制台 TypeError）。
4. 不动下拉，改一下这艘船的血量让表单重渲染，再关掉表单重新打开。**期待**：Sensor Phase 下拉**仍停在刚才提交的那一项**（修之前每次重开都跳回第一项）。控制台 `game.actors.getName("<船名>").getFlag("alien-evolved-automation","shipPhase")` 应当能看到 `{sensorPhase: "..."}`。
5. 打开一只**有**酸血的怪（`system.general.acidSplash.value` 是数字，例如 Drone），点 Acid Splash。**期待**：弹出伤害对话框 → 填 0 → 掷骰 → 出卡。**已知边界**：这张掷骰卡的署名由系统在 `YZEDiceRoller.mjs:398-401` 用字符串 id 生成，本任务改不了（那是 rollBus 独占的包裹目标），所以它仍可能署成当前选中的 token 或 GM——记下你观察到的实际署名即可，不算失败。
6. 打开一只**没有**酸血的怪（Neomorph / Harvester / Lion Worm / Scorpionid / Swarm 这类 `acidSplash.value` 是 `"-"` 的），点 Acid Splash。**期待**：**不弹**伤害对话框，直接出一张标题 "Acid Blood"、正文 "This Creature does not have Acid Blood" 的卡（中文世界里显示对应译文，因为用的是系统自带的 `ALIENRPG.AcidAttack` / `ALIENRPG.AcidBlood` 两个键），**署名是这只怪**。修之前这张卡是永远走不到的死代码。
7. 若装了 `token-action-hud-alien`：选中一只有酸血的怪，在 HUD 上点 Acid Splash。**期待**：正常弹对话框出卡（修之前控制台 TypeError、按钮完全无反应）。
8. 把一只**没有**酸血的怪拖到场景上两次，得到同一张基础卡的两只 token（它们默认非链接、名字相同）。分别选中每一只、在各自的 token 卡上点 Acid Splash。**期待**：两张卡的 `speaker.token` **不同**——控制台执行 `game.messages.contents.slice(-2).map(m => m.speaker.token)` 应当得到两个不同的 token id（修之前两张卡的 speaker 取决于当时选中了谁，分不出是哪一只）。
9. 新建一个 `item` 类型物品，把 `header.type.value` 设为 `9`（Mainframe），打开物品卡 → 四行技能标签里点 **RANGED COMBAT**。**期待**：掷骰（修之前只有上面三行 Comtech / Piloting / Observation 能点）。右键点它应当走 `rollComputerMod`，同样有反应。
10. 新建一个 `character`，什么都不填就打开卡。**期待**：库存页头部读作 `0kg / 0kg`，进度条不是满的红条，token 上**没有** Encumbered 状态图标。
11. 给它 STR 4，放一件 0.1kg 的物品、数量 3。**期待**：读作 `0.3kg / 16kg`，**不是** `0.30000000000000004kg`。
12. 给这个角色装上一件把 STR 修正到 5 的物品（`system.modifiers.attributes.str` 加 1）。**期待**：容量变成 `20kg`（修之前恒为 16kg，因为读的是基础值）。
13. 拖入 corerules 的 Pack Mule 天赋。**期待**：容量翻倍到 `40kg`。若世界开了 Babele 中文层、天赋名显示为中文，**容量仍然翻倍**——这是本条修复的核心。若没翻倍，控制台执行 `game.actors.getName("<角色名>").items.filter(i => i.type === "talent").map(i => [i.name, i.flags?.babele?.originalName])` 看原名有没有被保留；没有的话把本世界的天赋名用 `|` 追加进 `AEA.repair.packMuleAliases`。
14. 把一个 `system.header.npc` 为 true 的 actor 拖到场景上，**先把这个 token 改名**再复制粘贴一份。**期待**：新 token 是红色敌对且非链接（控制台 `canvas.tokens.controlled[0].document.actorLink === false`、`.disposition === -1`）。控制台里系统那条 `Cannot read properties of undefined (reading 'type')` **仍会出现**——那是上游 `alienrpg.mjs:366` 抛的，我们压不掉，只能把结果做对。
15. 打开模组设置里的特性开关菜单，把「修复：负重上限」（`encumbrance`）关掉，**不重载世界**，重新打开步骤 12 那个角色的卡。**期待**：容量退回 `16kg`（系统原行为），说明开关在执行时生效；再打开就能看到步骤 10 那个新角色又顶上了 Encumbered 图标。把开关调回 full，容量立刻回到 `20kg`。五个开关在菜单里都应显示中文名与说明（来自 `AEA.feature.<id>.name` / `.hint`），且都标着 GM 专用。

- [ ] **Step 44: 提交接线与验收记录**

```bash
git add scripts/main.mjs && git commit -m "$(cat <<'EOF'
chore(repairs): 五条表单类修复接进 main.mjs 的两个锚点

按锚点文本定位：五行 import 插在 /* AEA-ANCHOR: imports */ 下方，
五个成员插在 /* AEA-ANCHOR: repairs */ 下方。本次不新增任何生命周期行，
也不碰 api 对象与四个 ready 子锚点——init 的
for (const r of REPAIRS) r.register() 与 ready 的 r.install?.()
都是 main.mjs 自己的既有循环。

手工验收 15 条已在本机冒烟世界跑通：八条补丁 applied/ok；飞船四个 Submit
按钮出卡且宣告活过重渲染与重开；有酸血的怪正常掷骰、没酸血的怪出系统自带
提示卡且两只同名 token 署名可区分；Mainframe 的 RANGED COMBAT 行可点；
新角色不再一开卡就「已负重」；装备加成计入容量；中文世界里背包骡子仍然翻倍；
改名 token 仍被设成敌对非链接；关掉某个开关即时退回系统原行为。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

**UPSTREAM PR（五个独立 PR，各自自足；每条缺陷各带独立版本探针，所以不要合并成一个）**

- **PR A** `module/sheets/spacecraft-sheet.mjs:487`：`type: CONST.CHAT_MESSAGE_TYPES.OTHER` → `style: CONST.CHAT_MESSAGE_STYLES.OTHER`（`CHAT_MESSAGE_TYPES` 已从 Foundry 的 `common/constants.mjs` 删除，只剩 `CHAT_MESSAGE_STYLES`，:242）。同一个对象里的 `other:` 与 `user:` 两个键都不是 ChatMessage 的 schema 字段（schema 里叫 `author`，见 `common/documents/chat-message.mjs:49`），建议一并改掉；`speaker: { actor: actorID }` 建议换成 `ChatMessage.getSpeaker({ actor: this.actor })`。附带：`templates/actor/spacecraft-combat-phases.hbs` 四个 select 的 `name` 应改成真实 schema 路径，或者在 `_prepareContext` 里把宣告值放上上下文根（模板已经写了 `selected=sensorPhase`，只是没人赋值），否则宣告仍然活不过一次重渲染。
- **PR B** `module/sheets/creature-sheet.mjs:567-569`：`static async _onCreatureAcidRoll(event, target) { this.actor.creatureAcidRoll(this.actor, target); }`（把误当 actor 的 Event 换成真 actor）；配套把 `module/documents/actor.mjs:2563` 的 `if (dataset.dataset.roll !== 0)` 改成 `const rating = Number(dataset?.dataset?.roll ?? dataset?.roll); if (Number.isFinite(rating) && rating > 0)`，让 `:2599-2617` 那张已经写好本地化文案的卡真正可达，同时兼容 `token-action-hud-alien` 传来的 `{roll, label}` 形状。`:2604` 与 `YZEDiceRoller.mjs:400` 的 `ChatMessage.getSpeaker({actor: <string>})` 都建议改传 Actor 文档本身——传字符串在 `client/documents/chat-message.mjs:231-247` 里一个分支都不匹配。
- **PR C** `templates/item/item-sheet.hbs:64`：给 RANGED COMBAT 的 `<label>` 补上 `data-action='RollComputer'`，与 :52 / :56 / :60 三个兄弟一致（一个属性）。
- **PR D** `module/alienrpg.mjs:366`：`game.actors.find((i) => i.name === tokenData.name)` → `document.actor`，并对 `null` 早退。改名 token 今天会让整个回调抛 TypeError。
- **PR E**（负重）`module/sheets/character-sheet.mjs:483-506` 及 `synthetic-sheet.mjs:470` / `colony-sheet.mjs:331`：删掉重复的 `value` 键、`str.value` → `str.mod`、`enc.pct` 除零前先判 `max > 0`、`actorData.talents` 改成 `(actorData.talents ?? [])`（colony 那份从来没有 `talents`，而且没有任何调用点，等于一颗哑弹）。PR 说明里要写明**这会改变现有角色的容量数字**。Pack Mule 的按名比对是否改成按 id，牵扯到他们自己的内容包，留给上游决定。
