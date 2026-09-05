> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 15 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 15: 载具与飞船开火路径 + 装备启用闸门（两条修复共用 rollBus 的 `itemRoll` 目标）

**Files:**
- Create: `scripts/repairs/item-roll-source.mjs`（**只拍源码快照，不注册任何包裹**）
- Create: `scripts/repairs/vehicle-roll-path-repair.pure.mjs`
- Create: `scripts/repairs/vehicle-roll-path-repair.mjs`
- Create: `scripts/repairs/gear-active-gate.pure.mjs`
- Create: `scripts/repairs/gear-active-gate.mjs`
- Create: `test/repair-vehicle-roll-path.test.mjs`（脱桩真单测）
- Create: `test/repair-gear-active-gate.test.mjs`（脱桩真单测）
- Create: `test/repair-item-roll-source.test.mjs`（用共享桩 `installFoundryStub()`）
- Create: `test/repair-item-roll-stages.test.mjs`（用共享桩 `installFoundryStub()`）
- Modify: `lang/en.json`、`lang/cn.json`（往已有的 `AEA` 对象里**合并**键，嵌套结构，不要整体替换）
- Modify: `scripts/main.mjs`（只在 `/* AEA-ANCHOR: imports */` 与 `/* AEA-ANCHOR: repairs */` 两个锚点后各插两行；`api` 对象一个字不改，四个 ready 子锚点一行不加）

**Interfaces:**
- Consumes:
  - `import { SYSTEM_ID } from "../const.mjs";`（`SYSTEM_ID === "alienrpg"`，用来读系统的 `evolved` 世界设置）
  - `import { features } from "../kernel/features.mjs";` —— `features.register({id, default, gmOnly, requires, hint})`、`features.enabled(id) -> boolean`、`features.all() -> def[]`、`features.registerSettings()`
  - `import { patches } from "../kernel/patches.mjs";` —— `patches.register({id, type, target, minSystem, fixedIn, probe(), apply()})`、`patches.applyAll()`、`patches.status() -> [{id, type, target, applied, reason, fixedIn}]`
  - `import { rollBus } from "../kernel/rollbus.mjs";` —— **只用一个成员**：`rollBus.addStage(target, {id, order, around})`，`target` 取 `"itemRoll"`，`around(next, args, thisArg)`
  - `import { selftest } from "../kernel/selftest.mjs";` —— `selftest.register({id, label, run})`、`selftest.runAll()`
  - `import { resolver } from "../kernel/resolver.mjs";` —— `resolver.actorById(id, {warn}) -> Actor|null`（遗留裸 actor id 专用；乘员槽存的正是裸 id）
  - `test/stubs/foundry.mjs` 的 `installFoundryStub(options) -> ctx` / `uninstallFoundryStub()`。**桩只读不改**：不得在测试文件里另造 `globalThis.game`，不得用私有 Map 顶替 `game.settings`
- Produces:
  - `item-roll-source.mjs`：`export const ITEM_ROLL_PATH`、`export function captureItemRollSource()`、`export function itemRollSource()`
  - `vehicle-roll-path-repair.pure.mjs`：`pureRangeReasonKey(kind)`、`pureVehicleRangeMod({band,maxRange,minRange})`、`pureMountTakeover({actorType,itemType,weaponType})`、`pureVehicleRollIsBuggy(source)`、`pureManoeuvreControlMissing(html)`
  - `vehicle-roll-path-repair.mjs`：`export const vehicleRollPathRepair = { id, register() }`
  - `gear-active-gate.pure.mjs`：`pureGearInactiveKey()`、`pureGearGate({actorType,itemType,active})`、`pureGearGateIsBuggy(source)`
  - `gear-active-gate.mjs`：`export const gearActiveGate = { id, register() }`
  - 特性开关三个（GM 可各自关掉）：`vehicle-roll-path-repair`、`vehicle-manoeuvrability-control`、`gear-active-gate`
  - 补丁三条：`vehicle-roll-path-repair`(MIXED)、`vehicle-manoeuvrability-control`(HOOK)、`gear-active-gate`(MIXED)
  - 自检三条：`repair.vehicle-roll-path-repair`、`repair.vehicle-manoeuvrability-control`、`repair.gear-active-gate`
  - **两个修复对象都不导出 `install()`**：它们没有 ready 阶段的活儿要干（stage 在 `apply()` 里就挂好了）。`main.mjs` 的修复循环写的是 `r.install?.()`，可选链本来就允许缺席。

---

## 背景（假设你完全不懂 Foundry VTT，也不懂 Alien RPG）

**Foundry VTT** 是网页版桌面 RPG 平台。「系统（system）」实现某套规则，「模组（module）」在系统之上打补丁。我们写的模组装在 `Data/modules/alien-evolved-automation/`，要修的系统在 `Data/systems/alienrpg/`（版本 **4.1.13**，本任务引用的每一行都逐行打开核对过）。

需要的概念，只讲你这一任务用得到的部分：

- **Document / Actor / Item**：世界里的数据对象。`Actor` 是角色，也可以是**载具（`vehicles`）**或**飞船（`spacecraft`）**；`Item` 是角色身上的东西（武器、装备、天赋……）。`item.actor` 指向拥有它的 actor，`item.system` 是它的数据。
- **Hook**：全局事件总线。`Hooks.on("renderXxx", fn)` 让 `fn` 在某个界面每次渲染完成后被调用，返回一个数字句柄，`Hooks.off(name, fn)` 摘除。
- **libWrapper**：一个已安装的第三方模组，用来安全地替换别人的函数。**本任务一次都不会直接用它**——原因见下一节。
- **DataModel / StringField / NumberField**：Foundry 用 schema 描述文档数据。`StringField` 存的永远是字符串。本任务的头号陷阱：`weapon` 的 `system.header.active` 是 `StringField`，出厂值是字符串 `"false"`，另有取值 `"fLocker"`（意为「放进储物柜」，`module/sheets/character-sheet.mjs:638` 写入）。**这两个值在 JavaScript 里都是 truthy**，所以判断必须写 `=== "true"`，写 `if (active)` 会永远为真。（`spacecraftweapons` 的 `header.active` 反而是 `BooleanField`，见 `module/data/item-spacecraftweapons.mjs:13`，所以判定要同时容忍字符串与真布尔。）
- **DOMStringMap**：HTML 元素的 `element.dataset`。`data-roll="3"` 读回来是字符串 `"3"`，不要拿它做 `=== 3` 比较。
- **DialogV2 / FormDataExtended**：Foundry V13+ 的对话框 API。`foundry.applications.api.DialogV2.wait({window, content, buttons})` 弹一个模态框并 `await` 到用户点按钮；按钮的 `callback: (event, button) => new foundry.applications.ux.FormDataExtended(button.form).object` 把表单读成普通对象。系统自己就是这么写的（`module/documents/item.mjs:379-394`），我们照抄。

规则侧只需知道两件事：掷骰是数「黑色基础骰」和「黄色压力骰」各掷出几颗 6；载具/飞船上的武器由**乘员**开火，用乘员的 `rangedCbt`（远程战斗）技能。

---

## 契约 v3.1 里直接决定本任务形状的四条（逐字遵守，不要自作主张）

**（1）`rollBus.addStage()` 是介入 `Item#roll` 的唯一合法方式。** 契约 §4 K1 [v3.1] 写死：

> **[v3.1] rollBus 独占这四个 libWrapper 目标。** lib-wrapper 1.13.5 对同一目标的重复注册会抛
> `A wrapper for '<target>' (ID=<n>) has already been registered by <module>.`，而
> `CONFIG.Item.documentClass = alienrpgItem`（`alienrpg.mjs:138`）与 rollBus 要包的是同一个方法 ——
> 若某条修复也去 `libWrapper.register` 同一目标，**它会静默变成空操作**。
> 因此：**任何需要介入这四个目标的修复或特性，一律经 `rollBus.addStage()` 排队，不得自己 register。**

```js
// target 取值："yzeRoll" | "abilityRoll" | "itemRoll" | "pushRoll"
// around(next, args, thisArg) -> any；恰好调用一次 next(args)，或有意短路（其返回值即最终返回值）
// order 小的先执行（更靠外）；同 order 按注册顺序。order 0 留给 rollBus 自己的采集帧
```

`args` 是实参数组（对 `Item#roll` 就是 `[right, dataset]`），`thisArg` 是被调用的那个 `Item`。
本任务两条修复各挂一个 stage 到 `"itemRoll"`：闸门 `order: 10`（外），炮击接管 `order: 20`（内）。
**绝不要写 `libWrapper.register`**——写了不报错、不生效，是最难查的那种坏法。
两个 stage 的短路都是**有意的**（闸门拦下掷骰、炮击整条路径由我们接管），除此之外必须恰好调一次 `next(args)`，**永远不要调两次**。

**（2）`apply()` 里 addStage 一定赶得上。** 契约 §5 的 ready 段是四个有序子锚点：
`ready.registry` → `ready.patches` → `ready.rollbus` → `ready.cards`。
`patches.applyAll()`（我们的 `apply()` 在这里跑）严格早于 `rollBus.install()`（它在那一刻才把各 target 的 stage 链组好并做唯一一次 libWrapper 注册）。所以 stage 写在 `apply()` 里既赶得上链条组装，又保留了「probe 说缺陷没了就不装」的退休能力。**不要**把 addStage 提前到 `register()`——那样补丁退休时 stage 还在。

**（3）每条修复都要有开关和自检。** 契约 §7 [v3.1]：

> **[v3.1] 一期 b 的每一条修复也必须有开关**：在 `register()` 里 `features.register({id, default:"full", gmOnly:true})`，
> 并在 `apply()` 装上的包裹/钩子**执行时**查 `features.enabled(id)` —— 关掉即原样放行，不需要重载世界。

本任务有三条补丁，就有三个同名开关、三条自检。查开关的位置是 **stage 与钩子回调的第一行**，不是 `apply()` 里。

**（4）HOOK 型补丁可以自己挂钩子，但有三条约束。** 契约 §0.2 [v3.1]：

> **[v3.1] 唯一例外**：`type: "HOOK"` 的**已登记补丁**可以在自己的 `apply()` 里挂领域钩子
> （本节判给内核独占的三组除外）。

内核独占的三组是 `preCreateChatMessage`/`createChatMessage`、`diceSoNiceRollComplete`、`renderChatMessageHTML`/`renderChatMessage`，外加五个生命周期钩子。我们要挂的是 `renderalienrpgVehicleSheet`，不在其中。三条约束逐条落地：
- 只在 `apply()` 内挂**一次**（用模块级 `manoeuvreHookId` 守卫，重复 `applyAll()` 不重复挂）；
- 句柄存下来，退休时可 `Hooks.off("renderalienrpgVehicleSheet", injectManoeuvreControl)`；
- 回调**第一行**查 `features.enabled(MANOEUVRE_ID)`；
- 钩子名写进补丁 def 的 `target` 字段（`"Hooks:renderalienrpgVehicleSheet"`）供 `status()` 展示。

**另外一条来自 §3 的范围裁决（决定聊天记录归谁）：**

> 4. `actorUuid` 一律记**掷骰者**——即 `ChatMessage.getSpeaker` 会解析到的那个 actor（载具/飞船挂载武器开火时是**乘员**），**不是**武器的主人。武器主人（载具本身）经 `itemUuid` 的 `.parent` 可达，不为此新增字段。

系统自己就是这么干的：`module/documents/item.mjs:399`（载具）与 `:495`（飞船远程）在建卡之前把 `actorid` 重新赋成 `fCrew[shooter].firerID`。所以**我们调 `yzeRoll` 时第 9 个参数 `actorid` 必须传乘员的 id**，不是载具的 id。这一条不需要写代码去「设置」什么，但传错了会在二期返工，所以 Step 27 与手工验证 C 各盯一次。

---

## 已核对的源码事实（引用行号前都重新打开过文件）

**A. 载具右键是死路** —— `module/documents/item.mjs:82` 起是右键分支；`:169` 是
```js
if (this.actor.type !== "vehicles" && this.actor.type !== "spacecraft") {
```
右键分支里没有任何 else 处理载具。载具/飞船右键 → 弹出 Base Modifier 框 → 点 Roll → 框关掉 → **什么都不发生**，控制台也不报错。左键分支的同一个判断在 `:215`，那一处是合法的（`:348` 起的 else 才是载具分支）。

**B. 印刷的距离档修正从来没生效过** —— 两处，症状**不一样**，别写混：
- 载具（`:404-418`）：`rangeMod` 是下拉的**档位序号**（1..5）。`:408` 用 `Number(itemData.attributes.range.value - rangeMod) < 0` 当越界检查；`:412-415` 若低于最小射程则 `rangeMod = 2*(rangeMod - minrange)`，**否则一律置 0**；`:418` 才把它加进池子。也就是说 **+2/+1/0/−1/−2 这张印刷表在载具上永远是 0**。
- 飞船（`:499-504`）：只有 `if (Number.isNaN(rangeMod)) rangeMod = 0`，然后 `:504` 把**档位序号原样加进池子**：Extreme（5）凭空多 5 颗黑骰、Adjacent（1）多 1 颗——和印刷表正好反向，而且完全没有越界检查。
- 印刷表的权威出处就在系统自己的配置里：`module/helpers/config.mjs:416-422` 的 `ALIENRPG.vehicle_weapon_range_list` 五行带 `value: "2"/"1"/"0"/"-1"/"-2"`（即 `3 - band`），代码从来不读这个字段；Evolved 那张表 `:423-430` 干脆连 `value` 都没有。

**C. 新建的挂载武器开箱即死** —— `module/data/item-weapon.mjs:39-59`，`range.value`、`minrange.value`、`rounds.value` 三个 `NumberField` 的 `initial` 都是 `0`。于是：`rounds 0` → 载具表单 `module/sheets/vehicle-sheet.mjs:660-666` 直接发一条红字 `ALIENRPG.noAmmo`（"You Need To Reload !!"）并**不调用** `roll()`；把弹药填上后 `range 0` 又让 `:408` 恒成立 → 每个档位都报 Out of Range。**没有任何提示告诉 GM 缺的是哪个字段。**

**D. 载具挂载的弹药永不消耗** —— 载具/飞船分支调 `yze.yzeRoll(...)` 时只传到第 10 个参数 `itemid`（`:429-440`、`:515-526`），不传第 12 个 `moddata`；而扣弹逻辑整个关在 `module/helpers/YZEDiceRoller.mjs:571` 的 `if (moddata && moddata.weapontype === "1")` 里面。**而且这条路子补不回来**：`:572-573` 是 `game.actors.get(actorid).items.get(moddata.itemId)`，`actorid` 传的是**乘员**的 id，武器却挂在**载具**身上，硬传 `moddata` 只会拿到 `undefined` 然后崩。所以扣弹必须我们自己做。载具挂载的 `rounds` 是离散的导弹/炮弹枚数（表单以 `rounds <= 0` 判定「不能开火」），因此**每次远程射击扣 1 发**；`spacecraftweapons` 这个物品类型的 schema（`module/data/item-spacecraftweapons.mjs`，全文 86 行，只有 `range`，**没有 `rounds`、没有 `minrange`**）对它不扣、不提示。

**E. `roll()` 全文没有出现过 `header.active`** —— `grep -n 'header.active' module/documents/item.mjs` 零命中。未启用的武器照常开火，没有任何警告。（`armor` 类型不用管：`:67` 一进 `roll()` 就 `return`。）

**F. 载具的机动性（Manoeuvrability）没有任何代码读** —— `templates/actor/vehicle-crew.hbs:26-28` 驾驶员那一行是
```hbs
<h3 for="actor.system.skills.piloting.value" class="resource-label rollable Attr1 gSC8" data-action='RollAbility'
    data-actorid="{{ actor.id }}" data-roll="{{actor.system.skills.piloting.mod}}" data-label="…">
```
**没有 `data-mod`**。而 `module/documents/actor.mjs:187` 的 `abilityRoll(actor, dataset, rollMod)` 在 `:198` 算的是
```js
const modifier = Number(dataset?.mod ?? 0) + Number(dataset?.modifier ?? 0);
```
`:210` 再 `r1Data = Number(dataset.roll || 0) + Number(modifier || 0)`。也就是说：**只要那一行有 `data-mod`，机动性就会自动进池子**。载具头部的 `system.attributes.manoeuvrability.value`（`module/data/actor-vehicle.mjs:87-93`，`NumberField` initial 0）今天没有任何消费者。每个乘员行外面包着 `<div class="occupant …" data-crew-id="…">`（`vehicle-crew.hbs:15`），表单类名是 `alienrpgVehicleSheet`（`module/sheets/vehicle-sheet.mjs:14`），所以渲染钩子名是 `renderalienrpgVehicleSheet`。

**G. 乘员槽存的是裸 actor id** —— `module/data/actor-vehicle.mjs:131-136`：`crew.occupants` 是 `[{id: StringField, position: StringField}]`，`position` 取值见 `module/helpers/config.mjs:314`：`COMMANDER / PILOT / GUNNER / PASSENGER`（飞船另有 `CAPTAIN / SENSOR-OP / ENGINEER`）。系统用 `game.actors.get(id)` 解析（`:358`）；契约 §7 要求本模组一律走 `resolver.actorById(id, {warn})`。乘员为空时系统在 `:371` 发 `ALIENRPG.noCrewAssigned`。

**H. 谁会调 `roll()`** —— 载具表单 `module/sheets/vehicle-sheet.mjs:641-684`：左键对 `weapon` / `spacecraftweapons` 调 `item.roll(false, dataset)`，右键只对 `weapon` 调 `item.roll(true, dataset)`；飞船表单同形；角色表单 `module/sheets/character-sheet.mjs:754-781` 同形。另外热键栏宏 `module/alienrpg.mjs:563` 会调**无参**的 `item.roll()`——所以我们的代码必须容忍 `args` 是空数组、`right` 与 `dataset` 双双 `undefined`。

**I. 系统的载具开火对话框** —— `templates/dialog/roll-vehicle-weapon.hbs` 共四栏：`FirerSelect`（选项由调用方传的 `options` 对象给出，键是序号字符串）、`rangeMod`（`<select>`，选项来自 `config.vehicle_weapon_range_list` 或 `config.evolved_vehicle_weapon_range_list`，值是 1..5）、`modifier`（Base Mod）、`stressMod`。系统在 `:373-378` 用 `foundry.applications.handlebars.renderTemplate` 渲染它。Evolved 与否取自 `game.settings.get("alienrpg", "evolved")`（`vehicle-sheet.mjs:650`）。

---

## 明确不做（写下来是为了不被当成漏掉）

1. **不注入 `data-action="ItemActivate"`**。清单标题后半句「ship/vehicle gear must be activatable at all」已被**证伪**：三张表单都已注册该动作——`module/sheets/vehicle-sheet.mjs:34`、`module/sheets/spacecraft-sheet.mjs:37`、`module/sheets/character-sheet.mjs:42`（`ItemActivate: { handler: this._onItemActivate, buttons: [0, 2] }`），左键置 `"true"`、右键置 `"false"`。再注入一次就是双重绑定。
2. **不碰「物品 Modifiers 页签被无视」**（`module/data/actor-character.mjs:352`）。它会改动全服每个角色的骰池，清单风险栏要求默认关闭并带迁移标记，属三期 `talent-registry-and-effects`。
3. **不修载具库存行的行内编辑**（`module/sheets/vehicle-sheet.mjs:606-612` 整段监听器被注释掉了，改弹药数不保存）。本任务改由开火自动扣弹，物品卡上的 Rounds 字段本来就能改；行内编辑留给上游 PR。
4. **不接管飞船的「防御」掷骰**（`itemData.header.type.value === "2"`，`module/documents/item.mjs:529-611`）。它用的是另一个对话框模板 `templates/dialog/roll-spacecraft-defense.hbs`，**里面根本没有 `rangeMod` 字段**，于是 `Number(undefined)` → NaN → `:582` 的 `if (Number.isNaN(rangeMod)) rangeMod = 0` 把它归零——这条路径今天是**对的**，不需要修。`pureMountTakeover` 会显式把它放行给系统。
5. **弹药不做「补给骰」模型**。角色手持武器走的是 `YZEDiceRoller.mjs:571-611` 的补给骰（掷 `rounds` 颗黄骰、按出现的 1 的个数扣），载具挂载走不到那段（见事实 D），且离散枚数模型才配得上表单的 `rounds <= 0` 判定。这是一条**会改变现有战局数字**的决定，写进提交信息与上游 PR 描述。

---

- [ ] **Step 1: 写第一个失败测试 —— 距离档算术（纯函数）**

新建 `test/repair-vehicle-roll-path.test.mjs`。这是脱桩真单测：被测函数不许引用任何 Foundry 全局（契约 §0.1 的分层铁律）。

```js
import { describe, expect, it } from "vitest";
import {
  pureRangeReasonKey,
  pureVehicleRangeMod,
} from "../scripts/repairs/vehicle-roll-path-repair.pure.mjs";

describe("pureRangeReasonKey", () => {
  it("pins the two i18n keys the range check can hand back", () => {
    // 这是全仓库唯一一处写出这两个键字面量的地方：渲染层、测试与自检条目
    // 都从这个访问器取，改键名只改一处。
    expect(pureRangeReasonKey("outOfRange")).toBe("AEA.repair.outOfRange");
    expect(pureRangeReasonKey("badBand")).toBe("AEA.repair.badBand");
    expect(pureRangeReasonKey("nonsense")).toBeNull();
  });
});

describe("pureVehicleRangeMod", () => {
  const wide = { maxRange: 5, minRange: 1 };

  it("returns the printed band modifier +2/+1/0/-1/-2", () => {
    // 出处：systems/alienrpg/module/helpers/config.mjs:416-422 的
    // ALIENRPG.vehicle_weapon_range_list，五行的 value 正是 "2","1","0","-1","-2"。
    expect(pureVehicleRangeMod({ band: 1, ...wide }).mod).toBe(2);
    expect(pureVehicleRangeMod({ band: 2, ...wide }).mod).toBe(1);
    expect(pureVehicleRangeMod({ band: 3, ...wide }).mod).toBe(0);
    expect(pureVehicleRangeMod({ band: 4, ...wide }).mod).toBe(-1);
    expect(pureVehicleRangeMod({ band: 5, ...wide }).mod).toBe(-2);
  });

  it("refuses to fire beyond the mount's maximum range", () => {
    const r = pureVehicleRangeMod({ band: 4, maxRange: 3, minRange: 1 });
    expect(r.ok).toBe(false);
    expect(r.mod).toBe(0);
    expect(r.reasonKey).toBe(pureRangeReasonKey("outOfRange"));
  });

  it("subtracts 2 dice per band below the minimum, on top of the band modifier", () => {
    // band 1 (+2) with minimum 3 => two bands under => 2 + 2*(1-3) = -2
    expect(pureVehicleRangeMod({ band: 1, maxRange: 5, minRange: 3 }).mod).toBe(-2);
    // band 2 (+1) with minimum 3 => one band under => 1 + 2*(2-3) = -1
    expect(pureVehicleRangeMod({ band: 2, maxRange: 5, minRange: 3 }).mod).toBe(-1);
    // at or above the minimum there is no penalty
    expect(pureVehicleRangeMod({ band: 3, maxRange: 5, minRange: 3 }).mod).toBe(0);
  });

  it("treats the schema initial 0 as 'the GM never filled this in' and names the field", () => {
    // item-weapon.mjs:39-52 的 range/minrange 出厂都是 0，0 不是合法档位。
    const r = pureVehicleRangeMod({ band: 5, maxRange: 0, minRange: 0 });
    expect(r.ok).toBe(true);
    expect(r.mod).toBe(-2);
    expect(r.unconfigured).toEqual(["range", "minrange"]);
  });

  it("stays silent about a field the item type does not even have", () => {
    // spacecraftweapons 的 schema 里没有 minrange，调用方传 null 表示「字段不存在」。
    const r = pureVehicleRangeMod({ band: 4, maxRange: 5, minRange: null });
    expect(r.ok).toBe(true);
    expect(r.mod).toBe(-1);
    expect(r.unconfigured).toEqual([]);
  });

  it("rejects a band outside 1..5 or one that is not a number at all", () => {
    expect(pureVehicleRangeMod({ band: 0, ...wide }).ok).toBe(false);
    expect(pureVehicleRangeMod({ band: 9, ...wide }).reasonKey).toBe(pureRangeReasonKey("badBand"));
    expect(pureVehicleRangeMod({ band: undefined, ...wide }).ok).toBe(false);
    expect(pureVehicleRangeMod({ band: "", ...wide }).ok).toBe(false);
  });

  it("accepts the string a <select> hands back, because DOM values are strings", () => {
    expect(pureVehicleRangeMod({ band: "4", maxRange: "5", minRange: "1" }).mod).toBe(-1);
  });
});
```

- [ ] **Step 2: 跑一遍，看它失败**

Run: `npx vitest run test/repair-vehicle-roll-path.test.mjs`

Expected: FAIL，8 个用例全部报同一条 —— `Error: Failed to load url ../scripts/repairs/vehicle-roll-path-repair.pure.mjs (resolved id: …/scripts/repairs/vehicle-roll-path-repair.pure.mjs). Does the file exist?`（文件还没建）。

- [ ] **Step 3: 写出 `pureRangeReasonKey` 与 `pureVehicleRangeMod`**

新建 `scripts/repairs/vehicle-roll-path-repair.pure.mjs`，先只放这两个：

```js
/**
 * Pure half of the vehicle / spacecraft gunnery repair.
 * Contract §0.1: nothing in this file may touch a Foundry global.
 */

/** i18n keys this module can hand back. Single source of truth for the literals. */
const REASON_KEYS = {
  outOfRange: "AEA.repair.outOfRange",
  badBand: "AEA.repair.badBand",
};

/**
 * @param {"outOfRange"|"badBand"} kind
 * @returns {string|null}
 */
export function pureRangeReasonKey(kind) {
  return REASON_KEYS[kind] ?? null;
}

/** A legal range band is an integer 1..5 (Contact/Adjacent … Extreme). */
function legalBand(n) {
  return Number.isInteger(n) && n >= 1 && n <= 5;
}

/**
 * Alien RPG vehicle / spacecraft gunnery range arithmetic.
 *
 * Bands are 1..5 = Contact(Adjacent) / Short / Medium / Long / Extreme and the
 * printed table gives +2 / +1 / 0 / -1 / -2, i.e. (3 - band). The system ships
 * that very table in module/helpers/config.mjs:416-422 with a `value` field and
 * then never reads it: the vehicle branch (item.mjs:412-415) zeroes the modifier
 * unless the shot is under the minimum range, and the spacecraft branch
 * (item.mjs:501-504) adds the raw band INDEX into the pool instead.
 *
 * A shot below the mount's minimum range costs a further -2 dice per band, on
 * top of the printed modifier. A shot beyond the maximum range cannot be taken.
 *
 * `range.value` / `minrange.value` both ship with a schema initial of 0, which
 * is not a legal band: treat that as "the GM never filled this in" — stay
 * permissive so a fresh mount can still fire, and report the field name so the
 * caller can say which box is empty. `null`/`undefined` means the item type has
 * no such field at all (spacecraftweapons has neither minrange nor rounds):
 * permissive AND silent, because there is no box for the GM to fill.
 *
 * @param {object} input
 * @param {number|string} input.band       selected band, 1..5 (a <select> gives a string)
 * @param {number|string|null} input.maxRange item.system.attributes.range.value
 * @param {number|string|null} input.minRange item.system.attributes.minrange.value
 * @returns {{ok: boolean, mod: number, reasonKey: string|null, unconfigured: string[]}}
 */
export function pureVehicleRangeMod({ band, maxRange, minRange }) {
  const unconfigured = [];
  const b = Number(band);
  if (band === "" || band === null || band === undefined || !legalBand(b)) {
    return { ok: false, mod: 0, reasonKey: pureRangeReasonKey("badBand"), unconfigured };
  }

  let effMax = 5;
  if (maxRange !== null && maxRange !== undefined) {
    const max = Number(maxRange);
    if (legalBand(max)) effMax = max;
    else unconfigured.push("range");
  }

  let effMin = 1;
  if (minRange !== null && minRange !== undefined) {
    const min = Number(minRange);
    if (legalBand(min)) effMin = min;
    else unconfigured.push("minrange");
  }

  if (b > effMax) {
    return { ok: false, mod: 0, reasonKey: pureRangeReasonKey("outOfRange"), unconfigured };
  }

  let mod = 3 - b;
  if (b < effMin) mod += 2 * (b - effMin);
  return { ok: true, mod, reasonKey: null, unconfigured };
}
```

- [ ] **Step 4: 跑一遍，看它通过**

Run: `npx vitest run test/repair-vehicle-roll-path.test.mjs`
Expected: `8 passed`。

- [ ] **Step 5: 提交**

```bash
git add scripts/repairs/vehicle-roll-path-repair.pure.mjs test/repair-vehicle-roll-path.test.mjs && git commit -m "$(cat <<'EOF'
fix(repairs): 载具炮击改用印刷的距离档修正表

系统自己在 config.mjs:416-422 里带着这张表（value 为 2/1/0/-1/-2），
却从不读它：载具分支 item.mjs:412-415 把修正一律置 0，
飞船分支 :501-504 干脆把下拉的档位序号原样加进骰池，
Extreme 反而凭空多 5 颗黑骰。

新增纯函数 pureVehicleRangeMod：按 3-band 给印刷修正，
低于最小射程每档再 -2，超出最大射程直接拒绝开火。
schema 初始值 0 不是合法档位，按「GM 没填」放行并回报字段名；
字段整个不存在（spacecraftweapons 没有 minrange）则静默放行。
两个 i18n 键收在 pureRangeReasonKey 一处。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: 写第二个失败测试 —— 接管范围**

追加到 `test/repair-vehicle-roll-path.test.mjs`（把 `pureMountTakeover` 加进文件顶部那条 import）：

```js
describe("pureMountTakeover", () => {
  it("takes over vehicle mounts, ranged and close alike", () => {
    expect(pureMountTakeover({ actorType: "vehicles", itemType: "weapon", weaponType: "1" })).toBe(true);
    expect(pureMountTakeover({ actorType: "vehicles", itemType: "weapon", weaponType: "2" })).toBe(true);
  });

  it("takes over ranged ship armament of either item type", () => {
    expect(pureMountTakeover({ actorType: "spacecraft", itemType: "weapon", weaponType: "1" })).toBe(true);
    expect(pureMountTakeover({ actorType: "spacecraft", itemType: "spacecraftweapons", weaponType: "1" })).toBe(true);
  });

  it("leaves the ship DEFENCE roll to the system, which is already correct", () => {
    // roll-spacecraft-defense.hbs has no rangeMod field, so item.mjs:582's
    // NaN guard already zeroes it. Nothing to repair, so do not intercept.
    expect(pureMountTakeover({ actorType: "spacecraft", itemType: "weapon", weaponType: "2" })).toBe(false);
  });

  it("never touches gear carried by a person or a creature", () => {
    expect(pureMountTakeover({ actorType: "character", itemType: "weapon", weaponType: "1" })).toBe(false);
    expect(pureMountTakeover({ actorType: "synthetic", itemType: "weapon", weaponType: "1" })).toBe(false);
    expect(pureMountTakeover({ actorType: "creature", itemType: "weapon", weaponType: "1" })).toBe(false);
    expect(pureMountTakeover({ actorType: "vehicles", itemType: "talent", weaponType: "1" })).toBe(false);
  });
});
```

- [ ] **Step 7: 跑一遍，看它失败**

Run: `npx vitest run test/repair-vehicle-roll-path.test.mjs -t "pureMountTakeover"`
Expected: FAIL —— `SyntaxError: The requested module '/scripts/repairs/vehicle-roll-path-repair.pure.mjs' does not provide an export named 'pureMountTakeover'`。

- [ ] **Step 8: 写出 `pureMountTakeover`**

追加到 `scripts/repairs/vehicle-roll-path-repair.pure.mjs`：

```js
/** Actor types whose weapons are fixed mounts fired by a crew member. */
const MOUNT_ACTORS = new Set(["vehicles", "spacecraft"]);
/** Item types that can be a mount. */
const MOUNT_ITEMS = new Set(["weapon", "spacecraftweapons"]);

/**
 * Does this roll belong to the broken mount path we take over completely?
 *
 * `weaponType` is item.system.header.type.value: "1" = ranged, "2" = close /
 * (on a spacecraft) defence. The spacecraft defence dialog carries no range
 * select, so item.mjs:582 already zeroes the modifier and that path is correct
 * as shipped — we hand it straight back to the system.
 *
 * @returns {boolean} true = our stage handles it, false = call the system
 */
export function pureMountTakeover({ actorType, itemType, weaponType }) {
  if (!MOUNT_ACTORS.has(actorType)) return false;
  if (!MOUNT_ITEMS.has(itemType)) return false;
  if (actorType === "spacecraft") return weaponType === "1";
  return weaponType === "1" || weaponType === "2";
}
```

- [ ] **Step 9: 跑一遍，看它通过**

Run: `npx vitest run test/repair-vehicle-roll-path.test.mjs`
Expected: `12 passed`。

- [ ] **Step 10: 写第三个失败测试 —— 两个可注入的缺陷谓词**

契约 §4 K7 要求每条补丁带一个可运行的 `probe()`：返回 `true` 表示缺陷仍在（该装），`false` 表示上游已修（补丁自动退休并提示 GM 一次）。probe 必须是**可注入纯谓词的薄壳、且两个方向都有测试**——只测「4.1.13 → true」等于把补丁写死成永远安装，上游修好后会双重修复。

追加到 `test/repair-vehicle-roll-path.test.mjs`（同样把两个新名字加进顶部 import）：

```js
// 逐字取自 4.1.13 的 module/documents/item.mjs（:82 的右键段与 :404-418 的载具段），
// 只删掉与判定无关的中间行。
const SOURCE_4_1_13 = `
    if (right) {
      // ************************************
      // Right Click Roll so display modboxes
      // ************************************
      if (this.actor.type === "character" || actorData.header.synthstress) {
        modifier = Number(response.modifier);
      } else {
        if (this.actor.type !== "vehicles" && this.actor.type !== "spacecraft") {
          const r1Data = actorData.skills.rangedCbt.mod + itemData.attributes.bonus.value + modifier;
        }
      }
    } else {
      // ************************************
      // Normal Left Click Roll
      // ************************************
      let rangeMod = Number(response.rangeMod);
      const r1Data = Number(game.actors.get(tactorid).system.skills.rangedCbt.mod + itemData.attributes.bonus.value + modifier + rangeMod);
    }
`;

const SOURCE_FIXED = `
    if (right) {
      // ************************************
      // Right Click Roll so display modboxes
      // ************************************
      if (this.actor.type === "character" || actorData.header.synthstress) {
        modifier = Number(response.modifier);
      } else if (this.actor.type === "vehicles" || this.actor.type === "spacecraft") {
        return this._mountRoll(dataset);
      }
    } else {
      // ************************************
      // Normal Left Click Roll
      // ************************************
      const bandId = Number(response.rangeMod);
      const bandMod = Number(config.vehicle_weapon_range_list[bandId].value);
      const r1Data = Number(skill + bonus + modifier + bandMod);
    }
`;

describe("pureVehicleRollIsBuggy", () => {
  it("says yes to the 4.1.13 body we are repairing", () => {
    expect(pureVehicleRollIsBuggy(SOURCE_4_1_13)).toBe(true);
  });

  it("says no once BOTH symptoms are gone", () => {
    expect(pureVehicleRollIsBuggy(SOURCE_FIXED)).toBe(false);
  });

  it("still says yes when only the arithmetic was fixed and right-click is still dead", () => {
    const half = SOURCE_FIXED.replace(
      'else if (this.actor.type === "vehicles" || this.actor.type === "spacecraft") {',
      'else if (this.actor.type !== "vehicles" && this.actor.type !== "spacecraft") {',
    );
    expect(pureVehicleRollIsBuggy(half)).toBe(true);
  });

  it("still says yes when only right-click was fixed and the raw band is still added", () => {
    const half = SOURCE_FIXED.replace("+ modifier + bandMod", "+ modifier + rangeMod");
    expect(pureVehicleRollIsBuggy(half)).toBe(true);
  });
});

describe("pureManoeuvreControlMissing", () => {
  // vehicle-crew.hbs:26 as it renders today — no data-mod anywhere on the row.
  const PILOT_ROW = `<h3 class="resource-label rollable Attr1 gSC8" data-action='RollAbility' data-actorid="abc" data-roll="7" data-label="Ripley - Piloting">Piloting</h3>`;

  it("reports the shipped pilot row as missing the modifier hook", () => {
    expect(pureManoeuvreControlMissing(PILOT_ROW)).toBe(true);
  });

  it("reports a row that already carries data-mod as fine", () => {
    expect(pureManoeuvreControlMissing(PILOT_ROW.replace('data-roll="7"', 'data-roll="7" data-mod="2"'))).toBe(false);
  });

  it("passes no judgement on markup that is not a roll control", () => {
    expect(pureManoeuvreControlMissing('<div class="gSC7">Ripley</div>')).toBe(false);
    expect(pureManoeuvreControlMissing(undefined)).toBe(false);
  });
});
```

- [ ] **Step 11: 跑一遍，看它失败**

Run: `npx vitest run test/repair-vehicle-roll-path.test.mjs -t "IsBuggy"`
Expected: FAIL —— `SyntaxError: The requested module '/scripts/repairs/vehicle-roll-path-repair.pure.mjs' does not provide an export named 'pureVehicleRollIsBuggy'`。

- [ ] **Step 12: 写出两个缺陷谓词**

追加到 `scripts/repairs/vehicle-roll-path-repair.pure.mjs`：

```js
/**
 * Is the shipped `alienrpgItem#roll` still carrying the two defects we repair?
 * Injectable so both directions are unit-tested; the effect layer only passes
 * `Function.prototype.toString()` of the snapshot it took before anything
 * wrapped the method.
 *
 * Symptom 1 — arithmetic: the raw band index is added straight into the pool
 *   (item.mjs:418 / :504 both read `+ rangeMod`).
 * Symptom 2 — dead right-click: the right-click half of the function (the text
 *   before the "Normal Left Click Roll" banner) excludes vehicles and never
 *   handles them, so the dialog closes and nothing happens (item.mjs:169).
 *   If a future release strips comments the banner is gone; we then fall back
 *   to symptom 1 alone rather than guessing.
 *
 * @param {string} source
 * @returns {boolean} true = defect still present, install the repair
 */
export function pureVehicleRollIsBuggy(source) {
  const text = String(source ?? "");
  const cut = text.indexOf("Normal Left Click Roll");
  const rightHalf = cut === -1 ? "" : text.slice(0, cut);
  const rawBandAdded = text.includes("+ rangeMod");
  const rightClickDead =
    rightHalf.includes('!== "vehicles"') && !rightHalf.includes('=== "vehicles"');
  return rawBandAdded || rightClickDead;
}

/**
 * Does this rendered pilot row still lack the `data-mod` hook that
 * `abilityRoll` reads at actor.mjs:198? Anything that is not a roll control at
 * all gets `false`: we have nothing to say about it.
 *
 * @param {string|undefined} html  one row's outerHTML
 * @returns {boolean}
 */
export function pureManoeuvreControlMissing(html) {
  const s = String(html ?? "");
  const isRollControl =
    s.includes('data-action="RollAbility"') || s.includes("data-action='RollAbility'");
  if (!isRollControl) return false;
  return !s.includes("data-mod");
}
```

- [ ] **Step 13: 跑一遍，看它通过，然后提交**

Run: `npx vitest run test/repair-vehicle-roll-path.test.mjs`
Expected: `19 passed`。

```bash
git add scripts/repairs/vehicle-roll-path-repair.pure.mjs test/repair-vehicle-roll-path.test.mjs && git commit -m "$(cat <<'EOF'
fix(repairs): 补上接管范围判定与两个可注入的缺陷谓词

pureMountTakeover 把飞船「防御」掷骰显式放行给系统：
roll-spacecraft-defense.hbs 里根本没有 rangeMod 字段，
item.mjs:582 的 NaN 归零让那条路径本来就是对的，不该介入。

两个 probe 谓词都做成吃源码字符串的纯函数，各带正反两组夹具：
4.1.13 原文 -> true；两处症状都修好 -> false；只修一半 -> 仍 true。
避免把补丁写死成永远安装，上游修好后能自动退休。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 14: 写装备闸门的失败测试**

新建 `test/repair-gear-active-gate.test.mjs`：

```js
import { describe, expect, it } from "vitest";
import {
  pureGearGate,
  pureGearGateIsBuggy,
  pureGearInactiveKey,
} from "../scripts/repairs/gear-active-gate.pure.mjs";

describe("pureGearInactiveKey", () => {
  it("pins the one i18n key this gate can hand back", () => {
    expect(pureGearInactiveKey()).toBe("AEA.repair.gearInactive");
  });
});

describe("pureGearGate", () => {
  it("blocks a carried weapon whose active flag is the string 'false'", () => {
    const r = pureGearGate({ actorType: "character", itemType: "weapon", active: "false" });
    expect(r.allowed).toBe(false);
    expect(r.reasonKey).toBe(pureGearInactiveKey());
  });

  it("blocks 'fLocker' too, because both stored strings are truthy in JS", () => {
    // item-weapon.mjs:88 ships active as a StringField initial "false";
    // character-sheet.mjs:638 writes "fLocker" for footlocker storage.
    expect(pureGearGate({ actorType: "character", itemType: "weapon", active: "fLocker" }).allowed).toBe(false);
    expect(pureGearGate({ actorType: "synthetic", itemType: "item", active: "fLocker" }).allowed).toBe(false);
  });

  it("allows exactly the string 'true'", () => {
    expect(pureGearGate({ actorType: "character", itemType: "weapon", active: "true" }).allowed).toBe(true);
    expect(pureGearGate({ actorType: "synthetic", itemType: "item", active: "true" }).allowed).toBe(true);
  });

  it("allows a real boolean true, for item types that model the flag that way", () => {
    // item-spacecraftweapons.mjs:13 uses a BooleanField for header.active.
    expect(pureGearGate({ actorType: "character", itemType: "weapon", active: true }).allowed).toBe(true);
    expect(pureGearGate({ actorType: "character", itemType: "weapon", active: false }).allowed).toBe(false);
  });

  it("never gates a fixed mount on a vehicle or a ship", () => {
    expect(pureGearGate({ actorType: "vehicles", itemType: "weapon", active: "false" }).allowed).toBe(true);
    expect(pureGearGate({ actorType: "spacecraft", itemType: "spacecraftweapons", active: false }).allowed).toBe(true);
  });

  it("never gates a creature's natural attacks", () => {
    expect(pureGearGate({ actorType: "creature", itemType: "weapon", active: "false" }).allowed).toBe(true);
  });

  it("never gates an item type the activation rule does not cover", () => {
    expect(pureGearGate({ actorType: "character", itemType: "talent", active: "false" }).allowed).toBe(true);
    expect(pureGearGate({ actorType: "character", itemType: "critical-injury", active: "false" }).allowed).toBe(true);
    // armor never reaches a roll anyway: item.mjs:67 returns immediately.
    expect(pureGearGate({ actorType: "character", itemType: "armor", active: "false" }).allowed).toBe(true);
  });

  it("is permissive when the flag is absent, never blocking on missing data", () => {
    expect(pureGearGate({ actorType: "character", itemType: "weapon", active: undefined }).allowed).toBe(true);
    expect(pureGearGate({ actorType: "character", itemType: "weapon", active: null }).allowed).toBe(true);
  });
});

describe("pureGearGateIsBuggy", () => {
  it("says yes to a roll() that never mentions header.active", () => {
    const shipped = `async roll(right, dataset) {
      let hostile = this.actor.type;
      if (this.type === "armor") { return; }
    }`;
    expect(pureGearGateIsBuggy(shipped)).toBe(true);
  });

  it("says no once upstream checks the flag itself", () => {
    const fixed = `async roll(right, dataset) {
      if (this.system.header.active !== "true") {
        return ui.notifications.warn(game.i18n.localize("ALIENRPG.NotActive"));
      }
    }`;
    expect(pureGearGateIsBuggy(fixed)).toBe(false);
  });
});
```

- [ ] **Step 15: 跑一遍，看它失败**

Run: `npx vitest run test/repair-gear-active-gate.test.mjs`
Expected: FAIL —— `Error: Failed to load url ../scripts/repairs/gear-active-gate.pure.mjs`。

- [ ] **Step 16: 写出 `gear-active-gate.pure.mjs`**

新建 `scripts/repairs/gear-active-gate.pure.mjs`：

```js
/**
 * Pure half of the "inactive gear must not roll" repair.
 * Contract §0.1: no Foundry globals in this file.
 */

/** Actor types whose inventory is carried gear the activation rule covers. */
const GATED_ACTOR_TYPES = new Set(["character", "synthetic"]);

/**
 * Item types the rule covers AND that can reach a roll.
 * `armor` is deliberately absent: item.mjs:67 returns before doing anything,
 * so gating it would only produce a warning for a roll that never happens.
 * `item` is present because a hotbar macro calls item.roll() on any owned item
 * (alienrpg.mjs:563).
 */
const GATED_ITEM_TYPES = new Set(["weapon", "item"]);

/** The one i18n key this gate can hand back. */
export function pureGearInactiveKey() {
  return "AEA.repair.gearInactive";
}

/**
 * All items import Inactive and cannot be used in that state.
 *
 * `system.header.active` is a StringField whose shipped values are "true",
 * "false" and "fLocker" (item-weapon.mjs:88, character-sheet.mjs:638).
 * "false" and "fLocker" are BOTH truthy in JavaScript, so the comparison must
 * be against the literal string "true" — a plain truthiness test would let
 * every inactive item straight through, which is the entire defect.
 *
 * Vehicle and spacecraft mounts are NOT gated: they are fixed armament, not
 * carried gear, and their sheets already ship a working ItemActivate control
 * (vehicle-sheet.mjs:34, spacecraft-sheet.mjs:37).
 *
 * @param {object} input
 * @param {string|undefined} input.actorType   actor.type
 * @param {string|undefined} input.itemType    item.type
 * @param {string|boolean|undefined|null} input.active  item.system.header.active
 * @returns {{allowed: boolean, reasonKey: string|null}}
 */
export function pureGearGate({ actorType, itemType, active }) {
  const pass = { allowed: true, reasonKey: null };
  if (!GATED_ACTOR_TYPES.has(actorType)) return pass;
  if (!GATED_ITEM_TYPES.has(itemType)) return pass;
  if (active === undefined || active === null) return pass;
  if (active === "true" || active === true) return pass;
  return { allowed: false, reasonKey: pureGearInactiveKey() };
}

/**
 * Is the shipped `roll()` still ignoring the activation flag?
 * Injectable so both directions are unit-tested.
 *
 * @param {string} source
 * @returns {boolean} true = defect still present
 */
export function pureGearGateIsBuggy(source) {
  return !String(source ?? "").includes("header.active");
}
```

- [ ] **Step 17: 跑一遍，看它通过，然后提交**

Run: `npx vitest run test/repair-gear-active-gate.test.mjs`
Expected: `11 passed`。

```bash
git add scripts/repairs/gear-active-gate.pure.mjs test/repair-gear-active-gate.test.mjs && git commit -m "$(cat <<'EOF'
fix(repairs): 未启用的随身装备不得掷骰（纯判定层）

item.mjs:39 的 roll() 全文没有出现过 header.active，未启用武器照常开火。
新增 pureGearGate：weapon 的 system.header.active 是 StringField，
"false" 与 "fLocker" 在 JS 里都是 truthy，故必须与字面量 "true" 比较；
spacecraftweapons 那边是 BooleanField，因此真布尔也要容忍。

载具与飞船的固定挂载不在闸门范围内——三张表单的 ItemActivate 动作
（vehicle-sheet.mjs:34 / spacecraft-sheet.mjs:37 / character-sheet.mjs:42）
均已存在，清单中「ship/vehicle gear 无法启用」一说已证伪，
再注入 data-action 只会造成双重绑定。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 18: 写源码快照的失败测试（用共享桩）**

两条修复的 `probe()` 都靠读 `alienrpgItem#roll` 的**源码文本**判断缺陷还在不在。但 probe 在 ready 才跑、自检还会被 GM 反复调用，那时 `CONFIG.Item.documentClass.prototype.roll` 上可能已经是内核 rollBus 经 libWrapper 装的分发器——它的源码里没有系统的算术，读它会让每个 probe 都答「上游修好了」，两条修复静默退休。所以要在被包裹之前拍一张快照。

契约 §0.3：测试里的 Foundry 全局**只能**用共享桩 `test/stubs/foundry.mjs`，不得各自就地造 `globalThis.game` 或用私有 Map 顶替 `game.settings`。桩自己会装 `CONFIG` 这个全局；下面往里塞一条 `Item.documentClass` 属于**世界内容**，不是另造全局。

新建 `test/repair-item-roll-source.test.mjs`：

```js
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";

/** 一个形似 4.1.13 的 Item 类，源码里带着我们要认的那两处症状。 */
class ShippedItem {
  async roll(right, dataset) {
    let rangeMod = Number(dataset?.rangeMod);
    // Normal Left Click Roll
    if (this.actor.type !== "vehicles" && this.actor.type !== "spacecraft") return;
    return Number(this.bonus + rangeMod);
  }
}

describe("captureItemRollSource", () => {
  let mod;

  beforeEach(async () => {
    vi.resetModules(); // 模块级快照必须每个用例都是新的
    installFoundryStub();
    globalThis.CONFIG.Item = { documentClass: ShippedItem };
    mod = await import("../scripts/repairs/item-roll-source.mjs");
  });

  afterEach(() => uninstallFoundryStub());

  it("hands back the shipped method's source text", () => {
    expect(mod.captureItemRollSource()).toContain("Normal Left Click Roll");
    expect(mod.captureItemRollSource()).toContain("+ rangeMod");
  });

  it("keeps the first snapshot even after the property is replaced by a wrapper", () => {
    mod.captureItemRollSource();
    // 这是内核 rollBus 在 ready.rollbus 装上 libWrapper 之后的样子。
    globalThis.CONFIG.Item.documentClass.prototype.roll = function dispatcher() {
      return "libWrapper dispatcher";
    };
    expect(mod.captureItemRollSource()).toContain("Normal Left Click Roll");
    expect(mod.itemRollSource()).not.toContain("dispatcher");
  });
});

describe("captureItemRollSource with no system class in sight", () => {
  let mod;

  beforeEach(async () => {
    vi.resetModules();
    installFoundryStub();
    globalThis.CONFIG.Item = {}; // 系统还没把 documentClass 交出来
    mod = await import("../scripts/repairs/item-roll-source.mjs");
  });

  afterEach(() => uninstallFoundryStub());

  it("says 'I have nothing' instead of guessing", () => {
    // "" 是 probe 的「不知道」信号；probe 会把它当成「缺陷还在」，
    // 宁可多装一次修复，也不要静默退休。
    expect(mod.captureItemRollSource()).toBe("");
    expect(mod.itemRollSource()).toBe("");
  });
});
```

- [ ] **Step 19: 跑一遍，看它失败**

Run: `npx vitest run test/repair-item-roll-source.test.mjs`
Expected: FAIL —— `Error: Failed to load url ../scripts/repairs/item-roll-source.mjs`。

- [ ] **Step 20: 写出 `item-roll-source.mjs`**

新建 `scripts/repairs/item-roll-source.mjs`：

```js
/**
 * A one-shot snapshot of the system's shipped `alienrpgItem#roll` SOURCE TEXT.
 *
 * THIS MODULE REGISTERS NOTHING. Contract §4 K1 [v3.1] gives kernel rollBus the
 * exclusive libWrapper registration on the four roll entry points
 * ("yzeRoll" | "abilityRoll" | "itemRoll" | "pushRoll"); every interception in
 * this module goes through `rollBus.addStage()`. All this file does is remember
 * what `roll()` looked like BEFORE anything wrapped it.
 *
 * Why that matters: both repairs carry a `probe()` that decides "is the defect
 * still there?" by reading the shipped source. Probes run at ready
 * (patches.applyAll) and again every time the GM runs the self-test — by then
 * libWrapper's dispatcher owns the property and its source says nothing about
 * the system's arithmetic. Reading it live would make every probe answer
 * "upstream fixed it" and silently retire both repairs.
 *
 * The capture is safe by the contract's own ready order:
 *   ready.registry -> ready.patches -> ready.rollbus -> ready.cards
 * `register()` runs at init and `patches.applyAll()` at ready.patches, both
 * strictly before `rollBus.install()` at ready.rollbus, so either call site
 * sees the untouched method. Once a non-empty snapshot exists it is never
 * replaced, so a late call can never capture a wrapper.
 */

/** Metadata only — nothing here resolves or wraps this path. */
export const ITEM_ROLL_PATH = "CONFIG.Item.documentClass.prototype.roll";

let snapshot = "";

/**
 * Idempotent; retries only while it still has nothing.
 * @returns {string} the source text, or "" when the system class is not reachable
 */
export function captureItemRollSource() {
  if (snapshot) return snapshot;
  const fn = globalThis.CONFIG?.Item?.documentClass?.prototype?.roll;
  if (typeof fn === "function") snapshot = Function.prototype.toString.call(fn);
  return snapshot;
}

/** @returns {string} the captured source; "" means it was never captured */
export function itemRollSource() {
  return snapshot;
}
```

- [ ] **Step 21: 跑一遍，看它通过，然后提交**

Run: `npx vitest run test/repair-item-roll-source.test.mjs`
Expected: `3 passed`。

```bash
git add scripts/repairs/item-roll-source.mjs test/repair-item-roll-source.test.mjs && git commit -m "$(cat <<'EOF'
fix(repairs): 在被包裹之前给 alienrpgItem#roll 拍一张源码快照

两条修复的 probe 都靠读 roll() 的源码判断缺陷还在不在。
ready.rollbus 之后这个属性上是 rollBus 经 libWrapper 装的分发器，
读它会让每个 probe 都答「上游已修」，两条修复就静默退休了。

快照在 init 的 register() 里拍，patches.applyAll()（ready.patches）
还会补拍一次——按契约 §5 的 ready 子锚点顺序，两处都严格早于
ready.rollbus，拿到的一定是未被包裹的原方法。
拍到之后永不覆盖；拍不到就返回 ""，由 probe 当作「缺陷还在」。
本文件不注册任何包裹：那四个目标归内核 rollBus 独占。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 22: 写注册面与 stage 行为的失败测试（用共享桩）**

新建 `test/repair-item-roll-stages.test.mjs`。这一份把两条修复的**装配**（开关、补丁元数据、stage 排队、钩子幂等、自检）与 **stage 的实际行为**（放行 / 拦下 / 接管 / 开关即时生效）都钉住。

`rollBus` 在契约 §4 K1 里是一个普通对象字面量（`export const rollBus = {...}`），所以可以 `vi.spyOn(rollBus, "addStage")` 把 stage 截下来直接调用——不需要真的装 libWrapper，也不需要 rollBus 组链。

```js
import { afterEach, describe, expect, it, vi } from "vitest";
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";

/** 形似 4.1.13：源码里既有 "+ rangeMod"，右键段又排除了载具，且从不提 header.active。 */
class ShippedItem {
  async roll(right, dataset) {
    let rangeMod = Number(dataset?.rangeMod);
    if (this.actor.type !== "vehicles" && this.actor.type !== "spacecraft") return;
    // Normal Left Click Roll
    return Number(this.bonus + rangeMod);
  }
}

/** 上游把三处症状都修好之后的样子：查了 header.active、按表查修正、右键也走载具。 */
class FixedItem {
  async roll(right, dataset) {
    if (this.system.header.active !== "true") return;
    const bandId = Number(dataset?.rangeMod);
    const bandMod = Number(CONFIG.ALIENRPG.vehicle_weapon_range_list[bandId].value);
    // Normal Left Click Roll
    if (this.actor.type === "vehicles" || this.actor.type === "spacecraft") {
      return Number(this.bonus + bandMod);
    }
  }
}

let ctx;
let warn;
let features;
let patches;
let rollBus;
let selftest;
const staged = new Map();

/**
 * 走一遍 main.mjs 在 init 段做的事：先 register 每条修复，再 registerSettings，
 * 最后（ready.patches）applyAll。契约 §5 [v3.1] 明确 def 必须先在册、设置才注册得上。
 */
async function boot({ ItemClass = ShippedItem, applyPatches = true } = {}) {
  vi.resetModules();
  ctx = installFoundryStub();
  globalThis.CONFIG.Item = { documentClass: ItemClass };
  globalThis.CONFIG.ALIENRPG = { vehicle_weapon_range_list: { 4: { value: "-1" } } };
  warn = vi.spyOn(globalThis.ui.notifications, "warn").mockImplementation(() => {});

  ({ features } = await import("../scripts/kernel/features.mjs"));
  ({ patches } = await import("../scripts/kernel/patches.mjs"));
  ({ rollBus } = await import("../scripts/kernel/rollbus.mjs"));
  ({ selftest } = await import("../scripts/kernel/selftest.mjs"));

  staged.clear();
  vi.spyOn(rollBus, "addStage").mockImplementation((target, stage) => {
    staged.set(stage.id, { target, ...stage });
  });

  const { gearActiveGate } = await import("../scripts/repairs/gear-active-gate.mjs");
  const { vehicleRollPathRepair } = await import("../scripts/repairs/vehicle-roll-path-repair.mjs");

  gearActiveGate.register();
  vehicleRollPathRepair.register();
  features.registerSettings();
  if (applyPatches) await patches.applyAll();
}

afterEach(() => {
  vi.restoreAllMocks();
  uninstallFoundryStub();
});

describe("repair registration", () => {
  it("declares one GM-only switch per patch, all defaulting to full", async () => {
    await boot();
    const byId = Object.fromEntries(features.all().map((d) => [d.id, d]));
    for (const id of ["gear-active-gate", "vehicle-roll-path-repair", "vehicle-manoeuvrability-control"]) {
      expect(byId[id], id).toMatchObject({ default: "full", gmOnly: true });
    }
  });

  it("declares three patches with the right type and target metadata", async () => {
    await boot();
    const byId = Object.fromEntries(patches.status().map((p) => [p.id, p]));

    expect(byId["gear-active-gate"]).toMatchObject({
      type: "MIXED",
      target: "rollBus:itemRoll#gear-active-gate",
      fixedIn: null,
      applied: true,
    });
    expect(byId["vehicle-roll-path-repair"]).toMatchObject({
      type: "MIXED",
      target: "rollBus:itemRoll#vehicle-roll-path-repair",
      applied: true,
    });
    expect(byId["vehicle-manoeuvrability-control"]).toMatchObject({
      type: "HOOK",
      target: "Hooks:renderalienrpgVehicleSheet",
      applied: true,
    });
  });

  it("books both roll stages on rollBus's itemRoll target, gate outside gunnery", async () => {
    await boot();
    expect([...staged.keys()].sort()).toEqual(["gear-active-gate", "vehicle-roll-path-repair"]);
    expect(staged.get("gear-active-gate")).toMatchObject({ target: "itemRoll", order: 10 });
    expect(staged.get("vehicle-roll-path-repair")).toMatchObject({ target: "itemRoll", order: 20 });
    // order 0 是 rollBus 自己的采集帧，任何修复都不许占用它。
    expect(staged.get("gear-active-gate").order).toBeGreaterThan(0);
  });

  it("hooks the vehicle sheet exactly once even if applyAll runs twice", async () => {
    await boot({ applyPatches: false });
    const hookOn = vi.spyOn(globalThis.Hooks, "on");
    await patches.applyAll();
    await patches.applyAll();
    const mine = hookOn.mock.calls.filter(([name]) => name === "renderalienrpgVehicleSheet");
    expect(mine).toHaveLength(1);
  });

  it("reports both roll repairs installed and the sheet control as not yet observed", async () => {
    await boot();
    const byId = Object.fromEntries((await selftest.runAll()).map((r) => [r.id, r]));

    expect(byId["repair.gear-active-gate"]).toMatchObject({ ok: true });
    expect(byId["repair.gear-active-gate"].detail).toMatch(/^installed:/);
    expect(byId["repair.vehicle-roll-path-repair"].detail).toMatch(/^installed:/);
    expect(byId["repair.vehicle-manoeuvrability-control"]).toMatchObject({ ok: false });
    expect(byId["repair.vehicle-manoeuvrability-control"].detail).toMatch(/^not-yet-observed:/);
  });

  it("retires both roll repairs the moment upstream ships the fix", async () => {
    await boot({ ItemClass: FixedItem });
    expect(staged.size).toBe(0);

    const byId = Object.fromEntries((await selftest.runAll()).map((r) => [r.id, r]));
    expect(byId["repair.gear-active-gate"]).toMatchObject({ ok: true });
    expect(byId["repair.gear-active-gate"].detail).toMatch(/^upstream-fixed:/);
    expect(byId["repair.vehicle-roll-path-repair"].detail).toMatch(/^upstream-fixed:/);
  });
});

describe("stage behaviour", () => {
  const gate = () => staged.get("gear-active-gate").around;
  const gunnery = () => staged.get("vehicle-roll-path-repair").around;

  it("blocks an inactive carried weapon and never reaches the system roll", async () => {
    await boot();
    const item = {
      type: "weapon",
      name: "M41A",
      actor: { type: "character" },
      system: { header: { active: "false" } },
    };
    const next = vi.fn();

    const out = await gate()(next, [false, {}], item);

    expect(next).not.toHaveBeenCalled();
    expect(out).toBeUndefined();
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("gearInactive"));
  });

  it("lets an activated weapon through untouched, passing the original arguments on", async () => {
    await boot();
    const item = {
      type: "weapon",
      name: "M41A",
      actor: { type: "character" },
      system: { header: { active: "true" } },
    };
    const next = vi.fn(() => "system-result");
    const args = [true, { itemId: "abc" }];

    const out = await gate()(next, args, item);

    expect(next).toHaveBeenCalledTimes(1);
    expect(next).toHaveBeenCalledWith(args);
    expect(out).toBe("system-result");
    expect(warn).not.toHaveBeenCalled();
  });

  it("asks the switch at roll time, so turning it off needs no world reload", async () => {
    await boot();
    const enabled = vi.spyOn(features, "enabled").mockReturnValue(false);
    const item = {
      type: "weapon",
      name: "M41A",
      actor: { type: "character" },
      system: { header: { active: "false" } },
    };
    const next = vi.fn(() => "system-result");

    const out = await gate()(next, [false, {}], item);

    expect(enabled).toHaveBeenCalledWith("gear-active-gate");
    expect(next).toHaveBeenCalledTimes(1);
    expect(out).toBe("system-result");
    expect(warn).not.toHaveBeenCalled();
  });

  it("takes the mount roll over completely: the system path never runs", async () => {
    await boot();
    const vehicle = { type: "vehicles", name: "M577 APC", system: { crew: { occupants: [] } } };
    const mount = {
      type: "weapon",
      name: "30mm cannon",
      actor: vehicle,
      system: { header: { type: { value: "1" } }, attributes: {} },
    };
    const next = vi.fn();

    const out = await gunnery()(next, [false, {}], mount);

    expect(next).not.toHaveBeenCalled();
    expect(out).toBeUndefined();
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("noCrew"));
  });

  it("hands a person's own weapon straight back to the system", async () => {
    await boot();
    const item = {
      type: "weapon",
      name: "M41A",
      actor: { type: "character" },
      system: { header: { type: { value: "1" } } },
    };
    const next = vi.fn(() => "system-result");
    const args = [false, {}];

    const out = await gunnery()(next, args, item);

    expect(next).toHaveBeenCalledTimes(1);
    expect(next).toHaveBeenCalledWith(args);
    expect(out).toBe("system-result");
  });

  it("hands the ship DEFENCE roll back to the system, which is already correct", async () => {
    await boot();
    const ship = { type: "spacecraft", name: "Montero", system: { crew: { occupants: [] } } };
    const gun = {
      type: "weapon",
      name: "Point defence",
      actor: ship,
      system: { header: { type: { value: "2" } } },
    };
    const next = vi.fn(() => "system-result");

    const out = await gunnery()(next, [false, {}], gun);

    expect(next).toHaveBeenCalledTimes(1);
    expect(out).toBe("system-result");
  });
});
```

- [ ] **Step 23: 跑一遍，看它失败**

Run: `npx vitest run test/repair-item-roll-stages.test.mjs`
Expected: FAIL —— 12 个用例全部报 `Error: Failed to load url ../scripts/repairs/gear-active-gate.mjs`。

- [ ] **Step 24: 写出 `gear-active-gate.mjs`（副作用层）**

新建 `scripts/repairs/gear-active-gate.mjs`：

```js
import { features } from "../kernel/features.mjs";
import { patches } from "../kernel/patches.mjs";
import { rollBus } from "../kernel/rollbus.mjs";
import { selftest } from "../kernel/selftest.mjs";
import { pureGearGate, pureGearGateIsBuggy } from "./gear-active-gate.pure.mjs";
import { captureItemRollSource, itemRollSource } from "./item-roll-source.mjs";

const ID = "gear-active-gate";

/**
 * A rollBus stage on the "itemRoll" target, i.e. around
 * `CONFIG.Item.documentClass.prototype.roll(right, dataset)`.
 *
 * Contract §4 K1 [v3.1]: `around(next, args, thisArg)` — `args` is the argument
 * list, `thisArg` is the Item the roll was called on. Call `next(args)` exactly
 * once to let the roll continue, or short-circuit deliberately by not calling
 * it and returning your own value. Never call `next` twice.
 * rollBus owns the single libWrapper registration on this target; this repair
 * must never call libWrapper itself (it would be refused and become a silent
 * no-op).
 *
 * The switch is read HERE, at roll time, not at install time (contract §7), so
 * a GM who turns the repair off mid-session gets the system's own behaviour on
 * the very next click without reloading the world.
 */
function gearGateStage(next, args, item) {
  if (!features.enabled(ID)) return next(args);

  const gate = pureGearGate({
    actorType: item?.actor?.type,
    itemType: item?.type,
    active: item?.system?.header?.active,
  });
  if (gate.allowed) return next(args);

  ui.notifications.warn(game.i18n.format(gate.reasonKey, { name: item?.name ?? "" }));
  return undefined; // deliberate short-circuit: this roll stops here
}

export const gearActiveGate = {
  id: ID,

  /**
   * init stage. Declares the switch, the patch and the self-test — and installs
   * nothing. `main.mjs` drives this from its REPAIRS loop; there is no
   * `install()` because the stage is booked by `apply()` at ready.patches,
   * which the contract's ready order puts before `rollBus.install()`.
   */
  register() {
    features.register({ id: ID, default: "full", gmOnly: true, requires: [], hint: "" });

    // Snapshot the untouched source now, before anything can wrap it.
    captureItemRollSource();

    patches.register({
      id: ID,
      type: "MIXED",
      // Metadata only, for patches.status(): the registration lives in rollBus.
      target: `rollBus:itemRoll#${ID}`,
      minSystem: "4.1.13",
      fixedIn: null,
      probe: () => {
        const src = captureItemRollSource();
        // "" means we never got to read the class. Assume the defect is still
        // there rather than retiring on ignorance.
        return src === "" ? true : pureGearGateIsBuggy(src);
      },
      apply: () => rollBus.addStage("itemRoll", { id: ID, order: 10, around: gearGateStage }),
    });

    selftest.register({
      id: `repair.${ID}`,
      // Contract: label is an i18n KEY, localized by the runner — registration
      // happens at init, before i18nInit, when localize() would only echo it.
      label: `AEA.selftest.repair.${ID}`,
      run: () => {
        if (!features.enabled(ID)) {
          return { ok: true, detail: "disabled-by-gm: the switch is off, the system rolls unchanged" };
        }
        const src = itemRollSource();
        if (src === "") {
          return { ok: false, detail: "source-unavailable: alienrpgItem#roll was never captured" };
        }
        if (!pureGearGateIsBuggy(src)) {
          return { ok: true, detail: "upstream-fixed: roll() now reads header.active" };
        }
        const applied = patches.status().find((p) => p.id === ID)?.applied === true;
        return applied
          ? { ok: true, detail: "installed: gate staged on rollBus itemRoll (order 10)" }
          : { ok: false, detail: "defect-present-not-installed: the patch did not apply" };
      },
    });
  },
};
```

- [ ] **Step 25: 跑一遍，看它换一条失败信息**

Run: `npx vitest run test/repair-item-roll-stages.test.mjs`
Expected: 仍然 FAIL，但报的换成了 `Error: Failed to load url ../scripts/repairs/vehicle-roll-path-repair.mjs`——说明闸门这一半已经能加载了。

- [ ] **Step 26: 写出 `vehicle-roll-path-repair.mjs` 的开火部分**

新建 `scripts/repairs/vehicle-roll-path-repair.mjs`，先写到 `mountRoll` 为止：

```js
import { SYSTEM_ID } from "../const.mjs";
import { features } from "../kernel/features.mjs";
import { patches } from "../kernel/patches.mjs";
import { resolver } from "../kernel/resolver.mjs";
import { rollBus } from "../kernel/rollbus.mjs";
import { selftest } from "../kernel/selftest.mjs";
import { captureItemRollSource, itemRollSource } from "./item-roll-source.mjs";
import {
  pureManoeuvreControlMissing,
  pureMountTakeover,
  pureVehicleRangeMod,
  pureVehicleRollIsBuggy,
} from "./vehicle-roll-path-repair.pure.mjs";

const ID = "vehicle-roll-path-repair";
const MANOEUVRE_ID = "vehicle-manoeuvrability-control";

/**
 * Crew members who can fire. Vehicle and spacecraft crew slots store a BARE
 * actor id (actor-vehicle.mjs:131-136: crew.occupants[].id), never a uuid, so
 * contract §4 K4's legacy-path helper is the only correct way to resolve them —
 * §7 forbids game.actors.get() in module code.
 *
 * @returns {{id: string, name: string, position: string, actor: object}[]}
 */
function firingCrew(actor) {
  const out = [];
  for (const occupant of actor?.system?.crew?.occupants ?? []) {
    if (occupant.position === "PASSENGER") continue;
    const crewActor = resolver.actorById(occupant.id, { warn: true });
    if (!crewActor) continue; // deleted crew member: the system would throw here
    out.push({ id: crewActor.id, name: crewActor.name, position: occupant.position, actor: crewActor });
  }
  return out;
}

/**
 * The whole mount-firing path, replacing item.mjs:348-527 for the cases
 * pureMountTakeover() claims.
 *
 * Left-click and right-click now behave identically. That is not a shortcut:
 * the system's own left-click dialog (roll-vehicle-weapon.hbs) already carries
 * the Base Mod and Stress Mod boxes, and right-click was dead only because
 * item.mjs:169 excluded vehicles from the right-click branch.
 */
async function mountRoll(item, dataset) {
  const actor = item.actor;
  const crew = firingCrew(actor);
  if (!crew.length) {
    ui.notifications.warn(game.i18n.localize("AEA.repair.noCrew"));
    return undefined;
  }

  const attrs = item.system?.attributes ?? {};
  const weaponType = item.system?.header?.type?.value;
  // spacecraftweapons has no `rounds` field at all (item-spacecraftweapons.mjs).
  const rounds = attrs.rounds?.value;
  const tracksAmmo = weaponType === "1" && rounds !== undefined;

  if (tracksAmmo && Number(rounds) <= 0) {
    ui.notifications.warn(game.i18n.format("AEA.repair.noRounds", { name: item.name }));
    return undefined;
  }

  const options = {};
  crew.forEach((c, i) => {
    options[`${i}`] = `${c.name} (${c.position})`;
  });

  // Reuse the system's own dialog template so the UI and its translations stay
  // identical (item.mjs:373-378 renders the same file). We compute isEvolved
  // ourselves because a hotbar macro calls roll() with no dataset at all
  // (alienrpg.mjs:563).
  const isEvolved = game.settings.get(SYSTEM_ID, "evolved") === true;
  const content = await foundry.applications.handlebars.renderTemplate(
    "systems/alienrpg/templates/dialog/roll-vehicle-weapon.hbs",
    {
      config: CONFIG.ALIENRPG,
      actorData: actor.system,
      dataset: { ...(dataset ?? {}), isEvolved },
      options,
    },
  );

  const title = `${game.i18n.localize("ALIENRPG.DialTitle1")} ${item.name} ${game.i18n.localize("ALIENRPG.DialTitle2")}`;
  const response = await foundry.applications.api.DialogV2.wait({
    window: { title },
    content,
    rejectClose: false,
    buttons: [
      {
        label: "ALIENRPG.DialRoll",
        callback: (_event, button) => new foundry.applications.ux.FormDataExtended(button.form).object,
      },
      { label: "ALIENRPG.DialCancel", action: "cancel" },
    ],
  });
  if (!response || response === "cancel") return "cancelled";

  const band = pureVehicleRangeMod({
    band: response.rangeMod,
    maxRange: attrs.range?.value ?? null,
    minRange: attrs.minrange?.value ?? null,
  });
  if (band.unconfigured.length) {
    ui.notifications.warn(
      game.i18n.format("AEA.repair.unconfiguredWeapon", {
        name: item.name,
        fields: band.unconfigured.join(", "),
      }),
    );
  }
  if (!band.ok) {
    ui.notifications.warn(game.i18n.localize(band.reasonKey));
    return undefined;
  }

  const shooter = crew[Number(response.FirerSelect)] ?? crew[0];
  const modifier = Number(response.modifier) || 0;
  const stressMod = Number(response.stressMod) || 0;
  const skill = Number(shooter.actor.system?.skills?.rangedCbt?.mod ?? 0);
  const bonus = Number(attrs.bonus?.value ?? 0);
  const stress = Number(shooter.actor.system?.header?.stress?.value ?? 0);
  const synthetic = shooter.actor.type === "synthetic";

  // Same label the system builds at item.mjs:64 plus the :420 suffix, so the
  // chat card reads exactly as before.
  const damage = attrs.damage?.value ?? 0;
  const label = `${item.name} (${game.i18n.localize("ALIENRPG.Damage")} : ${damage}) (${actor.name}) `;

  // item.mjs:59-60 clears these two on entry; our stage short-circuits before
  // that body ever runs, so we must do it ourselves.
  game.alienrpg.rollArr.sCount = 0;
  game.alienrpg.rollArr.multiPush = 0;

  // Argument 9 is `actorid`, and it MUST be the crew member, not the vehicle:
  // YZEDiceRoller builds the card's speaker from it, and contract §3 ruling 4
  // says RollRecord.actorUuid records the shooter. The system does the same at
  // item.mjs:399 / :495 (actorid = fCrew[shooter].firerID). The vehicle stays
  // reachable through the record's itemUuid -> .parent.
  await game.alienrpg.yze.yzeRoll(
    synthetic ? "synthetic" : "character",
    actor.prototypeToken?.disposition === -1, // blind roll for a hostile vehicle, as item.mjs:78
    synthetic, // reRoll: synthetics push differently
    label,
    skill + bonus + modifier + band.mod,
    game.i18n.localize("ALIENRPG.Black"),
    stress + stressMod,
    game.i18n.localize("ALIENRPG.Yellow"),
    shooter.id,
    item.id,
  );
  game.alienrpg.rollArr.sCount = game.alienrpg.rollArr.r1Six + game.alienrpg.rollArr.r2Six; // item.mjs:442

  // Ammunition. The system can never do this here: it passes no `moddata`
  // (item.mjs:429-440), and YZEDiceRoller.mjs:572 would look the weapon up on
  // the GUNNER's actor while the mount lives on the vehicle. A vehicle mount's
  // `rounds` is a discrete shell/missile count (the sheet refuses to fire at
  // <= 0), so one shot spends one round.
  if (tracksAmmo) {
    await item.update({ "system.attributes.rounds.value": Math.max(0, Number(rounds) - 1) });
  }
  return undefined;
}
```

- [ ] **Step 27: 写出同一文件的 stage、机动性注入与注册面**

追加到 `scripts/repairs/vehicle-roll-path-repair.mjs`：

```js
/**
 * rollBus stage on "itemRoll". See the contract note in gear-active-gate.mjs:
 * `args` is the argument list ([right, dataset] here — the hotbar macro at
 * alienrpg.mjs:563 calls roll() with none, so both can be undefined) and
 * `thisArg` is the Item. Taking the mount path over is a deliberate
 * short-circuit: that whole branch is what we are replacing.
 */
async function mountStage(next, args, item) {
  if (!features.enabled(ID)) return next(args);

  const takeover = pureMountTakeover({
    actorType: item?.actor?.type,
    itemType: item?.type,
    weaponType: item?.system?.header?.type?.value,
  });
  if (!takeover) return next(args);

  return mountRoll(item, args?.[1]);
}

/** "unknown" | "missing" | "present" — what the last rendered pilot row looked like. */
let manoeuvreObservation = "unknown";
let manoeuvreInjections = 0;
/** Hook handle, kept so the patch can be retired with Hooks.off(). */
let manoeuvreHookId = null;

/**
 * Add a "PILOTING + Manoeuvrability" control next to each pilot row.
 *
 * vehicle-crew.hbs:26 renders the pilot row without `data-mod`, and
 * abilityRoll reads exactly that attribute at actor.mjs:198
 * (`Number(dataset?.mod ?? 0) + Number(dataset?.modifier ?? 0)`), so a clone of
 * the row carrying data-mod is all it takes. ApplicationV2 render hooks hand us
 * (app, element, context, options) with a real HTMLElement — not jQuery.
 *
 * Idempotent: a re-render rebuilds the DOM, our own row is excluded from the
 * selector, and we skip a row whose sibling already carries the control, so
 * nothing ever stacks up.
 */
function injectManoeuvreControl(app, element) {
  if (!features.enabled(MANOEUVRE_ID)) return; // contract §0.2 exception, condition 3

  const rows = element?.querySelectorAll?.(
    ".occupant h3[data-action='RollAbility']:not(.aea-manoeuvre)",
  );
  if (!rows?.length) return;

  const doc = app?.document ?? app?.actor;
  const manoeuvrability = Number(doc?.system?.attributes?.manoeuvrability?.value ?? 0);
  for (const row of rows) {
    const missing = pureManoeuvreControlMissing(row.outerHTML);
    manoeuvreObservation = missing ? "missing" : "present";
    if (!missing) continue; // upstream shipped the hook: leave the sheet alone
    if (!manoeuvrability) continue; // nothing to add
    if (row.parentElement?.querySelector(".aea-manoeuvre")) continue;

    const extra = document.createElement("h3");
    extra.className = "resource-label rollable aea-manoeuvre";
    extra.dataset.action = "RollAbility";
    extra.dataset.actorid = row.dataset.actorid ?? "";
    extra.dataset.roll = row.dataset.roll ?? "0";
    extra.dataset.mod = String(manoeuvrability);
    extra.dataset.label = `${row.dataset.label ?? ""} +${manoeuvrability}`;
    extra.textContent = game.i18n.localize("AEA.repair.pilotManoeuvre");
    row.after(extra);
    manoeuvreInjections += 1;
  }
}

/** Idempotent: contract §0.2's HOOK exception requires apply() to hook at most once. */
function installManoeuvreHook() {
  if (manoeuvreHookId !== null) return manoeuvreHookId;
  manoeuvreHookId = Hooks.on("renderalienrpgVehicleSheet", injectManoeuvreControl);
  return manoeuvreHookId;
}

export const vehicleRollPathRepair = {
  id: ID,

  /**
   * init stage. Two patches, two switches, two self-tests — and nothing is
   * installed here. No `install()` is exported: `apply()` books the stage at
   * ready.patches, which the contract's ready order puts before
   * rollBus.install(), and main.mjs's REPAIRS loop calls `r.install?.()`.
   */
  register() {
    features.register({ id: ID, default: "full", gmOnly: true, requires: [], hint: "" });
    features.register({ id: MANOEUVRE_ID, default: "full", gmOnly: true, requires: [], hint: "" });

    captureItemRollSource(); // snapshot before anything wraps roll()

    patches.register({
      id: ID,
      type: "MIXED",
      target: `rollBus:itemRoll#${ID}`, // metadata only; rollBus owns the registration
      minSystem: "4.1.13",
      fixedIn: null,
      probe: () => {
        const src = captureItemRollSource();
        return src === "" ? true : pureVehicleRollIsBuggy(src);
      },
      apply: () => rollBus.addStage("itemRoll", { id: ID, order: 20, around: mountStage }),
    });

    // Contract §0.2 [v3.1]: a registered patch of type "HOOK" may hook a domain
    // hook inside its own apply(). renderalienrpgVehicleSheet is not one of the
    // three hook groups the contract reserves for kernel modules, the hook name
    // is in `target` for status(), the handle is kept for retirement, and the
    // callback checks the switch on its first line.
    patches.register({
      id: MANOEUVRE_ID,
      type: "HOOK",
      target: "Hooks:renderalienrpgVehicleSheet",
      minSystem: "4.1.13",
      fixedIn: null,
      probe: () => manoeuvreObservation !== "present",
      apply: () => installManoeuvreHook(),
    });

    selftest.register({
      id: `repair.${ID}`,
      label: `AEA.selftest.repair.${ID}`,
      run: () => {
        if (!features.enabled(ID)) {
          return { ok: true, detail: "disabled-by-gm: the switch is off, the system rolls unchanged" };
        }
        const src = itemRollSource();
        if (src === "") {
          return { ok: false, detail: "source-unavailable: alienrpgItem#roll was never captured" };
        }
        if (!pureVehicleRollIsBuggy(src)) {
          return { ok: true, detail: "upstream-fixed: band table applied and right-click routed" };
        }
        const applied = patches.status().find((p) => p.id === ID)?.applied === true;
        return applied
          ? { ok: true, detail: "installed: mount rolls staged on rollBus itemRoll (order 20)" }
          : { ok: false, detail: "defect-present-not-installed: the patch did not apply" };
      },
    });

    selftest.register({
      id: `repair.${MANOEUVRE_ID}`,
      label: `AEA.selftest.repair.${MANOEUVRE_ID}`,
      run: () => {
        if (!features.enabled(MANOEUVRE_ID)) {
          return { ok: true, detail: "disabled-by-gm: the pilot row is left exactly as the system renders it" };
        }
        if (manoeuvreObservation === "present") {
          return { ok: true, detail: "upstream-fixed: the shipped pilot row already has data-mod" };
        }
        if (manoeuvreObservation === "missing") {
          return { ok: true, detail: `installed: control injected ${manoeuvreInjections} time(s)` };
        }
        return { ok: false, detail: "not-yet-observed: open a vehicle sheet with a PILOT, then re-run" };
      },
    });
  },
};
```

- [ ] **Step 28: 跑一遍，看它通过**

Run: `npx vitest run test/repair-item-roll-stages.test.mjs`
Expected: `12 passed`（6 条 registration + 6 条 stage behaviour）。

- [ ] **Step 29: 加 i18n 键（两个文件一起改）**

契约 §7：面向用户的字符串一律 `game.i18n.localize()`/`format()`；模组自有键前缀 `AEA.`、**嵌套结构**。§4 K5：特性显示名与说明固定为 `AEA.feature.<逐字的 feature id>.name` / `.hint`。自检条目的 `label` 是 i18n 键，也要在这里给出文案。

把下面这段**合并进** `lang/en.json` 已有的 `AEA` 对象（不要替换整个对象；如果某个子对象还不存在就新建）：

```json
{
  "AEA": {
    "feature": {
      "vehicle-roll-path-repair": {
        "name": "Vehicle and ship gunnery",
        "hint": "Apply the printed range-band modifier (+2/+1/0/-1/-2), revive the dead right-click roll on a mount, and spend one round of the mount's ammunition per ranged shot."
      },
      "vehicle-manoeuvrability-control": {
        "name": "Pilot rolls include Manoeuvrability",
        "hint": "Add a second PILOTING control to each pilot row on a vehicle sheet that also adds the vehicle's Manoeuvrability to the dice pool."
      },
      "gear-active-gate": {
        "name": "Inactive gear cannot roll",
        "hint": "Refuse to roll a weapon or item that has not been activated on the sheet, and say so instead of rolling silently."
      }
    },
    "repair": {
      "outOfRange": "Out of range: this mount cannot reach that band.",
      "badBand": "No range band was selected.",
      "noCrew": "No crew member is assigned to a firing position on this vehicle.",
      "noRounds": "{name} is out of ammunition.",
      "unconfiguredWeapon": "{name} has no {fields} set on its item sheet; the shot was allowed anyway.",
      "gearInactive": "{name} is not activated. Left-click its image on the sheet to activate it first.",
      "pilotManoeuvre": "PILOTING + Manoeuvrability"
    },
    "selftest": {
      "repair": {
        "vehicle-roll-path-repair": "Vehicle and ship gunnery path",
        "vehicle-manoeuvrability-control": "Pilot row carries Manoeuvrability",
        "gear-active-gate": "Inactive gear must not roll"
      }
    }
  }
}
```

同样合并进 `lang/cn.json`：

```json
{
  "AEA": {
    "feature": {
      "vehicle-roll-path-repair": {
        "name": "载具与飞船炮击",
        "hint": "按印刷的距离档修正表（+2/+1/0/-1/-2）算骰池，救活挂载武器上那条右键死路径，每次远程射击扣一发弹药。"
      },
      "vehicle-manoeuvrability-control": {
        "name": "驾驶掷骰计入机动性",
        "hint": "在载具表单的驾驶员那一行再加一个驾驶控件，把载具的机动性一并加进骰池。"
      },
      "gear-active-gate": {
        "name": "未启用的装备不得掷骰",
        "hint": "尚未在角色卡上启用的武器与装备拒绝掷骰，并明确提示，而不是照常掷出去。"
      }
    },
    "repair": {
      "outOfRange": "超出射程：这件挂载打不到这个距离档。",
      "badBand": "没有选择距离档。",
      "noCrew": "这辆载具没有乘员被指派到射击位。",
      "noRounds": "{name} 已经没有弹药了。",
      "unconfiguredWeapon": "{name} 的物品卡上没有填 {fields}，本次射击仍然放行。",
      "gearInactive": "{name} 尚未启用。请先在角色卡上左键点击它的图标。",
      "pilotManoeuvre": "驾驶 + 机动性"
    },
    "selftest": {
      "repair": {
        "vehicle-roll-path-repair": "载具与飞船炮击路径",
        "vehicle-manoeuvrability-control": "驾驶员那一行带上机动性",
        "gear-active-gate": "未启用的装备不得掷骰"
      }
    }
  }
}
```

- [ ] **Step 30: 在 `main.mjs` 的两个锚点接线**

`scripts/main.mjs` 由更早的任务建立，里面有十个锚点注释。本任务只碰其中两个，**不往任何生命周期钩子里加代码，也一个字不改 `api` 对象**——`register()` 与 `install?.()` 由 `REPAIRS` 循环统一驱动。

先确认锚点在位（按锚点文本定位，不要用行号）：

```bash
grep -n "AEA-ANCHOR: imports" scripts/main.mjs
grep -n "AEA-ANCHOR: repairs" scripts/main.mjs
```

改两处：

1. 在 `/* AEA-ANCHOR: imports */` **之后**加两行：

```js
import { gearActiveGate } from "./repairs/gear-active-gate.mjs";
import { vehicleRollPathRepair } from "./repairs/vehicle-roll-path-repair.mjs";
```

2. 在 `/* AEA-ANCHOR: repairs */` **之后**（即 `const REPAIRS = [ … ]` 数组内部）加两行：

```js
  gearActiveGate,
  vehicleRollPathRepair,
```

两个模块的 `register()` 内部已经调过 `features.register`、`patches.register`、`selftest.register`，`main.mjs` 不需要再调别的。两者都没有 `install()`，循环里的 `r.install?.()` 会安静跳过。数组内的先后顺序不影响行为：stage 链按 `order` 排（闸门 10 在外、炮击 20 在内），与数组顺序无关。

- [ ] **Step 31: 跑全套并提交**

Run: `npm test`
Expected: 全绿；本任务新增的四个测试文件合计 **45 passed**（`repair-vehicle-roll-path` 19 + `repair-gear-active-gate` 11 + `repair-item-roll-source` 3 + `repair-item-roll-stages` 12）。

```bash
git add -A && git commit -m "$(cat <<'EOF'
fix(repairs): 载具开火与装备闸门改挂 rollBus stage，并接线进 REPAIRS

契约 §4 K1 [v3.1]：yzeRoll/abilityRoll/itemRoll/pushRoll 这四个目标上
全模组只允许 rollBus 一处 libWrapper 注册（lib-wrapper 1.13.5 会拒绝
第二次注册，修复会静默变成空操作），任何介入一律走 rollBus.addStage。
两条修复因此各出一个 itemRoll stage：
- gear-active-gate（order 10）：character/synthetic 身上未启用的
  weapon/item 拒绝掷骰并给出可操作提示；载具与飞船的固定挂载不受限。
- vehicle-roll-path-repair（order 20）：载具的 weapon 与飞船的远程挂载
  完全接管——复用系统自己的 roll-vehicle-weapon.hbs 对话框，
  按印刷波段表算修正，左右键走同一条路（右键此前是死的），
  乘员经 resolver.actorById 解析（乘员槽存的是裸 actor id），
  yzeRoll 第 9 参传乘员 id（契约 §3 裁决 4：actorUuid 记掷骰者），
  远程射击每次扣一发弹药（系统在这条路上永远扣不了：不传 moddata，
  且 YZEDiceRoller.mjs:572 会去乘员身上找挂在载具上的武器）。
  飞船「防御」掷骰显式放行给系统，那条路径本来就是对的。

另加一条 HOOK 型补丁：renderalienrpgVehicleSheet 上给驾驶员那一行
补一个带 data-mod 的「驾驶 + 机动性」控件（abilityRoll 在 actor.mjs:198
读的正是 data-mod）。按契约 §0.2 [v3.1] 的例外条款办：只在 apply() 内
挂一次、句柄留着以便退休摘除、回调第一行查开关、钩子名写进 target。

三条补丁各有一个 gmOnly 开关（默认 full，掷骰那一刻才查，关掉不用重载
世界）、一条可注入纯谓词的 probe、一条同源断言的自检条目。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 32: MANUAL VERIFICATION A —— 补丁装上了没有**

桩无法忠实模拟真实的 libWrapper 链、DialogV2 与 ApplicationV2 渲染时序，所以下面五步在真 Foundry 里做（契约 §0.4）。

1. 打开一个装着 `alienrpg` 4.1.13 的世界，启用 `lib-wrapper` 与 `alien-evolved-automation`。游戏设置里确认 `Alien RPG` 的 **Evolved** 开着（默认开）。
2. F12 控制台执行：
   ```js
   game.modules.get("alien-evolved-automation").api.patches.status()
   ```
   **期望**：数组里有三行，`id` 分别是 `gear-active-gate`、`vehicle-roll-path-repair`、`vehicle-manoeuvrability-control`，每行的形状是 `{id, type, target, applied, reason, fixedIn}`；前两条 `type: "MIXED"`、`target` 分别是 `"rollBus:itemRoll#gear-active-gate"` 与 `"rollBus:itemRoll#vehicle-roll-path-repair"`，第三条 `type: "HOOK"`、`target: "Hooks:renderalienrpgVehicleSheet"`；三条 `applied` 都是 `true`。
3. 再执行：
   ```js
   await game.modules.get("alien-evolved-automation").api.selftest.runAll()
   ```
   **期望**：`repair.gear-active-gate` 与 `repair.vehicle-roll-path-repair` 两条 `ok: true`、`detail` 以 `installed:` 开头；`repair.vehicle-manoeuvrability-control` 这时还是 `ok:false` / `not-yet-observed:`（还没开过载具表单），Step 35 之后再看它。
4. 控制台再确认全模组在这个目标上只有 rollBus 一处注册：
   ```js
   libWrapper._modules?.get?.("alien-evolved-automation")
   ```
   或退而求其次，在「模组管理 → libWrapper → Active Wrappers」面板里查 `alienrpgItem.prototype.roll`。**期望**：`alien-evolved-automation` 在这个方法上只出现**一次**（rollBus 的 WRAPPER）。若出现两次，说明有人绕过了 `rollBus.addStage` 自己去 register 了，回头改回来。
5. **如果** `detail` 是 `source-unavailable:`：说明 `CONFIG.Item.documentClass` 在 init 与 ready.patches 两个时刻都读不到。把这条原话回报给计划主控——这不是本任务的判定逻辑错，而是加载顺序变了。

- [ ] **Step 33: MANUAL VERIFICATION B —— 载具炮击的四个症状**

1. 新建一个 `vehicles` actor。在 Crew 页签把一个 `character` 指派为 **GUNNER**，另一个指派为 **PILOT**。
2. 给载具加一件 `weapon`：Type 选 Ranged（`header.type.value === "1"`），Bonus 填 `2`，Rounds 填 `3`，**Range 与 Min Range 都留空（0）**。
3. 在载具库存里**左键**点这件武器 →
   **期望**：弹出系统原本那个对话框（Select Firer / Range Modifier / Base Mod / Stress Mod 四栏），Select Firer 里两个人都带位置后缀（如 `Ripley (GUNNER)`）。选 Extreme，Base Mod 填 0，点 Roll。
   **期望**：右下角一条黄色提示「… 的物品卡上没有填 range, minrange，本次射击仍然放行」；聊天卡上的黑骰数 = 炮手的 rangedCbt.mod **+ 2（武器 Bonus）− 2（Extreme 的印刷修正）**。
   **修复前**：黑骰数会多出 5 颗（飞船路径）或印刷修正整个不生效（载具路径），且没有任何字段提示。
4. 看载具库存行的 Rounds：**期望**从 3 变成 2。**修复前**：永远不动。
5. **右键**点同一件武器 → **期望**：弹出**同一个**对话框，填 Base Mod `+1` 点 Roll，出卡，黑骰数比上一步多 1。**修复前**：弹出 "Base Modifier" 框、点 Roll、框关掉、什么都不发生、控制台也没有报错——这一步是死路径被修好的直接证据。
6. 把 Range 填成 `3`，Min Range 填成 `1`，再左键、选 Long（4）→ **期望**：黄色提示「超出射程：这件挂载打不到这个距离档」，**没有**聊天卡。
7. 把 Rounds 改成 `0`，左键 → **期望**：系统表单自己会先拦一条红字 "You Need To Reload !!"（`vehicle-sheet.mjs:660-666`，Evolved 模式下它在我们之前跑）。把世界的 Evolved 关掉再试同一步 → **期望**：这次由我们拦，黄色提示「… 已经没有弹药了」。
8. 把 Crew 里的两个人都移除，左键武器 → **期望**：黄色提示「这辆载具没有乘员被指派到射击位」。

- [ ] **Step 34: MANUAL VERIFICATION C —— 掷骰记录归到乘员名下**

契约 §3 裁决 4 要求 `RollRecord.actorUuid` 记的是**掷骰者（乘员）**，不是武器的主人（载具）。这条一期不影响玩法，但记错了会在二期 `attack-context-binding` 上返工，所以现在就验一次。

1. 照 Step 33 的第 3 步再开一炮，出一张聊天卡。
2. 控制台执行：
   ```js
   const m = game.messages.contents.at(-1);
   game.modules.get("alien-evolved-automation").api.rollBus.recordOf(m)
   ```
3. **期望**：
   - `actorUuid` 指向**炮手那个 character**（不是载具）——把它丢进 `fromUuidSync(...)` 应当拿到炮手，`.name` 是炮手的名字；
   - `itemUuid` 指向**载具身上的那件挂载**，`fromUuidSync(record.itemUuid).parent.name` 是**载具**的名字（载具由此可达，不需要额外字段）；
   - `kind` 是 `"weapon"`，`pools.base` 与你在 Step 33 数出来的黑骰数一致。
4. 如果 `actorUuid` 指向的是载具，说明 `yzeRoll` 的第 9 个参数传错了（必须是 `shooter.id`，不是 `actor.id`）；回 Step 26 那一段核对。

- [ ] **Step 35: MANUAL VERIFICATION D —— 机动性控件**

1. 在载具头部把 **Manoeuvrability** 填 `2`，关掉再重新打开载具表单，切到 Crew 页签。
2. **期望**：驾驶员那一行 PILOTING 的**下面**多出一条「驾驶 + 机动性」。
3. 点原来的 PILOTING，记下黑骰数；点新的「驾驶 + 机动性」→ **期望**黑骰数正好多 2。
4. 在表单里随便改一个字段触发重渲染（例如改一下车名），再看 Crew 页签 → **期望**「驾驶 + 机动性」**只有一条**，不会每次渲染叠加。
5. 控制台再跑一次 `await game.modules.get("alien-evolved-automation").api.selftest.runAll()` → **期望** `repair.vehicle-manoeuvrability-control` 变成 `ok:true`、`detail` 以 `installed: control injected` 开头。
6. 把 Manoeuvrability 改回 `0`，重开表单 → **期望**不再出现那一条（没有修正可加就不加）。
7. 打开`游戏设置 → 模组设置 → Alien Evolved Automation`，把「驾驶掷骰计入机动性」改成 **Off**，重开载具表单 → **期望**那一条不再出现，且原本的 PILOTING 行完全没被动过。

- [ ] **Step 36: MANUAL VERIFICATION E —— 装备闸门与即时开关**

1. 打开一个 `character`，看库存里一件 `weapon`：出厂 `active` 是 `"false"`，图标是暗的。
2. 左键点**武器名**（不是图标）→ **期望**：黄色提示「… 尚未启用。请先在角色卡上左键点击它的图标」，**没有**聊天卡。**修复前**：照常掷骰，没有任何警告。
3. 左键点**图标**把它启用（变亮），再点武器名 → **期望**：正常弹对话框并掷骰。
4. 右键点图标（置为 `"false"`），把武器拖到热键栏做成宏再点那个宏 → **期望**：同一条黄色提示（宏走的是 `item.roll()` 无参路径，也被闸门覆盖）。
5. 打开`游戏设置 → 模组设置 → Alien Evolved Automation`，把「未启用的装备不得掷骰」改成 **Off**，**不要重载世界**，再点一次未启用的武器名 → **期望**：立刻恢复成系统原行为（照常掷骰）。这一步验证开关是在掷骰那一刻查的，不是安装那一刻。
6. 打开一个 `vehicles`，确认它那件 `active` 为 `"false"` 的挂载**照常可以开火**（固定挂载不受闸门管辖）。

---

**UPSTREAM PR 草案（`vehicle-roll-path-repair`，干净自足，值得单独发给 pwatson100/alienrpg）**

- 文件一 `module/helpers/config.mjs:423-430` —— 给 `ALIENRPG.evolved_vehicle_weapon_range_list` 的五行补上 `value: "2" / "1" / "0" / "-1" / "-2"`，与既有的 `vehicle_weapon_range_list`（`:416-422`）对齐。
- 文件二 `module/documents/item.mjs:404-418` 与 `:499-504` —— 两处把 `rangeMod` 从「下拉序号」改成「查表得到的 value」：先 `const bandId = Number(response.rangeMod)`，再 `const bandMod = Number((evolved ? config.evolved_vehicle_weapon_range_list : config.vehicle_weapon_range_list)[bandId].value)`；最小射程罚则改成在 `bandMod` 之上叠加 `2 * (bandId - minrange)`（而不是像现在这样把 `bandMod` 覆盖掉）；越界检查改成 `bandId > range.value`；最后用 `bandMod` 参与 `r1Data`。飞船那处还要补上今天完全没有的越界检查。
- 文件三 `module/documents/item.mjs:169` —— 去掉右键分支里 `if (this.actor.type !== "vehicles" && this.actor.type !== "spacecraft")` 这一层，让右键落到与左键相同的载具处理上。
- 文件四 `module/data/item-weapon.mjs:39-52` —— `range` 与 `minrange` 的 `initial` 从 `0` 改成 `1`（Adjacent），使新建挂载开箱即可用。
- 文件五 `module/sheets/vehicle-sheet.mjs:606-612` —— 取消注释那段行内编辑监听器（或改用 ApplicationV2 的 `change` 委托），让库存行里的弹药数能存下来。
- 附带一处小崩溃：`module/sheets/vehicle-sheet.mjs:660` 在 Evolved 分支对 `spacecraftweapons` 也读 `item.system.attributes.rounds.value`，而该类型的 schema 里没有 `rounds`（`module/data/item-spacecraftweapons.mjs` 全文 86 行），会抛 TypeError。加一层可选链即可。
- PR 描述里必须写明：**这会改变现有战局的骰池**（此前印刷波段修正从未生效，飞船更是反向多骰），建议随发行说明公告。
