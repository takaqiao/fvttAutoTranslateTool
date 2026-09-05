> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 6 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 6: K1 纯函数层 —— RollRecord 组装器（`kernel/record.mjs`）

**Files:**
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/kernel/record.mjs`
- Test: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/test/record.test.mjs`
- 本任务**一行都不改** `scripts/main.mjs`，理由见 Interfaces 末条。

**Interfaces:**

- **Consumes**：`scripts/const.mjs`（更早的任务已建立）的 `export const RECORD_VERSION = 1;`。**import 它，绝不硬写 `1`**。除此之外本文件不 import 任何东西。
  本任务**不使用** `test/stubs/foundry.mjs`：契约 §0.4 第一条把 `record.mjs` 判为「真单测，无桩或仅用最小对象字面量」。**不得** `import { installFoundryStub } from "./stubs/foundry.mjs"`，也**不得**在测试里自己造 `globalThis.game`。本文件根本不碰 Foundry，装了桩只会掩盖分层违规。
- **Produces**（前五个签名逐字取自契约 §4 K1，不得改动）：
  - `export function classifyKind(args, ctx) -> "attribute"|"skill"|"weapon"|"armor"|"supply"|"other"`
  - `export function successesOf(rollArr) -> number`
  - `export function banesOf(rollArr) -> number`
  - `export function pureReverseLabelKey(label, index) -> string|null`
  - `export function buildLabelIndex(localize, keys) -> {[localizedText]: string|null}` —— `localize` 是**注入进来的函数**（副作用层传 `game.i18n.localize` 的绑定版），`keys` 是 i18n 键数组；同一段译文被两个不同的键命中时，该译文映射到 `null`。
  - `export function buildRollRecord(input) -> RollRecord`，其中
    `input = {args, rollArr, refs, ctx, userId, worldTime, now, id, labelIndex}`
  - `export const LABEL_KEYS` —— **21 个 i18n 键的冻结数组**，是 `buildLabelIndex` 的第二个实参。
- **`LABEL_KEYS` 的属主判定（契约 v3.1 §4 K1）**：这张键表**全仓只允许存在这一份，属主是本文件**。`kernel/rollbus.mjs` 与 `scripts/main.mjs` 一律 `import { LABEL_KEYS, buildLabelIndex } from "./record.mjs"`（从 main.mjs 看是 `"./kernel/record.mjs"`），**不得**各自定义同名符号。上一轮 `record.mjs` 与 `rollbus.mjs` 各定义了一份、键数还不一致（26 vs 25），必然有一份成死代码，或者 main.mjs 从错误的模块 import 出一张与兜底分支不同源的表。二期要加键就加到本文件这一份里。
- **本任务与 `main.mjs` 的关系 —— 按契约 v3.1「每一行的插入属主」规则判定**：该规则是「清单里每一行的属主是**该行第一个被调用对象的建立者**」。契约 §5 的 i18nInit 那一行逐字是

  ```js
  rollBus.setLabelIndex(buildLabelIndex(game.i18n.localize.bind(game.i18n), LABEL_KEYS));
  ```

  第一个被调用对象是 `rollBus`，所以这一行**归 `kernel/rollbus.mjs` 的属主插入**（他动手时本文件的两个符号都已存在）。反过来，若本任务现在就往 `main.mjs` 插它，`main.mjs` 就会 import 一个还不存在的 `kernel/rollbus.mjs`、整个模组加载失败。**本任务的交付物就是让那一行引用的两个符号先就位**，并把它要用的 import 逐字写在这里供属主抄：

  ```js
  import { buildLabelIndex, LABEL_KEYS } from "./kernel/record.mjs";
  ```

  本任务的十个 `AEA-ANCHOR` 锚点插入数为零。`record.mjs` 也**不是** `api` 的八个槽之一（八个槽是 features / patches / resolver / registry / rollBus / diceBarrier / cards / selftest），所以本任务**不碰 `api` 对象，不增删任何键，不替换任何 `null`**。

---

**背景（动手前必须知道的八件事）**

读者不必了解 Foundry VTT 或 Alien RPG。需要知道的只有：Foundry 是一个跑在浏览器里的桌游平台，`game`、`ui`、`Hooks`、`CONFIG` 这些是它注入到全局作用域的对象；alienrpg 是跑在上面的游戏系统（版本 4.1.13），它的掷骰入口是一个静态方法 `yzeRoll`，掷完把结果写进一个全局可变对象再发一条聊天消息。本模组要做的是把每次掷骰的事实结构化成一条 `RollRecord` 存进那条消息。本文件负责其中**不碰 Foundry 的那一半**。

1. **这个文件是纯函数层。** 契约 §0.1 的分层铁律：`kernel/record.mjs` 里**不得出现任何 Foundry 全局**（`game` / `ui` / `canvas` / `CONFIG` / `Hooks` / `foundry` / `ChatMessage` / `Roll` / `libWrapper`）。它只吃普通对象、吐普通对象，所以 vitest 能直接跑，不需要任何测试桩。`const.mjs` 只有常量，import 它不破坏这条规则。**参数也只能是普通对象、数组与原始值**，绝不能是 Foundry 的 Document 或 Collection —— 副作用层负责先摊平再喂进来。Step 26 有一条护栏测试把这条铁律钉死。

2. **`args` 是被包裹函数的 12 个位置参数装成的同名对象。** 真实签名（`systems/alienrpg/module/helpers/YZEDiceRoller.mjs:31-44`，逐字）：

   ```js
   static async yzeRoll(actortype, blind, reRoll, label,
                        r1Dice, col1, r2Dice, col2,
                        actorid, itemid, tactorid, moddata)
   ```

   `actortype` 的取值来自各调用点的第一个实参（`module/documents/item.mjs` 12 处：`:114 :130 :175 :191 :294 :312 :330 :429 :515 :598 :641 :671`；`module/documents/actor.mjs` 7 处：`:305 :345 :359 :1315 :1524 :1633 :2597`；外加 `module/sheets/vehicle-sheet.mjs:1406`、`module/sheets/spacecraft-sheet.mjs:1510`）。穷举后只有十种：`character` `synthetic` `creature` `vehicles` `spacecraft` `colony` `planet` `territory`（这八个是 `actor.type`，见 `system.json` 的 `documentTypes.Actor`），外加两个系统自造的假类型 —— `supply`（`item.mjs:641` 的 `rollAmmo`、`actor.mjs:1633` 的消耗品检定）与 `item`（`item.mjs:665` 的 `rollComputer` 里 `const effectiveActorType = "item"`，且第 9 个参数存的是 `item.id` 而非 actor id，见 `:663`）。`synthetic` 若开了世界设置 `synthstress`，`actor.mjs:226` 会把它改写成 `character` 再传进来，所以到这里还看得见 `synthetic` 就是「不吃压力、不能推骰」的那种。

3. **`ctx` 是副作用层（`kernel/rollbus.mjs`）用四个包装器采集的上下文帧，摊平成普通对象后传进来。** 它存在的原因见契约 §2：`abilityRoll` 调 `yzeRoll` 时**只传 9 个参数**（`actor.mjs:305`、`:345`、`:359` 三处调用点），所以 `itemid`/`tactorid`/`moddata` 在技能与属性路径上全是 `undefined`；`dataset.attr` 更是在 `actor.mjs:194` 读进局部变量 `attrib` 之后就被丢弃，永远到不了 `yzeRoll`。这些东西只在**上游函数的入参**里存在。你能收到的形状固定为：

   ```js
   ctx = {
     attr:         string|null,   // dataset.attr，来自 alienrpgActor#abilityRoll 的第 2 个参数
     itemUuid:     string|null,   // this.uuid，来自 alienrpgItem#roll
     dataset:      object|null,   // 表单递给 abilityRoll / item.roll 的 dataset 的普通对象浅拷贝
     parentRollId: string|null,   // 来自 alienrpgActor#pushRoll 的第 5 个参数（被推的那条消息）的记录 id
     pushCount:    number,        // 同上，父记录的 push.count + 1
     labelKey:     string|null,   // 由 buildRollRecord 自己算好后塞进副本再交给 classifyKind
   }
   ```

   `ctx` 可能整个是 `null` 或 `{}`：从 GM 宏直接调 `game.alienrpg.yze.yzeRoll(...)` 就没有任何上游帧。所有分支都必须能在 `ctx` 缺席时给出答案。

   **`ctx.actor` / `ctx.token` 不进本文件。** 副作用层的四个包装器还会在帧上挂 `ctx.actor` 与 `ctx.token` 两个真正的 Foundry Document，但它们的用途是喂给 `kernel/resolver.mjs` 的 `resolver.refs(actor, token)` 产出 `refs = {actorUuid, tokenUuid}`；**Document 本身绝不能越过分层线**。本层只认 `refs` 里那两个字符串，**绝不自己拼 uuid**，也**绝不用 actor id 兜底伪造 `tokenUuid`** —— 契约 §4 K1 的诚实边界写死了：聊天卡是 `ChatMessage.getSpeaker({actor: actorid})` 建的（`YZEDiceRoller.mjs:398-401`），token 在消息存在之前就被丢掉了，拿不到就如实写 `null`。

4. **`ctx.attr` 的真实取值是字面量 `"attribute"`，不是属性名。** 4.1.13 的全部模板都写死 `data-attr='attribute'`（`templates/actor/character-header.hbs:40/45/50/55`、`templates/actor/synthetic-header.hbs:38-56`、`templates/actor/spacecraft-general.hbs:25-43` 等 20 余处，无一例外）。它在 `actor.mjs:278` 的 `if (attrib && actor.type !== "synthetic" && !rollMod)` 里只当布尔标记用，含义是「这是属性检定，弹属性对话框」。所以一期 `record.attr` 的实际取值只有 `"attribute"` 或 `null`。**照录不改**：不许从 label 反推属性名，那是猜。

5. **`rollArr` 是系统的单一可变全局 `game.alienrpg.rollArr`**（`module/alienrpg.mjs:96-106` 定义），字段是 `{r1Dice, r1One, r1Six, r2Dice, r2One, r2Six, tLabel, sCount, multiPush}`。`r1*` 是基础骰（黑），`r2*` 是压力骰（黄）。`YZEDiceRoller.mjs:107-114` 在每次掷骰开头把前八个字段逐个清零（**`multiPush` 不清**）。真正填值的是内嵌函数 `buildChat`：`:444-445` 与 `:479-481` 填 `r1*`，`:517`/`:526`/`:532-533` 填 `r2*`。要点：
   - `sCount` 在 `:113` 被清零，掷骰期间恒为 0；调用点是在 `yzeRoll` **返回之后**才把它设成上一轮成功数的（如 `actor.mjs:196`、`item.mjs:126`）。**成功数绝不能算它。**
   - `multiPush` 只在 `:357` 被写，是「多次推骰的累计」，**也不能算进成功数**。
   - 补给掷骰（`actortype === "supply"`，以及 `label === localize("ALIENRPG.RadiationReduced")` 的辐射减免掷骰）的骰子被记进 **`r2*`**（`:517` `game.alienrpg.rollArr.r2Dice = mr.terms[0].number`），`r1*` 保持 0。所以补给检定「消耗几点」= `r2One`，`item.mjs:653-655` 正是照这个字段扣弹药的。
   - `r1Dice` 为负时 `:139` 会 `r2Dice = r2Dice + r1Dice`，**负修正直接吃掉压力骰**；`:154-157` 又把补给骰上限钳到 6。

6. **`pools` 与 `results` 一律取 `rollArr`，绝不取 `args`。** 这是契约 §3 的第 4 条范围裁决。除了上一条列的两处系统内部改写之外，还有一条硬理由：另一条修复（`roll-pool-integrity`）用 **MIXED** 类型的补丁钳制骰池，按 libWrapper 的顺序它跑在本模组 rollBus 的 **WRAPPER 内层**，所以汇点包装器看到的 `args.r1Dice` / `args.r2Dice` 是**钳制前**的请求值。把它们写进记录就是谎报玩家从未掷过的骰子，而下游的成功数行、推骰重掷数全读这两个字段。`args` 只用于 `classifyKind`、`label` 与 `itemid`。

7. **`labelIndex` 是「已本地化文本 → i18n 键」的反查表。** 它存在的原因：`yzeRoll` 收到的 `label` 已经是译文（`actor.mjs:189` `let label = dataset.label`，而模板里写的是 `data-label='{{localize "ALIENRPG.AbilityStr"}}'`；`actor.mjs:254` 的护甲分支更是直接 `label = game.i18n.localize("ALIENRPG.Armor")`），而纯函数层不许碰 `game.i18n`，反解译文又被设计文档明令禁止。反查表把这件事变成一次查表。**同一段译文对应两个键就是歧义，一律返回 `null`，绝不猜。**

8. **`LABEL_KEYS` 的入表判据（本轮定稿，21 键）—— 这是上一轮两个任务各造一张表、键数 26 vs 25 打架的收尾。** 判据只有一条，可逐键核验：

   > **某个键进表，当且仅当存在一条代码路径，把 `{{localize KEY}}` 或 `game.i18n.localize(KEY)` 的产物送进 `yzeRoll` 的第 4 个参数。**

   按这条判据把两份表的并集（27 个候选）逐条核过之后，**21 个进表，6 个出局**。出局的六个必须写进源码注释，否则后来的人会当成漏项补回去：

   | 出局的键 | 出处 | 出局理由（已在 4.1.13 源码里核过） |
   |---|---|---|
   | `ALIENRPG.Stress` | `character-header.hbs:23` | 挂在 `data-action='RollStress'` 上 → `character-sheet.mjs:1055 _onRollStress` → `actor.mjs:545 rollPanic` / `:762 rollPanicMod` / `:1061 rollStress` / `:1270 rollStressMod`。这四条自己 `ChatMessage.create(chatData)` 出卡（`rollStress` 结尾就是），**没有一条调 `yze.yzeRoll`** —— `actor.mjs` 里 yzeRoll 的全部 7 个调用点是 `:305 :345 :359 :1315 :1524 :1633 :2597`，没有一个落在这四个函数体内。契约 §3 裁决 1 说的就是这件事。 |
   | `ALIENRPG.Panic` | `character-enhanced-header.hbs:22` | 同上，同一个 `data-action='RollStress'`。 |
   | `ALIENRPG.Resolve` | `character-enhanced-general.hbs:140`、`synthetic-enhanced-general.hbs:158` | 挂在 `data-action='RollResolve'` 上 → `character-sheet.mjs:1074 _onRollResolve` → `actor.mjs:837 rollResolve` / `:1030 rollResolveMod`，同样不经 `yzeRoll`。 |
   | `ALIENRPG.Speed` | `creature-header.hbs:26` | 那一行写的是**字面量** `data-label='Speed'`，只有旁边的可见文字才是 `{{localize 'ALIENRPG.Speed'}}`。同一张卡的 `:42 'Mobility'`、`:48 'Observation'`、`:54 'Acid Splash'` 也都是字面量。字面量不随语言变，而 `localize("ALIENRPG.Speed")` 在英文世界碰巧等于它、在中文世界是「速度」—— 收进表里等于让 `labelKey` 的结果取决于世界语言。 |
   | `ALIENRPG.InventoryArmorHeader` | `creature-header.hbs:18`、`crt/crtui-creature-header.hbs:20` | 两处都带 `data-spbutt='armor'`。`abilityRoll` 的护甲块（`actor.mjs:251-254`）位于 `switch (actor.type)` **之外**、对任何 actor 类型都会跑，只要 `dataset.spbutt === "armor"` 就把 `label` 覆写成 `localize("ALIENRPG.Armor")`；`creatureAcidRoll` 在 `:2568-2569` 做同样的事。**这个键的译文永远到不了 `yzeRoll`。** |
   | `ALIENRPG.ArmorRating` | `vehicle-header.hbs:13` | 同上，也带 `data-spbutt='armor'`，label 同样被覆写成 `ALIENRPG.Armor`。 |

   **最后两个键不只是没用，收进来还有害。** `lang/en.json` 里 `ALIENRPG.Armor` 与 `ALIENRPG.InventoryArmorHeader` 都是 `"Armor"`，`lang/cn.json` 里都是 `"护甲"`。把它收进同一张表，`"Armor"`/`"护甲"` 就变成歧义、映射到 `null`，于是**所有护甲掷骰都丢掉 `labelKey`**，宏发起的护甲掷骰（没有 `dataset.spbutt`）就再也认不出是护甲。排除它之后，任何语言下护甲掷骰的 label 都是 `localize("ALIENRPG.Armor")`（系统在 `:254` / `:2569` 强制改写），反查稳定得到 `ALIENRPG.Armor`。Step 16 有一条测试直接读真语言包，把「加回去就会中毒」这件事钉成断言。

   剩下的 21 个键在 `en.json` 与 `cn.json` 里**两两不撞车**（Step 16 的真语言包测试对两种语言各断言一次），所以 `buildLabelIndex` 的歧义分支在出厂键表上不会触发 —— 它靠 Step 16 的注入式假 localizer 单测覆盖，那才是它真正要防的将来。

**契约 §3 的四条一期范围裁决，本任务必须逐条照做，不得自行「补全」：**
- `kind` 枚举**只有六个成员**：`"attribute"|"skill"|"weapon"|"armor"|"supply"|"other"`。`panic`/`stress`/`crit`/`ammo` 已从枚举移除 —— 那四条路径（`rollPanic` `actor.mjs:545`、`rollResolve` `:837`、`rollStress` `:1061`、重伤链 `:1780+`）根本不经过 `yzeRoll`，一期无从产出；二期各自补上时**追加**枚举成员而 `RECORD_VERSION` 不变（追加枚举向后兼容）。
- `consumed.ammo` 一期**恒为 `null`**：弹药子掷骰在 `YZEDiceRoller.mjs:571-611` 就地 `new Roll` 并直接 `weapon.update()` 扣弹（`:606-609`），扣了几发从不写进 `rollArr`。
- `targets` 一期**恒为 `[]`**，二期随 `attack-context-binding` 填。
- `push.parentRollId` 一期**只产出、不消费**（消费者是二期的 push-history 特性）。它是有意保留的 schema，**不是死字段**，后来的人不许因为「没人读」把它删掉。

---

- [ ] **Step 1: 写下 `classifyKind` 的失败测试**

在模组根目录 `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/` 下新建 `test/record.test.mjs`：

```js
import { describe, it, expect } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { classifyKind } from "../scripts/kernel/record.mjs";

/** A representative skill roll: a character rolls 5 black dice + 2 yellow stress dice. */
const SKILL_ARGS = {
  actortype: "character",
  blind: false,
  reRoll: false,
  label: "Heavy Machinery",
  r1Dice: 5,
  col1: "Black",
  r2Dice: 2,
  col2: "Yellow",
  actorid: "actor00000000001",
  itemid: undefined,
  tactorid: undefined,
  moddata: undefined,
};

describe("classifyKind", () => {
  it("has a defined branch for every actortype the system's call sites can pass", () => {
    // The five real Actor types that own a dice-pool sheet…
    expect(classifyKind({ ...SKILL_ARGS, actortype: "character" }, null)).toBe("skill");
    expect(classifyKind({ ...SKILL_ARGS, actortype: "synthetic" }, null)).toBe("skill");
    expect(classifyKind({ ...SKILL_ARGS, actortype: "creature" }, null)).toBe("skill");
    expect(classifyKind({ ...SKILL_ARGS, actortype: "vehicles" }, null)).toBe("skill");
    expect(classifyKind({ ...SKILL_ARGS, actortype: "spacecraft" }, null)).toBe("skill");
    // …three that exist in system.json but never reach abilityRoll…
    expect(classifyKind({ ...SKILL_ARGS, actortype: "colony" }, null)).toBe("other");
    expect(classifyKind({ ...SKILL_ARGS, actortype: "planet" }, null)).toBe("other");
    expect(classifyKind({ ...SKILL_ARGS, actortype: "territory" }, null)).toBe("other");
    // …plus the two pseudo-types the system invents.
    expect(classifyKind({ ...SKILL_ARGS, actortype: "supply" }, null)).toBe("supply");
    expect(classifyKind({ ...SKILL_ARGS, actortype: "item" }, null)).toBe("other");
  });

  it("classifies an attribute roll from the ctx.attr marker", () => {
    // Every template writes the LITERAL string "attribute" into data-attr (character-header.hbs:40
    // and 20 more); actor.mjs:194 reads it into `attrib` and :278 uses it as a boolean gate. It
    // never reaches yzeRoll, so the abilityRoll wrapper hands it over in ctx.
    expect(classifyKind(SKILL_ARGS, { attr: "attribute" })).toBe("attribute");
    expect(classifyKind(SKILL_ARGS, { attr: "" })).toBe("skill");
    expect(classifyKind(SKILL_ARGS, { attr: null })).toBe("skill");
  });

  it("classifies an armor roll from the dataset flag or from the reversed label key", () => {
    // actor.mjs:251-254 — `if (dataset.spbutt === "armor") { … label = localize("ALIENRPG.Armor");
    //                        r2Data = 0; reRoll = true; }`. actor.mjs:1339 and :1358 set that same
    // dataset.spbutt when an armor ITEM is clicked, so the armor branch must outrank the weapon one.
    expect(classifyKind(SKILL_ARGS, { dataset: { spbutt: "armor" } })).toBe("armor");
    // A macro-driven roll has no ctx.dataset, but the label still reverses to ALIENRPG.Armor,
    // because the system rewrote the label itself before yzeRoll saw it.
    expect(classifyKind(SKILL_ARGS, { labelKey: "ALIENRPG.Armor" })).toBe("armor");
    // Defensive second key: creature-header.hbs:18 writes InventoryArmorHeader into data-label.
    // Today actor.mjs:254 always overwrites it (which is why LABEL_KEYS excludes it and the reverse
    // lookup can never produce it), but if a future system version stops overwriting, we still cope.
    expect(classifyKind(SKILL_ARGS, { labelKey: "ALIENRPG.InventoryArmorHeader" })).toBe("armor");
    expect(classifyKind(SKILL_ARGS, { dataset: { spbutt: "ammo" } })).toBe("skill");
  });

  it("ranks armor above the attribute marker", () => {
    // An armor roll on a sheet can carry a stale dataset.attr; the armor branch is the truth.
    expect(classifyKind(SKILL_ARGS, { attr: "attribute", dataset: { spbutt: "armor" } })).toBe("armor");
  });

  it("treats a roll that carries an item id or an item uuid as a weapon roll", () => {
    // Ten call sites in item.mjs pass a 10th argument (`itemid`): :114 :130 :175 :191 :294
    // :312 :330 :429 :515 :598 — the last one also passes an 11th (`tactorid`, the target).
    expect(classifyKind({ ...SKILL_ARGS, itemid: "item00000000001" }, null)).toBe("weapon");
    expect(classifyKind(SKILL_ARGS, { itemUuid: "Actor.a1.Item.i1" })).toBe("weapon");
    expect(classifyKind({ ...SKILL_ARGS, actortype: "creature", itemid: "item00000000001" }, null)).toBe("weapon");
  });

  it("does not mistake pushRoll's literal 0 item id for an item", () => {
    // actor.mjs:1325 passes the NUMBER 0 in the itemid slot of its push re-roll.
    expect(classifyKind({ ...SKILL_ARGS, reRoll: "push", itemid: 0 }, null)).toBe("skill");
    expect(classifyKind({ ...SKILL_ARGS, itemid: "" }, null)).toBe("skill");
  });

  it("keeps supply ahead of everything else", () => {
    // item.mjs:641 rollAmmo passes actortype "supply" for an ammo check.
    expect(classifyKind({ ...SKILL_ARGS, actortype: "supply", itemid: "item00000000001" }, { attr: "attribute" })).toBe("supply");
  });

  it("falls back to other for a missing or unknown actortype", () => {
    expect(classifyKind({}, null)).toBe("other");
    expect(classifyKind({ ...SKILL_ARGS, actortype: undefined }, null)).toBe("other");
    expect(classifyKind({ ...SKILL_ARGS, actortype: "hostile" }, null)).toBe("other");
    expect(classifyKind(undefined, undefined)).toBe("other");
  });
});
```

（`node:fs` 与 `node:url` 这两行 import 现在还没人用，Step 16 与 Step 26 会用到；一次写好省得回头改文件头。）

- [ ] **Step 2: 跑它，看它失败**

在模组根目录跑：`npx vitest run test/record.test.mjs -t "classifyKind"`
Expected: FAIL —— `Error: Failed to resolve import "../scripts/kernel/record.mjs" from "test/record.test.mjs". Does the file exist?`（文件还不存在，整份测试文件加载失败）。

- [ ] **Step 3: 写出 `classifyKind`**

新建 `scripts/kernel/record.mjs`，全文如下：

```js
import { RECORD_VERSION } from "../const.mjs";

/**
 * Pure layer of K1 (CONTRACT §4 K1). NOTHING in this file may touch a Foundry global
 * (game / ui / canvas / CONFIG / Hooks / foundry / ChatMessage / Roll / libWrapper), and no
 * parameter may be a Foundry Document or Collection — only plain objects, arrays, primitives.
 * kernel/rollbus.mjs flattens everything before calling in here; in particular it converts its
 * ctx.actor / ctx.token Documents into `refs` with resolver.refs() and passes only the uuids.
 */

/** Actor types that own a dice-pool sheet and reach yzeRoll through abilityRoll / item.roll. */
const POOL_ACTOR_TYPES = new Set(["character", "synthetic", "creature", "vehicles", "spacecraft"]);

/** Actor types that exist in system.json but have no dice pool; listed so the mapping is total. */
const NON_POOL_ACTOR_TYPES = new Set(["colony", "planet", "territory"]);

/**
 * The i18n keys that mean "this was an armor roll".
 * Only ALIENRPG.Armor is reachable through the reverse lookup today: every sheet that writes
 * InventoryArmorHeader into data-label also writes data-spbutt='armor', and actor.mjs:251-254
 * (plus creatureAcidRoll at :2568-2569) then overwrites the label with localize("ALIENRPG.Armor")
 * before yzeRoll sees it — which is why LABEL_KEYS excludes InventoryArmorHeader. The second key
 * stays here as a cheap safety net in case a later system version stops overwriting.
 */
const ARMOR_LABEL_KEYS = new Set(["ALIENRPG.Armor", "ALIENRPG.InventoryArmorHeader"]);

/** True for a usable id/uuid/marker: a non-empty string and nothing else. */
function nonEmpty(value) {
  return typeof value === "string" && value.length > 0;
}

/**
 * @param {object} args the 12 yzeRoll parameters by name
 * @param {object|null} ctx the flattened context frame from kernel/rollbus.mjs
 * @returns {"attribute"|"skill"|"weapon"|"armor"|"supply"|"other"}
 */
export function classifyKind(args, ctx) {
  const actortype = typeof args?.actortype === "string" ? args.actortype : "";
  if (actortype === "supply") return "supply"; // consumables + ammo checks; 0 black dice, N yellow
  if (actortype === "item") return "other"; // item.mjs:665 rollComputer — arg 9 holds an ITEM id
  if (NON_POOL_ACTOR_TYPES.has(actortype)) return "other"; // colony / planet / territory: no pool
  if (!POOL_ACTOR_TYPES.has(actortype)) return "other"; // missing or unknown
  // Armor first: actor.mjs:251-254 overwrites the label and zeroes the stress pool, a sheet can
  // hand over a stale dataset.attr alongside it, and actor.mjs:1339/:1358 set spbutt='armor' on an
  // armor ITEM click — which would otherwise look like a weapon roll.
  if (ctx?.dataset?.spbutt === "armor" || ARMOR_LABEL_KEYS.has(ctx?.labelKey)) return "armor";
  if (nonEmpty(args?.itemid) || nonEmpty(ctx?.itemUuid)) return "weapon";
  if (nonEmpty(ctx?.attr)) return "attribute";
  return "skill";
}
```

- [ ] **Step 4: 跑它，看它通过**

Run: `npx vitest run test/record.test.mjs -t "classifyKind"`
Expected: PASS —— `8 passed`。

- [ ] **Step 5: 提交 classifyKind**

这一个 TDD 循环已经绿了，先落一笔再进下一个。契约要求每个逻辑单元各自提交，
不要攒到任务末尾一次性提交 —— 循环之间互不依赖，分开提交才能单独回退。

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation"
git add scripts/kernel/record.mjs test/record-classify.test.mjs
git commit -m "feat(record): classifyKind 按 yzeRoll 的 12 个入参判定掷骰种类" -m "v1 枚举只收 attribute/skill/weapon/armor/supply/other —— panic/stress/crit/ammo
四条路径根本不经过 yzeRoll，一期无从产出，二期各自的特性再追加枚举成员。
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: 写下 `successesOf` / `banesOf` 的失败测试**

把 `test/record.test.mjs` 里那行 record.mjs 的 import 改成
`import { classifyKind, successesOf, banesOf } from "../scripts/kernel/record.mjs";`
并在文件末尾追加：

```js
/** The exact shape of game.alienrpg.rollArr right after YZEDiceRoller.mjs:107-114 clears it. */
const CLEARED_ROLL_ARR = {
  r1Dice: 0, r1One: 0, r1Six: 0,
  r2Dice: 0, r2One: 0, r2Six: 0,
  tLabel: "", sCount: 0, multiPush: 0,
};

describe("successesOf / banesOf", () => {
  it("adds base sixes to stress sixes and ignores sCount and multiPush", () => {
    // sCount is zeroed at YZEDiceRoller.mjs:113 and only written AFTER the roll returns
    // (actor.mjs:196, item.mjs:126); multiPush is a running push tally written at :357.
    // Neither is a success.
    const rollArr = { ...CLEARED_ROLL_ARR, r1Dice: 5, r1One: 1, r1Six: 2, r2Dice: 3, r2One: 1, r2Six: 1, sCount: 4, multiPush: 3 };
    expect(successesOf(rollArr)).toBe(3);
  });

  it("counts only stress ones as banes", () => {
    expect(banesOf({ ...CLEARED_ROLL_ARR, r1One: 2, r2One: 3 })).toBe(3);
  });

  it("reads a cleared rollArr as zero", () => {
    expect(successesOf(CLEARED_ROLL_ARR)).toBe(0);
    expect(banesOf(CLEARED_ROLL_ARR)).toBe(0);
  });

  it("survives a missing or garbage rollArr instead of returning NaN", () => {
    expect(successesOf(undefined)).toBe(0);
    expect(banesOf(null)).toBe(0);
    expect(successesOf({ r1Six: "2", r2Six: null })).toBe(2);
    expect(banesOf({ r2One: -3 })).toBe(0);
    expect(successesOf({ r1Six: Number.NaN, r2Six: 1 })).toBe(1);
  });
});
```

- [ ] **Step 7: 跑它，看它失败**

Run: `npx vitest run test/record.test.mjs -t "successesOf"`
Expected: FAIL —— `SyntaxError: The requested module '../scripts/kernel/record.mjs' does not provide an export named 'successesOf'`。

- [ ] **Step 8: 写出 `successesOf` / `banesOf`**

在 `scripts/kernel/record.mjs` 的 `nonEmpty()` 下面插入 `count()`：

```js
/** Coerce one rollArr field to a non-negative integer; garbage and NaN become 0. */
function count(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.trunc(n));
}
```

并在文件末尾追加两个导出：

```js
/**
 * Successes = every 6 on either colour. Never sCount (zeroed at YZEDiceRoller.mjs:113)
 * and never multiPush (a push tally, YZEDiceRoller.mjs:357).
 *
 * On a PUSHED roll this counts only the newly rolled sixes: the kept sixes are not re-rolled
 * (actor.mjs:1313 `reRoll1 = r1Dice - r1Six`) and live in the parent record. Consumers that want
 * the cumulative total walk push.parentRollId back through the chain and sum.
 * @param {object} rollArr snapshot of game.alienrpg.rollArr
 */
export function successesOf(rollArr) {
  return count(rollArr?.r1Six) + count(rollArr?.r2Six);
}

/**
 * Banes = every 1 on a yellow stress die. On a supply check the same field is the number of
 * units consumed, because YZEDiceRoller.mjs:511-517 files supply dice (and radiation-reduced
 * dice) under r2* — which is exactly the field item.mjs:653-655 subtracts rounds by.
 * @param {object} rollArr snapshot of game.alienrpg.rollArr
 */
export function banesOf(rollArr) {
  return count(rollArr?.r2One);
}
```

- [ ] **Step 9: 跑它，看它通过**

Run: `npx vitest run test/record.test.mjs -t "successesOf"`
Expected: PASS —— `4 passed`。

- [ ] **Step 10: 提交 successesOf / banesOf**

这一个 TDD 循环已经绿了，先落一笔再进下一个。契约要求每个逻辑单元各自提交，
不要攒到任务末尾一次性提交 —— 循环之间互不依赖，分开提交才能单独回退。

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation"
git add scripts/kernel/record.mjs test/record-counts.test.mjs
git commit -m "feat(record): 成功数与厄运数一律从 rollArr 计数" -m "系统自己从不读 Roll#total（两个骰子类的 get total() 返回的是池子大小），
所以计数只能从 rollArr 的六个字段来，且绝不解析渲染后的文本。
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 11: 写下 `pureReverseLabelKey` 的失败测试**

把 record.mjs 的 import 改成
`import { classifyKind, successesOf, banesOf, pureReverseLabelKey } from "../scripts/kernel/record.mjs";`
并追加：

```js
/**
 * A label index as buildLabelIndex() produces it: localized text -> i18n key, with `null` marking
 * a text that two different keys localize to. Every value below is the real string in
 * systems/alienrpg/lang/en.json.
 *
 * NOTE the "Armor": null entry. This fixture is deliberately built from a key set that INCLUDES
 * ALIENRPG.InventoryArmorHeader, which the shipped LABEL_KEYS excludes (see Step 18) — it exists
 * so the ambiguous branch has an end-to-end fixture. Step 21 covers the shipped behaviour, where
 * "Armor" is unambiguous.
 */
const EN_INDEX = {
  "Armor": null, // ALIENRPG.Armor and ALIENRPG.InventoryArmorHeader are both "Armor"
  "Radiation": "ALIENRPG.Radiation",
  "Radiation Reduced": "ALIENRPG.RadiationReduced",
  "Heavy Machinery": "ALIENRPG.SkillheavyMach",
  "Strength": "ALIENRPG.AbilityStr",
};

const CN_INDEX = {
  "护甲": "ALIENRPG.Armor",
  "辐射": "ALIENRPG.Radiation",
  "辐射减少": "ALIENRPG.RadiationReduced",
  "重型机械": "ALIENRPG.SkillheavyMach",
};

describe("pureReverseLabelKey", () => {
  it("maps a localized label back to its i18n key in any language", () => {
    expect(pureReverseLabelKey("Radiation", EN_INDEX)).toBe("ALIENRPG.Radiation");
    expect(pureReverseLabelKey("Heavy Machinery", EN_INDEX)).toBe("ALIENRPG.SkillheavyMach");
    expect(pureReverseLabelKey("辐射减少", CN_INDEX)).toBe("ALIENRPG.RadiationReduced");
    expect(pureReverseLabelKey("护甲", CN_INDEX)).toBe("ALIENRPG.Armor");
  });

  it("returns null for an ambiguous text instead of guessing", () => {
    expect(pureReverseLabelKey("Armor", EN_INDEX)).toBeNull();
  });

  it("returns null for anything that is not in the index", () => {
    // A weapon roll's label is the item's name; a spacecraft skill roll's label is
    // "{{actor.name}} - {{localize 'ALIENRPG.Skillpiloting'}}" (spacecraft-general.hbs:120),
    // so composite labels never reverse — that is correct, not a bug.
    expect(pureReverseLabelKey("M41A Pulse Rifle", EN_INDEX)).toBeNull();
    expect(pureReverseLabelKey("Nostromo - Piloting", EN_INDEX)).toBeNull();
    expect(pureReverseLabelKey("", EN_INDEX)).toBeNull();
    expect(pureReverseLabelKey("Radiation", undefined)).toBeNull();
    expect(pureReverseLabelKey(undefined, EN_INDEX)).toBeNull();
    expect(pureReverseLabelKey(42, EN_INDEX)).toBeNull();
  });

  it("trims the label before looking it up", () => {
    // Datasets carry whitespace straight from the .hbs templates.
    expect(pureReverseLabelKey("  Radiation  ", EN_INDEX)).toBe("ALIENRPG.Radiation");
  });

  it("accepts a Map as the index as well as a plain object", () => {
    const map = new Map([["Radiation", "ALIENRPG.Radiation"], ["Armor", null]]);
    expect(pureReverseLabelKey("Radiation", map)).toBe("ALIENRPG.Radiation");
    expect(pureReverseLabelKey("Armor", map)).toBeNull();
  });

  it("never returns a prototype member for a label like toString or constructor", () => {
    expect(pureReverseLabelKey("toString", EN_INDEX)).toBeNull();
    expect(pureReverseLabelKey("constructor", EN_INDEX)).toBeNull();
  });
});
```

- [ ] **Step 12: 跑它，看它失败**

Run: `npx vitest run test/record.test.mjs -t "pureReverseLabelKey"`
Expected: FAIL —— `SyntaxError: The requested module '../scripts/kernel/record.mjs' does not provide an export named 'pureReverseLabelKey'`。

- [ ] **Step 13: 写出 `pureReverseLabelKey`**

追加到 `scripts/kernel/record.mjs`：

```js
/**
 * Reverse a label that arrived ALREADY LOCALIZED back to its i18n key.
 *
 * Why this exists: yzeRoll's 4th parameter is display text, not a key — actor.mjs:189 reads
 * `dataset.label` straight off the sheet (the template put `{{localize "ALIENRPG.AbilityStr"}}`
 * in there), and :254 / :273 compare it against localize("ALIENRPG.Armor" | "ALIENRPG.Radiation").
 * Downstream features need the KEY (a radiation card must not get a second success line on top of
 * the system's own total at YZEDiceRoller.mjs:288-334), and this layer may not touch game.i18n.
 * So the effect layer localizes LABEL_KEYS once with buildLabelIndex() and hands the table here.
 *
 * @param {string} label the localized text
 * @param {Record<string,string|null>|Map<string,string|null>|undefined} index text -> key,
 *        with null marking a text that two keys share
 * @returns {string|null} the i18n key, or null when unknown or ambiguous
 */
export function pureReverseLabelKey(label, index) {
  if (typeof label !== "string" || !index) return null;
  const text = label.trim();
  if (!text) return null;
  const hit =
    typeof index.get === "function"
      ? index.get(text)
      : Object.prototype.hasOwnProperty.call(index, text)
        ? index[text]
        : null;
  return typeof hit === "string" && hit.length > 0 ? hit : null;
}
```

- [ ] **Step 14: 跑它，看它通过**

Run: `npx vitest run test/record.test.mjs -t "pureReverseLabelKey"`
Expected: PASS —— `6 passed`。

- [ ] **Step 15: 提交 pureReverseLabelKey**

这一个 TDD 循环已经绿了，先落一笔再进下一个。契约要求每个逻辑单元各自提交，
不要攒到任务末尾一次性提交 —— 循环之间互不依赖，分开提交才能单独回退。

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation"
git add scripts/kernel/record.mjs test/record-labelkey.test.mjs
git commit -m "feat(record): 已本地化标签反查 i18n 键，歧义返回 null" -m "yzeRoll 的 label 入参在传进来时已经本地化，要恢复 labelKey 只能反查。
同一文本对应多个键时返回 null —— 宁可留空也不猜错。
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 16: 写下 `buildLabelIndex` 与 `LABEL_KEYS` 的失败测试**

把 record.mjs 的 import 改成
`import { classifyKind, successesOf, banesOf, pureReverseLabelKey, buildLabelIndex, LABEL_KEYS } from "../scripts/kernel/record.mjs";`
然后在文件末尾追加：

```js
/** The English strings behind the six keys EN_INDEX is built from (real lang/en.json values). */
const EN_STRINGS = {
  "ALIENRPG.Armor": "Armor",
  "ALIENRPG.InventoryArmorHeader": "Armor", // excluded from LABEL_KEYS; kept here on purpose
  "ALIENRPG.Radiation": "Radiation",
  "ALIENRPG.RadiationReduced": "Radiation Reduced",
  "ALIENRPG.SkillheavyMach": "Heavy Machinery",
  "ALIENRPG.AbilityStr": "Strength",
};

/** Foundry's game.i18n.localize returns the KEY itself when the key is missing. */
const fakeLocalize = (key) => EN_STRINGS[key] ?? key;

/** Where the game system lives relative to us: Data/modules/<us>/test -> Data/systems/alienrpg. */
const systemLangFile = (lang) =>
  fileURLToPath(new URL(`../../../systems/alienrpg/lang/${lang}.json`, import.meta.url));

/** Foundry lang files are nested objects; flatten to the dotted keys localize() actually takes. */
function flattenLang(path) {
  const flat = {};
  const walk = (node, prefix) => {
    for (const [k, v] of Object.entries(node)) {
      const dotted = prefix ? `${prefix}.${k}` : k;
      if (v && typeof v === "object") walk(v, dotted);
      else flat[dotted] = v;
    }
  };
  walk(JSON.parse(readFileSync(path, "utf8")), "");
  return flat;
}

const SYSTEM_LANGS_PRESENT = existsSync(systemLangFile("en")) && existsSync(systemLangFile("cn"));

describe("buildLabelIndex", () => {
  it("maps each localized text to the key that produced it", () => {
    const index = buildLabelIndex(fakeLocalize, ["ALIENRPG.Radiation", "ALIENRPG.SkillheavyMach"]);
    expect({ ...index }).toEqual({
      "Radiation": "ALIENRPG.Radiation",
      "Heavy Machinery": "ALIENRPG.SkillheavyMach",
    });
  });

  it("maps a text that two different keys share to null", () => {
    const index = buildLabelIndex(fakeLocalize, ["ALIENRPG.Armor", "ALIENRPG.InventoryArmorHeader"]);
    expect(index["Armor"]).toBeNull();
    // Order must not matter, and a third key hitting the same text must not un-poison it.
    const reversed = buildLabelIndex(fakeLocalize, ["ALIENRPG.InventoryArmorHeader", "ALIENRPG.Armor"]);
    expect(reversed["Armor"]).toBeNull();
    // The same key listed twice is not an ambiguity.
    expect(buildLabelIndex(fakeLocalize, ["ALIENRPG.Armor", "ALIENRPG.Armor"])["Armor"]).toBe("ALIENRPG.Armor");
  });

  it("skips keys the localizer echoes back untranslated, and trims what it keeps", () => {
    const index = buildLabelIndex(fakeLocalize, ["ALIENRPG.NoSuchKey"]);
    expect({ ...index }).toEqual({});
    expect(buildLabelIndex((k) => `  ${EN_STRINGS[k]}  `, ["ALIENRPG.Radiation"])["Radiation"]).toBe("ALIENRPG.Radiation");
    expect({ ...buildLabelIndex(() => "   ", ["ALIENRPG.Radiation"]) }).toEqual({});
  });

  it("returns an empty index instead of throwing on bad input", () => {
    expect({ ...buildLabelIndex(undefined, ["ALIENRPG.Radiation"]) }).toEqual({});
    expect({ ...buildLabelIndex(fakeLocalize, undefined) }).toEqual({});
    expect({ ...buildLabelIndex(fakeLocalize, ["", null, 7]) }).toEqual({});
    // A localizer that throws on one key must not lose the others.
    const flaky = (k) => { if (k === "ALIENRPG.Armor") throw new Error("i18n exploded"); return EN_STRINGS[k] ?? k; };
    expect({ ...buildLabelIndex(flaky, ["ALIENRPG.Armor", "ALIENRPG.Radiation"]) }).toEqual({ "Radiation": "ALIENRPG.Radiation" });
  });

  it("produces exactly the EN_INDEX fixture the reverse lookup is tested against", () => {
    expect({ ...buildLabelIndex(fakeLocalize, Object.keys(EN_STRINGS)) }).toEqual(EN_INDEX);
  });

  // Reads the real installed game system, so it is skipped where alienrpg is absent.
  it.skipIf(!SYSTEM_LANGS_PRESENT)("resolves every LABEL_KEYS entry unambiguously in real alienrpg 4.1.13 lang files", () => {
    expect(LABEL_KEYS.length).toBe(21);
    for (const lang of ["en", "cn"]) {
      const flat = flattenLang(systemLangFile(lang));
      // A typo'd key would silently never reverse — nothing else would ever notice.
      expect(LABEL_KEYS.filter((k) => typeof flat[k] !== "string"), `missing keys in ${lang}.json`).toEqual([]);
      const index = buildLabelIndex((k) => flat[k] ?? k, LABEL_KEYS);
      // No two shipped keys share a translation, in EITHER language: every key keeps its entry.
      expect(Object.keys(index).length, `collision in ${lang}.json`).toBe(LABEL_KEYS.length);
    }
    const en = flattenLang(systemLangFile("en"));
    const enIndex = buildLabelIndex((k) => en[k] ?? k, LABEL_KEYS);
    expect(enIndex["Heavy Machinery"]).toBe("ALIENRPG.SkillheavyMach");
    expect(enIndex["Radiation Reduced"]).toBe("ALIENRPG.RadiationReduced");
    // The armor roll keeps its key, in every language, because actor.mjs:254 rewrites the label
    // to localize("ALIENRPG.Armor") before yzeRoll sees it.
    expect(enIndex["Armor"]).toBe("ALIENRPG.Armor");
    // …and this is exactly what re-adding the excluded sibling would cost: "Armor" (and "护甲")
    // become ambiguous, so EVERY armor roll loses its labelKey. Do not put it back.
    expect(en["ALIENRPG.InventoryArmorHeader"]).toBe("Armor");
    expect(buildLabelIndex((k) => en[k] ?? k, [...LABEL_KEYS, "ALIENRPG.InventoryArmorHeader"])["Armor"]).toBeNull();
  });
});
```

- [ ] **Step 17: 跑它，看它失败**

Run: `npx vitest run test/record.test.mjs -t "buildLabelIndex"`
Expected: FAIL —— `SyntaxError: The requested module '../scripts/kernel/record.mjs' does not provide an export named 'buildLabelIndex'`。

- [ ] **Step 18: 写出 `LABEL_KEYS` 与 `buildLabelIndex`**

追加到 `scripts/kernel/record.mjs`。21 个键全部在 `systems/alienrpg/lang/en.json` 与 `cn.json` 里查证过，出处逐条注明；被排除的六个键连同理由一起写进注释，防止后来的人当漏项补回去：

```js
/**
 * Every i18n key whose LOCALIZED text can end up in yzeRoll's `label` parameter.
 * The effect layer localizes exactly this list once, at Foundry's i18nInit, and feeds the result
 * to buildLabelIndex(). This is the ONLY copy of the key list in the module (CONTRACT §4 K1
 * [v3.1]): kernel/rollbus.mjs and scripts/main.mjs import it from here and must not declare a
 * second one. Phase 2 adds its keys to THIS array.
 *
 * Membership rule, applied key by key against alienrpg 4.1.13: a key belongs here iff some code
 * path puts `{{localize KEY}}` / game.i18n.localize(KEY) into yzeRoll's 4th argument.
 *
 * DELIBERATELY EXCLUDED — all six were audited and rejected; do not "restore" them:
 *  - ALIENRPG.Stress (character-header.hbs:23) and ALIENRPG.Panic (character-enhanced-header.hbs:22)
 *    sit on data-action='RollStress' -> character-sheet.mjs:1055 -> actor.mjs rollPanic :545 /
 *    rollPanicMod :762 / rollStress :1061 / rollStressMod :1270. All four build their own
 *    ChatMessage and none of them calls yze.yzeRoll.
 *  - ALIENRPG.Resolve (character-enhanced-general.hbs:140, synthetic-enhanced-general.hbs:158)
 *    sits on data-action='RollResolve' -> character-sheet.mjs:1074 -> actor.mjs rollResolve :837 /
 *    rollResolveMod :1030. Same story.
 *  - ALIENRPG.Speed: creature-header.hbs:26 writes the LITERAL data-label='Speed' (only the
 *    visible text is localized). Same for 'Mobility' :42, 'Observation' :48, 'Acid Splash' :54.
 *    A literal does not change with the language, so keying off it would make labelKey depend on
 *    which language the world runs in.
 *  - ALIENRPG.InventoryArmorHeader (creature-header.hbs:18, crt/crtui-creature-header.hbs:20) and
 *    ALIENRPG.ArmorRating (vehicle-header.hbs:13): all three carry data-spbutt='armor', and the
 *    armor block at actor.mjs:251-254 — which sits OUTSIDE the switch on actor.type, so it runs
 *    for every actor type — overwrites the label with localize("ALIENRPG.Armor") before yzeRoll
 *    sees it (creatureAcidRoll does the same at :2568-2569). Worse than useless: en.json gives
 *    both ALIENRPG.Armor and ALIENRPG.InventoryArmorHeader the text "Armor" (cn.json: "护甲"), so
 *    including the sibling makes that text ambiguous and EVERY armor roll loses its labelKey.
 */
export const LABEL_KEYS = Object.freeze([
  // The 12 skills. templates/actor/character-skills.hbs:8 renders
  // {{localize (alienConcat "ALIENRPG.Skill" key)}} over CONFIG.ALIENRPG.skills
  // (module/helpers/config.mjs:52-64); crt/crtui-character-skills.hbs:7 is the same.
  "ALIENRPG.SkillheavyMach",
  "ALIENRPG.SkillcloseCbt",
  "ALIENRPG.Skillstamina",
  "ALIENRPG.SkillrangedCbt",
  "ALIENRPG.Skillmobility",
  "ALIENRPG.Skillpiloting",
  "ALIENRPG.Skillcommand",
  "ALIENRPG.Skillmanipulation",
  "ALIENRPG.SkillmedicalAid",
  "ALIENRPG.Skillobservation",
  "ALIENRPG.Skillsurvival",
  "ALIENRPG.Skillcomtech",
  // Creature-only roll: crt/crtui-creature-header.hbs:65 localizes it into data-label, and
  // data-action='creatureAcidRoll' reaches yzeRoll at actor.mjs:2597.
  "ALIENRPG.SkillAcidSplash",
  // The 4 attributes (character-header.hbs:40/45/50/55 and every other header template).
  "ALIENRPG.AbilityStr",
  "ALIENRPG.AbilityWit",
  "ALIENRPG.AbilityAgl",
  "ALIENRPG.AbilityEmp",
  // Armor. actor.mjs:254 forces this exact key onto every armor roll's label, in every language.
  "ALIENRPG.Armor",
  // creature-header.hbs:34 / crtui-creature-header.hbs:36 — data-spbutt='armorVfire', NOT 'armor',
  // so the :254 rewrite does not touch it and the localized text really does reach yzeRoll.
  "ALIENRPG.ArmorVsFire",
  // Radiation. actor.mjs:273 compares the label against the first; actor.mjs:1515 localizes the
  // second straight into the label slot, and YZEDiceRoller.mjs:154/511 branch on it.
  "ALIENRPG.Radiation",
  "ALIENRPG.RadiationReduced",
]);

/**
 * Build the localized-text -> i18n-key table that pureReverseLabelKey() consumes.
 *
 * Pure by injection: `localize` is passed in (the effect layer binds game.i18n.localize at
 * i18nInit, after Babele has had its say), so this file still touches no Foundry global.
 *
 * @param {(key:string)=>string} localize
 * @param {string[]} keys typically LABEL_KEYS
 * @returns {Record<string,string|null>} null-prototype map; a text produced by two different
 *          keys maps to null (ambiguous — pureReverseLabelKey refuses to guess)
 */
export function buildLabelIndex(localize, keys) {
  const index = Object.create(null); // null prototype: a label of "__proto__" cannot poison it
  if (typeof localize !== "function" || !Array.isArray(keys)) return index;
  const owner = new Map(); // localized text -> the first key that produced it
  for (const key of keys) {
    if (typeof key !== "string" || key.length === 0) continue;
    let text;
    try {
      text = localize(key);
    } catch {
      continue; // one broken key must not cost us the other 20
    }
    if (typeof text !== "string") continue;
    text = text.trim();
    if (!text || text === key) continue; // Foundry echoes the key back when it is missing
    if (!owner.has(text)) {
      owner.set(text, key);
      index[text] = key;
    } else if (owner.get(text) !== key) {
      index[text] = null; // sticky: once ambiguous, always ambiguous
    }
  }
  return index;
}
```

- [ ] **Step 19: 跑它，看它通过**

Run: `npx vitest run test/record.test.mjs -t "buildLabelIndex"`
Expected: PASS —— `6 passed`（本机装着 alienrpg 4.1.13，最后那条读真语言包的用例应当真的跑起来并通过；若显示 `5 passed | 1 skipped`，说明 `Data/systems/alienrpg/lang/en.json` 或 `cn.json` 不在预期位置，先确认路径再往下走）。

- [ ] **Step 20: 提交 LABEL_KEYS / buildLabelIndex**

这一个 TDD 循环已经绿了，先落一笔再进下一个。契约要求每个逻辑单元各自提交，
不要攒到任务末尾一次性提交 —— 循环之间互不依赖，分开提交才能单独回退。

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation"
git add scripts/kernel/record.mjs test/record-labelindex.test.mjs
git commit -m "feat(record): LABEL_KEYS 按可达性收键，buildLabelIndex 注入式构建反查表" -m "收键判据是可达性不是数量：一个键进表，当且仅当它本地化后的文本可能出现在
yzeRoll 第四个参数的位置上。少收一个键会让 labelKey 静默变 null。
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 21: 写下 `buildRollRecord` 的失败测试**

把 record.mjs 的 import 改成
`import { classifyKind, successesOf, banesOf, pureReverseLabelKey, buildLabelIndex, LABEL_KEYS, buildRollRecord } from "../scripts/kernel/record.mjs";`
并追加：

```js
/** Every key of RollRecord v1, sorted — the schema the whole plan rests on (CONTRACT §3). */
const RECORD_KEYS = [
  "actorUuid", "at", "attr", "banes", "consumed", "id", "itemUuid", "kind",
  "label", "labelKey", "pools", "push", "results", "successes", "targets",
  "tokenUuid", "userId", "v",
];

const SKILL_INPUT = {
  args: SKILL_ARGS,
  rollArr: { ...CLEARED_ROLL_ARR, r1Dice: 5, r1One: 1, r1Six: 2, r2Dice: 2, r2One: 0, r2Six: 1, tLabel: "Heavy Machinery" },
  refs: { actorUuid: "Scene.scene00000001.Token.token000000001.Actor.actor00000000001", tokenUuid: "Scene.scene00000001.Token.token000000001" },
  ctx: null,
  userId: "user00000000001",
  worldTime: 1200,
  now: 1757030400000,
  id: "roll00000000001",
  labelIndex: EN_INDEX,
};

describe("buildRollRecord", () => {
  it("emits exactly the 18 keys of RollRecord v1", () => {
    const record = buildRollRecord(SKILL_INPUT);
    expect(Object.keys(record).sort()).toEqual(RECORD_KEYS);
    expect(record.v).toBe(1);
  });

  it("records a plain skill roll", () => {
    const record = buildRollRecord(SKILL_INPUT);
    expect(record.kind).toBe("skill");
    expect(record.label).toBe("Heavy Machinery");
    expect(record.labelKey).toBe("ALIENRPG.SkillheavyMach");
    expect(record.pools).toEqual({ base: 5, stress: 2 });
    expect(record.results).toEqual({ baseSixes: 2, baseOnes: 1, stressSixes: 1, stressOnes: 0 });
    expect(record.successes).toBe(3);
    expect(record.banes).toBe(0);
    expect(record.push).toEqual({ count: 0, pushable: true, parentRollId: null });
    expect(record.actorUuid).toBe(SKILL_INPUT.refs.actorUuid);
    expect(record.tokenUuid).toBe(SKILL_INPUT.refs.tokenUuid);
    expect(record.userId).toBe("user00000000001");
    expect(record.id).toBe("roll00000000001");
    expect(record.at).toEqual({ worldTime: 1200, real: 1757030400000 });
  });

  it("threads attr, itemUuid and the push chain out of ctx", () => {
    const attribute = buildRollRecord({ ...SKILL_INPUT, ctx: { attr: "attribute", dataset: { attr: "attribute" } } });
    expect(attribute.kind).toBe("attribute");
    expect(attribute.attr).toBe("attribute"); // the literal marker every template writes

    const weapon = buildRollRecord({
      ...SKILL_INPUT,
      ctx: { itemUuid: "Actor.actor00000000001.Item.item00000000001" },
    });
    expect(weapon.kind).toBe("weapon");
    expect(weapon.itemUuid).toBe("Actor.actor00000000001.Item.item00000000001");

    const pushed = buildRollRecord({
      ...SKILL_INPUT,
      args: { ...SKILL_ARGS, reRoll: "push", itemid: 0 },
      ctx: { parentRollId: "roll00000000001", pushCount: 1 },
      id: "roll00000000002",
    });
    expect(pushed.push).toEqual({ count: 1, pushable: false, parentRollId: "roll00000000001" });
  });

  it("never fabricates a token uuid and is unmoved by anything Document-shaped on ctx", () => {
    // CONTRACT §4 K1: rollbus captures ctx.actor / ctx.token as real Foundry Documents, converts
    // them with resolver.refs() and passes ONLY the two uuid strings in here. If a Document ever
    // leaks through it must change nothing, and a missing token stays null — the card is built
    // from ChatMessage.getSpeaker({actor: actorid}) (YZEDiceRoller.mjs:398-401), so the token is
    // genuinely unknown on most paths and guessing it would corrupt K4.
    const leaked = buildRollRecord({
      ...SKILL_INPUT,
      refs: { actorUuid: "Actor.actor00000000001", tokenUuid: null },
      ctx: { actor: { id: "actor00000000001" }, token: { id: "token000000001" }, attr: "attribute" },
    });
    expect(Object.keys(leaked).sort()).toEqual(RECORD_KEYS);
    expect(leaked.actorUuid).toBe("Actor.actor00000000001");
    expect(leaked.tokenUuid).toBeNull();
    expect(leaked.kind).toBe("attribute");
  });

  it("records an armor roll, classifying it from the dataset when the index is ambiguous", () => {
    const record = buildRollRecord({
      ...SKILL_INPUT,
      args: { ...SKILL_ARGS, reRoll: true, label: "Armor", r2Dice: 0 },
      ctx: { dataset: { spbutt: "armor" } },
      rollArr: { ...CLEARED_ROLL_ARR, r1Dice: 4, r1Six: 1, tLabel: "Armor" },
    });
    expect(record.kind).toBe("armor");
    expect(record.labelKey).toBeNull(); // EN_INDEX poisons "Armor" on purpose; the dataset decided
    expect(record.push.pushable).toBe(false); // actor.mjs:256 forces reRoll = true on armor
  });

  it("classifies a macro-driven armor roll from the label alone with the shipped LABEL_KEYS", () => {
    // The reason ALIENRPG.InventoryArmorHeader is excluded from LABEL_KEYS: with it gone, "Armor"
    // is unambiguous, so an armor roll fired from a GM macro — which has no ctx.dataset at all —
    // still comes out as kind "armor". actor.mjs:254 guarantees the label really is
    // localize("ALIENRPG.Armor") in every language, so this holds in a Chinese world too.
    const shipped = buildLabelIndex(fakeLocalize, LABEL_KEYS);
    expect(shipped["Armor"]).toBe("ALIENRPG.Armor");
    const record = buildRollRecord({
      ...SKILL_INPUT,
      args: { ...SKILL_ARGS, reRoll: true, label: "Armor", r2Dice: 0 },
      ctx: null,
      rollArr: { ...CLEARED_ROLL_ARR, r1Dice: 4, r1Six: 1, tLabel: "Armor" },
      labelIndex: shipped,
    });
    expect(record.labelKey).toBe("ALIENRPG.Armor");
    expect(record.kind).toBe("armor");
  });

  it("leaves the phase-2 and unreachable fields explicitly empty", () => {
    const record = buildRollRecord(SKILL_INPUT);
    expect(record.targets).toEqual([]); // CONTRACT §3 decision 3: filled in phase 2
    expect(record.consumed).toEqual({ ammo: null }); // §3 decision 2: the ammo sub-roll never reaches rollArr
    expect(record.itemUuid).toBeNull();
    expect(record.attr).toBeNull();
  });

  it("takes the pools from what was actually rolled, not from what was requested", () => {
    // CONTRACT §3 decision 4. A -2 modifier makes r1Dice negative; YZEDiceRoller.mjs:139 then
    // eats two stress dice. The roll-pool-integrity MIXED patch clamps args further, INSIDE our
    // WRAPPER, so args are pre-clamp values and writing them into the record would be a lie.
    const record = buildRollRecord({
      ...SKILL_INPUT,
      args: { ...SKILL_ARGS, r1Dice: -2, r2Dice: 5 },
      rollArr: { ...CLEARED_ROLL_ARR, r1Dice: 0, r2Dice: 3, r2Six: 1 },
    });
    expect(record.pools).toEqual({ base: 0, stress: 3 });
    expect(record.successes).toBe(1);
  });

  it("records a supply check: no base pool, and the stress ones are the units consumed", () => {
    const record = buildRollRecord({
      ...SKILL_INPUT,
      args: { ...SKILL_ARGS, actortype: "supply", reRoll: true, label: "Rounds Supply", r1Dice: 0, r2Dice: 8 },
      // YZEDiceRoller.mjs:154-157 caps a supply pool at 6, so 8 requested became 6 rolled.
      rollArr: { ...CLEARED_ROLL_ARR, r1Dice: 0, r2Dice: 6, r2One: 2, r2Six: 1, tLabel: "Rounds Supply" },
    });
    expect(record.kind).toBe("supply");
    expect(record.pools).toEqual({ base: 0, stress: 6 });
    expect(record.successes).toBe(1);
    expect(record.banes).toBe(2);
    expect(record.push).toEqual({ count: 0, pushable: false, parentRollId: null });
  });

  it("derives the push count from reRoll when no ctx frame supplies one", () => {
    const push = (reRoll) => buildRollRecord({ ...SKILL_INPUT, args: { ...SKILL_ARGS, reRoll } }).push;
    expect(push(false).count).toBe(0);
    expect(push("push").count).toBe(1);
    expect(push("mPush").count).toBe(2);
    expect(push("push").pushable).toBe(false);
    expect(push("mPush").pushable).toBe(false);
  });

  it("marks pushable only for the actor types the system's push handler can service", () => {
    const pushable = (over, ctx = null) =>
      buildRollRecord({ ...SKILL_INPUT, args: { ...SKILL_ARGS, ...over }, ctx }).push.pushable;
    // alienrpg.mjs:472 resolves game.actors.get(message.speaker.actor) and :486-492 switches on
    // its .type with only `case "character"` and `default: return`.
    expect(pushable({ actortype: "character" })).toBe(true);
    // actor.mjs:235 puts the PILOT's id in the actorid slot, so a vehicle card's speaker is that
    // character and the switch services it.
    expect(pushable({ actortype: "vehicles" })).toBe(true);
    expect(pushable({ actortype: "spacecraft" })).toBe(true);
    expect(pushable({ actortype: "creature" })).toBe(false);
    // No stress track. A synthstress synthetic arrives as "character" (actor.mjs:226).
    expect(pushable({ actortype: "synthetic" })).toBe(false);
    expect(pushable({ actortype: "supply" })).toBe(false);
    expect(pushable({ actortype: "item" })).toBe(false);
    expect(pushable({ reRoll: true })).toBe(false); // armor / radiation / synthetic-pilot rolls
    expect(pushable({}, { pushCount: 1 })).toBe(false); // already pushed once
  });

  it("builds a record when no actor could be resolved", () => {
    const record = buildRollRecord({ ...SKILL_INPUT, refs: {} });
    expect(record.actorUuid).toBeNull();
    expect(record.tokenUuid).toBeNull();
    expect(record.kind).toBe("skill");
    expect(record.successes).toBe(3);
  });

  it("builds a zero record from an empty input", () => {
    // Real case: templates/actor/spacecraft-general.hbs:25 has data-attr but no data-label,
    // so dataset.label is undefined and the label must degrade to "" rather than "undefined".
    const record = buildRollRecord({});
    expect(Object.keys(record).sort()).toEqual(RECORD_KEYS);
    expect(record.kind).toBe("other");
    expect(record.label).toBe("");
    expect(record.labelKey).toBeNull();
    expect(record.pools).toEqual({ base: 0, stress: 0 });
    expect(record.results).toEqual({ baseSixes: 0, baseOnes: 0, stressSixes: 0, stressOnes: 0 });
    expect(record.successes).toBe(0);
    expect(record.banes).toBe(0);
    expect(record.push).toEqual({ count: 0, pushable: false, parentRollId: null });
    expect(record.at).toEqual({ worldTime: 0, real: 0 });
  });
});
```

- [ ] **Step 22: 跑它，看它失败**

Run: `npx vitest run test/record.test.mjs -t "buildRollRecord"`
Expected: FAIL —— `SyntaxError: The requested module '../scripts/kernel/record.mjs' does not provide an export named 'buildRollRecord'`。

- [ ] **Step 23: 写出 `buildRollRecord`**

追加到 `scripts/kernel/record.mjs` 末尾：

```js
/**
 * The actor types whose roll cards the system's push handler can actually service:
 * alienrpg.mjs:472 resolves `message.speaker.actor` and :486-492 switches on its `type`,
 * returning on `default`. Vehicle and spacecraft rolls put the PILOT's id in the actorid slot
 * (actor.mjs:235), so their speaker is a character.
 */
const PUSHABLE_ACTOR_TYPES = new Set(["character", "vehicles", "spacecraft"]);

/**
 * Push state.
 * - count: how many times THIS card has already been pushed (a first roll is 0).
 * - pushable: whether this card may still be pushed. It is a RULES judgement, not a copy of the
 *   system's render gate: YZEDiceRoller.mjs:176-181 flips reRoll to true whenever a stress 1
 *   shows, which then suppresses the system's own Push button at :378 — that is the defect the
 *   push-correctness feature exists to fix, not the rule.
 * - parentRollId: the id of the record this roll was pushed FROM (null on a first roll).
 *   CONTRACT §3 ruling: phase 1 PRODUCES this field and does not consume it; the consumer is the
 *   phase-2 push-history feature. Do not prune it as dead schema.
 */
function normalizePush(ctx, args) {
  const declared = Number(ctx?.pushCount);
  const pushed =
    Number.isFinite(declared) && declared > 0
      ? Math.trunc(declared)
      : args?.reRoll === "mPush"
        ? 2
        : args?.reRoll === "push"
          ? 1
          : 0;
  const pushable =
    pushed === 0 && !args?.reRoll && PUSHABLE_ACTOR_TYPES.has(typeof args?.actortype === "string" ? args.actortype : "");
  return { count: pushed, pushable, parentRollId: ctx?.parentRollId ?? null };
}

/**
 * Assemble RollRecord v1 (CONTRACT §3).
 * @param {object} input
 * @param {object} input.args      the 12 yzeRoll parameters by name
 * @param {object} input.rollArr   snapshot of game.alienrpg.rollArr taken by kernel/rollbus.mjs
 * @param {{actorUuid?:string|null, tokenUuid?:string|null}} input.refs produced by
 *        resolver.refs(actor, token); this layer never builds a uuid and never guesses a token
 * @param {object|null} input.ctx  flattened context frame from kernel/rollbus.mjs
 * @param {string} input.userId
 * @param {number} input.worldTime
 * @param {number} input.now
 * @param {string} input.id
 * @param {Record<string,string|null>|Map|undefined} input.labelIndex from buildLabelIndex()
 * @returns {object} RollRecord v1
 */
export function buildRollRecord(input) {
  const args = input?.args ?? {};
  const rollArr = input?.rollArr ?? {};
  const refs = input?.refs ?? {};
  const ctx = input?.ctx ?? {};
  const raw = args.label;
  const label = typeof raw === "string" ? raw : raw === undefined || raw === null ? "" : String(raw);
  const labelKey = pureReverseLabelKey(label, input?.labelIndex);
  return {
    v: RECORD_VERSION,
    id: String(input?.id ?? ""),
    actorUuid: refs.actorUuid ?? null,
    tokenUuid: refs.tokenUuid ?? null,
    userId: String(input?.userId ?? ""),
    // labelKey is folded into a COPY of ctx so a macro-driven armor roll, which has no dataset,
    // can still classify from its label without mutating the caller's frame.
    kind: classifyKind(args, { ...ctx, labelKey }),
    label,
    labelKey,
    attr: nonEmpty(ctx.attr) ? ctx.attr : null,
    itemUuid: nonEmpty(ctx.itemUuid) ? ctx.itemUuid : null,
    // Dice actually rolled, not dice requested: a negative base pool eats stress dice
    // (YZEDiceRoller.mjs:139), a supply pool is capped at 6 (:154-157), and the
    // roll-pool-integrity MIXED patch clamps args inside our WRAPPER. CONTRACT §3 decision 4.
    pools: { base: count(rollArr.r1Dice), stress: count(rollArr.r2Dice) },
    results: {
      baseSixes: count(rollArr.r1Six),
      baseOnes: count(rollArr.r1One),
      stressSixes: count(rollArr.r2Six),
      stressOnes: count(rollArr.r2One),
    },
    successes: successesOf(rollArr),
    banes: banesOf(rollArr),
    push: normalizePush(ctx, args),
    targets: [], // CONTRACT §3 decision 3 — phase 2 (attack-context-binding)
    consumed: { ammo: null }, // §3 decision 2 — the ammo sub-roll at :571-611 never writes rollArr
    at: { worldTime: count(input?.worldTime), real: count(input?.now) },
  };
}
```

- [ ] **Step 24: 跑它，看它通过**

Run: `npx vitest run test/record.test.mjs -t "buildRollRecord"`
Expected: PASS —— `13 passed`。

- [ ] **Step 25: 提交 buildRollRecord**

这一个 TDD 循环已经绿了，先落一笔再进下一个。契约要求每个逻辑单元各自提交，
不要攒到任务末尾一次性提交 —— 循环之间互不依赖，分开提交才能单独回退。

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation"
git add scripts/kernel/record.mjs test/record-build.test.mjs
git commit -m "feat(record): RollRecord v1 组装器，pools 取自 rollArr 而非 args" -m "骰池钳制补丁是 MIXED、跑在 rollBus 的 WRAPPER 内层，所以 args 是钳制前的值，
只有 rollArr 反映真正掷出去的骰子。这条弄反了整条链的数字都会偏。
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 26: 加一条分层护栏测试，并亲眼看它能变红**

契约 §0.1 是本文件唯一无法靠功能测试兜住的约束：将来有人往这里加一行 `game.settings.get(...)`，功能测试全绿，模组却会在 vitest 里炸掉、并把纯层拖进 Foundry 依赖。这条护栏扫描源码本身。它**不是**红-绿循环的一环（它现在就该绿），所以本步要求你先人为把它弄红一次，确认它真的有效。

先追加到 `test/record.test.mjs` 末尾：

```js
describe("layering guard (CONTRACT §0.1)", () => {
  const SOURCE = readFileSync(fileURLToPath(new URL("../scripts/kernel/record.mjs", import.meta.url)), "utf8");
  // Strip block comments and line comments first: the JSDoc in this file legitimately NAMES
  // the forbidden globals when it explains where the data comes from.
  const CODE = SOURCE.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, "$1");

  it("mentions no Foundry global outside comments", () => {
    for (const global of ["game", "ui", "canvas", "CONFIG", "Hooks", "foundry", "ChatMessage", "Roll", "libWrapper"]) {
      expect(CODE, `record.mjs must not reference the Foundry global "${global}"`).not.toMatch(new RegExp(`\\b${global}\\b`));
    }
  });

  it("imports nothing but const.mjs", () => {
    const specifiers = [...CODE.matchAll(/from\s+"([^"]+)"/g)].map((m) => m[1]);
    expect(specifiers).toEqual(["../const.mjs"]);
  });
});
```

Run: `npx vitest run test/record.test.mjs -t "layering guard"`
Expected: PASS —— `2 passed`。

现在验证它抓得住违规：在 `scripts/kernel/record.mjs` 的 import 行下面临时加一行 `const probe = game.settings;`，再跑同一条命令。
Expected: FAIL —— `AssertionError: record.mjs must not reference the Foundry global "game"`。
看到红之后**把那一行删掉**，重跑，回到 `2 passed`。

- [ ] **Step 27: 跑整个测试套件**

Run: `npm test`（等价于 `vitest run`，跑上所有已存在的测试文件）
Expected: PASS。其中 `test/record.test.mjs` 一栏是 `39 passed`（classifyKind 8 + successesOf/banesOf 4 + pureReverseLabelKey 6 + buildLabelIndex 6 + buildRollRecord 13 + layering guard 2）。若这台机器上 `Data/systems/alienrpg/lang/` 不存在，则是 `38 passed | 1 skipped`。更早任务的测试文件必须一条都没被弄红 —— 本任务只新增文件，不改任何已有文件、不往 `main.mjs` 插任何一行。

- [ ] **Step 28: 提交**

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation"
git add scripts/kernel/record.mjs test/record.test.mjs
git commit -F - <<'MSG'
feat(kernel): 落地 RollRecord v1 的纯函数层（K1）

按契约 §3 与 §4 K1 实现记录组装器，全程不碰任何 Foundry 全局，参数一律普通对象，
另加一条扫描源码的分层护栏测试把这条铁律钉死。本文件不挂钩子、不进 api 八槽、
不往 main.mjs 插任何一行。

- classifyKind(args, ctx) 对各调用点能传进来的十种 actortype 全部有明确分支：
  五种有骰池的真 Actor 类型 + 三种没有骰池的 + supply（消耗品／弹药）+ item
  （rollComputer，第 9 参存的是 item id，item.mjs:663-665）。attribute 认的是
  ctx.attr 这个标记——全部模板都把 data-attr 写成字面量 "attribute"，且它在
  actor.mjs:194 读完即丢，只有 abilityRoll 包装器拿得到，所以本层照录不反推属性名。
  armor 排在 weapon 前面：点击护甲物品时 actor.mjs:1339/:1358 会写 dataset.spbutt。
- LABEL_KEYS 定稿为 21 键，并按契约 v3.1 认定本文件是全仓唯一属主（rollbus.mjs
  与 main.mjs 一律 import，不得再定义同名符号）。入表判据只有一条：存在代码路径把
  localize(KEY) 的产物送进 yzeRoll 的第 4 个参数。据此逐条核过 27 个候选，排除六个：
  Stress／Panic 挂在 data-action='RollStress' 上，走 rollPanic/rollStress，那几条
  自己发卡、从不调 yzeRoll；Resolve 同理走 rollResolve；Speed 在 creature-header.hbs:26
  写的是字面量 data-label='Speed'（同卡的 Mobility/Observation/Acid Splash 也是），
  字面量不随语言变，收进来会让 labelKey 取决于世界语言；InventoryArmorHeader 与
  ArmorRating 所在的三处都带 data-spbutt='armor'，而 actor.mjs:251-254 的护甲块在
  switch(actor.type) 之外、对任何类型都跑，会把 label 覆写成 localize("ALIENRPG.Armor")，
  所以它们的译文根本到不了 yzeRoll。后两个还有害：en.json 里 Armor 与
  InventoryArmorHeader 都是 "Armor"、cn.json 里都是 "护甲"，收进同一张表会让该文本
  变成歧义、映射为 null，于是所有护甲掷骰都丢掉 labelKey，宏发起的护甲掷骰再也认不出。
  排除之后护甲检定在任何语言下都能稳定反查到 ALIENRPG.Armor。
- buildLabelIndex(localize, keys)：localize 靠注入，i18nInit 阶段由 rollbus 的属主
  调用一次。同一段译文被两个键命中即歧义、该文本映射到 null。一条用例直接读真语言包
  的 en 与 cn 两份，断言 21 键全部存在、两种语言下都不撞车，并把「把
  InventoryArmorHeader 加回去就会毒掉 Armor」这件事钉成断言，防止后来人当漏项补回。
- pureReverseLabelKey 把已本地化的 label 反查回 i18n 键：yzeRoll 的第 4 个参数是译文
  不是键，而下游要靠键区分辐射卡。歧义与未命中一律 null，绝不猜。
- successesOf 只读 r1Six + r2Six：sCount 在 YZEDiceRoller.mjs:113 已被清零，
  multiPush 是推骰累计（:357），两者都不是成功数。推骰卡只计新掷出的 6，
  保留的 6 在父记录里（actor.mjs:1313），累计值由消费者顺 parentRollId 求和。
- banesOf 只读 r2One；补给与辐射减免检定的骰子被系统记在 r2*（:511-517），所以同一个
  字段就是消耗掉的补给点数，item.mjs:653-655 正是照它扣的弹药。
- pools/results 一律取 rollArr 而非 args（契约 §3 裁决 4）：负修正会吃掉压力骰（:139）、
  补给骰上限 6（:154-157），且 roll-pool-integrity 的 MIXED 补丁跑在本层 WRAPPER 内侧，
  args 拿到的是钳制前的值，写进记录就是谎报玩家从未掷过的骰子。
- actorUuid/tokenUuid 只从 refs 抄，本层不拼 uuid、不拿 actor id 伪造 token：卡是
  getSpeaker({actor: actorid}) 建的，token 在消息存在前就丢了，拿不到就如实写 null。
- push.pushable 按规则判定，不照抄系统的渲染门（:176-181 出压力 1 就把 reRoll 翻成
  true，从而在 :378 吞掉推骰按钮）；可推的 actortype 限于 alienrpg.mjs:486-492 那个
  switch 真能服务的三种。

按契约 §3 的四条一期范围裁决：kind 枚举六个成员（panic/stress/crit/ammo 不经过
yzeRoll，二期追加）、consumed.ammo 恒 null、targets 恒空、push.parentRollId 一期
只产出不消费（消费者是二期 push-history，不得当死字段删掉）。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
```
