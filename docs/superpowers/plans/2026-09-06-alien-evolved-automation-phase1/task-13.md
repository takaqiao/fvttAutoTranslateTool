> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 13 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 13: stress-panic-math-repair —— 压力／恐慌状态机的六处缺陷、四条补丁

**Files:**
- Create: `scripts/features/stress-panic-math-repair.pure.mjs`
- Create: `scripts/features/stress-panic-math-repair.mjs`
- Test: `test/stress-panic-math-repair.test.mjs`
- Modify: `lang/en.json`、`lang/cn.json`
- Modify: `scripts/main.mjs`（**只加两行**：`/* AEA-ANCHOR: imports */` 之后一行 `import`、`/* AEA-ANCHOR: features */` 之后一行数组成员。十个锚点一个不删不改，`export const api` 那八个键一个不碰，四个 `ready.*` 子锚点一行不插）

**Interfaces:**

- Consumes：
  - `scripts/const.mjs` 的 `MID`（= `"alien-evolved-automation"`，libWrapper 注册时的包名）。
  - `kernel/features.mjs` 的 `features.register(def)` / `features.enabled(id)`。**调用时机（契约 §5 [v3.1]）**：`main.mjs` 的 `init` 段**第一件事**就是 `for (const f of FEATURES) f.register()`，`features.registerSettings()` 排在它**之后** —— 所以本特性的 def 在生成设置项时已经在册，运行期 `features.enabled("stress-panic-math-repair")` 读到的是一个**已注册**的世界设置，不会抛「未注册键」。本任务**不得**自己调 `features.registerSettings()`。
  - `kernel/patches.mjs` 的 `patches.register(def)` / `patches.status()`（形状 `[{id, type, target, applied, reason, fixedIn}]`）。`await patches.applyAll()` 由 K7 属主插在 `/* AEA-ANCHOR: ready.patches */`，**排在 `for (const f of FEATURES) await f.install()` 之前**，所以本特性的四条包装器是被 `applyAll()` 装上的，`install()` 是空函数。
  - `kernel/registry.mjs` 的 `registry.table("stressResponse")` / `("panicResponse")` / `("panic")` —— 这三个键的绑定由 `await registry.resolveAll()` 解析，它由 K2 属主插在 `/* AEA-ANCHOR: ready.registry */`，是四个 ready 子锚点里**最靠前**的一个，因此本任务在 `install()` 与运行期读到的绑定必定已解析完毕。若某个键未绑定，本任务的做法是**如实报错**（`ui.notifications.error` + 自检条目红），绝不 `game.tables.getName()` 兜底。
  - `kernel/resolver.mjs` 的 `resolver.actorById(id, {warn})` —— 遗留裸 actor id 的唯一解析入口（契约 §7：永不 `game.actors.get()`）。
  - `kernel/selftest.mjs` 的 `selftest.register({id, label, run})`，`label` 传 **i18n 键**而非已本地化文本（登记发生在 `init`，早于 `i18nInit`，此刻语言包尚未加载）。
  - `test/stubs/foundry.mjs` 的 `installFoundryStub(options)` / `uninstallFoundryStub()`（契约 §0.3 [v3.1]）。桩由 Task 1 独占实现，**本任务只读不改，禁止在测试里另造 `globalThis.game` / `globalThis.ui`，也禁止用私有 Map 顶替 `game.settings`**。本任务只依赖桩的三条行为：`installFoundryStub()` 幂等、`ui.notifications.*` 记进 `ctx.notifications`（条目形状 `{type, message}`）、`game.i18n.localize` 可用。
  - Foundry 全局：`game.tables`、`game.user`、`game.settings`、`CONFIG.statusEffects`、`CONFIG.sounds.dice`、`Roll`、`ChatMessage`、`ui.notifications`、`libWrapper`、`foundry.applications.handlebars.renderTemplate`（系统本体在 `module/documents/actor.mjs:1029`、`:1254` 用的就是这个路径）。
  - `main.mjs` 已有的 `const FEATURES = [ /* AEA-ANCHOR: features */ ];`（Task 1 逐字建立）。
  - **不 Consumes `rollBus`**：契约 §4 [v3.1] 把 `yzeRoll` / `abilityRoll` / `itemRoll` / `pushRoll` 四个 libWrapper 目标判给 `rollBus` 独占，任何别的介入必须走 `rollBus.addStage()`。本任务包裹的四个目标是 `rollResolve`、`rollStress`、`toggleStatusEffect`、`rollPanic`，**与那四个没有交集**，所以照常 `libWrapper.register`，不经 `addStage`。

- Produces：
  - `stress-panic-math-repair.pure.mjs`（只导出 `pure*`，不引用任何 Foundry 全局）：`pureResponseMax`、`pureResponseLevel`、`pureCurrentResponseLevel`、`pureResponsePlan`、`pureRemainingResponseEntries`、`pureHighestResponseLevel`、`pureLegacyPanicTableName`，以及返回 `true|false|null` 的 `pureRollResolveIsBuggy` / `pureRollStressIsBuggy` / `pureStatusRecountIsBuggy` / `purePanicTableLookupIsBuggy`（签名见各步）
  - `stress-panic-math-repair.mjs`：`repairedRollResponse(kind, actor, dataset)`、`makeResponseWrapper(kind)`、`statusLevelRecountWrapper(wrapped, statusId, options)`、`installPanicTableShim(tables, boundTable, legacyName)`、`panicTableLookupWrapper(wrapped, ...args)`、`probeVerdict(suffix)` 与四个 `probe*()`、`export const stressPanicMathRepairFeature = {id, register(), install()}`
  - 四条 `patches.register`（`type:"MIXED"`，id 为 `stress-panic-math-repair.rollResolve` / `.rollStress` / `.statusLevelRecount` / `.panicTableLookup`）；六条 `selftest.register`（四条 `.probe` 加 `.tableCeilings`、`.panicTableLookup.live`）；i18n 键 `AEA.feature.stress-panic-math-repair.name` / `.hint` 与 `AEA.stressPanic.noTarget`
  - `main.mjs` 里的两行接线（一行 import、一个 `FEATURES` 数组成员）

---

**背景（不需要 Alien RPG 或 Foundry 前置知识）**

Foundry VTT 是网页版桌游平台；"系统"（这里是 `alienrpg` 4.1.13）提供规则实现，"模组"（我们）在其上打补丁。角色是 `Actor` 文档，`actor.system` 是它的结构化数据（字段由 schema 声明，写不存在的字段会被 Foundry 拒绝并告警），`actor.effects` 是它身上的状态效果集合，每个效果带一个 `statuses` 字符串集合。`RollTable` 是可掷骰的结果表。我们用第三方库 `libWrapper` 包裹系统方法：`MIXED` 包装器签名是 `(wrapped, ...原参数)`，`wrapped()` 调用原实现。

压力机制有两条平行判定，算术相同（`1d6 + 压力等级 − Resolve`），查不同的表：**Stress Response**（Evolved 版，8 行，档位 0–7）与 **Panic Response**（13 行，档位 0–12）。共用一条规则：**掷出你已有的反应时向上升一档，而不是原地重来；且永远不能超过表的最高一行。** 系统把它们实现成 `alienrpgActor` 上的 `rollResolve`（压力反应，`module/documents/actor.mjs:837`）与 `rollStress`（恐慌反应，`:1061`）。调用点：角色卡按钮（`character-sheet.mjs:1083` / `:1070`、`synthetic-sheet.mjs:1070` / `:1057`）、飞船与载具的船员面板（`spacecraft-sheet.mjs:455`、`vehicle-sheet.mjs:451`），以及 `YZEDiceRoller.mjs:208` 的自动恐慌递归（`autopanic` 且 `evolved` 打开时调 `rollResolve`）。第三条是 1e 版的 `rollPanic`（`:545`），走另一张表，由 `character-sheet.mjs:1064`、`spacecraft-sheet.mjs:449`、`vehicle-sheet.mjs:445` 与 `YZEDiceRoller.mjs:229` 调用。

**六处已逐行核对的缺陷**

**(a) `actor.mjs:1100-1113` —— 钳制块与升级块顺序颠倒**

```js
if (rollTotal < 0) { rollTotal = 0; }
else { if (rollTotal >= 12 || oldPanic >= 12) { rollTotal = 12; } }   // :1103-1108 先钳到 12
if (rollTotal <= oldPanic) { rollTotal = oldPanic + 1; }              // :1110-1113 再升一档 → 13
customResults = await table.getResultsForRoll(rollTotal);             // :1115
altDescription = customResults[0].description;                        // :1117
```

`rollResolve` 的 `:871-883` 是**先升档再钳制**，顺序对；`rollStress` 把两块写反了。出货表最后一行区间是 **12–20**，所以 `getResultsForRoll(13)` 仍返回 Catatonic，第一次并不崩；`switch`（`:1118-1235`）只有 `case 0`–`case 11` 加一个 `case 12`，12 以上一律落 `default:`。角色已带 catatonic 时 `status` 保持空串、`if (status)`（`:1237`）整块跳过；GM 每手动清一次 catatonic 再恐慌，`:1240` 就把越界值 +1 写回 `panic.lastRoll`，13→14→…；到 **21** 时 `getResultsForRoll(21)` 返回空数组，`:1117` 的 `customResults[0].description` 抛 TypeError。修法：块序改对，档位就永远停在 12。

**(b)(c) `actor.mjs:856-861` —— 先解引用后判空，而且读的字段根本不存在**

（`:856` 读 `general.stressresponse.lastRoll`、`:857` 读 `header.resolve.calculatedMax`，而合成人的判空守卫在 `:860-861` 才出现。）

`actor-synthetic.mjs` 的 `schema.general` 里**没有 `stressresponse` 字段**（只有 `panic`(:226)、`addpanic`(:238) 等），所以 `.general.stressresponse` 是 `undefined`，`.lastRoll` 直接抛：合成人角色卡点 Resolve（`synthetic-sheet.mjs:1070`）什么都不发生，只在控制台留一行错。而即使普通 character，`actor-character.mjs:243-248` 的 `stressresponse` 也**只声明了 `value`**（initial −1），写回处（`:1001`）用的正是 `stressresponse.value` —— 于是 `:856` 读 `.lastRoll` 恒为 `undefined`，`rollTotal <= undefined` 恒为 false，**升档守卫永远不生效**。

**(d) `actor.mjs:893` —— 接收者写错**

（`:892-893` 是 `effectid = "jumpy"` 紧跟 `if (await this.hasCondition(effectid))`。）

同一 switch 里其余六处（`:910`/`:923`/`:936`/`:949`/`:962`/`:978`）全写 `actor.hasCondition(...)`。**正确的是 `actor`**：`:853-855` 有 `if (dataset.action === "CrewPanic") actor = game.actors.get(dataset.crewpanic)`，actor 被换成了船员；而 `spacecraft-sheet.mjs:455` / `vehicle-sheet.mjs:451` 是 `await this.actor.rollResolve(this.actor, dataset)`，`this` 是飞船／载具（模板 `spacecraft-general.hbs:136`、`vehicle-crew.hbs:32` 带 `data-action="CrewPanic"` 与 `data-crewpanic="{{actor.id}}"`，后者是**裸 actor id**）。于是按船员面板的恐慌按钮时 `:893` 查的是飞船有没有 jumpy（永远没有），走 else 设 `status = "jumpy"`，最后 `:999` 的 `toggleStatusEffect("jumpy")` 把船员身上**已有的** jumpy **切掉**：掷出坏结果反而变好，还少拿 +1 压力。在自己的角色卡上 `this === actor`，所以只在飞船／载具面板现形。

**(e) `actor.mjs:2369-2378` —— 移除状态后按显示名重算等级**

```js
for (const effect of this.effects) {
  for (const effectList of CONFIG.statusEffects) {
    if (effectList.id === effect.name.toLowerCase()) { hitList.push(effectList.tableNumber); }
  }
}
hitList.sort().reverse();
await this.update({ "system.general.panic.lastRoll": hitList[0] });
```

三处叠在一起：(1) `effectList.id`（如 `"jumpy"`）拿去和 ActiveEffect 的**已本地化显示名**比，英文世界靠巧合相等，**中文世界永远不等** —— hitList 为空，先写进一个 `undefined`，再落到 `hitList.length === 0` 分支（`:2379-2385`）把等级重置成 −1，玩家身上还挂着 Frenzy，恐慌等级却被清零（正是契约 §7「永不比对显示名」要防的）；(2) `hitList.sort()` 是**字典序**，`[2, 10]` 排成 `["10","2"]`，reverse 后取到 2；(3) 它把 `tableNumber: 0` 的条目也算进去，而 `module/helpers/config.mjs:231-237` 的 `keepingguard` 是 `resp:"stress", tableNumber:0`，身上只剩 Keeping Guard 时应当算 −1。这一条不修，(a) 修好之后 `lastRoll` 仍会被持续污染。

**(f) `actor.mjs:554` —— 1e 恐慌表按显示名查找**

（`:554` 是 `const table = game.tables.getName("Panic Table")`，`:555-557` 取不到就 `ui.notifications.error(..."ALIENRPG.NoPanicTable")`。）

契约 §4 K2 要求声明 `panic` 绑定键，但一期本来没有任务接管这一行；Babele（中文汉化层）把表改名后，1e 世界的恐慌按钮（`character-sheet.mjs:1064`，条件是系统设置 `alienrpg.evolved` 为假）只会弹「找不到恐慌表」。整个 `rollPanic`（`:545-761`）只有 `:554`/`:555`/`:571` 三处用到 `table`，且 `:554` 位于该方法**第一次 `await`（`:570` 的 `new Roll(modRoll).evaluate()`）之前**，所以不必整体接管：装一个**只拦一次、拦完立刻自我卸载**的 `game.tables.getName` 蒙皮即可，异步窗口为零。

**实现路线**：(a)(b)(c)(d) 都在函数内部，外层包装够不着，所以 `rollResolve` 与 `rollStress` 用 libWrapper `MIXED` **整体接管**（特性关掉时回落 `wrapped()`），接管版把算术抽成纯函数、把 120 行 switch 压成数据表，顺带把 `:845`/`:1067` 那两处 `game.tables.getName("Stress Response Table")` / `("Panic Response Table")` 一并换成 `registry.table()`。(e) 相反：删除动作是对的，只有尾部重算是错的，所以在 `wrapped()` 返回 `false`（＝确实删了效果）之后**重算并覆盖**。(f) 用蒙皮。actor 解析一律经 `resolver.actorById`。四个 `probe()` 一律是**一行**：把 `init` 阶段的源码快照喂给同名纯谓词；纯谓词各带三条 fixture（4.1.13 原文 → `true`，改对之后 → `false`，认不出 → `null`）。

---

- [ ] **Step 1: 写会失败的档位算术测试**

新建 `test/stress-panic-math-repair.test.mjs`：

```js
import { describe, expect, it } from "vitest";
import {
  pureResponseLevel,
  pureResponseMax,
} from "../scripts/features/stress-panic-math-repair.pure.mjs";

describe("pureResponseMax", () => {
  it("knows the top row of the shipped Stress Response Table", () => {
    expect(pureResponseMax("stress")).toBe(7);
  });
  it("knows the top row of the shipped Panic Response Table", () => {
    expect(pureResponseMax("panic")).toBe(12);
  });
});

describe("pureResponseLevel", () => {
  const S = pureResponseMax("stress");
  const P = pureResponseMax("panic");

  it("escalates one row when the roll matches the level you already have", () => {
    expect(pureResponseLevel({ die: 3, stress: 0, resolve: 0, current: 3, max: S })).toBe(4);
  });
  it("keeps a higher roll as rolled", () => {
    expect(pureResponseLevel({ die: 6, stress: 0, resolve: 0, current: 2, max: S })).toBe(6);
  });
  it("never exceeds the top row of the stress response table", () => {
    expect(pureResponseLevel({ die: 6, stress: 9, resolve: 2, current: 2, max: S })).toBe(S);
  });
  it("stays at the top row of the panic table instead of walking off it", () => {
    // Defect (a): actor.mjs:1100-1113 clamps first and escalates second, so it returns 13.
    expect(pureResponseLevel({ die: 3, stress: 0, resolve: 0, current: 12, max: P })).toBe(P);
  });
  it("pulls a stored level that already ran off the table back onto it", () => {
    expect(pureResponseLevel({ die: 1, stress: 0, resolve: 0, current: 20, max: P })).toBe(P);
  });
  it("never drops below zero", () => {
    expect(pureResponseLevel({ die: 1, stress: 0, resolve: 5, current: -1, max: S })).toBe(0);
  });
  it("treats a missing stored level as one below the floor", () => {
    expect(pureResponseLevel({ die: 2, stress: 0, resolve: 0, current: null, max: S })).toBe(2);
  });
});
```

- [ ] **Step 2: 跑它，看它失败**

Run: `npx vitest run test/stress-panic-math-repair.test.mjs`

Expected: FAIL —— `Failed to resolve import "../scripts/features/stress-panic-math-repair.pure.mjs" from "test/stress-panic-math-repair.test.mjs"`，9 条用例一条都没执行。

- [ ] **Step 3: 写档位算术**

新建 `scripts/features/stress-panic-math-repair.pure.mjs`。契约 §6 要求 `.pure.mjs` 只导出 `pure*` 函数，所以两张档位→状态映射与两个上限都是模块私有，对外只经 `pureResponseMax(kind)`。

```js
// Pure layer for stress-panic-math-repair.
// Must not reference any Foundry global (game / ui / canvas / CONFIG / Hooks /
// foundry / ChatMessage / Roll / libWrapper). Only pure* functions are exported.

/** Top row of the shipped Stress Response Table (rows 0-7). */
const STRESS_RESPONSE_MAX = 7;
/** Top row of the shipped Panic Response Table (rows 0-12). */
const PANIC_RESPONSE_MAX = 12;

/** Stress Response level -> status effect id, transcribed from actor.mjs:887-996. */
const STRESS_RESPONSE_STATUS = Object.freeze({
  1: "jumpy", 2: "tunnelvision", 3: "aggravated", 4: "shakes",
  5: "frantic", 6: "deflated", 7: "messup",
});

/** Panic Response level -> status effect id, transcribed from actor.mjs:1118-1235. */
const PANIC_RESPONSE_STATUS = Object.freeze({
  1: "spooked", 2: "noisy", 3: "twitchy", 4: "loseitem", 5: "paranoid", 6: "hesitant",
  7: "freeze", 8: "seekcover", 9: "scream", 10: "flee", 11: "frenzy", 12: "catatonic",
});

export function pureResponseMax(kind) {
  return kind === "panic" ? PANIC_RESPONSE_MAX : STRESS_RESPONSE_MAX;
}

/**
 * Escalate FIRST, clamp SECOND. actor.mjs:1100-1113 does it the other way round,
 * which produces a level one past the top row of the table.
 */
export function pureResponseLevel({ die, stress, resolve, current, min = 0, max }) {
  const floor = Number(min);
  const ceiling = Number(max);
  const cur = Number.isFinite(current) ? Number(current) : floor - 1;

  let level = Number(die) + (Number(stress) - Number(resolve));
  if (!Number.isFinite(level)) level = floor;
  if (level <= cur) level = cur + 1;
  if (level < floor) level = floor;
  if (level > ceiling) level = ceiling;
  return level;
}
```

- [ ] **Step 4: 跑它，看它通过**

Run: `npx vitest run test/stress-panic-math-repair.test.mjs`

Expected: PASS — `9 passed`。

- [ ] **Step 5: 提交**

```bash
git add scripts/features/stress-panic-math-repair.pure.mjs test/stress-panic-math-repair.test.mjs && git commit -m "feat(stress-panic-math-repair): 档位算术改为先升档后钳制" -m "actor.mjs:1100-1113 把钳制块写在升级块前面，满级角色每次恐慌都算出 13。
出货表最后一行是 12-20 会吸收 13，所以首次不崩；GM 每清一次 catatonic 再恐慌就 +1，
到 21 时 getResultsForRoll 返回空数组、:1117 的 customResults[0] 抛异常。
rollResolve(:871-883) 的顺序本来就是对的，照它来。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: 写会失败的「当前档位读取」测试**

追加到 `test/stress-panic-math-repair.test.mjs`，并把顶部 import 里补上 `pureCurrentResponseLevel`：

```js
describe("pureCurrentResponseLevel", () => {
  it("reads the stress level from the field the system actually writes", () => {
    // actor.mjs:1001 writes system.general.stressresponse.value; :856 reads .lastRoll.
    expect(pureCurrentResponseLevel({ stressresponse: { value: 4 } }, "stress")).toBe(4);
  });
  it("returns null instead of throwing when the actor has no stressresponse field", () => {
    // actor-synthetic.mjs declares panic/addpanic/cash under general, but no stressresponse.
    expect(pureCurrentResponseLevel({ panic: { value: 0, lastRoll: 0 } }, "stress")).toBeNull();
  });
  it("reads the panic level from panic.lastRoll", () => {
    expect(pureCurrentResponseLevel({ panic: { lastRoll: 6 } }, "panic")).toBe(6);
  });
  it("returns null when the general sub-tree is missing entirely", () => {
    expect(pureCurrentResponseLevel(undefined, "panic")).toBeNull();
    expect(pureCurrentResponseLevel({}, "stress")).toBeNull();
  });
});
```

- [ ] **Step 7: 跑它，看它失败**

Run: `npx vitest run test/stress-panic-math-repair.test.mjs -t "pureCurrentResponseLevel"`

Expected: FAIL —— `TypeError: pureCurrentResponseLevel is not a function`，4 条用例全红。

- [ ] **Step 8: 实现 pureCurrentResponseLevel**

追加到 `scripts/features/stress-panic-math-repair.pure.mjs`。参数是 `actor.system.general` 这一层的**普通子树**（Foundry 的 SchemaField 读出来就是普通数字与对象，所以测试里的对象字面量是忠实的）：

```js
/**
 * Read the level the actor currently sits at, from the plain system.general sub-tree.
 * "stress" -> general.stressresponse.value  (actor.mjs:856 wrongly reads .lastRoll,
 *             which no schema declares, so the escalation guard is dead)
 * "panic"  -> general.panic.lastRoll
 * Returns null when the field is absent, which is exactly what a synthetic looks
 * like: actor-synthetic.mjs has no stressresponse field at all.
 */
export function pureCurrentResponseLevel(general, kind) {
  if (!general) return null;
  const raw = kind === "panic" ? general.panic?.lastRoll : general.stressresponse?.value;
  return Number.isFinite(raw) ? Number(raw) : null;
}
```

- [ ] **Step 9: 跑它，看它通过**

Run: `npx vitest run test/stress-panic-math-repair.test.mjs`

Expected: PASS — `13 passed`。

- [ ] **Step 10: 写会失败的「档位效果计划」「剩余档位」测试**

追加到测试文件，import 里补上 `pureResponsePlan`、`pureRemainingResponseEntries`、`pureHighestResponseLevel`：

```js
describe("pureResponsePlan (stress response)", () => {
  it("applies a new response", () => {
    expect(pureResponsePlan({ kind: "stress", level: 4, has: new Set() }))
      .toEqual({ status: "shakes", remove: null, stressDelta: 0, noteKey: null, writeLevel: true });
  });
  it("adds a stress level and swaps the description when you already have it", () => {
    expect(pureResponsePlan({ kind: "stress", level: 4, has: new Set(["shakes"]) }))
      .toEqual({ status: null, remove: null, stressDelta: 1, noteKey: "ALIENRPG.YouAlreadyHaveThis", writeLevel: false });
  });
  it("cannot become Jumpy while Deflated", () => {
    expect(pureResponsePlan({ kind: "stress", level: 1, has: new Set(["deflated"]) }))
      .toEqual({ status: null, remove: null, stressDelta: 0, noteKey: "ALIENRPG.CantGetJumpy", writeLevel: false });
  });
  it("removes Jumpy when Deflated arrives", () => {
    // actor.mjs:967-971: the deflated branch toggles jumpy off before setting status.
    expect(pureResponsePlan({ kind: "stress", level: 6, has: new Set(["jumpy"]) }))
      .toEqual({ status: "deflated", remove: "jumpy", stressDelta: 0, noteKey: null, writeLevel: true });
  });
  it("never loses a condition the target already has (defect d)", () => {
    // On the crew-panic button the system checked the SHIP's conditions, decided the
    // crew member did not have Jumpy, and toggled Jumpy back off again.
    const plan = pureResponsePlan({ kind: "stress", level: 1, has: new Set(["jumpy"]) });
    expect(plan.status).toBeNull();
    expect(plan.stressDelta).toBe(1);
  });
  it("does not ask for a level write on an actor whose schema has no such field", () => {
    // A synthetic with synthstress ticked can roll a stress response, but
    // actor-synthetic.mjs has nowhere to store the level.
    expect(pureResponsePlan({ kind: "stress", level: 4, has: new Set(), hasLevelField: false }))
      .toEqual({ status: "shakes", remove: null, stressDelta: 0, noteKey: null, writeLevel: false });
  });
});

describe("pureResponsePlan (panic response)", () => {
  it("Spooked costs one extra stress level", () => {
    expect(pureResponsePlan({ kind: "panic", level: 1, has: new Set() }))
      .toEqual({ status: "spooked", remove: null, stressDelta: 1, noteKey: null, writeLevel: true });
  });
  it("Scream relieves one stress level", () => {
    expect(pureResponsePlan({ kind: "panic", level: 9, has: new Set() }))
      .toEqual({ status: "scream", remove: null, stressDelta: -1, noteKey: null, writeLevel: true });
  });
  it("does nothing when the panic response is already active", () => {
    // actor.mjs:1122-1124: the panic switch's "already have it" branch is empty.
    expect(pureResponsePlan({ kind: "panic", level: 12, has: new Set(["catatonic"]) }))
      .toEqual({ status: null, remove: null, stressDelta: 0, noteKey: null, writeLevel: false });
  });
});

describe("pureRemainingResponseEntries", () => {
  const catalog = [
    { id: "jumpy", resp: "stress", tableNumber: 1 },
    { id: "flee", resp: "panic", tableNumber: 10 },
    { id: "keepingguard", resp: "stress", tableNumber: 0 },
  ];
  it("matches by status id, never by the effect's translated display name (defect e)", () => {
    // actor.mjs:2372 compares effectList.id against effect.name.toLowerCase(); in a
    // Chinese world effect.name is 惊跳 and nothing ever matches.
    const effects = [{ name: "惊跳", statuses: new Set(["jumpy"]) }];
    expect(pureRemainingResponseEntries(effects, catalog))
      .toEqual([{ tableNumber: 1, resp: "stress" }]);
  });
  it("skips unknown ids and effects with no statuses", () => {
    expect(pureRemainingResponseEntries([{ statuses: new Set(["bleeding"]) }, { name: "x" }], catalog)).toEqual([]);
    expect(pureRemainingResponseEntries(undefined, catalog)).toEqual([]);
  });
});

describe("pureHighestResponseLevel", () => {
  const rows = [
    { tableNumber: 2, resp: "panic" },
    { tableNumber: 10, resp: "panic" },
    { tableNumber: 7, resp: "stress" },
  ];
  it("picks the numerically highest, not the lexicographically highest (defect e)", () => {
    // actor.mjs:2377 does hitList.sort().reverse(), i.e. ["10","2"] -> 2.
    expect(pureHighestResponseLevel(rows, "panic")).toBe(10);
    expect(pureHighestResponseLevel(rows, "stress")).toBe(7);
  });
  it("returns the system's own -1 sentinel when nothing is left", () => {
    expect(pureHighestResponseLevel([], "panic")).toBe(-1);
  });
  it("ignores conditions that carry no response level, like Keeping Guard", () => {
    // config.mjs:231-237 gives keepingguard resp:"stress" with tableNumber 0.
    expect(pureHighestResponseLevel([{ tableNumber: 0, resp: "stress" }], "stress")).toBe(-1);
  });
});
```

- [ ] **Step 11: 跑它，看它失败**

Run: `npx vitest run test/stress-panic-math-repair.test.mjs -t "pureResponsePlan"`

Expected: FAIL —— `TypeError: pureResponsePlan is not a function`，`-t` 过滤到的 9 条全红。

- [ ] **Step 12: 实现 pureResponsePlan、pureRemainingResponseEntries、pureHighestResponseLevel**

追加到 `scripts/features/stress-panic-math-repair.pure.mjs`：

```js
/**
 * What a given response level does to the target. `has` is the set of status ids the
 * TARGET already carries — built from the actor the roll belongs to, never from `this`
 * (actor.mjs:893 uses `this`, which on the crew-panic buttons is the craft).
 * `hasLevelField` is false when the schema has nowhere to store the level.
 */
export function pureResponsePlan({ kind, level, has, hasLevelField = true }) {
  const owns = (id) => Boolean(has && typeof has.has === "function" && has.has(id));
  const map = kind === "panic" ? PANIC_RESPONSE_STATUS : STRESS_RESPONSE_STATUS;
  const status = map[level] ?? null;
  const idle = { status: null, remove: null, stressDelta: 0, noteKey: null, writeLevel: false };

  if (!status) return idle;

  if (owns(status)) {
    // Panic: actor.mjs's "already have it" branches are empty. Stress: +1 stress and a note.
    if (kind === "panic") return idle;
    return { status: null, remove: null, stressDelta: 1, noteKey: "ALIENRPG.YouAlreadyHaveThis", writeLevel: false };
  }

  if (kind === "stress") {
    if (status === "jumpy" && owns("deflated")) {
      return { status: null, remove: null, stressDelta: 0, noteKey: "ALIENRPG.CantGetJumpy", writeLevel: false };
    }
    return {
      status,
      remove: status === "deflated" && owns("jumpy") ? "jumpy" : null,
      stressDelta: 0,
      noteKey: null,
      writeLevel: Boolean(hasLevelField),
    };
  }

  const stressDelta = level === 1 ? 1 : level === 9 ? -1 : 0;
  return { status, remove: null, stressDelta, noteKey: null, writeLevel: Boolean(hasLevelField) };
}

/**
 * Response-table rows for the statuses still on an actor. `effects` is any iterable of
 * {statuses: Set<string>}, `catalog` any iterable of {id, tableNumber, resp}. Matching is
 * by status id; actor.mjs:2372 matches against the effect's localized display name.
 */
export function pureRemainingResponseEntries(effects, catalog) {
  const byId = new Map();
  for (const entry of catalog ?? []) if (entry?.id) byId.set(entry.id, entry);
  const out = [];
  for (const effect of effects ?? []) {
    for (const id of effect?.statuses ?? []) {
      const entry = byId.get(id);
      if (entry) out.push({ tableNumber: entry.tableNumber, resp: entry.resp });
    }
  }
  return out;
}

/**
 * Highest level still on the actor after a removal. -1 is the system's own sentinel
 * (actor.mjs:2379-2385). Levels <= 0 do not count: keepingguard is resp:"stress",
 * tableNumber 0, and carrying it is not a stress response.
 */
export function pureHighestResponseLevel(entries, resp) {
  let best = -1;
  for (const entry of entries ?? []) {
    if (entry?.resp !== resp) continue;
    const n = Number(entry.tableNumber);
    if (Number.isFinite(n) && n > 0 && n > best) best = n;
  }
  return best;
}
```

- [ ] **Step 13: 跑它，看它通过**

Run: `npx vitest run test/stress-panic-math-repair.test.mjs`

Expected: PASS — `27 passed`。

- [ ] **Step 14: 提交**

```bash
git add scripts/features/stress-panic-math-repair.pure.mjs test/stress-panic-math-repair.test.mjs && git commit -m "feat(stress-panic-math-repair): 补齐当前档位、档位效果计划与剩余档位三组纯函数" -m "actor.mjs:856 读 general.stressresponse.lastRoll，character schema(:243-248) 只有 value，
升档守卫恒为 undefined；actor-synthetic.mjs 的 general 下没有 stressresponse，:856 先解引用后判空(:860)。
档位效果计划把 120 行 switch 压成数据表，要求调用方传入「目标角色」的状态集合，
从结构上封死 :893 用 this 的接收者错误；hasLevelField 防止往合成人写不存在的字段。
剩余档位替换 :2369-2378 的按显示名匹配 + 字典序排序，并忽略 tableNumber 0 的 keepingguard。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 15: 为四个探针写会失败的双向测试**

契约要求每个补丁带一个可运行的 `probe()`（`true` = 缺陷仍在、该装；`false` = 上游已修、退休）。终检的硬要求是：探针不得是「恒为真」的字符串匹配，必须是可注入源码的纯谓词，并且**两个方向都有 fixture**。这里再加第三个方向 `null`＝「这不是我审过的那段代码，别乱猜」。

追加到测试文件，import 里补上四个 `pure*IsBuggy` 与 `pureLegacyPanicTableName`：

```js
// Verbatim excerpts of alienrpg 4.1.13; "FIXED" variants are what the upstream
// repair would plausibly look like, and must switch the verdict to false.
const ROLL_RESOLVE_4113 = `
    oldStress = actor.getRollData().general.stressresponse.lastRoll;
    resolve = actor.getRollData().header.resolve.calculatedMax;
    if (rollTotal <= oldStress) { rollTotal = oldStress + 1; }
      case 1: { effectid = "jumpy"; if (await this.hasCondition(effectid)) {
      case 2: { effectid = "tunnelvision"; if (await actor.hasCondition(effectid)) {`;
const ROLL_RESOLVE_FIXED = ROLL_RESOLVE_4113
  .replace(".stressresponse.lastRoll", ".stressresponse.value")
  .replace("await this.hasCondition(effectid)", "await actor.hasCondition(effectid)");

const ROLL_STRESS_4113 = `
    if (rollTotal < 0) {
      console.log("Can't go Lower than 0");
      rollTotal = 0;
    } else {
      if (rollTotal >= 12 || oldPanic >= 12) {
        console.log("Can't go heigher than 12");
        rollTotal = 12;
      }
    }
    if (rollTotal <= oldPanic) {
      console.log("you already have condition =>: ", rollTotal, "existing level: ", oldPanic);
      rollTotal = oldPanic + 1;
    }`;
// The repaired body is the same two blocks in rollResolve's order (:871-883).
const ROLL_STRESS_FIXED = [
  ROLL_STRESS_4113.slice(ROLL_STRESS_4113.indexOf("    if (rollTotal <= oldPanic)")),
  ROLL_STRESS_4113.slice(0, ROLL_STRESS_4113.indexOf("    if (rollTotal <= oldPanic)")),
].join("");

const STATUS_RECOUNT_4113 = `
      for (const effect of this.effects) {
        for (const effectList of CONFIG.statusEffects) {
          if (effectList.id === effect.name.toLowerCase()) {
            hitList.push(effectList.tableNumber);
          }
        }
      }
      hitList.sort().reverse();
      await this.update({ "system.general.panic.lastRoll": hitList[0] });`;
const STATUS_RECOUNT_FIXED = STATUS_RECOUNT_4113
  .replace("if (effectList.id === effect.name.toLowerCase()) {", "if (effectList.id === [...effect.statuses][0]) {")
  .replace("hitList.sort().reverse();", "hitList.sort((a, b) => b - a);");

const ROLL_PANIC_4113 = `
    const table = game.tables.getName("Panic Table");
    if (!table) {
      return ui.notifications.error(game.i18n.localize("ALIENRPG.NoPanicTable"));
    }`;
const ROLL_PANIC_FIXED = ROLL_PANIC_4113.replace(
  'game.tables.getName("Panic Table")', 'await fromUuid(game.settings.get("alienrpg", "panicTableUuid"))');

const UNRECOGNISED = "async somethingElse() { return 42; }";

describe("patch probes are injectable predicates, verified in both directions", () => {
  it("pureLegacyPanicTableName is the literal the system looks up", () => {
    expect(pureLegacyPanicTableName()).toBe("Panic Table");
  });

  it("pureRollResolveIsBuggy: 4.1.13 -> true, repaired -> false, foreign body -> null", () => {
    expect(pureRollResolveIsBuggy(ROLL_RESOLVE_4113)).toBe(true);
    expect(pureRollResolveIsBuggy(ROLL_RESOLVE_FIXED)).toBe(false);
    expect(pureRollResolveIsBuggy(UNRECOGNISED)).toBeNull();
  });
  it("pureRollStressIsBuggy: clamp-before-escalate -> true, swapped -> false, foreign -> null", () => {
    expect(pureRollStressIsBuggy(ROLL_STRESS_4113)).toBe(true);
    expect(pureRollStressIsBuggy(ROLL_STRESS_FIXED)).toBe(false);
    expect(pureRollStressIsBuggy(UNRECOGNISED)).toBeNull();
  });

  it("pureStatusRecountIsBuggy: name matching -> true, id matching -> false, foreign -> null", () => {
    expect(pureStatusRecountIsBuggy(STATUS_RECOUNT_4113)).toBe(true);
    expect(pureStatusRecountIsBuggy(STATUS_RECOUNT_FIXED)).toBe(false);
    expect(pureStatusRecountIsBuggy(UNRECOGNISED)).toBeNull();
  });
  it("purePanicTableLookupIsBuggy: getName -> true, uuid -> false, foreign -> null", () => {
    expect(purePanicTableLookupIsBuggy(ROLL_PANIC_4113)).toBe(true);
    expect(purePanicTableLookupIsBuggy(ROLL_PANIC_FIXED)).toBe(false);
    expect(purePanicTableLookupIsBuggy(UNRECOGNISED)).toBeNull();
  });
  it("each defect marker fires on its own", () => {
    expect(pureRollResolveIsBuggy(ROLL_RESOLVE_FIXED.replace("await actor.hasCondition", "await this.hasCondition"))).toBe(true);
    expect(pureStatusRecountIsBuggy(STATUS_RECOUNT_FIXED.replace("hitList.sort((a, b) => b - a);", "hitList.sort().reverse();"))).toBe(true);
    expect(purePanicTableLookupIsBuggy(ROLL_PANIC_4113.replace('"Panic Table"', "'Panic Table'"))).toBe(true);
  });
});
```

- [ ] **Step 16: 跑它，看它失败**

Run: `npx vitest run test/stress-panic-math-repair.test.mjs -t "injectable predicates"`

Expected: FAIL —— `TypeError: pureLegacyPanicTableName is not a function`，6 条用例全红。

- [ ] **Step 17: 实现四个纯谓词**

追加到 `scripts/features/stress-panic-math-repair.pure.mjs`：

```js
/** The display name actor.mjs:554 hard-codes for the 1e panic table. */
const LEGACY_PANIC_TABLE_NAME = "Panic Table";

export function pureLegacyPanicTableName() {
  return LEGACY_PANIC_TABLE_NAME;
}

/**
 * Each predicate takes Function.prototype.toString() of the untouched system
 * method and answers: true = the defect is still there (install the patch),
 * false = upstream fixed it (retire the patch), null = this is not the body we
 * audited, so refuse to guess. `null` must never install a patch.
 */
export function pureRollResolveIsBuggy(source) {
  const src = String(source ?? "");
  if (!src.includes("hasCondition")) return null;         // not rollResolve as we know it
  const readsMissingField = src.includes("general.stressresponse.lastRoll");  // defects (b)(c)
  const wrongReceiver = src.includes("await this.hasCondition(");             // defect (d)
  return readsMissingField || wrongReceiver;
}

export function pureRollStressIsBuggy(source) {
  const src = String(source ?? "");
  const clampAt = src.indexOf("Can't go heigher than 12");
  const escalateAt = src.indexOf("you already have condition");
  if (clampAt < 0 || escalateAt < 0) return null;
  return clampAt < escalateAt;                                                // defect (a)
}

export function pureStatusRecountIsBuggy(source) {
  const src = String(source ?? "");
  if (!src.includes("hitList")) return null;
  return src.includes("effect.name.toLowerCase()") || src.includes("hitList.sort().reverse()"); // defect (e)
}

export function purePanicTableLookupIsBuggy(source) {
  const src = String(source ?? "");
  if (!src.includes("NoPanicTable")) return null;
  return src.includes(`getName("${LEGACY_PANIC_TABLE_NAME}")`)
      || src.includes(`getName('${LEGACY_PANIC_TABLE_NAME}')`);                // defect (f)
}
```

- [ ] **Step 18: 跑它，看它通过**

Run: `npx vitest run test/stress-panic-math-repair.test.mjs`

Expected: PASS — `33 passed`。

- [ ] **Step 19: 提交**

```bash
git add scripts/features/stress-panic-math-repair.pure.mjs test/stress-panic-math-repair.test.mjs && git commit -m "feat(stress-panic-math-repair): 四个探针改成可注入源码的三值纯谓词" -m "终检要求 probe 必须是可注入源码的纯谓词并给出两个方向的 fixture。这里给三个方向：
4.1.13 原文 -> true、改对之后的写法 -> false、认不出的函数体 -> null。null 永远不装补丁，
并在自检里单独报出来，避免上游改写后我们悄悄重复修复（设计文档 §7 二号风险）。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 20: 为副作用层写会失败的测试**

只测**共享桩能忠实模拟**的部分：四条在 `new Roll()` 与模板渲染**之前**就返回的早退分支、两条特性开关关掉后的直通、以及 `toggleStatusEffect` 包装器的非移除短路。

为什么不测完整路径：契约 §0.3 的桩不产真骰点，且完整路径要走 `game.settings.get("core", "rollMode")`（桩对**未注册**的键按约定抛错）与 `foundry.applications.handlebars.renderTemplate`，这两样桩不承诺。完整路径因此交给 Step 36–39 的手工验证，而不是编造一个跑不动的单测（契约 §0.4）。

`vi.mock` 换掉的是**本模组自己的内核模块**（隔离单元），Foundry 全局一律来自共享桩 —— 契约 §0.3 [v3.1] 禁止的是就地造 `globalThis.game` / 用私有 Map 顶替 `game.settings`，不是禁止 mock 兄弟模块。

把测试文件顶部改成下面这样（`vi.hoisted` 保证容器在 `vi.mock` 工厂之前求值；直接引用普通顶层变量会踩模块求值顺序的坑）：

```js
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";
// ... 保留 Step 1/6/10/15 已经写好的那一整串 pure* import，原样不动 ...

const state = vi.hoisted(() => ({ table: null, crew: null, actorByIdCalls: [], enabled: true }));

vi.mock("../scripts/kernel/features.mjs", () => ({
  features: { register: () => {}, enabled: () => state.enabled },
}));
vi.mock("../scripts/kernel/patches.mjs", () => ({
  patches: { register: () => {}, status: () => [] },
}));
vi.mock("../scripts/kernel/selftest.mjs", () => ({ selftest: { register: () => {} } }));
vi.mock("../scripts/kernel/registry.mjs", () => ({ registry: { table: () => state.table } }));
vi.mock("../scripts/kernel/resolver.mjs", () => ({
  resolver: {
    actorById: (id, opts) => { state.actorByIdCalls.push([id, opts]); return state.crew; },
  },
}));

import {
  makeResponseWrapper,
  repairedRollResponse,
  statusLevelRecountWrapper,
} from "../scripts/features/stress-panic-math-repair.mjs";
```

在文件末尾追加：

```js
const anyTable = { getResultsForRoll: async () => [], results: { contents: [] } };

describe("repairedRollResponse early exits", () => {
  let ctx;
  beforeEach(() => {
    ctx = installFoundryStub();
    state.table = anyTable;
    state.crew = null;
    state.actorByIdCalls = [];
    state.enabled = true;
  });
  afterEach(() => uninstallFoundryStub());

  it("reports an unbound table instead of throwing", async () => {
    state.table = null;
    await repairedRollResponse("stress", { type: "character", system: {} }, {});
    // §0.3 fixes the entry shape as {type, message}; the text itself comes from the
    // language pack, so assert the channel and that nothing else was attempted.
    expect(ctx.notifications.map((n) => n.type)).toEqual(["error"]);
    expect(state.actorByIdCalls).toEqual([]);
  });

  it("returns quietly for a synthetic without synthstress instead of throwing on a missing field", async () => {
    // actor.mjs:856 dereferences general.stressresponse.lastRoll before the :860 guard.
    const synthetic = {
      type: "synthetic", name: "Ash",
      system: { header: { synthstress: false }, general: { panic: { lastRoll: 0 } } },
      update: vi.fn(),
    };
    await expect(repairedRollResponse("stress", synthetic, {})).resolves.toBeUndefined();
    expect(synthetic.update).not.toHaveBeenCalled();
    expect(ctx.notifications).toEqual([]);
  });

  it("acts on the crew member from dataset.crewpanic, not on the craft it was called with", async () => {
    // spacecraft-sheet.mjs:455 passes the ship; dataset.crewpanic is the crew member's bare id.
    state.crew = {
      type: "synthetic", name: "Bishop",
      system: { header: { synthstress: false }, general: { panic: { lastRoll: 0 } } },
      update: vi.fn(),
    };
    const craft = { type: "spacecraft", name: "Montero", system: { header: {}, general: {} }, update: vi.fn() };
    await expect(
      repairedRollResponse("stress", craft, { action: "CrewPanic", crewpanic: "xyz" }),
    ).resolves.toBeUndefined();
    expect(state.actorByIdCalls).toEqual([["xyz", { warn: true }]]);
    expect(craft.update).not.toHaveBeenCalled();
    expect(state.crew.update).not.toHaveBeenCalled();
  });

  it("refuses to fall back to the craft when the crew id does not resolve", async () => {
    state.crew = null;
    const craft = { type: "spacecraft", system: { header: {}, general: {} }, update: vi.fn() };
    await expect(repairedRollResponse("stress", craft, { action: "CrewPanic", crewpanic: "gone" })).resolves.toBeUndefined();
    expect(craft.update).not.toHaveBeenCalled();
    expect(state.actorByIdCalls).toEqual([["gone", { warn: true }]]);   // AEA.stressPanic.noTarget
    expect(ctx.notifications.map((n) => n.type)).toEqual(["error"]);
  });

  it("hands the roll straight back to the system when the feature is switched off", async () => {
    // Contract §7 [v3.1]: every repair must be switchable at execution time, no reload.
    state.enabled = false;
    const dataset = { action: "CrewPanic", crewpanic: "xyz" };
    const craft = { type: "spacecraft", name: "Montero" };
    const wrapped = vi.fn(async () => "system return value");
    await expect(makeResponseWrapper("stress").call(craft, wrapped, craft, dataset))
      .resolves.toBe("system return value");
    expect(wrapped).toHaveBeenCalledWith(craft, dataset);
    expect(state.actorByIdCalls).toEqual([]);
    expect(ctx.notifications).toEqual([]);
  });
});

describe("statusLevelRecountWrapper", () => {
  beforeEach(() => { state.enabled = true; });

  it("recounts nothing unless the system actually deleted an effect", async () => {
    // actor.mjs returns false only on the branch that ran deleteEmbeddedDocuments.
    const actor = { type: "character", system: { general: { panic: { lastRoll: 4 } } }, effects: [], update: vi.fn() };
    const wrapped = vi.fn(async () => true);
    await expect(statusLevelRecountWrapper.call(actor, wrapped, "jumpy")).resolves.toBe(true);
    expect(wrapped).toHaveBeenCalledWith("jumpy", {});
    expect(actor.update).not.toHaveBeenCalled();
  });

  it("does not recount at all when the feature is switched off", async () => {
    state.enabled = false;
    const actor = { type: "character", system: { general: { panic: { lastRoll: 4 } } }, effects: [], update: vi.fn() };
    const wrapped = vi.fn(async () => false);
    await expect(statusLevelRecountWrapper.call(actor, wrapped, "jumpy", { active: false })).resolves.toBe(false);
    expect(wrapped).toHaveBeenCalledWith("jumpy", { active: false });
    expect(actor.update).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 21: 跑它，看它失败**

Run: `npx vitest run test/stress-panic-math-repair.test.mjs`

Expected: FAIL —— `Failed to resolve import "../scripts/features/stress-panic-math-repair.mjs"`，整个文件 40 条用例一条都没执行。

- [ ] **Step 22: 写副作用层的判定与包装器**

新建 `scripts/features/stress-panic-math-repair.mjs`。本步只写到 `statusLevelRecountWrapper` 为止；`register()`、补丁清单与蒙皮在后面两步补。

```js
import { MID } from "../const.mjs";
import { features } from "../kernel/features.mjs";
import { registry } from "../kernel/registry.mjs";
import { resolver } from "../kernel/resolver.mjs";
import {
  pureCurrentResponseLevel,
  pureHighestResponseLevel,
  pureLegacyPanicTableName,
  pureRemainingResponseEntries,
  pureResponseLevel,
  pureResponseMax,
  pureResponsePlan,
} from "./stress-panic-math-repair.pure.mjs";

const FEATURE_ID = "stress-panic-math-repair";

const KIND = {
  stress: {
    registryKey: "stressResponse",
    template: "systems/alienrpg/templates/chat/stress-response-roll.hbs",
    levelPath: "system.general.stressresponse.value",
  },
  panic: {
    registryKey: "panicResponse",
    template: "systems/alienrpg/templates/chat/panic-response-roll.hbs",
    levelPath: "system.general.panic.lastRoll",
  },
};

/**
 * The crew-panic buttons call this.actor.rollResolve(this.actor, dataset) — `this` is the
 * craft; the real subject is the bare actor id in dataset.crewpanic (vehicle-crew.hbs:32),
 * which must go through resolver.actorById, never game.actors.get. Returns null rather
 * than falling back to the craft: rolling the crew's response against the ship is the bug.
 */
function resolveTarget(actor, dataset) {
  if (dataset?.action === "CrewPanic" && dataset.crewpanic) {
    return resolver.actorById(dataset.crewpanic, { warn: true });
  }
  return actor ?? null;
}

/** Status ids currently on the actor, from its own active effects. */
function conditionSet(actorDoc) {
  const set = new Set();
  for (const effect of actorDoc.effects ?? []) {
    for (const statusId of effect.statuses ?? []) set.add(statusId);
  }
  return set;
}

export async function repairedRollResponse(kind, actor, dataset) {
  const cfg = KIND[kind];
  const table = registry.table(cfg.registryKey);
  if (!table) return void ui.notifications.error(game.i18n.localize("ALIENRPG.NoResolveTable"));

  const target = resolveTarget(actor, dataset);
  if (!target) return void ui.notifications.error(game.i18n.localize("AEA.stressPanic.noTarget"));
  // Guard BEFORE reading anything: actor-synthetic.mjs has no general.stressresponse,
  // so actor.mjs:856 throws today instead of returning at :861.
  if (target.type === "synthetic" && !target.system?.header?.synthstress) return undefined;

  const header = target.system?.header ?? {};
  const baseStress = target.type === "synthetic" ? 0 : Number(header.stress?.value ?? 0);
  const extraStress = kind === "panic" ? Number(dataset?.stressMod ?? 0) : 0;
  const resolveMod = kind === "panic" ? 0 : Number(dataset?.resolveMod ?? 0);
  // :1083 reads header.resolve.value while :857 reads calculatedMax; unify on
  // calculatedMax so equipment and talent bonuses count on both tracks.
  const resolve = Number(header.resolve?.calculatedMax ?? header.resolve?.value ?? 0);

  const roll = await new Roll("1d6").evaluate();
  const level = pureResponseLevel({
    die: roll.total,
    stress: baseStress + extraStress,
    resolve: resolve + resolveMod,
    current: pureCurrentResponseLevel(target.system?.general, kind),
    min: 0,
    max: pureResponseMax(kind),
  });

  const rows = await table.getResultsForRoll(level);
  const row = rows?.[0] ?? table.results.contents.at(-1);
  if (!row) return void ui.notifications.error(game.i18n.localize("ALIENRPG.NoResolveTable"));

  const storedLevel = kind === "panic"
    ? target.system?.general?.panic?.lastRoll
    : target.system?.general?.stressresponse?.value;
  const plan = pureResponsePlan({
    kind,
    level,
    has: conditionSet(target),
    hasLevelField: storedLevel !== undefined,
  });

  if (plan.stressDelta && header.stress?.value !== undefined) {
    await target.update({
      "system.header.stress.value": Math.max(0, Number(header.stress.value) + plan.stressDelta),
    });
  }
  if (plan.remove) await target.toggleStatusEffect(plan.remove);
  if (plan.status) {
    await target.toggleStatusEffect(plan.status);
    if (plan.writeLevel) await target.update({ [cfg.levelPath]: level });
  }

  const html = await foundry.applications.handlebars.renderTemplate(cfg.template, {
    actorname: target.name,
    img: row.img,
    name: row.name,
    description: plan.noteKey ? game.i18n.localize(plan.noteKey) : row.description,
    resolve,
    modifier: kind === "panic" ? extraStress : baseStress - (resolve + resolveMod),
    result: level,
  });

  const chatData = {
    user: game.user.id,
    speaker: ChatMessage.getSpeaker({ actor: target }),
    content: html,
    sound: CONFIG.sounds.dice,
  };
  ChatMessage.applyRollMode(chatData, game.settings.get("core", "rollMode"));
  return ChatMessage.create(chatData);
}

/** libWrapper MIXED wrapper factory. Switched off -> the system runs untouched. */
export function makeResponseWrapper(kind) {
  return function aeaResponseWrapper(wrapped, actor, dataset) {
    if (!features.enabled(FEATURE_ID)) return wrapped(actor, dataset);
    return repairedRollResponse(kind, actor ?? this, dataset);
  };
}

/**
 * Defect (e): the removal is correct, only the recount after it is wrong, so let the
 * system run and then overwrite the level. actor.mjs returns false ONLY from the branch
 * that deleted effects; the invalid-statusId throw at :2348 happens before anything is
 * deleted, so it must propagate untouched.
 */
export function statusLevelRecountWrapper(wrapped, statusId, options = {}) {
  if (!features.enabled(FEATURE_ID)) return wrapped(statusId, options);
  const actor = this;
  return (async () => {
    const result = await wrapped(statusId, options);
    if (result !== false) return result;
    if (!actor.system?.general?.panic) return result;

    const remaining = pureRemainingResponseEntries(actor.effects, CONFIG.statusEffects);
    const update = { "system.general.panic.lastRoll": pureHighestResponseLevel(remaining, "panic") };
    if (actor.type === "character") {
      update["system.general.stressresponse.value"] = pureHighestResponseLevel(remaining, "stress");
    }
    await actor.update(update);
    return result;
  })();
}
```

- [ ] **Step 23: 跑它，看它通过**

Run: `npx vitest run test/stress-panic-math-repair.test.mjs`

Expected: PASS — `40 passed`。

- [ ] **Step 24: 补上补丁清单、探针与 register()**

追加到 `scripts/features/stress-panic-math-repair.mjs`。契约 §4 K7 的要点：`apply()` 是**无参自装器**，自己调 `libWrapper.register(MID, target, fn, type)`；`target`/`type` 只是 `status()` 展示用的元数据。

`register()` 的调用时刻是 `init` 段最前面的 `for (const f of FEATURES) f.register()`（契约 §5 [v3.1]），比 `features.registerSettings()` 早，所以 def 一定先于设置项生成而在册。源码快照必须在这一刻取：`patches.applyAll()` 要到 `ready.patches` 才装壳，此时读到的还是系统原文。快照读的是 `CONFIG.Actor.documentClass.prototype`，此刻它已经是 `alienrpgActor` —— Foundry 服务端按优先级发脚本标签（系统 esmodule 优先级 6、普通模组 esmodule 优先级 8），所以系统的 `Hooks.once("init")`（`module/alienrpg.mjs:72`，`:123` 赋 `CONFIG.Actor.documentClass`）注册并执行在本模组的 init 回调之前。

在 import 区补上 `patches`、`selftest` 与四个纯谓词：

```js
import { patches } from "../kernel/patches.mjs";
import { selftest } from "../kernel/selftest.mjs";
import {
  purePanicTableLookupIsBuggy,
  pureRollResolveIsBuggy,
  pureRollStressIsBuggy,
  pureStatusRecountIsBuggy,
} from "./stress-panic-math-repair.pure.mjs";
```

在 `statusLevelRecountWrapper` 之后追加：

```js
/** Untouched system method sources, snapshotted at init before anyone wraps them. */
const snapshots = new Map();

const PATCH_SPECS = [
  { suffix: "rollResolve", method: "rollResolve",
    target: "CONFIG.Actor.documentClass.prototype.rollResolve",
    predicate: pureRollResolveIsBuggy, wrapper: makeResponseWrapper("stress") },
  { suffix: "rollStress", method: "rollStress",
    target: "CONFIG.Actor.documentClass.prototype.rollStress",
    predicate: pureRollStressIsBuggy, wrapper: makeResponseWrapper("panic") },
  { suffix: "statusLevelRecount", method: "toggleStatusEffect",
    target: "CONFIG.Actor.documentClass.prototype.toggleStatusEffect",
    predicate: pureStatusRecountIsBuggy, wrapper: statusLevelRecountWrapper },
];

/** -> true (defect present) | false (upstream fixed) | null (body not recognised). */
export function probeVerdict(suffix) {
  const spec = PATCH_SPECS.find((s) => s.suffix === suffix);
  if (!spec) return null;
  return spec.predicate(snapshots.get(spec.method) ?? "");
}

export const probeRollResolve = () => probeVerdict("rollResolve") === true;
export const probeRollStress = () => probeVerdict("rollStress") === true;
export const probeStatusLevelRecount = () => probeVerdict("statusLevelRecount") === true;

function registerPatch(spec) {
  patches.register({
    id: `${FEATURE_ID}.${spec.suffix}`,
    type: "MIXED",
    target: spec.target,          // metadata only; apply() installs the wrapper itself
    minSystem: "4.1.13",
    fixedIn: null,
    probe: () => probeVerdict(spec.suffix) === true,
    apply: () => libWrapper.register(MID, spec.target, spec.wrapper, "MIXED"),
  });

  selftest.register({
    id: `${FEATURE_ID}.${spec.suffix}.probe`,
    label: `AEA.feature.${FEATURE_ID}.name`,
    run() {
      const verdict = probeVerdict(spec.suffix);
      const row = patches.status().find((p) => p.id === `${FEATURE_ID}.${spec.suffix}`);
      if (verdict === null) {
        return { ok: false, detail: `${spec.suffix}: unrecognised ${spec.method} body — the system source changed, re-audit before trusting this patch` };
      }
      const applied = Boolean(row?.applied);
      // The direction that must always hold: a patch may only be installed while the
      // defect is genuinely present. The reverse is legitimate whenever the GM turned
      // the feature or the patch off, so `reason` is reported rather than asserted.
      const ok = !applied || verdict === true;
      return {
        ok,
        detail: `${spec.suffix}: defect=${verdict} applied=${applied} type=${row?.type ?? "-"} target=${row?.target ?? "-"} reason=${row?.reason ?? "-"}`,
      };
    },
  });
}

export const stressPanicMathRepairFeature = {
  id: FEATURE_ID,

  register() {
    const proto = CONFIG.Actor?.documentClass?.prototype;
    for (const spec of PATCH_SPECS) {
      snapshots.set(spec.method, String(proto?.[spec.method] ?? ""));
    }

    features.register({ id: FEATURE_ID, default: "full", gmOnly: false, requires: [], hint: "" });

    for (const spec of PATCH_SPECS) registerPatch(spec);

    selftest.register({
      id: `${FEATURE_ID}.tableCeilings`,
      label: `AEA.feature.${FEATURE_ID}.name`,
      async run() {
        const lines = [];
        let ok = true;
        for (const kind of ["stress", "panic"]) {
          const table = registry.table(KIND[kind].registryKey);
          if (!table) { ok = false; lines.push(`${kind}: registry key "${KIND[kind].registryKey}" unbound`); continue; }
          const top = Math.max(...table.results.contents.map((r) => Number(r.range?.[1] ?? 0)));
          const ceiling = pureResponseMax(kind);
          if (top < ceiling) { ok = false; lines.push(`${kind}: bound table tops out at ${top}, module ceiling is ${ceiling}`); }
          else lines.push(`${kind}: table top ${top} >= ceiling ${ceiling}`);
        }
        return { ok, detail: lines.join("; ") };
      },
    });
  },

  // ready stage: nothing to install. All four patches are installed by patches.applyAll(),
  // which main.mjs runs at /* AEA-ANCHOR: ready.patches */ — before the FEATURES install loop.
  install() {},
};
```

- [ ] **Step 25: 跑全量测试，确认没打红别的文件**

Run: `npm test`

Expected: PASS —— `test/stress-panic-math-repair.test.mjs` 仍是 `40 passed`，其余测试文件的通过数与本步之前一致。

- [ ] **Step 26: 提交**

```bash
git add scripts/features/stress-panic-math-repair.mjs test/stress-panic-math-repair.test.mjs && git commit -m "feat(stress-panic-math-repair): MIXED 接管 rollResolve/rollStress，并在 toggleStatusEffect 后重算档位" -m "前两个方法整体接管而非在外面补刀：块序、读字段、判空次序、接收者四处缺陷都在函数内部，
外层包装够不着。表查找一律经 registry（顺带替掉 :845/:1067 两处 getName），裸 actor id 经 resolver.actorById，
船员 id 解析不到时如实报错而不是回退到载具本身。
第三个补丁只接尾巴：:2369-2378 按译名匹配（中文世界恒不匹配 -> 等级清成 -1）且字典序排序（10 输给 2），
只在 wrapped() 返回 false 时重算，:2348 的非法 statusId 抛错原样透传。
顺带把恐慌的 Resolve 从 header.resolve.value(:1083) 统一到 calculatedMax(:857)。
源码快照在 init 取：系统 esmodule 的脚本优先级高于普通模组，此刻 CONFIG.Actor.documentClass
已是 alienrpgActor，而 patches.applyAll() 要到 ready.patches 才装壳。
三个包装器执行时都查 features.enabled()，关掉即原样放行，不需要重载世界。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 27: 为 1e 恐慌表蒙皮写会失败的测试**

做法不是接管两百行的 `rollPanic`，而是在包装器里临时替换 `game.tables.getName` **并在第一次被调用时立刻自我卸载**：`:554` 位于 `rollPanic` 第一次 `await`（`:570` 的 1d6）之前，所以蒙皮存活期内没有任何异步窗口。

追加到测试文件末尾（import 里补 `installPanicTableShim`）：

```js
describe("installPanicTableShim", () => {
  it("hands the bound table to the first lookup of the legacy name, then uninstalls itself", () => {
    const legacy = { id: "legacy" };
    const bound = { id: "bound" };
    const tables = { getName: (name) => (name === "Panic Table" ? legacy : null) };
    const original = tables.getName;
    installPanicTableShim(tables, bound);
    expect(tables.getName("Panic Table")).toBe(bound);
    expect(tables.getName).toBe(original);
    expect(tables.getName("Panic Table")).toBe(legacy);
  });

  it("passes any other name straight through", () => {
    const tables = { getName: (name) => ({ id: name }) };
    installPanicTableShim(tables, { id: "bound" });
    expect(tables.getName("Critical Injuries").id).toBe("Critical Injuries");
  });

  it("restore() is idempotent and never clobbers a later reassignment", () => {
    const tables = { getName: () => null };
    const restore = installPanicTableShim(tables, { id: "bound" });
    restore();
    const replacement = () => "someone else's";
    tables.getName = replacement;
    restore();
    expect(tables.getName).toBe(replacement);
  });
});
```

- [ ] **Step 28: 跑它，看它失败**

Run: `npx vitest run test/stress-panic-math-repair.test.mjs -t "installPanicTableShim"`

Expected: FAIL —— `TypeError: installPanicTableShim is not a function`，3 条用例全红。

- [ ] **Step 29: 实现蒙皮、包装器与第四条补丁**

在 `scripts/features/stress-panic-math-repair.mjs` 里，把下面两个函数插在 `PATCH_SPECS` 定义**之前**（`PATCH_SPECS` 会引用 `panicTableLookupWrapper`；函数声明会提升，但放在前面更好读）：

```js
/**
 * Make game.tables.getName("Panic Table") answer with the document bound to the registry
 * key, then take itself back out on the first call. actor.mjs:554 runs before rollPanic's
 * first await (:570), so the shim's whole life is one synchronous stretch.
 * Returns an idempotent restore().
 */
export function installPanicTableShim(tables, boundTable, legacyName = pureLegacyPanicTableName()) {
  const original = tables.getName;
  let restored = false;
  const restore = () => {
    if (restored) return;
    restored = true;
    tables.getName = original;
  };
  tables.getName = function aeaPanicTableShim(name, ...rest) {
    restore();
    return name === legacyName ? boundTable : original.call(this, name, ...rest);
  };
  return restore;
}

/** Defect (f): rollPanic looks its table up by display name, so Babele breaks it. */
export function panicTableLookupWrapper(wrapped, ...args) {
  if (!features.enabled(FEATURE_ID)) return wrapped(...args);
  const bound = registry.table("panic");
  if (!bound) return wrapped(...args);
  const restore = installPanicTableShim(game.tables, bound);
  try {
    return wrapped(...args);
  } finally {
    restore();
  }
}
```

在 `PATCH_SPECS` 数组末尾加第四行：

```js
  { suffix: "panicTableLookup", method: "rollPanic",
    target: "CONFIG.Actor.documentClass.prototype.rollPanic",
    predicate: purePanicTableLookupIsBuggy, wrapper: panicTableLookupWrapper },
```

在 `probeStatusLevelRecount` 后面加一行导出：

```js
export const probePanicTableLookup = () => probeVerdict("panicTableLookup") === true;
```

并在 `register()` 里 `tableCeilings` 那条 `selftest.register` 之后，再加一条现场断言：

```js
    selftest.register({
      id: `${FEATURE_ID}.panicTableLookup.live`,
      label: `AEA.feature.${FEATURE_ID}.name`,
      run() {
        const bound = registry.table("panic");
        if (!bound) return { ok: false, detail: 'registry key "panic" is unbound' };
        // Going through getName here asserts the shim; every real lookup uses registry.table().
        const restore = installPanicTableShim(game.tables, bound);
        let seen;
        try { seen = game.tables.getName(pureLegacyPanicTableName()); } finally { restore(); }
        const ok = seen?.id === bound.id;
        return {
          ok,
          detail: ok
            ? `the shim hands rollPanic the bound table (${bound.id})`
            : `shim returned ${seen?.id ?? "nothing"}, expected ${bound.id}`,
        };
      },
    });
```

- [ ] **Step 30: 跑它，看它通过**

Run: `npx vitest run test/stress-panic-math-repair.test.mjs`

Expected: PASS — `43 passed`。

- [ ] **Step 31: 提交**

```bash
git add scripts/features/stress-panic-math-repair.mjs test/stress-panic-math-repair.test.mjs && git commit -m "feat(stress-panic-math-repair): 1e 恐慌表改走 registry 绑定，不再按显示名查" -m "actor.mjs:554 硬写 game.tables.getName(\"Panic Table\")，Babele 改名后 1e 恐慌按钮只弹「找不到恐慌表」。
契约 §4 K2 要求声明 panic 键，终检把这一行点名并进本任务。不接管两百行的 rollPanic：
:554 在该方法第一次 await(:570) 之前，所以装一个只拦第一次、拦完立刻卸载的 getName 蒙皮，
异步窗口为零，restore 幂等且不覆盖他人的重新赋值。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 32: 加 i18n 键**

合并进 `lang/en.json` 已有的顶层 `AEA` 对象（语言包是嵌套结构，顶层只有 `AEA` 一个键）：

```json
{
  "AEA": {
    "feature": {
      "stress-panic-math-repair": {
        "name": "Stress and panic maths",
        "hint": "Escalate before clamping, read the level from the field the system writes, stop synthetics throwing, act on the crew member rather than the craft, recount by status id instead of translated name, and find the 1e panic table by binding."
      }
    },
    "stressPanic": {
      "noTarget": "No actor to roll a stress or panic response for."
    }
  }
}
```

合并进 `lang/cn.json`：

```json
{
  "AEA": {
    "feature": {
      "stress-panic-math-repair": {
        "name": "压力与恐慌算术",
        "hint": "先升档再钳制、按系统实际写入的字段读当前档位、合成人不再抛异常、船员面板一律作用于船员而非载具、移除状态后按状态 id 而非译名重算档位、1e 恐慌表按绑定而非显示名查找。"
      }
    },
    "stressPanic": {
      "noTarget": "没有可供判定压力／恐慌反应的角色。"
    }
  }
}
```

- [ ] **Step 33: 在 main.mjs 接线（两行，按锚点文本定位）**

`scripts/main.mjs` 由 Task 1 逐字建立，带**十个**锚点注释：`/* AEA-ANCHOR: imports */`、`/* AEA-ANCHOR: features */`、`/* AEA-ANCHOR: repairs */`、`/* AEA-ANCHOR: init */`、`/* AEA-ANCHOR: i18nInit */`、`/* AEA-ANCHOR: diceSoNiceReady */`，以及四个有序的 ready 子锚点 `ready.registry` / `ready.patches` / `ready.rollbus` / `ready.cards`。骨架里 `init` 段已在跑 `for (const f of FEATURES) safely(..., () => f.register())`、`ready` 段已在跑 `for (const f of FEATURES) await safely(..., () => f.install())`。

本任务**只用两个锚点，各插一行**，其余八个锚点、`export const api` 那八个键、以及四个生命周期钩子体一个字都不动：

1. 找到 `/* AEA-ANCHOR: imports */` 这一行，在它的**下一行**插入：

```js
import { stressPanicMathRepairFeature } from "./features/stress-panic-math-repair.mjs";
```

2. 找到 `const FEATURES = [` 里面那一行 `/* AEA-ANCHOR: features */`，在它的**下一行**插入（缩进两个空格，行尾逗号）：

```js
  stressPanicMathRepairFeature,
```

**禁止**在 `init` / `ready` 块里再写一次 `stressPanicMathRepairFeature.register()` 或 `.install()`：那会让 `features.register` 重复登记、`patches.register` 重复登记，`patches.applyAll()` 随后对同一目标二次 `libWrapper.register`，lib-wrapper 会抛 `A wrapper for '...' has already been registered by alien-evolved-automation`，ready 阶段直接炸。生命周期调用**只**由那两个 `for..of` 循环承担。

- [ ] **Step 34: 跑全量测试**

Run: `npm test`

Expected: PASS —— 全部测试文件通过；`test/stress-panic-math-repair.test.mjs` 是 `43 passed`。若 Task 1 带了「语言包顶层唯一键为 AEA」的护栏测试，它也必须仍是绿的（本任务新加的两组键都嵌在 `AEA` 之下）。

- [ ] **Step 35: 提交**

```bash
git add scripts/main.mjs lang/en.json lang/cn.json && git commit -m "feat(stress-panic-math-repair): 加入 FEATURES 数组并补齐中英文案" -m "main.mjs 只加两行：AEA-ANCHOR: imports 之后一行 import、AEA-ANCHOR: features 之后一行数组成员。
十个锚点一行未动，api 的八个键一个未碰，四个 ready 子锚点一行未插。
生命周期调用由 Task 1 的骨架统一承担（init 段 for..of FEATURES f.register()、ready 段 f.install()），
禁止重复直调，否则 patches.applyAll() 会对同一目标二次注册而抛错。
特性名与说明走 AEA.feature.stress-panic-math-repair.name/.hint，嵌套在顶层 AEA 之下。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 36: MANUAL VERIFICATION 1 —— 满级恐慌不再走出表外**

在 Foundry 里打开这个世界（系统设置 `alienrpg` 的 `evolved` 打开），然后：

1. 打开一个 character 角色卡，按 F12 执行 `game.actors.getName("<角色名>").update({"system.general.panic.lastRoll": 12})`
2. 点角色卡上的恐慌（Panic）按钮。
3. **期望**：卡上结果数字是 **12**，内容是 Panic Response Table 最后一行（Catatonic）；控制台无异常；`...system.general.panic.lastRoll` 仍是 12。**今天**（本特性设为 off 可复现）：数字是 13 并被写回；每清一次 catatonic 再恐慌就 +1，到 21 时抛 `TypeError: Cannot read properties of undefined (reading 'description')`。

- [ ] **Step 37: MANUAL VERIFICATION 2 —— 合成人可以点 Resolve**

1. 新建一个 synthetic 角色，勾选 "Human Panic, Push, etc."（`synthstress`），点 Resolve。
   **期望**：出现一张压力反应聊天卡；控制台既无 `TypeError`，也**没有** schema 校验警告（形如 `does not exist in the schema`）—— 合成人没有 `general.stressresponse`，本模组因此不去写它。
2. 再新建一个**没勾** synthstress 的 synthetic，点 Resolve。**期望**：什么都不发生且控制台无报错（这是系统本来就有的提前返回，只是被 `:856` 的解引用抢在了前面）。**今天**：两种情况都只留 `TypeError: Cannot read properties of undefined (reading 'lastRoll')`，永远没有卡。

- [ ] **Step 38: MANUAL VERIFICATION 3 —— 升档、船员面板、移除后重算**

1. 在 character 角色卡上点 Resolve 若干次，拿到某个压力反应（比如 Shakes，档位 4），再点 Resolve。
   **期望**：结果只会是 5、6、7，绝不回到 ≤4；若掷出的档位正是 4，卡上写 "You already have this"，且角色的压力值 +1。
2. 打开一艘 spacecraft，在 Crew 面板放入一个**已带 Jumpy** 的船员，点该船员那一行的恐慌按钮（`spacecraft-general.hbs:136`／`vehicle-crew.hbs:32` 那个带 `data-action="CrewPanic"` 的按钮）。
   **期望**：船员的 Jumpy 图标**保留**、船员压力 +1、卡上写 "You already have this"；飞船本身状态不变。
3. 把界面语言切到中文（`lang/cn.json` 生效，且系统本体有中文汉化时状态效果显示为译名）。给一个角色依次加上 Flee（档位 10）与 Spooked（档位 1），然后手动点掉 Spooked。
   **期望**：`game.actors.getName("<角色名>").system.general.panic.lastRoll` 是 **10**。
   **今天**：是 **−1**（译名匹配不上，整段重置）；英文世界则得到 **1**（字典序）。
4. 接上一步，再手动点掉 Flee，只留一个 Keeping Guard。
   **期望**：`system.general.stressresponse.value` 是 **−1**，不是 0（Keeping Guard 的 `tableNumber` 是 0，不是一档压力反应）。

- [ ] **Step 39: MANUAL VERIFICATION 4 —— 1e 恐慌表改名后仍能查到**

1. 在 Foundry 的设置里把系统设置 `alienrpg` 的 `evolved` **关掉**（切回 1e 规则），重载世界。
2. 在 Roll Tables 侧栏里把出厂的 "Panic Table" 改名成任意别的名字（例如 "恐慌表"）——这就是 Babele 汉化层在中文世界里造成的等价状态。
3. 打开模组的绑定重绑菜单，确认 `panic` 键仍绑在这张（已改名的）表上；若显示未绑定，手动绑上。
4. 在 character 角色卡上点恐慌按钮。
   **期望**：正常弹出 1e 恐慌聊天卡。**今天**（本特性设为 off 后重载可复现）：右上角弹出 "No Panic Table"，什么都不发生。

- [ ] **Step 40: 跑模组自检并提交记录**

F12 控制台执行：

```js
await game.modules.get("alien-evolved-automation").api.selftest.runAll();
```

Expected：以下六条全部 `ok: true` —— `stress-panic-math-repair.rollResolve.probe`、`.rollStress.probe`、`.statusLevelRecount.probe`、`.panicTableLookup.probe`、`.tableCeilings`、`.panicTableLookup.live`。

`tableCeilings` 的 detail 应写 `stress: table top 7 >= ceiling 7; panic: table top 20 >= ceiling 12`；四条 `.probe` 的 detail 里应是 `defect=true applied=true type=MIXED`。若任何一条写 `unrecognised ... body`，说明系统源码已经变了：**先停下来重新审计那段代码**，不要把补丁留在原地。若 `tableCeilings` 或 `.live` 报 `unbound`，说明 `registry.resolveAll()` 没在 `ready.registry` 跑到或绑定没配，先修绑定再继续。再执行 `api.patches.status().filter(p => p.id.startsWith("stress-panic-math-repair"))`，确认四行都在、`applied` 均为 `true`。

```bash
git commit --allow-empty -m "test(stress-panic-math-repair): 本机冒烟四组 + 自检六条通过" -m "满级恐慌停在 12；勾了 synthstress 的合成人正常出卡且无 schema 警告，没勾的静默返回；
压力反应逐档上升；飞船船员面板上已带 Jumpy 的船员按恐慌后 Jumpy 保留且压力 +1；
中文世界下移除状态后 panic.lastRoll 重算为 10 而非 -1；1e 世界改名后恐慌按钮仍能出卡。
自检六条全绿，patches.status() 四行 applied 均为 true。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```
