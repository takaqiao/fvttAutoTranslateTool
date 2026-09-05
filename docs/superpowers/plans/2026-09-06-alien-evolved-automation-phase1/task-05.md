> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 5 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 5: K2 DocRegistry —— id 化一切文档查找

仓库根目录：`C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation`。下面所有路径都相对它。

**Files:**
- Create: `scripts/kernel/registry.mjs`
- Create: `templates/registry-rebind.hbs`
- Create: `test/registry-choose.test.mjs`
- Create: `test/registry-foundry.test.mjs`
- Modify: `styles/alien-evolved-automation.css`（追加重绑面板的四条规则）
- Modify: `lang/en.json`、`lang/cn.json`（追加 `AEA.registry.*` 与两个 `AEA.selftest.*` 键）
- Modify: `scripts/main.mjs`，四处，每处都按锚点注释的**逐字文本**定位（不用行号）：
  1. `/* AEA-ANCHOR: imports */` 之后加 `import { registry } ...`，并**幂等地**补 `import { selftest } ...`（已有就不重复加）；
  2. `export const api = {...}` 里把 `registry: null` 换成 `registry,`（**只换自己这一个槽，不增删键、不替换 api 对象本身**）；
  3. `/* AEA-ANCHOR: init */` 之后插 `registry.registerSettings()` + 13 次 `registry.declare(...)` + 2 条 `selftest.register(...)`；
  4. `/* AEA-ANCHOR: ready.registry */` 之后插 `await registry.resolveAll();`。

**Interfaces:**

- Consumes:
  - `scripts/const.mjs :: MID`（值 `"alien-evolved-automation"`）、`SETTING_BINDINGS`（值 `"registryBindings"`）、`SYSTEM_ID`（值 `"alienrpg"`）。这三个是纯字符串常量，测试文件也直接 import 它们。
  - `test/stubs/foundry.mjs :: installFoundryStub(options) -> ctx` / `uninstallFoundryStub()`。这个桩由**另一位属主**实现并附带 `test/stub-fidelity.test.mjs` 逐条守卫，**本任务只读不改，也绝不在自己的测试文件里就地造 `globalThis.game` / `globalThis.foundry` / 私有 Map 顶替 `game.settings`**。本文件依赖的桩行为，全部出自契约 §0.3：
    - `installFoundryStub()` 幂等（第二次调用先隐式卸载再装），`uninstallFoundryStub()` 未安装时是安全空操作；
    - 桩装在 `globalThis` 上的全局包含 `game` / `ui` / `Hooks` / `CONFIG` / `libWrapper` / `logger` / `foundry.{utils, applications}`；
    - `game.settings.register/get/set` 由 `ctx.settings`（Map）真支撑，`get` 读未注册键**抛错**，`set` 返回 Promise，`registerMenu` 记进 `ctx.menus`；
    - `ctx.documents`（Map: uuid → 文档）支撑 `fromUuidSync` 与 `fromUuid`，未命中返回 `null`；
    - `ctx` 字段固定含 `isGM`、`userId`、`i18n`、`settings`、`menus`、`notifications`、`documents`、`world`；这些字段由 `installFoundryStub(options)` 的**同名选项**设定，因此本文件用 `installFoundryStub({ world, i18n, isGM })` 喂夹具，用 `ctx.documents` / `ctx.menus` / `ctx.notifications` 读回观察值。
    - `ctx.menus` 与 `ctx.notifications` 的**条目容器形状**契约没有钉死（Map 或数组都可能），所以本文件用一个 8 行的 `asList()` 归一化后再断言——这是读 ctx，不是补桩。
  - `scripts/kernel/selftest.mjs :: selftest.register({id, label, run})`，`run()` 返回 `{ok:boolean, detail:string}`。**`label` 传的是 i18n 键**（形如 `AEA.selftest.<id>`）而不是已本地化文本：登记发生在 `init`，那时语言包还没加载，`game.i18n.localize()` 只会把键原样回声；由 `selftest.runAll()` 在运行时本地化。`register` 保存 def 原样即可，本任务不传惰性 getter（getter 会让 def 无法被序列化展示）。
  - `scripts/main.mjs`（由骨架任务建立）已含**十个逐字锚点注释**，本任务只用其中三个：`/* AEA-ANCHOR: imports */`（文件顶部 import 区）、`/* AEA-ANCHOR: init */`（在 `Hooks.once("init", ...)` 体内，且**排在两个 `register()` 循环之后**）、`/* AEA-ANCHOR: ready.registry */`（在 `Hooks.once("ready", async () => {...})` 体内，紧跟 `await waitForWorldSettled({ timeoutMs: 10000, pollMs: 100 });`，是 ready 段四个有序子锚点里的**第一个**，其后依次是 `ready.patches` / `ready.rollbus` / `ready.cards`）。
  - `scripts/main.mjs` 的 `export const api = { features: null, patches: null, resolver: null, registry: null, rollBus: null, diceBarrier: null, cards: null, selftest: null }`：八个槽的字面量由骨架任务一次写出，**本任务只把 `registry` 那一个 `null` 换成自己 import 的 `registry`**，其余七个槽（含 `selftest`）不碰。
  - **启动排序闸门的唯一属主是 `main.mjs` 的 `waitForWorldSettled`**，它在 `ready.registry` 子锚点之前。本任务的 `resolveAll()` 里**不得**再等第二遍（两道守卫串起来会把新世界的最坏启动延迟翻倍）。
  - `styles/alien-evolved-automation.css` 已存在且已写进 `module.json` 的 `styles` 数组。
  - `lang/en.json` / `lang/cn.json` 已存在，且各自只有一个顶层键 `"AEA"`（有一条护栏测试断言这一点，所以新键必须并进这个对象里，不能新开顶层键）。

- Produces:
  - `export function pureChooseBinding(candidates, guesses)` —— `candidates = [{uuid, name}]`，`guesses = [String]`，返回 `{uuid, name}` 或 `null`。
  - `export const registry = { declare(key, {kind, docType, guess}), registerSettings(), resolveAll(), table(key), folder(key), journal(key), tableByLegacyName(name), bind(key, uuid), bindings(), unbound() }`（成员表与契约 §4 K2 逐字一致，不多不少）。
  - 世界设置 `alien-evolved-automation.registryBindings`，形状 `{ [key]: {uuid, name, boundAt, boundBy} }`。**`registerSettings()` 只注册这一个世界设置加一个 `registerMenu`**，不给每个键各注册一个键。
  - 设置菜单 `alien-evolved-automation.registryRebind`（`restricted: true`，即只有 GM 能看见）。
  - `main.mjs` 的 `api.registry` 从 `null` 变成本模块的 `registry` 对象（自检与手工验证都从 `game.modules.get(MID).api.registry` 进来）。
  - **必须声明的 13 个键（冻结，一期不增不减）**：`panic`、`stressResponse`、`panicResponse`、`critInjury1e`、`critInjuryEvolved`、`critInjurySynthetic`、`critInjuryXeno`、`shipMinorComponent`、`shipMajorComponent`、`folderAlienTables`、`folderCreatureTables`、`folderMotherTables`、`journalMother`。
  - 测试文件 `test/registry-choose.test.mjs`（15 条，覆盖那 1 个 `pure*`）与 `test/registry-foundry.test.mjs`（22 条，覆盖夹具面、9 个副作用方法与重绑菜单的注册）。
  - 自检条目 `registry-bindings`、`registry-legacy-name`。

- **验收硬指标（会被后续任务依赖，别削）**：
  - `registerSettings()` **不得在调用时刻快照** `declared` 或绑定表。`main.mjs` 的 init 段是先 `registry.registerSettings()`、后 13 次 `registry.declare(...)`；重绑面板的行与下拉必须在**渲染时**才从 `declared` / `bindings()` / 世界文档集合现算。Step 6 有一条专门的用例守这一点。
  - `await registry.resolveAll();` 必须落在 `ready.registry` 子锚点，即**早于** `ready.patches`（`await patches.applyAll();`）、`ready.rollbus`、`ready.cards`。Step 20 有一条静态顺序守卫。

- 跑不了单测的断言 → 替代覆盖（这张表是为了让「有人后来删掉一条自检」立刻显形；每一行左边都不许没有右边）：

  | 断言 | 为什么 vitest 覆盖不了 | 替代覆盖 |
  |---|---|---|
  | 13 个键在真实世界里全部绑上 | 需要系统真的导入冒险包并建出这些文档 | selftest `registry-bindings` + MANUAL VERIFICATION A.3 |
  | 改名后按 uuid 仍能取到表 | 需要真实的 Foundry 文档与 uuid 解析 | MANUAL VERIFICATION B.6 |
  | 怪物卡存的英文表名仍能落到已绑定的表 | 同上，且要真实的 actor 数据 | selftest `registry-legacy-name` + MANUAL VERIFICATION B.8 |
  | `resolveAll()` 真的排在 `patches.applyAll()` 之前 | `main.mjs` 挂真实生命周期钩子，不进 vitest | Step 20 的静态顺序守卫（读文件比下标）+ MANUAL VERIFICATION A.3 |
  | 重绑面板真的渲染、下拉真的按类型过滤、提交真的写库、行数在渲染时才现算 | 契约禁止 jsdom，ApplicationV2 的渲染管线桩不出来 | MANUAL VERIFICATION C.12 / C.14 |
  | `restricted: true` 真的把面板对玩家藏起来 | 需要第二个非 GM 客户端 | MANUAL VERIFICATION C.16 |
  | 自检条目的 label 在面板里显示成本地化文本 | 桩里没有语言包，`localize` 只会回声键名 | MANUAL VERIFICATION A.4 |

---

**一期兑现边界（先读这段，免得把「占位」读成「漏做」）**

13 个键**全部**在一期声明并绑定——绑定本身就是产出：它让 GM 的手动重绑面板、自检、以及后续接管都有现成的数据。但**接管**（把系统的按名查找改接注册表）是别的模块的活，一期只兑现其中一部分。逐条说清楚：

| 键 | 一期有没有运行期消费者 | 说明 |
|---|---|---|
| `critInjury1e` / `critInjuryEvolved` / `critInjurySynthetic` | **有** | 重伤链接管特性会用它们替掉 `actor.mjs:1812-1817`（EV 与 1e 的三段或链）与 `:1830`（合成人）。 |
| `folderCreatureTables` / `folderMotherTables` | **有** | 怪物卡两个下拉的数据源修复要用：`module/helpers/rollTableData.mjs:7` 与 `:24` 在文件夹缺失时直接对 `undefined` 取 `.contents` 抛异常，整张怪物卡打不开。 |
| `panic` / `stressResponse` / `panicResponse` | **没有**（只声明与绑定） | 这三条掷骰路径（`actor.mjs:554` `rollPanic`、`:845` `rollResolve`、`:1067` `rollStress`）**不经过 `yzeRoll`**，一期的掷骰总线看不见它们，接管归二期的恐慌链特性。**验收时「键已绑定、系统行为未变」是预期结果，不是缺陷。** |
| `shipMinorComponent` / `shipMajorComponent` | **没有**（只声明与绑定） | `actor.mjs:1848` / `:1855` 的飞船部件损伤表，归二期的飞船阶段特性。同上，键绑上但行为不变是预期。 |
| `critInjuryXeno` | 绑定在一期，消费在二期 | 它的运行期入口是 `actor.mjs:1839` 的动态名（怪物卡 Roll Crit），归二期的怪物攻击/重伤抽取管线。 |
| `folderAlienTables` | **没有**（只声明与绑定） | 唯一调用点 `module/apps/init.mjs:49` 是系统自己的首次导入判断，一期不接管。 |
| `journalMother` | **没有**（只声明与绑定） | `init.mjs:81` / `:107` 与 `module/alienrpg.mjs:574 / 592 / 613` 的 MU/TH/ER 日志链，归二期。 |
| `tableByLegacyName()` | 一期唯一的调用者就是本任务自己的 `registry-legacy-name` 自检 | 真正的运行期调用者是 `actor.mjs:1839` 那条动态名路径，在二期。方法一期就得存在，因为它是「英文字面量 → 键 → uuid」这条链的**唯一**合法入口，晚做会让二期的调用方各自发明一套。 |

一句话：**一期 K2 的产出是「绑定这层基础设施本身」，不是「所有按名查找都已被替换」。**

---

**背景一：这个任务替换掉的是哪些查找（逐条重新打开源码核对，系统 4.1.13）**

Foundry 里 `game.tables.getName("X")` 按**显示名**找随机表。显示名会变：Babele（一个把文档改写成本地语言的翻译模组）会改它，GM 手动重命名也会改它，中文汉化包导入进来时它本来就是中文的。名字一变，下面每一条都静默失效。契约 §4 K2 把这些点收进 13 个语义键：

`systems/alienrpg/module/documents/actor.mjs`：

| 行 | 源码 | 归属键 |
|---|---|---|
| 554 | `const table = game.tables.getName("Panic Table");`（`rollPanic` 内） | `panic` |
| 845 | `const table = game.tables.getName("Stress Response Table");` | `stressResponse` |
| 1067 | `const table = game.tables.getName("Panic Response Table");` | `panicResponse` |
| 1812 | `atable = game.tables.getName(game.i18n.localize("ALIENRPG.EVCriticalInjuries")) \|\| game.tables.getName("EV - Critical Injuries");` | `critInjuryEvolved` |
| 1815-1817 | `game.i18n.localize("ALIENRPG.CriticalInjuries")` → `"Critical Injuries"` → `"Critical injuries"` 三连或 | `critInjury1e` |
| 1830 | `game.tables.getName("Critical Injuries on Synthetics") \|\| game.tables.getName("critical injuries on synthetics");` | `critInjurySynthetic` |
| 1839 | `atable = game.tables.getName(dataset.atttype);`（creature 分支） | **动态名** → `tableByLegacyName()`，一期能命中的是 `critInjuryXeno` |
| 1848 | `atable = game.tables.getName("Spaceship Minor Component Damage");` | `shipMinorComponent` |
| 1855 | `atable = game.tables.getName("Spaceship Major Component Damage");` | `shipMajorComponent` |
| 2497 | `game.tables.contents.find((b) => b.name === targetTable);`（`targetTable = dataset.atttype`） | **动态名**，每怪物一张攻击表，二期的 creature-attack 特性接手 |

`systems/alienrpg/module/helpers/rollTableData.mjs`（喂怪物卡两个下拉框；文件夹缺失时 `folder.contents` 直接抛异常，怪物卡打不开）：

| 行 | 源码 | 归属键 |
|---|---|---|
| 7 | `game.folders.contents.find((x) => x.name === "Alien Creature Tables")` —— 填 `system.rTables`（攻击表下拉） | `folderCreatureTables` |
| 24 | `game.folders.contents.find((x) => x.name === "Alien Mother Tables")`，再 `.filter(x => x.name.startsWith("Critical Injuries"))` —— 填 `system.cTables`（重伤表下拉） | `folderMotherTables` |

`systems/alienrpg/module/apps/init.mjs`：

| 行 | 源码 | 归属键 |
|---|---|---|
| 49 | `!game.folders.getName("Alien Tables")` | `folderAlienTables` |
| 81 / 107 | `game.journal.getName(welcomeJournalEntry).show()`，`welcomeJournalEntry = "MU/TH/ER Instructions."`（`init.mjs:14`） | `journalMother` |

同一篇日志另有三处用法：`module/apps/migratefolders.js:120`，以及 `module/alienrpg.mjs:574 / 592 / 613`（`showReleaseNotes()` 里 `releaseNoteName = "MU/TH/ER Instructions."`，第 592 行直接取 `.id` —— 日志被改名就抛 TypeError）。

**`critInjuryXeno` 与那个动态名的关系（13 个键里最需要解释的一个）。** 创造物卡上的 Roll Crit 按钮把 `system.cTables` 的**字面表名**塞进 `data-atttype`（`templates/actor/creature-general.hbs:19` 的 `<select name='system.cTables' data-atttype='{{system.cTables}}'>` 与 `:22` 的按钮），`rollCrit` 的 creature 分支就拿它去 `getName`。而 `system.cTables` 的候选来自 Mother 文件夹里所有 `Critical Injuries*` 开头的表（`rollTableData.mjs:24-30`），实际只有一张有意义：`Critical Injuries on Xenomorphs`。所以 `tableByLegacyName("Critical Injuries on Xenomorphs")` → 键 `critInjuryXeno` → 绑定的 uuid。这条路径成立的原因是：`system.cTables` 存的是 **actor 数据里的英文字面量**，Babele 不会改它；而表本身可能已被改名——正是必须按 uuid 取的那一半。

**`tableByLegacyName()` 返回 `null` 的语义定死为「本模组不介入，系统按原样走」。** 调用方必须原样放行：**不得**自己回退去调 `getName()` 或 `contents.find()`，**不得**吞掉系统自己的 `ui.notifications.warn`，**不得**替系统猜表。「永不按显示名查找」这条纪律约束的是本模组新写的代码，不是尚未被包裹的系统原代码。逐怪物的攻击表（`actor.mjs:1839` 带怪物专名时、与 `:2497`）一期不接管，二期由 creature-attack 特性以 actor flag 存 uuid 解决，不为此扩 registry 键。

**不在替换范围内**：`module/actor/old-*.js` 与 `module/actor/old-rollTableData.js` 是 1e 遗留层的死代码；`actor.mjs:715 / 1592 / 1596` 的 `actor.items.getName(...)` 找的是系统自己生成的临时物品，不是世界内容。

**一期已知边界（写进文档而不是假装能做到）**：一个键只绑一个文档。若某个世界同时存在 `Critical Injuries on Xenomorphs` 与 `EV - Critical Injuries on Xenomorphs`，首次绑定按 guess 顺序取前者，两个名字都会映射到这同一个绑定；GM 用重绑面板改成哪张，`tableByLegacyName` 就返回哪张。逐怪物的变体精度属二期。

**背景二：启动排序的闸门在 main.mjs，不在这里**

要绑的文档不是世界自带的，是系统在 `ready` 阶段现建的，所以「什么时候解析」是个真问题：

1. **系统的冒险导入。** `module/apps/init.mjs:49` 在 `ready` 里判断 `!game.settings.get("alienrpg", "imported") && game.user.isGM && !game.folders.getName("Alien Tables")`，成立就跑 `FirstTimeSetup()`——那才是把 Panic Table、Alien Tables 文件夹、MU/TH/ER 日志建到世界里的一步，完成后 `:78` 把世界设置 `imported` 置 `true`（该设置在 `init.mjs:24` 注册，world scope、Boolean、默认 `false`）。**关键**：`:49` 那个分支调 `FirstTimeSetup()` 时**没有 await**，所以钩子顺序再怎么排都救不了，只能轮询它的完成标志。
2. **文件夹迁移。** `module/apps/migratefolders.js:12` 把世界设置 `alienrpg.ARPGSemaphore`（`module/helpers/settings.mjs:209` 注册，world scope、String、默认 `""`）设成 `"busy"`，`:118` 清回 `""`。所以「世界已安定」= `imported === true` **且** `ARPGSemaphore !== "busy"`。
3. **Babele 完成。** 若 Babele 模组激活，它在 `ready` 期间改写文档名，`game.babele.initialized` 为 `true` 表示已完成。

这三件事的等待**已经由 `main.mjs` 的 `await waitForWorldSettled({timeoutMs: 10000, pollMs: 100})` 统一承担**，它排在 `/* AEA-ANCHOR: ready.registry */` 之前。所以 `resolveAll()` 里**只做一次廉价的诊断断言**：读一次那两个系统设置，若世界仍未安定就 `console.warn` 一条，然后照常解析。**绝不再等第二遍**——两道守卫串起来，新世界首次进入的最坏延迟会翻倍，而且两套判据一旦漂移就会产生只在新世界复现的间歇故障。

那么文档来晚了怎么办？答案是重绑面板上的「自动识别」按钮：它重跑 `resolveAll()`，把这时已经存在的文档补绑上。这是设计好的恢复路径，不是遗漏。

---

- [ ] **Step 1: Write the failing test**

创建 `test/registry-choose.test.mjs`：

```js
import { describe, expect, it } from "vitest";
import { pureChooseBinding } from "../scripts/kernel/registry.mjs";

describe("pureChooseBinding", () => {
  it("returns the single exact match", () => {
    const candidates = [
      { uuid: "RollTable.a", name: "Panic Table" },
      { uuid: "RollTable.b", name: "Stress Response Table" },
    ];
    expect(pureChooseBinding(candidates, ["Panic Table"])).toEqual({
      uuid: "RollTable.a",
      name: "Panic Table",
    });
  });

  it("exact beats case-insensitive", () => {
    const candidates = [
      { uuid: "RollTable.lower", name: "panic table" },
      { uuid: "RollTable.exact", name: "Panic Table" },
    ];
    expect(pureChooseBinding(candidates, ["Panic Table"]).uuid).toBe("RollTable.exact");
  });

  it("case-insensitive is used when no exact match exists", () => {
    const candidates = [{ uuid: "RollTable.lower", name: "critical injuries on synthetics" }];
    expect(pureChooseBinding(candidates, ["Critical Injuries on Synthetics"]).uuid).toBe("RollTable.lower");
  });

  it("case-insensitive beats trimmed", () => {
    const candidates = [
      { uuid: "RollTable.pad", name: "  Panic Table  " },
      { uuid: "RollTable.case", name: "PANIC TABLE" },
    ];
    expect(pureChooseBinding(candidates, ["Panic Table"]).uuid).toBe("RollTable.case");
  });

  it("trimmed is the last resort", () => {
    const candidates = [{ uuid: "RollTable.pad", name: " Panic Table " }];
    expect(pureChooseBinding(candidates, ["Panic Table"]).uuid).toBe("RollTable.pad");
  });

  it("returns null rather than guessing when two candidates tie", () => {
    const candidates = [
      { uuid: "RollTable.a", name: "Panic Table" },
      { uuid: "RollTable.b", name: "Panic Table" },
    ];
    expect(pureChooseBinding(candidates, ["Panic Table"])).toBeNull();
  });

  it("does not fall through to a weaker tier after an ambiguous stronger tier", () => {
    const candidates = [
      { uuid: "RollTable.a", name: "Panic Table" },
      { uuid: "RollTable.b", name: "Panic Table" },
      { uuid: "RollTable.c", name: " panic table " },
    ];
    expect(pureChooseBinding(candidates, ["Panic Table"])).toBeNull();
  });

  it("honours guess order inside one tier", () => {
    const candidates = [
      { uuid: "RollTable.ev", name: "EV - Critical Injuries" },
      { uuid: "RollTable.classic", name: "Critical Injuries" },
    ];
    expect(
      pureChooseBinding(candidates, ["EV - Critical Injuries", "Critical Injuries"]).uuid,
    ).toBe("RollTable.ev");
    expect(
      pureChooseBinding(candidates, ["Critical Injuries", "EV - Critical Injuries"]).uuid,
    ).toBe("RollTable.classic");
  });

  it("lets a stronger tier on a later guess beat a weaker tier on an earlier guess", () => {
    const candidates = [
      { uuid: "RollTable.ev", name: "EV - Critical Injuries" },
      { uuid: "RollTable.classic", name: "Critical Injuries" },
    ];
    // guess[0] would only match "EV - Critical Injuries" case-insensitively,
    // guess[1] matches "Critical Injuries" exactly — the exact tier wins.
    expect(
      pureChooseBinding(candidates, ["ev - critical injuries", "Critical Injuries"]).uuid,
    ).toBe("RollTable.classic");
  });

  it("keeps the panic and panicResponse keys apart", () => {
    // The two shipped tables differ by one word; a substring or prefix matcher would
    // bind both keys to whichever table it saw first.
    const candidates = [
      { uuid: "RollTable.panic", name: "Panic Table" },
      { uuid: "RollTable.presp", name: "Panic Response Table" },
      { uuid: "RollTable.sresp", name: "Stress Response Table" },
    ];
    expect(pureChooseBinding(candidates, ["Panic Table"]).uuid).toBe("RollTable.panic");
    expect(pureChooseBinding(candidates, ["Panic Response Table"]).uuid).toBe("RollTable.presp");
    expect(pureChooseBinding(candidates, ["Stress Response Table"]).uuid).toBe("RollTable.sresp");
  });

  it("keeps the four crit-injury keys apart in one world", () => {
    const candidates = [
      { uuid: "RollTable.c1e", name: "Critical Injuries" },
      { uuid: "RollTable.cev", name: "EV - Critical Injuries" },
      { uuid: "RollTable.csyn", name: "Critical Injuries on Synthetics" },
      { uuid: "RollTable.cxeno", name: "Critical Injuries on Xenomorphs" },
    ];
    expect(pureChooseBinding(candidates, ["Critical Injuries", "Critical injuries"]).uuid).toBe("RollTable.c1e");
    expect(pureChooseBinding(candidates, ["EV - Critical Injuries"]).uuid).toBe("RollTable.cev");
    expect(
      pureChooseBinding(candidates, ["Critical Injuries on Synthetics", "critical injuries on synthetics"]).uuid,
    ).toBe("RollTable.csyn");
    expect(
      pureChooseBinding(candidates, ["Critical Injuries on Xenomorphs", "EV - Critical Injuries on Xenomorphs"]).uuid,
    ).toBe("RollTable.cxeno");
  });

  it("treats the same uuid listed twice as one candidate, not as ambiguity", () => {
    const candidates = [
      { uuid: "RollTable.a", name: "Panic Table" },
      { uuid: "RollTable.a", name: "Panic Table" },
    ];
    expect(pureChooseBinding(candidates, ["Panic Table"]).uuid).toBe("RollTable.a");
  });

  it("skips malformed candidates and guesses", () => {
    const candidates = [
      null,
      { uuid: "", name: "Panic Table" },
      { uuid: "RollTable.a", name: 7 },
      { uuid: "RollTable.good", name: "Panic Table" },
    ];
    expect(pureChooseBinding(candidates, [null, "", 42, "Panic Table"]).uuid).toBe("RollTable.good");
  });

  it("returns null for empty or non-array input", () => {
    expect(pureChooseBinding([], ["Panic Table"])).toBeNull();
    expect(pureChooseBinding([{ uuid: "RollTable.a", name: "Panic Table" }], [])).toBeNull();
    expect(pureChooseBinding(null, null)).toBeNull();
  });

  it("touches no Foundry global (the contract's layering rule)", () => {
    expect(globalThis.game).toBeUndefined();
    expect(() => pureChooseBinding([{ uuid: "RollTable.a", name: "X" }], ["X"])).not.toThrow();
  });
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `npx vitest run test/registry-choose.test.mjs`

Expected: FAIL —— `Error: Failed to load url ../scripts/kernel/registry.mjs`，整个文件 0 条用例执行（文件还不存在）。

- [ ] **Step 3: 实现 pureChooseBinding（只写纯函数）**

创建 `scripts/kernel/registry.mjs`：

```js
/**
 * K2 · DocRegistry — every RollTable / Folder / JournalEntry lookup resolved by id.
 *
 * The system looks everything up by display name (module/documents/actor.mjs:554, 845,
 * 1067, 1812-1817, 1830, 1839, 1848, 1855, 2497; module/helpers/rollTableData.mjs:7, 24;
 * module/apps/init.mjs:49, 81, 107; module/apps/migratefolders.js:120;
 * module/alienrpg.mjs:574, 592, 613). A Babele translation layer or one manual rename
 * silently kills all of it. Names are used exactly once — to make the first guess.
 */

/**
 * Name matchers in strength order. A stronger tier always wins over a weaker one.
 * @type {Array<(candidateName: string, guess: string) => boolean>}
 */
const MATCHERS = [
  (a, b) => a === b,
  (a, b) => a.toLowerCase() === b.toLowerCase(),
  (a, b) => a.trim().toLowerCase() === b.trim().toLowerCase(),
];

/**
 * Choose one document for a key from a list of candidates and a list of name guesses.
 * Pure: never touches game / ui / canvas / CONFIG / Hooks / foundry / ChatMessage / Roll,
 * and both parameters are plain arrays of plain objects — never a Foundry Collection.
 *
 * Exact match wins over case-insensitive, which wins over trimmed. Inside one tier the
 * guesses are tried in order. A tie inside the winning tier returns null — the registry
 * would rather report the key as unbound and let the GM pick than bind the wrong table.
 *
 * @param {Array<{uuid: string, name: string}>} candidates
 * @param {string[]} guesses
 * @returns {{uuid: string, name: string}|null}
 */
export function pureChooseBinding(candidates, guesses) {
  if (!Array.isArray(candidates) || !Array.isArray(guesses)) return null;
  const pool = candidates.filter(
    (c) => c && typeof c.uuid === "string" && c.uuid.length > 0 && typeof c.name === "string",
  );
  if (pool.length === 0) return null;

  for (const matches of MATCHERS) {
    for (const guess of guesses) {
      if (typeof guess !== "string" || guess.length === 0) continue;
      const hits = [];
      for (const c of pool) {
        if (!matches(c.name, guess)) continue;
        if (!hits.some((h) => h.uuid === c.uuid)) hits.push(c);
      }
      if (hits.length === 1) return { uuid: hits[0].uuid, name: hits[0].name };
      if (hits.length > 1) return null; // ambiguous — never guess
    }
  }
  return null;
}
```

- [ ] **Step 4: Run it and watch it pass**

Run: `npx vitest run test/registry-choose.test.mjs`
Expected: PASS —— `Tests 15 passed (15)`。

- [ ] **Step 5: Commit**

```bash
git add scripts/kernel/registry.mjs test/registry-choose.test.mjs && git commit -F - <<'EOF'
feat(kernel): K2 pureChooseBinding —— 首次绑定的名字猜测内核

三档匹配强度：精确 > 忽略大小写 > 去首尾空白；强档永远压过弱档，
同档内按 guess 顺序取先命中者。任一档内出现两个候选即判歧义返回 null，
宁可让该键落到未绑定由 GM 手选，也绝不猜。

两条防混淆用例落地：Panic Table 与 Panic Response Table 不得互串；
一期四张重伤表（1e / EV / 合成人 / 异形）在同一个世界里各归各的键。

名字只在首次绑定时用这一次，之后一律走 uuid。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

- [ ] **Step 6: 为 registry 对象、世界安定断言与动态名映射写失败测试**

创建 `test/registry-foundry.test.mjs`。全局**全部**由公共桩 `installFoundryStub(options)` 提供，本文件**从不给 `globalThis` 赋值**：夹具世界经 `installFoundryStub({ world, i18n, isGM })` 喂进去，观察值经 `ctx.documents` / `ctx.menus` / `ctx.notifications` 读回来。第一条用例专门守住这层夹具面——它红了就说明公共桩与契约 §0.3 有出入，按「其余任务只读不改」的规矩报回桩的属主，**不要在本文件里就地补桩**。

```js
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";
import { MID, SETTING_BINDINGS, SYSTEM_ID } from "../scripts/const.mjs";

/** The stub context returned by installFoundryStub(). */
let ctx;
/** The console.warn spy — the registry reports every diagnostic there. */
let warnSpy;

/**
 * Contract §0.3 pins WHAT lands in ctx.menus / ctx.notifications but not the container
 * shape (a Map and an array are both legal readings). Normalise before asserting.
 */
const asList = (x) => {
  if (x instanceof Map) return [...x.values()];
  if (Array.isArray(x)) return x;
  return Object.values(x ?? {});
};

// Plain fixture documents. `documentName` is what Foundry calls the document class;
// a Folder's `type` is the document type it is allowed to hold.
const table = (uuid, name) => ({ uuid, name, documentName: "RollTable" });
const folder = (uuid, name, type = "RollTable") => ({ uuid, name, documentName: "Folder", type });
const journal = (uuid, name) => ({ uuid, name, documentName: "JournalEntry" });

/**
 * Install the shared stub with this test's fixture world, then register the two SYSTEM
 * settings the world-settled assertion reads — through the stub's own game.settings API.
 * @param {{docs?: object[], imported?: boolean, semaphore?: string, isGM?: boolean, i18n?: Record<string,string>}} o
 */
async function seedWorld({ docs = [], imported = true, semaphore = "", isGM = true, i18n = {} } = {}) {
  const world = {};
  for (const doc of docs) world[doc.uuid] = doc;
  ctx = installFoundryStub({ world, i18n, isGM });
  // ctx.documents is the uuid map that backs fromUuidSync (contract §0.3); seed it too so
  // uuid resolution and the world collections cannot drift apart inside one test.
  for (const [uuid, doc] of Object.entries(world)) ctx.documents.set(uuid, doc);

  const { settings } = globalThis.game;
  settings.register(SYSTEM_ID, "imported", { scope: "world", config: false, type: Boolean, default: false });
  settings.register(SYSTEM_ID, "ARPGSemaphore", { scope: "world", config: false, type: String, default: "" });
  await settings.set(SYSTEM_ID, "imported", imported);
  await settings.set(SYSTEM_ID, "ARPGSemaphore", semaphore);
}

/** Fresh module state per test: `declared` and the binding cache are module-level. */
async function loadRegistry() {
  vi.resetModules();
  return import("../scripts/kernel/registry.mjs");
}

/** The current value of the module's world setting, read through the stub. */
const storedBindings = () => globalThis.game.settings.get(MID, SETTING_BINDINGS);

beforeEach(() => {
  warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  uninstallFoundryStub();
  vi.restoreAllMocks();
});

describe("shared stub fixture surface", () => {
  it("provides the world collections, uuid resolution, i18n, user and ApplicationV2", async () => {
    // If this one fails, the shared stub does not match contract §0.3 — report it to the
    // stub's owner. Do NOT patch globals in this file; the contract makes the stub read-only
    // for every task but its owner.
    await seedWorld({ docs: [table("RollTable.panic", "Panic Table")], i18n: { "AEA.probe": "ok" } });
    expect(globalThis.game.tables.contents.map((d) => d.uuid)).toContain("RollTable.panic");
    expect(globalThis.foundry.utils.fromUuidSync("RollTable.panic").name).toBe("Panic Table");
    expect(globalThis.game.i18n.localize("AEA.probe")).toBe("ok");
    expect(globalThis.game.user.isGM).toBe(true);
    expect(typeof globalThis.foundry.applications.api.ApplicationV2).toBe("function");
    expect(typeof globalThis.foundry.applications.api.HandlebarsApplicationMixin).toBe("function");
  });
});

describe("registry.registerSettings", () => {
  it("registers the world binding setting with an empty default", async () => {
    await seedWorld();
    const { registry } = await loadRegistry();
    registry.registerSettings();
    expect(storedBindings()).toEqual({});
  });

  it("is idempotent — a second call does not wipe existing bindings", async () => {
    await seedWorld();
    const { registry } = await loadRegistry();
    registry.registerSettings();
    await globalThis.game.settings.set(MID, SETTING_BINDINGS, { panic: { uuid: "RollTable.x" } });
    registry.registerSettings();
    expect(storedBindings().panic.uuid).toBe("RollTable.x");
  });

  it("does not snapshot the key set — a key declared afterwards still binds", async () => {
    // main.mjs calls registerSettings() first and declares the 13 keys after it. A
    // registerSettings() that froze `declared` (or precomputed the panel's choices) would
    // leave every key invisible to unbound() and to the rebind panel.
    await seedWorld({ docs: [table("RollTable.panic", "Panic Table")] });
    const { registry } = await loadRegistry();
    registry.registerSettings();
    registry.declare("panic", { kind: "table", guess: ["Panic Table"] });
    expect(registry.unbound()).toEqual(["panic"]);
    await registry.resolveAll();
    expect(registry.table("panic").uuid).toBe("RollTable.panic");
    expect(registry.bindings().panic.name).toBe("Panic Table");
  });
});

describe("registry.declare", () => {
  it("rejects an unknown kind and an empty key", async () => {
    await seedWorld();
    const { registry } = await loadRegistry();
    expect(() => registry.declare("panic", { kind: "compendium", guess: [] })).toThrow(/kind/);
    expect(() => registry.declare("", { kind: "table", guess: [] })).toThrow(/key/);
  });

  it("touches no setting and no menu, so init may declare in any order", async () => {
    await seedWorld();
    const { registry } = await loadRegistry();
    registry.declare("panic", { kind: "table", guess: ["Panic Table"] });
    expect(asList(ctx.menus)).toHaveLength(0);
    expect(registry.bindings()).toEqual({}); // reads through the missing setting without throwing
  });
});

describe("registry.resolveAll world-settled assertion", () => {
  it("says nothing when the system import has already finished", async () => {
    await seedWorld({ docs: [table("RollTable.panic", "Panic Table")], imported: true, semaphore: "" });
    const { registry } = await loadRegistry();
    registry.registerSettings();
    registry.declare("panic", { kind: "table", guess: ["Panic Table"] });

    await registry.resolveAll();

    expect(warnSpy.mock.calls.some((c) => String(c[0]).includes("not settled"))).toBe(false);
    expect(registry.table("panic").uuid).toBe("RollTable.panic");
  });

  it("warns once and still resolves when the world is not settled", async () => {
    // main.mjs owns the actual wait (waitForWorldSettled before the ready.registry anchor);
    // getting here unsettled means that gate timed out, which the GM must be able to see.
    await seedWorld({ docs: [table("RollTable.late", "Panic Table")], imported: false, semaphore: "busy" });
    const { registry } = await loadRegistry();
    registry.registerSettings();
    registry.declare("panic", { kind: "table", guess: ["Panic Table"] });

    await registry.resolveAll();

    const notSettled = warnSpy.mock.calls.filter((c) => String(c[0]).includes("not settled"));
    expect(notSettled).toHaveLength(1);
    expect(String(notSettled[0][0])).toContain("ARPGSemaphore");
    expect(registry.table("panic").uuid).toBe("RollTable.late"); // resolved anyway
  });
});

describe("registry.resolveAll binding", () => {
  it("binds tables, folders and journals and reports what is still unbound", async () => {
    await seedWorld({
      docs: [
        table("RollTable.panic", "Panic Table"),
        folder("Folder.alien", "Alien Tables"),
        journal("JournalEntry.mother", "MU/TH/ER Instructions."),
      ],
    });
    const { registry } = await loadRegistry();
    registry.registerSettings();
    registry.declare("panic", { kind: "table", guess: ["Panic Table"] });
    registry.declare("folderAlienTables", { kind: "folder", docType: "RollTable", guess: ["Alien Tables"] });
    registry.declare("journalMother", { kind: "journal", guess: ["MU/TH/ER Instructions."] });
    registry.declare("critInjuryEvolved", { kind: "table", guess: ["EV - Critical Injuries"] });

    const result = await registry.resolveAll();

    expect(registry.table("panic").uuid).toBe("RollTable.panic");
    expect(registry.folder("folderAlienTables").uuid).toBe("Folder.alien");
    expect(registry.journal("journalMother").uuid).toBe("JournalEntry.mother");
    expect(registry.table("critInjuryEvolved")).toBeNull();
    expect(registry.unbound()).toEqual(["critInjuryEvolved"]);
    expect(result.unbound).toEqual(["critInjuryEvolved"]);
    // exactly ONE toast, not one per unbound key
    expect(asList(ctx.notifications)).toHaveLength(1);
    const listed = warnSpy.mock.calls.find((c) => String(c[0]).includes("unbound key(s)"));
    expect(listed?.[1]).toEqual(["critInjuryEvolved"]);
  });

  it("filters folder candidates by docType so a same-named journal folder is not ambiguous", async () => {
    await seedWorld({
      docs: [folder("Folder.tables", "Alien Tables", "RollTable"), folder("Folder.journals", "Alien Tables", "JournalEntry")],
    });
    const { registry } = await loadRegistry();
    registry.registerSettings();
    registry.declare("folderAlienTables", { kind: "folder", docType: "RollTable", guess: ["Alien Tables"] });

    await registry.resolveAll();
    expect(registry.folder("folderAlienTables").uuid).toBe("Folder.tables");
  });

  it("expands an ALIENRPG.* guess through game.i18n at resolve time", async () => {
    // declare() runs at init where translations are not loaded, so a localized guess is
    // declared as its key and expanded here — mirroring actor.mjs:1815.
    await seedWorld({ docs: [table("RollTable.zh", "重伤")], i18n: { "ALIENRPG.CriticalInjuries": "重伤" } });
    const { registry } = await loadRegistry();
    registry.registerSettings();
    registry.declare("critInjury1e", {
      kind: "table",
      guess: ["Critical Injuries", "ALIENRPG.CriticalInjuries"],
    });

    await registry.resolveAll();
    expect(registry.table("critInjury1e").uuid).toBe("RollTable.zh");
  });

  it("re-resolves a key whose stored uuid no longer resolves", async () => {
    await seedWorld({ docs: [table("RollTable.fresh", "Panic Table")] });
    const { registry } = await loadRegistry();
    registry.registerSettings();
    registry.declare("panic", { kind: "table", guess: ["Panic Table"] });
    await globalThis.game.settings.set(MID, SETTING_BINDINGS, {
      panic: { uuid: "RollTable.deleted", name: "Panic Table", boundAt: 1, boundBy: "gm1" },
    });

    await registry.resolveAll();
    expect(registry.bindings().panic.uuid).toBe("RollTable.fresh");
  });

  it("keeps a valid stored uuid even when the document has since been renamed", async () => {
    await seedWorld({ docs: [table("RollTable.translated", "恐慌表")] });
    const { registry } = await loadRegistry();
    registry.registerSettings();
    registry.declare("panic", { kind: "table", guess: ["Panic Table"] });
    await globalThis.game.settings.set(MID, SETTING_BINDINGS, {
      panic: { uuid: "RollTable.translated", name: "Panic Table", boundAt: 1, boundBy: "gm1" },
    });

    await registry.resolveAll();
    expect(registry.table("panic").uuid).toBe("RollTable.translated");
    expect(registry.unbound()).toEqual([]);
  });

  it("does not write the world setting from a non-GM client", async () => {
    await seedWorld({ docs: [table("RollTable.panic", "Panic Table")], isGM: false });
    const { registry } = await loadRegistry();
    registry.registerSettings();
    registry.declare("panic", { kind: "table", guess: ["Panic Table"] });

    await registry.resolveAll();
    expect(storedBindings()).toEqual({});
    expect(registry.table("panic").uuid).toBe("RollTable.panic"); // in-memory only
  });
});

describe("registry accessors", () => {
  it("refuses to hand a table back through folder() or journal()", async () => {
    await seedWorld({ docs: [table("RollTable.panic", "Panic Table")] });
    const { registry } = await loadRegistry();
    registry.registerSettings();
    registry.declare("panic", { kind: "table", guess: ["Panic Table"] });
    await registry.resolveAll();

    expect(registry.folder("panic")).toBeNull();
    expect(registry.journal("panic")).toBeNull();
    expect(registry.table("neverDeclared")).toBeNull();
    expect(warnSpy.mock.calls.some((c) => String(c[0]).includes("never declared"))).toBe(true);
  });

  it("bind() stores a uuid and rejects the wrong document type", async () => {
    await seedWorld({ docs: [table("RollTable.chosen", "恐慌表"), folder("Folder.alien", "Alien Tables")] });
    const { registry } = await loadRegistry();
    registry.registerSettings();
    registry.declare("panic", { kind: "table", guess: ["Panic Table"] });
    await registry.resolveAll();
    expect(registry.unbound()).toEqual(["panic"]);

    const entry = await registry.bind("panic", "RollTable.chosen");
    expect(entry.uuid).toBe("RollTable.chosen");
    expect(entry.name).toBe("恐慌表");
    expect(entry.boundBy).toBe(globalThis.game.user.id);
    expect(registry.table("panic").uuid).toBe("RollTable.chosen");
    expect(storedBindings().panic.uuid).toBe("RollTable.chosen");

    await expect(registry.bind("panic", "Folder.alien")).rejects.toThrow(/Folder/);
    await expect(registry.bind("panic", "RollTable.missing")).rejects.toThrow(/resolve/);
    await expect(registry.bind("neverDeclared", "RollTable.chosen")).rejects.toThrow(/declared/);
  });

  it("bindings() hands back a copy, not the live store", async () => {
    await seedWorld({ docs: [table("RollTable.panic", "Panic Table")] });
    const { registry } = await loadRegistry();
    registry.registerSettings();
    registry.declare("panic", { kind: "table", guess: ["Panic Table"] });
    await registry.resolveAll();

    const snapshot = registry.bindings();
    snapshot.panic.uuid = "RollTable.tampered";
    expect(registry.table("panic").uuid).toBe("RollTable.panic");
  });

  it("bindings() returns {} instead of throwing before registerSettings ran", async () => {
    await seedWorld();
    const { registry } = await loadRegistry();
    expect(registry.bindings()).toEqual({});
  });
});

describe("registry.tableByLegacyName", () => {
  it("maps a creature's stored cTables name onto the key that owns it", async () => {
    // actor.mjs:1839 does game.tables.getName(dataset.atttype); dataset.atttype is the
    // English literal held in the creature's system.cTables, which Babele never rewrites.
    await seedWorld({ docs: [table("RollTable.xeno", "异形重伤表")] }); // already translated
    const { registry } = await loadRegistry();
    registry.registerSettings();
    registry.declare("critInjuryXeno", {
      kind: "table",
      guess: ["Critical Injuries on Xenomorphs", "EV - Critical Injuries on Xenomorphs"],
    });
    await registry.bind("critInjuryXeno", "RollTable.xeno");

    expect(registry.tableByLegacyName("Critical Injuries on Xenomorphs").uuid).toBe("RollTable.xeno");
    expect(registry.tableByLegacyName("critical injuries on xenomorphs").uuid).toBe("RollTable.xeno");
    expect(registry.tableByLegacyName("EV - Critical Injuries on Xenomorphs").uuid).toBe("RollTable.xeno");
  });

  it("returns null for 'None', for an unmapped creature table and for junk", async () => {
    await seedWorld();
    const { registry } = await loadRegistry();
    registry.registerSettings();
    registry.declare("critInjuryXeno", { kind: "table", guess: ["Critical Injuries on Xenomorphs"] });
    registry.declare("folderMotherTables", { kind: "folder", docType: "RollTable", guess: ["Alien Mother Tables"] });

    expect(registry.tableByLegacyName("None")).toBeNull();
    expect(registry.tableByLegacyName("Drone Attacks")).toBeNull(); // phase 2 owns per-creature tables
    expect(registry.tableByLegacyName("Alien Mother Tables")).toBeNull(); // folder key, not a table
    expect(registry.tableByLegacyName("")).toBeNull();
    expect(registry.tableByLegacyName(null)).toBeNull();
    expect(warnSpy.mock.calls.some((c) => String(c[0]).includes("maps to no declared key"))).toBe(true);
  });
});
```

- [ ] **Step 7: Run it and watch it fail**

Run: `npx vitest run test/registry-foundry.test.mjs`

Expected: FAIL —— 20 条用例里 19 条红（第一条 `shared stub fixture surface` 应当**绿**，它只碰桩不碰 registry；若它也红，先解决桩的问题再往下走）。红的那些绝大多数报 `TypeError: Cannot read properties of undefined (reading 'registerSettings')`（`registry.mjs` 目前只导出 `pureChooseBinding`，`registry` 是 `undefined`）；`rejects an unknown kind and an empty key` 那条报 `AssertionError: expected error to match /kind/` —— 它抛的其实是同一个 `TypeError`。

- [ ] **Step 8: 实现 registry 对象、世界安定断言与动态名映射**

在 `scripts/kernel/registry.mjs` 顶部（文件注释块之后、`MATCHERS` 之前）插入 import：

```js
import { MID, SETTING_BINDINGS, SYSTEM_ID } from "../const.mjs";
```

在文件末尾（`pureChooseBinding` 之后）追加：

```js
// ---------------------------------------------------------------------------
// Side-effect layer
// ---------------------------------------------------------------------------

const KINDS = new Set(["table", "folder", "journal"]);
const DOCUMENT_NAME = { table: "RollTable", folder: "Folder", journal: "JournalEntry" };

/** key -> {key, kind, docType, guess} */
const declared = new Map();
/** In-memory mirror of the world setting; null until first read. */
let cache = null;
let settingsReady = false;

function syncUuid(uuid) {
  if (typeof uuid !== "string" || !uuid) return null;
  const fn = globalThis.foundry?.utils?.fromUuidSync ?? globalThis.fromUuidSync;
  if (typeof fn !== "function") return null;
  try {
    return fn(uuid) ?? null;
  } catch {
    return null;
  }
}

function readBindings() {
  if (cache) return cache;
  try {
    cache = globalThis.game?.settings?.get(MID, SETTING_BINDINGS) ?? {};
  } catch {
    cache = {}; // the setting is not registered yet — an empty table is the honest answer
  }
  return cache;
}

/**
 * declare() runs during `init`, where translations are not loaded yet, so a guess that has
 * to be localized is declared as its i18n KEY and expanded here, at resolve time. Mirrors
 * the system's own dual lookup at module/documents/actor.mjs:1812 and :1815.
 * @param {string[]} guess
 * @returns {string[]}
 */
function expandGuesses(guess) {
  const out = [];
  for (const g of guess) {
    if (typeof g !== "string" || !g) continue;
    if (g.startsWith("ALIENRPG.")) {
      const localized = globalThis.game?.i18n?.localize?.(g);
      // A missing translation makes localize() echo the key back — that is not a name.
      if (typeof localized === "string" && localized && localized !== g) out.push(localized);
      continue;
    }
    out.push(g);
  }
  return out;
}

/**
 * Gather {uuid, name} pairs for one declaration. Folders are filtered by `docType`
 * (a Folder's `type` is the document type it holds), so a JournalEntry folder named
 * "Alien Tables" cannot make the RollTable folder key ambiguous.
 * @param {{kind: string, docType: string|null}} def
 * @returns {Array<{uuid: string, name: string}>}
 */
function collectCandidates(def) {
  const g = globalThis.game;
  const source =
    def.kind === "table" ? g?.tables?.contents : def.kind === "journal" ? g?.journal?.contents : g?.folders?.contents;
  if (!Array.isArray(source)) return [];
  return source
    .filter((doc) => !def.docType || doc?.type === def.docType)
    .map((doc) => ({ uuid: doc?.uuid, name: doc?.name }));
}

function typedGet(key, kind) {
  const def = declared.get(key);
  if (!def) {
    console.warn(`${MID} | registry: key "${key}" was never declared`);
    return null;
  }
  if (def.kind !== kind) {
    console.warn(`${MID} | registry: key "${key}" is a ${def.kind}, not a ${kind}`);
    return null;
  }
  return syncUuid(readBindings()[key]?.uuid);
}

/**
 * Diagnostic ONLY — it never waits. The boot-order gate is owned by main.mjs, whose ready
 * handler awaits waitForWorldSettled() before the ready.registry anchor; a second wait here
 * would double the worst-case startup delay on a fresh world. What this reports is the
 * case where that gate timed out: the tables this registry binds are created by the
 * system's own adventure import (module/apps/init.mjs FirstTimeSetup, which sets
 * `alienrpg.imported` at :78) and can be rewritten by its folder migration
 * (module/apps/migratefolders.js, which parks `alienrpg.ARPGSemaphore` on "busy" at :12
 * and clears it at :118). init.mjs:49 calls FirstTimeSetup() WITHOUT awaiting it.
 */
function assertWorldSettled() {
  let imported;
  let semaphore;
  try {
    imported = globalThis.game.settings.get(SYSTEM_ID, "imported");
    semaphore = globalThis.game.settings.get(SYSTEM_ID, "ARPGSemaphore");
  } catch {
    return; // not the alienrpg system, or those settings are gone — nothing to judge
  }
  if (imported === true && semaphore !== "busy") return;
  console.warn(
    `${MID} | registry: resolving while the world is not settled ` +
      `(imported=${imported}, ARPGSemaphore="${semaphore}") — some bindings may be missing; ` +
      `re-run them from Game Settings with the "Auto-detect" button`,
  );
}

export const registry = {
  /**
   * Declare a semantic key. Called 13 times from main.mjs during `init`.
   * Pure bookkeeping — it touches no setting, so init may call it before or after
   * registerSettings().
   * @param {string} key
   * @param {{kind: "table"|"folder"|"journal", docType?: string|null, guess?: string[]}} def
   */
  declare(key, { kind, docType = null, guess } = {}) {
    if (typeof key !== "string" || key.length === 0) {
      throw new Error(`${MID} | registry.declare: key must be a non-empty string`);
    }
    if (!KINDS.has(kind)) {
      throw new Error(`${MID} | registry.declare: unknown kind "${kind}" for key "${key}"`);
    }
    declared.set(key, { key, kind, docType, guess: Array.isArray(guess) ? [...guess] : [] });
    return declared.get(key);
  },

  /** Register the world setting and the GM rebind menu. Called once from `init`. */
  registerSettings() {
    if (settingsReady) return;
    const g = globalThis.game;
    if (!g?.settings) throw new Error(`${MID} | registry.registerSettings: game.settings is not available yet`);
    g.settings.register(MID, SETTING_BINDINGS, {
      scope: "world",
      config: false,
      type: Object,
      default: {},
    });
    settingsReady = true;
  },

  /**
   * Resolve every declared key. The caller (main.mjs) has already awaited the boot-order
   * gate, so this only asserts and resolves — it never waits.
   * @returns {Promise<{bound: number, unbound: string[]}>}
   */
  async resolveAll() {
    if (!settingsReady) console.warn(`${MID} | registry: resolveAll() ran before registerSettings()`);
    assertWorldSettled();

    const store = structuredClone(readBindings());
    let dirty = false;

    for (const def of declared.values()) {
      const current = store[def.key];
      // A binding that still resolves is never recomputed — names may drift freely.
      if (current?.uuid && syncUuid(current.uuid)) continue;
      const pick = pureChooseBinding(collectCandidates(def), expandGuesses(def.guess));
      // No pick: keep whatever is stored. unbound() judges by resolvability, so a
      // temporarily absent document heals itself on the next resolveAll().
      if (!pick) continue;
      store[def.key] = {
        uuid: pick.uuid,
        name: pick.name,
        boundAt: Date.now(),
        boundBy: globalThis.game?.user?.id ?? null,
      };
      dirty = true;
    }

    cache = store;
    if (dirty && globalThis.game?.user?.isGM) {
      await globalThis.game.settings.set(MID, SETTING_BINDINGS, store);
    }

    const missing = this.unbound();
    if (missing.length > 0 && globalThis.game?.user?.isGM) {
      console.warn(`${MID} | registry: ${missing.length} unbound key(s):`, missing);
      globalThis.ui?.notifications?.warn(
        globalThis.game.i18n.format("AEA.registry.unboundWarn", { count: missing.length }),
      );
    }
    return { bound: Object.keys(store).length, unbound: missing };
  },

  /** @returns {object|null} a RollTable document */
  table(key) {
    return typedGet(key, "table");
  },
  /** @returns {object|null} a Folder document */
  folder(key) {
    return typedGet(key, "folder");
  },
  /** @returns {object|null} a JournalEntry document */
  journal(key) {
    return typedGet(key, "journal");
  },

  /**
   * Map a legacy DISPLAY NAME onto an already-bound key. This is the dynamic path:
   * module/documents/actor.mjs:1839 does `game.tables.getName(dataset.atttype)`, where
   * dataset.atttype is the English literal stored in a creature's `system.cTables`
   * (templates/actor/creature-general.hbs:19). That literal lives in actor data, so
   * Babele never rewrites it — which is exactly why matching it against the declared
   * guess lists works while matching it against live table names does not.
   *
   * A null result means "this module does not take over; the system's own code runs as
   * it always did". Callers must pass it through untouched: they MUST NOT fall back to
   * getName() or contents.find(), MUST NOT swallow the system's own ui.notifications.warn,
   * and MUST NOT guess a table on the system's behalf. Per-creature attack tables
   * (actor.mjs:1839 with a monster-specific name, and :2497) are phase 2's work.
   * @param {string} name
   * @returns {object|null} a RollTable document
   */
  tableByLegacyName(name) {
    if (typeof name !== "string" || !name || name === "None") return null;
    const candidates = [];
    for (const def of declared.values()) {
      if (def.kind !== "table") continue;
      // Reuse the pure matcher by treating each (key, guess) pair as a candidate whose
      // "uuid" is the key; pureChooseBinding dedupes by that, so a key with several
      // guesses still counts once and two keys claiming one name stay ambiguous.
      for (const guess of expandGuesses(def.guess)) candidates.push({ uuid: def.key, name: guess });
    }
    const pick = pureChooseBinding(candidates, [name]);
    if (!pick) {
      console.warn(`${MID} | registry: legacy table name "${name}" maps to no declared key`);
      return null;
    }
    return typedGet(pick.uuid, "table");
  },

  /**
   * Bind a key to a specific document uuid. Used by the GM rebind panel.
   * @param {string} key
   * @param {string} uuid
   */
  async bind(key, uuid) {
    const def = declared.get(key);
    if (!def) throw new Error(`${MID} | registry.bind: key "${key}" was never declared`);
    const doc = syncUuid(uuid);
    if (!doc) throw new Error(`${MID} | registry.bind: "${uuid}" does not resolve to a document`);
    const expected = DOCUMENT_NAME[def.kind];
    if (doc.documentName !== expected) {
      throw new Error(`${MID} | registry.bind: "${uuid}" is a ${doc.documentName}, expected ${expected}`);
    }
    const entry = {
      uuid,
      name: doc.name,
      boundAt: Date.now(),
      boundBy: globalThis.game?.user?.id ?? null,
    };
    const next = { ...readBindings(), [key]: entry };
    cache = next;
    await globalThis.game.settings.set(MID, SETTING_BINDINGS, next);
    return entry;
  },

  /** @returns {object} a copy of the whole binding table */
  bindings() {
    return structuredClone(readBindings());
  },

  /** @returns {string[]} declared keys with no currently resolvable binding */
  unbound() {
    const store = readBindings();
    return [...declared.keys()].filter((key) => !syncUuid(store[key]?.uuid));
  },
};
```

- [ ] **Step 9: Run it and watch it pass**

Run: `npx vitest run test/registry-foundry.test.mjs`
Expected: PASS —— `Tests 20 passed (20)`，整文件耗时 1 秒以内（这里没有任何定时器等待：闸门在 main.mjs，本文件只断言）。

- [ ] **Step 10: Commit**

```bash
git add scripts/kernel/registry.mjs test/registry-foundry.test.mjs && git commit -F - <<'EOF'
feat(kernel): K2 registry 对象、世界安定断言与遗留名映射

registerSettings/declare/resolveAll/table/folder/journal/tableByLegacyName/
bind/bindings/unbound 落地，绑定表存世界设置
alien-evolved-automation.registryBindings，形状 {key: {uuid, name, boundAt, boundBy}}。

启动排序的闸门只有一个属主：main.mjs 的 waitForWorldSettled，它在 ready 段的
registry 子锚点之前。resolveAll 这边只做一次廉价诊断——读一次 alienrpg.imported
与 alienrpg.ARPGSemaphore，未安定就 warn 一条再照常解析。不在这里等第二遍：
两道守卫串起来会把新世界的最坏启动延迟翻倍，判据漂移还会造成只在新世界复现的
间歇故障。文档来晚了走重绑面板的「自动识别」重解析。

declare 不碰任何设置，也不被 registerSettings 快照——init 段是先注册设置后声明
十三个键，快照会让所有键对面板与 unbound() 隐形，一条用例守住这点。
文件夹候选按 docType 过滤，同名的日志文件夹不再让文件夹键判歧义。guess 里以
ALIENRPG. 开头的项在 resolveAll 时才经 game.i18n 展开——declare 跑在 init，
那时翻译还没加载。

tableByLegacyName 把 actor.mjs:1839 的动态表名映射到已绑定的键：那个名字取自
creature 的 system.cTables 字面量，Babele 不改它，所以按名字找键、按 uuid 取表。
返回 null 的语义定死为「本模组不介入，系统原样走」，调用方不得自行回退 getName、
不得吞掉系统的 warn、不得替系统猜表。

已绑定且 uuid 仍可解析的键不重算；解析不到时保留旧条目而不是删掉。
非 GM 客户端只读不写世界设置。测试全部经公共桩 installFoundryStub 喂夹具，
本文件不给 globalThis 赋任何值。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

- [ ] **Step 11: 为 GM 重绑面板写失败测试**

在 `test/registry-foundry.test.mjs` 末尾追加一个 describe 块：

```js
describe("registry rebind menu", () => {
  it("registers a GM-only settings menu whose type is an application class", async () => {
    await seedWorld();
    const { registry } = await loadRegistry();
    registry.registerSettings();

    const menus = asList(ctx.menus);
    expect(menus).toHaveLength(1);
    // ctx.menus entries carry the registerMenu(namespace, key, config) arguments; the
    // config is either the entry itself or its `config`/`cfg` field depending on the stub.
    const entry = menus[0];
    const cfg = entry?.config ?? entry?.cfg ?? entry;
    expect(JSON.stringify([entry?.namespace, entry?.ns, entry?.module])).toContain("alien-evolved-automation");
    expect(JSON.stringify([entry?.key, entry?.name, cfg?.key])).toContain("registryRebind");
    expect(cfg.restricted).toBe(true);
    expect(typeof cfg.type).toBe("function");
    expect(cfg.name).toBe("AEA.registry.menuName");
    expect(cfg.label).toBe("AEA.registry.menuLabel");
    expect(cfg.hint).toBe("AEA.registry.menuHint");
  });

  it("registers the menu only once even if registerSettings is called again", async () => {
    await seedWorld();
    const { registry } = await loadRegistry();
    registry.registerSettings();
    registry.registerSettings();
    expect(asList(ctx.menus)).toHaveLength(1);
  });
});
```

- [ ] **Step 12: Run it and watch it fail**

Run: `npx vitest run test/registry-foundry.test.mjs -t "registers a GM-only settings menu"`

Expected: FAIL —— `AssertionError: expected [] to have a length of 1 but got +0`。`registerSettings()` 目前只注册了世界设置，没注册菜单。

- [ ] **Step 13: 实现重绑面板、模板、样式与 i18n 键**

**13a —— 在 `scripts/kernel/registry.mjs` 的 `export const registry = {` 之前插入下面这段：**

`ApplicationV2` 是 Foundry V13 引入的窗口基类，`HandlebarsApplicationMixin` 给它加上 Handlebars 模板渲染。两者都住在 `foundry.applications.api` 下，而 `foundry` 这个全局在模块被 import 的那一刻还不一定存在，所以类必须**惰性**构造——只在 `init` 阶段 `registerSettings()` 时才求值。契约 §4 K2 没有给这个类命名，所以它不导出，只经 `registerMenu` 引用，模块外不可见。

```js
/** @type {Function|null} */
let AppClass = null;

/**
 * Lazily build the GM rebind window. ApplicationV2 lives on the `foundry` global, which
 * does not exist at module-import time, so the class is created on first use (during `init`).
 */
function registryBindingApp() {
  if (AppClass) return AppClass;
  const { ApplicationV2, HandlebarsApplicationMixin } = globalThis.foundry.applications.api;

  AppClass = class RegistryBindingApp extends HandlebarsApplicationMixin(ApplicationV2) {
    /** Re-run automatic detection, then redraw. Bound to the app instance by ApplicationV2. */
    static async onAutoBind() {
      await registry.resolveAll();
      this.render();
    }

    /** ApplicationV2 form handler: (event, form, formData). */
    static async onSubmit(event, form, formData) {
      const submitted = formData?.object ?? {};
      const current = registry.bindings();
      for (const [key, uuid] of Object.entries(submitted)) {
        if (!uuid || current[key]?.uuid === uuid) continue;
        await registry.bind(key, uuid);
      }
      globalThis.ui?.notifications?.info(globalThis.game.i18n.localize("AEA.registry.saved"));
      this.render();
    }

    static DEFAULT_OPTIONS = {
      id: "aea-registry-rebind",
      tag: "form",
      classes: ["aea", "aea-registry-rebind"],
      window: { title: "AEA.registry.title", icon: "fas fa-link", resizable: true },
      position: { width: 680, height: "auto" },
      form: { handler: RegistryBindingApp.onSubmit, closeOnSubmit: false, submitOnChange: false },
      actions: { autoBind: RegistryBindingApp.onAutoBind },
    };

    static PARTS = { body: { template: `modules/${MID}/templates/registry-rebind.hbs` } };

    /**
     * Rows are computed HERE, at render time — never snapshotted in registerSettings().
     * main.mjs registers the setting first and declares the 13 keys afterwards, so a
     * snapshot would render an empty panel forever.
     */
    async _prepareContext() {
      const bound = registry.bindings();
      const rows = [...declared.values()].map((def) => {
        const entry = bound[def.key];
        const candidates = collectCandidates(def)
          .filter((c) => typeof c.uuid === "string" && typeof c.name === "string")
          .sort((a, b) => a.name.localeCompare(b.name))
          .map((c) => ({ uuid: c.uuid, name: c.name, selected: c.uuid === entry?.uuid }));
        return { key: def.key, kind: def.kind, bound: Boolean(entry?.uuid), candidates };
      });
      return { rows };
    }
  };
  return AppClass;
}
```

并把 `registerSettings()` 整个方法替换为：

```js
  /**
   * Register the world setting and the GM rebind menu. Called once from `init`.
   * It registers exactly ONE world setting plus ONE menu — never one setting per key —
   * and it must NOT snapshot `declared` or the binding table: main.mjs declares the 13
   * keys AFTER this call, and the panel reads them at render time.
   */
  registerSettings() {
    if (settingsReady) return;
    const g = globalThis.game;
    if (!g?.settings) throw new Error(`${MID} | registry.registerSettings: game.settings is not available yet`);
    g.settings.register(MID, SETTING_BINDINGS, {
      scope: "world",
      config: false,
      type: Object,
      default: {},
    });
    g.settings.registerMenu(MID, "registryRebind", {
      name: "AEA.registry.menuName",
      label: "AEA.registry.menuLabel",
      hint: "AEA.registry.menuHint",
      icon: "fas fa-link",
      type: registryBindingApp(),
      restricted: true,
    });
    settingsReady = true;
  },
```

**13b —— 创建 `templates/registry-rebind.hbs`：**

```hbs
<section class="aea-registry-rebind">
  <p class="notes">{{localize "AEA.registry.intro"}}</p>
  <table>
    <thead>
      <tr>
        <th>{{localize "AEA.registry.colKey"}}</th>
        <th>{{localize "AEA.registry.colKind"}}</th>
        <th>{{localize "AEA.registry.colBinding"}}</th>
      </tr>
    </thead>
    <tbody>
      {{#each rows}}
      <tr class="{{#unless this.bound}}aea-unbound{{/unless}}">
        <td><code>{{this.key}}</code></td>
        <td>{{this.kind}}</td>
        <td>
          <select name="{{this.key}}">
            <option value="">{{localize "AEA.registry.none"}}</option>
            {{#each this.candidates}}
            <option value="{{this.uuid}}" {{#if this.selected}}selected{{/if}}>{{this.name}}</option>
            {{/each}}
          </select>
        </td>
      </tr>
      {{/each}}
    </tbody>
  </table>
  <footer class="form-footer">
    <button type="button" data-action="autoBind">
      <i class="fas fa-wand-magic-sparkles"></i> {{localize "AEA.registry.autoBind"}}
    </button>
    <button type="submit"><i class="fas fa-floppy-disk"></i> {{localize "AEA.registry.save"}}</button>
  </footer>
</section>
```

**13c —— 追加到 `styles/alien-evolved-automation.css` 末尾：**

```css
.aea-registry-rebind table { width: 100%; border-collapse: collapse; }
.aea-registry-rebind th, .aea-registry-rebind td { padding: 2px 4px; text-align: left; }
.aea-registry-rebind tr.aea-unbound { outline: 1px solid var(--color-level-error, #b33); }
.aea-registry-rebind tr.aea-unbound code { color: var(--color-level-error, #b33); }
```

**13d —— 把这一段合并进 `lang/en.json` 里那个唯一的顶层 `"AEA"` 对象（与已有的二级段并列，逗号别漏）：**

```json
    "registry": {
      "menuName": "Document Bindings",
      "menuLabel": "Configure Bindings",
      "menuHint": "Bind the roll tables, folders and journal entry the automation needs. Bindings are stored by document id, so renaming or translating a document never breaks them.",
      "title": "Alien Evolved: Automation — Document Bindings",
      "intro": "Every lookup resolves by document id. Pick the correct document for any key shown as unbound.",
      "colKey": "Key",
      "colKind": "Type",
      "colBinding": "Bound document",
      "none": "— unbound —",
      "autoBind": "Auto-detect",
      "save": "Save",
      "saved": "Bindings saved.",
      "unboundWarn": "Alien Evolved: Automation — {count} document binding(s) could not be resolved. Open Game Settings and configure them."
    }
```

**13e —— 把这一段合并进 `lang/cn.json` 的 `"AEA"` 对象：**

```json
    "registry": {
      "menuName": "文档绑定",
      "menuLabel": "配置绑定",
      "menuHint": "为自动化所需的随机表、文件夹与日志建立绑定。绑定按文档 id 保存，改名或翻译都不会打断。",
      "title": "Alien Evolved: Automation —— 文档绑定",
      "intro": "所有查找一律按文档 id 解析。请为显示为「未绑定」的键选择正确的文档。",
      "colKey": "键",
      "colKind": "类型",
      "colBinding": "已绑定文档",
      "none": "—— 未绑定 ——",
      "autoBind": "自动识别",
      "save": "保存",
      "saved": "绑定已保存。",
      "unboundWarn": "Alien Evolved: Automation —— 有 {count} 个文档绑定未能解析，请到「游戏设置」中手动配置。"
    }
```

- [ ] **Step 14: Run it and watch it pass**

Run: `npx vitest run test/registry-choose.test.mjs test/registry-foundry.test.mjs`
Expected: PASS —— `test/registry-choose.test.mjs` 15 passed、`test/registry-foundry.test.mjs` 22 passed。

- [ ] **Step 15: Commit**

```bash
git add scripts/kernel/registry.mjs templates/registry-rebind.hbs styles/alien-evolved-automation.css lang/en.json lang/cn.json test/registry-foundry.test.mjs && git commit -F - <<'EOF'
feat(kernel): K2 GM 重绑面板（ApplicationV2）

registerSettings 里挂 restricted: true 的设置菜单——一个世界设置加一个菜单，
不给每个键各注册一个键。面板逐键列出 kind、当前绑定，未绑定行加 aea-unbound 类
（样式表里给一条红色描边）；下拉列出该类型的全部世界文档（文件夹按 docType
过滤），提交走 registry.bind。「自动识别」按钮重跑 resolveAll —— 这是文档晚到时
的既定恢复路径。

行与下拉都在 _prepareContext 里现算，registerSettings 不快照 declared：
init 段是先注册设置、后声明十三个键，快照会让面板永远是空表。

ApplicationV2 住在 foundry.applications.api 上，模块 import 时该全局尚不存在，
因此类惰性构造，只在 init 阶段 registerSettings() 时求值——这样 registry.mjs
仍可被 vitest 直接 import。契约 §4 K2 没给这个类命名，故不导出，只经
registerMenu 引用。

i18n 键并进语言包唯一的顶层 AEA 对象，二级段 registry 取小写，与其余段一致。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

- [ ] **Step 16: 在 main.mjs 顶部补两行 import**

打开 `scripts/main.mjs`，逐字搜索这行注释（**搜文本，不要用行号**）：

```
/* AEA-ANCHOR: imports */
```

它在文件顶部的 import 区。做两件事：

1. 在**这行注释的下一行**插入：

   ```js
   import { registry } from "./kernel/registry.mjs";
   ```

2. 再**先搜索** `import { selftest }`。**已经存在就什么都不做**；不存在才在同一位置补一行：

   ```js
   import { selftest } from "./kernel/selftest.mjs";
   ```

   （`selftest` 这个内核模块的属主是另一个任务，它落地时也会加同一行。两边都写成「有则跳过」，谁先谁后都不会出现重复声明。）

若 `scripts/kernel/selftest.mjs` 此刻尚不存在，本步骤仍然照做——这是 `main.mjs` 的既定装配。要知道的后果是：在那个文件落地之前，整个模组在 Foundry 里加载会失败；vitest 不受影响，因为本任务的测试文件一个都不 import `main.mjs`。

- [ ] **Step 17: 把 api 里 registry 那一个 null 换成 registry**

同一个文件里找到骨架写好的 api 字面量（八个槽，值全是 `null`）：

```js
export const api = {
  features: null, patches: null, resolver: null, registry: null,
  rollBus: null, diceBarrier: null, cards: null, selftest: null,
};
```

把 `registry: null` 改成 `registry,`（ES 的对象简写，等价于 `registry: registry`）。改完那一行读作：

```js
  features: null, patches: null, resolver: null, registry,
```

**只动这一个槽。** 不要替换 `api` 这个对象本身、不要增删键、不要顺手把 `selftest: null` 也填上——那七个槽各有自己的属主任务。

- [ ] **Step 18: 在 init 锚点下注册设置并声明 13 个键**

逐字搜索：

```
/* AEA-ANCHOR: init */
```

它在 `Hooks.once("init", () => { ... })` 的处理器函数体里，且**排在两个 `register()` 循环之后**（`Hooks.once(name, fn)` 是 Foundry 的一次性事件注册；`init` 在世界数据加载前触发，是注册设置的唯一合法时机）。把下面整块插在**这行注释的下一行**。

块内先 `registerSettings()` 后 13 次 `declare()`，与契约的 init 顺序一致；这样排是安全的，因为 `registerSettings()` 不快照 `declared`（面板在渲染时才现算行）。与同一个 init 体里别人插的行也无先后依赖。

```js
  registry.registerSettings();

  // 13 semantic keys. Names are guesses used exactly once, at first bind; everything
  // afterwards resolves by uuid, so a rename or a Babele translation cannot break the
  // chain. A guess written as an "ALIENRPG.*" key is localized at resolve time, because
  // translations are not loaded during init. See the lookups these replace:
  // module/documents/actor.mjs:554, 845, 1067, 1812-1817, 1830, 1839, 1848, 1855;
  // module/helpers/rollTableData.mjs:7, 24; module/apps/init.mjs:49, 81, 107.
  registry.declare("panic", { kind: "table", guess: ["Panic Table"] });
  registry.declare("stressResponse", { kind: "table", guess: ["Stress Response Table"] });
  registry.declare("panicResponse", { kind: "table", guess: ["Panic Response Table"] });
  registry.declare("critInjury1e", { kind: "table", guess: ["Critical Injuries", "Critical injuries", "ALIENRPG.CriticalInjuries"] });
  registry.declare("critInjuryEvolved", { kind: "table", guess: ["EV - Critical Injuries", "ALIENRPG.EVCriticalInjuries"] });
  registry.declare("critInjurySynthetic", { kind: "table", guess: ["Critical Injuries on Synthetics", "critical injuries on synthetics"] });
  registry.declare("critInjuryXeno", { kind: "table", guess: ["Critical Injuries on Xenomorphs", "EV - Critical Injuries on Xenomorphs"] });
  registry.declare("shipMinorComponent", { kind: "table", guess: ["Spaceship Minor Component Damage"] });
  registry.declare("shipMajorComponent", { kind: "table", guess: ["Spaceship Major Component Damage"] });
  registry.declare("folderAlienTables", { kind: "folder", docType: "RollTable", guess: ["Alien Tables"] });
  registry.declare("folderCreatureTables", { kind: "folder", docType: "RollTable", guess: ["Alien Creature Tables"] });
  registry.declare("folderMotherTables", { kind: "folder", docType: "RollTable", guess: ["Alien Mother Tables"] });
  registry.declare("journalMother", { kind: "journal", guess: ["MU/TH/ER Instructions."] });
```

- [ ] **Step 19: 在 ready.registry 子锚点下插解析调用**

逐字搜索：

```
/* AEA-ANCHOR: ready.registry */
```

它在 `Hooks.once("ready", async () => { ... })` 的处理器体里，紧跟在 `await waitForWorldSettled({ timeoutMs: 10000, pollMs: 100 });` 之后，是 ready 段**四个有序子锚点的第一个**（后面依次是 `ready.patches`、`ready.rollbus`、`ready.cards`，各有各的属主任务）。把这一行插在**这行注释的下一行**：

```js
  await registry.resolveAll();
```

**只插这一行，只插在这个子锚点下。** 位置是硬要求，不是风格：解析必须早于 `ready.patches` 段的 `await patches.applyAll();`，否则补丁装上时注册表还是空的；也必须晚于 `waitForWorldSettled`，否则系统的冒险导入还没把表建出来。本任务**不**在 `resolveAll()` 内部再等一次（理由见「背景二」）。

- [ ] **Step 20: 验证 main.mjs 的装配（语法 + 顺序 + api 槽）并跑全部测试**

Run（一整条命令，四项检查）：

```bash
node --check scripts/main.mjs && node -e "
const s = require('fs').readFileSync('scripts/main.mjs','utf8');
const at = (needle) => s.indexOf(needle);
if (at('import { registry }') < 0) throw new Error('registry import missing');
if (/registry:\s*null/.test(s)) throw new Error('api.registry is still null');
if (at('registry.registerSettings();') < 0) throw new Error('registerSettings not inserted');
if ((s.match(/registry\.declare\(/g) || []).length !== 13) throw new Error('expected exactly 13 declare() calls');
const resolve = at('await registry.resolveAll();');
if (resolve < 0) throw new Error('resolveAll not inserted');
const gate = at('waitForWorldSettled');
if (!(gate >= 0 && gate < resolve)) throw new Error('resolveAll must come after waitForWorldSettled');
const patches = at('AEA-ANCHOR: ready.patches');
if (patches >= 0 && resolve > patches) throw new Error('resolveAll must precede the ready.patches anchor');
console.log('main.mjs wiring ok');
" && npx vitest run test/registry-choose.test.mjs test/registry-foundry.test.mjs
```

Expected: PASS —— `node --check` 无输出（ESM 语法解析通过），随后打印 `main.mjs wiring ok`，再 15 passed + 22 passed。

- [ ] **Step 21: Commit**

```bash
git add scripts/main.mjs && git commit -F - <<'EOF'
feat(main): 装配 K2 —— import、api 槽、init 段十三个键、ready 段解析

三张恐慌/压力主表（panic / stressResponse / panicResponse）、四张分型重伤表
（critInjury1e / critInjuryEvolved / critInjurySynthetic / critInjuryXeno）、
两张飞船部件损伤表（shipMinorComponent / shipMajorComponent）、
三个文件夹（folderAlienTables / folderCreatureTables / folderMotherTables，
docType 一律 RollTable）、一篇日志（journalMother）。

四处插入全部按锚点文本定位而不是行号：imports 锚点加 registry 的 import
（selftest 那行幂等，已有就跳过）；api 八槽里只把 registry 的 null 换成对象，
不增删键；init 锚点插 registerSettings 与十三次 declare；ready.registry 子锚点
插 await registry.resolveAll()。

解析这一行的位置是硬要求：晚于 waitForWorldSettled（系统的冒险导入才建出这些
文档），早于 ready.patches 段的 patches.applyAll（补丁装上时注册表不能是空的）。
加了一条静态顺序守卫防止后来者插错位置。

guess 里的名字只在首次绑定时用一次。critInjury1e / critInjuryEvolved 各带一个
ALIENRPG.* 键，由 resolveAll 在 ready 时展开——照系统 actor.mjs:1812/1815 的双查
写法，中文世界首次绑定也能命中。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

- [ ] **Step 22: 把绑定完整性与遗留名映射登记为两条自检**

真实的 `ready` 队列、Babele 与系统导入的竞态，vitest 的桩只能复述作者的假设。按契约 §0.4，这两条断言进 `selftest`。

打开 `scripts/main.mjs`，逐字搜索 `/* AEA-ANCHOR: init */`（就是 Step 18 用过的那个锚点），把下面整块插在 **Step 18 插入的最后一行 `registry.declare("journalMother", ...)` 的下一行**。

`label` 传的是 **i18n 键本身**，不是 `game.i18n.localize(...)` 的结果，也不是惰性 getter：登记发生在 `init`，那时 Foundry 还没加载语言包，`localize()` 只会把键原样回声；由 `selftest.runAll()` 在运行时本地化。存键还有一个好处——def 仍然是可序列化的普通对象。两条 `register` 与同一锚点下的其它行没有先后依赖。**不新增 import**：`selftest` 已在 Step 16 补好。

```js
  selftest.register({
    id: "registry-bindings",
    label: "AEA.selftest.registryBindings",
    run: () => {
      const missing = registry.unbound();
      const total = Object.keys(registry.bindings()).length;
      return {
        ok: missing.length === 0,
        detail: missing.length === 0
          ? `${total} key(s) bound, 0 unbound`
          : `unbound: ${missing.join(", ")}`,
      };
    },
  });

  selftest.register({
    id: "registry-legacy-name",
    label: "AEA.selftest.registryLegacyName",
    run: () => {
      // The name a creature stores in system.cTables must reach a bound table, or the
      // Roll Crit button on every xenomorph stays dead (actor.mjs:1839).
      const legacy = "Critical Injuries on Xenomorphs";
      const table = registry.tableByLegacyName(legacy);
      return {
        ok: Boolean(table),
        detail: table ? `"${legacy}" -> ${table.uuid} (${table.name})` : `"${legacy}" -> nothing`,
      };
    },
  });
```

- [ ] **Step 23: 给两条自检加语言包键**

把这两行合并进 `lang/en.json` 的 `AEA.selftest` 段（若 `AEA` 下还没有 `selftest` 段，就新建这个段，与 `registry` 段平级）：

```json
      "registryBindings": "Registry: every declared key is bound",
      "registryLegacyName": "Registry: the xenomorph crit table name still maps to a bound table"
```

同样合并进 `lang/cn.json` 的 `AEA.selftest` 段：

```json
      "registryBindings": "注册表：所有声明的键都已绑定",
      "registryLegacyName": "注册表：异形重伤表的遗留表名仍能映射到已绑定的表"
```

- [ ] **Step 24: 验证语法与语言包仍合法**

Run:

```bash
node --check scripts/main.mjs && node -e "
for (const f of ['lang/en.json','lang/cn.json']) {
  const o = JSON.parse(require('fs').readFileSync(f,'utf8'));
  const k = Object.keys(o);
  if (k.length !== 1 || k[0] !== 'AEA') throw new Error(f + ' top-level keys: ' + k.join(','));
  if (!o.AEA.registry || !o.AEA.registry.menuName) throw new Error(f + ' missing AEA.registry.menuName');
  if (!o.AEA.selftest || !o.AEA.selftest.registryBindings || !o.AEA.selftest.registryLegacyName) throw new Error(f + ' missing AEA.selftest keys');
  console.log(f, 'ok');
}"
```

Expected: PASS —— 打印 `lang/en.json ok` 与 `lang/cn.json ok`。这条同时证明：两个语言包仍是合法 JSON、顶层仍然只有 `AEA` 一个键（护栏要求）、本任务新加的四个键都在。

- [ ] **Step 25: Commit**

```bash
git add scripts/main.mjs lang/en.json lang/cn.json && git commit -F - <<'EOF'
feat(kernel): 把绑定完整性与遗留名映射登记为两条自检

registry-bindings 断言 unbound() 为空并列出缺哪几个键；
registry-legacy-name 断言 "Critical Injuries on Xenomorphs" 仍能经
tableByLegacyName 落到一张已绑定的表——这条一断，异形卡上的 Roll Crit
（actor.mjs:1839）就整条哑掉。

两条都登记在 init 锚点、label 传 i18n 键而不是本地化文本：登记跑在 init，
那时语言包还没加载，localize 只会回声裸键；本地化由 runAll 在运行时做，
def 因此保持可序列化。

真实 ready 队列与 Babele 竞态桩测不了，按契约 §0.4 走 selftest 而不是编造单测。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

- [ ] **Step 26: MANUAL VERIFICATION（排序、改名不打断与面板，都依赖真实 ready 队列）**

**A. 首次绑定与排序**

1. 在本机 Foundry 里**新建**一个用 `alienrpg` 4.1.13 的世界（必须是新世界：只有 `alienrpg.imported` 为 `false` 时系统才会跑 `FirstTimeSetup()`，这正是要验的竞态）。启用本模组，进入世界。
2. 系统会自动导入冒险包并弹出 MU/TH/ER 日志。等它弹完，按 F12 打开控制台执行：

   ```js
   const api = game.modules.get("alien-evolved-automation").api;
   console.log("api.registry present:", Boolean(api.registry));
   console.table(api.registry.bindings());
   console.log("unbound:", api.registry.unbound());
   console.table(await api.selftest.runAll());
   ```
3. **期望观察**：`api.registry present: true`（若打出 `false`，说明 Step 17 的 api 槽没换）；第一张表有 13 行；九个表键的 `uuid` 形如 `RollTable.<16 位 id>`，三个 folder 键形如 `Folder.<16 位 id>`，`journalMother` 形如 `JournalEntry.<16 位 id>`；`unbound: []`；自检表里 `registry-bindings` 与 `registry-legacy-name` 两行 `ok` 都是 `true`。控制台**没有**以 `alien-evolved-automation | registry:` 开头的黄色警告（特别是不应出现 `resolving while the world is not settled`——出现它说明 main.mjs 的就绪守卫超时了，也不应出现 `resolveAll() ran before registerSettings()`——出现它说明 init 段那块没插上），右下角**没有**黄色 toast。
   若 `unbound` 里出现键，多半是这个世界的表本来就是译名、名字对不上，走下面的 C 手动绑。
4. **期望观察（自检标签已本地化）**：在中文客户端上，第三张表的 `label` 列读作「注册表：所有声明的键都已绑定」与「注册表：异形重伤表的遗留表名仍能映射到已绑定的表」，**不是** `AEA.selftest.registryBindings` 这样的裸键。本任务传给 `selftest.register` 的 `label` 是键，本地化是 `selftest.runAll()` 的职责——若这里显示裸键，缺陷在 `selftest.mjs`（它没有对 `def.label` 调 `game.i18n.localize`），报到那边去修，不要改回在 `init` 里 localize。
5. **期望观察（ready 顺序）**：控制台执行 `game.modules.get("alien-evolved-automation").api.patches?.status?.()`。若补丁模块已落地，它应当已经 `applied`——而绑定表在第 3 步已经是满的，说明 `resolveAll()` 确实跑在 `applyAll()` 之前。若补丁模块尚未落地（`api.patches` 为 `null`），跳过这条，Step 20 的静态顺序守卫已经守住了插入位置。

**B. 改名不打断**

6. 在 Roll Tables 侧边栏把 `Panic Table` 重命名为 `恐慌表`。刷新页面（F5）。控制台执行 `game.modules.get("alien-evolved-automation").api.registry.table("panic")?.name`。
   **期望观察**：打出 `恐慌表`——绑定按 uuid 保存，改名不影响；`api.registry.unbound()` 仍是 `[]`。**这一步是整个 K2 的存在理由**：换成系统原本的 `game.tables.getName("Panic Table")`，此刻会返回 `undefined`，恐慌链整条静默失效。
7. 再把 `Critical Injuries on Xenomorphs` 重命名为 `异形重伤表`，刷新页面。
8. 控制台执行 `game.modules.get("alien-evolved-automation").api.registry.tableByLegacyName("Critical Injuries on Xenomorphs")?.name`。
   **期望观察**：打出 `异形重伤表`。这条证明动态名那一路也扛住了改名：怪物卡里 `system.cTables` 存的仍是英文字面量，键映射不变，取表按 uuid。

**C. 重绑面板**

9. 删掉世界里的 `恐慌表`（右键 → Delete）。刷新页面。
10. **期望观察**：右下角出现一条黄色 toast，中文世界读作「Alien Evolved: Automation —— 有 1 个文档绑定未能解析，请到「游戏设置」中手动配置。」只出现**一条**，不是每个未绑定键一条。
11. 打开 Game Settings → Configure Settings → Module Settings，找到 Alien Evolved: Automation 下的「配置绑定 / Configure Bindings」按钮，点开。
12. **期望观察**：窗口标题为「Alien Evolved: Automation —— 文档绑定」，表格**恰好 13 行**（这一条同时证明 `registerSettings()` 没有在 init 时快照键集——它跑在 13 次 `declare()` 之前，若快照了，这里会是 0 行）；`panic` 那一行的下拉显示「—— 未绑定 ——」，整行带红色描边（`aea-unbound`）；其余 12 行下拉各自选中了对应文档；三个 folder 行的下拉里**只有 RollTable 文件夹**，没有日志或场景的文件夹。
13. 从 `panic` 行的下拉里挑另一张表（例如 `Stress Response Table`），点「保存」。
14. **期望观察**：出现「绑定已保存。」提示，窗口重绘后 `panic` 行选中了刚挑的那张表，红色描边消失。控制台 `api.registry.table("panic").name` 与所选一致。
15. 点「自动识别」按钮。
    **期望观察**：面板重绘，控制台没有新的报错。这个按钮重跑 `resolveAll()`，是文档比模组晚到（守卫超时、或 GM 事后才导入冒险包）时的恢复路径。
16. 用普通玩家账号（非 GM）登录同一世界，打开 Game Settings → Module Settings。
    **期望观察**：**看不到**「配置绑定」按钮（`restricted: true`）。控制台执行 `game.modules.get("alien-evolved-automation").api.registry.table("panic")?.name` 仍能取到正确的表——玩家客户端只读绑定表，不写世界设置。

**D. 一期兑现边界的验收口径（别把预期读成缺陷）**

17. 控制台执行 `game.modules.get("alien-evolved-automation").api.registry.bindings().shipMinorComponent`。
    **期望观察**：打出一个带 `uuid` / `name` / `boundAt` / `boundBy` 的对象——键**已绑定**。同时，在飞船卡上点部件损伤按钮，系统的行为与没装本模组时**完全一样**（它仍然走 `actor.mjs:1848` 的 `getName`）。`panic` / `stressResponse` / `panicResponse` / `folderAlienTables` / `journalMother` 同理：一期只声明与绑定、不接管。**「键已绑定但对应功能行为未变」是预期结果，不是缺陷**——接管归后续的恐慌链、飞船阶段与 Mother 链任务。
