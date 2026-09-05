> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 17 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 17: 出厂数据修复 —— 三张 EV 神谕表 + NPC 名录校验器

> **分组理由**：这两条修复都不改代码，只改**世界里的文档数据**，因此共用同一套纪律（仅 GM、幂等、记账带 applied-version、`importAdventure` 之后重跑），也共用同一个「先算差异、再让 GM 过目、最后写入」的流程。它们与其它任务触碰的文件零重叠。

**Files:**
- Create: `scripts/repairs/ev-oracle-tables.pure.mjs`
- Create: `scripts/repairs/ev-oracle-tables.mjs`
- Create: `scripts/repairs/npc-roster.pure.mjs`
- Create: `scripts/repairs/npc-roster.mjs`
- Create: `scripts/repairs/repair-log.mjs`（世界设置 `dataRepairs` 的**唯一属主**）
- Create: `scripts/repairs/reimport-guard.mjs`（`importAdventure` 监听的幂等安装器，两条修复共用）
- Create: `test/repair-ev-oracle-tables.test.mjs`
- Create: `test/repair-npc-roster.test.mjs`
- Create: `test/repair-log.test.mjs`
- Create: `test/repair-reimport-guard.test.mjs`
- Modify: `scripts/main.mjs`（**只在三个锚点各插固定几行**：`imports` 三行 import、`repairs` 两个数组成员、`init` 一行 `repairLog.registerSettings();`。不碰 `api` 对象，不碰四个 `ready.*` 子锚点，不碰 `i18nInit` / `diceSoNiceReady`。）
- Modify: `lang/en.json`, `lang/cn.json`

**Interfaces:**

- **Consumes**（全部来自别处已建好的符号，本任务不新建它们）：
  - `import { MID, SETTING_DATA_REPAIRS } from "../const.mjs";` —— `MID === "alien-evolved-automation"`，`SETTING_DATA_REPAIRS === "dataRepairs"`。
  - `import { features } from "../kernel/features.mjs";` —— 用到 `features.register(def)` 与 `features.enabled(id)`；`def = {id, default:"full"|"prompt"|"off", gmOnly, requires:[], hint:""}`，`enabled(id)` 返回 `mode(id) !== "off"`。
  - `import { patches } from "../kernel/patches.mjs";` —— 用到 `patches.register(def)`；`def = {id, type, target, minSystem, fixedIn, probe(), apply()}`，`type` 取 `"WRAPPER"|"MIXED"|"OVERRIDE"|"DATA"|"HOOK"`；`probe()` 返回 `true` 表示「该在这个世界布防」，`apply()` 是无参安装器，可返回 Promise。`patches.status()` 返回 `[{id, type, target, applied, reason, fixedIn}]`。K7 只有 `register` / `applyAll` / `status` 三个方法，**没有撤销或退休回调**。
  - `import { selftest } from "../kernel/selftest.mjs";` —— 用到 `selftest.register({id, label, run})`。**`label` 是 i18n 键（形如 `AEA.selftest.<id>`）而不是已本地化文本**：登记发生在 `init`，此刻语言包尚未加载；`register` 原样保存整个 def，本地化由 `runAll()` 调 `game.i18n.localize(def.label)` 负责。`run()` 返回 `{ok:boolean, detail:string}`，可 async。
  - `import { resolver } from "../kernel/resolver.mjs";` —— 只用 `resolver.actorById(id, {warn})`，这是契约给「手里只有一个裸 actor id」的遗留路径准备的入口。本任务的模块**不得**直接写 `game.actors.get(...)`。
  - `import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";`（测试用）—— 共享测试桩，**只读不改**，禁止在自己的测试文件里就地造 `globalThis.game` 或用私有 Map 顶替 `game.settings`。本任务依赖它这几条已定死的行为：`game.settings.register/get/set` 由 `ctx.settings` 真支撑（`register` 把 `default` 播种进去并写进 `game.settings.settings` 这个 Map，`get` 读**未注册**键抛 `Error`，`set` 返回 Promise）；`Hooks.callAll(name, ...)` 真的按注册顺序同步调用 `Hooks.on`/`Hooks.once` 的回调，`Hooks.off` 可用；`ctx.hooks.on` 的条目形状是 `[{name, fn, once}]`；`game.user = {id, isGM}`。
- **Produces**：
  - `scripts/repairs/ev-oracle-tables.pure.mjs`：`pureOracleTableIds()`、`pureOraclePlan(tables)`、`pureOracleTablesPresent(tables)`、`pureOracleTablesAreBuggy(tables)`
  - `scripts/repairs/npc-roster.pure.mjs`：`pureRosterReference()`、`pureRosterActorIds()`、`pureRosterDiff(actorData, reference?)`、`pureRosterActorsPresent(actorDatas)`、`pureRosterIsStale(actorDatas, reference?)`、`pureEscapeText(value)`
  - `scripts/repairs/repair-log.mjs`：`DATA_REPAIR_VERSION`、`repairLog.registerSettings()`、`repairLog.record(id, {version, at, by, count})`
  - `scripts/repairs/reimport-guard.mjs`：`makeReimportGuard(handler)` → `{install(), retire(), hookId}`
  - `scripts/repairs/ev-oracle-tables.mjs`：`export const evOracleTablesRepair = { id: "ev-oracle-tables", register() }`、`export async function onAdventureImported()`
  - `scripts/repairs/npc-roster.mjs`：`export const npcRosterRepair = { id: "npc-roster", register() }`、`export async function onAdventureImported()`
  - 两条 features 开关：`ev-oracle-tables`、`npc-roster`（默认 `"full"`，`gmOnly: true`）
  - **四条** patches 条目：`ev-oracle-tables`（`type:"DATA"`）、`ev-oracle-tables.reimport`（`type:"HOOK"`，`target:"importAdventure"`）、`npc-roster`（`"DATA"`）、`npc-roster.reimport`（`"HOOK"`）
  - 两条 selftest 条目：`data.ev-oracle-tables`、`data.npc-roster`

**这两个修复模块**不**导出 `install()`**：它们在 `ready` 阶段没有任何订阅要做，真正的动作发生在 `patches.applyAll()` 调用的 `apply()` 里，而 `applyAll()`（`AEA-ANCHOR: ready.patches`）排在 `REPAIRS` 的 install 循环之前。骨架把那一行写成 `r.install?.()`（可选调用）正是为了容纳这种修复。

**本任务与 `ready` 段执行顺序的关系（说明，不是要你插代码）**：`ready` 段的四个有序子锚点 `ready.registry` → `ready.patches` → `ready.rollbus` → `ready.cards` 由各自的内核属主任务填，本任务**一行都不插**。本任务的两条修复也**不读任何 registry 绑定**（三张表按不可变 `_id` 直取，见下面的 §7 例外），所以 `resolveAll` 早于 `applyAll` 这条顺序对本任务不是承重的；本任务只要求 `applyAll()` 会被调用一次。

**跑不动单测的部分 → 由谁兜底**（这张表是给复核的人看的：少掉一条 selftest 条目就会在这里露出一行没有对应物的断言）：

| 无法单测的断言 | 兜底 |
|---|---|
| 三张表在真实世界里被写对了 | selftest `data.ev-oracle-tables` + 手工验证 2 |
| 二元神谕修好后能抽到 Strong yes | 手工验证 3 |
| d66 卡上不再一次冒出两个结果 | 手工验证 4 |
| 幂等（重进世界不再弹框） | 手工验证 5 |
| 三个 NPC 字段被写对了 | selftest `data.npc-roster` + 手工验证 7 |
| 改 `type` 后 actor 数据没丢 | 手工验证 8（无法单测：需要真实 DataModel 重新校验） |
| `importAdventure` 之后自动重跑 | 手工验证 9（无法单测：桩没有真实 Adventure 导入时序） |
| 监听只挂一次、可摘除 | `test/repair-reimport-guard.test.mjs`（桩的 `Hooks` 真派发、`Hooks.off` 可用） |
| 开关关掉后监听原样放行 | 两个测试文件里的 `onAdventureImported` 门测试 + 手工验证 11 |
| 记账写进 `dataRepairs` 世界设置 | `test/repair-log.test.mjs`（桩的设置后端是真实现）+ 手工验证 5 |
| 玩家端完全不动手 | 手工验证 10 |
| 真实 `DialogV2` 交互 | 手工验证 1（桩只提供 `foundry.applications.api.{ApplicationV2, HandlebarsApplicationMixin}`，没有 `DialogV2`） |

**前置依赖（跑测试前先确认）**：本任务的登记形状测试会 `import` `scripts/kernel/features.mjs`、`patches.mjs`、`selftest.mjs`、`resolver.mjs`、`scripts/const.mjs` 与 `test/stubs/foundry.mjs`。这六个文件由别的任务交付。如果你的工作树里还没有它们，相关用例会以 `Failed to load url ../scripts/kernel/features.mjs` 之类的信息失败 —— 那是任务合并顺序问题，不是本任务的缺陷：先把内核任务合进来再跑。前 18 步（两个 `.pure.mjs` 与它们的单测）没有任何依赖，任何时候都能跑。

---

#### 背景：Foundry 概念（读者不需要懂 Foundry，这里够用了）

- **RollTable / TableResult**：随机表文档。`formula` 是掷骰式（`"1d6"`、`"10*1d6+1d6"`）；每一行是一个嵌入文档 `TableResult`，`range` 是 `[下界, 上界]`。抽表时 Foundry 返回**所有**满足 `range[0] <= 点数 <= range[1]` 的行 —— **范围重叠就会一次返回多行**，卡上于是出现两个答案。
- **d66**：掷两颗 d6，第一颗当十位、第二颗当个位，所以只有 36 个可能值：11–16、21–26、31–36、41–46、51–56、61–66。表里写 `"10*1d6+1d6"` 就是这个意思。
- **文档 `_id`**：16 位随机串，文档一辈子不变。合集包（compendium）里的文档被导入世界时带 `keepId: true`（见 `systems/alienrpg/module/apps/init.mjs:166`），所以世界里的表和包里的表是同一个 `_id`。**按 `_id` 找文档是稳定的，按显示名找不是**（Babele 之类的汉化层会把名字换掉）。
- **Adventure（冒险包）与再导入**：`systems/alienrpg/module/apps/init.mjs:48` 的 `Hooks.on("ready")` 判断 —— 首次进世界就跑 `FirstTimeSetup()` 全量导入。设置里的 "Re-Import"（`ReImport()`，`init.mjs:135`）只**创建缺失**的文档（`partition(d => collection.has(d._id))` 取补集再 `createDocuments(..., {keepId:true, keepEmbeddedId:true})`，`init.mjs:166`），**不覆盖已有的**；但从合集里手动跑完整的 Adventure Importer **会覆盖**。所以修复必须能重复施加，并在 `importAdventure` 之后自动重跑。
- **`importAdventure` 钩子**：Foundry 在一个 Adventure 导入完成后触发的**领域钩子**（不是生命周期钩子）。系统自己也挂了它（`systems/alienrpg/module/apps/init.mjs:101`，写作 `(created, updated) => {...}`，而 Foundry 核心实际传的是 `(adventure, formData, created, updated)`）—— 这正是我们的回调**一个参数都不接**的原因：这个钩子的入参列表在版本间漂移过，而「把整个计划重跑一遍」一个参数都不需要。
- **`game.settings.register(module, key, {scope:"world"})`**：世界级设置，只有 GM 能写。用它记录「已修到哪一版」。**设置只能在 `init` 生命周期钩子里注册。**
- **生命周期**：Foundry 依次触发 `init` → `i18nInit` → `setup` → `ready`。`game.i18n` 到 `i18nInit` 才就绪，所以 `init` 阶段调 `game.i18n.localize()` 只会拿回键名本身 —— 这就是 selftest 的 `label` 存**键**而不是存文本的原因。

#### 你要修的数据事实（用 `classic-level` 直接读 `modules/alien-evolved-corerules/packs/alien-evolved-core-rules` 这个 LevelDB 逐条核实过；模组版本 `1.0.2`，只有一个 `Adventure` 类型的包）

**A. 三张神谕表**

| 文档 | 现状 | 应为 | 后果 |
|---|---|---|---|
| RollTable `6HcfWwkEJ1Y0KzMy`「EV - 50. LS - BINARY RESPONSE MATRIX」 | `formula: "1d4"`，四行 `[1,1]"Strong no"` / `[2,3]"No"` / `[4,5]"Yes"` / `[6,6]"Strong yes"` | `formula: "1d6"` | 1d4 只出 1–4：Strong no 25%、No 50%、Yes 25%、**Strong yes 概率为 0**；「否」合计 75% |
| RollTable `dnm74JBwOH9AMQ6v`「EV - 52. LS - SECTION MATRIX」的 TableResult `c9g3UmwQpBCBjYDE`（"Cryosleep"） | `range: [25, 266]` | `range: [25, 26]` | 36 个 d66 结果里有 **24 个**（31 及以上）会同时命中这一行，卡上永远多出一个 Cryosleep |
| RollTable `YnhWA69wqLLp2jsB`「EV - 53. LS - ACCESS MATRIX」的 TableResult `RfiCDLTS1aVIP3d1`（"Stairs/Ladder"） | `range: [31, 362]` | `range: [31, 36]` | 41 及以上共 **18 个**（正好一半）会多出一个 Stairs/Ladder |

同表里 `0wZOQbZwu0RUQDG4`（"Corridor"）的 `[11, 26]` 是**合法的宽行**（覆盖 11–16 与 21–26 共 12 个结果），绝不能一起「修」掉 —— 这就是为什么计划要按「上界恰好等于 266 / 362」来定位，而不是按「上界大于 66」。

整个包的 `formula` 直方图：`1d6` 50 张、`10*1d6+1d6` 45 张、`2d6` 19 张、`3d6` 2 张、`1d1` 5 张、`1D6` 1 张、**`1d4` 恰好 1 张** —— 就是那张二元神谕。所以「formula 恰好是 1d4」是一个零误报的探针。

**B. NPC 名录 —— 权威出处在包内**

JournalEntry `MZu96EjrzlxhTRXc`「Alien Evolved GM Guide」的页面 `TGaKQLUMk69avd5b`「11. CAMPAIGN  PLAY」（名字中间是两个空格）逐行印着 `NPC | Attributes | Health | Skills | Talents | Gear`。下面每一条都对着这张表核过。

| Actor `_id` | 名字 | 现状 | 表上印的 | 判定 |
|---|---|---|---|---|
| `ZCfDvo9HjdAzpnNe` | EV - COLONY MANAGER | agl 3 / wit 4 | Agility **4**, Wits **3**（其余 str 2、emp 5、health 3、六项技能、三件装备逐项吻合） | **修**：两项对调了。旁证：包里存着的派生快照 `agl.mod = 4`、`wit.mod = 3`，与 `value` 恰好反着，而这个 actor 身上没有任何改属性的物品 |
| `DukQ40yWrGH8jmKP` | EV - SQUAD LEADER | `skills.command.value` 1 | Command **2**（其余 str 5/agl 3/wit 3/emp 3、health 4、另四项技能、天赋 Field Commander、三件装备逐项吻合） | **修**：只有这一项不符 |
| `881anZR8zz4RnlB2` | EV - ANDROID, COVERT | `type: "character"`，`system.header.npc: false` | 属性、生命、五项技能与表上完全吻合 | **修**：同包的两个兄弟 `9SL7d6ixOihyqwwL`「ANDROID, CURIOUS」与 `3R9Y97rfqLb0MdHv`「ANDROID, REFURBISHED」都是 `type:"synthetic"` / `npc:true`；`synthetic` 是 `systems/alienrpg/system.json:49` 声明过的合法 actor 类型；而且它是全名录里唯一 `npc:false` 的 NPC |
| `IxstyK7Y6gwdaiI0` | EV - COMPANY BUREAUCRAT | str 2 / agl 3 / wit 5 / emp 4；observation 3、command 3、manipulation 4；天赋 Cunning；装备 手电、磁带录音机、PDT | 表上的 **Corporate executive** 一行是：Strength 2, Agility 3, Wits 5, Empathy 4 / 3 / Observation 3, Command 3, Manipulation 4 / **Cunning** / Penlight, PDT, voice recorder | **不修**：这不是几个字段打错，而是**整行装错了人** —— 出厂的「公司官僚」身上是「企业主管」那一行的属性、技能、天赋和装备。真正的 Company bureaucrat 是 Wits 4 / Empathy 5、Observation 3, Comtech 3, Manipulation 2, Medical Aid 2、天赋 Personal Safety。逐字段去「修」只会造出一个两边都不像的杂交体 |

**`type` 变更为什么可控（本轮在系统源码里逐项核过）**：`character` 与 `synthetic` 两个 DataModel 的顶层字段完全一致（`header` / `attributes` / `skills` / `general` / `consumables` / `adhocitems`，见 `module/data/actor-character.mjs:12,64,85,109,254,285` 与 `module/data/actor-synthetic.mjs:12,65,86,110,250,281`）；`attributes` 与 `skills` 两边都是由**同一份** `CONFIG.ALIENRPG.attributes` / `CONFIG.ALIENRPG.skills` 归约生成的（`actor-character.mjs:65/86`、`actor-synthetic.mjs:66/87`），键集按构造相同；`header` 两边都有 `health{value,max,mod,calculatedMax,label}`、`stress`、`resolve`、`npc`，`synthetic` 只**多**一个布尔 `synthstress`（`actor-synthetic.mjs:61`）。所以这次 `type` 变更预期是无损的（多出来的字段取初值 `false`）。**预期无损不等于已验证**：它仍然单独隔离、最后施加、在确认框里标红，并且手工验证 8 给了回退方案。

**范围纪律**：参考数据只收录上表判定为「修」的三条。公司官僚记在参考数据的 `notRepaired` 段里（只是数据，不参与差异计算），这样以后有人重读这份文件时不会好心把它加回去；它的正解是上游整条替换，见文末的 UPSTREAM PR。**宁可漏修，不可误修。**

---

- [ ] **Step 1：写第一个失败测试 —— 神谕修复计划**

新建 `test/repair-ev-oracle-tables.test.mjs`。夹具用的是从包里读出来的真实值（只保留与判定相关的行）：

```js
import { describe, expect, it } from "vitest";
import { pureOraclePlan, pureOracleTableIds } from "../scripts/repairs/ev-oracle-tables.pure.mjs";

const binary = {
  _id: "6HcfWwkEJ1Y0KzMy",
  formula: "1d4",
  results: [
    { _id: "PT5VjLPIEgGpASuQ", range: [1, 1] },
    { _id: "Oi6yEIb9F5XycNaK", range: [2, 3] },
    { _id: "jdUqqQtFlHZam4H3", range: [4, 5] },
    { _id: "AaPCoHJprUcnE8Ul", range: [6, 6] },
  ],
};
const section = {
  _id: "dnm74JBwOH9AMQ6v",
  formula: "10*1d6+1d6",
  results: [
    { _id: "Ol8H2fzmg5iCErzk", range: [21, 22] },
    { _id: "D8uWWQ1GsqGlIIwZ", range: [23, 24] },
    { _id: "c9g3UmwQpBCBjYDE", range: [25, 266] },
    { _id: "6QqEoCLaqHyit3Rr", range: [31, 32] },
  ],
};
const access = {
  _id: "YnhWA69wqLLp2jsB",
  formula: "10*1d6+1d6",
  results: [
    { _id: "0wZOQbZwu0RUQDG4", range: [11, 26] },
    { _id: "RfiCDLTS1aVIP3d1", range: [31, 362] },
    { _id: "doHLsnChDpz8ELGu", range: [41, 43] },
  ],
};

describe("pureOraclePlan", () => {
  it("changes the binary oracle's formula from 1d4 to 1d6", () => {
    expect(pureOraclePlan([binary])).toEqual([
      { tableId: "6HcfWwkEJ1Y0KzMy", kind: "formula", from: "1d4", to: "1d6" },
    ]);
  });

  it("clamps only the row whose upper bound is the typo'd 266", () => {
    expect(pureOraclePlan([section])).toEqual([
      { tableId: "dnm74JBwOH9AMQ6v", kind: "range", resultId: "c9g3UmwQpBCBjYDE", from: [25, 266], to: [25, 26] },
    ]);
  });

  it("clamps only the row whose upper bound is the typo'd 362", () => {
    expect(pureOraclePlan([access])).toEqual([
      { tableId: "YnhWA69wqLLp2jsB", kind: "range", resultId: "RfiCDLTS1aVIP3d1", from: [31, 362], to: [31, 36] },
    ]);
  });

  it("leaves the legitimately wide 11-26 Corridor row alone", () => {
    expect(pureOraclePlan([access]).some((p) => p.resultId === "0wZOQbZwu0RUQDG4")).toBe(false);
  });

  it("is idempotent: an already-repaired pack yields an empty plan", () => {
    const fixed = [
      { ...binary, formula: "1d6" },
      { ...section, results: section.results.map((r) => (r._id === "c9g3UmwQpBCBjYDE" ? { ...r, range: [25, 26] } : r)) },
      { ...access, results: access.results.map((r) => (r._id === "RfiCDLTS1aVIP3d1" ? { ...r, range: [31, 36] } : r)) },
    ];
    expect(pureOraclePlan(fixed)).toEqual([]);
  });

  it("ignores tables it does not own, whatever their formula", () => {
    expect(pureOraclePlan([{ _id: "someOtherTable", formula: "1d4", results: [] }])).toEqual([]);
  });

  it("does not throw when a target result id is absent", () => {
    expect(() => pureOraclePlan([{ _id: "dnm74JBwOH9AMQ6v", formula: "10*1d6+1d6", results: [] }])).not.toThrow();
  });

  it("pureOracleTableIds lists exactly the three target tables", () => {
    expect(pureOracleTableIds().sort()).toEqual(
      ["6HcfWwkEJ1Y0KzMy", "YnhWA69wqLLp2jsB", "dnm74JBwOH9AMQ6v"].sort()
    );
  });
});
```

- [ ] **Step 2：跑一遍，看它失败**

Run: `npx vitest run test/repair-ev-oracle-tables.test.mjs`

Expected: FAIL —— `Error: Failed to load url ../scripts/repairs/ev-oracle-tables.pure.mjs (resolved id: ...). Does the file exist?`

- [ ] **Step 3：写出 `ev-oracle-tables.pure.mjs`**

新建 `scripts/repairs/ev-oracle-tables.pure.mjs`。这个文件**不引用任何 Foundry 全局**，vitest 直接跑它。

```js
/**
 * Pure layer of the `ev-oracle-tables` data repair.
 *
 * Nothing in this file touches a Foundry global, so vitest runs it directly
 * against plain object literals.
 *
 * Every target is addressed by document id and by the exact wrong value, never
 * by display name: a translation layer can rename any of these tables, and
 * matching on the wrong value is what makes the repair idempotent and
 * self-retiring. Once a value is right — whether we fixed it or an upstream
 * corerules release did — the plan for it is empty.
 */

const ORACLE_TARGETS = {
  // "EV - 50. LS - BINARY RESPONSE MATRIX": four rows spread across 1..6 but a
  // 1d4 formula, so "Strong yes" (row [6,6]) can never come up and the two
  // negative rows take 75% of the probability mass. Across the whole shipped
  // pack "1d4" occurs exactly once, on this table, so this guard cannot
  // misfire on a table we do not own.
  "6HcfWwkEJ1Y0KzMy": { formula: { from: "1d4", to: "1d6" } },

  // "EV - 52. LS - SECTION MATRIX", row "Cryosleep": a d66 upper bound typed as
  // 266 instead of 26, so all 24 outcomes of 31 or more also hit this row.
  dnm74JBwOH9AMQ6v: { ranges: { c9g3UmwQpBCBjYDE: { from: 266, to: 26 } } },

  // "EV - 53. LS - ACCESS MATRIX", row "Stairs/Ladder": 362 instead of 36, so
  // all 18 outcomes of 41 or more also hit this row. The neighbouring
  // "Corridor" row really is [11,26] and must be left alone — which is why we
  // match on the exact wrong bound rather than on "upper bound above 66".
  YnhWA69wqLLp2jsB: { ranges: { RfiCDLTS1aVIP3d1: { from: 362, to: 36 } } },
};

/** @returns {string[]} the document ids this repair owns. */
export function pureOracleTableIds() {
  return Object.keys(ORACLE_TARGETS);
}

/**
 * @param {Array<{_id:string, formula:string, results:Array<{_id:string, range:number[]}>}>} tables
 * @returns {Array<{tableId:string, kind:"formula"|"range", resultId?:string, from:any, to:any}>}
 */
export function pureOraclePlan(tables) {
  const plan = [];
  for (const table of tables ?? []) {
    const spec = ORACLE_TARGETS[table?._id];
    if (!spec) continue;

    if (spec.formula && table.formula === spec.formula.from) {
      plan.push({ tableId: table._id, kind: "formula", from: spec.formula.from, to: spec.formula.to });
    }

    for (const [resultId, bound] of Object.entries(spec.ranges ?? {})) {
      const result = (table.results ?? []).find((row) => row?._id === resultId);
      if (!result) continue; // absent row: nothing to plan, the caller stays quiet
      if (result.range?.[1] !== bound.from) continue;
      plan.push({
        tableId: table._id,
        kind: "range",
        resultId,
        from: [result.range[0], result.range[1]],
        to: [result.range[0], bound.to],
      });
    }
  }
  return plan;
}
```

- [ ] **Step 4：跑一遍，看它通过**

Run: `npx vitest run test/repair-ev-oracle-tables.test.mjs`

Expected: `8 passed`。

- [ ] **Step 5：写第二个失败测试 —— 补丁探针与自检断言（两个方向都测）**

`probe()` 和自检条目都必须是「一个可注入的纯谓词的薄壳」，而且**两个方向都要有夹具**，否则一个永远返回 true 的探针会在上游修好之后重复施加。把文件顶部的 import 改为：

```js
import {
  pureOraclePlan,
  pureOracleTableIds,
  pureOracleTablesAreBuggy,
  pureOracleTablesPresent,
} from "../scripts/repairs/ev-oracle-tables.pure.mjs";
```

并追加到文件末尾：

```js
describe("oracle predicates", () => {
  it("pureOracleTablesPresent is true when at least one target table is in the world", () => {
    expect(pureOracleTablesPresent([access])).toBe(true);
  });

  it("pureOracleTablesPresent is false for a world that has none of them", () => {
    expect(pureOracleTablesPresent([])).toBe(false);
    expect(pureOracleTablesPresent([{ _id: "someOtherTable", formula: "1d6", results: [] }])).toBe(false);
  });

  it("pureOracleTablesAreBuggy is true for the shipped 1.0.2 data", () => {
    expect(pureOracleTablesAreBuggy([binary, section, access])).toBe(true);
  });

  it("pureOracleTablesAreBuggy is false once every value is right", () => {
    const fixed = [
      { ...binary, formula: "1d6" },
      { ...section, results: section.results.map((r) => (r._id === "c9g3UmwQpBCBjYDE" ? { ...r, range: [25, 26] } : r)) },
      { ...access, results: access.results.map((r) => (r._id === "RfiCDLTS1aVIP3d1" ? { ...r, range: [31, 36] } : r)) },
    ];
    expect(pureOracleTablesAreBuggy(fixed)).toBe(false);
  });
});
```

- [ ] **Step 6：跑一遍，看它失败**

Run: `npx vitest run test/repair-ev-oracle-tables.test.mjs`

Expected: FAIL —— `SyntaxError: The requested module '/scripts/repairs/ev-oracle-tables.pure.mjs' does not provide an export named 'pureOracleTablesAreBuggy'`（整个文件加载失败，8 条已通过的用例也一并变红）。

- [ ] **Step 7：补上两个谓词**

追加到 `scripts/repairs/ev-oracle-tables.pure.mjs`：

```js
/**
 * The patch probe: "should this repair stand guard in this world?" — i.e. is
 * the shipped corerules pack imported here at all. It deliberately does NOT
 * ask whether the data is currently wrong: an adventure re-import can make it
 * wrong again mid-session, so the repair must stay armed in a world we already
 * repaired. Whether the data is right *now* is the selftest's question,
 * answered by pureOracleTablesAreBuggy below.
 *
 * @param {Array<{_id:string}>} tables
 */
export function pureOracleTablesPresent(tables) {
  return (tables ?? []).some((table) => Object.hasOwn(ORACLE_TARGETS, table?._id ?? ""));
}

/** The selftest assertion: is any owned value still wrong? */
export function pureOracleTablesAreBuggy(tables) {
  return pureOraclePlan(tables).length > 0;
}
```

- [ ] **Step 8：跑一遍，看它通过**

Run: `npx vitest run test/repair-ev-oracle-tables.test.mjs`

Expected: `12 passed`。

- [ ] **Step 9：提交**

```bash
git add scripts/repairs/ev-oracle-tables.pure.mjs test/repair-ev-oracle-tables.test.mjs && git commit -m "$(cat <<'EOF'
fix(repairs): 三张 EV 神谕表的修复计划（纯层、幂等、探针两个方向都可测）

用 classic-level 直接读 alien-evolved-corerules 的 LevelDB 逐条核实：
- 6HcfWwkEJ1Y0KzMy「EV-50 二元回应矩阵」formula 是 1d4，四行却铺到 6，
  于是「否」占 75%、Strong yes 概率为 0。整个包里 1d4 恰好只此一张。
- dnm74JBwOH9AMQ6v 的 c9g3UmwQpBCBjYDE（Cryosleep）range 是 [25,266]
  （应为 [25,26]），36 个 d66 结果里有 24 个会多出一行。
- YnhWA69wqLLp2jsB 的 RfiCDLTS1aVIP3d1（Stairs/Ladder）range 是 [31,362]
  （应为 [31,36]），正好一半的结果会多出一行。
同表的 0wZOQbZwu0RUQDG4（Corridor）[11,26] 是合法宽行，所以计划按
「上界恰好等于 266/362」定位，而不是按「上界大于 66」。

计划只按文档 id 与「恰好等于错误值」定位，所以可以反复施加：
一旦值已正确（无论是我们改的还是 corerules 更新改的），计划就是空的。
探针拆成两个纯谓词，present 决定该不该布防、areBuggy 供自检断言，
两个方向各有夹具。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 10：写第三个失败测试 —— NPC 名录差异**

新建 `test/repair-npc-roster.test.mjs`：

```js
import { describe, expect, it } from "vitest";
import {
  pureRosterActorIds,
  pureRosterDiff,
  pureRosterReference,
} from "../scripts/repairs/npc-roster.pure.mjs";

// Exactly as the pack ships it: Agility and Wits are swapped.
const colonyManager = {
  _id: "ZCfDvo9HjdAzpnNe",
  name: "EV - COLONY MANAGER",
  type: "character",
  system: {
    attributes: { str: { value: 2 }, agl: { value: 3 }, wit: { value: 4 }, emp: { value: 5 } },
    skills: { command: { value: 3 } },
  },
};

const covertAndroid = {
  _id: "881anZR8zz4RnlB2",
  name: "EV - ANDROID, COVERT",
  type: "character",
  system: { header: { npc: false } },
};

describe("pureRosterDiff", () => {
  it("reports the swapped Agility and Wits on the colony manager", () => {
    expect(pureRosterDiff(colonyManager)).toEqual([
      { path: "system.attributes.agl.value", from: 3, to: 4, risky: false },
      { path: "system.attributes.wit.value", from: 4, to: 3, risky: false },
    ]);
  });

  it("is idempotent: a repaired colony manager yields no diff", () => {
    const fixed = {
      ...colonyManager,
      system: {
        ...colonyManager.system,
        attributes: { str: { value: 2 }, agl: { value: 4 }, wit: { value: 3 }, emp: { value: 5 } },
      },
    };
    expect(pureRosterDiff(fixed)).toEqual([]);
  });

  it("puts the covert android's npc flag first and its type change last, flagged risky", () => {
    const d = pureRosterDiff(covertAndroid);
    expect(d[0]).toEqual({ path: "system.header.npc", from: false, to: true, risky: false });
    expect(d.at(-1)).toEqual({ path: "type", from: "character", to: "synthetic", risky: true });
  });

  it("returns nothing for an actor the reference does not know", () => {
    expect(pureRosterDiff({ _id: "notInRoster", system: {} })).toEqual([]);
  });

  it("never invents a field the reference does not list", () => {
    const listed = Object.keys(pureRosterReference().entries.ZCfDvo9HjdAzpnNe.fields);
    expect(pureRosterDiff(colonyManager).every((d) => listed.includes(d.path))).toBe(true);
  });

  it("does not offer to repair the company bureaucrat, which is recorded as notRepaired", () => {
    const bureaucrat = {
      _id: "IxstyK7Y6gwdaiI0",
      name: "EV - COMPANY BUREAUCRAT",
      type: "character",
      system: { attributes: { wit: { value: 5 } }, skills: { comtech: { value: 0 } } },
    };
    expect(pureRosterDiff(bureaucrat)).toEqual([]);
    expect(pureRosterReference().notRepaired.IxstyK7Y6gwdaiI0).toBeDefined();
  });

  it("pureRosterActorIds lists exactly the three repairable actors", () => {
    expect(pureRosterActorIds().sort()).toEqual(
      ["881anZR8zz4RnlB2", "DukQ40yWrGH8jmKP", "ZCfDvo9HjdAzpnNe"].sort()
    );
  });
});
```

- [ ] **Step 11：跑一遍，看它失败**

Run: `npx vitest run test/repair-npc-roster.test.mjs`

Expected: FAIL —— `Error: Failed to load url ../scripts/repairs/npc-roster.pure.mjs (resolved id: ...). Does the file exist?`

- [ ] **Step 12：写出 `npc-roster.pure.mjs` 的参考数据与差异计算**

新建 `scripts/repairs/npc-roster.pure.mjs`。参考数据直接写在这个纯文件里（而不是一个 `.json`），这样既不必依赖 import attributes 的语法可用性，又能让 `pure*` 之外没有别的导出。

```js
/**
 * Pure layer of the `npc-roster` data repair.
 *
 * Every "should be" below was checked against a source that ships inside the
 * same module: JournalEntry MZu96EjrzlxhTRXc "Alien Evolved GM Guide", page
 * TGaKQLUMk69avd5b "11. CAMPAIGN  PLAY" (two spaces in that name), which
 * reprints the NPC table as "NPC | Attributes | Health | Skills | Talents |
 * Gear". Nothing here rests on a book I cannot open.
 */

const ROSTER_REFERENCE = {
  sourceModule: "alien-evolved-corerules",
  sourceVersion: "1.0.2",
  sourceDocument: "JournalEntry MZu96EjrzlxhTRXc / page TGaKQLUMk69avd5b",

  entries: {
    // Printed: Strength 2, Agility 4, Wits 3, Empathy 5 | Health 3 | Heavy
    // Machinery 1, Stamina 2, Comtech 1, Survival 1, Command 3, Manipulation 2.
    // Everything matches the shipped actor except Agility and Wits, which are
    // swapped. Corroborated inside the pack: the stored derived snapshots read
    // agl.mod = 4 and wit.mod = 3 — the printed values — and this actor carries
    // no attribute-modifying item that could explain the difference.
    ZCfDvo9HjdAzpnNe: {
      label: "EV - COLONY MANAGER",
      fields: {
        "system.attributes.agl.value": 4,
        "system.attributes.wit.value": 3,
      },
    },

    // Printed: Close Combat 2, Stamina 2, Ranged Combat 3, Survival 1,
    // Command 2. Attributes, health, the other four skills, the Field Commander
    // talent and all three gear items match; only Command is one short.
    DukQ40yWrGH8jmKP: {
      label: "EV - SQUAD LEADER",
      fields: {
        "system.skills.command.value": 2,
      },
    },

    // Attributes, health and all five skills match the printed row. What does
    // not match is the pack's own internal consistency: its two siblings —
    // 9SL7d6ixOihyqwwL "ANDROID, CURIOUS" and 3R9Y97rfqLb0MdHv "ANDROID,
    // REFURBISHED" — are both type "synthetic" with header.npc true, and
    // "synthetic" is a declared Actor type in systems/alienrpg/system.json:49.
    // This is also the only NPC in the roster shipped with npc:false, which via
    // systems/alienrpg/module/alienrpg.mjs:366-375 means every one of its tokens
    // is linked to the one base actor and they all share a single health pool.
    "881anZR8zz4RnlB2": {
      label: "EV - ANDROID, COVERT",
      fields: {
        "system.header.npc": true,
      },
      // Changing an Actor's `type` re-validates `system` against a different
      // DataModel, so it is applied last, on its own, and flagged in the dialog.
      // The two models were compared field by field and synthetic is a superset
      // of character (same header/attributes/skills/general/consumables/
      // adhocitems, plus one extra boolean header.synthstress), so this is
      // expected to be lossless — expected, not proven, hence "risky".
      riskyFields: {
        type: "synthetic",
      },
    },
  },

  /**
   * Known-wrong but deliberately NOT repaired. Data only: pureRosterDiff never
   * reads this section. It is recorded so a later reader does not helpfully
   * promote it into `entries`.
   */
  notRepaired: {
    IxstyK7Y6gwdaiI0: {
      label: "EV - COMPANY BUREAUCRAT",
      reason:
        "The shipped actor is not a damaged bureaucrat, it is the printed " +
        "'Corporate executive' row filed under the wrong name: Wits 5, " +
        "Empathy 4, Observation 3, Command 3, Manipulation 4, the Cunning " +
        "talent and the executive's penlight / voice recorder / PDT all match " +
        "that row exactly. The printed bureaucrat has Wits 4, Empathy 5, " +
        "Observation 3, Comtech 3, Manipulation 2, Medical Aid 2 and the " +
        "Personal Safety talent. Patching four fields would produce a hybrid " +
        "matching neither NPC, so the fix belongs upstream, as a whole-row " +
        "replacement.",
    },
  },
};

/** @returns {object} a copy of the reference dataset, safe for callers to read. */
export function pureRosterReference() {
  return structuredClone(ROSTER_REFERENCE);
}

/** @returns {string[]} the actor ids this repair is willing to write to. */
export function pureRosterActorIds() {
  return Object.keys(ROSTER_REFERENCE.entries);
}

function readPath(obj, path) {
  let cursor = obj;
  for (const segment of path.split(".")) {
    if (cursor === null || cursor === undefined) return undefined;
    cursor = cursor[segment];
  }
  return cursor;
}

/**
 * Diff one shipped roster actor against the reference.
 *
 * The reference lists only fields with an in-pack source, so an upstream
 * release that fixes an entry makes the diff shrink rather than making this
 * validator offer to "repair" a now-correct actor back to something wrong.
 *
 * @param {object} actorData a plain actor object (`actor.toObject()` at runtime)
 * @param {object} [reference] injectable for tests; defaults to the shipped one
 * @returns {Array<{path:string, from:any, to:any, risky:boolean}>}
 */
export function pureRosterDiff(actorData, reference = ROSTER_REFERENCE) {
  const entry = reference?.entries?.[actorData?._id];
  if (!entry) return [];

  const diffs = [];
  for (const [path, want] of Object.entries(entry.fields ?? {})) {
    const have = readPath(actorData, path);
    if (have === want) continue;
    diffs.push({ path, from: have, to: want, risky: false });
  }
  for (const [path, want] of Object.entries(entry.riskyFields ?? {})) {
    const have = readPath(actorData, path);
    if (have === want) continue;
    diffs.push({ path, from: have, to: want, risky: true });
  }
  return diffs;
}
```

- [ ] **Step 13：跑一遍，看它通过**

Run: `npx vitest run test/repair-npc-roster.test.mjs`

Expected: `7 passed`。

- [ ] **Step 14：写第四个失败测试 —— 名录探针、自检谓词与 HTML 转义**

把文件顶部的 import 改为：

```js
import {
  pureEscapeText,
  pureRosterActorIds,
  pureRosterActorsPresent,
  pureRosterDiff,
  pureRosterIsStale,
  pureRosterReference,
} from "../scripts/repairs/npc-roster.pure.mjs";
```

并追加到文件末尾：

```js
describe("roster predicates", () => {
  it("pureRosterActorsPresent is true when one roster actor is in the world", () => {
    expect(pureRosterActorsPresent([covertAndroid])).toBe(true);
  });

  it("pureRosterActorsPresent is false for a world that has none of them", () => {
    expect(pureRosterActorsPresent([])).toBe(false);
    expect(pureRosterActorsPresent([{ _id: "IxstyK7Y6gwdaiI0" }])).toBe(false);
  });

  it("pureRosterIsStale is true for the shipped 1.0.2 roster", () => {
    expect(pureRosterIsStale([colonyManager, covertAndroid])).toBe(true);
  });

  it("pureRosterIsStale is false once every listed field matches", () => {
    const fixedManager = {
      ...colonyManager,
      system: {
        ...colonyManager.system,
        attributes: { str: { value: 2 }, agl: { value: 4 }, wit: { value: 3 }, emp: { value: 5 } },
      },
    };
    const fixedAndroid = { ...covertAndroid, type: "synthetic", system: { header: { npc: true } } };
    expect(pureRosterIsStale([fixedManager, fixedAndroid])).toBe(false);
  });
});

describe("pureEscapeText", () => {
  it("neutralises the five HTML-significant characters", () => {
    expect(pureEscapeText(`<b>Ash & "Bishop"'s</b>`)).toBe(
      "&lt;b&gt;Ash &amp; &quot;Bishop&quot;&#39;s&lt;/b&gt;"
    );
  });

  it("turns null and undefined into an empty string", () => {
    expect(pureEscapeText(null)).toBe("");
    expect(pureEscapeText(undefined)).toBe("");
  });
});
```

- [ ] **Step 15：跑一遍，看它失败**

Run: `npx vitest run test/repair-npc-roster.test.mjs`

Expected: FAIL —— `SyntaxError: The requested module '/scripts/repairs/npc-roster.pure.mjs' does not provide an export named 'pureEscapeText'`（整个文件加载失败，7 条已通过的用例也一并变红）。

- [ ] **Step 16：补上三个函数**

追加到 `scripts/repairs/npc-roster.pure.mjs`：

```js
/**
 * The patch probe: is the shipped roster imported into this world at all? Like
 * the oracle repair, this deliberately does not ask whether the data is
 * currently wrong — an adventure re-import can overwrite a repaired actor
 * mid-session, so the repair must stay armed in an already-repaired world too.
 *
 * @param {Array<{_id:string}>} actorDatas
 */
export function pureRosterActorsPresent(actorDatas) {
  return (actorDatas ?? []).some((data) => Object.hasOwn(ROSTER_REFERENCE.entries, data?._id ?? ""));
}

/** The selftest assertion: does any listed field still disagree? */
export function pureRosterIsStale(actorDatas, reference = ROSTER_REFERENCE) {
  return (actorDatas ?? []).some((data) => pureRosterDiff(data, reference).length > 0);
}

/**
 * Escape text for interpolation into the confirmation dialog's HTML. Actor
 * names are world data and a GM can rename them to anything at all, so they are
 * never dropped into markup raw.
 */
export function pureEscapeText(value) {
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return String(value ?? "").replace(/[&<>"']/g, (character) => map[character]);
}
```

- [ ] **Step 17：跑一遍，看它通过**

Run: `npx vitest run test/repair-npc-roster.test.mjs`

Expected: `13 passed`。

- [ ] **Step 18：提交**

```bash
git add scripts/repairs/npc-roster.pure.mjs test/repair-npc-roster.test.mjs && git commit -m "$(cat <<'EOF'
feat(repairs): NPC 名录参考数据与差异计算（纯层）

权威出处在包内：JournalEntry MZu96EjrzlxhTRXc「Alien Evolved GM Guide」
的页面 TGaKQLUMk69avd5b「11. CAMPAIGN  PLAY」逐行印着 NPC 总表。
按它核对后只收录三条：
- 殖民地管理者 agl 与 wit 对调（包内旁证：存下来的 agl.mod=4 / wit.mod=3
  正是印刷值，而它身上没有任何改属性的物品）；
- 班长 command 1→2（其余属性、技能、天赋、装备逐项吻合）；
- 潜伏仿生人 header.npc false→true，type character→synthetic
  （同包两个兄弟仿生人都是 synthetic+npc:true，synthetic 也是 system.json
  里声明过的合法类型；npc:false 让它的每个 token 共享同一条血量）。

公司官僚移出修复范围并记进 notRepaired：出厂那个 actor 根本不是坏掉的
「公司官僚」，而是印刷版「企业主管」整行装错了人（属性、三项技能、Cunning
天赋、手电/录音机/PDT 全对得上企业主管）。逐字段去修只会造出两边都不像的
杂交体，正解是上游整条替换。宁可漏修，不可误修。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 19：写第五个失败测试 —— 记账模块**

`dataRepairs` 这个世界设置的**唯一属主**是 `scripts/repairs/repair-log.mjs`：别的修复一律经 `repairLog.record()` 记账，**不得**各自 `game.settings.register` 这个键，也**不得**写 `game.settings.settings.has()` 守卫（守卫会把「两个模块都来认领同一个键」这种计划缺陷悄悄吃掉，而不是让它暴露）。共享测试桩的设置后端是真实现（`register` 播种默认值、`get` 读未注册键抛错、`set` 返回 Promise），所以这一段有真单测，不靠手工验证。

新建 `test/repair-log.test.mjs`：

```js
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";
import { MID, SETTING_DATA_REPAIRS } from "../scripts/const.mjs";
import { DATA_REPAIR_VERSION, repairLog } from "../scripts/repairs/repair-log.mjs";

const KEY = `${MID}.${SETTING_DATA_REPAIRS}`;

describe("repairLog", () => {
  beforeEach(() => installFoundryStub());
  afterEach(() => uninstallFoundryStub());

  it("registers dataRepairs as a hidden world setting defaulting to an empty object", () => {
    repairLog.registerSettings();
    expect(game.settings.settings.has(KEY)).toBe(true);
    const def = game.settings.settings.get(KEY);
    expect(def.scope).toBe("world");
    expect(def.config).toBe(false);
    expect(game.settings.get(MID, SETTING_DATA_REPAIRS)).toEqual({});
  });

  it("cannot record before the setting is registered", async () => {
    await expect(repairLog.record("ev-oracle-tables", { count: 1 })).rejects.toThrow();
  });

  it("stamps version, timestamp, author and count", async () => {
    repairLog.registerSettings();
    const before = Date.now();
    await repairLog.record("ev-oracle-tables", { count: 3 });
    const entry = game.settings.get(MID, SETTING_DATA_REPAIRS)["ev-oracle-tables"];
    expect(entry.version).toBe(DATA_REPAIR_VERSION);
    expect(entry.count).toBe(3);
    expect(entry.by).toBe(game.user.id);
    expect(entry.at).toBeGreaterThanOrEqual(before);
  });

  it("keeps sibling entries when a second repair records", async () => {
    repairLog.registerSettings();
    await repairLog.record("ev-oracle-tables", { count: 3 });
    await repairLog.record("npc-roster", { version: 2, at: 42, by: "u1", count: 4 });
    const log = game.settings.get(MID, SETTING_DATA_REPAIRS);
    expect(Object.keys(log).sort()).toEqual(["ev-oracle-tables", "npc-roster"]);
    expect(log["ev-oracle-tables"].count).toBe(3);
    expect(log["npc-roster"]).toEqual({ version: 2, at: 42, by: "u1", count: 4 });
  });
});
```

- [ ] **Step 20：跑一遍，看它失败**

Run: `npx vitest run test/repair-log.test.mjs`

Expected: FAIL —— `Error: Failed to load url ../scripts/repairs/repair-log.mjs (resolved id: ...). Does the file exist?`

- [ ] **Step 21：写记账模块**

新建 `scripts/repairs/repair-log.mjs`：

```js
import { MID, SETTING_DATA_REPAIRS } from "../const.mjs";

/**
 * Bookkeeping for the shipped-data repairs.
 *
 * This module is the ONE owner of the world setting `dataRepairs`
 * (SETTING_DATA_REPAIRS). No other repair may call
 * `game.settings.register(MID, SETTING_DATA_REPAIRS, ...)`, and no one — this
 * module included — guards that call with `game.settings.settings.has(...)`:
 * a guard would silently absorb a second module claiming the same key instead
 * of making the double claim blow up where a reviewer can see it. Everyone else
 * records through `repairLog.record()`.
 *
 * Foundry only accepts setting registration during the `init` lifecycle hook,
 * so `registerSettings()` is called exactly once, from scripts/main.mjs at the
 * `AEA-ANCHOR: init` anchor.
 */

/** Bump when a repair's target values change, so a stamp tells you what ran. */
export const DATA_REPAIR_VERSION = 1;

export const repairLog = {
  registerSettings() {
    game.settings.register(MID, SETTING_DATA_REPAIRS, {
      scope: "world", // world-scoped: only a GM may write it
      config: false, // bookkeeping, not a knob; never shown in the settings sheet
      type: Object,
      default: {},
    });
  },

  /**
   * Stamp one repair's run into the world log.
   *
   * @param {string} id repair id, e.g. "ev-oracle-tables"
   * @param {{version?:number, at?:number, by?:string, count?:number}} [stamp]
   * @returns {Promise<{version:number, at:number, by:string, count:number}>}
   */
  async record(id, { version = DATA_REPAIR_VERSION, at = Date.now(), by = game.user.id, count = 0 } = {}) {
    const log = { ...(game.settings.get(MID, SETTING_DATA_REPAIRS) ?? {}) };
    log[id] = { version, at, by, count };
    await game.settings.set(MID, SETTING_DATA_REPAIRS, log);
    return log[id];
  },
};
```

- [ ] **Step 22：跑一遍，看它通过**

Run: `npx vitest run test/repair-log.test.mjs`

Expected: `4 passed`。

- [ ] **Step 23：提交**

```bash
git add scripts/repairs/repair-log.mjs test/repair-log.test.mjs && git commit -m "$(cat <<'EOF'
feat(repairs): 出厂数据修复的记账模块（dataRepairs 世界设置的唯一属主）

registerSettings() 由 main.mjs 在 init 锚点调一次；其它修复一律经
repairLog.record(id, {version, at, by, count}) 记账，不得各自 register
这个键，也不写 settings.has() 守卫——守卫会把「两个模块认领同一个键」
这种计划缺陷悄悄吃掉，而不是让它暴露。

共享测试桩的设置后端是真实现（register 播种默认值、get 读未注册键抛错、
set 返回 Promise），所以这一层有真单测：注册形状、未注册即记账会抛、
版本/时间/作者/条数四个字段、以及第二条记账不冲掉第一条。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 24：写第六个失败测试 —— `importAdventure` 监听的幂等安装器**

契约的通则是「特性与修复不得自己挂钩子」，**唯一例外**是 `type: "HOOK"` 的**已登记补丁**可以在自己的 `apply()` 里挂**领域钩子**（内核独占的三组 —— 聊天消息创建、Dice So Nice 完成、聊天卡渲染 —— 与五个生命周期钩子除外）。用这条例外要满足三件事：**只挂一次**、**钩子名写进补丁 def 的 `target` 供 `status()` 展示**、**回调第一行查 `features.enabled(id)`**。这个安装器负责「只挂一次」和「句柄可摘除」，两条修复共用它。

新建 `test/repair-reimport-guard.test.mjs`：

```js
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";
import { makeReimportGuard } from "../scripts/repairs/reimport-guard.mjs";

describe("makeReimportGuard", () => {
  let ctx;
  beforeEach(() => {
    ctx = installFoundryStub();
  });
  afterEach(() => uninstallFoundryStub());

  const importListeners = () => ctx.hooks.on.filter((row) => row.name === "importAdventure");

  it("hooks importAdventure once however many times install() is called", () => {
    const guard = makeReimportGuard(() => {});
    const first = guard.install();
    expect(guard.install()).toBe(first);
    expect(importListeners()).toHaveLength(1);
    expect(importListeners()[0].once).toBeFalsy();
  });

  it("dispatches the handler when the hook fires", () => {
    const handler = vi.fn();
    makeReimportGuard(handler).install();
    Hooks.callAll("importAdventure");
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it("retire() takes the handle back off, and install() may then re-arm", () => {
    const handler = vi.fn();
    const guard = makeReimportGuard(handler);
    guard.install();
    expect(guard.retire()).toBe(true);
    expect(guard.hookId).toBeNull();
    Hooks.callAll("importAdventure");
    expect(handler).not.toHaveBeenCalled();
    guard.install();
    Hooks.callAll("importAdventure");
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it("retire() on an unarmed guard is a safe no-op", () => {
    expect(makeReimportGuard(() => {}).retire()).toBe(false);
  });
});
```

- [ ] **Step 25：跑一遍，看它失败**

Run: `npx vitest run test/repair-reimport-guard.test.mjs`

Expected: FAIL —— `Error: Failed to load url ../scripts/repairs/reimport-guard.mjs (resolved id: ...). Does the file exist?`

- [ ] **Step 26：写安装器**

新建 `scripts/repairs/reimport-guard.mjs`：

```js
/**
 * Idempotent installer for the one domain hook the shipped-data repairs take:
 * `importAdventure`.
 *
 * House rule: features and repairs never hook anything on their own. The single
 * exception is a registered patch of `type: "HOOK"`, which may hook a domain
 * hook from inside its own `apply()` provided (a) it hooks exactly once,
 * (b) the hook name is in the patch def's `target` so `patches.status()` shows
 * it, (c) the handle stays removable, and (d) the callback's first line checks
 * the repair's feature switch. This factory owns (a) and (c); each caller owns
 * (b) and (d).
 *
 * `importAdventure` fires after an Adventure document finishes importing into
 * the world. It is not one of the three hook groups the kernel owns
 * (chat-message creation, Dice So Nice completion, chat-card rendering), so a
 * patch may take it. Note that its argument list is not something to rely on:
 * the system's own handler reads it as `(created, updated)`
 * (systems/alienrpg/module/apps/init.mjs:101) while Foundry core passes
 * `(adventure, formData, created, updated)` — which is why every handler passed
 * in here takes no arguments and simply re-runs its whole plan.
 *
 * @param {Function} handler the zero-argument callback to install
 * @returns {{install:()=>number, retire:()=>boolean, hookId:number|null}}
 */
export function makeReimportGuard(handler) {
  let hookId = null;
  return {
    install() {
      if (hookId !== null) return hookId; // already standing: never hook twice
      hookId = Hooks.on("importAdventure", handler);
      return hookId;
    },
    retire() {
      if (hookId === null) return false;
      Hooks.off("importAdventure", handler);
      hookId = null;
      return true;
    },
    get hookId() {
      return hookId;
    },
  };
}
```

- [ ] **Step 27：跑一遍，看它通过**

Run: `npx vitest run test/repair-reimport-guard.test.mjs`

Expected: `4 passed`。

- [ ] **Step 28：提交**

```bash
git add scripts/repairs/reimport-guard.mjs test/repair-reimport-guard.test.mjs && git commit -m "$(cat <<'EOF'
feat(repairs): importAdventure 监听的幂等安装器

契约通则是修复不得自己挂钩子，唯一例外是 type:"HOOK" 的已登记补丁可以在
apply() 里挂领域钩子，条件有三：只挂一次、钩子名写进 def.target 供
status() 展示、回调第一行查 features.enabled(id)。这个工厂负责第一条
和「句柄可摘除」，调用方负责另外两条。

回调一个参数都不接是有意的：系统自己那个 handler 按 (created, updated)
读（apps/init.mjs:101），而 Foundry 核心传的是 (adventure, formData,
created, updated)——入参列表漂移过，而重跑整个计划一个参数都不需要。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 29：加 i18n 键**

语言包是**嵌套结构、只有一个顶层键 `AEA`**（Foundry 会把嵌套对象拍平成带点的键，所以 `selftest.data["ev-oracle-tables"]` 拍平后就是 `AEA.selftest.data.ev-oracle-tables`）。往 `lang/en.json` 里**已有的** `AEA` 对象中并入下面这些成员（不要新增第二个顶层键）：

```json
{
  "feature": {
    "ev-oracle-tables": {
      "name": "Repair the EV oracle tables",
      "hint": "Fixes three shipped Alien Evolved oracle tables whose roll formula and row ranges make some results unreachable and others appear twice. Only the formula and the numeric ranges change; no text is touched."
    },
    "npc-roster": {
      "name": "Repair the shipped NPC roster",
      "hint": "Corrects three shipped Alien Evolved NPCs whose stats disagree with the NPC table reprinted in this module's own GM guide journal."
    }
  },
  "data": {
    "oracleTitle": "Repair the EV oracle tables?",
    "oracleBody": "{n} shipped table value(s) are wrong. Repairing them changes only the roll formula and the numeric ranges; no text is touched.",
    "rosterTitle": "Repair the shipped NPC roster?",
    "rosterBody": "{n} field(s) on {m} shipped NPC(s) disagree with the NPC table in this module's GM guide.",
    "apply": "Repair",
    "skip": "Leave as is",
    "risky": "risky - changes the actor type",
    "done": "Repaired {n} value(s).",
    "reapplied": "An adventure import overwrote shipped data; the repairs were re-applied.",
    "oracleClean": "All three EV oracle tables read as printed.",
    "oracleAbsent": "The EV oracle tables are not in this world; nothing to check.",
    "rosterClean": "Every checked field on the shipped NPCs reads as printed.",
    "rosterAbsent": "The shipped NPC roster is not in this world; nothing to check."
  },
  "selftest": {
    "data": {
      "ev-oracle-tables": "Shipped data: EV oracle tables",
      "npc-roster": "Shipped data: NPC roster"
    }
  }
}
```

往 `lang/cn.json` 的 `AEA` 对象里并入同样的结构：

```json
{
  "feature": {
    "ev-oracle-tables": {
      "name": "修复 EV 神谕表",
      "hint": "修正三张出厂的 Alien Evolved 神谕表：它们的掷骰式与行区间让一部分结果永远抽不到、另一部分每次都多冒出来。只改掷骰式与数字区间，不动任何文字。"
    },
    "npc-roster": {
      "name": "修复出厂 NPC 名录",
      "hint": "修正三个出厂 Alien Evolved NPC 的数值 —— 它们与本模组自带 GM 指南日志里转载的 NPC 总表不符。"
    }
  },
  "data": {
    "oracleTitle": "修复 EV 神谕表？",
    "oracleBody": "出厂数据里有 {n} 处错误。修复只改掷骰式与数字区间，不动任何文字。",
    "rosterTitle": "修复出厂 NPC 名录？",
    "rosterBody": "{m} 个出厂 NPC 共 {n} 个字段与 GM 指南里的 NPC 总表不符。",
    "apply": "修复",
    "skip": "保持原样",
    "risky": "有风险 —— 会改变 actor 类型",
    "done": "已修复 {n} 处。",
    "reapplied": "一次冒险导入覆盖了出厂数据，修复已重新施加。",
    "oracleClean": "三张 EV 神谕表都与印刷版一致。",
    "oracleAbsent": "本世界里没有这几张 EV 神谕表，无需检查。",
    "rosterClean": "受检的出厂 NPC 字段都与印刷版一致。",
    "rosterAbsent": "本世界里没有出厂 NPC 名录，无需检查。"
  },
  "selftest": {
    "data": {
      "ev-oracle-tables": "出厂数据：EV 神谕表",
      "npc-roster": "出厂数据：NPC 名录"
    }
  }
}
```

- [ ] **Step 30：写第七个失败测试 —— 神谕修复的登记形状与开关门**

`register()` 会不会把该登记的都登记上、`label` 是不是**键**而不是已本地化文本、`importAdventure` 回调有没有先查开关 —— 这四件是这类任务最常见的失手处，而且都能用共享桩忠实测出来。把 `test/repair-ev-oracle-tables.test.mjs` 顶部的 vitest import 补成 `import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";`，再加一行 `import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";`，然后追加：

```js
describe("evOracleTablesRepair.register()", () => {
  beforeEach(() => {
    // Fresh module state per test: the kernel registries are module-level
    // singletons, so a second register() into the same registry would duplicate.
    vi.resetModules();
    installFoundryStub();
  });
  afterEach(() => uninstallFoundryStub());

  it("registers one feature switch and one DATA patch under the id ev-oracle-tables", async () => {
    const { evOracleTablesRepair } = await import("../scripts/repairs/ev-oracle-tables.mjs");
    const { features } = await import("../scripts/kernel/features.mjs");
    const { patches } = await import("../scripts/kernel/patches.mjs");

    expect(evOracleTablesRepair.id).toBe("ev-oracle-tables");
    evOracleTablesRepair.register();

    expect(features.all().map((def) => def.id)).toContain("ev-oracle-tables");
    const entry = patches.status().find((p) => p.id === "ev-oracle-tables");
    expect(entry).toBeDefined();
    expect(entry.type).toBe("DATA");
    expect(entry.fixedIn).toBeNull();
  });

  it("registers a second, HOOK-typed patch whose target names the hook it takes", async () => {
    const { evOracleTablesRepair } = await import("../scripts/repairs/ev-oracle-tables.mjs");
    const { patches } = await import("../scripts/kernel/patches.mjs");

    evOracleTablesRepair.register();

    const entry = patches.status().find((p) => p.id === "ev-oracle-tables.reimport");
    expect(entry).toBeDefined();
    expect(entry.type).toBe("HOOK");
    expect(entry.target).toBe("importAdventure");
  });

  it("registers a selftest entry whose label is a raw i18n key, not localized text", async () => {
    const { selftest } = await import("../scripts/kernel/selftest.mjs");
    const spy = vi.spyOn(selftest, "register");
    const { evOracleTablesRepair } = await import("../scripts/repairs/ev-oracle-tables.mjs");

    evOracleTablesRepair.register();

    const def = spy.mock.calls.map((call) => call[0]).find((d) => d?.id === "data.ev-oracle-tables");
    expect(def).toBeDefined();
    expect(typeof def.run).toBe("function");
    // Registration happens during `init`, before game.i18n exists, so the def
    // must carry the key and let the runner localize it. A getter would also
    // make the def unserializable, so assert it is a plain string property.
    expect(def.label).toBe("AEA.selftest.data.ev-oracle-tables");
    expect(Object.getOwnPropertyDescriptor(def, "label").get).toBeUndefined();
  });
});

describe("ev-oracle onAdventureImported()", () => {
  beforeEach(() => {
    vi.resetModules();
    installFoundryStub();
  });
  afterEach(() => uninstallFoundryStub());

  it("checks the feature switch before touching anything", async () => {
    const { features } = await import("../scripts/kernel/features.mjs");
    const enabled = vi.spyOn(features, "enabled").mockReturnValue(false);
    const { onAdventureImported } = await import("../scripts/repairs/ev-oracle-tables.mjs");

    await expect(onAdventureImported()).resolves.toBe(0);
    expect(enabled).toHaveBeenCalledWith("ev-oracle-tables");
  });
});
```

- [ ] **Step 31：跑一遍，看它失败**

Run: `npx vitest run test/repair-ev-oracle-tables.test.mjs -t "register"`

Expected: FAIL —— `Error: Failed to load url ../scripts/repairs/ev-oracle-tables.mjs (resolved id: ...). Does the file exist?`

- [ ] **Step 32：写神谕修复的副作用层**

新建 `scripts/repairs/ev-oracle-tables.mjs`：

```js
import { features } from "../kernel/features.mjs";
import { patches } from "../kernel/patches.mjs";
import { selftest } from "../kernel/selftest.mjs";
import { repairLog } from "./repair-log.mjs";
import { makeReimportGuard } from "./reimport-guard.mjs";
import {
  pureOraclePlan,
  pureOracleTableIds,
  pureOracleTablesAreBuggy,
  pureOracleTablesPresent,
} from "./ev-oracle-tables.pure.mjs";

const ID = "ev-oracle-tables";

/**
 * Read the live copies of the three tables this repair owns.
 *
 * RECORDED DEVIATION from the house rule "never look a document up by display
 * name — always go through the registry". The registry's key list is frozen at
 * 13 system documents for this release and its `kind` enum has no slot for a
 * third-party pack, so these three tables have no registry key to go through.
 * The sanctioned fallback for a shipped-pack document is what this does:
 * address it by immutable document `_id`. That satisfies what the rule actually
 * forbids — `getName()` and display-name comparison — because a translation
 * layer (Babele) renames these tables freely while `_id` never changes. The ids
 * survive import because the system imports the adventure with `keepId: true`
 * (systems/alienrpg/module/apps/init.mjs:166). When the registry grows an
 * item/pack kind in phase two, this lookup moves behind it.
 */
function liveTables() {
  const found = [];
  for (const id of pureOracleTableIds()) {
    const table = game.tables.get(id);
    if (!table) continue; // pack not imported, or this world deleted the table
    found.push(table.toObject());
  }
  return found;
}

async function askGm(title, body) {
  const answer = await foundry.applications.api.DialogV2.confirm({
    window: { title },
    content: body,
    yes: { label: game.i18n.localize("AEA.data.apply") },
    no: { label: game.i18n.localize("AEA.data.skip") },
    rejectClose: false, // closing the dialog resolves null instead of throwing
  });
  return answer === true;
}

async function writePlan(plan) {
  const byTable = new Map();
  for (const step of plan) {
    if (!byTable.has(step.tableId)) byTable.set(step.tableId, []);
    byTable.get(step.tableId).push(step);
  }
  for (const [tableId, steps] of byTable) {
    const table = game.tables.get(tableId);
    if (!table) continue;
    const formulaStep = steps.find((step) => step.kind === "formula");
    if (formulaStep) await table.update({ formula: formulaStep.to });
    const rangeSteps = steps.filter((step) => step.kind === "range");
    if (rangeSteps.length) {
      // TableResults are embedded documents, updated through their parent.
      await table.updateEmbeddedDocuments(
        "TableResult",
        rangeSteps.map((step) => ({ _id: step.resultId, range: step.to }))
      );
    }
  }
}

/**
 * One repair pass. Safe to call any number of times: an empty plan writes
 * nothing, asks nothing and stamps nothing.
 *
 * @param {{silent?:boolean}} [options] silent skips the confirmation dialog,
 *   used when re-applying after an adventure import so the GM is not asked the
 *   same question twice in one session.
 * @returns {Promise<number>} how many values were changed
 */
async function repairOnce({ silent = false } = {}) {
  if (!game.user.isGM) return 0; // world documents are GM-writable only
  // Checked here rather than at registration, so flipping the switch takes
  // effect on the next pass without reloading the world.
  if (!features.enabled(ID)) return 0;

  const plan = pureOraclePlan(liveTables());
  if (!plan.length) return 0;

  if (!silent) {
    const ok = await askGm(
      game.i18n.localize("AEA.data.oracleTitle"),
      `<p>${game.i18n.format("AEA.data.oracleBody", { n: plan.length })}</p>`
    );
    if (!ok) return 0;
  }

  await writePlan(plan);
  await repairLog.record(ID, { count: plan.length });
  ui.notifications.info(game.i18n.format("AEA.data.done", { n: plan.length }));
  return plan.length;
}

/**
 * The `importAdventure` callback. Exported so the switch gate is directly
 * testable, and so a GM can re-run it from the console.
 *
 * First line checks the feature switch, as required of anything a patch's
 * apply() installs: turning the repair off must stop it dead with no world
 * reload. (repairOnce checks again — it is also called straight from apply().)
 */
export async function onAdventureImported() {
  if (!features.enabled(ID)) return 0;
  const changed = await repairOnce({ silent: true });
  if (changed) ui.notifications.info(game.i18n.localize("AEA.data.reapplied"));
  return changed;
}

const reimportGuard = makeReimportGuard(onAdventureImported);

export const evOracleTablesRepair = {
  id: ID,

  /** Called during `init`. Registers only; touches no document. */
  register() {
    features.register({
      id: ID,
      default: "full",
      gmOnly: true, // only a GM can write world documents
      requires: [],
      hint: "",
      // Display name and hint come from AEA.feature.<id>.name / .hint.
      // This repair always asks the GM before writing, so the "full" and
      // "prompt" modes behave identically here; only "off" changes anything.
    });

    patches.register({
      id: ID,
      type: "DATA",
      target: "RollTable 6HcfWwkEJ1Y0KzMy / TableResult c9g3UmwQpBCBjYDE + RfiCDLTS1aVIP3d1",
      // The version gate compares the *system* version, and shipped-module data
      // does not track it, so minSystem/fixedIn cannot carry this decision:
      // probe() does. minSystem is the release this was verified against.
      minSystem: "4.1.13",
      fixedIn: null,

      // "Should this repair stand guard here?" — not "is the data wrong right
      // now". An adventure re-import can make it wrong again mid-session, so
      // the repair stays armed in a world we already fixed. In a world without
      // the pack, probe is false and patches records it as retired — which is
      // exactly the outcome we want there: install nothing. Whether the data is
      // currently right is the selftest entry's question, below; both share one
      // pure predicate.
      probe: () => pureOracleTablesPresent(liveTables()),

      apply: () => repairOnce(),
    });

    patches.register({
      id: `${ID}.reimport`,
      type: "HOOK",
      // The hook name lives in `target` so patches.status() shows which hook
      // this patch took — the condition attached to the "a registered HOOK
      // patch may hook from apply()" exception.
      target: "importAdventure",
      minSystem: "4.1.13",
      fixedIn: null,
      probe: () => pureOracleTablesPresent(liveTables()),
      // The system's own importer runs on `ready` and a GM can run the full
      // Adventure Importer at any time, either of which overwrites these
      // documents. The guard hooks at most once and keeps the handle removable.
      apply: () => {
        reimportGuard.install();
      },
    });

    selftest.register({
      id: `data.${ID}`,
      // An i18n KEY, not localized text: register() runs during `init` and
      // game.i18n only exists from `i18nInit` onward, so localizing here would
      // freeze the raw key in. The runner localizes it.
      label: "AEA.selftest.data.ev-oracle-tables",
      run: () => {
        const tables = liveTables();
        if (!pureOracleTablesPresent(tables)) {
          return { ok: true, detail: game.i18n.localize("AEA.data.oracleAbsent") };
        }
        if (!pureOracleTablesAreBuggy(tables)) {
          return { ok: true, detail: game.i18n.localize("AEA.data.oracleClean") };
        }
        return {
          ok: false,
          detail: pureOraclePlan(tables)
            .map((step) => `${step.tableId} ${step.kind}: ${JSON.stringify(step.from)} -> ${JSON.stringify(step.to)}`)
            .join("; "),
        };
      },
    });
  },
};
```

- [ ] **Step 33：跑一遍，看它通过**

Run: `npx vitest run test/repair-ev-oracle-tables.test.mjs`

Expected: `16 passed`。

- [ ] **Step 34：提交**

```bash
git add scripts/repairs/ev-oracle-tables.mjs test/repair-ev-oracle-tables.test.mjs lang/en.json lang/cn.json && git commit -m "$(cat <<'EOF'
feat(repairs): EV 神谕表修复的副作用层（开关 + 两条补丁 + 自检）

- features.register 一个可关的开关（gmOnly），执行时查 enabled，
  查在执行时而不是登记时，所以改档不必重载世界；
- patches.register 两条：type:"DATA" 那条只写数据；type:"HOOK" 那条把
  importAdventure 监听挂上，target 写的就是钩子名，供 status() 展示——
  这是「已登记 HOOK 补丁可在 apply() 里挂领域钩子」这条例外的附带条件。
  监听经幂等安装器挂，句柄可摘除，回调第一行查 features.enabled；
- probe 问的是「该不该在这个世界布防」（三张表在不在），不是「现在错没错」：
  再导入会把数据重新弄坏，已修好的世界里也必须布防。「现在错没错」由自检
  条目回答，两者共用同一个纯谓词；
- selftest.register 一条 data.ev-oracle-tables，label 存的是 i18n 键
  而不是文本，也不是 getter：登记发生在 init、game.i18n 要到 i18nInit
  才就绪，本地化由 runAll 负责，getter 还会让 def 无法序列化展示。

按不可变 _id 找文档、绝不按显示名（这是对「一律走 registry」的一处有记录的
偏离：registry 13 键一期冻结、kind 里没有第三方包的位置，模块 JSDoc 里写明
了原因）；记账走 repairLog.record，本模块不碰那个世界设置的注册。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 35：写第八个失败测试 —— 名录修复的登记形状与开关门**

把 `test/repair-npc-roster.test.mjs` 顶部的 vitest import 补成 `import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";`，再加一行 `import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";`，然后追加：

```js
describe("npcRosterRepair.register()", () => {
  beforeEach(() => {
    vi.resetModules();
    installFoundryStub();
  });
  afterEach(() => uninstallFoundryStub());

  it("registers one feature switch and one DATA patch under the id npc-roster", async () => {
    const { npcRosterRepair } = await import("../scripts/repairs/npc-roster.mjs");
    const { features } = await import("../scripts/kernel/features.mjs");
    const { patches } = await import("../scripts/kernel/patches.mjs");

    expect(npcRosterRepair.id).toBe("npc-roster");
    npcRosterRepair.register();

    expect(features.all().map((def) => def.id)).toContain("npc-roster");
    const entry = patches.status().find((p) => p.id === "npc-roster");
    expect(entry).toBeDefined();
    expect(entry.type).toBe("DATA");
    expect(entry.fixedIn).toBeNull();
  });

  it("registers a second, HOOK-typed patch whose target names the hook it takes", async () => {
    const { npcRosterRepair } = await import("../scripts/repairs/npc-roster.mjs");
    const { patches } = await import("../scripts/kernel/patches.mjs");

    npcRosterRepair.register();

    const entry = patches.status().find((p) => p.id === "npc-roster.reimport");
    expect(entry).toBeDefined();
    expect(entry.type).toBe("HOOK");
    expect(entry.target).toBe("importAdventure");
  });

  it("registers a selftest entry whose label is a raw i18n key, not localized text", async () => {
    const { selftest } = await import("../scripts/kernel/selftest.mjs");
    const spy = vi.spyOn(selftest, "register");
    const { npcRosterRepair } = await import("../scripts/repairs/npc-roster.mjs");

    npcRosterRepair.register();

    const def = spy.mock.calls.map((call) => call[0]).find((d) => d?.id === "data.npc-roster");
    expect(def).toBeDefined();
    expect(typeof def.run).toBe("function");
    expect(def.label).toBe("AEA.selftest.data.npc-roster");
    expect(Object.getOwnPropertyDescriptor(def, "label").get).toBeUndefined();
  });
});

describe("npc-roster onAdventureImported()", () => {
  beforeEach(() => {
    vi.resetModules();
    installFoundryStub();
  });
  afterEach(() => uninstallFoundryStub());

  it("checks the feature switch before touching anything", async () => {
    const { features } = await import("../scripts/kernel/features.mjs");
    const enabled = vi.spyOn(features, "enabled").mockReturnValue(false);
    const { onAdventureImported } = await import("../scripts/repairs/npc-roster.mjs");

    await expect(onAdventureImported()).resolves.toBe(0);
    expect(enabled).toHaveBeenCalledWith("npc-roster");
  });
});
```

- [ ] **Step 36：跑一遍，看它失败**

Run: `npx vitest run test/repair-npc-roster.test.mjs -t "register"`

Expected: FAIL —— `Error: Failed to load url ../scripts/repairs/npc-roster.mjs (resolved id: ...). Does the file exist?`

- [ ] **Step 37：写名录修复的副作用层**

新建 `scripts/repairs/npc-roster.mjs`：

```js
import { features } from "../kernel/features.mjs";
import { patches } from "../kernel/patches.mjs";
import { resolver } from "../kernel/resolver.mjs";
import { selftest } from "../kernel/selftest.mjs";
import { repairLog } from "./repair-log.mjs";
import { makeReimportGuard } from "./reimport-guard.mjs";
import {
  pureEscapeText,
  pureRosterActorIds,
  pureRosterActorsPresent,
  pureRosterDiff,
  pureRosterIsStale,
} from "./npc-roster.pure.mjs";

const ID = "npc-roster";

/**
 * Resolve the live copies of the actors this repair owns.
 *
 * All we have is a bare actor id from the shipped pack, which is exactly the
 * legacy shape `resolver.actorById` exists for; this module never calls
 * `game.actors.get` itself. `warn: false` because a world that never imported
 * the pack is a normal, quiet case, not a misconfiguration.
 *
 * RECORDED DEVIATION, same as the oracle repair: the house rule routes document
 * lookup through the registry, whose 13 keys are frozen for this release and
 * cover no third-party pack document. Addressing a shipped actor by immutable
 * `_id` is the sanctioned fallback — it satisfies what the rule forbids
 * (`getName()` and display-name comparison), and the ids survive import because
 * the system imports with `keepId: true`
 * (systems/alienrpg/module/apps/init.mjs:166).
 */
function liveRoster() {
  const found = [];
  for (const id of pureRosterActorIds()) {
    const actor = resolver.actorById(id, { warn: false });
    if (!actor) continue;
    found.push({ actor, data: actor.toObject() });
  }
  return found;
}

function pendingDiffs() {
  return liveRoster()
    .map(({ actor, data }) => ({ actor, diffs: pureRosterDiff(data) }))
    .filter((row) => row.diffs.length > 0);
}

async function askGm(title, body) {
  const answer = await foundry.applications.api.DialogV2.confirm({
    window: { title },
    content: body,
    yes: { label: game.i18n.localize("AEA.data.apply") },
    no: { label: game.i18n.localize("AEA.data.skip") },
    rejectClose: false,
  });
  return answer === true;
}

function diffListHtml(rows) {
  const riskyLabel = game.i18n.localize("AEA.data.risky");
  return rows
    .map(({ actor, diffs }) => {
      const items = diffs
        .map((diff) => {
          const suffix = diff.risky ? ` <em>(${pureEscapeText(riskyLabel)})</em>` : "";
          return (
            `<li><code>${pureEscapeText(diff.path)}</code>: ` +
            `${pureEscapeText(JSON.stringify(diff.from))} &rarr; ` +
            `${pureEscapeText(JSON.stringify(diff.to))}${suffix}</li>`
          );
        })
        .join("");
      return `<li><b>${pureEscapeText(actor.name)}</b><ul>${items}</ul></li>`;
    })
    .join("");
}

async function repairOnce({ silent = false } = {}) {
  if (!game.user.isGM) return 0;
  if (!features.enabled(ID)) return 0;

  const rows = pendingDiffs();
  const total = rows.reduce((sum, row) => sum + row.diffs.length, 0);
  if (!total) return 0;

  if (!silent) {
    const ok = await askGm(
      game.i18n.localize("AEA.data.rosterTitle"),
      `<p>${game.i18n.format("AEA.data.rosterBody", { n: total, m: rows.length })}</p>` +
        `<ul>${diffListHtml(rows)}</ul>`
    );
    if (!ok) return 0;
  }

  for (const { actor, diffs } of rows) {
    const safe = diffs.filter((diff) => !diff.risky);
    if (safe.length) {
      // Foundry expands dotted keys, so one update carries every safe field.
      const update = {};
      for (const diff of safe) update[diff.path] = diff.to;
      await actor.update(update);
    }
    // Risky fields last and one at a time: changing an Actor's `type` makes
    // Foundry re-validate `system` against a different DataModel. For this one
    // case synthetic is a superset of character (same header / attributes /
    // skills / general / consumables / adhocitems, with attributes and skills
    // built from the same CONFIG.ALIENRPG maps on both sides, plus one extra
    // boolean header.synthstress), so nothing should be dropped — but it is
    // still the one change worth isolating and showing to the GM.
    for (const diff of diffs.filter((row) => row.risky)) {
      await actor.update({ [diff.path]: diff.to });
    }
  }

  await repairLog.record(ID, { count: total });
  ui.notifications.info(game.i18n.format("AEA.data.done", { n: total }));
  return total;
}

/** See the oracle repair: exported so the switch gate is directly testable. */
export async function onAdventureImported() {
  if (!features.enabled(ID)) return 0;
  const changed = await repairOnce({ silent: true });
  if (changed) ui.notifications.info(game.i18n.localize("AEA.data.reapplied"));
  return changed;
}

const reimportGuard = makeReimportGuard(onAdventureImported);

export const npcRosterRepair = {
  id: ID,

  register() {
    features.register({
      id: ID,
      default: "full",
      gmOnly: true,
      requires: [],
      hint: "",
    });

    patches.register({
      id: ID,
      type: "DATA",
      target: "Actor ZCfDvo9HjdAzpnNe / DukQ40yWrGH8jmKP / 881anZR8zz4RnlB2",
      minSystem: "4.1.13",
      fixedIn: null,
      probe: () => pureRosterActorsPresent(liveRoster().map((row) => row.data)),
      apply: () => repairOnce(),
    });

    patches.register({
      id: `${ID}.reimport`,
      type: "HOOK",
      target: "importAdventure",
      minSystem: "4.1.13",
      fixedIn: null,
      probe: () => pureRosterActorsPresent(liveRoster().map((row) => row.data)),
      apply: () => {
        reimportGuard.install();
      },
    });

    selftest.register({
      id: `data.${ID}`,
      label: "AEA.selftest.data.npc-roster",
      run: () => {
        const datas = liveRoster().map((row) => row.data);
        if (!pureRosterActorsPresent(datas)) {
          return { ok: true, detail: game.i18n.localize("AEA.data.rosterAbsent") };
        }
        if (!pureRosterIsStale(datas)) {
          return { ok: true, detail: game.i18n.localize("AEA.data.rosterClean") };
        }
        return {
          ok: false,
          detail: pendingDiffs()
            .flatMap(({ actor, diffs }) =>
              diffs.map(
                (diff) => `${actor.name} ${diff.path}: ${JSON.stringify(diff.from)} -> ${JSON.stringify(diff.to)}`
              )
            )
            .join("; "),
        };
      },
    });
  },
};
```

- [ ] **Step 38：跑一遍，看它通过**

Run: `npx vitest run test/repair-npc-roster.test.mjs`

Expected: `17 passed`。

- [ ] **Step 39：提交**

```bash
git add scripts/repairs/npc-roster.mjs test/repair-npc-roster.test.mjs && git commit -m "$(cat <<'EOF'
feat(repairs): NPC 名录修复的副作用层

与神谕表同一套纪律：GM 门禁、执行时查 features.enabled、计划为空不动手、
记账走 repairLog.record、两条补丁（DATA 写数据 + HOOK 挂 importAdventure
静默重放，target 写钩子名、监听经幂等安装器、回调第一行查开关）、
自检 label 存 i18n 键。

裸 actor id 一律经 resolver.actorById(id, {warn:false}) 解析，
本模组不直接调 game.actors.get；确认框里的 actor 名字过 pureEscapeText，
因为 GM 可以把名字改成任何东西。

有风险的 type 变更单独最后逐条施加：改 Actor#type 会让 Foundry 拿另一个
DataModel 重新校验 system。这一处已逐项核对：synthetic 是 character 的
超集（header/attributes/skills/general/consumables/adhocitems 同名，
attributes 与 skills 两边都由同一份 CONFIG.ALIENRPG 归约生成，
synthetic 只多一个布尔 header.synthstress），预期无损，
但仍然隔离出来并在确认框里标红。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 40：接到 `main.mjs` 上（三个锚点，共六行）**

`scripts/main.mjs` 已经存在，里面有十个 `/* AEA-ANCHOR: ... */` 注释。**一律按锚点文本定位，不用行号**，并且**只动属于本任务的三个锚点**。

1. 在 `/* AEA-ANCHOR: imports */` 这一行**之后**加三行：

```js
import { repairLog } from "./repairs/repair-log.mjs";
import { evOracleTablesRepair } from "./repairs/ev-oracle-tables.mjs";
import { npcRosterRepair } from "./repairs/npc-roster.mjs";
```

2. 在 `/* AEA-ANCHOR: repairs */` 这一行**之后**加两行数组成员（它在 `const REPAIRS = [` 之内）：

```js
  evOracleTablesRepair,
  npcRosterRepair,
```

3. 在 `Hooks.once("init", ...)` 里的 `/* AEA-ANCHOR: init */` 这一行**之后**加一行：

```js
  repairLog.registerSettings();
```

这一行是 `dataRepairs` 世界设置的**唯一注册点**（Foundry 只在 `init` 接受设置注册）。它排在锚点之后、也就排在骨架里那两个 `f.register()` / `r.register()` 循环之后，这不要紧：`dataRepairs` 与特性档位设置是两个互不相干的键，`init` 阶段没有任何代码读它，第一次读发生在 `ready` 的 `patches.applyAll()` 里。

**不要做的事**：不碰 `export const api = { ... }` 那个八槽对象（本任务不是内核模块的属主，一个槽都不换、一个键都不加）；不往 `ready.registry` / `ready.patches` / `ready.rollbus` / `ready.cards` 四个子锚点插任何东西；不碰 `i18nInit` 与 `diceSoNiceReady`。两个修复模块也**不导出 `install()`** —— `ready` 段那句 `r.install?.()` 会安静地跳过它们，真正的动作发生在 `patches.applyAll()`（`ready.patches` 子锚点）调用的 `apply()` 里，而那一句排在 install 循环之前。

- [ ] **Step 41：确认接线落在了正确的位置**

Run:
```bash
grep -n "AEA-ANCHOR\|repairLog\|evOracleTablesRepair\|npcRosterRepair" scripts/main.mjs && git diff --stat scripts/main.mjs
```

Expected：
- 三行 import 紧跟在 `/* AEA-ANCHOR: imports */` 之后；
- `evOracleTablesRepair,` 与 `npcRosterRepair,` 紧跟在 `/* AEA-ANCHOR: repairs */` 之后；
- `repairLog.registerSettings();` 紧跟在 `/* AEA-ANCHOR: init */` 之后；
- 本任务的四个标识符**不出现**在任何 `AEA-ANCHOR: ready.*` 之后，也不出现在 `api` 那个对象字面量里；
- `git diff --stat` 显示 `scripts/main.mjs` 是 **6 insertions(+), 0 deletions(-)**。若有删除行，说明你动了不属于本任务的东西，撤回重做。

- [ ] **Step 42：跑全套**

Run: `npm test`

Expected：`test/repair-ev-oracle-tables.test.mjs` 16 passed、`test/repair-npc-roster.test.mjs` 17 passed、`test/repair-log.test.mjs` 4 passed、`test/repair-reimport-guard.test.mjs` 4 passed，且**仓库里原有的每一个测试文件仍然全绿**（本任务不改任何既有测试、不改任何既有源文件，只在 `main.mjs` 里加六行）。

- [ ] **Step 43：MANUAL VERIFICATION（本机冒烟机；VPS 才是发布权威）**

真实的 `DialogV2` 交互、文档写库、`type` 变更后的 DataModel 重校验、以及 Adventure Importer 的时序都不是共享桩能忠实模拟的，按纪律它们只能靠自检条目加下面这些逐条动作来验收。

前置：一个**装了 `alien-evolved-corerules`（版本 1.0.2）且已经导入过冒险包**的世界。

1. **以 GM 身份进入世界。** 期待：`ready` 之后先后弹出两个对话框 ——「修复 EV 神谕表？」说有 **3 处**错误；「修复出厂 NPC 名录？」说 **3 个** NPC 共 **4 个**字段不符，列表里潜伏仿生人的 `type` 那一行带「有风险 —— 会改变 actor 类型」。两个都点「修复」。
2. **控制台核对三张表：**
   - `game.tables.get("6HcfWwkEJ1Y0KzMy").formula` → 期待 `"1d6"`
   - `game.tables.get("dnm74JBwOH9AMQ6v").results.get("c9g3UmwQpBCBjYDE").range` → 期待 `[25, 26]`
   - `game.tables.get("YnhWA69wqLLp2jsB").results.get("RfiCDLTS1aVIP3d1").range` → 期待 `[31, 36]`
   - `game.tables.get("YnhWA69wqLLp2jsB").results.get("0wZOQbZwu0RUQDG4").range` → 期待仍是 `[11, 26]`（这行合法，绝不能被改）
3. **在侧栏抽 20 次「EV - 50. LS - BINARY RESPONSE MATRIX」。** 期待：**能抽到 "Strong yes"**（修之前概率为 0）。
4. **抽 10 次「EV - 52. LS - SECTION MATRIX」。** 期待：**每张卡只有一个结果**；修之前 36 个结果里有 24 个会额外冒出一个 "Cryosleep"。同样抽 10 次「EV - 53」，期待不再多出 "Stairs/Ladder"。
5. **重启世界。** 期待：**两个对话框都不再弹**（幂等）。控制台跑
   `game.settings.get("alien-evolved-automation", "dataRepairs")` → 期待看到 `ev-oracle-tables` 与 `npc-roster` 两个键，每个带 `version: 1`、`at`、`by`、`count`。
6. **跑自检：** `await game.modules.get("alien-evolved-automation").api.selftest.runAll()` → 期待 `data.ev-oracle-tables` 与 `data.npc-roster` 两条都是 `ok: true`，`detail` 是中文的「都与印刷版一致」，且 `label` 显示为中文。若 `label` 显示成 `AEA.selftest.data.ev-oracle-tables` 这样的裸键，说明语言包里少了这个键（def 里存的**本来就是**键，本地化由 `runAll()` 负责，这一点不要改回 getter）。
7. **控制台核对三个 NPC：**
   - `game.actors.get("ZCfDvo9HjdAzpnNe").system.attributes.agl.value` → 期待 `4`；`...wit.value` → 期待 `3`
   - `game.actors.get("DukQ40yWrGH8jmKP").system.skills.command.value` → 期待 `2`
   - `game.actors.get("881anZR8zz4RnlB2").type` → 期待 `"synthetic"`；`...system.header.npc` → 期待 `true`
   - `game.actors.get("IxstyK7Y6gwdaiI0").system.attributes.wit.value` → 期待仍是 `5`（公司官僚**不在**修复范围内，动了就是错）
8. **打开潜伏仿生人的角色卡。** 期待：属性 7/6/3/1、五项技能（近战 3、重型机械 2、远程 2、通讯技术 2、生存 1）、生命 7、身上的 "Take Control" 物品**都还在**，卡片以合成人（synthetic）样式打开。
   - 再把它拖到场景上放两个 token，期待两个 token 各自独立（不再共享同一条血量），并且默认是敌对。
   - **若 `type` 变更导致任何数据丢失**：立刻 `game.actors.get("881anZR8zz4RnlB2").update({type:"character"})` 回退，并把 `881anZR8zz4RnlB2` 的 `riskyFields` 从 `npc-roster.pure.mjs` 里去掉（只保留 `system.header.npc`），改由文末的上游 PR B 用「导出 → 删除 → 以 synthetic 类型 keepId 重建」的路径解决。
9. **测试再导入的重放：** 控制台把一张表改回错的 —— `await game.tables.get("6HcfWwkEJ1Y0KzMy").update({formula:"1d4"})` —— 然后打开合集里的 "Alien Evolved Core Rules" 冒险包、手动跑一次 Adventure Importer。期待：导入完成后自动弹出「一次冒险导入覆盖了出厂数据，修复已重新施加」的提示（**不再弹确认框**），且 `game.tables.get("6HcfWwkEJ1Y0KzMy").formula` 又是 `"1d6"`。
10. **以玩家身份进入同一世界。** 期待：**完全不弹任何对话框**、没有任何写库、控制台无报错。
11. **测试开关：** 以 GM 打开模组设置，把「修复 EV 神谕表」调到「关闭」，控制台把表改回错的（同第 9 步），**不重载世界**，直接手动跑一次 Adventure Importer → 期待**不弹框、不修**（开关是在执行时查的）；`await api.selftest.runAll()` 里 `data.ev-oracle-tables` 变成 `ok: false` 并在 `detail` 里列出那处差异。改回「全自动」后重进世界 → 期待又弹框并修好。
12. **检查补丁面板数据：** `game.modules.get("alien-evolved-automation").api.patches.status().filter(p => ["ev-oracle-tables","ev-oracle-tables.reimport","npc-roster","npc-roster.reimport"].includes(p.id))` → 期待**四条**：两条 `type: "DATA"`（`target` 是可读的英文文档 id 串）、两条 `type: "HOOK"` 且 `target` 恰好是 `"importAdventure"`，四条都 `applied: true`。

- [ ] **Step 44：提交**

```bash
git add scripts/main.mjs && git commit -m "$(cat <<'EOF'
chore(repairs): 把两条出厂数据修复接到 main.mjs 的三个锚点上

按锚点文本定位，共六行：imports 锚点后三行 import、repairs 锚点后两个
数组成员、init 锚点后一行 repairLog.registerSettings()——那是 dataRepairs
世界设置的唯一注册点，Foundry 只在 init 接受设置注册。

不碰 api 那个八槽对象（本任务不是内核模块的属主），不碰四个 ready.* 子锚点，
不碰 i18nInit 与 diceSoNiceReady。两个模块都不导出 install()：它们在 ready
阶段没有订阅，真正的动作在 patches.applyAll()（ready.patches 子锚点）调用的
apply() 里，而它排在 install 循环之前，骨架把那一行写成 r.install?.() 正是
为了容纳这种修复。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

**UPSTREAM PR（两个，都是纯数据 PR，发给 `pwatson100` 的 `alien-evolved-corerules` 仓库）**

- **PR A —— 三处掷骰表手误**（`packs/alien-evolved-core-rules`）：RollTable `6HcfWwkEJ1Y0KzMy` 的 `formula` 由 `"1d4"` 改为 `"1d6"`；TableResult `c9g3UmwQpBCBjYDE` 的 `range` 由 `[25,266]` 改为 `[25,26]`；TableResult `RfiCDLTS1aVIP3d1` 的 `range` 由 `[31,362]` 改为 `[31,36]`。三处都是多打了一位。描述里附上可复现的现象即可：二元神谕当前 75% 偏向「否」且 "Strong yes" 概率为 0；EV-52 的 36 个 d66 结果里有 24 个会多返回一个 "Cryosleep"；EV-53 有 18 个会多返回一个 "Stairs/Ladder"。同时说明 EV-53 的 `0wZOQbZwu0RUQDG4` `[11,26]` 是合法宽行，不要一起改。
- **PR B —— 四个 Actor**（同一个包）：
  - `ZCfDvo9HjdAzpnNe`「EV - COLONY MANAGER」：`system.attributes.agl.value` 3→4、`system.attributes.wit.value` 4→3（与本模组自带的 GM 指南日志 `MZu96EjrzlxhTRXc` / 页 `TGaKQLUMk69avd5b` 里印的 "Colony manager Strength 2, Agility 4, Wits 3, Empathy 5" 一致；包里存下来的 `agl.mod=4` / `wit.mod=3` 也是这么写的）。
  - `DukQ40yWrGH8jmKP`「EV - SQUAD LEADER」：`system.skills.command.value` 1→2（同一页印的是 "Command 2"）。
  - `881anZR8zz4RnlB2`「EV - ANDROID, COVERT」：`type` `character`→`synthetic`、`system.header.npc` `false`→`true`。描述里要说明连带效果：`systems/alienrpg/module/alienrpg.mjs:366-375` 的 `preCreateToken` 只对 `system.header.npc` 为真的 actor 设 `HOSTILE` 与 `actorLink:false`，所以现在这个 android 的每一个 token 都链接到同一个基础 actor、共享一条血量；同包的另外两个 android 都是 `synthetic` + `npc:true`。
  - `IxstyK7Y6gwdaiI0`「EV - COMPANY BUREAUCRAT」：**整条替换，不是改字段**。现在这个 actor 是印刷版 "Corporate executive" 那一行（Wits 5、Empathy 4、Observation 3、Command 3、Manipulation 4、Cunning 天赋、penlight/录音机/PDT），被填在了 "Company bureaucrat" 的名字下。真正的 Company bureaucrat 是 Strength 2, Agility 3, Wits 4, Empathy 5 / Health 3 / Observation 3, Comtech 3, Manipulation 2, Medical Aid 2 / Personal Safety / PDT + Seegson P-DAT。建议要么把这个 actor 改名为 "EV - CORPORATE EXECUTIVE"、另建一个真正的 bureaucrat，要么整条重填。本模组**不**自动修它 —— 逐字段打补丁只会造出两边都不像的杂交体。
