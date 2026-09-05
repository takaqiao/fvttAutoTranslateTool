> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 3 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 3: K7 补丁自退休内核 `kernel/patches.mjs` 与自检套件 `kernel/selftest.mjs`

**Files:**
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/kernel/patches.mjs`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/kernel/selftest.mjs`
- Modify: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/main.mjs`（只动三处，全部按锚点原文定位、**不按行号**：`/* AEA-ANCHOR: imports */` 之后加两行 import；`export const api` 字面量里把 `patches: null` 与 `selftest: null` 两个槽换成简写属性；`/* AEA-ANCHOR: ready.patches */` 之后加一行 `await patches.applyAll();`）
- Modify: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/lang/en.json`、`lang/cn.json`（在顶层 `AEA` 对象内追加 `Patch` 与 `Selftest` 两个子对象，共 3 个键）
- Test: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/test/patches.test.mjs`
- Test: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/test/selftest.test.mjs`
- Test: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/test/main-lifecycle.test.mjs`（追加一个 `describe` 块，3 条用例）

本任务是内核任务，**不注册任何特性、也不注册任何真实补丁**，因此不往 `main.mjs` 的 `FEATURES` / `REPAIRS` 两个数组里加任何东西，也**不往** `/* AEA-ANCHOR: init */`、`/* AEA-ANCHOR: i18nInit */`、`/* AEA-ANCHOR: diceSoNiceReady */`、`/* AEA-ANCHOR: ready.registry */`、`/* AEA-ANCHOR: ready.rollbus */`、`/* AEA-ANCHOR: ready.cards */` 这六个锚点里插任何东西。真实补丁属于各自的修复文件。

**Interfaces:**

- Consumes:
  - `scripts/const.mjs` → `MID === "alien-evolved-automation"`、`I18N === "AEA"`（该文件已存在，只导出契约列出的那批常量，不得增删）。
  - `scripts/main.mjs` → 已存在，已逐字写好十个锚点注释。本任务只用到其中两个：顶部 import 区末尾的 `/* AEA-ANCHOR: imports */`，以及 `Hooks.once("ready", async () => { ... })` 体内第二个 ready 子锚点 `/* AEA-ANCHOR: ready.patches */`。ready 体的首行已经是 `await waitForWorldSettled({ timeoutMs: 10000, pollMs: 100 });`（世界就绪守卫：等 Babele 翻译层与系统 `apps/init.mjs` 的首次冒险导入落定）。文件里还有一个模块级 `export const api = { ... }` 对象字面量，八个键的值全是 `null`，占两行：`  features: null, patches: null, resolver: null, registry: null,` 与 `  rollBus: null, diceBarrier: null, cards: null, selftest: null,`。init 钩子体的最后一句是 `publishApi()`，它执行 `game.modules.get(MID).api = api`，**按引用**发布，因此对象里的值在模块求值时就已确定。
  - `test/stubs/foundry.mjs` → 三个导出：`installFoundryStub(options) -> ctx`、`uninstallFoundryStub()`、`foundryStubContext()`。这个桩由别的任务独占实现，**本任务只读不改，禁止在自己的测试文件里就地造 `globalThis.game` 或用私有 Map 顶替 `game.settings`**。本任务用到的 `options`：`isGM`（默认 `true`）、`systemVersion`（默认 `"4.1.13"`）、`i18n`（`key -> text` 词典）。本任务用到的 `ctx` 成员：`ctx.isGM`（可写，写完即刻反映到 `game.user.isGM`）、`ctx.notifications`、`ctx.wrappers`、`ctx.hooks.once`。本任务依赖的桩行为，全部来自契约 §0.3 的行为契约：`game.settings` 由 `ctx.settings` 真后端支撑（`get` 未注册键抛错、`register` 播种默认值）、`game.i18n` 提供 `localize` 与 `format`（`format` 会把 `{ids}` 这类占位替换掉，未知键原样回声键名）、`libWrapper.register` 记进 `ctx.wrappers` 且**同一目标重复注册时抛出与真实 libWrapper 同形的错误**、`game.system.version` 由 `options.systemVersion` 决定、`game.modules.get(MID)` 返回一个可写对象。
- Produces:
  - `export function pureVersionApplies(systemVersion, {minSystem, fixedIn}) -> boolean`（纯函数，不碰任何 Foundry 全局）
  - `export const patches = { register(def), applyAll(), status() }`
  - `def = {id, type:"WRAPPER"|"MIXED"|"OVERRIDE"|"DATA"|"HOOK", target:string|null, minSystem:string|null, fixedIn:string|null, probe():boolean, apply():void|Promise}` —— 字段就是这七个，**不多不少**（契约 §4/K7 定死）。
  - `applyAll()` → `Promise<{applied: string[], skipped: string[], retired: string[]}>`
  - `status()` → `[{id, type, target, applied, reason, fixedIn}]`（**六个键，逐字固定**）。`reason` 取值集合：`"pending" | "applied" | "below-min-system" | "fixed-in-release" | "probe-clean" | "probe-error" | "apply-error" | "libwrapper-missing"`。`type` / `target` / `fixedIn` 是只读元数据，供诊断面板与手工验收读取。
  - `export const selftest = { register({id, label, run}), runAll(), results() }`，`runAll()` → `Promise<[{id, label, ok, detail}]>`。`label` 收的是 **i18n 键**（形如 `AEA.Selftest.<something>`），`register()` 原样保存、绝不在登记时本地化；本地化发生在 `runAll()` 里。
  - `scripts/kernel/selftest.mjs` **由本任务唯一创建**。其余任何任务只 `import { selftest } from "../kernel/selftest.mjs"` 后调 `register()`，**不得重复创建该文件**。`patch:` 这个 id 前缀由补丁内核独占：`patches.register()` 会自动为每个补丁登记一条 `patch:<id>` 自检条目，别的任务的自检 id 不得以 `patch:` 开头（撞车会在 `register()` 里直接抛 `duplicate selftest id`）。**这条自动登记同时替一期 b 的每条修复满足了「必须有一条 selftest 条目」的要求**——修复作者只要 `patches.register()`，探针条目就有了，不必也不该再手写一条同名的。
  - `main.mjs` 顶部 `import { patches } ...` 与 `import { selftest } ...` 两行：**这两行由本任务落地，是全模组唯一一处 import 它们进 main.mjs 的地方**。别的任务若要在 `main.mjs` 里调 `selftest.register(...)`，直接用这个裸标识符即可，不要再写第二遍 import（同名 import 两次是 `SyntaxError: Identifier 'selftest' has already been declared`）。
  - `api.patches`、`api.selftest` 两个槽：本任务把 `export const api` 字面量里这两个 `null` 换成对应对象。**不替换 `api` 这个对象本身，不增删键，不在 init 里给 `api` 赋值。**

  单测证明不了的三条断言，各自的去处（这张表就是它们的审计线索，删掉任何一条自检条目都会让这里出现孤行）：

  | 单测覆盖不到的断言 | 覆盖它的东西 |
  |---|---|
  | 我们自己实现的版本比较真的等于 `foundry.utils.isNewerVersion` | 自检条目 `patch:version-compare-parity` + 下面的 MANUAL VERIFICATION 第 1 步 |
  | `apply()` 在真实 libWrapper 上确实以**零参**被调用、被包裹的系统函数照常执行 | MANUAL VERIFICATION 第 5 步 |
  | 退休提示只弹给 GM、整场只弹一次（真实 `ui.notifications` 的行为） | 桩层由 `test/patches.test.mjs` 覆盖判断逻辑，真实弹窗由 MANUAL VERIFICATION 第 4 步覆盖 |

---

**这个内核为什么必须存在**

系统 `alienrpg` 有 269 条已坐实的缺陷，本模组要补其中一部分。危险不在于补，而在于**上游把同一个洞补上之后我们还在补**——两边各修一次，结果是双重修复，症状比原缺陷更难查。系统自己帮不上忙：它的迁移块 `systems/alienrpg/module/alienrpg.mjs:311-326` 整段是注释掉的（那十几行以 `// const currentVersion = game.settings.get("alienrpg", "systemMigrationVersion")` 起头的代码从未启用）。所以每个补丁必须自带两道闸：

- **版本区间**（`minSystem` / `fixedIn`）——静态的、便宜的、看清单就能判断的。
- **`probe()`** ——运行时的行为断言。**返回 `true` 表示缺陷仍在**（该装补丁），返回 `false` 表示上游已经修好了（退休，并提示 GM 一次）。`probe` 比版本号可靠：上游可能在一个没写进 changelog 的补丁版里修掉，也可能换一种方式修而函数名不变。

一个真实的 probe 长什么样，看系统 `module/helpers/alienRPGBaseDice.mjs:13` 与 `:43`：`AlienRPGBaseDie` 与 `AlienRPGStressDie` 两个类的 `get total()` 都写成 `return this.results.length`，也就是返回**骰池大小**而不是点数和。对应的 probe 就是「造一个已知结果的骰子实例，问它 `total` 是不是等于池子大小」——行为断言，跟版本号无关。（真正的补丁属于修复包，不在本任务范围；这里只建承载它的内核。）

**写 probe 的人必须遵守的一条**：`probe()` 的两个分支都要能被测到。做法是把判断抽成一个接受「被检查的东西」作为参数的纯函数，probe 只负责把系统里的真家伙喂给它：`probeXxx()` 就是 `pureXxxIsBuggy(SomeSystemClass.prototype.method)`。这样单测可以喂两份替身——一份复刻 4.1.13 的坏行为（断言返回 `true`），一份是修好的行为（断言返回 `false`）。一个硬写成 `return true` 的 probe 能通过所有现存测试，然后永远双重修复下去。这条规则下面会原样写进 `patches.register()` 的 JSDoc，因为那是补丁作者真正会读到的地方。

**`apply()` 的语义（定死）**

```js
// def = {
//   id:        string
//   type:      "WRAPPER" | "MIXED" | "OVERRIDE" | "DATA" | "HOOK"
//   target:    string|null   // 纯元数据；applyAll 绝不据此替你注册任何东西
//   minSystem: string|null, fixedIn: string|null
//   probe():   boolean       // true = 缺陷仍在（该装）；false = 上游已修（退休）
//   apply():   void|Promise  // 无参自装器：自己调 libWrapper.register(...)、
//                            // 自己挂钩子、或自己改数据。applyAll 只调它一次。
// }
```

`apply()` 是**无参自装器**，不是包装函数本身。理由：修复包里有三类修复根本不是函数包裹——纯数据修复、宏安装、`preCreateToken` 钩子。靠 `target`/`type` 驱动的自动注册表达不了它们，只有「你自己装，我只负责判断该不该装、并记录你装没装成」这一种语义能覆盖全部五类。`type` 里的 `DATA` 与 `HOOK` 就是给后两类用的。

**libWrapper 是什么**：一个前置库模组，用来让多个模组安全地包裹同一个函数而不互相踩。调用形态是 `libWrapper.register(packageId, target, fn, type)`，`target` 是一条点分路径字符串（比如 `"CONFIG.Actor.documentClass.prototype.rollStress"`），`type` 三选一：

- `WRAPPER` —— 你的 `fn` 第一个参数是 `wrapped`，你**必须**调用它；libWrapper 保证这类包裹排在最前面（最外层）。
- `MIXED` —— 同样收 `wrapped`，但允许你在某些分支下不调用它；跑在 `WRAPPER` 的内层。
- `OVERRIDE` —— 完全替换，不收 `wrapped`；同一个 target 上只允许一个 `OVERRIDE`。

补丁的 `apply()` 自己调它，`type`/`target` 只是写进 def 供诊断阅读。

**一条必须写进 JSDoc 的禁令**：lib-wrapper 对**同一个包 + 同一个目标**的第二次注册会抛
`A wrapper for '<target>' (ID=<n>) has already been registered by <module>.`。本模组的 RollBus 内核**独占**这四个目标：
`game.alienrpg.yze.yzeRoll`、`CONFIG.Actor.documentClass.prototype.abilityRoll`、`CONFIG.Item.documentClass.prototype.roll`、`CONFIG.Actor.documentClass.prototype.pushRoll`。
任何补丁若在 `apply()` 里对这四个之一调 `libWrapper.register`，都会撞车。因为 `patches.applyAll()` 排在 `rollBus.install()` **之前**，撞车时先装上的是补丁、后面 `rollBus.install()` 才抛——一次崩掉整个 ready。要介入这四个目标的修复，一律经 `rollBus.addStage(target, {id, order, around})` 排队。这条禁令在 `register()` 的 JSDoc 里写死，并由 Step 6 里那条「碰撞记成 apply-error」的用例守住下限：撞车至少不会被误报成 `applied`。

（另一条给 `DATA` 类补丁的告诫：改原型时不能改用子类换名字。`RollTerm.fromData` 按 `data.class` 的名字匹配，改名会让历史聊天卡静默降级成普通 `Die`、丢掉骰面图。）

**关于「补丁的开关」**：契约 §7 要求一期 b 的每条修复都有开关，且**关掉即原样放行、不需要重载世界**。这个开关**不在本内核里**——`applyAll()` 刻意**不查** `features`，`patches.mjs` 因此对 `features.mjs` 零依赖。修复自己在 `register()` 里 `features.register({id, default:"full", gmOnly:true})`，并在 `apply()` 装上的那个包裹/钩子**执行时**查 `features.enabled(id)`。这样 GM 中途关掉一条修复，下一次掷骰就已经原样放行；若改成「applyAll 看开关决定装不装」，开关只在下次 ready 生效，与契约要求的热生效相反。

**为什么本任务同时建自检套件**：设计文档要求「需要 Foundry 的部分做模组内自检套件，每条断言同时就是补丁的『缺陷还在不在』探针」——一份代码两用。这个共用点就在 `patches.register()` 上：每注册一个补丁，自动生成一条同名 `patch:<id>` 自检条目。

**一期明确不做的两件事**（写在这里，免得被当成漏项）：设计文档 §6.2 的「ready 时可选执行自检」与 §6.3 的 269 条集中回归基线，一期都不做。依据是契约本身：§5 的三段调用清单是穷举的、ready 段里没有 `selftest.runAll()`；§1 冻结了 `const.mjs`，加不进 `SETTING_SELFTEST_ON_READY` 这种开关键；§4 的 `selftest` 也没有批量登记入口。一期自检的唯一入口就是 GM 手工在控制台调 `api.selftest.runAll()`；集中回归基线由「每条补丁自带 `probe()` + 自动生成的 `patch:<id>` 条目」替代，推到二期再做。

---

- [ ] **Step 1: 写会失败的版本比较测试**

新建 `test/patches.test.mjs`：

```js
// test/patches.test.mjs
import { describe, it, expect } from "vitest"
import { pureVersionApplies } from "../scripts/kernel/patches.mjs"

describe("pureVersionApplies", () => {
	it("treats minSystem as inclusive", () => {
		expect(pureVersionApplies("4.1.13", { minSystem: "4.1.13", fixedIn: null })).toBe(true)
		expect(pureVersionApplies("4.1.14", { minSystem: "4.1.13", fixedIn: null })).toBe(true)
		expect(pureVersionApplies("4.1.12", { minSystem: "4.1.13", fixedIn: null })).toBe(false)
	})

	it("treats fixedIn as exclusive — the named release already carries the fix", () => {
		expect(pureVersionApplies("4.1.13", { minSystem: "4.1.13", fixedIn: "4.2.0" })).toBe(true)
		expect(pureVersionApplies("4.1.99", { minSystem: "4.1.13", fixedIn: "4.2.0" })).toBe(true)
		expect(pureVersionApplies("4.2.0", { minSystem: "4.1.13", fixedIn: "4.2.0" })).toBe(false)
		expect(pureVersionApplies("4.2.1", { minSystem: "4.1.13", fixedIn: "4.2.0" })).toBe(false)
	})

	it("treats a null fixedIn as 'nobody has fixed it yet'", () => {
		expect(pureVersionApplies("9.9.9", { minSystem: "4.1.13", fixedIn: null })).toBe(true)
		expect(pureVersionApplies("9.9.9", { minSystem: "4.1.13" })).toBe(true)
	})

	it("treats a null minSystem as 'every version has it'", () => {
		expect(pureVersionApplies("1.0.0", { minSystem: null, fixedIn: "4.2.0" })).toBe(true)
		expect(pureVersionApplies("1.0.0", {})).toBe(true)
	})

	it("compares numeric parts numerically, not as strings", () => {
		// the case a naive string compare gets wrong: "4.1.9" < "4.1.13"
		expect(pureVersionApplies("4.1.9", { minSystem: "4.1.13" })).toBe(false)
		expect(pureVersionApplies("4.1.13", { minSystem: "4.1.9" })).toBe(true)
		expect(pureVersionApplies("4.10.0", { minSystem: "4.9.0" })).toBe(true)
	})

	it("matches Foundry's asymmetric handling of a missing trailing part", () => {
		// Foundry: isNewerVersion("1.2.0", "1.2") === true, isNewerVersion("1.2", "1.2.0") === false
		expect(pureVersionApplies("1.2.0", { minSystem: "1.2" })).toBe(true)
		expect(pureVersionApplies("1.2", { minSystem: "1.2.0" })).toBe(true)
		expect(pureVersionApplies("1.2", { fixedIn: "1.2.0" })).toBe(true)
		expect(pureVersionApplies("1.2.0", { fixedIn: "1.2" })).toBe(false)
	})

	it("compares a non-numeric part as a string, the way Foundry does", () => {
		expect(pureVersionApplies("4.2.0-beta", { minSystem: "4.2.0" })).toBe(true)
		expect(pureVersionApplies("4.1.13", { minSystem: "4.1.13-beta" })).toBe(false)
	})

	it("refuses to decide when the system version is unknown", () => {
		expect(pureVersionApplies(null, { minSystem: "4.1.13" })).toBe(false)
		expect(pureVersionApplies(undefined, { minSystem: "4.1.13" })).toBe(false)
		expect(pureVersionApplies("", { minSystem: "4.1.13" })).toBe(false)
	})

	it("touches no Foundry global", () => {
		expect(globalThis.game).toBeUndefined()
		expect(pureVersionApplies("4.1.13", { minSystem: "4.1.13" })).toBe(true)
	})
})
```

第 6 条那两行不对称是**故意**照抄 Foundry 的实际行为，不是笔误：Foundry 的 `isNewerVersion` 只遍历 `v1` 的段，`v0` 段不够时直接判 v1 更新，所以 `"1.2.0"` 比 `"1.2"` 新，而 `"1.2"` 不比 `"1.2.0"` 新。若我们自作聪明补零对齐，补丁在 `fixedIn: "1.2"` 这种写法下的退休时机就会跟 Foundry 自己的依赖检查错开。

- [ ] **Step 2: 跑它，看它红**

Run: `cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation" && npx vitest run test/patches.test.mjs`
Expected: FAIL — `Error: Failed to load url ../scripts/kernel/patches.mjs (resolved id: .../scripts/kernel/patches.mjs). Does the file exist?`，9 条用例一条都不执行。

- [ ] **Step 3: 写 pureVersionApplies**

新建 `scripts/kernel/patches.mjs`：

```js
// scripts/kernel/patches.mjs
import { MID, I18N } from "../const.mjs"

/**
 * WRAPPER / MIXED / OVERRIDE describe a libWrapper registration the patch's own
 * apply() performs. DATA is a one-shot mutation (a document update, a prototype
 * poke, a macro install); HOOK is a patch that attaches a Foundry hook. Nothing
 * in this file branches on the value except the libWrapper availability check —
 * it is diagnostic metadata.
 */
const TYPES = ["WRAPPER", "MIXED", "OVERRIDE", "DATA", "HOOK"]

/** Types whose apply() is expected to call libWrapper.register itself. */
const LIBWRAPPER_TYPES = ["WRAPPER", "MIXED", "OVERRIDE"]

/** id -> normalized def */
const REGISTERED = new Map()
/** id -> {applied:boolean, reason:string} */
const STATE = new Map()
/** The GM is told about retirements once per session, not once per patch. */
let retirementNotified = false

/** Foundry's Number.isNumeric: not null/undefined/"", and not NaN. */
function isNumericPart(part) {
	if (part === null || part === undefined || part === "") return false
	return +part === +part
}

/**
 * Faithful reimplementation of foundry.utils.isNewerVersion — true iff v1 is
 * strictly newer than v0. No semver dependency: Foundry's own comparison is NOT
 * semver (it walks v1's dot-separated parts only, compares numeric parts
 * numerically and everything else as strings, and calls v1 newer the moment v0
 * runs out of parts). Matching it exactly matters, because Foundry uses the same
 * rule for the dependency checks in module.json.
 */
function isNewerVersion(v1, v0) {
	if (typeof v1 === "number" && typeof v0 === "number") return v1 > v0
	const v1Parts = String(v1 ?? "").split(".")
	const v0Parts = String(v0 ?? "").split(".")
	for (let i = 0; i < v1Parts.length; i++) {
		const p1 = v1Parts[i]
		const p0 = v0Parts[i]
		if (p0 === undefined) return true
		if (isNumericPart(p1) && isNumericPart(p0)) {
			if (Number(p1) !== Number(p0)) return Number(p1) > Number(p0)
		} else if (p1 !== p0) {
			return p1 > p0
		}
	}
	return false
}

/**
 * Is a patch in range for this system version?
 *   minSystem — inclusive lower bound; null means "every version has the defect"
 *   fixedIn   — exclusive upper bound; the named release already carries the fix,
 *               so it and everything after are out of range. null means nobody
 *               has fixed it yet.
 * Pure: touches no Foundry global.
 */
export function pureVersionApplies(systemVersion, { minSystem, fixedIn } = {}) {
	if (systemVersion === null || systemVersion === undefined || systemVersion === "") return false
	if (minSystem && isNewerVersion(minSystem, systemVersion)) return false
	if (fixedIn && !isNewerVersion(fixedIn, systemVersion)) return false
	return true
}
```

- [ ] **Step 4: 跑它，看它绿**

Run: `npx vitest run test/patches.test.mjs`
Expected: PASS — 9 passed。

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "$(cat <<'EOF'
feat(kernel/patches): pureVersionApplies 版本区间判定

不引 semver：Foundry 自己的 isNewerVersion 就不是 semver —— 它只遍历 v1 的
点分段，数字段按数字比、其余按字符串比，v0 段用尽时直接判 v1 更新。因此
"1.2.0" 比 "1.2" 新，而 "1.2" 不比 "1.2.0" 新。这份不对称是照抄的，不是笔误：
Foundry 用同一套规则做 module.json 的依赖检查，我们补零对齐反而会跟它错开。

minSystem 含端点，fixedIn 不含端点（上游标称修好的那个版本本身已经带修复）。
systemVersion 未知时一律返回 false —— 拿不准就不装补丁。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: 追加会失败的补丁注册表测试**

把 `test/patches.test.mjs` 顶部的第一行 import 改成 `import { describe, it, expect, beforeEach, afterEach, vi } from "vitest"`，在它下面加一行 `import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs"`，并在文件末尾追加：

```js
const PKG = "alien-evolved-automation"
const RETIRED_MESSAGE = "Upstream fixed these, so their patches retired: {ids}"

function makeDef(overrides = {}) {
	return {
		id: "demo",
		type: "DATA",
		target: null,
		minSystem: "4.1.13",
		fixedIn: null,
		probe: () => true,
		apply() {},
		...overrides,
	}
}

/** The six-key diagnostic row status() must produce, with makeDef's defaults. */
function row(overrides = {}) {
	return { id: "demo", type: "DATA", target: null, applied: false, reason: "pending", fixedIn: null, ...overrides }
}

/**
 * The contract pins the ctx field NAME (ctx.notifications, ctx.wrappers) but not
 * the key each row uses for severity / package id, so read both spellings rather
 * than let this task go red over a stub key name it does not own.
 */
function infoNotices(ctx) {
	return ctx.notifications.filter((n) => (n.type ?? n.level) === "info").map((n) => n.message)
}
function wrapperRows(ctx) {
	return ctx.wrappers.map((w) => ({ pkg: w.module ?? w.pkg, target: w.target, type: w.type }))
}

describe("patches registry", () => {
	let ctx
	let patches

	beforeEach(async () => {
		vi.resetModules()
		ctx = installFoundryStub({
			isGM: true,
			systemVersion: "4.1.13",
			i18n: { "AEA.Patch.Retired": RETIRED_MESSAGE },
		})
		;({ patches } = await import("../scripts/kernel/patches.mjs"))
	})

	afterEach(() => uninstallFoundryStub())

	it("normalizes a def and starts it pending", () => {
		patches.register(makeDef())
		expect(patches.status()).toEqual([row()])
	})

	it("reports exactly the six diagnostic keys, metadata included", () => {
		patches.register(makeDef({ id: "meta", type: "MIXED", target: "CONFIG.Actor.documentClass.prototype.rollStress", fixedIn: "4.3.0" }))
		const [entry] = patches.status()
		// Pinned literally: a diagnostics panel reads type/target/fixedIn off this row,
		// so dropping one of them is a breaking change, not a refactor.
		expect(Object.keys(entry).sort()).toEqual(["applied", "fixedIn", "id", "reason", "target", "type"])
		expect(entry).toEqual({
			id: "meta",
			type: "MIXED",
			target: "CONFIG.Actor.documentClass.prototype.rollStress",
			applied: false,
			reason: "pending",
			fixedIn: "4.3.0",
		})
	})

	it("rejects a def that cannot be honoured", () => {
		expect(() => patches.register(makeDef({ id: "" }))).toThrow(/string id/)
		expect(() => patches.register(makeDef({ type: "PATCH" }))).toThrow(/invalid type/)
		expect(() => patches.register(makeDef({ probe: null }))).toThrow(/probe/)
		expect(() => patches.register(makeDef({ apply: null }))).toThrow(/apply/)
		patches.register(makeDef())
		expect(() => patches.register(makeDef())).toThrow(/duplicate patch id/)
	})

	it("accepts all five types, including the two that are not function wrapping", () => {
		for (const type of ["WRAPPER", "MIXED", "OVERRIDE", "DATA", "HOOK"]) {
			expect(() => patches.register(makeDef({ id: `t-${type}`, type }))).not.toThrow()
		}
		expect(patches.status().map((p) => p.type)).toEqual(["WRAPPER", "MIXED", "OVERRIDE", "DATA", "HOOK"])
	})

	it("skips a patch whose minSystem is above the running system, without probing", async () => {
		let probed = false
		patches.register(makeDef({ id: "future", minSystem: "4.2.0", probe: () => { probed = true; return true } }))
		const result = await patches.applyAll()
		expect(result).toEqual({ applied: [], skipped: ["future"], retired: [] })
		expect(probed).toBe(false)
		expect(patches.status()).toEqual([row({ id: "future", reason: "below-min-system" })])
	})

	it("retires a patch whose fixedIn release is already running, without probing", async () => {
		let probed = false
		patches.register(makeDef({ id: "landed", fixedIn: "4.1.13", probe: () => { probed = true; return true } }))
		const result = await patches.applyAll()
		expect(result).toEqual({ applied: [], skipped: [], retired: ["landed"] })
		expect(probed).toBe(false)
		expect(patches.status()).toEqual([row({ id: "landed", reason: "fixed-in-release", fixedIn: "4.1.13" })])
	})

	it("probes first, then calls the patch's own zero-arg installer", async () => {
		const order = []
		patches.register(
			makeDef({
				id: "live",
				type: "WRAPPER",
				target: "CONFIG.Actor.documentClass.prototype.rollStress",
				probe: () => {
					order.push("probe")
					return true
				},
				apply: (...args) => {
					// applyAll must pass NOTHING: apply is an installer, not a wrapper fn
					order.push(`apply(${args.length})`)
					libWrapper.register(
						PKG,
						"CONFIG.Actor.documentClass.prototype.rollStress",
						function (wrapped, ...rest) {
							return wrapped(...rest)
						},
						"WRAPPER"
					)
				},
			})
		)

		const result = await patches.applyAll()

		expect(result.applied).toEqual(["live"])
		expect(order).toEqual(["probe", "apply(0)"])
		// the wrapper reached libWrapper because the PATCH registered it, not applyAll
		expect(wrapperRows(ctx)).toEqual([
			{ pkg: PKG, target: "CONFIG.Actor.documentClass.prototype.rollStress", type: "WRAPPER" },
		])
		expect(patches.status()).toEqual([
			row({ id: "live", type: "WRAPPER", target: "CONFIG.Actor.documentClass.prototype.rollStress", applied: true, reason: "applied" }),
		])
	})

	it("awaits an async installer before reporting it applied", async () => {
		const done = []
		patches.register(
			makeDef({
				id: "slow",
				apply: async () => {
					await Promise.resolve()
					done.push("finished")
				},
			})
		)
		const result = await patches.applyAll()
		expect(done).toEqual(["finished"])
		expect(result.applied).toEqual(["slow"])
	})

	it("never touches libWrapper on behalf of a DATA patch", async () => {
		const apply = vi.fn()
		patches.register(makeDef({ id: "prototype-poke", type: "DATA", target: null, apply }))
		const result = await patches.applyAll()
		expect(result.applied).toEqual(["prototype-poke"])
		expect(apply).toHaveBeenCalledTimes(1)
		expect(ctx.wrappers).toHaveLength(0)
	})

	it("retires a patch whose probe reports the defect gone, and tells the GM once", async () => {
		patches.register(makeDef({ id: "upstream-fixed-it", probe: () => false }))
		patches.register(makeDef({ id: "also-fixed", probe: () => false }))

		const first = await patches.applyAll()
		expect(first.retired).toEqual(["upstream-fixed-it", "also-fixed"])

		expect(infoNotices(ctx)).toEqual(["Upstream fixed these, so their patches retired: upstream-fixed-it, also-fixed"])

		await patches.applyAll()
		expect(infoNotices(ctx)).toHaveLength(1)
	})

	it("keeps the retirement notice off a player's screen", async () => {
		ctx.isGM = false
		patches.register(makeDef({ id: "upstream-fixed-it", probe: () => false }))
		expect((await patches.applyAll()).retired).toEqual(["upstream-fixed-it"])
		expect(ctx.notifications).toEqual([])
	})

	it("skips a patch whose probe throws, and leaves the system untouched", async () => {
		const spy = vi.spyOn(console, "error").mockImplementation(() => {})
		const apply = vi.fn()
		patches.register(makeDef({ id: "boom", apply, probe: () => { throw new Error("probe blew up") } }))
		const result = await patches.applyAll()
		expect(result.skipped).toEqual(["boom"])
		expect(apply).not.toHaveBeenCalled()
		expect(patches.status()).toEqual([row({ id: "boom", reason: "probe-error" })])
		expect(spy).toHaveBeenCalled()
		spy.mockRestore()
	})

	it("skips a patch whose apply throws rather than reporting it applied", async () => {
		const spy = vi.spyOn(console, "error").mockImplementation(() => {})
		patches.register(makeDef({ id: "bad-apply", apply: () => { throw new Error("nope") } }))
		const result = await patches.applyAll()
		expect(result).toEqual({ applied: [], skipped: ["bad-apply"], retired: [] })
		expect(patches.status()).toEqual([row({ id: "bad-apply", reason: "apply-error" })])
		spy.mockRestore()
	})

	it("reports a patch that collides on an already-wrapped target as apply-error, never as applied", async () => {
		const spy = vi.spyOn(console, "error").mockImplementation(() => {})
		// lib-wrapper refuses a second registration of the same target by the same
		// package. This is the failure mode a patch hits when it tries to wrap one of
		// the four roll entry points the RollBus kernel owns. The one thing this layer
		// must guarantee is that such a patch is NOT filed as "applied" — a patch that
		// claims success while installing nothing is invisible forever.
		libWrapper.register(PKG, "CONFIG.Actor.documentClass.prototype.rollStress", (w, ...a) => w(...a), "WRAPPER")
		patches.register(
			makeDef({
				id: "collides",
				type: "WRAPPER",
				target: "CONFIG.Actor.documentClass.prototype.rollStress",
				apply: () =>
					libWrapper.register(PKG, "CONFIG.Actor.documentClass.prototype.rollStress", (w, ...a) => w(...a), "WRAPPER"),
			})
		)
		const result = await patches.applyAll()
		expect(result).toEqual({ applied: [], skipped: ["collides"], retired: [] })
		expect(patches.status()[0].reason).toBe("apply-error")
		expect(patches.status()[0].applied).toBe(false)
		spy.mockRestore()
	})

	it("refuses to run a libWrapper-shaped patch when libWrapper is gone", async () => {
		const spy = vi.spyOn(console, "error").mockImplementation(() => {})
		const apply = vi.fn()
		patches.register(makeDef({ id: "needs-lw", type: "MIXED", target: "Some.thing.method", apply }))
		delete globalThis.libWrapper
		const result = await patches.applyAll()
		expect(result.skipped).toEqual(["needs-lw"])
		expect(apply).not.toHaveBeenCalled()
		expect(patches.status()).toEqual([
			row({ id: "needs-lw", type: "MIXED", target: "Some.thing.method", reason: "libwrapper-missing" }),
		])
		spy.mockRestore()
	})

	it("never installs the same patch twice when applyAll runs again", async () => {
		const apply = vi.fn()
		patches.register(makeDef({ id: "live", apply }))
		await patches.applyAll()
		const second = await patches.applyAll()
		expect(apply).toHaveBeenCalledTimes(1)
		expect(second).toEqual({ applied: [], skipped: ["live"], retired: [] })
		expect(patches.status()).toEqual([row({ id: "live", applied: true, reason: "applied" })])
	})

	it("retires a patch when the world has been upgraded past its fixedIn release", async () => {
		uninstallFoundryStub()
		vi.resetModules()
		ctx = installFoundryStub({ isGM: true, systemVersion: "4.2.0", i18n: { "AEA.Patch.Retired": RETIRED_MESSAGE } })
		;({ patches } = await import("../scripts/kernel/patches.mjs"))

		patches.register(makeDef({ id: "live", fixedIn: "4.2.0" }))
		const result = await patches.applyAll()

		expect(result).toEqual({ applied: [], skipped: [], retired: ["live"] })
		expect(infoNotices(ctx)[0]).toBe("Upstream fixed these, so their patches retired: live")
	})
})
```

`apply(${args.length})` 那条断言是这一层最重要的一条：它把「`applyAll` 绝不给 `apply` 传参」钉死。若把 `apply` 当包装函数交给 libWrapper，libWrapper 会以 `apply(wrapped, ...args)` 调它，被包裹的系统函数从此再也不会被调用——所有补丁静默变成黑洞。

- [ ] **Step 7: 跑它，看它红**

Run: `npx vitest run test/patches.test.mjs -t "normalizes a def and starts it pending"`
Expected: FAIL — `TypeError: Cannot read properties of undefined (reading 'register')`（`patches` 导出还不存在，`beforeEach` 里的解构拿到 `undefined`）。

- [ ] **Step 8: 写 register / applyAll / status**

在 `scripts/kernel/patches.mjs` 末尾追加：

```js
function mark(bucket, id, applied, reason) {
	STATE.set(id, { applied, reason })
	bucket.push(id)
}

export const patches = {
	/**
	 * @param {{id:string, type:"WRAPPER"|"MIXED"|"OVERRIDE"|"DATA"|"HOOK",
	 *          target?:string|null, minSystem?:string|null, fixedIn?:string|null,
	 *          probe:()=>boolean, apply:()=>void|Promise<void>}} def
	 *
	 *   probe() returning TRUE means the defect is still present — install the patch.
	 *   FALSE means upstream fixed it — retire, and tell the GM once.
	 *
	 *   RULE FOR PATCH AUTHORS: both branches of probe() must be testable. Put the
	 *   judgement in a pure predicate that takes the thing under inspection as an
	 *   argument — pureXxxIsBuggy(SomeClass.prototype.method) — and let probe() be
	 *   the one line that feeds it the real system object. Unit-test that predicate
	 *   with two fixtures: the 4.1.13 body (expect true) and a corrected body
	 *   (expect false). A probe hardcoded to `return true` passes every test that
	 *   exists and then double-fixes forever.
	 *
	 *   apply() is a ZERO-ARGUMENT installer. It receives nothing and is called
	 *   exactly once; it is responsible for its own libWrapper.register call, its
	 *   own hook, or its own data change. `target` and `type` are metadata only.
	 *   Unknown keys on the def are ignored rather than rejected.
	 *
	 *   DO NOT libWrapper.register any of these four targets from apply():
	 *     game.alienrpg.yze.yzeRoll
	 *     CONFIG.Actor.documentClass.prototype.abilityRoll
	 *     CONFIG.Item.documentClass.prototype.roll
	 *     CONFIG.Actor.documentClass.prototype.pushRoll
	 *   The RollBus kernel owns them, and lib-wrapper throws on a second
	 *   registration of the same target by the same package. Queue up with
	 *   rollBus.addStage(target, {id, order, around}) instead.
	 *
	 *   This kernel deliberately does NOT consult features.enabled(): a repair's
	 *   on/off switch has to take effect without a world reload, so the check
	 *   belongs inside the wrapper/hook that apply() installs, at call time.
	 * @returns {object} the normalized def
	 */
	register(def) {
		if (!def || typeof def.id !== "string" || def.id.length === 0) {
			throw new Error(`${MID} | patches.register needs a non-empty string id`)
		}
		if (REGISTERED.has(def.id)) {
			throw new Error(`${MID} | duplicate patch id: ${def.id}`)
		}
		if (!TYPES.includes(def.type)) {
			throw new Error(`${MID} | patch ${def.id} has invalid type ${def.type}; expected one of ${TYPES.join(", ")}`)
		}
		if (typeof def.probe !== "function") {
			throw new Error(`${MID} | patch ${def.id} needs a runnable probe() — a patch with no defect assertion can never retire`)
		}
		if (typeof def.apply !== "function") {
			throw new Error(`${MID} | patch ${def.id} needs an apply()`)
		}
		const normalized = {
			id: def.id,
			type: def.type,
			target: typeof def.target === "string" && def.target.length > 0 ? def.target : null,
			minSystem: def.minSystem ?? null,
			fixedIn: def.fixedIn ?? null,
			probe: def.probe,
			apply: def.apply,
		}
		REGISTERED.set(normalized.id, normalized)
		STATE.set(normalized.id, { applied: false, reason: "pending" })
		return normalized
	},

	/**
	 * Called once from main.mjs during the ready hook, after the world settle gate.
	 * Async because a patch's installer may be async.
	 * @returns {Promise<{applied:string[], skipped:string[], retired:string[]}>}
	 */
	async applyAll() {
		const result = { applied: [], skipped: [], retired: [] }
		const systemVersion = game.system?.version ?? null

		for (const def of REGISTERED.values()) {
			if (STATE.get(def.id)?.applied) {
				// Already installed this session. Running the installer again would
				// stack a second wrapper on the same target, or repeat a data change.
				result.skipped.push(def.id)
				continue
			}

			// Cheap static gate first: never probe a patch that is out of range.
			const aboveMin = pureVersionApplies(systemVersion, { minSystem: def.minSystem, fixedIn: null })
			if (!aboveMin) {
				mark(result.skipped, def.id, false, "below-min-system")
				continue
			}
			if (!pureVersionApplies(systemVersion, { minSystem: def.minSystem, fixedIn: def.fixedIn })) {
				mark(result.retired, def.id, false, "fixed-in-release")
				continue
			}

			let defectPresent
			try {
				defectPresent = def.probe() === true
			} catch (err) {
				console.error(`${MID} | probe threw for patch ${def.id}; leaving the system untouched`, err)
				mark(result.skipped, def.id, false, "probe-error")
				continue
			}

			if (!defectPresent) {
				mark(result.retired, def.id, false, "probe-clean")
				continue
			}

			if (LIBWRAPPER_TYPES.includes(def.type) && typeof libWrapper === "undefined") {
				console.error(
					`${MID} | patch ${def.id} is a ${def.type} patch but libWrapper is not available; not installing it`
				)
				mark(result.skipped, def.id, false, "libwrapper-missing")
				continue
			}

			try {
				// Zero arguments, on purpose. apply() installs itself.
				await def.apply()
			} catch (err) {
				// Includes the lib-wrapper "already been registered" throw, which is what
				// a patch gets for trying to wrap a target the RollBus kernel owns.
				console.error(`${MID} | apply threw for patch ${def.id}`, err)
				mark(result.skipped, def.id, false, "apply-error")
				continue
			}

			mark(result.applied, def.id, true, "applied")
		}

		notifyRetired(result.retired)
		return result
	},

	/**
	 * One diagnostic row per registered patch, in registration order. The shape is
	 * fixed by the interface contract; type/target/fixedIn are read-only metadata a
	 * diagnostics panel and the manual acceptance steps rely on.
	 * @returns {Array<{id:string, type:string, target:string|null, applied:boolean,
	 *                  reason:string, fixedIn:string|null}>}
	 *   reason is one of: "pending" (registered, applyAll has not reached it yet)
	 *   | "applied" | "below-min-system" | "fixed-in-release" | "probe-clean"
	 *   (upstream fixed it) | "probe-error" | "apply-error" | "libwrapper-missing".
	 */
	status() {
		return [...REGISTERED.values()].map((def) => {
			const state = STATE.get(def.id) ?? { applied: false, reason: "pending" }
			return {
				id: def.id,
				type: def.type,
				target: def.target,
				applied: state.applied,
				reason: state.reason,
				fixedIn: def.fixedIn,
			}
		})
	},
}

function notifyRetired(retired) {
	if (retired.length === 0 || retirementNotified) return
	if (!game.user?.isGM) return
	retirementNotified = true
	const ids = retired.join(", ")
	const message = game.i18n?.format?.(`${I18N}.Patch.Retired`, { ids }) ?? `${MID} | patches retired: ${ids}`
	ui.notifications?.info(message)
}
```

- [ ] **Step 9: 跑它，看它绿**

Run: `npx vitest run test/patches.test.mjs`
Expected: PASS — 26 passed（9 条纯函数 + 17 条注册表）。

- [ ] **Step 10: 补上退休提示的语言键并跑全套**

`lang/en.json` 顶层 `AEA` 对象内追加：

```json
		"Patch": {
			"Retired": "Upstream fixed these, so their patches retired: {ids}"
		}
```

`lang/cn.json` 顶层 `AEA` 对象内追加：

```json
		"Patch": {
			"Retired": "上游已修复以下缺陷，对应补丁已自动退休：{ids}"
		}
```

Run: `npm test`
Expected: PASS — 现有测试文件全绿，其中 `test/patches.test.mjs` 26 passed；语言包护栏测试仍绿（两份语言包压平后键集依然一一对应，且顶层仍只有 `AEA` 一个键）。

- [ ] **Step 11: 提交**

```bash
git add -A && git commit -m "$(cat <<'EOF'
feat(kernel/patches): 无参自装器语义、版本闸、probe 优先与六键诊断行

apply() 是无参自装器，不是包装函数本身：它自己调 libWrapper.register、自己挂
钩子、自己改数据，applyAll 只负责判断该不该装并记录装没装成。target/type 纯属
元数据。理由是修复包里有三类修复根本不是函数包裹（纯数据修复、宏安装、
preCreateToken 钩子），靠 target/type 驱动的自动注册表达不了它们；type 因此扩到
五个，多出 DATA 与 HOOK。测试里专门断言 apply 收到的参数个数是 0 —— 若把它当
包装函数交给 libWrapper，被包裹的系统函数将永远不再被调用。

status() 按契约返回 {id, type, target, applied, reason, fixedIn} 六个键并有一条
逐字锁死键集的用例：诊断面板要读 type/target/fixedIn，只报 id/applied/reason 的
三键行会让「这条补丁包了谁」无从查起。

先过便宜的静态闸再跑 probe：minSystem 之下直接 skip（below-min-system），
fixedIn 已发布则直接退休（fixed-in-release），两种情况都不触发 probe —— 在上游
已经改过的代码上跑我们的行为断言，本身就可能抛。

apply 抛异常一律降级为 skip 并 console.error，其中包含 lib-wrapper 的
「already been registered」—— 那是补丁去抢 RollBus 独占的四个掷骰目标时的报错。
这一层保证撞车的补丁不会被记成 applied：一条自称装好、实则什么也没装的补丁
永远查不出来。JSDoc 里点名那四个目标，并指向 rollBus.addStage。

本内核刻意不查 features.enabled：修复的开关必须免重载生效，所以查询点在
apply() 装上的那个包裹/钩子里，不在这儿。patches.mjs 因此对 features.mjs 零依赖。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 12: 写会失败的自检套件测试**

新建 `test/selftest.test.mjs`：

```js
// test/selftest.test.mjs
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest"
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs"

describe("kernel/selftest.mjs without any Foundry present", () => {
	let selftest

	beforeEach(async () => {
		vi.resetModules()
		// No Foundry stub on purpose: registering and running must work before the
		// world exists, and must never throw for want of a global.
		;({ selftest } = await import("../scripts/kernel/selftest.mjs"))
	})

	it("runs with no Foundry globals installed at all", async () => {
		expect(globalThis.game).toBeUndefined()
		selftest.register({ id: "a", label: "AEA.Selftest.First", run: () => ({ ok: true, detail: "fine" }) })
		await expect(selftest.runAll()).resolves.toEqual([
			{ id: "a", label: "AEA.Selftest.First", ok: true, detail: "fine" },
		])
	})

	it("rejects an entry that cannot be run", () => {
		expect(() => selftest.register({ run: () => ({ ok: true }) })).toThrow(/string id/)
		expect(() => selftest.register({ id: "a" })).toThrow(/run/)
		selftest.register({ id: "a", run: () => ({ ok: true }) })
		expect(() => selftest.register({ id: "a", run: () => ({ ok: true }) })).toThrow(/duplicate selftest id/)
	})

	it("falls back to the id when no label is given, and to '' when no detail is given", async () => {
		selftest.register({ id: "bare", run: () => ({ ok: true }) })
		await expect(selftest.runAll()).resolves.toEqual([{ id: "bare", label: "bare", ok: true, detail: "" }])
	})

	it("treats anything that is not an explicit ok:true as a failure", async () => {
		selftest.register({ id: "undef", run: () => undefined })
		selftest.register({ id: "truthy", run: () => ({ ok: 1, detail: "not a boolean" }) })
		const results = await selftest.runAll()
		expect(results.map((r) => [r.id, r.ok])).toEqual([["undef", false], ["truthy", false]])
	})

	it("awaits an async check and turns a thrown error into a failed result", async () => {
		selftest.register({ id: "slow", run: async () => ({ ok: true, detail: "waited" }) })
		selftest.register({ id: "boom", run: () => { throw new Error("kaboom") } })
		const results = await selftest.runAll()
		expect(results[0]).toEqual({ id: "slow", label: "slow", ok: true, detail: "waited" })
		expect(results[1].ok).toBe(false)
		expect(results[1].detail).toMatch(/kaboom/)
	})

	it("remembers the last run and starts empty", async () => {
		expect(selftest.results()).toEqual([])
		selftest.register({ id: "a", run: () => ({ ok: true, detail: "d" }) })
		const results = await selftest.runAll()
		expect(selftest.results()).toEqual(results)
	})
})

describe("kernel/selftest.mjs label localization", () => {
	let selftest

	beforeEach(async () => {
		vi.resetModules()
		installFoundryStub({ i18n: { "AEA.Selftest.Known": "Known check" } })
		;({ selftest } = await import("../scripts/kernel/selftest.mjs"))
	})

	afterEach(() => uninstallFoundryStub())

	it("localizes the label at run time, not at register time", async () => {
		// register() happens during Foundry's `init` hook, which runs BEFORE
		// `i18nInit` — the language files are not loaded yet, so localizing there
		// would freeze the raw key into the entry forever. The stored def therefore
		// keeps the key, and runAll() is what turns it into text.
		const stored = selftest.register({ id: "known", label: "AEA.Selftest.Known", run: () => ({ ok: true }) })
		expect(stored.label).toBe("AEA.Selftest.Known")

		selftest.register({ id: "unknown", label: "AEA.Selftest.NoSuchKey", run: () => ({ ok: true }) })
		const results = await selftest.runAll()
		expect(results.map((r) => r.label)).toEqual(["Known check", "AEA.Selftest.NoSuchKey"])
	})
})
```

- [ ] **Step 13: 跑它，看它红**

Run: `npx vitest run test/selftest.test.mjs`
Expected: FAIL — `Error: Failed to load url ../scripts/kernel/selftest.mjs (resolved id: .../scripts/kernel/selftest.mjs). Does the file exist?`，7 条用例一条都不执行。

- [ ] **Step 14: 写 kernel/selftest.mjs**

新建 `scripts/kernel/selftest.mjs`：

```js
// scripts/kernel/selftest.mjs
/**
 * The module's in-world assertion suite. THIS FILE IS CREATED ONCE, HERE.
 * Everything else in the module only does `import { selftest }` and calls
 * register(); nobody creates a second suite and nobody re-declares this file.
 *
 * Half of this module cannot be unit tested: the real libWrapper chain, the 3D
 * dice animation, chat render timing. That half gets registered here instead, and
 * a GM runs it from the console against a live world:
 *
 *   await game.modules.get("alien-evolved-automation").api.selftest.runAll()
 *
 * That console call is the ONLY entry point in phase 1 — nothing runs the suite
 * automatically during `ready`.
 *
 * A `run()` returns {ok, detail}. `ok` means "the check completed and the world
 * matches what this module believes about it" — it is NOT "the system is bug
 * free". A patch probe registered here reports ok:true whether the defect is
 * present or gone; what it reports in `detail` is which of the two it found.
 * ok:false means the check could not decide, which is the state that needs a
 * human.
 *
 * LABELS ARE i18n KEYS. register() runs during Foundry's `init` hook, which is
 * BEFORE `i18nInit`: no language file is loaded yet, so localizing at that moment
 * would bake the raw key into the entry. The def is stored verbatim and runAll()
 * does the localization, falling back to the key itself when there is no game
 * object (unit tests) or no translation (Foundry echoes unknown keys anyway).
 *
 * ID NAMESPACE: ids beginning with "patch:" belong to the patch kernel, which
 * registers one entry per patch automatically. Nothing else may claim that prefix.
 */
import { MID } from "../const.mjs"

/** id -> {id, label, run} */
const REGISTERED = new Map()
/** Results of the most recent runAll(). */
let LAST = []

/** Localize a label key if there is a live Foundry to do it; otherwise echo it. */
function localizeLabel(label) {
	const i18n = globalThis.game?.i18n
	if (typeof i18n?.localize !== "function") return label
	const text = i18n.localize(label)
	return typeof text === "string" && text.length > 0 ? text : label
}

export const selftest = {
	/**
	 * @param {{id:string, label?:string, run:()=>({ok:boolean, detail?:string})|Promise<{ok:boolean, detail?:string}>}} entry
	 *   label is an i18n KEY (shaped like "AEA.Selftest.<something>"), not localized
	 *   text. It is stored exactly as given; runAll() localizes it. Omit it and the
	 *   id is used as the label.
	 * @returns {object} the normalized entry
	 */
	register({ id, label, run } = {}) {
		if (typeof id !== "string" || id.length === 0) {
			throw new Error(`${MID} | selftest.register needs a non-empty string id`)
		}
		if (REGISTERED.has(id)) {
			throw new Error(`${MID} | duplicate selftest id: ${id}`)
		}
		if (typeof run !== "function") {
			throw new Error(`${MID} | selftest ${id} needs a runnable run()`)
		}
		const entry = { id, label: typeof label === "string" && label.length > 0 ? label : id, run }
		REGISTERED.set(id, entry)
		return entry
	},

	/** @returns {Promise<Array<{id:string, label:string, ok:boolean, detail:string}>>} */
	async runAll() {
		const results = []
		for (const entry of REGISTERED.values()) {
			const label = localizeLabel(entry.label)
			let outcome
			try {
				outcome = await entry.run()
			} catch (err) {
				results.push({ id: entry.id, label, ok: false, detail: `threw: ${err?.message ?? err}` })
				continue
			}
			results.push({
				id: entry.id,
				label,
				// Only an explicit boolean true passes. A truthy 1 is a bug in the check.
				ok: outcome?.ok === true,
				detail: typeof outcome?.detail === "string" ? outcome.detail : "",
			})
		}
		LAST = results
		return results
	},

	/** Results of the most recent runAll(), or [] if it has never run. */
	results() {
		return LAST
	},
}
```

- [ ] **Step 15: 跑它，看它绿**

Run: `npx vitest run test/selftest.test.mjs`
Expected: PASS — 7 passed。

- [ ] **Step 16: 提交**

```bash
git add -A && git commit -m "$(cat <<'EOF'
feat(kernel/selftest): 模组内自检套件（全模组唯一一份）

模组有一半测不了：真实的 libWrapper 链、DsN 动画、聊天渲染时序。这一半登记到
这里，由 GM 在真实世界的控制台里跑 api.selftest.runAll()。本文件只在这一处创建，
其余模块一律 import 后 register，不再造第二个套件；patch: 前缀留给补丁内核独占。
一期没有任何自动跑法，那条控制台调用就是唯一入口。

label 收的是 i18n 键而不是已本地化文本：register 发生在 init 钩子里，那时
i18nInit 还没跑、语言包尚未加载，此刻 localize 只会把键名原样回声回来并被永久
冻进条目。因此 register 原样保存整个 def，本地化推迟到 runAll —— 没有 game
对象（单测）或没有译文时回落到键名本身，与 Foundry 对未知键的行为一致。

run() 返回 {ok, detail}。ok 的含义是「这条检查跑完了，且世界与本模组的认知一致」，
不是「系统没 bug」——补丁探针无论缺陷在不在都报 ok:true，它在 detail 里说清楚
自己看到的是哪一种；ok:false 表示这条检查没能下判断，那才是要人来看的状态。
只有显式的 boolean true 算通过，truthy 的 1 不算：检查本身写错了要能被发现。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 17: 追加会失败的「一份代码两用」测试**

在 `test/patches.test.mjs` 的 `describe("patches registry")` **内部**末尾追加三条用例（`selftest` 是模块级单例状态，必须跟 `patches` 用同一次 `vi.resetModules()` 之后的动态 import 取，所以在用例里现取，不要加到文件顶部的静态 import 区）：

```js
	it("turns every registered patch into a self-test entry — one body of code, two uses", async () => {
		const { selftest } = await import("../scripts/kernel/selftest.mjs")
		patches.register(makeDef({ id: "still-broken", probe: () => true }))
		patches.register(makeDef({ id: "fixed-upstream", probe: () => false }))
		await patches.applyAll()

		const results = await selftest.runAll()
		const byId = Object.fromEntries(results.map((r) => [r.id, r]))

		expect(byId["patch:still-broken"].ok).toBe(true)
		expect(byId["patch:still-broken"].detail).toContain("DATA patch still-broken")
		expect(byId["patch:still-broken"].detail).toMatch(/defect still present/)
		expect(byId["patch:still-broken"].detail).toMatch(/applied/)

		expect(byId["patch:fixed-upstream"].ok).toBe(true)
		expect(byId["patch:fixed-upstream"].detail).toMatch(/defect gone/)
		expect(byId["patch:fixed-upstream"].detail).toMatch(/probe-clean/)
	})

	it("reports a probe that cannot decide as a failed self-test rather than a passing one", async () => {
		const { selftest } = await import("../scripts/kernel/selftest.mjs")
		const spy = vi.spyOn(console, "error").mockImplementation(() => {})
		patches.register(makeDef({ id: "undecidable", probe: () => { throw new Error("cannot tell") } }))
		await patches.applyAll()

		const entry = (await selftest.runAll()).find((r) => r.id === "patch:undecidable")
		expect(entry.ok).toBe(false)
		expect(entry.detail).toMatch(/cannot tell/)
		spy.mockRestore()
	})

	it("checks our version comparison against Foundry's own, and refuses to pass without it", async () => {
		const { selftest } = await import("../scripts/kernel/selftest.mjs")
		// Drive both branches explicitly rather than leaning on whatever the shared
		// stub happens to ship: absent reference => parity UNPROVEN, present
		// reference => real case-by-case comparison.
		const saved = foundry.utils.isNewerVersion
		delete foundry.utils.isNewerVersion

		const entry = (await selftest.runAll()).find((r) => r.id === "patch:version-compare-parity")
		expect(entry).toBeDefined()
		expect(entry.ok).toBe(false)
		expect(entry.detail).toMatch(/isNewerVersion/)

		foundry.utils.isNewerVersion = (v1, v0) => {
			const a = String(v1).split(".")
			const b = String(v0).split(".")
			for (let i = 0; i < a.length; i++) {
				if (b[i] === undefined) return true
				const na = Number(a[i])
				const nb = Number(b[i])
				if (Number.isFinite(na) && Number.isFinite(nb)) {
					if (na !== nb) return na > nb
				} else if (a[i] !== b[i]) return a[i] > b[i]
			}
			return false
		}
		const again = (await selftest.runAll()).find((r) => r.id === "patch:version-compare-parity")
		expect(again.ok).toBe(true)
		expect(again.detail).toMatch(/7 cases/)

		if (saved === undefined) delete foundry.utils.isNewerVersion
		else foundry.utils.isNewerVersion = saved
	})
```

- [ ] **Step 18: 跑它，看它红**

Run: `npx vitest run test/patches.test.mjs -t "one body of code, two uses"`
Expected: FAIL — `TypeError: Cannot read properties of undefined (reading 'ok')`（`byId["patch:still-broken"]` 是 `undefined`：`patches.register` 还没往自检套件里登记任何东西）。

- [ ] **Step 19: 把探针接进自检套件**

在 `scripts/kernel/patches.mjs` 顶部的 `import { MID, I18N } from "../const.mjs"` 之后追加一行：

```js
import { selftest } from "./selftest.mjs"
```

在 `patches.register` 里的 `STATE.set(normalized.id, { applied: false, reason: "pending" })` 之后、`return normalized` 之前插入：

```js
		// One body of code, two uses: the same assertion that decides whether to
		// install the patch is also the assertion a GM can re-run after a system
		// upgrade to find out which patches should retire. This is also the
		// selftest entry every phase-1b repair is required to have — registering a
		// patch gets you one, so do not hand-write a second entry for the same id.
		const descriptor = `${normalized.type} patch ${normalized.id}${normalized.target ? ` on ${normalized.target}` : ""}`
		selftest.register({
			id: `patch:${normalized.id}`,
			label: `${I18N}.Selftest.PatchProbe`,
			run: () => {
				let present
				try {
					present = normalized.probe() === true
				} catch (err) {
					return { ok: false, detail: `${descriptor}: probe could not decide: ${err?.message ?? err}` }
				}
				const state = STATE.get(normalized.id) ?? { applied: false, reason: "pending" }
				return {
					ok: true,
					detail: present
						? `${descriptor}: defect still present in the running system; patch state: ${state.reason}`
						: `${descriptor}: defect gone upstream; patch state: ${state.reason}`,
				}
			},
		})
```

- [ ] **Step 20: 加模块级的版本比较对照条目**

在 `scripts/kernel/patches.mjs` 的**文件末尾**追加：

```js
/**
 * [v1, v0, expected] — expected is what foundry.utils.isNewerVersion returns for
 * "is v1 strictly newer than v0". The middle rows are the asymmetries a naive
 * semver comparison gets wrong, and they are exactly the rows that decide when a
 * patch with a two-part fixedIn retires.
 */
const VERSION_PARITY_CASES = [
	["4.1.13", "4.1.13", false],
	["4.1.13", "4.1.9", true],
	["4.1.9", "4.1.13", false],
	["1.2.0", "1.2", true],
	["1.2", "1.2.0", false],
	["4.10.0", "4.9.0", true],
	["4.2.0-beta", "4.2.0", true],
]

// Registered at module scope rather than from a hook: it is a property of this
// file, not of any one patch. selftest touches no Foundry global at import time,
// so importing this module under vitest is safe; the run() body is what needs a
// live Foundry, and it is not executed until someone calls runAll().
selftest.register({
	id: "patch:version-compare-parity",
	label: `${I18N}.Selftest.VersionParity`,
	run: () => {
		const reference = globalThis.foundry?.utils?.isNewerVersion
		if (typeof reference !== "function") {
			return { ok: false, detail: "foundry.utils.isNewerVersion is unavailable, so parity is unproven" }
		}
		const mismatches = []
		for (const [v1, v0, expected] of VERSION_PARITY_CASES) {
			const ours = isNewerVersion(v1, v0)
			const theirs = reference(v1, v0) === true
			if (ours !== theirs || theirs !== expected) {
				mismatches.push(`isNewerVersion("${v1}","${v0}") ours=${ours} foundry=${theirs} expected=${expected}`)
			}
		}
		return mismatches.length === 0
			? { ok: true, detail: `${VERSION_PARITY_CASES.length} cases match foundry.utils.isNewerVersion` }
			: { ok: false, detail: mismatches.join("; ") }
	},
})
```

这条是本模组唯一一处「拿真 Foundry 验我们自己的实现」：对照必须发生在真实客户端里，Step 17 的第三条用例只是把两个分支都走一遍，喂进去的参照实现是测试自己写的替身，不构成背书。

- [ ] **Step 21: 补上两个自检标签的语言键**

`lang/en.json` 顶层 `AEA` 对象内追加：

```json
		"Selftest": {
			"PatchProbe": "Patch defect probe",
			"VersionParity": "Version comparison matches foundry.utils.isNewerVersion"
		}
```

`lang/cn.json` 顶层 `AEA` 对象内追加：

```json
		"Selftest": {
			"PatchProbe": "补丁缺陷探针",
			"VersionParity": "版本比较与 foundry.utils.isNewerVersion 一致"
		}
```

- [ ] **Step 22: 跑全套，看它全绿**

Run: `npm test`
Expected: PASS — 全套绿；`test/patches.test.mjs` 29 passed，`test/selftest.test.mjs` 7 passed；语言包护栏测试仍绿。

- [ ] **Step 23: 提交**

```bash
git add -A && git commit -m "$(cat <<'EOF'
feat(kernel/patches): 每个补丁的 probe 自动登记为自检条目

设计文档要求「每条自检断言同时就是补丁的缺陷探针」，共用点落在
patches.register 上：注册补丁的同时生成一条 patch:<id> 自检条目。这同时替一期 b
的每条修复满足了「必须有一条 selftest 条目」的要求 —— 注册补丁就有，不必也不该
再手写一条同名的。条目的 ok 表示「这条探针跑出了结论」，detail 里先写清是哪条
补丁（type/id/target），再说结论是「缺陷仍在」还是「上游已修」；探针抛异常才算
ok:false —— 那是唯一需要人来看的状态。

另加一条模块级条目 patch:version-compare-parity，在真实客户端里把我们自己实现的
isNewerVersion 与 foundry.utils.isNewerVersion 逐例对照；参照函数不存在时明确报
ok:false 并说明「parity unproven」，不会静默通过 —— 拿同一作者写的副本当基准
等于自己验自己。

两条自检的 label 都是 i18n 键，译文进 lang/*.json 的 AEA.Selftest 下。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 24: 写会失败的接线测试**

打开 `test/main-lifecycle.test.mjs`，确认顶部 import 区已有下面三行，**缺哪行补哪行**（同名重复 import 会报 `SyntaxError: Identifier 'readFileSync' has already been declared`，下一步跑测试立刻能看到）：

```js
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest"
import { readFileSync } from "node:fs"
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs"
```

在**文件末尾**追加一个独立的 `describe` 块（独立是为了不依赖别处 `describe` 内部的变量作用域）：

```js
describe("main.mjs patch wiring", () => {
	let ctx

	beforeEach(() => {
		vi.resetModules()
		ctx = installFoundryStub({ isGM: true, systemVersion: "4.1.13" })
		// The ready hook waits on the system's own semaphore before doing anything.
		// systems/alienrpg/module/helpers/settings.mjs:209 registers it as a world
		// String defaulting to "", and module/apps/migratefolders.js sets it to
		// "busy" while the first adventure import runs. Seeding the registered
		// default lets the settle gate pass at once instead of burning its timeout.
		game.settings.register("alienrpg", "ARPGSemaphore", {
			name: "Semaphore Flag",
			scope: "world",
			config: false,
			type: String,
			default: "",
		})
	})

	afterEach(() => uninstallFoundryStub())

	/**
	 * Import main.mjs while capturing the lifecycle callbacks it registers, so the
	 * test can await the async `ready` body. Wrapping the stub's own Hooks.once is
	 * shape-independent — it does not assume how ctx.hooks stores its rows.
	 */
	async function loadMainCapturingLifecycle() {
		const captured = {}
		const realOnce = Hooks.once
		Hooks.once = (name, fn) => {
			captured[name] = fn
			return realOnce.call(Hooks, name, fn)
		}
		try {
			const mod = await import("../scripts/main.mjs")
			return { mod, captured }
		} finally {
			Hooks.once = realOnce
		}
	}

	it("exposes the patch kernel and the self-test suite on the public api", async () => {
		const { mod, captured } = await loadMainCapturingLifecycle()
		const { patches } = await import("../scripts/kernel/patches.mjs")
		const { selftest } = await import("../scripts/kernel/selftest.mjs")

		// The api literal carries real objects from module evaluation onward — no
		// hook has to fire for a console user to reach them.
		expect(mod.api.patches).toBe(patches)
		expect(mod.api.selftest).toBe(selftest)
		// All eight contract keys survive: this task replaces two nulls, it does not
		// rebuild the object or add keys.
		expect(Object.keys(mod.api)).toEqual([
			"features", "patches", "resolver", "registry",
			"rollBus", "diceBarrier", "cards", "selftest",
		])

		captured.init()
		// publishApi hangs the same object off the module document, so the console
		// path a GM uses reaches the very same kernels.
		expect(game.modules.get("alien-evolved-automation").api.patches).toBe(patches)
		expect(game.modules.get("alien-evolved-automation").api.selftest).toBe(selftest)
	})

	it("applies patches during ready, never during init", async () => {
		const { mod, captured } = await loadMainCapturingLifecycle()
		captured.init()

		const applied = []
		mod.api.patches.register({
			id: "probe-demo",
			type: "DATA",
			target: null,
			minSystem: "4.1.13",
			fixedIn: null,
			probe: () => true,
			apply: () => applied.push("ran"),
		})
		expect(applied).toEqual([])

		await captured.ready()

		expect(applied).toEqual(["ran"])
		expect(mod.api.patches.status()).toEqual([
			{ id: "probe-demo", type: "DATA", target: null, applied: true, reason: "applied", fixedIn: null },
		])
	})

	it("applies patches after the world settle gate and at its own ready sub-anchor", () => {
		// A source-order guard, not a behaviour test: probes are behaviour assertions
		// against a live world, so applyAll must never run before Babele and the
		// system's first adventure import have settled, and it must sit at the
		// ready.patches sub-anchor so the neighbouring kernels keep their order.
		// Nothing else in the suite catches someone inserting a line above the gate.
		const source = readFileSync(new URL("../scripts/main.mjs", import.meta.url), "utf8")
		const ready = source.indexOf('Hooks.once("ready"')
		expect(ready).toBeGreaterThan(-1)
		const gate = source.indexOf("waitForWorldSettled(", ready)
		const anchor = source.indexOf("/* AEA-ANCHOR: ready.patches */", ready)
		const apply = source.indexOf("patches.applyAll()", ready)
		expect(gate).toBeGreaterThan(ready)
		expect(anchor).toBeGreaterThan(gate)
		expect(apply).toBeGreaterThan(anchor)
		// exactly one call site, so nobody can "fix" an ordering complaint by adding a second
		expect(source.split("patches.applyAll()").length - 1).toBe(1)
	})
})
```

- [ ] **Step 25: 跑它，看它红**

Run: `npx vitest run test/main-lifecycle.test.mjs`
Expected: FAIL — 新加的 3 条全红，其余原有用例仍绿：
- `exposes the patch kernel...` → `AssertionError: expected null to be { register: [Function], applyAll: [Function], status: [Function] }`（`mod.api.patches` 还是 Task 1 写下的 `null`）
- `applies patches during ready...` → `TypeError: Cannot read properties of null (reading 'register')`
- `applies patches after the world settle gate...` → `AssertionError: expected -1 to be greater than <数字>`（源码里还搜不到 `patches.applyAll()`，`indexOf` 返回 `-1`）

- [ ] **Step 26: 在 main.mjs 加两行 import**

打开 `scripts/main.mjs`，找到顶部那行逐字写作 `/* AEA-ANCHOR: imports */` 的锚点注释，在它**之后**另起两行：

```js
import { patches } from "./kernel/patches.mjs";
import { selftest } from "./kernel/selftest.mjs";
```

这两行是全模组唯一一处把它们引进 `main.mjs` 的地方。别的任务在 `main.mjs` 里调 `selftest.register(...)` 时直接用裸标识符，不要写第二遍。

- [ ] **Step 27: 把两个内核填进 api 字面量的对应槽**

在同一个文件里找到 `export const api = {` 那个对象字面量。它有两行、八个键、值全是 `null`。**只改两处子串，不动键名、不动键数、不替换这个对象**：

- 第一行 `  features: null, patches: null, resolver: null, registry: null,` 里，把 `patches: null` 改成 `patches`
- 第二行 `  rollBus: null, diceBarrier: null, cards: null, selftest: null,` 里，把 `selftest: null` 改成 `selftest`

改完这两行长这样：

```js
  features: null, patches, resolver: null, registry: null,
  rollBus: null, diceBarrier: null, cards: null, selftest,
```

用的是对象简写属性，键名一字未变。Step 24 那条 `Object.keys(mod.api)` 断言就是守这一点的：八个键、顺序不变。

- [ ] **Step 28: 在 ready.patches 子锚点后调用 applyAll**

在同一个文件的 `Hooks.once("ready", async () => { ... })` 体里找到那行逐字写作 `/* AEA-ANCHOR: ready.patches */` 的子锚点注释，在它**之后**另起一行（缩进两个空格，与该函数体其余行一致）：

```js
  await patches.applyAll();
```

只插在这个子锚点后面，不要插到 `/* AEA-ANCHOR: ready.registry */`、`ready.rollbus`、`ready.cards` 三个兄弟锚点旁边——四个子锚点在文件里的先后顺序就是执行顺序，各插各的位就自动得到「世界就绪守卫 → 文档绑定解析 → 装补丁 → RollBus 上线 → 卡片挂点上线 → 特性与修复 install」这条链。装补丁必须在世界就绪守卫之后（probe 是对活着的世界做的行为断言，翻译层与冒险导入没落定就跑，结论不作数），也必须在 RollBus 上线之前（骰池钳制那类 MIXED 补丁要排在 RollBus 的 WRAPPER 内层，libWrapper 按注册先后分层）。

- [ ] **Step 29: 跑全套，看它全绿**

Run: `npm test`
Expected: PASS — 全套绿；`test/patches.test.mjs` 29 passed，`test/selftest.test.mjs` 7 passed，`test/main-lifecycle.test.mjs` 比上一次多 3 条通过用例。

- [ ] **Step 30: 提交**

```bash
git add -A && git commit -m "$(cat <<'EOF'
feat(main): ready.patches 子锚点调用 applyAll，patches 与 selftest 填进 api 两个槽

按锚点原文接线，不按行号：imports 锚点后两行 import，api 字面量里把
patches: null 与 selftest: null 两个槽换成简写属性，ready.patches 子锚点后一行
await patches.applyAll()。不替换 api 这个对象、不增删键 —— 八个槽各有各的属主，
谁也别动别人的那一个。用字面量而不是 init 里赋值，是因为 publishApi 按引用发布，
字面量在模块求值时就已定型，控制台拿到的和模块内部是同一批对象。

补丁在 ready 里装，且夹在世界就绪守卫与 rollBus.install() 之间：probe 是行为
断言，要等 Babele 与冒险包导入落定才有意义；而骰池钳制那类 MIXED 补丁必须排在
RollBus 的 WRAPPER 内层，libWrapper 按注册先后分层，晚注册就翻不过来。四个 ready
子锚点在文件里的先后顺序就是执行顺序，各插各的位即可。

另加一条源码顺序守卫用例：读 main.mjs 源文本，断言 ready 体里
waitForWorldSettled → ready.patches 锚点 → patches.applyAll 依次出现，且
applyAll 全文只有一个调用点。行为测试抓不到「有人把一行插到守卫上面」，只有
顺序断言抓得到。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

- [ ] **MANUAL VERIFICATION（Task 3 收尾，必做）**

有三件事单测证明不了：我们自己写的版本比较是否真的等于 `foundry.utils.isNewerVersion`；`libWrapper` 在真实 Foundry 里是否真的接受「补丁自装器」的调用形态；真实的 `ui.notifications` 是否只弹给 GM。下面六步逐条覆盖它们。全程用契约钉死的 api 路径书写：`game.modules.get("alien-evolved-automation").api.<成员>`。

1. 启动 Foundry，进入一个启用了本模组的 `alienrpg` 世界（系统版本 4.1.13），按 F12 打开控制台，执行：
   ```js
   await game.modules.get("alien-evolved-automation").api.selftest.runAll();
   ```
   预期返回一个数组，里面**至少**有一条 `{id: "patch:version-compare-parity", ok: true, detail: "7 cases match foundry.utils.isNewerVersion"}`，且它的 `label` 是已本地化的文本（中文界面「版本比较与 foundry.utils.isNewerVersion 一致」），不是裸键 `AEA.Selftest.VersionParity`——后者说明 `runAll()` 的本地化没生效或语言键漏写。
   若 `ok:false`，**停下来改实现并补测试**，不要继续——它说明本模组的版本比较与当前 Foundry 已经不一致，所有补丁的退休时机都会错开。`detail` 里会逐条列出哪几组对不上。

2. 控制台执行 `game.modules.get("alien-evolved-automation").api.patches.status()`，预期返回 `[]`（此刻尚未注册任何真实补丁），且不抛错。

3. 用系统里一个已知的真实缺陷验证「probe 写得出来」。系统 `module/helpers/alienRPGBaseDice.mjs:13` 与 `:43` 把 `AlienRPGBaseDie` / `AlienRPGStressDie` 的 `get total()` 写成 `return this.results.length`，返回的是骰池大小而不是点数和；这两个类在 `module/alienrpg.mjs:119-120` 注册为 `CONFIG.Dice.terms["b"]` 与 `["s"]`。在控制台执行下面这段**只读**探针：
   ```js
   const die = new CONFIG.Dice.terms.b({ number: 3, faces: 6 });
   die.results = [{ result: 6, active: true }, { result: 1, active: true }, { result: 4, active: true }];
   die.total;   // 缺陷仍在 => 3（池子大小）；上游已修 => 11（点数和）
   ```
   预期在 4.1.13 上看到 `3`。这就是将来那个补丁的 `probe()` 要断言的事实。

4. 验证退休路径与 GM 过滤。临时在 `scripts/main.mjs` 的 `/* AEA-ANCHOR: init */` 锚点注释之后插入一条一次性补丁（`patches` 已在文件顶部 import，直接可用），重载世界：
   ```js
   patches.register({
   	id: "retire-smoke-test", type: "DATA", target: null,
   	minSystem: "4.1.13", fixedIn: null,
   	probe: () => false, apply: () => {},
   });
   ```
   预期：世界载入完成后，GM 屏幕右上角弹出一条**蓝色 info** 通知，内容为「上游已修复以下缺陷，对应补丁已自动退休：retire-smoke-test」（英文界面为 "Upstream fixed these, so their patches retired: retire-smoke-test"）；控制台执行 `game.modules.get("alien-evolved-automation").api.patches.status()` 返回
   `[{id: "retire-smoke-test", type: "DATA", target: null, applied: false, reason: "probe-clean", fixedIn: null}]`；
   执行 `await game.modules.get("alien-evolved-automation").api.selftest.runAll()` 时该补丁对应的条目是
   `{id: "patch:retire-smoke-test", ok: true, detail: "DATA patch retire-smoke-test: defect gone upstream; patch state: probe-clean", label: "补丁缺陷探针"}`。
   再用一个**玩家**账号登录同一世界，预期**不弹**这条通知。

5. 验证自装器这条链真的通。把第 4 步插入的那段改成下面这条，**注意 target 用的是 `rollStress`，不是任何一个 RollBus 独占的掷骰入口**（`yzeRoll` / `abilityRoll` / `Item#roll` / `pushRoll` 四个已被 RollBus 用 libWrapper 占住，补丁再注册同一目标会让 `rollBus.install()` 抛错、一次崩掉整个 ready）：
   ```js
   patches.register({
   	id: "wrapper-smoke-test", type: "WRAPPER",
   	target: "CONFIG.Actor.documentClass.prototype.rollStress",
   	minSystem: "4.1.13", fixedIn: null,
   	probe: () => true,
   	apply: () => {
   		libWrapper.register(
   			"alien-evolved-automation",
   			"CONFIG.Actor.documentClass.prototype.rollStress",
   			function (wrapped, ...args) {
   				console.warn("AEA smoke wrapper hit");
   				return wrapped(...args);
   			},
   			"WRAPPER"
   		);
   	},
   });
   ```
   （`rollStress` 在 `systems/alienrpg/module/documents/actor.mjs:1061`，由角色卡上的压力骰按钮触发，且不经过 `yzeRoll`。）重载世界，在任意一张角色卡上点一次压力骰。预期两件事同时发生：控制台出现 `AEA smoke wrapper hit`，**且压力骰的聊天卡照常出现在聊天栏**。第二件是关键——它证明 `apply()` 被当成自装器调用（收到零个参数、自己注册了包装函数），而不是被当成包装函数交给 libWrapper。若卡片没出现，说明 `applyAll` 把 `apply` 本身交给了 libWrapper，被包裹的系统函数从此再也不会被调用。另外执行 `game.modules.get("alien-evolved-automation").api.patches.status()`，预期该行的 `type` 是 `"WRAPPER"`、`target` 是 `"CONFIG.Actor.documentClass.prototype.rollStress"`、`applied` 是 `true`——这三个字段就是六键诊断行存在的理由。

6. 验证完把第 4/5 步插进 `main.mjs` 的代码删掉，确认 `npm test` 仍全绿后再提交——真正的补丁属于修复包，不属于 `main.mjs`。
