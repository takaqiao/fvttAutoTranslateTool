> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 2 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 2: K5 特性开关内核 `kernel/features.mjs`

**Files:**
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/kernel/features.mjs`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/templates/feature-settings.hbs`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/test/features.test.mjs`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/test/features-wiring.test.mjs`
- Modify: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/main.mjs`（**恰好三处**，全部按锚点文本定位、绝不按行号：`/* AEA-ANCHOR: imports */` 之后加一行 import、`api` 字面量里把 `features: null,` 换成 `features,`、`/* AEA-ANCHOR: init */` 之后加一行调用）
- Modify: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/lang/en.json`、`lang/cn.json`（**只在已有的 `AEA` 对象里追加** `settings` 与 `mode` 两个子对象，不动别人写的键）
- Modify: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/styles/alien-evolved-automation.css`（追加设置面板样式）
- **不得修改**：`test/stubs/foundry.mjs`。共享测试桩由建立仓库骨架的那个任务独占实现，并由 `test/stub-fidelity.test.mjs` 逐条守卫；本任务**只读不改**，也**禁止**在自己的测试文件里就地造 `globalThis.game` 或用私有 Map 顶替 `game.settings`。

**Interfaces:**

- **Consumes:**
  - `scripts/const.mjs` → `MID = "alien-evolved-automation"`、`SETTING_FEATURES = "features"`、`SETTING_SCHEMA = "settingsSchemaVersion"`、`I18N = "AEA"`。
  - `test/stubs/foundry.mjs` → `installFoundryStub(options)`、`uninstallFoundryStub()`。桩已按契约把 Foundry 全局装在 `globalThis` 上（`game` / `ui` / `Hooks` / `ChatMessage` / `Roll` / `CONFIG` / `CONST` / `libWrapper` / `logger` / `foundry.{utils, applications}`），并保证下列本任务真正依赖的行为：**幂等**（重复 `installFoundryStub()` 会先隐式卸载再装，`uninstallFoundryStub()` 未安装时是安全空操作）；**设置是真后端**（`game.settings.register/get/set` 由桩内部的设置表支撑，`get` 未注册键**抛错**，`set` 返回 Promise，`registerMenu` 被记录）；**钩子会派发**（`Hooks.callAll(name, ...args)` 真正调用经 `Hooks.on` / `Hooks.once` 注册的回调，`once` 只触发一次）；`installFoundryStub({ moduleId, isGM })` 按选项决定模组 id 与当前用户是否 GM。**本任务不读桩的 `ctx` 内部字段**：需要观察「我的代码调了 `game.settings.register*` 什么参数」时一律用 `vi.spyOn` 挂在桩自己的对象上（spy 默认透传，既能取到入参又不破坏真后端），这样断言的是真实 Foundry API 面，而不是某个假货的私有账本。
  - `scripts/main.mjs` → 文件里逐字存在的三个装配点：`/* AEA-ANCHOR: imports */`、`export const api = {` 那个八槽全 `null` 的对象字面量、`/* AEA-ANCHOR: init */`。
- **Produces:**
  - `export function pureResolveMode(id, modes, defs)` → `"full"|"prompt"|"off"`（纯函数，不碰任何 Foundry 全局）
  - `export const features = { register(def), registerSettings(), mode(id), enabled(id), all() }` —— **只有这五个成员**。特性的 `install()` 由 `main.mjs` 里的 `FEATURES` 数组循环调用，**K5 不提供 `installAll()`**；`features.shouldRun(id)` / `features.confirm(id)` 是**二期符号**，本任务**不得发明**。Step 6 有一条用例把成员集合钉成恰好这五个。
  - `def = {id, default:"full"|"prompt"|"off", gmOnly:false, requires:[], hint:""}`，`register()` 返回归一化后的 def。
  - i18n 约定（契约钉死）：特性显示名取 `AEA.feature.<id>.name`、说明取 `AEA.feature.<id>.hint`，`<id>` 逐字等于 `features.register({id})` 里的 id；`game.i18n.has()` 为假时显示名退回裸 id。`def.hint` 是可选的**覆盖键**，只有在 `AEA.feature.<id>.hint` 不存在时才用。
  - 两个 world 设置 `alien-evolved-automation.features` / `alien-evolved-automation.settingsSchemaVersion`（都是 `config: false`），以及唯一一个 `restricted: true` 的设置菜单 `alien-evolved-automation.featuresMenu`。
  - **装配承诺**：`api.features` 从 `null` 变成本模块导出的 `features` 对象；`init` 阶段多出一行 `features.registerSettings();`。**不增删 `api` 的键，不替换 `api` 这个对象本身，不动其余七个 `null`，不占用四个 `ready` 子锚点中的任何一个**（`ready.registry` / `ready.patches` / `ready.rollbus` / `ready.cards` 分别属于另外四个内核，本内核在 `ready` 阶段没有任何调用）。

---

**给不熟悉 Foundry 的实现者：这一节出现的 Foundry 概念**

- **模组（module）**：一个放在 `Data/modules/<id>/` 下的目录，`module.json` 是它的清单。Foundry 启动时按清单加载 `esmodules` 里列出的 ES 模块。
- **生命周期钩子**：Foundry 用一个全局的 `Hooks` 事件总线广播启动阶段。`Hooks.once("init", fn)` 注册一个只跑一次的处理器；Foundry 在合适的时刻 `Hooks.callAll("init")`。`init` 是**唯一**允许注册设置的阶段。本模组的 `main.mjs` 由前一个任务建立，里面的生命周期钩子各带逐字的锚点注释。
- **设置（settings）**：`game.settings.register(namespace, key, config)` 声明一项设置；之后才能 `game.settings.get(namespace, key)` / `await game.settings.set(...)`。对未注册的 key 调 `get` 会**抛错**。`scope: "world"` 表示整个世界共用一个值、只有 GM 能写；`config: true` 表示直接摆在设置面板上，`config: false` 表示藏起来、由代码或自定义窗口读写。
- **设置菜单（settings menu）**：`game.settings.registerMenu(namespace, key, {…, type: SomeApplicationClass, restricted: true})` 在设置面板上摆一个**按钮**，点开是一个自定义窗口。`restricted: true` 是唯一真正把这个按钮限制给 GM 的地方。
- **ApplicationV2**：Foundry V13+ 的窗口基类，在 `foundry.applications.api` 下。`HandlebarsApplicationMixin` 给它加上 Handlebars 模板渲染能力，`static PARTS` 声明各模板片段，`_prepareContext()` 返回喂给模板的数据，`DEFAULT_OPTIONS.form.handler` 是表单提交处理器。
- **i18n**：`game.i18n.localize("A.B.C")` 按点分键取译文；`game.i18n.has(key)` 判断键存不存在。查找**大小写敏感**。语言包是 `lang/*.json`，嵌套结构，本模组自有的键一律在顶层 `AEA` 之下。

---

**为什么不照抄系统 `module/helpers/settings.mjs` 的写法**

先把那份文件从头读一遍（217 行，一个默认导出的函数；`wc -l` 与 `grep -c "game.settings.register("` 分别是 217 与 20，`grep -c "type: Boolean"` 是 11）。它的形态是：一个函数体里连着 20 次 `game.settings.register("alienrpg", ...)`，其中 11 次 `type: Boolean` 的开关，其余是 `config: false` 的颜色／字体字符串与内部标志。四个具体问题，逐条对应本模组的做法：

1. **平铺 boolean 不可组合。** 一个 boolean 只能表达开／关，表达不了「提示我一下再动手」。本模组的设计要求是「默认全自动，逐项可降级」，最小单位就是三态 `full|prompt|off`，boolean 从形状上就不够。
2. **没有依赖关系，只能半跑。** 系统里 `autopanic`（`settings.mjs:26-34`）和恐慌表查找之间有隐含依赖，但设置层不知道，关掉一个另一个照跑。本模组的特性彼此依赖成链，所以 `def.requires` 必须存在，且**前置关闭要传递性地把后继整条禁掉**——半跑的自动化比不跑更危险，因为玩家会以为它跑了。
3. **权限写反。** `settings.mjs:6-13` 的 `registerMenu(..., { restricted: false })`（`restricted: false` 在第 12 行）是唯一真正管权限的地方，而它写的是 `false`，任何玩家都能打开这个配置面板；反过来 `:21` 与 `:33` 两处 `game.settings.register` 上写着 `restricted: true`——`register` 根本没有这个选项，那是无声的空操作（`grep -n restricted` 列出的另外 7 处 `restricted: false` 也一样，全在 `register` 上，全是空操作）。Foundry 的实际规则是：`scope: "world"` 的设置本来就只有 GM 能写、只在 GM 的设置页显示；要限制**菜单**必须在 `registerMenu` 上写 `restricted: true`。本模组因此只有一个 `registerMenu({restricted: true})` 加两个 `config: false` 的 world 设置——玩家看不到、点不开、写不了。
4. **`onChange: () => location.reload()`。** 系统的 `evolved` 开关（`settings.mjs:14-25`，`onChange` 在 `:22-24`）一改就强制整页刷新。本模组不这么干：`features.mode(id)` 每次调用都现读设置，改档立刻生效，不需要重载。

还有一个只有本模组需要的东西系统完全没有：**存储形状的版本号**。系统直接把值裸存进 setting，将来想换形状就没有迁移锚点（它的迁移块 `module/alienrpg.mjs:311-326` 整段是注释掉的，`// const currentVersion = game.settings.get(...)` 开头那十几行）。本模组把三态表存在 `SETTING_FEATURES`，把形状版本存在 `SETTING_SCHEMA`，读的时候按版本号归一化。第一次真正的迁移就在这次交付里：schema 0 的历史值是裸 boolean，`true → "full"`、`false → "off"`。

迁移**只在内存里做，不在 `init` 期写盘**：`features.mjs` 是内核模块，只有它自己拥有的领域钩子才允许它挂，而它一个领域钩子都没有，所以它不能挂 `ready` 去补写盘；而在 `init` 里调 `game.settings.set` 时机不可靠。所以归一化发生在每次读取时，真正的落盘发生在 GM 从设置菜单点保存的那一刻（同时把 `SETTING_SCHEMA` 盖成当前版本）。

**两个设置而不是一个**：`SETTING_FEATURES` 存 `{[id]: "full"|"prompt"|"off"}` 这**一个对象**，`SETTING_SCHEMA` 存它的形状版本号。要害是「**不为每条特性各注册一个键**」——本任务在 Step 6 有一条用例把本模组命名空间下的 `register` 调用数钉死为 2，无论注册了多少条特性都不许涨。

**`registerSettings()` 不许快照 def 集合（验收条款）**

`init` 段的顺序是「先 `for (const f of FEATURES) f.register()`，再 `features.registerSettings()`」，但**本任务不得依赖这个顺序来保证正确性**：`registerSettings()` 只注册一个 `Object` 类型的设置加一个菜单按钮，**绝不能**在被调用的那一刻遍历 `features.all()` 去生成什么东西。菜单的行、每行的 `choices`、以及提交时要写回哪些 id，**全部在菜单被渲染／提交的那一刻**才从 `features.all()` 现读。Step 11 有一条用例专门守这一点：在 `registerSettings()` **之后**才注册的特性，必须照样出现在菜单上下文里。

**`prompt` 档一期只落到一半，本任务在 UI 层如实止血**

`mode(id)` 的返回值是三态，所以 `pureResolveMode` 与 `mode()` **必须**完整支持 `"prompt"`：存量世界里可能已经存着 `"prompt"`，读到就原样返回、写回时原样保存，**不做迁移**。但一期**不提供任何确认入口**：所有特性侧的检查都只调 `features.enabled(id)`，而 `enabled` 对 `"prompt"` 与 `"full"` 同为 `true`——GM 选「提示并确认」得到的行为与「全自动」逐字相同。真正让它有意义的确认卡属于二期。

因此本任务的处置是：**菜单一期只提供 `Full` 与 `Off` 两个选项**，绝不把一个空承诺摆在 GM 面前；同时，当某个特性的存量值**已经**是 `"prompt"`，该行才多出第三个选项并选中它，标签逐字写明「预留档，当前等同全自动」——这样保存表单永远不会把别人存进去的值悄悄改掉。两个分支各有一条单测。另外 `register()` 在收到 `default: "prompt"` 时打一条 `console.warn`，免得后续任务的作者以为它已经能用。

**关于本任务的可验证性**

Foundry 侧的三件事——菜单被正确注册、渲染上下文算得对、保存回写对——全部用 `installFoundryStub()` 写成了真测试。**唯一测不了的是浏览器把这个窗口画出来的样子**，那是需要人眼判断的视觉结果。本任务因此只给收尾的 MANUAL VERIFICATION，**不登记自检条目**：`selftest.register({id, label, run})` 的 `run()` 返回 `{ok, detail}`，它判断不了「这个窗口好不好看、译名有没有显示出来」；而 `kernel/selftest.mjs` 由另一个任务建立，本任务不能 `import` 一个还不存在的文件。

---

- [ ] **Step 1: 写会失败的 `pureResolveMode` 测试**

新建 `test/features.test.mjs`：

```js
import { describe, it, expect } from "vitest"
import { pureResolveMode } from "../scripts/kernel/features.mjs"

const defs = [
	{ id: "roll-record", default: "full", requires: [] },
	{ id: "success-line", default: "full", requires: ["roll-record"] },
	{ id: "push-correctness", default: "full", requires: ["success-line"] },
	{ id: "shy", default: "prompt", requires: [] },
]

describe("pureResolveMode", () => {
	it("returns the feature's own default when nothing is stored", () => {
		expect(pureResolveMode("roll-record", {}, defs)).toBe("full")
		expect(pureResolveMode("shy", {}, defs)).toBe("prompt")
	})

	it("lets a stored value win over the default", () => {
		expect(pureResolveMode("shy", { shy: "full" }, defs)).toBe("full")
		expect(pureResolveMode("roll-record", { "roll-record": "off" }, defs)).toBe("off")
	})

	it("falls back to the default when the stored value is not a legal mode", () => {
		expect(pureResolveMode("roll-record", { "roll-record": "banana" }, defs)).toBe("full")
		expect(pureResolveMode("roll-record", { "roll-record": true }, defs)).toBe("full")
	})

	it("turns a feature off when its direct dependency is off", () => {
		expect(pureResolveMode("success-line", { "roll-record": "off" }, defs)).toBe("off")
	})

	it("turns a feature off when a TRANSITIVE dependency is off", () => {
		// push-correctness -> success-line -> roll-record, and only the last one is off
		expect(pureResolveMode("push-correctness", { "roll-record": "off" }, defs)).toBe("off")
	})

	it("does not disable a dependent just because its dependency is on prompt", () => {
		expect(pureResolveMode("push-correctness", { "roll-record": "prompt" }, defs)).toBe("full")
	})

	it("returns off for an unregistered id — fail closed, never fail open", () => {
		expect(pureResolveMode("never-registered", { "never-registered": "full" }, defs)).toBe("off")
	})

	it("returns off for a missing dependency rather than pretending it is satisfied", () => {
		const broken = [{ id: "a", default: "full", requires: ["ghost"] }]
		expect(pureResolveMode("a", {}, broken)).toBe("off")
	})

	it("breaks a dependency cycle by failing closed", () => {
		const cyclic = [
			{ id: "x", default: "full", requires: ["y"] },
			{ id: "y", default: "full", requires: ["x"] },
		]
		expect(pureResolveMode("x", {}, cyclic)).toBe("off")
		expect(pureResolveMode("y", {}, cyclic)).toBe("off")
	})

	it("accepts defs as an array or as a Map", () => {
		const asMap = new Map(defs.map((d) => [d.id, d]))
		expect(pureResolveMode("success-line", { "roll-record": "off" }, asMap)).toBe("off")
	})

	it("touches no Foundry global", () => {
		expect(globalThis.game).toBeUndefined()
		expect(pureResolveMode("roll-record", {}, defs)).toBe("full")
	})
})
```

最后一条是分层铁律的可执行版本：`pure*` 函数不得引用任何 Foundry 全局，所以在**没装桩**的进程里它必须照常工作（这个 `describe` 声明在文件最前面，跑在任何 `installFoundryStub()` 之前）。它同时守住另一件事——这个文件顶部是**静态** import `features.mjs`，如果 `features.mjs` 在模块顶层碰了 `game` 或 `foundry`，整个测试文件在加载期就会炸。

- [ ] **Step 2: 跑它，看它红**

Run: `npx vitest run test/features.test.mjs`
Expected: FAIL — `Error: Failed to load url ../scripts/kernel/features.mjs (resolved id: .../scripts/kernel/features.mjs). Does the file exist?`，11 条用例一条都不执行。

- [ ] **Step 3: 写 `pureResolveMode`**

新建 `scripts/kernel/features.mjs`：

```js
import { MID, SETTING_FEATURES, SETTING_SCHEMA, I18N } from "../const.mjs"

/**
 * The three legal modes.
 *   full   — run the automation
 *   prompt — RESERVED. Phase 1 offers no confirmation entry point at all, and
 *            enabled() is true for both "prompt" and "full", so a phase-1
 *            feature does not distinguish the two: it just runs. An existing
 *            stored "prompt" is returned as-is and saved back as-is (no
 *            migration), and the settings menu only shows it on a row that
 *            already stores it. The confirm card that would give it meaning is
 *            phase 2; features.shouldRun() / features.confirm() are phase-2
 *            symbols and must not be invented here.
 *   off    — stay out of the way
 */
const MODES = ["full", "prompt", "off"]

/**
 * Shape version of the SETTING_FEATURES payload.
 *   0 — historical: a flat map of booleans, one per feature
 *   1 — current: a flat map of "full" | "prompt" | "off"
 */
const FEATURES_SCHEMA_VERSION = 1

/** id -> normalized def. Module-level, so tests must vi.resetModules() between cases. */
const REGISTERED = new Map()

/** Accept defs as an array (what features.all() returns), a Map, or a plain object. */
function indexDefs(defs) {
	if (defs instanceof Map) return defs
	const index = new Map()
	const list = Array.isArray(defs) ? defs : Object.values(defs ?? {})
	for (const def of list) {
		if (def && typeof def.id === "string") index.set(def.id, def)
	}
	return index
}

function resolveOne(id, modes, index, seen) {
	const def = index.get(id)
	if (!def) return "off" // unknown feature: fail closed
	if (seen.has(id)) return "off" // dependency cycle: fail closed

	seen.add(id)

	const stored = modes?.[id]
	const fallback = MODES.includes(def.default) ? def.default : "full"
	const own = MODES.includes(stored) ? stored : fallback
	if (own === "off") return "off"

	for (const required of def.requires ?? []) {
		// A fresh branch set per dependency: two features may legitimately share a
		// dependency without that counting as a cycle.
		if (resolveOne(required, modes, index, new Set(seen)) === "off") return "off"
	}
	return own
}

/**
 * Effective mode of one feature, after its whole `requires` chain is taken into
 * account. Off anywhere in the chain means off here — a half-running automation
 * is worse than none, because players assume it ran.
 *
 * Pure: no Foundry global is touched. `modes` is the stored {id: mode} map,
 * `defs` the registry contents.
 */
export function pureResolveMode(id, modes, defs) {
	return resolveOne(id, modes, indexDefs(defs), new Set())
}
```

- [ ] **Step 4: 跑它，看它绿**

Run: `npx vitest run test/features.test.mjs`
Expected: PASS — 11 passed。

- [ ] **Step 5: 提交**

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation"
git add -A && git commit -m "$(cat <<'EOF'
feat(kernel/features): pureResolveMode 依赖链求值

三态 full/prompt/off，依赖链上任何一环为 off 就整条禁掉，且是传递性的：
push-correctness -> success-line -> roll-record，只关最后一个，前两个也全灭。
半跑的自动化比不跑更危险，玩家会以为它跑了。

未注册 id、缺失依赖、依赖成环一律返回 off（fail closed，绝不 fail open）。
最后一条用例在未装 Foundry 桩的进程里跑，作为分层铁律的可执行断言：
文件顶部是静态 import，模块顶层一旦碰 game/foundry，整个测试文件加载期就炸。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: 追加会失败的注册表与设置测试（不含菜单）**

先把 `test/features.test.mjs` 顶部的 import 改成：

```js
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest"
import { pureResolveMode } from "../scripts/kernel/features.mjs"
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs"
```

再在文件末尾追加：

```js
/**
 * Redirect only the two i18n methods, keeping the shared stub's own object
 * identity. A private fake i18n object would make this file agree with nothing
 * but itself, and the shared stub is off limits to this task.
 */
function useI18n(table) {
	vi.spyOn(game.i18n, "has").mockImplementation((key) => Object.hasOwn(table, key))
	vi.spyOn(game.i18n, "localize").mockImplementation((key) => table[key] ?? key)
}

/** Fresh module instance per case: REGISTERED is module-level state. */
async function loadFeatures() {
	vi.resetModules()
	return (await import("../scripts/kernel/features.mjs")).features
}

describe("features registry (Foundry-facing half)", () => {
	let features

	beforeEach(async () => {
		installFoundryStub({ moduleId: "alien-evolved-automation", isGM: true })
		features = await loadFeatures()
	})

	afterEach(() => {
		vi.restoreAllMocks()
		uninstallFoundryStub()
	})

	it("normalizes a sparse def and hands it back", () => {
		const def = features.register({ id: "demo" })
		expect(def).toEqual({ id: "demo", default: "full", gmOnly: false, requires: [], hint: "" })
		expect(features.all()).toEqual([def])
	})

	it("exposes exactly the five contracted members", () => {
		// shouldRun / confirm are phase-2 symbols; installAll() does not exist
		// because main.mjs owns the FEATURES array loop.
		expect(Object.keys(features).sort()).toEqual(["all", "enabled", "mode", "register", "registerSettings"])
	})

	it("refuses a def with no id, a duplicate id, and an id containing a dot", () => {
		expect(() => features.register({})).toThrow(/string id/)
		features.register({ id: "demo" })
		expect(() => features.register({ id: "demo" })).toThrow(/duplicate feature id/)
		// the menu encodes each row as name="modes.<id>" and expandObject splits on
		// dots, so a dotted id would silently nest and be dropped on save
		expect(() => features.register({ id: "a.b" })).toThrow(/must not contain a dot/)
	})

	it("warns that a prompt default has no consumer in phase 1", () => {
		const spy = vi.spyOn(console, "warn").mockImplementation(() => {})
		features.register({ id: "demo", default: "prompt" })
		expect(spy).toHaveBeenCalledWith(expect.stringContaining("prompt"))
		expect(features.all()[0].default).toBe("prompt")
	})

	it("registers two hidden world settings and not one key per feature", () => {
		const register = vi.spyOn(game.settings, "register")
		features.register({ id: "one" })
		features.register({ id: "two" })
		features.register({ id: "three" })
		features.registerSettings()

		const ours = register.mock.calls.filter((call) => call[0] === "alien-evolved-automation")
		expect(ours.map((call) => call[1])).toEqual(["features", "settingsSchemaVersion"])

		const modes = ours[0][2]
		expect(modes.scope).toBe("world")
		expect(modes.config).toBe(false)
		expect(modes.default).toEqual({})

		const schema = ours[1][2]
		expect(schema.scope).toBe("world")
		expect(schema.config).toBe(false)
		expect(schema.default).toBe(0)
	})

	it("reads the stored mode and re-reads it after a change, with no reload", async () => {
		features.register({ id: "demo", default: "full" })
		features.registerSettings()
		expect(features.mode("demo")).toBe("full")
		expect(features.enabled("demo")).toBe(true)

		await game.settings.set("alien-evolved-automation", "features", { demo: "prompt" })
		expect(features.mode("demo")).toBe("prompt")
		expect(features.enabled("demo")).toBe(true) // phase 1: prompt behaves like full

		await game.settings.set("alien-evolved-automation", "features", { demo: "off" })
		expect(features.enabled("demo")).toBe(false)
	})

	it("returns off for an id nobody registered", () => {
		features.registerSettings()
		expect(features.mode("ghost")).toBe("off")
	})

	it("falls back to defaults when the settings are not registered yet", () => {
		const spy = vi.spyOn(console, "warn").mockImplementation(() => {})
		features.register({ id: "a", default: "full" })
		features.register({ id: "b", default: "off" })
		// no registerSettings() call: game.settings.get throws, defaults must survive
		expect(features.mode("a")).toBe("full")
		expect(features.mode("b")).toBe("off")
		expect(spy).toHaveBeenCalledTimes(1) // warned once, not once per read
	})

	it("migrates a schema-0 boolean payload to tri-state modes on read", async () => {
		features.register({ id: "demo", default: "full" })
		features.registerSettings()

		await game.settings.set("alien-evolved-automation", "settingsSchemaVersion", 0)
		await game.settings.set("alien-evolved-automation", "features", { demo: false })
		expect(features.mode("demo")).toBe("off")

		await game.settings.set("alien-evolved-automation", "features", { demo: true })
		expect(features.mode("demo")).toBe("full")
	})

	it("ignores a stray boolean once the payload is already at schema 1", async () => {
		features.register({ id: "demo", default: "off" })
		features.registerSettings()
		await game.settings.set("alien-evolved-automation", "settingsSchemaVersion", 1)
		await game.settings.set("alien-evolved-automation", "features", { demo: false })
		expect(features.mode("demo")).toBe("off") // from the def default, not from `false`
		await game.settings.set("alien-evolved-automation", "features", { demo: true })
		expect(features.mode("demo")).toBe("off")
	})
})

describe("features registry on a player client", () => {
	let features

	beforeEach(async () => {
		installFoundryStub({ moduleId: "alien-evolved-automation", isGM: false })
		features = await loadFeatures()
	})

	afterEach(() => {
		vi.restoreAllMocks()
		uninstallFoundryStub()
	})

	it("hides a gmOnly feature from a player client", () => {
		expect(game.user.isGM).toBe(false) // arrange precondition, driven by the stub option
		features.register({ id: "gm-thing", default: "full", gmOnly: true })
		features.register({ id: "shared-thing", default: "full" })
		features.registerSettings()
		expect(features.mode("gm-thing")).toBe("off")
		expect(features.enabled("gm-thing")).toBe(false)
		expect(features.enabled("shared-thing")).toBe(true)
	})
})
```

- [ ] **Step 7: 跑它，看它红**

Run: `npx vitest run test/features.test.mjs`
Expected: FAIL — 22 条里 11 passed（全是 `pureResolveMode` 那一组）、11 failed。11 条的报错全部是 `TypeError: Cannot read properties of undefined (reading 'register')` 之类：`features.mjs` 目前只导出 `pureResolveMode`，`(await import(...)).features` 是 `undefined`。

- [ ] **Step 8: 写 `features` 对象与两个设置注册（先不含菜单）**

在 `scripts/kernel/features.mjs` 末尾追加：

```js
/** Warn about missing settings once per session, not once per read. */
let warnedNoSettings = false

/**
 * In-memory migration. Nothing is written here: this kernel owns no domain hook,
 * so it cannot attach a ready hook to write the normalized shape back, and
 * writing settings during init is unreliable. The normalized shape is persisted
 * the moment a GM saves the menu.
 */
function normalizeModes(raw, schema) {
	const out = {}
	if (!raw || typeof raw !== "object") return out
	for (const [id, value] of Object.entries(raw)) {
		if (MODES.includes(value)) {
			out[id] = value // an existing "prompt" survives untouched
			continue
		}
		// schema 0 stored one boolean per feature: true meant on, false meant off.
		if (schema < FEATURES_SCHEMA_VERSION && typeof value === "boolean") {
			out[id] = value ? "full" : "off"
		}
	}
	return out
}

/** Read the stored mode map, normalized to the current schema. Never throws. */
function storedModes() {
	let raw = {}
	let schema = 0
	try {
		raw = game.settings.get(MID, SETTING_FEATURES) ?? {}
		schema = Number(game.settings.get(MID, SETTING_SCHEMA)) || 0
	} catch (err) {
		// Called before registerSettings(): game.settings.get throws for an
		// unregistered key. Defaults are the right answer, and one warning is
		// cheaper than a crash at init or a warning on every single read.
		if (!warnedNoSettings) {
			warnedNoSettings = true
			console.warn(`${MID} | feature settings unavailable, using per-feature defaults`, err)
		}
		return {}
	}
	return normalizeModes(raw, schema)
}

export const features = {
	/**
	 * @param {{id:string, default?:"full"|"prompt"|"off", gmOnly?:boolean, requires?:string[], hint?:string}} def
	 *   hint is an OPTIONAL override i18n key; the normal source of a feature's
	 *   hint is AEA.feature.<id>.hint.
	 * @returns {object} the normalized def
	 */
	register(def) {
		if (!def || typeof def.id !== "string" || def.id.length === 0) {
			throw new Error(`${MID} | features.register needs a non-empty string id`)
		}
		if (def.id.includes(".")) {
			// the settings menu names each row "modes.<id>" and foundry.utils.expandObject
			// splits on dots, so a dotted id would nest and then be dropped on save
			throw new Error(`${MID} | a feature id must not contain a dot: ${def.id}`)
		}
		if (REGISTERED.has(def.id)) {
			throw new Error(`${MID} | duplicate feature id: ${def.id}`)
		}
		if (def.default === "prompt") {
			console.warn(
				`${MID} | feature "${def.id}" defaults to prompt, but no phase-1 feature consumes prompt: it will behave like full`,
			)
		}
		const normalized = {
			id: def.id,
			default: MODES.includes(def.default) ? def.default : "full",
			gmOnly: def.gmOnly === true,
			requires: Array.isArray(def.requires) ? [...def.requires] : [],
			hint: typeof def.hint === "string" ? def.hint : "",
		}
		REGISTERED.set(normalized.id, normalized)
		return normalized
	},

	/**
	 * Called once from main.mjs during the init hook. Registers ONE object-valued
	 * world setting plus its shape version — never one key per feature — and
	 * deliberately does NOT read the def set here: the menu rows, their choices
	 * and the save handler all read features.all() lazily, at render/submit time,
	 * so a feature registered after this call still shows up.
	 */
	registerSettings() {
		game.settings.register(MID, SETTING_FEATURES, {
			name: `${I18N}.settings.featuresName`,
			hint: `${I18N}.settings.featuresHint`,
			scope: "world",
			config: false,
			type: Object,
			default: {},
		})
		game.settings.register(MID, SETTING_SCHEMA, {
			name: `${I18N}.settings.schemaName`,
			scope: "world",
			config: false,
			type: Number,
			default: 0,
		})
	},

	/** @returns {"full"|"prompt"|"off"} */
	mode(id) {
		const def = REGISTERED.get(id)
		if (!def) return "off"
		// world-scope settings are already GM-write-only; this hides GM-only
		// automation from a player client that would otherwise try to run it.
		if (def.gmOnly && !game.user?.isGM) return "off"
		return pureResolveMode(id, storedModes(), REGISTERED)
	},

	/** Phase 1: "prompt" counts as enabled — there is no confirmation entry point yet. */
	enabled(id) {
		return features.mode(id) !== "off"
	},

	all() {
		return [...REGISTERED.values()]
	},
}
```

- [ ] **Step 9: 跑它，看它绿**

Run: `npx vitest run test/features.test.mjs`
Expected: PASS — 22 passed。

- [ ] **Step 10: 提交**

```bash
git add -A && git commit -m "$(cat <<'EOF'
feat(kernel/features): 特性注册表、三态读取与 schema 0 迁移

一个 Object 型 world 设置存 {id: full|prompt|off}，外加一个形状版本号，而不是
N 个平铺 boolean（系统 module/helpers/settings.mjs 是 217 行里 20 次 register、
其中 11 次 type: Boolean）。用例把本命名空间下的 register 调用数钉死为 2，注册
再多特性也不许涨。

register 拒收空 id、重复 id 与含点的 id：菜单把每行编码成 name="modes.<id>"，
foundry.utils.expandObject 按点拆层，含点的 id 会静默嵌套然后在保存时被丢掉。

mode() 每次现读设置，改档即时生效，不像系统 evolved 开关那样 location.reload()。
gmOnly 在玩家端直接返回 off。设置未注册时 game.settings.get 会抛，这里兜住并只
警告一次，返回各自的 default。

schema 0 的裸 boolean 在读取时归一化为三态（true -> full, false -> off），迁移
只在内存里做：本内核不拥有任何领域钩子，不能挂 ready 补写盘（系统那份
module/alienrpg.mjs:311-326 的迁移块整段是注释掉的，没有可抄的锚点），落盘发生
在 GM 点保存时。存量的 "prompt" 原样保留，不做迁移。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 11: 追加会失败的设置菜单测试**

在 `test/features.test.mjs` 末尾追加：

```js
describe("feature settings menu", () => {
	let features
	let registerMenu

	beforeEach(async () => {
		installFoundryStub({ moduleId: "alien-evolved-automation", isGM: true })
		features = await loadFeatures()
		registerMenu = vi.spyOn(game.settings, "registerMenu")
	})

	afterEach(() => {
		vi.restoreAllMocks()
		uninstallFoundryStub()
	})

	/**
	 * The menu class as it was handed to Foundry. _prepareContext is called off
	 * the prototype rather than on a constructed window: constructing a real
	 * ApplicationV2 is a browser concern and is covered by MANUAL VERIFICATION.
	 */
	function menuClass() {
		const ours = registerMenu.mock.calls.filter((call) => call[0] === "alien-evolved-automation")
		expect(ours).toHaveLength(1)
		expect(ours[0][1]).toBe("featuresMenu")
		return ours[0][2]
	}

	function prepareContext(MenuClass) {
		return MenuClass.prototype._prepareContext.call({})
	}

	it("registers exactly one GM-restricted menu whose type is a class", () => {
		features.registerSettings()
		const menu = menuClass()
		expect(menu.restricted).toBe(true) // the ONE place that actually gates it to GMs
		expect(typeof menu.type).toBe("function")
		expect(typeof menu.type.DEFAULT_OPTIONS.form.handler).toBe("function")
	})

	it("builds a menu context from the contracted AEA.feature.<id>.name / .hint keys", async () => {
		useI18n({
			"AEA.feature.child-feature.name": "Child feature",
			"AEA.feature.child-feature.hint": "Only runs while its parent runs.",
			"AEA.mode.full": "Full",
			"AEA.mode.off": "Off",
		})
		features.register({ id: "parent", default: "full" })
		features.register({ id: "child-feature", default: "full", requires: ["parent"] })
		features.registerSettings()
		await game.settings.set("alien-evolved-automation", "features", { parent: "off" })

		const context = await prepareContext(menuClass().type)
		const child = context.features.find((f) => f.id === "child-feature")
		const parent = context.features.find((f) => f.id === "parent")

		expect(child.label).toBe("Child feature")
		expect(child.hint).toBe("Only runs while its parent runs.")
		expect(child.requires).toBe("parent")
		expect(child.blocked).toBe(true) // user asked for full, parent vetoed it
		expect(child.choices.find((c) => c.value === "full").selected).toBe(true)

		// no AEA.feature.parent.name key exists, so the raw id shows through
		expect(parent.label).toBe("parent")
		expect(parent.hint).toBe("")
		expect(parent.blocked).toBe(false)
		expect(context.buttons).toEqual([
			{ type: "submit", icon: "fa-solid fa-floppy-disk", label: "AEA.settings.save" },
		])
	})

	it("lists a feature registered AFTER registerSettings — nothing is snapshotted", async () => {
		features.registerSettings()
		features.register({ id: "late-comer", default: "full" })

		const context = await prepareContext(menuClass().type)
		expect(context.features.map((f) => f.id)).toEqual(["late-comer"])
		expect(context.features[0].choices.map((c) => c.value)).toEqual(["full", "off"])
	})

	it("offers Full and Off only, never prompt as a new choice", async () => {
		features.register({ id: "demo", default: "full" })
		features.registerSettings()

		const context = await prepareContext(menuClass().type)
		expect(context.features[0].choices.map((c) => c.value)).toEqual(["full", "off"])
	})

	it("keeps an already-stored prompt selectable so saving cannot rewrite it", async () => {
		features.register({ id: "demo", default: "full" })
		features.registerSettings()
		await game.settings.set("alien-evolved-automation", "features", { demo: "prompt" })

		const MenuClass = menuClass().type
		const row = (await prepareContext(MenuClass)).features[0]
		expect(row.choices.map((c) => c.value)).toEqual(["full", "prompt", "off"])
		expect(row.choices.find((c) => c.value === "prompt").selected).toBe(true)

		// re-submitting the rendered form must round-trip the value untouched
		await MenuClass.DEFAULT_OPTIONS.form.handler({}, {}, { object: { "modes.demo": "prompt" } })
		expect(game.settings.get("alien-evolved-automation", "features")).toEqual({ demo: "prompt" })
	})

	it("writes tri-state modes and stamps the schema version when the GM saves", async () => {
		features.register({ id: "demo", default: "full" })
		features.register({ id: "other", default: "full" })
		features.registerSettings()
		await game.settings.set("alien-evolved-automation", "settingsSchemaVersion", 0)

		const handler = menuClass().type.DEFAULT_OPTIONS.form.handler
		await handler({}, {}, { object: { "modes.demo": "off", "modes.other": "prompt", "modes.ghost": "full" } })

		// the unregistered "ghost" key is dropped rather than persisted
		expect(game.settings.get("alien-evolved-automation", "features")).toEqual({ demo: "off", other: "prompt" })
		expect(game.settings.get("alien-evolved-automation", "settingsSchemaVersion")).toBe(1)
		expect(features.mode("demo")).toBe("off")
		expect(features.mode("other")).toBe("prompt")
	})
})
```

「不快照」那一条是本任务的验收条款：`init` 段的既定顺序是先 `f.register()` 再 `features.registerSettings()`，但正确性不许依赖这个顺序。倒数第二条与最后一条是「一期不摆空承诺、但也不改写存量值」这条决定的可执行形式。而 `builds a menu context ...` 的意义不只是「取得到译文」：它把 i18n 键的**大小写与分段**钉死——Foundry 的 `game.i18n.has` 大小写敏感，`AEA.Feature.<id>` 与 `AEA.feature.<id>.name` 是两棵完全不同的树，写错的后果是设置面板里每一行都显示裸的 kebab-case id 而没有任何报错。

- [ ] **Step 12: 跑它，看它红**

Run: `npx vitest run test/features.test.mjs`
Expected: FAIL — 28 条里 22 passed、6 failed，全部集中在菜单上，报错都来自 `menuClass()` 里的 `expect(ours).toHaveLength(1)`：`AssertionError: expected [] to have a length of 1 but got +0` —— `registerSettings()` 还没有调 `game.settings.registerMenu`。

- [ ] **Step 13: 写设置菜单类（ApplicationV2）**

在 `scripts/kernel/features.mjs` 的 `registerSettings()` 里、**两个 `game.settings.register` 之前**插入：

```js
		game.settings.registerMenu(MID, "featuresMenu", {
			name: `${I18N}.settings.menuName`,
			label: `${I18N}.settings.menuLabel`,
			hint: `${I18N}.settings.menuHint`,
			icon: "fa-solid fa-sliders",
			type: buildMenuClass(),
			restricted: true, // the ONE place that actually gates the menu to GMs
		})
```

并在文件末尾追加：

```js
/**
 * A feature's display name and hint. The keys are fixed by contract as
 * AEA.feature.<id>.name / AEA.feature.<id>.hint, where <id> is verbatim the id
 * passed to features.register — kebab case and all. Foundry's i18n lookup is
 * case sensitive, so this is the one spelling that resolves.
 */
function labelOf(def) {
	const key = `${I18N}.feature.${def.id}.name`
	return game.i18n.has(key) ? game.i18n.localize(key) : def.id
}

function hintOf(def) {
	const key = `${I18N}.feature.${def.id}.hint`
	if (game.i18n.has(key)) return game.i18n.localize(key)
	if (def.hint && game.i18n.has(def.hint)) return game.i18n.localize(def.hint)
	return ""
}

/**
 * Phase 1 offers Full and Off only. "prompt" is a legal mode that pureResolveMode
 * and mode() understand, but nothing consumes it yet — the confirm card that
 * would give it meaning is phase 2 — so the menu must not offer it as a NEW
 * choice: a GM who picked it would get full automation, silently. It is still
 * rendered, clearly labelled, when it is ALREADY the stored value, so that
 * submitting the form can never rewrite somebody's stored setting behind them.
 */
function choicesFor(own) {
	return MODES.filter((mode) => mode !== "prompt" || own === "prompt").map((mode) => ({
		value: mode,
		label: game.i18n.localize(`${I18N}.mode.${mode}`),
		selected: own === mode,
	}))
}

/**
 * Persist the submitted modes. ApplicationV2 calls a form handler as
 * (event, form, formData) where formData.object is a FLAT map keyed by the input
 * `name` attributes — "modes.<feature-id>" — hence expandObject. The def set is
 * read here, at submit time, never snapshotted at registerSettings() time.
 */
async function onSubmitFeatureMenu(_event, _form, formData) {
	const submitted = foundry.utils.expandObject(formData?.object ?? {})
	const next = {}
	for (const def of features.all()) {
		const value = submitted.modes?.[def.id]
		if (MODES.includes(value)) next[def.id] = value
	}
	await game.settings.set(MID, SETTING_FEATURES, next)
	// Saving is the only moment the stored shape is guaranteed current.
	await game.settings.set(MID, SETTING_SCHEMA, FEATURES_SCHEMA_VERSION)
}

/**
 * Built lazily rather than at module scope: `foundry.applications.api` only
 * exists once Foundry's client bundle has run, and this file is imported by
 * vitest where the module top level must not touch any Foundry global at all.
 *
 * ApplicationV2 is Foundry V13+'s window base class; HandlebarsApplicationMixin
 * adds template rendering, where PARTS names each template fragment.
 */
function buildMenuClass() {
	const { ApplicationV2, HandlebarsApplicationMixin } = foundry.applications.api

	return class FeatureSettingsMenu extends HandlebarsApplicationMixin(ApplicationV2) {
		static DEFAULT_OPTIONS = {
			id: "aea-feature-settings",
			tag: "form",
			classes: ["aea", "aea-feature-settings"],
			window: {
				title: `${I18N}.settings.menuTitle`,
				icon: "fa-solid fa-sliders",
				contentClasses: ["standard-form"],
			},
			position: { width: 620, height: "auto" },
			form: { handler: onSubmitFeatureMenu, closeOnSubmit: true, submitOnChange: false },
		}

		static PARTS = {
			body: { template: `modules/${MID}/templates/feature-settings.hbs`, scrollable: [""] },
			footer: { template: "templates/generic/form-footer.hbs" },
		}

		/** Reads features.all() on every render: the row set is never cached. */
		async _prepareContext() {
			const modes = storedModes()
			return {
				features: features.all().map((def) => {
					const own = MODES.includes(modes[def.id]) ? modes[def.id] : def.default
					const effective = pureResolveMode(def.id, modes, REGISTERED)
					return {
						id: def.id,
						label: labelOf(def),
						hint: hintOf(def),
						gmOnly: def.gmOnly,
						requires: def.requires.join(", "),
						// true when the user asked for it but a dependency vetoed it
						blocked: effective === "off" && own !== "off",
						choices: choicesFor(own),
					}
				}),
				buttons: [{ type: "submit", icon: "fa-solid fa-floppy-disk", label: `${I18N}.settings.save` }],
			}
		}
	}
}
```

- [ ] **Step 14: 写模板、语言键与样式**

新建 `templates/feature-settings.hbs`：

```hbs
{{!-- templates/feature-settings.hbs --}}
<p class="hint">{{localize "AEA.settings.menuHint"}}</p>
<div class="aea-feature-list">
	{{#each features as |feature|}}
	<div class="form-group aea-feature-row" data-feature-id="{{feature.id}}">
		<label>
			{{feature.label}}
			{{#if feature.gmOnly}}<i class="fa-solid fa-user-shield" data-tooltip="{{localize "AEA.settings.gmOnly"}}"></i>{{/if}}
		</label>
		<div class="form-fields">
			<select name="modes.{{feature.id}}">
				{{#each feature.choices as |choice|}}
				<option value="{{choice.value}}" {{#if choice.selected}}selected{{/if}}>{{choice.label}}</option>
				{{/each}}
			</select>
		</div>
		{{#if feature.hint}}<p class="hint">{{feature.hint}}</p>{{/if}}
		{{#if feature.requires}}
		<p class="hint aea-requires">{{localize "AEA.settings.requires"}} {{feature.requires}}</p>
		{{/if}}
		{{#if feature.blocked}}
		<p class="aea-blocked">{{localize "AEA.settings.disabledByDependency"}}</p>
		{{/if}}
	</div>
	{{/each}}
</div>
```

`lang/en.json`：**保留文件里已有的键**，在顶层 `AEA` 对象内追加下面两个子对象。二级段用小写（`settings`、`mode`），与契约钉死的 `AEA.feature.<id>.name` 保持同一种拼写习惯：

```json
		"settings": {
			"menuName": "Automation Features",
			"menuLabel": "Configure Features",
			"menuHint": "Set each automation to run on its own, or to stay out of the way.",
			"menuTitle": "Alien Evolved: Automation — Features",
			"featuresName": "Feature modes",
			"featuresHint": "Per-feature full / prompt / off map. Edited through the Configure Features menu.",
			"schemaName": "Settings schema version",
			"save": "Save Features",
			"gmOnly": "Runs on the Game Master's client only.",
			"requires": "Requires:",
			"disabledByDependency": "Held off: something this feature depends on is switched off."
		},
		"mode": {
			"full": "Full — run it",
			"prompt": "Prompt — reserved, currently behaves like Full",
			"off": "Off — leave it alone"
		}
```

`lang/cn.json`：同样只追加，键必须与英文**一字不差**（两份语言包压平后的键集必须完全相同）：

```json
		"settings": {
			"menuName": "自动化特性",
			"menuLabel": "配置特性",
			"menuHint": "为每一项自动化选择：自己跑，或者别管。",
			"menuTitle": "异形进化版：自动化 — 特性",
			"featuresName": "特性档位",
			"featuresHint": "逐特性的 全自动／提示／关闭 映射，通过「配置特性」菜单编辑。",
			"schemaName": "设置结构版本",
			"save": "保存特性设置",
			"gmOnly": "仅在 GM 客户端运行。",
			"requires": "依赖：",
			"disabledByDependency": "已挂起：该特性依赖的某一项被关掉了。"
		},
		"mode": {
			"full": "全自动 — 直接执行",
			"prompt": "提示 — 预留档，当前等同全自动",
			"off": "关闭 — 不要插手"
		}
```

`styles/alien-evolved-automation.css` 末尾追加：

```css
.aea-feature-settings .aea-feature-row {
	border-bottom: 1px solid var(--color-border-light-tertiary, rgba(0, 0, 0, 0.15));
	padding: 0.4rem 0;
}

.aea-feature-settings .aea-feature-row:last-child {
	border-bottom: none;
}

.aea-feature-settings .aea-requires {
	font-style: italic;
	opacity: 0.75;
}

.aea-feature-settings .aea-blocked {
	color: var(--color-level-warning, #c9a227);
	font-size: var(--font-size-12, 0.75rem);
	margin: 0.15rem 0 0;
}
```

- [ ] **Step 15: 跑它，看它全绿**

Run: `npx vitest run test/features.test.mjs`
Expected: PASS — 28 passed。（若 Step 13 的 `labelOf`/`hintOf` 漏写，`builds a menu context ...` 会报 `AssertionError: expected 'child-feature' to be 'Child feature'`。）

- [ ] **Step 16: 提交**

```bash
git add -A && git commit -m "$(cat <<'EOF'
feat(kernel/features): 单一 GM 限定设置菜单，Full/Off 两档

一个 registerMenu(restricted: true) 加两个 config:false 的 world 设置。系统那份
module/helpers/settings.mjs 的 :6-13 把 registerMenu 写成 restricted: false，任何
玩家都能打开配置面板；而 :21/:33 两处 game.settings.register 上写的 restricted
根本不是有效选项，是无声空操作（同文件另有 7 处同样的空操作）。

特性名与说明固定走 AEA.feature.<id>.name / .hint，<id> 逐字等于 register 的 id。
Foundry 的 i18n 查找大小写敏感，写成 AEA.Feature.<id> 会静默退化成显示裸 id 而
不报错，所以专门有一条用例把大小写与分段钉死。

prompt 档一期没有任何消费者（确认卡在二期），因此菜单只提供 Full/Off 两项，不把
空承诺摆给 GM；只有当存量值已经是 prompt 时那一行才多出第三项并原样保存，保证
提交表单不会改写别人存进去的值。

菜单行、每行 choices、以及提交时写回哪些 id，全部在渲染/提交那一刻现读
features.all()，registerSettings() 不快照 def 集合：有一条用例专门在
registerSettings() 之后才注册特性，验证它照样出现在菜单上下文里。

菜单类在 registerSettings() 内惰性构建：foundry.applications.api 要等客户端
bundle 跑完才存在，放模块顶层会让 vitest 在 import 期就炸。测试用
MenuClass.prototype._prepareContext.call({}) 直接调方法本体，不构造 ApplicationV2
实例——窗口真能画出来属于人工验收。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 17: 写会失败的 main.mjs 接线测试**

`main.mjs` 由前一个任务建立，是**唯一**允许挂生命周期钩子的地方。它交付时长这样（逐字，锚点是定位依据）：

```js
import { MID } from "./const.mjs";
/* AEA-ANCHOR: imports */

export const api = {
  features: null, patches: null, resolver: null, registry: null,
  rollBus: null, diceBarrier: null, cards: null, selftest: null,
};

const FEATURES = [
  /* AEA-ANCHOR: features */
];
const REPAIRS = [
  /* AEA-ANCHOR: repairs */
];

Hooks.once("init", () => {
  for (const f of FEATURES) safely(`feature ${f.id} register`, () => f.register());
  for (const r of REPAIRS)  safely(`repair ${r.id} register`,  () => r.register());
  /* AEA-ANCHOR: init */
  publishApi();
});
```

本任务往三个装配点各插一处：import 锚点后一行 import、`api` 里把 `features: null,` 换成 `features,`、`init` 锚点后一行 `features.registerSettings();`。注意 `api` 是**模块级 const 字面量**，本任务改的是字面量本身而不是运行期赋值，`publishApi()` 只是把这同一个对象挂到模组上，所以两者之间没有先后关系。而 `init` 锚点在两个 `register()` 循环**之后**，这正是契约要的顺序：特性 def 先进册，再注册设置。

新建 `test/features-wiring.test.mjs`：

```js
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest"
import { readFile } from "node:fs/promises"
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs"

const MID = "alien-evolved-automation"

describe("main.mjs wiring for the feature kernel", () => {
	beforeEach(() => {
		vi.resetModules() // main.mjs calls Hooks.once at import time
		installFoundryStub({ moduleId: MID, isGM: true })
	})

	afterEach(() => {
		vi.restoreAllMocks()
		uninstallFoundryStub()
	})

	it("registers the feature settings inside the init hook, not at import time", async () => {
		const register = vi.spyOn(game.settings, "register")
		const registerMenu = vi.spyOn(game.settings, "registerMenu")
		await import("../scripts/main.mjs")
		expect(registerMenu).not.toHaveBeenCalled()

		Hooks.callAll("init")

		expect(registerMenu.mock.calls.filter((c) => c[0] === MID).map((c) => c[1])).toEqual(["featuresMenu"])
		const keys = register.mock.calls.filter((c) => c[0] === MID).map((c) => c[1])
		expect(keys).toContain("features")
		expect(keys).toContain("settingsSchemaVersion")
	})

	it("fills the features slot of the api without disturbing the other seven", async () => {
		expect(game.modules.get(MID)).toBeTruthy() // arrange precondition for publishApi()
		const mod = await import("../scripts/main.mjs")
		Hooks.callAll("init")

		expect(Object.keys(mod.api)).toEqual([
			"features",
			"patches",
			"resolver",
			"registry",
			"rollBus",
			"diceBarrier",
			"cards",
			"selftest",
		])
		expect(typeof mod.api.features?.enabled).toBe("function")
		expect(mod.api.features.mode("nobody-registered-this")).toBe("off")
		// published by reference, so the console path sees the same object
		expect(game.modules.get(MID).api.features).toBe(mod.api.features)
	})

	it("keeps features.registerSettings() after the FEATURES register loop", async () => {
		const source = await readFile(new URL("../scripts/main.mjs", import.meta.url), "utf8")
		expect(source).toContain("/* AEA-ANCHOR: init */")
		expect(source).toContain("features.registerSettings();")
		// defs must be in the registry before any settings/menu work happens
		expect(source.indexOf("for (const f of FEATURES)")).toBeLessThan(source.indexOf("features.registerSettings();"))
	})
})
```

第三条守的是装配顺序：`main.mjs` 里 `for (const f of FEATURES)` 第一次出现的位置就是 `init` 里的 register 循环（第二次出现在 `ready` 里的 install 循环），`features.registerSettings();` 必须排在它之后。这条断言的是文件本身，不需要跑起 Foundry。

- [ ] **Step 18: 跑它，看它红**

Run: `npx vitest run test/features-wiring.test.mjs`
Expected: FAIL — 3 failed：

- `registers the feature settings inside the init hook` → `AssertionError: expected [] to deeply equal [ 'featuresMenu' ]`
- `fills the features slot of the api ...` → `AssertionError: expected 'undefined' to be 'function'`（`api.features` 还是 `null`）
- `keeps features.registerSettings() after the FEATURES register loop` → `AssertionError: expected '…' to contain 'features.registerSettings();'`

- [ ] **Step 19: 改 `main.mjs`，三处按锚点插入**

全部按**原文文本**定位，不要按行号；**不得**新增或删除 `api` 的键，**不得**替换 `api` 这个对象本身，**不得**动其余七个 `null`，**不得**往四个 `ready` 子锚点（`ready.registry` / `ready.patches` / `ready.rollbus` / `ready.cards`）插任何东西——本内核在 `ready` 阶段没有调用。

1. **import**：在逐字的 `/* AEA-ANCHOR: imports */` 的**下一行**插入：
   ```js
   import { features } from "./kernel/features.mjs";
   ```
2. **api 槽位**：在 `export const api = {` 的字面量里，把 `features: null,` 这一处文本**替换**为：
   ```js
   features,
   ```
   替换后 `api` 仍是八个键、顺序不变，只有 `features` 这一个从 `null` 变成本任务导出的对象。
3. **init 段**：在逐字的 `/* AEA-ANCHOR: init */` 的**下一行**（同一个 `Hooks.once("init", …)` 函数体内、两个 `register()` 循环之后、`publishApi()` 之前）插入：
   ```js
   features.registerSettings();
   ```

- [ ] **Step 20: 跑接线测试，看它绿**

Run: `npx vitest run test/features-wiring.test.mjs`
Expected: PASS — 3 passed。

- [ ] **Step 21: 跑全套**

Run: `npm test`
Expected: PASS — 全部测试文件通过，其中 `test/features.test.mjs` 28 passed、`test/features-wiring.test.mjs` 3 passed。其余文件的条数由先前的任务决定，本任务不对它们做数字断言；只要出现 failed 就停下，先看是不是两份语言包键集不一致（追加 `settings`/`mode` 两个子对象时必须两份同步），再看是不是误动了 `api` 的其它槽位。

- [ ] **Step 22: 提交**

```bash
git add -A && git commit -m "$(cat <<'EOF'
feat(main): init 段接入 features.registerSettings()，api.features 落位

三处按锚点文本插入 main.mjs，不按行号：/* AEA-ANCHOR: imports */ 之后一行
import、api 字面量里把 features: null 换成 features、/* AEA-ANCHOR: init */
之后一行 features.registerSettings()。api 的八个键不增不删，其余七个 null 不动，
四个 ready 子锚点一个都不占——本内核在 ready 阶段没有调用。

init 锚点在两个 register 循环之后，正是契约要的顺序：特性 def 先进册，再注册
设置。接线测试另有一条直接断言 main.mjs 源文件里
"for (const f of FEATURES)" 出现在 "features.registerSettings();" 之前，把这个
装配顺序钉在文件上而不是靠任务执行次序。

api 是模块级 const 字面量，本次改的是字面量本身；测试顺带断言
game.modules.get(MID).api.features 与导出的 api.features 是同一个对象引用。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

- [ ] **MANUAL VERIFICATION（Task 2 收尾，必做）**

单测已经覆盖了菜单注册、渲染上下文、档位选项与保存回写；证明不了的只有一件事——它在浏览器里画出来的样子。模组目录是 `Data/modules/alien-evolved-automation/`，世界要装着 `alienrpg` 系统并启用本模组。**下面第 1 步的临时代码与临时译名一律用 `aea-demo-*` 前缀的 id**，绝不与后续任务要注册的真实特性 id 撞车；第 8 步逐字还原。

1. 临时在 `scripts/main.mjs` 的 `/* AEA-ANCHOR: init */` 之后、`features.registerSettings();` **之前**插入三行：
   ```js
   features.register({ id: "aea-demo-a", default: "full" });
   features.register({ id: "aea-demo-b", default: "full", requires: ["aea-demo-a"] });
   features.register({ id: "aea-demo-c", default: "full", gmOnly: true });
   ```
   同时在 `lang/en.json` 的 `AEA` 下（若还没有 `feature` 这个子对象就新建）加：
   ```json
   "feature": {
   	"aea-demo-a": { "name": "Demo A", "hint": "Temporary, for verification only." },
   	"aea-demo-b": { "name": "Demo B", "hint": "Only runs while Demo A runs." },
   	"aea-demo-c": { "name": "Demo C (GM only)", "hint": "Temporary, for verification only." }
   }
   ```
   并在 `lang/cn.json` 的同一位置加**同样的键**（值写中文），否则两份语言包键集不一致。
2. 重载世界（F5）。以 **GM** 身份进入 **Game Settings → Configure Settings → Alien Evolved: Automation**。预期看到**一个**按钮「Configure Features」（中文界面为「配置特性」），而不是三个复选框、也不是三个下拉框直接摊在面板上。
3. 点开它。预期：一个窗口、三行，每行一个下拉框，选项**恰好两个**：`Full — run it` / `Off — leave it alone`（中文「全自动 — 直接执行」／「关闭 — 不要插手」）；**不应该**出现 `Prompt`。每行标题是第 1 步写的译名（**不是** `aea-demo-a` 这样的裸 id——若看到裸 id，说明语言包里的键写成了 `AEA.Feature.*` 或漏了 `.name` 那一层）；`aea-demo-c` 那行标题后带一个盾牌图标；`aea-demo-b` 那行下方有斜体的 `Requires: aea-demo-a`。
4. 把 `aea-demo-a` 改成 `Off`，点 **Save Features**。重新打开菜单，预期 `aea-demo-b` 那行下方出现黄色一行「Held off: something this feature depends on is switched off.」，而它自己的下拉框仍停在 `Full`（用户的选择没被改写，只是被依赖否决了）。
5. 控制台执行 `game.modules.get("alien-evolved-automation").api.features.mode("aea-demo-b")`，预期返回 `"off"`；执行 `game.settings.get("alien-evolved-automation", "settingsSchemaVersion")`，预期返回 `1`（保存那一刻盖上了当前形状版本）。
6. 验证 prompt 存量值不被改写：控制台执行
   ```js
   await game.settings.set("alien-evolved-automation", "features", { "aea-demo-c": "prompt" });
   ```
   重新打开菜单，预期 `aea-demo-c` 那一行——**只有这一行**——出现**三个**选项，并选中「Prompt — reserved, currently behaves like Full」。什么都不改，直接点 **Save Features**，再执行
   ```js
   game.settings.get("alien-evolved-automation", "features")["aea-demo-c"];
   ```
   预期仍是 `"prompt"`（保存没有把它悄悄改成 full）。
7. 用一个**玩家**账号登录同一世界，打开 Game Settings，预期**看不到**「Configure Features」按钮；控制台执行 `game.modules.get("alien-evolved-automation").api.features.mode("aea-demo-c")` 预期返回 `"off"`。
8. **清理（不可跳过）**：先在 GM 客户端控制台执行
   ```js
   await game.settings.set("alien-evolved-automation", "features", {});
   ```
   清掉演示值，再在仓库里执行
   ```bash
   cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation"
   git checkout -- scripts/main.mjs lang/en.json lang/cn.json
   git status --short
   ```
   预期 `git status --short` 对这三个文件没有任何输出——第 1 步插入的三行注册代码与两份语言包里的 `AEA.feature.aea-demo-*` 三组键全部还原（`git checkout --` 是把它们退回上一次提交的状态，Step 22 已经提交过正式内容，所以退回后剩下的正是正式内容）。最后执行 `npm test`，预期仍全绿（`test/features.test.mjs` 28 passed、`test/features-wiring.test.mjs` 3 passed）。真正的特性注册属于后续任务。
