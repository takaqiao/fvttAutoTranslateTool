> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 1 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 1: 模组脚手架与 vitest 工具链

**Files:**
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/package.json`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/vitest.config.mjs`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/.gitignore`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/module.json`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/lang/en.json`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/lang/cn.json`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/styles/alien-evolved-automation.css`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/const.mjs`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/main.mjs`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/test/stubs/foundry.mjs`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/test/lib/discipline-scan.mjs`
- Test: `test/const.test.mjs`
- Test: `test/manifest.test.mjs`
- Test: `test/stub-fidelity.test.mjs`
- Test: `test/stub-world.test.mjs`
- Test: `test/main-lifecycle.test.mjs`
- Test: `test/main-boot-gate.test.mjs`
- Test: `test/discipline.test.mjs`

**Interfaces:**

- **Consumes（本轮全部重新打开核对过，可直接引用）：**
  - `C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg/system.json` → `id: "alienrpg"`、`title: "Alien Evolved"`、`version: "4.1.13"`；语言表里中文那条写的是 `"lang": "cn"`（`system.json:164`），不是 `zh-CN`。
  - 本机已装依赖的清单实测值（直接读四份 `module.json`）：`lib-wrapper` 版本 `1.13.5.1`、`compatibility {minimum:"0.6.5", verified:"14"}`；`socketlib` 版本 `v1.1.4`、`{minimum:"11", verified:"14"}`；`dice-so-nice` 版本 `6.2.9`、`{minimum:14, verified:"14.365"}`；`yze-combat` 版本 `1.7.1`、`{minimum:14, verified:14, maximum:14}`。后两者把 Foundry 主版本下限钉死在 14，所以它们只能是软依赖，否则 V13 用户根本装不上本模组。
  - 系统的两个世界设置：`alienrpg.ARPGSemaphore` 注册于 `module/helpers/settings.mjs:209`（`scope:"world"`、`type: String`、`config:false`、`default:""`），`module/apps/migratefolders.js:12` 把它置为 `"busy"`、`:118` 清回 `""`；`alienrpg.imported` 注册于 `module/apps/init.mjs:24`（`scope:"world"`、`type: Boolean`、`default:false`），`module/apps/init.mjs:78` 在导入完成后置 `true`。
  - `module/apps/init.mjs:48` 的 `Hooks.on("ready", () => {...})` 在 `:50` 调用 `async` 的 `FirstTimeSetup()` 却**不 `await`**（`:74` 是它的定义）。全新世界里，本模组要绑定的那些 RollTable 在任何 ready 时刻都还不存在。
  - 真实 Foundry V13/V14 里 `preCreateChatMessage` 的第一个参数是**已用 data 构造完毕、尚未落库**的文档实例，唯一有效的改写方式是 `document.updateSource({...})`：本机两个已装模组就是这么写的——`modules/dice-chronicle/scripts/tracker.js:13` 的 `msg.updateSource({ 'flags.dice-chronicle.targets': targets })`、`modules/pf2e-target-helper/src/main.js:97` 的 `message.updateSource({ [\`flags.${moduleName}.saves.roll\`]: saves, ... })`。本任务的桩按这个语义实现。
  - lib-wrapper 1.13.5.1 的重复注册错误原文（从 `modules/lib-wrapper/lib-wrapper.js` 里取出）：``A wrapper for '${f}' (ID=${h}) has already been registered by ${c.type_plus_id}.``，触发条件是 `_e(c, u, g)` —— 即**同一个包**对**同一个目标**注册第二次就抛（不同包互相叠加是允许的）。桩照这个条件实现。
  - `module/helpers/YZEDiceRoller.mjs:31` 的 `static async yzeRoll(actortype, blind, reRoll, label, r1Dice, col1, r2Dice, col2, actorid, itemid, tactorid, moddata)`，`:416` 自行 `await ChatMessage.create(chatData)`、`:417` `return`。本任务不碰它，只是桩里的 `ChatMessage.create` 必须能承载这条路径。

- **Produces:**
  - `scripts/const.mjs` → `MID`、`FLAG_ROLL`、`HOOK_ROLL_RESOLVED`、`RECORD_VERSION`、`SETTING_FEATURES`、`SETTING_BINDINGS`、`SETTING_SCHEMA`、`SETTING_DATA_REPAIRS`、`SETTING_DICE_TIMEOUT`、`SYSTEM_ID`、`I18N`（共 11 个，多一个都不行，有护栏断言）
  - `scripts/main.mjs` → **十条逐字锚点注释**，顺序固定：`/* AEA-ANCHOR: imports */`、`/* AEA-ANCHOR: features */`、`/* AEA-ANCHOR: repairs */`、`/* AEA-ANCHOR: init */`、`/* AEA-ANCHOR: i18nInit */`、`/* AEA-ANCHOR: diceSoNiceReady */`、`/* AEA-ANCHOR: ready.registry */`、`/* AEA-ANCHOR: ready.patches */`、`/* AEA-ANCHOR: ready.rollbus */`、`/* AEA-ANCHOR: ready.cards */`
  - `scripts/main.mjs` → `export const api`：键集与键序恒为八个内核槽 `features, patches, resolver, registry, rollBus, diceBarrier, cards, selftest`，本任务交付时**八个槽全是 `null`**
  - `scripts/main.mjs` → `const FEATURES` 与 `const REPAIRS` 两个**多行、不导出**的数组，以及 init 里两条 `register()` 循环、ready 里两条 `install()` 循环
  - `scripts/main.mjs` → `export function safely(what, fn)`、`export async function waitForWorldSettled({timeoutMs, pollMs}) -> "settled"|"timeout"`
  - `test/stubs/foundry.mjs` → **只有三个导出**：`installFoundryStub(options) -> ctx`、`uninstallFoundryStub()`、`foundryStubContext() -> ctx|null`；`ctx` 字段恒为 18 个：`moduleId`、`isGM`、`userId`、`systemVersion`、`i18n`、`settings`、`registered`、`menus`、`hooks`、`notifications`、`wrappers`、`messages`、`rolls`、`templates`、`modules`、`documents`、`babele`、`world`
  - `test/lib/discipline-scan.mjs` → `BANNED`、`stripComments(source)`、`findViolations(file, source, allowlist)`
  - `module.json` → 硬依赖只有 `lib-wrapper`；`socketlib` / `dice-so-nice` / `yze-combat` 都在 `recommends`
  - `lang/en.json`、`lang/cn.json` → **嵌套结构**，顶层唯一键是 `AEA`

**背景（读的人可能完全不懂 Foundry）**：Foundry VTT 是一个浏览器里跑的桌游平台。一个「模组（module）」就是 `Data/modules/<id>/` 下的一个目录，根上必须有 `module.json` 清单；Foundry 启动时读清单，按 `esmodules` 数组把 JS 当 ES module 加载进页面。清单里的 `relationships.requires` 会让 Foundry 在依赖缺失时拒绝启用本模组，`relationships.recommends` 只是在安装界面提示一句。Foundry 的版本比较不是 semver：`compatibility.minimum: "13"` 指 Foundry 主版本 13，`verified: "14"` 是「作者在 14 上实测过」。「钩子（Hook）」是 Foundry 的全局事件总线，`Hooks.once(name, fn)` 让 `fn` 在该事件首次触发时跑一次；启动期依次触发 `init` → `i18nInit` → `setup` → `ready`，其中 `diceSoNiceReady` 只有装了 Dice So Nice 这个 3D 骰子模组时才会触发。Foundry 客户端 API 没有 npm 包，所以凡是碰 `game` / `ui` / `Hooks` 的代码在 Node 里都必须靠手写的桩才能测。

**本任务建立的五条纪律（后面十七个任务全靠它们才能互不打架）**：

1. **仓库边界。** 本模组是**独立 git 仓库**，根目录是 `Data/modules/alien-evolved-automation/`。后续所有任务里写的相对路径都以它为根，与 `Data/systems/alienrpg/`（系统源码，只读参考）无关。Step 1 显式 `git init`，避免本模组的文件变成系统仓库里的未跟踪文件。
2. **`socketlib` 一期是软依赖。** 一期完全没有它的消费者（要用它的执行器内核排在二期），不该强迫用户装一个用不上的库。清单里写进 `recommends`，二期再提升为 `requires`。
3. **十条锚点是唯一的接线坐标，且 ready 段是有序的四条子锚点。** `main.mjs` 由本任务一次写全，后续任何任务往里插代码，一律**引用锚点原文定位**，**永不按行号**。上一轮的教训是：只有一条 `ready` 锚点、四个任务都说「插在它之后」，谁先谁后完全看谁最后动手，文件头部会整个倒序——`cards.init()` 跑到 `registry.resolveAll()` 前面去。所以 ready 段拆成 `ready.registry` → `ready.patches` → `ready.rollbus` → `ready.cards` 四条**按执行顺序排好**的子锚点，每个属主只往自己那条的下一行插一句，顺序由文件形状保证而不是由动手顺序保证。本任务有两条测试直接读 `main.mjs` 源文本：一条断言十条锚点各自恰好出现一次，一条断言它们在文件里的先后就是上面这个顺序。**写文件头注释时不要把带 `/* */` 定界符的锚点原文再抄一遍**，提到锚点时只写 `AEA-ANCHOR: init` 这样的裸名，否则「恰好一次」那条断言会红。
4. **`api` 是八个槽，本任务交付时全为 `null`。** 本任务动手时七个内核文件根本不存在，`import` 它们会让整个模组加载失败，所以只能先写空槽。每个内核模块的属主任务做三件事、一件不多：在 `AEA-ANCHOR: imports` 后加一行自己的 import、把 `api` 里**自己那一个** `null` 换成 import 进来的对象、在自己的 ready 子锚点后插自己的调用。**不得**替换 `api` 这个对象本身、**不得**增删键——`publishApi()` 是按引用挂上去的一行赋值，谁重建对象谁就把别人的槽抹掉。本任务的测试冻结键集与键序。
5. **测试桩全模组共用一份，且行为被逐条守卫。** `test/stubs/foundry.mjs` 是唯一的 Foundry 全局桩，导出面固定为三个符号。后续任务只许通过 `installFoundryStub(options)` 的 `options` 声明自己要的夹具，**不许**在自己的测试文件里就地造 `globalThis.game`、也不许用私有 Map 顶替 `game.settings`——各造各的假货，等于每个组只跟自己的假设一致。桩的行为不是「大概像」而是有契约的：`test/stub-fidelity.test.mjs` 一条用例守一条契约，本任务负责写全。

---

- [ ] **Step 1: 建仓库与工具链骨架**

```bash
mkdir -p "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/kernel" \
         "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/features" \
         "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/repairs" \
         "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/lang" \
         "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/styles" \
         "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/templates" \
         "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/test/stubs" \
         "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/test/lib"
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation"
git init -b main
printf 'node_modules/\npackage-lock.json\n.DS_Store\n' > .gitignore
cat > package.json <<'EOF'
{
	"name": "alien-evolved-automation",
	"version": "0.1.0",
	"private": true,
	"type": "module",
	"description": "Rules automation for the alienrpg (Alien Evolved) Foundry VTT game system.",
	"scripts": {
		"test": "vitest run",
		"test:watch": "vitest"
	}
}
EOF
cat > vitest.config.mjs <<'EOF'
import { defineConfig } from "vitest/config"

export default defineConfig({
	test: {
		environment: "node",
		include: ["test/**/*.test.mjs"],
		globals: false,
		restoreMocks: true,
	},
})
EOF
npm install --save-dev vitest
```

Expected: `node_modules/` 出现，`npx vitest --version` 打印版本号。

`"type": "module"` 让 Node 把 `.mjs`/`.js` 当 ES module；`include` 只收 `*.test.mjs`，所以 `test/stubs/foundry.mjs` 与 `test/lib/discipline-scan.mjs` 不会被当测试跑。vitest 是唯一 devDependency——不引 jsdom，因此本模组任何测试都不得假装有真实 `document`。

- [ ] **Step 2: 写会失败的 const 测试**

```js
// test/const.test.mjs
import { describe, it, expect } from "vitest"
import * as C from "../scripts/const.mjs"

const CONTRACTED = {
	MID: "alien-evolved-automation",
	FLAG_ROLL: "roll",
	HOOK_ROLL_RESOLVED: "aea.rollResolved",
	RECORD_VERSION: 1,
	SETTING_FEATURES: "features",
	SETTING_BINDINGS: "registryBindings",
	SETTING_SCHEMA: "settingsSchemaVersion",
	SETTING_DATA_REPAIRS: "dataRepairs",
	SETTING_DICE_TIMEOUT: "diceTimeoutMs",
	SYSTEM_ID: "alienrpg",
	I18N: "AEA",
}

describe("scripts/const.mjs", () => {
	it("exports every contracted identifier with the contracted value", () => {
		for (const [name, value] of Object.entries(CONTRACTED)) {
			expect(C[name], `const.mjs must export ${name}`).toBe(value)
		}
	})

	it("exports nothing beyond the contract", () => {
		expect(Object.keys(C).sort()).toEqual(Object.keys(CONTRACTED).sort())
	})
})
```

第二条断言是有意的护栏：接口契约禁止发明符号，`const.mjs` 多一个导出就红。任何后续任务想加常量，必须先改契约。

- [ ] **Step 3: 跑它，看它红**

Run: `cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation" && npx vitest run test/const.test.mjs`
Expected: FAIL — `Error: Failed to load url ../scripts/const.mjs (resolved id: .../scripts/const.mjs). Does the file exist?`（整个文件加载失败，两条用例都不执行）

- [ ] **Step 4: 写 const.mjs**

```js
// scripts/const.mjs
/** Module id. Doubles as the libWrapper package name and the flag namespace. */
export const MID = "alien-evolved-automation"

/** Key under message.flags[MID] where the RollRecord lives. */
export const FLAG_ROLL = "roll"

/** Hook every downstream feature subscribes to instead of wrapping the roller itself. */
export const HOOK_ROLL_RESOLVED = "aea.rollResolved"

/** RollRecord schema version. Migration anchor — bump only with a migration. */
export const RECORD_VERSION = 1

/** world setting key: per-feature full/prompt/off map. */
export const SETTING_FEATURES = "features"

/** world setting key: document registry bindings. */
export const SETTING_BINDINGS = "registryBindings"

/** world setting key: stored settings shape version. */
export const SETTING_SCHEMA = "settingsSchemaVersion"

/** world setting key: which one-shot data repairs have already run in this world. */
export const SETTING_DATA_REPAIRS = "dataRepairs"

/**
 * client setting key: how long the dice barrier waits for a 3D dice animation
 * before giving up. The default (4000 ms) is declared where the setting is
 * registered, not here — this file only owns the key.
 */
export const SETTING_DICE_TIMEOUT = "diceTimeoutMs"

/** The game system this module automates. */
export const SYSTEM_ID = "alienrpg"

/**
 * Prefix for every i18n key this module owns. Foundry flattens the language JSON
 * at load, so the nested file {"AEA": {"Boot": {"Ready": "..."}}} is looked up as
 * "AEA.Boot.Ready".
 */
export const I18N = "AEA"
```

- [ ] **Step 5: 跑它，看它绿**

Run: `npx vitest run test/const.test.mjs`
Expected: PASS — 2 passed。

- [ ] **Step 6: 提交**

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation"
git add -A && git commit -m "$(cat <<'EOF'
chore(scaffold): 建立独立仓库骨架与 vitest 工具链，落地 const.mjs

本模组是独立 git 仓库，根在 Data/modules/alien-evolved-automation，与系统源码
仓库无关；后续任务的相对路径一律以它为根。

vitest 为唯一 devDependency，不引 jsdom —— 因此本模组任何测试都不得假装有真实
document。测试只收 test/**/*.test.mjs，所以 test/stubs/ 与 test/lib/ 下的辅助
文件不会被当用例跑。

const.mjs 逐字照接口契约 §1 写死这 11 个常量，并加一条「不得多出导出」的护栏
断言：后续任务想加常量，必须先改契约。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: 写会失败的清单与语言包测试**

```js
// test/manifest.test.mjs
import { describe, it, expect } from "vitest"
import { existsSync, readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { MID, SYSTEM_ID, I18N } from "../scripts/const.mjs"

const root = fileURLToPath(new URL("../", import.meta.url))
const read = (rel) => JSON.parse(readFileSync(root + rel, "utf8"))

const manifest = read("module.json")

/** Foundry flattens a nested language file into dotted keys at load time. */
function flatten(value, prefix = "", out = {}) {
	for (const [key, child] of Object.entries(value)) {
		const path = prefix ? `${prefix}.${key}` : key
		if (child && typeof child === "object" && !Array.isArray(child)) flatten(child, path, out)
		else out[path] = child
	}
	return out
}

describe("module.json", () => {
	it("declares the contracted identity", () => {
		expect(manifest.id).toBe(MID)
		expect(manifest.title).toBe("Alien Evolved: Automation")
		expect(manifest.esmodules).toEqual(["scripts/main.mjs"])
		expect(manifest.socket).toBe(true)
	})

	it("targets Foundry 13 minimum, 14 verified", () => {
		expect(manifest.compatibility.minimum).toBe("13")
		expect(manifest.compatibility.verified).toBe("14")
	})

	it("requires the alienrpg system at 4.1.13 or newer", () => {
		const sys = manifest.relationships.systems.find((r) => r.id === SYSTEM_ID)
		expect(sys).toBeDefined()
		expect(sys.type).toBe("system")
		expect(sys.compatibility.minimum).toBe("4.1.13")
	})

	it("hard-requires only lib-wrapper; socketlib is a phase-1 recommendation", () => {
		expect(manifest.relationships.requires.map((r) => r.id)).toEqual(["lib-wrapper"])
		expect(manifest.relationships.recommends.map((r) => r.id).sort()).toEqual([
			"dice-so-nice",
			"socketlib",
			"yze-combat",
		])
		for (const rel of [...manifest.relationships.requires, ...manifest.relationships.recommends]) {
			expect(rel.type).toBe("module")
		}
	})

	it("only points at files that exist", () => {
		const referenced = [
			...manifest.esmodules,
			...manifest.styles,
			...manifest.languages.map((l) => l.path),
		]
		for (const rel of referenced) {
			expect(existsSync(root + rel), `module.json references missing file ${rel}`).toBe(true)
		}
	})
})

describe("language files", () => {
	it("ships exactly en and cn", () => {
		expect(manifest.languages.map((l) => l.lang)).toEqual(["en", "cn"])
	})

	it("is nested under a single AEA root, so every flattened key is namespaced", () => {
		for (const file of ["lang/en.json", "lang/cn.json"]) {
			const raw = read(file)
			expect(Object.keys(raw), `${file} must have exactly one top-level key`).toEqual([I18N])
			for (const key of Object.keys(flatten(raw))) {
				expect(key.startsWith(`${I18N}.`), `${file} → ${key} is outside the ${I18N}. namespace`).toBe(true)
			}
		}
	})

	it("keeps cn.json key-for-key identical to en.json", () => {
		expect(Object.keys(flatten(read("lang/cn.json"))).sort()).toEqual(
			Object.keys(flatten(read("lang/en.json"))).sort()
		)
	})

	it("has no empty translations", () => {
		for (const file of ["lang/en.json", "lang/cn.json"]) {
			for (const [key, value] of Object.entries(flatten(read(file)))) {
				expect(typeof value, `${file} → ${key}`).toBe("string")
				expect(value.trim().length, `${file} → ${key} is empty`).toBeGreaterThan(0)
			}
		}
	})
})
```

语言代码用 `cn` 而不是 `zh-CN`：系统 `alienrpg` 的 `system.json:164` 里中文条目就是 `"lang": "cn"`，两边必须一致才对得上。语言包用嵌套结构而不是扁平点分键——Foundry 两种都吃，加载时统一压平成点分键，但嵌套写法在两份文件之间做对照时肉眼能看出结构差异。测试里的 `flatten` 就是复现 Foundry 的这一步。

- [ ] **Step 8: 跑它，看它红**

Run: `npx vitest run test/manifest.test.mjs`
Expected: FAIL — `Error: ENOENT: no such file or directory, open '.../module.json'`（整个文件在加载期就抛，9 条用例全不执行）

- [ ] **Step 9: 写 module.json、两份语言包与样式表**

```json
// module.json
{
	"id": "alien-evolved-automation",
	"title": "Alien Evolved: Automation",
	"description": "Completes the rules automation of the alienrpg (Alien Evolved) game system: a single roll interception point, an id-based document registry, and self-retiring patches for upstream defects.",
	"version": "0.1.0",
	"authors": [
		{ "name": "takaqiao" }
	],
	"compatibility": {
		"minimum": "13",
		"verified": "14"
	},
	"esmodules": [
		"scripts/main.mjs"
	],
	"styles": [
		"styles/alien-evolved-automation.css"
	],
	"languages": [
		{ "lang": "en", "name": "English", "path": "lang/en.json" },
		{ "lang": "cn", "name": "中文", "path": "lang/cn.json" }
	],
	"relationships": {
		"systems": [
			{
				"id": "alienrpg",
				"type": "system",
				"compatibility": { "minimum": "4.1.13", "verified": "4.1.13" }
			}
		],
		"requires": [
			{
				"id": "lib-wrapper",
				"type": "module",
				"manifest": "https://github.com/ruipin/fvtt-lib-wrapper/releases/latest/download/module.json",
				"compatibility": { "minimum": "1.13.5", "verified": "1.13.5.1" }
			}
		],
		"recommends": [
			{
				"id": "socketlib",
				"type": "module",
				"reason": "Needed when player-initiated actions start writing to GM-owned documents. Nothing in this release uses it yet."
			},
			{
				"id": "dice-so-nice",
				"type": "module",
				"reason": "Holds automation back until the 3D dice have landed. Without it the module resolves immediately."
			},
			{
				"id": "yze-combat",
				"type": "module",
				"reason": "Year Zero card initiative. Foundry V14 only, so it can never be a hard requirement."
			}
		]
	},
	"socket": true,
	"url": "https://github.com/takaqiao/alien-evolved-automation",
	"manifest": "https://github.com/takaqiao/alien-evolved-automation/releases/latest/download/module.json",
	"download": "https://github.com/takaqiao/alien-evolved-automation/releases/download/0.1.0/module.zip"
}
```

两处需要解释：

- **`socketlib` 在 `recommends` 而不是 `requires`。** 这个版本没有任何代码用到它——用它的执行器内核排在二期。`requires` 会让缺它的用户根本启用不了本模组，为一个用不上的库付这个代价不合理。二期提升为 `requires`。
- **`"socket": true` 现在没有消费者。** 之所以先写上：`module.json` 的 socket 声明只在 Foundry 启动时读一次，二期再加就要求所有人重启服务器，成本高于现在多一行。

```json
// lang/en.json
{
	"AEA": {
		"ModuleTitle": "Alien Evolved: Automation",
		"Boot": {
			"Ready": "Alien Evolved: Automation is ready."
		}
	}
}
```

```json
// lang/cn.json
{
	"AEA": {
		"ModuleTitle": "异形进化版：自动化",
		"Boot": {
			"Ready": "异形进化版：自动化已就绪。"
		}
	}
}
```

```css
/* styles/alien-evolved-automation.css
   Everything this module renders lives under .aea / .aea-mount so nothing leaks
   into the system's own sheets, which style by bare element selectors in places.
   .aea-mount is the per-chat-card container the card kernel creates; each feature
   later renders into its own child of it. */
.aea,
.aea-mount {
	font-family: inherit;
	color: inherit;
}

.aea-mount:empty {
	display: none;
}
```

- [ ] **Step 10: 跑它，看它绿**

Run: `npx vitest run test/manifest.test.mjs`
Expected: PASS — 9 passed。

- [ ] **Step 11: 提交**

```bash
git add -A && git commit -m "$(cat <<'EOF'
feat(manifest): 落地 module.json、嵌套双语语言包与样式表

依赖关系按本机已装的 V14 模组实测格式写：relationships {systems, requires,
recommends}，每条 {id, type, compatibility}。硬依赖只留 lib-wrapper：

- socketlib 一期没有任何消费者（要用它的执行器内核在二期），写进 requires 等于
  逼用户装一个用不上的库，故降为 recommends，二期再提升；
- dice-so-nice 6.2.9 与 yze-combat 1.7.1 本来就只能软依赖 —— 两者的清单都把
  Foundry 主版本下限写死为 14，硬依赖会让 V13 用户装不上。

语言包用嵌套结构，顶层唯一键 AEA；Foundry 加载时会压平成点分键，测试里的
flatten 复现同一步，并强制 cn.json 与 en.json 压平后键集完全相同、无空译文、
无落在 AEA. 命名空间之外的键。语言代码取 cn 而非 zh-CN，与系统 system.json:164
一致。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 12: 写会失败的桩契约测试**

这个文件是接口契约 §0.3「桩的行为契约」的逐条守卫：**一条用例守一条契约**。九个下游任务都要靠这些行为，桩必须先被钉死。

```js
// test/stub-fidelity.test.mjs
import { describe, it, expect, afterEach, vi } from "vitest"
import { installFoundryStub, uninstallFoundryStub, foundryStubContext } from "./stubs/foundry.mjs"

/** Frozen by contract §0.3. A task that needs a new fixture adds an option, not a ctx field. */
const CTX_FIELDS = [
	"moduleId", "isGM", "userId", "systemVersion", "i18n", "settings", "registered", "menus",
	"hooks", "notifications", "wrappers", "messages", "rolls", "templates", "modules", "documents",
	"babele", "world",
]

const OWNED_GLOBALS = [
	"game", "ui", "Hooks", "ChatMessage", "Roll", "libWrapper", "foundry",
	"CONFIG", "CONST", "logger", "canvas", "fromUuid", "fromUuidSync",
]

describe("test/stubs/foundry.mjs — contract §0.3", () => {
	afterEach(() => uninstallFoundryStub())

	it("exports exactly three symbols and installs nothing at import time", async () => {
		const surface = await import("./stubs/foundry.mjs")
		expect(Object.keys(surface).sort()).toEqual([
			"foundryStubContext",
			"installFoundryStub",
			"uninstallFoundryStub",
		])
		expect(foundryStubContext()).toBeNull()
		expect(globalThis.game).toBeUndefined()
	})

	it("installs every global it owns and takes them all away again", () => {
		const ctx = installFoundryStub()
		for (const key of OWNED_GLOBALS) expect(globalThis[key], `${key} should exist`).toBeDefined()
		expect(game.system.id).toBe("alienrpg")
		expect(game.modules.get("alien-evolved-automation").active).toBe(true)
		expect(game.user.isGM).toBe(true)
		expect(foundryStubContext()).toBe(ctx)

		uninstallFoundryStub()
		for (const key of OWNED_GLOBALS) expect(globalThis[key], `${key} should be gone`).toBeUndefined()
		expect(foundryStubContext()).toBeNull()
	})

	it("is idempotent: a second install replaces the first, a second uninstall is a no-op", () => {
		const first = installFoundryStub({ userId: "u1" })
		const second = installFoundryStub({ userId: "u2" })
		expect(second).not.toBe(first)
		expect(foundryStubContext()).toBe(second)
		expect(game.user.id).toBe("u2")

		uninstallFoundryStub()
		expect(() => uninstallFoundryStub()).not.toThrow()
		expect(foundryStubContext()).toBeNull()
	})

	it("carries exactly the eighteen contracted ctx fields", () => {
		const ctx = installFoundryStub()
		expect(Object.keys(ctx).sort()).toEqual([...CTX_FIELDS].sort())
	})

	it("backs game.settings with ctx.settings and refuses unregistered keys", async () => {
		const ctx = installFoundryStub()
		game.settings.register("ns", "k", { scope: "world", config: false, type: Number, default: 3 })
		expect(game.settings.get("ns", "k")).toBe(3)
		expect(ctx.settings.get("ns.k")).toBe(3)

		const outcome = game.settings.set("ns", "k", 7)
		expect(typeof outcome.then).toBe("function")
		await outcome
		expect(game.settings.get("ns", "k")).toBe(7)
		expect(ctx.settings.get("ns.k")).toBe(7)
		expect(ctx.registered.get("ns.k").scope).toBe("world")
		expect(() => game.settings.get("ns", "missing")).toThrow(/not registered/)
		await expect(game.settings.set("ns", "missing", 1)).rejects.toThrow(/not registered/)
	})

	it("records menus and exposes Foundry's two settings maps", () => {
		const ctx = installFoundryStub()
		game.settings.registerMenu("ns", "m", { label: "Open", restricted: true })
		expect(ctx.menus.get("ns.m").label).toBe("Open")
		expect(game.settings.menus).toBe(ctx.menus)
		expect(game.settings.settings).toBe(ctx.registered)
	})

	it("really dispatches hooks, honours once, and lets call() be vetoed", () => {
		const ctx = installFoundryStub()
		const seen = []
		Hooks.once("ready", () => seen.push("once"))
		Hooks.on("ready", () => seen.push("on"))
		Hooks.callAll("ready")
		Hooks.callAll("ready")
		expect(seen).toEqual(["once", "on", "on"])
		expect(ctx.hooks.once).toEqual([{ name: "ready", fn: expect.any(Function), once: true }])
		expect(ctx.hooks.on).toEqual([{ name: "ready", fn: expect.any(Function), once: false }])
		expect(ctx.hooks.calls.map((c) => c.name)).toEqual(["ready", "ready"])

		const after = []
		Hooks.on("veto", () => false)
		Hooks.on("veto", () => after.push("reached"))
		expect(Hooks.call("veto", { id: "r1" })).toBe(false)
		expect(after).toEqual([])
		expect(Hooks.callAll("veto")).toBe(true)
		expect(after).toEqual(["reached"])
		expect(ctx.hooks.calls.at(-2)).toEqual({ name: "veto", args: [{ id: "r1" }] })
	})

	it("resolves uuids out of ctx.documents and returns null on a miss", async () => {
		const ctx = installFoundryStub()
		const doc = { uuid: "Actor.abc", name: "Ripley" }
		ctx.documents.set(doc.uuid, doc)
		expect(fromUuidSync("Actor.abc")).toBe(doc)
		expect(foundry.utils.fromUuidSync("Actor.abc")).toBe(doc)
		expect(fromUuidSync("Actor.nope")).toBeNull()
		await expect(fromUuid("Actor.abc")).resolves.toBe(doc)
		await expect(fromUuid("Actor.nope")).resolves.toBeNull()
	})

	it("creates a chat message through preCreate then create, where only updateSource lands", async () => {
		const ctx = installFoundryStub()
		const order = []
		Hooks.on("preCreateChatMessage", (doc, data, operation, userId) => {
			order.push(`pre:${userId}`)
			expect(data.content).toBe("hi")
			expect(operation.rollMode).toBe("public")
			data.flags = { ignored: true }
			doc.updateSource({ "flags.alien-evolved-automation.roll": { id: "r1" } })
		})
		Hooks.on("createChatMessage", (doc) => order.push(`create:${doc.content}`))

		const message = await ChatMessage.create({ content: "hi" }, { rollMode: "public" })
		expect(order).toEqual(["pre:stub-user", "create:hi"])
		expect(message.flags["alien-evolved-automation"].roll).toEqual({ id: "r1" })
		expect(message.getFlag("alien-evolved-automation", "roll")).toEqual({ id: "r1" })
		expect(message.flags.ignored).toBeUndefined()
		expect(ctx.messages).toEqual([message])

		Hooks.on("preCreateChatMessage", () => false)
		await expect(ChatMessage.create({ content: "no" })).resolves.toBeUndefined()
		expect(ctx.messages).toHaveLength(1)
	})

	it("reports the system version and the Foundry generation", () => {
		installFoundryStub()
		expect(game.system.version).toBe("4.1.13")
		expect(game.release.generation).toBe(14)

		installFoundryStub({ systemVersion: "4.2.0", generation: 13 })
		expect(game.system.version).toBe("4.2.0")
		expect(game.release.generation).toBe(13)
	})

	it("records libWrapper registrations and refuses a second one on the same target", () => {
		const ctx = installFoundryStub()
		const fn = () => {}
		libWrapper.register("alien-evolved-automation", "A.b.c", fn, "WRAPPER")
		expect(ctx.wrappers).toEqual([
			{ module: "alien-evolved-automation", target: "A.b.c", fn, type: "WRAPPER" },
		])

		expect(() => libWrapper.register("alien-evolved-automation", "A.b.c", fn, "MIXED")).toThrow(
			/A wrapper for 'A\.b\.c' \(ID=\d+\) has already been registered by/
		)
		expect(ctx.wrappers).toHaveLength(1)
		expect(() => libWrapper.register("other-module", "A.b.c", fn, "WRAPPER")).not.toThrow()
		expect(ctx.wrappers).toHaveLength(2)
	})

	it("localizes, formats, records notifications, and never invents dice", () => {
		const ctx = installFoundryStub({ i18n: { "X.y": "hello {who}" } })
		expect(game.i18n.localize("X.y")).toBe("hello {who}")
		expect(game.i18n.localize("X.missing")).toBe("X.missing")
		expect(game.i18n.has("X.y")).toBe(true)
		expect(game.i18n.has("X.missing")).toBe(false)
		expect(game.i18n.format("X.y", { who: "world" })).toBe("hello world")

		ui.notifications.info("hi")
		ui.notifications.warn("careful")
		expect(ctx.notifications).toEqual([
			{ type: "info", message: "hi" },
			{ type: "warn", message: "careful" },
		])

		const roll = new Roll("2d6")
		expect(ctx.rolls).toEqual([roll])
		expect(roll.total).toBeNull()

		expect(foundry.utils.expandObject({ "modes.a-b": "off", "modes.c": "full" }))
			.toEqual({ modes: { "a-b": "off", c: "full" } })
		expect(foundry.utils.flattenObject({ modes: { a: 1 } })).toEqual({ "modes.a": 1 })
		expect(foundry.utils.isNewerVersion).toBeUndefined()
	})
})
```

两处刻意为之，不是遗漏：

- 最后那句 `expect(foundry.utils.isNewerVersion).toBeUndefined()`——补丁内核要自己实现版本比较，如果桩里放一份「我自己写的 isNewerVersion」再拿它做基准，就是自己验自己，等于没验。真正的对照放在真实 Foundry 里跑的自检条目里。
- `data.flags = { ignored: true }` 那两行——它把 `preCreateChatMessage` 的语义钉死成真实 Foundry 的样子：钩子拿到的第一个参数是**已经用 `data` 构造好、尚未落库**的文档实例，改第二个参数 `data` 对最终落库的文档**没有任何影响**，唯一有效的写法是 `doc.updateSource({...})`。本机两个已装模组就是这么写的（`modules/dice-chronicle/scripts/tracker.js:13`、`modules/pf2e-target-helper/src/main.js:97`）。桩照这个语义实现，任何往 `data` 上写 flag 的实现都会在自己的测试里立刻变红——这正是想要的效果。

`libWrapper` 那条同理：lib-wrapper 1.13.5.1 的判据是**同一个包**对**同一个目标**注册第二次就抛（源码里的条件是 `_e(c, u, g)`，报文原文是 ``A wrapper for '${f}' (ID=${h}) has already been registered by ${c.type_plus_id}.``）。整个模组共用 `MID` 这一个包名，所以两处代码抢同一个目标时后一处会抛——桩必须复现这个失败，否则「谁抢了谁」只能等到真实世界里才发现。

- [ ] **Step 13: 跑它，看它红**

Run: `npx vitest run test/stub-fidelity.test.mjs`
Expected: FAIL — `Error: Failed to load url ./stubs/foundry.mjs (resolved id: .../test/stubs/foundry.mjs). Does the file exist?`，12 条用例全不执行。

- [ ] **Step 14: 写 test/stubs/foundry.mjs（核心全局）**

```js
// test/stubs/foundry.mjs
/**
 * A hand-written stand-in for the Foundry VTT client globals.
 *
 * Foundry ships as a browser app; there is no npm package for its client API, so
 * anything that touches `game`, `ui`, `Hooks`, `ChatMessage`, `Roll`, `libWrapper`
 * or `foundry.*` cannot be unit tested without a stub. This file provides the
 * smallest honest one: real recording behaviour and real dispatch, no pretend
 * business logic.
 *
 * THIS FILE HAS EXACTLY THREE EXPORTS and installs nothing at import time.
 * `import "./stubs/foundry.mjs"` for side effects will NOT give you globals —
 * call installFoundryStub() from beforeEach and uninstallFoundryStub() from
 * afterEach. Other tasks declare fixtures through `options` and MUST NOT build
 * their own private fakes of anything this file already provides: no
 * `globalThis.game = {...}` in a test file, no private Map standing in for
 * `game.settings`. test/stub-fidelity.test.mjs guards every behaviour below.
 *
 * Deliberate omission: foundry.utils.isNewerVersion. The patch kernel implements
 * its own version comparison, and checking it against a copy written by the same
 * author in the same repo would prove nothing. That one is verified against the
 * real Foundry by a registered self-test entry instead.
 *
 * Usage:
 *   vi.resetModules()                    // so the module under test re-runs top-level code
 *   const ctx = installFoundryStub()     // install globals BEFORE importing it
 *   const { thing } = await import("../scripts/kernel/thing.mjs")
 *   ...
 *   uninstallFoundryStub()
 */

const MODULE_ID = "alien-evolved-automation"

const INSTALLED_GLOBALS = [
	"game",
	"ui",
	"Hooks",
	"ChatMessage",
	"Roll",
	"libWrapper",
	"foundry",
	"CONFIG",
	"CONST",
	"logger",
	"canvas",
	"fromUuid",
	"fromUuidSync",
]

function setProperty(obj, path, value) {
	const parts = String(path).split(".")
	let target = obj
	for (let i = 0; i < parts.length - 1; i++) {
		const key = parts[i]
		if (typeof target[key] !== "object" || target[key] === null) target[key] = {}
		target = target[key]
	}
	target[parts[parts.length - 1]] = value
	return obj
}

function getProperty(obj, path) {
	return String(path)
		.split(".")
		.reduce((o, k) => (o === undefined || o === null ? undefined : o[k]), obj)
}

/** Foundry turns a flat form payload {"a.b": 1} into a nested object {a:{b:1}}. */
function expandObject(flat) {
	const out = {}
	for (const [key, value] of Object.entries(flat ?? {})) setProperty(out, key, value)
	return out
}

/** The inverse: {a:{b:1}} -> {"a.b": 1}. Foundry uses it on language files. */
function flattenObject(obj, prefix = "", out = {}) {
	for (const [key, value] of Object.entries(obj ?? {})) {
		const path = prefix ? `${prefix}.${key}` : key
		if (value && typeof value === "object" && !Array.isArray(value)) flattenObject(value, path, out)
		else out[path] = value
	}
	return out
}

/** Deep-merge `source` into `target` in place. Arrays replace, they do not concat. */
function mergeInto(target, source) {
	for (const [key, value] of Object.entries(source ?? {})) {
		if (value && typeof value === "object" && !Array.isArray(value)) {
			if (!target[key] || typeof target[key] !== "object") target[key] = {}
			mergeInto(target[key], value)
		} else {
			target[key] = value
		}
	}
	return target
}

function randomID(length = 16) {
	const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
	let out = ""
	for (let i = 0; i < length; i++) out += chars[Math.floor(Math.random() * chars.length)]
	return out
}

const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }
function escapeHTML(text) {
	return String(text).replace(/[&<>"']/g, (c) => HTML_ESCAPES[c])
}

/** Enough of ApplicationV2 that a settings-menu class can be declared and inspected. */
class StubApplicationV2 {
	static DEFAULT_OPTIONS = {}
	constructor(options = {}) {
		this.options = options
	}
	async render() {
		return this
	}
	async close() {
		return this
	}
	async _prepareContext() {
		return {}
	}
}

const StubHandlebarsApplicationMixin = (Base) =>
	class extends Base {
		static PARTS = {}
	}

let installed = null

/**
 * @param {object}  [options]
 * @param {boolean} [options.isGM]           default true
 * @param {string}  [options.userId]         default "stub-user"
 * @param {string}  [options.systemVersion]  default "4.1.13"
 * @param {number}  [options.generation]     Foundry major version, default 14
 * @param {object}  [options.i18n]           ALREADY-FLATTENED key -> text, as Foundry stores it
 * @param {Array}   [options.modules]        extra packages, e.g. [{id:"babele", active:true}]
 * @param {object}  [options.babele]         value for game.babele
 * @returns {object} ctx
 */
export function installFoundryStub(options = {}) {
	// Idempotent: a second install implicitly tears the first one down, so a test
	// that installs inside an `it` on top of a beforeEach cannot leak globals.
	if (installed) uninstallFoundryStub()

	const moduleId = options.moduleId ?? MODULE_ID
	const ctx = {
		moduleId,
		isGM: options.isGM ?? true,
		userId: options.userId ?? "stub-user",
		systemVersion: options.systemVersion ?? "4.1.13",
		i18n: { ...(options.i18n ?? {}) },
		settings: new Map(),
		registered: new Map(),
		menus: new Map(),
		hooks: { once: [], on: [], calls: [] },
		notifications: [],
		wrappers: [],
		messages: [],
		rolls: [],
		templates: [],
		modules: new Map([[moduleId, { id: moduleId, active: true }]]),
		documents: new Map(),
		babele: options.babele ?? undefined,
		world: null,
	}
	for (const pkg of options.modules ?? []) {
		ctx.modules.set(pkg.id, { active: true, ...pkg })
	}

	// --- AEA-STUB: world fixture ---------------------------------------------

	// --- hooks: record AND dispatch -----------------------------------------
	// ctx.hooks.once / ctx.hooks.on are append-only records for assertions;
	// `listeners` is the live registry the dispatcher walks.
	const listeners = new Map()
	let hookId = 0

	function addListener(name, fn, once) {
		const entry = { id: ++hookId, name, fn, once }
		if (!listeners.has(name)) listeners.set(name, [])
		listeners.get(name).push(entry)
		ctx.hooks[once ? "once" : "on"].push({ name, fn, once })
		return entry.id
	}

	/**
	 * Foundry has two dispatch modes: `Hooks.call` is interruptible — the first
	 * listener returning exactly `false` stops the chain and the whole call
	 * reports false (that is how preCreate* hooks veto a document) — while
	 * `Hooks.callAll` always runs every listener.
	 */
	function dispatch(name, args, interruptible) {
		ctx.hooks.calls.push({ name, args })
		for (const entry of [...(listeners.get(name) ?? [])]) {
			if (entry.once) listeners.set(name, (listeners.get(name) ?? []).filter((e) => e.id !== entry.id))
			const outcome = entry.fn(...args)
			if (interruptible && outcome === false) return false
		}
		return true
	}

	const user = {
		get id() {
			return ctx.userId
		},
		get isGM() {
			return ctx.isGM
		},
		set isGM(value) {
			ctx.isGM = value
		},
		name: "Stub User",
		targets: new Set(),
	}

	globalThis.game = {
		system: {
			id: "alienrpg",
			get version() {
				return ctx.systemVersion
			},
		},
		release: { generation: options.generation ?? 14 },
		modules: ctx.modules,
		// --- AEA-STUB: world collections ---
		get babele() {
			return ctx.babele
		},
		get user() {
			return user
		},
		i18n: {
			localize: (key) => (key in ctx.i18n ? ctx.i18n[key] : key),
			format: (key, data = {}) =>
				(key in ctx.i18n ? ctx.i18n[key] : key).replace(/\{(\w+)\}/g, (_, name) => String(data[name] ?? "")),
			has: (key) => key in ctx.i18n,
		},
		// ctx.settings IS the backing store — never shadow game.settings with a
		// private Map in a test, read ctx.settings instead. game.settings.settings
		// and .menus are the two Maps real Foundry exposes under the same names.
		settings: {
			settings: ctx.registered,
			menus: ctx.menus,
			register(namespace, key, config) {
				const id = `${namespace}.${key}`
				ctx.registered.set(id, config)
				if (!ctx.settings.has(id)) ctx.settings.set(id, config.default)
			},
			registerMenu(namespace, key, config) {
				ctx.menus.set(`${namespace}.${key}`, config)
			},
			get(namespace, key) {
				const id = `${namespace}.${key}`
				if (!ctx.registered.has(id)) throw new Error(`stub: setting "${id}" is not registered`)
				return ctx.settings.get(id)
			},
			async set(namespace, key, value) {
				const id = `${namespace}.${key}`
				if (!ctx.registered.has(id)) throw new Error(`stub: setting "${id}" is not registered`)
				ctx.settings.set(id, value)
				return value
			},
		},
	}

	globalThis.ui = {
		notifications: {
			notify: (message, type = "info") => ctx.notifications.push({ type, message }),
			info: (message) => ctx.notifications.push({ type: "info", message }),
			warn: (message) => ctx.notifications.push({ type: "warn", message }),
			error: (message) => ctx.notifications.push({ type: "error", message }),
		},
	}

	globalThis.Hooks = {
		once: (name, fn) => addListener(name, fn, true),
		on: (name, fn) => addListener(name, fn, false),
		off: (name, fn) => listeners.set(name, (listeners.get(name) ?? []).filter((e) => e.fn !== fn)),
		call: (name, ...args) => dispatch(name, args, true),
		callAll: (name, ...args) => dispatch(name, args, false),
	}

	globalThis.ChatMessage = class ChatMessage {
		constructor(data = {}) {
			Object.assign(this, data)
			this.id ??= randomID()
			this.documentName = "ChatMessage"
			this.flags = { ...(data.flags ?? {}) }
		}
		/** The only supported way to mutate a document from inside a preCreate hook. */
		updateSource(changes = {}) {
			mergeInto(this, expandObject(changes))
			return this
		}
		getFlag(scope, key) {
			return getProperty(this.flags?.[scope] ?? {}, key)
		}
		async setFlag(scope, key, value) {
			setProperty(this.flags, `${scope}.${key}`, value)
			return this
		}
		/**
		 * Mirrors the real creation workflow: build the document from `data`, fire
		 * the interruptible preCreate hook with (document, data, operation, userId)
		 * SYNCHRONOUSLY, bail out if a listener returns false, then store and fire
		 * create. Mutating `data` inside the hook has NO effect — exactly as in
		 * Foundry, where the document was already built before the hook ran.
		 */
		static async create(data = {}, operation = {}) {
			const doc = new ChatMessage(data)
			if (Hooks.call("preCreateChatMessage", doc, data, operation, ctx.userId) === false) return undefined
			ctx.messages.push(doc)
			Hooks.callAll("createChatMessage", doc, operation, ctx.userId)
			return doc
		}
		static getSpeaker({ actor = null, token = null, alias = "" } = {}) {
			return { scene: null, actor: actor?.id ?? actor, token: token?.id ?? token, alias }
		}
		static getSpeakerActor(speaker = {}) {
			if (speaker.token && speaker.scene) {
				return fromUuidSync(`Scene.${speaker.scene}.Token.${speaker.token}`)?.actor ?? null
			}
			return game.actors?.get?.(speaker.actor) ?? null
		}
		static getWhisperRecipients() {
			return [user]
		}
	}

	/**
	 * Records construction; it does NOT simulate dice. `total` is permanently null.
	 * Any function under test that needs real dice values must take them as an
	 * injectable argument (fn(..., { roll = defaultRoll } = {})) so that no test
	 * ever asserts against invented randomness.
	 */
	globalThis.Roll = class Roll {
		constructor(formula, data = {}) {
			this.formula = formula
			this.data = data
			this.terms = []
			this._evaluated = false
			ctx.rolls.push(this)
		}
		async evaluate() {
			this._evaluated = true
			return this
		}
		get total() {
			return null
		}
	}

	/**
	 * lib-wrapper 1.13.5.1 throws when the SAME package registers the SAME target
	 * twice (different packages stacking on one target is fine). Reproduced here
	 * verbatim, because this whole module registers under one package id: two
	 * pieces of our own code reaching for one target is a real failure, and a stub
	 * that silently accepted it would hide it until a live world.
	 */
	globalThis.libWrapper = {
		register: (module, target, fn, type = "MIXED") => {
			if (ctx.wrappers.some((w) => w.module === module && w.target === target)) {
				throw new Error(
					`A wrapper for '${target}' (ID=${ctx.wrappers.length}) has already been registered by module:${module}.`
				)
			}
			ctx.wrappers.push({ module, target, fn, type })
			return ctx.wrappers.length
		},
		unregister: (module, target) => {
			const index = ctx.wrappers.findIndex((w) => w.module === module && w.target === target)
			if (index >= 0) ctx.wrappers.splice(index, 1)
		},
		unregister_all: (module) => {
			for (let i = ctx.wrappers.length - 1; i >= 0; i--) {
				if (ctx.wrappers[i].module === module) ctx.wrappers.splice(i, 1)
			}
		},
	}

	globalThis.foundry = {
		utils: {
			expandObject,
			flattenObject,
			setProperty,
			getProperty,
			randomID,
			escapeHTML,
			mergeObject: (target, other) => mergeInto(target, expandObject(other)),
			deepClone: (value) => structuredClone(value),
			fromUuidSync: (uuid) => ctx.documents.get(uuid) ?? null,
		},
		applications: {
			api: {
				ApplicationV2: StubApplicationV2,
				HandlebarsApplicationMixin: StubHandlebarsApplicationMixin,
			},
		},
	}

	// libWrapper targets are strings, but code that reaches for the document class
	// directly needs something to reach. These are inert placeholders.
	globalThis.CONFIG = {
		ALIENRPG: {},
		Dice: { terms: {} },
		Actor: { documentClass: class StubActorDocument {} },
		Item: { documentClass: class StubItemDocument {} },
		sounds: { dice: "sounds/dice.wav" },
	}
	globalThis.CONST = { CHAT_MESSAGE_STYLES: { OTHER: 0, OOC: 1, IC: 2, EMOTE: 3, WHISPER: 4, ROLL: 5 } }
	globalThis.logger = console

	globalThis.canvas = { scene: null, tokens: { placeables: [], controlled: [], get: () => null } }
	globalThis.fromUuidSync = (uuid) => ctx.documents.get(uuid) ?? null
	globalThis.fromUuid = async (uuid) => ctx.documents.get(uuid) ?? null

	installed = ctx
	return ctx
}

export function uninstallFoundryStub() {
	for (const key of INSTALLED_GLOBALS) delete globalThis[key]
	installed = null
}

export function foundryStubContext() {
	return installed
}
```

- [ ] **Step 15: 跑它，看它绿**

Run: `npx vitest run test/stub-fidelity.test.mjs`
Expected: PASS — 12 passed。

- [ ] **Step 16: 提交**

```bash
git add -A && git commit -m "$(cat <<'EOF'
test(stubs): 落地共用 Foundry 全局桩，行为按契约 §0.3 逐条守卫

Foundry 客户端 API 没有 npm 包，凡是碰 game/ui/Hooks/ChatMessage/Roll/
libWrapper/foundry.* 的代码都测不了。这个桩只做两件事：装卸全局、如实记录并派发
调用，不假装任何业务逻辑。全模组共用这一份 —— 各任务只许通过 options 声明夹具，
不许在自己的测试里就地造 globalThis.game 或用私有 Map 顶替 game.settings。

上一轮的问题不是桩不存在，而是桩「装哪些全局名」有约定、「表现成什么样」没有，
于是九个任务各自补各自的。这次 test/stub-fidelity.test.mjs 一条用例守一条契约：
幂等、ctx 十八个字段冻结、game.settings 由 ctx.settings 真后端支撑且未注册键抛错、
Hooks 真派发且 call 可中断、ctx.documents 支撑 fromUuidSync/fromUuid、
ChatMessage.create 先同步派发 preCreate 再派发 create、game.system.version 与
game.release.generation 可读、libWrapper 重复注册抛错、Roll.total 恒 null。

三处按真实实现逐字复现：
- 改 preCreateChatMessage 的第二个参数 data 对落库结果无效，唯一有效的写法是
  doc.updateSource() —— 与本机 dice-chronicle:13、pf2e-target-helper:97 一致；
- libWrapper 的判据是同一个包对同一个目标注册第二次就抛（报文取自
  lib-wrapper 1.13.5.1 源码），整个模组共用一个包名，所以这个失败必须能被测出来；
- Roll 桩不模拟骰子，需要真骰值的被测函数一律把掷骰做成可注入参数。

有意不提供 foundry.utils.isNewerVersion —— 补丁内核要自己实现版本比较，拿同一
作者写的副本当基准等于自己验自己。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 17: 写会失败的世界夹具测试**

```js
// test/stub-world.test.mjs
import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs"

/**
 * One fixture shared by every downstream task: two unlinked tokens of the same
 * base actor (the "three Drones" case the whole token-identity story exists for),
 * one linked token, a world item on an actor, a RollTable, a journal and a folder.
 */
const WORLD = {
	actors: [
		{ _id: "actorDrone", name: "Drone", type: "creature" },
		{
			_id: "actorRipley",
			name: "Ripley",
			type: "character",
			items: [{ _id: "itemPulse", name: "M41A Pulse Rifle", type: "weapon" }],
		},
	],
	tables: [
		{ _id: "tablePanic", name: "Panic Table", results: [{ _id: "resKeep", range: [1, 6], text: "Keeping it together" }] },
	],
	journal: [{ _id: "journalMother", name: "MU/TH/ER Instructions." }],
	folders: [{ _id: "folderTables", name: "Alien Tables", type: "RollTable" }],
	macros: [{ _id: "macroDraw", name: "Draw Panic" }],
	users: [{ _id: "userGM", name: "Stub User" }],
	scenes: [
		{
			_id: "sceneNostromo",
			name: "Nostromo",
			active: true,
			tokens: [
				{ _id: "tokenDroneA", name: "Drone A", actorId: "actorDrone", actorLink: false },
				{ _id: "tokenDroneB", name: "Drone B", actorId: "actorDrone", actorLink: false },
				{ _id: "tokenRipley", name: "Ripley", actorId: "actorRipley", actorLink: true },
			],
		},
	],
}

describe("installFoundryStub({world})", () => {
	afterEach(() => uninstallFoundryStub())

	it("leaves every world collection empty rather than undefined when no fixture is given", () => {
		installFoundryStub()
		for (const name of ["actors", "items", "tables", "journal", "folders", "scenes", "macros", "users"]) {
			expect(game[name], `game.${name} must exist`).toBeDefined()
			expect(game[name].contents, `game.${name} must be empty`).toEqual([])
		}
		expect(canvas.scene).toBeNull()
		expect(canvas.tokens.placeables).toEqual([])
	})

	describe("with the shared fixture", () => {
		beforeEach(() => installFoundryStub({ world: WORLD, canvas: { controlled: ["tokenDroneB"] } }))

		it("files each fixture into the collection Foundry would put it in", () => {
			expect(game.actors.get("actorDrone").name).toBe("Drone")
			expect(game.actors.getName("Ripley").id).toBe("actorRipley")
			expect(game.tables.getName("Panic Table").id).toBe("tablePanic")
			expect(game.journal.getName("MU/TH/ER Instructions.").id).toBe("journalMother")
			expect(game.folders.getName("Alien Tables").type).toBe("RollTable")
			expect(game.macros.getName("Draw Panic").id).toBe("macroDraw")
			expect(game.scenes.get("sceneNostromo").tokens.contents).toHaveLength(3)
			expect(game.actors.contents.map((a) => a.id)).toEqual(["actorDrone", "actorRipley"])
			expect(game.actors.size).toBe(2)
		})

		it("resolves uuids both sync and async, embedded documents included", async () => {
			expect(fromUuidSync("Actor.actorRipley").name).toBe("Ripley")
			expect(fromUuidSync("Actor.actorRipley.Item.itemPulse").name).toBe("M41A Pulse Rifle")
			expect(fromUuidSync("RollTable.tablePanic").name).toBe("Panic Table")
			expect(fromUuidSync("Scene.sceneNostromo.Token.tokenDroneA").name).toBe("Drone A")
			expect(fromUuidSync("Actor.nope")).toBeNull()
			await expect(fromUuid("Actor.actorDrone")).resolves.toBe(game.actors.get("actorDrone"))
			expect(game.actors.get("actorRipley").items.get("itemPulse").uuid)
				.toBe("Actor.actorRipley.Item.itemPulse")
		})

		it("gives two unlinked tokens of one base actor two distinct actor uuids", () => {
			const a = game.scenes.get("sceneNostromo").tokens.get("tokenDroneA")
			const b = game.scenes.get("sceneNostromo").tokens.get("tokenDroneB")
			expect(a.uuid).toBe("Scene.sceneNostromo.Token.tokenDroneA")
			expect(a.actor.isToken).toBe(true)
			expect(a.actor.token).toBe(a)
			expect(a.actor.uuid).not.toBe(b.actor.uuid)
			expect(a.actor.uuid).toBe("Scene.sceneNostromo.Token.tokenDroneA.Actor.actorDrone")

			const linked = game.scenes.get("sceneNostromo").tokens.get("tokenRipley")
			expect(linked.actor).toBe(game.actors.get("actorRipley"))
			expect(linked.actor.isToken).toBe(false)
		})

		it("returns placeables from getActiveTokens, documents only when asked", () => {
			const drone = game.actors.get("actorDrone")
			const placeables = drone.getActiveTokens()
			expect(placeables).toHaveLength(2)
			expect(placeables.map((t) => t.document.id)).toEqual(["tokenDroneA", "tokenDroneB"])
			expect(drone.getActiveTokens(false, true).map((d) => d.documentName)).toEqual(["Token", "Token"])
			expect(game.actors.get("actorRipley").getActiveTokens()).toHaveLength(1)
			expect(canvas.tokens.controlled[0].document.id).toBe("tokenDroneB")
			expect(canvas.tokens.placeables).toHaveLength(3)
			expect(canvas.scene.id).toBe("sceneNostromo")
		})

		it("exposes RollTable results as a collection and records renderTemplate calls", async () => {
			const table = game.tables.get("tablePanic")
			expect(table.results.get("resKeep").text).toBe("Keeping it together")
			expect(table.results.contents[0].uuid).toBe("RollTable.tablePanic.TableResult.resKeep")

			const ctx = installFoundryStub({ world: WORLD })
			const html = await foundry.applications.handlebars.renderTemplate("modules/x/a.hbs", { n: 1 })
			expect(html).toContain("modules/x/a.hbs")
			expect(ctx.templates).toEqual([{ path: "modules/x/a.hbs", data: { n: 1 } }])
		})
	})
})
```

- [ ] **Step 18: 跑它，看它红**

Run: `npx vitest run test/stub-world.test.mjs`
Expected: FAIL — 6 条用例全红。第一条报 `TypeError: Cannot read properties of undefined (reading 'contents')`（`game.actors` 还不存在），最后一条报 `TypeError: Cannot read properties of undefined (reading 'renderTemplate')`。

- [ ] **Step 19: 扩桩——世界文档夹具、uuid 解析、canvas 与模板渲染**

对 `test/stubs/foundry.mjs` 做五处改动，全部按原文匹配定位，不按行号。

**(a)** 在 `let installed = null` 这一行**之前**插入文档与集合的实现：

```js
/**
 * A world document. Foundry addresses documents by uuid: a top-level one is
 * "Actor.<id>", an embedded one appends its own segment, e.g.
 * "Actor.<id>.Item.<id>" or "Scene.<id>.Token.<id>".
 */
class StubDocument {
	constructor(documentName, data = {}, parent = null) {
		Object.assign(this, data)
		this.documentName = documentName
		this.id = data._id ?? randomID()
		this._id = this.id
		this.parent = parent
		this.flags = { ...(data.flags ?? {}) }
	}
	get uuid() {
		return this.parent ? `${this.parent.uuid}.${this.documentName}.${this.id}` : `${this.documentName}.${this.id}`
	}
	getFlag(scope, key) {
		return getProperty(this.flags?.[scope] ?? {}, key)
	}
	async setFlag(scope, key, value) {
		setProperty(this.flags, `${scope}.${key}`, value)
		return this
	}
	async update(changes = {}) {
		mergeInto(this, expandObject(changes))
		return this
	}
	updateSource(changes = {}) {
		mergeInto(this, expandObject(changes))
		return this
	}
}

/** Behaves like Foundry's document collections for the members code actually uses. */
class StubCollection extends Map {
	get contents() {
		return [...this.values()]
	}
	getName(name) {
		return this.contents.find((d) => d.name === name) ?? null
	}
	find(fn) {
		return this.contents.find(fn)
	}
	filter(fn) {
		return this.contents.filter(fn)
	}
	map(fn) {
		return this.contents.map(fn)
	}
}

function collect(documentName, entries, parent, index) {
	const collection = new StubCollection()
	for (const entry of entries ?? []) {
		const doc = new StubDocument(documentName, entry, parent)
		collection.set(doc.id, doc)
		index.set(doc.uuid, doc)
	}
	return collection
}
```

**(b)** 把 `	// --- AEA-STUB: world fixture ---------------------------------------------` 这一行整行替换成世界的组装：

```js
	// --- world fixture -------------------------------------------------------
	const world = options.world ?? {}
	const index = ctx.documents
	const actors = collect("Actor", world.actors, null, index)
	const tables = collect("RollTable", world.tables, null, index)
	for (const table of tables.contents) table.results = collect("TableResult", table.results, table, index)
	const journal = collect("JournalEntry", world.journal, null, index)
	const folders = collect("Folder", world.folders, null, index)
	const items = collect("Item", world.items, null, index)
	const macros = collect("Macro", world.macros, null, index)
	const users = collect("User", world.users, null, index)
	const scenes = collect("Scene", world.scenes, null, index)

	const allTokens = () => scenes.contents.flatMap((s) => s.tokens?.contents ?? [])

	for (const actor of actors.contents) {
		actor.items = collect("Item", actor.items, actor, index)
		actor.isToken = false
		actor.token = null
		// Real Foundry hands back Token placeables by default and TokenDocuments
		// only when asked; resolver code branches on exactly that difference.
		actor.getActiveTokens = (linked = false, asDocument = false) => {
			const docs = allTokens().filter((t) => t.parent?.active && t.actorId === actor.id)
			return asDocument ? docs : docs.map((t) => t.object)
		}
	}

	for (const scene of scenes.contents) {
		scene.tokens = collect("Token", scene.tokens, scene, index)
		for (const token of scene.tokens.contents) {
			const base = actors.get(token.actorId) ?? null
			if (token.actorLink || !base) {
				token.actor = base
			} else {
				// An unlinked token owns a synthetic copy of the base actor. Its uuid
				// is the token's own path plus the actor segment, which is the ONLY
				// thing that keeps three tokens of one base actor apart.
				const synthetic = Object.create(base)
				synthetic.isToken = true
				synthetic.token = token
				synthetic.name = token.name ?? base.name
				Object.defineProperty(synthetic, "uuid", { get: () => `${token.uuid}.Actor.${base.id}` })
				index.set(synthetic.uuid, synthetic)
				token.actor = synthetic
			}
			token.object = {
				id: token.id,
				name: token.name,
				document: token,
				get actor() {
					return token.actor
				},
			}
		}
	}
	ctx.world = { actors, items, tables, journal, folders, macros, users, scenes }

	const controlledTokens = (options.canvas?.controlled ?? [])
		.map((id) => allTokens().find((t) => t.id === id)?.object)
		.filter(Boolean)
```

**(c)** 把 `		// --- AEA-STUB: world collections ---` 这一行整行替换成六个世界集合：

```js
		actors,
		items,
		tables,
		journal,
		folders,
		macros,
		users,
		scenes,
```

**(d)** 把这一行

```js
	globalThis.canvas = { scene: null, tokens: { placeables: [], controlled: [], get: () => null } }
```

替换成按夹具填充的版本（没有夹具时它自然退化成上面那个空壳）：

```js
	globalThis.canvas = {
		scene: scenes.contents.find((s) => s.active) ?? null,
		tokens: {
			get placeables() {
				return allTokens().map((t) => t.object)
			},
			controlled: controlledTokens,
			get: (id) => allTokens().find((t) => t.id === id)?.object ?? null,
		},
	}
```

**(e)** 在 `globalThis.foundry` 的 `applications: {` 块里，`api: { ... },` 之后加一行 handlebars 命名空间（V14 把 `renderTemplate` 挪到了这里）：

```js
			handlebars: {
				renderTemplate: async (path, data) => {
					ctx.templates.push({ path, data })
					return `<div data-aea-template="${path}"></div>`
				},
			},
```

- [ ] **Step 20: 跑它，看它绿**

Run: `npx vitest run test/stub-world.test.mjs test/stub-fidelity.test.mjs`
Expected: PASS — 2 个测试文件，18 passed（world 6 + fidelity 12）。

- [ ] **Step 21: 提交**

```bash
git add -A && git commit -m "$(cat <<'EOF'
test(stubs): 桩补世界文档夹具、uuid 解析、canvas 与模板渲染

上一版桩只有全局，没有文档，于是每个下游任务都得自己造 game.actors / fromUuid /
RollTable —— 各造各的，一处形状猜错就是「只跟自己的假设一致」。这次一次补齐，
下游只声明 options.world，不再自己打补丁；没给 world 时八个集合是空集合而不是
undefined，取值路径与有夹具时完全一致。

关键的一条是非链接 token：它的 token.actor 是基础 actor 的合成副本，uuid 形如
Scene.<s>.Token.<t>.Actor.<a>，isToken 为 true、token 指回 TokenDocument。同一个
基础 actor 的两个非链接 token 因此有两个不同的 actor uuid —— 三只同源 Drone 的
压力不该一起涨，这个夹具就是那条判据的唯一来源。

getActiveTokens 默认回 Token 放置物、传 asDocument 才回 TokenDocument，与真实
Foundry 一致：解析 token 的代码正是在这个差别上分支的。

renderTemplate 挂在 foundry.applications.handlebars 下（V14 的位置），只记录
调用并回一段带模板路径的确定字符串，不引 handlebars，也不假装渲染。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 22: 写会失败的生命周期与锚点测试**

```js
// test/main-lifecycle.test.mjs
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs"

const source = () => readFileSync(fileURLToPath(new URL("../scripts/main.mjs", import.meta.url)), "utf8")

/** The api key set AND key order are frozen: every kernel gets exactly one slot. */
const KERNEL_SLOTS = [
	"features", "patches", "resolver", "registry", "rollBus", "diceBarrier", "cards", "selftest",
]

/**
 * Every later task inserts its wiring line next to one of these, quoting it
 * verbatim. The order below is the order they must appear in the file: the ready
 * sub-anchors encode the phase's execution order, so an insert can never land in
 * front of a step it depends on.
 */
const ANCHORS = [
	"/* AEA-ANCHOR: imports */",
	"/* AEA-ANCHOR: features */",
	"/* AEA-ANCHOR: repairs */",
	"/* AEA-ANCHOR: init */",
	"/* AEA-ANCHOR: i18nInit */",
	"/* AEA-ANCHOR: diceSoNiceReady */",
	"/* AEA-ANCHOR: ready.registry */",
	"/* AEA-ANCHOR: ready.patches */",
	"/* AEA-ANCHOR: ready.rollbus */",
	"/* AEA-ANCHOR: ready.cards */",
]

const LOOPS = [
	"for (const f of FEATURES) safely(`feature ${f.id} register`, () => f.register())",
	"for (const r of REPAIRS) safely(`repair ${r.id} register`, () => r.register())",
	"for (const f of FEATURES) await safely(`feature ${f.id} install`, () => f.install())",
	"for (const r of REPAIRS) await safely(`repair ${r.id} install`, () => r.install?.())",
]

describe("scripts/main.mjs", () => {
	let ctx

	beforeEach(() => {
		vi.resetModules()
		ctx = installFoundryStub()
	})

	afterEach(() => uninstallFoundryStub())

	const hook = (name) => ctx.hooks.once.find((h) => h.name === name)

	it("attaches exactly the four lifecycle hooks it owns, in boot order", async () => {
		await import("../scripts/main.mjs")
		expect(ctx.hooks.once.map((h) => h.name)).toEqual(["init", "i18nInit", "diceSoNiceReady", "ready"])
	})

	it("attaches no domain hooks — those belong to the kernel module that owns them", async () => {
		await import("../scripts/main.mjs")
		expect(ctx.hooks.on).toEqual([])
	})

	it("publishes the api object by reference during init", async () => {
		const mod = await import("../scripts/main.mjs")
		expect(game.modules.get("alien-evolved-automation").api).toBeUndefined()

		hook("init").fn()

		// Identity, not a copy: a kernel task fills its slot by editing the literal,
		// and anything already holding a reference must see the same object.
		expect(game.modules.get("alien-evolved-automation").api).toBe(mod.api)
		mod.api.features = { marker: true }
		expect(game.modules.get("alien-evolved-automation").api.features).toEqual({ marker: true })
	})

	it("carries exactly the eight kernel slots, in the contracted order, all null on delivery", async () => {
		const mod = await import("../scripts/main.mjs")
		expect(Object.keys(mod.api)).toEqual(KERNEL_SLOTS)
		expect(Object.values(mod.api)).toEqual(KERNEL_SLOTS.map(() => null))
	})

	it("runs the whole ready phase with both module tables still empty", async () => {
		await import("../scripts/main.mjs")
		await expect(hook("ready").fn()).resolves.toBeUndefined()
	})

	it("carries each of the ten wiring anchors verbatim, exactly once", () => {
		const text = source()
		for (const anchor of ANCHORS) {
			expect(text.split(anchor).length - 1, `${anchor} must appear exactly once`).toBe(1)
		}
		expect(text.split("AEA-ANCHOR").length - 1).toBe(ANCHORS.length)
	})

	it("keeps the ten anchors in the contracted order", () => {
		const text = source()
		const positions = ANCHORS.map((a) => text.indexOf(a))
		expect(positions).toEqual([...positions].sort((x, y) => x - y))
	})

	it("runs all four module-table loops through safely()", () => {
		const text = source()
		for (const loop of LOOPS) expect(text, `main.mjs must contain: ${loop}`).toContain(loop)
	})
})
```

`vi.resetModules()` + 动态 `import()` 是必需的：`main.mjs` 在模块顶层就调 `Hooks.once`，所以桩必须先装好，import 才能发生；而 ES module 有缓存，不 reset 的话第二个用例里 `main.mjs` 的顶层代码根本不会再跑一遍。

后三条断言是给后面十七个任务用的护栏。`FEATURES` / `REPAIRS` 是不导出的模块内常量（外部要看清单读 `api.features.all()`），所以循环本身没法在 Node 里注入假特性来跑；改用两条源文本断言把「四条循环都走 `safely()`」钉住，而 `safely()` 的隔离行为在下一组用例里被真测。这不是「假装测过」——它守的是「有人把 `safely` 从循环里拿掉」这个具体退化。

- [ ] **Step 23: 跑它，看它红**

Run: `npx vitest run test/main-lifecycle.test.mjs`
Expected: FAIL — 8 条用例全红。前五条报 `Error: Failed to load url ../scripts/main.mjs (resolved id: .../scripts/main.mjs). Does the file exist?`，后三条报 `Error: ENOENT: no such file or directory, open '.../scripts/main.mjs'`。

- [ ] **Step 24: 写 scripts/main.mjs（骨架、十条锚点、八槽 api、两张模块表）**

```js
// scripts/main.mjs
/**
 * Alien Evolved: Automation — module entry point.
 *
 * HOOK OWNERSHIP. A Hook is Foundry's global event bus: `Hooks.once(name, fn)`
 * runs fn the first time the named event fires. This module splits them in two:
 *
 *   - LIFECYCLE hooks are attached ONLY here, and only these four:
 *     init, i18nInit, diceSoNiceReady, ready. (Phase 1 uses no `setup` hook; if
 *     one is ever needed it gets its own anchor, added here and nowhere else.)
 *   - DOMAIN hooks (preCreateChatMessage, createChatMessage,
 *     diceSoNiceRollComplete, renderChatMessageHTML) are attached by the kernel
 *     module that owns them, inside its own init()/install(), exactly once each.
 *     Features never attach hooks at all.
 *
 * HOW TO WIRE SOMETHING IN. This file carries ten anchor comments, named
 * AEA-ANCHOR: imports / features / repairs / init / i18nInit / diceSoNiceReady /
 * ready.registry / ready.patches / ready.rollbus / ready.cards. Insert your line
 * by quoting the anchor text — never by line number — and only at YOUR anchor:
 *
 *   - an import goes on the line below the imports anchor;
 *   - a FEATURES / REPAIRS member goes on the line above its array's anchor, one
 *     per line, always with a trailing comma;
 *   - an init-phase call goes below the init anchor;
 *   - a ready-phase call goes below its own ready sub-anchor.
 *
 * The four ready sub-anchors are already in execution order, which is why they
 * exist: registry bindings must resolve before patches apply (a data repair reads
 * a bound table), patches must register with libWrapper before the roll bus does
 * (the dice-pool clamp is an inner wrapper, and the sink wrapper must observe
 * post-clamp dice), and the card mount must exist before any feature installs
 * into it. Inserting "somewhere after the ready anchor" is what reversed this
 * order last time; there is no longer a single ready anchor to insert after.
 *
 * Canonical phase contents, for reference while inserting:
 *   init            FEATURES/REPAIRS register loops (already here, and first:
 *                   settings and menus are built from definitions already on file)
 *                   features.registerSettings(); registry.registerSettings()
 *                   registry.declare() x13; patches.register() xN
 *                   selftest.register() xN; then publishApi() (already last)
 *   i18nInit        rollBus.setLabelIndex(buildLabelIndex(...))
 *   diceSoNiceReady diceBarrier.init()
 *   ready           waitForWorldSettled() (already here, first)
 *                   registry.resolveAll() -> patches.applyAll() ->
 *                   rollBus.install() -> cards.init()
 *                   then the two install loops (already here, last)
 *
 * diceSoNiceReady only fires when the Dice So Nice module is installed. Anything
 * that must exist in a world without it — a setting registration, for instance —
 * belongs at init, not there.
 *
 * Do not repeat an anchor's full comment form anywhere else in this file:
 * test/main-lifecycle.test.mjs asserts each appears exactly once, and that all
 * ten appear in the order listed above.
 */
import { MID, SYSTEM_ID } from "./const.mjs"
/* AEA-ANCHOR: imports */

/**
 * Public API surface, reachable at runtime as
 * game.modules.get("alien-evolved-automation").api
 *
 * The key set and key order are frozen at these eight kernel slots. The task that
 * creates a kernel file does exactly three things: adds its import below the
 * imports anchor, replaces its own `null` here with the imported object, and
 * inserts its call below its own ready sub-anchor. Never replace this object,
 * never add or remove a key — publishApi() hands out this exact reference, so a
 * task that rebuilds the object erases every other task's slot.
 */
export const api = {
	features: null,     // owner: the task that creates scripts/kernel/features.mjs
	patches: null,      // owner: the task that creates scripts/kernel/patches.mjs
	resolver: null,     // owner: the task that creates scripts/kernel/resolver.mjs
	registry: null,     // owner: the task that creates scripts/kernel/registry.mjs
	rollBus: null,      // owner: the task that creates scripts/kernel/rollbus.mjs
	diceBarrier: null,  // owner: the task that creates scripts/kernel/dice-barrier.mjs
	cards: null,        // owner: the task that creates scripts/kernel/cards.mjs
	selftest: null,     // owner: the task that creates scripts/kernel/selftest.mjs
}

/**
 * Every feature module, in install order. A feature task imports its module below
 * the imports anchor and appends the module object here; nothing else.
 * Shape: {id, register(), install()}. Not exported — read api.features.all().
 */
const FEATURES = [
	/* AEA-ANCHOR: features */
]

/**
 * Every repair package, in install order. Same shape as FEATURES except that
 * install() is optional — a pure data repair has nothing to install.
 */
const REPAIRS = [
	/* AEA-ANCHOR: repairs */
]

/**
 * Attach the api object to the module document. `game.modules` is Foundry's
 * collection of installed packages; hanging an `api` property on our own entry is
 * the conventional way one package exposes functions to another. This assigns the
 * reference, it does not copy: whatever the eight slots hold at any moment is
 * what a console user sees.
 */
function publishApi() {
	const mod = game.modules.get(MID)
	if (mod) mod.api = api
}

/**
 * Run one boot step in isolation. A single broken feature must not take the whole
 * module down with it: the rest of the automation is still worth having, and a
 * console error names the culprit precisely. Returns whatever fn returned, so an
 * async install() can still be awaited by the caller — register() must be
 * synchronous, install() may return a Promise.
 *
 * Exported so vitest can drive all three outcomes directly.
 */
export function safely(what, fn) {
	try {
		const outcome = fn()
		if (outcome && typeof outcome.then === "function") {
			return outcome.catch((error) => console.error(`${MID} | ${what} failed`, error))
		}
		return outcome
	} catch (error) {
		console.error(`${MID} | ${what} failed`, error)
		return undefined
	}
}

Hooks.once("init", () => {
	for (const f of FEATURES) safely(`feature ${f.id} register`, () => f.register())
	for (const r of REPAIRS) safely(`repair ${r.id} register`, () => r.register())
	/* AEA-ANCHOR: init */
	publishApi()
})

Hooks.once("i18nInit", () => {
	/* AEA-ANCHOR: i18nInit */
})

Hooks.once("diceSoNiceReady", () => {
	/* AEA-ANCHOR: diceSoNiceReady */
})

Hooks.once("ready", async () => {
	/* AEA-ANCHOR: ready.registry */
	/* AEA-ANCHOR: ready.patches */
	/* AEA-ANCHOR: ready.rollbus */
	/* AEA-ANCHOR: ready.cards */
	for (const f of FEATURES) await safely(`feature ${f.id} install`, () => f.install())
	for (const r of REPAIRS) await safely(`repair ${r.id} install`, () => r.install?.())
	console.log(`${MID} | ${game.i18n.localize("AEA.Boot.Ready")}`)
})
```

两点说明：

- 两条 `register()` 循环排在 init 锚点**之前**是有意的。特性档位设置与文档绑定菜单都要按已登记的定义生成条目；若设置注册跑在登记之前，那时一条定义都没有，所有特性档位的设置项永远注册不上，运行期查 `features.enabled(id)` 读未注册键会直接抛。
- `publishApi()` 排在 init 体的**最后一句**，也是有意的：它是唯一的发布点，只此一处、只做一次引用赋值。
- `SYSTEM_ID` 现在只 import 进来还没用到，下一组步骤的就绪守卫要读系统命名空间下的两个设置。

- [ ] **Step 25: 跑它，看它绿**

Run: `npx vitest run test/main-lifecycle.test.mjs`
Expected: PASS — 8 passed。

- [ ] **Step 26: 提交**

```bash
git add -A && git commit -m "$(cat <<'EOF'
feat(main): 四个生命周期钩子、十条锚点、八槽 api 与两张模块表

main.mjs 只挂生命周期钩子（init / i18nInit / diceSoNiceReady / ready），领域钩子
（preCreateChatMessage、createChatMessage、diceSoNiceRollComplete、
renderChatMessageHTML）由拥有它的内核模块在自己的 init()/install() 里各挂一次，
特性一律不挂钩子。一期不使用 setup 阶段，文件头写明了这一条。

锚点从四条扩到十条，ready 段拆成四条有序子锚点。上一轮只有一条 ready 锚点、四个
任务都说「插在它之后」，谁最后动手谁排最前，函数体会整个倒序：cards.init() 跑到
registry.resolveAll() 前面，骰池钳制补丁也不再是 rollBus 的内层包裹。现在顺序由
文件形状固定 —— registry -> patches -> rollbus -> cards，每人只往自己那条子锚点
下一行插一句。另外补了 imports / features / repairs 三条锚点：顶部 import 区与两张
数组本来没有定位文本，四个任务各引用了四种写法。

api 的键集与键序冻结为八个内核槽，本次交付全是 null —— 内核文件此刻还不存在，
import 它们会让整个模组加载失败。每个内核任务只做三件事：加自己的 import、把自己
那个 null 换成对象、在自己的 ready 子锚点后插调用。publishApi() 是唯一发布点、按
引用赋值、排在 init 体最后一句。

两条 register() 循环排在 init 锚点之前：设置与菜单要按已登记的定义生成，登记必须
在前，否则特性档位设置永远注册不上、运行期读未注册键直接抛。四条循环一律走
safely()，单条抛错只记 console.error 并继续，不让一条坏特性带走整个模组。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 27: 写会失败的隔离与就绪守卫测试**

```js
// test/main-boot-gate.test.mjs
import { describe, it, expect, afterEach, vi } from "vitest"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs"

async function load(options) {
	vi.resetModules()
	const ctx = installFoundryStub(options)
	const mod = await import("../scripts/main.mjs")
	return { ctx, mod }
}

describe("safely", () => {
	afterEach(() => uninstallFoundryStub())

	it("returns the value of a step that succeeds", async () => {
		const { mod } = await load()
		expect(mod.safely("step", () => 42)).toBe(42)
	})

	it("swallows a synchronous throw and names the step", async () => {
		const { mod } = await load()
		const error = vi.spyOn(console, "error").mockImplementation(() => {})
		expect(mod.safely('feature "boom" register', () => { throw new Error("nope") })).toBeUndefined()
		expect(error.mock.calls[0][0]).toContain('feature "boom" register failed')
		error.mockRestore()
	})

	it("swallows a rejected promise and names the step", async () => {
		const { mod } = await load()
		const error = vi.spyOn(console, "error").mockImplementation(() => {})
		await expect(mod.safely("repair r1 install", () => Promise.reject(new Error("nope")))).resolves.toBeUndefined()
		expect(error.mock.calls[0][0]).toContain("repair r1 install failed")
		error.mockRestore()
	})
})

describe("waitForWorldSettled", () => {
	afterEach(() => {
		vi.useRealTimers()
		uninstallFoundryStub()
	})

	it("settles immediately when Babele is absent and the system settings do not exist", async () => {
		const { mod } = await load()
		await expect(mod.waitForWorldSettled({ timeoutMs: 1000, pollMs: 10 })).resolves.toBe("settled")
	})

	it("waits while the system holds its semaphore at busy, then settles", async () => {
		const { mod } = await load()
		game.settings.register("alienrpg", "ARPGSemaphore", { scope: "world", config: false, type: String, default: "" })
		await game.settings.set("alienrpg", "ARPGSemaphore", "busy")
		vi.useFakeTimers()

		let outcome = null
		const pending = mod.waitForWorldSettled({ timeoutMs: 5000, pollMs: 100 }).then((r) => (outcome = r))

		await vi.advanceTimersByTimeAsync(300)
		expect(outcome).toBeNull()

		await game.settings.set("alienrpg", "ARPGSemaphore", "")
		await vi.advanceTimersByTimeAsync(200)
		await pending
		expect(outcome).toBe("settled")
	})

	it("waits for a fresh world's adventure import, which the system never awaits", async () => {
		const { mod } = await load()
		game.settings.register("alienrpg", "imported", { scope: "world", config: false, type: Boolean, default: false })
		vi.useFakeTimers()

		let outcome = null
		const pending = mod.waitForWorldSettled({ timeoutMs: 5000, pollMs: 100 }).then((r) => (outcome = r))
		await vi.advanceTimersByTimeAsync(300)
		expect(outcome).toBeNull()

		await game.settings.set("alienrpg", "imported", true)
		await vi.advanceTimersByTimeAsync(200)
		await pending
		expect(outcome).toBe("settled")
	})

	it("waits for Babele when it is installed, and settles when it reports initialized", async () => {
		const { ctx, mod } = await load({ modules: [{ id: "babele", active: true }], babele: { initialized: false } })
		vi.useFakeTimers()

		let outcome = null
		const pending = mod.waitForWorldSettled({ timeoutMs: 5000, pollMs: 100 }).then((r) => (outcome = r))
		await vi.advanceTimersByTimeAsync(300)
		expect(outcome).toBeNull()

		ctx.babele.initialized = true
		await vi.advanceTimersByTimeAsync(200)
		await pending
		expect(outcome).toBe("settled")
	})

	it("gives up loudly instead of hanging the ready hook forever", async () => {
		const { mod } = await load()
		game.settings.register("alienrpg", "ARPGSemaphore", { scope: "world", config: false, type: String, default: "" })
		await game.settings.set("alienrpg", "ARPGSemaphore", "busy")
		const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
		vi.useFakeTimers()

		const pending = mod.waitForWorldSettled({ timeoutMs: 500, pollMs: 100 })
		await vi.advanceTimersByTimeAsync(700)
		await expect(pending).resolves.toBe("timeout")
		expect(warn).toHaveBeenCalled()
		warn.mockRestore()
	})

	it("is the first statement of the ready hook, called with the contracted arguments", () => {
		const text = readFileSync(fileURLToPath(new URL("../scripts/main.mjs", import.meta.url)), "utf8")
		const call = "await waitForWorldSettled({ timeoutMs: 10000, pollMs: 100 })"
		expect(text).toContain(call)
		expect(text.indexOf(call)).toBeLessThan(text.indexOf("/* AEA-ANCHOR: ready.registry */"))
	})
})
```

最后一条是接线断言而不是行为断言：契约把这句的**参数值**也写死了（10000 / 100），而超时与轮询间隔一旦退回成看不见的实现默认值，评审就看不出这道闸等多久。它同时钉住「守卫在四条 ready 子锚点之前」——任何插在子锚点下的调用都必须发生在世界落定之后。

- [ ] **Step 28: 跑它，看它红**

Run: `npx vitest run test/main-boot-gate.test.mjs`
Expected: FAIL — 9 条用例全红。`safely` 那三条报 `TypeError: mod.safely is not a function`？不会——`safely` 已经导出，这三条应当**绿**；实际红的是 `waitForWorldSettled` 那六条，报 `TypeError: mod.waitForWorldSettled is not a function`，最后一条报 `AssertionError: expected '…' to contain 'await waitForWorldSettled({ timeoutMs: 10000, pollMs: 100 })'`。所以本步的预期是：**3 passed, 6 failed**。

- [ ] **Step 29: 给 main.mjs 加就绪守卫**

在 `scripts/main.mjs` 里做两处按原文匹配的改动，**不要按行号盲改**。

第一处：在 `export function safely(what, fn) { ... }` 这个函数之后、`Hooks.once("init", () => {` 之前插入：

```js
/** How long the ready gate waits before giving up and resolving anyway. */
const READY_GATE_TIMEOUT_MS = 10000
/** How often the gate re-reads the world's settle flags. */
const READY_GATE_POLL_MS = 100

function sleep(ms) {
	return new Promise((resolve) => setTimeout(resolve, ms))
}

/** Read a world setting that may not be registered at all. Never throws. */
function peekSetting(namespace, key) {
	try {
		return game.settings.get(namespace, key)
	} catch {
		return undefined
	}
}

/**
 * Babele is a translation module that rewrites document names at ready. Resolving
 * our document bindings before it has run would match against untranslated names
 * — or against nothing. Module ids sort "alien-evolved-automation" < "babele", so
 * our ready hook is registered first and fires first: hook order alone cannot fix
 * this, which is why the gate polls a state flag instead.
 */
function babeleSettled() {
	if (!game.modules?.get("babele")?.active) return true
	return game.babele?.initialized === true
}

/**
 * The alienrpg system imports its adventure pack (which is where every RollTable
 * this module binds to actually lives) from its own ready hook at
 * systems/alienrpg/module/apps/init.mjs:48. That hook calls the async
 * FirstTimeSetup() at :50 WITHOUT awaiting it, so on a fresh world the tables do
 * not exist yet at any ready-hook time no matter how the hooks are ordered.
 *
 * Two observable flags, both in the system's own namespace:
 *   - "ARPGSemaphore" is held at the string "busy" while the folder migration
 *     runs (systems/alienrpg/module/apps/migratefolders.js:12 sets it, :118 clears it)
 *   - "imported" flips to true when FirstTimeSetup finishes (init.mjs:78)
 * Either one being unreadable (setting not registered, system not loaded) counts
 * as settled — this gate delays us, it must never block us.
 */
function systemImportSettled() {
	if (peekSetting(SYSTEM_ID, "ARPGSemaphore") === "busy") return false
	const imported = peekSetting(SYSTEM_ID, "imported")
	return imported === undefined || imported === true
}

/**
 * Hold the ready phase until the world has stopped moving underneath us, or until
 * the deadline passes. Exported so it can be unit tested against fake timers; it
 * is deliberately NOT hung on `api`, whose eight slots belong to the kernels.
 *
 * This is the module's ONLY boot gate. No kernel may add a second wait of its own
 * — two gates with drifting criteria stack their timeouts on a fresh world.
 *
 * @returns {Promise<"settled"|"timeout">}
 */
export async function waitForWorldSettled({ timeoutMs = READY_GATE_TIMEOUT_MS, pollMs = READY_GATE_POLL_MS } = {}) {
	const deadline = Date.now() + timeoutMs
	while (!(babeleSettled() && systemImportSettled())) {
		if (Date.now() >= deadline) {
			console.warn(
				`${MID} | ready gate timed out after ${timeoutMs}ms waiting for Babele and the alienrpg adventure import; ` +
					`continuing anyway. Document bindings resolved now may be wrong.`
			)
			return "timeout"
		}
		await sleep(pollMs)
	}
	return "settled"
}
```

第二处，就是后续任务要反复做的那种锚点式插入：找到 `	/* AEA-ANCHOR: ready.registry */` 这一行，在它的**上一行**插入

```js
	await waitForWorldSettled({ timeoutMs: 10000, pollMs: 100 })
```

改完后 ready 的钩子体应当逐字是这样（十条锚点仍然各出现一次）：

```js
Hooks.once("ready", async () => {
	await waitForWorldSettled({ timeoutMs: 10000, pollMs: 100 })
	/* AEA-ANCHOR: ready.registry */
	/* AEA-ANCHOR: ready.patches */
	/* AEA-ANCHOR: ready.rollbus */
	/* AEA-ANCHOR: ready.cards */
	for (const f of FEATURES) await safely(`feature ${f.id} install`, () => f.install())
	for (const r of REPAIRS) await safely(`repair ${r.id} install`, () => r.install?.())
	console.log(`${MID} | ${game.i18n.localize("AEA.Boot.Ready")}`)
})
```

- [ ] **Step 30: 跑它，看它绿**

Run: `npx vitest run test/main-boot-gate.test.mjs test/main-lifecycle.test.mjs`
Expected: PASS — 2 个测试文件，17 passed（gate 9 + lifecycle 8）。

- [ ] **Step 31: 提交**

```bash
git add -A && git commit -m "$(cat <<'EOF'
feat(main): ready 阶段加唯一就绪守卫，等 Babele 与系统冒险导入落定

默认钩子顺序是错的，两条独立原因：
1. 模组 id 排序 "alien-evolved-automation" < "babele"，我们的 ready 先注册先触发，
   跑在 Babele 改名之前；
2. 系统自己的 ready 钩子（apps/init.mjs:48）在 :50 调 async 的 FirstTimeSetup()
   却不 await，全新世界里那些 RollTable 在任何 ready 时刻都还不存在。

所以守卫不能靠钩子排位，只能轮询状态位：Babele 用 game.babele.initialized（未装
即视为已落定），系统用它自己命名空间下的 ARPGSemaphore != "busy"（settings.mjs:209
注册、migratefolders.js:12 置 busy、:118 清空）与 imported（init.mjs:24 注册、
:78 置 true）。两者任一读不到（设置未注册）都算落定 —— 这道闸只许拖慢我们，不许
卡死我们。

它是全模组唯一的启动闸，注释里写死了这一条：任何内核不得再加第二道等待，两套
判据一旦漂移会在新世界里叠出双倍超时。超时 10 秒后 console.warn 并照常放行，绝不
让 ready 永远挂着。调用点逐字带参数 {timeoutMs: 10000, pollMs: 100}，免得等多久
退回成看不见的实现默认值；一条测试断言它排在四条 ready 子锚点之前。

导出为具名函数以便用假定时器真测四条路径，但不挂到 api 上：那八个槽是内核的。
safely 也导出，三条用例分别覆盖返回值、同步抛错、Promise 拒绝。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 32: 写会失败的横切纪律测试**

接口契约 §7 有四条「永不」：永不 `getName()` 比对显示名、永不 `game.actors.get(speaker.actor)`、永不解析渲染文本、永不读 `Roll#total`。这四条是全计划的横切纪律，但十七个任务里没有任何一个拥有它们的**自动执行**——本轮就是靠人工阅读才抓到一处漏网。这一步给它们一个会红的守卫。

```js
// test/discipline.test.mjs
import { describe, it, expect } from "vitest"
import { readdirSync, readFileSync, statSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { findViolations } from "./lib/discipline-scan.mjs"

const root = fileURLToPath(new URL("../", import.meta.url))

/**
 * Explicit, reviewed exceptions. Adding one REQUIRES editing this file, which is
 * the point: a discipline you can opt out of silently is not a discipline.
 * Shape: {file, id, reason}. Empty today — nothing under scripts/ needs an out.
 */
const ALLOWLIST = []

function walk(rel, out = []) {
	for (const entry of readdirSync(root + rel)) {
		const child = `${rel}/${entry}`
		if (statSync(root + child).isDirectory()) walk(child, out)
		else if (child.endsWith(".mjs")) out.push(child)
	}
	return out
}

const FIXTURE = [
	'// game.actors.get("commented") must not count',
	'const a = game.actors.get(speaker.actor)',
	'const t = game.tables.getName("Panic Table")',
	'const n = roll.total',
	'const s = element.innerText',
].join("\n")

describe("test/lib/discipline-scan.mjs", () => {
	it("flags each banned call in a fixture", () => {
		const ids = findViolations("fixture.mjs", FIXTURE).map((v) => v.id).sort()
		expect(ids).toEqual(["actorsGet", "domText", "getName", "rollTotal"])
	})

	it("ignores the same text inside comments", () => {
		const commented = "/* game.actors.get(x) roll.total */\n// element.innerText\nconst ok = 1"
		expect(findViolations("fixture.mjs", commented)).toEqual([])
	})
})

describe("contract §7 discipline across scripts/**/*.mjs", () => {
	it("finds no banned call", () => {
		const findings = []
		for (const file of walk("scripts")) {
			findings.push(...findViolations(file, readFileSync(root + file, "utf8"), ALLOWLIST))
		}
		expect(findings, `contract §7 violations:\n${JSON.stringify(findings, null, 2)}`).toEqual([])
	})

	it("has no stale allowlist entry", () => {
		for (const entry of ALLOWLIST) {
			const hits = findViolations(entry.file, readFileSync(root + entry.file, "utf8"))
			expect(
				hits.some((h) => h.id === entry.id),
				`${entry.file} no longer contains ${entry.id}; delete this allowlist entry`
			).toBe(true)
		}
	})
})
```

- [ ] **Step 33: 跑它，看它红**

Run: `npx vitest run test/discipline.test.mjs`
Expected: FAIL — `Error: Failed to load url ./lib/discipline-scan.mjs (resolved id: .../test/lib/discipline-scan.mjs). Does the file exist?`，4 条用例全不执行。

- [ ] **Step 34: 写 test/lib/discipline-scan.mjs**

```js
// test/lib/discipline-scan.mjs
/**
 * A tiny source scanner for the four cross-cutting "never do this" rules in
 * interface contract §7. It is not a JS parser: it blanks out comments, then
 * matches four literal call shapes line by line.
 *
 * Known and accepted limitation: text that merely LOOKS like a comment inside a
 * string literal (a "https://…" URL, say) gets blanked too. That can only cause a
 * missed finding, never a false one, so the scanner never blocks honest code.
 */

export const BANNED = [
	{
		id: "getName",
		pattern: "\\.getName\\s*\\(",
		why: "Look documents up through the registry by bound uuid; never compare display names.",
	},
	{
		id: "actorsGet",
		pattern: "game\\.actors\\.get\\s*\\(",
		why: "Resolve actors through the resolver; a bare actor id loses unlinked-token identity.",
	},
	{
		id: "rollTotal",
		pattern: "\\.total\\b",
		why: "Read successes from the RollRecord; the system's dice pool has no meaningful Roll#total.",
	},
	{
		id: "domText",
		pattern: "\\.(innerText|textContent)\\b",
		why: "Never parse rendered card text; the record is the source of truth.",
	},
]

/** Replace every comment with spaces, preserving line numbers and total length. */
export function stripComments(source) {
	return String(source)
		.replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, " "))
		.replace(/\/\/[^\n]*/g, (m) => " ".repeat(m.length))
}

/**
 * @param {string} file      path used in the report, e.g. "scripts/kernel/cards.mjs"
 * @param {string} source    file text
 * @param {Array}  allowlist [{file, id}] reviewed exceptions
 * @returns {Array} [{file, line, id, why}]
 */
export function findViolations(file, source, allowlist = []) {
	const lines = stripComments(source).split("\n")
	const found = []
	for (const rule of BANNED) {
		lines.forEach((line, i) => {
			if (new RegExp(rule.pattern).test(line)) found.push({ file, line: i + 1, id: rule.id, why: rule.why })
		})
	}
	return found.filter((v) => !allowlist.some((a) => a.file === v.file && a.id === v.id))
}
```

- [ ] **Step 35: 跑它，看它绿**

Run: `npx vitest run test/discipline.test.mjs`
Expected: PASS — 4 passed。`scripts/` 下此刻只有 `const.mjs` 与 `main.mjs`，两者都不含被禁调用（`main.mjs` 里的 `game.modules.get(...)` 与 `game.settings.get(...)` 都不在禁用表内），所以第三条用例是真绿而不是空跑。

- [ ] **Step 36: 跑全套并提交**

Run: `npm test`
Expected: PASS — 7 个测试文件，50 passed（const 2 + manifest 9 + stub-fidelity 12 + stub-world 6 + main-lifecycle 8 + main-boot-gate 9 + discipline 4）。

```bash
git add -A && git commit -m "$(cat <<'EOF'
test(discipline): 给契约 §7 的四条「永不」加自动守卫

四条横切纪律 —— 永不 getName() 比对显示名、永不 game.actors.get(speaker.actor)、
永不解析渲染文本、永不读 Roll#total —— 此前没有任何任务拥有它们的自动执行，
只能靠十七个任务各自自觉；本轮正是靠人工阅读才抓到一处漏网。

扫描器先把注释整块替换成等长空格（保留行号），再逐行匹配四种调用形状。它不是
JS 解析器，字符串里长得像注释的文本也会被抹掉 —— 这只会漏报不会误报，绝不挡住
正常代码。例外走 test/discipline.test.mjs 里的显式 ALLOWLIST，加例外必须改测试；
另有一条用例断言每条例外今天仍然命中，过期的例外会自己变红。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

- [ ] **MANUAL VERIFICATION（Task 1 收尾，必做）**

vitest 只证明代码在 Node 里能跑，证明不了 Foundry 认这份清单。桩再忠实也不是真实的 libWrapper 链、真实的语言包加载、真实的启动时序。请在本机 Foundry 里做完这七步：

1. 启动 Foundry，进入 **Configuration → Add-on Modules**，确认列表里出现 **Alien Evolved: Automation**，版本 0.1.0，且**没有**红色的 "Module not compatible" 横幅。
2. 打开一个 `alienrpg` 系统的世界 → **Game Settings → Manage Modules**，勾选本模组。预期：条目下方只列出 **lib-wrapper** 一条硬依赖（已装，故不弹「缺少依赖」对话框）；socketlib、dice-so-nice、yze-combat 不出现在硬依赖里。
3. 保存并重载后按 F12 打开控制台，预期看到一行 `alien-evolved-automation | Alien Evolved: Automation is ready.`（若界面语言设为中文，则是「异形进化版：自动化已就绪。」）。若打印的是键名 `AEA.Boot.Ready` 而不是译文，说明 `module.json` 的 `languages[].path` 写错，或 `lang` 代码不是 `cn`，或语言包嵌套结构写坏了。控制台里同时**不应**出现任何红色报错。
4. 在控制台执行
   ```js
   Object.keys(game.modules.get("alien-evolved-automation").api);
   ```
   预期**按这个顺序**返回八个键：`["features","patches","resolver","registry","rollBus","diceBarrier","cards","selftest"]`，且每个值此刻都是 `null`——内核尚未落地，这是正确状态，不是缺陷。若返回 `undefined`，说明 `init` 钩子里的 `publishApi()` 没跑到。
5. 验证就绪守卫在真实世界里不会拖慢启动：控制台执行
   ```js
   game.settings.get("alienrpg", "ARPGSemaphore");   // 预期 ""（不是 "busy"）
   game.settings.get("alienrpg", "imported");        // 预期 true（世界已导入过冒险包）
   ```
   两者如预期，说明第 3 步那行 ready 日志是**立刻**打出来的而不是等了 10 秒；控制台里也不应出现 `ready gate timed out` 警告。若真出现了该警告，先确认这两个设置的实际取值再改守卫，不要直接调大超时。
6. 验证十条锚点确实进了发布出去的文件（后面十七个任务全靠它们定位）。在模组目录执行：
   ```bash
   grep -c "AEA-ANCHOR" "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/main.mjs"
   grep -n "AEA-ANCHOR" "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/main.mjs"
   ```
   预期第一条打印 `10`；第二条打印出的十行顺序必须是 imports、features、repairs、init、i18nInit、diceSoNiceReady、ready.registry、ready.patches、ready.rollbus、ready.cards，且 `ready.registry` 那一行的**上一行**是 `await waitForWorldSettled({ timeoutMs: 10000, pollMs: 100 })`。
7. 验证 libWrapper 真实存在且本模组还没占用任何目标（后续内核要在这个前提上叠包裹）。控制台执行：
   ```js
   libWrapper.version;
   ```
   预期打印 `1.13.5.1` 一类的版本串而不是 `undefined`。这一步确认硬依赖真的被 Foundry 加载了——桩里的 `libWrapper` 是假的，只有这一眼能证明真货在。
