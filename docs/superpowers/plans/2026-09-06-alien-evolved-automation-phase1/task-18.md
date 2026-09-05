> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 18 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 18: 抽表路径 —— D66 十位修正、抽表工具与怪物卡文件夹守卫

> **分组理由**：三条修复是**同一条动作链**上的三段——「找到表 → 给一个修正 → 抽一次」。十位修正的算术是 enricher 骰子图标和 GM 抽表对话框共用的那段；文件夹解析是抽表对话框和怪物卡下拉共用的那段。分开做会写出两份不一致的实现。三条修复各自成文件（`<repair-id>.mjs` / `<repair-id>.pure.mjs` 同 stem），后两条 `import` 前面那条的引擎与纯函数。

**读者须知（本任务假定你不了解 Foundry VTT，也不了解异形 RPG）**：Foundry 是浏览器里的桌游平台；**世界（world）**是一份存档，里面有角色、随机表、宏等文档；**系统（system）**是规则实现（这里是 `alienrpg` 4.1.13）；**模组（module）**是外挂在系统之上的补充代码（我们写的就是模组）。本模组的原则：不改系统源码，只在运行时接管系统写错了的行为，每一条接管都带一个可运行的「缺陷还在不在」探针（`probe()`），上游哪天修好就自动退休；每一条接管都带一个 GM 可关的开关，关掉即刻恢复系统原行为、不需要重载世界。

**Files**（模组根目录 `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/`）:
- Create: `scripts/repairs/d66-roll-composer.pure.mjs`
- Create: `scripts/repairs/d66-roll-composer.mjs`
- Create: `scripts/repairs/table-draw-tools-and-macros.pure.mjs`
- Create: `scripts/repairs/table-draw-tools-and-macros.mjs`
- Create: `scripts/repairs/creature-table-folders.pure.mjs`
- Create: `scripts/repairs/creature-table-folders.mjs`
- Create: `test/repair-d66-roll-composer.test.mjs`
- Create: `test/repair-table-draw-tools.test.mjs`
- Create: `test/repair-creature-table-folders.test.mjs`
- Modify: `lang/en.json`、`lang/cn.json`（合并进已有的 `AEA` 对象，不新增顶层键）
- Modify: `scripts/main.mjs`（只在两个锚点注释后各插三行；该文件由更早的任务建立，一律**按锚点文本定位**，不按行号）

**Interfaces:**

- **Consumes**
  - `import { MID, SYSTEM_ID } from "../const.mjs";` —— `MID === "alien-evolved-automation"`，`SYSTEM_ID === "alienrpg"`。
  - `import { features } from "../kernel/features.mjs";` —— `features.register({id, default, gmOnly, requires})` 登记一个三档开关（`"full" | "prompt" | "off"`）；`features.enabled(id) -> boolean`，其定义是「`mode(id) !== "off"` 且 `requires` 链上没有 `"off"`」。**`gmOnly` 只表示这条开关是 GM 才能改的世界设置，它不改变 `enabled()` 对玩家的返回值**——所以凡是只该给 GM 看的界面，必须自己另外查 `game.user.isGM`。`features.mode()` 每次现读设置，改档立刻生效。
  - `import { patches } from "../kernel/patches.mjs";` —— `patches.register(def)`，`def = {id, type, target, minSystem, fixedIn, probe(), apply()}`。`probe()` **同步**返回 `true` 表示「缺陷仍在，该装」，`false` 表示「上游已修，退休」；`apply()` 是**无参安装器**（可以是 async），自己负责装监听器／挂钩子／改函数，`patches.applyAll()` 只调它。`patches.status()` 返回**全部已登记补丁**（含尚未 `applyAll()` 的），每条**恰好六个字段**：`{id, type, target, applied, reason, fixedIn}`；`reason` 只有七个取值：`"ok"`（已装）、`"disabled"`、`"version-not-applicable"`、`"probe-false"`、`"libwrapper-missing"`、`"error"`、`"pending"`（已登记、`applyAll()` 尚未跑到）。**`patches.register()` 不会自动登记任何自检条目**——自检一律显式登记。
  - `import { registry } from "../kernel/registry.mjs";` —— `registry.folder(key) -> Folder|null`。本任务用两个键：`"folderMotherTables"`、`"folderCreatureTables"`；它们由更早的任务 `declare()`，并在 `ready` 阶段的 `registry.resolveAll()` 里解析成 uuid 绑定。`resolveAll()` 排在 `patches.applyAll()` **之前**，而本任务只在「用户点开界面」时读绑定，所以绑定一定已就绪。
  - `import { selftest } from "../kernel/selftest.mjs";` —— `selftest.register({id, label, run})`。**`label` 是 i18n 键、不是已本地化文本**（登记发生在 `init`，那时语言包还没加载，`localize()` 只会回声键名）；`register` 原样保存 def，`runAll()` 才调 `game.i18n.localize(def.label)`。`run()` 返回 `{ok:boolean, detail:string}`，可以是 async。自检 id 的命名空间是 `repair.<修复id>.<方面>`。
  - 测试里 `import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";` —— 契约 §0.3 的共享桩，由建立仓库骨架的那个任务独占实现，**只读不改**；**禁止**在测试里自造 `globalThis.game` 或用私有 Map 顶替 `game.settings`。本任务依赖桩的这几条已定行为：`installFoundryStub()` 幂等（第二次先隐式卸载再装）；`game.settings` 由 `ctx.settings` 真支撑，`get` 读**未注册**键会抛 `Error`；`game.i18n` 有 `has`/`localize`/`format`，未知键回声键名，可用 `installFoundryStub({ i18n })` 注入词典；`game.release.generation` 默认 `14`、可用 `installFoundryStub({ generation })` 覆盖；`game.user = {id, isGM}` 且 `isGM` 可写；`ChatMessage.create` 把文档推进 `ctx.messages`；`ui.notifications.*` 记进 `ctx.notifications`（`[{type, message}]`）；`libWrapper.register` 记进 `ctx.wrappers`；**桩的 `Roll` 不做真随机，`total` 恒为 `null`**，所以任何需要真骰值的函数必须把掷骰做成可注入参数。
  - `scripts/main.mjs` 里由更早任务逐字写好的十个锚点注释，本任务只用到 `/* AEA-ANCHOR: imports */` 与 `const REPAIRS = [` 内的 `/* AEA-ANCHOR: repairs */`。`init` 段已有 `for (const r of REPAIRS) safely(..., () => r.register());`，`ready` 段已有 `for (const r of REPAIRS) await safely(..., () => r.install?.());`。

- **Produces**
  - `scripts/repairs/d66-roll-composer.pure.mjs`（**不得引用任何 Foundry 全局**，vitest 直接跑）
    - `export function pureIsD66Formula(formula) -> boolean`
    - `export function pureComposeD66({tens, ones, tensMod}) -> {total:number|null, noEncounter:boolean}`
    - `export function pureResolveDrawTotal(rows, total) -> number|null`
    - `export function pureUnreachableRows(formula, rows) -> Array<{resultId:string|null, range:number[]}>`
    - `export function pureDrawHandlerVerdict(source) -> "buggy"|"fixed"|"unknown"`
    - `export function pureDrawMessageOptions(generation, mode) -> {messageMode:string} | {rollMode:string} | {}`
    - `export function pureEscapeHtml(text) -> string`
  - `scripts/repairs/d66-roll-composer.mjs`
    - `export const REPAIR_ID = "d66-roll-composer";`
    - `export async function drawFromTable(table, {modifier, mode, count, roll, makeRoll}) -> Promise<void>`
    - `export const d66RollComposerRepair = { id, register() };`（**故意没有 `install()`**：安装由 `patches.applyAll()` 调 `apply()` 完成，`main.mjs` 的 `r.install?.()` 对它是空操作）
  - `scripts/repairs/table-draw-tools-and-macros.pure.mjs`
    - `export function pureMacroPackMissing(packs, systemId) -> boolean`
    - `export function pureMacroPlan(existing, specs) -> {create:string[], update:string[], kept:Array<{kind:string,name:string}>}`
    - `export function pureTablesInFolder(folder) -> object[]`
  - `scripts/repairs/table-draw-tools-and-macros.mjs`
    - `export const REPAIR_ID = "table-draw-tools-and-macros";`
    - `export function macroCommand(kind) -> string`
    - `export async function openDrawDialog(kind) -> Promise<void>`（`kind` 为 `"mother" | "creature"`）
    - `export const tableDrawToolsAndMacrosRepair = { id, register() };`（同样没有 `install()`）
  - `scripts/repairs/creature-table-folders.pure.mjs`
    - `export function pureTableChoices(tables, {prefix}) -> Record<string, {key:string, label:string}>`
    - `export function pureFolderLookupVerdict(source) -> "buggy"|"fixed"|"unknown"`
  - `scripts/repairs/creature-table-folders.mjs`
    - `export const REPAIR_ID = "creature-table-folders";`
    - `export const creatureTableFoldersRepair = { id, register() };`
  - 六条显式自检条目：`repair.d66-roll-composer.takeover`、`repair.d66-roll-composer.unreachableRows`、`repair.table-draw-tools-and-macros.tools`、`repair.table-draw-tools-and-macros.folderBindings`、`repair.table-draw-tools-and-macros.worldMacros`、`repair.creature-table-folders.override`。
  - i18n 键（全部嵌在唯一顶层键 `AEA` 之下）：`AEA.feature.<三个修复 id>.{name,hint}`、`AEA.tableDraw.*`、`AEA.selftest.repair.*`。

- **不接触的东西**（写清楚，免得两处都改互相打架）
  - **不往 `api` 上加任何键**。`scripts/main.mjs` 的 `export const api` 是八个内核成员的冻结面（`features / patches / resolver / registry / rollBus / diceBarrier / cards / selftest`），仓库里有一条护栏测试逐字断言它恰好这八个键。本任务三条修复都不出现在 `api` 上；世界宏改用**动态 import 本修复模块**进来（做法见 Step 24）。
  - **不自定义任何钩子名**。一期 `aea.*` 命名空间下只有 `HOOK_ROLL_RESOLVED` 一个钩子，它是掷骰总线的，与本任务无关。
  - **不包裹 `yzeRoll` / `abilityRoll` / `itemRoll` / `pushRoll` 这四个 libWrapper 目标**——它们由掷骰总线独占，第二次注册会被 lib-wrapper 拒绝并静默变成空操作。本任务一次 `libWrapper.register` 都不调。
  - `macros/gmRollYZEDiceMacro.js:34` 那条崩溃（GM 骰子器在压力骰出 1 时整张卡消失）根子在 `module/helpers/YZEDiceRoller.mjs:198-242`：宏没传 `actorid`，自动恐慌分支里 `game.actors.get(undefined)` 返回 `undefined`，下一行读 `.getRollData()` 就炸。那是一期五条特性里 `roll-pool-integrity` 的自动恐慌守卫要修的，**不要在这里再修一遍**。
  - **不做「按角色的压力 − 决心自动填修正」**。系统在 `module/documents/actor.mjs:867` 与 `:1094` 用 `roll.total + (aStress - resolve)` 算这个修正，而 `rollResolve` / `rollStress` 这两个方法由「压力与恐慌算术」那条修复整体接管。在本任务的对话框里再实现一遍，等于同一条规则有两份实现。本任务只提供**手填**的修正框。
  - `module/actor/old-rollTableData.js` 与 `old-actor-sheet.js` / `old-spacecraft-sheet.js` 是系统里的遗留副本，当前 sheet 注册链不经过它们，**不碰**。

---

**Foundry 术语（本任务真正用到的那几个）**

- **RollTable（随机表）**：一份文档，有一条掷骰式 `formula`（例如 `"1d6"`、`"10*1d6+1d6"`）和若干行 `results`，每行有 `range: [下界, 上界]`（闭区间）。`table.draw({roll})` 抽一行并发一张聊天卡。
- **`RollTable#draw` 一定会重掷你给的 Roll**。我逐行读过本机 Foundry V14 build 367 的 `client/documents/roll-table.mjs`：`draw()`（`:98`）在没有现成 `results` 时转调 `this.roll({roll})`；`roll()` 在 `:287-288` 先 `roll.reroll({minimize})` / `{maximize}` 做一次范围守卫，然后 `:302-309` 是
  `while (!results.length) { if (iter >= 10000) { ui.notifications.error("TABLE.DrawMaximumIterations"); break } roll = await roll.reroll(); results = this.getResultsForRoll(roll.total); iter++ }`。
  所以「抽到我算好的那一格」的唯一可靠做法是传一个**常量式** Roll（`new Roll("43")`）：常量式 reroll 之后还是 43，minimize/maximize 也是 43，守卫和循环都过得去。
- **`getResultsForRoll(value)`（`:341-342`）**：`this.results.filter(r => !r.drawn && Number.between(value, ...r.range))`——**已被抽掉的行（`drawn`）被排除在外**。这决定了一个关键防呆：若我们算出的点数落在**两行之间的空档**里（落在全部行的下界之下或上界之上会被 `:295` 的 `TABLE.NoPossibleResults` 守卫挡住并直接返回，落在空档里不会），而常量式 Roll 每次 reroll 都是同一个数，那个 `while` 就要空转到第 10000 次才报错跳出。所以我们**必须**在传进去之前，把点数对齐到一个仍可抽的行上。
- **enricher（富文本增强器）**：Foundry 允许把 `@DRAW[uuid]{名字}` 这样的标记渲染成可点的元素。系统在 `module/helpers/enricher.mjs:8-24`（`@DRAW`）与 `:26-41`（`@TEXTDRAW`）里把它渲染成 `<span class="draw-from-table" data-uuid="…">`（第三段存在时还带 `data-roll`），点击由 `:153` 的 `$(document).on("click", ".draw-from-table", drawFromRollableTable)` 接管。
- **事件捕获阶段（capture phase）**：`document.addEventListener("click", fn, true)` 里那个 `true` 让 `fn` 在事件**下行**到目标之前就执行，早于任何冒泡阶段的监听器。jQuery 的委托监听器（`$(document).on`）挂在冒泡阶段，所以在捕获阶段调 `stopImmediatePropagation()` 就能让它**永远收不到**这次点击。这是接管一个匿名 jQuery 委托的唯一干净办法。V14 仍以普通全局脚本加载 jQuery（`dist/server/express.mjs` 的 `CORE_VIEW_SCRIPTS` 第一项就是 `scripts/jquery.min.js`），所以 `jQuery._data` 也在。
- **DialogV2**：Foundry V13+ 的对话框。`foundry.applications.api.DialogV2.wait({window, content, buttons, rejectClose:false})` 返回 Promise。我读过 `client/applications/api/dialog.mjs`：`_onSubmit`（`:264-277`）取 `const result = (await button?.callback?.(event, target, this)) ?? button?.action`，`wait`（`:405-425`）把它 resolve 出来；直接关窗则 resolve 成 `null`。**每个按钮都必须有 `action`**；`callback` 的第二个参数是被点的 `<button>` 元素，`button.form` 就是表单。`new foundry.applications.ux.FormDataExtended(button.form).object` 把表单读成一个对象，`<input type="number">` 已经是 number，`<input type="checkbox">` 已经是 boolean。**注意**：`_renderHTML`（`:205-215`）自己就 `document.createElement("form")` 并把 `content` 放进去，所以 `content` 里**不要再套一层 `<form>`**——直接写 `<div class="form-group">` 即可。
- **Macro（宏）**：世界里的一段脚本文档，`macro.command` 是一段字符串代码，可以拖到快捷栏。执行时被包进 `new foundry.utils.AsyncFunction(...)`（`client/documents/macro.mjs:145-146`），所以宏正文里可以直接写 `await`。**世界里的宏是系统 `macros/*.js` 的冻结副本**——改系统源文件对已经存在的世界毫无作用。
- **`foundry.utils.getRoute(path)`**（`common/utils/helpers.mjs:698-705`）：把 `"modules/x/y.mjs"` 变成带路由前缀的绝对路径（`/modules/x/y.mjs`，装了 `ROUTE_PREFIX` 就是 `/前缀/modules/x/y.mjs`）。Foundry 的 express 路由（`dist/server/express.mjs`）正是把数据目录挂在这个前缀下，所以这条路径与浏览器加载 `main.mjs` 时用的 URL 同源同形。
- **ES 模块的同一性**：`import()` 一个**已经被加载过的 URL**，拿回的是**同一个模块实例**，不是新副本。本任务两处依赖这一点：世界宏 `import` 本修复模块（拿到的 `features` / `registry` 就是 `main.mjs` 那两个单例），以及第三条修复 `import` 系统的 `rollTableData.mjs`（拿到的类对象就是 `creature-sheet.mjs` 正在用的那个）。URL 必须逐字一致，所以一律用 `getRoute()` 拼。
- **`getSceneControlButtons` 钩子**：左侧场景工具栏渲染前发出（`client/applications/ui/scene-controls.mjs:392`）。参数是一个以控件组名为键的对象，每组有 `tools`（也是对象）。工具的字段见同文件 `:14-32` 的 `SceneControlTool` typedef：`{name, order, title, icon, visible, button, onChange}`，`title` 可以直接传 i18n 键。V14 里 token 那一组的键是 `"tokens"`（取自 `client/canvas/layers/tokens.mjs:158` 的 `layerOptions.name`），V13 是 `"token"`。

---

**你要修的事实（每一条我都逐行打开确认过；行号对应 `systems/alienrpg` 4.1.13 与本机 Foundry V14 build 367）**

1. `module/helpers/enricher.mjs:112` —— `const roll = formula ? new Roll(formula) : new Roll(\`${table.formula} + ${modifier}\`)`。表的式子是 `10*1d6+1d6`，于是修正被加到了**总数**上。而规则原文（核心书「战役玩法」章）是：**「Modify the tens digit roll for the factors below. A result of 0 or less indicates no encounter.」**，例子是「GM 掷出 3（十位）和 6（个位），十位 3 因为外缘 −3 降到 0，所以没有遭遇」；同章「General Colony Encounters」写着「**Add a +1 to the tens digit if the colony is established**」。修正加在十位、不是总数。加在总数上不只抽错行，还会让大量点数落到所有行的范围之外，被上面那个 while 循环默默重掷，GM 得到的行和他输入的修正毫无算术关系。
2. `module/helpers/enricher.mjs:101` 的 `drawFromRollableTable` 是模块内的局部 `async function`，只经 `:153` 的 `$(document).on(...)` 注册，**不是任何对象的属性**——libWrapper 只能包裹「某个对象/原型上的一个属性」，这里无 target 可用。所以只能用 document 捕获阶段接管。
3. `module/helpers/enricher.mjs:19-21` —— `@DRAW[uuid]{名字}{1d4}` 的第三段会写成 `data-roll` 属性，`:111-112` 于是用 `new Roll(data-roll)` 并**完全忽略修正**。我们的接管必须照样忽略它（而且更进一步：不弹那个没用的修正框）。
4. `module/helpers/enricher.mjs:113` 的 `table.draw({ roll })` **不传任何可见性选项**，所以图标抽出来的恐慌表结果玩家全看得见。
5. 出厂数据（`modules/alien-evolved-corerules`，我用脚本解出整包核对过）：RollTable `zB9xnHVp8ehurZZF`「EV - 20. STAR SYSTEM ENCOUNTERS」`formula: "10*1d6+1d6"`，37 行，其中 `BY8GnOLMElYEOn8B` 的 `range` 是 `[0, 10]`、正文是 `<p>None</p>`——**这个式子永远掷不出 ≤10 的数**，「无遭遇」这一行一次都没被抽到过。同包 `cR826F30mhD3LnDb`「EV - 23. GENERAL COLONY ENCOUNTERS」`formula: "10*1d6+1d6"`，42 行，除 11..66 外还有六行：`7IO6YSzbbZ3DHLX4`[71,71]「Starship crew off-duty」、`tHWs5vKxGpy9tUsr`[72,72]「Thugs」、`ngEBtSqfTIXXKDtD`[73,73]「Security patrol」、`n8sf1qimKKJwTh57`[74,74]「Colonial official with entourage」、`AdgYzcm7djuTMqdo`[75,75]「Accident in progress」、`zJeGMXgTSCzPj7Vm`[76,76]「Colonists on strike or protesting」——那正是「已建成殖民地十位 +1」才能到达的六行，同样一次都抽不到。这两处是「十位修正缺失」的**可运行证据**。
6. `system.json` 的 `packs` 数组里**只有一项**：`{"name":"alien-rpg-system","label":"Alien RPG System","type":"Adventure","module":"AlienRPG","system":"alienrpg"}`。**没有 Macro 类型的合集**，所以系统文档里说的「右键合集 → Import All Content」根本无从下手。
7. `macros/gmRollMotherTables.js:4` / `macros/gmRollCreatureTables.js:5` 按**文件夹显示名**筛表（`t.folder.name === 'Alien Mother Tables'` / `'Alien Creature Tables'`）。汉化层或 GM 手动改名之后，下拉就是空的。
8. 两个宏的 `:25` 判的都是 `game.tables.size > 0`——**全世界有没有表**，而不是这个下拉里有没有东西。于是空下拉照样给出 Draw 按钮，点下去 `game.tables.get("")` 得到 `undefined`，`:34` 的 `table.formula` 抛 TypeError。
9. 两个宏的 `:39` 是 `roll.evaluate({ async: false })`——我查过本机 V14 的 `client/dice/roll.mjs`，`evaluate` 只有 async 版本，`async:false` 这个选项早已不存在，这行返回一个没人 await 的 Promise，`:40` 的 `table.draw` 拿到的是一个未求值的 Roll。两个宏还都用 `new Dialog(...)`，而 `Dialog` 在 `client/client.mjs` 里标着 `@deprecated since v13 until v16`。
10. **V14 的两处 API 改名**（本任务要同时兼容 V13 与 V14，模组清单写的是 minimum 13 / verified 14）：`RollTable#draw` 的 `rollMode` 选项在 `client/documents/roll-table.mjs:134-137` 被判为「deprecated in favor of the `messageMode` option」（since 14, until 16），V14 取值是 `"public"|"gm"|"blind"|"self"`，V13 的旧取值是 `"publicroll"|"gmroll"|"blindroll"|"selfroll"`（映射见 `client/dice/roll.mjs:1155-1158` 的 `_mapLegacyRollMode`）。所以本任务**不调**弃用面，改为按 `game.release.generation` 选项名。
11. `module/helpers/rollTableData.mjs:7` 与 `:24` —— `game.folders.contents.find((x) => x.name === "Alien Creature Tables")` 与 `… === "Alien Mother Tables"`，紧接着 `:9` / `:27` 就 `folder.contents`。**文件夹缺失（或被改名、被汉化）时 `folder` 是 `undefined`，`folder.contents` 直接抛 TypeError**，而这两个静态方法的调用点是 `module/sheets/creature-sheet.mjs:129-130`（`context.rTables = alienrpgrTableGet.rTableget()` / `context.cTables = alienrpgrTableGet.cTableget()`）在 `_prepareContext` 里——**怪物卡因此整张打不开**。这两处消费的正是 `folderCreatureTables` / `folderMotherTables` 两个 registry 键，所以归本任务收口，成为第三条修复 `creature-table-folders`。注意两个方法的命名是反的：`rTableget()` 读的是**创造物**文件夹（无过滤），`cTableget()` 读的是 **Mother** 文件夹并过滤 `name.startsWith("Critical Injuries")`。
12. `alienrpgrTableGet` **没有**挂在 `game.alienrpg` 上（`module/alienrpg.mjs:74-84` 只暴露了 `alienrpgActor / alienrpgItem / ActiveEffect / ActorSheets / yze / registerSettings / rollItemMacro / ModuleImport / ImportFormWrapper`），所以 libWrapper 同样无 target 可用。但 `creature-sheet.mjs:4` 是 `import { alienrpgrTableGet } from "../helpers/rollTableData.mjs"` 且 `:129` 是**属性访问**（不是解构），所以只要拿到那个类对象、换掉它的两个静态方法，调用点立刻走我们的实现。拿到类对象的办法就是上面说的「同 URL 动态 import 拿到同一模块实例」。

---

- [ ] **Step 1: 写第一个失败测试 —— D66 十位算术与行对齐**

新建 `test/repair-d66-roll-composer.test.mjs`：

```js
import { describe, expect, it } from "vitest";
import {
  pureComposeD66,
  pureIsD66Formula,
  pureResolveDrawTotal,
  pureUnreachableRows,
} from "../scripts/repairs/d66-roll-composer.pure.mjs";

describe("pureIsD66Formula", () => {
  it("recognises the two spellings a d66 table can carry", () => {
    expect(pureIsD66Formula("10*1d6+1d6")).toBe(true);
    expect(pureIsD66Formula(" 10 * 1D6 + 1d6 ")).toBe(true);
    expect(pureIsD66Formula("1d66")).toBe(true);
  });

  it("rejects everything else", () => {
    expect(pureIsD66Formula("1d6")).toBe(false);
    expect(pureIsD66Formula("2d6")).toBe(false);
    expect(pureIsD66Formula("")).toBe(false);
    expect(pureIsD66Formula(undefined)).toBe(false);
  });
});

describe("pureComposeD66", () => {
  it("applies the modifier to the TENS digit, not to the total", () => {
    // Outer Rim is -3. Tens 5, ones 4 => tens becomes 2 => 24, not 54-3=51.
    expect(pureComposeD66({ tens: 5, ones: 4, tensMod: -3 })).toEqual({ total: 24, noEncounter: false });
  });

  it("reaches rows 71-76, which 10*1d6+1d6 can never produce", () => {
    expect(pureComposeD66({ tens: 6, ones: 3, tensMod: 1 })).toEqual({ total: 73, noEncounter: false });
  });

  it("matches the worked example in the rules: tens 3 minus 3 is no encounter", () => {
    // "The GM rolls a 3 (tens digit) and a 6 (ones digit), the tens digit (3) is
    // reduced to 0 because of the Outer Rim modifier and so there is no encounter."
    // The total keeps the ones digit so a table carrying a [0,10] "None" row can
    // still be drawn from normally.
    expect(pureComposeD66({ tens: 3, ones: 6, tensMod: -3 })).toEqual({ total: 6, noEncounter: true });
  });

  it("clamps a tens digit driven below zero rather than producing a negative total", () => {
    expect(pureComposeD66({ tens: 1, ones: 1, tensMod: -5 })).toEqual({ total: 1, noEncounter: true });
  });

  it("is the identity when the modifier is 0", () => {
    expect(pureComposeD66({ tens: 4, ones: 6, tensMod: 0 })).toEqual({ total: 46, noEncounter: false });
    expect(pureComposeD66({ tens: 1, ones: 1, tensMod: 0 })).toEqual({ total: 11, noEncounter: false });
  });

  it("refuses dice outside 1..6", () => {
    expect(pureComposeD66({ tens: 0, ones: 3, tensMod: 0 })).toEqual({ total: null, noEncounter: false });
    expect(pureComposeD66({ tens: 3, ones: 7, tensMod: 0 })).toEqual({ total: null, noEncounter: false });
    expect(pureComposeD66({})).toEqual({ total: null, noEncounter: false });
  });
});

describe("pureResolveDrawTotal", () => {
  const rows = [
    { resultId: "a", range: [11, 16] },
    { resultId: "b", range: [31, 36] },
  ];

  it("leaves a total that a row already covers alone", () => {
    expect(pureResolveDrawTotal(rows, 14)).toBe(14);
    expect(pureResolveDrawTotal(rows, 36)).toBe(36);
  });

  it("snaps a total below every row up to the lowest bound", () => {
    expect(pureResolveDrawTotal(rows, 5)).toBe(11);
  });

  it("snaps a total above every row down to the highest bound", () => {
    expect(pureResolveDrawTotal(rows, 76)).toBe(36);
  });

  it("snaps a total in a gap to the nearer side of the gap", () => {
    expect(pureResolveDrawTotal(rows, 20)).toBe(16);
    expect(pureResolveDrawTotal(rows, 25)).toBe(31);
  });

  it("returns null when there is nothing left to draw", () => {
    expect(pureResolveDrawTotal([], 14)).toBe(null);
    expect(pureResolveDrawTotal([{ resultId: "junk", range: [] }], 14)).toBe(null);
    expect(pureResolveDrawTotal(rows, Number.NaN)).toBe(null);
  });
});

describe("pureUnreachableRows", () => {
  it("flags EV-20's no-encounter row, which 10*1d6+1d6 can never produce", () => {
    const rows = [
      { resultId: "BY8GnOLMElYEOn8B", range: [0, 10] },
      { resultId: "aky2prjY4yMPTOGh", range: [12, 12] },
    ];
    expect(pureUnreachableRows("10*1d6+1d6", rows)).toEqual([
      { resultId: "BY8GnOLMElYEOn8B", range: [0, 10] },
    ]);
  });

  it("flags EV-23's rows 71-76", () => {
    const rows = [
      { resultId: "x", range: [66, 66] },
      { resultId: "7IO6YSzbbZ3DHLX4", range: [71, 71] },
      { resultId: "zJeGMXgTSCzPj7Vm", range: [76, 76] },
    ];
    expect(pureUnreachableRows("10*1d6+1d6", rows).map((r) => r.resultId)).toEqual([
      "7IO6YSzbbZ3DHLX4",
      "zJeGMXgTSCzPj7Vm",
    ]);
  });

  it("says nothing about a table whose rows all sit inside 11..66", () => {
    const rows = [
      { resultId: "x", range: [11, 16] },
      { resultId: "y", range: [61, 66] },
    ];
    expect(pureUnreachableRows("10*1d6+1d6", rows)).toEqual([]);
  });

  it("only judges the shipped spelling, because 1d66 is a 66-sided die and reaches 1..66", () => {
    const rows = [{ resultId: "x", range: [0, 10] }];
    expect(pureUnreachableRows("1d66", rows)).toEqual([]);
    expect(pureUnreachableRows("2d6", rows)).toEqual([]);
  });
});
```

- [ ] **Step 2: 跑一遍，看它失败**

Run: `npx vitest run test/repair-d66-roll-composer.test.mjs`
Expected: FAIL —— 整个文件在加载期就报 `Failed to load url ../scripts/repairs/d66-roll-composer.pure.mjs`，17 条用例一条都不执行。

- [ ] **Step 3: 写出四个纯函数**

新建 `scripts/repairs/d66-roll-composer.pure.mjs`。契约 §0.1：本文件**不得**出现 `game` / `ui` / `Hooks` / `CONFIG` / `Roll` / `foundry` 之类的 Foundry 全局，vitest 直接跑它。

```js
/**
 * Pure layer of the `d66-roll-composer` repair.
 *
 * A d66 result is two d6 read as a two-digit number, 11..66. The shipped tables
 * express that as "10*1d6+1d6". The rules modify the TENS DIGIT, never the total:
 *   "Modify the tens digit roll for the factors below. A result of 0 or less
 *    indicates no encounter." (Outer Rim/Frontier -3, Uncharted Space -5)
 *   "Add a +1 to the tens digit if the colony is established."
 * The system instead builds `${table.formula} + ${modifier}`
 * (module/helpers/enricher.mjs:112), which picks the wrong row AND throws away
 * roughly half of all modified rolls: RollTable#roll silently re-rolls until the
 * total lands inside some row's range (client/documents/roll-table.mjs:302-309).
 */

/** Strip whitespace and case so the two shipped spellings compare cleanly. */
function normalize(formula) {
  return String(formula ?? "").replaceAll(" ", "").toLowerCase();
}

/**
 * @param {string} formula
 * @returns {boolean} whether this table should be read as two d6 digits
 */
export function pureIsD66Formula(formula) {
  const f = normalize(formula);
  return f === "10*1d6+1d6" || f === "1d66";
}

/**
 * Compose a d66 result with the modifier on the tens digit.
 *
 * A modified tens digit of 0 or less means "no encounter". We still return a
 * total (the bare ones digit, 1..6) because EV-20 ships a [0,10] "None" row
 * precisely for that outcome — the caller draws it from the table when a row
 * catches the total, and posts its own line when none does.
 *
 * @param {{tens:number, ones:number, tensMod:number}} input
 * @returns {{total:number|null, noEncounter:boolean}} total is null for invalid dice
 */
export function pureComposeD66({ tens, ones, tensMod } = {}) {
  const isDie = (n) => Number.isInteger(n) && n >= 1 && n <= 6;
  const t = Number(tens);
  const o = Number(ones);
  if (!isDie(t) || !isDie(o)) return { total: null, noEncounter: false };

  const modified = t + (Number(tensMod) || 0);
  if (modified <= 0) return { total: o, noEncounter: true };
  return { total: modified * 10 + o, noEncounter: false };
}

/**
 * Move a total onto a row that can actually be drawn.
 *
 * We hand RollTable#draw a CONSTANT-formula Roll so the row we computed is the
 * row that comes out. The price is that `roll.reroll()` inside
 * client/documents/roll-table.mjs:302-309 keeps producing the same number, so a
 * total that no available row covers makes that loop spin until its 10000-iteration
 * bail-out. Snapping first is what makes the constant-Roll technique safe.
 *
 * `rows` must already exclude rows marked drawn, mirroring
 * RollTable#getResultsForRoll (client/documents/roll-table.mjs:341-342).
 *
 * @param {Array<{resultId:string|null, range:number[]}>} rows
 * @param {number} total
 * @returns {number|null} null when no row is usable at all
 */
export function pureResolveDrawTotal(rows, total) {
  const usable = (rows ?? []).filter(
    (r) => Array.isArray(r?.range) && Number.isFinite(r.range[0]) && Number.isFinite(r.range[1])
  );
  if (!usable.length || !Number.isFinite(total)) return null;

  for (const row of usable) {
    const lo = Math.min(row.range[0], row.range[1]);
    const hi = Math.max(row.range[0], row.range[1]);
    if (total >= lo && total <= hi) return total;
  }

  // Nothing covers it: take the closest bound of the closest row. Ties go to the
  // row listed first, so the choice is deterministic.
  let best = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const row of usable) {
    const lo = Math.min(row.range[0], row.range[1]);
    const hi = Math.max(row.range[0], row.range[1]);
    const candidate = total < lo ? lo : hi;
    const distance = Math.abs(total - candidate);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = candidate;
    }
  }
  return best;
}

/**
 * Rows a shipped d66 table carries that its own formula can never reach.
 *
 * This is the runnable evidence that the tens-digit modifier is missing:
 * EV-20 "STAR SYSTEM ENCOUNTERS" (zB9xnHVp8ehurZZF) carries a [0,10] "None" row
 * and EV-23 "GENERAL COLONY ENCOUNTERS" (cR826F30mhD3LnDb) carries rows 71..76,
 * while 10*1d6+1d6 only ever produces 11..66.
 *
 * Judged only for the shipped spelling: "1d66" is a 66-sided die in Foundry and
 * really does reach 1..66, so it is a different defect and not ours to report.
 *
 * @param {string} formula
 * @param {Array<{resultId:string|null, range:number[]}>} rows
 * @returns {Array<{resultId:string|null, range:number[]}>}
 */
export function pureUnreachableRows(formula, rows) {
  if (normalize(formula) !== "10*1d6+1d6") return [];
  const out = [];
  for (const row of rows ?? []) {
    const [lo, hi] = row?.range ?? [];
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) continue;
    if (hi < 11 || lo > 66) out.push({ resultId: row.resultId ?? null, range: [lo, hi] });
  }
  return out;
}
```

- [ ] **Step 4: 跑一遍，看它通过**

Run: `npx vitest run test/repair-d66-roll-composer.test.mjs`
Expected: PASS —— 17 passed。

- [ ] **Step 5: 提交**

```bash
git add scripts/repairs/d66-roll-composer.pure.mjs test/repair-d66-roll-composer.test.mjs && git commit -m "$(cat <<'EOF'
fix(d66-roll-composer): 修正加在十位而不是总数上（纯层）

enricher.mjs:112 拼的是 `${table.formula} + ${modifier}`，把修正加到了总数上。
规则原文是「Modify the tens digit roll... A result of 0 or less indicates no
encounter」，书里的例子就是十位 3 减 3 变 0、没有遭遇。加在总数上除了抽错行，
还会让大量点数落在所有行的范围之外，被 roll-table.mjs:302-309 的 while 循环默默重掷。

pureUnreachableRows 把「缺陷还在不在」变成可运行的断言：corerules 的
zB9xnHVp8ehurZZF（EV-20）有一行 [0,10]「None」，cR826F30mhD3LnDb（EV-23）有
71-76 六行，而 10*1d6+1d6 只掷得出 11..66，这些行至今一次都没被抽到过。

pureResolveDrawTotal 是常量式 Roll 能安全使用的前提：常量 reroll 之后还是同一个数，
点数一旦落在两行之间的空档里，roll-table.mjs 的 while 要空转到第一万次才跳出。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: 写第二个失败测试 —— 探针判定、消息选项与转义**

追加到 `test/repair-d66-roll-composer.test.mjs` 末尾（新增的三个名字加到文件顶部已有的那条 import 里，或者另写一条 import，两者都行）：

```js
import {
  pureDrawHandlerVerdict,
  pureDrawMessageOptions,
  pureEscapeHtml,
} from "../scripts/repairs/d66-roll-composer.pure.mjs";

// Verbatim body of alienrpg 4.1.13's module/helpers/enricher.mjs:101-113.
const SHIPPED_HANDLER = `async function drawFromRollableTable(event) {
  event.preventDefault()
  const uuid = event.currentTarget.getAttribute("data-uuid")
  if (!uuid) { return }
  const table = await fromUuid(uuid)
  const myF = async (uuid, modifier) => {
    if (table instanceof RollTable) {
      const formula = event.currentTarget.getAttribute("data-roll")
      const roll = formula ? new Roll(formula) : new Roll(\`\${table.formula} + \${modifier}\`)
      await table.draw({ roll })
    }
  }
}`;

// What a fixed upstream handler would look like: the modifier reaches the tens
// digit and the total handed to draw() is already decided.
const FIXED_HANDLER = `async function drawFromRollableTable(event) {
  const table = await fromUuid(event.currentTarget.getAttribute("data-uuid"))
  if (table instanceof RollTable) {
    const total = composeD66(tens + modifier, ones)
    await table.draw({ roll: await new Roll(String(total)).evaluate() })
  }
}`;

describe("pureDrawHandlerVerdict", () => {
  it("calls the shipped 4.1.13 handler buggy", () => {
    expect(pureDrawHandlerVerdict(SHIPPED_HANDLER)).toBe("buggy");
  });

  it("calls a handler that no longer adds the modifier to the formula fixed", () => {
    expect(pureDrawHandlerVerdict(FIXED_HANDLER)).toBe("fixed");
  });

  it("says unknown when there is no handler source to read", () => {
    expect(pureDrawHandlerVerdict(null)).toBe("unknown");
    expect(pureDrawHandlerVerdict("")).toBe("unknown");
    expect(pureDrawHandlerVerdict("function unrelated(){ return 1 }")).toBe("unknown");
  });
});

describe("pureDrawMessageOptions", () => {
  it("uses the v14 messageMode vocabulary on a v14 client", () => {
    expect(pureDrawMessageOptions(14, "gm")).toEqual({ messageMode: "gm" });
    expect(pureDrawMessageOptions(15, "public")).toEqual({ messageMode: "public" });
  });

  it("uses the legacy rollMode vocabulary on a v13 client", () => {
    expect(pureDrawMessageOptions(13, "gm")).toEqual({ rollMode: "gmroll" });
    expect(pureDrawMessageOptions(13, "public")).toEqual({ rollMode: "publicroll" });
  });

  it("passes nothing at all when no mode was chosen, so Foundry uses the world's", () => {
    expect(pureDrawMessageOptions(14, null)).toEqual({});
    expect(pureDrawMessageOptions(undefined, null)).toEqual({});
  });
});

describe("pureEscapeHtml", () => {
  it("makes a table name safe to interpolate into markup", () => {
    expect(pureEscapeHtml(`Ash & <b>"Bishop"</b>`)).toBe("Ash &amp; &lt;b&gt;&quot;Bishop&quot;&lt;/b&gt;");
    expect(pureEscapeHtml(undefined)).toBe("");
  });
});
```

- [ ] **Step 7: 跑一遍，看它失败**

Run: `npx vitest run test/repair-d66-roll-composer.test.mjs`
Expected: FAIL —— `SyntaxError: The requested module '../scripts/repairs/d66-roll-composer.pure.mjs' does not provide an export named 'pureDrawHandlerVerdict'`，整个文件加载失败。

- [ ] **Step 8: 补上这三个纯函数**

追加到 `scripts/repairs/d66-roll-composer.pure.mjs`：

```js
/**
 * The composition we are here to replace, written whitespace-free so a
 * reformat upstream does not fool it. `drawFromRollableTable` is a module-local
 * function, so the only way to read it is Function.prototype.toString() on the
 * handler jQuery stored when the system called
 * $(document).on("click", ".draw-from-table", ...) at
 * module/helpers/enricher.mjs:153.
 */
const BUGGY_MARKERS = [
  "${table.formula}+${modifier}",
  'table.formula+"+"+modifier',
  "table.formula+'+'+modifier",
];

/**
 * Decide, from the system's own handler source, whether the defect is still there.
 *
 * Three-valued on purpose. "unknown" means we could not read a handler at all —
 * jQuery gone, or upstream moved to addEventListener. The caller installs on
 * "unknown" as well as on "buggy", because our takeover is a complete
 * replacement for the action rather than a partial correction, so running it
 * against an unreadable handler is safe; retiring on an unreadable handler would
 * silently drop the fix. The selftest entry reports the verdict verbatim so the
 * situation is visible rather than guessed at.
 *
 * @param {string|null} source
 * @returns {"buggy"|"fixed"|"unknown"}
 */
export function pureDrawHandlerVerdict(source) {
  if (typeof source !== "string" || source.trim() === "") return "unknown";
  const flat = source.replace(/\s+/g, "");
  if (BUGGY_MARKERS.some((marker) => flat.includes(marker))) return "buggy";
  if (flat.includes("RollTable") || flat.includes(".draw(")) return "fixed";
  return "unknown";
}

/** v14 message modes, and the legacy roll modes v13 still expects. */
const LEGACY_MODE = { public: "publicroll", gm: "gmroll", blind: "blindroll", self: "selfroll" };

/**
 * Pick the option name RollTable#draw wants on this client.
 *
 * V14 deprecated the `rollMode` option in favour of `messageMode`
 * (client/documents/roll-table.mjs:134-137, "since 14, until 16"); passing the
 * old name still works but logs a compatibility warning on every draw. V13 only
 * knows the old name. Passing neither lets Foundry apply the world's current
 * setting, which is what an unmodified click should do.
 *
 * @param {number|undefined} generation  game.release.generation
 * @param {"public"|"gm"|"blind"|"self"|null} mode
 * @returns {{messageMode:string}|{rollMode:string}|{}}
 */
export function pureDrawMessageOptions(generation, mode) {
  if (!mode) return {};
  if ((Number(generation) || 0) >= 14) return { messageMode: mode };
  return { rollMode: LEGACY_MODE[mode] ?? "publicroll" };
}

/**
 * Escape a document name for interpolation into markup. Table names are authored
 * by whoever owns the world, so they are not hostile input, but they do contain
 * ampersands and quotes often enough to break an <option> label.
 * Written here rather than taken from foundry.utils so the draw engine stays
 * testable outside Foundry.
 */
export function pureEscapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
```

- [ ] **Step 9: 跑一遍，看它通过**

Run: `npx vitest run test/repair-d66-roll-composer.test.mjs`
Expected: PASS —— 24 passed。

- [ ] **Step 10: 提交**

```bash
git add scripts/repairs/d66-roll-composer.pure.mjs test/repair-d66-roll-composer.test.mjs && git commit -m "$(cat <<'EOF'
feat(d66-roll-composer): 探针判定改成可注入的纯谓词，并把 V14 的选项改名收进纯层

probe 不再去扫世界里的表（那会在没装 corerules 的世界里直接判「上游已修」，
把修复静默退掉），改为读系统自己那个 jQuery 委托处理器的源码字符串，交给
pureDrawHandlerVerdict 判定，两个方向都有夹具：4.1.13 的原文 -> buggy，
修好之后的写法 -> fixed，读不到 -> unknown（照装，并在自检里如实报出）。

pureDrawMessageOptions 把 V14 的 messageMode / V13 的 rollMode 这条改名收在一处：
roll-table.mjs:134-137 判了 rollMode 弃用，本模组这个弃用面一次都不碰。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 11: 写第三个失败测试 —— 抽表引擎与登记（用共享桩）**

追加到 `test/repair-d66-roll-composer.test.mjs` 末尾。

关于这一段测试的诚实性，三句话：(1) `RollTable` 是一份 Foundry 文档，桩里没有、也不该有；我们传进去的是一个**参数位置上的测试替身**（一个普通对象），不是伪造的全局——契约禁止的是各任务自造 Foundry 全局。(2) 桩的 `Roll` 不做真随机、`total` 恒为 `null`，所以掷骰经 `drawFromTable` 的 `roll` 参数注入，交给 `table.draw` 的那个常量式 Roll 经 `makeRoll` 参数注入；真实运行时两者都用默认实现。(3) `features.enabled()` 要读世界设置，而桩的 `game.settings.get` 对**未注册**的键会抛错，所以登记测试里必须照 `init` 的真实顺序先 `register()` 再 `features.registerSettings()`。

```js
import { afterEach, beforeEach } from "vitest";
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";
import { d66RollComposerRepair, drawFromTable } from "../scripts/repairs/d66-roll-composer.mjs";
import { features } from "../scripts/kernel/features.mjs";
import { patches } from "../scripts/kernel/patches.mjs";

/** A stand-in for a RollTable document: only the members drawFromTable touches. */
function fakeTable({ formula = "10*1d6+1d6", rows = [], replacement = true, name = "EV - 20" } = {}) {
  const table = {
    id: "tbl",
    name,
    formula,
    replacement,
    results: rows.map((r) => ({ id: r.id, range: r.range, drawn: false })),
    draws: [],
    async draw(options) {
      table.draws.push(options);
      // Mirrors RollTable#draw lines 108-114: a table without replacement marks
      // the rows it produced as drawn, and getResultsForRoll then skips them.
      if (!table.replacement) {
        for (const row of table.results) {
          if (options.roll.total >= row.range[0] && options.roll.total <= row.range[1]) row.drawn = true;
        }
      }
      return { roll: options.roll, results: [] };
    },
  };
  return table;
}

/** Hands back the queued numbers in order, standing in for real dice. */
function queuedRolls(values) {
  const queue = [...values];
  return async () => queue.shift();
}

/** Stands in for `new Roll(String(total)).evaluate()`. */
async function fakeRoll(total) {
  return { formula: String(total), total, async reroll() { return this; } };
}

describe("drawFromTable", () => {
  let ctx;
  beforeEach(() => {
    ctx = installFoundryStub({ i18n: { "AEA.tableDraw.noEncounter": "{table}: no encounter." } });
  });
  afterEach(() => uninstallFoundryStub());

  it("hands the table a constant-formula Roll carrying the tens-modified result", async () => {
    const table = fakeTable({ rows: [{ id: "r24", range: [24, 24] }] });
    await drawFromTable(table, { modifier: -3, roll: queuedRolls([5, 4]), makeRoll: fakeRoll });

    expect(table.draws).toHaveLength(1);
    expect(table.draws[0].roll.formula).toBe("24");
    expect(table.draws[0].roll.total).toBe(24);
    // No mode was asked for, so Foundry is left to apply the world's own setting.
    expect(Object.keys(table.draws[0])).toEqual(["roll"]);
  });

  it("draws the table's own no-encounter row when it ships one", async () => {
    // EV-20 carries range [0,10] with the text "None".
    const table = fakeTable({ rows: [{ id: "none", range: [0, 10] }, { id: "r36", range: [36, 36] }] });
    await drawFromTable(table, { modifier: -3, roll: queuedRolls([3, 6]), makeRoll: fakeRoll });

    expect(table.draws[0].roll.total).toBe(6);
    expect(ctx.messages).toHaveLength(0);
  });

  it("posts its own line when the table has no row for a no-encounter result", async () => {
    const table = fakeTable({ name: "EV - 23", rows: [{ id: "r11", range: [11, 11] }] });
    await drawFromTable(table, { modifier: -3, roll: queuedRolls([3, 6]), makeRoll: fakeRoll });

    expect(table.draws).toHaveLength(0);
    expect(ctx.messages).toHaveLength(1);
    expect(ctx.messages[0].content).toContain("no encounter");
  });

  it("snaps a modified total on a non-d66 table instead of letting Foundry re-roll it", async () => {
    const table = fakeTable({ formula: "1d6", rows: [{ id: "a", range: [1, 3] }, { id: "b", range: [4, 6] }] });
    await drawFromTable(table, { modifier: 4, roll: queuedRolls([5]), makeRoll: fakeRoll });

    expect(table.draws[0].roll.total).toBe(6);
  });

  it("never re-uses a drawn row on a table without replacement", async () => {
    const table = fakeTable({
      formula: "1d6",
      replacement: false,
      rows: [{ id: "a", range: [1, 3] }, { id: "b", range: [4, 6] }],
    });
    await drawFromTable(table, { count: 2, roll: queuedRolls([2, 2]), makeRoll: fakeRoll });

    expect(table.draws.map((d) => d.roll.total)).toEqual([2, 4]);
  });

  it("warns instead of spinning when every row has already been drawn", async () => {
    const table = fakeTable({ formula: "1d6", replacement: false, rows: [{ id: "a", range: [1, 6] }] });
    await drawFromTable(table, { count: 2, roll: queuedRolls([3, 3]), makeRoll: fakeRoll });

    expect(table.draws).toHaveLength(1);
    expect(ctx.notifications.at(-1).type).toBe("warn");
  });

  it("speaks the legacy rollMode vocabulary on a v13 client", async () => {
    installFoundryStub({ generation: 13 });
    const table = fakeTable({ formula: "1d6", rows: [{ id: "a", range: [1, 6] }] });
    await drawFromTable(table, { mode: "gm", roll: queuedRolls([3]), makeRoll: fakeRoll });

    expect(table.draws[0].rollMode).toBe("gmroll");
    expect(table.draws[0].messageMode).toBeUndefined();
  });
});

describe("d66RollComposerRepair.register", () => {
  let ctx;
  beforeEach(() => {
    ctx = installFoundryStub();
  });
  afterEach(() => uninstallFoundryStub());

  it("registers a switchable feature and a patch that has not been applied yet", () => {
    // The real init order, per the module's lifecycle: defs first, settings second.
    d66RollComposerRepair.register();
    features.registerSettings();

    expect(features.all().map((f) => f.id)).toContain("d66-roll-composer");
    expect(features.enabled("d66-roll-composer")).toBe(true);

    const entry = patches.status().find((p) => p.id === "d66-roll-composer");
    expect(entry).toBeTruthy();
    expect(Object.keys(entry).sort()).toEqual(["applied", "fixedIn", "id", "reason", "target", "type"]);
    expect(entry.type).toBe("OVERRIDE");
    expect(entry.target).toContain("enricher.mjs:153");
    expect(entry.applied).toBe(false);
    expect(entry.reason).toBe("pending");
    // Nothing is installed at register time, and this repair never uses libWrapper.
    expect(ctx.wrappers).toHaveLength(0);
  });
});
```

- [ ] **Step 12: 跑一遍，看它失败**

Run: `npx vitest run test/repair-d66-roll-composer.test.mjs`
Expected: FAIL —— `Failed to load url ../scripts/repairs/d66-roll-composer.mjs`，整个文件加载失败（连之前那 24 条也不执行）。

- [ ] **Step 13: 写抽表引擎与捕获阶段接管**

新建 `scripts/repairs/d66-roll-composer.mjs`：

```js
import { MID } from "../const.mjs";
import { features } from "../kernel/features.mjs";
import { patches } from "../kernel/patches.mjs";
import { selftest } from "../kernel/selftest.mjs";
import {
  pureComposeD66,
  pureDrawHandlerVerdict,
  pureDrawMessageOptions,
  pureEscapeHtml,
  pureIsD66Formula,
  pureResolveDrawTotal,
  pureUnreachableRows,
} from "./d66-roll-composer.pure.mjs";

export const REPAIR_ID = "d66-roll-composer";

let takeoverInstalled = false;

/** Default dice source. Injectable so the engine is testable without real dice. */
async function evaluateFormula(formula) {
  const evaluated = await new Roll(String(formula)).evaluate();
  return Number(evaluated.total);
}

/** Default constant-Roll factory. Injectable for the same reason. */
async function constantRoll(total) {
  return new Roll(String(total)).evaluate();
}

/**
 * The rows that are still drawable, shaped for the pure layer.
 * Mirrors RollTable#getResultsForRoll (client/documents/roll-table.mjs:341-342),
 * which filters out rows already marked drawn.
 */
function availableRows(table) {
  return Array.from(table?.results ?? [])
    .filter((row) => !row.drawn)
    .map((row) => ({
      resultId: row.id ?? row._id ?? null,
      range: Array.isArray(row.range) ? row.range : [],
    }));
}

/** GM user ids, for whispering a message we compose ourselves. */
function gmUserIds() {
  return (game.users?.contents ?? []).filter((user) => user.isGM).map((user) => user.id);
}

async function postNoEncounter(table, mode) {
  const line = game.i18n.format("AEA.tableDraw.noEncounter", { table: table.name });
  await ChatMessage.create({
    content: `<div class="aea-table-draw"><p>${pureEscapeHtml(line)}</p></div>`,
    whisper: mode === "gm" || mode === "blind" ? gmUserIds() : [],
  });
}

/**
 * Draw from a table with a modifier that actually lands on the row we computed.
 *
 * On a d66 table the modifier applies to the TENS DIGIT, per the rules; on any
 * other table it applies to the total. Either way the total is snapped onto an
 * available row and handed to RollTable#draw as a CONSTANT-formula Roll:
 * RollTable#draw forwards to RollTable#roll, which always calls roll.reroll()
 * (client/documents/roll-table.mjs:302-309), so an ordinary evaluated Roll would
 * simply be thrown away, while re-rolling "24" is still 24.
 *
 * @param {object} table                     a RollTable document
 * @param {object} [options]
 * @param {number} [options.modifier=0]
 * @param {"public"|"gm"|"blind"|"self"|null} [options.mode=null]  null = the world's setting
 * @param {number} [options.count=1]         clamped to 1..20
 * @param {(formula:string)=>Promise<number>} [options.roll]      dice source; injected by tests
 * @param {(total:number)=>Promise<object>} [options.makeRoll]    constant-Roll factory; injected by tests
 */
export async function drawFromTable(
  table,
  { modifier = 0, mode = null, count = 1, roll = evaluateFormula, makeRoll = constantRoll } = {}
) {
  if (!table) return;
  const draws = Math.min(20, Math.max(1, Number(count) || 1));
  const messageOptions = pureDrawMessageOptions(game.release?.generation, mode);
  const mod = Number(modifier) || 0;

  for (let i = 0; i < draws; i++) {
    const rows = availableRows(table);
    if (!rows.length) {
      ui.notifications.warn(game.i18n.format("AEA.tableDraw.noRows", { table: table.name }));
      return;
    }

    let total = null;
    let noEncounter = false;
    if (pureIsD66Formula(table.formula)) {
      const tens = await roll("1d6");
      const ones = await roll("1d6");
      const composed = pureComposeD66({ tens, ones, tensMod: mod });
      total = composed.total;
      noEncounter = composed.noEncounter;
    } else {
      total = Number(await roll(table.formula)) + mod;
    }
    if (!Number.isFinite(total)) return;

    const snapped = pureResolveDrawTotal(rows, total);
    // "No encounter" that the table itself has a row for (EV-20's [0,10] "None")
    // is drawn from the table, so the GM reads the text the designers wrote.
    if (noEncounter && snapped !== total) {
      await postNoEncounter(table, mode);
      continue;
    }
    if (snapped === null) return;
    await table.draw({ roll: await makeRoll(snapped), ...messageOptions });
  }
}

/**
 * Read the source of the click handler the system registered with jQuery.
 * `drawFromRollableTable` is a module-local function (enricher.mjs:101) with no
 * addressable path, so jQuery's own event registry is the only way to reach it.
 * Returns null when there is nothing to read, which the pure verdict reports as
 * "unknown" rather than guessing.
 */
function systemDrawHandlerSource() {
  const clickHandlers = globalThis.jQuery?._data?.(globalThis.document, "events")?.click ?? [];
  const entry = Array.from(clickHandlers).find((handler) => handler?.selector === ".draw-from-table");
  return entry?.handler ? String(entry.handler) : null;
}

/**
 * Take over the enricher's dice icon.
 *
 * A capture-phase listener on `document` runs before any bubble-phase delegate
 * on `document`, and stopImmediatePropagation there means the system's jQuery
 * handler never sees the click. When the GM switches this repair off we simply
 * do not intercept, so the stock behaviour comes back on the very next click
 * with no reload.
 *
 * This is a DOM listener, not a Foundry hook, so the kernel's hook-ownership
 * rules do not apply to it; it is still installed exactly once (see apply()).
 */
function onDrawIconCapture(event) {
  const icon = event.target?.closest?.(".draw-from-table");
  if (!icon) return;
  if (!features.enabled(REPAIR_ID)) return;

  event.preventDefault();
  event.stopImmediatePropagation();

  const uuid = icon.getAttribute("data-uuid");
  if (!uuid) return;
  const pinnedFormula = icon.getAttribute("data-roll");
  const wantsModifier = event.shiftKey;

  void (async () => {
    const table = await fromUuid(uuid);
    if (!(table instanceof RollTable)) return;
    // @DRAW[uuid]{name}{1d4} pins its own formula and the system ignores the
    // modifier entirely in that case (enricher.mjs:111-112). So do we — and we
    // do not put up a modifier box that would do nothing.
    if (pinnedFormula) {
      await table.draw({ roll: await new Roll(pinnedFormula).evaluate() });
      return;
    }
    if (!wantsModifier) {
      await drawFromTable(table);
      return;
    }
    const modifier = await promptModifier(table);
    if (modifier === null) return;
    await drawFromTable(table, { modifier });
  })();
}

async function promptModifier(table) {
  const isD66 = pureIsD66Formula(table.formula);
  const label = game.i18n.localize(isD66 ? "AEA.tableDraw.tensMod" : "AEA.tableDraw.modifier");
  // DialogV2 wraps `content` in its own <form> (client/applications/api/dialog.mjs:205-215),
  // so we must NOT nest another one here.
  const response = await foundry.applications.api.DialogV2.wait({
    window: { title: table.name },
    content:
      `<div class="form-group"><label>${label}</label>` +
      `<input type="number" name="modifier" value="0" step="1" autofocus /></div>` +
      `<p class="notes">${game.i18n.localize("AEA.tableDraw.modifierHint")}</p>`,
    rejectClose: false,
    buttons: [
      {
        action: "draw",
        default: true,
        label: game.i18n.localize("AEA.tableDraw.draw"),
        callback: (_event, button) => new foundry.applications.ux.FormDataExtended(button.form).object,
      },
      { action: "cancel", label: game.i18n.localize("AEA.tableDraw.cancel") },
    ],
  });
  if (!response || response === "cancel") return null;
  return Number(response.modifier) || 0;
}

export const d66RollComposerRepair = {
  id: REPAIR_ID,

  /** Called from main.mjs during `init`, through the REPAIRS array. */
  register() {
    // Contract §7: every behaviour change the GM might disagree with is switchable,
    // and the switch is read at execution time, never at install time. `gmOnly`
    // marks the setting as GM-editable; it does not turn the repair off for players,
    // which matters here because players click these dice icons too.
    features.register({ id: REPAIR_ID, default: "full", gmOnly: true, requires: [] });

    patches.register({
      id: REPAIR_ID,
      // OVERRIDE because we replace the system's click behaviour wholesale. It uses
      // no libWrapper — there is no addressable target — and it registers no Foundry
      // hook either; it installs one DOM listener.
      type: "OVERRIDE",
      target: 'document capture-phase click on ".draw-from-table" (module/helpers/enricher.mjs:153)',
      minSystem: "4.1.13",
      fixedIn: null,
      probe: () => pureDrawHandlerVerdict(systemDrawHandlerSource()) !== "fixed",
      apply: () => {
        // Idempotent twice over: the flag guards it, and addEventListener with the
        // same function reference and the same capture flag is a no-op by spec.
        if (takeoverInstalled) return;
        document.addEventListener("click", onDrawIconCapture, true);
        takeoverInstalled = true;
      },
    });

    selftest.register({
      id: "repair.d66-roll-composer.takeover",
      label: "AEA.selftest.repair.d66-roll-composer.takeover",
      run: () => {
        const verdict = pureDrawHandlerVerdict(systemDrawHandlerSource());
        return {
          ok: takeoverInstalled || verdict === "fixed",
          detail: `system handler: ${verdict}; capture takeover installed: ${takeoverInstalled}`,
        };
      },
    });

    selftest.register({
      id: "repair.d66-roll-composer.unreachableRows",
      label: "AEA.selftest.repair.d66-roll-composer.unreachableRows",
      run: () => {
        const offenders = [];
        for (const table of game.tables?.contents ?? []) {
          const rows = Array.from(table.results ?? []).map((row) => ({
            resultId: row.id ?? null,
            range: row.range,
          }));
          const unreachable = pureUnreachableRows(table.formula, rows);
          if (unreachable.length) offenders.push(`${table.name} (${unreachable.length})`);
        }
        if (!offenders.length) return { ok: true, detail: "no d66 table carries rows outside 11-66" };
        return {
          ok: takeoverInstalled,
          detail: takeoverInstalled
            ? `now reachable through the tens-digit composer: ${offenders.join(", ")}`
            : `unreachable, and the composer is not installed: ${offenders.join(", ")}`,
        };
      },
    });

    console.debug(`${MID} | ${REPAIR_ID} registered`);
  },
};
```

- [ ] **Step 14: 跑一遍，看它通过**

Run: `npx vitest run test/repair-d66-roll-composer.test.mjs`
Expected: PASS —— 32 passed。

- [ ] **Step 15: 提交**

```bash
git add scripts/repairs/d66-roll-composer.mjs test/repair-d66-roll-composer.test.mjs && git commit -m "$(cat <<'EOF'
feat(d66-roll-composer): 用捕获阶段接管 enricher 的骰子图标

enricher.mjs:101 的 drawFromRollableTable 是模块内的局部函数，只经 :153 的
$(document).on 注册，不是任何对象的属性，libWrapper 无 target 可用。
document 捕获阶段 + stopImmediatePropagation 是唯一能抢在 jQuery 冒泡委托
之前的干净办法；这是 DOM 监听、不是 Foundry 钩子，且同一函数引用重复
addEventListener 按规范就是空操作，再加一个标志位，重复 apply() 也安全。
关掉这条修复时我们不拦截，系统原行为下一次点击立刻回来，不需要重载世界。

抽表一律传常量式 Roll（draw 转调 roll，roll-table.mjs:302-309 一定会 reroll，
只有常量式 reroll 之后还是同一个数），传之前先用 pureResolveDrawTotal 对齐到
一个仍可抽的行，否则那个 while 会在空档上空转到第一万次。
data-roll 存在时按系统原样忽略修正，并且不弹那个不起作用的修正框。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 16: 加两份语言包的键**

语言包是**嵌套**结构，顶层唯一键是 `AEA`（仓库里有一条护栏测试，任何顶层新键、任何 `AEA.` 以外的扁平键、或两份文件键不对齐都会让它变红）。

把下面这些合并进 `lang/en.json` 已有的 `AEA` 对象（`feature` / `tableDraw` / `selftest` 三个子对象若已存在就合并进去，不要覆盖）：

```json
{
  "AEA": {
    "feature": {
      "d66-roll-composer": {
        "name": "D66 tens-digit modifier",
        "hint": "Applies a table modifier to the tens digit the way the rules do, instead of to the total, and stops the silent re-rolls a modified total causes. Switch it off to hand table draws straight back to the system."
      },
      "table-draw-tools-and-macros": {
        "name": "Table draw tools",
        "hint": "GM buttons for the Mother and creature table folders, plus two world macros. The tables come from the module's folder bindings rather than from a folder's display name, so renaming or translating a folder cannot empty the list."
      },
      "creature-table-folders": {
        "name": "Creature sheet table dropdowns",
        "hint": "Fills the two table dropdowns on a creature sheet from the module's folder bindings. Without it the sheet reads folders by display name and throws, which stops the sheet opening at all when a folder is missing or renamed."
      }
    },
    "tableDraw": {
      "title": "Draw from a table",
      "table": "Table",
      "modifier": "Modifier",
      "tensMod": "Tens-digit modifier",
      "modifierHint": "On a D66 table the modifier applies to the tens digit, and a modified tens digit of 0 or less means no encounter. On any other table it applies to the total.",
      "count": "Draws",
      "gmOnly": "GM only",
      "draw": "Draw",
      "cancel": "Cancel",
      "noTables": "That table folder is not bound yet, or it holds no tables. Bind it in the module's registry panel.",
      "noRows": "{table} has no undrawn rows left.",
      "noEncounter": "{table}: no encounter.",
      "folderUnbound": "The {key} folder is not bound, so that dropdown is empty. Bind it in the module's registry panel.",
      "tool": {
        "mother": "Mother tables",
        "creature": "Creature tables",
        "installMacros": "Install the Alien table macros"
      },
      "macroMother": "Alien - Mother table draw",
      "macroCreature": "Alien - Creature table draw",
      "macrosInstalled": "Macros: {created} created, {updated} updated, {kept} left alone.",
      "macroKept": "{name} has been edited by hand, so it was left alone.",
      "macroNeedsModule": "This macro needs the Alien Evolved: Automation module to be active."
    },
    "selftest": {
      "repair": {
        "d66-roll-composer": {
          "takeover": "The enricher dice icon is served by the tens-digit composer",
          "unreachableRows": "D66 tables carrying rows their own formula cannot reach"
        },
        "table-draw-tools-and-macros": {
          "tools": "The GM table-draw buttons are installed",
          "folderBindings": "The Mother and creature table folders resolve by id",
          "worldMacros": "The two world macros exist and are current"
        },
        "creature-table-folders": {
          "override": "The creature sheet's table dropdowns come from folder bindings"
        }
      }
    }
  }
}
```

把下面这些合并进 `lang/cn.json` 已有的 `AEA` 对象（键必须与上面**逐键一一对应**）：

```json
{
  "AEA": {
    "feature": {
      "d66-roll-composer": {
        "name": "D66 十位修正",
        "hint": "按规则把表格修正加在十位上而不是总数上，并消掉总数被改后引发的静默重掷。关掉它，抽表立刻交回系统原样处理。"
      },
      "table-draw-tools-and-macros": {
        "name": "抽表工具",
        "hint": "在场景控件里给出 Mother 系列表与生物表两个抽表按钮，并可一键在世界里创建两个宏。表的来源是模组的文件夹绑定，不是文件夹显示名，改名或汉化都不会让下拉变空。"
      },
      "creature-table-folders": {
        "name": "怪物卡表格下拉",
        "hint": "怪物卡上那两个表格下拉改由模组的文件夹绑定填充。没有它，系统按文件夹显示名查找并直接取 contents，文件夹缺失或改名时会抛错，整张怪物卡都打不开。"
      }
    },
    "tableDraw": {
      "title": "抽表",
      "table": "表格",
      "modifier": "修正",
      "tensMod": "十位修正",
      "modifierHint": "D66 表的修正加在十位上，十位修正后为 0 或更低即「无遭遇」；其它表的修正加在总数上。",
      "count": "抽取次数",
      "gmOnly": "仅 GM 可见",
      "draw": "抽取",
      "cancel": "取消",
      "noTables": "这个表格文件夹还没有绑定，或者里面没有表。请在模组的注册表面板里绑定它。",
      "noRows": "{table} 里已经没有可抽的行了。",
      "noEncounter": "{table}：无遭遇。",
      "folderUnbound": "{key} 文件夹尚未绑定，这个下拉因此是空的。请在模组的注册表面板里绑定它。",
      "tool": {
        "mother": "Mother 系列表",
        "creature": "生物表",
        "installMacros": "安装异形抽表宏"
      },
      "macroMother": "异形 - Mother 表抽取",
      "macroCreature": "异形 - 生物表抽取",
      "macrosInstalled": "宏：新建 {created} 个，更新 {updated} 个，保留 {kept} 个。",
      "macroKept": "{name} 被手工改过，已保持原样未动。",
      "macroNeedsModule": "这个宏需要 Alien Evolved: Automation 模组处于启用状态。"
    },
    "selftest": {
      "repair": {
        "d66-roll-composer": {
          "takeover": "enricher 的骰子图标已由十位修正器接管",
          "unreachableRows": "D66 表里自身掷骰式抽不到的行"
        },
        "table-draw-tools-and-macros": {
          "tools": "GM 抽表按钮已安装",
          "folderBindings": "Mother 与生物表两个文件夹按 id 解析成功",
          "worldMacros": "两个世界宏存在且是最新版"
        },
        "creature-table-folders": {
          "override": "怪物卡的表格下拉来自文件夹绑定"
        }
      }
    }
  }
}
```

- [ ] **Step 17: 写第四个失败测试 —— 宏计划、合集探针与文件夹取表**

新建 `test/repair-table-draw-tools.test.mjs`：

```js
import { describe, expect, it } from "vitest";
import {
  pureMacroPackMissing,
  pureMacroPlan,
  pureTablesInFolder,
} from "../scripts/repairs/table-draw-tools-and-macros.pure.mjs";

describe("pureMacroPackMissing", () => {
  it("is true for the packs alienrpg 4.1.13 actually ships", () => {
    // system.json declares exactly one pack and it is an Adventure, so the
    // documented "right-click the Macro compendium and import" workflow has
    // nothing to right-click.
    const packs = [{ type: "Adventure", packageName: "alienrpg" }];
    expect(pureMacroPackMissing(packs, "alienrpg")).toBe(true);
    expect(pureMacroPackMissing([], "alienrpg")).toBe(true);
  });

  it("is false once the system itself ships a Macro compendium", () => {
    const packs = [
      { type: "Adventure", packageName: "alienrpg" },
      { type: "Macro", packageName: "alienrpg" },
    ];
    expect(pureMacroPackMissing(packs, "alienrpg")).toBe(false);
  });

  it("does not count another package's Macro compendium", () => {
    const packs = [{ type: "Macro", packageName: "some-other-module" }];
    expect(pureMacroPackMissing(packs, "alienrpg")).toBe(true);
  });
});

describe("pureMacroPlan", () => {
  const specs = [
    { kind: "mother", name: "Alien - Mother table draw", command: "COMMAND-V2" },
    { kind: "creature", name: "Alien - Creature table draw", command: "COMMAND-V2" },
  ];

  it("creates both macros in a world that has none", () => {
    expect(pureMacroPlan([], specs)).toEqual({ create: ["mother", "creature"], update: [], kept: [] });
  });

  it("does nothing when the world already carries the current command", () => {
    const existing = [
      { kind: "mother", name: "m", command: "COMMAND-V2", installedCommand: "COMMAND-V2" },
      { kind: "creature", name: "c", command: "COMMAND-V2", installedCommand: "COMMAND-V2" },
    ];
    expect(pureMacroPlan(existing, specs)).toEqual({ create: [], update: [], kept: [] });
  });

  it("updates a macro that is still exactly as we last wrote it", () => {
    const existing = [
      { kind: "mother", name: "m", command: "COMMAND-V1", installedCommand: "COMMAND-V1" },
      { kind: "creature", name: "c", command: "COMMAND-V2", installedCommand: "COMMAND-V2" },
    ];
    expect(pureMacroPlan(existing, specs)).toEqual({ create: [], update: ["mother"], kept: [] });
  });

  it("leaves a macro the GM has edited alone, and says so", () => {
    const existing = [
      { kind: "mother", name: "My tweaked draw", command: "GM EDIT", installedCommand: "COMMAND-V1" },
      { kind: "creature", name: "c", command: "COMMAND-V2", installedCommand: "COMMAND-V2" },
    ];
    expect(pureMacroPlan(existing, specs)).toEqual({
      create: [],
      update: [],
      kept: [{ kind: "mother", name: "My tweaked draw" }],
    });
  });
});

describe("pureTablesInFolder", () => {
  // Folder#contents is direct children only (client/documents/folder.mjs:53),
  // and getSubfolders(true) walks the tree (:364).
  const table = (id, name) => ({ documentName: "RollTable", id, name });
  const child = { contents: [table("t3", "Deep")], getSubfolders: () => [] };
  const folder = {
    contents: [table("t1", "Top"), { documentName: "Actor", id: "a1", name: "Not a table" }],
    getSubfolders: (recursive) => (recursive ? [child] : []),
  };

  it("returns the RollTables under the folder and its sub-folders", () => {
    expect(pureTablesInFolder(folder).map((t) => t.id)).toEqual(["t1", "t3"]);
  });

  it("ignores documents that are not RollTables", () => {
    expect(pureTablesInFolder(folder).every((t) => t.documentName === "RollTable")).toBe(true);
  });

  it("is empty for a missing folder, so an unbound key can never throw", () => {
    expect(pureTablesInFolder(null)).toEqual([]);
    expect(pureTablesInFolder(undefined)).toEqual([]);
    expect(pureTablesInFolder({})).toEqual([]);
  });
});
```

- [ ] **Step 18: 跑一遍，看它失败**

Run: `npx vitest run test/repair-table-draw-tools.test.mjs`
Expected: FAIL —— `Failed to load url ../scripts/repairs/table-draw-tools-and-macros.pure.mjs`，10 条用例一条都不执行。

- [ ] **Step 19: 写这一条修复的纯层**

新建 `scripts/repairs/table-draw-tools-and-macros.pure.mjs`（同样不得引用任何 Foundry 全局）：

```js
/**
 * Pure layer of the `table-draw-tools-and-macros` repair.
 */

/**
 * Does the system still ship no Macro compendium?
 *
 * alienrpg 4.1.13's system.json declares exactly one pack —
 * {name:"alien-rpg-system", type:"Adventure"} — so the documented
 * "right-click the compendium, Import All Content" route to the GM macros does
 * not exist. The day upstream ships a Macro pack, this goes false and the patch
 * retires itself.
 *
 * @param {Array<{type:string, packageName:string}>} packs  pack metadata
 * @param {string} systemId
 * @returns {boolean}
 */
export function pureMacroPackMissing(packs, systemId) {
  return !(packs ?? []).some((pack) => pack?.type === "Macro" && pack?.packageName === systemId);
}

/**
 * Decide what to do with the world's copies of our macros.
 *
 * A world macro is a frozen copy, so we offer rather than overwrite. We can only
 * tell "the GM customised this" from "this is ours, just from an older release"
 * because we record the command we wrote in a flag: if the live command still
 * equals the recorded one, the macro is untouched and safe to refresh.
 *
 * @param {Array<{kind:string, name:string, command:string, installedCommand:string|null}>} existing
 * @param {Array<{kind:string, name:string, command:string}>} specs
 * @returns {{create:string[], update:string[], kept:Array<{kind:string, name:string}>}}
 */
export function pureMacroPlan(existing, specs) {
  const plan = { create: [], update: [], kept: [] };
  for (const spec of specs ?? []) {
    const found = (existing ?? []).find((macro) => macro?.kind === spec.kind);
    if (!found) {
      plan.create.push(spec.kind);
      continue;
    }
    if (found.command === spec.command) continue;
    if (found.command !== found.installedCommand) {
      plan.kept.push({ kind: spec.kind, name: found.name });
      continue;
    }
    plan.update.push(spec.kind);
  }
  return plan;
}

/**
 * Every RollTable under a folder, including its sub-folders.
 *
 * Folder#contents is direct children only (client/documents/folder.mjs:53-57),
 * and getSubfolders(true) walks the whole tree (:364-370). The shipped macros
 * instead filtered `t.folder.name === 'Alien Mother Tables'`
 * (macros/gmRollMotherTables.js:4), which both misses sub-folders and empties
 * itself the moment anyone renames or translates the folder.
 *
 * Takes the Folder document as a parameter and touches no global, so it is a
 * pure function testable with plain object literals. Membership is decided by
 * `documentName` rather than `instanceof RollTable` for the same reason.
 *
 * @param {object|null} folder
 * @returns {object[]} the RollTable documents, in tree order
 */
export function pureTablesInFolder(folder) {
  if (!folder) return [];
  const folders = [folder, ...(folder.getSubfolders?.(true) ?? [])];
  const out = [];
  for (const entry of folders) {
    for (const doc of entry?.contents ?? []) {
      if (doc?.documentName === "RollTable") out.push(doc);
    }
  }
  return out;
}
```

- [ ] **Step 20: 跑一遍，看它通过**

Run: `npx vitest run test/repair-table-draw-tools.test.mjs`
Expected: PASS —— 10 passed。

- [ ] **Step 21: 提交**

```bash
git add scripts/repairs/table-draw-tools-and-macros.pure.mjs test/repair-table-draw-tools.test.mjs lang/en.json lang/cn.json && git commit -m "$(cat <<'EOF'
feat(table-draw-tools-and-macros): 纯层（合集探针、宏计划、文件夹取表）与两份语言包

pureMacroPackMissing 把「系统还没发 Macro 合集」变成可运行的探针：
system.json 的 packs 只有 alien-rpg-system 一项且是 Adventure 类型，
所以系统文档说的「右键合集 → Import All Content」无从下手；哪天上游发了
Macro 合集，这条就自动退休。

pureTablesInFolder 收口出厂宏那条按文件夹显示名筛表的写法
（gmRollMotherTables.js:4 / gmRollCreatureTables.js:5，汉化或改名即空），
并顺带把子文件夹也算进来；它只吃 Folder 文档参数、不碰全局，所以能用
对象字面量做真单测。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 22: 追加失败测试 —— 宏正文与登记**

追加到 `test/repair-table-draw-tools.test.mjs` 末尾：

```js
import { afterEach, beforeEach } from "vitest";
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";
import {
  macroCommand,
  tableDrawToolsAndMacrosRepair,
} from "../scripts/repairs/table-draw-tools-and-macros.mjs";
import { features } from "../scripts/kernel/features.mjs";
import { patches } from "../scripts/kernel/patches.mjs";

describe("macroCommand", () => {
  it("reaches the module by importing this very file, not through any api surface", () => {
    const command = macroCommand("mother");
    expect(command).toContain(
      'foundry.utils.getRoute("modules/alien-evolved-automation/scripts/repairs/table-draw-tools-and-macros.mjs")'
    );
    expect(command).toContain('openDrawDialog("mother")');
    // The module api is frozen at its eight kernel members, and the aea.* hook
    // namespace holds exactly one hook, which is not this one.
    expect(command).not.toContain(".api.");
    expect(command).not.toContain("Hooks.call");
  });

  it("says something useful when the module is not active", () => {
    // With the module off the language files are not loaded, so the macro needs a
    // literal English fallback of its own.
    expect(macroCommand("creature")).toContain("AEA.tableDraw.macroNeedsModule");
    expect(macroCommand("creature")).toContain("This macro needs the Alien Evolved: Automation module");
  });
});

describe("tableDrawToolsAndMacrosRepair.register", () => {
  beforeEach(() => installFoundryStub());
  afterEach(() => uninstallFoundryStub());

  it("registers a switchable feature and a hook patch that has not been applied yet", () => {
    tableDrawToolsAndMacrosRepair.register();
    features.registerSettings();

    expect(features.all().map((f) => f.id)).toContain("table-draw-tools-and-macros");

    const entry = patches.status().find((p) => p.id === "table-draw-tools-and-macros");
    expect(entry).toBeTruthy();
    expect(Object.keys(entry).sort()).toEqual(["applied", "fixedIn", "id", "reason", "target", "type"]);
    // A hook patch must name the hook it installs, so patches.status() can show it.
    expect(entry.type).toBe("HOOK");
    expect(entry.target).toBe("getSceneControlButtons");
    expect(entry.applied).toBe(false);
    expect(entry.reason).toBe("pending");
  });
});
```

- [ ] **Step 23: 跑一遍，看它失败**

Run: `npx vitest run test/repair-table-draw-tools.test.mjs`
Expected: FAIL —— `Failed to load url ../scripts/repairs/table-draw-tools-and-macros.mjs`，整个文件加载失败（连之前那 10 条也不执行）。

- [ ] **Step 24: 写抽表对话框、场景控件与宏安装器**

新建 `scripts/repairs/table-draw-tools-and-macros.mjs`：

```js
import { MID, SYSTEM_ID } from "../const.mjs";
import { features } from "../kernel/features.mjs";
import { patches } from "../kernel/patches.mjs";
import { registry } from "../kernel/registry.mjs";
import { selftest } from "../kernel/selftest.mjs";
import { drawFromTable } from "./d66-roll-composer.mjs";
import { pureEscapeHtml } from "./d66-roll-composer.pure.mjs";
import {
  pureMacroPackMissing,
  pureMacroPlan,
  pureTablesInFolder,
} from "./table-draw-tools-and-macros.pure.mjs";

export const REPAIR_ID = "table-draw-tools-and-macros";

const KINDS = [
  { kind: "mother", folderKey: "folderMotherTables", nameKey: "AEA.tableDraw.macroMother" },
  { kind: "creature", folderKey: "folderCreatureTables", nameKey: "AEA.tableDraw.macroCreature" },
];

let hooksInstalled = false;

/**
 * The command body stored in the world's copy of our macro.
 *
 * A macro is a frozen string in the world database, so it must depend on the
 * smallest, most stable thing available. The module's public `api` object is
 * frozen at its eight kernel members and does not carry repairs, and inventing
 * an `aea.*` hook name is not allowed either. What is left — and what is in fact
 * the most robust of the three — is importing this very file by URL: importing a
 * URL that has already been loaded returns the SAME module instance, so the
 * `features` and `registry` singletons the dialog uses are the ones main.mjs set
 * up. foundry.utils.getRoute (common/utils/helpers.mjs:698) supplies the route
 * prefix, which is what makes the URL identical to the one the browser used.
 *
 * Macro bodies are wrapped in an AsyncFunction (client/documents/macro.mjs:145),
 * so top-level `await` is legal here.
 */
export function macroCommand(kind) {
  return [
    `const mod = game.modules.get("${MID}");`,
    `if (!mod?.active) {`,
    `  const key = "AEA.tableDraw.macroNeedsModule";`,
    `  ui.notifications.error(game.i18n.has(key) ? game.i18n.localize(key)`,
    `    : "This macro needs the Alien Evolved: Automation module to be active.");`,
    `} else {`,
    `  const url = foundry.utils.getRoute("modules/${MID}/scripts/repairs/${REPAIR_ID}.mjs");`,
    `  const repair = await import(url);`,
    `  await repair.openDrawDialog("${kind}");`,
    `}`,
  ].join("\n");
}

/**
 * The GM-facing draw dialog: pick a table from a bound folder, give a modifier,
 * choose how many draws, and keep it off the players' screens by default.
 *
 * @param {"mother"|"creature"} kind
 */
export async function openDrawDialog(kind) {
  if (!features.enabled(REPAIR_ID)) return;
  const spec = KINDS.find((entry) => entry.kind === kind) ?? KINDS[0];
  const tables = pureTablesInFolder(registry.folder(spec.folderKey));
  if (!tables.length) {
    ui.notifications.warn(game.i18n.localize("AEA.tableDraw.noTables"));
    return;
  }

  const options = tables
    .map((table) => `<option value="${table.id}">${pureEscapeHtml(table.name)}</option>`)
    .join("");

  // DialogV2 supplies the <form> itself (client/applications/api/dialog.mjs:205-215).
  const response = await foundry.applications.api.DialogV2.wait({
    window: { title: game.i18n.localize("AEA.tableDraw.title") },
    content:
      `<div class="form-group"><label>${game.i18n.localize("AEA.tableDraw.table")}</label>` +
      `<select name="tableId">${options}</select></div>` +
      `<div class="form-group"><label>${game.i18n.localize("AEA.tableDraw.modifier")}</label>` +
      `<input type="number" name="modifier" value="0" step="1" /></div>` +
      `<div class="form-group"><label>${game.i18n.localize("AEA.tableDraw.count")}</label>` +
      `<input type="number" name="count" value="1" min="1" max="20" step="1" /></div>` +
      `<div class="form-group"><label>${game.i18n.localize("AEA.tableDraw.gmOnly")}</label>` +
      `<input type="checkbox" name="gmOnly" checked /></div>` +
      `<p class="notes">${game.i18n.localize("AEA.tableDraw.modifierHint")}</p>`,
    rejectClose: false,
    buttons: [
      {
        action: "draw",
        default: true,
        label: game.i18n.localize("AEA.tableDraw.draw"),
        callback: (_event, button) => new foundry.applications.ux.FormDataExtended(button.form).object,
      },
      { action: "cancel", label: game.i18n.localize("AEA.tableDraw.cancel") },
    ],
  });
  if (!response || response === "cancel") return;

  const table = game.tables.get(response.tableId);
  if (!table) return;

  await drawFromTable(table, {
    modifier: Number(response.modifier) || 0,
    count: Number(response.count) || 1,
    // The shipped macros broadcast publicly unless the GM remembered to flip the
    // world roll mode first and back afterwards.
    mode: response.gmOnly ? "gm" : "public",
  });
}

/** The world's copies of our macros, shaped for the pure planner. */
function installedMacros() {
  return (game.macros?.contents ?? [])
    .map((macro) => ({
      kind: macro.flags?.[MID]?.macroKind ?? null,
      name: macro.name,
      command: macro.command,
      installedCommand: macro.flags?.[MID]?.installedCommand ?? null,
      doc: macro,
    }))
    .filter((entry) => typeof entry.kind === "string");
}

function macroSpecs() {
  return KINDS.map((entry) => ({
    kind: entry.kind,
    name: game.i18n.localize(entry.nameKey),
    command: macroCommand(entry.kind),
  }));
}

async function installMacros() {
  if (!features.enabled(REPAIR_ID)) return;
  const specs = macroSpecs();
  const existing = installedMacros();
  const plan = pureMacroPlan(existing, specs);

  for (const kind of plan.create) {
    const spec = specs.find((entry) => entry.kind === kind);
    await Macro.create({
      name: spec.name,
      type: "script",
      scope: "global",
      command: spec.command,
      flags: { [MID]: { macroKind: kind, installedCommand: spec.command } },
    });
  }
  for (const kind of plan.update) {
    const spec = specs.find((entry) => entry.kind === kind);
    const found = existing.find((entry) => entry.kind === kind);
    await found?.doc?.update({
      command: spec.command,
      [`flags.${MID}.installedCommand`]: spec.command,
    });
  }
  for (const kept of plan.kept) {
    ui.notifications.info(game.i18n.format("AEA.tableDraw.macroKept", { name: kept.name }));
  }
  ui.notifications.info(
    game.i18n.format("AEA.tableDraw.macrosInstalled", {
      created: plan.create.length,
      updated: plan.update.length,
      kept: plan.kept.length,
    })
  );
}

/**
 * Add three GM buttons to the token tool group.
 * The control group key is the token layer's name: "tokens" on v14
 * (client/canvas/layers/tokens.mjs:158), "token" on v13.
 */
function addSceneControls(controls) {
  // `gmOnly` on the feature def only means the setting is GM-editable, so the
  // GM-only surface has to be guarded here explicitly.
  if (!game.user.isGM) return;
  if (!features.enabled(REPAIR_ID)) return;
  const tokens = controls.tokens ?? controls.token;
  if (!tokens?.tools) return;
  const base = Object.keys(tokens.tools).length;

  tokens.tools.aeaMotherDraw = {
    name: "aeaMotherDraw",
    order: base + 1,
    title: "AEA.tableDraw.tool.mother",
    icon: "fa-solid fa-dice-d6",
    button: true,
    onChange: () => void openDrawDialog("mother"),
  };
  tokens.tools.aeaCreatureDraw = {
    name: "aeaCreatureDraw",
    order: base + 2,
    title: "AEA.tableDraw.tool.creature",
    icon: "fa-solid fa-spider",
    button: true,
    onChange: () => void openDrawDialog("creature"),
  };
  tokens.tools.aeaInstallMacros = {
    name: "aeaInstallMacros",
    order: base + 3,
    title: "AEA.tableDraw.tool.installMacros",
    icon: "fa-solid fa-scroll",
    button: true,
    onChange: () => void installMacros(),
  };
}

export const tableDrawToolsAndMacrosRepair = {
  id: REPAIR_ID,

  /** Called from main.mjs during `init`, through the REPAIRS array. */
  register() {
    features.register({ id: REPAIR_ID, default: "full", gmOnly: true, requires: [] });

    patches.register({
      id: REPAIR_ID,
      // HOOK: a registered patch may install a domain hook from its own apply().
      // The three conditions that come with that permission are met below:
      // the hook name is the def's `target` so status() shows it, apply() is
      // idempotent, and the callback's first lines check the switch.
      type: "HOOK",
      target: "getSceneControlButtons",
      minSystem: "4.1.13",
      fixedIn: null,
      probe: () =>
        pureMacroPackMissing(
          (game.packs?.contents ?? []).map((pack) => ({
            type: pack.metadata?.type,
            packageName: pack.metadata?.packageName,
          })),
          SYSTEM_ID
        ),
      apply: () => {
        if (hooksInstalled) return;
        Hooks.on("getSceneControlButtons", addSceneControls);
        hooksInstalled = true;
      },
    });

    selftest.register({
      id: "repair.table-draw-tools-and-macros.tools",
      label: "AEA.selftest.repair.table-draw-tools-and-macros.tools",
      run: () => ({
        ok: hooksInstalled,
        detail: hooksInstalled
          ? "getSceneControlButtons listener installed"
          : "not installed this session",
      }),
    });

    selftest.register({
      id: "repair.table-draw-tools-and-macros.folderBindings",
      label: "AEA.selftest.repair.table-draw-tools-and-macros.folderBindings",
      run: () => {
        const unbound = KINDS.filter((entry) => !registry.folder(entry.folderKey)).map((e) => e.folderKey);
        return {
          ok: unbound.length === 0,
          detail: unbound.length ? `unbound: ${unbound.join(", ")}` : "both table folders are bound",
        };
      },
    });

    selftest.register({
      id: "repair.table-draw-tools-and-macros.worldMacros",
      label: "AEA.selftest.repair.table-draw-tools-and-macros.worldMacros",
      run: () => {
        const plan = pureMacroPlan(installedMacros(), macroSpecs());
        return {
          ok: plan.create.length === 0,
          detail: plan.create.length
            ? `not installed yet: ${plan.create.join(", ")}`
            : `installed; ${plan.update.length} stale, ${plan.kept.length} edited by the GM`,
        };
      },
    });

    console.debug(`${MID} | ${REPAIR_ID} registered`);
  },
};
```

- [ ] **Step 25: 跑一遍，看它通过**

Run: `npx vitest run test/repair-table-draw-tools.test.mjs`
Expected: PASS —— 13 passed。

- [ ] **Step 26: 提交**

```bash
git add scripts/repairs/table-draw-tools-and-macros.mjs test/repair-table-draw-tools.test.mjs && git commit -m "$(cat <<'EOF'
feat(table-draw-tools-and-macros): 场景控件三个按钮 + 世界内建宏，取代不可用的出厂宏

出厂宏三处坏掉：按文件夹显示名筛表（gmRollMotherTables.js:4 /
gmRollCreatureTables.js:5，汉化即空）、用 game.tables.size 判空下拉（两处 :25，
空下拉照样给 Draw 按钮，点下去 :34 的 table.formula 抛 TypeError）、
roll.evaluate({async:false})（两处 :39，这个选项在现在的 Roll 上早已不存在）；
而 system.json 的 packs 只有一项 Adventure，根本没有 Macro 合集可导入。

改为场景控件三个按钮加一键在世界里创建宏：表一律走 registry 的
folderMotherTables / folderCreatureTables 绑定并含子文件夹，抽取默认仅 GM 可见，
宏「提供而非覆盖」——只有 command 仍等于我们上次写入的那份时才刷新，
GM 改过的一律保留并提示。

宏正文不走 api（api 冻在八个内核成员上），也不自定义钩子名，而是按
foundry.utils.getRoute 拼出本文件的 URL 再 import 它：同一 URL 的 import
拿回的是同一个模块实例，所以宏里用到的 features / registry 就是 main.mjs
那两个单例。补丁 def 的 target 就写钩子名 getSceneControlButtons，
apply() 用标志位保证重复调用不重复挂，回调头两行查 isGM 与 features.enabled。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 27: 写第五个失败测试 —— 怪物卡文件夹守卫的纯层**

新建 `test/repair-creature-table-folders.test.mjs`：

```js
import { describe, expect, it } from "vitest";
import {
  pureFolderLookupVerdict,
  pureTableChoices,
} from "../scripts/repairs/creature-table-folders.pure.mjs";

// Verbatim body of alienrpg 4.1.13's module/helpers/rollTableData.mjs:6-21.
const SHIPPED_STATIC = `static rTableget() {
  const folder = game.folders.contents.find((x) => x.name === "Alien Creature Tables")
  const aTables = folder.contents
  const lTables = {}
  lTables[0] = { key: "None", label: "None" }
  for (let index = 0; index < aTables.length; index++) {
    const counter = index + 1
    lTables[counter] = { key: aTables[index].name, label: aTables[index].name }
  }
  return lTables
}`;

// What a fixed upstream version would look like: no display-name lookup at all.
const FIXED_STATIC = `static rTableget() {
  const folder = game.folders.get(game.settings.get("alienrpg", "creatureTableFolder"))
  const aTables = folder?.contents ?? []
  const lTables = { 0: { key: "None", label: "None" } }
  return lTables
}`;

describe("pureFolderLookupVerdict", () => {
  it("calls the shipped 4.1.13 static buggy", () => {
    expect(pureFolderLookupVerdict(SHIPPED_STATIC)).toBe("buggy");
  });

  it("calls a static that no longer looks a folder up by display name fixed", () => {
    expect(pureFolderLookupVerdict(FIXED_STATIC)).toBe("fixed");
  });

  it("says unknown before the class has been read", () => {
    expect(pureFolderLookupVerdict(null)).toBe("unknown");
    expect(pureFolderLookupVerdict("")).toBe("unknown");
  });
});

describe("pureTableChoices", () => {
  const tables = [
    { id: "t1", name: "Critical Injuries - Xeno" },
    { id: "t2", name: "Panic Roll" },
    { id: "t3", name: "Critical Injuries - Synthetic" },
  ];

  it("keeps the shape the creature sheet template already expects", () => {
    // The system returns an object keyed 0,1,2,... with {key,label} entries, and
    // "None" always occupies slot 0 (module/helpers/rollTableData.mjs:12-19).
    expect(pureTableChoices(tables)).toEqual({
      0: { key: "None", label: "None" },
      1: { key: "Critical Injuries - Xeno", label: "Critical Injuries - Xeno" },
      2: { key: "Panic Roll", label: "Panic Roll" },
      3: { key: "Critical Injuries - Synthetic", label: "Critical Injuries - Synthetic" },
    });
  });

  it("applies the name prefix filter the Mother-folder static uses", () => {
    // cTableget filters `name.startsWith("Critical Injuries")` at :27.
    expect(pureTableChoices(tables, { prefix: "Critical Injuries" })).toEqual({
      0: { key: "None", label: "None" },
      1: { key: "Critical Injuries - Xeno", label: "Critical Injuries - Xeno" },
      2: { key: "Critical Injuries - Synthetic", label: "Critical Injuries - Synthetic" },
    });
  });

  it("still returns the None slot when there is nothing to offer", () => {
    // This is the whole point: an unbound folder must leave the sheet openable.
    expect(pureTableChoices([])).toEqual({ 0: { key: "None", label: "None" } });
    expect(pureTableChoices(null)).toEqual({ 0: { key: "None", label: "None" } });
  });

  it("uses the table name as both key and label, because the stored value is looked up by name", () => {
    const choices = pureTableChoices([{ id: "t1", name: "Panic Roll" }]);
    expect(choices[1].key).toBe(choices[1].label);
  });
});
```

- [ ] **Step 28: 跑一遍，看它失败**

Run: `npx vitest run test/repair-creature-table-folders.test.mjs`
Expected: FAIL —— `Failed to load url ../scripts/repairs/creature-table-folders.pure.mjs`，7 条用例一条都不执行。

- [ ] **Step 29: 写第三条修复的纯层**

新建 `scripts/repairs/creature-table-folders.pure.mjs`：

```js
/**
 * Pure layer of the `creature-table-folders` repair.
 *
 * module/helpers/rollTableData.mjs:7 and :24 look a Folder up by display name
 * (`game.folders.contents.find(x => x.name === "Alien Creature Tables")` and
 * `… === "Alien Mother Tables"`) and then dereference `folder.contents` at :9
 * and :27 with no guard. A missing, renamed or translated folder therefore
 * throws a TypeError inside module/sheets/creature-sheet.mjs:129-130, which runs
 * in _prepareContext — so the whole creature sheet fails to open.
 */

/** The literal markers, whitespace-stripped so a reformat upstream cannot fool us. */
const BUGGY_MARKERS = [
  'game.folders.contents.find((x)=>x.name==="AlienCreatureTables")',
  'game.folders.contents.find((x)=>x.name==="AlienMotherTables")',
  "game.folders.contents.find((x)=>x.name==='AlienCreatureTables')",
  "game.folders.contents.find((x)=>x.name==='AlienMotherTables')",
];

/**
 * Decide, from the system's own static-method source, whether the defect is still there.
 *
 * Three-valued for the same reason as the draw-handler verdict: "unknown" means
 * we have not been able to read the method yet (the class is reached by dynamic
 * import inside apply(), so before the first apply there is nothing to read).
 * The caller installs on "unknown" as well as on "buggy" — our replacement is a
 * complete substitute rather than a partial correction, and retiring on an
 * unreadable method would silently drop a fix for a crash.
 *
 * @param {string|null} source
 * @returns {"buggy"|"fixed"|"unknown"}
 */
export function pureFolderLookupVerdict(source) {
  if (typeof source !== "string" || source.trim() === "") return "unknown";
  const flat = source.replace(/\s+/g, "");
  if (BUGGY_MARKERS.some((marker) => flat.includes(marker))) return "buggy";
  if (flat.includes("lTables") || flat.includes("key:")) return "fixed";
  return "unknown";
}

/**
 * Build the choice object the creature sheet template already consumes.
 *
 * Shape and content are copied from the system verbatim
 * (module/helpers/rollTableData.mjs:10-20): an object keyed "0","1","2",… whose
 * entries are `{key, label}`, with `{key:"None", label:"None"}` always at 0.
 * `key` stays the table's display NAME rather than its id on purpose — the value
 * the sheet stores is later resolved by name at module/documents/actor.mjs:1839,
 * so switching to ids here would silently break every creature already set up.
 *
 * @param {Array<{name:string}>|null} tables
 * @param {{prefix?:string|null}} [options] prefix filters by name, as cTableget does
 * @returns {Record<string, {key:string, label:string}>}
 */
export function pureTableChoices(tables, { prefix = null } = {}) {
  const chosen = (tables ?? []).filter((table) => {
    const name = String(table?.name ?? "");
    return prefix ? name.startsWith(prefix) : name.length > 0;
  });
  const out = { 0: { key: "None", label: "None" } };
  chosen.forEach((table, index) => {
    out[index + 1] = { key: table.name, label: table.name };
  });
  return out;
}
```

- [ ] **Step 30: 跑一遍，看它通过**

Run: `npx vitest run test/repair-creature-table-folders.test.mjs`
Expected: PASS —— 7 passed。

- [ ] **Step 31: 追加失败测试 —— 第三条修复的登记**

追加到 `test/repair-creature-table-folders.test.mjs` 末尾：

```js
import { afterEach, beforeEach } from "vitest";
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";
import { creatureTableFoldersRepair } from "../scripts/repairs/creature-table-folders.mjs";
import { features } from "../scripts/kernel/features.mjs";
import { patches } from "../scripts/kernel/patches.mjs";

describe("creatureTableFoldersRepair.register", () => {
  let ctx;
  beforeEach(() => {
    ctx = installFoundryStub();
  });
  afterEach(() => uninstallFoundryStub());

  it("registers a switchable feature and an override patch that has not been applied yet", () => {
    creatureTableFoldersRepair.register();
    features.registerSettings();

    expect(features.all().map((f) => f.id)).toContain("creature-table-folders");
    expect(features.enabled("creature-table-folders")).toBe(true);

    const entry = patches.status().find((p) => p.id === "creature-table-folders");
    expect(entry).toBeTruthy();
    expect(Object.keys(entry).sort()).toEqual(["applied", "fixedIn", "id", "reason", "target", "type"]);
    expect(entry.type).toBe("OVERRIDE");
    expect(entry.target).toContain("rollTableData.mjs");
    expect(entry.applied).toBe(false);
    expect(entry.reason).toBe("pending");
    // This repair swaps two static methods; it never touches libWrapper, because
    // the class is not reachable from any global path.
    expect(ctx.wrappers).toHaveLength(0);
  });
});
```

- [ ] **Step 32: 跑一遍，看它失败**

Run: `npx vitest run test/repair-creature-table-folders.test.mjs`
Expected: FAIL —— `Failed to load url ../scripts/repairs/creature-table-folders.mjs`，整个文件加载失败（连之前那 7 条也不执行）。

- [ ] **Step 33: 写第三条修复的副作用层**

新建 `scripts/repairs/creature-table-folders.mjs`：

```js
import { MID } from "../const.mjs";
import { features } from "../kernel/features.mjs";
import { patches } from "../kernel/patches.mjs";
import { registry } from "../kernel/registry.mjs";
import { selftest } from "../kernel/selftest.mjs";
import { pureTablesInFolder } from "./table-draw-tools-and-macros.pure.mjs";
import { pureFolderLookupVerdict, pureTableChoices } from "./creature-table-folders.pure.mjs";

export const REPAIR_ID = "creature-table-folders";

/**
 * The system module that owns the two static methods. Reached by dynamic import
 * because `alienrpgrTableGet` is not exposed on any global: module/alienrpg.mjs:74-84
 * publishes nine other things on game.alienrpg but not this class, and
 * module/sheets/creature-sheet.mjs:4 imports it directly. Importing the same URL
 * the system used returns the SAME module instance, so replacing a static on the
 * class object is seen by creature-sheet.mjs:129-130, which does a property
 * lookup at call time rather than destructuring.
 */
const SOURCE_PATH = "systems/alienrpg/module/helpers/rollTableData.mjs";

/** { rTableget, cTableget } once installed; also our only window on the source. */
let originals = null;
const warned = new Set();

function staticSource() {
  return originals ? String(originals.rTableget) : null;
}

function warnUnboundOnce(folderKey) {
  if (warned.has(folderKey)) return;
  warned.add(folderKey);
  ui.notifications.warn(game.i18n.format("AEA.tableDraw.folderUnbound", { key: folderKey }));
}

/**
 * Build the dropdown choices for one bound folder.
 * An unbound folder yields the "None"-only object plus one warning per session,
 * which is the entire point: the sheet must still open.
 */
function choicesFor(folderKey, { prefix = null } = {}) {
  const folder = registry.folder(folderKey);
  if (!folder) {
    warnUnboundOnce(folderKey);
    return pureTableChoices([], { prefix });
  }
  return pureTableChoices(pureTablesInFolder(folder), { prefix });
}

export const creatureTableFoldersRepair = {
  id: REPAIR_ID,

  /** Called from main.mjs during `init`, through the REPAIRS array. */
  register() {
    features.register({ id: REPAIR_ID, default: "full", gmOnly: true, requires: [] });

    patches.register({
      id: REPAIR_ID,
      type: "OVERRIDE",
      target: "alienrpgrTableGet.rTableget / .cTableget (module/helpers/rollTableData.mjs:6, :23)",
      minSystem: "4.1.13",
      fixedIn: null,
      // Before the first apply() there is nothing to read, so the verdict is
      // "unknown" and we install; from then on the selftest reports the real one.
      probe: () => pureFolderLookupVerdict(staticSource()) !== "fixed",
      apply: async () => {
        if (originals) return; // idempotent
        const module = await import(foundry.utils.getRoute(SOURCE_PATH));
        const cls = module?.alienrpgrTableGet;
        if (typeof cls?.rTableget !== "function" || typeof cls?.cTableget !== "function") return;

        originals = { rTableget: cls.rTableget, cTableget: cls.cTableget };
        // Switch checked at call time: turning the repair off restores the stock
        // behaviour on the next sheet open, with no reload.
        cls.rTableget = () =>
          features.enabled(REPAIR_ID)
            ? choicesFor("folderCreatureTables")
            : originals.rTableget.call(cls);
        cls.cTableget = () =>
          features.enabled(REPAIR_ID)
            ? choicesFor("folderMotherTables", { prefix: "Critical Injuries" })
            : originals.cTableget.call(cls);
      },
    });

    selftest.register({
      id: "repair.creature-table-folders.override",
      label: "AEA.selftest.repair.creature-table-folders.override",
      run: () => {
        const verdict = pureFolderLookupVerdict(staticSource());
        const bound = ["folderCreatureTables", "folderMotherTables"].filter((key) => registry.folder(key));
        return {
          ok: !!originals && bound.length === 2,
          detail: `system source: ${verdict}; installed: ${!!originals}; bound folders: ${bound.length}/2`,
        };
      },
    });

    console.debug(`${MID} | ${REPAIR_ID} registered`);
  },
};
```

- [ ] **Step 34: 跑一遍，看它通过**

Run: `npx vitest run test/repair-creature-table-folders.test.mjs`
Expected: PASS —— 8 passed。

- [ ] **Step 35: 提交**

```bash
git add scripts/repairs/creature-table-folders.pure.mjs scripts/repairs/creature-table-folders.mjs test/repair-creature-table-folders.test.mjs && git commit -m "$(cat <<'EOF'
fix(creature-table-folders): 怪物卡两个表格下拉改走文件夹绑定，文件夹缺失不再打不开卡

rollTableData.mjs:7 与 :24 按文件夹显示名查找（"Alien Creature Tables" /
"Alien Mother Tables"），紧接着 :9 / :27 就取 folder.contents，文件夹缺失、
改名或被汉化时抛 TypeError；调用点在 creature-sheet.mjs:129-130 的
_prepareContext 里，于是整张怪物卡打不开。这两处消费的正是本任务已认领的
folderCreatureTables / folderMotherTables 两个 registry 键。

alienrpgrTableGet 没挂在 game.alienrpg 上（alienrpg.mjs:74-84），libWrapper
无 target；但 creature-sheet.mjs:129 是属性访问而不是解构，所以按同一 URL
动态 import 拿到同一个类对象、换掉两个静态方法就够了。未绑定时返回只含
「None」的选项表并每次会话提示一次，绝不让表单准备抛错。

key 仍保留表的显示名而不是 id：存进角色的值日后由 actor.mjs:1839 按名查表，
改成 id 会静默弄坏所有已配置好的怪物。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 36: 在 `main.mjs` 里接线**

`scripts/main.mjs` 由更早的任务建立，里面已经逐字写好十个锚点注释、四个生命周期钩子、以及两个显式数组 `FEATURES` 与 `REPAIRS`。`init` 段已有 `for (const r of REPAIRS) safely(..., () => r.register());`（它排在 `features.registerSettings()` **之前**，这正是本任务需要的顺序：三条 def 先在册，设置才生成得出来）；`ready` 段已有 `for (const r of REPAIRS) await safely(..., () => r.install?.());`。

**本任务不往任何生命周期钩子体内插代码，也不碰 `export const api`**，只在两个锚点后各插三行，一律**按锚点文本定位、不按行号**：

1. 找到逐字文本 `/* AEA-ANCHOR: imports */`，在它**下一行**插入：

```js
import { d66RollComposerRepair } from "./repairs/d66-roll-composer.mjs";
import { tableDrawToolsAndMacrosRepair } from "./repairs/table-draw-tools-and-macros.mjs";
import { creatureTableFoldersRepair } from "./repairs/creature-table-folders.mjs";
```

2. 找到逐字文本 `/* AEA-ANCHOR: repairs */`（它在 `const REPAIRS = [` 与 `];` 之间），在它**下一行**插入（数组每行一项，末尾带逗号）：

```js
  d66RollComposerRepair,
  tableDrawToolsAndMacrosRepair,
  creatureTableFoldersRepair,
```

三条的先后顺序无所谓：它们之间没有依赖，`patches.applyAll()` 按登记顺序装，而三条的 `apply()` 互不影响。**不要**动 `init` 段既有的行序。

- [ ] **Step 37: 跑全套**

Run: `npx vitest run test/repair-d66-roll-composer.test.mjs test/repair-table-draw-tools.test.mjs test/repair-creature-table-folders.test.mjs`
Expected: PASS —— 53 passed。

Run: `npm test`
Expected: 全绿。特别留意仓库骨架那几条护栏测试：它们断言 `const.mjs` 没有多余导出、`main.mjs` 的 `api` 恰好八个键、两份语言包的顶层键**只有** `AEA`、每个压平后的键都以 `AEA.` 开头、且 `cn.json` 与 `en.json` 键对键完全一致。如果语言包那条红了，多半是 Step 16 的 JSON 合并错位（把新对象贴到了 `AEA` 外面，或者两份文件漏了某个键）；如果 `api` 那条红了，说明有人误把本任务的东西挂上去了——本任务一个键都不加。

- [ ] **Step 38: MANUAL VERIFICATION（在本机冒烟世界里做，逐条对照）**

前提：世界里启用了 `alienrpg` 4.1.13、本模组、以及 `alien-evolved-corerules`；模组的注册表面板里 `folderMotherTables` 与 `folderCreatureTables` 已绑定。控制台里的 `api` 一律写成 `game.modules.get("alien-evolved-automation").api`。

1. **补丁装上了**：控制台跑 `api.patches.status()` → 期待三条 `d66-roll-composer`、`table-draw-tools-and-macros`、`creature-table-folders` 都在，每条恰好六个字段，`applied` 为 `true`、`reason` 为 `"ok"`。
2. **十位修正对话框**：找一段带 `@DRAW[...]` 的日志（corerules 的 solo 章节里到处都是），**按住 shift 左键**那个骰子图标 → 期待弹出一个标题是表名、字段写着「十位修正」的对话框，下面一行说明「D66 表的修正加在十位上……」。填 `-3`，点「抽取」。
3. **无遭遇真的能抽到了**：用「EV - 20. STAR SYSTEM ENCOUNTERS」把第 2 步重复十几次 → 期待：十位掷到 1/2/3 时抽到表里内容为 **None** 的那一行（**修之前这一行的概率是 0，一次都出不来**）；出别的结果时，行号的十位 = 掷出的十位 − 3。
4. **71-76 六行真的能抽到了**：用「EV - 23. GENERAL COLONY ENCOUNTERS」填 `+1` 重复十几次 → 期待能抽到 71「Starship crew off-duty」到 76「Colonists on strike or protesting」这六行（修之前一次都抽不到）。
5. **不按 shift**：直接点骰子图标 → 期待一次普通抽取，没有对话框。
6. **`data-roll` 那一支**：在任意日志里临时写一条 `@DRAW[某表uuid]{名字}{1d4}` → shift+左键 → 期待**不弹修正框**，直接按 `1d4` 抽（与系统原行为一致，因为系统在这一支本来就忽略修正）。
7. **开关能关**：设置 → 模组设置 → 本模组的特性菜单，把「D66 十位修正」改成「关闭」，**不重载世界**，再 shift+左键那个图标 → 期待弹出的是**系统原本那个**写着「Modifier」的对话框。改回「全自动」，再点一次 → 期待又变回「十位修正」。
8. **玩家也享受得到**：用一个玩家账号登录同一世界，shift+左键同一个图标 → 期待弹出的是**我们的**「十位修正」对话框（这条开关是 GM 才能改的世界设置，但行为对所有人生效）。
9. **场景控件**：GM 客户端左侧 token 那一组工具的底部 → 期待出现三个新按钮：「Mother 系列表」「生物表」「安装异形抽表宏」。切到第 8 步那个玩家账号 → 期待这三个按钮**不出现**。
10. **抽表对话框**：点「Mother 系列表」→ 期待下拉里是**绑定文件夹**（含其子文件夹）里的表。把那个文件夹改成一个中文名，刷新页面再点 → 期待下拉里**内容不变**（这正是出厂宏会变空的地方）。
11. **仅 GM 可见**：勾着「仅 GM 可见」抽一次 → 期待玩家客户端**看不到**这张卡；取消勾选再抽一次 → 期待玩家能看到。控制台无 `rollMode ... is deprecated` 的兼容性警告。
12. **抽取次数**：把「抽取次数」填 3 抽一次 → 期待出现三张卡。
13. **不重复抽同一行不会卡死**：随便挑一张表，在表的设置里**关掉** "Draw with replacement"，然后用对话框连抽到只剩一两行 → 期待每次都正常出卡、最后给出黄色提示「…里已经没有可抽的行了」，**界面不卡死、控制台没有 `TABLE.DrawMaximumIterations`**。
14. **装宏**：点「安装异形抽表宏」→ 期待世界的宏目录里出现两个宏，并弹出「宏：新建 2 个，更新 0 个，保留 0 个」。把其中一个宏的 command 改一个字，再点一次 → 期待提示「… 被手工改过，已保持原样未动」，**内容没被覆盖**。
15. **宏能用**：把宏拖到快捷栏点一下 → 期待打开对应的抽表对话框（宏正文是 `await import(foundry.utils.getRoute(...))`，走的是同一个模块实例，所以对话框里读到的绑定与第 10 步完全一致）。
16. **模组停用时宏会说话**：停用本模组、重载世界，再点那个宏 → 期待一条**红色**提示「This macro needs the Alien Evolved: Automation module to be active.」（模组停用时语言包没加载，所以显示的是宏里内置的英文兜底句），而不是一个静默失败或一条 404 报错。
17. **怪物卡打得开**：重新启用模组，打开任意一个 `creature` 类型的角色卡 → 期待卡正常打开，卡上那两个表格下拉里有内容（一个来自生物表文件夹，另一个是 Mother 文件夹里以 "Critical Injuries" 开头的表）。
18. **怪物卡在文件夹缺失时也打得开**：在注册表面板里把**生物表文件夹解绑**，重载页面，再打开同一张怪物卡 → 期待卡**照样打开**，那个下拉里只有一项「None」，并有一条黄色提示「folderCreatureTables 文件夹尚未绑定…」。**修之前这里是 TypeError、整张卡打不开**。把它重新绑定、重载，下拉内容恢复。
19. **抽表工具在解绑后的降级**：让 Mother 文件夹保持解绑状态，点「Mother 系列表」→ 期待一条黄色提示「这个表格文件夹还没有绑定…」，**不出空下拉、不抛 TypeError**（出厂宏正是在这里炸的）。
20. **自检**：控制台跑 `await api.selftest.runAll()` → 期待看到本任务的六条，`label` 都是**已本地化的中文/英文句子**（不是 `AEA.selftest.…` 这样的裸键）：`repair.d66-roll-composer.takeover`（`ok:true`，detail 里 `system handler: buggy`、`capture takeover installed: true`）、`repair.d66-roll-composer.unreachableRows`（`ok:true`，detail 里列出 EV-20 与 EV-23）、`repair.table-draw-tools-and-macros.tools`（`ok:true`）、`repair.table-draw-tools-and-macros.folderBindings`（`ok:true`）、`repair.table-draw-tools-and-macros.worldMacros`（`ok:true`）、`repair.creature-table-folders.override`（`ok:true`，detail 里 `system source: buggy`）。**不应该**出现任何 `patch:…` 开头的条目——补丁登记不会自动产生自检条目。

- [ ] **Step 39: 提交**

```bash
git add scripts/main.mjs && git commit -m "$(cat <<'EOF'
chore(main): 把三条抽表修复登记进 REPAIRS

按锚点文本插入：AEA-ANCHOR: imports 后补三条 import，AEA-ANCHOR: repairs
后补三项。不碰 export const api（八个内核键冻结），不往任何生命周期钩子体内
插代码，也不调整 init 段既有的行序。

三条修复都不导出 install()——安装由 patches.applyAll() 调各自的 apply() 完成，
ready 段的 r.install?.() 对它们是空操作。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

**UPSTREAM PR（三条，做完本任务后提给 pwatson100）**

- **PR A —— `module/helpers/enricher.mjs:109-113`**：把 `new Roll(\`${table.formula} + ${modifier}\`)` 换成 d66 感知的组装——先判 `table.formula` 是不是 `10*1d6+1d6`，是则把修正加到十位、组成常量式 `new Roll(String(tens * 10 + ones))` 再交给 `table.draw({roll})`，否则维持现状；顺便给 `table.draw` 传一个可见性选项。附上可复现的证据：`zB9xnHVp8ehurZZF`（EV-20）的 `[0,10]` 行与 `cR826F30mhD3LnDb`（EV-23）的 71-76 六行，在当前掷骰式下概率恒为 0；规则原文是「Modify the tens digit roll... A result of 0 or less indicates no encounter」。
- **PR B —— `macros/gmRollMotherTables.js` 与 `macros/gmRollCreatureTables.js`**：`roll.evaluate({ async: false })`（两处 `:39`）→ `await roll.evaluate()`；`if (game.tables.size > 0)`（两处 `:25`）→ 判**这个下拉**里有没有项；`t.folder.name === '...'`（`:4` / `:5`）→ 按 folder id 取，或至少在为空时给出可读提示；`new Dialog` → `foundry.applications.api.DialogV2`（`Dialog` 已标 deprecated since v13 until v16）；两个宏都在 `table.draw` 时显式传可见性。**另外要提醒他**：世界里的宏是 `macros/*.js` 的冻结副本，改源文件对已有世界无效，需要配合 `Macro.updateDocuments` 或在发行说明里让 GM 重新导入。
- **PR C —— `module/helpers/rollTableData.mjs:6-21` 与 `:23-38`**：把 `game.folders.contents.find(x => x.name === "...")` 换成一个可配置的文件夹引用（世界设置存 folder id），并把 `folder.contents` 改成 `folder?.contents ?? []`。附证据：文件夹缺失或被汉化时 `creature-sheet.mjs:129-130` 抛 TypeError，整张怪物卡打不开；这是纯崩溃修复，与本地化项目直接相关。
