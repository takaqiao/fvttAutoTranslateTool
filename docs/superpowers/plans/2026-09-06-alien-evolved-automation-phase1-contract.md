# 接口契约 v3.2 —— 所有任务必须逐字遵守

> **v3 说明**：v2 落地后，三个终检从不同角度撞上同一批真实缺口（渲染扇出点无符号、labelIndex 无属主、
> 特性 install() 无调用点、diceBarrier.init() 无调用点、token 来源不全、status() 形状自相矛盾、pools 取值歧义）。
> 本版逐条补齐。带 **[v3]** 的是那一轮新增，带 **[v2]** 的是更早一轮的改动。冲突处一律以最高版本号为准。
>
> **v3.1 说明**：18 个任务各自定稿后，接线点名审计抓出 20 个装配级缺陷 —— 其中
> `registry.resolveAll()` 与 `cards.init()` **没有任何任务插入**、`api` 八个槽只有三个有装配人、
> `LABEL_KEYS` 被两个模块各定义一份、以及 **rollBus 与两条 P0 修复抢同一个 libWrapper 目标导致后者静默失效**。
> 带 **[v3.1]** 的条目是那一轮的裁决。
>
> **v3.2 说明**：收尾轮的三个关卡又抓出三条。其中两条「T3/T7 整行替换 api」经主控**逐行复核为误判**
> （两个任务写的都是单槽子串替换，关卡把后面的「改完长这样」示意块读成了替换指令）；
> 真正成立的是 `around` 的短路语义在写进 v3.1 时被漏掉半句，而两条 P0 修复正需要它。
> 另裁决三条散落项：selftest 的 label 走 i18n 键、桩负责提供模板通路、LABEL_KEYS 按可达性逐键裁决。

本文件由主控作者，**不得改动**。任务之间的类型一致性完全靠它。
任何任务若需要一个此处未定义的符号，说明任务划分错了 —— 在你的产出里明确指出，不要自行发明。

## 0. 仓库与工具链

```
C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\alien-evolved-automation\
├─ module.json
├─ package.json                     # vitest 唯一 devDependency
├─ vitest.config.mjs
├─ lang/en.json  lang/cn.json       # 嵌套结构，模组自有键一律 AEA.* 前缀   [v2]
├─ styles/alien-evolved-automation.css                                      [v2]
├─ templates/*.hbs
├─ scripts/
│  ├─ main.mjs                      # 生命周期钩子的唯一挂载处；export const api
│  ├─ const.mjs
│  ├─ kernel/
│  │  ├─ features.mjs   (K5)
│  │  ├─ patches.mjs    (K7)
│  │  ├─ resolver.mjs   (K4)
│  │  ├─ registry.mjs   (K2)
│  │  ├─ record.mjs     (K1 纯函数层)
│  │  ├─ rollbus.mjs    (K1 副作用层)
│  │  ├─ dice-barrier.mjs (K6)
│  │  ├─ cards.mjs      (K8)
│  │  └─ selftest.mjs   (自检套件)                                          [v2]
│  ├─ features/*.mjs                # 一期五条特性
│  └─ repairs/*.mjs                 # 一期 b 修复包
└─ test/
   ├─ stubs/foundry.mjs             # Foundry 全局桩，导出见 §0.3
   └─ *.test.mjs
```

### 0.1 分层铁律

`kernel/record.mjs`、以及各模块中以 `pure` 开头导出的函数**不得引用任何 Foundry 全局**
（`game` / `ui` / `canvas` / `CONFIG` / `Hooks` / `foundry` / `ChatMessage` / `Roll` / `libWrapper`）。
这些是 vitest 直接跑的部分。副作用层负责把 Foundry 的东西喂给它们。

### 0.2 钩子归属 **[v2]**

v1 写「main.mjs 是唯一挂 Hooks 的地方」，三个组同时指出这不可实现。修订为：

- **生命周期钩子**（`init` / `i18nInit` / `setup` / `ready` / `diceSoNiceReady`）**只**在 `main.mjs` 挂。
- **领域钩子**由拥有它的内核模块自己挂，且只在该模块的 `init()` / `install()` 里挂一次：
  - `preCreateChatMessage`、`createChatMessage` → `rollbus.mjs`
  - `diceSoNiceRollComplete` → `dice-barrier.mjs`
  - `renderChatMessageHTML`（V14）与 `renderChatMessage`（V13 回退）→ `cards.mjs`
- 特性与修复**不得在模块顶层或 `install()` 里挂钩子**；它们订阅 `HOOK_ROLL_RESOLVED`，或经
  `cards.registerAction` / `cards.onRender` 注册。
  **[v3.1] 唯一例外**：`type: "HOOK"` 的**已登记补丁**可以在自己的 `apply()` 里挂领域钩子
  （本节判给内核独占的三组除外）。这条例外让「飞船阶段提交」「preCreateToken」这类非函数包裹的修复
  有合法落点 —— 它与 §4 K7 的 `apply()` 自装语义配套，两处不再自相矛盾。

### 0.3 测试桩 **[v2]**

`test/stubs/foundry.mjs` 导出三个符号，所有任务统一使用，不得各自就地造全局：

```js
export function installFoundryStub(options = {}) // -> ctx，装 globalThis 上的 game/ui/Hooks/
                                                 //    ChatMessage/Roll/CONFIG/CONST/libWrapper/
                                                 //    logger/foundry.{utils,applications}
export function uninstallFoundryStub()           // 还原
export function foundryStubContext()             // -> ctx|null，取当前桩上下文
```

**[v3.1] 桩的行为契约**（每条都是被测代码真正依赖的行为。桩由 Task 1 独占实现并交付
`test/stub-fidelity.test.mjs` 逐条守卫；**其余任务只读不改，禁止在自己的测试文件里就地造
`globalThis.game` 或用私有 Map 顶替 `game.settings`**）：

- **幂等**：`installFoundryStub()` 可重复调用（第二次先隐式卸载再装）；`uninstallFoundryStub()` 未安装时是安全空操作。
- **设置是真后端**：`game.settings.register/get/set` 由 `ctx.settings`（Map）支撑；`get` 未注册键抛错，
  `set` 返回 Promise；`registerMenu` 记进 `ctx.menus`。
- **钩子会派发**：`Hooks.callAll(name, ...args)` 真正调用经 `Hooks.on/once` 注册的回调，并把调用记进 `ctx.hooks.calls`。
- **文档可解析**：`ctx.documents`（Map: uuid 到文档）支撑 `fromUuidSync` 与 `fromUuid`；未命中返回 `null`。
- **消息可创建**：`ChatMessage.create` 把文档推进 `ctx.messages`，并**先同步派发 `preCreateChatMessage`
  再派发 `createChatMessage`**；传入的文档实例带可用的 `updateSource()`。
- **版本可读**：`game.system.version`、`game.release.generation` 由 `options` 指定，默认 `"4.1.13"` 与 `14`。
- **[v3.2] 模板通路由桩提供**：`foundry.applications.handlebars = { loadTemplates, renderTemplate }`，
  两者把调用记进 `ctx.templates`，`renderTemplate` 返回可由 `options` 配置的 HTML 串（默认空串）。
  这是桩的职责而非被测代码的探针位 —— `ctx.templates` 本来就在字段表里，且已有两个消费者。
  任何任务在自己的测试文件里给 `globalThis.foundry.applications.handlebars` 赋值都属于「就地补桩」，禁止。
- **包裹可观察**：`libWrapper.register` 记进 `ctx.wrappers`，重复注册同一目标时抛出与真实 libWrapper 同形的错误。
- `ctx` 字段固定为：`moduleId`、`isGM`、`userId`、`systemVersion`、`i18n`、`settings`、`registered`、`menus`、
  `hooks{once,on,calls}`、`notifications`、`wrappers`、`messages`、`rolls`、`templates`、`modules`、`documents`、`babele`、`world`。

### 0.4 测试诚实性

- `pure*` 函数与 `record.mjs`：**真单测**，无桩或仅用最小对象字面量。
- 需要 Foundry 但可用桩忠实模拟的：用 `installFoundryStub()` 写真测试。
- 桩无法忠实模拟的（真实的 libWrapper 链、DsN 动画、渲染时序）：写进 `selftest.mjs`
  **并且**在任务里给出 **MANUAL VERIFICATION** 步骤，逐条列出在 Foundry 里要做的动作与要观察的结果。
- **禁止**为跑不动的东西编造单测。

测试命令：`npm test`（= `vitest run`）。单条：`npx vitest run test/<file> -t "<name>"`。

## 1. scripts/const.mjs（Task 1 建立，全部任务引用）

```js
export const MID = "alien-evolved-automation";
export const FLAG_ROLL = "roll";
export const HOOK_ROLL_RESOLVED = "aea.rollResolved";
export const RECORD_VERSION = 1;
export const SETTING_FEATURES = "features";
export const SETTING_BINDINGS = "registryBindings";
export const SETTING_SCHEMA = "settingsSchemaVersion";
export const SETTING_DATA_REPAIRS = "dataRepairs";        // [v2]
export const SETTING_DICE_TIMEOUT = "diceTimeoutMs";      // [v2] 默认 4000
export const SYSTEM_ID = "alienrpg";
export const I18N = "AEA";                                 // [v2] 模组自有 i18n 键前缀
```

`const.mjs` **不得**导出以上之外的符号（Task 1 带一条护栏断言）。

## 2. 被包裹函数的精确签名（源码事实，勿改）

```js
// systems/alienrpg/module/helpers/YZEDiceRoller.mjs:31
static async yzeRoll(
  actortype, blind, reRoll, label,
  r1Dice, col1, r2Dice, col2,
  actorid, itemid, tactorid, moddata
)
```

- 类对象同时可达 `game.alienrpg.yze`（`alienrpg.mjs:79`）。22 个调用点全部 `yze.yzeRoll(...)`，零解构。
- 内部在 `:416` 自行 `ChatMessage.create(chatData)`，`:417` `return`（返回 `undefined`）。
- `chatData.flags = { tactorid }`（扁平，非命名空间）。
- 两个不建消息的提前返回：`:117`（NoAttribute）、`:141`（NoDice）。
- 自动恐慌在 `:198-242` 递归触发 `rollResolve` / `rollPanic`，**恐慌卡先于触发卡创建**。
- 全局 `game.alienrpg.rollArr = { r1Dice, r1One, r1Six, r2Dice, r2One, r2Six, tLabel, sCount, multiPush }`（`alienrpg.mjs:96-106`），每次调用在 `YZEDiceRoller.mjs:107-114` 逐字段清零。
- **[v2] 关键事实**：`abilityRoll` 调 `yzeRoll` 时**只传 9 个参数**（见 `actor.mjs:305-315`、`345-355`、`359-371`），
  因此技能／属性路径上 `itemid`、`tactorid`、`moddata` **全部是 `undefined`**。
  `dataset.attr` 只存在于 `abilityRoll` 的入参里（`actor.mjs:194` 读进局部变量后丢弃），永远到不了 `yzeRoll`。

## 3. RollRecord（v1，schema 定死）

```js
/**
 * @typedef {object} RollRecord
 * @property {1} v
 * @property {string} id
 * @property {string|null} actorUuid
 * @property {string|null} tokenUuid
 * @property {string} userId
 * @property {"attribute"|"skill"|"weapon"|"armor"|"supply"|"other"} kind   // [v2] 见下
 * @property {string} label
 * @property {string|null} labelKey
 * @property {string|null} attr
 * @property {string|null} itemUuid
 * @property {{base:number, stress:number}} pools
 * @property {{baseSixes:number, baseOnes:number, stressSixes:number, stressOnes:number}} results
 * @property {number} successes
 * @property {number} banes
 * @property {{count:number, pushable:boolean, parentRollId:string|null}} push
 * @property {string[]} targets
 * @property {{ammo:number|null}} consumed
 * @property {{worldTime:number, real:number}} at
 */
```

存放位置：`message.flags[MID][FLAG_ROLL]`。

**[v3.1] 写入机制定死**：rollBus 在 `preCreateChatMessage(document, data, options, userId)` 里**只能**用

```js
document.updateSource({ [`flags.${MID}.${FLAG_ROLL}`]: record });
```

钩子的第一个参数是**尚未落库的文档实例**：直接给 `data` 赋值不会进库，`document.flags[...] = x` 也不会。
本机两个已装模组（`dice-chronicle/scripts/tracker.js:14`、`pf2e-target-helper/src/main.js:97`）用的都是这个写法。
**禁止**在 `createChatMessage` 之后用 `message.update()` 回填 —— 那是多一次写库，而且卡片会闪一下。

**[v2] 三条 v1 范围裁决**（都是起草阶段查证出来的真实限制，写进文档而不是假装能做到）：

1. `kind` 枚举**去掉** `panic` / `stress` / `crit` / `ammo`：这四条路径（`rollPanic` `actor.mjs:545`、
   `rollResolve` `:840`、`rollStress` `:1061`、重伤链 `:1780+`）根本不经过 `yzeRoll`，一期无从产出。
   它们在二期各自的特性里补，届时枚举**追加**成员，`RECORD_VERSION` 不变（追加枚举向后兼容）。
2. `consumed.ammo` 一期**恒为 `null`**：弹药子掷骰在 `YZEDiceRoller.mjs:571-611` 就地 `new Roll` 并直接
   `weapon.update()` 扣弹（`:606-609`），扣了几发从不写进 `rollArr`，包装器在 `preCreateChatMessage` 时刻看不见。二期随 `ammo-economy` 补。
3. `targets` 一期恒为 `[]`，二期随 `attack-context-binding` 填。

`attr`、`labelKey`、`push.parentRollId` 三个字段**可以**在一期产出，但必须靠 §4/K1 的四入口包裹拿到，见下。

## 4. 各内核模块的导出（签名不得改动）

### K5 `kernel/features.mjs`
```js
export function pureResolveMode(id, modes, defs)   // 纯：算上 requires 传递闭包后的有效档
export const features = {
  register(def),        // def = {id, default:"full"|"prompt"|"off", gmOnly:false, requires:[], hint:""}
  registerSettings(),   // init 调用
  mode(id),             // -> "full"|"prompt"|"off"
  enabled(id),          // -> boolean（mode !== "off" 且 requires 链上无 "off"）
  all(),                // -> def[]
};
```
显示名与说明走 i18n：`AEA.feature.<id>.name` / `AEA.feature.<id>.hint`，`game.i18n.has()` 为假时退回裸 id。 **[v2]**

### K7 `kernel/patches.mjs`
```js
export function pureVersionApplies(systemVersion, {minSystem, fixedIn})  // 纯 -> boolean
export const patches = {
  register(def),   // 见下
  applyAll(),      // -> {applied:string[], skipped:string[], retired:string[]}
  status(),        // [v3] -> [{id, type, target, applied, reason, fixedIn}]
};
```

**[v2] def 语义定死（v1 的歧义是三个组冲突的根源）**：

```js
// def = {
//   id:       string
//   type:     "WRAPPER" | "MIXED" | "OVERRIDE" | "DATA" | "HOOK"   // DATA/HOOK 是 v2 新增
//   target:   string|null    // 仅元数据，供 status() 展示；applyAll 不据此注册
//   minSystem: string, fixedIn: string|null
//   probe():  boolean        // true = 缺陷仍在（该装）；false = 上游已修（退休并提示 GM 一次）
//   apply():  void|Promise   // 无参安装器，自己负责调 libWrapper.register(MID, target, fn, type)
//                            // 或自己挂钩子、或自己改数据。applyAll 只调它。
// }
```

理由：一期 b 有三类修复不是函数包裹（纯数据修复、宏安装、`preCreateToken` 钩子），
`target`/`type` 驱动的自动注册无法表达它们。**`apply()` 自装** 是唯一能覆盖全部五类的语义。

### K4 `kernel/resolver.mjs`
```js
export function purePickRef(speaker, lookups)  // 纯；lookups = {token(id), messageActor(), actor(id)}
                                               // -> {actorUuid, tokenUuid, source:"token"|"message"|"actor"|"none"}
export const resolver = {
  fromSpeaker(speaker),          // -> Actor|null
  refs(actor, token),            // -> {actorUuid, tokenUuid}
  actorOf(record),               // -> Actor|null
  soleToken(actor),              // [v3.1] -> Token|null：getActiveTokens() 恰好一个时返回它，否则 null；不猜
  actorById(id, {warn = true}),  // [v2] 遗留路径专用：载具乘员槽存的是裸 actor id
};                               //      （module/data/actor-vehicle.mjs 的 crew.occupants[].id）
```

### K2 `kernel/registry.mjs`
```js
export function pureChooseBinding(candidates, guesses)  // 纯；candidates=[{uuid,name}] -> {uuid,name}|null
                                                        // 精确匹配 > 忽略大小写 > 去空白；歧义返回 null，不猜
export const registry = {
  declare(key, {kind:"table"|"folder"|"journal", docType:null, guess:[String]}),  // [v2] docType 供文件夹过滤
  registerSettings(),   // [v2] init 调用，显式注册世界设置与重绑菜单
  resolveAll(),         // async，ready 阶段
  table(key), folder(key), journal(key),
  tableByLegacyName(name),   // [v2] 动态路径：把一个遗留显示名映射到已绑定的键
  bind(key, uuid),      // async
  bindings(), unbound(),
};
```

**[v2] 必须声明的绑定键扩到 13 个**（v1 只给了 8 个，起草阶段在源码里数出 13 个按名查找点）：

`panic`、`stressResponse`、`panicResponse`、`critInjury1e`、`critInjuryEvolved`、
`critInjurySynthetic`、`critInjuryXeno`、`shipMinorComponent`、`shipMajorComponent`、
`folderAlienTables`、`folderCreatureTables`、`folderMotherTables`、`journalMother`。

`actor.mjs:1839` 的 `game.tables.getName(dataset.atttype)` 是**运行时动态名**，走 `tableByLegacyName()`。

### K1 `kernel/record.mjs`（纯）
```js
export function classifyKind(args, ctx)            // -> RollRecord["kind"]
export function successesOf(rollArr)               // -> number
export function banesOf(rollArr)                   // -> number
export function pureReverseLabelKey(label, index)  // [v2] 已本地化文本 -> i18n 键；歧义返回 null
export function buildLabelIndex(localize, keys)   // [v3] 纯：localize 是注入的函数，keys 是 i18n 键数组
                                                  //      -> {localizedText: key}；同文本多键则该文本映射到 null
export const LABEL_KEYS = [/* ... */]             // [v3.1] 反查标签用的 i18n 键表，纯数据
// [v3.2] 收键的判据是**可达性，不是数量**：一个键进表，当且仅当它本地化后的文本
//        可能出现在 §2 那 12 个参数里的 `label` 位置上（依据是 22 个调用点的实际传参，
//        不是「有没有人给它写断言」）。每次增删都要在任务里写明对应的调用点。
//        少收一个键 = 那条标签反查不到、`RollRecord.labelKey` 静默变 null。
// [v3.1] LABEL_KEYS 只此一份，属主是 record.mjs。rollbus.mjs 与 main.mjs 一律 import 它，
//        **不得**各自定义同名符号（上一轮 record 与 rollbus 各定义了一份，键数还不一致）。
export function buildRollRecord(input)             // -> RollRecord
// input = {args, rollArr, refs, ctx, userId, worldTime, now, id, labelIndex}
// args = §2 那 12 个同名参数组成的对象（技能路径上后三个是 undefined）
// ctx  = §4/K1 四入口包裹采集到的上下文，见下
```

### K1 `kernel/rollbus.mjs`

**[v2] 包裹四个入口，不是一个。** `yzeRoll` 是汇点，但它丢掉了三样下游必需的东西
（`attr`、`itemUuid`、推骰父卡），这三样只在上游函数的入参里存在：

```js
export const rollBus = {
  install(),            // ready：libWrapper 注册四处
  recordOf(message),    // -> RollRecord|null
  depth(),              // -> number，当前 yzeRoll 调用栈深度（用于恐慌递归归属）
  context(),            // -> ctx|null，当前上下文帧
  setLabelIndex(index), // [v3] main.mjs 在 i18nInit 阶段注入 buildLabelIndex() 的产物
  addStage(target, {id, order = 0, around}),  // [v3.1] 见下
};
```

**[v3.1] rollBus 独占这四个 libWrapper 目标。** lib-wrapper 1.13.5 对同一目标的重复注册会抛
`A wrapper for '<target>' (ID=<n>) has already been registered by <module>.`，而
`CONFIG.Item.documentClass = alienrpgItem`（`alienrpg.mjs:138`）与 rollBus 要包的是同一个方法 ——
若某条修复也去 `libWrapper.register` 同一目标，**它会静默变成空操作**（上一轮有两条 P0 修复正是如此）。

因此：**任何需要介入这四个目标的修复或特性，一律经 `rollBus.addStage()` 排队，不得自己 register。**

**[v3.2] 短路是一等语义，不是违例。** `gear-active-gate` 要在装备未激活时拦下整次掷骰、
`vehicle-roll-path-repair` 要接管整条炮击路径 —— 两条 P0 修复都必须能不调 `next` 就返回。
`rollBus` 的链式执行器必须：阶段不调 `next` 时**原样返回该阶段的返回值、绝不代它调用下一层**；
阶段调用 `next` 两次时抛错并指名该阶段 id。这两条各配一条单测。

```js
// target 取值："yzeRoll" | "abilityRoll" | "itemRoll" | "pushRoll"
// around(next, args, thisArg) -> any
//   [v3.2] 要么恰好调用一次 next(args)（可改 args、可改返回值），
//          要么**有意短路**：不调 next，其返回值即整条链的最终返回值。
//          两次调用 next 永远禁止。
// order 小的先执行（更靠外）；同 order 按注册顺序
```

| 包裹目标 | 类型 | 采集什么 |
|---|---|---|
| `game.alienrpg.yze.yzeRoll` | WRAPPER | 汇点：置栈深度、armed 标志，内层返回后**同步**快照 `rollArr` |
| `CONFIG.Actor.documentClass.prototype.abilityRoll` | WRAPPER | `ctx.attr = dataset.attr`、`ctx.actor`、`ctx.token` |
| `CONFIG.Item.documentClass.prototype.roll` | WRAPPER | `ctx.itemUuid = this.uuid`、`ctx.dataset = dataset`、**[v3]** `ctx.actor = this.actor`、`ctx.token = this.actor?.token ?? soleActiveToken(this.actor)` |
| `CONFIG.Actor.documentClass.prototype.pushRoll` | WRAPPER | `ctx.parentRollId = recordOf(message)?.id`、`ctx.pushCount`、**[v3]** `ctx.actor` / `ctx.token` 从父记录的 `actorUuid`/`tokenUuid` 继承 |

上下文帧是一个栈（`ctx` 随 `depth()` 进出），因为恐慌会递归。

**[v3] `tokenUuid` 的诚实边界**：卡是 `ChatMessage.getSpeaker({actor: actorid})` 建的（`YZEDiceRoller.mjs:398-401`），
token 在消息存在之前就被丢掉。所以 `tokenUuid` 只在三种情况下非空：掷骰源自 token 表单、
actor 在当前场景恰好只有一个活动 token（`soleActiveToken`）、或推骰从父记录继承。其余情况为 `null` 并如实记录，
**不得**用 `game.actors.get()` 兜底伪造。

**[v3] `pools` 的取值来源定死**：`buildRollRecord` 的 `pools.base` / `pools.stress` 一律取自
`rollArr.r1Dice` / `rollArr.r2Dice`（实际掷出的池子），**绝不取自 `args.r1Dice` / `args.r2Dice`**。
理由：骰池钳制补丁是 MIXED 类型、按 libWrapper 顺序跑在 rollBus 的 WRAPPER **内层**，
所以汇点包装器看到的 `args` 是**钳制前**的值，只有 `rollArr` 反映真正掷出去的骰子。
发出钩子：`Hooks.callAll(HOOK_ROLL_RESOLVED, record, message)`。

### K6 `kernel/dice-barrier.mjs`
```js
export function pureBarrierPlan({dsnActive, isRecipient, timeoutMs})  // 纯 -> {wait:boolean, timeoutMs:number}
export function pureIsRecipient({whisper, blind, userId, isGM})       // [v2] 纯，可单测
export const diceBarrier = { init(), awaitDice(message) /* -> Promise<void> */ };
```
超时读 `SETTING_DICE_TIMEOUT`，默认 4000ms。

### K8 `kernel/cards.mjs`
```js
export const cards = {
  init(),
  mount(element, message),          // 幂等，返回模组自有的 <div class="aea-mount">
  render(target, templatePath, data),// [v2] 特性塞内容的唯一入口；[v3.1] 返回 Promise，语义见下
  registerAction(name, handler),    // handler(event, {action, message, messageId, element, mount})  [v2]
  onRender(name, handler),          // [v3] 渲染扇出点，见下
};
```
- **禁止**使用系统的 `dmgBtn-container` 挂点：它只在 `YZEDiceRoller.mjs:378` 的
  `if (!reRoll || reRoll === "mPush")` 分支内发出，推骰卡／NPC 卡／怪物卡上不存在。
- **[v3.1] `render(target, templatePath, data)` 只替换 `target` 自身的 `innerHTML`**，绝不触碰它的父节点、
  兄弟节点或挂点上的其它内容；返回 Promise（内部 `await renderTemplate`）。
  一期有两条特性共用同一个挂点，所以**每条特性必须先在 `mount` 下建自己的子元素**
  （类名 `aea-<feature-id>`，用 `mount.querySelector` 复用、没有才创建），再对那个子元素调 `render()`。
  直接对 `mount` 调 `render()` 会清掉另一条特性的内容，症状还取决于 `onRender` 的注册顺序。
- **[v3] `onRender(name, handler)` 是特性拿到挂点的唯一通路。** §0.2 把渲染钩子判给 `cards.mjs` 独占、且禁止特性自挂钩子，
  所以必须由 cards 在自己的渲染钩子里按注册顺序回调各特性：
  `handler({ message, element, mount, record })`，其中 `record` 是 `rollBus.recordOf(message)` 的结果（可能为 `null`）。
  handler 同步执行、异常被 cards 捕获并记 `console.error` 后继续调用后一个，绝不让一条特性打断整张卡的渲染。
  `mount` 由 cards 惰性创建（只有 handler 第一次访问才插入 DOM），因此不产生空挂点。
- **[v2]** 委托监听挂在 `document` 上、以 `.aea-mount` 作用域守卫，而非字面挂 `#chat-log`：
  ChatLog 重渲染会整体替换 `#chat-log` 元素，且聊天弹出窗口有第二个同 id 元素。行为是 `#chat-log` 委托的严格超集。

### 自检 `kernel/selftest.mjs` **[v2]**
```js
export const selftest = {
  register({id, label, run}),   // run() -> {ok:boolean, detail:string}，可 async
                                // [v3.2] label 是 i18n 键（`AEA.selftest.<id>`），不是英文字面量。
                                // register 原样保存 def、**不得**在登记时本地化；
                                // runAll()/results() 返回的条目里由运行器 localize(def.label)。
                                // 混用字面量会让自检面板中英分裂。
  runAll(),                     // -> [{id, label, ok, detail}]
  results(),
};
```
`patches` 的 `probe()` 与自检条目共用同一批断言：一份代码两用（设计文档 §6.2）。

## 5. main.mjs 的生命周期顺序 **[v2 有改动]**

**[v3] `main.mjs` 由 Task 1 建立，且必须一次写出全部四个生命周期钩子与四个锚点注释。**
后续任何任务往里插代码，一律**按锚点文本定位**（不用行号），插在锚点注释的上一行或下一行：

**[v3.1] 锚点从四个扩到十个，ready 段拆成有序子锚点。** 上一轮四个任务都声明「插在 ready 锚点之后」，
谁先谁后完全靠运气；而 import 区与 FEATURES/REPAIRS 数组根本没有锚点，四个任务各自引用了四种定位文本。
Task 1 必须逐字写出下面这个骨架（`safely` 与 `waitForWorldSettled` 由 Task 1 实现）：

```js
// scripts/main.mjs —— Task 1 必须逐字写出全部十个锚点
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
Hooks.once("i18nInit",        () => { /* AEA-ANCHOR: i18nInit */ });
Hooks.once("diceSoNiceReady", () => { /* AEA-ANCHOR: diceSoNiceReady */ });
Hooks.once("ready", async () => {
  await waitForWorldSettled({ timeoutMs: 10000, pollMs: 100 });
  /* AEA-ANCHOR: ready.registry */
  /* AEA-ANCHOR: ready.patches */
  /* AEA-ANCHOR: ready.rollbus */
  /* AEA-ANCHOR: ready.cards */
  for (const f of FEATURES) await safely(`feature ${f.id} install`, () => f.install());
  for (const r of REPAIRS)  await safely(`repair ${r.id} install`,  () => r.install?.());
});
```

**[v3.1] `api` 的装配规则**：Task 1 逐字写出八个键、值全为 `null`（此刻内核文件尚不存在，import 会让整个模组加载失败）。
每个内核模块的属主任务做三件事、一件不多：在 `/* AEA-ANCHOR: imports */` 后加一行 import、
把 `api` 里自己那一个 `null` 换成该对象、在自己的 ready 子锚点后插自己的调用。
**不得**替换 `api` 这个对象本身，**不得**增删键。

**[v3.1] 每个 ready 子锚点的归属与内容（一一对应，无歧义）**：

| 子锚点 | 属主任务 | 插入内容 |
|---|---|---|
| `ready.registry` | K2 DocRegistry | `await registry.resolveAll();` |
| `ready.patches`  | K7 Patches     | `await patches.applyAll();` |
| `ready.rollbus`  | K1 RollBus     | `rollBus.install();` |
| `ready.cards`    | K8 Cards       | `cards.init();` |

各阶段最终要包含的调用（每一行都必须在某个任务里有明确的插入步骤，没有主人的行就是计划缺陷）：

**[v3.1] init 段顺序修正**：v3 把 `features.registerSettings()` 排在 `f.register()` 之前，
而 `registerSettings` 要按已注册的 def 生成设置项 —— 那时一条 def 都没有，所有特性档位设置永远注册不上，
运行期 `features.enabled(id)` 读未注册键会抛。**注册 def 必须排在注册设置之前**：

```
init            → for (const f of FEATURES) f.register()      // [v3.1] 移到最前
                  for (const r of REPAIRS)  r.register()      // [v3.1]
                  features.registerSettings()                 // [v3.1] 此刻 def 已在册
                  registry.registerSettings(); registry.declare(...) ×13
                  patches.register(...) ×N; selftest.register(...) ×N
i18nInit        → rollBus.setLabelIndex(buildLabelIndex(game.i18n.localize.bind(game.i18n), LABEL_KEYS))  // [v3]
diceSoNiceReady → diceBarrier.init()
ready           → await waitForWorldSettled({timeoutMs: 10000, pollMs: 100});
                  await registry.resolveAll(); await patches.applyAll();
                  rollBus.install(); cards.init();
                  for (const f of FEATURES) f.install()       // [v3]
                  for (const r of REPAIRS)  r.install?.()     // [v3]
```

**[v3] `FEATURES` / `REPAIRS` 是 `main.mjs` 里的两个显式数组**（`import` 各模块后列进去），
不是注册表方法 —— 契约不发明 `features.installAll()`。每个特性/修复任务的最后一步就是把自己加进对应数组。

**[v3] `api` 必须暴露内核全部成员**，供自检与手工验证使用：
`api = { features, patches, resolver, registry, rollBus, diceBarrier, cards, selftest }`。

`ready` 必须**排在 Babele 与系统冒险导入之后**。做法：在 `Hooks.once("ready")` 里
先 `await` 一个就绪守卫 —— 检测 `game.babele?.initialized`（不存在则跳过），
并读 `game.settings.get(SYSTEM_ID, "ARPGSemaphore")` 确认系统的 `apps/init.mjs` 首次导入已完成，
再解析绑定。守卫本身要有超时与降级（超时则照常解析并记警告）。

## 6. 特性与修复的模块形状 **[v2]**

```js
// scripts/features/<feature-id>.mjs
export const <camelId>Feature = {
  id: "<feature-id>",       // 与 features.register 的 id 一致
  register(),               // init 阶段：features.register(def)，必要时 patches.register(def)
  install(),                // ready 阶段：订阅 HOOK_ROLL_RESOLVED / cards.registerAction
};
```
纯逻辑放同名 `<feature-id>.pure.mjs`，只导出 `pure*` 函数。修复包同形，放 `scripts/repairs/`。

## 7. 通用要求（每个任务隐含包含）

- 代码与标识符全英文；面向用户的字符串一律 `game.i18n.localize()`，键写进 `lang/en.json` 与 `lang/cn.json`，前缀 `AEA.`，嵌套结构。
- 任何 RollTable／状态／文档查找一律经 `registry`，**永不 `getName()`、永不比对显示名**。
- 任何 actor 解析一律经 `resolver`（遗留裸 id 用 `resolver.actorById`），**永不 `game.actors.get(speaker.actor)`**。
- 成功数一律读 `RollRecord`，**永不解析渲染文本、永不读 `Roll#total`**。
- 每个特性必须 `features.register()` 并在执行前查 `features.enabled(id)`。
- 每个系统缺陷修复必须 `patches.register()` 并带可运行的 `probe()`；同一断言登记进 `selftest.register()`。
- **[v3.1] 一期 b 的每一条修复也必须有开关**：在 `register()` 里 `features.register({id, default:"full", gmOnly:true})`，
  并在 `apply()` 装上的包裹/钩子**执行时**查 `features.enabled(id)` —— 关掉即原样放行，不需要重载世界。
  设计决策 2（默认全自动、逐项可关）对修复同样成立；上一轮有五条修复既无开关也无自检。
- **[v2] `socketlib` 一期写进 `relationships.recommends` 而非 `requires`** —— 一期没有任何消费者（K3 Executor 在二期），不该强迫用户装一个用不上的库。二期提升为 `requires`。
- 提交信息中文正文，结尾附：
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  ```
