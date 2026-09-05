# Alien Evolved: Automation —— 设计文档

- 日期：2026-09-05
- 状态：已定稿，待实施计划
- 目标系统：`alienrpg` 4.1.13（标题 "Alien Evolved"，作者 pwatson100），Foundry V13–V14
- 配套文档：`2026-09-05-alien-evolved-automation-gap-inventory.md`（116 条特性完整差集清单）

---

## 1. 缘起与普查结论

要做一个 Foundry 模组，把 Alien RPG（Evolved 版）规则书要求、而 `alienrpg` 系统尚未自动化的部分补齐。

普查方式是双向遍历后取差集：

| 侧 | 材料 | 产出 |
|---|---|---|
| 规则侧 | `alien-evolved-corerules` 全书正文 88.7 万字符，切成 36 块 | **1726 条规则原子** |
| 系统侧 | 35k 行源码，划成 21 个审计区 | **631 条已实现能力** |

631 条能力按实现程度分布：full 137 / partial 231 / broken 138 / data-only 74 / manual-only 31 / legacy-1e 20。

差集经三轮收敛与验证：

```
1726 条规则原子
  → 1268 条候选缺口       （逐块对照能力索引，每条要求 grep 证据）
  → 391 条候选特性         （48 个批次合并同类项）
  → 119 条定稿特性         （6 个主题去重、定级、定落点）
  → 116 条存活             （对抗反驳杀掉 3 条；29 条实现路径被判需重做）
```

其中 42 条被判「系统本体写错了」的特性，又做了一轮逐条复现核查（要求给出具体操作路径与玩家可见症状）：

- **327 条独立缺陷断言 → 269 条坐实 / 43 条误判 / 14 条属于没有入口能调到的死代码 / 1 条存疑**
- 按严重度：崩溃 37、静默算错数 53、功能是死的 106、观感 36、无影响 95
- **19 条特性因此改级**，42 条 broken 中只剩 9 条真 broken

最终分层：**P0 17 · P1 42 · P2 42 · P3 15**，其中 **35 条够格作为独立 PR 提给上游**。

### 1.1 核心结论：这不是功能缺失问题，是地基缺失问题

116 条特性里约 90 个才是独立工作量——清单按章节拼装，导致同一个东西被独立发现多次（注册表 4 次、时钟 5 次、对抗判定 5 次、Broken 状态 2 次）。更要紧的是，**几乎每一条最终都撞上同一批不存在的地基**。

四条已由人工 grep 复核的事实：

1. **系统不发任何掷骰钩子。** 全系统仅 5 处 `Hooks.call`（`active-effect.mjs:106`、四处 `drop*SheetData`），无一与掷骰有关。唯一可拦截点是 `yze.yzeRoll`。
   *好消息*：`export class yze { static async yzeRoll }`，22 个活跃调用点（另有 7 个 1e 遗留层）**全部写作 `yze.yzeRoll(...)`，零解构**；各处 `import { yze }` 拿到同一个类对象引用，`game.alienrpg.yze` 也是它 —— 改静态属性对全部调用点生效。

2. **掷骰结果只存在一个全局可变对象里。** `game.alienrpg.rollArr`（`alienrpg.mjs:96-106`），`YZEDiceRoller.mjs:107-114` 在每次掷骰开头逐字段清零。而 `actor.mjs:316/356/370` 在**未 await** 的调用之后立刻读它，读到的是零。
   `pushRoll`（`actor.mjs:1302-1327`）从这个全局算重掷池并使用 `rollArr.tLabel` 而非传入的 message —— 所以「推骰会重掷别人的骰子」不是推骰的 bug，是这个全局的 bug；而且它**已经先扣了压力**才出错。

3. **系统完全没有 socket。** `grep -rni socket module/` 零命中。因此一切「玩家动作自动改 GM 拥有的文档」（自动扣血、自动上状态、生成 token、向旁观者广播恐慌）今天**不是半成品，是完全阻塞**。清单里这些条目标注的 `manual-only` / `none` 低估了工作量。

4. **一切文档查找按显示名。** `game.tables.getName("Panic Table")`（`actor.mjs:554`）、`"Stress Response Table"`（:845）、`"Panic Response Table"`（:1067）、重伤链（:1812-1817, :1830）、`getName(dataset.atttype)`（:1839）、两处飞船部件表（:1848, :1855）、`contents.find(b => b.name === targetTable)`（:2497），外加 `game.folders.getName("Alien Tables")` 与 `game.journal.getName("MU/TH/ER Instructions.")`。
   怪物 actor 的 `system.rTables` / `system.cTables` 直接存字面表名字符串。
   这些表在**世界里**（由 adventure pack 导入），所以这不只是 Babele 合集翻译的问题——手改一个表名同样会打断整条链。

### 1.2 已知的样本缺陷（人工坐实，用作校准）

- `module/documents/actor.mjs:1743` —— 补给消耗写回是 `Number("system.consumables.<x>.value" - tNum || 0)`，其中第一项是模板字符串。模板字符串减数字 = `NaN`，`NaN || 0` → 写入字面 `0`。**只要消耗后还有剩余，该项消耗品直接被清零**，静默无报错。
- `module/documents/actor.mjs:1104-1112` —— 恐慌上限：`rollTotal >= 12 || oldPanic >= 12` 钳到 12，紧接着 `if (rollTotal <= oldPanic)` 又设为 `oldPanic + 1` = **13**。`table.getResultsForRoll(13)` 返回空数组，下一行 `customResults[0].description` 抛异常。**角色恐慌满级后再恐慌就崩**。
- `module/documents/actor.mjs:1876-1941` —— 重伤表结果按 `split(/[:] |<br>/)` 切碎后取 `testArray[3]/[5]/[9]` 三个魔法下标，与**带尾随空格**的本地化字符串比对，其中两处用 **EN DASH U+2013**，另有裸字面量 `"Shift"`。表格文案被翻译或改一个标点 → `cFatal` 恒 `false`、`healTime` 恒 `0`，静默失效。

### 1.3 两类已证伪的误判（写给后续实施者，避免重犯）

- **DOMStringMap 陷阱**：`character-sheet.mjs:758` 是 `const dataset = target.dataset`，这是真正的 `DOMStringMap`。赋值会把 Number/boolean **强制转成字符串**，所以 `dataset.shootrangeMod = Number(x)` 之后 `=== "1"` 成立、`dataset.conserveammo = true` 之后 `=== "true"` 成立。两条「死代码」断言据此被推翻。判断任何类型不匹配缺陷前，**先追对象来源**：DOMStringMap 会转换，普通对象不会。
- **Foundry 全局**：`globalThis.logger = console` 由 Foundry 提供，所以 `YZEDiceRoller.mjs:191` 的 `logger.warn` **不会**抛 ReferenceError。断言「未定义标识符」前先确认 Foundry 是否提供该全局。

---

## 2. 已锁定的决策

| # | 决策 | 理由 |
|---|---|---|
| 1 | 全量普查，**分期补全** | 用户明确要求「遍历」；但设计只锁定第一期 |
| 2 | **默认全自动，逐项可降级** | 每个特性有独立开关，可降为「提示并确认」的聊天卡，或完全关闭 |
| 3 | **双轨 i18n**：代码与标识符全英文，`lang/en.json` + `lang/cn.json` | 可公开发布；且**任何 RollTable / 状态 / 物品查找一律走 id 或模组自有注册表，永不按显示名** |
| 4 | 系统缺陷 **模组打补丁 + 并行提上游 PR** | 补丁带版本探针，上游合并后自动退休，不会双重修复 |
| 5 | **本机是冒烟机，VPS 是发布权威** | 沿用现有汉化项目纪律；本地跑通不作为诊断证据 |

---

## 3. 架构

### 3.1 模组身份

| | |
|---|---|
| id | `alien-evolved-automation` |
| title | Alien Evolved: Automation |
| 仓库 | `AppData\Local\FoundryVTT\Data\modules\alien-evolved-automation`，独立 git 仓库 |
| flag 域 | `alien-evolved-automation` |
| 对外钩子前缀 | `aea.` |
| 公开 API | `game.modules.get("alien-evolved-automation").api` |
| 兼容 | 系统 `alienrpg >= 4.1.13`；Foundry min 13 / verified 14 |

**依赖**

- 硬依赖：`lib-wrapper` ≥1.13.5、`socketlib` ≥1.1.4
- 软依赖：`dice-so-nice`（缺失时栅栏立即 resolve）、`yze-combat`（**V14 独占，故只能可选**，否则 V13 用户无法安装）

### 3.2 内核八件套

116 条特性最终都落在这八个组件上。不先建它们，每条特性都会各自造一个私有版本。

#### K1 · RollBus —— 唯一掷骰拦截点

libWrapper `WRAPPER` 包 `game.alienrpg.yze.yzeRoll`。

**必须处理的四件事**（都来自源码事实，不是猜测）：

1. **不能「await 原函数后再盖 flag」**。`yzeRoll` 自己在 `:416` 调 `ChatMessage.create`、`:417` 返回 `undefined`。包装器恢复执行时消息早已创建。
   → 正确做法：在 `preCreateChatMessage` 钩子里快照 `rollArr`（此时 `buildChat` 已在 `:131`/`:171` 跑完，数据完整），由包装器设置的重入标志控制归属。
2. **两个不建消息的提前返回**：`:117`（NoAttribute）与 `:141`（NoDice）。包装器必须容忍「这次调用没有产生消息」。
3. **重入**：自动恐慌在 `:198-242` 触发 `rollResolve` / `rollPanic`，**恐慌卡会先于触发卡创建**。归属逻辑必须用调用栈深度而非「最近一条消息」。
4. **快照必须同步**。内层调用返回后到读 `rollArr` 之间不能有任何 `await`，否则并发的另一名玩家的掷骰会覆盖它。

**记录 schema（v1）—— 整个计划里最贵、最难改回去的决定**

```js
flags["alien-evolved-automation"].roll = {
  v: 1,                    // schema 版本，迁移锚点
  id: string,              // 本次掷骰的稳定键，Executor 幂等性依赖它
  actorUuid: string,       // Resolver 输出：非链接 token 给 token actor
  tokenUuid: string|null,
  userId: string,
  kind: "attribute"|"skill"|"weapon"|"armor"|"supply"|"panic"|"stress"|"crit"|"ammo"|"other",
  label: string,           // 原样传入的 label
  labelKey: string|null,   // 可还原时给出 i18n 键（绝不反解已翻译文本）
  attr: string|null,       // dataset.attr —— 系统在 actor.mjs:194 读完即丢弃
  itemUuid: string|null,
  pools:   { base: number, stress: number },
  results: { baseSixes, baseOnes, stressSixes, stressOnes },
  successes: number,       // baseSixes + stressSixes
  banes: number,           // stressOnes
  push: { count: number, pushable: boolean, parentRollId: string|null },
  targets: string[],       // 二期填充
  consumed: { ammo: number|null },
  at: { worldTime: number, real: number }
}
```

**不变量**：

- 成功数**只从此记录读取**，绝不解析渲染后的文本（"Sixes" 是 `lang/en.json:500`，会被 Babele 改写），也绝不读 `Roll#total`（见 K7 说明）。
- actor 一律用 uuid 解析，绝不用 `game.actors.get(speaker.actor)`。

**对外钩子**：`aea.rollResolved(record, message)`。后续所有特性订阅这一个 seam，不再各自包装 `yzeRoll`。

#### K2 · DocRegistry —— id 化一切查找

替换 §1.1(4) 列出的全部按名查找。**清单中 4 条独立「注册表」特性在此合并为 1 条。**

- 绑定表存为 world setting：`{ key: { uuid, name, boundAt, boundBy } }`
- 解析时机：`ready`，且必须**在 Babele 完成与系统的冒险导入之后**（两者都挂 `ready`，钩子顺序依注册顺序而定 —— 需显式排序，见 §3.3）
- 首次绑定才按名字猜测；之后一律用 uuid。提供 GM 重绑面板
- 需要的绑定键：`panic`、`stressResponse`、`panicResponse`、各类型重伤表、每个怪物的攻击表、Mother 系列表、`Alien Tables` 等文件夹、`MU/TH/ER Instructions.` 日志

#### K3 · Executor —— 玩家动作改 GM 文档（二期启用）

基于 `socketlib`。**难点不是 socket 本身，是幂等与可撤销**：

- **幂等键** = `rollRecord.id + actionId`。聊天卡重渲染、双击、断线重连都不能扣两次血
- **权限策略表**：哪些动作允许玩家发起、哪些必须 GM 确认
- **可撤销账本**：记录每次施加的前值，GM 不认可时可回滚

#### K4 · Resolver —— uuid 优先解析（一期就位）

- 怪物 `prototypeToken.actorLink = false`（`actor.mjs:93`），且 `preCreateToken`（`alienrpg.mjs:366-376`）强制 npc 非链接
- 系统各处存 `actor.id` 并用 `game.actors.get()`，聊天卡用 `ChatMessage.getSpeaker({ actor: actorid })`（`YZEDiceRoller.mjs:399-401`；`actor.mjs:2513, 2605`）—— **token 信息被整个丢弃**
- 后果：三只 Drone 在场时，任何自动扣血/上状态/加压力都落到共享的基础 actor，三只一起变，**静默无报错**
- 解析顺序：`canvas.tokens.get(speaker.token)?.actor` → `ChatMessage.getSpeakerActor()` → `game.actors.get()`（仅作兜底并记警告）

#### K5 · Features —— 逐特性 full / prompt / off

- 一张注册表：特性 id、默认档、是否仅 GM、依赖（前置关闭时整个禁用而非半跑）、设置 schema 版本
- 一个设置菜单，不是 95 个平铺 boolean
- 系统的先例是反面教材：20 个 boolean 挤在一个函数里，若干 world 设置 `restricted: false`（玩家可写），`evolved` 改动会强制 `location.reload()`
- 与 K7 共用同一张表，才能区分「这个特性关着是因为上游已修复」与「因为它的表没绑定」

#### K6 · DiceBarrier —— 骰子落地再结算

`awaitDice(message)`。系统在 `alienrpg.mjs:381` 留了 `diceSoNiceRollComplete` 钩子且**函数体是空的**。

四种必须正确处理的情况：

1. DsN 未安装或本用户禁用 → 立即 resolve，绝不 await 一个永不触发的钩子
2. 消息是密语/盲骰且本客户端非接收者 → 无动画也无完成事件（`YZEDiceRoller.mjs:408-415` 有四条路径会设置 whisper/blind）
3. 用户开了 DsN 的立即动画或「对他人隐藏」
4. 硬超时，丢事件不能卡死管线

与 K3 交互：动画在掷骰者客户端播放，写入在 GM 客户端执行 —— 栅栏必须跨客户端协调。

#### K7 · Patches —— 补丁自退休

每个补丁声明：

```js
registerPatch({
  id, systemRange, target, type: "WRAPPER" | "MIXED" | "OVERRIDE",
  probe: () => boolean,   // 运行时断言：这个缺陷还在不在
  apply, fixedIn
})
```

`ready` 时先跑 `probe` 再装补丁；`probe` 说缺陷已消失就自动停用并提示 GM。

系统自身在这方面帮不上忙：它的迁移块（`alienrpg.mjs:311-326`）**是注释掉的**。

**关于 `Die#total`**：`alienRPGBaseDice.mjs:12-15/42-45` 两个类的 `get total()` 都返回 `this.results.length`（池子大小）。复核结论是这**不是崩溃也不是算错数**——系统自己的判定路径从不读 `total`，只有内联骰（`[[8db]]`，全书 39 处）、手打 `/r` 和第三方模组看得见，所以定级 P2 而非 P0。修的时候必须 `Object.defineProperty` 在**现有原型**上，不能改名做子类：`RollTerm.fromData` 按 `data.class` 名匹配，改名会让历史聊天卡静默降级成普通 Die、丢掉骰面图。

#### K8 · Cards —— 自有聊天卡契约

- 自有 Handlebars 模板；聊天日志上**一个**按 `data-action` 分发的委托监听（不是每次渲染重新绑定）
- 自有渲染钩子绑定：V14 用 `renderChatMessageHTML`。系统仍绑已废弃的 `renderChatMessage`（`alienrpg.mjs:464`），该 shim 标注 "since 13, until 15"，V15 会消失
- **不能复用系统的 `dmgBtn-container` 挂点**：它只在 `if (!reRoll || reRoll === "mPush")`（`:378`）分支内发出，**推骰卡、NPC 卡、怪物卡上根本不存在**。必须自己注入挂点
- 系统的 Push 处理器还对非 `character` 类型直接 `return`（`:487-493`），并用 `ev.target.previousElementSibling.checked` 读多推复选框 —— 任何卡片改动都会打断它

### 3.3 启动顺序契约

隐式顺序会产生间歇性、不可复现的失败：

1. `init`：注册设置（K5）、声明注册表键（K2）、注册补丁（K7，**先 probe 后 apply**）
2. `diceSoNiceReady`：探测 DsN 能力（K6）
3. `ready`（显式排在 Babele 与系统冒险导入**之后**）：解析注册表绑定、安装 K1 包装器、启动 K3 socket
4. 特性声明式注册进模组自有 API 命名空间，不各自挂钩子

---

## 4. 分期路线图

### 一期 · 掷骰说真话并留下记录

内核：K1、K2、K4、K5、K6、K7、K8。**K3（Executor）不在一期。**

> 关键简化：一期**不写任何 actor 数据**，因此不需要 socket 层，风险最小。
> 但 **K4 必须在一期就位** —— 记录里存的必须是 uuid 而非 `actor.id`，否则 schema 定错，后面全部返工。

| 特性 | 现状 | 玩家立刻可见的改变 |
|---|---|---|
| `roll-record-and-success-line` | 做了一半 | 不必再心算两行 "Sixes"，一行合并成功数；给出 `aea.rollResolved` |
| `roll-pool-integrity` | 做了一半 | −4 修正不再从压力骰扣（规则原文：「永远不能低于一颗基础骰，修正永不影响压力骰」）；GM 掷骰出现压力 1 时不再整张卡消失 |
| `stress-panic-math-repair` | 做了一半 | 恐慌满级再恐慌不再抛异常；合成人点 Resolve 不再报错 |
| `crit-result-parse-hardening` | 做了一半 | 重伤不再因表格文案被翻译而静默变成「不致命、恢复 0」 |
| `push-correctness` | 做了一半 | 推骰绑回它自己那张卡 |

### 一期 b · 崩溃修复包（并行，触碰文件与主干零重叠）

`shipphase-submit-crash` · `vehicle-roll-path-repair` · `ev-oracle-table-data-repair`（二元神谕现 75% 偏向「否」）· `gear-active-gate` · `d66-roll-composer` · `encumbrance-capacity-fix` · `npc-roster-repair` · `upstream-bug-repair-pack` · `table-draw-tools-and-macros`

### 二期 · 一发子弹打完整条链（加 K3）

`attack-context-binding` → `damage-application` → `ammo-economy` → `broken-and-death-chain` → `creature-attack-draw-pipeline`

**尾巴挂 `time-clock`，唯一消费者是死亡骰倒计时。** 时钟是第二大扇出节点（15+ 特性挂它）；先建成裸原语则无消费者可验证 API，十个特性会照着一个没验过的接口写。

### 三期 · 状态会留下来（时钟的收成）

持续时间引擎 · 战斗状态注册表 · 生存状态 · 重伤计时 · 辐射 · 休息与医疗 · 压力消解 · 恐慌反应的实际效果 · 跨角色恐慌广播（需 K3）

**天赋注册表与修正账本必须同期发布** —— 二者是干掉自由文本 "Base Mod" 输入框的两半，只发一半比不发更糟。

### 四期 · 对抗判定原语

从 `zone-model` 中抽出便宜的一半：**对抗判定 = 比较两条掷骰记录**。一处解锁侦测对抗、防御反应、擒抱、飞船 COMMAND、Evade 五条。地图与邻接留到最后。

---

## 5. 明确不做（四个陷阱）

共同形态：高观感 / 低每场分钟数 / 各拖四到五个前置。

| 不做 | 成本 | 理由 |
|---|---|---|
| `zone-model` | c5 | 12 条特性隐式依赖的空间基质。做一半 = 代码库里两套不兼容的邻接模型 |
| 飞船／载具战斗簇（6 条） | c3–c5 | 一整套平行规则（自有棋盘、阶段结构、行动经济、伤害轨、尺度规则），与地面循环只共享掷骰记录。早期只取 `shipphase-submit-crash` |
| `chargen-wizard` | c5 | 每角色一次。先把它的 5 个数据集当**独立可用件**做（`attribute-skill-bounds-guard` c1 今天就能挡住技能点到 9） |
| `first-edition-converter` | — | 失败模式静默且不可恢复：未转换的 Armor 12 在 Evolved 尺度下仍读作 12。改用书面清单 + 校验器 |
| `hazard-engine` | c5 | 同上形态；其组成部分在三期已分别落地 |

---

## 6. 测试策略

环境未安装 Quench。但 K7 的补丁自检本来就要写 —— 一份代码两用。

1. **纯函数层跑真单测（vitest，TDD 正常走）**：记录组装、注册表解析、成功数计算、重伤行解析。这些不碰 Foundry API。
2. **需要 Foundry 的部分做模组内自检套件**：`ready` 时可选执行；每条断言同时就是 K7 的「缺陷还在不在」探针。
3. **回归基线**：把 **269 条已坐实缺陷**写成断言清单。系统升级后重跑，直接得知上游修了哪些、该退休哪个补丁。

---

## 7. 风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| 掷骰记录 schema 定错 | 全部后续特性返工 | 一期定死并带 `v` 字段；K4 保证存 uuid；先写纯函数单测 |
| 系统版本漂移 | 补丁双重修复或与重写后的实现冲突 | K7 的 `probe` + `systemRange` + `fixedIn`；自检失败即停用并提示 |
| Babele 层改名 | 整条恐慌/重伤链静默失效 | K2 全面 id 化；绝不反解已翻译文本 |
| `yzeRoll` 重入与并发 | 掷骰记录张冠李戴 | 调用栈深度归属；快照必须同步；幂等键 |
| 非链接 token | 伤害/状态落到共享基础 actor，三只一起变 | K4 uuid 优先解析，兜底路径记警告 |
| V13/V14 API 分歧 | 渲染钩子与 DialogV2 差异 | K8 自有绑定；`yze-combat` 只做软依赖 |
| 29 条特性的实现路径已被判需重做 | 按原方案实施会撞空 | 清单中逐条标注；实施计划前必须重新设计其 seam |

---

## 8. 本文档的实施范围

**第一份实施计划只覆盖：内核 K1/K2/K4/K5/K6/K7/K8 + 一期五条特性 + 一期 b 修复包。**

二期及以后的内容写在本文档中，仅作为架构决策的上下文（确保一期不做出会让后续无法落地的选择），不属于第一份计划的范围。
