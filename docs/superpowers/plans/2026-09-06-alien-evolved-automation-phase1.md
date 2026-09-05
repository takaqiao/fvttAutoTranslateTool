# Alien Evolved: Automation 一期实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建成 `alien-evolved-automation` 模组的内核七件套，交付"掷骰说真话并留下记录"，并修掉一期 b 的九条系统缺陷 —— 全程不写任何 actor 数据。

**Architecture:** 模组不分叉系统，只经 libWrapper 包裹与钩子订阅。内核提供七个原语（掷骰拦截与每条消息的掷骰记录、id 化文档注册表、uuid 优先的 actor/token 解析、逐特性 full/prompt/off 注册表、落骰栅栏、带版本探针的补丁注册表、自有聊天卡契约），一期五条特性与一期 b 九条修复全部构建在这七个原语之上，不各自造私有版本。Executor（socketlib GM 执行层）属于二期，因此一期完全不需要跨客户端写入。

**Tech Stack:** JavaScript ES modules（无构建步骤，Foundry 直接加载）· Foundry VTT V13–V14 · libWrapper 1.13.5 · vitest（唯一 devDependency）· Handlebars（Foundry 内置）

**Spec:** `C:/Users/Taka/Desktop/fvtt/docs/superpowers/specs/2026-09-05-alien-evolved-automation-design.md`
**接口契约 v2（与本计划同等强制）：** 见本文件 §全局约束下方的"接口契约"一节，或 `docs/superpowers/plans/2026-09-05-alien-evolved-automation-contract.md`
**差集清单（每条特性的核实细节、风险与逐条缺陷复核）：** `docs/superpowers/specs/2026-09-05-alien-evolved-automation-gap-inventory.md`

---

## Global Constraints

以下是全项目级要求，**每个任务的要求隐含包含本节全部条目**。数值与字符串均逐字取自设计文档与接口契约 v2。

**身份与版本**
- 模组 id：`alien-evolved-automation`；title：`Alien Evolved: Automation`
- 目标系统：`alienrpg`，`relationships.systems[].compatibility.minimum = "4.1.13"`
- Foundry：`compatibility.minimum = "13"`，`verified = "14"`
- 仓库：`C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation`，独立 git 仓库，默认分支 `main`
- 依赖：`lib-wrapper` ≥ 1.13.5 为 `relationships.requires`；`socketlib` ≥ 1.1.4 一期写 `relationships.recommends`（一期无消费者，二期 Executor 落地时提升为 requires）；`dice-so-nice` 与 `yze-combat` 为 `recommends`（`yze-combat` 1.7.1 的 `compatibility.minimum` 是 14，做硬依赖会挡住 V13 用户）

**编码与本地化**
- 代码、标识符、文件路径、提交标题一律英文；说明性散文用中文
- 面向用户的字符串一律 `game.i18n.localize()`；键前缀 `AEA.`，`lang/en.json` 与 `lang/cn.json` 同步，嵌套结构
- 语言码用 `cn`（不是 `zh-CN`）

**四条不可违反的查找纪律**
- 任何 RollTable／文件夹／日志／状态查找一律经 `registry`，**永不 `game.tables.getName()`、永不比对显示名**
- 任何 actor 解析一律经 `resolver`；遗留裸 id 走 `resolver.actorById(id)`，**永不 `game.actors.get(speaker.actor)`**
- 成功数一律读 `RollRecord`，**永不解析渲染后的文本、永不读 `Roll#total`**
- 每个特性执行前查 `features.enabled(id)`；每个系统缺陷修复必须 `patches.register()` 且带可运行的 `probe()`

**测试纪律**
- `pure*` 函数与 `record.mjs` 不得引用任何 Foundry 全局，用 vitest 直接真测
- 可用桩忠实模拟的用 `installFoundryStub()`（`test/stubs/foundry.mjs`）
- 桩无法忠实模拟的（真实 libWrapper 链、DsN 动画、渲染时序）写进 `selftest.register()`，**并且**在任务里给出 MANUAL VERIFICATION 步骤
- **禁止**为跑不动的东西编造单测

**提交**
- 每个任务末尾提交；提交正文中文，结尾附：
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  ```

---

## 合稿裁决（执行前必读）

计划经四轮起草与四道审计收敛。下面是主控在合稿时逐条裁决的事项，**优先级高于任何单个任务正文里的相反写法**。

### 一、`api` 槽的替换只能是单槽子串替换

`scripts/main.mjs` 里的 `export const api` 占两行、八个槽，由六个任务分别填自己那一个。

- **正确**：`patches: null,` → `patches,`（只匹配自己那一个槽的子串）
- **禁止**：整行替换。任务正文里「改完那一行长这样」的示意块**只是示意**，用来让你确认改对了位置，**不是可以整段粘贴的替换文本** —— 粘贴它会把别人已经填好的槽重新写回 `null`，而且症状取决于任务执行顺序。

（审计曾把 Task 3 与 Task 7 的示意块误读为整行替换并报了两个 blocker；主控逐行复核后确认两个任务写的都是单槽替换，误判驳回。但示意块被整段粘贴的风险是真的，故立此约束。）

### 二、`rollBus.addStage()` 的 `around` 允许有意短路

契约 v3.1 只写了「必须恰好调用一次 `next(args)`」，漏掉了短路那半句。**两条 P0 修复必须能短路**：`gear-active-gate` 要在装备未激活时拦下整次掷骰，`vehicle-roll-path-repair` 要接管整条炮击路径。

裁决：`around` 要么恰好调用一次 `next`，要么**不调 `next` 直接返回**（其返回值即整条链的最终返回值）。调用两次永远禁止。`rollBus` 的链式执行器为这两条各配一条单测。

### 三、`selftest.register` 的 `label` 是 i18n 键，不是英文字面量

键名 `AEA.selftest.<条目 id>`，两份 lang 文件都要补。`register()` 原样保存 def、**不在登记时本地化**；`runAll()` / `results()` 返回时由运行器 `localize(def.label)`。

这条对**全部十八个任务**生效，不只是被点名的那两个 —— 混用字面量会让自检面板中英分裂。

### 四、模板通路由测试桩提供，任何任务不得就地补桩

`test/stubs/foundry.mjs` 负责装 `foundry.applications.handlebars = { loadTemplates, renderTemplate }`，调用记进 `ctx.templates`。任务在自己的测试文件里给这个全局赋值属于「就地补桩」，禁止 —— 已经有两个任务在各自补，正是这条禁令要防的。

### 五、`LABEL_KEYS` 按可达性逐键裁决，不按数量

三轮草稿分别收了 26 / 25 / 21 个键。数量不是判据。**判据是**：一个键进表，当且仅当它本地化后的文本可能出现在 `yzeRoll` 第四个参数 `label` 的位置上，依据是接口契约 §2 那 22 个调用点的实际传参。每次增删都要在任务正文里写明对应的调用点。

少收一个键的后果是静默的：那条标签反查不到 i18n 键 → `RollRecord.labelKey` 变 `null`，而 `labelKey` 是已发货的 schema 字段。

### 六、没装 Dice So Nice 时，超时设置项不出现（接受，非缺陷）

`SETTING_DICE_TIMEOUT` 由 `diceBarrier.init()` 注册，而 `diceSoNiceReady` 只在装了 DsN 时触发。所以未装 DsN 的世界里，GM 在设置面板看不到这一项；装上 DsN 后它才出现。

裁决：**接受这个行为**。该键的唯一读者就是 `diceBarrier`，未装 DsN 时零消费者、不会抛错。写进 README 与 Task 8 的手工验证清单即可，不为它扩契约。

---

## 四轮收敛记录

留档，说明这份计划里那些看起来啰嗦的约束各自是为什么存在的。

| 轮次 | 抓到什么 |
|---|---|
| 起草 | 48 个开放问题。其中两条推翻了上游文档：**设计文档 §1.2 关于恐慌溢出的断言是错的**（出货表最后一行 `12-20` 会吸收 13，真正抛异常要爬到 21）；**「包一个入口」的架构不成立**（`abilityRoll` 调 `yzeRoll` 只传 9 个参数，`attr`/`itemUuid`/推骰父卡全拿不到）→ 改为四入口包裹。 |
| 修订 | 三类符号漂移。最严重的是 `patches.register` 的 `apply()` 被四个组实现成两种互斥语义 → 定死为「无参自装器」，因为一期 b 有三类修复根本不是函数包裹。 |
| 定稿 | 20 个装配级缺陷。`registry.resolveAll()` 与 `cards.init()` **没有任何任务插入**；`api` 八个槽只有三个有装配人；`LABEL_KEYS` 被两个模块各定义一份。→ 锚点从 4 个扩到 10 个、ready 段拆成有序子锚点。 |
| 收尾 | 「静默空操作」专项通过：`rollBus` 独占的四个 libWrapper 目标上，上一轮撞车的三处（T11 `pushRoll`、T12 `yzeRoll`、T15 私造 seam）全部改道 `addStage()`。剩余三条中两条为审计误判、一条真缺（短路语义），见上文裁决二。 |

**为什么这些错值得四轮去找**：它们全都不报错。libWrapper 重复注册让修复静默变空操作、`api` 槽被整行覆盖让内核成员静默变 `null`、`LABEL_KEYS` 少一键让 `labelKey` 静默变 `null`、锚点顺序颠倒让 `cards.init()` 跑到 `registry.resolveAll()` 前面 —— 没有一条会在控制台留下痕迹，全部要到实际开桌时才以「某个功能就是不生效」的形式暴露。

---

## 任务索引

十八个任务分文件存放在 `2026-09-06-alien-evolved-automation-phase1/` 目录下（合计 1,198,975 字符，单文件不便逐任务执行）。
每个任务自带完整上下文，实施者只需读自己那一个 + 本文件 + 接口契约。

| # | 分段 | 任务 | 文件 |
|---|---|---|---|
| 1 | 内核 | 模组脚手架与 vitest 工具链 | [`task-01.md`](2026-09-06-alien-evolved-automation-phase1/task-01.md) |
| 2 | 内核 | K5 特性开关内核 `kernel/features.mjs` | [`task-02.md`](2026-09-06-alien-evolved-automation-phase1/task-02.md) |
| 3 | 内核 | K7 补丁自退休内核 `kernel/patches.mjs` 与自检套件 `kernel/selftest.mjs` | [`task-03.md`](2026-09-06-alien-evolved-automation-phase1/task-03.md) |
| 4 | 内核 | K4 Resolver —— uuid 优先的 actor/token 解析 | [`task-04.md`](2026-09-06-alien-evolved-automation-phase1/task-04.md) |
| 5 | 内核 | K2 DocRegistry —— id 化一切文档查找 | [`task-05.md`](2026-09-06-alien-evolved-automation-phase1/task-05.md) |
| 6 | 内核 | K1 纯函数层 —— RollRecord 组装器（`kernel/record.mjs`） | [`task-06.md`](2026-09-06-alien-evolved-automation-phase1/task-06.md) |
| 7 | 内核 | K1 副作用层 —— RollBus 四入口包裹（`kernel/rollbus.mjs`）与 main.mjs 五处接线 | [`task-07.md`](2026-09-06-alien-evolved-automation-phase1/task-07.md) |
| 8 | 内核 | K6 DiceBarrier —— 骰子落地再结算 | [`task-08.md`](2026-09-06-alien-evolved-automation-phase1/task-08.md) |
| 9 | 内核 | K8 Cards —— 模组自有的聊天卡契约与渲染扇出 | [`task-09.md`](2026-09-06-alien-evolved-automation-phase1/task-09.md) |
| 10 | 一期特性 | 掷骰卡上的合并成功数行（`roll-record-and-success-line`） | [`task-10.md`](2026-09-06-alien-evolved-automation-phase1/task-10.md) |
| 11 | 一期特性 | 推骰绑回它自己那张卡（`push-correctness`） | [`task-11.md`](2026-09-06-alien-evolved-automation-phase1/task-11.md) |
| 12 | 一期特性 | roll-pool-integrity —— 基础骰下限、压力骰不受修正影响、GM 掷骰不再凭空消失 | [`task-12.md`](2026-09-06-alien-evolved-automation-phase1/task-12.md) |
| 13 | 一期特性 | stress-panic-math-repair —— 压力／恐慌状态机的六处缺陷、四条补丁 | [`task-13.md`](2026-09-06-alien-evolved-automation-phase1/task-13.md) |
| 14 | 一期特性 | crit-result-parse-hardening —— 重伤结果不再靠切碎富文本按下标取值 | [`task-14.md`](2026-09-06-alien-evolved-automation-phase1/task-14.md) |
| 15 | 一期 b 修复包 | 载具与飞船开火路径 + 装备启用闸门（两条修复共用 rollBus 的 `itemRoll` 目标） | [`task-15.md`](2026-09-06-alien-evolved-automation-phase1/task-15.md) |
| 16 | 一期 b 修复包 | 五条表单类修复 —— 飞船阶段提交、异形酸血、主机技能行、改名 token、负重 | [`task-16.md`](2026-09-06-alien-evolved-automation-phase1/task-16.md) |
| 17 | 一期 b 修复包 | 出厂数据修复 —— 三张 EV 神谕表 + NPC 名录校验器 | [`task-17.md`](2026-09-06-alien-evolved-automation-phase1/task-17.md) |
| 18 | 一期 b 修复包 | 抽表路径 —— D66 十位修正、抽表工具与怪物卡文件夹守卫 | [`task-18.md`](2026-09-06-alien-evolved-automation-phase1/task-18.md) |

**接口契约 v3.2**（与本计划同等强制，签名不得改动）：[`2026-09-06-alien-evolved-automation-phase1-contract.md`](2026-09-06-alien-evolved-automation-phase1-contract.md)

## 差集清单 → 任务 对照表

计划里的实现 id 比清单更细（有些一条清单项拆成了两三条独立补丁）。这张表保证每条清单项都找得到落点，
也保证验收时能反查「清单里这条到底做没做」。清单原文见 [`../specs/2026-09-05-alien-evolved-automation-gap-inventory.md`](../specs/2026-09-05-alien-evolved-automation-gap-inventory.md)。

| 差集清单 id | 任务 | 计划内实现 id |
|---|---|---|
| `roll-record-and-success-line` | [Task 10](2026-09-06-alien-evolved-automation-phase1/task-10.md) | roll-record-and-success-line |
| `push-correctness` | [Task 11](2026-09-06-alien-evolved-automation-phase1/task-11.md) | push-correctness |
| `roll-pool-integrity` | [Task 12](2026-09-06-alien-evolved-automation-phase1/task-12.md) | roll-pool-integrity |
| `stress-panic-math-repair` | [Task 13](2026-09-06-alien-evolved-automation-phase1/task-13.md) | stress-panic-math-repair（拆成四条补丁） |
| `crit-result-parse-hardening` | [Task 14](2026-09-06-alien-evolved-automation-phase1/task-14.md) | crit-result-parse-hardening |
| `gear-active-gate` | [Task 15](2026-09-06-alien-evolved-automation-phase1/task-15.md) | gear-active-gate |
| `vehicle-roll-path-repair` | [Task 15](2026-09-06-alien-evolved-automation-phase1/task-15.md) | vehicle-roll-path-repair |
| `shipphase-submit-crash` | [Task 16](2026-09-06-alien-evolved-automation-phase1/task-16.md) | spacecraft-phase-submit + spacecraft-phase-persist |
| `upstream-bug-repair-pack` | [Task 16](2026-09-06-alien-evolved-automation-phase1/task-16.md) | creature-acid-args + mainframe-skill-row + token-defaults-by-reference |
| `encumbrance-capacity-fix` | [Task 16](2026-09-06-alien-evolved-automation-phase1/task-16.md) | 负重容量修复 |
| `ev-oracle-table-data-repair` | [Task 17](2026-09-06-alien-evolved-automation-phase1/task-17.md) | ev-oracle-tables |
| `npc-roster-repair` | [Task 17](2026-09-06-alien-evolved-automation-phase1/task-17.md) | npc-roster |
| `d66-roll-composer` | [Task 18](2026-09-06-alien-evolved-automation-phase1/task-18.md) | d66-roll-composer |
| `table-draw-tools-and-macros` | [Task 18](2026-09-06-alien-evolved-automation-phase1/task-18.md) | 抽表工具 + 怪物卡文件夹守卫 |
