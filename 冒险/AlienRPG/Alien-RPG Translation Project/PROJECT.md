# 异形 RPG（Alien RPG）汉化项目 · 主文档

> 这是本项目的**唯一长期入口**。新会话请先读 §1，再按需跳读。
> §8 裁决日志**只追加、不重写**——被推翻的结论加脚注保留，不许改成"一直是对的"。

**发版状态（2026-08-29）：Phase 0/1/2 已完成。`alienrpg-cn 0.1.0` **已公开发布** ——
系统 UI 600 键 + 系统自带 Adventure 包全译，仓库 https://github.com/takaqiao/alienrpg-cn 。
`alien-evolved-starterset-cn 0.1.0` / `alien-evolved-corerules-cn 0.1.0` 仍是**未发布的空壳**：
两个内容包（34 万字 / 229 万字）**一个字都还没译**。
⚠ 已发布的这一版**没做过实机冒烟**，见 §7.1。**

> ⚠ **本行就是「抬头」**。版本号全文只写两处：**本行**与 **§2 的版本矩阵**。别处一律指过来。
> EC 项目实测：§1 里那个「第二真相」曾停在旧版本达四个发布版没人发现。
> ⚠ **今后谁往文档最前面加节，先想一下这一行还在不在前 40 行里**——判据取前 40 行。

---

## 1. 快速跟进（新会话必读）

### 1.1 这是什么

把 Foundry VTT 的 **Alien RPG（Alien Evolved）** 整套汉化成简体中文，方法论照搬
`C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project`（下称 **EC 项目**）。
不是照抄内容——**世界观不同，术语表严禁与 EC 或 PF2E 合并**——照搬的是管线、闸门和纪律。

**三个交付模块**（各自独立 git 仓，与本项目外仓分开）：

| 槽位 | 模块 id | 覆盖 | 体量 |
|---|---|---|---|
| `1-系统汉化插件` | `alienrpg-cn` | 系统 UI（590 键）+ 系统自带 Adventure 包 + 五个插件的 lang + 运行时补丁 + CJK 字体 + **唯一一份全局 babele mapping** | 7.6 万字 + UI |
| `2-新手包汉化插件` | `alien-evolved-starterset-cn` | Starter Set 内容包 | 34.1 万字 |
| `3-核心书汉化插件` | `alien-evolved-corerules-cn` | Core Rules 内容包 | 228.7 万字 |

合计 **2,703,509** 字（原始 HTML 字符）。可见正文约 98 万字——**按可见字数排工，不按原始字数**，
最极端的一页 `CR-J-14b EQUIPMENT` 是 59,662 原始 / 8,438 可见（14%）。

### 1.2 三条必须先知道的事

**① 三个包都是单 Adventure 文档包，而且是「导入型」模块。**
`alien-evolved-corerules/module/init.js` 在首次 `ready` 时把整个 Adventure 导入世界。
Babele 包装的是 `CONFIG.DatabaseBackend._getDocuments`，所以**导入时确实会拿到译文**——
但 `alien-evolved-*` 按字母序排在 `babele` 之前，它的 `ready` 钩子先注册也先触发，
**首次启动的自动导入拿到的是英文**。汉化模块必须提供重新导入路径或世界文档重译宏。
这是与 EC 最大的结构差异（EC 的内容一直待在合集里，读时翻译）。

**② 系统自带的 `lang/cn.json` 不是可以打补丁的底子，是重译目标。**
它有两层：**A 层**（TI-130 / Discord `Tian#7972`，2020-11→2021-01，人工，质量好——属性、技能、
职业、距离、整张 Panic7-15 表）和 **B 层**（维护者 2022-2026 的批量"更新所有语言文件"，机翻，约 107 键有缺陷）。
B 层的招牌错误：

| 键 | 英文 | 现有中文 |
|---|---|---|
| `ALIENRPG.GMONLY` | GM Only | 仅限**通用汽车** |
| `ALIENRPG.Overwatch` | Overwatch | **守望先锋** |
| `ALIENRPG.SelectFirer` | Select Firer | 选择**消防员** |
| `ALIENRPG.HOWMANYDICE` | How Many Base Die? | 有多少**基地死亡**？ |
| `ALIENRPG.Utility` | Utility | **公用事业** |
| `ALIENRPG.Homebrew` | Homebrew | **家酿** |
| `ALIENRPG.None` | None | **莫** |
| `ALIENRPG.OneTurn` | One Turn | **一斡** |

按实际引用统计：**487 个活键里约 162 个今天渲染成英文**。
`zh-tw.json` 是同一个简体祖本的**不完整** S→T 转换（511 键里 101 个值压根没转），
不是独立来源，**不要拿它挖备选译法**。

**③ 有两条运行时硬耦合，翻错就是静默破坏玩法。**

- **重伤表解析**：`module/documents/actor.mjs` 的 `case "character"` 分支把重伤 RollTable 的
  **结果正文**按 `/[:] |<br>/gi` 切开，再按**固定下标**读格子。
  ⚠ **这里是两个 switch，不是一个；而且最危险的那条腿根本不在 switch 里。**

  | 代码 | 读哪一格 | 与什么比 | 落空时 |
  |---|---|---|---|
  | `switch(testArray[3])` | FATAL | `localize('ALIENRPG.Yes')` + 三个后缀（`' '` / `', –1 '` / `', –2 '`）| `default` → `cFatal = false`。**静默** |
  | `switch(testArray[5])` | TIME LIMIT | `localize('ALIENRPG.None' / 'OneRound' / 'OneTurn' / 'OneShift' / 'OneDay')` + `' '` | `default` → `healTime = 0`。**静默** |
  | 两个 switch **之前**的那条 `if` 链（读 `testArray[9]`）| HEALING TIME | 先 `!== localize('ALIENRPG.Permanent')`，再 `=== 'Shift'`——**裸字面量，比较侧没有 i18n 键** | **不静默**：落进 `[[NdM]]` 正则，`.match()` 返回 null，`null[1]` 抛**未捕获 TypeError**，整次重伤检定死在半路，聊天卡不发、伤害不落 |

  ⚠ 两个 `Yes` 后缀用的是 **EN DASH U+2013** 而不是 ASCII 连字符——判据必须用 `\u2013` 转义写死。
  ⚠ **`ALIENRPG.Shift` 是输出键，该翻；表格里那个 `Shift` 格子是比较侧，不许翻。**两者名字像，作用相反。
  ⚠ **"healTime 恒为 0" 有两个来源，别混为一谈。** 经典模式（`Critical injuries` 表）恒为 0，是现有
  cn.json 把这组键翻了一半造成的，**翻译能修**；进化版（`EV - Critical Injuries` 表）实测在 index 5
  吐的是 `–` / `Shift` / `Stretch` / `Round`，五个 case 一个都不命中，**与语言无关、今天英文环境下
  也恒为 0**——那是上游缺陷，别顺手"修"成 `One Shift `，那是改玩法。
  ⚠ 同一个文件里**还有三个分支**（`synthetic` / `creature` 与 `spacecraft`）做同样的固定下标解析，
  但用**另一个切分器** `/[:] |<br \/>/gi`——**没有** `<br />`→`<br>` 归一化，所以裸 `<br>` 在那里根本不切——
  读的下标也不同，另外覆盖四张表。登记在册子的 `crit_parse_lockstep.other_branches`。
  译文与以上所有格子必须锁步，由 `qa/scan_crit_lockstep.py` 看守。
- **怪物按表名找攻击表**：actor 的 `system.rTables` / `system.cTables` 存的是**字面表名字符串**
  （`'EV - Chestburster Attacks'` 等），26 个 actor 存的是字面 `'None'`。
  翻了表名不同步 actor 字段，攻击检定当场断。由 `qa/scan_name_lookup_traps.py` 看守。

同类的硬冻结面**全部登记在 `7-其他内容/DO-NOT-TRANSLATE.json`，一律以那个文件为准，别从本文档转抄计数**。
册子今天是 **11 节 + 5 条上游缺陷 + C1–C9 九条更正**（**以文件为准**）：首次导入查找名 · RollTable 字面名 ·
文件夹名 · 物品名 · T-EXACT 对齐 · 重伤锁步 · actor 表名引用 · enricher 标签 · 标题锚点 ·
`macro_commands` · `name_substring_tests`。**闸门在运行时读它**，不是读文档。
⚠ **`qa/build_register.py` 今天已经生成不出这个文件了**——册子顶层的 `_generator_status` 就写着这句。
重跑它会**删掉** C1–C9、两个 `gate_tests` 块，以及 `scan_crit_lockstep.py` 用来避免误报的
`rows_with_nonstandard_split`。**在生成器改回同步之前，那份 JSON 是唯一真相，不要跑生成器。**

**这四条比上面两条更狠**（都是 2026-08-29 从源码重新推导出来的，勘察轮没抓到）：

| # | 事实 | 后果 |
|---|---|---|
| ① | `Alien Creature Tables`（`rollTableData.mjs:7`）与 `Alien Mother Tables`（`:24`）用 `find()` 取文件夹后**无守卫**地读 `folder.contents` | 翻了这两个名字 ⇒ `_prepareContext` 里抛异常 ⇒ **整张怪物卡渲染不出来，是一个空窗口**，不是下拉框退化。<br>⚑ 而 `Alien Sub-Tables` **没有被引用**（`Folder#contents` 非递归），**可以翻** |
| ② | 一批物品名是用 `.toUpperCase() === "..."` 比的——**不是 2 个，是 6 个**（`PACK MULE` `TAKE CONTROL` `NERVES OF STEEL` `TOUGH` `HARDENED` `STOIC`，`actor-character.mjs` 与 `actor-synthetic.mjs` 各带六个；名单以册子为准）| **双语尾巴当场致命**：`"驮马 Pack Mule".toUpperCase() !== "PACK MULE"`。全部必须 T-FROZEN。<br>⚑ 其中 `HARDENED`（+1 最大生命，两个模式都吃）与 `STOIC` 在 corerules 包里**有活文档**——翻了当场改人物数值，**不报错** |
| ③ | `actor.mjs:1890` 的裸字面量 `Shift`，由 `EV - Critical Injuries` 的两行（14-14、15-15）产出 | 翻了它 ⇒ 落进 `.match(...)[1]` 而 match 为 null ⇒ **未捕获 TypeError，整个进化版重伤检定当场死在半路**。<br>⚑ 但 `ALIENRPG.Shift` 是**输出**键，**应该**翻——两者名字像，作用相反 |
| ④ | 346 个 `data-hash` 深链锚点（153 个不同），而**没有任何一个标题带显式 `id`** | `_makeHeadingNode` 取 `heading.id \|\| slugifyHeading(heading)`，而 `slugify` 非严格模式**原样保留 CJK** ⇒ 翻了标题，指向它的每一条深链都静默滚到页首。<br>**唯一不用改两端的修法：在译后标题上补一个 `id=` 保留英文 slug** |

另：`ALIENRPG.Permanent` **没有**尾随空格（它是串里最后一格），其余 7 个键各带**恰好一个**。

### 1.3 已定的口径（业主 2026-08-29 拍板，不要再问）

| # | 决定 |
|---|---|
| 1 | 三个模块**全部公开发布**。风险已当面提出并由业主裁定接受。`compendium/en/*` 仍然排除出发布 zip——那是上游英文原文，装进去就从"翻译覆盖层"变成"原文再分发" |
| 2 | 语言代码一律 **`cn`**（不是 zh-Hans / zh-CN）。系统自己声明的就是 `cn`，本机 13 个包用 `cn`、仅 2 个用 `zh-CN` |
| 3 | ⚑ **2026-08-29 已推翻并重定**（业主补料后）：术语主轴改为**本书中译优先**。见 §1.4 |
| 4 | **异形**，永不用「异型」（四部电影 4601 条双语对照里 异形 56 / 异型 0；维基与 2024 院线版《异形：夺命舰》同）。并且编码一条中文才有的分野：**异形 = 生物（名词）／外星 = 形容词**（"外星飞船"不是"异形飞船"） |
| 5 | 命名**三档制**，见 §3.4 |
| 6 | **本机是冒烟机，VPS 是发布权威**。§2 矩阵记 VPS 实际版本，发版前跑版本对齐断言 |

### 1.4 术语来源优先级（2026-08-29 定，**取代 §1.3 第 3 条的旧版**）

业主补料之前只有 zh.wikipedia（讲**电影**的通用百科）和 CnSCG 字幕（一条血脉）两类证据。
补料之后多了三类更强的，其中一类**就是这本书本身的中译**。⇒ 主轴重排：

| 级 | 来源 | 说明 |
|---|---|---|
| **1** | **《异形RPG：进化版》中文连载**（第 1、2 章，25,615 字）| **同一本书、同一版本、术语成体系**。⚠ **论坛上只有这两章，没有别的了**（业主确认），所以它覆盖不到的地方要往下顺延 |
| **2** | **Romulus 2024 · Alien: Earth 2025 字幕** | 最新、面向大陆观众；Romulus 有院线版 |
| **3** | **Alien: Isolation 官方简中**（16.3 万键 / 675 万字）| **官方**本地化，且是**界面长度**的词条，比电影台词更贴 RPG 术语 |
| **4** | 系统 `lang/cn.json` **A 层**（TI-130 2020-21 人工层）| 玩家今天在 Foundry 里看到的就是它 |
| **5** | Covenant 2017 · Prometheus 2012 字幕 | 独立血脉 |
| **6** | CnSCG 异形 1/2/3/4 字幕 | **四部折成一票**（同一翻译血脉）；含台湾腔与错译 |
| **7** | zh.wikipedia | 讲电影的通用百科，**不是讲这个 RPG 的**。降为兜底 |

**业主钉死的两条例外**（第 4 级压过第 1 级，因为玩家已经在看它们）：

| 英文 | 采纳 | 而本书中译作 |
|---|---|---|
| **Wits** | **机智** | ~~智力~~ |
| **Empathy** | **共情** | ~~同理心~~ |

理由：这两个词出现在每张角色卡、每次检定、每条天赋描述里；「机智」更贴 Wits 的字面
（机敏/急智），且两词都是二字，卡片窄栏友好。

**其余由主控按上表综合裁定，不再逐条问业主**（业主原话：「术语你综合判断最优解作为主轴即可」）。
裁定结果落进 `7-其他内容/glossary/glossary_alien.json`，依据摊在
`7-其他内容/glossary/SOURCE-CONFLICTS.md`，**争议保留在 `.disputes.json`**。

⚠ 由此产生的**具体改动**（相对 Phase 0 那版维基主轴的词表）：
`Weyland-Yutani` 韦兰-尤坦尼集团 → **韦兰德-汤谷公司**（简称**韦汤**）·
`Nostromo` 诺史莫号 → **诺斯特罗莫号** · `Sulaco` 苏拉克号 → **萨拉科号** ·
`Synthetic` → **合成人**（`Android` → 仿生人、`Robot` → 机器人，三词分开）·
`atmosphere processor` 大气处理厂 → **大气处理器** · `Hadley's Hope` → **哈德利的希望** ·
新增 `Round/Stretch/Shift` = **轮 / 节 / 班**，并连带把 `Turn` 从 ~~轮次~~ 改成 **节** ·
`Cinematic/Campaign Mode` = **电影模式 / 战役模式** ·
`Frontier/Outer Veil/Core Systems/Outer Rim` = **边境 / 外层帷幕 / 核心星系 / 外环** ·
`Evolved Edition` = **进化版**。
`Facehugger` = **抱脸虫** 的裁定不变（本书前两章未涉及生命周期，仍靠百度/维基两支）。

---

## 2. 版本矩阵

> **这是版本号的第二处也是最后一处。**由 `qa/assert_resolutions.py` 的 `R-version-matrix` 机械看守。

| 模块 | 本仓版本 | 已发布 | 上游依赖 | 上游版本（本机实测） |
|---|---|---|---|---|
| `alienrpg-cn` | 0.1.0 | **v0.1.0 · 2026-08-29 公开发布** | system `alienrpg` | 4.1.13 |
| `alien-evolved-starterset-cn` | 0.1.0 | —（空壳，正文未译） | module `alien-evolved-starterset` | 1.0.2 |
| `alien-evolved-corerules-cn` | 0.1.0 | —（空壳，正文未译） | module `alien-evolved-corerules` | 1.0.2 |
| （运行时依赖） | — | — | `babele` | 2.9.1 |

> **Phase 0 收尾状态（2026-08-29）**：三个模块的骨架齐了（`module.json` / `register.js` / 发版工作流），
> `compendium/en` 已按上表这三个上游版本抽全，并**一个不落**地快照进 `7-其他内容/english-baseline/`。
> 三个仓的 `compendium/cn` 都还只有 `.gitkeep`，三个仓都是 **0 commit、0 tag**——"已发布"列的 `—` 是字面意思。
> ⚠ 模组一的发版工作流**还缺一道"有东西可发"的闸**（模组二、三都有），照现状打 tag 会发出一个空壳，见 §8。

**插件覆盖面**（都由 `alienrpg-cn` 的 gated lang 条目承载）：

| 模块 | 版本 | 汉化通道 | 体量 |
|---|---|---|---|
| `alien-mu-th-ur` | 2.0.0 | lang + 运行时补丁 | 9.2k + 2.0k |
| `motion_tracker` | 1.5.5 | 纯 lang（全项目最干净的面） | 3.5k |
| `motion-tracker-multideck` | 1.0.2 | 纯运行时（无任何 i18n 管线） | 2.2k |
| `token-action-hud-alien` | 1.3.0 | **无需单独做**——它的标签全是 `ALIENRPG.*` 键，翻系统即翻它 | 14 字符 |
| `terminal` | 4.0.11 | 纯运行时；先只做玩家可见切片 | 1.6k（全量 33k） |

> ⚠ **VPS 版本尚未录入。** 发版前必须填这一列并跑对齐断言——本机装的不等于 VPS 装的。

---

## 3. 硬约束（不可违背）

### 3.1 声明式 mapping，唯一真相源

mapping 数据**只写在** `4-常用脚本/extract/mappings.mjs`。它被两处消费：运行时交给
`babele.registerMapping()`，抽取器**解释**同一份数据决定抽哪些字段。这保证"CN 里这条路径存在吗"
与"Babele 会去找这个键吗"是同一个问题。

`babele-mappings.js` 是**生成物**，头部写死 DO-NOT-EDIT-BY-HAND 并注明再生成命令。
**改 mappings.mjs 或 runtime-converters.js 必须在同一个 commit 里重跑 `generate_runtime.mjs`，
并且自定义 converter 的 extract 方向与 translate 方向必须同一个 commit 一起改。**

⚠ **`registerMapping` 推的是全局层，会并进每一个 Babele 管理的包。**
所以**每一个 variant 字段都要带 `exists` 守卫**，不只是 `description` 那几个——
`character` / `creature` / `weapon` 是 dnd5e/pf2e 里最常见的 type 键，
不带守卫的 `notes` / `weaponClass` / `capacity` / `appearance` 会污染别的系统的导出。

⚠ **只有 hub 模块 `alienrpg-cn` 调 `registerMapping`。** 三个模块各调一次会三重合并。

### 3.2 基线取最新上游发布版，且历史必须归档

`compendium/en/` 永远只有**当前**英文。重抽之前必须先把旧的整份快照进
`7-其他内容/english-baseline/<pkg>-<ver>/`，**全部包，一个不落**。

> EC 项目最贵的一个 bug：只快照了 10 个包里的 3 个，三道漂移闸在 30% 的语料上跑了几个月，
> 报告一直是干净的。**"基线缺一个包"和"扫过了、干净"在输出上长得一模一样。**
> ⇒ 每个吃基线的脚本都必须打印覆盖行（扫了几个包 / 仓里几个包 / 缺几个），并支持 `--strict-coverage`。

⚠ 两个内容模块的 manifest 是 S3 上的 `-latest.json`，**没有版本锚点**，
`1.0.2` 可能被无声替换。每轮跟版都要重新快照三个包。

### 3.3 术语表是本项目专属

`7-其他内容/glossary/glossary_alien.json`。**严禁并入 EC 的 `glossary_ec.json` 或 PF2E 主表**——
世界观不同。跨系统表只能当**低优先级建议源**，每一条命中都要人确认。

术语裁定阶梯（强→弱）：
**同一条目自己的 `name` 字段 > 同一卷已译页 > 全库多数 > 术语表 > 任务书里的摘录表**。

⚠ **字幕语料算一票，不算四票。** 四部电影全是 CnSCG（圣城家园）同一条翻译血脉，
跨片"一致"是共同出处的产物。其中若干是台湾腔（士官长 / 太空梭 / 电波枪 / 复仇161星 / 伟伦优达尼），
若干是错译（APC→焚化炉、dropship 直接抹掉、stasis→静态平衡、Ellen→海伦）。
Facehugger / Chestburster / Ovomorph / Drone / Warrior / USCM / Acheron / Hadley's Hope 在字幕里**零命中**。

⚠ **绝不从裸中文词频得结论。** 每条术语结论都要带**英文闸**的三桶计数
（gated_hit / cn_only / en_only），见 `4-常用脚本/tm/term_gate.py`。
EC 实测：160 次「阶位」看着像 Tier 残留，英文其实是 Rank；167 次「闪电」里只有 23 条是真缺陷。

### 3.4 命名三档制

| 档 | 规则 | 覆盖 |
|---|---|---|
| **T-FROZEN** | 保持英文，逐字节不动 | **覆盖面以 `7-其他内容/DO-NOT-TRANSLATE.json` 为准，不在这里列举。**册子里挂 `T-FROZEN` 的节是：首次导入查找名 · 硬编码 RollTable 名 · 文件夹名 · 物品名 · `macro_commands` · `None` 哨兵 |
| **T-EXACT** | 纯中文，且**与 lang 键逐字节相等**，不许带英文尾巴 | 12 个 skill-stunts 物品名，必须等于 `ALIENRPG.Skill<key>` 的值 |
| **T-BILINGUAL** | `中文 English`，**一个 ASCII 空格**分隔，不用括号 | 其余专有名词的 `name` 与页名：怪物、装备、天赋、行星、表格 |
| **T-LATIN** | 保持拉丁字母，但**不是**因为有代码在比对它 | 型号与船级前缀：`USCSS` · `UAS` · `USS` · `LV-426` · `M41A` 一类。**与 T-FROZEN 分开是有意的**——T-FROZEN 是一条**可机检的证明**（每一条都必须在 `DO-NOT-TRANSLATE.json` 里找得到对应查找点），把型号塞进去会逼出一张手维护的例外表，而**带例外表的闸门就是不再闸的闸门** |
| **T-PLAIN** | 裸中文 | 正文散文、`{label}` 内文本 |

> ⚠ **T-FROZEN 的三个数曾经在本文档里写错，一律以册子为准（以文件为准）**：
> RollTable 名是 **11** 不是 10——第 11 个**不在模组 JS 里**，而在 corerules 包自带的那个 Macro 的
> `command` 字段里（`EV - 48a. LS - DANGER EVENT DETAIL`），且它对 `.formula` 的解引用无守卫；
> 文件夹名是 **3** 不是 2，而 `Alien Sub-Tables` 从没被代码点过名，**反而可以翻**；
> 物品名是 **6** 不是 2——`actor-character.mjs` 与 `actor-synthetic.mjs` **各带六个**
> `Attrib.name.toUpperCase() === "..."` 比较：`PACK MULE` · `TAKE CONTROL` · `NERVES OF STEEL` ·
> `TOUGH` · `HARDENED` · `STOIC`。其中 `HARDENED`（+1 最大生命，**两个模式都吃**）与 `STOIC`
> （WIT > STR 时把 Stamina 的属性换成 WIT）在 corerules 包里**有活文档**——翻了当场改人物数值且不报错。
>
> ⚠ **`Macro.command` 是一整个字段级的冻结。** Babele 默认就翻 `command`
> （`babele/script/mapping/default-mappings.js:163`）。抽取器今天不发这个字段——
> **那是承重的，不是碰巧的**。宏对话框的文案本来就走 `game.i18n.localize('ALIENRPG.*')`，
> 要加新串请加进 `lang/cn.json`，不要动 `command`。
>
> ⚠ **另有两条不是"整串冻结"、而是"局部必须留英文"的约束**，也在册子里，写正则前先看：
> ① `name_substring_tests`——弹药重量按**物品名**判：`includes(' RPG ')` / `startsWith('RPG')` /
> `endsWith('RPG')` **三选一必须仍然成立**（数据腿已实测是死的，名字是唯一活路；今天唯一命中
> `M5A3 RPG Launcher`）。T-BILINGUAL 的英文尾巴恰好保住它，**纯中文改名会让每发弹药从 0.5 kg
> 静默掉到 0.25 kg**，和 `PACK MULE` 同一类静默数值缺陷。
> ② `rolltable_names.prefix_filters`——`Critical Injuries on Xenomorphs` 是被
> `folder.contents.filter(x => x.name.startsWith('Critical Injuries'))` 取用的，
> 所以它的英文前缀必须留在**偏移 0**，**不能**写成 `中文 English`，否则它掉出怪物卡的重伤表下拉框。
>
> T-EXACT 为什么不能带英文尾巴：`character-skills.hbs:15` 发 `data-pmbut='{{skill.description}}'`，
> 而 `system.skills.<skl>.description` 每次 `prepareDerivedData` 都被
> `game.i18n.localize('ALIENRPG.Skill<key>')` 覆写，然后 `character-sheet.mjs` 拿它去
> `game.items.getName()`。**包里的物品名和 lang 文件跨两个模块耦合。**
> ⚠ **爆炸半径不止 character / synthetic 两张卡。**同一个 `game.items.getName(dataset.pmbut)`
> 跑在**四张卡**上：character / synthetic 被 `if (!evolved)` 关在**经典模式**里，
> 而 **spacecraft / vehicle 无守卫，两个模式都跑**——所以在进化版世界里，能被这条打坏的
> 恰恰是最容易被漏掉的那两张。（另：`_stuntBtn` 的第二条腿是死的，见 §7.3，不要为它建 cn.json 块。）

### 3.5 落盘纪律

- **落盘前必须三方合并。** `apply_translations.py` 是**整叶覆盖**；同一 base 生成的多个并行批次
  按顺序落，先落的会被静默回滚（EC 实测某轮 350 条、另一轮 681 条路径被争抢）。
  base 取**当前** `compendium/cn` 的值，每个批次当 diff 重放，真冲突进 `resolutions.json` 并写明理由。
- **`--force` 语义。** 不带 `--force` 时，已有中文的路径会被**跳过**并报 `skipped(existing)`，**不报错**。
  所有补漏 / 回填 / 重对齐 / 术语统一批次按定义都打在已有中文上，**必须带 `--force`，否则是静默空跑**。
  验收标准定在**缺陷**上而不是闸上：批次落地后 BLOCK / TRUNCATED 必须**肉眼可见地下降**；没动就是没写进去。
- **只有主控写 `compendium/cn`。** 译者和复核只写自己单元目录里的 `batch.json`。
  并且**波次运行期间主控也不许写**——agent 正在读它，会读到写了一半的 JSON。
- **每一次写都过闸。** compendium 三道（EN 源漂移 / 无 CJK / markup 多重集），
  lang 四道（键在新 EN 里存在 / 占位符一致 / HTML 标签 1:1 / 内联 markup 目标一致）。
- **译者自闸到零拒绝**：`apply_translations.py --dry` 必须打印 0 rejections 才允许交付。

### 3.6 markup 是功能，不是装饰

`@X[...]` 括号内、`[[/cmd ...]]` 内的一切都是机器件，**逐字复制**；只有尾部 `{label}` 是散文。
结构标签**计数**必须与英文完全相等——不许合并、拆分、丢段。功能性 class 属性逐字复制。

⚠ 本语料的链接语法是 **v10+ 的 `@UUID[Item.<id>]{label}`**，不是 `@Item[...]`。
针对后者写的正则会**零命中并放行一个链接全毁的文件**。
⚠ corerules 用 `class="content-link"`，starterset 用 `class="ev-content-link"`——**两个包不一样**，
写正则时两个都要匹配。
但 **class 的差异只影响 CSS，不影响功能**：Foundry 的委托选择器是 `a[data-link]` 而不是 class
（`foundry.mjs:35921/:35925`），且两个模块都没给 `.ev-content-link` 写样式。
⚠ 带显式 `{Label}` 的 `@UUID` 是 **42 条**（39 Item + 2 JournalEntry + 1 Actor），
其中 **39 条 Item 链接里有 12 条指向 starterset 包的物品**——
**这是跨两个 git 仓的同步点**，改 starterset 的物品名会让 corerules 正文里的链接文字过期。
（另：`@UUID[Actor.d8GcCHXV1KpltgK1]{STARCUB SHUTTLE}` 上游就是断的。）
⚠ corerules 正文里另有 33 个 `@TEXTDRAW[RollTable.<id>]{label}` + 3 个 `@DRAW[...]`，
它们的 `{label}` 复制的是 **RollTable 名**，是表名的第二处书写地点。
⚠ 第 5 章 GEAR & TECH 里 39 个 `@UUID[Item.<id>]{Label}` 的 label 是**显式**的——
翻了物品名**不会**自动更新链接文字，这是 39 个硬同步点。

### 3.7 文档纪律

- **版本号只写两处**（抬头 + §2 矩阵），别处一律指过去。
- **轮次相关的数字一律不转抄**：规则条数看 `RESOLUTIONS.assertions.json`，脚本数看 `ls`，
  哈希当场跑 `sha256sum`。任务书里的数字是**线索不是基线**——实测与任务书不符时报实测值和差额。
- **§8 只追加**。后一轮推翻前一轮，给那一行**加脚注**，原文保留。
  把历史行改写成"一直是对的"会毁掉教训本身。
- **每轮收尾按 `git status` + `git log -1` 重写 §1 和 §2**，不许转抄上一轮的。

---

## 4. 目录与脚本索引

```
Alien-RPG Translation Project/
├── PROJECT.md              ← 本文件，唯一长期入口
├── PARALLEL-RUNBOOK.md     ← 多 agent 并行翻译运行手册（冷读即可上手）
├── 1-系统汉化插件/          alienrpg-cn        独立 git 仓
├── 2-新手包汉化插件/        alien-evolved-starterset-cn   独立 git 仓
├── 3-核心书汉化插件/        alien-evolved-corerules-cn    独立 git 仓
├── 4-常用脚本/             永久工具，从 EC 移植后重新指向
│   ├── extract/   mappings.mjs（唯一真相源）· extract_en.mjs
│   ├── tm/        build_glossary · fill_twin · term_gate
│   ├── qa/        apply_translations（唯一写入者）· apply_lang · lang_gap · flatten_lang
│   │              · capture_baseline · validate_translations · assert_resolutions
│   │              · scan_*（漂移/残留/markup/覆盖/术语）
│   │              · 【本项目新增】scan_name_lookup_traps · scan_table_ref_sync
│   │                · scan_crit_lockstep · measure_ratio
│   │              · verify_extract_leaves（抽取器的独立复算）· check_glossary_alien
│   │              · gate_fixtures/{build_fixture,build_gaps,run_gates}  ← 两道硬闸的双向回测 harness
│   ├── parallel/  prep_units · collect_* · diff_* · tagseq · prose_survival · merge_batches
│   └── release/   runtime-converters.js · generate_runtime.mjs
├── 5-临时脚本/<日期>-<轮次>/  一次性探针，**永不删除**
│   └── findings/            ⚠ .gitignore 里绝不许有能吞掉它的 `**/*.json` 规则
├── 6-工作区/                纯中间产物，可删可重跑
│   └── raw-dumps/           三个包的 LevelDB 原始导出
└── 7-其他内容/
    ├── DO-NOT-TRANSLATE.json    ⚠ 闸门运行时读它
    ├── RESOLUTIONS.assertions.json
    ├── glossary/            glossary_alien{,.provenance,.disputes,.pending}.json
    ├── english-baseline/<pkg>-<ver>/   历史英文快照 + LOCAL-PATCHES.md
    ├── findings/            人裁过、重跑不出来的产物
    └── reports/             可再生，gitignored
```

**会话级临时区**：`$env:ALIEN_PARALLEL_ROOT` 指向 scratchpad 下的波次工作目录。
**不在 git 里**——会话结束时没落地的东西就没了。

---

## 5. SOP

*（⚠ **Phase 0 已完成，本节仍然是空的——这是 Phase 1 开工前必须先补的第一件事**，
别在没有 SOP 的情况下开翻译波次。当前先记住 §3 的硬约束和 §7 的冒烟清单。）*

### 5.4 发版前全套扫描

*（待 Phase 1 建立后逐条填入。至少包含：`lang_gap` 五桶归零含 UNREACHABLE ·
`flatten_lang` 三数相等 · `validate_translations` 覆盖率 · markup 漂移 LINK/BLOCK/PLACEHOLDER=0 ·
`scan_name_lookup_traps` · `scan_crit_lockstep` · `scan_table_ref_sync` ·
`assert_resolutions` 主闸与 `--selftest` **两个数都要报**。）*

---

## 6. 年表与发版一览

| 日期 | 轮次 | 做了什么 | 发版 |
|---|---|---|---|
| 2026-08-28 | 勘察轮 | 11 agent 全面勘察：EC 方法论提取 · 系统 i18n 面 · 五个插件 i18n 面 · Babele mapping 设计 · 内容清单（98 个工作单元）· 字幕术语挖掘 · 网络术语调研；三路对抗式复核推翻 31 条断言、补出 23 条遗漏 | — |
| 2026-08-29 | Phase 0 | 目录脚手架 · 移植 EC 脚本并逐个重指向 · 业主拍板六条口径 · 补料后重排术语来源优先级（§1.4）· 三包英文抽取 + 独立复算 + 三包基线快照 · 硬冻结册子 · `glossary_alien` v0.2 · 两道硬闸建成并做了**双向**回测。5 路建设 + 3 路对抗式复核，推翻 23 条断言、补出 20 条遗漏，详见 §8 | — |
| 2026-08-29 | Phase 0 收尾 | **Phase 0 完成**。三个仓仍是 0 commit / 0 tag，`compendium/cn` 全空。Phase 1（翻译波次）尚未开始 | — |
| 2026-08-29 | Phase 1 | 系统 UI 590 键重译（覆盖 584，另加 16 个 en.json 没有、代码却在引用的键）· 运行时补丁两支。⚠ 分片险些静默互相覆盖（162 键无人认领 / 57 键多方认领 / `apply_lang.py` 无冲突检测），由对抗式复核救回 | — |
| 2026-08-29 | Phase 2 | 系统自带 Adventure 包全译 81.8 KB。单元按 `<h1>` 字节区间切，铺满断言 66,228/66,228 无缝无叠。长度比 band 首次在真实语料实测（中位 0.36） | — |
| 2026-08-29 | **发版** | **`alienrpg-cn v0.1.0` 公开发布**。产物逐项核对：zip 27 个成员 · 排除清单生效（`compendium/en/*`、`lang/en.json`、`lang/_baseline.json`、`lang_keep_english.json`、`.github/*` 均不在包内）· 7 个必备文件齐全 · manifest 可下载。⚠ **未做实机冒烟** | ✅ |

---

## 7. 待办与冒烟清单

### 7.1 冒烟验证清单（⚠ 最高优先级）

> EC 项目**从首版至今每一次发版都是在没做冒烟的情况下发出去的**，它自己的文档里写着这句话。
> 本项目**不重复这个错误**——每个 Phase 的退出条件都含冒烟，且冒烟不许延后。

⚠⚠ **必须开两个世界。** `module/helpers/settings.mjs` 的 `evolved` 设置是
`scope:"world", default:true, restricted:true, onChange:location.reload()`——
**Evolved 与 Classic(1e) 是世界级二选一**，同一个世界里看不到另一套的卡片。
而 `lang/en.json` 的 590 键**两套规则的 UI 串都在里面**（`ALIENRPG.Evolved` 的说明就是
"Switches from Alien Classic sheets and rules to Alien Evolved sheets and rules"）。
⇒ 冒烟必须 **Evolved 世界一遍 + Classic 世界一遍**，否则有一半卡片的标签从没被人看过。
内容包（corerules / starterset）都是 Evolved 内容，只在 Evolved 世界验。

以下缺陷类**对一切静态检查不可见**，只有真开一个世界才看得见：

- [ ] 9 个首次导入查找名——装上汉化后**首次**导入三个包，确认 Adventure / 欢迎日志 / 场景都找得到
- [ ] 重伤 healTime——中文环境下掷一次重伤，确认恢复时间不是 0
- [ ] 怪物攻击表——中文环境下让异形攻击一次，确认 `rTables` 找得到表
- [ ] 枚举型字段被当散文翻译（EC 实测：104 个中文值写进地形枚举，运行时移动开销直接坏掉，
      而每一项静态检查都是绿的）
- [ ] `#anchor` 链接是否还跳得动（翻标题会改 Foundry 的 slug）
- [ ] MU/TH/UR 命令解析：`HELP` / `HACK` / `CERBERUS` / `SPECIAL ORDER 937` / 裸 `937` / `/M <中文消息>` 往返
- [ ] terminal 的 `help` / `ls` / `ssh` 表格边框是否还对齐（CJK 双宽 vs `padEnd`）
- [ ] 十二个 skill-stunts 物品名是否与 lang 值逐字节相等（做成断言，不靠眼看）

### 7.2.0 ⛔ 已确认拿不到的（**不要再去找**）

| 材料 | 结论 | 日期 |
|---|---|---|
| **异形 RPG 第一版（非进化版）中文翻译** | **不存在可溯源的副本。** 业主查证：纯美苹果园那批帖子年代太久，信息已经消失，无法溯源 | 2026-08-29 |
| 純美蘋果園 topic=121082【翻译】异形RPG潜行规则（和其它）| 同上，随该站旧内容一并失效 | 2026-08-29 |
| 《异形RPG：进化版》第 3 章及以后的中译 | **论坛上只连载了第 1、2 章，没有别的**（业主确认）| 2026-08-29 |

⇒ **别再为这三项开搜索轮。** 它们是**已裁定的负结果**，写在这里是为了防止未来的会话
（包括未来的我）看到词表里的 `pending` 就以为是遗漏、又去找一遍。

**由此产生的结构性后果 —— 本项目就是中文异形 RPG 的第一份术语表。**
`Facehugger` / `Chestburster` / `Ovomorph` / `Drone` / `Warrior` / `Praetorian` 这一组
在 11,088 对字幕里**零命中**（电影里根本没人说过这些词），进化版前两章也够不到生命周期（那在第 10 章），
而 1e 中译已确认不存在。⇒ **它们没有任何中文先例可循。**

处理办法（不再等外部材料）：

1. **按构词法自裁**，并在 `provenance.json` 里把 `source_level` 标成 `0-本项目自定`，
   **不许伪装成有出处**。已定：`Facehugger = 抱脸虫`（业主裁定）。
2. **组内自洽优先于逐词最优**：这是一个**生命周期序列**（卵 → 抱脸虫 → 破胸体 → 成体），
   六个词要读起来像一套，不能一个用「虫」一个用「体」一个用「者」。
3. **落进 `disputes.json` 并在第 10 章翻译时最终定稿** —— 那一章的正文会给出每个形态的
   完整描述，届时按描述回头校准比现在凭词典拍板准。
4. ⚠ **一旦定稿就进 `RESOLUTIONS.assertions.json` 钉死**，因为后面 25 个 creature actor、
   123 张 RollTable 和大量正文都会引用它们，改一次的爆炸半径很大。

### 7.2 外部材料：已到货的、还挖不了的、真缺的

⚑ **2026-08-29 大批到货，这一节整节重写。** 上一版按价值排序的 7 项里，**1–6 项都已由业主提供**
（其中第 3 项转成了 §7.2.0 的负结果）。原始文件落在 `C:\Users\Taka\Desktop\fvtt\AlienRPG\`，
派生物落在 `7-其他内容/reference/` 与 `7-其他内容/glossary/`。

**已到货，且已经挖进语料**：

| 材料 | 落在哪 | 状态 |
|---|---|---|
| 《异形：夺命舰》Romulus 2024（简体 / 简体&英文 / 繁体 / 英文，ass + srt）| `AlienRPG/Alien.Romulus.2024.…-FLUX/` | ✅ 已对齐进 `reference/subtitles/`。**独立血脉**，术语优先级第 2 级。⚠ 手上这份是 WEB-DL 随片轨，不是院线公映轨，专名上可能有出入 |
| Alien: Earth 2025 S01E01–E04（ass）| `AlienRPG/Alien.Earth.S01E01-04.WEB.DDP5.1/` | ✅ 已对齐。独立血脉，第 2 级 |
| Covenant 2017（ass）· Prometheus 2012（srt，gb18030）| `AlienRPG/` 根目录 | ✅ 已对齐。各自独立血脉，第 5 级 |
| **Alien: Isolation 官方简中** | `AlienRPG/简体汉化/` | ✅ 已挖成 `glossary/mined_alien_isolation.json`，读取器是 `tm/read_isolation.py`。⚠ 三条实测事实见下 |
| 《异形RPG：进化版》第 1、2 章中译 | `reference/cn-fan-translation/*.txt` | ✅ 已挖成 `glossary/mined_fan_translation.json`。术语优先级**第 1 级**。⚠ 只有这两章，§7.2.0 已裁定 |
| bilibili cv18100844 的设定/时间线中译（原第 4 项）| `reference/cn-fan-translation/《异形RPG》版异形宇宙设定翻译·时间线·第一部分….txt` | ⚠ **到了，但只有第一部分，而且是 1e 的、不是进化版的**（业主自己在文件名上标了）。只当低置信度旁证 |

> ⚠ **Alien: Isolation 那棵树的三条实测事实，不要再去重新发现**（`tm/read_isolation.py` 头部有完整记录）：
> ① 磁盘上 768 个 `.TXT`，**只有 64 个是不同的**——DLC 目录下是 11 份逐字节副本，md5 实测 768 路径 → 64 哈希。
> 按 768 个数会把每一项统计**乘以 12**。② 编码是 **UTF-16LE with BOM**，不是 utf-8-sig。
> ③ **这棵树没有英文侧**（目录叫 ENGLISH，里面全是中文），所以只有两道真英文闸——键名本身是英文 slug，
> 以及本地化没翻、内嵌在中文值里的英文专名。实测**只有约 19.5% 的活键**过得了闸，剩下八成是不透明行号，
> **只能当旁证，不能单独定案**（§3.3 的"绝不从裸中文词频得结论"在这里咬得最紧）。
> 去重后的真实规模与流传的那个 12 倍数**以 `mined_alien_isolation.json` 和读取器为准**。

**已到货，但今天还挖不了 —— 卡在 OCR**：

| 材料 | 落在哪 | 为什么还没用上 |
|---|---|---|
| Alien (1979) DC 蓝光官方字幕 · Aliens (1986) SE 蓝光官方字幕，两部各带**繁中国语**与**繁中粤语**两轨 | `AlienRPG/*.idx` + `AlienRPG/*.sub` | ⛔ **VobSub 是位图字幕，不是文本**。`.sub` 里是行程编码的图像，`.idx` 只有时间戳与调色板，`grep` 不到一个字。**这恰恰就是原第 6 项要的"非 CnSCG 的独立血脉"，是这批材料里价值最高的一份，但今天一条都没进语料。** |

> **⚠ 这是一条未完成项，不是一个已完成的来源。别在术语裁定里引用它。**
>
> **规模（实测自 `.idx` 里 `timestamp:` 的条数，不是估的）**：中文侧四轨共 **3,796** 条——
> Alien 1979 国语 747 / 粤语 771，Aliens 1986 国语 1,137 / 粤语 1,141。
> 英文侧两轨另有 1,877 条，但**英文侧不必 OCR**：本机两部片都已有英文文本字幕，按时间码对齐即可建英文闸。
>
> **成本形状（估算，不是实测——本机没跑过，`vobsub2srt` 与 tesseract 中文包都还没装）**：
> 一次 VobSub → 文本的 OCR（Subtitle Edit 或 `vobsub2srt` + Tesseract `chi_tra`）→ 繁转简（OpenCC）→
> 按时间码与英文文本轨对齐 → **人工校对**。CJK 位图 OCR 的错字率决定了校对不能省。
> 按每轨一个工作单元估，**四轨约四个工作单元**，其中人工校对占大头。
>
> ⚠ 两条是**粤语**配音轨的字幕，用词与国语轨会系统性不同。它们不是彼此的副本，但也**不许各算一票**——
> 同一张蓝光盘的两条轨，按 §3.3 的血脉规则折算；`tm/term_vote.py` 的 `LINEAGE` 表届时要加一条，
> 别让它掉进 `other:` 桶里当独立血脉计票。

**仍然真缺的**：

| 材料 | 为什么还想要 |
|---|---|
| **Alien RPG Evolved Edition 核心书 PDF**（DriveThruRPG 付费）| **唯一一项还没有任何替代品的材料。**今天只能看到 Foundry 暴露出来的英文键名，看不到书里对 Resolve / Stress Response / Panic Response 的定义；也校不了第 10 章的生命周期形态描述——而 §7.2.0 说了，那一章正是六个形态词最终定稿的地方 |

⇒ **除 OCR 之外，不要再为外部材料开搜索轮。** §7.2.0 那三项是已裁定的负结果；
其余 1–6 项都到货了。剩下的工作是**挖**（OCR + 对齐），不是**找**。

### 7.3 已知的上游 bug（不是我们的锅，但会被当成我们的锅）

- `module/sheets/active-effect-config.mjs:21,24` 指向 `systems/alienrpglates/...`——
  一个不存在的系统 id，一次失败的查找替换。**ActiveEffect 配置表当前就是坏的**，
  它那三个模板（以及里面 6 个 `DRAW_STEEL.*` 遗留键）**根本加载不到**。
  ⇒ 不要花力气翻它们，也不要为此在 cn.json 里加 `DRAW_STEEL` 块。**应当给上游报 bug。**
- `module/helpers/effects.mjs:15,20` 用 `alienrpgct.Passive` / `alienrpgct.Inactive`——
  这个命名空间**在任何包的任何语言文件里都不存在**，两处会渲染成裸键。
  ⇒ 我们的 cn.json 加一个顶层 `alienrpgct` 块就能修。
- `motion_tracker/module.json` 的 `es` 条目指向 `lang/en.json`（西班牙语用户看到英文），
  而 `lang/es.json` 就在旁边没人用。
- `ALIENRPG.RollManShipMinorCrit`：英文说 "between 1 and 66"，中文说"介于 1 和 44 之间"——数值不符。
- `ALIENRPG.Seepage106`：英文说 page 105，中文说第 106 面。上游 issue #137 在 2021 年就报过，至今没修。
- ⚑ **`alien-evolved-corerules` 的欢迎日志查找今天就是坏的，与汉化无关。**
  `module/init.js:11` 写的 `CORE RULES - HOW TO USE THIS MODULE` 用的是 **ASCII 空格**，
  而包里那篇 JournalEntry 的名字**五个间隔全是 U+00A0**。
  ⇒ `getName()` 返回 null，`init.js:115` 的 `.show()` 当场抛异常，
  于是下一行的 `Hooks.off("importAdventure")` **永远不会执行**。
  **报上游时把这条一起报**——否则将来有人会把它算到汉化头上。
- `character-sheet.mjs` 的 `_stuntBtn` 第二条腿（`"ALIENRPG." + name.replace(/\s+/g,"")`）
  在 4.1.13 里是**死的**——那 12 个派生键在 en.json 里一个都不存在。
  ⇒ **不要**为它们在 cn.json 里建块。
- `game.tables.getName()` 的 10 个字面名里有 **2 个在任何包里都匹配不到文档**
  （`Critical Injuries` 大写 I、`critical injuries on synthetics`）——死腿，登记下来是为了防止有人拿去复用。
  经典模式的回落实际落在小写的 `Critical injuries` 上。

### 7.3.1 本机环境陷阱（2026-08-28 那次模组勘察的结论，直接继承）

- ⚠ **`motion_tracker` 不能走包管理器更新。** 本机装的是 1.5.5（manifest 指向 main 分支 raw，滚动安装）；
  Foundry 包库上发布的"最新版"是 **1.5.4 且 `maximum 13`，V14 硬锁**。
  而 `motion-tracker-multideck` 硬依赖 `motion_tracker >= 1.5.5`。
  **点一次"更新"会同时废掉运动追踪器和多层甲板伴侣。**
- 系统 `macros/` 文件夹里的宏没有 compendium，UI 里看不到，要手动导入。
- `yze-combat` 会崩场景（上游 issue #93，重复 statusEffect id），且系统本就自带抽牌先攻——不要装。
- 官方还有 Heart of Darkness（155 篇日志）/ CMOM（181 篇）等 1e 内容包，**本机未安装**。
  它们是同一套 Babele 工作流的天然续作，但属于 1e，要开独立的 Classic 世界。

### 7.4 上游回馈

`pwatson100/alienrpg` **接受 PR**——issue #282 里维护者对"我想翻译成波兰语该怎么开始"的回答
就是 "just open a PR"，issue #241 显示他也接受 issue 附件里的 zip。
已有至少四位外部贡献者的翻译被合入。

⇒ **先发我们自己的覆盖模块（不被上游阻塞），等它经过一次发版验证后，
把能 1:1 映射到 en.json 590 键的那个子集提 PR 上去**，顺带报 §7.3 的两个 bug。
上游若合并，我们的覆盖对那些键退化成无害的 no-op，且仍然赢得 merge，没有冲突态要管。

---

## 8. 裁决日志（**只追加**）

### 2026-08-28 · 勘察轮

**11 个 agent 并行勘察 + 3 路对抗式复核。** 复核推翻 31 条断言、补出 23 条遗漏。
原始 finding 全文落盘在 `7-其他内容/findings/2026-08-29-survey/`（11 个 JSON）。

被推翻的重要断言（**复核方为准**）：

| 原断言 | 实际 |
|---|---|
| Item 变体 `_when {type: 'crit-inj'}` 提供前向兼容 | `crit-inj` 是**文件名**，注册的类型是 `critical-injury`。该变体永不命中，是死代码 |
| `system.modifiers.*.label` 是派生字段（`prepareDerivedData` 会覆写） | **13 个 item 类型全部覆写了 base-item 的 `prepareDerivedData` 为空体**，且 `feature` 根本不是注册类型。DO-NOT-TRANSLATE 的结论只剩一条腿（模板从不渲染它） |
| `system.header.type.value` 是 '0'/'1' 枚举 | 实测跨三包只有 '1'–'9'。引用的 '0'/'1' switch 被 `type === 'spacecraft-crit'` 守着，而三个包里**零个** spacecraft-crit 物品 |
| `sigItem` / `relOne` / `relTwo` 全是 'None' | 全错。sigItem 有真内容（Toy dinosaur / Company ID badge），relOne/relTwo 存的是**人物姓氏**（MacWhirr / Hirsch / Sigg），且必须跟着译名走 |
| 正文链接是 `@Item[...]` / `@JournalEntry[...]` | 是 v10+ 的 `@UUID[Item.<id>]{label}`。按原断言写的正则会零命中并放行链接全毁的文件 |
| 三个包"无需自定义 mapping 即可直接用 Adventure 默认层" | Babele 默认 Item 指 `system.description.value`、Actor 指 `system.details.biography.value`，本系统两个都不用——**约 27.8 万字会静默保持英文** |
| `EVCriticalInjuries` 缺键 | 是 JSON `null`。且 `localize` 的失败机制是**先落回英文 fallback 字典**，不是返回键名 |
| `DRAW_STEEL.*` 是需要修的活面 | 那三个模板**根本加载不到**（见 §7.3），是死面。不要为它加 cn.json 块 |
| 重伤 switch 用 ASCII 连字符 | 用的是 **EN DASH U+2013**。照原文抄的判据永远不会命中，静默丢掉 `cFatal=true` 分支 |
| 96 个工作单元 / 8 张零文本表 118 条结果 | 98 个单元 / 9 张表 165 条结果 |
| 5 个锚点单元约 11.8 万字 | 11.8 万是**可见**字数，与全项目所有单元用的**原始**字数不同币种；串行关键路径被低估 66-87% |

复核补出的重要遗漏：

- **9 个首次导入查找名**——三个包都做无守卫的 `===` 比对 Adventure 名/欢迎日志名/场景名。
  比勘察发现的 RollTable 陷阱更严重，因为它直接砸掉首次导入。
- **corerules 正文里 33 个 `@TEXTDRAW` + 3 个 `@DRAW`**，`{label}` 复制的是表名——表名的第二处书写地点。
- **42 个内嵌物品与包级物品 `_id` 相同**（corerules 38 / starterset 4）。恰好同名所以安全结论仍成立，
  但 Babele 的默认 Item 匹配顺序是 **`_id` 优先**，这 42 个根本不是按名字匹配的。
- **`module/` 下除 62 个 `.mjs` 外还有 12 个 `.js`（5,428 行）从未被扫**，其中两个在活的 import 图里
  且含用户可见英文：`devmsg.js:24` 的 `alias: "Alien RPG News"`（GM 就绪私聊的常驻英文署名）。
- **`alienrpgct.Passive` / `.Inactive`** 命名空间在任何包里都不存在（见 §7.3）。
- **`creature-header.hbs:26,42,48,54` 四个硬编码 `data-label`**（Speed / Mobility / Observation /
  Acid Splash），它们会顺着拼接点流进对话框标题和聊天 flavour，产出「检定 Speed 检定」。
  而报告明说"零个 data-tooltip/aria-label 问题"。
- **Dice So Nice 注册串**（colorset description 'Yellow'/'AlienBlack'、dice-system name
  'Alien RPG - Blank'/'Alien RPG - Full Dice'）——一整类从未被审计的用户可见面。
- **`"{MISSING_CREW}"`** 字面占位符被当船员名渲染在飞船/载具卡上。
- **Babele 自己的 `importAdventure` 钩子 + `syncImportedAdventureTokenNames` 世界设置（默认 true）**
  会在导入后用 Actor 的 `prototypeToken.name` 重写场景 token 名——会覆盖 `Scene.tokens` mapping 写的东西。
- **Babele 2.9.1 自己没有 `cn` 语言**（只有 ar/en/it/es/de/fr/pl/zh-tw/ja），
  所以中文世界里 Babele 的设置界面是英文的。

**方法论结论（从 EC 继承，本项目直接生效）**：见 §3 全节。
其中最贵的三条是 §3.2 的基线全量快照、§3.5 的三方合并与 `--force` 语义、
§3.3 的英文闸计数。这三条 EC 都是栽过之后才写下来的。

### 2026-08-29 · Phase 0

业主拍板六条口径（见 §1.3）。核心书**公开发布**——风险已当面提出（Free League 已授权七种语言
不含中文；异形 IP 归 20 世纪影业；该模块 `protected: true` 是付费内容），由业主裁定接受。
唯一保留的技术性收口：`compendium/en/*` 排除出所有发布 zip。

建立目录脚手架，移植 EC 的 69 个脚本（26 个含硬编码 EC 路径待重指向）。
把勘察轮的 11 份 finding 与 252 条术语种子落进 `7-其他内容/`——
**不放在 `5-临时脚本/` 下**，因为 EC 的 `.gitignore` 里那条 `4-临时脚本/**/*.json`
曾把人裁过、重跑不出来的 finding 静默挡在仓外。

### 2026-08-29 · Phase 0 收尾（同日第二轮，**接在上一条之后，不覆盖它**）

**5 路建设 + 3 路对抗式复核。** 复核推翻 **23** 条断言、补出 **20** 条遗漏
（mapping **4 / 6** · register **7 / 5** · skeleton **12 / 9**）。
原始 finding 全文落盘在 `7-其他内容/findings/2026-08-29-phase0/`（8 个 JSON：5 份建设 + 3 份复核）。

**建成了什么**（数字一律以产物为准，这里只记形状）：

| 产物 | 落在哪 | 状态 |
|---|---|---|
| 三包英文抽取 | 三个仓的 `compendium/en/` + `_source.json` | 5,023 叶 / 2,705,264 字符，`--strict-mappings` 干净 |
| **独立复算** | `4-常用脚本/qa/verify_extract_leaves.py` | 用 Python 重新实现 `_variants` / `_when` / 每个 converter / key 分配，读**原始 dump** 而不是 LevelDB：两侧 5,023 / 2,705,264 **逐路径相等**，only-in-raw 0、only-in-extractor 0、值不等 0 |
| 英文基线快照 | `7-其他内容/english-baseline/<pkg>-<ver>/` | **三个包一个不落**——§3.2 那条 EC 最贵的 bug 的直接对策 |
| 唯一真相源 mapping | `4-常用脚本/extract/mappings.mjs` | 69 行 FIELD_CENSUS 全部与 dump 实测相符；`BABELE_DEFAULTS` 与装机的 Babele 2.9.1 **JSON 严格相等（含键序）** |
| 硬冻结册子 | `7-其他内容/DO-NOT-TRANSLATE.json` | 11 节 + 5 条上游缺陷 + C1–C9 |
| 术语表 | `7-其他内容/glossary/glossary_alien{,.provenance,.disputes,.pending}.json` | 本轮产出 **v0.2**：346 词条 / 9 争议 / 32 pending，**零条来自裸中文词频**，字幕语料正确折成一票。⚑ 同日晚些已被 **v1.0** 取代（Phase 0 那一版的争议快照留在 `glossary_alien.disputes.phase0.json`）——**当前数一律以 `provenance.json` 的 `_meta` 为准，别引用这一行的数** |
| 三个模块骨架 | `1-` / `2-` / `3-` 三个仓 | `module.json` · `register.js` · 发版工作流齐；`compendium/en` 已由**两道**独立的闸排除出发版 zip |
| 两道硬闸 | `qa/scan_name_lookup_traps.py` · `qa/scan_crit_lockstep.py` | 见下 |

#### 闸门证明（本轮最有价值的产出）

**两位闸门作者都做了双向回测**，harness 是
`4-常用脚本/qa/gate_fixtures/{build_fixture,build_gaps,run_gates}.py`（已从 scratchpad 收进项目，可重跑），
全部 stdout 落在册子的 `gate_tests_2026_08_29` 与 `gate_tests_2026_08_29_pass2` 两个块里。

| 方向 | 夹具 | `scan_name_lookup_traps` | `scan_crit_lockstep` |
|---|---|---|---|
| Day-one（真项目，`compendium/cn` 空）| — | checked=0 viol=0 exit=0 | checked=8 viol=0 exit=0 |
| **特异性**（GOOD：三个包一份**合规**的全量中译）| 第一轮 | checked=123 viol=0 exit=0 | checked=132 viol=0 exit=0 |
| **特异性** | 第二轮（补完之后）| checked=195 viol=0 exit=0 | checked=190 viol=0 exit=0 |
| **敏感性**（单点注入）| 第一轮 · 17 处注入 | checked=124 **viol=14** exit=1（FROZEN_TRANSLATED 9 / EXACT_MISMATCH 1 / TABLE_REF_DANGLING 4）| checked=129 **viol=9** exit=1（CELL_MISMATCH 5 / LANG_BAD 1 / SPLIT_SHAPE 1 / ROLL_SHAPE 1 / SHIFT_TRANSLATED 1）|
| **敏感性** | 第二轮 · 23 个变异体 | 每个变异体在**管它那条规则的那道闸**上 exit=1、在另一道上 exit=0；唯一例外是故意的 `M13_bad_json`（不可解析的 JSON，两道都炸）||

⚠ **光测坏样本是不够的——建 GOOD 夹具本身就抓到了一个假阳性。**
`EV - Critical Injuries` 的 11-11 行在两个包里都以一个多余的 `<br />` 结尾，切出 **11** 段而不是 10，
而闸只容忍 expected−1，于是**一份忠实于上游的译文会被判死**（这就是更正 C3）。
⇒ **闸门回测必须双向**：只测坏样本的闸，会把"忠实"当成"违规"，而这种误报会训练人去忽略闸。

#### 七条真实的、静默的、会毁掉玩法的改动，在补完之前**两道闸都放行**（violations=0 / exit=0）

这是本轮最贵的一条发现。补完之后七条全部被拦下：

| 夹具 | 是什么 | 补完后被谁抓住 |
|---|---|---|
| `G1_danger_event_table_translated` | corerules 那个 Macro 的 `command` 里 `getName()` 的目标表名被翻 | `rolltable_names` 第 11 条 |
| `G2_hardened_stoic_talents` | `HARDENED` / `STOIC` 天赋名被翻——**当场改人物数值，不报错** | `item_names` 第 3–6 条 |
| `G3_spacecraft_crit_split` | 飞船重伤行里的 `<br /><br />` 被"整理"成一个 | `crit_parse_lockstep.other_branches` |
| `G4_synthetic_crit_split` | 合成人重伤行里冒号变成全角 `：` | 同上 |
| `G5_rpg_launcher_name` | `M5A3 RPG Launcher` 改名丢掉 ` RPG ` | `name_substring_tests` |
| `G6_macro_command_translated` | `Macro.command` 被发进 `compendium/cn` 并翻译 | `macro_commands` |
| `G7_xenomorph_crit_split` | 异形重伤行里冒号变成全角 | `other_branches` |

**为什么七条全逃过了闸**（根因，写下来是为了别再犯）：

1. **只扫了模组 JS。** 名字查找**也住在包内容里**——corerules 自带的那个 Macro 的 `command` 字段
   就在调 `game.tables.getName()`，而 Babele **默认就翻 `command`**。G1 / G6 都是这一条。
2. **重伤解析只登记了 `case "character"` 一条腿。** 同一个文件里还有 `synthetic` / `creature` 与
   `spacecraft` 两个分支，**切分器不同**（`/[:] |<br \/>/gi`，没有 `<br />`→`<br>` 归一化）、
   **固定下标不同**，另外覆盖四张表。G3 / G4 / G7 都是这一条。
3. **只登记了"整串相等"，没登记"子串测试"。** 弹药重量是按物品名 `includes(' RPG ')` 判的。G5 是这一条。

#### 被推翻的重要断言（**复核方为准**）

| 原断言 | 实际 |
|---|---|
| 册子由 `build_register.py` 生成，`cite()` 失败即构建失败，"citation 漂移会让构建失败，而不是让文档撒谎" | **生成器已经生成不出这个文件了。**重跑它会删掉 C1–C9、两个 `gate_tests` 块，以及 `scan_crit_lockstep.py` 赖以避免误报的 `rows_with_nonstandard_split`。已在 JSON 顶层加 `_generator_status` 自曝警告。**这是本轮最高价值的遗留风险** |
| 硬编码 RollTable 名 **10** 个 | **11**。第 11 个在 corerules 包自带 Macro 的 `command` 里，`.formula` 解引用**无守卫**，而且抛在 async 对话框回调里，**用户侧完全静默** |
| 冻结物品名 **2** 个（`PACK MULE` / `TAKE CONTROL`）| **6**。`actor-character.mjs` 与 `actor-synthetic.mjs` **各带六个** `toUpperCase() === "..."`；漏掉的四个里 `HARDENED`（+1 最大生命，**两个模式都吃**）与 `STOIC` 在 corerules 包里有活文档 |
| 重伤锁步 = 一个 switch、两张表 | **两个 switch**（`[3]`→cFatal、`[5]`→healTime）**外加一条不在 switch 里的 `if` 链**（`[9]`，`Permanent` / 裸 `Shift`）；另有三个分支、另外四张表、另一个切分器 |
| 裸 `Shift` 与两个 switch 同类，落空就静默取默认值 | **不同类**。两个 switch 的 `default` 静默落到 `cFatal=false` / `healTime=0`；`Shift` 被翻则落进 `[[NdM]]` 正则，`.match()` 返回 null，`null[1]` 抛**未捕获 TypeError**——整次重伤检定死在半路 |
| corerules 欢迎日志的 NBSP 版本"已经是坏的，所以冻结 ASCII 空格形式" | **正好反了。**字节转储证明 `init.js:11` 的 JS 字面量间隔全是 U+00A0，与那篇 JournalEntry **逐字节相同**，查找是**活的**。旧条目会冻结一个永不命中的串，还会用"反正已经死了"授权去翻那个日志名，**真的砸掉核心书导入后的流程** |
| corerules 欢迎日志"**五个** U+00A0" | **六个**（同一条目的 `string_codepoints` 字段本来就是对的，错的是散文）。一个只靠逐字节吃饭的册子，码位数差一就是同类缺陷 |
| `system.attributes.class.value`（weaponClass）是自由文本 | 在 JS 里被拿去和字面量比——**正是 EC 那个地形枚举的失效模式** |
| `Alien Tables` / `Alien Sub-Tables` 安全 | `Alien Tables` 是首次导入的**否定守卫**（`!game.folders.getName('Alien Tables')`），翻了它世界**每次启动都重导一遍**。而 `Alien Sub-Tables` 确实安全，`Folder#contents` 非递归 |
| 三个 `update.js` 里的 `compared_at` 是活比对点 | 三处**全都够不到**：`updateModule()` 开头 `updateAssets` 是空数组并直接 `return`。同一片死区里还藏着**四个未登记的 `getName()`**，上游一旦填回 `updateAssets` 就活过来 |
| T-EXACT 断裂只影响 character / synthetic 的 stunts 面板 | 同一个 `game.items.getName(dataset.pmbut)` 跑在**四张卡**上。character / synthetic 被 `if (!evolved)` 关在**经典模式**里；**spacecraft / vehicle 无守卫，两个模式都跑**——进化版里能坏的恰恰是原文没提的那两张 |
| `i18nInit` 在 `init` 钩子**之前**触发 | 正好相反，`init` 先触发。结论侥幸没错，但按这个反的模型去"简化"补丁就会坏 |
| 在 `setup` / `ready` 里注册 mapping 会"**直接报错**" | `#assertConfigurable` 在还没有 session 时**提前返回**，`setup` 期注册不报错也能用。把例外当铁律会掩盖真正的钩子顺序问题 |
| `ALIENRPG.conditions` 有 20 项且不含 `fatigued` | 对 `config.mjs:161` 做括号配对是 **21** 项，`fatigued` 就在里面。术语表因此漏了 Fatigued，已补 |

#### 复核补出的重要遗漏（选载）

- **把 `compendium/cn` 注册给 Babele 会同时开出第二条 mapping 通道。** 同一个目录既当
  `translationDirectories` 又当 `mappingDirectories`，而 Babele **纯按文件名**决定一份 `.json` 是哪种——
  `mappings.json` / `mapping.json` 在那个目录里是**保留字**。
- **模组一的发版工作流没有"有东西可发"的闸**（模组二、三都有）。按当前骨架它会**发出一个空壳**：
  `lang/cn.json` 3 字节 `{}`，四个 plugins lang 也是 3 字节，`compendium/cn` 只有 `.gitkeep`，
  而每一道现存检查都会通过。
- **模组 lang 文件能覆盖系统键，但永远删不掉系统键。** `mergeObject` 默认 `applyOperators=false`，
  `-=key` 语法是死的，`performDeletions` 在 V14 已弃用。要让某个键落回英文，**只能把英文写进去**。
- **两个运行时补丁器用裸 `Handlebars.compile(source)` 预注册 partial**，
  而 Foundry 自己用 `{preventIndent: true}`——被 `{{> ...}}` 内联时输出会被重新缩进。
- **`Scene.grid.units`（`'m'` × 11 / `'ft'` × 4）是全语料唯一一个既没被覆盖、又用户可见的字符串叶**，
  两份名册里都没有它。
- **`Actor.system.modifiers.<skill>.label` 不在 DO-NOT-TRANSLATE 名册里，而它的兄弟 `.ability` 在。**
- **`Adventure.name` / `JournalEntry.name` 本身就是查找键**，且好几个读取点**没有 null 检查**——
  Babele 一旦翻了包索引，首次导入就先炸，轮不到那些 RollTable 字面名。
- **`Critical Injuries on Xenomorphs` 不能用标准 T-BILINGUAL 形式**：它是被
  `name.startsWith('Critical Injuries')` 过滤取用的，英文前缀必须留在偏移 0（见 §3.4 表下注解）。

#### 方法论结论（**新增，本项目自己栽出来的**）

1. **闸门必须双向回测，而且 GOOD 夹具要从 `compendium/en` 和册子本身派生。**
   只测坏样本的闸会把忠实译文判死（C3 就是这么现形的）；而 GOOD 从册子派生意味着
   **往册子里冻一个新串，夹具下一次自动带上它**，闸和登记表不会各走各的。
2. **"扫过了、干净"和"扫的面不全"在输出上长得一模一样。**
   EC 是在**基线包数**上栽的（§3.2），本项目是在**扫描面**上栽的——只扫模组 JS 没扫包内容里的
   `Macro.command`，只扫重伤解析的一条腿没扫另外三条。
   ⇒ **每道闸都要能报出它覆盖了哪些面**，而不只是报违规数；`checked=` 计数是这条的最小实现。
3. **生成器与产物一旦脱钩，必须在产物里写一条自曝的警告。**
   册子的 `_generator_status` 是这条的第一次应用：它不修好生成器，但它保证下一个人不会因为读了
   "由脚本生成"这句话就去重跑，从而静默删掉九条更正和两个回测块。
4. **业主补料会推翻上一轮的术语主轴——先看有没有"这本书自己的中译"再定优先级。**
   Phase 0 的第一版词表以维基为主轴；补料之后 §1.4 整个重排，
   `Weyland-Yutani` / `Nostromo` / `Sulaco` / `Synthetic` 等一批已定条目当场改写。
   ⇒ 术语主轴在拿到全部可得材料之前**不要钉死**，但**每一条都要留 provenance**，否则重排时无法批量回溯。

### 2026-08-29 · 词表重铸轮（业主补料之后，**取代 Phase 0 那份维基主轴的词表**）

业主一次性补齐了勘察轮列的第 1–6 项，其中三样把证据格局整个换掉了：
**《异形RPG：进化版》第 1、2 章中译**（就是这本书本身）· **Romulus 2024 / Alien: Earth 2025 字幕**
（最新、面向大陆）· **Alien: Isolation 官方简中**（768 个 TXT / 16.3 万键 / 675 万字）。
⇒ 业主拍板把主轴从 zh.wikipedia 换成**本书中译优先**（§1.4），并钉死两条例外
（Wits=机智、Empathy=共情），其余授权主控自裁。

**结果**：404 条（原 346），重裁 128 条、新增 61 条、6 条降入 pending、7 条从 pending 升出。
胜出层级 `L1=140 / L2=11 / L3=8 / L4=48 / L5=1 / L6=50 / L7=6 / 惯例=101 / FROZEN=31 /
LATIN=4 / RULING=4`——**zh.wikipedia 从主轴掉到只剩 6 条胜出**（全是片名与两条 Queen 相关）。
⚠ 条数以 `glossary_alien.provenance.json` 的 `_meta.count` 为准，本段的数只是当轮快照。

**词表现在是可重跑的构建物，不是手改的文件**：`4-常用脚本/tm/build_glossary_v1.py`
（+ 四个模块）读三份 miner 产物、`lang/{en,cn}.json`、`DO-NOT-TRANSLATE.json` 与
`RULINGS.json`，产出五个文件；**16 条不变式在写盘之前把闸**，任何一条不过就打印
`NOTHING WRITTEN` 并 `exit 1`。连跑两次五个文件逐字节相同（已实测 md5 相同）。

**两条本轮才发现、且都会静默坏事的**：

| # | 事实 | 后果 |
|---|---|---|
| ① | 一级源（本书中译）把天赋 `Hardened` 译作 **硬汉**，而 `actor-character.mjs:449` / `actor-synthetic.mjs:439` 在比 `Attrib.name.toUpperCase() === "HARDENED"`，比对对象是**今天就发货的文档**（corerules 物品 + actor `EV - MINING WILDCATTER`）| `'硬汉'.toUpperCase()` 还是 `'硬汉'` ⇒ **角色静默少 1 点最大生命，不报错**。⇒ **最高级别的来源也会把游戏翻坏**，一级源不是免检 |
| ② | 一级源另有四条把专名当普通词翻的硬伤：`Hydr8tion` → 8号药剂（把 leetspeak 的 8 读成型号）· `Armat Model 37A2` → 军用型37A2（厂商名读成形容词「军用」）· `Watatsumi DV-303` → 海神DV-303（日本公司名意译）· `Lasalle Bionational` → 生物国家（丢掉前名） | 已全部隔离在 `confidence: low`。**采纳一级源要过 T-FROZEN 闸与专名闸，不能整体信任** |

**词表自身的七条 Phase 0 缺陷全部修掉**，其中最有方法论价值的一条：
`_meta.lockstep_literals` 里的 actor.mjs 行号**不再转抄，改成每次构建按模式从装好的
`actor.mjs` 现推**。Phase 0 抄错过三个锚点，v0.2.1 手改之后 disputes 文件里**又留了一个旧的**
——这正是 §3.7「轮次数字一律不转抄」要防的形状。现在上游挪了行，下一次构建报的是新行号而不是撒谎。
同时把两个 switch 分开写清：`switch(testArray[3])` 管 cFatal（EN DASH U+2013 在 :1906/:1912，
带 codepoint 转储自证），`switch(testArray[5])` 管 healTime（:1940 落 0）。
**两种失效模式性质不同**：healTime 那条是**静默归零**，裸 `Shift` 那条是**抛异常整个检定当场死**。

**主控裁定四条**（业主授权「术语你综合判断最优解作为主轴即可」），依据与反方论点全文在
`7-其他内容/glossary/RULINGS.json`——**决策与证据推导分开存放**，
这样下一轮读得出哪条是证据说的、哪条是人拍的：

| # | 词 | 采纳 | 一句话理由 |
|---|---|---|---|
| R1 | Queen | **女王** | 真正要用的是复合词**异形女王**；且异形女王是孤雌生殖的蜂巢统治者本人，「王后」在中文里是「国王的配偶」，指向一个不存在的国王。反方 王后 全部来自 CnSCG 一条血脉，是**一票在自己内部打架** |
| R2 | Armor Piercing | **破甲** | 现行 `穿甲弹` 是**范畴错误**不是风格差异：这条属性同时挂在刀、气矢枪和**异形酸血**上，酸血不是「弹」。属性名按最宽的挂载面取 |
| R3 | Comtech | **通信科技** | 两个候选各错一半：一级源 `计算机科学` **窄于英文**（砍掉通讯与电子），四级源 `科技` **宽到不指任何东西**。英文本身是 Com+tech 的缩合 |
| R4 | Captain | **船长**（军舰另作**舰长**）| 按域分裂而非二选一。`USCSS` 的 C 就是 Commercial，三大战役框架里「太空卡车司机」排第一。二级/五级源那两处 `舰长` 指的都是具名大船，**采纳域分裂之后两边证据都被解释了，没有一方被推翻** |

**另有一条不在任务书里、自行改正的**：`Turn` 由 ~~轮次~~ 改 **节**。`轮次` 与 `Round` 的 `轮`
只差一个字，且它命名的单位这游戏里根本没有。两条独立佐证：一级源的时间表把中间单位写作
`节 (Stretch) 5-10分`；而 healTime 的 switch 把 case 排成 `OneRound(1) < OneTurn(2) < OneShift(3)`,
把 Turn 放在 Stretch 的位置上。⇒ `ALIENRPG.OneTurn` 现行的 `一斡`（B 层机翻）要改成 `一节`，
**且必须与它那几个 RollTable 单元格同一个 commit**，否则 healTime 静默归零。

**`Facehugger = 抱脸虫` 从「零命中的猜测」升级成有闸的证据**：Alien: Isolation 的键
`*_KILLED_BY_FACEHUGGER` → `被抱脸虫所杀`，3 个键命中。`异形` 同理，现在有 72 个键闸命中，
不再靠维基引用。**但 Chestburster / Ovomorph / Drone / Warrior / Praetorian 仍是零证据**，
按 §7.2.0 在第 10 章定稿。

⚠ **构建器会吃自己的输出**（`build_glossary_v1.py` 读 `provenance.json` / `pending.json` 又写它们）。
已验证幂等，但首跑**就地吃掉了 Phase 0 的 provenance**，是事后才补档的
（`glossary_alien.disputes.phase0.json`）。⇒ **改构建器之前先备份这两个文件。**

⚠ **落到 `lang/cn.json` 的重写清单里有几条是锁步关键**：
`ALIENRPG.OneRound` 一回合→一轮、`ALIENRPG.OneTurn` 一斡→一节 是 healTime switch 的 case，
**必须与对应 RollTable 单元格同一个 commit**。非锁步的还有
`SkillcloseCbt` 肉搏→近战 · `WepTypeMelee` 近战→肉搏 · `Engaged` 近战→接战 ·
`SkillheavyMach` 机械→重型机械 · `Skillobservation` 观察→侦察 · `Skillsurvival` 求生→生存 ·
`Skillcomtech` 科技→通信科技。**这些进 Phase 1 的任务书。**

### 2026-08-29 · Phase 1（系统 UI，业主优先级 #1）

**产出**：`1-系统汉化插件/lang/cn.json` **31,052 B / 600 键**，外加运行时补丁。
en.json 590 键里**覆盖 584**，6 条有意不写（理由写死在 `qa/assemble_lang.py` 的
`DELIBERATELY_UNWRITTEN` 里）；另有 **16 个 en.json 根本没有、但代码在引用**的键，
它们今天在**任何语言下**都渲染成裸键，我们是唯一修它们的地方。

**分片翻译差点整批静默损坏，被对抗式复核救回来。** 四个 agent 各翻一片，交上来时：
- **162 / 590 键没有任何分片认领**，其中包含本阶段**受命要修的四个招牌缺陷**
  （`GMONLY` 仅限通用汽车 · `Overwatch` 守望先锋 · `HOWMANYDICE` 有多少基地死亡？ · `Utility` 公用事业）；
- **57 键被两到三个分片同时认领，其中 29 条中文各不相同**；
- 而项目自带的 `qa/apply_lang.py:206` 是**裸 set_path，没有任何跨批次冲突检测**——
  实测 505 条进、487 条报 "applied"、实际只有 446 条不同：**59 次静默覆盖，零日志**。

⇒ 这是 §3.5「整叶覆盖静默回滚并行批次」那条 EC 教训**在 lang 通道上的同型复发**。
复核补写了第五个分片 `lang-gap.json`（156 键）、把 57 条重叠逐条判给唯一属主、
并从落败方删除。**装配改用新写的 `qa/assemble_lang.py`，它把「同一个键出现在两个分片」
当硬错误而不是当合并。**

**装配器的 11 条不变式当场抓到两个会静默毁掉 UI 的写入**（这两条是本轮最有价值的产出）：

| 键 | 为什么写不得 |
|---|---|
| `ALIENRPG.Effect.Temporary` | 父键 `ALIENRPG.Effect` 在 en.json 里是**字符串** `'Effect'`，且有 **7 个活读者**（4 张重伤聊天卡 + 2 张物品卡）。写子键会让 `mergeObject` 把那个字符串整个替换成对象，**7 处标签同时变裸键** |
| `ALIENRPG.General.NeedToImportActor` | 同型。父键 `ALIENRPG.General` = `'General'`，**7 个活读者** |

⇒ **这两个键是上游 bug，在英文下也永远解析不了**（`helpers/effects.mjs:11` 与三张 sheet 引用了
父键为字符串的点号键）。我们不写，改走运行时补丁，并报上游。
**这正是 EC 那条「顶层点号键 + 嵌套值互相顶掉，77% 的 lang 文件是死的还发出去了」的教训**——
区别是这次**在写盘之前就被闸住了**。

**装配后按 Foundry 真实语义做了合并仿真**（系统先加载、模块后合并、`getProperty` 点号下降）：
写入 600 键**零不可达**，被我们从字符串变成对象而**毁掉的系统键 0 个**，最终运行时 620 键。

**一条不变式被证明是错的，已改**：原第 8 条要求中文首尾空白与英文逐字节相同。
那对拉丁文成立（英文靠空格分词），**对中文不成立**——中文数字两侧不加空格，
`你受到3点生命值伤害` 才是对的，`你受到 3 点生命值伤害` 是错的。
改成：**可以不同，但必须在 `6-工作区/phase1/WHITESPACE-EXCEPTIONS.json` 里逐条申报并写明拼接点**，
且反向检查申报了却已不再有差异的条目（防豁免表腐烂）。当前 12 条，每条都核过调用点。

**运行时补丁**：复核发现 `plugins-hardcoded-cn.mjs` **完全没有语言闸也没有系统闸**，
且在 `init` 注册译好的 partial——而它的兄弟文件自己的推导就证明 `init` 读不到真实语言。
表填满之后会**把中文推进英文世界**。已修（加 `game.system.id === 'alienrpg' && game.i18n.lang === 'cn'` 双闸）。
另外两个补丁文件用了**同一个 `__alienCnPatched` 哨兵**，先装的会让后装的以为已经装过——已拆成两个名字。
新增 `qa/adversarial_hardcoded_patch.mjs`：**125 项检查全绿**，其中爆炸半径那一组是拿
**295 个已装模块 + 全部系统 + 核心共 4,238 个真实文件**测的，确认每条规则的锚点都只有一个属主。

**⚠ 交付顺序的硬约束（本轮定，写进发版清单）**：
`alienrpg-cn` **不能只发 Phase 1 就发**。12 个 T-EXACT 技能名是一条链：
`lang/cn.json` 的 `ALIENRPG.Skill<key>` → 每次 `prepareDerivedData` 写进
`system.skills.<skl>.description` → `character-skills.hbs:15` 发成 `data-pmbut` →
`character-sheet.mjs:1119` 拿去 `game.items.getName()`。
而**那 12 个 `skill-stunts` 物品就在系统自带的 Adventure 包里**（Phase 2，7.6 万字），
且**与 lang 同属 `alienrpg-cn` 这一个模块**。
⇒ 只发 lang 会让 `getName("重型机械")` 返回 null，`:1118` 抛异常，
`:1127` 的 catch 落到 `<h2>No Stunts Entered</h2>`。
**Phase 2 必须与 Phase 1 同一个发布版**。（缓解：character/synthetic 两张卡的这条路径
被 `if (!game.settings.get("alienrpg","evolved"))` 挡着，只在经典模式生效；
但 creature/colony 两张卡没有这层闸，且经典模式本来就要支持。）

### 2026-08-29 · Phase 2（系统自带 Adventure 包）

**产出**：`1-系统汉化插件/compendium/cn/alienrpg.alien-rpg-system.json` **81,802 B**。
9 个单元、11 个 agent（6 期刊段 + 表/物品/外壳 + 跨单元一致性 + 质量对抗）。

**切单元的方式改了，并且这次是结构性保证的。** Phase 1 的四个分片按「命名空间描述」切，
四个 agent 各自判断边界，结果 **162 键无人认领、57 键被多方认领**。
Phase 2 改成**按 `<h1>` 字节偏移切**，`prep_sys_units.py` 直接断言分段铺满整页：
`66,228/66,228 bytes, no gaps, no overlaps`。**单元之间不可能重叠，因为区间不相交。**

**装配后的结构核验（我自己跑的，不采信 agent 自述）**：
标签多重集 **2148 / 2148 完全相同** · 90 个标题**全部带显式 `id=`**（保留英文 slug，
否则译标题会静默切断所有 `data-hash` 深链）· `class` 属性零差异 ·
tables 3/3 · items 26/26 · folders 7/7 · macros 4/4，无缺失。

**T-EXACT 链逐字节验证通过，12/12 零失配**——这正是 Phase 1 不能单独发版的原因：
`lang/cn.json` 的 `ALIENRPG.Skill<key>` 与包内 12 个 `skill-stunts` 物品名必须字节相等。
近战 / 指挥 / **通信科技** / 重型机械 / 操控 / 医疗 / 机动 / 侦察 / 驾驶 / 远程战斗 / 耐力 / 生存。
26 个 `notes` 全部保持上游那个字面量 `[object Object]` 未动。

**册子的精度得到验证**：三个冻结文件夹名（`Alien Tables` / `Alien Creature Tables` /
`Alien Mother Tables`）保持英文，而 **`Alien Sub-Tables` 被正确译成「异形子表」**——
它没有被任何代码引用（`Folder#contents` 非递归）。册子不是「一律冻结」的偷懒表，是逐条查过的。

**复核抓到的三条真缺陷，都在 Phase 1 的产物里，不在 Phase 2**：

| 键 | 原值 | 改为 | 依据 |
|---|---|---|---|
| `ALIENRPG.Pwr` | ~~威力~~ | **电力** | `character-inventory.hbs:39-41` 的 Items 行是 `Pwr\|Food\|Water`——**异形的消耗品三件套**。Pwr 是 Power 的缩写，与 `ALIENRPG.Power=电力` 同指。威力 是 power 的另一个义项 ⇒ **B 层同型错误在我们自己的产出里复发了** |
| `ALIENRPG.Rounds` | ~~备弹~~ | **弹数** | 与 `Rds`（EN `Reloads`）**同为 备弹，两列同名**。`YZEDiceRoller.mjs:422` 拼作 `Rounds + Supply` ＝弹数补给 |
| `ALIENRPG.Rds` | 备弹 | 备弹（不变） | `character-inventory.hbs:34` 经典模式下顶替 `ammo`(弹药) 当列头，指备用弹匣 |

⇒ 教训：**「同一个中文对应两个不同英文」要当缺陷筛，但不能一刀切**——
实测 29 组同中文里只有 1 组是真缺陷，其余 28 组是同一个英文词的大小写/同义变体，本就该同译。

**⚠ 册子里有一列是「陷阱字段」，已改名**：`exact_match_to_lang` 的
`lang_cn_shipped_by_upstream` 记的是**上游 4.1.13 自带 cn.json** 的样子，
**不是** T-EXACT 的目标值。实测 12 条里 **7 条**与我们交付的 lang 不同
（重型机械/近战/远程战斗/操控/侦察/生存/通信科技）——照它改包内物品名会产出 7 个错名。
已改名为 `lang_cn_upstream_4_1_13_reference_only` 并加 `⚠_TARGET_WARNING` 与
`target_is`（指向运行时读 lang 文件）。**闸门本来就读 lang 文件，错的只是文档字段。**

**主控裁定两条**（都属「手册把东西叫成用户看不到的名字」这一类）：
- **R5 宏名**：U1 把 4 个宏名译成中文，J5/J6 却用英文引用它们 ⇒ 读者在中文侧边栏里找不到。
  裁定**保留中文宏名**（`command` 字段本就冻结且已排除出 mapping，`name` 无代码引用），
  并把 J5/J6 的引用改成与 U1 逐字节相同。
- **R6 产品名按域分裂**：`Alien RPG` / `Alien Evolved` 指**Foundry 系统/模块**时保持英文
  （它们是用户在 UI 里看到的产品标识，且 `Alien Evolved` 就是 `system.json` 的 `title`，硬编码）；
  指**作品/书**时用 `《异形》…`。SYS-I1 的 12 处书目引用由 `《异形RPG》规则书` 改为 `《异形》规则书`。

**长度比 band 首次在真实语料上实测，写进 `7-其他内容/RATIO-BAND.json`**：
6 个单元中位 **0.36**，区间 0.32–0.44，无一段异常短（⇒ 没有截断）。
TRUNCATED 阈值定在 **< 0.25**，INFLATED 定在 **> 0.60**。
⚠ 这一次的样本**全是软件手册体裁**；starterset / corerules 是散文与表格，
**Phase 4 落地后必须重量并追加**（不要覆盖本次）。
⚠ 长度比只是候选筛不是判据——EC 实测有一条确凿缺陷落在 0.377，正正好在正常带里。

**闸门状态**：`scan_name_lookup_traps` checked=25 violations=0 ·
`scan_crit_lockstep` checked=8 violations=0（8 个键仍为英文，等 Phase 5 与重伤表一起翻）。

⇒ **`alienrpg-cn` 的 Phase 1 + Phase 2 已齐，具备发版条件**——但**发版前必须先做 §7.1 的冒烟**，
尤其是首次导入的 9 个查找名与经典模式下的技能炫技按钮。

### 2026-08-29 · Phase 4 备料时推翻的一条复用断言（**记下来，因为复核也confirm错了**）

勘察轮的内容清单说：**「Map Pins 的 17 页是 Hope's Last Day 里位置段落的 100% 复制，
纯粘贴，零新翻译」**，而那一轮的对抗式复核**明确 confirm 了这条**
（原话：「Map Pins really is 100% contained in Hope's Last Day」）。

**Phase 4 备料时实测：不是 100%，是 60.8%。**

| | |
|---|---|
| Map Pins 可见文本合计 | 15,118 字 |
| 在 Hope's Last Day 里逐字找得到的 | 9,189 字（**60.8%**）|
| **整页命中的页数** | **0 / 17** |
| 完全无重叠（0%）的页 | **3 页**：Asset/Claims Office · Assistant Manager Office · Geological Managers Office |

⇒ 若按原断言当"纯粘贴"处理，会**静默丢掉约 5,900 字**，并让那 3 页**整页不翻**发出去。
而且发不出任何告警——覆盖率会显示 100%，因为那些页"有译文"（粘来的）。

**方法论教训（比这条数据本身重要）**：
对抗式复核**确认了一条假断言**。最可能的原因是它用了**与原报告同样的度量方式**去验证，
而不是独立重算。⇒ **复核"确认"过的复用类断言，落地前仍要自己量一遍**——
复用断言的失败方向是**静默漏译**，不是报错，属于 §3.2「扫了但没扫到 vs 扫了很干净，输出长得一样」那一类。

**处理**：Map Pins 按**正常翻译单元**派工（17 个单元，ST-J09…ST-J25），
其中 61% 可用 TM 从 Hope's Last Day 的已译段落回填，剩下的照常翻。
**不是**纯粘贴单元。

⚑ 顺带修正另一条：勘察轮说 `EV - Critical Injuries` 只在 corerules 里。
实测 **starterset 也有**，且 `Critical injuries` / `Critical Injuries on Xenomorphs` /
`Critical Injuries on Synthetics` 四张表**两个包都有**。
⇒ 重伤表**不能单包翻**，两个包必须同时动，见下条。

### 2026-08-29 · 重伤表的两条铁律（Phase 4 定，Phase 5 执行）

重伤表的结果正文是**定长字段记录**：
`<strong>INJURY: </strong>Sprained Ankle <br /><strong>FATAL: </strong>No <br />…<strong>HEALING TIME: </strong>Shift`

`actor.mjs` 剥标签后按 `/[:] |<br \/>/gi` 切，再读**固定下标**——偶数位是字段名、奇数位是值。

| # | 铁律 | 违反的后果 |
|---|---|---|
| **一** | 字段名后的分隔符必须是 **ASCII 冒号 + 空格**。字段名本身可以译（`伤势: ` / `致命: ` / `时限: ` / `效果: ` / `恢复时间: `），但那两个字节不能动 | 写成全角「：」⇒ split 形状改变，**所有固定下标错位**。前一轮的闸门实测过这个形状：`checked` 从 131 掉到 128 |
| **二** | `FATAL` / `TIME LIMIT` / `HEALING TIME` 三格的**值**在 lang 那 8 个键翻过来之前保持英文。`INJURY` 名与 `EFFECTS` 正文可以译 | 单方面译值 ⇒ healTime 静默归零；裸 `Shift` 那一支更狠，直接抛未捕获异常 |

⇒ **Phase 4 只译 INJURY 名与 EFFECTS 正文**，值与 lang 键留到 Phase 5，
届时 starterset + corerules + `lang/cn.json` **同一个 commit** 一起翻，由 `scan_crit_lockstep.py` 看守。

### 2026-08-29 · ⚠ 撤回上面那条「Map Pins 复用断言被推翻」——**错的是我，不是勘察轮**

> **本节是对同日上一条 §8「Phase 4 备料时推翻的一条复用断言」的完整撤回。**
> 按 §3.7 只追加纪律，那一条原文保留在上面，不删——因为它记录的是一次**真实的判断失误**，
> 而失误本身比结论更有教学价值。

**事实**：Map Pins 的 17 页**确实是 Hope's Last Day 的 100% 子集**，勘察轮说得对，
那一轮的对抗式复核确认它也对。**我报的 60.8% 是量错的。**

**根因**：我的归一化把 HTML 标签**删掉**而不是**换成空格**。

```python
re.sub(r'<[^>]+>', '', s)     # 我写的 —— 错
re.sub(r'<[^>]+>', ' ', s)    # 正确
```

删标签会把**跨标签边界的相邻词黏成一个词**。同一段文字在两个文档里的标签结构不同，
黏法就不同，于是被判成"不一样"。实证（我当初报「0% 重叠」的三页之一）：

| 归一化 | Asset/Claims Office 的开头 | 在 HLD 里找得到吗 |
|---|---|---|
| 删标签 | `Asset/Claims OfficeIf searched, a successful…` | **False** |
| 换空格 | `Asset/Claims Office If searched, a successful…` | **True** |

`OfficeIf` —— 标题和下一句黏成了一个词。改成换空格之后，两种量法（整页包含 / 30 字滑窗覆盖）
**都给出 100.0%，覆盖率低于 5% 的页 0 个**。

**这条属于 §3.7 已经写过的那一类：「校验脚本自己会静默出错」。**
EC 项目栽的是 Windows glob 返回反斜杠导致 `.replace('/cn/','/en/')` 静默失效、
拿中文文件跟自己比、全绿。**这次是同一形状的另一种：归一化不对称，导致同一段文字判成不同。**
两次都不报错，两次都给出一个**看起来很具体、很可信**的错误数字。

⚠ **我不但把错的数字写进了主文档，还在它上面加了一条"方法论教训"，
说对抗式复核"确认了一条假断言、因为它用了和原报告一样的度量方式"。**
那条教训整个是错的——复核当时用的量法是对的，是我的量法坏了。**教训要反过来读**：

> **当你的测量推翻了一个"原报告 + 独立复核"都同意的结论时，
> 先怀疑你的测量，而不是先怀疑他们两个。**
> 两个独立来源一致、而你一个人不一致，先验上你才是离群的那个。
> 本项目的判据纪律（§3.3 英文闸、§3.2 覆盖行）都要求**证据带自证**，
> 这一条要补进去：**推翻共识的测量，必须先给出"我的量法是对的"的正证**——
> 比如换一种完全不同的算法复算一次，两种量法结论一致才算数。

**代价**：Phase 4 的任务书里把这条假更正当事实写给了 20 个 agent，让它们"把每一页当真活干"。
实际后果**不严重**——它们照常翻译，只是没有走 TM 回填的捷径；
真正的损失是**同一段英文被翻了两遍、措辞不同**（复核实测 69 对同英文双中文、30 个块被翻 2-4 次），
这正是"当成新内容翻"必然产生的副作用。复核已修掉大部分，剩余由装配前的统一处理。

⇒ **`4-常用脚本` 里所有做文本比对的脚本，标签一律换空格，不许删。**
已在 `tm/build_subtitle_corpus.py` 与本轮的比对脚本里统一。

---

### 图内文字（新手包 0.2.0）：从 6 张变成 10 张，以及一次规则性错误

上游把一部分文字直接排进了位图像素，Babele 与任何文本管线都够不着。
处理办法是照原样重画中文版、再把译文里的 `<img src>` 指过去（`4-常用脚本/release/make_charts.py`
与 `make_minimap.py`），**只改译文包的 src，不动上游模块一个字节**。

**清点漏了。** 一开始只认出 6 张流程图。实际上同一日志里还有 `actions`（完整动作/快速动作）、
`resolvecalc`（D6 + 压力等级 − 精神强度）、`zones`（五段距离带）三张同类小图，
以及站图 `mini-map`（27 条英文房间名）。⇒ **清点这类图不能靠"看上去像流程图"，
要把该日志引用的每一张图逐张打开看过**。`guns` / `mini-office-map` / `mini-shuttle-map`
逐张确认过是纯美术、无文字，保持上游引用。

#### 压力反应检定图：我把规则改了

`stressresponseroll.webp` 在 `alien-evolved-starterset/images/journal/` 下**不存在**。
我据此判定"上游图片缺失"，然后**按包内已译的压力反应表自己重建了一张**，
做成「掷 1D6 查表」的六档。

**这是规则错误。** 用户拿出了真图：判定式是 **D6 + 压力等级 − 精神强度**，
结果范围 **≤0 到 7+ 共八档**——我丢掉了 `≤0 镇定` 与 `7+ 失误` 两档，
还把"骰子加减修正后的结果"错写成"1D6 的点数"。

**根因不是翻译，是溯源。** 那张图确实存在，只是上游放在 **`alien-evolved-corerules`** 下，
而新手包的页面 src 指向 `starterset/…`（上游自己的路径 bug，英文原版这张图同样是破的）。
我只在 starterset 一个模块里找了一次，就下了"不存在"的结论，
并且在图里标注"属于对上游的补全"——**把一个检索失败包装成了一条结论**。

> **教训：判定"上游没有某资源"之前，必须在**全部已装模块**里搜一遍文件名，
> 而不是只在引用它的那个模块里搜。**同名资源跨模块存放在本项目里是常态**
> （新手包与核心书共享大量图）。

改正后的八行与包内 `Stress Response Table` 的 `range`（`0-0` … `7-10`）逐行对齐，
描述文字直接复用表里已交付的同一段中文，图与表不会出现两种说法。

#### `src="` 在 `json.dumps` 的结果上永远匹配不到

查"译文里引用了哪些图"时，我在 `json.dumps(pack)` 的输出上找 `src="([^"]+)"`，
得到 **0 条**——序列化之后 HTML 里的引号是 `\"`，这个模式一次都匹配不上。
第一次只是让我误判了接线状态；**第二次它进了发布闸**
（`2-新手包汉化插件/.github/workflows/_assert_images.py`），
使那道闸 `refs=0`、**永远通过**——一道假闸比没有闸更危险。

是脚本里那条**反向检查**（"包里有、但没人引用的图"把 10 张全报成孤儿）把它暴露的。

⇒ **两条纪律**：
> 1. 要在 JSON 里找 HTML 内容，**遍历字符串叶子、在未转义的值上匹配**，
>    不要在序列化结果上做正则。这和 §3.7 那条"删标签 vs 换空格"是同一族：
>    **在错误的表示层上做测量**。
> 2. **每道闸都要有反向测试**：不但要验"正常情况放行"，还要验
>    "把它该拦的东西造出来，它确实拦得住"。`_assert_images.py` 已双向测过
>    （抽掉 `zones.png` → 退出码 1）。同时加了一条自检：`refs` 为空即报错退出。

#### 出货面变了，README 的旧说法作废

0.1.0 的 README 写着"本仓**只含译文**，不含上游的任何美术"。0.2.0 起这句话不成立。
已改写并**把 10 张图分成两类说明**：9 张是从零重画的（HTML+CSS，未使用上游像素，
骰面符号复用**系统** `alienrpg` 自带资源），`mini-map.png` 是**上游美术的衍生物**
（底图是上游 webp，只换了房间名）。后者性质不同，README 里单独声明并给了撤下的途径。

---

### 覆盖率盘点（除核心书外）：一个真缺口，和一次同坑复发

问「除 core 之外是不是都汉完了」，实测下来**基本是，但漏了一整类**。

| 目标 | 实测 |
|---|---|
| 系统 `alienrpg` 界面 | 590 键 / 已译 575。余下 15 个全部在 DO-NOT-TRANSLATE 登记过 |
| `alien-mu-th-ur` | 275 / 269（6 个 MOTHER 关键词必须留 ASCII） |
| `motion_tracker` | 69 / 69 |
| `token-action-hud-alien`、`motion-tracker-multideck` | 上游没有 `lang/`，无界面文本 |
| 系统合集包 | 26 物品 / 4 宏 / 1 日志全译；3 表名 + 3 文件夹名按设计冻结 |
| **`alien-evolved-*` 的 19 个 `ALIENRPG.*` 键** | **完全未译，连对应 lang 文件都没出** |

那 19 个键是**技能炫技清单（12）与天赋描述（7）的正文**，系统自己的 `en.json`
里没有，由两个 Evolved 模块补给系统，两包逐字节相同。已补
`1-系统汉化插件/lang/plugins/evolved-stunts-cn.json`，按模块门控声明两次。

#### 顺带发现：这条查找在中文下**本来就不可能命中**

    character-skills.hbs:15   data-pmbut='{{skill.description}}'
    actor-character.mjs:468   skills[x].description = localize(CONFIG.ALIENRPG.skills[x].name)
    character-sheet.mjs:1113  localize("ALIENRPG." + 去空格(那个名字))

key 是用**已经本地化的技能名**拼的。英文 `Close Combat` → `ALIENRPG.CloseCombat`，
正好命中上游提供的键；中文「近战」→ `ALIENRPG.近战`，上游没有，按钮永远显示
「未录入炫技」。这是系统自身的 i18n 缺陷，**只在英文下成立**。

处理：额外提供 12 个**中文键别名**，内容与英文键逐字节相同。
别名跟着技能名走，改词表会**静默**打断它，所以配了
`qa/scan_stunt_aliases.py`（正向 + 反向都测过：把「近战」改成「格斗」立刻报两条）。

⇒ **凡是「拿本地化后的字符串再去查表」的代码路径，翻译一定会打断它。**
   §3.4 的命名分层管的是「名字本身要不要翻」，这条补的是另一半：
   **名字被翻之后，谁还在用它当键**。发现一条就登记一条。

#### 同一个坑，第三次

量「译文比英文多出多少 `id=` 属性」时写了：

    for cnp in glob.glob("[12]-*/compendium/cn/*.json"):
        enp = cnp.replace("/cn/", "/en/")

**Windows 的 glob 返回反斜杠**，`replace` 一次都没生效，等于拿中文文件跟自己比，
干脆利落地报了 **0**。而直接点名比对那两个字段，明明是 EN 0 / CN 1。

这就是 §3.7 记着的 EC 项目那次，一模一样。**记在文档里没能阻止它复发。**
所以这次把解药写进代码而不是写进文档 —— `qa/scan_html_fidelity.py` 的 `pair()`：

    en_path = cn_path.replace(os.sep + "cn" + os.sep, os.sep + "en" + os.sep)
    assert en_path != cn_path, "路径配对失效：…别用写死的 '/cn/'"

> **配对路径的函数必须断言「配出来的确实不一样」。**
> 静默失效的比对不会报错，只会给你一个漂亮的 0；而 0 正是你想看到的数字，
> 所以不会去怀疑它。这一族（§3.7 删标签 vs 换空格、`json.dumps` 上找 `src="`、
> 反斜杠路径）的共同点都是**在错误的表示层上测量**，且全都**朝"没问题"的方向错**。

实测结果：系统包 14 个字段被译文管线凭空加了 `id=`（英文基准是光秃秃的 `<h2>`），
新手包 0 个。已清掉。这不只是脏 —— `character-sheet.mjs:1125` 拿
`startsWith("<h2>No Stunts Entered</h2>")` 做字符串比较，多一个属性判断就废。

#### 同一段英文，两种中文（新手包 6 组）

同一件物品在 Adventure 的 `items/` 与 actor 内嵌各存一份，并行翻译时落进不同切片、
被当成两段新内容各翻一遍。§8 记过成因（Phase 4 复核实测 69 对），但**没有闸门**，
于是 v0.2.0 带着 6 组发了出去：玩家在物品栏看到「其他所有 PC」，点开角色卡看到
「其余所有玩家角色」。已按全项目用词计数定稿，并补 `qa/scan_dup_renderings.py`。

⇒ **成因写进文档 ≠ 问题被治住。写过成因的缺陷，都要配一道能跑的闸。**

#### 上游缺陷登记：`system.notes` 存的就是 `"[object Object]"`

系统包 26 件物品 + 新手包 20 处的 `system.notes` 字段，值是字符串
`"[object Object]"`。用 `classic-level` 直接读上游 LevelDB 确认过：**上游存的就是这个**
（`base-item.mjs:8` 还留着旧 `SchemaField` 的注释，第 11 行已改成 `HTMLField()`，
显然是某次迁移时把对象赋进了字符串字段）。

我们的抽取器如实抄下、译文如实保留、Babele 写回去与原值相同 —— **无害**。
一度怀疑是我们管线弄坏的，不是。留此存照，免得将来有人"修"它、凭空编出内容。

---

### `yze-combat`：系统官方支持的先攻方案，但它是**替换**不是叠加

`alienrpg` 系统内建了对接钩子（`alienrpg.mjs:500`，顶层作用域，不在任何 hook 里）：

    Hooks.once("yzeCombatReady", (yzec) => yzec.register({
      actorSpeedAttribute: "system.attributes.speed.value",
      duplicateCombatantOnCombatStart: true,
    }));

装上 yze-combat 就自动写好这两项 —— 前者让「速度」属性生效，后者让**速度 2 的生物
开局自动复制成两个参战者**，正是异形生物每轮行动两次的规则。

**谁覆盖谁（实测，非推断）**：两边都**无条件**设置同一批 CONFIG——

| | 系统 `alienrpg.mjs:87,114` | 模块 `yze-combat.js` init |
|---|---|---|
| `CONFIG.Combat.documentClass` | `AlienRPGCombat` | `YearZeroCombat` |
| `CONFIG.ui.combat` | `AlienRPGCTContext` | `YearZeroCombatTracker` |
| `CONFIG.Combatant.documentClass` | —— | `YearZeroCombatant` |

系统那边没有「yze 在就让路」的判断。顺序由内核决定：
`dist/server/views/view.mjs` 里先发 `system.esmodules`、再发各 `module.esmodules`，
ES 模块按文档顺序执行 ⇒ **模块的 `init` 后跑，yze 全赢**。

**因此丢掉的**：系统原生 `rollInitiative` 会把抽到的牌以图片发进聊天栏
（`systems/alienrpg/images/cards/card-{1..10}.png`；装了核心书则换成
`modules/alien-evolved-corerules/images/cards/`，两处各 10 张，都在）。
yze 的默认牌堆是**扑克牌**（`cards/light-soft/spades-ace.webp`），异形牌面不再出现。
可补救：yze 的先攻牌堆是真正的 Cards 文档，自建一副用系统那 10 张图并在设置里指过去。

系统的速度克隆逻辑（`AlienRPGCombat.createEmbeddedDocuments`，`combat.mjs:177`）
同样被顶掉，但 yze 的 `duplicateCombatantOnCombatStart` 顶上了，功能不丢、换了实现。

**规则契合度**（全部实测默认值）：牌堆 10 张值 1–10 ✓ ｜ 排序默认升序（低牌先手）✓
｜ `SlowAndFastActions` 默认开 ✓ ｜ `ShowAmbushed` 默认开（遭伏击者先攻加牌堆张数
⇒ 本轮最后行动）✓。

**术语错位**：YZE 通称 Slow/Fast Action，《异形》**进化版改称 Full/Quick**
（本项目交付值：完整动作 / 快速动作）。`yzecombat-cn.json` 按进化版译；
跑经典版把 `YZEC.CombatTracker.SlowAction` 改回「慢速动作」即可。

⚠ 上游 yze-combat 在 v14 + alienrpg 下有 statusEffect 重复注册崩溃，
用 takaqiao fork（`module.json` 的 description 里写明了）。
