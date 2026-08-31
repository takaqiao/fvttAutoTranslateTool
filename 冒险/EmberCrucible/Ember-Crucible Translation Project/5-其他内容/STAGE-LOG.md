# 阶段日志（2026-08-06 ~ 2026-08-09 · 阶段 0–27）

> 这是 `PROJECT.md` 第 6 节的**完整归档**，只追加、不重写。
> 主文档里保留的是压缩年表；需要查某一阶段的原始记录（当时的测量数字、
> 走过的弯路、被推翻的判断）时来这里。
>
> **这些记录反映的是当时的状态，不是现在的状态。** 现状一律以 `PROJECT.md`
> 第 1 节为准；仍然生效的硬约束在第 3 节，仍然生效的裁决在第 8 节。

---

### 2026-08-06 · 阶段 0：基建与全面测量

**范围**：项目重组、版本差异精确测量、babele 2.9.1 兼容性排查、外部参考调研。

**做了什么**

- 建立本项目目录结构，把两个汉化仓库从 GitHub 正式 clone 到 `1-` / `2-`
- 用 `classic-level` 直读 LevelDB packs，抽出 crucible 0.10.1 与 ember 0.6.0 的英文原版
- 三方 diff（新版英文 / 旧版英文 / 现有译文），得出第 2 节的全部数字
- 逐字段测量 ember 战役包的真实缺口（16065 条可译串，已完成 39%）
- 通读 babele 2.9.1 源码，查清 API 变更与三个真实故障（第 3.1 节）
- 调研 Padhiver/Crucible-FR，确认可抄与不可抄的部分（第 4 节）
- 盘点本地 6 份 glossary，确定以 `glossary_crucible_merged.json` 为基底

**关键发现**

1. **babele 2.9.1 的源包回退**：ember 战役里 82.4% 的 actor 内嵌物品字符带 `compendiumSource`。
   改用默认递归 `document` mapping 后，约 60.9 万字符（`crucible.talent` 41.8 万 + `crucible.adversary-talents` 10.8 万 + `ember.crucible-*` 6.1 万 + 其余）**自动翻译，零成本**。
   → ember 战役待译量从初估 330 万字符降到约 159 万。
2. **早期基于本地陈旧副本得出的三条结论是错的**，已作废：
   - ~~ember 汉化 0% 生效（顶层 key 未改名）~~ → v1.0.15 已是 `Ember Early Access`，是本地装的 1.0.0 太旧
   - ~~ember 96 万字符正文不可达~~ → v1.0.15 的 `emberPages` 已覆盖 overview/exposition/summary/content.*
   - ~~ember lang 没打包~~ → v1.0.15 已有 `languages` 字段并发货
   **教训：任何测量都必须先确认基线是最新发布版，不能拿本地已安装副本当基线。**
3. 查出三个真实故障（第 3.1 节）＋ 两个数据问题（损坏的 `-en.json`、垃圾 `path:{}` 字段）
4. `ember.adventure`（dnd5e 版）与 `ember.crucible-adventure` 的日志正文**逐字节相同** —— 翻一份等于翻两份，值级 TM 命中 79.7%
5. crucible 0.10.1 的 `system.json` 声明了 `crafting` 包但**没有实际发货**（目录不存在），忽略即可

**产出**

- `4-临时脚本/2026-08-06/` 下的全部探针脚本（可复现上述所有数字）
- 本文件

**遗留**

- 管线改造未开始
- 翻译正文未开始

---

### 2026-08-06 · 阶段 1：通用抽取器与英文基准

**范围**：把抽取器泛化并用独立实现交叉验证；产出两个包的权威英文基准。

**做了什么**

- 写了 `3-常用脚本/extract/mappings.mjs`（mapping 数据的唯一真源）与 `extract_en.mjs`（**解释型**抽取器）。
  抽取器不再硬编码字段，而是直接解释 mapping 数据 —— 运行时交给 `registerMapping()` 的是同一份数据，
  因此「转换器读 A 键、抽取器写 B 键」这类漂移在结构上不可能发生。
- 抽出 crucible 0.10.1（15 包）与 ember 0.6.0（10 包）的英文基准
- 用 Padhiver/Crucible-FR 的 `compendium/en`（独立实现，同样基于 0.10.1）做交叉校验

**关键发现**

1. **Crucible 的 `system.description` 是多态的** —— 大多数 item 类型（talent / ancestry / archetype /
   background / taxonomy / spell）存**纯字符串**，equipment 类存 `{public, private}` 对象。
   第一版把它当成 `system.description.public` 抽，结果 `crucible.talent` 一个包就漏掉约 8 万字符，
   `crucible.ancestry` 只抽到 53 字符（实际 2500）。已改为多态转换器 `crucibleDescription`。
   **这个 bug 只有靠交叉校验才发现得了 —— 单看自己的输出完全正常。**
2. **交叉校验最终结果**：修正后 5089 串 / 767,890 字符 vs 他们的 5150 / 769,456。
   按条目 `name` 配对后**零字段差异、零字符缺口**；总数那 61 串的差是重复条目的计数方式不同，非缺失。
3. **`crucible.pregens` 有 16 个 actor 但只有 8 个不同名字**（每个预设角色两份，内容确实不同）。
   旧抽取器按名字去重，静默丢掉了 8 个。babele 匹配顺序是 `_id` → `name`，所以重复项需要 `_id` 键。
   抽取器现在的策略是：**内容完全相同才按名字合并，确实不同才退回 `_id` 键** —— 既不丢内容，
   也不无谓地把翻译面积翻倍。
4. **`crucible-fr` 是混合式而非纯手写**（见第 4 节更正）。

**产出**

- `3-常用脚本/extract/mappings.mjs`、`extract_en.mjs`
- `5-其他内容/english-baseline/crucible-0.10.1/`（15 包 + `_source.json`）
- `5-其他内容/english-baseline/ember-0.6.0/`（10 包 + `_source.json`）
- `5-其他内容/english-baseline/crucible-0.9.1-legacy/`（旧仓库自带基准，归档用于算 drift）
- `5-其他内容/english-baseline/ember-cn-v1.0.15-shipped-en/`（从 `compendium/cn` 移出的三份错位英文文件，含那个损坏 JSON）
- 两个仓库的 `compendium/en/` 已更新为权威基准
- `4-临时脚本/2026-08-06/crosscheck_vs_crucible_fr.py`

**遗留**

- 管线改造未开始（`registerMapping` 尚未接进两个 register 文件）
- 翻译正文未开始

---

### 2026-08-06 · 阶段 2：术语表构建（顺带查出一批既有译文缺陷）

**范围**：建立 `glossary_ec.json`；构建过程同时充当一次全量术语 QA。

**做了什么**

- 写了 `3-常用脚本/tm/build_glossary.py`（每次升级后重跑）
- 三层合成：基底 `glossary_crucible_merged.json`（4602）＋ 从已发布译文按结构配对挖出的 EN→CN 对
  ＋ 尚无中文的英文名词清单

**结果**

| 项 | 数量 |
|---|---|
| `glossary_ec.json` 术语总数 | **5522**（基底 4602 + 已发布译文新增 920） |
| 基底 vs 已发布：仅双语格式差异（自动采用已发布） | 2817 |
| 基底 vs 已发布：**真实分歧，待裁决** | **20** |
| 已发布译文内部：双语后缀时加时不加 | 209 |
| 已发布译文内部：**同名异译，待裁决** | **59** |
| 当前基准中出现、尚无中文的名词 | 2419 |

**关键发现 —— 已发布译文里的实际缺陷**

1. **双语格式脚本有 bug，把中文复制了一遍而不是接英文**：
   `Anvil` → `铁砧 铁砧`、`Harbinger of Disease` → `疫病先驱 疫病先驱`、`Huiana` → `惠安娜 惠安娜`
2. **机翻残留**：`Harbinger of Madness` → `疯狂先驱者 先驱者 of 疯狂`
3. **中文部分被截断**：`Welcome To Crucible` → `欢迎来到 Welcome To Crucible`（「欢迎来到」后面没了）；
   `What is Crucible` → `什么是《 What is Crucible`（书名号悬空）
4. **人名被当普通名词译掉**：`Corvana Vortest` → `法师`
5. **专名未译**：`Hulg'run Lineage` → `Hulg'run血统 Hulg'run Lineage`
6. **同一专名多种写法**：`Cevher` 家族有 杰夫赫尔 / 杰夫赫 / 切夫赫尔 三种；
   `Bite` 啃咬(24 处) vs 噬咬(1 处)；`Inflection:` 屈折 vs 词缀；`Claw Hammer` 爪锤 vs 羊角锤

**策略**：格式类差异（2817 + 209）可脚本批量归一；真实分歧（20 + 59）逐条裁决后写回术语表，
并记入第 8 节。**不静默择一** —— 脚本对这类只保留现状值并列进 `glossary_ec.disputes.json`。

**产出**

- `3-常用脚本/tm/build_glossary.py`
- `5-其他内容/glossary/glossary_ec.json`（扁平 `{en: cn}`，供 QA/TM 工具直接消费）
- `5-其他内容/glossary/glossary_ec.provenance.json`（每条来源与候选）
- `5-其他内容/glossary/glossary_ec.disputes.json`（20 + 59 待裁决 + 209 格式不一致）
- `5-其他内容/glossary/glossary_ec.pending.json`（2419 条待补中文）

**遗留**

- 79 条待裁决术语未裁决（需结合上下文，放在翻译批次里一并处理）
- 上述已发布译文缺陷未修复（属阶段 3 管线改造后的 QA 批次）

---

### 2026-08-06 · 阶段 3：babele 2.9.1 管线改造 + 首批实译

**范围**：两个模块的运行时改造、译文 schema 迁移、验收工具、crucible 首批翻译。

**做了什么**

*管线*
- 两个模块的注册都迁到 `babele.init`，加 `game.modules.get('babele')?.active` 守卫
- crucible-cn：删掉 `SUPPORTED_PACKS` / `DEFAULT_MAPPINGS.ActiveEffect` 死补丁；
  修掉 `game.babele.converters.actions_converter(...)` 那个 TypeError（改由 babele 原生 `document` 转换器承担）
- ember_cn：删掉 `emberPages` / `emberAdventureJournals` / `emberActions` 手写遍历转换器、
  失效的 `_tableResults` 补丁、以及 `patchBabeleImportAdventureHook`
  （2.9.1 自身已用 `token.delta?.name` 可选链，该补丁无事可做）。
  **其余防御性代码（数据形状迁移、清洗、causticPhial 补丁、syncOwnedItems API）原样保留。**
- 新增 `3-常用脚本/release/generate_runtime.mjs`：从 `mappings.mjs` + `runtime-converters.js`
  **生成**两个仓库的 `babele-mappings.js`。全项目只有一份 mapping 定义，抽取端与运行时不可能漂移。
- 顺手修掉一个遗留 bug：crucible-cn 的 `i18nInit` 里 `game.i18n.translations.sort = "tri"` ——
  "tri" 是法语，从 crucible-fr 抄来忘了改，现为「排序」

*译文 schema 迁移*（`3-常用脚本/qa/migrate_cn_schema.mjs`）
- 删除各 pack 文件里的 `mapping` 块（compendium-local 优先级高于 `registerMapping()`，留着会盖掉新映射）
- crucible：`categories` 从 `{<_id>: {name}}` 转成 `nameCollection` 要的 `{"EN名": "译名"}`（10 处）
- ember：页面键 `soverview`→`overview` 等重命名 1410 处；清掉 1447 个垃圾 `path: {}` 字段；
  250 个 actor 的 `prototypeToken`/`biography*` 归一；1004 处 item description 归一；
  622 处位置型 `actionname[]`/`actiondesc[]` 数组按英文基准的动作 id 顺序重建为 id 键映射
- 12 处找不到英文对照的动作译文**不丢弃**，改存到 `_legacyActions` 键下待人工救回（babele 会忽略未知键）

*验收与应用工具*
- `3-常用脚本/qa/validate_translations.py` —— 拿英文基准逐路径核对译文。
  因为基准就是「解释运行时 mapping」产出的，所以「CN 这条路径在 EN 里存不存在」＝「babele 找不找得到这条译文」。
  同时输出机读待译清单到 `5-其他内容/reports/*/todo/`，翻译批次直接由它驱动。
- `3-常用脚本/qa/apply_translations.py` —— 批量回写译文，三道闸：英文源不匹配则拒、
  无中文则拒、**行内标记不匹配则拒**（`@UUID[...]` / `@Check[...]` / HTML 标签；
  `@UUID[目标]{标签}` 只比对目标，标签本就该翻译）

**关键发现**

1. **`crucible.pregens` 的 8 对重复角色只差在 `items`，且 id 键那份一件物品都没有**。
   抽取器原本给它们各分配 `_id` 键，等于同样的姓名/血统/背景要翻两遍，而且 `_id` 匹配优先于 `name`，
   以后改名字键的译文会静默失效。改为**深合并到名字键、标量真冲突才退回 `_id`** 之后：
   `pregens` 覆盖率 59%→83%、`playtest` 72%→91%，crucible 总待译从 1356 条降到 834 条。
2. **第 2 节原有的待译数字确实偏低**。用完整基准重算后 crucible 是 **4717 叶 / 待译 834 条 / 11.1 万字符**
   （原记 493 条 / 7.8 万）。ember 是 **35157 叶 / 覆盖 38%**（该数字含会被源包回退自动翻译的内嵌物品，偏保守）。
3. 26 处 crucible 孤儿译文（CN 有值但英文基准无此路径 → babele 永远不会应用），待清理。

*首批实译*（crucible，40 条）
- `crucible.taxonomy` 8 条、`crucible.archetype` 16 条、`crucible.summons` 15 条、`crucible.macros` 1 条 → **四个包收尾**
- 术语裁决见第 8 节

**结果**

| 包 | 改造前 | 阶段 3 末 |
|---|---|---|
| crucible 总覆盖 | 74%（含漏字段） | **91%**（4717 叶 / 待译 411 条 / 7.9 万字符） |
| 已 100% 的包 | 3 | **11 / 15** |

本阶段实译并收尾的包：`taxonomy` `archetype` `summons` `macros`
`adversary-equipment`(整包新增 112 条) `affixes`(176 条) `adversary-talents`(65 条)。

**又发现的既有缺陷**

1. **`lang/cn.json` 里 `Burrow` 译作 `掘穴 շարժ作`** —— 混进了**亚美尼亚字符** `շարժ`。
   显然是某次机器处理的污染，必须修。
2. **`Toughness` 与 `Fortitude` 在 lang 里都译作「坚韧」** —— 一个是属性、一个是防御，撞名。
   当前处理：正文里 Fortitude 一律写作「坚韧**防御**」以消歧；根治需要改 lang 并做一次全库扫。
3. **`Electricity` 三处不一致**：lang 作「电力」，而词缀名作「电击伤害 / 电抗性 / 电能转换」。
   已裁定**以 lang 的 UI 标签为准**（玩家在角色卡上看到的就是它），正文统一用「电力」；
   词缀名的清扫列入待办。
4. **`Rune: Lightning` 系列被上游改名为 `Rune: Storm`** —— 30 条译文已用 `port_orphans.py` 移植过去，未丢失。
5. 一个旧翻译流程的 bug：`playtest` 里有条目键本身被翻译成了中文
   （`items.吞噬思维 Devour Thoughts`），成为永不生效的死译文。
6. 既有译文对 `Tier` 有 **阶 / 阶级 / 阶位** 三种写法，对 essence 有 精髓 / 精华 两种。
   本阶段新译一律用**阶 / 精华 / 构筑法术**；旧的清扫列入待办。

**工具增补**

- `3-常用脚本/qa/port_orphans.py` —— 按改名规则把孤儿译文移植到新路径，
  移植不了的**留在原地并列出**，绝不静默丢弃
- `apply_translations.py` 修了一个真 bug：创建缺失容器时未参照英文结构，
  导致 EN 是数组的 `effects` 在 CN 里被建成 `{"0": {...}}` 字典，babele 永远读不到。
  现在按英文结构镜像创建 list/dict。

**遗留**

- ember 尚未开始实译
- 26 处孤儿译文（上游真删除的物品）留在原地，无害但可清
- 上面第 2、3、6 条既有缺陷未清扫
- **改造后未在真实 Foundry 世界里冒烟验证过**

---

### 2026-08-06 · 阶段 4：crucible compendium 收官

**范围**：把 crucible 的 15 个合集包全部译完。

**做了什么**

分批实译并逐批走 `validate → 翻译 → apply → validate` 闭环：
`adversary-equipment`(112，整包新增) · `affixes`(176) · `adversary-talents`(65) ·
`equipment`(85) · `talent`(126) · `playtest`(15) · `pregens`(5) · `rules`(19)。

**结果：crucible compendium 完成**

| | 数值 |
|---|---|
| 叶级覆盖 | 4556 / 4717 = **97%** |
| 内联 100% 的包 | **13 / 15** |
| 剩余 161 条的真实残余 | **0** |

那 161 条全部位于 `playtest` / `pregens` 的 actor 内嵌物品上，条目名（Backstab、Dagger、
Longbow…）都能在已 100% 的 `crucible.talent` / `equipment` 里找到，
由 babele 的通用回退自动取译文。**不需要内联翻译，重复翻反而会造成同名异译。**

**关键发现：通用回退比源包回退覆盖更广**

`pregens` 的 271 件内嵌物品**全都没有 `_stats.compendiumSource`**，所以源包回退不会触发。
但 babele 还有第二级：`DocumentConverter._genericTranslationSource` 调用
`runtime.translatedPackFor(documentType, data)`，它会扫描**每一个已注册的已翻译包**，
找出 `hasTranslation(data)` 为真的那个——匹配候选是 `_id` → `name` → `sourceId`。
于是一件仅仅**名叫**「Backstab」的内嵌物品，即使毫无来源信息，也会命中
`crucible.talent` 的「Backstab」条目。

→ 新增 `3-常用脚本/qa/resolve_generic_fallback.py`，从待译数字里扣掉这部分。
**没有它，会白白重复翻译 161 条（crucible）乃至上千条（ember）。**

**又修的两个工具缺陷**

1. `validate_translations.py` 只写不删待译清单，包做到 100% 后仍残留上一轮的文件，
   下一批会照着陈旧清单干活。已改为每次运行先清空。
2. `apply_translations.py` 的标记校验漏了 Foundry 的内联指令
   `[[/hazard 25 reflex health]]{标签}` / `[[/skillCheck wilderness 14]]`。
   补上后立刻抓到我自己译文里的一处真错误（给英文原文没有加粗的「创伤/疯狂」加了 `<strong>`）。

**顺带清掉的一类隐蔽缺陷：外来文字污染**

新增 `3-常用脚本/qa/scan_foreign_script.py`，扫描西里尔/亚美尼亚/希伯来/阿拉伯/天城/泰/格鲁吉亚字符。
这类污染在 diff 里几乎看不见，也躲得过所有结构校验，但玩家一眼就能看到。全库共 24 处：

| 类型 | 数量 | 例子 |
|---|---|---|
| 杂散字符（已自动清除） | 20 | `卡罗ว์ Carrow`（泰文，出现 15+ 次）、`奈姆ե Na'me`（亚美尼亚）、ember lang 的 `发现ים`（希伯来）、crucible lang 的 `掘穴 շարժ作`（亚美尼亚） |
| 整词被外语替换（已人工补译） | 4 | `发现一处 недавно被某头…`→最近、`检定来 удерживать 它们`→拉住、`石棺 леж在`→横陈在 |

清理后全库外来文字残留为 **0**。此扫描应纳入每次发版前的例行检查。

**遗留**

- crucible `lang/cn.json` 仍缺 293 新 key + 11 drift + 15 未译（明细见 `reports/crucible/lang_gap.json`）
- **冒烟验证仍未做** —— 尤其要验证上面那条「通用回退」在真实世界里确实生效

---

### 2026-08-06 · 阶段 5：crucible lang 收官 + 一批硬错误

**范围**：把 `lang/cn.json` 补齐到 0 缺口；顺带清掉 lang 与 compendium 里几处会真正出错的东西。

**做了什么**

*工具*
- 新增 `3-常用脚本/qa/lang_gap.py`（lang 的三方 diff，此前只有一次性探针，结果无法复现）
  与 `3-常用脚本/qa/apply_lang.py`（lang 版的回写闸门）。
  `2-Crucible汉化插件/lang/lang_keep_english.json` 记录**有意保留英文**的 key（DC / ∞ / ??? 等），
  避免它们每轮都被报成「未译」。

*lang 翻译*
- NEW 293 条、DRIFT 11 条、UNTRANSLATED 15 条（其中 2 条判定为有意保留）全部处理完
- STALE 32 条清除。其中 22 条是上游把 `ACTION.TABS.Description` 之类改成小写键
  （`ACTION.TABS.description`），旧译文可直接复用
- `<repo>/lang/en.json` 基准同步到 0.10.1。复验：1842 键，四类缺口全为 0

*本轮新译里值得记的判断*
- `DEFENSES.Madness`/`Wounds` 上游改名为 `Rallying Threshold`/`Healing Threshold`
  → 集结阈值 / 治疗阈值（`RESOURCES.Madness`/`Wounds` 仍是疯狂 / 创伤，两者是不同的东西）
- 新增的 `HAZARD.*`（38 条）按 compendium 既有译法用「危害」；`Danger Level` = 危险等级
- `ITEM.COMPOSED_NAME` 前后缀改成中文语序：前缀 `{prefixes}{name}`、后缀 `{name}·{suffixes}`

*lang 既有缺陷清扫（58 条）*
- 垃圾标记：`授予里程碑"}`、`Crucible 血统卡##`、`受训#endregion`
- 日文汉字：`支配状態` → 支配
- 半汉化：三条资源提示里的 `Toughness/Strength/Presence/Wisdom` 仍是英文
- 硬伤误译：`Major`→少校（军衔）、`Block {block}`→区、`Affix`→镶嵌、`Glance`→一瞥、
  `Brute`→蛮汉、`Consume`→吞吃、`Engaging`→啮合、`Composed`（作曲）、`Repellent`→驱避剂、
  `Attractive`→迷人的、`Crucible Skill`→坩埚技能
- `SPELL.INFLECTIONS.NegateAdj` 的值是**一整段效果描述**（英文只是 "Negated"）
- 既定译名回归：`Restrained` 受拘束→受缚、`Signature` 签名→招牌、critical hit 重击→暴击、
  `Fortitude` 坚韧→**坚韧防御**（根治与 `Toughness` 的撞名）

*compendium 硬错误（49 条 / 89 处替换，脚本见 `4-临时脚本/2026-08-06/fix_storm_inflection_batches.py`）*
1. **rules「符文」页的 `@Embed[...runeLightning000...]` 是坏链** —— 上游已把该物品改名为
   `runeStorm0000000`，译文里的 id 从未跟着改，玩家看到的是一个加载不出来的嵌入块
2. **Surgeweaver 的 Shocked 持续时间译错**：英文 3 Rounds，译文写成 1 轮（4 处，规则数值错误）
3. **`six-foot radius` 译成「六码 / 六码尺」**（3 处，单位错误）
4. `Storm Proficiency` 的译文把 `@Action[...]` 链接替换成了 `<strong>充能</strong>`，链接丢失
5. `Rune: Lightning`→`Rune: Storm` 改名的遗留：8 个条目名仍写着「符文：闪电 Rune: Lightning」，
   20 处正文仍称「闪电符文」
6. `Inflection` 在 talent 包被译成「词缀」，与 `Affix`（词缀）完全撞名 —— 11 个条目名，
   统一为「屈折：X」，X 采用 affixes 包 `adjective` 的用词（编构/限定/遁避/延展/否定/拉拽/推挤/迅捷/反应/重塑）
7. `Gesture: Sense` 译文末尾多一段英文早已删除的「试玩测试说明」
8. rules「施法总览」页机翻残留「加速 箭头 of 火焰」

**关键发现**

1. **`port_orphans.py` 只搬路径、不改内容。** 改名类 drift 处理完，译文里的旧名字会原样留下 ——
   `Rune: Storm` 这一处就留了 28 个「闪电」和 8 个 `Rune: Lightning` 后缀。已写进第 5.4 节例行检查。
2. **markup 闸门能倒查出既有译文的缺陷。** 这轮 `--force` 回写时被拦下 4 条，全是**本来就坏的**：
   多包一层 `<strong>`、丢掉 `@Action` 链接、多一段英文没有的正文。
   → 用 apply 工具重写一遍旧译文，本身就是一次结构体检。
3. **`lang_gap.py` 的 UNTRANSLATED 判据需要白名单**，否则 `DC`、`∞`、`???` 这类
   本就不该翻的会永远挂在报告里，掩盖真正的漏翻。

**复验**

| 项 | 结果 |
|---|---|
| `lang_gap.py` | NEW 0 / DRIFT 0 / UNTRANSLATED 0 / STALE 0（1842 键） |
| `validate_translations.py` | 4556 / 4717 = 97%，与阶段 4 持平（未回退） |
| `resolve_generic_fallback.py` | 真实残余 **0** |
| `scan_foreign_script.py` | 0 处 |
| 新译文残留英文 / 长度异常 | 0 / 0 |

*顺带修掉的一个发版阻塞*

`module.json` 的 `download` 写的是 `.../0.9.0/crucible-cn-0.9.0.zip`，而 `release.yml`
既打包 `module.zip`，又有一步 **校验 download 必须匹配 `.../<tag>/module.zip`** ——
照原样打 `0.9.0` 的 tag，工作流会在校验步直接失败，release 根本不会产出。
已把 download 改为 `.../0.9.0/module.zip`。同时按第 5.5 节的既定策略给打包加了两条排除：
`compendium/en/*`（1.1 MB 英文基准，运行时永远不会被 fetch）与 `lang/lang_keep_english.json`。

**遗留**

- **冒烟验证仍未做** —— crucible-cn 发版前唯一的硬门槛
- 全库术语统一未做：`Tier`/`Presence`/`essence`/`Wisdom` 约 142 条（已有决议，机械清扫）；
  `Electricity` 33 条（三种写法各占三分之一，需先裁决）。
  数据见 `5-其他内容/reports/crucible/terminology_sweep_pending.md`
- 26 处孤儿译文仍在原地（无害）

---

### 2026-08-06 · 阶段 6：全库术语统一 + 一场没预料到的体检

**范围**：把第 8 节已裁决的术语在全库落实；顺带发现并修掉一批「覆盖率看不见」的缺陷。

**新增工具**

| 脚本 | 干什么 |
|---|---|
| `qa/unify_terms.py` | 按规则表做术语统一。**只在英文原文确实出现该术语时才替换**，并支持正则搭配（`电力(?=伤害\|抗性…)`），避免把 `原始电能的混沌之力`、`针对其心灵的攻击`（译自 mind）这类误伤 |
| `qa/scan_markup_drift.py` | 扫译文与英文之间的**标记**差异，分 LINK / BLOCK / INLINE / PLACEHOLDER / TRUNCATED 五类 |
| `qa/restore_enrichers.py` | 把被写成裸中文的 enricher 还原回去，四级策略：lang 译名 → 搭配变体 → 人工对照表 → 段落对齐 |

规则与对照表进 git：`5-其他内容/glossary/unify_rules.2026-08-06.json`、
`5-其他内容/glossary/enricher_surface_forms.json`。

**术语统一（183 条译文）**

伤害类型是玩家在角色卡上天天看的词，9 个里 4 个不一致，本轮全部定名：

| 术语 | 定为 | 理由 |
|---|---|---|
| `Electricity` | **电击** | 电力像市电、电能像物理量。连带把状态 `Shocked` 改为**感电**，否则伤害类型与状态撞名 |
| `Radiant` | **光耀** | 正文多数已用；辉光/光辉两个词互相太像 |
| `Poison` | **毒素** | 毒药指物（`Poison Vial 毒药瓶` 保持不动）；状态 `Poisoned` 仍是中毒 |
| `Psychic` | **灵能** | lang 与正文多数已一致，清掉 心灵/精神 |

外加已有决议的机械清扫：`Tier`→阶（74 处，另把「1个阶」「每个阶」顺成「1 阶」「每阶」）、
`Presence`→存在（35 处）、`essence`→精华（15 处）、`Wisdom`→感知（2 处）。
lang 侧独立处理：`DAMAGE.*` 三个标签、`Tier` 名词形态统一为**阶数**、
以及 `ABILITIES.*Abbr` 六个缩写（`Pre`→「预备」、`Tou`→「图」这种机翻）改为 敏/智/存/力/韧/感。

**关键发现：覆盖率 100% 不等于内容完整**

`validate_translations.py` 是**按路径**算覆盖率的 —— 一条路径只要有中文值就算已译，
哪怕译文把英文十段里的六段直接丢了。新增的 TRUNCATED 检测按纯文本长度比抓这类漏译
（本库译文/英文长度比中位数 0.31，低于 0.22 判为整段漏译），一抓抓出 12 条、约 2700 中文字：

- `Combat/Movement`：整节「生物碰撞」「强制移动」没译，`Burrow` 移动类型也没有
- `Adversaries/Overview`：「重要对手」「天赋成长」「技能成长」「装备成长」四节全缺，
  威胁类别表还少一整列（额外专注）
- `Equipment/Equipment Overview`：「注入」「物品堆叠」两节没译，
  且价值公式被写成 `(1 + 稀有度^3)`，英文是 `((1 + 稀有度)^3)` —— **算错了**
- `Conditions/Invisible`：英文整页规则被替换成一句「更进阶规则计划在未来更新中推出」
- `Overwatch` 天赋的译文，内容其实是 `Inquisitor` 的描述（串了）

以上全部补译完毕。

**关键发现：163 处 enricher 在翻译时被写成了裸文字**

`@Condition[exposed]` 这类标记不带标签，渲染出来的就是 lang 里的译名 —— 但旧译文把它们
直接写成了「暴露」两个字，于是玩家看到的是一段没法点、没有说明浮窗的普通文字。
`restore_enrichers.py` 逐级还原，163 → **3**（剩下 3 条在结构差异较大的规则页里，留待下一轮）。
顺带修掉的具体问题：

- `Resources` 页的「恢复 / 休息」链接指向的是**某个预设角色身上的动作实例**
  （`@Action[Compendium.crucible.pregens.Actor.iPMperuo6ZvBLnp9 recover]`），应为默认动作
- `Reactive Strike` 在译文里有 脱离打击 / 脱离交战打击 / 脱离攻击 / 反击 四种写法，全部归一到 enricher
- `Heroism` 页的法术名是机翻残留「组合 射线 of 生命」，英文那里是 `@Spell[life.ray.compose]` 链接
- 12 个符文熟练度天赋引用的动作链接，全被写成了加粗中文

**复验**

| 项 | 阶段 5 末 | 现在 |
|---|---|---|
| lang 四类缺口 | 0 | 0 |
| compendium 真实残余 | 0 | 0 |
| 外来文字 | 0 | 0 |
| 术语规则复跑 | — | **0 处待改** |
| LINK（坏链） | 163 | **3** |
| TRUNCATED（整段漏译） | 12 | **0** |
| BLOCK（段落数不符） | 73 | 64 |
| INLINE（加粗漂移） | 338 | 294 |

**遗留**

- **冒烟验证仍未做**
- BLOCK 64 / INLINE 294：段落合并、多包一层 `<strong>` 之类，不影响功能，观感层面的清扫
- 剩余 3 处 LINK 在 `Initiative and Turn Order` / `Skills` / `Adversaries Overview` 页
- ember 尚未开工

---

### 2026-08-06 · 阶段 7：ember lang 收官 + 法语社区包侦察

**ember `lang/cn.json` 补齐**：NEW 47 + UNTRANSLATED 3（`X` / `Y` / `uniqueMilestoneIdentifier`
判定为有意保留，进 `1-Ember汉化插件/lang/lang_keep_english.json`），并顺手统一了
`Attunement` 的译名（原本 同调 / 调谐 混用 → 一律**同调**）。
现在 486 键，四类缺口全 0，基准已同步。

11 个同调的译名沿用 compendium 既有写法：深渊 / 阿肯 / 灵气 / 科拉 / 余烬之心 /
卢克萨鲁姆 / 玛伊斯 / 奥比斯 / 普里莫迪斯 / 拉根 / 西格纳拉。

**法语社区包 `ember-fr` + `outils` 的评估**

结论：**译文没用，侦察结果很有用**。

*没有借鉴价值的部分*
- 他们的英文基准**比我们旧**：还写着 `Rune of Lightning`（我们已是 `Rune of Storm`），
  我们的基准多出 1423 串新内容（Crystallath、Juggernaut、新页面等）。不能当基准。
- 他们走的是**手写转换器**路线（`ember_journals_converter` / `ember_scene_levels_converter` …），
  正是第 8 节决议要避开的；我们的声明式 `registerMapping` 更好，不回头。
- 法语译文本身与中文无关。

*有借鉴价值的部分（已抽成英文清单存进 `5-其他内容/reference/ember-fr-recon/`）*

1. **`ember-hardcoded-strings-en.json` —— 148 条 babele 够不到的硬编码字符串，分 13 类**
   （prefixes / attunements / languages / soundscapes / advantages / criticals / knowledge /
   tooltips / ageAbbreviations / sectionHeaders / actionButtons / actionTooltips / dialogTitles）。
   这些写死在 Ember 的 `scripts/ember.mjs` 与模板里（例如 `"Day {DayOfCampaign}"`、
   `"Ancestry Details"`、`"Culture Details"`），babele 完全碰不到，只能靠 monkey-patch。
   **这是我们此前完全没盘点过的一整类内容** —— 不看这个包，只能靠一遍遍开世界慢慢发现。
   他们的补丁点：`EmberCalendar` 的日期/时间格式化器、TextEditor enrichers、section headers、
   action buttons、GM headers、`ui.notifications`、`DialogV2`、adventure importer、tag labels、
   calendar 时间按钮，以及 `crucible.CONFIG.knowledge / languageCategories / languages`。
2. **字体问题（已实测确认，对中文比对法语严重得多）**
   Ember 的 `styles/ember.css` 把 `--font-header` 设为 `Pirate Scroll`，h1/h2/h3 全用它。
   实测字形表：`PirateScroll.otf` 99 个码位、**CJK 0 个**；`Vollkorn.ttf` 1222 个码位、**CJK 0 个**。
   → 中文标题会整片走 fallback 甚至豆腐块。法语包的做法是换成 Cinzel；
   中文必须换成带 CJK 的字体（且不能依赖 Google Fonts CDN，国内取不到）。
3. `ember-terms-en.json`（412 条 Ember 专名）与 `ember-auto-terms-en.json`（3889 条名词/标题）
   —— 只取英文键，作为中文术语表 Ember 部分的待译底稿。
4. 他们的 `verificateur-glossaire.html` 是术语校验器，我们已有 Python 版，不需要。

**遗留**

- ember 6 个零汉化小包（`character` 164 串 / `crucible-effects` 42 / `dnd5e-effects` 42 /
  `crucible-affixes` 4 / `crucible-adversary` 94 / `crucible-character` 115）
- 硬编码字符串（148 条）与字体替换：需要在 `ember_cn` 里加一个补丁脚本，属新工作项
- ember 战役正文

---

### 2026-08-06 · 阶段 8：ember 四个小包收尾（自动循环第 1 轮）

**做完的**：`crucible-effects`(47) · `dnd5e-effects`(47) · `crucible-affixes`(6) · `crucible-items`(8)
—— 四个包全部 **100%**，标记漂移 0，外来文字 0。

- 两个 effects 包路径完全相同、47 条里 37 条内容一致，10 条按各自机制分叉
  （crucible 用 @Condition / [[/counterspell]]，dnd5e 用 &amp;reference[] / 长休 / 劣势）。
  一次翻译覆盖两包，分叉的 10 条各自另译。
- 11 个同调的祝福沿用既定译名：深渊 / 阿肯 / 灵气 / 科拉 / 余烬之心 / 卢克萨鲁姆 /
  玛伊斯 / 奥比斯 / 普里莫迪斯 / 拉根 / 西格纳拉。

**顺手查明的一件事，会显著影响第 8 项的排期**

`ember.crucible-adventure` 虽然显示 65% 已译，但它带着和 crucible 一模一样的那几类历史缺陷，
而且规模大约是 5 倍：**LINK 827 · BLOCK 592 · INLINE 721 · TRUNCATED 80**。
`ember.crucible-character` 另有 LINK 44 / INLINE 27。

→ 也就是说战役正文不是「翻剩下的 35%」那么简单，已译的 65% 还得过一遍
`restore_enrichers.py` + TRUNCATED 补译。好消息是 crucible 那轮的工具与
`enricher_surface_forms.json` 可以直接复用。

---

### 2026-08-06 · 阶段 9：ember.character 收尾（自动循环第 2 轮）

**做完的**：`ember.character` 274 条 → **100%**。漂移 0，外来文字 0。
内容是 11 个同调 × 5 阶（含阶位效果）、16 个血统、16 种文化、77 项特性/道途，以及 9 条描述。

**判断与决议**

1. **同调阶位命名格式**定为 `灵气 5：飓风 Aura 5: Hurricane` —— 中文用全角冒号，
   英文原名整体后缀，与 §8「专有名词双语并列」一致。
2. **血统与文化沿用 compendium 已有的双语译名**（阿尔提拉 Altyra / 龙裔 Drakon /
   印记裔 Signborn / 荆芽灵 Thornling / 疾行者 Strider …），没有另起炉灶。
3. **上游有两处拼写笔误**：`Heart 4: Verdant` 的效果键是 `Verdent`，
   `Orbis 3: Frenzied` 的效果键是 `Frenizied`。键必须照抄，否则 babele 匹配不上；
   译文按正确词义给（苍翠 / 狂乱）。
4. **补齐了上一轮的风格不一致**：阶段 8 那四个包的条目名当时写成了纯中文，
   而 ember 既有译文（以及 crucible）的惯例是条目名双语并列。已把 44 个条目名改为双语
   （`万花筒困惑 Kaleidoscopic Confusion` 这种形式）。子字段如 `adjective` 保持纯中文，
   与 crucible.affixes 的做法一致。
5. 文件夹名服从既定术语，而非 compendium 里的旧写法：`Ancestries` 用**血统**（不是祖裔）、
   `Attunements` 用**同调**（不是调谐）—— 战役包里还残留着旧写法，列入后续清扫。

**ember 进度**：13594 / 35157 = 39%；6 个小包里已完成 5 个。

---

### 2026-08-06 · 阶段 10：ember 小包全系列收官（自动循环第 3 轮）

**7a 完成**：`crucible-character`(80) 与 `crucible-adversary`(99) 译完，
至此 ember 的 **7 个小包全部 100%**（character / crucible-character / crucible-adversary /
crucible-effects / dnd5e-effects / crucible-affixes / crucible-items）。

**顺带修掉的三类历史缺陷**

1. **13 条描述是 v1.0.15 时代的占位译文** —— 内容写着「BETA TWO：该天赋尚未按 Crucible 机制更新」，
   而英文侧早已补全。包括全部 11 个同调的阶位说明（每个都是 5 阶的完整增益表，
   带 2 个 `@Action` 链接）、`Lunar Tattoos` 与 `Exceptional Taste`。全部按英文重译，约 1800 字。
   —— 这类缺陷 `validate_translations.py` 看不见（路径有中文值就算 100%），
   是 TRUNCATED 检测抓出来的。
2. **`@Embed[... inline overview]` 里的 `overview` 被当成正文译成了「概览」** ——
   那是段落 id，不是可译文本，译错等于嵌入块加载不出来。
   38 处（crucible-character 32 + 战役包 6），按英文 token 还原。
3. `Ashka Lineage` 的描述丢了两个 `@Condition` 链接，并多出一段英文早已删除的
   「ALPHA ONE：此特性尚未自动化」。已重译。

**判断**

- 同调阶位说明的行文统一为「阶位 N：……」，与 `ember.character` 里
  「灵气 5：飓风」的阶位命名呼应。
- 术语一律服从 crucible 侧：Empowered 强化 / Weakened 虚弱 / Deadly 致命 /
  Void 虚空 / Psychic 灵能 / Radiant 光耀 / Electricity 电击 / Fortitude 坚韧防御 /
  Healing Threshold 治疗阈值 / Lineage 血统 / Attunement 同调。
- **只有 @Embed / @Action 里的 `{标签}` 该翻译，方括号内的目标与参数一律照抄** ——
  这条已经被踩到两次（crucible 的 `runeLightning000`、ember 的 `inline overview`），
  写进第 8 节决议。

**7 个小包的漂移**：48 → **15**（1 BLOCK + 14 INLINE，都是加粗数量差，观感层面）。
外来文字 0，JSON 全部可解析。

**ember 总进度**：13740 / 35157 = 39%。剩下的几乎全在战役包。

---

### 2026-08-06 · 阶段 11：Ember 运行时补丁 —— 硬编码字符串与字体（自动循环第 4 轮）

**7b + 7c 一并完成**，两者同属「babele 与 i18n 都够不到」的同一类问题。

**新增 `scripts/ember-hardcoded-cn.mjs`**

先核对了法语包给出的 148 条清单对 ember 0.6.0 是否仍然成立：62 条能在源码里直接搜到，
其余是运行时拼出来的（`` `Attunement: ${attunement.name}` `` 这种）。查清了它们的来源 ——
Ember 注册了一批 TextEditor 富文本增强器（`[[/attunement X]]`、`[[/language X]]`、
`[[/knowledge X]]`、`[[/soundscape X]]`、`@Advantage[N]`、`@Critical…` 等），
标签在 JS 里拼好，两条汉化通道都碰不到。

补丁分三个层次，都是**只读不写**（不改 Ember 的数据，停用模块即恢复）：

1. **包装增强器** —— 只包 Ember 自己注册的那些，翻译其返回节点里的文本与 tooltip 属性；
2. **改写 `crucible.CONFIG.languages / knowledge`** —— Ember 往里塞的条目 label 是硬编码英文，
   会出现在角色卡下拉框里，这是唯一能改的地方；
3. **渲染钩子** —— 对 Ember 自己的界面根元素做一次 DOM 遍历，处理分节标题与按钮。

用桩环境跑了覆盖测试：**143 / 147**。未覆盖的 4 条是纪年缩写 `AC / AB / AT / AS`，
**有意保留** —— 它们是 Ember 历法的纪元代号，和 DC 一样属于记号，译开反而认不出来。

译名复用既有决议：11 个同调沿用 compendium 译名；34 个知识领域里有 30 条直接对齐
crucible lang 的 `KNOWLEDGE.*`（改一处要两边一起改，已在文件里注明）；
`Arcden` 用第 8 节裁定的**奥克登语**。

顺带发现 crucible lang 的 `LANGUAGES.Sign` 译作「标记」是错的 —— 那是**手语**（sign language），
补丁里按手语处理，crucible 侧的 lang 待下轮一并改。

**新增 `styles/ember-cn.css`**

实测过 Ember 的两个字体文件里一个 CJK 字形都没有（PirateScroll.otf 99 码位、
Vollkorn.ttf 1222 码位，CJK 均为 0），而回退链末端只有一个 `serif`。

做法是**不替换原字体**，只在回退链后面补中文字体。浏览器的字体回退是逐字形的，
所以拉丁字母仍由 Pirate Scroll / Vollkorn 渲染，只有中文字符落到后面，Ember 的视觉风格得以保留。
不打包字体文件（一套中文字体动辄 10 MB 以上），也不挂 Google Fonts（国内取不到会静默失败），
只用系统已有字体，按 Windows / macOS / Linux 排列。

另外压掉了标题 0.2em 的字距（那是为拉丁小型大写设计的，中文会散开），
并把正文行高提到 1.7。

**冒烟验证清单增补**：这两项都**无法靠脚本证实**，必须在真实世界里看：
① 同调/语言/知识的富文本标签是否变中文；② 角色卡分节标题与事件按钮；
③ 中文标题是否还掉进宋体默认样式；④ 控制台有无本模块的警告。

---

### 2026-08-06 · 阶段 12：crucible 标记漂移清扫（自动循环第 5 轮）

**新增 `qa/fix_bold_drift.py`** —— 修「译文比英文多包了 `<strong>`」。

没有单一规则能判断某个加粗是不是多余的，得先知道英文里哪些词该加粗。办法是**自举**：
先扫加粗数量本来就对得上的条目，按出现顺序把 EN 粗体词与 CN 粗体词配对，
学出一张「英文粗体词 → 中文粗体词」对应表（242 条，支持度 ≥ 2）；
再拿它判定多包的条目里哪些加粗是译者自己加的。

第一版会**拆过头** —— 对应表里没有的词会被误判为多余，结果加粗数掉到英文之下（只修好 70/253）。
改成**只拆掉超出的那几个**（不在预期集合的排前面，够数就停）后到 **250/253**。
拆多了比多包一层更糟，这条写进了脚本注释。

**本轮清扫结果**

| 类别 | 之前 | 现在 | 做法 |
|---|---|---|---|
| INLINE 加粗漂移 | 294 | **44** | 自举对应表 + 最小化拆解 |
| BLOCK 段落差异 | 64 | **13** | 见下 |
| LINK 坏链 | 3 | 3 | 都在结构差异大的规则页里，留待下轮 |

BLOCK 拆成四类分别处理：

- **21 条是英文里的空 `<p></p>`，译文没照抄** —— 这不是漏译。改的是**检测器**而不是译文：
  `scan_markup_drift.py` now 先剔掉空段落再数，免得真正的段落差异被噪音淹没。
  往译文里塞空段落对玩家没有任何价值。
- **16 条译文是裸文本、没有 `<p>` 包裹** —— 真缺陷，Foundry 渲染时的间距与样式会不对。已补。
- **17 条译文丢掉了末尾的「Playtest Notes」小节** —— 只有 4 种不同的段落（都是一句话的
  「尚未实现自动化」），已补译。
- 剩 13 条是段落合并/拆分与一个表格结构差异，需逐条判断，留待下轮。

**顺带**：7d 的 `LANGUAGES.Sign` 由「标记」改为**手语**（sign language 被当成了「标记」）。

**复验**：覆盖率 4556/4717 与真实残余 0 均未回退，lang 四类缺口 0，外来文字 0。

---

### 2026-08-06 · 阶段 13：战役包机械清扫 + 一个改变排期判断的发现（自动循环第 6 轮）

**做了什么**

先用这几轮练熟的工具链扫战役包，再顺着异常查根因。

| 项 | 之前 | 现在 |
|---|---|---|
| INLINE 加粗漂移 | 734 | **265** |
| LINK | 822 | 707 |
| TRUNCATED | 80 | 69 |

工具增补：
- `restore_enrichers.py` 加 `--lang-repo / --lang-package` —— 修 ember 时术语表得从
  crucible-cn 取，因为 `@Condition` / `@Action` 的名字都归 crucible 管。
- 同一脚本加 `ember_candidates()` —— Ember 自己的增强器按 token 现算候选写法。
  其中 `@CriticalSuccess[13]` 的 DC **不出现在渲染结果里**，译文那侧没有任何线索，
  只能靠段落对齐从英文同一段取回 N。
- `fix_bold_drift.py` 加反向模式（英文有粗体、译文没有），同一张自举对应表反过来用。

**踩到并修正的一个自己造成的回退**：还原 enricher 时 `candidate_forms` 会把
`<strong>词</strong>` 整个吃掉，导致 INLINE 从 233 涨回 279。先假设是「英文把 enricher
包在 strong 里」，一查 0 命中，假设不成立；真正原因是候选写法吞掉了本该保留的加粗。
用反向补加粗模式补回。**教训：修 A 类漂移时要顺带复查 B 类，两者会互相牵动。**

**关键发现：战役包剩下的 LINK 与 BLOCK 是同一个问题，而且不是「链接坏了」**

`@Advantage[2]` 缺 501 处怎么都还原不了，顺着查下去：

```
英文 <li class="advantage"> 共 708 处，译文 0 处
<li> 总数：英文 10558 / 译文 8351   （少 2207）
<p>  总数：英文 46788 / 译文 39330  （少 7458）
```

**译文里整块内容不存在。** 已译的 65% 是对着更早版本的 Ember 翻的，
那时还没有「知识检定 → 获得恩惠」这类列表项，所以不是译错，是压根没有。

这改变了第 8 项的排期判断：战役包的真实工作量**不是「翻剩下的 35%」**，
已译的 65% 里还缺着约 2200 个列表项与 7400 个段落。逐页比对补齐是唯一的办法，
机械工具到此为止 —— 剩下的 LINK 707 / BLOCK 584 会随着补齐内容一起消失，
单独去修它们没有意义。

**复验**：覆盖率 13740/35157 未回退，外来文字 0，JSON 全部可解析。

---

### 2026-08-06 · 阶段 14：战役正文首批 —— Ushna Dredging Docks（自动循环第 7 轮）

**切批方式确定**：按 journal 切，一轮一批。战役包共 76 组 journal，
未译字符合计约 148 万，最大的几组是 `Disturbed Earth`(22.8万)、
`Arctus Plateau Gazetteer`(8.9万)、`The Winding Trail`(7.7万)。
先挑一个自成一体、规模合适的：`Ushna Dredging Docks`（21 条 / 2.7 万英文字符，0% 已译）。

**本批完成 13/21 条**（约 8.7 千英文字符）：区域总览、地名志参考、工头办公室、仓库、茅厕
及各自标题与两个分类名。剩下 Stone Dock / Loading Area / Lake Orial / Dock Road
四页（约 1.8 万字符）下一轮做完。

**处理要点**

- 专名一律沿用战役包既有译法：乌什纳疏浚船坞 / 奥里亚尔湖 / 阿克图里安 / 贝雅克 /
  雷德拉克 / 阿克图斯高原 / 亚纳克；新出现的人名按音译：雅弗塔·乌什纳、巴亚尔·乌什纳、
  奥奇尔·乌什纳、拉拉。
- `<sup class="system-swap-inline">` 里 dnd5e / crucible 双轨技能检定的结构原样保留 ——
  那是 Ember 用来在两套系统间切换显示的机制，动了就两边都坏。
- 纯 `@Embed` 的页面（地名志参考）不含中文，被「无中文」闸拦下。这是既有例外，
  按阶段 10 的办法直接写入并自校标记。

**一个术语决议**：`Globlin` 由「泥砾精」改为音译**格布林**。
原译是意译，遇到 `Mud Globlin` / `Paint Globlin` 这类前缀构词就没法组合
（「泥浆泥砾精」读不通）；战役包里还另有一处把它误译成「绘画地精」——
地精是另一种生物。改成音译后：泥浆格布林 / 颜料格布林。已同步 3 处。

---

### 2026-08-06 · 阶段 15：Ushna Dredging Docks 收官（自动循环第 8 轮）

补完剩余 4 页（石砌码头 / 装卸区 / 奥里亚尔湖 / 码头路，约 1.8 万英文字符），
该 journal **21/21 全部完成**，回写零拒绝、无新增漂移。

**本批的判断**

- **英文源里有代词错误**：Yaphta 在同一页里明确标注 she/her，却有两处误用 his/he
  （小标题 "About his anxieties?"、以及 "One side of **his** mouth turns up"）。
  中文的他/她之分会把这个错误直接暴露出来，因此按角色自述的代词译作「她」。
  —— 这类源文错误不照抄，属既有做法（阶段 12 也修过 Shocked 持续时间的数值错误）。
- 新出现的专名按音译并沿用已有姓氏：雅弗塔·乌什纳 / 巴亚尔·乌什纳 / 奥奇尔·乌什纳；
  机械与场景术语按功能定名：疏浚轮 / 疏浚爪 / 沉淀池 / 蚌壳斗 / 移动式轮舟。
- `Agrimage Circle` → 农术师环会（`Agrimage` 沿用阶段 10 的「农术师」）；
  `Ordani trading houses` → 奥尔达尼商行；`Redsai's Roadman` → 雷德赛的巡路人。

**进度**：战役包 12865/19826 = 65%（本轮 +21 条 / 约 2.7 万英文字符）；
ember 总计 13760/35157 = 39%。76 组 journal 里第 1 组完成。

---

### 2026-08-06 · 阶段 16：Arcturel Dives 第一批（自动循环第 9 轮）

该组 40 条 / 5.9 万英文字符，一轮做不完，先取「店铺 / NPC / 办公」这一类共 10 页
（约 1.56 万字符）：三层矿井办公室、扳力修理行、霍布·科雷尔的非凡坐骑行、
沃特伯恩酿酒坊、佐迪·特拉斯克的公寓、餐区、储物柜区、后勤办公室、地名志参考，
以及标题与三个分类名。**22/40 完成**，回写零拒绝。

**新定的专名**（沿用既有风格：地名意译、人名音译）

阿克图瑞尔矿渊 Arcturel Dives（`The Dives` 单独出现时作「矿渊」）· 三层矿井 ·
扳力修理行 · 霍布·科雷尔 · 卡奥尔 · 詹恩 · 伦纳里 · 沃特伯恩酿酒坊 · 卡拉尔酒 ·
聚归馆（Rallyhome，酒馆）· 掌灯人 Lamplighters · 沃格巢 · 震颤感知。

**一个术语决议**：`Waterborne` 作**家族名**时音译为**沃特伯恩**。
战役包里原有一处 `Waterborne Whiskey` 被当成普通词译作「水运威士忌」——
但这家酒是沃特伯恩家族酿的，那处应改为「沃特伯恩威士忌」，列入后续清扫。

**顺带修掉检测器的一个漏洞**

本批回写后 PLACEHOLDER 从 20 涨到 22，查出是 `&Reference[surprise]{Surprised}` ——
dnd5e 的 `&Reference[…]{标签}` 与 Foundry 的 `@UUID[…]{标签}` 同理，标签是给玩家看的文字、
本就该译，但 `scan_markup_drift.py` 的 `strip_labels` 只认 `@xxx[…]` 和 `[[…]]`。
补上之后 PLACEHOLDER **22 → 1** —— 一并清掉了 19 条同类的历史误报。
crucible 侧复扫确认不受影响（仍为 0）。

**进度**：战役包 12886/19826 = 65%；ember 总计 13781/35157 = 39%。

---

### 2026-08-06 · 阶段 17：Arcturel Dives 第二批（自动循环第 10 轮）

矿坑主体 4 页：枢纽厅、聚集处、空矿坑、在采矿坑（约 1.36 万英文字符）。
**30/40 完成**，回写零拒绝，无新增漂移。

**新定的专名与术语**

枢纽厅 The Hub · 聚集处 · 空矿坑 · 在采矿坑 · 副储藏室 · 斯基瑟 Skither（沿用既有）·
装置 The Device · 通电／断电 Powered/Depowered · 矿渊矿井效应 · 万德伦 Wandren ·
地形撞击 · 重重摔落 · 伤害阈值。

**本批的复用**：沃格巢与「沟壑判定」两段在多页里逐字重复，抽成常量拼装，
既保证同一段文字在各页完全一致，也避免手抄出错 —— 这一批 4 页里有 3 页共用沃格巢段落。

**进度**：战役包 12894/19826 = 65%；ember 总计 13789/35157 = 39%。
`Arcturel Dives` 还剩 5 页（区域总览 8.0K、最后一坑 7.8K、储藏室 5.2K、
阿沃达的药剂 4.7K、副储藏室 4.2K，合计约 3.0 万字符），约需两轮。

---

### 2026-08-06 · 阶段 18：Arcturel Dives 第三批（自动循环第 11 轮）

储藏室 / 副储藏室 / 阿沃达的灵药铺（约 1.4 万英文字符）。**36/40 完成**，
回写零拒绝，无新增漂移。只剩「区域总览」与「最后一坑」两页（合计约 1.6 万字符）。

**新定的专名与术语**

阿沃达的灵药铺 · 埃梅琳·阿沃达 · 水术师 · 浓缩矿尘 · 金属碎解剂 · 阿布里克斯 Aburyx ·
落水洞深处 · 沃特伯恩家族酿酒坊 · 有毒空气 · 隐藏的开关 · 窸窣的斯基瑟；
药水类沿用通行译法：炼金银 / 抗毒剂 / 恒开花 / 滑腻油 / 警觉药水 / 治疗药水 /
抗性药水 / 水下呼吸药水 / 阿特里坎疾病 / 抗真菌药剂 / 抗凝补剂 / 溶解补剂 / 护心灵药。

**进度**：战役包 12900/19826 = 65%；ember 总计 13795/35157 = 39%。

---

### 2026-08-06 · 阶段 19：Arcturel Dives 收官 + 一整卷被改名埋掉的旧译文（自动循环第 12 轮）

**做完的**

1. `Arcturel Dives` 最后两页（区域总览 8.0K、最后的矿坑 7.8K），**40/40**，回写零拒绝。
2. 查出并修掉 **18 处「译文把标记内部的目标译成了中文」** —— 一整类此前无人检测的失效。
3. 发现 `Arcturel Upper` 是今天 `Arcturel Tradeway` 的旧名，**28 页译文一直躺在孤儿路径上**，
   移植后该 journal 待译归零（约 5 万英文字符不必重译）。

**关键发现 1：覆盖率把「没有可译文本的条目」也算成了待译**

`<p>@Embed[JournalEntry.x.JournalEntryPage.y inline]</p>` 这类条目整条都是机械标记，
永远不可能含中文，却被 `validate_translations.py` 一直记为待译 —— 全库 298 条。
`ember.crucible-character` 因此长期显示 90%，其实早已做完。

已给 `validate_translations.py` 加 `translatable()` 判据（剥掉标签与标记后还有没有拉丁字母；
`{标签}` **不**剥，因为标签是可见文字），这类条目改记为 `n/a` 一列，不进 leaves、不进待译清单。
crucible 侧复跑 4717/4556 与 n/a 0 完全未变，说明判据没有误伤。

**关键发现 2：13 条译文把中文写进了标记内部，且所有既有检查都看不见**

剥离 n/a 那批时，反查出 11 条「英文是纯标记、中文却有字」的条目 —— 全是把
`@Embed[JournalEntry…]` 的 **`JournalEntry` 这个文档类型关键字**译成了「日志条目」
（有一条连 `JournalEntryPage` 都成了「日志条目Page」）。嵌入块直接加载不出来。

为什么一直没被发现：路径上有中文 → 覆盖率算 100%；`apply_translations.py` 的标记闸只作用于
**经它写入**的值，v1.0.15 时代的旧译文从没过过闸。

新增 `qa/scan_markup_targets.py`（全库扫「方括号里有中文」，第 8 节那条决议的机械化形式），
并按严重度分两类，否则报告没法读：

| 判定 | 数量 | 说明 |
|---|---|---|
| BROKEN | 33 → **15** | 标识符被译：`@Embed[日志条目…]`、`@Condition[目盲/失聪/恐慌/倒地/受拘束/失能/破碎/未察觉]`、`[[/language 径道语]]` |
| by-design | 79 | `readaloud="中文"`（可见旁白）、`[[/r …#分身被摧毁]]`（掷骰说明）、`[[/item 战镐]]`（dnd5e 按角色身上**已被翻译**的物品名解析，中文才是运行时存在的那个名字） |

修复走 `4-临时脚本/2026-08-06/fix_translated_markup_targets.py`：拿英文基准按**同类同序**
对齐后把英文原文抄回，再经 `apply_translations.py` 的标记闸复核。加了一道形状判据
（中文段只能顶替一个不含空格的 token），否则会把 `swoopingStrike00}和@UUID[` 里的「和」
换成英文的 ` and `、把 `readaloud= "卡莉…"` 整段吃掉 —— 两处都真的被拦下了。

剩下的 15 条各有原因，不是漏修：11 条在 `crucible-character` 的孤儿条目上（英文侧已无该条目，
babele 永远不会应用）、2 条同类孤儿在战役包、1 条是**上游自己的错**
（`@UUID[…swoopingStrike00}` 英文侧就是 `}`）、1 条所在页缺整块内容被闸拦下（属第 8c 项）。

顺带把 `apply_translations.py` 的「无中文则拒」改为**只对有可译文本的条目生效** ——
否则纯标记条目的正确值（不含中文）永远写不进去，这 10 条根本没法修。

**关键发现 3：`Arcturel Upper` / `Arcturel Lower` 是两卷被改名埋掉的旧译文**

`compendium/cn` 里有两个英文基准中不存在的 journal，共 47 页、6.4 万字符，一直计在 611 条孤儿里。
逐页比对后查明是 0.6.0 的改名：

| 旧名 | 今名 | 页数 | 处置 |
|---|---|---|---|
| `Arcturel Lower` | `Arcturel Dives` | 19 | 本轮已全部重译，**移植时被「目标已有译文」闸拦下 33 条**，不覆盖 |
| `Arcturel Upper` | `Arcturel Tradeway` | 28 | **移植** |

决定移植的依据不是「有译文」，而是**结构逐段对得上**：28 页里 27 页的
`<p>`/`<li>`/`<section>` 计数与今天的英文完全一致，全卷 `<p>` 238/240。
这与阶段 13 那批「已译 65% 却缺 2200 个 `<li>`」的情况**不是一回事**，所以不会把
「看起来 100%、实则缺整块」的问题引进来。抽读 `Rallyhome` 全页确认译文质量可用。

移植后逐项清理（每一步都过标记闸）：

- **5 处结构漂移**：`Scene.emberArcturelUnd` 已被上游换成 `emberArcturelLow`（升降机/悬空大道/
  安保室 3 处，其中安保室的两个标签还互相串了），`Silver Beam Foyer` 缺一整个
  `complex-check` 列表与两个文化链接（该页正是唯一结构对不上的那页）
- **6 处旧译残留**：整句英文没译（`都 eager to draw a crowd…`）、三个英文书名、
  两处 dnd5e 分支里漏译的 `check`、`gold/silver` 没译成金币/银币。
  这批是靠新写的残留英文探针抓出来的 —— `gp/sp`、`CG`、`he/him`、`&amp;reference[…]`
  是全库既有写法，要先排除掉才看得见真的漏译
- **术语统一**：`Rallyhome` 拉力之家→**聚归馆**（阶段 16 定名）13 处、
  `Arcturian` 阿克图瑞安→**阿克图里安**（对齐文化页条目名）10 处。
  另有 4 处因所在条目**本来就**与英文结构不符被闸拦下（`Arctus Plateau Gazetteer.Arcturel`
  一页 CN 有 85 个 `<p>`、EN 只有 48），留给第 8c 项
- 补 5 个分类名（总览 / 聚归馆 / 银光束总部 / 贸易道各处 / 底腹区各处）

**复验**

| 项 | 阶段 18 末 | 现在 |
|---|---|---|
| 战役包覆盖 | 12900 / 19826 | **12955 / 19605 = 66%**（分母减 221 条 n/a） |
| 待译字符 | 1,236,195 | **1,159,596**（−7.7 万） |
| LINK | 707 | **689** |
| BLOCK / INLINE / TRUNCATED / PLACEHOLDER | 584 / 265 / 69 / 1 | 584 / 265 / 69 / 1（移植零净增） |
| 外来文字 | 0 | 0 |
| markup targets BROKEN | （无此检查） | 15，全部有已知原因 |

**遗留**

- `Arcturel Tradeway` 的 28 页是**移植来的旧译文**，已过全部机械检查并抽读一页，
  但没有逐页通读。列为第 8f 项。
- 其余 995 条孤儿（场景注记、actor 内嵌物品）没有改名规则可套，仍留在原地
- `The Tunnels`、`Waterborne Family Distillery` 两页今天已无同名页，未移植
- 上游自身缺陷两处，未跟改：`@Condition[exhaustion` 少右括号（2 处）、
  `@UUID[…swoopingStrike00}` 用了 `}`。照抄英文，等上游修
- `The Clever Fox (Engraved Sign)` 的 `description.private`：英文已被上游清空成
  `<p></p><p></p>`，中文还留着一整段旧版描述。无害（玩家看到的是更多内容），未删，留档待定

---

### 2026-08-06 · 阶段 20：并行翻译管线 + 第 1 批 5 卷 + 全库术语统一

**范围**：把「一次一卷」改成「一次多卷并行」，跑通第一批并验证质量；顺带把剩余工作量测准。

**并行管线怎么搭的**（工作目录 `scratchpad\parallel\`，产物是 batch，不进 git）

| 件 | 作用 |
|---|---|
| `BRIEF.md` | 译者须知：标记硬规则 / 既定译名 / 文风 / 自检命令。**每个 agent 必读** |
| `probe.py` | 按英文词查全库既有中文写法；`--names` 查某卷页名对照 |
| `residue.py` | 查译文里残留的英文（`gp`/`AC`/`he/him`/`&reference[…]` 等既有写法先排除掉） |
| 每单元一个目录 | `todo.json`（待译）+ `already_translated.json`（同卷已译页，术语锚点） |

关键设计有两条：

1. **译者自己过闸门**。agent 交付前必须反复跑 `apply_translations.py --dry` 到 0 拒绝 ——
   标记类错误在返回主控之前就被挡掉，主控不必逐条复核。第 1 批 162 条落盘时**零拒绝**。
2. **谁都不许写 `compendium/cn`**。译者与审校只写自己目录下的 batch，落盘只由主控做。
   这样任何一个 agent 出问题都不会污染译文库，回退只是丢掉一个 batch 文件。

编排：每单元一个译者 → 一个**对抗式审校**（提示词要求「默认假设译者出了错，去证实它」，
能改的直接改并重跑闸门）→ 最后一个跨单元术语核对 agent。

**第 1 批结果**：`Glitter in the Dark`(44) · `Lantern Roads`(26) · `The Book Of Tales`(31) ·
`An Old Friend`(24) · `Ancient Paths`(37) —— 162 条 / **14.9 万英文字符**，11 个 agent，约 63 分钟。

| 项 | 之前 | 之后 |
|---|---|---|
| 战役包覆盖 | 12955 / 19605 | **13117 / 19605 = 67%** |
| 待译字符 | 1,159,596 | **1,010,355**（−149,241） |
| LINK / BLOCK / INLINE / TRUNCATED | 689 / 584 / 265 / 69 | **687 / 582 / 264 / 69**（不升反降） |
| 外来文字 | 0 | 0 |

审校结论 3 GOOD + 2 FIXABLE，**critical 0**。跨卷核对改掉 12 处冲突
（影匣→暗箱、内层领域→内界、事件结算→事件结束、卢森特→辉耀等）。

**关键发现：agent 查出我给的术语表有三条是错的**

我写 `BRIEF.md` 时从本会话刚做完的 Arcturel 那两卷里抽术语，抽出来的东西**只在那两卷里成立**：

| 我写的 | 实际全库 | 判定 |
|---|---|---|
| Inkaro Pearl 印卡罗珍珠 | 因卡罗 126 : 印卡罗 39，glossary_ec 亦作因卡罗 | 我错 |
| Amalthea 阿玛尔忒娅 | 阿玛尔忒亚 161 : 13，演员条目名就是阿玛尔忒亚·石艺 | 我错 |
| 引号用「」 | “” 2618 : 「」44 | 我错 |

三个 agent 各自独立查到并**按既有译文而不是按我的表**处理，在报告里逐条列出依据 ——
这正是「审校要能推翻上游指令」的价值。BRIEF 现已加一段：本文件与既有译文冲突时以既有译文为准，
判断依据强弱是 **同名条目/物品的 `name` 字段 > 同卷已译页 > 全库多数 > glossary_ec > 本表**。

**顺势做掉的全库术语统一**（`unify_rules.2026-08-06c.json`，含挂了两个阶段的第 8b 项）

| 术语 | 统一为 | 处置量 | 依据 |
|---|---|---|---|
| `Attunement` | **同调** | 调谐 418 → 0 | lang 与小包早已是同调，角色卡上显示的就是它，正文必须跟 |
| `Ancestry` | **血统** | 祖裔 149 → 0 | 阶段 9 已定名 |
| `The Dives` | **矿渊** | 底层区 21 → 0 | journal/场景/分类名都是矿渊；底层区把专名译丢了 |
| `Inkaro` | **因卡罗** | 印卡罗 39 → 0 | 改 4 个物品条目名比改 126 处正文便宜 |
| `Amalthea` | **阿玛尔忒亚** | 13 → 0 | 演员条目名 |
| `Rallyhome` | **聚归馆** | 补上阶段 19 被拦下的 1 处 | |
| 引号 | **“”** | 44 对 → 0 | |

残余 `调谐` 46 处全部落在**死路径**（`Players' Guide.Attunement Progression` 是孤儿页）或
**内容早已与英文脱节的页**（`Ember Background` 的中文引用的 UUID 英文侧已不存在）——
属第 8c 项，不在术语统一的射程内。

**修掉的两个工具 bug**

1. `resolve_generic_fallback.py` 一直在读错目录 —— 它用 `'Crucible' in 仓库路径` 判断报告目录，
   而**项目根目录本身就叫 `Ember-Crucible Translation Project`**，于是 ember 仓库也命中，
   静默去读了 crucible 的清单。这个脚本对 ember **从来没跑对过**。改成只看仓库目录名后，
   首次拿到 ember 的真实数字：战役包 6650 条待译里 **363 条 babele 会自动取译文**，不该翻。
2. `unify_terms.py --write` 的写回路径对**带点的条目名**（`Patch 0.5` 这类）会 KeyError 崩掉，
   而且是写到一半崩。改成逐级下探时把后续段用 `.` 拼回去重试。

**把剩余工作量测准了**（此前只有阶段 0 的粗估）

*战役包*（扣掉 babele 自动解析后 1,111,337 字符）：

| 桶 | 条数 | 字符 |
|---|---|---|
| journals（正文） | 1274 | 660,227 |
| actors.items（战役独有 NPC 能力） | 2598 | 265,109 |
| tables（随机表结果） | 586 | 82,580 |
| actors（传记/原型名） | 496 | 64,980 |
| scenes（场景注记） | 1058 | 16,194 |
| items / effects / actions | 255 | 21,438 |

*第 8c 项（已译页里缺失的整块内容）*：新增 `4-临时脚本/2026-08-06/measure_8c.py`，
按「英文区块数 − 中文区块数」的比例折算字符 —— **398 个条目、3742 个区块、约 53.3 万字符**。
其中一批 `Area Overview` 缺得异常狠（Yakoshta 71/73、Mythspire 38/40、Toothbreaker 49/52），
根因是上游把 `Gameplay Details` 页**并进了** `Area Overview`，而旧的 `Gameplay Details` 中文页
正躺在孤儿路径上 —— 约 4.5 万字符可以直接搬过去补。

*dnd5e 孪生包 `ember.adventure`*：名义上 831 万字符，实际**不是翻译工作量**。
`measure_twin_tm.py` 实测：

| | 条数 | 字符 |
|---|---|---|
| 现在就能用精确匹配 TM 直接填 | 8497 | 7,302,532（**88%**） |
| crucible 版同一句也还没译，随 crucible 侧推进自动覆盖 | 4694 | 871,961 |
| **孪生包独有、必须单独翻** | **1213** | **142,340** |

→ 孪生包需要的是**一个填充脚本 + 一轮清理**，不是一轮轮翻译。

**孤儿页面（第 8g 项）**：`match_orphan_pages.py` 用**标记指纹**（UUID 与内联命令会原样抄进译文，
是天然的指纹）而不是页名来配对 —— 页名恰恰是改掉的那个东西。42 个孤儿页面里两个强匹配：
`In The Behemoth's Wake → In the Behemoth's Wake`（0.99，只改了大小写，7913 字已译）、
`Orb Room → Service Room`（0.96）。已挂进第 2 批相应工作目录当底稿。

**剩余总量与轮次**：常规待译 111 万 + 第 8c 项 53 万 + 孪生包独有 14 万 ≈ **179 万字符**。
按第 1 批的口径（每译者约 3 万字符），一轮 9 个单元约 27 万 → **7 轮左右**。

---

### 2026-08-06 · 阶段 21：并行第 2 批（9 单元 / 27.4 万字符）+ 两类新缺陷

**第 2 批**：`The Winding Trail`(33) · `Gamemaster's Guide`(20) · `The Expedition Challenge`(31) ·
`Chapter 2 Events`(33) · `Players' Guide`(7) · `Mythspire Observatory`(12) ·
随机表三批(291+170+125) —— **722 条 / 27.4 万英文字符**，19 个 agent。

| 项 | 之前 | 之后 |
|---|---|---|
| 战役包覆盖 | 13117 / 19605 | **13839 / 19605 = 71%** |
| 待译字符 | 1,010,355 | **736,524**（−273,831） |
| LINK / BLOCK / INLINE / TRUNCATED | 687 / 582 / 264 / 69 | **684 / 579 / 264 / 69**（再次不升反降） |
| 外来文字 | 0 | 0 |

722 条落盘**零拒绝**。三个单元用上了挂进工作目录的**旧路径译文**当底稿：
`In the Behemoth's Wake` 那页（指纹相似度 0.99，只改过大小写）逐段与今日英文对齐后改了 8 类问题 ——
页名 `wake` 误译成「苏醒」、`Randomization Procedure` 小标题整条漏译、
多出一整段英文早已删除的正文、dnd5e 分支 4 段留英文、三个 h3 物种名留英文。
**这正是「旧译文只能当底稿、不能照抄」的实证**。

**中途两次撞上 session limit**。第一次 9 个单元只完成 3 个、10 个 agent 挂掉；
用 `resumeFromRunId` 续跑，已完成的从缓存回放、失败的重跑，磁盘上没有半截文件需要清理。
**这条要记住：并行批次必须可断点续跑，工作单元之间不能有共享可变状态。**

**agent 改了公共脚本，并且是对的**

`Gamemaster's Guide` 的页名自带小数点（`Patch 0.5.1` … `Patch 0.6.0`），
而 `apply_translations.py` 用 `path.split('.')` 拆路径，于是这 10 条一律被判「no English source」，
而 path 又必须逐字节照抄、无法从 batch 侧绕开。译者在脚本里加了 `split_path()`：
仍然先走原来的朴素拆分，**只有解析不到时**才按最长匹配键回退遍历英文结构。

我复核了这个改动：纯增量 37 行；拿第 1 批 5 个 batch 回归，结果与改动前逐字节一致
（44/26/31/24/37，全部 0 拒绝）；20 条补丁页在标记闸下 0 拒绝，
这本身就证明新解析确实落在正确的英文条目上（落错页的话标记签名对不上，一定会被拒）。
**予以保留** —— 这是 `unify_terms.py` 那个「带点条目名」bug 的同源问题，两处都修了。

**关键发现 1：中文坐在错误的活路径上（比没翻更糟）**

`Mythspire Observatory` 的译者读正文时发现 `Ancient Lift` 的中文写的是另一间厅室。
这类缺陷所有既有检查都看不见：路径上有中文 → 覆盖率算 100% → 永不进待译清单 →
玩家读到的是别的房间。

据此写了 `detect_swapped_pages.py`：用**标记指纹**（`@UUID`/`[[/…]]` 会原样抄进译文，
是天然指纹）把全库 19605 条缩到 **4 个候选**，逐条读正文定性：

| 候选 | 结论 |
|---|---|
| `Spellbreaker Tower / Storage` | **真错位** —— 该卷两间储藏室开头 readaloud 逐字相同；中文实为 `jyEjb9CXfSzRRZCf`（水/酒/灯油那间）的译文 |
| `Lightless Halls / Stone Bowl` | 假警报（英文被改短，中文留着旧段落） |
| `Aedir Signalpost / Lookout Post` | 假警报（同上） |
| `Lightless Halls / Void Bridge` | 假警报（属第 8c 项） |

试过给它加自动判据，**两个方向都会判错**：按「英文本页要有足够标记」过滤会把唯一的真错位滤掉；
按译文/英文长度比 <0.9 判错位，则正好把真的（1.07）判成假、把假的（0.61）判成真。
最终定性只能靠读正文，这条实测结论写进了脚本文件头，免得下次再试一遍。

处理：**删掉错的译文，让它退回待译**。留着的话覆盖率会一直把它算作已译；
删掉之后玩家看到的是英文，而它会重新出现在待译清单里被排进下一轮。

**关键发现 2：译文里留着英文早已删除的内容（第 8c 项的镜像）**

`measure_stale_extra.py`：**189 条 / 776 个多余区块 / 约 4.8 万字符**。
上游把页面改短或改写，中文留着旧段落，玩家会读到已被删除的规则与场景。
`measure_8c.py` 只找「中文比英文短」，看不见这一类。

**风格归一**：引号在第 1 批后已归零；破折号同理 —— 全库**无空格 3798 : 带空格 127**，
带空格那批是我在 BRIEF 里写错后引入的，本轮一并归一（60 条 / 183 处）。

---

### 2026-08-07 · 阶段 22：并行第 3、4 批 + 自动循环

**第 3 批**（12 单元 / 449 条 / **30.8 万英文字符**）：`Disturbed Earth` 按页切 8 块、
`Arctus Plateau Gazetteer` 切 3 块、12 个小卷合成 1 块。25 个 agent 全部完成。

| 项 | 之前 | 之后 |
|---|---|---|
| 战役包覆盖 | 13838 / 19605 | **14287 / 19605 = 73%** |
| 待译字符 | 739,237 | **431,900**（−307,337） |
| LINK / BLOCK / INLINE / TRUNCATED | 682 / 577 / 263 / 68 | 682 / 577 / 263 / 68（**零净增**） |

**同卷跨块一致性是这一批的真问题**。8 个 agent 同翻一卷，跨块核对查出 12 组分叉，
最典型的是 `Terrane`：两块作「特雷恩」、四块作「泰兰」，**同一个 Actor UUID 在不同块挂了两种标签**。
按全库多数（88:35）统一为泰兰；`Earthen Henge` 则按 Gazetteer 的页名字段统一为「土石环阵」。
→ 决议：切块并行时，**跨块核对不是可选项**。

**修掉一条自相矛盾的规则**：`apply_translations.py` 的标记闸把整个
`@Embed[Actor.x readaloud="…"]` 当作必须逐字保留的机关，于是那段**要念给玩家听的旁白**
根本无法翻译 —— 而库里已有 22 处是翻过的。译者为满足「0 拒绝」照抄了英文，
同时把译好的中文**另存上报**（`_pending_embed_readaloud.NOT-A-BATCH.json`），没有硬闯也没有悄悄放弃。

两处同源修复（比对前把 `="…"` 的值抹平，参数名与括号内其余部分照常比对）：
`qa/apply_translations.py` 的 `markup_signature()`、`qa/scan_markup_drift.py` 的 LINK 判据
（后者不改的话，每翻一段 readaloud 就多报一条 LINK —— 本轮实测 682→686，改后回到 682）。
全库扫下来这条旧规则一共挡住 **3 处 1298 字符**，已补译，现为 0。

**第 4 批**（11 单元 / 2955 条 / **33 万英文字符**）：`actors` 桶 —— actor 身上的内嵌物品与能力。
性质与前三批不同（不是战役正文，是武器名/天赋名/动作描述/传记），风险从「漏译整段」
变成「同名异译」与「机制数值错」，提示词与审校要点相应改写。

**自动循环**（阶段 22 起）：用 `CronCreate` 每 3 小时唤醒一次，读 `PARALLEL-RUNBOOK.md`
自行判断该 resume、该落盘还是该发下一轮。两条实测出来的经验：

1. **额度窗口的起点会随使用漂移**，不是固定的 5 小时格点（实测重置时刻 03:30 → 14:00 → 19:05）。
   所以循环周期要短于窗口（3 小时），靠「空转一次」兜住漂移，而不是去猜下一次重置在几点。
2. **额度总是在审校阶段用完** —— 译者先跑、吃掉大半窗口，审校排在后面。
   于是「batch 齐了」极易被误读成「这批做完了」。第 4 批就是这个形态：
   11 个译者全完成、11 个审校挂了 9 个。已把「batch 齐但审校没跑成 → 也要 resume」
   写进手册第 1 节的分支表。

> ⚠️ `CronCreate` 的任务是**会话级**的，Claude 一退出就没了，不写盘。

**待裁决的一批 Ember/Crucible 跨包分裂**（第 4 批译者报上来，规则已备好
`glossary/unify_rules.2026-08-07.json`，待第 4 批落盘后执行）：

| 词 | 分裂 | 裁决 |
|---|---|---|
| `Kinesis` | crucible 念力 53 / ember 念动力 58，两边各自内部一致 | **念力**（第 8 节既有决议）+ 改 crucible lang |
| `Gesture: Aspect` | 条目名化相 18，lang 却写「相位」 | **化相** + 改 lang |
| `Glyphweaving` | 同一张角色卡上「符纹编织新手」与「符文编织熟练工」并列 | **符文编织**（UI 标签） |
| `Willpower` | lang 意志力 vs 正文意志防御 8 处 | **意志力防御**（对齐坚韧防御构词） |
| `Rune: Earth` 里的 `Storm` | ember 9 处误译作「闪电」 | **风暴** —— 这条是错译，不是风格 |
| `Presence` | 能力值义「存在」vs 气场 | **不动** —— 「存在」是极常用词、「威严气场」是既有决议，机械替换必然误伤 |

---

### 2026-08-07 · 阶段 23：第 4 批落盘（89%）+ 第 5 批（收尾 + 补缺块）

**第 4 批落盘**（actors 桶 11 单元 / 3106 条 / 33 万英文字符）：

| 项 | 之前 | 之后 |
|---|---|---|
| 战役包覆盖 | 14287 / 19605 | **17393 / 19605 = 89%** |
| 待译字符 | 431,900 | **101,106**（−330,794） |
| LINK / BLOCK / INLINE / TRUNCATED | 682 / 577 / 263 / 68 | 682 / 577 / 263 / 68（**零净增**） |

跨块核对改了 **341 处**。方法上有个值得记的判断：`.name` 要按**「路径形状」归组**再比，
而不是只按英文归组 —— `items.X.name`（条目名）、`items.X.actions.<id>.name`（动作名）、
`effects[].name`（效果名）在库里是三套不同惯例，混在一起比会得出错误结论。
查出 91 处同名异译 + 8 处术语硬伤，其中 **`科拉施法屈折` 应为 `科兰施法屈折`**
（英文是 `Coran`，不是 `Cora` 爆裂之月）—— **库里现存那份译文也是错的**，属继承来的缺陷。

**六组 Ember/Crucible 跨包分裂已裁决并执行**（`unify_rules.2026-08-07.json`）。
裁决前先查了 crucible 的 `lang/cn.json`（玩家界面真正显示的标签）与两个包的条目名计数：

| 词 | 证据 | 裁决 |
|---|---|---|
| `Kinesis` | crucible 包 53 处/14 条目名「念力」，ember 包 58 处/11 条目名「念动力」，两边各自内部一致 | **念力**（第 8 节既有决议）+ 改 lang |
| `Gesture: Aspect` | 条目名 18 个「化相」，lang 却写「相位」 | **化相** + 改 lang |
| `Glyphweaving` | UI 标签「符文编织」，条目名却有「符纹编织新手」 | **符文编织** |
| `Willpower` | lang「意志力」vs 正文「意志防御」 | **意志力防御**（对齐坚韧防御构词） |
| `Rune: Storm` | ember 22 处误译作「闪电」 | **风暴** —— 错译，不是风格 |
| `Presence` | 「存在」1371 处（极常用词），「威严气场」是既有决议 | **不动**，机械替换必然误伤 |

> 这轮暴露出一件事：**crucible 自己的 lang 与 compendium 条目名不一致**，
> 玩家在法术构筑界面看到的标签和天赋列表里的名字不是同一个词。
> 以后定名要把 `lang/cn.json` 与 `compendium` 的条目名**一起**核对，只看一边会漏。

**第 5 批已落盘**（11 单元）：战役包收尾 3 个（场景注记 1058 / 物品 255 / 各卷零碎 528）+
**第 8c 项补缺块 8 个**（58 条 / 约 27 万字符的缺失内容）。

| 项 | 之前 | 之后 |
|---|---|---|
| 战役包覆盖 | 17393 / 19605 = 89% | **19234 / 19605 = 98%** |
| 常规待译字符 | 101,106 | **48,363** |
| 8c 缺失区块 / 估算字符 | 3738 / 532,047 | **1772 / 259,801** |
| LINK / BLOCK / INLINE / TRUNCATED | 682 / 577 / 263 / 68 | **624 / 519 / 230 / 51** |

**补缺块差点白做 —— 被「已有中文则跳过」闸静默跳过了。**
第 8c 项的目标路径本来就有中文（那正是它的定义），不加 `--force` 落盘时
`apply_translations.py` 会报 `skipped (existing)` 而**不是报错**。
第一次落盘后 8 个 fill 单元一条都没写进去，而覆盖率因为同批另外 3 个单元在涨，
表面上一切正常 —— **是标记漂移一动不动才暴露的**（补缺块若真写进去了，
BLOCK 与 TRUNCATED 必然明显下降）。加 `--force` 重落之后四项漂移一起降了。
→ 已写进手册第 3 节，并立下验收判据：**补缺块落盘后 BLOCK/TRUNCATED 没降就是没写进去**。

补缺块与前四批任务性质不同：每条同时给出今天的英文与现有中文，
agent 要把缺掉的区块**插进**现有译文，而不是从零重译。铁律是**已有中文原样保留**
（那是别人校对过的），审校的第二条检查项就是「有没有被无故重写」。
这批的验收有个天然抓手：**补全前这些条目必然过不了标记闸**（缺的区块里带着 `@UUID` 与标签），
所以「0 拒绝」直接证明区块补齐了。

**场景注记那个单元查出的两件事**（该 agent 已完成）：

1. **上游把 scene notes 的键从英文文本改成了随机 ID**，于是 CN 包里 298 条注记译文
   全部成了死键 —— 又一次「改名把译文埋掉」，与阶段 19/20 的孤儿卷、孤儿页同源。
   本批按新 ID 重新落键，逐字沿用旧译。
2. 世界地图上有一批**把专名当普通词处理**的旧译，建议单独安排一次定名：
   `Break`→破坏、`Catch`→接住、`Hoist`→升起、`Wheel`→轮子、`Sail`→航行、`Crown`→王冠、
   `Mordant`→腐蚀性的、`WINDBARE`→光秃秃的、`Karon Mounts`→卡隆坐骑、
   `The Sword Range`→剑的射程、`CASCAAL GOLDS`→卡斯卡尔金币，
   以及 `KAUSTIC HINTERLANDS`→「KAUSTIC 边域」（英文没译完）。

**一条要改的操作经验**：`resume` 之前必须先看 `batch_status.py`。
前几轮失败的译者都是一个字没写，但第 5 批出现了**写到一半被掐断**的（fill-8 写了 12 条里的 4 条）。
留着的话重跑的 agent 可能在半截文件上接着写。已写进手册第 2 节。

---

### 2026-08-08 · 阶段 24：第 6 批 —— 8c + 8j 合并为「页面重对齐」，两项欠账清零

**范围**：把「补缺块」与「删多余块」合并成一个任务，493 页一次做完；顺带换掉一个把单元规模
低估 5.6 倍的切分方式，并查出两类闸门看不见的缺陷。

**结果**

| 项 | 之前 | 之后 |
|---|---|---|
| 第 8c 项：中文缺块 | 339 条 / 1772 块 / 259,801 字符 | **0 / 0 / 0** |
| 第 8j 项：中文多出内容 | 186 条 / 762 块 / 47,467 字符 | **0 / 0 / 0** |
| LINK / BLOCK / INLINE / TRUNCATED | 624 / 519 / 230 / 51 | **232 / 26 / 112 / 20** |
| class 属性漂移 | 412 条 | **48 条** |
| 战役包覆盖 | 19234 / 19605 | **19253 / 19605** |
| 外来文字 / 错位页候选 | 0 / 0 | 0 / 0 |

512 条（493 页 + 19 条残余）落盘**零拒绝**。BLOCK 降 95% 正是阶段 23 立下的验收判据 ——
补缺块若没真写进去，这个数字不会动。

**为什么把 8c 与 8j 合并**

两张清单有 **32 条路径重叠** —— 上游把那些页改写过（既加块又删块）。分给两个 agent 会让
两份 batch 争同一条 path，落盘时一个静默覆盖另一个。而闸门是多重集**相等**比较，
本来就同时管两个方向，所以「补」和「删」本就是同一个任务：**把页面对齐到今天的英文**。

**换掉整页重写：改用页文件 + 逐块编辑**（`prep_realign.py` / `collect_realign.py` / `diff_realign.py`）

两条实测逼出来的：

1. `prep_8c.py` 按 `est_chars`（缺失内容 26 万）切单元，但交完整替换值的 agent 必须重新产出
   **整页** —— 全量 145 万字符，**低估 5.6 倍**。阶段 23 的 fill-8 写了 12 条里的 4 条被掐断，根因在此。
2. 整页重写会把已校对的译文洗掉，阶段 23 只能靠审校去兜。

新格式里 `<i>.cn.html` 一开始就是现有译文的**逐字节副本**，agent 用 Edit 局部改 ——
没动过的字节保持没动过是**格式保证的**，authored 输出从 145 万降到 26 万。
`collect_realign.py` 装配回 batch 并报 **UNTOUCHED**（还与原文逐字节相同 = 没做）。

**这个格式把「额度耗尽」从事故降成了普通中断。** 前两轮 13 个 agent 全灭、`agents_done` 为 0、
无任何缓存，但磁盘上分别攒下 219 / 422 页。**判断进度只能看磁盘**（collect + 闸门），
不能看 workflow 返回值 —— 返回值是 agent 正常 return 才有的东西，被杀就必然是空的。
共四轮（含 1 次 API 断连）跑完 25 个 agent。

**两类闸门看不见的缺陷**

1. **块补对了但插错位置** —— 闸门比对的是**无序多重集**，位置错了照样 0 拒绝，玩家读到的段落顺序是乱的。
   `tagseq.py`（由 realign-8 的审校 agent 写出，已提升为常驻工具）逐位比对块骨架。
   全库 493 页只有 7 页对不上，且**审校跑完的单元基本为 0**。
2. **class 属性漂移** —— 闸门的签名只取标签名（`TAGNAME` 只捕获 `<(/?)(\w+)`），
   于是 `<ul class="complex-check">` 和裸 `<ul>` 一模一样。而这些 class 是功能性的：
   `ul.complex-check` / `li.advantage` 决定检定结果怎么渲染，`sup.system-swap-inline` 是双轨显示。
   新增 `qa/scan_class_drift.py`。首测 412 条，落盘后降到 **48** —— 89% 与 8c/8j 根因同源，自行消失了。

**三次我自己判错、靠证据纠回来的**

1. 「12 个单元全都过度编辑」—— 是**我的工具**在骗我。`diff_realign.py` 的分块正则比 `measure_8c.py`
   多切了 `section`/`h3`/`ul`，`+` 虚高约 2 倍；插入与修改相邻时被 difflib 并成一个 `replace`，`~` 被撑大。
   12 个单元**整齐划一**地异常，指向的是度量而非译者。跨单元核对 agent 报的
   「realign-4 自述严重失实」踩的是同一个坑（拿 diff 口径比译者的 measure_8c 口径），**不采信**。
2. 「丢了 `section.block gamemaster` = GM 内容泄露给玩家」—— **错**。那 36 条的中文里
   **根本没有那段 GM 内容**（属 8c 缺块），不存在泄露。判据：拿该段的
   `<span class="reference">` 坐标去中文里搜，搜不到就是缺块，不是丢包裹层。
3. 「句子存活率只有 0.70–0.87 = 译文被洗掉」—— **也不成立**。两个合法原因：老中文里带的是
   **dnd5e 式命令**（`[[/check 16 perception]]`），按今天的英文改成 `[[/skillCheck …]]` 后散文一字未动也算「丢失」；
   以及上游把整页换掉了 —— `Lantern Roads/Impromptu Jail` 标着「应补 1 块」，
   老中文写的却是**今天英文里根本不存在的偷听对话**，重译才是对的。
   → **`measure_8c.py` / `measure_stale_extra.py` 只比块数，上游「换内容但块数不变」时它们几乎看不见**；
   这批能抓到全靠闸门比对标记多重集。`prose_survival.py` 只能用来挑候选，不能当判据。

**质量**：12 个单元 8 GOOD / 4 FIXABLE，**critical 0**。三份审校的 `rewrites_reverted` 全为 0，
全批 493 页只还原了 **1 处**真正的重写、挪正 2 处错位、恢复 1 处误删。
跨单元核对裁决并改掉 **55 处**分叉（碎齿暴徒 / 霍伦多尔 / 颜料格布林 / 陶里克 / 万德伦巡逻者 /
`Event Outcome` 事件结果 等），并复查前几轮已统一项 **零违规新增**，引号与破折号新块 0 违规。

**顺带修的**：`apply_translations.py` 末尾 `main()` 加 `if __name__ == '__main__':` 守卫，
好让 prep 复用它的 `split_path()` —— 页名带点的 `Patch 0.2.0` 用朴素 `split('.')` 解析不到，
正是阶段 21 修过的那个 bug，我在新脚本里又踩了一次。CLI 行为不变。

**同日落盘：dnd5e 孪生包 `ember.adventure` 0% → 90%**（`tm/fill_twin.py`）

两部战役是同一冒险的两个系统版本，日志正文逐字节相同。crucible 侧推到 98% 后，
孪生包的精确匹配 TM 覆盖率从 88% 涨到 **98%** —— 这 817 万字符不需要译者，只需要一次查表。

| | 之前 | 之后 |
|---|---|---|
| `ember.adventure` | 0 / 14359 | **12981 / 14359 = 90%** |
| 全库 | 58% | **95%** |
| 待译字符 | 836 万 | **90 万** |

12981 条落盘**零拒绝**。脚本**不直接写 `compendium/cn`**，只出 batch 交给
`apply_translations.py` —— 这 817 万字符照样过三道闸，而不是走一条没人检查的私有写入路径。

**顺序是有讲究的**：先落第 6 批再跑，弃填数从 554 降到 **143**（被弃填的里 66% 正是第 6 批
修好的页），多填了 421 条。反过来做就白少填。

键的设计踩了一次自己记过的坑：最初按「最后一段路径 + 英文」做键，这会把
`items.X.name` / `items.X.actions.<id>.name` / `effects[].name` 三种惯例重新混成一堆 ——
正是阶段 23 警告过的。改用**结构路径**（滤掉实体名，只留 `actors.items.actions.name` 这种骨架）后，
歧义键 697 → **530**。

QA：外来文字 0；新增 12981 条只带来 **10 条**漂移、**1 条** class 漂移。
那 9 条 TRUNCATED **不是新缺陷**，是 crucible 侧自己的译文已陈旧（上游换了内容但块数没变），
TM 忠实地复制了过来 —— 也就是说 **`TRUNCATED` 可以当「内容被换掉」的探针用**，
而 `measure_8c` / `measure_stale_extra` 只比块数，看不见这一类。

---

### 2026-08-09 · 阶段 25：按新政策自行裁决并统一 9 组术语

**范围**：项目所有者定下「不一致由主控自行裁决、不必上报」后的第一轮执行。

**结果**：两个仓库共 **236 处替换 / 221 条目**落盘，QA **零回归**
（覆盖率、标记漂移 232/26/112/29、外来文字 0、class 漂移 49 全部与统一前一致 ——
术语替换只动中文词、不动结构，本就该如此）。

| 组 | 裁决 | 处置 |
|---|---|---|
| `Electricity` 电能 | 电击 | 13（加 `electrical energy` 守卫后） |
| `Electricity` 误译成闪电 | 电击 | 17 + crucible 6 |
| `Concluding the Event` | 事件结束 | 146 + 7 |
| `Event Outcome` | 事件结果 | 2 |
| `Marlstone Manor` | 马尔斯通庄园 | 4 |
| `Fernis Ossa` | 费尔尼斯 | 16 |
| `Young Cheliceraeth` | 幼年螯蛛艾斯 | 8 |
| `Horrendor` | 霍伦多尔 | 8 |
| `Yakoshta` | 雅科什塔 | 19 |
| 四个 `Vista:` 场景 name | 见第 8 节 | 8（两个包各 4） |

**评审这一步救回两次误伤**（都是 `--review` 抽查发现的，不是事后补救）：

1. `原始电能的混沌之力` 差点被改成「原始电击」—— 那句翻的是英文 `raw electrical energy`
   这个散文短语，而同一条里的伤害类型本来就已经是「电击」。加 `unless` 守卫后
   ember 少改 5 处、crucible 从 7 处降到 0。
2. `Concluding the Event` 与 `Event Outcome` 在同一条目里大量共现，
   不互加 `unless` 会把对方的译名改坏。

**修掉 `unify_terms.py` 的两个缺陷**

1. **`readaloud="…"` 里的专名会让整批统一自我中止**。`MARKUP` 把整个
   `@Embed[Actor.x readaloud="……"]` 当一个标记，于是「费尼斯·奥萨」出现在旁白里时，
   替换后签名对不上，脚本报「标记被改动了，中止」。
   阶段 22 已裁定那段旁白**要翻译**，并同源修了 `apply_translations.markup_signature`
   与 `scan_markup_drift` 的 LINK 判据 —— **本文件是第三处，当时漏了**。已补 `QUOTED_PARAM` 抹平。
2. **JSON 里的 `\b` 是退格符，不是正则单词边界**。规则写成 `"\bYakoshta\b"`（少一个反斜杠）时
   正则变成 `'\x08Yakoshta\x08'`，一条也匹配不上，而脚本**只安静地报「0 处」**——
   看起来就像「本来就没有要改的」。已加载入时校验，直接报错拦下并说明原因。

**15 条被既有漂移挡住**：这些条目的中文标记与英文本来就对不上（`[[/skill …]]` vs `[[/check …]]`、
`<sup>/<sub>` 计数、`@CriticalSuccess[12]` 等），与本次替换无关 ——
**拿未修改的现值过闸，同样 15 条全拒**，据此确认是存量缺陷。
残留的 事件结尾 6 / 费尼斯 7 / 惊惧者 4 全部落在这 15 条里，数字自洽。
它们属 LINK 232 / INLINE 112 的一部分，修完漂移后重跑本轮规则即可收尾。

---

### 2026-08-09 · 阶段 26：第 7 批「标记签名修复」+ 孪生包连带释放 71 万字符

**范围**：修掉「中文携带的标记与今天的英文对不上」这一整类欠账，并兑现它对孪生包的连带收益。

**这类欠账为什么存在**：闸门是**写入时**校验的，所以闸门存在之前进库的、
或者写的时候英文还是另一副样子的内容，会永久违反它而**不被任何检查发现**。
`measure_8c` / `measure_stale_extra` 只比 `<p>`/`<li>` 块数，坏在块**内部**时它们全看不见。

**结果**

| 项 | 之前 | 之后 |
|---|---|---|
| 签名与英文失配（ember 战役包） | 357 条 / 79.1 万字符 | **1**（上游英文自己缺右方括号） |
| LINK / BLOCK / INLINE / TRUNCATED | 232 / 26 / 112 / 29 | **1 / 0 / 2 / 23** |
| class 漂移 | 49 | **3** |
| markup BROKEN | 14 | **13** |
| `ember.adventure`（孪生包） | 12981 / 14359 = 90% | **13123 / 14359 = 91%** |
| 孪生包待译字符 | 85.7 万 | **14.8 万** |
| 全库待译字符 | 90.4 万 | **19.4 万** |

356 条落盘零拒绝。**class 漂移从 49 掉到 3，印证了它与签名失配本就是同一个根因** ——
上一阶段把它当独立项列的，实际是同一批条目的另一个侧面。

**最常见的一种是双系统分支被压成单支**：
`<sup class="system-swap-inline"><sub data-system="dnd5e">…</sub><sub data-system="crucible">…</sub></sup>`
中文只留了一支 —— **另一个系统的读者什么都看不到**。

**顺序收益（与第 6 批同构）**：修完立刻重跑 `fill_twin.py`，弃填 **143 → 1**，
142 条 / 约 71.3 万字符自动填上。**那 71 万字符从来不是孪生包的翻译债**，
而是 crucible 侧译文与自己英文的标记失配 —— 先修 crucible 侧，孪生包自己就好了。

**质量**：10 GOOD / 2 FIXABLE，critical 0。`rewrites_reverted` 与 `branch_swaps_fixed`
**均为 0** —— 后者（dnd5e 的 `<sub>` 里写了 crucible 用语）是本批闸门看不见、
专门交给审校查的那条。跨单元核对改 157 处，其中含 3 组 **id ↔ 标签错配**
（中文语序调换后 id 没跟着换，签名相等、闸门看不见），以及 11 处「事件结束」、
14 处「雅科什塔」等**已统一项的既有违反**。

**又一个闸门盲区：`&Reference[...]` 从来没进过签名**

`MARKUP` 只认 `@` 开头，于是 `&Reference[Restrained]` 被译成 `&Reference[受拘束]` 时
闸门照样报 0 拒绝，而 enricher 查不到这个键 —— 玩家看到裸文本而不是规则链接。
`scan_markup_targets.py` 同理也漏了。由第 7 批跨单元核对 agent 查出，复核属实。

全库 77 条：位置对齐修好 69 条（英文与中文的 `&Reference` 数量相同时，第 i 个中文键取第 i 个英文键），
数量不等的 8 条留给人读。顺带纠正一条**不是中文、而是抄错成前一个引用**的键
（`Grappled` 应为 `Incapacitated`）。已把 `&Reference` 加进 `markup_signature`，
以后再犯会被当场拒绝。

**crucible-cn 侧也有同类欠账 78 条**（旧判据下就有，与 `&Reference` 无关，此前从未测过）。
差异集中在 `<strong>` 44 与 `<p>` 34，两个方向都有 —— 极端例是
`Character Mechanics.pages.Skills` 的中文有 108 个 `<p>` 对英文 26、还多出一整个 `<table>`，
**玩家会读到一张上游早已删除的技能表**。该包标着「已完成、待发 0.9.0」，
所以这是发版前必须收的尾。已另开批次处理。

---

### 2026-08-09 · 阶段 27：crucible-cn 侧签名失配清零 + 一批包级收尾

**范围**：把第 7 批的同一判据施加到 crucible 仓库（此前从未按签名判据测过），并处理审校上报的包级问题。

| 项 | 之前 | 之后 |
|---|---|---|
| crucible 签名失配 | 78 条 | **0** |
| crucible LINK / BLOCK / INLINE / TRUNCATED | — | **全 0** |
| crucible class 漂移 | — | **0** |

78 条落盘零拒绝，9 单元全部 GOOD/FIXABLE、critical 0，
`rewrites_reverted` 与 `wrong_deletions_fixed` **均为 0**（审校第一条检查项就是「删对了吗」——
这批删除多，**误删比不删更糟**）。

**顺带修好的规则错误**（旧译停在废弃规则上，玩家会照错的打）：
`Subtle Extrication` 旧译「不获得 2 个恩惠」而今日英文是 `+1 Bane`（**方向与数值都反**）；
`Executioner's Strike` 旧译无条件致命+流血，今日以「目标生命值低于一半」为前提；
`Strong Grip` 旧译「空出一只手」而今日是「施放需要双手的法术」（**语义相反**）且整段缺失；
`Rallying Cry` 旧译要掷一个今天已不存在的威吓检定。

**包级收尾**：删 2 个死条目（`Lightning Potency`/`Spellcraft`，上游改名 Storm 后 babele 永远匹配不到）；
3 条 `Spellcraft` 名漏「施法」（依据同条目 `adjective`，`Arrow` 的 adjective 本就是「箭形」，
name 却写「箭头」）；「合集包 包」双重翻译 9 处；`资料库`→`合集包` 11 处；
合集 `label` 两处（`祖裔`→`血统`、`adversary-equipment` 整个没译）；
`池塘地表`→`池塘水面`（地表 = 陆地表面）。

**两条操作教训**

1. **替换有先后依赖**：先做 `资料库→合集包`、后做「合集包 包」清理，会**新造出** 3 处
   「合集包包」。我第一遍就这么错了，是复验步骤抓回来的 —— 机械替换后必须复验，
   不能只看「应改的都改了」。
2. **跨仓库的术语统一要两个仓库都跑**：阶段 20 的 `祖裔→血统` 只跑了 ember，
   crucible 留下 10 处，直到本批审校在正文里读到「自定义祖裔」才暴露。

**修掉自己引入的工具缺陷**：`prep_sigfix.py` 写的 index 键是 `missing`/`surplus`，
而 `diff_realign.py` / `prose_survival.py` 硬编码读 `missing_blocks`/`extra_blocks` ——
两者在 sigfix 单元上直接 KeyError 崩溃（crucible 那批的审校 agent 是自己在 scratchpad
打补丁才跑通的）。已改成兼容两套键名。

**闸门仍有一个已知盲区**（审校证实，本轮未修）：`<strong>` **数量相等但加粗压在错词上**
—— 签名是多重集，看不见落点；`tagseq.py` 比的是块骨架，也看不见行内标记落点。
建议补一个「行内 `<strong>` 落点」检查：把中英各自被 `<strong>` 包裹的词按序取出比对，
或至少标出「中文有 `</strong><strong>` 相邻而英文没有」的条目。

---

