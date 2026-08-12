# Ember / Crucible 汉化项目 · 主文档

> 这是本项目的**唯一长期入口**。新会话请先读第 1 节，再按需跳读。
> 阶段日志（第 6 节）只追加、不重写，用来做长期校对与断点续做。

---

## 1. 快速跟进（新会话必读）

**翻译、术语裁决与全部自动检查已清零（2026-08-12）。剩下的只有冒烟验证。**

> ⚑ **读缺陷表时的硬规矩**：标 ✅ 的项**必须拿数据复验**，不能照抄上一版状态。
> 本轮复查就抓到我把 E / J 两项误标为已完成（实际 disputes 积压 20 条、地图针脚 14 个仍是普通词）。
> 而且术语类结论**一律要带英文闸的计数**：裸词频既会漏也会误伤 ——
> 库内「电能」25 处、「阶位」160 处看着像 Electricity / Tier 的残留，
> 加上英文闸才看出它们译的是 `electrical energy` 与 `Rank`，与那两个术语无关。

| | 英文键 / 中文缺口 | lang 键（Foundry 实际查得到） | 标记五项 | 功能性 class | 丢数字 | 外来文字 | 死键 |
|---|---|---|---|---|---|---|---|
| crucible-cn | 4607 / **0** | 1842 / **1842** | **全 0** | **0** | **0** | 0 | **0** |
| ember_cn | 34865 / **0** | 486 / **486** | **全 0** | **0** | **0** | 0 | **0** |

「标记五项」＝ LINK / BLOCK / INLINE / PLACEHOLDER / TRUNCATED。
ember 的运行时补丁（硬编码字符串 + 日历月名 + 中文字体回退）已就位。

> ⚠ **上一版（0.9.0 / 1.1.0）里 ember 的 lang 有 77% 是死的。**
> `lang/cn.json` 顶层写成了 `"EMBER.CALENDAR": { "WORLD": "世界地图" }` 这种
> 「键带点、值却是嵌套对象」的混合形态。Foundry 的 `getProperty` 先试整键
> `EMBER.CALENDAR.WORLD`（不存在），再按点下探 `translations["EMBER"]`（也不存在，
> 只有 `"EMBER.CALENDAR"`）—— **两条路都断**，回落英文。486 个键里只有 114 个真正生效，
> 项目所有者实测报上来的日历 tooltip（cosmos/world/region map、rewind time、
> party sheet、codex）全在死掉的那 372 个里。crucible 侧同病但只有 46 个。
> 已全部拍平；`lang_gap.py` 加了 `UNREACHABLE` 一列复刻 Foundry 的查找语义，不会再静默。
>
> **教训**：校验必须复刻被验证系统的查找语义。当时那个覆盖率脚本自己会递归展开嵌套，
> 于是一路报「486 键缺口 0」—— **工具替缺陷打了掩护**。

**下一步**（按顺序）：

1. **全局验证** —— 第 5.4 节那一整套，每项都应为 0。**别只跑前五项**：第 6 项
   （英文有、中文整条不存在）是所有其它检查都覆盖不到的方向。
2. **冒烟验证** —— 清单在第 7 节末尾。唯一无法靠脚本证实的环节，至今没做过。
   重点看日历那排 tooltip（1.1.1 修的就是它）。
3. ~~发下一版~~ —— `0.9.2` / `v1.1.2` 已于 2026-08-12 发出。

> **发版状态**
>
> | 版本 | 日期 | 说明 |
> |---|---|---|
> | `crucible-cn 0.9.2` / `ember_cn_unofficial 1.1.2` | 2026-08-12 | 20 条积压术语分歧 + 14 个世界地图针脚 + 两处中文与英文不符（详见第 7 节 E / J 项）。发版前跑完 5.4 全套（含反方向缺口检查）均为 0；下载回包核对：ember 7.23 MB / crucible 0.36 MB，`compendium/en` 与嵌套旧副本均未混入，`download` 指向本 tag，包内抽查译文正确 |
> | `crucible-cn 0.9.1` / `ember_cn_unofficial 1.1.1` | 2026-08-12 | 缺陷表清零后发。Actions 自动建 release，manifest 与 zip 实测 HTTP 200；下载回包核对：ember 7.23 MB / crucible 0.36 MB，`compendium/en` 与嵌套旧副本均未混入，包内 `lang/cn.json` 抽查键值正确 |
> | `crucible-cn 0.9.0` / `ember_cn_unofficial 1.1.0` | 2026-08-10 | 首版，应项目所有者要求提前发 |
>
> 发版怎么做见第 5.5 节。发版前**必做**：`flatten_lang.py`（不加 `--write`）确认两个仓库
> `拍平前 == 拍平后 == 英文键数` 三者相等 —— 这一步就是为了挡住 1.1.0 那次 77% lang 失效。

**怎么干活**：一轮 10–12 个并行单元、约 30 万英文字符，译者自检闸门 + 对抗式审校 +
跨单元术语核对。**操作手册是 `PARALLEL-RUNBOOK.md`**；做法与硬约束的来龙去脉见 `5-其他内容/STAGE-LOG.md` 阶段 20。
八轮实测约 9000 条 / 190 万英文字符落盘**零拒绝**，标记漂移全程只降不升
（LINK 689→0 / BLOCK 584→0 / INLINE 265→0）。

> **别把 BLOCK / INLINE 漂移一概当观感问题。** 阶段 24 查出 `class` 属性漂移
> （`section.block gamemaster` / `ul.complex-check` / `sup.system-swap-inline`）是**功能性**的，
> 而闸门的签名只取标签名、看不见 class —— 用 `qa/scan_class_drift.py` 单独扫。

### ⚑ 优先级与 dnd5e 来源的作用域（2026-08-09 项目所有者定）

**Ember 是世界包，同时支持 Crucible 与 dnd5e 两套规则。**

- **主线是 `crucible` 系统 + Ember 的 crucible 侧**（`2-Crucible汉化插件` 与
  `ember.crucible-adventure`）。这是项目真正要交付的东西，质量与进度都以它为准。
- **dnd5e 侧（`ember.adventure`）是附带项** —— 顺带一起翻，但**不得为它牺牲主线**。

由此得出一条硬性作用域：
**`dnd-simplified-chinese-babele-patch` 之类的 dnd5e 中文来源，只能作用于 dnd5e 侧。**
它对 `crucible.*` 与 `ember.crucible-adventure` **没有任何权威**。
这不是保守，是规则体系不同：拿模块比对我们已译的 name，752 条不一致里有 **542 条**落在
Crucible 侧 —— 那些英文只是碰巧同名（`Dagger`/`Longbow`），条目却是 Crucible 自己的。
整包照搬会把主线污染掉。`tm/fill_twin_names.py` 因此只填 `ember.adventure` 的**空槽**，
从不覆盖既有译文。

### ⚑ 术语与前后不一致：由主控自行裁决，不要来问（2026-08-09 项目所有者定）

**任何上下文 / 前后译文不一致的情况，反复核对译文、词义、上下文之后自行统一即可，不必上报。**
这条覆盖此前「不静默择一、列进 disputes 待裁决」的做法 —— 那条现在只适用于**证据真的不足**时。

裁决时仍走既定的依据阶梯（强 → 弱）：
**同名条目/物品的 `name` 字段 > 同卷已译页 > 全库多数写法 > `glossary_ec.json` > `BRIEF.md` 的表**。
另外三条实测出来的注意事项：

- **先查英文再判中文**。中文写法不同不等于错 —— 英文本来就不同的场合很多。
  例：库里 167 处「闪电」，其中 123 处英文确实是 `Lightning`（忠实），44 处是「闪电般迅捷」这类比喻，
  真正错的只有**英文写 `Electricity` 却译成「闪电」的 23 处**。不做这一步会误伤大片。
- **改动面小的那边优先**（阶段 20 的 `Inkaro`：改 4 个物品条目名而不是 126 处正文）。
- **`name` 字段与正文冲突时，多数情况该改 `name`**，因为正文改动面通常大得多；
  但要连 `lang/cn.json` 一起看 —— 阶段 23 的教训是 crucible 自己的 lang 与 compendium 条目名就不一致。

裁决后**必须**：① 写进第 8 节决议记录（附证据与计数）；② 用 `qa/unify_terms.py` 执行
（它只在**英文原文确实出现该术语**时才改，正是上面第一条的机械保障）；③ 复跑 QA 全套。

**翻译时必须遵守的既定译名**（避免和已完成的 11 个包冲突）：
`Kinesis`念力 · `Warden`守林者 · `Guardian`守护者 · `Swarm`(archetype)群集 · `Tier`阶 ·
`Electricity`**电击**（状态 `Shocked` 作**感电**；`Lightning` 才是闪电，别混）·
`Bludgeoning`钝击 · `Fire`火焰 · `Corruption`腐化 · `Fortitude`**强韧** ·
`Toughness`坚韧 · `Wisdom`感知 · `Presence`存在 · `Willpower`**意志** · `Health`生命值 ·
`Boon`**恩惠骰**（对 `Bane`祸骰）· `Accurate`精准 · `Stride`步幅 · `Arrow`箭矢 ·
`inflection`屈折 · `gesture`手势 · `rune`符文 · `spellcraft`施法 · `essence`精华 · `compose spells`构筑法术。
完整表见 `5-其他内容/glossary/glossary_ec.json`。

**要做全局验证就跑第 5.4 节那一整套**（每一项都应为 0）。翻译批次的做法在
`PARALLEL-RUNBOOK.md`；回写一律走 `qa/apply_translations.py`（三道闸：英文源漂移 /
无中文 / 标记破损），**任何情况下都不要直接改 `compendium/cn`**。

一条容易踩的：追平类批次改的是**已有中文**的条目，落库必须带 `--force`，
否则是 `applied 0 / skipped(existing) N` 的静默空跑。

改过 `3-常用脚本/extract/mappings.mjs` 或 `release/runtime-converters.js` 之后，必须重跑：
```powershell
node "$P\3-常用脚本\release\generate_runtime.mjs"      # 重新生成两个仓库的 babele-mappings.js
python "$P\4-临时脚本\2026-08-06\crosscheck_vs_crucible_fr.py"   # 交叉校验抽取器
```

**关键路径**：
```
抽英文基准 → 算 diff → 管线改造(声明式 mapping) → TM/回源预填 → 分批翻译 → 三轮校准 → 发版
```

**三个必须知道的事实**（不知道会做错方向）：

1. **babele 2.9.1 会自动回源翻译内嵌文档**。Adventure 里 actor 内嵌的 item，只要带 `_stats.compendiumSource` / `flags.core.sourceId`，babele 就会去它原属合集的译文里取。ember 战役里 82.4% 的内嵌物品字符有来源包 —— 只要用**默认递归 `document` mapping**（而不是手写遍历转换器），这部分自动就翻好了。**不要退回手写转换器**，那会白白丢掉约 60 万字符的免费收益。
2. **英文基准必须存档**。每次系统/模块升级，只有拿旧版英文才能算出"哪些英文原文改了"（drift）。基准存两处：各插件仓库 `compendium/en/`（当前版本，进 git）＋ `5-其他内容/english-baseline/<包>-<版本>/`（历史快照）。
3. **术语表是 `5-其他内容/glossary/glossary_ec.json`**，基底来自 `glossary_crucible_merged.json`（4602 条已裁决译名）。**不要另起炉灶**，也**不要并入 PF2E 主表**（`fvtt\glossary.json`）—— Crucible/Ember 是 Foundry 自有世界观，PF2 的译法会污染（例：`Restrained` 本项目作「受缚」，PF2 作「受制」）。

---

## 2. 项目总结

把 Foundry VTT 的 **Crucible 系统**和 **Ember 战役模块**汉化成简体中文，通过两个 Babele 翻译模块交付。

### 版本矩阵

| 组件 | 类型 | 版本 | 位置 |
|---|---|---|---|
| crucible | 系统 | **0.10.1** | `%LOCALAPPDATA%\FoundryVTT\Data\systems\crucible` |
| ember | 模块（**付费/protected**） | **0.6.0** | `…\Data\modules\ember` |
| babele | 模块（翻译框架） | **2.9.1** | `…\Data\modules\babele` |
| crucible-cn | 汉化模块（本项目） | **0.9.2**（2026-08-12 发布） | `2-Crucible汉化插件\` |
| ember_cn_unofficial | 汉化模块（本项目） | **1.1.2**（2026-08-12 发布） | `1-Ember汉化插件\` |

两个汉化仓库：
- https://github.com/takaqiao/crucible-cn
- https://github.com/takaqiao/ember_cn_unofficial

### 汉化的两条通道

- **`lang/cn.json`** —— 界面字符串（Foundry 原生 i18n），走 module.json 的 `languages` 字段
- **Babele `compendium/cn/*.json`** —— 合集内容（天赋/装备/日志/战役正文），走 `babele.register()`

crucible 侧两条都用；ember 侧两条也都用。

---

## 3. 须知（踩过的坑与硬约束）

### 3.1 babele 2.9.1 的三个真实故障

1. **`crucible-cn/babele-register.js` 有真 bug**
   `adventure_items_converter` 内部调 `game.babele.converters.actions_converter(items, translations)`。
   2.9.1 的 `.converters` getter 返回的是 `ConverterRegistry.snapshot()` —— 值是 **`FunctionalConverter` 对象**，不是函数。
   → 冒险模组里带 `actions` 的内嵌物品会抛 `TypeError: not a function`。

2. **`crucible-cn` 的 `SUPPORTED_PACKS` / `DEFAULT_MAPPINGS.ActiveEffect` 补丁已是死代码**
   2.9.1 的默认 mapping 原生支持 `ActiveEffect`（还带 `changes` 的 `structured` 转换器）。该补丁块可删。

3. **`ember_cn/register.js` 的 `_tableResults` 补丁全是死代码**
   2.9.1 已无 `_tableResults` 转换器，改由 `document` + `documentType: "TableResult"` 处理，identity 是
   `_identity: {export: ["range","_id"], match: ["_id","range"]}`。
   这正是 table results 至今 0% 的原因。

### 3.2 babele 2.9.1 的关键能力（务必用上）

- **注册钩子是 `babele.init`**，不是 Foundry 的 `init`。babele 在自己的 `init` 里 `game.babele = …` 然后同步 `Hooks.callAll('babele.init')`。现在两个模块都挂在 Foundry `init` 上，靠模块加载顺序侥幸能跑 —— 要迁走。
- **`window.Babele` 仍然存在**（`babele.js` 末尾 `window.Babele = BabeleFacade`），旧的 `typeof Babele !== 'undefined'` 守卫不会失效。
- **`registerMapping(mapping)`** —— 声明式追加/覆盖全局文档 mapping 层，比手写转换器好得多。
- **`_variants` + `_when`** —— 按字段值分支 mapping，正好对应 ember 的 13 种 page 子类型：
  ```json
  "_variants": [{ "_when": {"path": "type", "equals": "ember.location"}, "overview": "system.overview" }]
  ```
- **递归 `document` 转换器 + 源包回退**（见第 1 节事实 1）。`fallbackPolicy` 可选
  `source-first`（默认）/ `owner-package-before-generic` / `owner-package-first`。
- **`_packs-folders.json`** —— 能翻合集**文件夹名**。crucible-cn 完全没做，Crucible-FR 做了。
- **多源优先级** —— 同一个 collection 有多份译文文件时可设优先级（`setSourcePriority`）。
- **诊断接口** —— `game.babele.inspectMapping(type)`、`await game.babele.sourceDiagnostics()`、`cacheDiagnostics()`。验收时用。
- **损坏的译文文件不会拖垮整体**：`TranslationLoader.#loadJsonFile` 有 try/catch，只 `console.error` 并跳过该 collection。

### 3.3 ember 的自定义 JournalEntryPage 子类型

ember 的正文**不在 `text.content` 里**。13 种子类型及其正文字段（实测自 ember 0.6.0）：

| 子类型 | 页数 | 正文字段 |
|---|---|---|
| `ember.location` | 115 | `system.overview`、`system.exposition`、`system.terrain` |
| `ember.biome` | 27 | `system.overview`、`system.exposition`、`system.terrain` |
| `ember.lore` | 167 | `system.content.overview`、`system.content.gamemaster`、`system.pronunciation` |
| `ember.deity` | 74 | `system.content.overview`、`system.content.gamemaster`、`system.subtitle`、`system.pronunciation` |
| `ember.questEvent` | 229 | `system.overview`、`system.exposition`、`system.summary`、`system.outcomes[].label\|summary` |
| `ember.standaloneEvent` | 18 | 同上 |
| `ember.culture` | 28 | `system.content.overview`、`system.banner.caption`、`system.pronunciation` |
| `ember.ancestry` | 18 | `system.content.overview`、`system.height`、`system.lifespan`、`system.origin`、`system.pronunciation` |
| `ember.cosmos` | 11 | `system.content.overview`、`system.content.gamemaster`、`system.subtitle` |
| `ember.organization` | 21 | `system.content.overview`、`system.content.gamemaster`、`system.pronunciation` |
| `ember.characterClass` | 13 | `system.content.overview` |
| `ember.quest` | 20 | `system.overview` |
| `ember.questFlowchart` | 20 | 仅 `name` |
| `text`（原生） | 727 | `text.content` |

实测 `system.overview` 与 `text.content` **741 处全部不同**，不是冗余镜像 —— 两者都要翻。

### 3.4 译名规范

- **专有名词用双语并列**：`申特月神殿 Shent Moon Temple`、`古冢 Barrows`。
  这是既有 v1.0.15 已在用的风格，保持一致。
- **术语优先级**：本项目 glossary > 从既有译文提取的 TM > PF2E 主表（仅作建议，命中须人工确认）。
- **HTML/富文本标记必须原样保留**：`@UUID[...]`、`@Check[...]`、`<section class="block gamemaster">`、
  `<span class="reference">⬢ s.3204.2870</span>`、`<figure>/<figcaption>` 等。
  这些是 Foundry 的功能性标记，改坏了会导致链接失效或样式崩。

### 3.5 其他约束

- **ember 是付费模块**（`module.json` 里 `"protected": true`）。`ember_cn_unofficial` 公开仓库放着整部战役的完整中文正文。
  Padhiver 的 Crucible-FR 有 Ember 转换器但**没有公开发布 Ember 译文**，大概率就是这个原因。是否改私有/只发 diff 由项目所有者决定，此处仅记录。
- **`ember_cn` v1.0.15 发布包里 `ember.crucible-adventure-en.json` 是损坏 JSON**（第 44 行多一个 `}`）。
  它躺在 `compendium/cn/` 里，每次开世界都会被 fetch + 解析失败（11.4 MB 白流量）。要移出到 `compendium/en/` 并重新生成。
- **`ember_cn` 的译文里有 1447 个页面条目带垃圾字段 `path: {}`** —— 是 mapping 保留字 `path` 被当成翻译字段写进去了，要清掉。
- Windows 目录名不能含 `/`，所以项目目录用连字符：`Ember-Crucible Translation Project`。

---

## 4. 目录与脚本索引

```
Ember-Crucible Translation Project\
├── PROJECT.md                  ← 本文件
├── 1-Ember汉化插件\            ← ember_cn_unofficial 的 git clone（可直接发版）
├── 2-Crucible汉化插件\         ← crucible-cn 的 git clone（可直接发版）
├── 3-常用脚本\
│   ├── extract\   从 LevelDB packs 抽英文原版
│   ├── tm\        翻译记忆库构建 / 预填 / 去重
│   ├── qa\        术语校验 / markup 完整性 / 残留英文 / 覆盖率
│   └── release\   打包 zip / 改 module.json / 发 GitHub release
├── 4-临时脚本\<日期>\          ← 一次性探针，按日期归档，不删（可复现结论）
└── 5-其他内容\
    ├── glossary\          glossary_ec.json + 来源谱系 + 冲突裁决
    ├── english-baseline\  历史版本英文快照（跨版本算 drift）
    ├── reports\           每阶段 diff / QA 报告
    └── reference\         babele 2.9.1 API 要点、Crucible-FR 参考实现摘录
```

### 脚本索引

`$P` = 项目根目录；`<repo>` = `1-Ember汉化插件` 或 `2-Crucible汉化插件`。

| 脚本 | 干什么 | 怎么调 |
|---|---|---|
| `extract/mappings.mjs` | **mapping 数据的唯一真源**。抽取器解释它，运行时文件由它生成。改了它必须重跑下面两条 | 不直接执行 |
| `extract/extract_en.mjs` | 解释 mapping，从 LevelDB packs 抽英文基准 | `node extract_en.mjs --package <foundry包目录> --out <输出目录> [--target crucible\|ember] [--pack <名>]` |
| `release/runtime-converters.js` | 三个自定义转换器的**翻译方向**实现（抽取方向在 `extract_en.mjs` 里）。两边必须同一次提交一起改 | 不直接执行 |
| `release/generate_runtime.mjs` | 由上面两个文件**生成**两个仓库的 `babele-mappings.js` | `node generate_runtime.mjs` |
| `tm/build_glossary.py` | 合成 `glossary_ec.json`，并产出待裁决 / 待补清单 | `python build_glossary.py` |
| `qa/validate_translations.py` | **核心验收**：拿英文基准逐路径核对译文，输出覆盖率 + 机读待译清单 | `python validate_translations.py --repo <repo> --out <报告目录>` |
| `qa/lang_gap.py` | `lang/cn.json` 的三方 diff：NEW / DRIFT / UNTRANSLATED / STALE | `python lang_gap.py --repo <repo> --package <foundry包> --out <报告目录> [--sync-baseline]` |
| `qa/apply_lang.py` | lang 批次回写。四道闸：key 不存在 / 占位符 / HTML 标签 / 行内标记；`--clean-stale` 清上游已删的 key | `python apply_lang.py --repo <repo> --package <foundry包> --batch <batch.json> [--clean-stale] [--dry]` |
| `qa/unify_terms.py` | 按规则表统一术语。只在**英文原文确实出现该术语**时才改，支持正则搭配限定 | `python unify_terms.py --repo <repo> [--package <foundry包>] --rules <rules.json> [--review <md>] [--write]` |
| `qa/scan_markup_drift.py` | 扫译文与英文的标记差异：LINK / BLOCK / INLINE / PLACEHOLDER / TRUNCATED | `python scan_markup_drift.py --repo <repo> [--kind LINK,TRUNCATED] [--out <json>]` |
| `qa/scan_markup_targets.py` | 扫**方括号内部**被译成中文的标记（链接/嵌入块会静默失效，覆盖率与漂移检查都看不见）。分 BROKEN / by-design 两类 | `python scan_markup_targets.py --repo <repo> [--repo <另一个>] [--json <out>]` |
| `qa/restore_enrichers.py` | 把被写成裸中文的 `@Condition[...]` 等标记还原回去 | `python restore_enrichers.py --repo <repo> --package <foundry包> --surface-forms <json> [--write]` |
| `qa/resolve_generic_fallback.py` | 从待译数字里**扣掉 babele 会自动解析的部分**。翻译前必跑，否则重复劳动 | `python resolve_generic_fallback.py --repo <repo>` |
| `qa/apply_translations.py` | 批量回写译文。三道闸：英文源漂移 / 无中文 / 标记破损 一律拒 | `python apply_translations.py --repo <repo> --pack <pack.json> --batch <batch.json> [--force] [--dry]` |
| `qa/scan_foreign_script.py` | 扫外来文字污染（西里尔 / 亚美尼亚 / 希伯来 / 泰文等机翻残留） | `python scan_foreign_script.py --repo <repo> [--repo <另一个>] [--fix]` |
| `qa/port_orphans.py` | 上游改名后把孤儿译文移植到新路径；移植不了的**留在原地并列出** | `python port_orphans.py --repo <repo> --rules <rules.json> [--dry]` |
| `qa/migrate_cn_schema.mjs` | 一次性 schema 迁移（已执行完毕，保留备查） | `node migrate_cn_schema.mjs --repo <repo> --package <foundry包> --target <crucible\|ember> [--dry]` |
| `qa/flatten_lang.py` | 把 `lang/cn.json` 拍平成 Foundry 真查得到的扁平点号键，并按英文侧逐键复核。**含 `foundry_lookup()`（复刻 `getProperty`）** | `python flatten_lang.py --repo <repo> --english <英文 lang 文件> [--write]` |
| `qa/scan_content_coverage.py` | 靠「跨语言不变量」（英文正文里的数字）找中文没跟上的条目。**认中文数字与 decade/dozen 这类倍数量词**，否则会逼出「2 个十年」那种坏中文 | `python scan_content_coverage.py --repo <repo> [--out <json>]` |
| `qa/fix_bold_drift.py` | 加粗漂移的机械修复（阶段 20 用过，保留备查） | `python fix_bold_drift.py --repo <repo> [--write]` |
| `tm/fill_twin.py` | crucible-adventure → ember.adventure 单向 TM 填充（两套规则、同一场战役，日志正文逐字节相同） | `python fill_twin.py [--out <batch.json>] [--report <json>]` |
| `qa/prune_dead.py` | 删中文包里英文包已没有的键（babele 永远查不到）。顺带揪出**键名混进中文**的条目 | `python prune_dead.py --repo <repo> [--write]` |
| `qa/propagate_fix.py` | 把某次提交里修好的译文推到**英文逐字相同**的同源副本上。三方 diff 挑不出那些副本（它们的英文没变过） | `python propagate_fix.py --repo <repo> --english <英文包目录> --since <commit> [--write]` |
| `tm/fill_missing.py` | 用全库 TM 补**中文侧整条不存在的键**。这类缺口所有既有扫描都发现不了 | `python fill_missing.py --repo <repo> [--repo <另一个>] --out-dir <批次目录> [--report <json>]` |

`3-常用脚本/parallel/` 下的 20 余个脚本（切单元 / 收批次 / 逐单元核对）**统一由 `PARALLEL-RUNBOOK.md` 说明**，此处不重复。

**批次文件格式**（喂给 `apply_translations.py`）：扁平 `{"<待译清单里的 path>": "<中文>"}`。
待译清单位于 `5-其他内容/reports/<crucible|ember>/todo/*.todo.json`。

**运行前提**：`node` 能从 `C:/Users/Taka/Desktop/fvtt` 解析到 `classic-level`；Python 3.14。
PowerShell 写批次文件会带 BOM，`apply_translations.py` 已按 `utf-8-sig` 读取。

### 外部参考实现

**Padhiver/Crucible-FR** — https://github.com/Padhiver/Crucible-FR
法语社区汉化，已对齐 crucible 0.10.1，值得抄的地方：

- `Hooks.once("babele.init")` + `game.modules.get("babele")?.active` 守卫
- `module.json` 里 babele 依赖直接写 `minimum: 2.9.1`，`compatibility.maximum: 14.999`
- 仓库里同时有 `compendium/en` 和 `compendium/fr`（英文基准进 git —— 与本项目做法一致）
- **`crucible._packs-folders.json`**（翻合集文件夹名，本项目缺）
- 转换器拆成独立 ES 模块 `scripts/converters-crucible.js` / `converters-ember-core.js`
- 他们的 `compendium/en` 可用来**交叉校验本项目抽取器有没有漏字段**
- 他们也做了 Ember 转换器（`emberPages`/`emberTables`/`emberSceneLevels`/`emberTableResults` 等），
  但**没公开发布 Ember 译文**

他们的策略是**混合式**：`compendium/en/mappings.json` 里只覆写 `Adventure.journals/scenes/macros/tables`、
`JournalEntry.pages`、`RollTable.results` 等，而 **`Adventure.actors` 和 `Actor.items` 保持 babele 默认** ——
所以他们同样拿到了源包回退。（早先记录说他们拿不到，是错的，已更正。）

**本项目与之的差别**：他们对 journal pages 用手写 `pages_converter`，本项目改用声明式子类型键
（`JournalEntryPage.ember.location` 等）。两者都能工作；声明式的好处是抽取器可以直接解释同一份数据，
不会出现「转换器读 A 键、抽取器写 B 键」的漂移。另外作者在 `converters-ember-core.js` 文件头自注
「本轮没在游戏里重测过」，所以他们的实现不宜盲信。

GitHub 上除本项目外**没有任何其他语言的 Ember 翻译**。

---

## 5. 标准操作 SOP

### 5.0 ⚑ 升级追平方案（Ember / Crucible 出新版后照这个走）

> 这一节是**发版之后**要落实的长期方案。核心洞察：
> **找「中文没跟上英文」不要靠启发式，靠旧版英文与新版英文的直接 diff。**

此前用过的间接信号各有盲区，而且盲区**重叠**：

| 检查 | 判据 | 盲在哪 |
|---|---|---|
| `validate_translations.py` | 路径上有没有中文 | 内容对不对完全看不见 |
| `measure_8c` / `measure_stale_extra` | `<p>`/`<li>` **块数** | 上游换内容但块数不变 → 沉默 |
| `scan_markup_drift` 的 `TRUNCATED` | 中文 < 英文 **0.22 倍** | 上游改写但长度相当 → 沉默 |
| 标记签名 | 标记多重集 | 只有标记跟着变才响 |

而 `EN_old != EN_new` 是**直接证据**。工具：`qa/scan_en_drift.py`。

**每次上游升级的标准动作**

```powershell
# 1. 先把「当前」英文归档成「旧版」——这一步漏了，下次就没得比
Copy-Item -Recurse "<repo>\compendium\en" "5-其他内容\english-baseline\<包>-<旧版本号>"
# 2. 抽新版英文，覆盖进 compendium/en
node "$P\3-常用脚本\extract\extract_en.mjs" --package <foundry包目录> --out "<repo>\compendium\en"
# 3. 重打 LOCAL-PATCHES.md 里记的上游笔误补丁（重抽会覆盖掉）
# 4. 三方 diff：英文变过、且中文更贴合旧英文的，就是要重译的
python "$P\3-常用脚本\qa\scan_en_drift.py" --repo <repo> --baseline "5-其他内容\english-baseline\<包>-<旧版本号>" --out <报告>
# 5. 按报告里的 items 切单元、发并行批次（沿用 PARALLEL-RUNBOOK 的页文件工装）
# 6. QA 全套 + 冒烟验证
```

**报告分四档**，按该管的先后：

| 档 | 含义 | 怎么处理 |
|---|---|---|
| `stale` | 英文变过 **且中文长度更贴合旧英文** | **优先重译**。这是主产物 |
| `changed`（非 stale） | 英文变过、但中文长度已贴合新英文 | 多半是上一轮已重译过，抽查即可 |
| `gone` | 上游删了、中文还在 | 死文本，清掉（babele 匹配不到，无害但占体积） |
| `new` | 上游新增 | 走正常翻译流程 |

**两个工具抓的不是同一类东西，都要跑**：

| 工具 | 抓什么 | 抓不到什么 |
|---|---|---|
| `scan_en_drift.py` | 英文**变过**、中文停在旧版 | 英文没变、但译文自始就不全 |
| `scan_content_coverage.py` | 英文里的**数字**中文没有 | 不含数字的漏译 |

那个「更贴合旧英文」的判别很关键：本库译文/英文纯文本长度比中位数是 **0.31**，
比较 `中文/旧英文` 与 `中文/新英文` 哪个更接近 0.31 即可。
**首测就靠它把 crucible 的 295 条压到 44 条、ember 的 921 条压到 280 条** ——
否则上千条无从下手。

> ⚠ **归档旧英文是这套方案的命门。** 没有旧基准，本方案退化成启发式。
> `compendium/en/` 只保存当前版本，历史快照必须另存到
> `5-其他内容/english-baseline/<包>-<版本>/`（第 1 节事实 2 就是讲这个）。

### 5.1 系统/模块升级后的例行检测

1. 记录新版本号，更新第 2 节版本矩阵
2. 用 `3-常用脚本/extract` 抽新版英文 → `5-其他内容/english-baseline/<包>-<新版本>/`
3. 拿**旧版**英文基准 + 新版英文 + 现有译文，算三方 diff：
   - `NEW` 新增条目（要翻）
   - `STALE` 上游删除的条目（可清理）
   - `DRIFT` 英文原文改动（要重翻）
   - `PARTIAL` 条目在但字段缺（要补）
4. 报告落 `5-其他内容/reports/`
5. 把新版英文覆盖进各仓库 `compendium/en/`

### 5.2 翻译批次

1. **先 TM/回源预填，再实译** —— 避免重复劳动与前后不一致
2. 按 pack 或按 journal 切批，单批控制在可校对的规模
3. 每批产出后立刻进三轮校准，不要攒着

### 5.3 三轮校准（每批都做）

- **R1 术语一致性** —— 对照 `glossary_ec.json` 强制校验，命中即替换/告警
- **R2 上下文一致性** —— 同一地点/人物/任务在不同页面的措辞对齐；
  `@UUID` / `@Check` / `<section class>` / `<span class="reference">` 标记完整性
- **R3 机械扫描** —— 残留英文、markup 漂移、长度异常（中译文长度通常为英文 0.4–0.7 倍，超出范围要看）

### 5.4 全库验证（发版前必跑，也是新会话做全局核查的入口）

一次跑完，**每一项都应为 0**。`<repo>` 取 `1-Ember汉化插件` / `2-Crucible汉化插件`，
`<pkg>` 取对应的 Foundry 包目录（`…\Data\systems\crucible` / `…\Data\modules\ember`）。

```powershell
$P = "C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
$Q = "$P\3-常用脚本\qa"

# 1. lang —— 必须四项全 0，尤其 UNREACHABLE（有中文但 Foundry 查不到＝键形态错）
python "$Q\lang_gap.py"      --repo <repo> --package <pkg> --out <reportDir>
#    另跑一遍拍平自检：拍平前 == 拍平后 == 英文键数，三者相等才算过
python "$Q\flatten_lang.py"  --repo <repo> --english <pkg>\lang\en.json

# 2. 标记五项 LINK / BLOCK / INLINE / PLACEHOLDER / TRUNCATED
python "$Q\scan_markup_drift.py"   --repo <repo>
# 2b. 方括号内部被译成中文的标记（链接/嵌入块会静默失效）
python "$Q\scan_markup_targets.py" --repo <repo>
# 2c. class 漂移 —— 闸门的签名只取标签名、看不见 class，
#     而 section.block gamemaster / ul.complex-check / sup.system-swap-inline 都是功能性的
python "$Q\scan_class_drift.py"    --repo <repo> --out <reportDir>\class_drift.json

# 3. 内容覆盖（中文有没有丢掉英文里的数字）
python "$Q\scan_content_coverage.py" --repo <repo>
# 4. 外来文字污染（西里尔/亚美尼亚/希伯来/泰文等机翻残留）
python "$Q\scan_foreign_script.py"   --repo <repo>
# 5. 死键（中文有、英文没有 —— babele 永远查不到，纯占体积）
python "$Q\prune_dead.py"            --repo <repo>          # 加 --write 才真删
# 6. 反方向：英文有、中文整条不存在（**上面每一项都覆盖不到这类缺口**）
python "$P\3-常用脚本\tm\fill_missing.py" --repo <repo> --out-dir <批次目录>
```

> ⚠ **第 6 项是 2026-08-12 才补上的盲区。** 第 1–5 项全都以「中文里的某条」为起点，
> 中文里压根没有的条目不在它们的定义域内 —— 所以库里一度报「覆盖率 99%」，
> 而 crucible 的两个预生角色几乎整体没译。**别再只跑前五项就宣布干净。**

> `port_orphans.py` 只搬路径、不改译文内容。上游改名后**译文里的旧名字不会自动更新** ——
> `Rune: Lightning`→`Rune: Storm` 就是这么留下 28 处「闪电」的。改名类 drift 处理完必须回头
> 搜一遍旧名字。

### 5.5 发版

**两个仓库都有 tag 触发的 `.github/workflows/release.yml`，正常情况不需要手工打包。**

1. 先跑 5.4 全套 + 冒烟验证
2. 改 `module.json` 的 `version` 与 `download`
   —— **tag 形态不同**：crucible 不带 `v`（`0.9.2`），ember 带 `v`（`v1.1.2`），
   两边 `module.json` 的 `download` 就是这么写的，workflow 会逐字校验，写错直接失败
3. 更新 `.github/release-body-template.md`（Actions 拿它当 release 正文）
4. 推 main → 打 tag → push tag，Actions 自动打包建 release
5. 发完**下载回包核对**：manifest/zip HTTP 200；`unzip -l` 确认没有 `compendium/en/`
   与嵌套的上一版副本（1.1.0 就因为 `zip -r` 是**追加**而不是重建，把 22 MB 陈旧副本发了出去）
6. 在第 6 节年表补一行

Actions 不可用时的兜底手工流程：`zip -r` 前先 `rm -f module.zip`，排除
`.git/* .github/* release/* compendium/en/* lang/lang_keep_english.json *.zip *.bak`，
然后 `gh release create <tag> module.json module.zip --notes-file .github/release-body-template.md`。

---

## 6. 阶段年表

> **完整记录已归档到 `5-其他内容/STAGE-LOG.md`**（1509 行，阶段 0–27 的原始测量、
> 走过的弯路、被推翻的判断）。这里只留一句话年表，用来定位「哪一步发生在什么时候」。
> 仍然生效的硬约束看第 3 节，仍然生效的裁决看第 8 节 —— 不要回头去日志里翻结论，
> 那里的数字是当时的状态。

| 阶段 | 日期 | 做了什么 |
|---|---|---|
| 0–2 | 08-06 | 基建、通用抽取器与英文基准、术语表构建 |
| 3–4 | 08-06 | babele 2.9.1 管线改造（声明式 `registerMapping`）+ crucible compendium 收官 |
| 5–7 | 08-06 | crucible `lang` 收官、全库术语统一、ember `lang` 收官 |
| 8–10 | 08-06 | ember 四个小包收尾 |
| 11 | 08-06 | Ember 运行时补丁：硬编码字符串 + 中文字体回退 |
| 12–13 | 08-06 | 标记漂移清扫；查出「显示 100% 实则缺整块」那类欠账 |
| 14–19 | 08-06 | 战役正文首批（Ushna Dredging Docks、Arcturel Dives）+ 一整卷被改名埋掉的旧译文 |
| 20–24 | 08-06~08 | **并行翻译管线**（页文件 + Edit 局部改）第 1–6 批；8c/8j 合并为「页面重对齐」并清零 |
| 25 | 08-09 | 按「自行裁决」新政策统一 9 组术语 |
| 26–27 | 08-09 | 标记签名失配两侧清零；孪生包连带释放 71 万字符 |
| — | 08-10 | **首版发布** `crucible-cn 0.9.0` / `ember_cn_unofficial 1.1.0` |
| — | 08-12 | lang 键形态修复、追平批、TM 补齐缺口、死键清理；发 `0.9.1` / `1.1.1` |
| — | 08-12 | 20 条积压术语分歧裁完、世界地图针脚重译、两处中文与英文不符；PROJECT.md 2239→644 行；发 `0.9.2` / `1.1.2` |

## 7. 现状与唯一未做项

**排期表已并入第 6 节年表**（原表 48 行全部 ✅，只是历史）。当前状态见第 1 节。

| # | 事项 | 状态 |
|---|---|---|
| 9 | **真实 Foundry 世界冒烟验证** | ⬜ **唯一还没做的事**。脚本证不了，必须开世界看，清单见本节末 |
| 8f | `Arcturel Tradeway` 28 页 | 🔶 待译归零、结构核对通过，但**逐页通读没做**（内容是从改名前的旧路径移植来的） |

> ⚠️ **不要拿 `validate_translations.py` 的百分比当真实缺口。**
> crucible 显示 97%、ember 显示 99%，但那 436 条待译**全部**由 babele 通用回退
> 从已译包按名字自动取译文。动手前先跑：
> ```powershell
> python "$P\3-常用脚本\qa\resolve_generic_fallback.py" --repo <repo> --also <另一个 repo>
> ```
> 重复翻译只会制造同名异译。**两个包的真实残余都是 0。**

### 待清扫的既有缺陷（2026-08-12 全部清完）

| # | 问题 | 现状 |
|---|---|---|
| A | `lang/cn.json` 的 `Burrow` 含亚美尼亚字符 | ✅ 已修 |
| B | `Toughness` / `Fortitude` 同译「坚韧」 | ✅ **2026-08-12**：`Fortitude` 改 **强韧**，三项防御成为 强韧/反射/意志，各占一词。同时提到两者的 2 页（Defenses、Ability Scores）手工处理 —— Willpower 的公式就是 `(坚韧+存在)/4`，用的是属性，自动替换必然误伤 |
| C | `Electricity` 三种写法 | ✅ **英文闸下已归零**：英文写 `Electricity` 的条目里中文全是「电击」。库内另有 电能 25 / 电力 8，译的是 `electrical energy` 等别的英文，**不是残留**（2026-08-12 复核） |
| D | `Tier` 阶/阶级/阶位；essence 精髓/精华 | ✅ **英文闸下 Tier 只剩「阶」**（另 1 处「阶数」是已裁的名词形）。库内 阶位 160 译的是 `Rank`，不是 Tier。`essence` 的 精髓 8 处**当时确实没做**，2026-08-12 已并入 精华（132:8，同一个 essence，无术语/散文之分） |
| E | 已发布译文双语格式不一致、同名异译 | ✅ **2026-08-12 全部裁完**。一是 `glossary_ec.disputes.json` 里积压的 20 条（该文件已转为裁决记录）：Agrimage→农艺法师、Thornling→荆芽灵、Aberin→阿伯林、House Cevher→杰夫赫尔、Ordain→奥尔丹（225 处「奥丹」）、Hulg'run→赫尔格伦 等；二是本轮新裁的 Boon→恩惠骰、Fortitude→强韧、Willpower→意志、Accurate→精准、Arrow→箭矢、Stride→步幅、maximum Action→最大动作，另修 5 个 lang 标签。详见第 8 节 |
| F | 孤儿译文（babele 匹配不到 key 的死文本） | ✅ **2026-08-12**：`prune_dead.py` 清掉 crucible 74 / ember 1435 条，省 480 KB，有效译文键数前后一字不差。含 4 条**把译名当成了键**的（`items.吞噬思维 Devour Thoughts`，整条对玩家不存在） |
| G | `Rune: Lightning`→`Storm` 改名后译文仍写「闪电」 | ✅ 已修 |
| H | `Inflection` 与 `Affix` 撞名 | ✅ 已修 |
| I | `TRUNCATED`（中文是照更早英文写的缩写版） | ✅ 23 条已补译，复测 **0** |
| J | 世界地图专名被当普通词译掉 | ✅ **2026-08-12**：那批「死键」里的 300 条是照坏基线补出来的重复品（见 F 项）；活的 283 条地图针脚译名逐条核对无误。同批另修两处**真的内容错**：`The Waterworks`（运河隧道迷宫）有 3 处被写成了 `The Waterworks Office`（另一栋楼），统一为 水务工程；`Wedgelands` 页中文**凭空多出**英文里没有的「切夫赫尔庄园（由同名商会所有）」一句，已删 |
| K | **英文变过、中文没跟上**（旧版英文 diff 查出） | ✅ ember 侧 378 条（08-10）+ crucible 侧 55 条（08-12）。crucible 那批另经 `propagate_fix.py` 推到 24 个**英文逐字相同的同源副本** —— 那些包的英文没变过，三方 diff 永远挑不出它们 |
| L | 上游已删、中文还留着的死文本 | ✅ 与 F 项同批清完 |
| M | **中文从来就没覆盖全英文**（数字覆盖检查查出） | ✅ 修的是**扫描器**：原判据只认阿拉伯数字，会把「三层矿井」「第一军团」「二十年」全报成缺失（并已真的逼出过「2 个十年」这种坏中文）。加了中文数字折算 + 实体剥离 + `decade/dozen/score/century` 单位换算后，两个仓库都归零 |
| **N** | **中文侧整条不存在的键**（新发现） | ✅ **2026-08-12**：crucible 158 / ember 414 条。**任何既有扫描都发现不了** —— 覆盖率、残留、签名、drift 全是拿「中文里的某条」去比对，中文里压根没有的条目不在它们的定义域内，所以库里一直报「覆盖率 99%」而预生角色 Fizzit/Zarajah 几乎整体没译。500 条由 TM 精确命中补齐，72 条会话内人工翻译 |
| **O** | **`lang/cn.json` 键形态错，Foundry 查不到** | ✅ **2026-08-12**：ember 486 键里 372 个失效、crucible 46 个。详见第 1 节的警告块 |

### 冒烟验证怎么做（第 9 项）

管线改造过但**没有在真实世界里跑过**。开一个 Crucible 世界，控制台执行：

```js
game.babele.inspectMapping('Item')                    // 应看到 crucibleDescription / crucibleActions
game.babele.inspectMapping('JournalEntryPage', {data:{type:'ember.location'}})  // 应命中子类型层
await game.babele.sourceDiagnostics()                 // 每个 collection 的译文来源与重叠
game.babele.cacheDiagnostics()
```
重点确认：① 合集里天赋/装备显示中文；② 导入 playtest 冒险后，角色身上的内嵌物品也是中文
（这条验证的是源包回退）；③ 控制台无 `TypeError`。

阶段 5 之后补充四个要看的点（都是这轮改动可能出问题的地方）：

- **动作卡上的消耗标签**：`{action}动` / `{focus}专` / `{heroism}英` / `武{action}动`
  原本是 `2A`/`1F`/`W2A`，改成中文后字更宽，确认没有换行或截断
- **物品名拼装**：品质/词缀会拼成「精良长剑」「长剑·腐蚀」，看一眼语序是否可接受
  （`ITEM.COMPOSED_NAME.Prefix` / `.Suffix`）
- **法术名拼装**：`迅捷的 燃烧的打击` —— 屈折形容词与法术名之间的空格是系统硬编码的，改不掉
- **rules「符文」页**：那个原本坏掉的 `@Embed` 现在应该能正常渲染出「符文：风暴」

阶段 24–27 之后再补三个（都是这几轮大改动的直接产物）：

- **GM 专属块是否仍对玩家隐藏**：第 6 批补回了大量
  `<section class="block gamemaster">`。以玩家身份看一页 `Area Overview`，
  确认「游戏主持人摘要」没有暴露出来。
- **双系统分支只显示当前系统那一支**：第 7 批补回了大量
  `<sup class="system-swap-inline">`。在 Crucible 世界里看一段带检定的正文，
  确认只出现 `[[/skillCheck …]]` 那一支，不会两套规则并排显示。
- **孪生包不会被白 fetch**：`compendium/cn/ember.adventure.json` 有约 **9 MB**。
  在 **Crucible** 世界里开控制台看 Network，确认 babele 没有去拉它
  （dnd5e 版的包在 Crucible 世界里根本不存在，拉了就是纯浪费流量）。

---

## 8. 决议记录

| 日期 | 决议 | 理由 |
|---|---|---|
| 2026-08-06 | 术语表以 `glossary_crucible_merged.json`（4/16，4602 条）为基底 | 是本地所有 crucible 术语表的超集，且冲突已裁决。另一份 `glossary_adaptive_crucible.json` 与之逐条相同 |
| 2026-08-06 | **不**并入 PF2E 主表 `fvtt\glossary.json`（10942 条） | 不同世界观。例：`Restrained` 本项目「受缚」vs PF2「受制」。仅作低优先级建议源 |
| 2026-08-06 | 术语冲突采用新版裁决 | `Restrained` 受拘束→**受缚**；`Arcden` 阿克登语→**奥克登语**；`jurtak` 尤塔克→**尤尔塔克**；`Hulg'run Lineage` →**赫尔格伦血统**；`Ken Crystals` 感晶→**肯水晶**；`House Cevher` 切夫赫尔→**杰夫赫尔**（共 28 条，全表见 `5-其他内容/glossary/`） |
| 2026-08-06 | 管线用 babele 声明式 `registerMapping` + `_variants`，而非手写遍历转换器 | 手写转换器拿不到 2.9.1 的源包回退，会白丢约 60.9 万字符的免费翻译 |
| 2026-08-06 | 英文基准同时存仓库 `compendium/en/` 与 `5-其他内容/english-baseline/` | 前者是当前版本、跟译文一起进 git；后者是历史快照、用于跨版本算 drift。用途不同，不是重复 |
| 2026-08-06 | 专有名词保持 `中文 English` 双语并列格式 | v1.0.15 既有译文已是此风格，改动会造成大面积不一致 |
| 2026-08-06 | 抽取器改为**解释 mapping 数据**，而非硬编码字段 | 运行时与抽取端共用同一份 mapping，结构上杜绝键名漂移 |
| 2026-08-06 | 保持既有键形状（`description` 多态、`actions` 按 id、`ancestry` 等嵌套对象），不改名 | 改键形状会让 crucible-cn 现有约 4600 条译文全部失配 |
| 2026-08-06 | 每次抽取器改动后必须跑一次 `crosscheck_vs_crucible_fr.py` | 多态 description 那个 bug 单看自己输出完全正常，只有独立实现对照才暴露 |
| 2026-08-06 | ember 的 `-en.json` 从 `compendium/cn/` 移出到 `english-baseline/` | 它们躺在 babele 注册目录里会被每次开世界 fetch（其中一份 11.4 MB 且是损坏 JSON） |
| 2026-08-06 | 各 pack 文件的 `mapping` 块一律删除，改用全局 `registerMapping()` | compendium-local 映射优先级高于注册层，留着会静默盖掉新映射并继续调用已不存在的转换器 |
| 2026-08-06 | 同名文档若内容可合并则合并到名字键，标量冲突才用 `_id` 键 | babele 匹配 `_id` 优先于 `name`；分开发键既翻倍工作量，又会让日后改名字键的译文静默失效 |
| 2026-08-06 | `@UUID[目标]{标签}` 的标签**要翻译**，校验只比对目标 | 标签是玩家看到的可见文字；早期把整段当作不可变标记，导致合法译文被拒 |
| 2026-08-06 | `Kinesis` → **念力**（非术语表里的「念动力」） | 既有译文已用「念力术师 Kineturge」「符文：念力 Rune: Kinesis」「念力熟练度」，服从既有用法 |
| 2026-08-06 | `Warden` → **守林者**（非术语表里的「典狱长」） | 语境是「召唤自然先祖魂灵的战斗德鲁伊」，且要避开 `Guardian`＝守护者的碰撞 |
| 2026-08-06 | `Swarm`（archetype）→ **群集**，非「虫群」 | 该 archetype 描述是「多个生物以单一群集实体行动」，并不限于虫类；具体生物名如 Insect Swarm 仍用「虫群」 |
| 2026-08-06 | 新增译名：`Automaton` 自动人偶 / `Dust Devil` 尘卷风 / `Mud Elemental` 泥浆元素 / `Stone Elemental` 岩石元素 / `Constrictor` 绞杀者 / `Juggernaut` 碾压者 / `Telekinetic` 念力者 / `Deep Behemoth` 深层巨兽 / `Lightweaver` 织光者 / `Prankster` 恶作剧者 / `Ancestral Guardian` 先祖守护者 / `Ancestral Ward` 先祖庇护 / `Ancestral Spirit` 先祖之灵 / `Rune: Storm` 符文：风暴 | 对齐既有 taxonomy 风格（土元素/火元素/冰霜元素、元素微粒）与 archetype 风格（成年龙兽/狂战士/掘穴者） |
| 2026-08-06 | crucible-cn 的 `i18nInit` 里 `sort = "tri"` 改为「排序」 | "tri" 是法语，抄 crucible-fr 时的遗留 |
| 2026-08-06 | `DEFENSES.Madness`/`Wounds` → **集结阈值 / 治疗阈值** | 上游 0.10.1 把这两个防御改名为 Rallying/Healing Threshold；`RESOURCES.Madness`/`Wounds`（疯狂 / 创伤）是另一组东西，不动 |
| 2026-08-06 | `Hazard` → **危害**，`Danger Level` → 危险等级 | 服从 compendium 既有的「环境危害 / 配置危害」 |
| 2026-08-06 | `Inflection` → **屈折**，且 talent 条目名统一为「屈折：X」 | 原译「词缀」与 `Affix`（词缀）完全撞名，玩家无法区分 |
| 2026-08-06 | 屈折/施法构件用词以 `crucible.affixes` 的 `adjective` 为准：编构 / 限定 / 遁避 / 延展 / 否定 / 拉拽 / 推挤 / 迅捷 / 反应 / 重塑 | 那一组 10 条是唯一内部自洽的；talent 侧原本是作曲 / 判定 / 闪避 / 推开 / 迅捷化，且「闪避」与 `Dodge` 撞名 |
| 2026-08-06 | `Critical Hit` → **暴击**（lang 里原有的「重击」改掉） | glossary 与 compendium 正文都是暴击，只有 lang 一处例外；且「重击」易与 `Strike`（打击）混淆 |
| ~~2026-08-06~~ | ~~`Fortitude` 的 lang 标签由「坚韧」改为「坚韧防御」~~ | **已被 2026-08-12 推翻** → `Fortitude` 改 **强韧**。加后缀只是把撞名藏起来：正文里的裸「坚韧」仍然分不清指属性还是指防御 |
| 2026-08-06 | `Signature`（天赋树）→ **招牌**，非「签名」 | `TALENT.WARNINGS.Banned` 早已写作「招牌天赋」，节点标签却是「签名」 |
| 2026-08-06 | 紧凑标签译成中文：`{action}A`→`{action}动`、`F`→专、`H`→英、`W`→武、`{value}R`→`{value}轮` | 目标是完整汉化；但字宽会变，列入冒烟验证清单 |
| 2026-08-06 | `DC` / `∞` / `???` 有意保留原样，记入 `lang/lang_keep_english.json` | DC 是中文桌游圈通用写法；后两个是符号。不进白名单的话每轮都会被报成漏翻 |
| 2026-08-06 | 物品名拼装：前缀 `{prefixes}{name}`、后缀 `{name}·{suffixes}` | 英文的 "Sword of Flame" 语序在中文里不成立；间隔号是国内游戏常见写法，且不会与词缀名内部的字冲突 |
| 2026-08-06 | 中文没有 zero/two/few/many 复数形态，这些键一律填与 `other` 相同的值 | Intl.PluralRules 对 zh 只会取 other；填上只是为了让 `lang_gap.py` 不再报缺口 |
| 2026-08-06 | `Electricity` → **电击**，状态 `Shocked` 连带改为 **感电** | 电力像市电、电能像物理量；伤害类型与状态不能同名 |
| 2026-08-06 | `Radiant` → **光耀**；`Poison`（伤害类型）→ **毒素**；`Psychic` → **灵能** | 都是正文多数写法；辉光/光辉太像，毒药指物不指伤害类型 |
| 2026-08-06 | `Tier` 作独立名词时用 **阶数**（最低阶数 / 阶数配置 / 附魔阶数），计数时用 **阶**（1 阶 / 每阶） | 单说「阶」在 UI 标签里不成词，但正文计数必须与 compendium 的「1 阶」一致 |
| 2026-08-06 | `ABILITIES.*Abbr` 六个缩写定为 敏 / 智 / 存 / 力 / 韧 / 感 | 原本是机翻（`Pre`→「预备」、`Tou`→「图」）；属性框位置窄，单字最合适 |
| 2026-08-06 | `Formidable Presence` 译作 **威严气场**，不套用 `Presence`＝存在 | 这里的 presence 是「气场」的日常义，不是属性值 |
| 2026-08-06 | 被写成裸中文的 enricher 一律还原成 `@Condition[...]` / `@Action[...]` 等标记 | 这类标记不带标签、渲染出来就是 lang 译名；写成裸字等于让玩家失去可点链接与说明浮窗 |
| 2026-08-06 | enricher 方括号内的**目标与参数一律照抄**，只有 `{标签}` 可译 | 已经踩到两次：crucible 的 `@Embed[...runeLightning000...]`、ember 的 `@Embed[... inline overview]` 被译成「概览」。两次都是链接/嵌入块直接失效，且 diff 里看不出来 |
| 2026-08-06 | `Globlin` → **格布林**（音译），不用意译「泥砾精」 | 遇到 `Mud Globlin` / `Paint Globlin` 这类前缀构词时意译没法组合；战役包里还有一处误译成「绘画地精」，而地精是另一种生物 |
| 2026-08-06 | `Waterborne` 作家族名时音译为**沃特伯恩** | 该家族经营酿酒坊；原有一处 `Waterborne Whiskey` 被当普通词译作「水运威士忌」，应改为沃特伯恩威士忌 |
| 2026-08-06 | `The Last Pit` → **最后的矿坑**（此前记为「最后一坑」） | 与同组的 `The Empty Pit` 空矿坑 / `The Active Mine Pit` 在采矿坑 用同一个「矿坑」构词；「最后一坑」的量词读不顺 |
| 2026-08-06 | `Bright Lord` → **辉耀领主** | 沿用 v1.0.15 旧译文里的写法（该角色只在书信中被这样称呼）。`For Other Fortunes` 则保持英文原样，与战役包已译各页一致 |
| 2026-08-06 | 「没有可译文本」的条目（整条都是 `@Embed`/`@UUID` 之类）不计入覆盖率分母 | 它们永远不可能含中文，计进去等于让每个包都永远显示未完成，并把死条目塞进驱动翻译批次的待译清单。全库 298 条 |
| 2026-08-06 | `[[/item 中文名]]` 属**正常**，不算标记被译坏 | dnd5e 的 `/item` 按角色身上的物品名解析，而 babele 已经把那些名字翻成中文了 —— 此处英文反而会失配。`readaloud="…"`、`[[/r …#掷骰说明]]` 同理 |
| 2026-08-06 | 孤儿译文是否移植，看**结构是否逐段对得上**，而不是看有没有译文 | `Arcturel Upper` 28 页 `<p>` 238/240 与今天的英文一致 → 移植；若像阶段 13 那批缺 2200 个 `<li>`，移植只会制造「显示 100% 实则缺整块」的新债 |
| 2026-08-06 | 并行翻译：译者**自己**把 `apply_translations.py --dry` 跑到 0 拒绝才算交付；任何 agent 都不许写 `compendium/cn` | 标记类错误在返回主控前就被挡掉，主控不必逐条复核；落盘只由主控做，单个 agent 出问题只需丢掉一个 batch 文件 |
| 2026-08-06 | 术语冲突的判断依据强弱：**同名条目/物品的 `name` 字段 > 同卷已译页 > 全库多数写法 > glossary_ec > 交给译者的术语表** | 第 1 批三个 agent 各自独立查出我给的术语表有三条（Inkaro/Amalthea/引号）与全库不符，按既有译文处理是对的。术语表是二手摘录，会抽错 |
| 2026-08-06 | `Attunement` → **同调**（战役包 418 处调谐一并改掉） | lang/cn.json 与 character 小包早已是同调，玩家在角色卡上看到的就是它；正文与 UI 不一致比选哪个词更糟 |
| 2026-08-06 | `Inkaro` → **因卡罗**，改 4 个物品条目名而不是 126 处正文 | 统一方向选「改动面小的那边」；统一后 `@UUID` 标签与物品名自然对齐 |
| 2026-08-06 | 引号一律 **“”** | 全库 2618 : 44，44 处是后来一批按风格说明写的，与存量不符 |
| 2026-08-06 | 孤儿页面用**标记指纹**配对，不用页名 | 页名恰恰是上游改掉的那个东西；而 `@UUID`/`[[/…]]` 会被原样抄进译文，是天然指纹。`In The Behemoth's Wake` 只改了大小写就成了孤儿，指纹相似度 0.99 |
| 2026-08-08 | 第 8c 项与第 8j 项**合并**为「页面重对齐」，不分两批做 | 两张清单有 32 条路径重叠（上游改写过的页），分开做会让两份 batch 争同一条 path、落盘时静默互相覆盖；而闸门是多重集**相等**比较，本就同时管两个方向 |
| 2026-08-08 | 并行单元改用**页文件 + Edit 局部改**，不再交整页替换值 | 按「缺失内容」切会把单元规模低估 5.6 倍（26 万 vs 145 万），是阶段 23 fill-8 被掐断的根因；且整页重写会把已校对的译文洗掉。新格式下「没动过的字节保持没动过」是格式保证的。副作用：额度耗尽从事故降成普通中断 |
| 2026-08-08 | `fill_twin.py` 的 TM 键用**结构路径**，不用最后一段路径 | 只取最后一段会把 `items.X.name` / `items.X.actions.<id>.name` / `effects[].name` 三种惯例混成一堆 —— 正是阶段 23 警告过的错。改用结构骨架后歧义键 697→530 |
| 2026-08-08 | 先落第 6 批，**再**跑孪生包 TM 填充 | 被弃填的 554 条里 66% 正是第 6 批在修的 8c/8j 页；顺序反了少填 421 条 |
| **2026-08-09** | **术语与前后不一致由主控自行裁决并统一，不上报项目所有者** | 项目所有者明示。此前「不静默择一、列进 `glossary_ec.disputes.json` 待裁决」的做法降级为**仅证据不足时**适用。裁决仍走既定依据阶梯，且必须写进本表 + 用 `unify_terms.py` 执行 + 复跑 QA |
| 2026-08-09 | `Electricity` → **电击** 再次确认；`Lightning` → 闪电，两者**不是同一个词** | 库内 电击 97 / 电能 48 / 电力 8；crucible `lang/cn.json` 里一个含「电」的键都没有，故按决议＋库内多数。**2026-08-12 复核已归零**：英文写 `Electricity` 的条目里中文全是「电击」；库内残下的「电能」25 / 「电力」8 译的是 `electrical energy` 等别的英文，不是残留 |
| **2026-08-11** | **推翻 08-09 的 `Reaper Ocean`→收割者海洋，改回「劫掠者海洋」** | **我当时判错了。** 上游对同一片海有两种拼写：`Reaver Ocean`（多数、正规）与 `Reaper Ocean`（少数几处，**是拼写错误**）。`Reaver` 就是劫掠者，原译「劫掠者海洋」本来就对。我当时只按 `\bReaper Ocean\b` 取样，没查 `Reaver`，还反过来推理「劫掠者对应 Raider」，把对的改成了错的。全库实测 劫掠者海洋 31 : 收割者海洋 6（后者全是我改出来的），已改回。**教训：定名前先查这个专名在英文侧有没有异体拼写，只按一种拼写取样会取到偏样本。** |
| 2026-08-09 | 库内 167 处「闪电」中，**123 处不动** | 那 123 处英文确实是 `Lightning`（忠实翻译），另有 44 处是「闪电般迅捷」这类比喻。`Rune: Lightning→Storm` 的改名只作用于**英文写 Storm** 的地方。**先查英文再判中文**，否则机械替换会误伤大片 |
| 2026-08-09 | `Electricity` 的「电能」替换加 `unless: electrical energy` | `Rune: Storm` 引导句是 `The chaotic force of raw **electrical energy**`，中文「原始电能的混沌之力」翻的是这个散文短语，而同条里的伤害类型 `deals **Electricity** damage` 中文**本来就是**「电击」。评审抽查抓到的；不加守卫会把正确译文改坏（ember 5 处、crucible 全部 7 处都属此类） |
| 2026-08-09 | `Concluding the Event` → **事件结束**；`Event Outcome` → **事件结果**；两者互加 `unless` | 是两个不同的词组，但在同一条目里大量共现（`Event Outcome` 的 61 条里就有 134 处「事件结束」）。不互相排除就会把对方的译名改坏。同时出现的条目一律跳过 —— 宁可漏改不可错改 |
| 2026-08-09 | `Marlstone Manor` → **马尔斯通庄园**（变体写全「马尔石庄园」而非裸「马尔石」） | name 字段即马尔斯通庄园，英文对得上的条目里 161:22。裸「马尔石」另有街区名之用，替换会误伤 |
| 2026-08-09 | `Fernis Ossa` → **费尔尼斯**；`Horrendor` → **霍伦多尔**；`Yakoshta` → **雅科什塔** | 均以 name 字段为准（204:50、82:13、6 处 name 对 2）。**「惊惧者」是另一个 actor `Harrower` 的名字**，挂在 Horrendor 上属挂错名，`glossary_ec` 里那条也是错的，已改 |
| 2026-08-09 | `Young Cheliceraeth` → **幼年螯蛛艾斯**，**不取 name 字段** | 这是依据阶梯的例外：孤立的 actor name「幼年螯蛛以太兽」只有 8 处，而 archetype name「螯蛛艾斯」、macro「切换螯蛛艾斯」、正文 44 处三方一致。改这一处 name 比改 44 处正文便宜（「改动面小的那边优先」） |
| **2026-08-12** | **复查发现：缺陷表里标 ✅ 的项必须拿数据验，不能照抄上一版状态** | 本轮我一度把 C/D/E/J 四项直接誊成 ✅，实测其中 **E 与 J 根本没做**（disputes 20 条积压、地图针脚 14 个仍是普通词），而 C/D 又被我误报成"没做"（裸计数 电能 25/阶位 160 看着像残留，加英文闸后全是别的英文：`electrical energy`、`Rank`）。**结论：术语类结论一律要带英文闸的计数，裸词频既会漏也会误伤。** |
| **2026-08-12** | 20 条积压 disputes 一次裁完，依据阶梯＝name 字段 > 同卷已译页 > 全库多数 | Agrimage→农艺法师(284:180)、Thornling→荆芽灵(313:148)、Aberin→阿伯林(216:31)、House Cevher→杰夫赫尔(913:6:4)、Ordain→奥尔丹(3062:225)、Arcturian→阿克图里安(248:3)、Reliquary→圣髑匣(29:10)、Wind Raider→风袭劫掠者(26:2)、Hulg'run→赫尔格伦。**注意**：库内的「念动力」译的是 `Telekinesis` 而非 `Kinesis`，两者是不同的词，disputes 里那条已过期 |
| **2026-08-12** | `essence`→**精华**；`Stride`→**步幅**；`maximum Action`→**最大动作** | essence 132:8，同一个英文、无术语/散文之分；Stride 67:28 且 lang 的 `ACTOR.StrideSpecific` 就是「步幅 {stride}」（英文闸必须**区分大小写**，小写 stride 是动词，「所跨步踏过的土地」那处不能动）；maximum Action 按 lang 的 `RESOURCES.Action`＝动作 |
| **2026-08-12** | `The Waterworks` → **水务工程**，与 `The Waterworks Office`（水务办公室）分开 | 是两个地点：前者是城区地下的运河隧道迷宫，后者是那栋楼。中文有 3 处把前者写成了后者，玩家按指示去「水务办公室」会走错地方。取地名志 landmark 名 水务工程 |
| **2026-08-12** | 两处**中文与英文不符**的实质错误 | ① `Wedgelands` 页中文凭空多出「例如切夫赫尔庄园（由同名商会所有）以及」一句，英文只有 Corpin Sanctuary，已删；② `Supplies and Demands` 的旁白整段被重排，且把 `thornling` 译成「农法师」——埃迪维尔是荆芽灵不是农艺法师，还把台词「幸好它是附了魔的」改写成了旁白「若不是埃迪维尔施加的魅惑」。已按英文重译。**两处的块数多重集都与英文相等，所以 BLOCK 检查一声不响 —— 块数相等不等于顺序与内容正确** |
| **2026-08-12** | 两个被截断的页名 | `欢迎来到 Welcome To Crucible`（中文半截是空的）→「欢迎来到 Crucible」；`什么是《 What is Crucible`（残留孤立书名号）→「什么是 Crucible」。库内系统名一律保留英文 Crucible，中文里已含该词，故不再缀英文尾巴 |
| **2026-08-12** | **`lang/cn.json` 一律写扁平点号键，禁止按点建嵌套** | Foundry 的 `getProperty` 先试整键、再按点下探。顶层带点又嵌套的混合形态**两条路都断**：ember 486 键里 372 个（77%）静默失效。`apply_lang.py` 的 `set_path` 当时就是按点建嵌套的，等于每写一条新译文就重造一次这个坑（Boon 统一批亲历：新值成了嵌套副本，与旧扁平键并存，UI 一个字没变还毫无报错），已改为 `root[dotted] = value` |
| **2026-08-12** | **校验必须复刻被验证系统的查找语义** | 上面那个缺陷之所以活了这么久，是因为校验脚本自己会递归展开嵌套，于是一路报「486 键缺口 0」。`lang_gap.py` 现在有 `foundry_lookup()` 与 `UNREACHABLE` 一列 |
| **2026-08-12** | `Boon` → **恩惠骰**（原 恩惠 57 / 惠骰 51 / 恩惠骰 4 三写并存） | lang 的 `DICE.Boons` 就是这三个字，玩家每次掷骰都看得到，且与 `DICE.Banes`＝祸骰对称；`惠骰` 在 lang 里没有任何锚点。**dnd5e 侧例外**：`ember.adventure.json` 里 boon 多是普通英文名词（metaphysical boon / the boons associated with…），只把 system-swap 嵌进来的 4 处「+N Boons」改掉 |
| **2026-08-12** | `Fortitude` → **强韧**（`Toughness` 保持 坚韧） | 缺陷表 B 项的根治。两者共用「坚韧」二字时，正文 120 处裸「坚韧」无从分辨指属性还是指防御，而 Willpower 的公式 `(坚韧+存在)/4` 用的正是属性 |
| **2026-08-12** | `Willpower`→**意志**、`Accurate`→**精准**、`Arrow`→**箭矢** | 意志 79:13；精准是 lang 自己的 `AccurateTooltip` 与 rules 攻击标签表的写法（`ACTION.TAG.Accurate` 的「精确」是同一文件内的自相矛盾）；箭矢是运行时按 `SPELL.GESTURES.Arrow` 渲染出的法术名，且中文的「箭头」指箭镞/光标 |
| **2026-08-12** | lang 标签：`Vocal` 声乐→**言语**、`Auditory` 听觉的→**听觉**、`Mechanical` 机械的→**机械**、`ARMOR.PROPERTIES.Natural` 自然→**天然**、`WEAPON.TAGS.Natural` 自然→**天生** | 声乐是误译（Silenced 页正文写的就是「言语标签」）；其余 40 余个 `ACTION.TAG` 一律是不带「的」的名词，那两个是仅有的例外；两个 Natural 分别对齐同文件的 `ARMOR.CATEGORIES.Natural`＝天然 与正文的「天生武器熟练度」 |
| **2026-08-12** | **抽取器的 `textCollection` 一律按 `text` 建键，不能用 `_id`** | Babele 的 `textCollection` 就是 `fieldCollection("text")`，运行时查 `translations[data.text]`。抽取器写成 `it._id ?? it.text`，于是英文基准按 ID 建键、中文包按文本建键，300 条**正在正常生效**的地图针脚译名被判成死键，`prune_dead` 差一步就整批删掉。**清理类操作前必须先验证「死」的判据本身没错** |
| **2026-08-12** | 数字覆盖检查必须认中文数字与倍数量词 | 只认阿拉伯数字会逼出坏中文：库里真出现过「2 个十年」（2 decades），历史上还逼出过「3 层矿井」「第 1 军团」。中文本来就该写 三/两/第一/二十年 —— 是规则错了，不是译文错了。`decade/dozen/score/century` 这类量词换算后的值也算可接受写法 |
| **2026-08-12** | 上游把 `{Persuasion}` 裸写在正文里，改英文基准而不是改译文 | 它前面没有 `@UUID[…]`、也不在任何 enricher 里，Foundry 会原样渲染花括号。中文写「魅力（游说）」比上游还正确，不该为迁就 PLACEHOLDER 那条正则改成「魅力{游说}」。记入 `LOCAL-PATCHES.md` 第 2 条 |
| **2026-08-12** | 新定专名：`Mial Mountain` 米亚尔山 / `inkaro pearl` 因卡罗珍珠 / `Dusk Hound` 暮猎犬 / `Winged Scavenger` 有翼食腐者 | 补最后 72 条纯缺译时定的，库内此前无任何写法。其余专名一律先查既有译法再落笔 |
| 2026-08-09 | 四个 `Vista:` 场景 name 定名 | `Ordain Streets` 授命街道→**奥尔丹街道**（同一份场景清单里 `Ordain Overview`/`Ordain Interiors` 都作奥尔丹，「授命」是把 ordain 当动词的机翻）；`Yakoshta`、`Arbore Sanctorus` 原本整个没译 → **雅科什塔** / **圣树庇护所**（后者取自既有 name 字段）；`Ordain Interiors` 室内装潢→**室内景**（interiors 指室内场景，上下文是「舒适休息室」「神秘者公寓」这类构图，不是装潢） |
