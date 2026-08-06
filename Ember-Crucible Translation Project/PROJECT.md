# Ember / Crucible 汉化项目 · 主文档

> 这是本项目的**唯一长期入口**。新会话请先读第 1 节，再按需跳读。
> 阶段日志（第 6 节）只追加、不重写，用来做长期校对与断点续做。

---

## 1. 快速跟进（新会话必读）

**当前状态**：阶段 0–20 完成。管线已改造到 babele 2.9.1 声明式映射，工具链就绪。
**crucible 侧全部译完**（15 个合集包残余 0、`lang/cn.json` 1842 键缺口 0），只剩冒烟验证与发版。
**ember 侧**：lang 486 键缺口 0，7 个小包 100%，运行时补丁（硬编码字符串 + 中文字体回退）已就位；
战役包 13117 / 19605 = **67%**，76 组 journal 里 8 组完成。

**翻译已改为并行推进**（阶段 20 起）：一轮 9 个单元、约 27 万英文字符，
译者自检闸门 + 对抗式审校 + 跨单元术语核对。做法与硬约束见阶段 20 日志。
剩余约 **179 万字符**（常规 111 万 + 第 8c 项 53 万 + 孪生包独有 14 万），约 7 轮。

**下一步**（按顺序）：
1. **冒烟验证** —— 见本节末尾。这是唯一无法靠脚本证实的环节，只能在真实 Foundry 世界里做。
2. 发 crucible-cn **0.9.0**
3. 发 ember_cn **1.1.0**
4. ember 战役正文继续并行推进。开新一轮前**先跑一遍孤儿清单**（第 8g 项）——
   阶段 19 靠它发现整整一卷 28 页译文躺在改名前的旧路径上，阶段 20 又靠标记指纹找出 42 个孤儿页面。

全库术语已统一，规则正文的整段漏译已补齐（见阶段 6）。剩下的 BLOCK / INLINE 漂移属观感层面，
不影响功能，可与冒烟验证一起决定要不要在本版处理。

### git 状态（2026-08-06）

两个插件仓库的改造与汉化**已合入各自 main，但尚未 push**。

| 仓库 | main 现在 | 改造前的 main（还原点） | 保留的分支 |
|---|---|---|---|
| `2-Crucible汉化插件` | `ea8783e` | **`085bfe6`** | `feat/babele-2.9.1-crucible-0.10.1` |
| `1-Ember汉化插件` | `ffe7f19` | **`cf58b4f`** | `feat/babele-2.9.1-pipeline` |

要回退到改造前：`git reset --hard <还原点>`。

**发版流程只由 tag 触发**（`0.9.0` / `v1.1.0` 这类），push main **不会**发布任何东西。
所以 push 是安全的；但**打 tag 前必须先做冒烟验证**。

项目本身（PROJECT.md + 脚本 + 术语表）已提交进 `Desktop\fvtt` 仓库 `b17a5f4`。

**翻译时必须遵守的既定译名**（避免和已完成的 11 个包冲突）：
`Kinesis`念力 · `Warden`守林者 · `Guardian`守护者 · `Swarm`(archetype)群集 · `Tier`阶 ·
`Electricity`电力 · `Bludgeoning`钝击 · `Fire`火焰 · `Corruption`腐化 · `Fortitude`坚韧防御 ·
`Toughness`坚韧 · `Wisdom`感知 · `Presence`存在 · `Willpower`意志力 · `Health`生命值 ·
`inflection`屈折 · `gesture`手势 · `rune`符文 · `spellcraft`施法 · `essence`精华 · `compose spells`构筑法术。
完整表见 `5-其他内容/glossary/glossary_ec.json`。

**怎么继续（照抄即可）**：

```powershell
$P = "C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
$SYS = "$env:LOCALAPPDATA\FoundryVTT\Data\systems\crucible"

# 0. lang 缺口（与 compendium 相互独立的一条线；当前为 0）
python "$P\3-常用脚本\qa\lang_gap.py" --repo "$P\2-Crucible汉化插件" --package $SYS --out "$P\5-其他内容\reports\crucible"
#    翻完回写： apply_lang.py --repo <repo> --package $SYS --batch <batch.json> [--clean-stale]
#    翻完之后再 lang_gap.py --sync-baseline，把 <repo>\lang\en.json 更新成新版基准

# 1. 看当前缺口 + 生成待译清单
python "$P\3-常用脚本\qa\validate_translations.py" --repo "$P\2-Crucible汉化插件" --out "$P\5-其他内容\reports\crucible"

# 2. 读某个包的待译项（清单在 5-其他内容\reports\crucible\todo\）
#    翻译成 {"<path>": "<中文>"} 的 JSON 批次文件

# 3. 回写（三道闸会拦下英文源漂移 / 无中文 / 标记破损）
python "$P\3-常用脚本\qa\apply_translations.py" --repo "$P\2-Crucible汉化插件" --pack <pack.json> --batch <batch.json>

# 4. 复验
python "$P\3-常用脚本\qa\validate_translations.py" --repo "$P\2-Crucible汉化插件" --out "$P\5-其他内容\reports\crucible"
```

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

### 版本矩阵（截至 2026-08-06）

| 组件 | 类型 | 版本 | 位置 |
|---|---|---|---|
| crucible | 系统 | **0.10.1** | `%LOCALAPPDATA%\FoundryVTT\Data\systems\crucible` |
| ember | 模块（**付费/protected**） | **0.6.0** | `…\Data\modules\ember` |
| babele | 模块（翻译框架） | **2.9.1** | `…\Data\modules\babele` |
| crucible-cn | 汉化模块（本项目） | 0.8.9.13 → 目标 0.9.0 | `2-Crucible汉化插件\` |
| ember_cn_unofficial | 汉化模块（本项目） | 1.0.15 → 目标 1.1.0 | `1-Ember汉化插件\` |

两个汉化仓库：
- https://github.com/takaqiao/crucible-cn
- https://github.com/takaqiao/ember_cn_unofficial

### 汉化的两条通道

- **`lang/cn.json`** —— 界面字符串（Foundry 原生 i18n），走 module.json 的 `languages` 字段
- **Babele `compendium/cn/*.json`** —— 合集内容（天赋/装备/日志/战役正文），走 `babele.register()`

crucible 侧两条都用；ember 侧两条也都用。

### 待译量（精确测量，非估算）

**crucible 0.10.1**

| 项 | 数量 | 现状 |
|---|---|---|
| lang 新增 key | 293 | ✅ 阶段 5 已补 |
| lang 英文原文改动 | 11 | ✅ 阶段 5 已重翻 |
| lang 未译（cn == en） | 15 | ✅ 13 条已译，2 条有意保留 |
| compendium 新增条目 | 132（含整包 `adversary-equipment` 53 条） | ✅ 阶段 3–4 已补 |
| compendium 失效条目 | 4 | ✅ |
| compendium 叶级覆盖率 | 88%（4238 串中 3745 已译） | → 97%（4556/4717），真实残余 0 |
| **compendium 待译** | **493 串 / 7.8 万字符** | ✅ 完成 |

> 这张表是阶段 0 的初测数据，保留用于对照。用完整基准重算后的准确数字见阶段 3 的关键发现 2。

**ember 0.6.0**（战役包 `ember.crucible-adventure`，16065 条可译串，已完成 39%）

| 项 | 待译串 | 待译字符 | 备注 |
|---|---|---|---|
| 已映射未翻（主要是 140 个新页面 `text.content`） | 644 | 822 K | 700 K 集中在新页面正文 |
| actor 内嵌 items —— 可回源自动翻 | — | −609 K | 采用默认 mapping 后免费 |
| actor 内嵌 items —— 无来源包（战役独有） | 727 件 | 220 K | 必须内联翻 |
| actor 内嵌 items —— 来源是 `dnd5e.*` | — | 405 K | Crucible 世界里该包不存在，回源失败 |
| table results | 251 | 77 K | 需补 mapping |
| `outcomes.label` / `outcomes.summary` | 548 | 42 K | 需补 mapping |
| subtitle / pronunciation / caption / terrain 等 | 265 | 4 K | 需补 mapping |
| **合计（Crucible 世界）** | | **约 159 万字符** | |

**ember 其他包**

| 包 | 状态 |
|---|---|
| `ember.adventure`（dnd5e 孪生战役） | 零汉化，7687 串 / 7.09 M 字符 —— 但日志正文与 crucible 版**逐字节相同**，值级 TM 命中 79.7% |
| `ember.character` | 零汉化，164 串 / 2.3 K |
| `ember.crucible-effects` | 零汉化，42 串 / 6.4 K |
| `ember.dnd5e-effects` | 零汉化，42 串 / 6.5 K |
| `ember.crucible-affixes` | 零汉化，4 串 / 1.5 K |
| `ember.crucible-items` / `ember.dnd5e-items` | 空包，无需处理 |
| `ember.crucible-adversary` | 94 串 / 10.3 K 待译 |
| `ember.crucible-character` | 115 串 / 14.1 K 待译 |
| `lang/cn.json` | 47 个新 key |

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

### 5.4 发版前例行检查

```powershell
$P = "C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
# 1. 真实残余（先扣掉 babele 会自动解析的部分，别重复翻译）
python "$P\3-常用脚本\qa\validate_translations.py"      --repo <repo> --out <reportDir>
python "$P\3-常用脚本\qa\resolve_generic_fallback.py"   --repo <repo>
# 2. 外来文字污染（西里尔/亚美尼亚/希伯来/泰文等机翻残留）
python "$P\3-常用脚本\qa\scan_foreign_script.py"        --repo <repo>
# 2b. 方括号里被译成中文的标记（链接/嵌入块静默失效，BROKEN 应为 0 或全部有已知原因）
python "$P\3-常用脚本\qa\scan_markup_targets.py"        --repo <repo>
# 3. 孤儿译文（上游改名/删除后失效的条目）
python "$P\3-常用脚本\qa\port_orphans.py"               --repo <repo> --rules <rules.json> --dry
# 4. lang 缺口（NEW/DRIFT/UNTRANSLATED/STALE 应全为 0）
python "$P\3-常用脚本\qa\lang_gap.py"                   --repo <repo> --package <foundry包> --out <reportDir>
```

> `port_orphans.py` 只搬路径、不改译文内容。上游改名后**译文里的旧名字不会自动更新** ——
> `Rune: Lightning`→`Rune: Storm` 就是这么留下 28 处「闪电」的。改名类 drift 处理完必须回头
> 搜一遍旧名字。

### 5.5 发版

1. 更新 `module.json` 的 `version` / `download` / `changelog`
2. `compendium/en/` 与 `release/` 不进发布 zip
3. 打 zip → 建 tag → 发 GitHub release
4. crucible-cn 仓库已有 `.github/workflows/release.yml` 自动化，复用它
5. 发完在第 6 节追加阶段日志

---

## 6. 阶段日志

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

## 7. 待办与排期

| # | 事项 | 状态 |
|---|---|---|
| 1 | 搭建项目目录、clone 两个仓库 | ✅ 完成 |
| 2 | 泛化抽取器（解释型，认 ember page 子类型 + scene 深层结构） | ✅ 完成 |
| 4 | 抽英文基准并用 Crucible-FR 交叉校验 | ✅ 完成（零字段缺口） |
| 3 | 构建 `glossary_ec.json` | ✅ 完成（5522 条） |
| 5 | babele 2.9.1 管线改造（两个模块） | ✅ 完成 |
| 5a | └ 用新基准重算 diff | ✅ 完成（见阶段 3） |
| 6 | crucible-cn **compendium** 全量补齐 | ✅ 完成（残余 0） |
| 6a | └ crucible **`lang/cn.json`** 补 293 新 key + 11 drift + 15 未译 | ✅ 完成（1842 键，缺口 0） |
| 6b | └ 清理 26 处孤儿译文（可选，无害） | ⬜ |
| 6c | └ 发 crucible-cn 0.9.0 | ⬜ 待冒烟验证后 |
| 5b | 修复已发布译文的格式类缺陷（2817+209 可脚本归一） | ⬜ |
| 5c | 裁决剩余真实术语分歧（59 处同名异译 + 20 处基底冲突） | ⬜ 部分已裁决，见第 8 节 |
| 5d | 全库术语统一：Tier/Presence/essence/Wisdom | ✅ 完成（阶段 6） |
| 5e | 伤害类型定名 Electricity/Radiant/Poison/Psychic | ✅ 完成（阶段 6，见第 8 节） |
| 5f | 还原被写成裸文字的 enricher | ✅ 163 → 3 |
| 5g | 补译被整段丢掉的规则正文（TRUNCATED） | ✅ 12 条 / 约 2700 字 |
| 5h | 段落结构与加粗漂移清扫（BLOCK 64 / INLINE 294） | ⬜ 观感层面，不影响功能 |
| 9 | **真实 Foundry 世界冒烟验证**（管线改造后必做） | ⬜ **下一步** |
| 7 | ember_cn **`lang/cn.json`** 47 新 key + 3 未译 | ✅ 完成（阶段 7，486 键缺口 0） |
| 7a | └ ember 小包：effects ×2 / affixes / items | ✅ 完成（4 包 100%） |
| 7a2 | └ ember 小包：character(274) | ✅ 完成（100%） |
| 7a3 | └ ember 小包：crucible-character / crucible-adversary | ✅ 完成 |
| 7a4 | └ 小包内 13 条占位译文补译 + 38 处 @Embed 参数还原 | ✅ 完成 |
| 7b | └ **Ember 硬编码字符串补丁** | ✅ 完成（143/147，4 条纪年缩写有意保留） |
| 7c | └ **字体回退链**：补中文字体，不替换原字体 | ✅ 完成（styles/ember-cn.css） |
| 7d | └ crucible `LANGUAGES.Sign` 误译「标记」 | ✅ 已改为手语 |
| 8 | ember 战役正文分批翻译（按 journal 切批，76 组 / 约 148 万字符） | ⬜ 进行中 |
| 8d | └ `Ushna Dredging Docks` 21 条 | ✅ 21/21 |
| 8e | └ `Arcturel Dives` 40 条 | ✅ 40/40（阶段 19 收官） |
| 8f | └ `Arcturel Tradeway` 28 页 | ✅ 待译归零，但内容是移植来的旧译文，**逐页通读未做** |
| 8h | └ **并行第 1 批**：Glitter / Lantern Roads / Book Of Tales / An Old Friend / Ancient Paths | ✅ 162 条 / 14.9 万字符 |
| 8i | └ **并行第 2 批**：Winding Trail / GM Guide / Expedition / Ch2 Events / Players' Guide / Mythspire + 随机表 ×3 | 🔶 进行中（9 单元 / 27.3 万字符） |
| 8a | └ 机械可修的部分（加粗 734→265、enricher 还原） | ✅ 完成 |
| 8c | └ **已译页里缺失的整块内容**：398 条 / 3742 区块 / 约 53.3 万字符 | ⬜ 需逐页比对补齐；其中约 4.5 万可从孤儿 `Gameplay Details` 页直接搬 |
| 8b | └ 战役包里 祖裔→血统 / 调谐→同调 的残留清扫 | ✅ 完成（阶段 20，含 The Dives/Inkaro/Amalthea/引号共 6 组） |
| 8g | └ 孤儿译文：整卷改名 ✅（阶段 19）；**页级改名** 42 个已用标记指纹配对，2 个强匹配已进第 2 批 | 🔶 |
| 10 | **dnd5e 孪生包 `ember.adventure`** | ⬜ 88% 可 TM 直接填，需写 `tm/fill_twin.py`；独有部分仅 14.2 万字符 |

### crucible compendium：✅ 已完成，不必再动

15 个包中 13 个内联 100%；`playtest` / `pregens` 剩下的 161 条是 actor 内嵌物品，
**由 babele 通用回退从已译包按名字自动取译文，残余为 0**。

> ⚠️ 不要因为 `validate_translations.py` 还显示 97% 就去补那 161 条。
> 先跑 `resolve_generic_fallback.py` 看真实残余。重复翻译会制造同名异译。

```powershell
python "$P\3-常用脚本\qa\resolve_generic_fallback.py" --repo "$P\2-Crucible汉化插件"
```

### crucible lang/cn.json：✅ 已完成

1842 键全部有中文，`lang_gap.py` 四类缺口均为 0，基准已同步到 0.10.1。
有意保留英文的 key 记在 `2-Crucible汉化插件/lang/lang_keep_english.json`（`DICE.DC`、`∞`、`???`），
清单会被 `lang_gap.py` 与 `apply_lang.py` 同时读取，不会再被误报成漏翻。

### 待清扫的既有缺陷（不阻塞发版，但发版前应处理）

| # | 问题 | 范围 |
|---|---|---|
| A | ~~`lang/cn.json` 的 `Burrow` 含亚美尼亚字符 `շարժ`~~ | ✅ 已修 |
| B | `Toughness` / `Fortitude` 同译「坚韧」 | lang ✅ 已改「坚韧防御」；正文仍有 35 条 |
| C | `Electricity`：电击 14 / 电能 12 / 电力 14 / 闪电 7 | ⬜ 需先裁决，见 `terminology_sweep_pending.md` |
| D | `Tier` 有 阶/阶级/阶位；essence 有 精髓/精华 | lang ✅ 已统一；正文 76 + 15 条待清 |
| E | 已发布译文双语格式不一致 209 处、同名异译 59 处 | 见 `glossary_ec.disputes.json` |
| F | 26 处孤儿译文（上游已删除的物品） | 无害，可清 |
| G | ~~`Rune: Lightning`→`Storm` 改名后译文仍写「闪电」~~ | ✅ 已修（28 处 + 8 个条目名） |
| H | ~~`Inflection` 在 talent 包译作「词缀」，与 `Affix` 撞名~~ | ✅ 已修（11 个条目名 + 正文 3 处） |

ember 战役正文的批次划分待 crucible 收尾后确定。

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
| 2026-08-06 | `Fortitude` 的 lang 标签由「坚韧」改为「**坚韧防御**」 | 根治与 `Toughness`（坚韧）的撞名；正文一直就写作「坚韧防御」 |
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
