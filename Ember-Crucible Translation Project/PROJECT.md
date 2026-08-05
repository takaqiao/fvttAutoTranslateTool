# Ember / Crucible 汉化项目 · 主文档

> 这是本项目的**唯一长期入口**。新会话请先读第 1 节，再按需跳读。
> 阶段日志（第 6 节）只追加、不重写，用来做长期校对与断点续做。

---

## 1. 快速跟进（新会话必读）

**当前状态**：阶段 0–4 完成。管线已改造到 babele 2.9.1 声明式映射，工具链就绪，
**crucible 的 15 个合集包全部译完（残余 0）**。

**下一步**（按顺序）：
1. **crucible `lang/cn.json`** —— 补 293 个新 key + 11 处英文改动 + 15 条未译。
   明细：`5-其他内容/reports/crucible/lang_gap.json`
2. **冒烟验证** —— 见本节末尾。这是唯一无法靠脚本证实的环节。
3. 发 crucible-cn **0.9.0**
4. 转 ember：lang 47 key + 6 个零汉化小包 → 发 1.1.0
5. ember 战役正文（分多会话）

**翻译时必须遵守的既定译名**（避免和已完成的 11 个包冲突）：
`Kinesis`念力 · `Warden`守林者 · `Guardian`守护者 · `Swarm`(archetype)群集 · `Tier`阶 ·
`Electricity`电力 · `Bludgeoning`钝击 · `Fire`火焰 · `Corruption`腐化 · `Fortitude`坚韧防御 ·
`Toughness`坚韧 · `Wisdom`感知 · `Presence`存在 · `Willpower`意志力 · `Health`生命值 ·
`inflection`屈折 · `gesture`手势 · `rune`符文 · `spellcraft`施法 · `essence`精华 · `compose spells`构筑法术。
完整表见 `5-其他内容/glossary/glossary_ec.json`。

**怎么继续（照抄即可）**：

```powershell
$P = "C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"

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

| 项 | 数量 |
|---|---|
| lang 新增 key | 293 |
| lang 英文原文改动 | 11 |
| lang 未译（cn == en） | 15 |
| compendium 新增条目 | 132（含整包 `adversary-equipment` 53 条） |
| compendium 失效条目 | 4 |
| compendium 叶级覆盖率 | 88%（4238 串中 3745 已译） |
| **compendium 待译** | **493 串 / 7.8 万字符** |

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
# 3. 孤儿译文（上游改名/删除后失效的条目）
python "$P\3-常用脚本\qa\port_orphans.py"               --repo <repo> --rules <rules.json> --dry
```

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
| 6a | └ crucible **`lang/cn.json`** 补 293 新 key + 11 drift + 15 未译 | ⬜ **下一步** |
| 6b | └ 清理 26 处孤儿译文（可选，无害） | ⬜ |
| 6c | └ 发 crucible-cn 0.9.0 | ⬜ 待冒烟验证后 |
| 5b | 修复已发布译文的格式类缺陷（2817+209 可脚本归一） | ⬜ |
| 5c | 裁决剩余真实术语分歧（59 处同名异译 + 20 处基底冲突） | ⬜ 部分已裁决，见第 8 节 |
| 7 | ember_cn lang 47 key + 小包补齐 → 发 1.1.0 | ⬜ |
| 9 | **真实 Foundry 世界冒烟验证**（管线改造后必做） | ⬜ |
| 8 | ember 战役正文分批翻译 | ⬜ 分多会话 |

### crucible compendium：✅ 已完成，不必再动

15 个包中 13 个内联 100%；`playtest` / `pregens` 剩下的 161 条是 actor 内嵌物品，
**由 babele 通用回退从已译包按名字自动取译文，残余为 0**。

> ⚠️ 不要因为 `validate_translations.py` 还显示 97% 就去补那 161 条。
> 先跑 `resolve_generic_fallback.py` 看真实残余。重复翻译会制造同名异译。

```powershell
python "$P\3-常用脚本\qa\resolve_generic_fallback.py" --repo "$P\2-Crucible汉化插件"
```

### crucible lang/cn.json 缺口（与 compendium 相互独立的一条工作线）

明细在 `5-其他内容/reports/crucible/lang_gap.json`：

| 项 | 数量 |
|---|---|
| NEW 新增 key（0.9.1→0.10.1） | 293 |
| DRIFT 英文原文改动 | 11 |
| UNTRANSLATED（cn 与 en 完全相同） | 15 |
| STALE 上游已删除 | 32（可清） |

`ACTION.TAG.MovementBurrow` 的亚美尼亚字符污染已修（`掘穴 շարժ作` → `掘穴`）。

### 待清扫的既有缺陷（不阻塞发版，但发版前应处理）

| # | 问题 | 范围 |
|---|---|---|
| A | ~~`lang/cn.json` 的 `Burrow` 含亚美尼亚字符 `շարժ`~~ | ✅ 已修 |
| B | `Toughness` / `Fortitude` 同译「坚韧」 | lang + 全库正文 |
| C | `Electricity`：lang「电力」vs 词缀名「电击」 | 词缀名 3 条 |
| D | `Tier` 有 阶/阶级/阶位 三种；essence 有 精髓/精华 两种 | 全库正文 |
| E | 已发布译文双语格式不一致 209 处、同名异译 59 处 | 见 `glossary_ec.disputes.json` |
| F | 26 处孤儿译文（上游已删除的物品） | 无害，可清 |

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
