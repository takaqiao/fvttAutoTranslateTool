# Crucible 汉化升级工作流

本文档描述把已有汉化升级到新版 Crucible 系统的完整流程。

## 目录结构约定

```
crucible-cn/
├── babele-register.js     # Babele 模块注册 + 自定义 converter
├── module.json            # 翻译模组清单
├── compendium/
│   ├── en/                # 旧版基线英文 (如 0.9.0)
│   ├── cn/                # 旧版基线中文 (双语: 中文+英文内嵌, 工作源)
│   ├── en_new/            # 新版英文 (extract 工具生成, 不覆盖 en/)
│   └── cn_new/            # 新版合并中文骨架 (merge 工具生成)
├── lang/
│   ├── cn.json            # UI 字符串中文
│   └── en.json            # UI 字符串英文
├── scripts/
│   ├── extract_en_compendium.mjs    # 从 Crucible 系统 LevelDB 抽取英文
│   ├── merge_cn_translation.py      # 合并旧中文到新英文结构
│   ├── find_untranslated_english.py # 查找未翻译的英文残留
│   └── normalize_adventure_translation.py
└── docs/
    └── workflow.md        # 本文件
```

> **重要**：`compendium/cn/` 是**双语工作源**（description 里中英并存）。发布时才剥离英文。别直接把这个版本用在 release 包里。

## 版本映射

| 工作目录字段 | 内容 | 典型版本 |
|---|---|---|
| `compendium/en/` | 基线英文 | Crucible 0.9.0 |
| `compendium/cn/` | 基线中文 (双语) | 对应 0.9.0 |
| `compendium/en_new/` | 抽取产物 | Crucible 0.9.1 |
| `compendium/cn_new/` | 合并产物 | 0.9.1 结构 + 复用 0.9.0 中文 |

每次完成一轮升级后:
1. `cn_new/` → `cn/`  (成为新基线)
2. `en_new/` → `en/`  (成为新基线)
3. 删除 `*_new` 目录
4. 更新 `module.json` 的 `version`

## 完整升级流程

### 0. 前提
- 新版 Crucible 系统解包到某个目录 (示例: `C:/Users/Taka/Desktop/fvtt/crucible/`)
- `classic-level` 已安装在 `C:/Users/Taka/Desktop/fvtt/node_modules/` (抽取脚本会从那里 resolve)

### 1. 抽取新版英文

```bash
cd C:/Users/Taka/Desktop/fvtt/crucible-cn
node scripts/extract_en_compendium.mjs
```

默认读 `C:/Users/Taka/Desktop/fvtt/crucible/packs/`，写 `compendium/en_new/`。

参数:
- `--system <path>` : Crucible 系统目录 (含 system.json 和 packs/)
- `--out    <path>` : 输出目录

脚本会:
- 按 `system.json` 里的 packs 列表依次处理
- 根据 pack 类型 (Item/Actor/JournalEntry/Adventure/ActiveEffect/Macro) 应用对应的 Babele mapping
- 自动从 LevelDB 按 key 前缀分桶 (`!items!`, `!actors!`, `!actors.items!`, `!journal!`, `!journal.pages!`, `!journal.categories!`, `!adventures!`, `!effects!`, `!macros!`, `!folders!`)
- Item pack 自动检测是否含 `system.actions` 决定用简单或带 actions 的 mapping
- 如遇系统新增的 pack (比如 0.9.1 加了 `affixes` 和 `macros`)，会自动按类型处理

### 2. 合并旧中文到新结构

```bash
python scripts/merge_cn_translation.py
```

默认:
- `--old-en` = `compendium/en/`
- `--old-cn` = `compendium/cn/`
- `--new-en` = `compendium/en_new/`
- `--out`    = `compendium/cn_new/`

合并规则 (按 JSON 路径逐字段):
- 新英文字段 == 旧英文字段 **且** 旧中文在该路径有值 → 复用旧中文
- 新英文字段 != 旧英文字段 → 保留新英文 (说明原文改了，需要重译)
- 新英文字段在旧英文中不存在 → 保留新英文 (说明是新增)

`label` / `mapping` / `folders` / `entries` 和任意嵌套 dict/list 都按同一套规则递归处理。粒度到**每个字符串叶子**，所以同一个 entry 里没改的字段仍然是中文，只有改动的那一个字段变成英文。

输出里会打印各文件的：总字符串数 / 复用 / 改动 / 新增 数量，以及总复用率。

### 3. 审阅 cn_new

重点看:
- rules 通常改动最多 (系统文档经常重写)
- 新增的 pack (affixes / macros 等) 全是英文
- 某个 entry 被完全删除了 → 在 cn_new 里不会出现，不用管
- 某个 entry 被重命名了 → 旧名在 cn_new 不出现，新名以全英文出现 → 如果是重命名，手动把旧 cn 条目的内容移过去

可用辅助脚本:
```bash
python scripts/find_untranslated_english.py
```

### 4. 质量优先翻译流程

**核心原则：质量 > 速度。上下文对齐是 FVTT 翻译的基准。** 以下每一步都要跑完才进入下一步，不要跳。

工具和配置都在 `crucible-cn/` 根目录，所有命令都从这里启动（`cd C:/Users/Taka/Desktop/fvtt/crucible-cn`），因为 translator 和 calibrator 都写死了从 cwd 读配置。

**4.1 术语预热 — `extract_glossary_only`**

`glossary_adaptive_crucible.json` 可能滞后于新版系统（affixes、新装备、新 talent 的术语还没进术语表）。先让模型扫一遍新内容提术语，后续翻译才有对齐依据。

```bash
cd C:/Users/Taka/Desktop/fvtt/crucible-cn
# 改 crucible_simple_config.json: run_mode = "extract_glossary_only"
python ../翻译工具/crucible_translator.py
# 完成后检查 glossary_adaptive_crucible.json 新增条目
```

**4.2 主翻译 — `translate`**

```bash
# 改 crucible_simple_config.json: run_mode = "translate"
python ../翻译工具/crucible_translator.py
```

compendium 默认双语输出（`keep_original = not is_system_lang_file`），`lang/cn.json` 按 path marker 走纯中文。

**4.3 漏词扫描 + 修复 — `repair_from_report`**

```bash
python scripts/find_untranslated_english.py --output release/untranslated-english-report.json
# 改 crucible_simple_config.json: run_mode = "repair_from_report"
python ../翻译工具/crucible_translator.py
```

修复阶段会注入 entry 级上下文（`repair_context_enabled: true`, `repair_context_max_chars: 12000`）。

**4.4 ~~质量审校 — `quality_review_and_repair`~~ 已封存 ⛔**

> **不要跑这一步。** 实测成本失控：单次跑会先扫 4629 条术语全表审校，再对每条带 12K entry 上下文 + 双语备份源 + 二次复审。短短 35 条候选已能烧掉数十美元。功能上 4.5 calibrator 已完全覆盖（batch 分组的上下文对齐反而更强），数值/markup 守卫靠 4.6 人工校对兜底即可。
>
> 保留配置项是为了万一需要单条字段定点修复时手动跑，**正式流程跳过**。

**4.5 ~~批次校准~~ → 已挪到 §5 剥离双语之后**

calibrator 不是 bilingual-aware（dry-run 实测会把 `<p>CN</p>\n<p>EN</p>` 双语剥成纯中文，并且改短中文）。所以**先剥离双语，再跑 calibrator**：在纯中文上对照 en_new 源校准，calibrator 擅长的就是这种格式。具体命令见 §5。

**4.6 人工校对**

剥离 + calibrate 跑完后，人工扫 `calibration_report_cn_new.json` + diff，重点看：
- rules / playtest 的长文本（系统改动最多）
- affixes 整包（全新）
- 数值、骰子、Fortune/Misfortune、@UUID/@Check 等宏标记是否被错改

### 5. 剥离双语 + 校准 + 替换基线 + 发版

#### 5.1 备份 cn_new 双语工作源

```bash
cp -r compendium/cn_new compendium/cn_new.bilingual.bak
```

这是双语工作源的最终备份，**永远不要丢**。下次升级要复用旧中文时还是从这里读。

#### 5.2 剥离双语（在 cn_new 原地剥）

```bash
python C:/Users/Taka/Desktop/fvtt/清洗修复工具/strip_bilingual_english.py \
    --target compendium/cn_new --recursive --in-place \
    --backup-dir compendium/cn_new.strip.bak \
    --report strip_bilingual_english_report.json
```

⚠️ strip 是内容启发式，不按 key 白名单。它只在"中文段 + HTML 平衡的英文段"结构里剥离，单行 name 类短字段天然保留。剥完后 diff 一遍 `cn_new.strip.bak` 看有没有误伤。

#### 5.3 calibrator 校准（剥离后的纯中文对照 en_new）

```bash
cd C:/Users/Taka/Desktop/fvtt/crucible-cn
# 先 dry-run 看前几个 batch
python ../翻译工具/crucible_calibrator.py
# 检查 calibration_report_cn_new.json 没问题再 apply
python ../翻译工具/crucible_calibrator.py --apply
```

配置：`crucible_calibrate_config.json`（`file_pairs` 预置 14 个 en_new ↔ cn_new 对）。剥离后 cn_new 是纯中文，calibrator 不会再误删英文，可以放心 apply。

#### 5.4 人工校对 + 替换基线

按 §4.6 扫报告 + diff 后再做替换：

```bash
# 备份旧基线
cp -r compendium/en compendium/en.bak
cp -r compendium/cn compendium/cn.bak

# 替换
rm -rf compendium/en compendium/cn
mv compendium/en_new compendium/en
mv compendium/cn_new compendium/cn

# 更新 module.json 的 "version"
```

> **重要**：`compendium/cn` 现在是**纯中文**的发版基线。下次升级 (§2 merge) 仍然要从 `compendium/cn_new.bilingual.bak` 读双语，因为 merge 需要原始英文做字段比对。

## 脚本说明

### extract_en_compendium.mjs
- 只读，不写 compendium/en/
- 幂等，可反复运行
- 默认输出到 compendium/en_new/
- 依赖 `classic-level` (从 fvtt/node_modules 复用)

### merge_cn_translation.py
- 只读 en/ cn/ en_new/，只写 cn_new/
- 幂等
- 纯 Python 标准库，无外部依赖

## 常见问题

**Q: en_new 里出现了 en/ 没有的 pack**
A: 正常，新版系统加了 pack。需要在 babele-register.js 里注册新 pack (如果需要自定义 converter)。

**Q: merge 报告显示某个文件 "no old EN"**
A: 全新 pack，所有条目都会以英文形式出现在 cn_new 里。

**Q: merge 报告显示某个 pack "changed" 数量很大**
A: 该 pack 的很多字段英文原文被改写了。这些需要重新翻译。常见于 rules/playtest。

**Q: 旧 cn 是纯中文 (剥离过英文) 可以当基线吗？**
A: 可以，但会损失原始英文信息。建议用双语版本作为基线。

**Q: merge 是按 entry 粒度还是字段粒度？**
A: **字段粒度**。同一个 entry 的 name 和 description 可以分别被复用/保留英文。
