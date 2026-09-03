# FVTT 模组汉化通用流程

把 FVTT 导出的英文 JSON 模组（journal/bestiary/addons 等）按 PF2e 中文社区惯例翻成中文，且在 FVTT 加载即用、达到精校质量。本流程从 FotRP（赤凰斗士）项目沉淀而成，已在 5 个 merged 文件（合计 ~5100 个新增字段、~4500 条 SRD 引用）上跑通并 0 残留通过 QA。

## 0. 适用范围

- **输入**：从 FVTT 导出（或上游 merge 推送）的英文 JSON 模组文件，含 entries/actors/items/folders 嵌套结构
- **目标**：原地把英文字段替换为中文，name 字段保留 `中文 English` 双语格式，HTML/UUID/Foundry enricher 全部不动
- **不适用于**：纯散文 PDF 翻译（用单独的 PDF→docx 管线）；冒险路线的"完全汉化版"PDF（用 PyMuPDF 抽取 + 段落对齐管线）

## 1. 翻译记忆（TM）来源优先级

每次翻译开始前，先从权威来源构建 TM 查找表，再用 TM 命中 → 旧项目记忆复用 → 剩余内容人工翻译。

**优先级（冲突时高级别覆盖低级别，即"高级别先查、低级别兜底"）：**

```
PF2 中文 Wiki > pf2e_compendium / pf2_cn > pf2e-compendium-extra-cn > 其他来源
```

其中 `pf2e_compendium` 的精确条目与 `pf2_cn` 属于同一层；两者冲突时，精确合集条目优先于从 i18n 键推导出的候选。`pf2e-compendium-extra-cn` 作为既有项目记忆，以稳定 Foundry ID 和完全一致的英文原文复用，不能按易变名称盲目覆盖新版内容。

具体路径：

| 优先级 | 源 | 路径 | 内容 | 条目量 |
|---|---|---|---|---|
| 最高 | **PF2 中文 Wiki** | `pf2wiki-scraper/out/glossary_wiki.json`、离线镜像或经核对的 Wiki 单页 | 社区现行规范译名；自动抓取结果必须先校验 | ~16K |
| 核心 | **pf2e_compendium (非 extra)** | `模组/pf2e_compendium_chn/compendium/pf2e.*.json` | PF2e 核心合集条目（武器、法术、护甲、专长等） | ~28K |
| 核心 | **pf2_cn** | `模组/pf2_cn/zh_Hans/*.json` | 系统 i18n UI 字符串与部分核心术语 | ~6K 可推导项 |
| 项目 | **pf2e-compendium-extra-cn** | 当前汉化包和历史版本 | 第三方模组既有译文；按稳定 ID / 完全一致原文复用 | 依项目而定 |
| 兜底 | **其他经审阅来源** | `术语表/glossary.json`、汉化 PDF、人工校对 | 仅在以上来源均无结果时使用 | 依来源而定 |

**为什么是这个优先级**：Wiki 的经核对页面用于确定社区现行译名；核心系统与合集提供完整规则文本；extra 项目记忆用于保留模组专名和已校对自定义内容。自动抓取的 Wiki 词表仍可能含噪音，因此“Wiki 最高”指已经核对过的页面或离线数据，不代表无条件信任未经校验的抓取结果。

**"非 extra" 是什么**：`pf2e_compendium/en-US/` 有 166 个文件，其中 73 个是 `pf2e.*` 核心 SRD（要用），93 个是第三方/同人模组（`battlezoo-*`、`botanical-bestiary.*`、`clerics.*`、`magus.*`、`impossible-lands.*`、`kctg-2e.*` 等，**不要用**——它们术语不规范、翻译质量参差）。

### 1.1 wiki 不稳定的应对策略

`pf2wiki-scraper/out/glossary_wiki.json` 由本地 scraper 跑出，**已知不稳定**（HTML 解析有失败、ZH/EN 配对有误标）。两种处理路径：

**路径 A（推荐，按需核对）**：对重命名、专名和争议术语逐条核对 PF2 中文 Wiki（`https://pf2.huijiwiki.com`），把确认结果保存成小型、可审计的项目词表；其余规则文本由核心合集补全。

**路径 B（Claude 手工校对，零外部调用）**：完全用 glossary.json 兜底，遇歧义术语在会话里向用户确认。零成本但可能与汉化组译名偏差。

**路径 C（重建 scraper，整页结构化）**：一次性投入大但后续稳定。仅在长期多项目使用时划算。

FotRP merge 流程实测：A 模式最经济。

### 1.2 Claude 手工裁决补充层

无论 wiki 是否命中，对**proper noun / 模组独有术语 / 跨文件出现频次 ≥ 3 的争议术语**应该过一次 Claude 人工核查：
- 跨文件搜索现有翻译（同名/相关名是否已有共识）
- 检查 glossary.json
- 与汉化组 PDF（如有）比对
- 实在歧义时让用户决断

这个步骤在阶段 7（术语一致性核查）跑一次即可。

## 2. 端到端管线

### 阶段 0：准备

```
mkdir -p _backup _tmp_cache
```

每次原地修改前把 OLD 备份到 `<工程>/_backup/<file>.<timestamp>.json`。

### 阶段 1：构建 3 源 TM

```bash
python 翻译流程/scripts/build_3source_tm.py
```

输出：`翻译流程/tm_cache/tm_3source.json`

每条目的结构：
```json
{
  "Halberd": {
    "name": "戟 Halberd",
    "description": "<p>...</p>",
    "source": "wiki | pf2e_compendium | pf2_cn",
    "all_sources": {"wiki": "戟", "pf2e_compendium": "戟 Halberd", ...}
  }
}
```

冲突时经核对的 Wiki 结果胜出；所有候选仍保留在 `all_sources` 供人工复核。

### 阶段 2：应用 TM（多策略 lookup）

```bash
python 翻译流程/scripts/apply_tm.py <input.json> <output.json>
```

按以下策略依次匹配 item key（直到命中）：

1. **直接命中**：`Halberd` → TM
2. **剥 FVTT 4 字符 ID 后缀**：`Halberd (K9Yf)` → 剥 `(K9Yf)` → `Halberd`
3. **剥所有括号短语**：`Plane Shift (Self Only)` → `Plane Shift`、`Bottled Lightning (Greater) (Infused)` → 迭代剥到 `Bottled Lightning (Greater)`
4. **反序 "Greater X" ↔ "X (Greater)"**：`Greater Alchemist's Fire` → `Alchemist's Fire (Greater)`
5. **符文剥离**：`+3 Greater Striking Kanabo` → `Kanabo`（先剥 `+\d+`，再剥 `Major|Greater|Lesser|Minor` + `Striking|Resilient|Disrupting|...`）
6. **撇号变体**：`Wild Wind's Gust` ↔ `Wild Winds Gust`
7. **后缀状态剥离**：`Charm (At Will)` → `Charm`、`Air Walk (Constant)` → `Air Walk`、`Telepathy 100 feet` → `Telepathy`
8. **Lore 模式合成**：`<X> Lore` → `<X 的中文翻译>学识 <X> Lore`（**注意：用「学识」不用「知识」**——PF2e zh-CN 标准是 `Lore` 单独时用「知识」，复合的 `<X> Lore` 用「`<X>`学识」）
9. **法术传统合成**：`Spontaneous Occult Spells` → `自发神秘法术`（前缀 + 传统 + 法术）
10. **自然攻击/通用 NPC 词条小字典兜底**：Claw / Jaws / Tail / Beak / Talon / Foot / Hoof / Tentacle / Pseudopod / Bite / Wing / Stinger / Mandibles / Tusk / Pincer / Slam / Spike

### 阶段 3：合并新旧（同上游 merge 时使用）

如果上游推了带旧译的 NEW 文件 + 新增内容，先 port 公共路径的旧值，再处理新增：

```bash
python 翻译流程/scripts/port_old_to_new.py <OLD.json> <NEW.json> <merged.json>
```

### 阶段 4：跨 entry 副本 port（处理 hmLe 类重复）

某些上游会推 entry 的 `(XXXX)` 副本，actor key 大量重叠。先 port 主 entry 的中文到副本：

```bash
python 翻译流程/scripts/port_entry_to_dup.py <file.json> <main_entry_key> <dup_entry_key>
```

### 阶段 5：人工补译剩余

经过阶段 2-4 后，剩下的就是模组真·新增内容（独有 NPC、特有能力、任务相关 publicNotes 等）。在会话里逐字段译。

约定：
- `name` 字段：`中文 English` 双语格式（空格分隔）
- `description`/`publicNotes`/`blurb`：纯中文
- `<p>`/`<strong>`/`<hr>`/`<em>` 等 HTML 标签：原样保留（包括 `<hr>` vs `<hr />` 差异——不要"规范化"，否则破坏 exact-match 替换）
- `@UUID[...]{label}`、`@Check[skill|dc:N]`、`@Damage[Xd6[type]]`、`@Template[burst|distance:N]`、`[[/r ...]]`、`@Localize[...]`：所有 enricher 原样保留，只译 `{label}` 部分

### 阶段 6：QA 三件套

```bash
python 翻译流程/scripts/scan_residue.py <target_dir>        # 扫真英文残留（自动发现 *.json）
python 翻译流程/scripts/scan_short_residue.py <target_dir>  # 扫 ≤5 字符短英文（前一个会漏）
python 翻译流程/scripts/audit_translations.py <target_dir>  # HTML/UUID/双语格式审计
```

所有 QA 脚本都接受 target_dir 作位置参数，自动跳过 `_backup` / `_tmp` / `_cache` / `_qa_reports` / `_pdf` / `_zh_synthetic` / `NEW` 子目录。

通过标准：
- 真英文残留 = 0（已剔除 @Localize 等 runtime token）
- 短英文残留 = 0
- HTML 标签平衡 0 错误
- UUID/enricher 结构 0 错误

### 阶段 7：术语一致性核查

```bash
python 翻译流程/scripts/term_consistency_check.py
```

跨文件对照核心术语，找出 e.g. `Yai` 在 bestiary 译为「巨鬼」但在 addons 译为「夜叉」这类不一致。**修复时按已核准的完整层级裁决**：经核对的 PF2 中文 Wiki > pf2e_compendium / pf2_cn > extra 项目记忆 > 其他经审阅来源。

### 阶段 8：部署 / 终审

```bash
python 翻译流程/scripts/qa_check.py <target_dir> [old_dir]
```

仅传 target_dir：JSON 有效性 + 每文件 UUID/enricher/HTML 计数  
传 old_dir：和旧文件对比 enricher 完整性是否守住（merge 场景关键）

## 3. 残留分类速查表

| 模式 | 是否真残留 | 处理方式 |
|---|---|---|
| `@Localize[PF2E.NPC.Abilities.Glossary.X]` | **不是** | runtime auto-render，留英文 |
| `@Damage[Xd6[type]]` 内部 type 词 | **不是** | enricher 内部，留英文 |
| `@Check[skill\|dc:N]` 内部 skill 名 | **不是** | enricher，留英文 |
| `<UUID 的 alphanumeric ID>` | **不是** | 系统 ID，不动 |
| `prototypeToken: "Han"` 类 actor token name | **是** | 译为「韩 Han」类双语格式 |
| `caption: "Book 2"` 类 entry 标签 | **是** | 译为「第二本 Book 2」 |
| `hp: "80 BT"` 类机制缩写 | **是**（推荐双语） | 「80 体能回合 80 BT」 |
| 自然攻击（Claw / Jaws）裸名 | **是** | 双语：「爪击 Claw」 |
| 已翻译的 "皮甲 Leather Armor" | **不是** | 双语合规格式 |
| 法术 `<X> Lore` 类 | **是** | 「<X 中文>学识 <X> Lore」 |

## 4. 多文件 merge 套路

如果上游一次推多个 merged 文件（如 5 个 addon/bestiary 同时更新）：

1. **结构 diff** 先：`python 翻译流程/scripts/diff_structures.py NEW/ OLD/`
   产出报告：每个文件 added_paths / removed_paths / changed_paths
   关键判断：**common path 上的英文值是否与 OLD 一致**——一致则可以直接 port OLD 中文；不一致说明上游改了原文需要重译

2. **trivial 文件不动**：如果 added/changed 都是 0，文件除了格式外完全等价，跳过

3. **port → TM → 在会话里译**：按上面的阶段管线走

## 5. 已知陷阱

- **不要把 folder 路径 (`/folders/`) 加进 SKIP_PATH_PATTERNS**：folder name 是用户可见的，需要翻译。早期版本的 `_tmp_final_scan.py` 误把 folder 跳过，导致 `Book 2 - Ready? Fight!` 等长期未译
- **prototypeToken 字段类型是字符串（addon）vs 字典（bestiary）**：addon 的 `prototypeToken: "Melodic Squall"` 直接是字符串；bestiary 的 `prototypeToken: {"name": "..."}`。处理时要分别判断
- **某些 entry 会有 `(XXXX)` 4 字符 ID 后缀的重复副本**（如 `Fists of the Ruby Phoenix: Addons (Book 2) (hmLe)`）——actor key 大部分重叠，需要单独 port
- **count_zh 阈值 `< 2`**：因为 2 字 PF2e 中文术语很常见（如「皮甲」「偷袭」「弱化」），残留检查器把它当 valid 双语
- **count_en 阈值 `> 5`**：会漏 4-5 字的短词如 "Cloak" / "Lure" / "Jian" / "Rock" / "Braid"。补一个 tight scan 用 alpha_count >= 2 抓
- **HTML 不要规范化**：保留 `<hr>` vs `<hr />` 的原始差异。否则 dict-based exact-match 会丢
- **ConvertFrom-Json / json.dump 会改键序**：备份用 `shutil.copy2` 保留 mtime，直接 file copy，不要走 load+dump
- **PowerShell 输出中文乱码**：用 `python -c "import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')"` 包一层
- **「Lore」译名**：单独的 `Lore` skill 用「知识」；`<X> Lore`（具名 Lore subskill）一律用「`<X>`学识」（如「过峡学识 Goka Lore」「战争学识 Warfare Lore」）

## 6. 已沉淀的术语裁决（FotRP）

| 英 | 中 | 来源 |
|---|---|---|
| Yai | 巨鬼 | bestiary 旧译，跨多文件一致；不要用「夜叉」（误译） |
| kaiju | 怪兽 | wiki + 汉化组 PDF |
| Mogaru | 魔加鲁 | 汉化组 PDF |
| Goka | 过峡 | wiki + 汉化组 PDF |
| Bonmu | 邦木 | 旧译统一（不是「邦穆」） |
| Tian-Shu | 天舒 | 汉化组 PDF |
| Tian-Hwan | 天奂 | 汉化组 PDF |
| Tian-Sing | 天兴 | 汉化组 PDF |
| Lore（独立） | 知识 | PF2e zh-CN |
| `<X> Lore`（具名） | `<X>`学识 | PF2e zh-CN |

## 7. 脚本清单（`翻译流程/scripts/`）

| 脚本 | 用途 |
|---|---|
| `build_3source_tm.py` | 从 wiki + pf2e_compendium + pf2_cn 构建优先级合并 TM |
| `apply_tm.py` | 多策略 TM 应用（直接/剥后缀/剥括号/反序/符文剥离/Lore合成/spell-trad合成/natural-attack） |
| `port_old_to_new.py` | 上游 merge 后把 OLD 中文 port 进 NEW 公共路径 |
| `port_entry_to_dup.py` | entry 副本之间 port 重叠 actor 翻译 |
| `diff_structures.py` | 结构化 diff: added/removed/changed paths |
| `scan_residue.py` | 真英文残留扫描（剔除 enricher 内部，识别双语格式） |
| `scan_short_residue.py` | 短英文残留扫描（≤5 字符，补 scan_residue 漏过的） |
| `audit_translations.py` | HTML 平衡 + UUID/enricher 结构审计 |
| `term_consistency_check.py` | 跨文件术语一致性核查 |
| `promote.py` | 备份 OLD + 把 merged 结果部署到工作目录 |

每个脚本都设计为可独立调用，参数通过 CLI 或顶部常量传入。

## 8. 关于 glossary.json（项目根）

项目根的 `glossary.json` 是历史沉淀的"主术语表"（10K+ 条），来源混合（wiki + AlphaStarguide repo + 手工整理）。**它是补充资源，不是主 TM**——主 TM 永远从上面 3 源动态构建。`glossary.json` 用于：
- TM 三源都没命中的术语兜底
- 项目特定的人物名/地名（如 "Mogaru = 魔加鲁"）
- 跨项目共享的低频但关键术语
