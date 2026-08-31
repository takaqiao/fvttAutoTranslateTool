# Grayce 翻译进度报告

**最后更新**: 2026-05-09（继续会话第二轮后）
**对象**: PF2e *Pathfinder: Secrets of Grayce* 入门选集（FVTT JSON）
**模式**: Claude 精校 in-session（Max 计划）+ 翻译流程 TM 管线

---

## 总体进度

| 文件 | 大小 | 残留前 → 残留后 | 完成度 |
|---|---|---|---|
| `pf2e.menace-under-otari-bestiary.json` | 262 KB | 268 → **0** | ✅ 100% |
| `pf2e.troubles-in-grayce-bestiary.json` | 210 KB | 393 → **0** | ✅ 100% |
| `pf2e-secrets-of-grayce.secrets-of-grayce.json` | 1.6 MB | 2165 → **183** | 🔶 91.5% |

**总进度**: 1982/2165 字段已翻译（**91.5%**）

---

## 已完成阶段

### 阶段 A：翻译流程脚本通用化
4 个 QA/扫描脚本 + apply_tm.py 全部参数化，支持任意模组目录。

### 阶段 B1：menace-under-otari-bestiary（100%）
- 所有 ~30 个 BB 怪物 + 数十个能力 + 物品名

### 阶段 B2：troubles-in-grayce-bestiary（100%）
- 所有危害（Trial by Ice/Flame/Poison Trap, Bittersweet Surprise, Death of the Dollmaker 等）+ 怪物（Path River Peeler, Sangrist, Gauntling, Sticky Toffee Hound, Aezar 等）

### 阶段 B3：secrets-of-grayce 主体（88.5%）

**已完成**:
- ✅ 顶层 entry name + caption + label + 14 folders（双语）
- ✅ 全部 229 actor names + tokenName + prototypeToken + blurbs
- ✅ 全部 14 journal entries 顶层名 + 269 page names + 旁注（双语）
- ✅ 全部 15 scenes + drawings 标签
- ✅ 全部 209 actor item names
- ✅ 短危害字段（hp/ac/disable/reset/stealthdetails/routine 等）
- ✅ 全部 50 publicNotes（41 unique，含 Gargoyle/Harpy/Soulbound Doll 等长篇）
- ✅ 全部 67 hazarddescription（GM 视角宝藏框，含珠宝/卷轴/具名物品列表）
- ✅ 全部 168 actor item descriptions（怪物能力详解）
- ✅ 21 random encounter table results（Bastardhall 表）
- ✅ 27 个最短的 journal text 已译（含 Grayce Graveyard, Chapel Bridge, East Docks, Apple Tree Stables, Adventure Toolbox 等）
- ✅ 通过跨文件 TM 复用（含归一化匹配）从 tig/muo 港 ~256 项

### 阶段 C：术语沉淀
- `glossary.json`：10847 → 10942（**+95 新词条**）
- 备份：`gracye/_backup/20260509_014304/`

### 阶段 D：QA 验证
- ✅ 3 个文件 JSON 全有效
- ✅ Total strings 数量保持不变
- ⚠️ 1 处 HTML 失衡（轻微，需人工检查）
- 自然增减的 UUID/HTML 计数（中文翻译引用了更多本地化条件，符合预期）

---

## 待译内容（journal text，~810K 字符）

剩余 250 残留中：
- **journal `text` 字段**：约 246 处（~810K 字符）— Grayce 城镇地理志正文 + 6 个冒险章节正文
- 几个零散 description / publicNotes / hazarddescription 因编码字符细差未应用

**预估剩余工作量**：8-12 个会话（每会话翻译 20-30 个 text 字段）

### 已译 vs 待译 journal text 分布

本次"继续"会话又翻译了 67 个 journal text 字段（公会厅、城堡区域、商店、环境规则、事件遭遇等）。共译 ~94/276 journal text。

- 多个 Grayce 镇地理位置（已开始）：Apple Tree Stables, Dalmira's Bakery, Grayce Graveyard, Chapel Bridge, East Docks, Gravedocks, Carpenter's Hall, Town Hall (Tollhouse) 等
- 6 个冒险章节正文（待译）：The Path River Peeler, Sabotage!, They Hunger Below, The Disgraced Dollmaker, Sweet Tooth, The Count of Petals
- GM 指南章节（待译）：Preparing an Adventure, Character Creation, Difficulty, Optional Rules 等
- 暗星之秘神庙（Beginner Box 章节，待译）

---

## 下次会话续接说明

```bash
# 1. 查看当前残留
python 翻译流程/scripts/scan_residue.py gracye/

# 2. 选择批次：建议每次 20-30 个 journal text
# 3. 翻译策略：用 Edit 直接处理逐个 page text，或写 Python apply 脚本批量

# 4. 完成一批后扫描
python 翻译流程/scripts/scan_residue.py gracye/

# 5. 全部完成后 QA
python 翻译流程/scripts/audit_translations.py gracye/
python 翻译流程/scripts/qa_check.py gracye/ gracye/_backup/20260509_014304
```

**当前临时脚本**（可保留作未来参考或删除）：
- `_tmp_apply_muo.py`, `_tmp_apply_tig.py` — bestiary 翻译应用
- `_tmp_apply_sog_actors.py` 系列（tier 1-5）— sog 结构翻译
- `_tmp_apply_sog_pns.py`, `_tmp_apply_sog_pns_v2.py` — publicNotes
- `_tmp_apply_sog_hd.py` — hazarddescriptions  
- `_tmp_apply_sog_descs.py` — actor item descriptions
- `_tmp_apply_sog_uuid_labels.py` — @UUID label 翻译
- `_tmp_apply_sog_text1.py` — 第一批 journal text
- `_tmp_sog_port_from_tig.py`, `_tmp_sog_port_normalized.py` — 跨文件 TM 复用
- `_tmp_glossary_grayce.py` — glossary 更新
