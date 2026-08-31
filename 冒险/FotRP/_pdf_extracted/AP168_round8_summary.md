# Round 8 — 跨书一致性扫描 + Wave drift 清理 (book 1 / book 2 / bestiary)

**会话日期**: 2026-05-19 (延续 Round 1-6)
**方向选择**: A1 (workflow_and_continuation.md §六) — 跨书一致性
**目标**: 用 bestiary 作 operational canonical, 把 book 1 / book 2 残留分歧对齐; 同时清 Round 5 没扫到的 b1/b2 Wave drift

**修改总数**: 86 处 (跨 8 个文件)
**QA**: JSON 全部 valid + HTML 全部平衡 + 残留 0

---

## 一、扫描方法

`round8_cross_book/` 目录下 9 个脚本:

1. **scan_bilingual_pairs.py** — 初步扫描 (失败, 因 JSON 全中文化, 双语对照稀疏)
2. **diag.py / diag2.py / diag3.py / diag4.py** — 诊断 JSON 结构 + name 字段格式
3. **build_name_map.py** — 从 bestiary 138 个 entries 抽出 108 个 actor 双语 name 映射
4. **full_scan.py** — 三大检查:
   - 历史 Wave canonical drift (b1+b2 未扫过, Round 5 仅扫 b3)
   - Bestiary canonical 名跨 journal 一致性 (en 出现, zh canonical=0 嫌疑分歧)
   - Bestiary 内部 3 处分歧
5. **inspect_divergences.py** — 嫌疑分歧的实际上下文 (确认真分歧 vs 假阳性)
6. **check_glossary.py** — glossary.json + glossary_wiki.json 核查 (这些 AP-specific NPC 名 glossary 静默)
7. **check_pf2_terms.py** — pf2cn 核查 (PF2 通用术语)
8. **check_dimension.py** — 维度 上下文区分 (prose vs Syndara 固定能力)
9. **check_tournament.py** — 锦标赛 上下文确认 (39 处全可 bulk replace)
10. **precise_count.py** — 候选 precise count
11. **investigate_bestiary.py** — 解决 bestiary 内部不一致剩余 1 处疑问
12. **apply.py** — 备份 + 应用
13. **qa.py** — JSON + HTML + 残留 QA

---

## 二、修改清单

### A. NPC 跨书命名分歧 (book 1/2 → bestiary canonical, 共 38 处)

权威裁决: 这些是 AP-specific NPC, pf2cn/AlphaStar/wiki/glossary 全静默. Bestiary 是 Foundry import 目标 (operational canonical), 故 journals 对齐 bestiary.

| 分歧 | book 1 用法 | bestiary canonical | 改后 | 处数 |
|---|---|---|---|---|
| Halspin the Stung | 蜂蛰哈尔斯丁 | 哈尔斯丁·刺痛 | 哈尔斯丁·刺痛 | 1 (b1_back) |
| Umbasi | 乌姆巴西 | 乌木巴西 | 乌木巴西 | 2 (b1_back) |
| Lantondo | 兰通多 | 兰托多 | 兰托多 | 10 (b1_back ×6, b2_back ×4) |
| Artus Rodrivan | 阿托斯·罗德里万 / 阿托斯 | 阿图斯·罗德里万 | 阿图斯·罗德里万 / 阿图斯 | 5 (b1_back ×3, b2_back ×2) |
| Ahmoza Twins | 阿姆扎双胞胎 / 阿姆扎 | 阿莫扎双子 | 阿莫扎双子 / 阿莫扎 | 8 (b1_ch1 ×3, b1_ch2 ×3, b1_back ×1, b2_back ×1) |
| Krankkiss | 克兰基斯 | 库拉克丝 | 库拉克丝 | 4 (b1_ch2) |
| Numoriz | 努莫里兹 | 路莫里兹 | 路莫里兹 | 5 (b1_ch2) |
| Paunnima | 波尼玛 | 保宁玛 | 保宁玛 | 3 (b1_ch2) |
| Rajna | 拉吉娜 | 拉贾娜 | 拉贾娜 | 4 (b1_ch2 ×3, bestiary 内部 ×1) |
| Jun | 君（Jun | 俊（Jun | 俊（Jun | 1 (b1_ch2) |

注: 阿托斯 / 阿姆扎 短形式分两轮 — 长全名先, 短形式后.

### B. Wave 历史 canonical drift 清 (b1+b2 leftover from Round 5)

| 修改 | 处数 | 备注 |
|---|---|---|
| 锦标赛 → 武道会 | 39 | b1_ch1 ×2, b1_back ×16, b2_ch1 ×6, b2_ch2 ×4, b2_ch3 ×1, b2_back ×10. Wave 7 N canonical, 39 处全部 context safe. |
| 跨维度场景 → 跨位面场景 | 1 | b1_back. Wave 8 BK (dimension→plane) prose. book 3 ch2/ch3 维度=7 全部是 Syndara 固定能力 (维度叠加/吞噬/连击/暗面之镜) — Wave 8 BK 保留. |

### C. Bestiary 内部分歧 (3 处)

| 分歧 | 改法 | 理由 |
|---|---|---|
| 娜迦毒液 / 娜迦裔毒液 | → 娜迦裔毒液 | pf2cn `TraitNagaji` = 娜迦裔, 与 trait canonical 一致 |
| 锚定射击 / 固定射击 | → 固定射击 | bestiary 3:1 majority + book1_ch1 一致 |
| 巨人摔角术 / 巨人摔角手 | → 巨人摔角手 | bestiary 2:1 majority + PF2 feat 称谓风格 (人称) |

---

## 三、保留 (不改) 的决策

### 3.1 Round 8 没做的项 (留给后续轮)

- **真气 → 斗气** (b1 ch1=2, b1 ch2=5, b1_back=11, b2 ch2=1, b2_back=4, b3_ch1=5, bestiary=10) — 需 prose context 判别 (pf2cn 法术名 真气 不可换). 留 A2 下轮.
- **选手 → 参赛者** (b1_back=1, b3_ch1=12, b3_ch2=2, b3_ch3=3, b3_back=2) — 「晋级选手」是 Wave 8 BK 保留固定术语, 需上下文判别. 留 A2.
- **您 → 你** (b3_ch1=1) — 单一处, 留 A2.

### 3.2 跨书但已一致 (无须改)

通过 cross_book_consistency.txt 验证, 以下 NPC 跨书一致:
- 雕匠辛达拉 (Syndara): 全部 53 处 ✓
- 千野·董 (Tino Tung): 全部 9 处 ✓
- 鹰虎 (Takatorra): 全部 18 处 ✓
- 苏塔奴 (Syu Tak-nwa): 全部 31 处 ✓
- 苏吉特·哈梅兰 (Surjit Hamelan): 全部 3 处 ✓
- 「正义的」亚彬 (Yabin the Just): 引号风格略异 (b1/b2 「」 vs bestiary "") 但中文一致 — 不改

### 3.3 不可改 (固定能力名 / Wave-stable)

- Syndara 维度叠加 / 维度吞噬 / 维度连击 / 维度抓握 / 维度暗面之镜 (book3) — Wave 8 BK 保留 [[fotrp_progress]]
- Hwanggot 花国 (主体) — glossary canonical
- Glass Lighthouse 琉璃光屋 — glossary canonical
- 飞行山脉 / 凯芬湾 — 项目内 Wave-stable, glossary 静默

---

## 四、跨书 NPC 命名分歧的根因模式

分析 11 处 NPC 分歧, 共两类模式:

### 4.1 早期书 (b1/b2) 用旧译名, bestiary 用更新译名
原因: bestiary 是 Wave 4-6 期间用 PDF 辅助 refined 的, books 1/2 停留在 Wave 1-3 早期版本.

例: 阿托斯 (旧 b1/b2) vs 阿图斯 (bestiary 修正)

### 4.2 Book 1 Ch 2 (格拉利昂最强队 backstory 段) 独立旧版本
b1_ch2 有一长段叙事介绍 8 名队员, 用了旧译名:
- 克兰基斯 / 努莫里兹 / 波尼玛 / 拉吉娜 / 阿姆扎双胞胎 / 君 (5 名队员 + Ahmoza twin 别名 + Jun)
这段叙事可能从原始翻译沿袭, 而 b2_back + bestiary 后来修正了名单格式 (用 "<li>名（English）" 风格).

---

## 五、影响 + 下一步

### 5.1 用户影响
- **Foundry 端**: bestiary 导入的 actor token 名不变 (除内部 description 文本里 3 处). 所以 import 后角色名一致.
- **Journal 端**: GM 阅读 backstory / handouts 时, NPC 名前后一致, 不会出现 "阿托斯" 突然变 "阿图斯".
- **跨书叙事**: 玩家 / GM 在 book 1 看到 backstory, book 3 看到角色登场 — 现在中文名一致.

### 5.2 下一步候选 (workflow_and_continuation.md §六 中尚未做的)

**A2 (Wave 7-8 manual prose)**: 真气 / 选手 / 您 — prose context 判别复杂, 但价值高 (~50 处). 建议下一轮.

**A3 (PDF 揭示具体段落)**: book 3 已做; book 1 / 2 OCR 未做. 价值中等.

**B1 (Bestiary 其他 9 个怪物名 wiki 核查)**: Canopy Elder / Gumiho / Inmyeonjo / Lophiithu / Orochi / Sanzuwu / Spirit Turtle / Sthira / Desecrated Guardian — 应跟 Round 6 (Bul-Gae → 天蚀狗) 一样查 wiki canonical.

**B2 (Stat block 数值核对)**: 抽样 boss 数据卡与 PDF 数字对照. 价值高, 工作量中.

---

## 六、产物清单

`_pdf_extracted/round8_cross_book/` 目录:
```
scan_bilingual_pairs.py             — 初步扫描 (失败 baseline)
diag.py / diag2.py / diag3.py / diag4.py — JSON 结构诊断
build_name_map.py + bestiary_name_map.txt — 108 actor 双语 name 映射
full_scan.py                        — 三大检查执行器
  → drift_scan.txt                  — Wave drift (b1/b2 leftover)
  → cross_book_consistency.txt      — 跨 journal NPC 一致性
  → internal_inconsistencies.txt    — Bestiary 内部 3 处分歧
inspect_divergences.py + divergence_contexts.txt — 嫌疑分歧 context
check_glossary.py + glossary_check.txt — Authority hierarchy 查询
check_pf2_terms.py + pf2cn_check.txt   — PF2 通用术语核查
check_dimension.py + dimension_contexts.txt — 维度 prose/固定能力区分
check_tournament.py + tournament_contexts.txt — 锦标赛 39 处 context 验证
investigate_bestiary.py + inv_out.txt — Bestiary 内部 1-hit 调查
precise_count.py + precise_count.txt — 86 处候选 precise count
apply.py + apply_report.txt         — 应用 + 报告
qa.py + qa_report.txt + qa_out.txt  — QA 验证
```

`_backup/20260519_round8_cross_book/` — 8 文件原始备份 (按需 rollback).

---

## 七、教训

### 7.1 假阳性: bestiary 用 "（八臂调和）" 后缀, journal 用 base 名
我的 cross_book_consistency 报告把 Arms of Balance 4 子成员标为"嫌疑分歧", 但实际 bestiary canonical 是 "兰雅·什瓦纳特丝（八臂调和）" 含队名后缀, journal 用 "兰雅·什瓦纳特丝" 不含后缀 — 这是结构差异不是命名分歧. 类似 Golarion's Finest 7 子成员.

教训: 跨文件 NPC name 检查时, 应允许 bestiary canonical 的"队名后缀"模式, 比对剥离后缀的 base 名.

### 7.2 HTML parser 自闭合标签
Python HTMLParser 对 `<hr />` 触发 startendtag, 但默认实现是 starttag + endtag — 我的 TagBalance class 处理了 starttag (跳过 self-closing 起始) 但 endtag handler 把 `</hr>` 当作不平衡的关闭. 修复: override `handle_startendtag` 为 no-op + endtag 中检测 self-closing 直接 return.

### 7.3 替换顺序 — 长字串先
"阿托斯·罗德里万" 含 "阿托斯", 必须先替换全名再替换短名, 否则短名先吃掉 prefix 导致全名匹配失败. apply.py 严格按长度顺序.

### 7.4 数据驱动的扫描 vs 启发式列表
本轮发现 11 处 NPC 分歧远超我最初设想的 4-5 处. 用 bestiary canonical 作 ground truth 自动对比 journal, 比手列 candidate term 更全. 推荐成为下次跨书扫描的标准操作.
