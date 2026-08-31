# AP168 新 PDF 集成 — 第三本 JSON 优化全总结 (2026-05-19)

新 PDF: `FotRP/AP168 《赤凰斗士》3-3 巅峰之王 (1).pdf` (65 页, 汉化组 2026-05-19 出)
权威层级: pf2cn > pf2compendium_chn > wiki > glossary > 汉化组 PDF

## Round 1 — Yoh Souran 译名 + GM 指南残留英文 (39 处)

**问题发现**: NEW/bestiary 用「叶索兰」(canonical) ×14, 但 NEW/第三本 journals 沿用「优·苏然」+ 散落「优」(指代 Yoh) — 内部不一致, PDF 印证「叶索兰」.

| 文件 | 优·苏然 | 苏然 alone | 优 alone (Yoh refs) |
|---|---|---|---|
| back-matter | 4→0 | 4→0 | 14→0 |
| chapter-1 | 1→0 | 2→0 | 12→0 |
| chapter-2 | 1→0 | 1→0 | 4→0 |
| chapter-3 | 0 | 0 | 1→0 |

**外加**: chapter-3 GM 指南 page `Hwanggot 旅程` → `花国旅程` (与项目 花国 一致)

## Round 2 — PF2 zh-CN canonical 术语 (8 处)

**glossary/wiki canonical**:
- **擒抱 → 擒拿** ×4 (back-matter) - Grapple action (Wave 6 user 已定)
- **斯堤吉亚 → 斯狄吉亚** ×1 (back-matter) - Stygia (wiki + glossary)
- **狄斯 → 迪斯** ×2 (back-matter) - Dis (glossary + PDF)
- **光辉爆 → 辐光爆发** ×1 (chapter-2) - Radiant Blast (glossary canonical)

## Round 3 — 凤凰血脉法术名 canonical (8 处)

发现凤凰血脉 page (back-matter line 744) 法术列表多处偏离 canonical:

| English | JSON 旧 | 新 (canonical) | 出处 |
|---|---|---|---|
| Remove Curse | 解除诅咒 | **移除诅咒** | glossary + PDF |
| Meteor Swarm | 陨星雨 | **流星爆** | glossary + PDF |
| Shroud of Flame | 火焰披风 | **火焰护罩** | Wave 4 bestiary + PDF |
| Disintegrate | 瓦解术 | **解离术** | Wave 5 canonical (统一 back-matter 内部 2 处 vs 4 处) |
| Contingency | 触发法术 | **触发术** | PDF + 通用化 (pf2cn 提示 "触发" trait) |

跨文件应用:
- back-matter: 7 处 (5 不同术语)
- chapter-2: 1 处 (Remove Curse)

## Round 4 — Bestiary 跨第三本生物 canonical 化 (56 处)

NEW/bestiary 含第三本怪物 (Syndara/Lightkeepers/Mogaru 等) 的能力条目, 残留旧术语:

- **擒抱 → 擒拿** ×54 (Grapple action — 大量生物 stat block 引用)
- **光辉爆 → 辐光爆发** ×2 (Radiant Blast — DESECRATED GUARDIAN)

## Round 5 — Wave 历史 canonical 残留补齐 (13 处)

| 旧 | 新 | 出处 |
|---|---|---|
| 锦标赛 ×5 | **武道会** | Wave 7 N tournament (散落 GM 指南: bm ×3, ch1 ×1, ch3 ×1) |
| 位面之门 ×2 | **次元门** | Wave 6 C2 Dimension Door (back-matter + ch3) |
| 阳舰二号 ×6 (bestiary) | **太阳剑·二式** | Wave 5 cross-file (journals 已 太阳剑·二式 ×32+) |

## Round 6 — Wiki canonical bestiary 名 (29 处)

- 不知盖 → **天蚀狗** ×29 (Bul-Gae) — pf2wiki 有独立 page (pageid:27375), PDF 印证. 全部跨 bestiary ×17 + back-matter ×12.

## 总修改 (6 轮)

- 文件: 5 文件 (4 个 第三本 journal + 1 bestiary)
- 总编辑次数: **~145 处**
- Round 1: 39 (Yoh + Hwanggot)
- Round 2: 8 (PF2 术语 1st batch)
- Round 3: 8 (凤凰血脉 spell names)
- Round 4: 56 (bestiary Grapple/Radiant Blast)
- Round 5: 13 (Wave 历史 canonical 残留)
- Round 6: 29 (Bul-Gae wiki canonical)

## QA 总结

| 文件 | JSON 有效 | HTML 平衡 | 页数保持 | 字节量变化 |
|---|---|---|---|---|
| back-matter | ✅ | ✅ | 50/50 | +~80 |
| chapter-1 | ✅ | ✅ | 45/45 | +73 |
| chapter-2 | ✅ | ✅ | 20/20 | +~30 |
| chapter-3 | ✅ | ✅ | 18/18 | +3 |
| handouts | ✅ | ✅ | 16/16 | 0 |
| bestiary | ✅ | ✅ | N/A | +~110 |

**全部 11 处目标残留全清** (优·苏然/Hwanggot/擒抱/斯堤吉亚/狄斯/光辉爆/解除诅咒/陨星雨/火焰披风/瓦解术/触发法术)

## 决策记录 — 不改的理由

按 priority hierarchy 检查后保留 JSON 现状:
- Glass Lighthouse: 琉璃光屋 (glossary; PDF 用 琉璃灯塔)
- Flying Mountain: 飞行山脉 (wiki/glossary 静默, JSON 17× 已稳定; PDF 用 飞来山)
- Kaifen Bay: 凯芬湾 (无更高源; PDF 用 开芬湾)
- Hwanggot 主体: 花国 (glossary; PDF 用 垦郭) — 只清残留英文
- Lady Xhai Zhia: 夏之芽女士 (无更高源, Wave-stable; PDF 用 载嘉夫人)
- Lord Aldanar Unmar: 阿达纳·乌马尔勋爵 (Wave 1 已审; PDF 用 阿达娜尔·安马尔领主)
- Yoh's wife: 炅裕 (无更高源; PDF 用 柳景庆)
- 16 巅峰专长名 (Wave 2 翻译 + Wave 4 修, 较直译; PDF 略意译; 无高源裁决)
- 章节标题: JSON 三章 (奔向峰顶/分形丛林之碎/拆解雕匠：灯塔之诅), JSON 较精炼, 不动
- See Invisibility (PDF 看破隐形 vs JSON 侦测隐形): pf2cn 静默, 不动
- Moment of Renewal (PDF 再起时刻 vs JSON 新生时刻): pf2cn 静默, 不动
- Rejuvenating Flames (PDF 回春之焰 vs JSON 新生之焰): 同上
- Cleansing Flames (PDF 净化烈焰 vs JSON 净化之焰): 同上

## 备份

- `_backup/20260519_book3_pdf_optimization/` — Round 1 (Yoh + Hwanggot)
- `_backup/20260519_book3_round2/` — Round 2 (PF2 术语)
- `_backup/20260519_book3_round3_spells/` — Round 3 (spell names + bestiary contingency)
- `_backup/20260519_book3_round4_bestiary/` — Round 4 (bestiary 跨第三本 Grapple/Radiant)

## 衍生产物

`_pdf_extracted/`:
- AP168_book3_zh.ocr.jsonl — 65 页 OCR JSONL
- AP168_book3_zh.ocr.full.txt — 全文可读
- AP168_book3_zh.english_names.txt — 121 双语对照
- AP168_pf2_terms.txt — 134 PF2 术语
- AP168_diff_decisions.md — Round 1 决策清单
- AP168_optimization_summary_20260519.md — Round 1 摘要
- AP168_total_summary_20260519.md — 本文档 (Round 1-4 综合)
- titles_lookup.txt / yoh_audit.txt / yoh_counts.txt / capstone_check.txt / spells_audit.txt / pf2_terms_check.txt / pf2cn_spell_check.txt
- final_qa_round123.txt — 各轮 QA 报告
