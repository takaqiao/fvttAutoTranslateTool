# AP168 新汉化组 PDF 集成与第三本 JSON 优化 — 2026-05-19

## 输入
- **新 PDF**: `FotRP/AP168 《赤凰斗士》3-3 巅峰之王 (1).pdf` (65 页, 2.5MB, 2026-05-19 出)
- **现 JSON**: `FotRP/需要翻译/NEW/第三本/*.json` (canonical, 5 文件)

## 处理流程
1. PDF 字体 DengXian 无 ToUnicode 映射 → PyMuPDF 文本提取乱码
2. 改用 Tesseract OCR (chi_sim) + 双栏切割 → 65 页全部 OCR (~6 分钟)
3. 提取 121 个双语对照
4. 按权威层级 (pf2cn > AlphaStar > wiki > glossary > 汉化组 PDF) 裁决每个分歧

## 修改总览 (NEW/第三本/, 备份 `_backup/20260519_book3_pdf_optimization/`)

### A. Yoh Souran 译名统一 (优·苏然 → 叶索兰)
**问题**: NEW/bestiary 用「叶索兰」(canonical), 但 NEW/第三本 journals 仍用「优·苏然」. 新 PDF 印证「叶索兰」正确.

| 文件 | 优·苏然 | 苏然alone | 优alone (Yoh refs) |
|---|---|---|---|
| back-matter | 4 → 0 | (合 14) | 14 → 0 |
| chapter-1 | 1 → 0 | 1 → 0 | 12 → 0 |
| chapter-2 | 1 → 0 | (合 5) | 4 → 0 |
| chapter-3 | 0 | 0 | 1 → 0 |

**结果**:
- back-matter: 18 处「叶索兰」
- chapter-1: 14 处「叶索兰」
- chapter-2: 5 处「叶索兰」
- chapter-3: 1 处「叶索兰」
- 总计 38 处统一, 与 bestiary canonical (14) 一致
- 「苏阿彦古」(儿子名) 正确保留 ×2 (无误改)

### B. GM 指南残留英文清理
- chapter-3 GM 指南 page (gmgcEd): `Hwanggot 旅程` → `花国旅程` (与项目其余 花国 用法一致)

## 决策记录 (保留现 JSON, 未改)

| 分歧 | JSON | 新 PDF | 决策依据 |
|---|---|---|---|
| Glass Lighthouse | 琉璃光屋 | 琉璃灯塔 | glossary `Glass Lighthouse: 琉璃光屋` |
| Flying Mountain | 飞行山脉 | 飞来山 | wiki/glossary 静默, 项目内 17× 已稳定 |
| Kaifen Bay | 凯芬湾 | 开芬湾 | 无更高源, 项目内 7× 已稳定 |
| Hwanggot 主体 | 花国 | 垦郭 | glossary `Hwanggot: 花国` |
| Lady Xhai Zhia | 夏之芽女士 | 载嘉夫人 | 无更高源, 项目内 Wave-stable |
| Lord Aldanar Unmar | 阿达纳·乌马尔勋爵 | 阿达娜尔·安马尔领主 | Wave 1 已审 |
| Quain | (未明确) | 夸英 | glossary `Quain: 岿安` 留待统一 |
| 章节标题 | "奔向峰顶/分形丛林之碎/拆解雕匠：灯塔之诅" | "趋向顶峰/分形丛林的碎片/拆解雕匠：灯塔的诅咒" | 都可, JSON 更精炼 |
| Wife 名 | 炅裕 | 柳景庆 | 无更高源, 留备查 |

## QA 结果
- ✅ JSON 有效: 5/5
- ✅ HTML 平衡: 0 不平衡
- ✅ 页数保持: 50/45/20/18/16
- ✅ 字节量微增 +76/+73/+22/+3 bytes (符合 +字符净增)
- ✅ 残余「优」全部在安全复合词中 (优雅×2 + 优势×1 + 优先×1 + 优越×2 + 优于×1)
- ✅ 苏阿彦古 (儿子名) 完整保留 ×2

## 备份
- `FotRP/_backup/20260519_book3_pdf_optimization/` — 5 文件 (备份于修改前)

## 衍生产物 (供后续参考)
- `_pdf_extracted/AP168_book3_zh.ocr.jsonl` — 65 页 OCR JSONL
- `_pdf_extracted/AP168_book3_zh.ocr.full.txt` — 全文可读
- `_pdf_extracted/AP168_book3_zh.english_names.txt` — 121 双语对照
- `_pdf_extracted/AP168_diff_decisions.md` — 决策清单
- `_pdf_extracted/qa_html.txt` — QA 报告
- `_pdf_extracted/final_qa.txt` — 跨文件最终核对
