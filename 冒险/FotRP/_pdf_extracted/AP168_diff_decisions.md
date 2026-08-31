# AP168 新汉化组 PDF vs 现 JSON 翻译差异 — 决策清单 (2026-05-19)

新 PDF: `FotRP/AP168 《赤凰斗士》3-3 巅峰之王 (1).pdf` (65 页, 2026/5/19 出)
比较对象: `FotRP/需要翻译/NEW/第三本/*.json` + `NEW/pf2e.fists-of-the-ruby-phoenix-bestiary.json` (canonical NEW)

权威层级: pf2cn (system/sf2_cn) > pf2compendium_chn (AlphaStar) > wiki > glossary > 汉化组 PDF

## 一、必须修复 (内部不一致 + 新 PDF 印证)

### 1. Yoh Souran 译名统一: 优·苏然 → 叶索兰

**证据**:
- bestiary (NEW canonical): **叶索兰** ×14 (含 prototypeToken.name)
- 新 PDF AP168: **叶索兰** (4 处明确双语 + 多处单独)
- back-matter NPC page: 优·苏然 ×4 + 苏然 ×4 (outlier!)
- chapter-1 出现 优·苏然 ×1 + 苏然 ×2 (outlier!)
- chapter-2 出现 优·苏然 ×1 + 苏然 ×1 (outlier!)
- 现 JSON 内部不一致, 与 bestiary + 新 PDF 都不符

**结论**: 把 NEW/第三本/ 所有 优·苏然/苏然/单独 "优" (指 Yoh) → 叶索兰. 保留 "苏阿彦古" (儿子名, 与 Yoh 名独立).

### 2. GM 指南残留英文 "Hwanggot" → 花国

**证据**:
- 现 JSON 用 **花国** (matches glossary "Hwanggot": "花国" ×3 occurrences)
- 新 PDF 用 **垦郭** (与 glossary 矛盾)
- chapter-3 GM 指南 page (gmgcEd) 第 1 行残留 "Hwanggot 旅程" 英文

**结论**: 改 "Hwanggot" → "花国" 在 chapter-3 GM 指南 page, 维持与项目 glossary 决策一致.

---

## 二、保留现 JSON (新 PDF 不动摇高优先级源决策)

### 3. Glass Lighthouse: 琉璃光屋 (JSON) vs 琉璃灯塔 (PDF)
- **glossary.json**: `"Glass Lighthouse": "琉璃光屋"`
- glossary > 汉化组 PDF → 保留 **琉璃光屋**
- (新 PDF 自己内部也不一致, 部分用 琉璃光屋)

### 4. Flying Mountain: 飞行山脉 (JSON 17×) vs 飞来山 (PDF)
- wiki/glossary 均无条目
- 项目内部一致用 "飞行山脉 THE FLYING MOUNTAINS" (page name 已稳定)
- 不动 (高优先级源静默时, 维护项目内一致性)

### 5. Kaifen Bay: 凯芬湾 (JSON 7×) vs 开芬湾 (PDF)
- 无更高源裁决
- 不动

### 6. Hwanggot 主体: 花国 (JSON + glossary) vs 垦郭 (PDF)
- glossary 优先
- 不动 (只清残留英文)

### 7. Lady Xhai Zhia: 夏之芽女士 (JSON Wave-stable) vs 载嘉夫人 (PDF)
- 无更高源裁决
- 现 JSON 描述性翻译已稳定 (跨第二本+第三本)
- 不动

### 8. Lord Aldanar Unmar: 阿达纳·乌马尔勋爵 (JSON) vs 阿达娜尔·安马尔领主 (PDF)
- Wave 1 已审过 "Lord Aldunar vs Aldanar"
- 不动

### 9. Quain: 岿安 (glossary) vs 夸英 (PDF)
- 现 JSON 在 第三本 chapter-2 还未明确, 保持现状

### 10. Chapter titles
- 第一章 "奔向峰顶" (JSON) vs "趋向顶峰" (PDF) — 都可, 不动
- 第二章 "分形丛林之碎" (JSON) vs "分形丛林的碎片" (PDF) — 都可, 不动
- 第三章 "拆解雕匠：灯塔之诅" (JSON) vs "拆解雕匠：灯塔的诅咒" (PDF) — JSON 更精炼, 不动

---

## 三、不建议批量改 (但建议人工抽查)

- **wife 炅裕 (JSON) vs 柳景庆 (PDF, p64)** — back-matter 仅出现 3 次, 但作为次要角色; PDF 用韩式名 (柳=Liu/Yu, 景庆=Kyoung-Yoo), 现 JSON 用 炅裕 (单字音译). 不批量改, 留备查.
- **道场对决 vs 大道场对决** — JSON "大道场对决" 对 "Grand Dojo" 更准确, 保留.
- **页 36-65 散落的 prose 表述细节** — 新 PDF 整体 prose 流畅度略好但和现 JSON 同等质量, 不批量改写.

---

## 总计修改

- NEW/第三本/back-matter-1ylYqjGKvevX3BgC.json: ~12 处 (优·苏然 ×4 + 苏然 ×4 + 散落"优" 指代 Yoh)
- NEW/第三本/chapter-1-xClvGtftweJDu3vX.json: ~3 处 (优·苏然 ×1 + 苏然 ×2)
- NEW/第三本/chapter-2-fwmZr935hxQLlBus.json: ~2 处 (优·苏然 ×1 + 苏然 ×1)
- NEW/第三本/chapter-3-VPgzvXimMH8NKzBk.json: 1 处 (GM 指南 "Hwanggot" → "花国")
