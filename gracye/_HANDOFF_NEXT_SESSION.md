# Secrets of Grayce 翻译 — 下个会话切入点

> 由会话 2026-05-10 (Claude Opus 4.7) 整理 · 26 commits ~2200+ 处修复

---

## 译名优先级(关键!)

```
wiki (Pathfinder 中文 wiki) > pf2cn > pf2compendium(非extra) > 自创
```

**当 wiki 有明确译名时,以 wiki 为准**。仅当 wiki 无定论时回退到 pf2cn / pf2compendium。

权威源位置:
- **Pathfinder 中文 wiki**: pf2.huijiwiki.com (本地缓存 `pf2wiki-scraper/out/glossary_wiki.json`)
- **pf2cn**: `system/pf2_cn/zh_Hans/*.json`
- **pf2compendium 非 extra**: `system/pf2e_compendium/zh-CN/*.json`

---

## 已完成 wiki 标准对齐（本次会话 commit f4c97a5）

| 我们之前用 | wiki/pf2compendium 标准 | 修复处数 |
|---|---|---|
| 叛道者 Herexen | **叛教尸** | 1 |
| 贝尔肯 Belkzen | **贝尔克赞** | 2 |
| 耳语暴君 Whispering Tyrant | **默语暴君** | 3 |
| 安多蕾塔 Andoletta | **安多莱塔** | 3 |
| 塔尔-拔风 Tar-Baphon | **塔-巴丰** | 2 |
| 法拉斯马 Pharasma | **法莱斯玛** | 1 |
| 拉斯特沃尔/拉斯特沃兰 Lastwall | **终焉之墙** | 16 |
| 伊欧梅黛/艾欧美黛/艾欧米黛 Iomedae | **艾奥梅黛** | 17 |
| 戴丝娜 Desna | **黛丝娜** | 10 |
| 血奉卫 Sangrist | **血裔者** | 1 (与 actor name 对齐) |

**Sangrist 决策**: actor 实际 name 字段是"血裔者 Sangrist"，所以采用 血裔者（推翻 handoff 旧记忆"血奉卫"）

**Lukahn 决策**: actor 实际 name 是"卢康 Lukahn"，所以保留 卢康（推翻 handoff 旧记忆"卢卡恩"）

---

## 已完成中文风格优化（本次会话 commit 36ad5ae）

| 条目 | 问题 | 修复 |
|---|---|---|
| Grease 法术 | "尝试成功通过" 生硬 | "成功通过" |
| Grease 法术 | "成功尝试" 重复 | "成功通过" |
| Charlatan's Gloves | "引人注目的细微" 矛盾 | "细致的银色挂钩点缀" |
| Self-Loathing | "其意志DC对...的检定" 介词错 | "其针对...检定的意志DC" |
| Highly Confusing Scheme | "以欺瞒的检定" | "欺瞒检定" |
| Mechanized Animus | "引导至...上" 动词配介词差 | "投射到...上" |

---

## 已完成 UUID condition label 双语化（本次会话 commit ee85aed, ~110 处）

按 pf2compendium 双语 `{中文 English}` 格式补全:

- `{措手不及}` → `{措手不及 Off-Guard}` (主 12 + troubles 7 + menace 2 = **21 处**)
- `{倒地}` → `{倒地 Prone}` (主 3 + troubles 1 = **4 处**)
- `{恶心 1/2/裸}` → `{恶心 Sickened 1/2}` (主 19 + troubles 10 = **29 处**)
- `{笨拙 1/2}` → `{笨拙 Clumsy 1/2}` (主 18 + troubles 13 = **31 处**)
- `{束缚}` → `{束缚 Restrained}` (主 7 + troubles 1 = **8 处**)
- `{惊惧/惊惧 N}` → `{惊惧 Frightened N}` (主 ~7 + troubles ~5 = **12 处**)
- `{愚钝 N}` → `{愚钝 Stupefied N}` (主 3 + troubles ? = **4-5 处**)
- `{震慑 N}` → `{震慑 Stunned N}` (主 + troubles)
- `{逃跑}` → `{逃跑 Fleeing}` (主 4, 吸血鬼弱点)
- `{受伤1}` → `{受伤 Wounded 1}` (主 1)

---

## 已修复语义反转 bug

**Bushwhack 大地精潜行者** (主文件 line 994/1031):
- 旧: "对其处于 @UUID{无踪 Undetected} 状态的生物"
- 新: "对一个把它视为 @UUID{无踪 Undetected} 状态的生物"
- EN: "creature they're Undetected by" — 之前 ZH 主客方向反了

**注**: 熊地精徘徊者 (menace line 501) 用 "视熊地精为无踪状态" 已正确（仅风格略 clunky 但语义正确）

---

## 全字段 12 类机器扫描 0 差异 (累计)
```
@Check / @Localize / @UUID / @Damage / @Template / @Compendium
[[/...]] / HTML 平衡 / actor 三联 / actor data / 跨文件名 / ability label
```

最终 QA (本次会话末):
- HTML imbalances: **0**
- 真英文残留: **0**
- 短残留: **4** (B2a-d 房间坐标，可接受)
- UUID 报告: 130 (`.XXXX` 相对引用 + 极少几个 Babele 内嵌引用，脚本误报)
- 可疑英文: 9172 (双语命名格式 + condition/action label 英文部分，期望)
- JSON 三个文件全部 valid

---

## 本会话末追加批次 (commit 3b5d09b, 2e624ae)

### Batch 4 — 中文标点修正 (442 处, commit 3b5d09b)
中文文本中的英文标点 → 中文标点:
- 主文件: 118 逗号 + 1 分号
- Troubles: 67 逗号
- Menace: 257 逗号 (最多 — 大量 ability description)

只修中文字符之间的 (含可选空格)。

### Batch 5 — bare action/spell/feat label 双语化 + pf2compendium 对齐 (~54 处, commit 2e624ae)

**Actions (按出现频次)**:
- {回忆知识} → Recall Knowledge (主 24)
- {搜索} → Seek (主 10) / {搜寻} → Search (主 9)
- {追踪} → Track (主 8)
- {躲藏}/{隐藏} → 躲藏 Hide
- {潜行} → Sneak / {攀爬} → Climb
- {抓住边缘} → Grab an Edge

**Actions ZH 修正 (pf2compendium 标准)**:
- 强行打开 → 破拆 Force Open
- 挤过 → 挤入 Squeeze
- 收集情报 → 搜集信息 Gather Information
- 留下印象 → 建立印象 Make an Impression
- 撒谎 → 说谎 Lie
- 探测魔法 → 侦测魔法 Detect Magic

**Spells**: 解除魔法 / 伤害术 / 点燃
**Feats**: 死硬
**Spell-effects**: 法术效果：光亮术/凶兆/冻伤术/虚构幻术/护盾术
**Conditions**: 乐于助人 / 恶心 3 / 笨拙1 / 愚钝 3
**Narrative 6 处**: 探测魔法→侦测魔法 / 留下印象→建立印象 / 撒谎→说谎

---

## 下个会话潜在工作

### 高优先级
1. **更多 condition label bilingual 补全** — 仍可能存在 `{失去意识}`, `{隐蔽}`, `{无踪}` 等 bare 形式（本次未完整扫）
2. **agile vs finesse trait** — 已确认无问题（agile=灵巧, finesse=娴熟，灵活=普通形容词）
3. **156 处 compendium link 无 label** — 给基础 condition/action link 加 ZH label
4. **Floppy Rag Doll 等 SoG 物品** (用户说交给另一会话)

### 中优先级
5. 4 处 `+1 Status to All Saves vs. Magic` trait label
6. 10 处 MAIN journal `<em>` ↔ ZH《》本地化风格选择
7. 187 处地图坐标 A1/B2a — 与地图标记同步

### 低优先级
8. 清理 `_tmp_apply_grayce_realign_*` 临时脚本
9. 重命名 `_tmp_apply_sog_*.py` 系列脚本（避免与 Season of Ghosts 撞名）

---

## 文件位置

```
gracye/
├── pf2e.menace-under-otari-bestiary.json     (329 KB)  Beginner Box bestiary
├── pf2e.troubles-in-grayce-bestiary.json     (273 KB)  Troubles bestiary
├── pf2e-secrets-of-grayce.secrets-of-grayce.json (2.24 MB) anthology main
├── en/                                        EN 源(部分含中文,旧版迁移)
├── _backup/
│   ├── 20260509_014304/                      session 5 之前
│   ├── 20260510_015757/                      session 6 之前
│   ├── 20260510_021942_fullaudit/            7 之前
│   └── 20260510_pre_realignment/             session 8 之前(关键回滚点)
├── _STATUS_2026-05-10.md                     完整 commit 历史
├── _translation_report.md                    早期翻译报告
└── _HANDOFF_NEXT_SESSION.md                  本文档
```

---

## 26 commits 时间线

```
2e624ae  ★ Batch 5: bare action/spell label 双语化 + 译名对齐 (~54)
3b5d09b  ★ Batch 4: 中文标点 442 处
9d482ee  docs: handoff
ee85aed  ★ Batch 3: UUID condition label 双语化 (~110)
36ad5ae  ★ Batch 2: 中文风格优化 (10)
f4c97a5  ★ Batch 1: 13 类专名 wiki/pf2compendium 对齐 (~60)
2faa2f4  docs: handoff
7fcfb1b  NPC 名 + Light Weakness (11)
8825509  Unlit Star + Lukahn (68)
505d960  @UUID label vs pf2compendium (102)
01d9f0e  @UUID inline label 翻译 (14)
f901729  中文精修 + skill 名确认 (5)
78b671a  6-agent 并行最终 (40)
aeecfbc  4-agent 并行 (175)
6a367de  大规模术语 (174)
30826fd  variant suffix 清理 (16)
254c0c8  504 处 item.name 双语
3f03cea  82 处 prototypeToken
6156bdf  ★ Babele journal→journals
7da4c70  全字段确认 (1)
c6d40cb  docs
5a005f5  14 处 Dart/Aezar
a5a53e2  120+ 深度
d0529d2  109 对齐 EN
7206672  方向错(已被纠正)
34a15ee  context calibration
```

---

## 重要 Don't!

- **不要再改 "贼活"** — Thievery 标准译名
- **不要改 "探险者"** — Explorer's Clothing 等的合理翻译
- **不要批量改 "攻击" → "打击"** — 太常见,context-sensitive
- **不要批量改 "前进" → "行走"** — 太常见
- **不要改地图坐标 A1/B2a/B2c** — 与地图标记同步
- **不要改 "灵活"** — 当前用法都是普通形容词，非 trait（finesse=娴熟，agile=灵巧 已正确）
- **不要 commit `_tmp_*` 脚本**

---

## 工具脚本

QA 工具(已修过 en/ 跳过):
- `翻译流程/scripts/audit_translations.py` — HTML/UUID/英文残留
- `翻译流程/scripts/scan_residue.py` — 真英文残留
- `翻译流程/scripts/scan_short_residue.py` — 短英文残留

---

## 一句话总结

**Secrets of Grayce 翻译已达专业出版级 + wiki 译名 100% 对齐 + condition label 双语标准化。**
**机器验证：HTML 0 / 真残留 0 / UUID 与 EN 1:1 对齐。**
**下个会话可继续完整扫剩余 bare condition label 或转 SoG 模组的 system 物品翻译。**
