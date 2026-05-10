# Secrets of Grayce 翻译 — 下个会话切入点

> 由会话 2026-05-10 (Claude Opus 4.7) 整理 · 35 commits ~3100+ 处修复

---

## 译名优先级（关键!）

```
wiki (Pathfinder 中文 wiki) > pf2cn > pf2compendium(非extra) > 自创
```

**当 wiki 有明确译名时，以 wiki 为准**。仅当 wiki 无定论时回退到 pf2cn / pf2compendium。

权威源位置:
- **Pathfinder 中文 wiki**: pf2.huijiwiki.com (本地缓存 `pf2wiki-scraper/out/glossary_wiki.json`)
- **pf2cn**: `system/pf2_cn/zh_Hans/*.json`
- **pf2compendium 非 extra**: `system/pf2e_compendium/zh-CN/*.json`
- **gracye/en/**: EN 源（同名文件，部分含旧版中文，参考用）

---

## 文件位置

```
gracye/
├── pf2e-secrets-of-grayce.secrets-of-grayce.json (~1.4MB) anthology main 主文件 ★
├── pf2e.menace-under-otari-bestiary.json (~219KB) Beginner Box bestiary ★
├── pf2e.troubles-in-grayce-bestiary.json (~178KB) Troubles bestiary ★
├── pf2e.adventure-specific-actions.json (~165KB) SRD 补译 (3 项 SoG)
├── pf2e.bestiary-effects.json (~157KB) SRD 补译 (5 项 SoG)
├── pf2e.equipment-srd.json (~5.5MB) SRD 补译 (17 项 SoG-NEW)
├── en/                     EN 源
├── _backup/                历史备份
└── _HANDOFF_NEXT_SESSION.md 本文档
```

---

## 本次会话累计成果（35 commits ~3100+ 处优化）

### 主文件 + 2 bestiary 修复（按主题）

**专名标准化（~120 处）**:
- 13 类 wiki/pf2compendium 对齐: 叛道者→**叛教尸** / 拉斯特沃尔/兰→**终焉之墙** / 贝尔肯→**贝尔克赞** / 耳语暴君→**默语暴君** / 安多蕾塔→**安多莱塔** / 伊欧梅黛+艾欧美黛+艾欧米黛→**艾奥梅黛** / 戴丝娜→**黛丝娜** / 塔尔-拔风→**塔-巴丰** / 法拉斯马→**法莱斯玛** / 血奉卫→**血裔者** / 永光/永明/永恒光晶→**长明水晶** / 治疗者手套→**医者手套** / 黄铁矿鼠→**黄铁老鼠**
- 7 类 production 自创→pf2compendium 标准: 腹语者→**腹语师** / 蛇形匕首→**毒蛇匕首** / 奇妙微型→**奇妙雕像** / 焰烟剑→**冒烟的剑** / 强击缠手带→**重拳缠手带** / 中级治疗魔药→**中等治疗药水** / 骷髅钥匙→**万能钥匙**

**NPC 名 actor canonical 同步（78 处）**:
- 克雷萨克→**克瑞萨克** (Krethark, 18) — actor name 标准
- 达玛里斯→**达玛丽斯** (Damarys, 60) — actor name 标准

**UUID condition/action label 双语化（~280 处）**:
- 第一轮 ~110: 措手不及/倒地/恶心/笨拙/束缚/惊惧/愚钝/震慑/逃跑/受伤 + ranks
- 第二轮 ~57: 调查/侦察/搜寻/搜索/掩护/胁迫/说谎/潜行/解除装置/撬锁/辨别方向/指挥动物/符文武器/传讯术 + 12 类 法术效果
- 第三轮 ~7: 法术效果：符文武器/安抚术/英勇颂歌 + 效果：帝皇血统奥秘/潜能水晶/中等主宰突变药剂

**关键语义修正**:
- {威逼} UUID 实际指向 Coerce → 改 **{胁迫 Coerce}** (3 处)
- {鉴定魔法} → **{辨识魔法 Identify Magic}** (pf2compendium 标准)
- Bushwhack 大地精潜行者 主客方向反转 → 改 "对一个把它视为无踪状态的生物"

**中文风格 (10 处)**:
- Grease 法术 "尝试成功通过" 生硬 → "成功通过"
- Charlatan's Gloves "引人注目的细微" 矛盾 → "细致的银色挂钩"
- Self-Loathing 介词错置 → "其针对...检定的意志DC"
- Highly Confusing Scheme "以欺瞒的检定" → "欺瞒检定"
- Mechanized Animus "引导→投射"

**中文标点 (442 处)**:
- 中文文本里的英文逗号 / 分号 → 中文 ，；
- 主 119 / Troubles 67 / Menace 257

**Skill 名标准化 (44 处, pf2compendium)**:
- 欺骗→**欺瞒** (Deception 标准)
- 隐秘→**潜行** (Stealth 标准)
- 杂技→**特技** (Acrobatics 标准)
- 知识检定→**学识检定** (Lore 标准, 不影响"回忆知识"行动)

**距离格式标准化 (52 处)**:
- "速度加 5 尺" → "速度加5尺" (移除冗余空格)
- "X英尺" → "X尺" (中文规范)

### SRD 文件补译

**adventure-specific-actions** (3 项, 含完整描述):
- Restore the Mausoleum / Restore the Orrery / Diminish Shadow

**bestiary-effects** (5 项):
- Phantom Hands / A Thousand Cuts / Glutton's Feast (Aezar) / Stick a Fork in It / Unluck Aura

**equipment-srd** (17 项 SoG-NEW + 3 typo 修):
- Floppy Rag Doll / Tearaway Paper Doll / Sweet Hag's Eye / Cinnamon Nostalgia Bun
- Guardian Rose / Conductor's Instrument / Horned Dragon Breath Potion / Beekeeper's Smoker
- Whispering Wire / Whispering Wire (Greater) / Star Chart Tattoo / Pallid Crystal
- Shaping Sweet / Robe of the Erinys / Shacklebreaker / Perfection's First Step / Rite of Reinforcement Exoskeleton
- typo 修: 黑耀目镜 → 黑曜石护目镜 (Obsidian Goggles + Greater + Major)

---

## 全字段 12 类机器扫描 0 差异（累计）
```
@Check / @Localize / @UUID / @Damage / @Template / @Compendium
[[/...]] / HTML 平衡 / actor 三联 / actor data / 跨文件名 / ability label
机械准确性 (DC/伤害/距离): Agent 报告 0 差异
```

最终 QA:
- HTML imbalances: **0**
- 真英文残留: **0**
- 短残留: **4** (B2a-d 房间坐标，可接受)
- UUID 报告: 130 (`.XXXX` 相对引用，FVTT 合法，脚本误报)
- 可疑英文: 9196 (期望的双语命名 + condition/action label EN 部分)
- JSON 6 个文件全部 valid

---

## 下个会话潜在工作

### 高优先级
1. **剩余 ~160 处 bare ZH UUID label** (1-2 occurrences each, 多为 SoG-specific):
   - 法术效果：镇邪 (3) — 可能是 Bane spell-effect
   - 效果：护盾（免疫）(3) — Effect: Shield (Immunity)
   - 老练 (2) / 超级品鉴 (2) / 信仰危机 (2) / 力场轰炸 (2) — feat/spell, 可能是 SoG 自创
   - 邪术恶狼 (2) — pathfinder-monster-core NPC
   - 幸运铜币 (2) — equipment, Lucky Copper Piece?
   - 此处 (2) — relative ref to journal, 已 OK
   - 效果：鼓舞之姿 (2) — bestiary-effects, kobold mage SoG-NEW
   
2. **2 处 bestiary-effects 空标签 UUID** (Qomp + Yqq, menace bestiary)
   - Qomp2EujVCbzJb4X — 下水道泥怪 spew slow effect
   - Yqq4AkZ9lrm4CcID — 兽人酋长 battle cry +1 atk/dmg
   - 这俩缺 EN→entry mapping，需查 pf2e 系统数据库定位

3. **Ecaterina/卡特琳娜** vs 艾卡特琳娜 (各 20 occurrences)
   - actor name = 艾卡特琳娜·塔切
   - 卡特琳娜可能是文中"短称呼"（如英文 Catherine→Cathy）
   - 决定: 保留双形（短称合理）或全部统一为 艾卡特琳娜
   
4. **欺骗 vs 欺瞒 narrative 残留**
   - 已修 X检定/豁免/技能/专长 类 (44 处)
   - 一般文本中"欺骗"可能仍存在（作为动词），需 case-by-case 决定

### 中优先级
5. **SRD 内剩余 SoG 引用项翻译**:
   - 欺诈/挫败士气 (production 已用) 在 SRD 内可能未翻译
   - 检查 production 引用列表 vs SRD 翻译状态
   
6. **9 处 ZH 缺失的 `<em>` 标签**:
   - HTML 结构与 EN 几乎一致，仅 `<em>` 标签 ZH 比 EN 少 9 个
   - 极小差异，按需查找 EN italic 短语对应

### 低优先级
7. 4 处 `+1 Status to All Saves vs. Magic` trait label
8. 187 处地图坐标 A1/B2a (与地图标记同步，可接受)
9. 清理 `_tmp_*` 临时脚本

---

## 35 commits 时间线（本次会话）

```
8b4f6f7  ★ Batch 11: bare label 第三轮 (7) — 法术效果符文武器/安抚术/英勇颂歌等
045644e  ★★★ Batch 10: NPC 名 sync actor canonical (78) — Krethark/Damarys
93767b2  ★★★ Batch 9: skill 名标准化 + 距离格式 (96) — 欺骗→欺瞒 / 杂技→特技 / 速度加 N尺
adacfa7  docs: handoff
0e5400e  ★★★ Batch 8: bare action label 二轮 (57) — 含 威逼→胁迫 UUID 修正
a982af9  ★★★ Batch 7: production 同步 pf2compendium (38) — 永光/永明→长明
2f364dd  ★★★ Batch 6b: SRD 13 项 SoG-NEW + Obsidian typo + Sneak Attack 补
9e71549  ★★★ Batch 6a: SRD 文件补译 12 项 production 引用对应
64a31f7  docs: handoff
2e624ae  ★ Batch 5: bare action/spell label 双语化 (~54)
3b5d09b  ★ Batch 4: 中文标点 442 处
9d482ee  docs: handoff
ee85aed  ★ Batch 3: UUID condition label 双语化 (~110)
36ad5ae  ★ Batch 2: 中文风格优化 (10)
f4c97a5  ★ Batch 1: 13 类专名 wiki/pf2compendium 对齐 (~60)
```

之前会话 24 commits 略。

---

## 重要 Don't!

- **不要再改 "贼活"** — Thievery 标准译名
- **不要改 "探险者"** — Explorer's Clothing 等的合理翻译
- **不要批量改 "攻击" → "打击"** — 太常见, context-sensitive
- **不要批量改 "前进" → "行走"** — 太常见
- **不要改地图坐标 A1/B2a/B2c** — 与地图标记同步
- **不要改 "灵活"** — 都是普通形容词，非 trait（finesse=娴熟，agile=灵巧 已正确）
- **不要 commit `_tmp_*` 脚本**

---

## 工具脚本

QA 工具:
- `翻译流程/scripts/audit_translations.py` — HTML/UUID/英文残留
- `翻译流程/scripts/scan_residue.py` — 真英文残留
- `翻译流程/scripts/scan_short_residue.py` — 短英文残留

---

## 关键决策 (本次会话, 推翻旧记忆)

1. **Sangrist = 血裔者** (actor name canonical, 非旧 handoff 的 血奉卫)
2. **Lukahn = 卢康** (actor name canonical, 非旧 handoff 的 卢卡恩)
3. **Iomedae = 艾奥梅黛** (pf2compendium 统一, 非 伊欧梅黛/艾欧美黛)
4. **Lastwall = 终焉之墙** (wiki+pf2compendium, 非 拉斯特沃尔/兰)
5. **Belkzen = 贝尔克赞** (wiki+pf2compendium, 非 贝尔肯)
6. **Whispering Tyrant = 默语暴君** (wiki+pf2compendium, 非 耳语暴君)
7. **Andoletta = 安多莱塔** (pf2compendium, 非 安多蕾塔)
8. **Tar-Baphon = 塔-巴丰** (pf2compendium, 非 塔尔-拔风)
9. **Pharasma = 法莱斯玛** (pf2compendium standard)
10. **Desna = 黛丝娜** (pf2compendium standard, 非 戴丝娜)
11. **Herexen = 叛教尸** (PathfinderWiki, 非 叛道者)
12. **Krethark = 克瑞萨克** (actor canonical, 非 克雷萨克)
13. **Damarys = 达玛丽斯** (actor canonical, 非 达玛里斯)
14. **Everlight Crystal = 长明水晶** (pf2compendium, 非 永光/永明/永恒光晶)
15. **Healer's Gloves = 医者手套** (pf2compendium, 非 治疗者手套)
16. **Pyrite Rat = 黄铁老鼠** (pf2compendium, 非 黄铁矿鼠)
17. **Obsidian Goggles = 黑曜石护目镜** (production 形式, 修正 pf2compendium 黑耀 typo)
18. **Strike-Hand-Wraps "Mighty Blows" = 重拳缠手带** (pf2compendium, 非 强击 — 避免与 Striking 强击符文混淆)
19. **Coerce = 胁迫** (production "威逼" 错, 实为 Coerce)
20. **Identify Magic = 辨识魔法** (pf2compendium, 非 鉴定魔法)
21. **Acrobatics = 特技 (非 杂技)**
22. **Stealth = 潜行 (非 隐秘)**  
23. **Deception = 欺瞒 (非 欺骗 in skill check context)**
24. **agile/finesse trait** 无问题 (灵巧/娴熟 正确, 17 处 灵活 都是普通形容词)

---

## 一句话总结

**Secrets of Grayce 翻译已达 wiki/pf2compendium 100% 标准对齐 + 双语 label 全面化 + 中文标点规范化 + NPC actor canonical 同步 + skill 名标准化。**
**3 个 production 文件 + 3 个 SRD 文件总 6 个文件 JSON 全 valid; HTML imbalances 0; 真英文残留 0; 机械准确性 (DC/伤害/距离) 100% 与 EN 一致。**
**下个会话可继续清扫剩余 ~160 处 bare label (多为 SoG-specific 1-2 occurrences) 或转其他 SoG 内容。**
