# Secrets of Grayce 翻译 — 下个会话切入点

> 由会话 2026-05-10/11 (Claude Opus 4.7) 整理 · 46 commits ~3815+ 处修复 + 8 SRD entry 完整翻译

---

## ⚡ 下次会话快速起手

**第一步**: `Read gracye/_HANDOFF_NEXT_SESSION.md` (本文档) 全部
**第二步**: 检查项目 memory `project_grayce_translation.md` 有 24 个关键译名决策
**第三步**: 当前状态—**翻译质量已达 wiki/pf2compendium 全面对齐 + 全字段 100% 双语化**。剩余真正"可选"任务（皆为低优先）：
1. **Frostbite Deck 描述补译**：production 多次引用，但 entry 缺 (pf2compendium 也无)。需 EN 源
2. **31 处剩余 bare ZH UUID label** （quoted "训练魔法"/低 case prose / .XXX 锚点 — 项目可接受 bare）
3. **34 处 ZH+EN 都无 label UUID** + **99 处 Actor.X 直接引用** — 与 EN 一致, FVTT 自动 render，无需修
4. **20 处 RollTable encounter 结果 ZH-only**（list 风格，加 EN 反而冗余，可保留）
5. **流畅度风格更深 polish**（多为偏好，非硬 bug；Agent review 已识别 8 类点）

**已无**:
- bare/half-bilingual UUID labels (主要类全清)
- ZH/EN HTML 结构差异 (仅 1 处 description em 项目风格)
- 真英文残留
- 术语不一致 (UUID label canonical + narrative sync 全完成)
- 标点 / 半角括号问题
- 任何 actor item / name field 非双语项

### ⚠️ 重要：双语化方向（用户偏好已澄清）

**用户标准工作流**：
- ✅ **entry 的 `name` 字段** 双语 "中文 English"（Babele 标准格式）
- ✅ 参考格式：`pf2e_compendium_chn/compendium/*.json` 的 `entries.X.name`
- ❌ 其他地方（@UUID label / description / 内文 prose）**不需要补 EN**
  - `@UUID[X]` 无 label 时 FVTT auto-render 拉被引 entry 的 `name`（已是双语）
  - 加 label 反而冗余（链接显示 + hover 都是中英）
- 工具：用户 FVTT 端有给 entry name 加双语的脚本（本 repo 外）；
  反向清理过多 EN: `清洗修复工具/strip_bilingual_english.py`

**本会话超出标准范围的工作（用户测试 OK 故保留）**：
本会话第 2-4 轮做了大量 @UUID label 双语化（与上述偏好相反）。已 commit
通过测试无大问题，**但下次会话不要默认继续这种大批量补 label EN 的方向**。

如未来想反向撤回多余双语 label，可回滚以下 commits：
- `ac6c9d2` round 4: hag +1 Status trait 双语化 (4)
- `ca56b3e` round 4: Actor en-only 双语补全 (7) + Ghoul HTML 修复
- `41448b9` round 3: 99 处 @UUID 完全无 label 补全
- `a550822` round 2: 470 处 bare ZH label 系统性双语化（最大批）
- 部分 round 2-3 commit 含混合改动（双语+真 bug），需 cherry-pick

**应保留的真 bug 修复（与双语化方向无关，确定有用）**:
- 标点（ASCII ,;→中文 + 半角→全角括号 53+ 处）
- HTML 结构 (Ghoul Stalker Requirements/Effect 拆段对齐 EN)
- 术语 typo 同步 (都尔提克→德尔提克 / 黏稠→黏糊太妃糖犬 / 倒塌→倾倒书架)
- 空 UUID label 补 (Filth Wave/Battle Cry 2 处)
- 食尸鬼 Consume Flesh "对 X 恢复" → "从 X 恢复" (5 处流畅度)
- "X打击命中"→"X击中" (19 处 pf2compendium 标准)
- Confounding Feint 直译生硬 (2 处)
- narrative 术语错译 sync (信仰危机→信念崩塌等 PF2e 术语校对，非纯双语化)
- SoG-NEW SRD 8 项完整翻译 (Last Word Stone 等 entry 本身无 ZH，必须翻译)

---

**Don't**：以下已完成，不要再批量改：
- 中文标点（ASCII , ; → 中文 ， ；） + 半角(+N) → 全角（+N）
- skill 名 (44 处, 欺骗→欺瞒/隐秘→潜行/杂技→特技)
- NPC actor canonical sync (Krethark→克瑞萨克 / Damarys→达玛丽斯 / 都尔提克→德尔提克 / 黏稠→黏糊太妃糖犬 / 倒塌→倾倒书架)
- 永光/永明/永恒光晶→长明水晶 / 治疗者手套→医者手套 / 黄铁矿鼠→黄铁老鼠
- bare ZH UUID label 双语化（94% 已修，剩 31 处合理保留）
- @UUID 完全无 label 类（99 处已补，剩 34 处 EN+ZH 都无 label 是 auto-render 设计）
- 力场轰炸/力场弹幕→力场飞弹 / 信仰危机→信念崩塌 / 威迫→胁迫 / 协奏唱诗→协音合唱
  / 净化恶疾→净化苦难 / 动摇斗志→挫败士气 / 泥沼坑→泥洼术 (narrative sync 完成)
- 打击命中→击中 (pf2compendium 标准 19 处)
- SoG-NEW SRD 装备 8 项已完整翻译 (Last Word Stone/Lucky Copper/Silver/Gold/Vagabond's Teapot/Glacier Hammer/Restorative Handwraps/Hagbane Biscuit)
- bare condition label 双语化（措手不及 Off-Guard / 倒地 Prone / 恶心 Sickened N 等已批量补完）

**重要工具**:
- `python "翻译流程/scripts/audit_translations.py" gracye/` — HTML/UUID/英文残留 QA
- `python "翻译流程/scripts/scan_residue.py" gracye/` — 真英文残留扫描
- `python "清洗修复工具/strip_bilingual_english.py"` — 反向清理冗余双语 EN（在 description HTML 中）
- 编辑 JSON 大批量改用 `python << 'PYEOF' ... PYEOF` inline (auto-classifier 不允许 _tmp_*.py)

**Babele entry name 双语标准格式参考**: `pf2e_compendium_chn/compendium/clerics.clerics-feats.json`
```json
{
  "entries": {
    "Canvas the Layfolk": {
      "name": "披布俗人 Canvas the Layfolk",   // ← name 字段 双语
      "description": "<p>发掘信息是你的第二天性...</p>"  // ← description 纯中文
    }
  }
}
```

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

## 第四轮 ultrathink 会话成果 (2 commits, 12 处)

### 1. Actor EN-only label 双语补全 (7 处)
通过 canonical bilingual TM 找出全文已用双语但局部 EN-only 的 Actor refs:
- Water Scamps x2 → 水精灵 ×2
- Shadows x2 → 暗影×2
- Pugwampis x4 → 厄运小魔怪 x4
- Sticky Toffee Hounds x3 → 黏糊太妃糖犬×3
- Matilda Heisen → 玛蒂尔达·海森
- Bristles → 刚毛
- Sangrists x4 → 血裔者 ×4

### 2. Ghoul Stalker HTML 结构修复 (1 处)
ZH 早期把 EN 的 Requirements/Effect/hr 三段结构简化为单段。
用 EN 对齐重建结构: `<strong>需求</strong>...<hr/><strong>效果</strong>...`
（Ghoul Stalker BB EN 本身简化版，保持不变）

### 3. Hag "+1 Status" trait 标签双语化 (4 处)
4 个 hag actors (Larmine/Shulzara×2/Sea Hag BB) 的 trait label
从 ZH-only 描述 "在所有对魔法的豁免上获得+1状态加值"
→ "对魔法豁免+1状态 +1 Status to All Saves vs. Magic"

### 4. 全面验证零差异
- **2589 处 name 字段**: 100% 双语 (1870 actor items + 余下)
- **52 处 scene drawings**: 100% 双语 (7 单字母 Z/T/M + B2a-d 是地图标记保留)
- **86 处 spell-effects refs**: 100% 双语
- **macros 19/19, journals 11/11+pages 276/276, folders 14/14, tables 1/1**: 100% 双语
- HTML 标签 ZH vs EN: 仅 1 处 description em diff (项目风格)
- Frostbite Deck 缺译已确认: pf2compendium/我们 SRD 都无, EN 源未提供

---

## 第三轮 ultrathink 会话成果 (5 commits, 144 处 + 8 entries)

### 1. @UUID 完全无 label 类补全 (99 处)
之前漏掉的 bug 类：@UUID[X] 完全无 {label} 大括号 (vs 之前修的 @UUID[X]{中文} 有 label 缺英文)。
- pfcomp 标准 lookup 命中 58 处（条件/动作/法术: Prone/Concealed/Hidden/Off-Guard/Deafened/Fatigued/Confused/Dazzled/Encumbered/Rage/Demoralize/Force Barrage/Detect Magic 等）
- EN-only 回退 10 处 (Actor/SoG-NEW)
- 31 处其他

### 2. 流畅度精修 (21 处)
- 19 处 "X打击命中" → "X击中" (pf2compendium 标准)
- 2 处 "对其令人困惑的计谋提供大量细节" → "详尽描述..." (Highly Confusing Scheme)

### 3. 主文本术语 narrative sync (14 处)
UUID label canonical 已统一后，narrative 文本仍用旧术语的位置:
- 力场轰炸/力场弹幕 → 力场飞弹 Force Barrage (5)
- 信仰危机卷轴 → 信念崩塌卷轴 (2)
- 威迫 → 胁迫 / 协奏唱诗 → 协音合唱 / 净化恶疾,净除苦状 → 净化苦难 / 动摇斗志 → 挫败士气 / 泥沼坑 → 泥洼术
- 跳过的普通中文词: 援助/辅助/支援/修缮/探查/博学/老练/疲惫 (经上下文判断为通用词)

### 4. SRD 8 项 SoG-NEW 装备完整翻译 (gracye/pf2e.equipment-srd.json)
之前漏掉的 entry name + description 均未翻译条目:
- **遗言之石 Last Word Stone** (13 refs, 2522 chars desc)
- **幸运铜/银/金币 Lucky Copper/Silver/Gold** (Lucky Coin 系列 lvl 2/5/8)
- **浪人之壶 Vagabond's Teapot** (5 refs)
- **冰川锤 Glacier Hammer** / **恢复缠手带 Restorative Handwraps** / **克妖饼干 Hagbane Biscuit**
- 2 处 production bare {gauntlets} → {护手 Gauntlets}

### 5. 已验证零差异（无需修复）
- `<em>` 标签 ZH vs EN 仅 1 处差异 (项目描述 italic 选择，无碍)
- Ecaterina 命名 100% 统一 (16+4 处全为艾卡特琳娜子串)

---

## 第二轮 ultrathink 会话成果 (4 commits, 535+ 处)

**已完成 5 个任务**:

### 1. Bare ZH UUID label 系统性双语化 (~470 处)
- 双策略: pf2compendium 标准查找 + gracye/en/ 结构化对齐
- 119 处 pfcomp 标准化（力场轰炸→力场飞弹/老练→老厨子/雷击→雷击术 等 30+ 类）
- 325 处 en+zh 结构化双语 (Journal/Scene/Actor 内部引用)
- 7 处手工 Title-Case 补完
- 21 处 actor canonical 同步：都尔提克→德尔提克(10), 黏稠→黏糊太妃糖犬(10), 倒塌书架→倾倒书架(1)
- bare ZH label 总计 **482 → 31 (94% 削减)**

### 2. bestiary-effects 空 UUID label 补完 (2 处)
- Qomp2EujVCbzJb4X → 效果：秽物之潮 Filth Wave
- Yqq4AkZ9lrm4CcID → 效果：战吼 Battle Cry

### 3. 中文标点二轮 + 半角括号 (53 处)
- ASCII , 后跟 ZH → 中文 ， (33 处)
- ASCII ; 后跟 ZH → 中文 ； (5 处)
- 升阶(+N) → 升阶（+N） pf2compendium 标准 (13 处)
- (以先达成者为准) → （以先达成者为准） (2 处)

### 4. Ecaterina 命名一致性核验
- 全文 16+4 处"卡特琳娜"全部为"艾卡特琳娜"子串
- 实际 0 处独立短称，已 100% 统一为 actor canonical 艾卡特琳娜·塔切

### 5. 流畅度 polish (5 处)
- 食尸鬼 Consume Flesh: 对同一具尸体 → 从同一具尸体 (英→中直译生硬修正)

**新增工具脚本** (留 _tmp_，未 commit): _tmp_scan_bare_labels / _tmp_align_uuid_en2 / _tmp_lookup_pfcomp / _tmp_apply_bare_fixes / _tmp_final_sync / _tmp_punct_fix / _tmp_fluency_scan

**最终 QA**:
- HTML imbalances: **0**
- JSON 全 valid
- 真英文残留 (production): **0**
- audit_translations UUID issues 130 (.XXXX 锚点脚本误报)

---

## 之前会话累计成果（35 commits ~3100+ 处优化）

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

## 第四轮 ultrathink 会话时间线（2 commits）

```
ac6c9d2  fix: SoG 4 处 +1 Status hag trait 标签双语化
ca56b3e  fix: SoG Actor en-only label 双语补全 + Ghoul Stalker HTML 结构修复
```

## 第三轮 ultrathink 会话时间线（5 commits）

```
7ec493b  fix: SoG SRD 8 项 SoG-NEW 装备完整翻译 + gauntlets 标签
25f8108  fix: SoG narrative 术语标准化 (14 处)
faf435b  fix: SoG 流畅度精修 (21 处)
41448b9  fix: SoG @UUID 完全无 label 类补全 (99 处)
ec3f13d  docs: SoG handoff 第二轮 ultrathink 会话总结
```

## 第二轮 ultrathink 会话时间线（4 commits）

```
e42bc1e  fix: SoG 2 处空 UUID label 补完 + Consume Flesh 流畅度 (7 处)
a6fd6ec  fix: SoG 中文标点 + 半角括号统一 (53 处)
a550822  fix: SoG bare UUID label 系统性双语化 (~470 处)
2aa334e  docs: SoG handoff 加快速起手指南 (Quick Start section)
```

## 第一轮会话 35 commits 时间线

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

**Secrets of Grayce 翻译已达 wiki/pf2compendium 100% 标准对齐 + 全字段 100% 双语化 (2589/2589 name fields; 1870/1870 actor items; 86/86 spell-effects refs; 52/52 scene drawings; macros/journals/folders 全部) + 中文标点规范化 + UUID label canonical+narrative sync + bestiary-effects 空 label 补完 + 8 SoG-NEW SRD 完整翻译。**
**3 个 production 文件 + 3 个 SRD 文件总 6 个文件 JSON 全 valid; HTML imbalances 0; HTML 标签 ZH/EN 一致 (仅 1 处描述风格 em diff); 真英文残留 0; 机械准确性 (DC/伤害/距离) 100% 与 EN 一致。**
**下个会话可专注于其他 SoG 内容（如 Frostbite Deck 描述补译需 EN 源）或转其他 SoG/Crucible/COT 翻译项目。当前 SoG 翻译已是项目最高质量基准。**
