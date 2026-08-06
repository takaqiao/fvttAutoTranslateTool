# Ember 战役正文汉化 · 译者须知

项目主文档：`C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\PROJECT.md`
（要看背景就读第 3 节和第 8 节；本文件已经把动手要用的都摘出来了）

`$P = "C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"`
`$REPO = "$P\1-Ember汉化插件"`，包名固定为 `ember.crucible-adventure.json`

---

## 1. 你的输入与输出

- 输入：`<你的工作目录>\todo.json` —— `{"journal": ..., "items": [{"path", "en", "chars"}]}`
- 参考：`<你的工作目录>\already_translated.json` —— **同一卷里已经译好的页面**。
  专有名词、章节标题、句式一律向它看齐，这是本卷的术语锚点，比任何别的来源优先。
- 可能存在：`<你的工作目录>\orphan_reference.json` —— **旧路径上的译文**。
  上游把某个页面改了名，旧译文就留在了老路径上（babele 取不到，等于死文本）。
  格式是 `{"今天的英文页名": {"orphan_cn_text": "旧中文", "match": 0.99}}`。
  **可以拿它当底稿，但必须逐段与今天的英文对齐后再用** —— 它是照着旧版英文翻的，
  已知问题包括：小标题漏译、多出英文早已删掉的整段、dnd5e/crucible 分支里整段留英文。
  照抄会把这些缺陷一起带进来。
- 输出：`<你的工作目录>\batch.json` —— 扁平 `{"<todo 里的 path 原样>": "<中文>"}`，UTF-8。

**path 必须逐字节照抄 todo.json 里的值**，不要改、不要补前缀。

## 2. 硬规则：标记必须原样保留

Foundry 的富文本标记是功能性的，改坏了链接就失效、样式就崩。

| 形态 | 规则 |
|---|---|
| `@UUID[目标]{标签}` `@Embed[目标]` `@Condition[broken]` `@Spell[illumination.ray]` `@Advantage[2]` | **方括号内部一律照抄**（那是 id 与参数，不是文字）。只有 `{标签}` 要译 |
| `[[/skillCheck awareness 14]]` `[[/gmroll 1d20]]` `[[/knowledge alchemy]]` `[[/eventState xxx]]` | 整段照抄 |
| `&amp;Reference[surprise]{Surprised}` | 方括号照抄，`{标签}` 要译（`{突袭}`） |
| `&amp;reference[damage threshold]` | 整段照抄，不译 |
| `&amp;Reference[Lightly Obscured]`（**没有** `{标签}`） | 方括号里就是 dnd5e 的引用键，整段照抄。渲染时由 dnd5e 自己出中文名。译了就断 |
| `<sup class="system-swap-inline"><sub data-system="dnd5e">…</sub> <sub data-system="crucible">…</sub></sup> | dnd5e / crucible 双轨显示机制，**结构原样保留**，两个分支的正文都要译 |
| `<section class="block gamemaster\|readaloud\|hazard\|exploration\|social">` `<ul class="complex-check">` `<li class="automatic-success\|advantage">` | class 照抄 |
| 标签数量 | `<p>` `<li>` `<strong>` `<div>` `<section>` `<h3>` `<h4>` 的**数量必须与英文完全一致**。英文有几段就译几段，不许合并、不许拆分、不许漏译整段 |

已经踩过两次的坑：`@Embed[… inline overview]` 里的 `overview` 是段落 id，被译成「概览」后嵌入块加载不出来；
`JournalEntry` 被译成「日志条目」同理。**方括号里看见英文单词，别管它是不是像人话，照抄。**

## 3. 译名与文风

- **专有名词双语并列只用于条目名 / 页名**（`name` 字段）：`最后的矿坑 The Last Pit`。
  正文里、以及 `@UUID[...]{标签}` 里的专名**只写中文**。
- **地名意译、人名音译**。新出现的专名先在 `already_translated.json` 里搜一遍，有就沿用。
- 引号用 **“”**（不要用「」）；破折号用 **`——`**，**前后都不加空格**（全库 3798 : 127）。

> **本文件与既有译文冲突时，以既有译文 + `glossary_ec.json` 为准。**
> 下面这张表是从既有译文里抽出来的，但抽错过（第 1 批就有三条被查出来是错的）。
> 你查到冲突时**按既有译文走**，然后在报告的 uncertain 里写明是哪一条、你选了什么、依据是什么。
> 判断依据的强弱顺序：同名条目/物品的 `name` 字段 > 同卷 already_translated > 全库多数写法 > glossary_ec > 本表。
- 章节标题的既定译法：
  `Area Map Context` 区域地图背景 · `Gameplay Details` 玩法细节 · `Levels & Elevation` 层级与高度 ·
  `Illumination` 照明 · `Terrain` 地形 · `Inhabitants` 居民 · `Enemies` 敌人 ·
  `Exploring the X` 搜查X · `Operating Hours` 营业时间 · `Tactics` 战术
- 常用词：check 检定 · Event 事件 · Area Map 区域地图 · Level(场景) 层级 · party 队伍 ·
  character 角色 · Hit Points 生命值 · Health 生命值 · AC · gp/sp 保持原样 ·
  `The character automatically succeeds on this check.` → 该角色此次检定自动成功。
- **英文源里的错误不照抄**（代词错、数值错、单位错）：按上下文改对，并在最终报告里列出来。

### 既定译名（与已完成的 11 个包一致，不许另起炉灶）

Kinesis 念力 · Warden 守林者 · Guardian 守护者 · Tier 阶 · Electricity 电击 · Shocked 感电 ·
Bludgeoning 钝击 · Fire 火焰 · Corruption 腐化 · Poison(伤害) 毒素 · Psychic 灵能 · Radiant 光耀 ·
Fortitude 坚韧防御 · Toughness 坚韧 · Wisdom 感知 · Presence 存在 · Willpower 意志力 ·
inflection 屈折 · gesture 手势 · rune 符文 · essence 精华 · Restrained 受缚 · Critical Hit 暴击 ·
Attunement 同调 · Lineage 血统 · Empowered 强化 · Weakened 虚弱 · Deadly 致命 · Void 虚空 ·
Healing Threshold 治疗阈值 · Rallying Threshold 集结阈值 · Hazard 危害

Ember 世界观专名：Arcturel 阿克图瑞尔 · Arcturian 阿克图里安（人作“阿克图里亚人”）·
The Dives 矿渊 · Ordani 奥尔达尼 · Railen 莱伦 · House Cevher 杰夫赫尔家族 · House Wandren 万德伦家族 ·
Tyraphem 提拉斐姆 · Vorg 沃格 · Skither 斯基瑟 · Aburyx 阿布里克斯 · Jobri 乔布里 ·
**Inkaro 因卡罗**（因卡罗珍珠 / 因卡罗池授权令 / 因卡罗水潭）· Globlin 格布林 · Waterborne 沃特伯恩 ·
Rallyhome 聚归馆 · Lyla Cevher 莱拉·杰夫赫尔 · Funar Cevher 富纳尔·杰夫赫 · Ankarist 安卡里斯特 ·
**Amalthea 阿玛尔忒亚** · Zodi Trask 佐迪·特拉斯克 · Emelyn Arvoda 艾梅琳·阿沃达 ·
Hob Korell 霍布·科雷尔 · Kilner 基尔纳 · Bright Lord 辉耀领主 · For Other Fortunes 保持英文不译

第 2 批新裁决（已落库，直接沿用）：Otherhood **异缘会**（`Otherhood of Fortune` 是专名，作
**幸运异姊会**）· Mayis **玛伊斯** · Kithil **基希尔** · `For Other Fortunes` **为了他人的财富** ·
`With Our Own Two Hands` **凭我们自己的双手** · Shard God **碎片女神** · fell drakes **凶恶龙兽**
（`邪龙` 是 Vile Dragon 的既定译名，别撞）· Repurposed Quarry **改造采石场** · `Bug Fixes` **错误修复** ·
Drakon **龙裔** / Drake **龙兽** / Drakeling **幼龙**（三个别混）· Level（Foundry 场景层级）**层级** ·
The Ordinate 按所指分用：机构作 **审序院**，地点/官署作 **法序议会**

第 1 批新裁决（已落库，直接沿用）：Chessman 棋士（Chessman construct 棋士构装体）·
Lucent 辉耀（构装体名，非音译）· Watcher 注视者 · Shadowbox 暗箱 · Inner Realm 内界 ·
Beacon Brigade 烽灯旅 · Otherhood 异缘会 · Terracini 特拉奇尼 · Seawall 海堤 ·
Attunement **同调**（全库已统一，正文/UI 一致）· Ancestry **血统** ·
`Concluding the Event` → **事件结束** · category `Overview` → **总览**（页名 Overview 则作“概览”）

全量术语表：`$P\5-其他内容\glossary\glossary_ec.json`（扁平 `{en: cn}`，5522 条，可以直接查）

## 4. 查既有译法的办法（**动手前先查，别自己造**）

```powershell
python "<parallel 目录>\probe.py" "Vorg" "Silver Beam" "Lyla"
```
按英文词在**整个战役包的中文译文**里找现成写法，输出带上下文。
查页名 / 条目名用 `python "<parallel 目录>\probe.py" --names "Lantern Roads"`。

## 5. 交付前必须自检（这一步不能省）

```powershell
python "$P\3-常用脚本\qa\apply_translations.py" --repo "$P\1-Ember汉化插件" `
  --pack ember.crucible-adventure.json --batch "<你的工作目录>\batch.json" --dry
```

三道闸：英文源漂移 / 无中文 / **标记不匹配**。
`REJECTED markup` 会打印 `{标记: (英文里几个, 你写了几个)}` —— 照着修，**改到 0 拒绝为止**。
`--dry` 不写盘，可以反复跑。

再跑一次残留英文自检：

```powershell
python "<parallel 目录>\residue.py" "<你的工作目录>\batch.json"
```
把漏译的整句、没译的小标题揪出来。`gp`/`sp`/`AC`/`DC`/`he`/`him`/`CG`/`&amp;reference[…]`
是全库既有写法，不算残留。

## 6. 返回什么

**不要**把译文贴回来。最终回复只要一段结构化摘要：

- 你负责的 journal 名、译了几条 / 多少字符
- `--dry` 的最终结果（必须是 0 拒绝）
- 你新定的专名清单（英文 → 中文，逐条列出，后面要做跨卷一致性检查）
- 你对英文源做出的改动（源文错误）
- 拿不准的地方
