# 第十六轮 · 主控独查：`distinct_terms` 空转藏了 184 处违规

## 起因

第十六轮第一段的复核单元报了一条：

> `a_distinct_terms()`（`assert_resolutions.py:177-186`）注释自陈「纯配置自检，不读库」——
> 它只检查一组术语的中文两两不同，从不扫 lang 也不扫 compendium。
> 所以 `R-region-area-map` 一直全绿，而 ember lang 的 `EMBER.CALENDAR.REGION`
> 反着写了「区域地图」四个发布版没人发现。

我读了源码，**指控成立**，docstring 白纸黑字。这是 `R-catwalk`（JSON 单反斜杠被当成
退格转义、正则匹配 0 叶却报绿）之后**第二次**「断言空转还报绿」。

⚠ 两次的形态不同，别只记住一次：
- `R-catwalk` 是**判据写坏了**（正则被 JSON 转义吃掉），`min_hits` 护栏能防。
- 这次是**判据压根没读库**，`min_hits` 防不住 —— 它连「命中数」这个概念都没有。
  通则应当是：**任何断言都必须能说出「我扫了多少个叶/键」**，说不出来的就是自检不是断言。

## 于是全扫了一遍四条 `distinct_terms` 规则

`R-round-turn` / `R-tier-rank-level` / `R-region-area-map` / `R-shard-god`，
拿上游英文（`FoundryVTT/Data/modules/ember/lang/en.json`、
`systems/crucible/lang/en.json`）做英文闸逐键比对。

### lang 侧：14 条违规

| 规则 | 仓 | 键 | 现值 | 应为 |
|---|---|---|---|---|
| R-region-area-map | ember | `EMBER.CALENDAR.REGION` | 区域地图 | 地区地图 |
| R-region-area-map | ember | `EMBER.BIOME.FIELDS.fillColor.hint` | 区域地图 | 地区地图 |
| R-region-area-map | ember | `EMBER.BIOME.FIELDS.sounds.environment.surface.hint` | 区域地图 | 地区地图 |
| R-region-area-map | ember | `EMBER.LOCATION.FIELDS.sounds.environment.neighbors.hint` | 区域地图 | 地区地图 |
| R-region-area-map | ember | `EMBER.QUEST.FIELDS.color.hint` | 区域地图 | 地区地图 |
| R-tier-rank-level | crucible | `SKILL.AcquiredRanks` | 获得的等级 | 获得的阶位 |
| R-tier-rank-level | crucible | `SKILL.Ranks` | 技能等级 | 技能阶位 |
| R-tier-rank-level | crucible | `SKILL.UnacquiredRanks` | 未习得等级 | 未习得阶位 |
| R-tier-rank-level | crucible | `TALENT.FIELDS.training.rank.label` | 训练等级 | 训练阶位 |
| R-tier-rank-level | crucible | `ARCHETYPE.FIELDS.skills.hint` 等 5 条 | 等级 | 阶位 |

⚠ **A/C 路产的 `ac-lang-ember.json` 只盖了 `EMBER.CALENDAR.REGION` 一条**，
另外 4 条 Region Map 是它和复核单元**双双漏掉的**。补批次 `mc-lang-region-map-rest.json`。

### compendium 侧：`Rank` 176 叶

英文闸 `\branks?\b`（**不区分大小写**）下，库内中文 **等级 223 叶 : 阶位 47 叶** ——
即 §8 2026-08-14d 那条「`Rank`=阶位 / `Tier`=阶 / `level`=等级 三分」的裁决
**从来没有被执行过**，因为守它的断言在空转。

但不能一把梭：`rank` 在本库是**两个词**。

- **游戏机制义**（→阶位）：`You gain the Novice rank in the Arcana skill` ·
  `Attunement Rank 1` · `Soulbound Rank` · `training rank` · `Bonus Rank Scale` ·
  `one rank of exhaustion` · `the created ooze never has rank superior to Normal`（敌手阶位）
- **普通名词义**（不动）：`denote their civic rank` 公民地位 · `within the ranks of Shard Gods` 行列 ·
  `stripped of his rank within the order` 教团职位 · `the social rank of individuals` 社会地位 ·
  `risen through the ranks` · `retains his rank as a Commander` 军衔 · `rank-and-file grunts`

做法：`probes/split_rank.py`，逐位对齐 + 上下文分类（GAME / COMMON），
计数不等的整叶跳过交人看；人看过的定论写进脚本里的 `MANUAL` 表（带预期次数护栏，
上游一改措辞就告警而不是悄悄改错）。

**结果：改写 161 叶 + 人裁 15 叶 = 176 叶，`apply_translations --force --dry` 零拒绝。**

剩 14 叶已逐叶裁为「本来就对，不动」：
`A Sage Welcome`（advance in level）· `Railen` / `Veiled Chain` / `Flame Guard`（组织内部位阶）·
`Grave Assignments`（dungeon level）· `Sage Advice`（advance in level）·
`Tethra Shùl.archetype` / `Necromancer`（EN 的 `threat rank` 是散文写法，
Crucible 定名是 `Threat Level`＝威胁等级，中文本就对）。

## 我自己在这条线上踩的两个坑（都记下来）

1. **`\bRanks?\b` 忘了 `re.IGNORECASE`。** 大写 `Rank` 只有 34 处、小写 `rank` 200+，
   于是 `crucible.rules` 的 Skill Checks 被误判成「EN×1 : CN×6 计数不等」，
   差点当成假阳性放过 —— 实际是 6:6 完全对齐。
   ⚠ **`split_region_area_map.py` 顶部已经用大字警告过同一个坑，我还是踩了。**
   说明这条教训写在单个脚本的注释里是不够的，应当进 PROJECT.md 的方法教训。
2. **上下文分类没剥 HTML 标签。** `Bonus Rank Scale` 在原文里是被 `<td>` 拆成三格的，
   按整串匹配一个都命中不到，16 叶因此判 UNKNOWN。剥标签后归零。

## 待办（交给断言路）

`distinct_terms` 必须改成读库的判据，且能报出「扫了多少键/叶」。
改完这 184 处应当全绿，**并且要做敏感度回测**：故意把一处改反，确认它变红。
