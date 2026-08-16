# crucible 0.10.1 → 0.10.2 上游增量（同一抽取器、同一 mapping，两次直读 LevelDB）

| pack | 新增叶 | 新增字符 | 消失叶 | 文本变化叶 |
|---|--:|--:|--:|--:|
| rules | 2 | 455 | 0 | 2 |
| playtest | 0 | 0 | 0 | 2 |
| pregens | 0 | 0 | 0 | 2 |
| talent | 0 | 0 | 0 | 4 |

合计：新增 2 叶 / 455 字符 · 消失 0 叶 · 文本变化 10 叶

## 新增（上游 0.10.2 才有的可译文本）

| pack | 条目 | 叶 | 字符 | cn 侧已有? |
|---|---|---|--:|---|
| rules | `Conditions` | `pages.Overrun.name` | 7 | **无** |
| rules | `Conditions` | `pages.Overrun.text` | 448 | **无** |

## 文本变化（键没变、英文原文变了 ⇒ 已有译文对应的是旧英文）

| pack | 条目 | 叶 | 旧字符 | 新字符 | cn 侧有译文? |
|---|---|---|--:|--:|---|
| rules | `Combat` | `pages.Engagement and Flanking.text` | 4274 | 4731 | 有（已过时） |
| rules | `Conditions` | `pages.Flanked.text` | 642 | 827 | 有（已过时） |
| playtest | `Playtest 1 - The Ring of Valor` | `actors.Zarajah.items.Counterspell.description` | 687 | 145 | 有（已过时） |
| playtest | `Playtest 1 - The Ring of Valor` | `actors.Fizzit.items.Counterspell.description` | 687 | 145 | 有（已过时） |
| pregens | `Fizzit` | `items.Counterspell.description` | 687 | 145 | 有（已过时） |
| pregens | `Zarajah` | `items.Counterspell.description` | 687 | 145 | 有（已过时） |
| talent | `Berserker` | `description` | 95 | 206 | 有（已过时） |
| talent | `Counterspell` | `description` | 687 | 145 | 有（已过时） |
| talent | `Duelist` | `description` | 520 | 555 | 有（已过时） |
| talent | `Eye of the Storm` | `description` | 201 | 141 | 有（已过时） |
