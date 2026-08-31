# 第十四轮 · 复核（2026-08-14）

把第十三轮遗留的 `REMAINING.json`（270 条）**逐条对当前 HEAD 重新核实**。

**为什么必须重核**：那份快照写于 08-14 04:16，而两个仓库 05:02 又各发了一版
（ember **v1.1.10** / crucible **0.9.7**，"运行时替换层覆盖面修复（严重档第一批）"），
其中相当一部分已经修掉。照抄 finding 会同时犯两个错：重复劳动，以及
把已经改对的地方按过期描述改坏。

## 结果

| 判定 | 条数 | 含义 |
|---|---|---|
| **OPEN** | **203** | 复现了，现在还在 |
| FIXED | 63 | v1.1.10 / 0.9.7 已修，逐条拿当前代码复核过 |
| UPSTREAM | 2 | 真问题，但根因在 crucible/ember 自己，两个汉化模块修不了 |
| INVALID | 1 | finding 的前提就是错的 |
| （漏）| 1 | `idx 264`，一个分片只返回了 19/20；主控已自行核实并判 OPEN，见 `MAINCTRL-PENDING.md` |

OPEN 的分档：**严重 13 / 一般 135 / 观感 55**。

> 复核用了 14 路并行 + 1 路归并；归并那一路**撞了 64k 输出上限失败**，
> 所以工作包是主控用 `cluster.py` 机械聚出来的（聚类只负责「一个文件只归一个 agent」，
> 语义去重留给拿到包的 agent）。原始 verdict 全在 `verdicts.json`。

## OPEN 的落点分布

| 文件 | 条数 |
|---|---|
| `ember-hardcoded-cn.mjs` | 90 |
| compendium（各包合计） | 38 |
| `babele-register.js` | 17 |
| `lang/cn.json`（两仓） | 16 |
| `register.js` | 12 |
| `mappings.mjs` + `babele-mappings.js` | 9 |
| `3-常用脚本/` 工具链 | 9 |
| 文档 | 6 |
| `module.json` / `.css` | 6 |

**压倒性地集中在代码面而不是译文面** —— 与第十三轮首次把插件代码纳入镜头时的结论一致。

## 13 条严重档

| idx | 落点 | 一句话 |
|---|---|---|
| 34 | compendium + 管线 | 遭遇战预置 token 覆盖名 **764 处**：管线已接上但**英文基线从未重抽**，一条译文都没落盘 |
| 41 | compendium + 管线 | 同上，任务事件遭遇模板里另有 **130 处** |
| 8 | `lang/cn.json` | `SPELL.RUNES.LifeAdj` 仍是「至关重要的」，会合成出「至关重要的打击」（Vital 取错义项） |
| 261 | `lang/cn.json` | `ui.notifications` 裸英文一条键都没加，实测 29 条 miss |
| 56 | ember 孪生包 | 屠龙毒药「8轮内造成6伤害」，与全库 8 处同句译法**机制含义相反** |
| 57 | ember 孪生包 | Kali Andrella 的 Control Water `the next round` 作「回合」，同刊另两张同法术卡作「轮」 |
| 63 | 四包 | 模板族译文碎片化（同英文串多种中文） |
| 80 | 两包 + 文档 | `Thayloc Courser` 一个生物三套中文身份，两包各 4+21 处 + 3 处裸英文 |
| 108 | `.mjs` | `CALENDAR_AGES` 四个纪元名 + 三个相对年份标签零覆盖 |
| 144 | `.mjs` | 通知条整面无入口，插件代码面 0 命中 |
| 156 | `.mjs` | 结构性不可达已解除，但仍有 5 处正文/标题落在表外 |
| 169 | `.mjs` | 12 个被上游 `prepareAbilities` 运行时覆写的天赋名仍无兜底 |
| 266 | `.mjs` + 文档 | `CALENDAR_MONTHS/DAYS` 20 条仍是零读者，注释里的因果**写反** |

## 工作包切法

按「**一个文件只归一个 agent**」切成 13 个包（`packages/*.json`，明细 `WORKPACKAGES.md`）。
这条是硬约束不是优化：本项目实测一轮里有 350 条路径被一个以上批次认领，
直接按顺序落盘会让先落的批次被后落的**静默回滚**。

- `ember-hardcoded-cn.mjs` 是同一个文件、90 条，切成**三段顺序**跑（表 → 闸 → 通道），
  后一段拿到前一段的产出摘要以免回改。
- compendium 与 lang 一律**只产批次文件**，不直接改 `compendium/cn` / `lang/cn.json`，
  由主控统一做三方合并后 `--force` 落盘。
- `generate_runtime.mjs` 与 `sync_twin_packs.py` 一律由主控最后跑。
