# `crucible-0.10.1-preupgrade-2026-08-15/` —— 升级前的**全包**英文快照

- **是什么**：2026-08-15 从 `2-Crucible汉化插件/compendium/en/` 原样拷下来的 **15 个包**
  （上游 `crucible 0.10.1` 系统 + 当前 `mappings.mjs` 抽取口径，**无任何本地补丁**）。
  叶数见 `_source.json`：en / cn 各 4 576 叶。

  > **crucible 侧没有本地补丁，这一点第二十轮逐叶验过。**
  > `LOCAL-PATCHES.md` 是 **ember 专用**的表（该表 2026-08-15 从 4 条订正为 **5** 条），
  > crucible 侧一条都没有 —— 15 个包与 `crucible-0.10.1/` 原始抽取对照，
  > **值不同的叶数为 0**（见下一节）。所以 crucible 重抽后**不需要**重打任何补丁。
- **拿它跑今天恒为 0 告警**，因为它就是当前英文。**这是对的，不是闸坏了。**
- **用途是下一次上游升级之后**当「旧英文」，让三条 drift 闸
  （`scan_dropped_terms` / `scan_number_drift` / `scan_marker_followup`）全包覆盖。

## 与已有的 `crucible-0.10.1/` 差在哪

逐叶实测：**只差 9 叶**，全在 `crucible.playtest.json`，且全是**新增路径**
（`encounterTokens`，来自 `mappings.mjs` 变更），没有一叶的值发生改变。

> **2026-08-15 第二十轮复核：上面这三个数字全部复现，一字不改。**
> 用 `4-临时脚本/2026-08-15-round20/diff_leaves.py` 逐包跑了 15 个包：
> `crucible.playtest.json` 当前 1 014 叶 / 快照 1 005 叶、仅当前有 **9** 条路径、
> 值不同 **0**；其余 14 个包路径与值**全 0 差异**。15 包合计 4 576 叶，与 `_source.json` 相符。
> （ember 侧同口径的那句「只差 10 叶」就没这么幸运 —— 那句漏数了 2 288 处路径差，
> 已在 `ember-0.6.0-preupgrade-2026-08-15/README.md` 里订正。）
既然如此为什么还要单截一份 —— 为了**一条统一规矩**：
`crucible-0.10.1/` 是「上游英文的原始抽取」，会被 `tm/build_glossary.py` 等按名字引用；
本目录是「升级前快照」，只服务三条 drift 闸，两者职责分开，
下一轮谁也不必再去判断「那份原始抽取和当前 compendium/en 到底还差多少」。

ember 侧同规格的那份是 `ember-0.6.0-preupgrade-2026-08-15/`（10 包），
两个目录的来龙去脉、以及 `english-baseline/` 下六个目录各自的用途，写在那份 README 里。

截快照的工具是 `3-常用脚本/qa/capture_baseline.py`
（第二十轮从 `4-临时脚本/2026-08-15-round19/` 挪过来的：它是常驻工具，不是临时产物）。
本目录 `_source.json` 里的 `capturedBy` 仍写着旧路径 —— **那是历史记录、不改**，
截这份快照的时候它确实在那儿。

## 顺带记一笔：crucible 侧今天的覆盖缺口

crucible 侧三条闸今天的基准是 `crucible-0.9.1-legacy/`（14 包），
缺 `crucible.adversary-equipment.json`（**106 条中文叶**）—— 那个包是 0.10.1 才有的，
0.9.1 里本来就不存在，**属于合理的缺口，不是漏截**。
现在三条闸每次都会把它点名报出来（「本次扫 14 个包 / 仓里共 15 个包 / 缺 1 个」），
不再是静默跳过。
