# `crucible-0.10.2/` —— 上游 crucible **0.10.2** 的英文基准（15 包）

- **是什么**：2026-08-17 从 `2-Crucible汉化插件/compendium/en/` 原样拷下来的 **15 个包 ＋ `_source.json`**。
  `_source.json` 的 `packageVersion` = **`0.10.2`**、`extractedAt` = `2026-08-16T19:45:03.382Z`。
- **叶数**：15 包合计 **4 743** 条英文叶（不含 `_source.json` 自身的 40 条元数据叶；
  连 `_source.json` 一起数是 4 783，两个数别混用）。
- **归档理由**：`PROJECT.md` 第 477 行与 §5.5 第 2 步要求英文基准**存两处** ——
  插件仓 `compendium/en/`（当前版本，进 git）＋ 本目录（历史快照）。
  0.10.2 跟版做完时只刷了前一处，本目录是补齐的第二处。

## 本目录**同时**是「原始抽取」和「升级前快照」（这一版不必再截 preupgrade）

`english-baseline/` 下 0.10.1 那两份是**两种东西**（`crucible-0.10.1/` 是 `extract_en.mjs`
的原始抽取、`crucible-0.10.1-preupgrade-2026-08-15/` 是 `capture_baseline.py` 从
`compendium/en` 截的升级前快照），而且**内容不同**（差 9 叶，见下节）。

本目录**没有这个二分**，因为两者在 0.10.2 上被实测证明等价：

```
node 3-常用脚本/extract/extract_en.mjs \
  --package C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible \
  --out <临时目录>
# 再与 2-Crucible汉化插件/compendium/en 逐包比
```

**2026-08-17 实跑结果**：15 个包**逐字节相同**（`byte=同`），4 743 叶
「仅一侧有 0 ／值不同 0」；唯一有差的是 `_source.json` 的 `extractedAt` 时间戳。
⇒ **crucible 侧没有任何本地补丁**（`LOCAL-PATCHES.md` 是 ember 专用表，crucible 一条都没有），
所以「从上游重抽」与「从 `compendium/en` 截」拿到的是同一份东西。
**下一次上游升级，直接拿本目录当「旧英文」即可，不需要另截一份 `-preupgrade-` 目录。**

## ⚠⚠ 做 0.10.1 的 diff 必须用 `-preupgrade-` 那份（本轮实测复现了那 9 叶）

| 拿谁当「旧英文」 | 0.10.1 → 0.10.2 算出来的差量 | 对不对 |
|---|---|---|
| `crucible-0.10.1-preupgrade-2026-08-15/` | **新增 2 叶 ／ 改写 10 叶** | ✅ 真值 |
| `crucible-0.10.1/` | 新增 **11** 叶 ／ 改写 10 叶 | ❌ **虚报 9 条新增** |

虚报的那 9 条**全部**是 `crucible.playtest.json` 的
`entries.Playtest 1 - The Ring of Valor.scenes.Arena of Valor.` 下的
`levels.Arena of Valor` ＋ `tokens.{Agnath,Belladonna,Duurath,Eliorwen,Fizzit,Kagura,Ulfen,Zarajah}`
—— 它们**不是 0.10.2 新增的内容**，是 `crucible-0.10.1/` 那份用**更旧一版 `extract_en.mjs`**
抽的、当时根本没抽 Scene 的 `levels`/`tokens` 字段（见 `PROJECT.md` §「抽取器根本没抽的字段」）。
**目录名看不出这件事，所以写在这里**：`crucible-0.10.1/` 是「老口径的原始抽取」，
不是「0.10.1 时刻的英文全貌」。

真差量的 12 叶（本轮跟版处理的就是这 12 条）：

- **新增 2**：`crucible.rules.json` → `entries.Conditions.pages.Overrun.{name,text}`
- **改写 10**：`playtest` 2（Fizzit / Zarajah 的 `Counterspell.description`）·
  `pregens` 2（同上两个 pregen）· `rules` 2（`Combat.pages.Engagement and Flanking.text` ／
  `Conditions.pages.Flanked.text`）· `talent` 4（`Berserker` / `Counterspell` / `Duelist` /
  `Eye of the Storm` 的 `description`）
- **删除 0**（compendium 侧；lang 侧另有增 8 ／删 5 ／改 2，不在本目录口径内）

> ⚠ **跟版差量要按「增 / 删 / 改」三个桶枚举。** 0.10.2 这一轮漏过一次第三桶
> （「改」），漏因是拿**已经升到 0.10.2 之后的上游**当「改前」基线 —— 基线被污染，
> 于是本该处理的改写被当成了背景噪声。**旧英文只能从 `english-baseline/` 里取，
> 不能从已升级的上游现抽。** 本目录存在的全部意义就是这个。

## 与其它目录的关系

见 `5-其他内容/english-baseline/README.md`（全目录索引与各自用途）。
