# `crucible-cn-0.9.0-shipped-en` —— crucible-cn `0.9.0` 发版当时的英文（15 包，含 `adversary-equipment`）

## 这份是从哪来的（可复现）

**从 `2-Crucible汉化插件` 自己的 git 历史里取的**，照 `ember-cn-v1.1.0-shipped-en/` 的体例：

```bash
cd 2-Crucible汉化插件
for f in $(git ls-tree -r --name-only 0.9.0 | grep '^compendium/en/.*\.json$'); do
  git show "0.9.0:$f" > "<本目录>/$(basename "$f")"
done
```

导出 16 个文件 = 15 个包 + 上游那份 `_source.json`（`packageVersion: crucible 0.10.1`,
`extractedAt: 2026-08-05T18:47:09Z`）。

**为什么是 `0.9.0` 而不是别的 tag** —— 逐 tag 数过 `compendium/en/`：

| tag 区间 | 打 tag 日期 | `compendium/en/` 里的包 | 含 `adversary-equipment` |
|---|---|---|---|
| `0.8.9.1` – `0.8.9.5` | 2026-03-22 起 | 12 | 否 |
| `0.8.9.6` – `0.8.9.13` | 2026-04-16 起 | 14 | 否 |
| **`0.9.0`** – `0.9.9` | **2026-08-09** 起 | **15 + `_source.json`** | **是** |

`crucible.adversary-equipment.json` 是被 `ea8783e`（2026-08-06,「对齐 crucible 0.10.1 +
改造到 babele 2.9.1」）一次性带进仓的，`git log --full-history -- compendium/en/crucible.adversary-equipment.json`
全库只有这**一个**提交。`0.9.0` 是**最早包含它的 tag**，再往前没有。

---

## ⚠ 先读这一条：这份基准答的是一个**零宽区间**

本目录的 15 个包与 `2-Crucible汉化插件/compendium/en/` 逐叶对照（2026-08-15 实测，
工具 `4-临时脚本/2026-08-15-round21/diff_baseline_dirs.py`）：

| | 合计 |
|---|---|
| 本目录叶数 | 4 567 |
| 当前 `compendium/en` 叶数 | 4 576 |
| 仅本目录有的路径 | **0** |
| 仅当前有的路径 | **9**（全在 `crucible.playtest.json` 的 `encounterTokens`，来自 `mappings.mjs` 变更） |
| 两侧都有**但值不同** | **0** |

也就是说：**本目录的英文 = `crucible-0.10.1/`（上游英文的原始抽取）逐字节相同**，
和 `crucible-0.10.1-preupgrade-2026-08-15/` 也只差那 9 条新增路径。

**推论（必须照实说）**：凡是「拿旧英文和新英文比差异」的闸，用本基准跑出来的 0
**不是「查过了没问题」，是「无从查起」** —— 两侧英文一个字都不差，差集本来就是空的。

---

## 那它还有什么用

三件事，都不是「答历史 drift」：

1. **把包覆盖补齐**。`crucible-0.9.1-legacy/`（14 包）跑五条闸时会点名
   「基准缺 `crucible.adversary-equipment.json`（106 条中文叶）」；换成本目录后
   五条闸都报「本次扫 **15** 个包 / 仓里 en 共 15 个包 / 基准缺 **0** 个」。
2. **`scan_marker_followup --all-leaves` 用它是真检查**（那个开关不要求「英文变过」）：
   2026-08-15 实跑纳入 4 567 对、标记完全一致 4 567、三档告警全 0，
   其中 `adversary-equipment` 106 对**确实逐条比过**。
   这一条是**「查过了没问题」**，与上面那句「无从查起」不是一回事。
3. **`scan_renamed_terms --mode cn-term` 的定义域覆盖到 15 包**（它的候选来自 `--old` 目录）。
   见下一节的变异回测。

---

## 五条闸 2026-08-15 实跑结果（口径逐条标注）

命令都在 `4-临时脚本/2026-08-15-round21/`，报告 `*_090.json`，词表一律显式传
`--glossary 5-其他内容/glossary/glossary_ec.json`（不传会静默跑成空词表）。

| 闸 | 包覆盖 | 结果 | 这个 0 是什么意思 |
|---|---|---|---|
| `scan_en_drift` | 15/15，缺 0 | 英文变过 **0**，可疑 0 | **无从查起**（两侧英文相同） |
| `scan_dropped_terms` | 15/15，缺 0 | `DROPPED_TERM_KEPT` **0** | **无从查起**（进闸条目 0；词表 1 924 词条、id→名 22 610，**不是空词表**） |
| `scan_number_drift` | 15/15，缺 0 | `STALE_NUMBER` **0** | **无从查起**（配对 4 567 对，但「英文数字变过」0） |
| `scan_marker_followup`（默认） | 15/15，缺 0 | `MARKER_STALE/MISSING/EXTRA` 全 **0** | **无从查起**（进闸对 0，它只看英文变过的） |
| `scan_marker_followup --all-leaves` | 15/15，缺 0 | 纳入 **4 567** 对，标记完全一致 4 567，全 **0** | ✅ **查过了没问题** |
| `scan_renamed_terms --mode cn-term` | 15/15，缺 0 | findings **0** / excluded 0（词表 7 974 条、ids 1 298） | **无从查起**（候选＝「旧英文里有、当前英文里没了」的专名，两侧相同 ⇒ 候选集恒空） |

### 上面那句「候选集恒空」是变异回测出来的，不是推断

`4-临时脚本/2026-08-15-round21/probe_renamed_b.py`（落盘脚本，可重跑）：
把 `compendium/{en,cn}` 整份拷进 scratchpad 造一个假仓，只在**假仓 en 侧**把
`crucible.adversary-equipment.json` 的条目 `Pseudopod` 改名成 `Ambulatory Bleb`，
cn 侧不动（＝模拟「上游改名、中文还留着旧译名 伪足」），`--old` 仍指本目录。

```
A 未注入: 扫 15 包 / 缺 0 · findings 0
B 已注入: 扫 15 包 / 缺 0 · findings 1  phrase='Pseudopod' cn='伪足'
          命中 crucible.adversary-equipment.json :: entries.Pseudopod.name
判定 PASS
```

⚠ 探针第一版挑的受害条目是 `Exoskeleton`，注入后照样报 0 —— 因为同包还有个
`Fused Exoskeleton` 含着它，`cur.has_phrase()` 仍为真、候选被跳过。
**受害条目必须挑「全库 en 里只出现 2 次（键 + name 值）」的**，否则探针自己会假绿。

---

## 与 `english-baseline/` 下其它 crucible 目录的分工（谁也替不了谁）

| 目录 | 包数 | 答哪个区间 | 今天跑出来是什么 |
|---|---|---|---|
| `crucible-cn-0.8.9.1-shipped-en/` | 12 | **2026-03-22 → 今天**（最长）。缺 `adversary-equipment`/`affixes`/`macros` | 英文变过 356 · 可疑 35 · `DROPPED_TERM_KEPT` 3（**逐条核过，3 条全是良性**，见该目录 README） |
| `crucible-0.9.1-legacy/` | 14 | **crucible 系统 0.9.1（≈2026-04-21）→ 今天**。缺 `adversary-equipment` | 英文变过 295 · 可疑 22 · 其余三闸 0 |
| **本目录** | **15** | **2026-08-09 → 今天 ＝ 零宽**（英文与当前逐字节相同） | 差异型闸恒 0；只用来补包覆盖 + 跑 `--all-leaves` / `cn-term` |
| `crucible-0.10.1/` | 15 | 不是 drift 基准，是「上游英文的原始抽取」，被 `tm/build_glossary.py` 等按名字引用 | —— |
| `crucible-0.10.1-preupgrade-2026-08-15/` | 15 | **留给下一次上游升级**当「旧英文」 | 今天恒 0（它就是当前英文） |

> `crucible-0.9.1-legacy/` 的内容 = tag `0.8.9.13` 的 `compendium/en/`
> **逐叶完全相同**（3 823 叶，路径与值 0 差异；2026-08-15 实测）。
> 所以它也是可从 git 复现的，不是孤本。

---

## ⚠ `adversary-equipment` 的历史 drift **今天无解，而且这是对的**

这个包是 crucible **0.10.1 才有的**；插件仓的 `compendium/en/` 又是在
`ea8783e`（已经对齐 0.10.1 之后）才建的。
⇒ **全世界（本项目范围内）不存在这个包的更早英文** —— 不在 git，不在任何基准目录。

所以「它的五条闸从来没跑过」这个缺口，**能补的部分**（包覆盖、`--all-leaves`
标记比对、`cn-term` 定义域）本目录已经补上；**不能补的部分**（它 2026-08-06 之前
的英文长什么样）要等**下一次上游升级**，届时靠
`crucible-0.10.1-preupgrade-2026-08-15/` 当旧英文，那时才第一次能对它做真正的 drift 比对。

**别把本目录当成「历史基准」去替换 `crucible-0.9.1-legacy/`** ——
那会把 295 条真实的英文变动一次性变成 0，看着更绿，实则丢掉全部历史区间。
