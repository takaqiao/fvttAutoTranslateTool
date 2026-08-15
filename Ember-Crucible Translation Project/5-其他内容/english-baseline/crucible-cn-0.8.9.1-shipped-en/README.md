# `crucible-cn-0.8.9.1-shipped-en` —— 插件仓里**最早**的一份英文（12 包，2026-03-22）

## 这份是从哪来的（可复现）

```bash
cd 2-Crucible汉化插件
for f in $(git ls-tree -r --name-only 0.8.9.1 | grep '^compendium/en/.*\.json$'); do
  git show "0.8.9.1:$f" > "<本目录>/$(basename "$f")"
done
```

12 个包，**没有 `_source.json`** —— 那会儿 `extract_en.mjs` 还不写这个文件，
所以本目录没有 provenance 元数据，只有 tag 名可考（tag `0.8.9.1` = 2026-03-22 07:28,
`module.json` 里 `crucible` 系统最低要求 `0.7.7`）。

同一 tag 的**中文**（`compendium/cn/`，12 包）也在 git 里，
`scan_renamed_terms --old-cn` 需要它时照上面的命令换 `cn` 即可导出。

## 为什么单独立这一份 —— `crucible-0.9.1-legacy/` **不是**最早的

2026-08-15 第二十一轮逐 tag 数过 `compendium/en/`：

| tag | 日期 | en 包数 |
|---|---|---|
| **`0.8.9.1`** | **2026-03-22** | **12** |
| `0.8.9.6` | 2026-04-16 | 14（多了 `affixes` / `macros`） |
| `0.8.9.13` | 2026-04-21 | 14 |
| `0.9.0` | 2026-08-09 | 15（多了 `adversary-equipment`）+ `_source.json` |

而 `crucible-0.9.1-legacy/` 的内容 **＝ tag `0.8.9.13` 的 `compendium/en/` 逐叶完全相同**
（3 823 叶，路径与值 0 差异，`4-临时脚本/2026-08-15-round21/diff_baseline_dirs.py` 实测）。
也就是说 legacy 答的是 **2026-04-21 → 今天**，而本目录能往前多答**一个月**。

本目录 vs `0.8.9.13`：共有路径上**值不同 75 叶**，且抽样看过全是**上游真的改了英文**
（`Common Clothing` 的 `labour→labor` 并删掉 "cannot be worn with armor"、
一批 Fine 装备的 `description.private` 整段重写成 affix 说明、
`Strong Grip` 加了 "melee" 限定、`Rune: Flame` 把 Intellect 加粗等），
**不是抽取口径噪声**。

## 代价：少 3 个包

本目录没有 `crucible.adversary-equipment.json`（106 中文叶）、
`crucible.affixes.json`（416）、`crucible.macros.json`（5）——
五条闸跑它时会照实报「本次扫 12 个包 / 仓里 en 共 15 个包 / 基准缺 3 个 /
合计未进闸的中文叶 527」。**所以它替不了 legacy，更替不了 15 包的那份**；
三份并存，各答各的区间（分工表见 `../crucible-cn-0.9.0-shipped-en/README.md`）。

---

## 2026-08-15 实跑结果（报告在 `4-临时脚本/2026-08-15-round21/*_08091.json`）

| 闸 | 本目录 | 对照：`crucible-0.9.1-legacy` |
|---|---|---|
| `scan_en_drift` | 配对 3 511 · **英文变过 356** · 「中文更贴合旧英文」**35** | 配对 3 740 · 变过 295 · 可疑 22 |
| `scan_dropped_terms` | **`DROPPED_TERM_KEPT` 3**（逐条核过，**3 条全良性**，见下） | 0 |
| `scan_number_drift` | `STALE_NUMBER` **0** · `MISSING_NUMBER` 0 · 良性 63 | 0 / 0 / 良性 43 |
| `scan_marker_followup` | 三档全 **0** | 三档全 0 |
| `scan_renamed_terms --mode cn-term` | findings **0**（候选 850 → 当前英文里没了的 **9** → 9 条都真的进了中文库搜索） | —— |

> 词表一律显式 `--glossary 5-其他内容/glossary/glossary_ec.json`
> （`scan_dropped_terms` 读到 1 924 词条 / id→名 22 610；
> `scan_renamed_terms` 读到 7 974 条 / ids 1 298）。**不传会静默跑成空词表。**

### `scan_renamed_terms` 的 0 是「查过了」而不是「无从查起」——有据

`4-临时脚本/2026-08-15-round21/probe_cnterm_domain.py`（import 真模块的函数，不复写判据）：

```
候选旧专名 850 个 ├ 当前英文里仍在 841  └ 当前英文里没了 9
9 个全都有中文写法可追：Disrupting Shot / Full Rest / Inquisitor's Strike /
Lightning Proficiency / Rune: Lightning / Spell Scroll (Legendary|Major|Minor) /
Toggle Talent Tree
```

⇒ 探测器 B **确实拿这 9 个旧译名在当前中文库里搜过**，一条残留都没有。
（同一探针对 `crucible-cn-0.9.0-shipped-en` 跑出来是「候选 1 179 → 没了的 **0**」，
那份的 0 就是**无从查起**。两个 0 含义不同，别混着报。）

### 那 3 条 `DROPPED_TERM_KEPT` —— 逐条核过，**全是良性，不要去改译文**

| # | 位置 | 报的是什么 | 核对结论 |
|---|---|---|---|
| 1 | `crucible.rules.json :: Combat.pages.Actions.text` | 旧英文 `Offhand`×2 / `Twohand`×1 没了，中文还有「副手」「双手持用」 | **上游只是加了连字符**：现英文写 `Main-Hand` / `Two-Handed` / `Off-Hand`（实测该叶各出现 1 次），中文「主手 / 双手持用 / 副手」译得对。判据的词切分把 `Offhand` 与 `Off-Hand` 当两个词 |
| 2 | `crucible.rules.json :: Combat.pages.Movement.text` | 旧 `Walk`×3 → 新×1，中文「行走」×2 | 现英文该叶有 `Walk`×1 + `walking`×1（"Movement while flying is faster than **walking**"），两处中文都被解释。默认口径只数大写词，小写的 `walking` 不进计数 |
| 3 | `crucible.rules.json :: Spellcraft.pages.Spellcraft Overview.text` | 旧 `Spellcraft`×9 → 新×7，中文「施法」×8 | 上游删掉的是「The Spellcraft system is still a work in progress…Playtest One…」那一段；**中文里根本没有这一段**（grep「进行中/开发/修订/迭代/试玩/Playtest」全 0）。中文是照新英文译的 |

---

## ⚠ 留给下一轮的活（本轮没做，因为不属于本单元的文件所有权）

`scan_en_drift` 的「中文更贴合**旧**英文」是**长度比启发式**，不是缺陷判定。
本目录报 35 条、legacy 报 22 条，两边取差集，**只有拿这份更早的基准才看得见的有 14 条**：

```
crucible.equipment.json :: Cloak of Kindly Visage.description.private
crucible.equipment.json :: Common Clothing.description.public
crucible.rules.json     :: Character Mechanics.pages.Defenses.text
crucible.rules.json     :: Combat.pages.Engagement and Flanking.text
crucible.rules.json     :: Conditions.pages.Broken.text
crucible.rules.json     :: Conditions.pages.Incapacitated.text
crucible.rules.json     :: Conditions.pages.Stunned.text
crucible.rules.json     :: Conditions.pages.Weakened.text
crucible.rules.json     :: Crafting.pages.Tradeskills Overview.text
crucible.rules.json     :: Equipment.pages.Weapons.text
crucible.rules.json     :: Spellcraft.pages.Inflections.text
crucible.rules.json     :: Welcome To Crucible.pages.Module Recommendations.text
crucible.rules.json     :: Welcome To Crucible.pages.Providing Feedback.text
crucible.rules.json     :: Welcome To Crucible.pages.What is Crucible.text
```

（反向：只有 legacy 看得见的 1 条 —— `crucible.rules.json :: Combat.pages.Actions.text`。）

这 14 条**逐叶读旧英文/新英文/中文三方对照**才能定性，属译文复核，需要能改
`compendium/cn` 的单元来做。**在有人逐条读过之前，别把它当成 14 个缺陷，也别当成 0。**
