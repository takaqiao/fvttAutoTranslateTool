# -*- coding: utf-8 -*-
"""删除型漂移闸：上游**删掉**一个术语，中文还留着它的译名。

    python scan_dropped_terms.py --repo <repo> --baseline <旧英文基准目录>
                                 --bindings <dump_bindings.mjs 的输出>
                                 [--glossary <glossary_ec.json>] [--out <json>] [--md <md>]
                                 [--all-case] [--min-sim 0.75] [--show 30]
                                 [--no-block-filter]

⚠ 两个口径必须照抄，跑偏过的都在这儿
====================================

**一、`--bindings` 是必填**（2026-08-15 从「强烈建议」升级；`--no-bindings` 才放行）
--------------------------------------------------------------------------------
同一个库、同一份基准，**带 81 条、不带 98 条**，差额全是假阳性：

    2026-08-15 实测 `1-Ember汉化插件` × `ember-cn-v1.0.15-shipped-en`
        带 --bindings  → DROPPED_TERM_KEPT **81**
        不带           → DROPPED_TERM_KEPT **98**（多报 20 叶、少报 3 叶）

多报的 20 叶全是「上游把明文术语换成**裸 `@UUID`**、中文侧标签仍是译名」这一形态。
实例 `The Bleak Archive.pages.Main Gallery.text`：英文的 `Multiattack` /
`Shadow Gait` / `Inner Antechamber` 都改成了不带 `{标签}` 的 `@UUID`，
中文写 `@UUID[...]{多重攻击}` 是**正当标签**（不写页面上就冒英文原名），
不给 idmap 时英文侧那三个词「凭空少了」，于是全被判成「中文停在旧版」。
（少报的 3 叶反过来：`Falar` / `Whisperer` / `characters` 靠 idmap 补回英文词才数得出。）
所以**不带 bindings 的数字既不是上界也不是下界，直接不可用**，脚本现在拒绝跑。

先跑同目录的 `dump_bindings.mjs`，把输出（可 `--bindings a.json --bindings b.json`
给多份）传进来。真要看不带 idmap 的对照，显式加 `--no-bindings`。

**二、基准目录选哪个 —— 别用 `english-baseline/ember-0.6.0/`**
--------------------------------------------------------------
`5-其他内容/english-baseline/` 下有六份，用途完全不同，本闸要的是**发版当时的旧英文**：

    ember-cn-v1.0.15-shipped-en/_repaired.json   ← ember 侧本闸的基准（就是它；只有 3 个包，见下）
    crucible-0.9.1-legacy/                       ← crucible 侧本闸的基准（14 包，缺 0.10.1 才有的 adversary-equipment）
    crucible-0.10.1/                             ← 当前上游英文，不是基准
    ember-0.6.0/                                 ← **当前上游英文的镜像，拿来 diff 等于没 diff**
    ember-0.6.0-preupgrade-2026-08-15/           ← 第十九轮新截的**全包**快照（10 包），给**下一次**升级用
    crucible-0.10.1-preupgrade-2026-08-15/       ← 同上（15 包）

**三、基准缺包 = 静默不扫（第十九轮 Y5）**
------------------------------------------
本闸只扫**基准目录里存在**的包：`scan()` 里 `if not os.path.exists(cur_p): continue`
会静默跳过，报告长得和「扫过、干净」一模一样。实测 2026-08-15：
`ember-cn-v1.0.15-shipped-en/` 只有 **3** 个包（`_repaired.json`＋adversary＋character），
仓里 10 个 en 包 / 9 个有中文，另外 **6 个包 15 147 条有中文的叶从未进过本闸**。
现在每次运行都印「本次扫 N 个包 / 仓里共 M 个包 / 缺 K 个」＋缺掉的中文叶数＋逐包对数，
并写进 `meta.coverage` / `meta.per_pack`；`--strict-coverage` 让缺包直接以退出码 3 结束
（默认只告警 —— 回测脚本会拿只有一个包的小仓跑，默认非零会把它们全打红）。

**这个洞补不了「过去」，只补得了「将来」**：基准是历史快照，那 6 个包在 v1.0.15
发版当时就没被捕获，「上游从那时到现在改了什么」对它们**根本无法回答**，不是脚本的错。
所以第十九轮的做法是**升级前先截全包快照**
（`3-常用脚本/qa/capture_baseline.py`，命名里写死上游版本号）。
拿 `*-preupgrade-2026-08-15/` 跑**今天**恒为 0 告警（它们就是当前英文），
那是对的、不是闸坏了 —— 它们的用途是下一次上游升级之后当「旧英文」。

`ember-0.6.0/` 与当前 `compendium/en/` 逐叶几乎相同：34,483 叶里**只有 10 叶英文变过**
（其余差异全是新增叶，本闸只看路径相同的对）。拿它当基准，「英文变过且有中文」
只剩 10 条 → 告警恒为 **0**，看报告的人会以为这一闸是假的 / 库是干净的。
用 `ember-cn-v1.0.15-shipped-en` 则是 933 条候选 → 81 告警。

⚠ 文档互相打架，别信文档信这里：`LOCAL-PATCHES.md` 写着「权威基准见 ember-0.6.0/」
（那说的是**核对上游原文**的权威副本，不是漂移基准），而
`ember-cn-v1.0.15-shipped-en/_source.json` 的注释又说自己「仅作历史留存」——
两处说明与本闸的实际用法都对不上。**因此本脚本把实际用的 baseline 绝对路径写进
报告 json 的 `meta.baseline`**（连同 repo / glossary / bindings / 命令行），
下一轮先看 `meta` 就不会像 2026-08-15 那轮一样先跑偏半天。

为什么必须单独设这一闸（§8 2026-08-13k 判据的结构盲区之一）
------------------------------------------------------------
`scan_en_drift.py` 挑出「英文变过」的条目后，此前的分诊判据是
**「新英文独有的 token 在不在中文里」**。它的方向是单向的 —— 只看**新增**。
上游把一个词**删掉**时，新英文没有任何独有 token，这条判据从头到尾一声不吭。

上一轮 1217 条 CHANGED 里唯一的真缺陷正是这个形态：

    旧：Increases in level grant additional Ability, Skill, and Talent points to spend
    新：Increases in level grant additional Ability and Talent points to spend
    中：等级提升会给予额外的能力、技能和天赋点数可供分配   ← 「技能」该删没删

数字改动那一路归 `scan_number_drift.py`，标记改动那一路归 `scan_marker_followup.py`，
**词被删掉**这一路就是本闸。

判据（先查英文再判中文）
------------------------
对**基准与当前英文里路径相同、且有中文**、且**英文变过**的每一条：

0. 词级相似度（匹配数/较短一侧长度）低于 `--min-sim` 的条目直接跳过 ——
   整段推倒重写时逐词数次数没有意义。
1. 归一到「玩家读到的词」：`@UUID/@Embed` 带 `{标签}` 的取标签、不带的用
   `--bindings` 里目标文档的英文名代替；其余 enricher（`@Condition[flanked]`）
   把方括号里的词展开；HTML 标签与属性丢掉。**这三条各自都踩过坑，见下。**
2. 逐词做**单复数归一**（`Skills`→`skill`），然后**数次数**：
   `旧英文出现 N 次 → 新英文出现 M 次`。
3. 只看 `M < N` 的词（掉了至少一次）。查 `glossary_ec.json` 拿它的既定译名；
   取译名开头那串汉字（库里的值常带中英对照尾巴，`'Skills' -> '技能 Skills'`）。
4. 译名**长度 ≥ 2 个汉字**、且**中文里出现次数 > M** 时报 `DROPPED_TERM_KEPT`。
   长度门槛是必须的：一字译名（`阶`/`轮`）在中文里几乎必然撞上，报了也没法用。
5. **逐位对齐过滤**（2026-08-15 第十八轮新增）：把第 4 步那条判据**缩到删除点所在的
   那一块上再判一次**，两次都成立才报。详见下一节。

⚠ 第 5 步：逐位对齐过滤（整叶计数的假阳性主类靠它压掉）
========================================================

第 1~4 步是**整叶**口径 —— 「英文这叶里 `event` 少了一次、中文这叶里『事件』还有 6 次」。
一叶常有几百个块，于是「英文在 A 段删了词」和「中文在 F 段正当地用了同一个译名」
分不开。第十六轮抽样 82 条里 **70 条**是这么来的（真缺陷 5 / 需先定译名 7）。

做法照抄 `4-临时脚本/2026-08-15-round16/probes/split_dives.py` 的切块法：
**按 HTML 标签切块，再逐块对齐**。

    ① 新英文与中文各 `TAG_SPLIT.split()`，块号一一对应；
    ② 旧英文→新英文用已有的 `SequenceMatcher` 逐词对齐，于是**每个旧词都能锚到
       一个新英文块号**（`anchor_old_to_new_blocks`）；
    ③ 对每个被删词干，取它所有删除点锚到的块，**逐块**重算 (旧 N, 新 M, 中 C)，
       任意一块仍满足 `M < C <= N` 才放行。

**为什么中文能跟新英文对齐（而不是跟旧英文）**：译文仓的中文一直被 markup 类闸
逼着跟当前英文保持结构同步，所以中文是「结构已迁到新版、文字可能停在旧版」。
2026-08-15 实测这条是硬的：告警叶 **77/77**（crucible 1/1）**新英文块数 == 中文块数**，
而**旧英文块数 == 中文块数只有 16/77** —— 中文跟的确实是新英文。

⚠ **对齐要用块数、不要用标签串**。同一批 77 叶里「标签串逐字节相同」只有 **20/77**
（中文侧 `id=` 锚点、`data-*` 属性与英文侧不一致），拿标签串当判据会把 57 叶
无谓地打回整叶口径。**块数才是干净的分界线。**

⚠ **`delete` 的候选块必须取删除点前后两块**。纯删除在新侧塌成一个**点**，
删除点正好在块尾时「后一个词」已经属于下一块了，只取后一块会把候选选歪、**漏报真缺陷**
（回测 TP7 就是专门卡这条）。`replace` 不存在这个问题，取新段覆盖的那些块即可。

⚠ **逐块单独判，不要把候选块并成一个集合再判**。并集等于在小一号的尺度上重新引入
整叶那种稀释；实测并集 40 叶 / 逐块 39 叶，逐块既更细也更少。

⚠ **块结构对不上时退回整叶口径，并计数**（`shape_mismatch_*`），不静默跳过。
ember 侧 757 叶里有 **2 叶**退回，成因是上游脏数据：`@Condition[exhaustion`
少了右方括号、又被 `</sub>` 从中间切开（`Toothbreaker Hideout.pages.Prison` /
`Players' Guide.pages.Region Exploration`），切块后 enricher 断成两截，
归一出来的词序列与整叶归一不一致。这两叶按整叶口径判，均未告警。

⚠ **块粒度是这条路的下限，别再往细里收**。剩下的 9 叶弱告警（① 同义改写类）局部
三元组是 `(旧2, 新1, 中2)` —— 与历史上那条真缺陷（`Ability, Skill, and Talent`
→ `Ability and Talent`，同块内另一句还留着小写 `skill`，回测 TP1）**数值上完全同形**。
所以「块内该词归零才报」之类的收紧必然连 TP1 一起杀掉。**第十八轮实测到此为止：
77 → 39，不是估的 12；再降就要拿灵敏度换。**

⚠ **必须数次数，不能只看「还在不在」**（本闸第一版就栽在这里）
--------------------------------------------------------------
第一版判据是「这个词在新英文里彻底消失了才算删」。拿上一轮那条真缺陷一测，
**报不出来**：`Character Creation.pages.Overview.text` 是整章一页，新英文里
另一句还留着一个小写 `skill`（“the skill and talent advancements”），
于是「彻底消失」永远不成立。改成数次数后 `skill 2→1`、中文「技能」仍是 2 次，
立刻报出。**长叶子上，词频差才是信号，词的有无不是。**

默认只认**旧英文里至少出现过一次首字母大写形式**的词（Foundry 的规则术语惯例：
`Skill` / `Talent` / `Bane`）。`--all-case` 放开到全部小写词 —— 噪声会涨一个量级，
只在专项复核时开。

已知假阳性（第十八轮逐位对齐上线后，四类里三类已被压掉）
--------------------------------------------------------
第十六轮的四类成因，以及各自现在的下场：

* ①**常用词撞名 / 同义改写**（当时 ≈46 条，最大头）：上游把 `characters` 改成
  `party`、把 `In this event,` 改写成别的说法，而中文那些命中来自**同叶别处的正当用法**。
  → **大部分被压掉**（删除点所在的中文块里没有那个译名）。**剩 9 叶压不掉**：
  删除点就在中文那一块里，且中文在同一块内重复了名词（英文用代词、中文还原成名词）。
  典型 `A Brush With Death.pages.Locating Kel Kornan`：英文
  `This is a social event… This event steers…` 合并成 `This Social Event occurs… and steers…`，
  局部 `event 旧3 新2`、中文「事件」3 次 —— 中文那 3 次全是对的。见上一节最后一条 ⚠。
* ②**整块删除、中文已跟进**（当时 ≈18 条）：上游删掉整节，中文那些块早就不在。
  → **全部被压掉**，因为删除点锚到的中文块里根本没有那个译名。
* ③**同形异义**（当时 ≈4 条）：英文删的是 `Maximum Focus`、中文的「专注」是
  「主要专注于…」的动词义；英文删的是表头 `Result: Critical Success`、中文的「成功」是
  「成功通过一次检定」。→ **被压掉**（两处汉字落在别的块里）。实例已复核：
  `The Winding Trail.pages.Giant Moonstone`（`Focus` 2→0）、
  `The Winding Trail.pages.A Promised Exit`（`Success` 3→0）。
* ④**词表一词多义**（当时 ≈2 条）：`glossary_ec` 只存一条主译名。→ **仍会报**，
  这是词表的口径问题，不是对齐能解决的。`Jahud`→`assassin`（贾胡德/刺客本项目同指）。
* 中文的**中英对照尾巴**（`天赋 Talent`）里本来就抄着英文，不影响本闸（只查汉字）。

回测（两份，都要过；本闸自身没有 `--selftest`）
------------------------------------------------
**一、`4-临时脚本/2026-08-15-round16/qa/backtest_dropped_terms.py`**（第 1~4 步）
双向 6/6 PASS。灵敏度 3/3：三项并列删中间一项（＝上一轮那条真缺陷，且同叶别处
还留着同一个词的小写形式）、整句删一个术语、公式里换掉一个属性名。
特异度 3/3 静默：裸词升级成 `@Condition[…]`、明文换成不带标签的 `@UUID`、整段重写。
第十八轮加过滤器后**复跑仍 6/6**。

**二、`4-临时脚本/2026-08-15-round18/backtest_block_filter.py`**（第 5 步）
双向 **12/12 PASS**（灵敏度 7/7 · 特异度 5/5）。⚠ **每个用例跑两遍** ——
带过滤器与 `--no-block-filter` —— 所以「结论是不是过滤器改的」有对照，不靠猜；
判定键在**目标词干**上，不看整叶有没有告警（注入文本里的 `relic`/`paladin`
在词表里也有译名，会自带无关命中）。

    灵敏度（必须仍然 REPORT）
      TP1 三项并列删中项，同块内另一句还留着小写形式  ← 卡「块内归零才报」这种过度收紧
      TP2 整句删术语        TP3 公式换属性名
      TP4 **整块删除、中文没跟**（块数不等 → 退回整叶口径，仍然报）
      TP5 `a Skill Check` → `a check`（ember 侧真缺陷原形）
      TP6 **别的块里有同名词的正当用法**（测锚点没把候选块选歪）
      TP7 **删除点正好落在块边界**（测 delete 取前后两块；只取后一块会漏报）
    特异度（必须 SILENT）
      FP1 裸词升级成 enricher   FP2 明文换成裸 @UUID   FP3 整段重写
      FP4 **整块删除、中文已跟进**  ← 带过滤器 SILENT，`--no-block-filter` 仍 REPORT
      FP5 **同义改写、中文命中全在同叶别处**  ← 同上，两个 REPORT/SILENT 对照就是过滤器的功效证明

注入走**副本树**，真库一个字节都不碰；跑完校验 4 个文件的 sha256（两份译文/基准 +
被测脚本本身），**实测 0 改动**。

⚠ 特异度还在**真库上逐条核过**，不是只看回测：第十八轮被压掉的 **38 叶 / 137 个候选块**
全部摊开三方原文人看（`4-临时脚本/2026-08-15-round18/suppressed_ember.txt`，
由 `probe_suppressed.py` 生成），**没有一条是压错的**。

⚠ 相似度门槛别用 `SequenceMatcher.ratio()`
------------------------------------------
`ratio()` = 2×匹配/(旧长+新长)，**纯删除会被它自己判成「重写」** ——
删掉一半句子时 ratio 只有 0.67，低于门槛就整条跳过，而删除恰恰是本闸要抓的。
回测 TP2「整句删术语」第一次就是这么被漏掉的。现用 **匹配数 / 较短一侧长度**：
纯删除 = 1.0，真重写才低。

当前库基线（2026-08-15 第十八轮，bindings 为 ember+crucible+dnd5e 三包合导）
--------------------------------------------------------------------------
* **crucible**（基准 `crucible-0.9.1-legacy`）：英文变过 295 条 → 整段重写跳过 24 →
  逐位对齐 271 叶 / 6950 块 → **告警 0 叶**（逐位对齐上线前是 1 叶）。
  ⚠ 那 1 叶的归因写在这里，免得下一轮当成「闸坏了」：
  `crucible.rules.json :: Character Mechanics.pages.Skills.text`，
  `training` 9→4 而中文「训练」仍 7、`ability` 5→2 而中文「能力」仍 3。
  **是同义改写造成的假阳性，不是缺陷** —— 上游删掉 `Skill Checks` 三节，
  中文其实早就没有那三节，两个计数全来自保留下来的正文。逐位对齐后删除点锚到的
  中文块里没有对应译名，两条都被压掉。**这一叶自动消失，正说明过滤器盖住了这一类；
  它要是还在，就是过滤器没盖住 ②/① 类，回头查 `anchor_old_to_new_blocks`。**
  历史上的真缺陷（`Surgeweaver` ×4 / `Rimecaller` ×2 / `Character Creation.pages.Overview`
  那条 `Skill`）已在第十六轮修掉，所以现在是 0。
* **ember**（基准 `5-其他内容/english-baseline/ember-cn-v1.0.15-shipped-en`）：
  英文变过 933 条 → 整段重写跳过 174 → 逐位对齐 757 叶 / 84262 块（2 叶退回整叶）
  → **告警 39 叶 / 43 条**（`--no-block-filter` 是 77 叶；不带 `--bindings` 会虚报成 98）。
  过滤器压掉 55 条、放行 43 条；叶级 77 → 39，**且是单调的**（压掉 38 叶、没有新增 1 叶）。
  43 条里 33 条局部 `新=0`（强：块内该词一个不剩，中文还留着译名），
  10 条局部 `新≥1`（弱：集中在 9 叶，即上面 ① 类剩下的那批）。
  报告 json 每条命中都带 `local` 字段（候选块号 + 局部 旧/新/中 三元组），
  先看 `local` 再看原文，比看整叶计数快一个量级。
  清单在 `4-临时脚本/2026-08-15-round18/on_ember.md`，属**译文侧**工作。
"""
from __future__ import annotations
import argparse
import collections
import difflib
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CJK = re.compile(r"[一-鿿]")
CJK_HEAD = re.compile(r"^[一-鿿·—－\-·、，。：；！？（）《》“”‘’…\s]+")
# `@UUID[…]` / `@Embed[…]`：方括号里是随机 id。**带 `{标签}` 时留标签、不带标签时
# 用目标文档的英文名代替** —— Foundry 在页面上渲染的就是目标名，那个词玩家看得见。
ID_MARKUP = re.compile(r"@(?:UUID|Embed)\[([^\]]*)\](?:\s*\{([^}]*)\})?", re.I)
# 其余 enricher（`@Condition[flanked]` / `@Spell[life.ray.compose]` / `[[/check …]]`）
# 方括号里是**有意义的词**，要留下来数 —— 见下面「必须把语义 enricher 展开」。
SEM_MARKUP = re.compile(r"@[A-Za-z]+\[([^\]]*)\]|&(?:amp;)?[A-Za-z]+\[([^\]]*)\]|\[\[([^\]]*)\]\]")
HTML_TAG = re.compile(r"<[^>]+>")
WORD = re.compile(r"[A-Za-z][A-Za-z'’]{2,}")
# 逐位对齐用：按 HTML 标签切块。标签本身是机械，结构没变时两侧块数必然相等。
TAG_SPLIT = re.compile(r"<[^>]+>")

DEFAULT_GLOSSARY = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "5-其他内容", "glossary", "glossary_ec.json"))


def load_idmap(paths):
    """`dump_bindings.mjs` 的输出 → {文档 id: 英文名}。给不带 `{标签}` 的 `@UUID` 用。"""
    idmap = {}
    for p in paths or []:
        b = json.load(open(p, encoding="utf-8"))
        for k, v in (b.get("ids") or {}).items():
            for o in (v if isinstance(v, list) else [v]):
                if isinstance(o, dict) and o.get("name"):
                    idmap.setdefault(k, o["name"])
                    break
    return idmap


def strip_machinery(s: str, idmap=None) -> str:
    """留下「玩家读到的词」，丢掉随机 id。

    ⚠ **语义 enricher 的方括号里必须展开来数**（本闸第二版栽在这里）：
    上游把裸词升级成 enricher 是很常见的一类改动 ——
    `a Flanked enemy` → `a @Condition[flanked] enemy`、
    `Composed Ray of Life` → `@Spell[life.ray.compose]`。
    把方括号整段抹掉的话，英文侧那个词就「凭空少了一次」，而中文照旧写着「夹击」
    「中毒」「生命」，于是被判成「中文停在旧版」。实测 crucible 侧 6 条告警里
    **5 条**都是这一个成因，展开后全部消失，只剩那 1 条真缺陷。

    ⚠ **裸 `@UUID` 顶替明文术语**（本闸第三版栽在这里，ember 侧的头号假阳性）：
    上游很爱把 `Incorporeal Movement` 这样的明文改写成**不带 `{标签}` 的**
    `@UUID[Actor.x.Item.y]` —— Foundry 渲染时自己去取目标文档名，玩家照样看到
    那个词；而我方译文必须写出 `{虚体移动}`，否则页面上会冒出英文原名。
    英文散文里那个词「消失了」，中文却理所当然还在。给 `--bindings`
    （`dump_bindings.mjs` 的输出）后，本函数会把裸 `@UUID` 换成目标文档的英文名，
    这一类就不再误报。**没给 `--bindings` 时这一类会大量误报，看到就先补上。**
    """
    def _uuid(m):
        label = m.group(2)
        if label:                       # 有 {标签}：标签就是玩家看到的文字
            return " " + label + " "
        tid = (m.group(1) or "").split("#")[0].strip().split(".")[-1]
        return " " + (idmap or {}).get(tid, "") + " "

    s = ID_MARKUP.sub(_uuid, s)
    s = SEM_MARKUP.sub(lambda m: " " + re.sub(r"[._:\-/#|]+", " ",
                                              next(g for g in m.groups() if g is not None)) + " ", s)
    return HTML_TAG.sub(" ", s)


def block_tokens(s, idmap=None):
    """按 HTML 标签切块 → `(词表, 每词所属块号, 块串列表)`。

    切法照抄 `4-临时脚本/2026-08-15-round16/probes/split_dives.py`：先 `TAG.split`，
    再逐块归一到「玩家读到的词」。块比整叶细一个量级，而标签是机械、两侧逐字节相同，
    所以结构没变时两侧块数必然相等；**不等的会被报出来（`shape_mismatch`）而不是静默跳过**。
    """
    parts = TAG_SPLIT.split(s)
    words, owner = [], []
    for bi, part in enumerate(parts):
        for w in WORD.findall(strip_machinery(part, idmap)):
            words.append(w)
            owner.append(bi)
    return words, owner, parts


def anchor_old_to_new_blocks(opcodes, old_len, new_owner):
    """旧英文的每个词 → 它在**新英文**里对应的块号集合。

    · `equal`   逐位对应，单块，最准；
    · `replace` 旧段整体对到新段覆盖的那些块；
    · `delete`  新侧是一个**点**，删除点正好落在块边界时前后两词分属两块，
                所以取 `{前一词的块, 后一词的块}` 两个候选 —— 少给会漏报真缺陷。
    """
    def blk(j):
        if not new_owner:
            return 0
        return new_owner[min(max(j, 0), len(new_owner) - 1)]

    anchor = [frozenset()] * old_len
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(i2 - i1):
                anchor[i1 + k] = frozenset({new_owner[j1 + k]})
        elif tag == "replace":
            s = frozenset(new_owner[j1:j2]) or frozenset({blk(j1)})
            for k in range(i1, i2):
                anchor[k] = s
        elif tag == "delete":
            s = frozenset({blk(j1 - 1), blk(j1)})
            for k in range(i1, i2):
                anchor[k] = s
    return anchor


def stem(w: str) -> str:
    """极简单复数归一：只处理 -ies/-es/-s，够用且不会把 `bonus` 砍成 `bonu`。

    ⚠ **必须先剥所有格**。`WORD` 正则把撇号收进词里，所以 `manor's` 会被 -s 规则
    砍成 `manor'`，与词表键 `manor` **不同桶**。第十八轮实测这一个 bug 同时造成两种错：
      · 假阳性 27 块 —— 所有格全部进不了同一个计数桶，看着像「英文把这个词删了」；
      · **漏报** —— 注入用例「上游删掉 `The Warden's`、中文仍写『守林者』」两边都不报，
        而只把 `Warden's` 换成 `Warden`、其余一字不改，立刻就报得出来。
    """
    w = w.lower()
    if w.endswith(("'s", "’s")):        # ← 先剥所有格，再做复数归一
        w = w[:-2]
    elif w.endswith(("'", "’")):        # `Sages'` 这种复数所有格
        w = w[:-1]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def cn_head(val: str) -> str:
    """`'技能 Skills'` → `'技能'`；不是以汉字开头的返回空串。"""
    m = CJK_HEAD.match(val or "")
    if not m:
        return ""
    return re.sub(r"\s+", "", m.group(0))


def load_glossary(path):
    """{英文词干: 汉字译名}，只收**单词**条目（多词短语不参与逐词比对）。

    按词干建键，`Skill` 与 `Skills` 才能合成同一个计数桶。同词干撞上多条时取
    最短的原词那条（`Skill` 优先于 `Skills`），因为词表里单数形式的译名更干净。
    """
    raw = json.load(open(path, encoding="utf-8"))
    out, picked = {}, {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        if not re.fullmatch(r"[A-Za-z][A-Za-z'’]{2,}", k):
            continue
        head = cn_head(v)
        if len(head) < 2:
            continue
        st = stem(k)
        if st not in out or len(k) < len(picked[st]):
            out[st], picked[st] = head, k
    return out


# --------------------------------------------------------------------- 载入
def load_json(path):
    raw = open(path, encoding="utf-8-sig").read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))


def leaves(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out[".".join(path)] = node


def baseline_packs(bdir):
    out = {}
    for f in sorted(os.listdir(bdir)):
        if not f.endswith(".json") or f == "_source.json":
            continue
        key = ("ember.crucible-adventure.json" if f == "_repaired.json"
               else f.replace("-en.json", ".json"))
        out.setdefault(key, os.path.join(bdir, f))
    return out


# ----------------------------------------------------------- 基准覆盖了几个包
def cjk_leaf_count(path):
    """一个 cn 包里**含中文**的叶数 —— 本闸的定义域就是这些叶。"""
    if not os.path.exists(path):
        return 0
    d = {}
    leaves(load_json(path).get("entries", {}), [], d)
    return sum(1 for v in d.values() if CJK.search(v))


def coverage(repo, baseline):
    """扫了几个包 / 仓里共几个包 / 基准缺几个包（连带缺掉多少条中文叶）。

    ⚠ **反空转第 (d) 型的探针**。前三种空转形态（判据写坏 / 压根不读库 /
    读旧报告快照）的防法对它全部无效：判据没问题、库也读了、报告是当场生成的 ——
    只是基准目录里**压根没有那个包**，`scan()` 里 `if not os.path.exists(cur_p):
    continue` 会**静默**跳过，报告上一片绿，实情是没扫。

    第十九轮实测：ember 侧基准 `ember-cn-v1.0.15-shipped-en/` 只装了 3 个包
    （`_repaired.json` + adversary + character），而仓里有 10 个 en 包 / 9 个有中文，
    另外 6 个包 **15 147 条有中文的叶**从未进过这三条闸 —— 三条闸却都报 0 告警。
    所以这三个数必须每次跟着结果一起印，不能只印告警数。

    补法在 `capture_baseline.py`（第十九轮）：**上游升级之前**从当前 `compendium/en`
    截一份全包快照，命名里写死上游版本号。已截：
    `english-baseline/ember-0.6.0-preupgrade-2026-08-15/`（10 包）与
    `english-baseline/crucible-0.10.1-preupgrade-2026-08-15/`（15 包）。
    """
    en_dir = os.path.join(repo, "compendium", "en")
    cn_dir = os.path.join(repo, "compendium", "cn")
    bmap = baseline_packs(baseline)
    repo_packs = sorted(f for f in os.listdir(en_dir)
                        if f.endswith(".json") and f != "_source.json")
    scanned, missing = [], []
    for f in repo_packs:
        p = bmap.get(f)
        if p and os.path.exists(p):
            scanned.append(f)
        else:
            cnp = os.path.join(cn_dir, f)
            missing.append({"pack": f, "cn_present": os.path.exists(cnp),
                            "cn_cjk_leaves": cjk_leaf_count(cnp)})
    return {
        "baseline": os.path.abspath(baseline),
        "repo_en_packs": len(repo_packs),
        "scanned_packs": scanned,
        "missing_packs": missing,
        "cn_cjk_leaves_uncovered": sum(m["cn_cjk_leaves"] for m in missing),
        # 基准里有、仓里已经没有的包（上游删包 / 改名）。不影响本次结果，但要看得见。
        "baseline_only_packs": [f for f in sorted(bmap) if f not in repo_packs],
    }


def print_coverage(cov):
    print(f"  包覆盖：本次扫 {len(cov['scanned_packs'])} 个包 / 仓里 en 共 "
          f"{cov['repo_en_packs']} 个包 / 基准缺 {len(cov['missing_packs'])} 个")
    if cov["missing_packs"]:
        print("  ⚠ 下列包**基准里没有，本闸一条也没看** —— 它们的 0 告警不是「干净」，是「没扫」：")
        for m in cov["missing_packs"]:
            tail = "" if m["cn_present"] else "（无 cn 文件）"
            print(f"       {m['pack']}  含中文的叶 {m['cn_cjk_leaves']}{tail}")
        print(f"     合计未进闸的中文叶 {cov['cn_cjk_leaves_uncovered']}")
        print("     历史快照补不回来（那些包发版当时就没捕获）；将来靠 "
              "3-常用脚本/qa/capture_baseline.py 在**升级前**截全包基准。")
    if cov["baseline_only_packs"]:
        print(f"  ⚠ 基准里有、仓里 en 已没有的包 {len(cov['baseline_only_packs'])} 个："
              f"{'、'.join(cov['baseline_only_packs'])}")


def norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------- 主流程
def scan(repo, baseline, gloss, capitalized_only=True, min_sim=0.75, idmap=None,
         block_filter=True, per_pack=None):
    en_dir = os.path.join(repo, "compendium", "en")
    cn_dir = os.path.join(repo, "compendium", "cn")
    findings = []
    stats = collections.Counter()

    for pack, oldpath in baseline_packs(baseline).items():
        cur_p = os.path.join(en_dir, pack)
        if not os.path.exists(cur_p):
            continue
        o, n, c = {}, {}, {}
        leaves(load_json(oldpath).get("entries", {}), [], o)
        leaves(load_json(cur_p).get("entries", {}), [], n)
        cnp = os.path.join(cn_dir, pack)
        if os.path.exists(cnp):
            leaves(load_json(cnp).get("entries", {}), [], c)
        if per_pack is not None:
            per_pack[pack] = {"baseline_leaves": len(o), "cur_en_leaves": len(n),
                              "cn_leaves": len(c), "changed_pairs": 0, "findings": 0}

        for path, new_en in n.items():
            old_en = o.get(path)
            cn = c.get(path)
            if old_en is None or not cn or not CJK.search(cn):
                continue
            if norm_ws(old_en) == norm_ws(new_en):
                continue
            stats["changed_pairs"] += 1
            if per_pack is not None:
                per_pack[pack]["changed_pairs"] += 1
            old_words = WORD.findall(strip_machinery(old_en, idmap))
            new_words = WORD.findall(strip_machinery(new_en, idmap))
            sm = difflib.SequenceMatcher(
                None, [w.lower() for w in old_words], [w.lower() for w in new_words],
                autojunk=False)
            # ⚠ 别用 `sm.ratio()`：它是 2*匹配/(旧长+新长)，**纯删除**会被它自己判成
            # 「重写」——删掉一半句子时 ratio 只有 0.67，而删除恰恰是本闸要抓的东西
            # （回测 TP2「整句删术语」就是这么被漏掉的）。改用
            # 匹配数 / 较短一侧的长度：纯删除 = 1.0，真重写才低。
            matched = sum(b.size for b in sm.get_matching_blocks())
            sim = matched / max(1, min(len(old_words), len(new_words)))
            if sim < min_sim:
                stats["rewritten_skipped"] += 1
                continue
            # ---- 逐位对齐：把新英文与中文各按 HTML 标签切块，块号一一对应 ----
            # 块结构对不上（块数不等 / 切块后词序列与整叶归一不一致）时**不静默跳过**，
            # 而是记 shape_mismatch 并退回整叶口径，让下一轮看得见有多少叶没盖住。
            ow_b, _o_owner, _o_parts = block_tokens(old_en, idmap)
            nw_b, n_owner, n_parts = block_tokens(new_en, idmap)
            cn_parts = TAG_SPLIT.split(cn)
            aligned = (block_filter and ow_b == old_words and nw_b == new_words
                       and len(cn_parts) == len(n_parts))
            if block_filter:
                if aligned:
                    stats["aligned_leaves"] += 1
                    stats["aligned_blocks"] += len(n_parts)
                elif len(cn_parts) != len(n_parts):
                    stats["shape_mismatch_blockcount"] += 1
                else:
                    stats["shape_mismatch_tokenize"] += 1
            anchor = (anchor_old_to_new_blocks(sm.get_opcodes(), len(old_words), n_owner)
                      if aligned else None)

            # 只看**真正被删掉的那几段**里的词。整叶数词会把「中文本来就比英文多提
            # 几次这个名词」当信号，噪声压过信号（实测 245 → 个位数）。
            deleted, del_spans = set(), []
            del_idx = collections.defaultdict(set)   # 词干 → 被删/被替换掉的旧词下标
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag in ("delete", "replace"):
                    deleted.update(stem(w) for w in old_words[i1:i2])
                    for k in range(i1, i2):
                        del_idx[stem(old_words[k])].add(k)
                    del_spans.append((
                        i1, i2,
                        " ".join(old_words[max(0, i1 - 6):i1]) + "  《删掉》 " +
                        " ".join(old_words[i1:i2]) + " 《/》  " +
                        " ".join(old_words[i2:i2 + 6]),
                        " ".join(new_words[max(0, j1 - 6):j2 + 6])))
            old_n = collections.Counter(stem(w) for w in old_words)
            new_n = collections.Counter(stem(w) for w in new_words)
            # 旧英文里至少出现过一次首字母大写形式的词干 = Foundry 的规则术语惯例
            cap = {stem(w) for w in old_words if w[0].isupper()}
            surface = {}
            for w in old_words:
                surface.setdefault(stem(w), w)
            hits = []
            for st in deleted:
                n_old, n_new = old_n.get(st, 0), new_n.get(st, 0)
                if n_new >= n_old:
                    continue                       # 没掉次数，不是删除型改动
                if capitalized_only and st not in cap:
                    continue
                term = gloss.get(st)
                if not term:
                    continue
                cn_c = cn.count(term)
                # 中文的次数要「和旧英文对得上、和新英文对不上」才算停在旧版：
                #   > n_new 说明比新英文多；<= n_old 说明没多到中文自己的行文习惯上去
                #  （中文爱重复名词，英文用代词，cn_c > n_old 基本都是这个原因）
                if not (n_new < cn_c <= n_old):
                    continue
                # ---- 逐位对齐过滤：把同一条判据**缩到删除点所在的那几块**上重判一次 ----
                # 整叶口径下「英文这里删了一个词」和「中文那边另一处正当地用了同一个译名」
                # 分不开（实测 70/82 假阳性都是这一类）。块级重判后中文那些别处的命中
                # 落在别的块里，不再参与，只剩「删除点对应的中文块里确实还留着译名」。
                local = None
                if aligned:
                    cand = set()
                    for k in del_idx.get(st, ()):
                        cand |= anchor[k]
                    # **逐块单独判**，不合并成一个大候选集：合并等于在小一号的尺度上
                    # 重新引入整叶那种稀释（实测合并 40 叶 / 逐块 39 叶，逐块更细也更少）。
                    residual = []
                    for b in sorted(cand):
                        g, gs = [b], {b}
                        o_l = sum(1 for k, w in enumerate(old_words)
                                  if stem(w) == st and (anchor[k] & gs))
                        n_l = sum(1 for j, w in enumerate(new_words)
                                  if stem(w) == st and n_owner[j] in gs)
                        c_l = sum(cn_parts[b].count(term) for b in g)
                        # ⚠ **块级只留下界。** 上界 `c_l <= o_l` 在块这个尺度上远比在叶尺度上
                        # 苛刻：块级 o_l 通常就是 1，而「英文用代词、中文还原名词」是本脚本
                        # 自己注释里写明的常态，一还原就 c_l=2 > o_l=1，整条被当假阳性压掉。
                        # 第十八轮实测：拿历史真缺陷 Surgeweaver 的原形注入
                        # （`half your Intellect score`→`half the ability score`，中文块内「智力」两次），
                        # **不带过滤器报、带过滤器不报**。上界留在叶级即可。
                        if n_l < c_l:
                            residual.append({"blocks": g, "en_old_n": o_l,
                                             "en_new_n": n_l, "cn_count": c_l})
                    if not residual:
                        stats["suppressed_by_block"] += 1
                        continue
                    stats["kept_by_block"] += 1
                    local = residual[0] if len(residual) == 1 else {
                        "blocks": sorted(cand),
                        "en_old_n": sum(r["en_old_n"] for r in residual),
                        "en_new_n": sum(r["en_new_n"] for r in residual),
                        "cn_count": sum(r["cn_count"] for r in residual),
                        "sites": residual}
                ctx = [d[2] for d in del_spans
                       if any(stem(w) == st for w in old_words[d[0]:d[1]])][:2]
                cn_ctx = [m for m in re.split(r"(?<=[。；！？])", re.sub(r"<[^>]+>", "", cn))
                          if term in m][:2]
                hits.append({"en": surface[st], "en_old_n": n_old, "en_new_n": n_new,
                             "cn_term": term, "cn_count": cn_c, "similarity": round(sim, 3),
                             "local": local,
                             "deleted_context": ctx, "cn_context": cn_ctx})
            hits.sort(key=lambda h: (h["en_new_n"] - h["cn_count"], -h["en_old_n"]))
            if not hits:
                stats["clean"] += 1
                continue
            stats["DROPPED_TERM_KEPT"] += 1
            if per_pack is not None:
                per_pack[pack]["findings"] += 1
            findings.append({
                "verdict": "DROPPED_TERM_KEPT", "repo": os.path.basename(repo),
                "pack": pack, "path": path, "dropped": hits,
                "old_en": old_en[:400], "new_en": new_en[:400], "cn": cn[:400],
            })
    return findings, stats


def main():
    ap = argparse.ArgumentParser(description="删除型漂移闸：英文删了词、中文还留着译名")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--glossary", default=DEFAULT_GLOSSARY)
    ap.add_argument("--bindings", action="append", default=[],
                    help="**必填**。dump_bindings.mjs 的输出，可重复。用来把不带 {标签} 的 @UUID "
                         "换成目标文档英文名；不给的话「上游把明文改成裸 @UUID」会大量误报"
                         "（实测 ember 侧 81 → 98）")
    ap.add_argument("--no-bindings", action="store_true",
                    help="显式承认「我就是要跑没有 idmap 的对照」。数字不可用于判缺陷。")
    ap.add_argument("--min-sim", type=float, default=0.75,
                    help="旧/新英文的词级相似度低于此值＝整段重写，逐词数次数没有意义，跳过")
    ap.add_argument("--all-case", action="store_true",
                    help="连小写词也查（噪声涨一个量级，专项复核才开）")
    ap.add_argument("--no-block-filter", action="store_true",
                    help="关掉逐位对齐过滤，退回整叶计数口径（实测假阳性会从 12 涨回 77，"
                         "只在与旧轮对数时用）")
    ap.add_argument("--strict-coverage", action="store_true",
                    help="基准缺包时以退出码 3 结束（默认只告警：回测脚本会拿单包小仓跑，"
                         "默认非零会把它们全打红）")
    ap.add_argument("--out")
    ap.add_argument("--md")
    ap.add_argument("--show", type=int, default=30)
    a = ap.parse_args()

    # --bindings 必填：不给时「上游把明文术语改成裸 @UUID」会大量误报，
    # 实测 ember 侧 81 → 98（多报 20 少报 3），数字既不是上界也不是下界。
    if not a.bindings and not a.no_bindings:
        print("✗ 必须给 --bindings：不给的话「上游把明文术语改成不带 {标签} 的 @UUID」"
              "会大量误报（实测 ember 侧 81 → 98，多报 20 叶）。\n"
              "  先跑同目录的 dump_bindings.mjs，再把输出传进来（可重复给多份）。\n"
              "  确实要看没有 idmap 的对照，显式加 --no-bindings。")
        return 2
    if not a.bindings:
        print("⚠ --no-bindings：本次没有 idmap，「明文→裸 @UUID」这一类会误报，"
              "结果只能当参考，不能据此判缺陷。")

    baseline_abs = os.path.abspath(a.baseline)
    if os.path.basename(baseline_abs.rstrip("\\/")) == "ember-0.6.0":
        print("⚠ 基准是 english-baseline/ember-0.6.0/：它与当前 compendium/en 逐叶几乎相同"
              "（34k 叶里只有 10 叶英文变过），本闸拿它 diff 会恒为 0 告警。\n"
              "  ember 侧本闸的基准应当是 english-baseline/ember-cn-v1.0.15-shipped-en/。")

    gloss = load_glossary(a.glossary)
    idmap = load_idmap(a.bindings)
    cov = coverage(a.repo, a.baseline)
    per_pack = {}
    findings, stats = scan(a.repo, a.baseline, gloss, not a.all_case, a.min_sim, idmap,
                           block_filter=not a.no_block_filter, per_pack=per_pack)
    print(f"{os.path.basename(a.repo)}  基准 {os.path.basename(a.baseline)}  "
          f"词表单词条目 {len(gloss)}  id→名 {len(idmap)}")
    print(f"  baseline = {baseline_abs}")
    # 反空转 (d)：基准缺包时 scan() 会静默跳过，报告与「扫过、干净」长得一模一样。
    print_coverage(cov)
    for pk, d in per_pack.items():
        print(f"     · {pk}  基准叶 {d['baseline_leaves']}  当前英文叶 {d['cur_en_leaves']}"
              f"  中文叶 {d['cn_leaves']}  英文变过 {d['changed_pairs']}  告警 {d['findings']}")
    print(f"  英文变过且有中文的条目 {stats['changed_pairs']}"
          f"  ·  整段重写跳过 {stats['rewritten_skipped']}  ·  无删词残留 {stats['clean']}")
    # 反空转：逐位对齐这一层必须自报「扫了多少叶、多少块、压掉多少条、放行多少条」，
    # 三个数任何一个是 0 都说明这一层根本没跑起来（本项目实测过三种空转形态）。
    if a.no_block_filter:
        print("  ⚠ --no-block-filter：整叶计数口径，假阳性主类（同义改写 / 整块删除）不会被压掉")
    else:
        print(f"  逐位对齐：对齐 {stats['aligned_leaves']} 叶 / {stats['aligned_blocks']} 块"
              f"  ·  块数不等退回整叶 {stats['shape_mismatch_blockcount']} 叶"
              f"  ·  切块后词序列不一致退回整叶 {stats['shape_mismatch_tokenize']} 叶")
        print(f"    块级重判：压掉 {stats['suppressed_by_block']} 条  ·  放行 {stats['kept_by_block']} 条")
    print(f"  **DROPPED_TERM_KEPT {stats['DROPPED_TERM_KEPT']}**")
    for f in findings[:a.show]:
        print(f"  {f['pack']} :: {f['path'][-70:]}")
        for h in f["dropped"]:
            print(f"     英文 {h['en']!r} {h['en_old_n']}→{h['en_new_n']} 次；"
                  f"中文 {h['cn_term']!r} 仍有 {h['cn_count']} 次")
    # 报告里**必须**记下所用 baseline 的绝对路径：english-baseline/ 下有四份目录，
    # 挑错一份（尤其 ember-0.6.0）结果会静悄悄地变成 0 告警，而 LOCAL-PATCHES.md
    # 与 _source.json 的两处说明彼此打架，事后光看数字复原不出当时用的是哪份。
    meta = {
        "tool": os.path.basename(__file__),
        "repo": os.path.abspath(a.repo),
        "baseline": baseline_abs,
        "baseline_packs": {k: os.path.abspath(v)
                           for k, v in baseline_packs(a.baseline).items()},
        # 缺包时报告要能自证「0 告警是没扫出来、还是根本没扫」。
        "coverage": cov,
        "per_pack": per_pack,
        "glossary": os.path.abspath(a.glossary),
        "bindings": [os.path.abspath(p) for p in a.bindings],
        "idmap_size": len(idmap),
        "min_sim": a.min_sim,
        "capitalized_only": not a.all_case,
        "block_filter": not a.no_block_filter,
        "argv": sys.argv,
    }
    if a.out:
        json.dump({"meta": meta, "stats": dict(stats), "findings": findings},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  -> {a.out}")
    if a.md:
        with open(a.md, "w", encoding="utf-8") as fh:
            fh.write(f"# 删除型漂移（scan_dropped_terms.py）— {os.path.basename(a.repo)}\n\n")
            fh.write(f"- baseline：`{meta['baseline']}`\n")
            fh.write(f"- 包覆盖：扫 {len(cov['scanned_packs'])} / 仓里 {cov['repo_en_packs']} / "
                     f"缺 {len(cov['missing_packs'])}"
                     f"（未进闸的中文叶 {cov['cn_cjk_leaves_uncovered']}）\n")
            fh.write(f"- bindings：{'、'.join(f'`{p}`' for p in meta['bindings']) or '**无（结果不可用）**'}\n")
            fh.write(f"- glossary：`{meta['glossary']}`\n")
            fh.write(f"- 英文变过且有中文 {stats['changed_pairs']} 条，**告警 {len(findings)}**\n\n")
            for f in findings:
                fh.write(f"## `{f['pack']}` `{f['path']}`\n\n")
                for h in f["dropped"]:
                    fh.write(f"- 英文删了 `{h['en']}` → 中文仍有 `{h['cn_term']}` ×{h['cn_count']}")
                    # 先看局部三元组再看下面的原文：整叶计数带着同叶别处的正当用法，
                    # 局部才是「删除点所在的那一块」的实况。`新=0` 是强证据。
                    L = h.get("local")
                    if L:
                        fh.write(f"（**局部** 块 {L['blocks']}：旧 {L['en_old_n']} → "
                                 f"新 {L['en_new_n']}，中文 {L['cn_count']}"
                                 f"{'，**强**' if L['en_new_n'] == 0 else ''}）")
                    else:
                        # 两种可能：跑了 --no-block-filter，或该叶块结构对不上退回整叶。
                        # 哪一种看报告头部的 stats / meta.block_filter，别在这里猜。
                        fh.write("（无局部口径，本条按**整叶**判）")
                    fh.write("\n")
                fh.write(f"\n- 旧英文：{f['old_en']}\n- 新英文：{f['new_en']}\n- 中文：{f['cn']}\n\n")
        print(f"  -> {a.md}")
    if a.strict_coverage and cov["missing_packs"]:
        print(f"✗ --strict-coverage：基准缺 {len(cov['missing_packs'])} 个包，"
              f"本次结果不构成全库结论。")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
