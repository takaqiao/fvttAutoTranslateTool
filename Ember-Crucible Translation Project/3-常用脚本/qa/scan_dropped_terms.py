# -*- coding: utf-8 -*-
"""删除型漂移闸：上游**删掉**一个术语，中文还留着它的译名。

    python scan_dropped_terms.py --repo <repo> --baseline <旧英文基准目录>
                                 --bindings <dump_bindings.mjs 的输出>
                                 [--glossary <glossary_ec.json>] [--out <json>] [--md <md>]
                                 [--all-case] [--min-sim 0.75] [--show 30]

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
`5-其他内容/english-baseline/` 下有四份，用途完全不同，本闸要的是**发版当时的旧英文**：

    ember-cn-v1.0.15-shipped-en/_repaired.json   ← ember 侧本闸的基准（就是它）
    crucible-0.9.1-legacy/                       ← crucible 侧本闸的基准
    crucible-0.10.1/                             ← 当前上游英文，不是基准
    ember-0.6.0/                                 ← **当前上游英文的镜像，拿来 diff 等于没 diff**

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

已知假阳性
----------
* **译名是常用词**：`Ability`→「能力」、`Level`→「等级」。上游把这一段里的
  `ability` 删了，中文却因为在讲别的事仍然写着「能力」。这一类靠人核，报告里
  一并给出旧/新/中三段原文就是为了两秒看完。
* **同义改写**：上游把 `Skill` 换成 `Proficiency`，中文照旧写「技能」其实是对的
  （两词在本项目的译名恰好相同）。判据看不出「换词」和「删词」的区别。
* **词表里一词多义**：`glossary_ec` 只存一条主译名，撞上多义词时取的可能不是
  这一处该用的那条。
* 中文的**中英对照尾巴**（`天赋 Talent`）里本来就抄着英文，不影响本闸（只查汉字）。

回测（2026-08-15，`4-临时脚本/2026-08-15-round16/qa/backtest_dropped_terms.py`）
--------------------------------------------------------------------------
双向 6/6 PASS。灵敏度 3/3：三项并列删中间一项（＝上一轮那条真缺陷，且同叶别处
还留着同一个词的小写形式）、整句删一个术语、公式里换掉一个属性名。
特异度 3/3 静默：裸词升级成 `@Condition[…]`、明文换成不带标签的 `@UUID`、整段重写。

⚠ 相似度门槛别用 `SequenceMatcher.ratio()`
------------------------------------------
`ratio()` = 2×匹配/(旧长+新长)，**纯删除会被它自己判成「重写」** ——
删掉一半句子时 ratio 只有 0.67，低于门槛就整条跳过，而删除恰恰是本闸要抓的。
回测 TP2「整句删术语」第一次就是这么被漏掉的。现用 **匹配数 / 较短一侧长度**：
纯删除 = 1.0，真重写才低。

当前库基线（2026-08-15）
------------------------
* crucible（基准 crucible-0.9.1-legacy）：英文变过 295 条 → 整段重写跳过 24 →
  **告警 8 叶**。逐叶人核：**7 真 1 假**。
  真：`Surgeweaver` ×4 副本（`half your Intellect score` →
  `half the ability score your Rune of Storm scales with`，中文仍写「你智力数值一半」）、
  `Rimecaller` ×2 副本（`Wisdom` 同理，中文仍写「你感知属性值一半」）、
  `Character Creation.pages.Overview`（上一轮那条 `Skill`）。
  假：`Character Mechanics.pages.Skills`（上游删掉 `Skill Checks` 三节，
  中文其实早就没有那三节，`训练`/`能力` 的计数来自保留下来的正文）。
* ember（基准 `5-其他内容/english-baseline/ember-cn-v1.0.15-shipped-en`，
  bindings 用 `4-临时脚本/2026-08-16-round16/seal-qa/bind_ember.json`）：
  英文变过 933 条 → 整段重写跳过 174 → **告警 81 叶**
  （2026-08-15 首跑记的是 82，其后译文侧改过一叶；不带 --bindings 会虚报成 98）。
  抽核确认真缺陷形态存在（`Players' Guide.pages.Area Maps`：
  `a Skill Check` → `a check`，中文仍写「一次技能检定」），也确认了常用词撞名的
  假阳性形态（`Bickering Priests` 删掉、中文的「牧师」来自另一句「若牧师或圣武士」）。
  完整清单在 `4-临时脚本/2026-08-15-round16/qa/drop_1-Ember汉化插件.md`，属**译文侧**工作。
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


def stem(w: str) -> str:
    """极简单复数归一：只处理 -ies/-es/-s，够用且不会把 `bonus` 砍成 `bonu`。"""
    w = w.lower()
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


def norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------- 主流程
def scan(repo, baseline, gloss, capitalized_only=True, min_sim=0.75, idmap=None):
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

        for path, new_en in n.items():
            old_en = o.get(path)
            cn = c.get(path)
            if old_en is None or not cn or not CJK.search(cn):
                continue
            if norm_ws(old_en) == norm_ws(new_en):
                continue
            stats["changed_pairs"] += 1
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
            # 只看**真正被删掉的那几段**里的词。整叶数词会把「中文本来就比英文多提
            # 几次这个名词」当信号，噪声压过信号（实测 245 → 个位数）。
            deleted, del_spans = set(), []
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag in ("delete", "replace"):
                    deleted.update(stem(w) for w in old_words[i1:i2])
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
                ctx = [d[2] for d in del_spans
                       if any(stem(w) == st for w in old_words[d[0]:d[1]])][:2]
                cn_ctx = [m for m in re.split(r"(?<=[。；！？])", re.sub(r"<[^>]+>", "", cn))
                          if term in m][:2]
                hits.append({"en": surface[st], "en_old_n": n_old, "en_new_n": n_new,
                             "cn_term": term, "cn_count": cn_c, "similarity": round(sim, 3),
                             "deleted_context": ctx, "cn_context": cn_ctx})
            hits.sort(key=lambda h: (h["en_new_n"] - h["cn_count"], -h["en_old_n"]))
            if not hits:
                stats["clean"] += 1
                continue
            stats["DROPPED_TERM_KEPT"] += 1
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
    findings, stats = scan(a.repo, a.baseline, gloss, not a.all_case, a.min_sim, idmap)
    print(f"{os.path.basename(a.repo)}  基准 {os.path.basename(a.baseline)}  "
          f"词表单词条目 {len(gloss)}  id→名 {len(idmap)}")
    print(f"  baseline = {baseline_abs}")
    print(f"  英文变过且有中文的条目 {stats['changed_pairs']}"
          f"  ·  整段重写跳过 {stats['rewritten_skipped']}  ·  无删词残留 {stats['clean']}")
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
        "glossary": os.path.abspath(a.glossary),
        "bindings": [os.path.abspath(p) for p in a.bindings],
        "idmap_size": len(idmap),
        "min_sim": a.min_sim,
        "capitalized_only": not a.all_case,
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
            fh.write(f"- bindings：{'、'.join(f'`{p}`' for p in meta['bindings']) or '**无（结果不可用）**'}\n")
            fh.write(f"- glossary：`{meta['glossary']}`\n")
            fh.write(f"- 英文变过且有中文 {stats['changed_pairs']} 条，**告警 {len(findings)}**\n\n")
            for f in findings:
                fh.write(f"## `{f['pack']}` `{f['path']}`\n\n")
                for h in f["dropped"]:
                    fh.write(f"- 英文删了 `{h['en']}` → 中文仍有 `{h['cn_term']}` ×{h['cn_count']}\n")
                fh.write(f"\n- 旧英文：{f['old_en']}\n- 新英文：{f['new_en']}\n- 中文：{f['cn']}\n\n")
        print(f"  -> {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
