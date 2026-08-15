# -*- coding: utf-8 -*-
"""数值**关系**被译反 —— 方向词、倍半、时间/距离单位。

为什么既有判据抓不到
--------------------
`scan_content_coverage` 用**数字多重集**比对英中两侧。「至少 3 点」译成「至多 3 点」，
数字多重集完全相等 —— 闸门一声不响，而规则整个反了。
`half`/`double`、`round`/`turn` 同理：字符层面挑不出毛病，语义整个反。
本判据补的是**数字与数字之间、数字与单位之间的关系**这一层。

四条轴
------
* `dir`      下界(at least / or more / greater than) ↔ 上界(at most / up to / no more than)
* `mag`      half / halve ↔ double / twice / triple
* `unit`     数字锚定的度量单位：3 rounds→3 轮、30 feet→30 尺、1 hour→1 小时 …
* `unit_rt`  **Round=轮 / Turn=回合** 的整叶计数失衡（不依赖数字，专抓 "each round"/"your turn"）

两种取样口径（都只报**硬冲突**，不报「中文没出现期望词」）
-----------------------------------------------------------
`leaf`   整叶级：英文侧**只有**方向 A，中文侧**只有**方向 B。
`anchor` 数字锚定：本库数字两侧一一对应（覆盖率判据全绿），
         取「英文里只出现一次、中文里也只出现一次」的数字当锚，只比锚点小窗内的关系词。
         长叶子里的单点译反靠这条抓。

判 `unit_rt` 的依据（不是拍脑袋）
--------------------------------
`crucible.rules.json > Combat > Initiative and Turn Order` 是本作对这两个词的**定义页**，
中文译作「轮与回合……**轮**指的是遭遇中角色们作出的一整组选择……**回合**则表示
在先攻顺序中轮到行动的那个角色所采取的动作」。全库计数也压倒性支持：
  EN 只提 round 的叶子：中文用「轮」**352** : 用「回合」**8**
  EN 只提 turn  的叶子：中文用「回合」**964** : 用「轮」**3**
所以那 8+3 是离群值，不是另一种译法。

已知的三个假阳性坑（都已在本脚本里堵掉，改判据时别踩回去）
------------------------------------------------------------
1. **英文否定式**：`never exceeds Shoddy` 是上界不是下界；`no larger than` 是上界。
   `no more than` 里含 `more than`、`no less than` 里含 `less than`，必须先消否定再扫肯定。
2. **中文否定式**：「不**能**超过三个」「不**得**大于一个 5 尺立方」是上界。
   只写 `(?<!不)超过` 挡不住「不能/不得/无法/不会 + 超过」。
3. **比较项调换**：EN `your result exceeds the Removal DC`（下界词）
   ↔ CN「移除DC**低于**你检定结果」（上界词）—— 两边把主宾对调了，**语义完全一致**。
   所以 `dir` 轴只在**同一个数字**两侧都是被比较项时才敢报（anchor 模式），
   leaf 模式必须配合人工逐条看。
4. 「以下」「以上」在白话里是「下文/上文」（"获得**以下**效果"），必须钉死在数字/量词后面。

铁律：只产出脚本与报告，绝不写 compendium/ 或 lang/。

用法：
  python scan_relation_direction.py --repo <仓库目录> [--repo <另一个>] --out <json>
  python scan_relation_direction.py --repo ... --axis unit_rt
  python scan_relation_direction.py --self-test          # 灵敏度自测（内存里注错，不碰磁盘）
  python scan_relation_direction.py --repo ... --inject inj.json   # 真库副本上注错回测
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKIP_KEYS = {"_id", "path", "_variants", "_when"}

# ---------------------------------------------------------------- 文本归一

TAG = re.compile(r"<[^>]+>")
# @UUID[...]{标签} 方括号内照抄不译，两侧字节相同；留着只会污染数字唯一性判定。
ENRICHER = re.compile(r"@\w+\[[^\]]*\](\{([^}]*)\})?")
INLINE_ROLL = re.compile(r"\[\[[^\]]*\]\](\{([^}]*)\})?")
ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")


def norm(s: str) -> str:
    s = ENRICHER.sub(lambda m: m.group(2) or " ", s)
    s = INLINE_ROLL.sub(lambda m: m.group(2) or " ", s)
    s = TAG.sub(" ", s)
    s = ENTITY.sub(" ", s)
    return re.sub(r"[ \t\r\n　]+", " ", s)


# ---------------------------------------------------------------- 轴 A: 方向

# 坑 1：先把英文否定式换成哨兵，再扫肯定式。
# 否定词表必须够宽 —— `cannot be less than 16` 漏掉 cannot 就会把上界读成下界。
_ENEG = (r"(?:no|not|never|rarely|seldom|hardly|barely|cannot|can't|won't|will\s+not|"
         r"shouldn't|should\s+not|mustn't|must\s+not|may\s+not|might\s+not|"
         r"doesn't|does\s+not|don't|do\s+not|didn't|isn't|is\s+not|aren't|are\s+not|"
         r"wasn't|weren't|couldn't|could\s+not|wouldn't|would\s+not)")
# 否定词到关系词之间可以隔 5 个词 —— `cannot use the same option more than once`
# 隔了 4 个，卡在 3 就会把上界读成下界（Suarrok 的 Uncanny Eye 实测踩过）。
EN_NEG_HI = re.compile(r"(?i)\b" + _ENEG + r"\s+(?:\w+\s+){0,5}?"
                       r"(?:(?:more|greater|larger|higher|bigger)\s+than|exceeds?)\b")
EN_NEG_LO = re.compile(r"(?i)\b" + _ENEG + r"\s+(?:\w+\s+){0,5}?"
                       r"(?:less|fewer|lower|smaller)\s+than\b")

EN_GE_PRE = re.compile(
    r"(?i)\b(?:at\s+least|a\s+minimum\s+of|minimum\s+of|greater\s+than|larger\s+than|"
    r"higher\s+than|bigger\s+than|more\s+than|exceeds?|exceeding|in\s+excess\s+of)\b")
EN_LE_PRE = re.compile(
    r"(?i)\b(?:at\s+most|a\s+maximum\s+of|maximum\s+of|less\s+than|fewer\s+than|"
    r"lower\s+than|smaller\s+than|up\s+to)\b")
# `within` / 「以内」故意**不进**方向词表：它们是射程描述词，全库上千处，
# 且从不参与 at least/at most 这一类误译。放进来 leaf 模式的假阳性从 3 条炸到 52 条。
# 「level 5+」也是下界写法（Hex 的 "5+ (24 hours)"）。加号在数字**后**才算，
# 「+1 Action」的加号在数字前，不算。
EN_GE_POST = re.compile(
    r"(?i)\bor\s+(?:more|higher|greater|above|better)\b|\band\s+(?:higher|above)\b"
    r"|(?<=\d)\+(?=[\s,)\.]|$)")
EN_LE_POST = re.compile(
    r"(?i)\bor\s+(?:less|fewer|lower|below|smaller)\b|\band\s+(?:lower|below)\b")

# 坑 2：中文否定式同样先消再扫。不能/不得/无法/不会/不可/不应/不宜 + 超过|大于|…
# 否定词与关系词之间可以隔一整个状语（「无法**通过这种方式获得**超过 3 级」），
# 所以窗口开到 12 个字，但不许跨标点 —— 跨句就不是同一个否定了。
_CNEG = r"[不无未][^，。；：！？、\n]{0,12}?"
CN_NEG_HI = re.compile(_CNEG + r"(?:超过|多于|高于|大于|超出)")
CN_NEG_LO = re.compile(_CNEG + r"(?:少于|低于|小于|不足)")

CN_GE_PRE = re.compile(r"至少|最少|不少于|不低于|不小于|大于|超过|多于|高于|超出")
CN_LE_PRE = re.compile(r"至多|最多|不超过|不多于|不高于|不大于|小于|少于|低于|不足")
# 坑 4：「以上/以下」只有钉在数字或量词后面才是方向词，否则是「上文/下文」。
NUMISH = r"[0-9０-９一二三四五六七八九十百千万两半]|点|级|阶|环|尺|呎|米|码|里|哩|磅|轮|回合|天|日|小时|分钟|秒|倍|个|次|名|件|张|颗|人|层"
CN_GE_POST = re.compile(r"(?:" + NUMISH + r")\s*(?:及|或)?以上|或更(?:高|多|大)")
CN_LE_POST = re.compile(r"(?:" + NUMISH + r")\s*(?:及|或)?以下|或更(?:低|少|小)")


GE_MARK, LE_MARK = "\x01", "\x02"


def _blank(ch):
    """同长替换：把否定短语整段换成 1 个哨兵 + 等量空格。
    长度不变，`anchor` 模式才能继续拿原始下标切窗口。"""
    return lambda m: ch + " " * (len(m.group(0)) - 1)


def neutralize_en(s: str) -> str:
    return EN_NEG_LO.sub(_blank(GE_MARK), EN_NEG_HI.sub(_blank(LE_MARK), s))


def neutralize_cn(s: str) -> str:
    return CN_NEG_LO.sub(_blank(GE_MARK), CN_NEG_HI.sub(_blank(LE_MARK), s))


def en_dir(win: str):
    """入参必须是 neutralize_en 之后的文本。两个方向都出现 -> None（判不了，不报）。"""
    ge = bool(EN_GE_PRE.search(win) or EN_GE_POST.search(win)) or GE_MARK in win
    le = bool(EN_LE_PRE.search(win) or EN_LE_POST.search(win)) or LE_MARK in win
    return "GE" if ge and not le else "LE" if le and not ge else None


def cn_dir(win: str):
    """入参必须是 neutralize_cn 之后的文本。"""
    ge = bool(CN_GE_PRE.search(win) or CN_GE_POST.search(win)) or GE_MARK in win
    le = bool(CN_LE_PRE.search(win) or CN_LE_POST.search(win)) or LE_MARK in win
    return "GE" if ge and not le else "LE" if le and not ge else None


# ---------------------------------------------------------------- 轴 B: 倍半

EN_HALF = re.compile(r"(?i)\b(?:half|halves|halved?|halving|one-half)\b")
EN_DOUBLE = re.compile(r"(?i)\b(?:doubles?|doubled|doubling|twice|two\s+times|"
                       r"triples?|tripled|three\s+times)\b")
CN_HALF = re.compile(r"一半|半数|减半|折半|除以二|除以\s*2|50\s*%|1/2")
CN_DOUBLE = re.compile(r"[两二双三]倍|加倍|翻倍|倍增|乘以[二三]|乘以\s*[23]|翻一番")


# ---------------------------------------------------------------- 轴 C: 单位

UNIT_EN = re.compile(
    r"(?i)(?<![\w.])(\d{1,4}(?:,\d{3})*)\s*[-‐‑– ]?\s*"
    r"(rounds?|turns?|feet|foot|ft\.?|miles?|mi\.?|pounds?|lbs?\.?|"
    r"hours?|hrs?\.?|minutes?|mins?\.?|seconds?|secs?\.?|days?|weeks?|months?|years?)"
    r"(?![\w])")

UNIT_KEY = {}
for _grp, _k in [(("round", "rounds"), "round"), (("turn", "turns"), "turn"),
                 (("feet", "foot", "ft", "ft."), "foot"),
                 (("mile", "miles", "mi", "mi."), "mile"),
                 (("pound", "pounds", "lb", "lbs", "lb.", "lbs."), "pound"),
                 (("hour", "hours", "hr", "hrs", "hr.", "hrs."), "hour"),
                 (("minute", "minutes", "min", "mins", "min.", "mins."), "minute"),
                 (("second", "seconds", "sec", "secs", "sec.", "secs."), "second"),
                 (("day", "days"), "day"), (("week", "weeks"), "week"),
                 (("month", "months"), "month"), (("year", "years"), "year")]:
    for _w in _grp:
        UNIT_KEY[_w] = _k

CN_UNIT_TOKENS = [
    (r"回合", "turn"), (r"轮次", "round"), (r"轮", "round"),
    (r"英尺", "foot"), (r"呎", "foot"), (r"尺", "foot"),
    (r"英里", "mile"), (r"哩", "mile"), (r"里", "mile"),
    (r"磅", "pound"),
    (r"小时", "hour"), (r"分钟", "minute"), (r"秒钟", "second"), (r"秒", "second"),
    (r"星期", "week"), (r"周", "week"), (r"个月", "month"), (r"月", "month"),
    (r"年", "year"), (r"天", "day"), (r"日", "day"),
]
CN_UNIT_RE = re.compile("|".join(f"(?P<u{i}>{p})" for i, (p, _) in enumerate(CN_UNIT_TOKENS)))
CN_UNIT_MAP = {f"u{i}": k for i, (_, k) in enumerate(CN_UNIT_TOKENS)}

# 中文换个粒度表达同一时间量是合法的，不算错。
UNIT_COMPAT = {
    "hour": {"hour", "minute"}, "minute": {"minute", "second", "hour"},
    "second": {"second", "minute"},
    "day": {"day", "hour", "week"}, "week": {"week", "day"},
    "month": {"month", "week", "day"}, "year": {"year", "month", "day"},
    "foot": {"foot"}, "mile": {"mile", "foot"}, "pound": {"pound"},
    "round": {"round"}, "turn": {"turn"},
}


def cn_unit_at(cn: str, pos: int, span: int = 6):
    tail = re.sub(r"^[\s 　（(]*", "", cn[pos:pos + span])
    m = CN_UNIT_RE.match(tail)
    if not m:
        return None
    for g, k in CN_UNIT_MAP.items():
        if m.group(g):
            return k
    return None


# ---------------------------------------------------------------- 轴 D: round/turn 计数

# `first turn it back on` 里的 turn 是动词，不是回合。后跟宾语/小品词的一律排除。
EN_TURN_N = re.compile(
    r"(?i)\b(?:\d+\s+turns?"
    r"|(?:your|their|his|her|its|the|each|every|next|this|one|a|another|following|"
    r"same|first|last|current)\s+turns?"
    r"|turns?\s+(?:ends?|begins?|starts?)|per\s+turn)\b"
    r"(?!\s+(?:it|them|him|her|me|us|you|on|off|back|around|away|into|over|up|down|"
    r"toward|towards|left|right|the|a|an|this|that|his|its|their)\b)")
EN_ROUND_N = re.compile(
    r"(?i)\b(?:\d+\s+rounds?"
    r"|(?:each|every|the|a|per|next|this|first|last|subsequent|following|combat|"
    r"current)\s+rounds?"
    r"|rounds?\s+(?:of\s+combat|ends?|begins?)|per\s+round)\b")
CN_TURN_N = re.compile(r"回合")
# 「轮」的干扰词：轮廓/轮子/轮流/轮到/车轮/齿轮…… 前后各排一遍。
CN_ROUND_N = re.compile(r"(?<![车齿转飞滑年月日火巡])轮(?![廓子胎椅船番流转到值班盘回岗换作唱])")


# ---------------------------------------------------------------- 数字锚点

NUM = re.compile(r"(?<![\w.])(\d{1,4}(?:,\d{3})*)(?![\d.]*[\w])")


def unique_numbers(en: str, cn: str):
    """英文里只出现一次、中文里也只出现一次的数字 -> {num: (en_s, en_e, cn_s, cn_e)}"""
    ec = collections.Counter(m.group(1) for m in NUM.finditer(en))
    cn_ms = list(NUM.finditer(cn))
    cc = collections.Counter(m.group(1) for m in cn_ms)
    out = {}
    for m in NUM.finditer(en):
        n = m.group(1)
        if ec[n] != 1 or cc.get(n) != 1:
            continue
        cm = next(x for x in cn_ms if x.group(1) == n)
        out[n] = (m.start(1), m.end(1), cm.start(1), cm.end(1))
    return out


# ---------------------------------------------------------------- 遍历


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        p = ".".join(path)
        out.append({"path": p,
                    "batch_path": p[len("entries."):] if p.startswith("entries.") else p,
                    "en": en, "cn": cn if isinstance(cn, str) else None})


def load_pairs(repo):
    en_d = os.path.join(repo, "compendium", "en")
    cn_d = os.path.join(repo, "compendium", "cn")
    rows = []
    for fn in sorted(os.listdir(en_d)):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        en = json.load(open(os.path.join(en_d, fn), encoding="utf-8-sig"))
        cp = os.path.join(cn_d, fn)
        cn = json.load(open(cp, encoding="utf-8-sig")) if os.path.isfile(cp) else {}
        sub = []
        walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], sub)
        for r in sub:
            r["pack"] = fn
            r["repo"] = os.path.basename(repo.rstrip("/\\"))
        rows.extend(sub)
    return rows


# ---------------------------------------------------------------- 判据


def check_pair(row, axes, modes, en_pre=48, en_post=26, cn_pre=12, cn_post=10,
               leaf_max=900, rt_max=1500):
    en_raw, cn_raw = row["en"], row.get("cn")
    if not cn_raw:
        return []
    en, cn = norm(en_raw), norm(cn_raw)
    hits = []

    def add(axis, mode, a, b, anchor, es, ee, cs, ce, pad=(90, 60, 40, 30)):
        hits.append({"axis": axis, "mode": mode, "en_side": a, "cn_side": b,
                     "anchor": anchor,
                     "en_ctx": en[max(0, es - pad[0]):ee + pad[1]].strip(),
                     "cn_ctx": cn[max(0, cs - pad[2]):ce + pad[3]].strip()})

    if "dir" in axes:
        # 否定式必须在**整叶**上先消掉：「角色无法通过这种方式获得超过 3 级」里
        # 否定词离关系词 9 个字，锚点小窗根本看不见它。同长替换保证下标不变。
        en_n, cn_n = neutralize_en(en), neutralize_cn(cn)
        if "leaf" in modes and len(en) <= leaf_max:
            de, dc = en_dir(en_n), cn_dir(cn_n)
            if de and dc and de != dc:
                add("dir", "leaf", de, dc, None, 0, min(len(en), 300),
                    0, min(len(cn), 300), (0, 0, 0, 0))
        if "anchor" in modes:
            for n, (es, ee, cs, ce) in unique_numbers(en, cn).items():
                # 窗口必须**连着锚点数字本身**，否则 `5+` 这种后置下界写法
                # 的 (?<=\d) 回顾断言会因为窗口从「+」起头而落空。
                ew = en_n[max(0, es - en_pre):ee + en_post]
                cw = cn_n[max(0, cs - cn_pre):ce + cn_post]
                de, dc = en_dir(ew), cn_dir(cw)
                if de and dc and de != dc:
                    add("dir", "anchor", de, dc, n, es, ee, cs, ce)

    if "mag" in axes and "leaf" in modes and len(en) <= leaf_max:
        eh, ed = bool(EN_HALF.search(en)), bool(EN_DOUBLE.search(en))
        ch, cd = bool(CN_HALF.search(cn)), bool(CN_DOUBLE.search(cn))
        if eh and not ed and cd and not ch:
            add("mag", "leaf", "HALF", "DOUBLE", None, 0, min(len(en), 300),
                0, min(len(cn), 300), (0, 0, 0, 0))
        elif ed and not eh and ch and not cd:
            add("mag", "leaf", "DOUBLE", "HALF", None, 0, min(len(en), 300),
                0, min(len(cn), 300), (0, 0, 0, 0))

    if "unit" in axes:
        uniq = unique_numbers(en, cn)
        for m in UNIT_EN.finditer(en):
            n, u = m.group(1), UNIT_KEY.get(m.group(2).lower())
            if not u or n not in uniq:
                continue
            es, ee, cs, ce = uniq[n]
            if m.start(1) != es:
                continue
            cu = cn_unit_at(cn, ce)
            if cu is None or cu == u or cu in UNIT_COMPAT.get(u, {u}):
                continue
            add("unit", "anchor", u, cu, n, es, ee, cs, ce, (70, 50, 30, 25))

    # 整叶计数只在规则文本长度内可靠。超过 rt_max 的基本都是 journal 长页，
    # `turn it back on`、「黄铜轮」、「多轮循环」这类非术语用法密度骤升，
    # 计数失衡不再说明问题（实测 1611/4225 字的两页全是假阳性）。
    # 长页里的 round/turn 错译得靠 `unit` 轴的数字锚定去抓。
    if "unit_rt" in axes and len(en) <= rt_max:
        r_en = list(EN_ROUND_N.finditer(en))
        t_en = list(EN_TURN_N.finditer(en))
        r_cn = len(CN_ROUND_N.findall(cn))
        t_cn = len(CN_TURN_N.findall(cn))
        # 中文侧**一个「轮」都没有**，却比英文多出「回合」—— 多出来的只能是 round 误译。
        if r_en and r_cn == 0 and t_cn > len(t_en):
            m = r_en[0]
            j = cn.find("回合")
            add("unit_rt", "leaf", f"round×{len(r_en)}/turn×{len(t_en)}",
                f"轮×0/回合×{t_cn}", None, m.start(), m.end(), j, j + 2, (60, 35, 28, 20))
        elif t_en and t_cn == 0 and r_cn > len(r_en):
            m = t_en[0]
            j = CN_ROUND_N.search(cn).start()
            add("unit_rt", "leaf", f"round×{len(r_en)}/turn×{len(t_en)}",
                f"轮×{r_cn}/回合×0", None, m.start(), m.end(), j, j + 1, (60, 35, 28, 20))
    return hits


def scan(rows, axes, modes, **kw):
    out = []
    for r in rows:
        for h in check_pair(r, axes, modes, **kw):
            h.update({"repo": r["repo"], "pack": r["pack"], "path": r["path"],
                      "batch_path": r["batch_path"]})
            out.append(h)
    return out


# ---------------------------------------------------------------- 灵敏度自测

SELF_TEST = [
    ("Deal at least 3 damage.", "造成至多 3 点伤害。", "dir", True),
    ("Deal at least 3 damage.", "造成至少 3 点伤害。", "dir", False),
    ("Deal at least 3 damage.", "造成 3 点或更多伤害。", "dir", False),
    ("Deal at least 3 damage.", "造成 3 点以下的伤害。", "dir", True),
    ("You may move up to 30 feet.", "你可以移动至少 30 尺。", "dir", True),
    ("You may move up to 30 feet.", "你最多可以移动 30 尺。", "dir", False),
    ("Targets with 5 or fewer wounds.", "创伤 5 点及以上的目标。", "dir", True),
    # 坑 1 英文否定式
    ("It almost never exceeds 2 quality.", "其品质几乎从不超过 2 。", "dir", False),
    ("It must be no larger than a 5-foot Cube.", "它不得大于一个 5 尺立方体。", "dir", False),
    ("She can have no more than 3 eyes.", "她控制的眼睛不能超过 3 个。", "dir", False),
    ("She can have no more than 3 eyes.", "她控制的眼睛不能少于 3 个。", "dir", True),
    ("Your armor class cannot be less than 16.", "你的护甲等级不会低于 16。", "dir", False),
    ("This should not exceed more than once every 10 minutes.",
     "频率不应超过每 10 分钟一次。", "dir", False),
    ("Concentration lasts longer with a slot of level 2 (up to 4 hours) or 5+ (24 hours).",
     "使用 2 环法术位时（最长 4 小时）；使用 5 环或更高法术位时（24 小时）。", "dir", False),
    # 坑 4 「以下」＝下文
    ("You need at least 2 of the following items.", "你需要至少 2 件以下物品。", "dir", False),
    ("Choose up to 3 from the list below.", "从下表中选择至多 3 项。", "dir", False),
    ("Gain at least 1 point. Spend at most 4 points.", "获得至少 1 点。花费至多 4 点。", "dir", False),
    # unit
    ("The effect lasts 3 rounds.", "该效果持续 3 回合。", "unit", True),
    ("The effect lasts 3 rounds.", "该效果持续 3 轮。", "unit", False),
    ("At the start of your next turn, take 2 damage.", "在你下一回合开始时，受到 2 点伤害。", "unit", False),
    ("Gain 4 turns of Haste.", "获得 4 轮迅捷。", "unit", True),
    ("Move 60 feet.", "移动 60 米。", "unit", False),
    ("Move 60 feet.", "移动 60 里。", "unit", True),
    ("It lasts 8 hours.", "持续 8 小时。", "unit", False),
    ("It lasts 8 hours.", "持续 8 天。", "unit", True),
    # mag
    ("You take half damage.", "你受到双倍伤害。", "mag", True),
    ("You take half damage.", "你受到一半伤害。", "mag", False),
    ("Damage is doubled.", "伤害减半。", "mag", True),
    ("Roll twice and take the higher.", "掷两次取较高值。", "mag", False),
    # unit_rt
    ("It may be Maintained for 1 Focus every subsequent round.",
     "可以维持，此后每个后续回合额外消耗 1 专注。", "unit_rt", True),
    ("It may be Maintained for 1 Focus every subsequent round.",
     "可以维持，此后每个后续轮次额外消耗 1 专注。", "unit_rt", False),
    ("Effects have a stated duration in combat Rounds. Reactions occur on another creature's turn.",
     "效果注明以战斗轮为单位的持续时间。反应在其他生物的回合执行。", "unit_rt", False),
    ("Once per Round you gain +1 Action.", "每回合你获得 +1 动作。", "unit_rt", True),
    ("Actions made during their next turn gain a Boon.", "他们在下一轮内的动作获得恩惠骰。", "unit_rt", True),
    ("Actions made during their next turn gain a Boon.", "他们在下一回合内的动作获得恩惠骰。", "unit_rt", False),
    ("It cannot use the same option more than once per round.",
     "它每轮不能使用同一个选项超过一次。", "dir", False),
    ("A character cannot gain more than 3 levels of Exhaustion this way.",
     "角色无法通过这种方式获得超过 3 级的力竭。", "dir", False),
    ("It recharges on a 5+, and creatures can see within 30 feet.",
     "该特性在掷出 5+ 时充能，生物只能看清 30 尺以内的范围。", "dir", False),
    ("They must first turn it back on before the round ends.",
     "他们必须先把它重新打开，然后这一轮才结束。", "unit_rt", False),
    # 「轮到」「轮流」不算「轮」
    ("Each creature acts on its turn in order.", "每个生物按顺序在轮到自己时行动的回合中行动。", "unit_rt", False),
]


def self_test():
    ok = True
    for en, cn, axis, expect in SELF_TEST:
        row = {"en": en, "cn": cn, "path": "t", "batch_path": "t", "pack": "t", "repo": "t"}
        hs = [h for h in check_pair(row, {"dir", "mag", "unit", "unit_rt"},
                                    {"leaf", "anchor"}) if h["axis"] == axis]
        got = bool(hs)
        if got != expect:
            ok = False
        print(f"{'OK  ' if got == expect else 'FAIL'} [{axis:<7}] expect={expect!s:<5} "
              f"got={got!s:<5} | {en}  ||  {cn}"
              + (f"   -> {hs[0]['en_side']}->{hs[0]['cn_side']}" if hs else ""))
    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", default=[])
    ap.add_argument("--axis", default="dir,mag,unit,unit_rt")
    ap.add_argument("--mode", default="leaf,anchor")
    ap.add_argument("--leaf-max", type=int, default=900)
    ap.add_argument("--rt-max", type=int, default=1500,
                    help="unit_rt 轴只看归一化后不超过这么长的叶子")
    ap.add_argument("--out")
    ap.add_argument("--show", type=int, default=60)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--inject", help="JSON: [[path_regex, cn_old, cn_new], ...] 内存注错回测")
    a = ap.parse_args()

    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.repo:
        ap.error("--repo required")

    axes = {x.strip() for x in a.axis.split(",") if x.strip()}
    modes = {x.strip() for x in a.mode.split(",") if x.strip()}

    rows = []
    for r in a.repo:
        rows.extend(load_pairs(r))

    injected = []
    if a.inject:
        for pat, old, new in json.load(open(a.inject, encoding="utf-8")):
            rx = re.compile(pat)
            for r in rows:
                if r["cn"] and rx.search(r["path"]) and old in r["cn"]:
                    r["cn"] = r["cn"].replace(old, new, 1)
                    injected.append({"path": r["path"], "pack": r["pack"],
                                     "from": old, "to": new})
                    break

    hits = scan(rows, axes, modes, leaf_max=a.leaf_max, rt_max=a.rt_max)
    by = collections.Counter(f"{h['axis']}/{h['mode']}" for h in hits)
    scanned = {"leaves": len(rows),
               "leaves_with_cn": sum(1 for r in rows if r["cn"]),
               "en_chars": sum(len(r["en"]) for r in rows)}
    payload = {"scanned": scanned, "counts": dict(sorted(by.items())),
               "total": len(hits), "injected": injected, "hits": hits}
    if a.out:
        open(a.out, "w", encoding="utf-8").write(
            json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"leaves={scanned['leaves']} withCN={scanned['leaves_with_cn']} "
          f"enChars={scanned['en_chars']}")
    for k, v in sorted(by.items()):
        print(f"  {k}: {v}")
    print(f"TOTAL={len(hits)}" + (f" -> {a.out}" if a.out else ""))
    if injected:
        found = {h["path"] for h in hits}
        for inj in injected:
            print(f"  INJECT {'CAUGHT' if inj['path'] in found else 'MISSED'}: "
                  f"{inj['from']} -> {inj['to']}  @ {inj['path']}")
    for h in hits[:a.show]:
        print(f"\n--- [{h['axis']}/{h['mode']}] {h['en_side']} -> {h['cn_side']} "
              f"n={h['anchor']} | {h['pack']} {h['path']}")
        print(f"  EN: {h['en_ctx'][:340]}")
        print(f"  CN: {h['cn_ctx'][:340]}")


if __name__ == "__main__":
    main()
