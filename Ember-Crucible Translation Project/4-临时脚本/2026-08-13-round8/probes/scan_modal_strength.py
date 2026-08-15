# -*- coding: utf-8 -*-
"""情态强度错配（must / should / may）—— 规则文本的**约束强度**在中文里被改了。

为什么既有判据全盲
------------------
「GM 可以给一个恩惠骰」与「GM 必须给一个恩惠骰」是两条不同的规则，
但两者的字数、标记、数字、块数、UUID、class 完全一样。
标记五项 / class 漂移 / 数字覆盖 / 外来文字 / 死键 / tokenName / 孪生分叉
没有一项能看见情态词。

三种子模式（--mode）
--------------------
  conflict （默认）英文单一情态档 ↔ 中文跨档冲突（must→可以 / may→必须 / 极性翻转）
  invented       英文侧一个情态词都没有，中文却写出「必须 / 不得」
  actor          英文侧情态句的施动者（the GM / the target / players）在中文里换了人

判据（conflict）
----------------
1. 每个叶子按 **HTML 块**（</p> </li> </h*> </td> <br> \\n）切开；
   en 块数 == cn 块数时 1:1 对齐，否则整叶做一个单元。
2. 块内再按句号切；en 句数 == cn 句数（且 >1）时 1:1 对齐（granularity=sent），
   否则用块（granularity=block）。单元英文长度限制在 [--min-chars, --max-chars]。
3. 英文侧情态档位**恰好一个**（多档位无法判定中文里的「必须」对应哪一句，跳过）。
4. 中文侧若含任何与英文档位**相容**的标记 → 放行。
5. 三道收紧闸（下面「回测教训」）全部通过后，才按 HARD 表报出。

档位
----
  OBLIG   must / shall / is required to / have to / got to / be sure to
  PERMIT  may / can / could / at your option / optionally / chooses to
  PROHIB  must not / cannot / may not / can no longer / is not allowed
  RECOM   should / ought to / recommended / feel free to
  NONOB   need not / does not have to / no need to
  EPIST   might / possibly / perhaps

回测教训（v1 报 410 条、v2 报 92 条，逐条人看后**假阳性率 100%**，
以下每一条都是被实测打出来的，删掉任何一条 FP 都会回涨）
-------------------------------------------------------------
 F1 中文「能不能 / 可不可以」是疑问式，`不能` 只是它的一半。
    实测 v2 的 92 条里 ~20 条是 “see if you can …” → 「看看能不能…」。
 F2 中文「不可X / 不得X」多半是成语：不可摧毁 / 不可想象 / 不可靠 / 妙不可言 /
    不得而知 / 万不得已 / 见不得人 / 并非不能。整串挖掉再判。
 F3 英文侧只要还剩否定/负极性词（not / no / none / without / unless / impossible /
    unable / hardly / fail / im-·un- 前缀形容词），中文的「无法 / 不能」多半忠实。
 F4 推断框架里的 must 是「想必」不是义务：clear that … must be / assured / seems。
 F5 英文残余里若已独立支持中文那个档位（requiring / necessary / before / only if /
    optional / allow / prohibit / recommend），说明中文没改强度，是在译别的词。
 F6 `can only` / `may only` / `only … can` 是受限许可，中文标准译法就是「必须…才能」。
 F7 固定搭配：have to do with / has to offer / cannot stress enough / may as well /
    must not only … but also（这里的 not 属于 not only，不是否定 must）。
 F8 句首 `Should a character …` 是条件倒装 = if，不是「应当」。
 F9 中文的「可 / 能 + 动词」（可包含、能看见）也是许可标记，不认它会把忠实译文报成
    「许可被丢掉」—— 实测这一条让 permit_drop 从 1480 条降到近乎全是噪声。

用法：
  python scan_modal_strength.py --repo <repoDir> [--repo <另一个>] --out <json>
  python scan_modal_strength.py --repo <repoDir> --mode invented --out <json>
  python scan_modal_strength.py --repo <repoDir> --mode actor    --out <json>
  python scan_modal_strength.py --repo <repoDir> --soft          # 连同同向弱化一起报
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

# ---------------------------------------------------------------- 文本规整

BLOCK_END = re.compile(
    r"</(?:p|li|h[1-6]|td|th|tr|div|blockquote|section|figcaption|dt|dd)>"
    r"|<br\s*/?>|\r?\n",
    re.I,
)
ENRICH_LABEL = re.compile(r"@\w+\[[^\]]*\]\{([^}]*)\}")
ENRICH_BARE = re.compile(r"@\w+\[[^\]]*\]")
ROLL_LABEL = re.compile(r"\[\[[^\]]*?\]\]\{([^}]*)\}")
ROLL_BARE = re.compile(r"\[\[.*?\]\]")
REF_AMP = re.compile(r"&[Rr]eference\[[^\]]*\](\{[^}]*\})?")
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")


def clean(s: str) -> str:
    """去掉 enricher 的方括号内容（只留 {标签}）与 HTML 标签。"""
    s = ROLL_LABEL.sub(r" \1 ", s)
    s = ROLL_BARE.sub(" ", s)
    s = REF_AMP.sub(" ", s)
    s = ENRICH_LABEL.sub(r" \1 ", s)
    s = ENRICH_BARE.sub(" ", s)
    s = TAG.sub(" ", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&mdash;", "—")
    s = s.replace("&rsquo;", "'").replace("&lsquo;", "'")
    s = s.replace("&ldquo;", '"').replace("&rdquo;", '"')
    return WS.sub(" ", s).strip()


def blocks(s: str) -> list:
    return [p for p in (clean(p) for p in BLOCK_END.split(s)) if p]


ABBR = re.compile(r"\b(?:e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr|St|No|Fig|approx|ft|lb)\.$", re.I)
EN_SENT = re.compile(r'(?<=[.!?])["\')\]]?\s+')
CN_SENT = re.compile(r'(?<=[。！？])["”’）]?\s*')


def en_sentences(s: str) -> list:
    out = []
    for piece in EN_SENT.split(s):
        if not piece.strip():
            continue
        if out and (ABBR.search(out[-1]) or re.search(r"\d\.$", out[-1])):
            out[-1] = out[-1] + " " + piece
        else:
            out.append(piece)
    return [x.strip() for x in out if x.strip()]


def cn_sentences(s: str) -> list:
    return [x.strip() for x in CN_SENT.split(s) if x.strip()]


# ---------------------------------------------------------------- 英文情态

EN_PAT = {
    "PROHIB": [
        r"\bmust\s+not\s+(?!only\b)", r"\bmustn['’]t\b", r"\bmay\s+not\s+(?!only\b)",
        r"\bmight\s+not\b", r"\bshall\s+not\b", r"\bcannot\b", r"\bcan\s*['’]t\b",
        r"\bcan\s+not\s+(?!only\b)", r"\bcan\s+no\s+longer\b", r"\bmay\s+no\s+longer\b",
        r"\bcan\s+never\b", r"\bmay\s+never\b", r"\b(?:is|are)\s+not\s+allowed\b",
        r"\b(?:is|are)\s+forbidden\b", r"\b(?:is|are)\s+prohibited\b",
    ],
    "NONOB": [
        r"\bneed\s+not\b", r"\bneedn['’]t\b", r"\bdo(?:es)?\s+not\s+need\s+to\b",
        r"\bdon['’]t\s+need\s+to\b", r"\bdoesn['’]t\s+need\s+to\b",
        r"\bdo(?:es)?\s+not\s+have\s+to\b", r"\bdon['’]t\s+have\s+to\b",
        r"\b(?:is|are)\s+not\s+required\s+to\b", r"\bno\s+need\s+to\b",
    ],
    "OBLIG": [
        r"\bmust\b", r"\bshall\b",
        r"\b(?:is|are|was|were|be|been|being)\s+required\s+to\b",
        r"\b(?:has|have|had)\s+to\b", r"\bmandatory\b",
        r"\b(?:'ve|'s|has|have|had)?\s*got\s+to\b", r"\bgotta\b",
        r"\bbe\s+sure\s+to\b", r"\bmake\s+sure\s+to\b", r"\bbe\s+certain\s+to\b",
        r"\b(?:is|are)\s+incumbent\s+upon\b", r"\b(?:is|are)\s+essential\b",
        r"\bnecessitat\w+", r"\bbe\s+careful\s+to\b",
    ],
    "RECOM": [
        r"\bshould\b", r"\bshouldn['’]t\b", r"\bought\s+to\b", r"\brecommended\b",
        r"\bwe\s+recommend\b", r"\bit\s+is\s+a\s+good\s+idea\b",
        r"\bit['’]s\s+a\s+good\s+idea\b", r"\bfeel\s+free\s+to\b",
        r"\b(?:is|are)\s+encouraged\s+to\b",
    ],
    "PERMIT": [
        r"\bmay\b", r"\bcan\b", r"\bcould\b",
        r"\b(?:is|are)\s+allowed\s+to\b", r"\b(?:is|are)\s+free\s+to\b",
        r"\bat\s+(?:your|their|his|her|its|the\s+\w+['’]s)\s+option\b",
        r"\boptionally\b", r"\bif\s+(?:you|they|he|she)\s+(?:so\s+)?choose",
        r"\b(?:choose|chooses|chose)\s+to\b", r"\bhave\s+the\s+option\b",
    ],
    "EPIST": [r"\bmight\b", r"\bpossibly\b", r"\bperhaps\b", r"\bmaybe\b"],
}
# F10 `must have been` / `there must be` 是推断（想必），不是义务 —— 归 EPIST，
# 所以 EPIST 必须排在 OBLIG **之前**（两者的词形不重叠，提前不影响别的档）。
EN_PAT["EPIST"] = [
    # 只认 must have + 过去分词（完成体推断）。`must have its elevation set` 是义务，
    # 不能一并吞掉 —— v5 的 3 条里 2 条就栽在这。
    r"\bmust\s+have\s+(?:been|come|gone|taken|known|seen|done|made|felt|left|found"
    r"|\w+ed)\b",
    r"\bthere\s+must\s+be\b", r"\bmust\s+be\s+some\b",
] + EN_PAT["EPIST"]
EN_ORDER = ["PROHIB", "NONOB", "EPIST", "OBLIG", "RECOM", "PERMIT"]
EN_RX = {k: re.compile("|".join(v), re.I) for k, v in EN_PAT.items()}

# F6 受限许可
RESTRICT_RX = re.compile(
    r"\b(?:can|may|could)\s+only\b"
    r"|\bonly\b[^.]{0,80}?\b(?:can|may|could|able\s+to)\b"
    r"|\b(?:can|may|could)\b[^.]{0,60}?\bonly\b"
    # F12 限定性关系从句门槛：「Any character who makes a successful check can X」
    # 的标准中译就是「角色必须通过检定，才能 X」—— 不是强度被改。
    r"|\b(?:any|each|every|only|a|an|the)\s+\w+\s+(?:who|which|that)\b"
    r"[^.]{0,90}?\b(?:can|may|could)\b",
    re.I,
)
# F8 条件倒装 Should / Were / Had
COND_INVERSION = re.compile(r"^\s*(?:should|were|had)\s+(?:a|an|the|any|no)?\s*\w", re.I)

# F7 固定搭配：先挖空
IDIOM_RX = re.compile(
    r"\b(?:has|have|had)\s+to\s+do\s+with\b"
    r"|\b(?:has|have|had)\s+to\s+offer\b"
    r"|\bcan(?:not|['’]t)\s+stress\b"
    r"|\b(?:may|might)\s+as\s+well\b"
    # 「could use X」= 需要 X，不是「可以用」
    r"|\bcould\s+(?:really\s+|all\s+|sure\s+)?use\b"
    # 「有什么话/信息 可说」—— have to 是「手上有可以…的」，不是义务
    r"|\b(?:information|news|stor(?:y|ies)|advice|wisdom|thing|things|what)\b"
    r"[^.]{0,40}?\b(?:has|have|had)\s+to\s+"
    r"(?:say|tell|share|offer|give|add|contribute|work\s+with)\b"
    # 口头禅
    r"|\b(?:must|have\s+to|had\s+to|got\s+to)\s+(?:admit|say|confess|tell\s+you)\b"
    r"|\bnot\s+only\b",
    re.I,
)

# F3 否定 / 负极性
NEG_CTX = re.compile(
    r"\bnot\b|n['’]t\b|\bno\b|\bnone\b|\bnever\b|\bnothing\b|\bnobody\b|\bneither\b"
    r"|\bwithout\b|\bnor\b|\bfew\b|\blittle\b|\bhardly\b|\bbarely\b|\bscarcely\b"
    r"|\bunless\b|\bexcept\b|\brefus\w*|\bfail(?:s|ed|ing|ure)?\b|\bprevent\w*"
    r"|\bstop(?:s|ped)?\s+\w+\s+from\b|\bcease\w*|\black\w*|\bdeni\w+|\bdeny\b"
    r"|\bunable\b|\bim(?:possib|passab|passib|penetrab|mobil|measurab|perceptib)\w+"
    r"|\bin(?:abilit|accessib|audib|visib|distinguishab|extricab|efficien|capab)\w+"
    r"|\bun(?:\w{2,}?)(?:able|ible)\b|\bungovernab\w+|\bunclear\b|\bobscur\w*"
    r"|\bindecipherab\w+|\bdisallow\w*|\brestrict\w*|\blimited\s+to\b|\btoo\s+\w+\s+to\b"
    # F13 语义否定的形容词（invented 模式的主要噪声源：ineffective / irreparable /
    # unlikely / -less 后缀都会在中文里正当地变成「无法 / 不可能」）
    r"|\bin(?:effectiv|effectu|complet|adequat|sufficien|access|escapab|animat)\w+"
    r"|\bir(?:reparab|reversib|revocab|resistib|rational)\w+"
    r"|\bunlikely\b|\bimprobab\w+|\bimplausib\w+|\bfutile\b|\bmoot\b|\bvain\b"
    r"|\bbeyond\s+\w+\b"
    r"|\b\w{3,}less\b|\bcease\w*|\bbarred\b|\bshut\b|\bsealed\b",
    re.I,
)

# F4 推断框架
EPI_FRAME = re.compile(
    r"\bclear(?:ly)?\b|\bseems?\b|\bseemed\b|\bappears?\b|\bappeared\b|\bassur\w+"
    r"|\brealiz\w+|\bsuspect\w*|\bconclud\w*|\bpresum\w*|\bsurmis\w*|\bdeduc\w*"
    r"|\bevident\w*|\bobvious\w*|\bapparent\w*|\bguess\w*|\bbeliev\w*|\binfer\w*"
    r"|\bknows?\b|\bsurely\b|\bcertainly\b|\bprobably\b|\blikely\b|\bposits?\b",
    re.I,
)

# F5 英文残余独立支持中文那个档位
IMPLIES = {
    "OBLIG": re.compile(
        r"\brequir\w*|\bneed\w*|\bnecessar\w+|\bnecessit\w+|\bprerequisite\w*"
        r"|\bmandat\w+|\bbefore\b|\buntil\b|\bin\s+order\s+to\b|\bobligat\w+"
        r"|\bforce[sd]?\b|\bforcing\b|\bcompel\w*|\bonly\s+if\b|\bdemand\w*"
        r"|\binsist\w*|\bessential\w*|\bvital\w*|\bimperative\w*|\bincumbent\b"
        r"|\bearned\b|\bexpects?\b|\bgo(?:es)?\s+to\s+great\s+lengths\b"
        r"|\btakes?\s+care\b|\bburden\b|\bimposed?\b", re.I),
    "PERMIT": re.compile(
        r"\boption(?:al|ally|s)?\b|\bchoos\w+|\bchoice\w*|\bchose\b|\bchosen\b"
        r"|\bat\s+will\b|\bfree(?:ly)?\b|\ballow\w*|\bpermit\w*|\bable\b|\babilit\w+"
        r"|\bcapable\b|\bafford\w*|\bmanage[sd]?\s+to\b|\bwelcome\s+to\b"
        r"|\bwish(?:es)?\s+to\b|\bwant(?:s)?\s+to\b|\bdecid\w+|\bdiscretion\b"
        r"|\brepeatable\b|\busable\b|\bavailable\b", re.I),
    "PROHIB": re.compile(
        r"\bprohibit\w*|\bforbid\w*|\bdisallow\w*|\bden(?:y|ies|ied)\b|\bimmune\b"
        r"|\bimpossib\w+|\bincapab\w+|\bblock\w*|\bbar(?:s|red)\b|\bseal\w*"
        r"|\block\w*|\bimpenetrab\w+|\bunbreakab\w+|\bimpassab\w+|\bimpassib\w+"
        r"|\bhold(?:s|ing)?\s+off\b|\bheld\s+off\b|\bwithheld\b"
        # 上限类：`up to its capacity` 的中译就是「不能超过其容量上限」
        r"|\bup\s+to\b|\bat\s+most\b|\bmaximum\b|\bcapacity\b|\blimit\w*"
        r"|\bexceed\w*|\bno\s+more\s+than\b|\bneither\b|\bnor\b"
        r"|\btakes?\s+(?:your|his|her|their|my)\s+breath\s+away\b", re.I),
    "RECOM": re.compile(
        r"\brecommend\w*|\badvis\w+|\bsuggest\w*|\bconsider\b|\bencourag\w*"
        r"|\bgood\s+idea\b|\bprefer\w*|\bideal\w*|\bbest\b", re.I),
    "NONOB": re.compile(r"\boption(?:al|ally)?\b|\bwithout\s+need\b|\bskip\w*", re.I),
    "EPIST": re.compile(
        r"\blikely\b|\bprobabl\w+|\bperhaps\b|\bpossib\w+|\bchance\b|\bmaybe\b"
        r"|\bseem\w*|\bappear\w*|\bunlikely\b|\bpotential\w*", re.I),
}

# ---------------------------------------------------------------- 中文强度

# F1 + F2：先整串挖掉的中文固定搭配 / 疑问式 / 成语
CN_IDIOM = re.compile(
    r"能不能|可不可以|行不行|是否能|能否|可否"
    r"|并非不能|不能不|无不|不无|不得不已"
    r"|不可(?:摧|摧毁|摧折|思议|想象|想像|名状|理喻|预测|预知|估量|替代|或缺|磨灭|逆转|抗拒"
    r"|避免|靠|多得|计数|胜数|限量|开交|言|同日|方物|一世|终日|燃)"
    r"|不得(?:而知|已|人|了|不)"
    r"|见不得|舍不得|由不得|怪不得|恨不得|巴不得|免不得|少不得|说不得|动不得|要不得|信不得"
    r"|坚不可摧|妙不可言|万不得已"
    r"|不能自已|无法无天"
)

CN_PAT = {
    "PROHIB": (r"不得|不可以|不可(?!能)|不能|不许|不准|禁止|严禁|不允许|不被允许|未被允许"
               r"|无法|无从|不让"),
    "NONOB": r"无需|无须|不必|毋须|不需要|没必要|非必需|并非必需",
    "OBLIG": r"必须|必需|务必|一定要|不得不|非得|亟需|亟须",
    "RECOM": r"应当|应该|理应|建议|最好(?!的|地)|宜于|不妨|本该|该当",
    # 严格许可标记：只有这些才会被当成「许可」证据报出冲突
    "PERMIT": (r"可以|(?<![不未经被])允许|能够|可选|亦可|也可(?!能)|均可|皆可|可自行"
               r"|听凭|自由选择|任选|可供选择|有权|大可|得以|可任意"),
    "EPIST": r"可能|也许|或许|大概|说不定|未必|多半|想必|必定|应是",
    # F9 / F11 弱档：这些中文词**歧义太大**，既不当冲突证据、也不当放行凭据
    # （它们不在 HARD/SOFT 表里，也不在 COMPATIBLE 里，命中即静默丢弃该单元）。
    #   PERMITW 可/能/会 + 动词  —— v3 首跑 55 条里 46 条是这么来的
    #   OBLIGW  需要/只能/才能   —— v4 24 条里 20 条是这么来的（"could use"→需要）
    #   PROHIBW 难以/并非/不容   —— 形容词性否定，不是禁令
    "PERMITW": (r"可(?![以能是爱怕靠惜怜观口谓知见恶敬悲喜贵恨疑])[一-鿿]"
                r"|能(?!不)[一-鿿]|会[一-鿿]"),
    "OBLIGW": r"需要|须要|只能|只得|只好|唯有|得先|要求|才能|才可|不得不已",
    "PROHIBW": r"不容|并非|难以|不容易|没能|未能|力所不及|谈不上",
}
CN_ORDER = ["NONOB", "PROHIB", "OBLIG", "RECOM", "EPIST", "PERMIT",
            "OBLIGW", "PROHIBW", "PERMITW"]
CN_RX = {k: re.compile(v) for k, v in CN_PAT.items()}


def _blank(m):
    return "　" * len(m.group(0))


def en_tiers(text: str):
    """返回 ({档位: [命中片段]}, 挖空后的残余文本)"""
    buf = IDIOM_RX.sub(_blank, text)
    got = {}
    for tier in EN_ORDER:
        hits = []

        def _sub(m):
            hits.append(m.group(0))
            return " " * len(m.group(0))

        buf = EN_RX[tier].sub(_sub, buf)
        if hits:
            got[tier] = hits
    return got, buf


def cn_tiers(text: str):
    buf = CN_IDIOM.sub(_blank, text)
    got = {}
    for tier in CN_ORDER:
        hits = []

        def _sub(m):
            hits.append(m.group(0))
            return "　" * len(m.group(0))

        buf = CN_RX[tier].sub(_sub, buf)
        if hits:
            got[tier] = hits
    return got


HARD = {
    ("OBLIG", "PERMIT"): ("必须→可以", "严重"),
    ("OBLIG", "PROHIB"): ("必须→不得（极性翻转）", "阻断"),
    ("OBLIG", "NONOB"): ("必须→无需（极性翻转）", "阻断"),
    ("PERMIT", "OBLIG"): ("可以→必须", "严重"),
    ("PERMIT", "PROHIB"): ("可以→不得（极性翻转）", "阻断"),
    ("RECOM", "OBLIG"): ("建议→必须", "严重"),
    ("RECOM", "PROHIB"): ("建议→不得（极性翻转）", "阻断"),
    ("NONOB", "OBLIG"): ("无需→必须（极性翻转）", "阻断"),
    ("PROHIB", "PERMIT"): ("不得→可以（极性翻转）", "阻断"),
    ("PROHIB", "OBLIG"): ("不得→必须（极性翻转）", "阻断"),
    ("EPIST", "OBLIG"): ("可能→必须", "严重"),
}
SOFT = {
    ("OBLIG", "RECOM"): ("必须→应当（弱化）", "一般"),
    ("RECOM", "PERMIT"): ("建议→可以（弱化）", "一般"),
    ("PERMIT", "RECOM"): ("可以→应当（强化）", "一般"),
}
# 中文侧出现这些档位之一 => 原意还在，放行。
# ⚠ 弱档（PERMITW / OBLIGW / PROHIBW）**不进这张表** —— 它们只是把歧义词从文本里
# 消掉，既不作证据也不作放行凭据。回测教训：把 PERMITW 当放行凭据时，
# 注入的「GM may -> 必须」被「才能/可…」顺手放行，漏报。
# 同理 RECOM 的相容集里**不能**放 OBLIG，否则 should->必须 永远报不出来。
COMPATIBLE = {
    "OBLIG": {"OBLIG", "RECOM"},
    "PERMIT": {"PERMIT", "EPIST", "RECOM"},
    "RECOM": {"RECOM", "PERMIT"},
    # 未必 / 可能不 —— 英文 `may not be clear` 的忠实中译，算相容
    "PROHIB": {"PROHIB", "NONOB", "EPIST"},
    "NONOB": {"NONOB", "PROHIB", "PERMIT", "EPIST"},
    "EPIST": {"EPIST", "PERMIT", "RECOM"},
}


# ---------------------------------------------------------------- 遍历

def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            sub = cn.get(k) if isinstance(cn, dict) else None
            walk(v, sub, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        p = ".".join(path)
        out.append({
            "path": p,
            "batch_path": p[len("entries."):] if p.startswith("entries.") else p,
            "en": en,
            "cn": cn if isinstance(cn, str) else None,
        })


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def collect(repo):
    en_dir = os.path.join(repo, "compendium", "en")
    cn_dir = os.path.join(repo, "compendium", "cn")
    rows = []
    for fn in sorted(os.listdir(en_dir)):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        cp = os.path.join(cn_dir, fn)
        if not os.path.isfile(cp):
            continue
        sub = []
        walk(load(os.path.join(en_dir, fn)).get("entries", {}),
             load(cp).get("entries", {}), ["entries"], sub)
        for r in sub:
            r["pack"] = fn
        rows.extend(sub)
    return rows


def units(en_s: str, cn_s: str):
    eb, cb = blocks(en_s), blocks(cn_s)
    if not eb or not cb:
        return
    if len(eb) != len(cb):
        yield ("leaf", "0", " ".join(eb), " ".join(cb))
        return
    for bi, (e, c) in enumerate(zip(eb, cb)):
        es, cs = en_sentences(e), cn_sentences(c)
        if len(es) == len(cs) and len(es) > 1:
            for si, (ee, cc) in enumerate(zip(es, cs)):
                yield ("sent", f"{bi}.{si}", ee, cc)
        else:
            yield ("block", str(bi), e, c)


# ---------------------------------------------------------------- 施动者

ACTOR_EN = [
    ("GM", r"\b(?:the\s+)?(?:GM|Game\s*master|Gamemaster|Narrator)\b"),
    ("TARGET", r"\bthe\s+target\b|\bthe\s+defender\b"),
    ("ALLY", r"\ban?\s+all(?:y|ies)\b|\byour\s+all(?:y|ies)\b|\beach\s+ally\b"),
    ("PLAYER", r"\bplayers?\b|\bthe\s+party\b"),
    ("ENEMY", r"\bthe\s+enemy\b|\bthe\s+attacker\b|\bthe\s+foe\b"),
]
ACTOR_CN = [
    ("GM", r"GM|游戏主持人|游戏主持|主持人|叙事者|裁判"),
    ("TARGET", r"目标|防御者"),
    ("ALLY", r"盟友|同伴|友军"),
    ("PLAYER", r"玩家|队伍|小队|你们"),
    ("ENEMY", r"敌人|攻击者|该敌|敌方"),
]
ACTOR_EN_RX = [(k, re.compile(v, re.I)) for k, v in ACTOR_EN]
ACTOR_CN_RX = [(k, re.compile(v)) for k, v in ACTOR_CN]


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--mode", default="conflict",
                    choices=["conflict", "invented", "actor"])
    ap.add_argument("--out")
    ap.add_argument("--soft", action="store_true")
    # invented 模式的 PROHIB 方向被英文的语义否定形容词淹没（ineffective / -less /
    # irreparable → 中文正当地写「无法」），默认只看「中文凭空写出必须」这一档。
    ap.add_argument("--invented-tiers", default="OBLIG",
                    help="invented 模式看哪些中文档位，逗号分隔（OBLIG / PROHIB）")
    ap.add_argument("--min-chars", type=int, default=15)
    ap.add_argument("--max-chars", type=int, default=400)
    ap.add_argument("--show", type=int, default=25)
    a = ap.parse_args()

    rows = []
    for repo in a.repo:
        tag = os.path.basename(os.path.normpath(repo))
        for r in collect(repo):
            r["repo"] = tag
            rows.append(r)

    scale = {"leaves_total": len(rows),
             "leaves_with_cn": sum(1 for r in rows if r["cn"]),
             "en_chars": sum(len(r["en"]) for r in rows)}

    table = dict(HARD)
    if a.soft:
        table.update(SOFT)
    found, stats, seen = [], collections.Counter(), set()

    for r in rows:
        if not r["cn"]:
            continue
        for gran, idx, e, c in units(r["en"], r["cn"]):
            if not (a.min_chars <= len(e) <= a.max_chars):
                stats["skip_len"] += 1
                continue
            stats["units"] += 1
            et, resid = en_tiers(e)

            if a.mode == "invented":
                if et:
                    continue
                if NEG_CTX.search(e):
                    stats["gate_negation"] += 1
                    continue
                ct = cn_tiers(c)
                want = {t.strip() for t in a.invented_tiers.split(",") if t.strip()}
                hard = {k: v for k, v in ct.items() if k in want}
                if not hard:
                    continue
                if any(IMPLIES[k].search(e) for k in hard):
                    stats["gate_implied"] += 1
                    continue
                key = (e, c)
                if key in seen:
                    stats["dup_twin"] += 1
                    continue
                seen.add(key)
                found.append({
                    "repo": r["repo"], "pack": r["pack"], "path": r["path"],
                    "batch_path": r["batch_path"], "granularity": gran, "unit": idx,
                    "kind": "英文无情态→中文写出" + "/".join(sorted(hard)),
                    "severity": "一般", "en_tier": "-", "cn_tier": "/".join(sorted(hard)),
                    "en_hit": [], "cn_hit": [x for v in hard.values() for x in v],
                    "en": e, "cn": c})
                continue

            if a.mode == "actor":
                if not et:
                    continue
                # 取**紧贴情态词之前**的那个主语，而不是整句里的所有施动者。
                # 回测教训：`If multiple player characters …, the Gamemaster should …`
                # 整句里 GM 与 PLAYER 同时出现，用「整句只有一个施动者」会直接漏掉。
                mpos = None
                for _t, _rx in EN_RX.items():
                    mm = _rx.search(e)
                    if mm and (mpos is None or mm.start() < mpos):
                        mpos = mm.start()
                if mpos is None:
                    continue
                head = e[max(0, mpos - 45):mpos]
                ea = [(rx.search(head).start(), k)
                      for k, rx in ACTOR_EN_RX if rx.search(head)]
                if not ea:
                    continue
                act = max(ea)[1]          # 离情态词最近的那个
                ca = {k for k, rx in ACTOR_CN_RX if rx.search(c)}
                if act in ca or not ca:
                    continue
                key = (e, c)
                if key in seen:
                    stats["dup_twin"] += 1
                    continue
                seen.add(key)
                found.append({
                    "repo": r["repo"], "pack": r["pack"], "path": r["path"],
                    "batch_path": r["batch_path"], "granularity": gran, "unit": idx,
                    "kind": f"施动者 {act} → {'/'.join(sorted(ca))}", "severity": "一般",
                    "en_tier": "/".join(sorted(et)), "cn_tier": "-",
                    "en_hit": [x for v in et.values() for x in v], "cn_hit": [],
                    "en": e, "cn": c})
                continue

            # --- mode == conflict
            if RESTRICT_RX.search(e):
                stats["gate_restrict"] += 1
                continue
            if COND_INVERSION.match(e):
                stats["gate_cond_inversion"] += 1
                continue
            if len(et) != 1:
                stats["skip_multi_or_none"] += 1
                continue
            stats["units_single_tier"] += 1
            etier = next(iter(et))
            ct = cn_tiers(c)
            if not ct:
                stats["cn_no_marker"] += 1
                continue
            if COMPATIBLE[etier] & set(ct):
                stats["ok"] += 1
                continue
            if etier in ("OBLIG", "RECOM", "EPIST") and EPI_FRAME.search(resid):
                stats["gate_epistemic"] += 1
                continue
            for ctier in ct:
                key = (etier, ctier)
                if key not in table:
                    continue
                # 否定/负极性只解释中文的「无法 / 不得 / 无需」，与中文写成
                # 「可以 / 必须」毫无关系 —— 所以这道闸只对 PROHIB/NONOB 结论生效。
                # （回测：`must be affixed ... to an immobile object` 里的 immobile
                #  曾经把注入的 must->可以 整条吞掉。）
                if ctier in ("PROHIB", "NONOB") and NEG_CTX.search(resid):
                    stats["gate_negation"] += 1
                    continue   # 不能 break：同一单元的另一个中文档位可能才是真冲突
                if IMPLIES[ctier].search(resid):
                    stats["gate_implied"] += 1
                    continue
                dedup = (e, c, ctier)
                if dedup in seen:
                    stats["dup_twin"] += 1
                    break
                seen.add(dedup)
                label, sev = table[key]
                found.append({
                    "repo": r["repo"], "pack": r["pack"], "path": r["path"],
                    "batch_path": r["batch_path"], "granularity": gran, "unit": idx,
                    "kind": label, "severity": sev,
                    "en_tier": etier, "cn_tier": ctier,
                    "en_hit": et[etier], "cn_hit": ct[ctier],
                    "en": e, "cn": c})
                break

    payload = {"criterion": f"modal_strength/{a.mode}", "scale": scale,
               "stats": dict(stats), "hits": len(found), "findings": found}
    txt = json.dumps(payload, ensure_ascii=False, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"mode={a.mode} scale={scale}")
        print(f"hits={len(found)} -> {a.out}")
        print("stats=", dict(stats))
        for f_ in found[: a.show]:
            print("-" * 70)
            print(f_["kind"], "|", f_["pack"], "|", f_["path"])
            print("EN:", f_["en"][:300])
            print("CN:", f_["cn"][:300])
    else:
        print(txt)


if __name__ == "__main__":
    main()
