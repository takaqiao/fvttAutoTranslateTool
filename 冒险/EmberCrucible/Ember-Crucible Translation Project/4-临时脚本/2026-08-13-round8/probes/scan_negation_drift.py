# -*- coding: utf-8 -*-
"""否定 / 条件反转判据 —— 英文侧的否定或条件从句在中文侧被丢掉（或凭空多出）。

为什么既有判据全盲
------------------
丢一个「不」字，规则的意思就反了，但：标记没变、class 没变、数字没变、
中文侧不缺键、没有外来文字、长度也正常。七轮既有判据一个都报不出来。

四层设计
--------
1) **块级 + 句级对齐**。本库 markup drift 判据已全绿，实测多块叶子里
   6792/6806 (99.8%) 英中 HTML 块数相同，所以可以把 <p>/<li>/<h*>/<td>
   切开后 1:1 对位；块内再按句号切，英中句数相同才切。
   **粒度是这个判据的命门**：整叶比对时，一页 4600 字、含 41 个「不」的规则页里
   把「生物不能穿过另一个生物的空间」改成「可以穿过」，判据完全看不见；
   句级对齐立刻报出来。回测里 4 个注入错误，块级只抓到 1 个，句级抓到 4 个。

2) **成对构式计数**，不是笼统的「否定数差值」。笼统差值假阳性极高
   （`is unable to see` -> 「目盲」一个否定字都没有，却是好译文）。按构式分对：
       unless / cannot / no longer / instead of / except / never / fails to /
       without / not，每对有自己的中文等价集合与权重。

3) **双闸**：
   a. 机制闸 —— 否定标记之后 WIN_EN 字符内必须出现机制词
      （伤害/检定/移动/动作/回合/距离/攻击/休息/复活…），
      排除叙事对白里的 "I don't know"。
   b. 预算闸 —— 该单元的英文**泛否定**计数必须严格大于中文**泛否定**计数。

4) **中文假朋友掩码**。中文泛否定用宽字集（不无没未非勿别禁免…）故意高估，
   但先抹掉「无论/不过/不仅/非常/无视」这类**不表否定**的词
   —— 回测里 Spellmute 的注入错误就是被句子里的「无论」白送一个否定额度挡掉的。

反方向（中文凭空多出否定）同样查，但英文侧的「否定语义白名单」开得很宽
（cannot/n't/indistinguishable/immobilize/beyond/rarely/prevent…），
因为英文用肯定词表达否定语义、中文用「无法」是**合法**的。全库反向 0 命中。

用法：
  python scan_negation_drift.py --repo <repoDir> [--repo <另一个>] --out <json>
  python scan_negation_drift.py --repo <repoDir> --mech-gate off --budget-gate off \
         --min-score 1 --granularity block          # 看召回上限 / 对比粒度
  python scan_negation_drift.py --repo <repoDir> --pack crucible.rules.json

配套：
  inject_negation_bugs.py  灵敏度回测（往临时副本注入 4 个已知否定错误）
  view_neg.py              把报告按条打印、高亮相关句子，便于人工核
  scan_negation_scope.py   否定作用域变体（1567 对 -> 81 报 -> 0 真，已排除）
"""
from __future__ import annotations
import argparse
import collections
import html
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKIP_KEYS = {"_id", "path", "_variants", "_when"}

# --------------------------------------------------------------------------
# 文本归一 + 块切分
# --------------------------------------------------------------------------
RE_ENRICHER_TARGET = re.compile(r"@\w+\[[^\]]*\]")   # @UUID[...] @Check[...] @Embed[...]
RE_ROLL = re.compile(r"\[\[.*?\]\]", re.S)            # [[/r 1d6]] [[/check ...]]
RE_BLOCK = re.compile(
    r"</?(?:p|li|h[1-6]|div|td|th|tr|blockquote|section|ul|ol|table|thead|tbody|"
    r"br|figcaption|caption|dt|dd|hr)\b[^>]*>", re.I)
RE_TAG = re.compile(r"<[^>]*>", re.S)
RE_WS = re.compile(r"\s+")


def _clean(t: str) -> str:
    t = RE_TAG.sub(" ", t).replace("{", " ").replace("}", " ")
    t = html.unescape(t).replace("’", "'").replace("‘", "'")
    return RE_WS.sub(" ", t).strip()


def blocks(s: str) -> list[str]:
    """切成块级文本；返回非空块列表。"""
    s = RE_ENRICHER_TARGET.sub(" ", s)
    s = RE_ROLL.sub(" ", s)
    return [t for t in (_clean(p) for p in RE_BLOCK.split(s)) if t]


# 句级再切分：块内英中句数相同时可以 1:1 对位，粒度比块更细
RE_SENT_EN = re.compile(r'(?<=[.!?;])\s+(?=[A-Z0-9("\'“])')
RE_SENT_CN = re.compile(r'(?<=[。！？；])\s*')


def _sents(e: str, c: str):
    se = [x.strip() for x in RE_SENT_EN.split(e) if x.strip()]
    sc = [x.strip() for x in RE_SENT_CN.split(c) if x.strip()]
    return (list(zip(se, sc)) if len(se) >= 2 and len(se) == len(sc) else None)


def units(en: str, cn: str, granularity: str = "sentence"):
    """对齐单元列表 [(标签, en, cn)]。

    block  -> 只切 HTML 块（块数不等时退回整叶）
    sentence -> 块内再按句号切；英中句数相同才切，否则保留整块
    """
    be, bc = blocks(en), blocks(cn)
    if len(be) >= 2 and len(be) == len(bc):
        pairs, mode = list(zip(be, bc)), "block"
    else:
        e, c = " ".join(be), " ".join(bc)
        if not e:
            return [], "leaf"
        pairs, mode = [(e, c)], "leaf"

    out = []
    for i, (e, c) in enumerate(pairs):
        ss = _sents(e, c) if granularity == "sentence" else None
        if ss:
            for j, (se, sc) in enumerate(ss):
                out.append((f"{mode}#{i}.s{j}", se, sc))
        else:
            out.append((f"{mode}#{i}", e, c))
    return out, mode


# --------------------------------------------------------------------------
# 机制词闸
# --------------------------------------------------------------------------
MECH_EN = re.compile(
    r"\b("
    r"damages?|checks?|saves?|saving|defenses?|defences?"
    r"|moves?|moved|movement|strides?|speed|distance|ranges?|spaces?|feet|foot"
    r"|actions?|turns?|rounds?|initiative|reactions?|combat|encounter"
    r"|attacks?|rolls?|hits?|miss(?:es)?|critical|crit"
    r"|spells?|talents?|abilit(?:y|ies)|skills?|effects?|damage"
    r"|bonus(?:es)?|penalt(?:y|ies)|resistances?|vulnerabilit\w*|immune|immunit\w*"
    r"|health|wounds?|morale|madness|focus|hit\s*points?|hp|dc|threshold"
    r"|targets?|creatures?|enem(?:y|ies)|all(?:y|ies)|tokens?|characters?|actors?"
    r"|gains?|lose|loses?|benefits?|appl(?:y|ies)|affects?|grants?"
    r"|conditions?|status(?:es)?|weapons?|armou?r|shields?|equip\w*|items?"
    r"|rank|tier|level|dice|die|d\d+|bane|boon|advantage|disadvantage"
    r"|uses?|used|require\w*|allow\w*|able|eligible|qualif\w*|permitted|counts?"
    r"|success|failure|succeed\w*|fail\w*|test|contest"
    r"|rest|rests|resting|reviv\w*|heal\w*|restor\w*|regain\w*|recover\w*"
    r"|cast|casts|casting|magic\w*|die|dies|dead|death|breathe|speak|see|hear"
    r"|stand|prone|engage\w*|disengage\w*|flank\w*|occupy|occupies|push\w*|pull\w*"
    r"|wall|walls|terrain|opportunit\w*|slot|slots|charge|charges|craft\w*|scroll\w*"
    r")\b", re.I)

# --------------------------------------------------------------------------
# 泛否定（预算闸）
# --------------------------------------------------------------------------
EN_NEG_ANY = re.compile(
    r"\b(?:unless|except|other\s+than|aside\s+from|apart\s+from|instead\s+of|"
    r"rather\s+than|in\s+place\s+of|without|never|no\s+longer|cannot|can\s*not|"
    r"can't|may\s+not|must\s+not|should\s+not|shall\s+not|will\s+not|would\s+not|"
    r"do\s+not|does\s+not|did\s+not|is\s+not|are\s+not|was\s+not|were\s+not|"
    r"be\s+not|has\s+not|have\s+not|had\s+not|\w+n't|unable|neither|nor|"
    r"none|nothing|nobody|no\s+one|fails?\s+to|failed\s+to|not|no(?=\s+\w))\b",
    re.I)
# 中文泛否定：**故意开宽**（含「不同/无论/非常」这类假朋友），高估中文否定量
CN_NEG_ANY = re.compile(r"[不无没未非勿别禁免缺拒]|除非|除了|以外|之外|否则")
# 先抹掉**确定不表否定**的假朋友，否则「无论」「不过」会白送中文一个否定额度，
# 把真的丢否定的句子挡在预算闸外（回测里 Spellmute 注入就是被「无论」挡掉的）。
CN_FALSE_FRIEND = re.compile(
    r"无论|不论|无视|无数|无尽|无穷|无比|无疑|无边|无垠|无限|无名|无形|无声|无息|"
    r"不过|不仅|不但|不只|不止|不断|不停|不时|不少|不禁|不由|不妨|不失为|不已|"
    r"不同|不一|不定|不外乎|非常|非但|莫非|无奈|无非|无妨|毫无疑问|"
    r"未来|未免|没什么|不错|不外|不日|不巧|不料")


def cn_neg_count(c: str) -> int:
    return len(CN_NEG_ANY.findall(CN_FALSE_FRIEND.sub("　", c)))

# --------------------------------------------------------------------------
# 成对构式：key -> (英文正则, 中文等价正则, 权重, 说明)
# --------------------------------------------------------------------------
PAIRS = {
    "unless": (
        re.compile(r"\bunless\b", re.I),
        re.compile(r"除非|否则|若非|不然|要不然|只有|仅当|仅在|仅限|才可|才能|才会|除了|以外|之外"),
        4, "unless 条件从句"),
    "cannot": (
        re.compile(r"\b(?:can\s*not|cannot|can't|may\s+not|must\s+not|"
                   r"(?:is|are|was|were|be|being|been)\s+unable|unable\s+to)\b", re.I),
        re.compile(r"无法|不能|不可|不得|不会|不许|禁止|没办法|无从|未能|尚未|无需|不必|不再|不受|没有|不"),
        3, "cannot / may not / unable"),
    "no_longer": (
        re.compile(r"\bno\s+longer\b", re.I),
        re.compile(r"不再|不复|失去|结束|移除|解除|退出|终止|停止|恢复"),
        3, "no longer"),
    "instead_of": (
        re.compile(r"\b(?:instead\s+of|rather\s+than|in\s+place\s+of)\b", re.I),
        re.compile(r"而不是|而非|不是|代替|取代|替代|改为|替换|而不|以外|之外|来代"),
        3, "instead of / rather than"),
    "except": (
        re.compile(r"\b(?:except(?:\s+for|\s+that)?|other\s+than|aside\s+from|"
                   r"apart\s+from)\b", re.I),
        re.compile(r"除|以外|之外|例外|另外|唯有|只有|仅"),
        3, "except / other than"),
    "never": (
        re.compile(r"\bnever\b", re.I),
        re.compile(r"从不|绝不|永不|从未|绝无|永远不|从来不|从来没|决不|不会|不曾|无一"),
        3, "never"),
    # `who fail to avoid the attack` 这类**限定性**从句：丢掉就等于把条件伤害
    # 写成无条件伤害，和 unless 一样危险，所以从 plain_not 里单拎出来加权。
    "fails_to": (
        re.compile(r"\bfail(?:s|ed|ing)?\s+to\b", re.I),
        re.compile(r"失败|未能|没能|未通过|不成功|未成功|失手|没有.{0,4}成功|无法"),
        3, "fails to（限定性从句）"),
    "without": (
        re.compile(r"\bwithout\b", re.I),
        re.compile(r"[无没未不免缺]|以外|之外"),
        2, "without"),
    "plain_not": (
        re.compile(r"\b(?:do\s+not|does\s+not|did\s+not|will\s+not|would\s+not|"
                   r"should\s+not|shall\s+not|is\s+not|are\s+not|was\s+not|"
                   r"were\s+not|be\s+not|has\s+not|have\s+not|had\s+not|\w+n't|"
                   r"neither|nor|none|nothing|nobody|no\s+one|fails?\s+to|"
                   r"failed\s+to|not|no(?=\s+\w))\b", re.I),
        re.compile(r"[不无没未非勿别禁免缺拒]"),
        1, "not / no / neither / fails to"),
}

# --------------------------------------------------------------------------
# 反方向：中文凭空多出否定
# --------------------------------------------------------------------------
CN_STRONG_NEG = re.compile(r"无法|不能|不可以|不得|不再|除非|不允许|禁止|不会")
# 英文侧「本来就带否定语义 / 限制语义」的宽白名单 —— 命中任何一个就不报
EN_NEG_SEMANTIC = re.compile(
    r"(?:\w+n't|\bcannot\b|\bcan\s*not\b)|"
    r"\b(?:un\w{3,}|in(?:distinguish|capab|abil|access|effect|visib|vulner|animat|"
    r"escap|evitab|ert|sufficient|complet|valid|frequent|tang|explic|"
    r"extricab|advertent|apt|different)\w*|im(?:possib|mobil|mune|permeab|passab|"
    r"perceptib|penetrab|movab|mortal|material)\w*|ir(?:resist|revers|regular)\w*|"
    r"dis(?:abl|advantag|allow|appear|arm|solv|rupt|regard|miss|lodg)\w*|non\w*|"
    r"beyond|rarely|seldom|hardly|barely|only|merely|lack\w*|prevent\w*|deny|"
    r"denies|denied|block\w*|stop\w*|halt\w*|cease\w*|refus\w*|resist\w*|avoid\w*|"
    r"escap\w*|ignor\w*|forbid\w*|prohibit\w*|restrict\w*|forfeit\w*|"
    r"motionless|silent|empt(?:y|ied)|void|absent|free\s+of|clear\s+of|safe\s+from|"
    r"protect\w*|shield\w*|reduce\w*|remove\w*|lose|loses|lost|losing|end(?:s|ed|ing)?|"
    r"expire\w*|until|before|instead|rather|besides|except\w*|unless|other\s+than|"
    r"require\w*|need\w*|must|limit\w*|cap\w*|maximum|at\s+most|no\w*|not|never|"
    r"fail\w*|difficult|impass\w*|too\s+\w+|less|fewer|minus|penalt\w*|"
    r"immun\w*|nullif\w*|negat\w*|cancel\w*|suppress\w*|deaf|blind|mute|"
    r"still|frozen|paralyz\w*|petrif\w*|helpless|trapped|stuck|bound|held)\b",
    re.I)


def gated_hits(text: str, rx: re.Pattern, gate: bool) -> int:
    if not gate:
        return len(rx.findall(text))
    n = 0
    for m in rx.finditer(text):
        if MECH_EN.search(text[m.end(): m.end() + WIN_EN]):
            n += 1
    return n


WIN_EN = 70


# --------------------------------------------------------------------------
def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                 path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else None))


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def clip(s, n):
    return s if len(s) <= n else s[:n] + f"…<+{len(s)-n}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--pack", default="all")
    ap.add_argument("--out")
    ap.add_argument("--mech-gate", choices=["on", "off"], default="on")
    ap.add_argument("--budget-gate", choices=["on", "off"], default="on")
    ap.add_argument("--reverse", choices=["on", "off"], default="on")
    ap.add_argument("--granularity", choices=["block", "sentence"],
                    default="sentence", help="句级对齐比块级细，灵敏度高得多")
    ap.add_argument("--min-score", type=int, default=3)
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--ctx", type=int, default=900)
    a = ap.parse_args()

    mg = a.mech_gate == "on"
    bg = a.budget_gate == "on"
    fwd, rev = [], []
    st = collections.Counter()
    per_pair = collections.Counter()

    for repo in a.repo:
        en_dir = os.path.join(repo, "compendium", "en")
        cn_dir = os.path.join(repo, "compendium", "cn")
        if not os.path.isdir(en_dir):
            print(f"!! 没有 {en_dir}")
            continue
        packs = (sorted(f for f in os.listdir(en_dir)
                        if f.endswith(".json") and not f.startswith("_"))
                 if a.pack == "all" else [x.strip() for x in a.pack.split(",")])
        for pack in packs:
            ep, cp = os.path.join(en_dir, pack), os.path.join(cn_dir, pack)
            if not (os.path.isfile(ep) and os.path.isfile(cp)):
                continue
            rows = []
            walk(load(ep).get("entries", {}), load(cp).get("entries", {}),
                 ["entries"], rows)
            for p, en, cn in rows:
                st["叶子总数"] += 1
                if not cn:
                    st["中文缺失（跳过）"] += 1
                    continue
                us, mode = units(en, cn, a.granularity)
                if not us:
                    continue
                st["参与比对叶"] += 1
                st["英文字符"] += len(en)
                st[f"对齐模式-{mode}"] += 1
                st["对齐单元数"] += len(us)

                for idx, e, c in us:
                    if not e:
                        continue
                    # ---------- 正向 ----------
                    gaps, score = {}, 0
                    for key, (rx_en, rx_cn, w, desc) in PAIRS.items():
                        ne = gated_hits(e, rx_en, mg)
                        if ne == 0:
                            continue
                        nc = len(rx_cn.findall(c))
                        if ne > nc:
                            gaps[key] = {"en_n": ne, "cn_n": nc,
                                         "gap": ne - nc, "desc": desc}
                            score += (ne - nc) * w
                    if gaps:
                        en_any = gated_hits(e, EN_NEG_ANY, mg)
                        cn_any = cn_neg_count(c)
                        if bg and en_any <= cn_any:
                            st["被预算闸拦下"] += 1
                        elif score >= a.min_score:
                            for k, v in gaps.items():
                                per_pair[k] += v["gap"]
                            st["**正向命中**"] += 1
                            fwd.append({
                                "repo": os.path.basename(repo), "pack": pack,
                                "path": p, "batch_path": p[len("entries."):]
                                if p.startswith("entries.") else p,
                                "unit": idx, "score": score,
                                "en_any": en_any, "cn_any": cn_any,
                                "gaps": gaps,
                                "en": clip(e, a.ctx), "cn": clip(c, a.ctx),
                            })
                        else:
                            st["分数不足"] += 1

                    # ---------- 反向 ----------
                    if a.reverse == "on" and len(e) >= 40:
                        ncs = CN_STRONG_NEG.findall(c)
                        if ncs and not EN_NEG_SEMANTIC.search(e):
                            st["**反向命中**"] += 1
                            rev.append({
                                "repo": os.path.basename(repo), "pack": pack,
                                "path": p, "batch_path": p[len("entries."):]
                                if p.startswith("entries.") else p,
                                "unit": idx,
                                "gaps": {"cn_strong_neg": {"hits": ncs}},
                                "en": clip(e, a.ctx), "cn": clip(c, a.ctx),
                            })

    fwd.sort(key=lambda f: (-f["score"], f["pack"], f["path"]))
    rev.sort(key=lambda f: (f["pack"], f["path"]))

    print("扫描规模：")
    for k in ("叶子总数", "参与比对叶", "对齐单元数", "对齐模式-block",
              "对齐模式-leaf", "中文缺失（跳过）", "英文字符"):
        print(f"  {k:16s} {st[k]}")
    print(f"\n闸门：机制闸={'on' if mg else 'off'} 预算闸={'on' if bg else 'off'} "
          f"阈值={a.min_score}")
    print(f"  被预算闸拦下 {st['被预算闸拦下']}   分数不足 {st['分数不足']}")
    print(f"\n正向命中 {len(fwd)} 条")
    for k, v in per_pair.most_common():
        print(f"    {k:12s} 缺口累计 {v}")
    print(f"反向命中 {len(rev)} 条")

    print("\n—— 正向 TOP ——")
    for f in fwd[: a.show]:
        keys = ",".join(f"{k}x{v['gap']}" for k, v in f["gaps"].items())
        print(f"  [{f['score']:>2}] {f['pack'][:24]:26s} {f['path'][-56:]:58s} "
              f"{f['unit']:9s} {keys}")
    print("\n—— 反向 TOP ——")
    for f in rev[: a.show]:
        print(f"       {f['pack'][:24]:26s} {f['path'][-56:]:58s} {f['unit']:9s} "
              f"{''.join(f['gaps']['cn_strong_neg']['hits'][:4])}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"stats": dict(st), "per_pair": dict(per_pair),
                       "forward": fwd, "reverse": rev}, fh,
                      ensure_ascii=False, indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
