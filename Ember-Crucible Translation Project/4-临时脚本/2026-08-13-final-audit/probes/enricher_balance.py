# -*- coding: utf-8 -*-
"""enricher 括号配对：只看「中文侧自己不平衡 / 或与英文侧的不平衡方向不同」的叶。

口径：
  Q1  CN 的 [ ] 数不相等，而 EN 相等        -> 中文把方括号弄丢/弄多了
  Q2  CN 的 { } 数不相等，而 EN 相等        -> 中文把花括号弄丢/弄多了
  Q3  两侧都不相等（上游本来就有裸方括号，如 "[名字]"）  -> 记数不报
  Q4  逐个 @Xxx[...]{...} 解析：CN 里出现「@Xxx[ 之后 200 字内没有 ]」的残缺 enricher
  Q5  CN 的 {..} 个数 > EN，且多出来的花括号不紧跟在 enricher 之后 -> 疑似正文里出现裸花括号
"""
import json, re, sys, collections
from pathlib import Path

ENRICHER_OPEN = re.compile(r"(@[A-Za-z]+|&(?:amp;)?[A-Za-z]+)\[")
FULL = re.compile(r"(@[A-Za-z]+|&(?:amp;)?[A-Za-z]+)\[[^\[\]]*\](\{[^{}]*\})?")
ROLL_OPEN = re.compile(r"\[\[")
ROLL_FULL = re.compile(r"\[\[.*?\]\]", re.S)
LABEL_AFTER = re.compile(r"(?:@[A-Za-z]+\[[^\[\]]*\]|&(?:amp;)?[A-Za-z]+\[[^\[\]]*\]|\[\[.*?\]\])\s*\{[^{}]*\}", re.S)


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def stray_braces(s):
    """抹掉所有合法 enricher 标签后仍剩下的花括号。"""
    t = LABEL_AFTER.sub("~", s)
    t = ROLL_FULL.sub("~", t)
    t = FULL.sub("~", t)
    return [m.group(0) for m in re.finditer(r"\{[^{}]{0,60}\}?|\}", t)]


def broken_enrichers(s):
    """@Xxx[ 后面 300 字内找不到 ] 的。"""
    out = []
    for m in ENRICHER_OPEN.finditer(s):
        tail = s[m.end():m.end() + 300]
        if "]" not in tail:
            out.append(s[m.start():m.start() + 120])
    # [[ 后面找不到 ]]
    for m in ROLL_OPEN.finditer(s):
        tail = s[m.end():m.end() + 300]
        if "]]" not in tail:
            out.append(s[m.start():m.start() + 120])
    return out


counts = collections.Counter()
rows = []
for repo in sys.argv[1:]:
    repo = Path(repo)
    for f in sorted((repo / "compendium" / "en").glob("*.json")):
        cnp = repo / "compendium" / "cn" / f.name
        if not cnp.exists():
            continue
        en = dict(leaves(json.loads(f.read_text(encoding="utf-8-sig"))))
        cn = dict(leaves(json.loads(cnp.read_text(encoding="utf-8-sig"))))
        for p, s in cn.items():
            e = en.get(p, "")
            cb, ce = s.count("["), s.count("]")
            eb, ee = e.count("["), e.count("]")
            cc, cd = s.count("{"), s.count("}")
            ec, ed = e.count("{"), e.count("}")
            tag = None
            if cb != ce and eb == ee:
                tag = "Q1"
            elif cc != cd and ec == ed:
                tag = "Q2"
            elif (cb != ce) and (eb != ee):
                counts["Q3"] += 1
            if tag:
                counts[tag] += 1
                rows.append((tag, repo.name, f.name, p, f"CN[{cb}/{ce}]{{{cc}/{cd}}} EN[{eb}/{ee}]{{{ec}/{ed}}}", s, e))
            br = broken_enrichers(s)
            ebr = broken_enrichers(e)
            if br and len(br) > len(ebr):
                counts["Q4"] += 1
                rows.append(("Q4", repo.name, f.name, p, f"残缺 enricher {br[:3]}", s, e))
            sb, seb = stray_braces(s), stray_braces(e)
            if len(sb) > len(seb):
                counts["Q5"] += 1
                rows.append(("Q5", repo.name, f.name, p, f"多余裸花括号 CN={sb[:4]} EN={seb[:4]}", s, e))

print(counts)
for tag, rn, pack, p, det, s, e in rows:
    print("-" * 96)
    print(f"[{tag}] {rn} {pack} | {p}")
    print("   ", det)
    if tag in ("Q1", "Q2"):
        # 定位不平衡的位置
        for side, txt in (("EN", e), ("CN", s)):
            depth = 0
            for i, ch in enumerate(txt):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth < 0:
                        print(f"    {side} 多出的 ] @{i}: ...{txt[max(0,i-90):i+40]}...")
                        depth = 0
            if depth > 0:
                print(f"    {side} 有 {depth} 个未闭合的 [")
    else:
        print("    EN:", e[:250])
        print("    CN:", s[:250])
