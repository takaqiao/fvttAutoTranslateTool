# -*- coding: utf-8 -*-
"""& 的转义审计。分类比裸计数细。

  A1  CN 有「裸 &」（不是合法实体、也不是 &Reference[ 这种 enricher 前缀）
      -> 浏览器会容错显示成 &，但若后面正好跟 amp; lt; 之类就会被吃掉；且 innerHTML 往返不稳定
  A2  EN 用 &Xxx[ / CN 用 &amp;Xxx[（或反过来）—— enricher 前缀的转义层数与英文侧不一致
  A3  CN 出现「双重转义」&amp;amp; / &amp;lt; / &amp;nbsp; 之类
  A4  CN 出现英文侧没有的 &nbsp;（不间断空格，中文排版里会变成多余空隙）
  A5  CN 里的 &amp; 数量少于 EN，且该叶英文的 & 两侧都是**专名**（疑似漏掉了并列号）
"""
import json, re, sys, collections
from pathlib import Path

ENTITY = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]{0,30}|#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6});")
ENRICHER_AMP = re.compile(r"&(?:amp;)?[A-Za-z]+\[")
DOUBLE = re.compile(r"&amp;(?:amp;|lt;|gt;|quot;|nbsp;|#\d+;|[a-zA-Z]{2,10};)")


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def bare_amps(s):
    out = []
    for i, ch in enumerate(s):
        if ch != "&":
            continue
        rest = s[i:]
        if ENTITY.match(rest) or ENRICHER_AMP.match(rest):
            continue
        out.append(s[max(0, i - 40):i + 40])
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
            if "&" not in s and "&" not in e:
                continue
            ba = bare_amps(s)
            if ba:
                counts["A1"] += 1
                rows.append(("A1", repo.name, f.name, p, ba[:3], e[:200]))
            cn_raw = len(re.findall(r"&[A-Za-z]+\[", s)) - len(re.findall(r"&amp;[A-Za-z]+\[", s))
            en_raw = len(re.findall(r"&[A-Za-z]+\[", e)) - len(re.findall(r"&amp;[A-Za-z]+\[", e))
            cn_esc = len(re.findall(r"&amp;[A-Za-z]+\[", s))
            en_esc = len(re.findall(r"&amp;[A-Za-z]+\[", e))
            if (cn_raw, cn_esc) != (en_raw, en_esc):
                counts["A2"] += 1
                rows.append(("A2", repo.name, f.name, p,
                             f"enricher& 裸/转义 CN=({cn_raw},{cn_esc}) EN=({en_raw},{en_esc})", e[:200]))
            d = DOUBLE.findall(s)
            de = DOUBLE.findall(e)
            if len(d) > len(de):
                counts["A3"] += 1
                rows.append(("A3", repo.name, f.name, p, f"双重转义 CN={d[:4]} EN={de[:4]}", s[:200]))
            if s.count("&nbsp;") > e.count("&nbsp;"):
                counts["A4"] += 1
                rows.append(("A4", repo.name, f.name, p,
                             f"&nbsp; CN={s.count('&nbsp;')} EN={e.count('&nbsp;')}", s[:200]))

print(counts)
seen = collections.Counter()
for tag, rn, pack, p, det, ctx in rows:
    seen[tag] += 1
    if seen[tag] > 12:
        continue
    print("-" * 96)
    print(f"[{tag}] {rn} {pack} | {p}")
    print("   ", det)
    print("    ctx:", ctx[:200])
