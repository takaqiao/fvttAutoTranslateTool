# -*- coding: utf-8 -*-
"""& 转义审计（收紧口径）：只报**中文侧引入的**转义差异。

  B1  该叶是 HTML 文本（含 < 标签），CN 里有裸 &，而 EN 同叶里没有裸 &
      -> 中文把 &amp; 写成了 &（HTML 里裸 & 后跟字母会被浏览器尝试当实体）
  B2  裸 & 出现在 enricher 标签 {…} 内部（标签会被塞进 HTML）
  B3  CN 用 &amp; 而 EN 该处用裸 &（多转义，会显示成字面 &amp;）—— 只在 HTML 叶里算
  B4  纯文本叶（name/label 等，不含 <）里 CN 写 &amp; 而 EN 写 & -> 会字面显示 "&amp;"
"""
import json, re, sys, collections
from pathlib import Path

ENTITY = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]{0,30}|#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6});")
ENRICHER_AMP = re.compile(r"&(?:amp;)?[A-Za-z]+\[")
LABEL = re.compile(r"(?:@[A-Za-z]+\[[^\[\]]*\]|&(?:amp;)?[A-Za-z]+\[[^\[\]]*\]|\[\[.*?\]\])\s*\{([^{}]*)\}", re.S)


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def bare(s):
    out = []
    for i, ch in enumerate(s):
        if ch != "&":
            continue
        if ENTITY.match(s[i:]) or ENRICHER_AMP.match(s[i:]):
            continue
        out.append(i)
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
            is_html = "<" in s or "<" in e
            bs, be = bare(s), bare(e)
            if is_html and bs and not be:
                counts["B1"] += 1
                rows.append(("B1", repo.name, f.name, p,
                             [s[max(0, i - 50):i + 50] for i in bs[:3]], e[:150]))
            for m in LABEL.finditer(s):
                if bare(m.group(1)):
                    counts["B2"] += 1
                    rows.append(("B2", repo.name, f.name, p, m.group(0)[:150], e[:150]))
                    break
            if is_html and s.count("&amp;") > e.count("&amp;") and len(be) > len(bs):
                counts["B3"] += 1
                rows.append(("B3", repo.name, f.name, p,
                             f"&amp; CN={s.count('&amp;')} EN={e.count('&amp;')}; 裸& CN={len(bs)} EN={len(be)}",
                             e[:150]))
            if not is_html and s.count("&amp;") > e.count("&amp;"):
                counts["B4"] += 1
                rows.append(("B4", repo.name, f.name, p, f"CN={s!r}", f"EN={e!r}"))

print(counts)
seen = collections.Counter()
for tag, rn, pack, p, det, ctx in rows:
    seen[tag] += 1
    if seen[tag] > 15:
        continue
    print("-" * 96)
    print(f"[{tag}] {rn} {pack} | {p}")
    print("    det:", det)
    print("    ctx:", str(ctx)[:200])
