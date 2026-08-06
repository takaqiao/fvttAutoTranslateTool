"""For every prepared work dir, attach the orphan CN pages of the same journal
that fingerprint-match an untranslated EN page.

Those are translations of a page upstream later renamed. They are usually 90%
usable and 100% dangerous to trust: they were written against an older English,
so the agent must diff them paragraph by paragraph rather than paste them.
"""
import json, os, re, sys
from collections import Counter

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parallel")
PACK = "ember.crucible-adventure.json"
CJK = re.compile(r"[一-鿿]")
MARKUP = re.compile(r"@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]")
TAG = re.compile(r"<\s*(/?)([a-zA-Z][a-zA-Z0-9]*)")

en = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "en", PACK), encoding="utf-8"))
cn = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "cn", PACK), encoding="utf-8"))
EJ = en["entries"]["Ember Early Access"]["journals"]
CJ = cn["entries"]["Ember Early Access"]["journals"]


def fp(s):
    return Counter(MARKUP.findall(s)) + Counter(f"<{a}{b.lower()}" for a, b in TAG.findall(s))


def sim(a, b):
    u = sum((a | b).values())
    return sum((a & b).values()) / u if u else 0.0


total = 0
for m in json.load(open(os.path.join(ROOT, sys.argv[1]), encoding="utf-8")):
    jn, d = m["unit"], m["dir"]
    ej, cj = EJ.get(jn), CJ.get(jn, {})
    if not ej:
        continue
    en_pages, cn_pages = ej.get("pages") or {}, cj.get("pages") or {}
    open_pages = {pn: p.get("text", "") for pn, p in en_pages.items()
                  if not CJK.search((cn_pages.get(pn) or {}).get("text") or "")}
    found = {}
    for pn, p in cn_pages.items():
        if pn in en_pages:
            continue
        text = p.get("text") or ""
        if not CJK.search(text):
            continue
        best, score = None, 0.0
        for opn, otext in open_pages.items():
            s = sim(fp(text), fp(otext))
            if s > score:
                best, score = opn, s
        # 多个孤儿可能都指向同一个英文页，只留匹配度最高的那个
        if best and score >= 0.4 and score > found.get(best, {}).get("match", 0):
            found[best] = {"orphan_page_name": pn, "match": round(score, 2),
                           "orphan_cn_name": p.get("name"), "orphan_cn_text": text}
    if found:
        json.dump(found, open(os.path.join(d, "orphan_reference.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        total += len(found)
        print(f"{jn:<28} {len(found)} 页可参考: " +
              ", ".join(f"{k}<-{v['orphan_page_name']}({v['match']})" for k, v in found.items())[:110])

print(f"\n共 {total} 页附上了旧路径译文")
