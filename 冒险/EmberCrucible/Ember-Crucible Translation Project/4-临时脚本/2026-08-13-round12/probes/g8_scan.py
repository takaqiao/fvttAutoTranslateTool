# -*- coding: utf-8 -*-
import re, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g8_tools import load, blocks, strip, pagename

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rows = load()

TAG = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)")
UUID = re.compile(r"@UUID\[[^\]]+\](?:\{[^}]*\})?")
IDAT = re.compile(r'\bid="([^"]*)"')
NUM = re.compile(r"\b\d+(?:\.\d+)?\b")

print("=== A. structural ===")
for r in rows:
    en, cn = r["en"], r["cn"] or ""
    k = r["k"]
    eb, cb = blocks(en), blocks(cn)
    msgs = []
    if len(eb) != len(cb):
        msgs.append(f"BLOCKCNT en={len(eb)} cn={len(cb)}")
    # tag multiset
    et = collections.Counter(m.group(0) for m in TAG.finditer(en))
    ct = collections.Counter(m.group(0) for m in TAG.finditer(cn))
    d = {t: (et[t], ct[t]) for t in set(et) | set(ct) if et[t] != ct[t]}
    if d:
        msgs.append(f"TAG {d}")
    # id attributes
    eids = collections.Counter(IDAT.findall(en))
    cids = collections.Counter(IDAT.findall(cn))
    if eids != cids:
        msgs.append(f"ID en={sorted(eids-cids)} cn={sorted(cids-eids)}")
    # uuid
    eu = collections.Counter(UUID.findall(en))
    cu = collections.Counter(UUID.findall(cn))
    # compare target part only
    def tgt(s):
        return collections.Counter(re.match(r"@UUID\[([^\]]+)\]", x).group(1) for x in s)
    if tgt(eu) != tgt(cu):
        a, b = tgt(eu), tgt(cu)
        msgs.append(f"UUID en-only={sorted((a-b).elements())} cn-only={sorted((b-a).elements())}")
    # numbers
    en_n = collections.Counter(NUM.findall(strip(en)))
    cn_n = collections.Counter(NUM.findall(strip(cn)))
    if en_n != cn_n:
        msgs.append(f"NUM en-only={sorted((en_n-cn_n).elements())} cn-only={sorted((cn_n-en_n).elements())}")
    if msgs:
        print(f"[{k}] " + " | ".join(msgs))

print()
print("=== B. per-block strong/em position ===")
for r in rows:
    eb, cb = blocks(r["en"]), blocks(r["cn"] or "")
    if len(eb) != len(cb):
        continue
    for i, (e, c) in enumerate(zip(eb, cb)):
        for tag in ("strong", "em", "b", "i", "a", "span"):
            ne = len(re.findall(rf"<{tag}\b", e, re.I))
            nc = len(re.findall(rf"<{tag}\b", c, re.I))
            if ne != nc:
                print(f"[{r['k']}] blk{i} <{tag}> en={ne} cn={nc}")
                print("   E:", strip(e)[:220])
                print("   C:", strip(c)[:220])

print()
print("=== C. gender per page ===")
pg = collections.defaultdict(lambda: {"en": collections.Counter(), "cn": collections.Counter()})
for r in rows:
    p, f = pagename(r["k"])
    if not p:
        continue
    e = " " + strip(r["en"]).lower() + " "
    c = strip(r["cn"] or "")
    for w in ("he", "him", "his", "himself"):
        pg[p]["en"][ "M" ] += len(re.findall(rf"\b{w}\b", e))
    for w in ("she", "her", "hers", "herself"):
        pg[p]["en"][ "F" ] += len(re.findall(rf"\b{w}\b", e))
    for w in ("they", "them", "their", "themselves"):
        pg[p]["en"][ "N" ] += len(re.findall(rf"\b{w}\b", e))
    pg[p]["en"]["god"] += len(re.findall(r"\bgod\b", e))
    pg[p]["en"]["goddess"] += len(re.findall(r"\bgoddess\b", e))
    pg[p]["cn"]["他"] += c.count("他")
    pg[p]["cn"]["她"] += c.count("她")
    pg[p]["cn"]["它"] += c.count("它")
    pg[p]["cn"]["祂"] += c.count("祂")
    pg[p]["cn"]["女神"] += c.count("女神")
for p in sorted(pg):
    e, c = pg[p]["en"], pg[p]["cn"]
    print(f"{p:<32} EN M={e['M']:<3} F={e['F']:<3} N={e['N']:<3} god={e['god']:<2} goddess={e['goddess']:<2} "
          f"|| CN 他={c['他']:<3} 她={c['她']:<3} 它={c['它']:<3} 祂={c['祂']:<3} 女神={c['女神']}")
