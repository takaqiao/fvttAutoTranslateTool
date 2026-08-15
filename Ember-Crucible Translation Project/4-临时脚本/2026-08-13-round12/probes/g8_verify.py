# -*- coding: utf-8 -*-
"""Re-run G8 scans against the post-batch CN values."""
import json, os, sys, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g8_tools import load, blocks, strip
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
B = json.load(open(os.path.join(ROOT, "4-临时脚本", "2026-08-13-round12", "batches",
                                "G8.1.ember.adventure.json"), encoding="utf-8"))
PRE = "Ember Early Access.journals.Deities."
rows = load()
for r in rows:
    nk = PRE + r["k"]
    if nk in B:
        r["cn"] = B[nk]

TAG = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)")
UUID = re.compile(r"@UUID\[([^\]]+)\]")
CJK = re.compile(r"[\u4e00-\u9fff]")

bad = 0
for r in rows:
    en, cn = r["en"], r["cn"] or ""
    eb, cb = blocks(en), blocks(cn)
    if len(eb) != len(cb):
        print("BLOCKCNT", r["k"], len(eb), len(cb)); bad += 1
    et = collections.Counter(m.group(0) for m in TAG.finditer(en))
    ct = collections.Counter(m.group(0) for m in TAG.finditer(cn))
    if et != ct:
        print("TAG", r["k"], {t: (et[t], ct[t]) for t in set(et) | set(ct) if et[t] != ct[t]}); bad += 1
    if collections.Counter(UUID.findall(en)) != collections.Counter(UUID.findall(cn)):
        print("UUID", r["k"]); bad += 1
print("structural problems:", bad)

print("\n-- residual untranslated blocks --")
for r in rows:
    eb, cb = blocks(r["en"]), blocks(r["cn"] or "")
    if len(eb) != len(cb):
        continue
    for i, (e, c) in enumerate(zip(eb, cb)):
        se, sc = strip(e), strip(c)
        if se and not CJK.search(sc) and re.search(r"[A-Za-z]{2}", se):
            print(" ", r["k"], i, se[:60], "||", sc[:60])

print("\n-- residual same-EN splits --")
m = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    eb, cb = blocks(r["en"]), blocks(r["cn"] or "")
    if not eb:
        eb, cb = [r["en"]], [r["cn"] or ""]
    if len(eb) != len(cb):
        continue
    for i, (e, c) in enumerate(zip(eb, cb)):
        se, sc = strip(e), strip(c)
        if se:
            m[se][sc].append(f"{r['k']}#{i}")
n = 0
for se in m:
    if len(m[se]) > 1:
        n += 1
        print(f"  EN: {se[:80]}")
        for sc in m[se]:
            print(f"     -> {sc[:80]}  {m[se][sc][:3]}")
print("splits:", n)

print("\n-- residual term counts --")
c = collections.Counter()
for r in rows:
    t = r["cn"] or ""
    for k in ["十九神", "影响与追随者", "神祇般的存在", "谕令", "莫里亚", "凡俗性与毁灭"]:
        c[k] += t.count(k)
print(dict(c))
