# -*- coding: utf-8 -*-
"""Block-level gender mismatch: EN block uses only she/her -> CN block uses bare 他."""
import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g8_tools import load, blocks, strip, pagename
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rows = load()
# bare 他 = not 其他 / 他们 / 他人 / 其它
BARE_TA = re.compile(r"(?<!其)他(?!们|人|方|处|日|乡)")
BARE_SHE = re.compile(r"她(?!们)")

for r in rows:
    eb, cb = blocks(r["en"]), blocks(r["cn"] or "")
    if not eb:
        eb, cb = [r["en"]], [r["cn"] or ""]
    if len(eb) != len(cb):
        continue
    for i, (e, c) in enumerate(zip(eb, cb)):
        et = " " + strip(e).lower() + " "
        ct = strip(c)
        m = len(re.findall(r"\b(?:he|him|his|himself)\b", et))
        f = len(re.findall(r"\b(?:she|her|hers|herself)\b", et))
        ta = len(BARE_TA.findall(ct))
        she = len(BARE_SHE.findall(ct))
        bad = None
        if f and not m and ta:
            bad = f"EN female-only(F={f}) but CN has 他 x{ta}"
        elif m and not f and she:
            bad = f"EN male-only(M={m}) but CN has 她 x{she}"
        if bad:
            print(f"\n[{r['k']}] blk{i}: {bad}")
            print("  E:", strip(e)[:400])
            print("  C:", strip(c)[:400])
