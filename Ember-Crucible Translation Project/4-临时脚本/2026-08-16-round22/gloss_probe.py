# -*- coding: utf-8 -*-
"""Round-22: for each of the 212 unique arrangement labels, strip the structural
suffix/prefix words and look the remaining *core* phrase up in glossary_ec.json
(and its pending/disputes siblings). Also try every contiguous multi-word
sub-phrase of the label, longest first, so `Ordain Docks Day` finds `Ordain Docks`
if that exists and otherwise falls back to `Ordain`.

Anti-空转: prints glossary size and, per label, how many sub-phrases were tried.
A glossary of size 0 => exit 2.
"""
import json, io, os, re, sys

sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
G = json.load(io.open(os.path.join(B, r"5-其他内容\glossary\glossary_ec.json"), encoding="utf-8"))
extra = {}
for fn in ["glossary_ec.pending.json"]:
    p = os.path.join(B, r"5-其他内容\glossary", fn)
    if os.path.exists(p):
        try:
            j = json.load(io.open(p, encoding="utf-8"))
            if isinstance(j, dict):
                extra[fn] = j
        except Exception as e:
            print("skip", fn, e)
print(f"glossary_ec entries={len(G)}; extra={[ (k,len(v)) for k,v in extra.items() ]}")
if not G:
    print("NO-GLOSSARY")
    sys.exit(2)

arr = json.load(io.open(os.path.join(B, r"4-临时脚本\2026-08-16-round22\soundscapes_r22.json"),
                        encoding="utf-8"))["arrLabels"]
print(f"labels={len(arr)}")

tried_total = 0
for lab in arr:
    toks = lab.split()
    found = []
    n = 0
    for size in range(len(toks), 0, -1):
        for i in range(0, len(toks) - size + 1):
            ph = " ".join(toks[i:i + size])
            ph = ph.strip("-").strip()
            if not ph:
                continue
            n += 1
            if ph in G:
                found.append((ph, G[ph], "ec"))
            for fn, j in extra.items():
                if ph in j:
                    found.append((ph, j[ph], fn))
        if found:
            break
    tried_total += n
    if found:
        print(f"{lab}  ||  " + " ; ".join(f"[{p}]={v}<{src}>" for p, v, src in found))
    else:
        print(f"{lab}  ||  --none-- (tried {n} sub-phrases)")
print(f"\ntotal sub-phrases tried={tried_total}")
