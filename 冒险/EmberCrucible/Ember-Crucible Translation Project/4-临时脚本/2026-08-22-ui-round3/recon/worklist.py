# -*- coding: utf-8 -*-
import io, json, os, re, collections
EMBER = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember"
OUT = os.path.dirname(os.path.abspath(__file__))
src = io.open(os.path.join(EMBER, "scripts/ember.mjs"), encoding="utf-8").read()
u = json.load(io.open(os.path.join(OUT, "parts_universe2.json"), encoding="utf-8"))
seg2l = u["seg2layers"]
decl = {m.group(1).split("/")[-1] for m in re.finditer(r'\bid:\s*"([A-Za-z0-9][A-Za-z0-9_./-]*)"', src)}
legbase = {m.group(1) for m in re.finditer(r'makeLegPoseParts\(\s*"([^"]+)"', src)}
POSES = ["Backward","Forward","Neutral","Sitting"]
lit = decl | legbase
RE = re.compile(r'(?<!^)([A-Z1-9])'); disp = lambda s: RE.sub(r' \1', s)

T1 = {"ears","mane","marks","scales","teeth","mouth","beard","fur","jaw","eyebrows","tail","collar","horns","eyes","face","hair","head","hood"}
T2 = {"hand","etherealHand","arm","etherealArm","forearm","etherealForearm","leg","legs","foot","torso"}

def tier(ls):
    if set(ls) & T1: return "T1"
    if set(ls) & T2: return "T2"
    return "REST"

rows = collections.defaultdict(lambda: collections.defaultdict(list))
stat = collections.Counter()
for seg, ls in sorted(seg2l.items()):
    t = tier(ls)
    if seg in lit: kind = "literal"
    elif any(seg.endswith(p) and seg[:-len(p)] in lit for p in POSES): kind = "legpose"
    elif seg.startswith("Marbled") and seg[len("Marbled"):] in lit: kind = "marbled"
    elif seg.endswith("Lower") and seg[:-5] in lit: kind = "lower"
    else: kind = "unused"
    stat[(t, kind)] += 1
    if kind in ("literal",):
        rows[t][ls[0] if len(ls)==1 else "+".join(sorted(ls))].append(seg)
for k in sorted(stat): print(k, stat[k])
json.dump({t: {l: v for l, v in sorted(rows[t].items())} for t in rows},
          io.open(os.path.join(OUT, "worklist.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
# 平铺：每个 tier 的字面量 id，按图层分组
for t in ("T1","T2"):
    segs = sorted({s for g in rows[t].values() for s in g})
    io.open(os.path.join(OUT, "ids_%s.txt" % t), "w", encoding="utf-8").write(
        "\n".join("%s\t%s\t%s" % (s, disp(s), ",".join(seg2l[s])) for s in segs))
    print(t, "字面量 id", len(segs))
