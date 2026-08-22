# -*- coding: utf-8 -*-
import io, json, os, collections
OUT = os.path.dirname(os.path.abspath(__file__))
u = json.load(io.open(os.path.join(OUT, "parts_universe2.json"), encoding="utf-8"))
lit = set(json.load(io.open(os.path.join(OUT, "litcheck.json"), encoding="utf-8"))["literal"])
seg2l = u["seg2layers"]
T1 = {"ears","mane","marks","scales","teeth","mouth","beard","fur","jaw","eyebrows","tail",
      "collar","horns","eyes","face","hair","head","hood"}
T2 = {"hand","etherealHand","arm","etherealArm","forearm","etherealForearm","leg","legs","foot","torso"}
import re
RE = re.compile(r'(?<!^)([A-Z1-9])')
disp = lambda s: RE.sub(r' \1', s)
t1 = sorted({s for s,ls in seg2l.items() if set(ls) & T1})
t2 = sorted({s for s,ls in seg2l.items() if (set(ls) & T2) and s not in t1})
rest = sorted(set(seg2l) - set(t1) - set(t2))
def rep(nm, xs):
    nl = [x for x in xs if x not in lit]
    print("%s: %d 条（其中 ember.mjs 无字面量 %d 条）" % (nm, len(xs), len(nl)))
    return nl
n1 = rep("T1 头脸族", t1); n2 = rep("T2 四肢躯干", t2); n3 = rep("其余（装备等）", rest)
print("T1 非字面量：", n1[:40])
print("T2 非字面量：", n2[:90])
json.dump({"T1": t1, "T2": t2, "REST": rest,
           "T1_display": [disp(s) for s in t1], "T2_display": [disp(s) for s in t2],
           "nonliteral_T1": n1, "nonliteral_T2": n2},
          io.open(os.path.join(OUT, "target.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
