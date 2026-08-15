# -*- coding: utf-8 -*-
import io, re, sys, os, json, collections
R = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
sys.path.insert(0, os.path.join(R, "4-临时脚本", "2026-08-13-final-audit", "probes"))
from cn_corpus import load_all, plain

leaves = load_all()
o = io.open(os.path.join(os.environ.get("OUTDIR", "."), "rep12.txt"), "w", encoding="utf-8")

o.write("=== exhaustion 「N级」写法精确统计（按 path 去孪生包重复）===\n")
forms = collections.Counter(); ex = collections.defaultdict(list)
seen = set()
for L in leaves:
    if L["path"] in seen: continue
    seen.add(L["path"])
    for m in re.finditer(r"([一二三四五1-9])\s?级\s?&(?:amp;)?[Rr]eference\[exhaustion\]", L["s"]):
        f = re.sub(r"[1-9]", "N", re.sub(r"[一二三四五]", "H", m.group(0)))
        forms[f] += 1
        ex[f].append((L["file"], L["path"][-70:]))
for k, v in forms.most_common():
    o.write("  %3d  %s\n" % (v, k))
    for e in ex[k][:6]:
        o.write("        %s | %s\n" % e)

o.write("\n=== EN 侧同处写法 ===\n")
def flat(d):
    out = {}
    def w(ob, p):
        if isinstance(ob, dict):
            for k, v in ob.items(): w(v, p + [str(k)])
        elif isinstance(ob, list):
            for i, v in enumerate(ob): w(v, p + ["[%d]" % i])
        elif isinstance(ob, str): out[".".join(p)] = ob
    w(d, [])
    return out
EN = {}
for repo in ["1-Ember汉化插件", "2-Crucible汉化插件"]:
    d = os.path.join(R, repo, "compendium", "en")
    for fn in os.listdir(d):
        if fn.endswith(".json") and fn != "_source.json":
            EN[fn] = flat(json.load(io.open(os.path.join(d, fn), encoding="utf-8")))
enf = collections.Counter()
for fn, mp in EN.items():
    for k, v in mp.items():
        for m in re.finditer(r"\w+\s+level[s]?\s+of\s+&(?:amp;)?[Rr]eference\[exhaustion\]|[Rr]eceive\s+\w+\s+&(?:amp;)?[Rr]eference\[exhaustion\]|\w+\s+&(?:amp;)?[Rr]eference\[exhaustion\]", v):
            enf[m.group(0)] += 1
for k, v in enf.most_common(20):
    o.write("  %3d  %s\n" % (v, k))

o.write("\n=== EN 侧 Rock Bottom / Drevin Markus 范围写法 ===\n")
for fn, mp in EN.items():
    for k, v in mp.items():
        if "Rock Bottom" in k or "Drevin Markus" in k:
            for m in re.finditer(r".{0,30}\d[\d,]*\s*[-\u2013]\s*\d[\d,]*.{0,20}", v):
                o.write("  [%s] %s :: %s\n" % (fn, k[-60:], m.group(0)))

o.write("\n=== EN 侧 ']] :' 空格 ===\n")
n = 0
for fn, mp in EN.items():
    for k, v in mp.items():
        n += len(re.findall(r"\]\]\s+[:,\.]", v))
o.write("  EN ']]' 后空格再标点: %d\n" % n)
o.close()
print("ok")
