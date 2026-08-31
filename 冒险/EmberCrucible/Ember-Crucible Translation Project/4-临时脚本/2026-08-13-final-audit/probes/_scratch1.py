# -*- coding: utf-8 -*-
import json, io, re, sys, os, collections
R = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
sys.path.insert(0, os.path.join(R, "4-临时脚本", "2026-08-13-final-audit", "probes"))
from cn_corpus import load_all, plain

leaves = load_all()

def flat(d):
    o = {}
    def w(ob, path):
        if isinstance(ob, dict):
            for k, v in ob.items(): w(v, path + [str(k)])
        elif isinstance(ob, list):
            for i, v in enumerate(ob): w(v, path + ["[%d]" % i])
        elif isinstance(ob, str): o[".".join(path)] = ob
    w(d, [])
    return o

EN = {}
for repo in ["1-Ember汉化插件", "2-Crucible汉化插件"]:
    d = os.path.join(R, repo, "compendium", "en")
    for fn in os.listdir(d):
        if fn.endswith(".json") and fn != "_source.json":
            EN[fn] = flat(json.load(io.open(os.path.join(d, fn), encoding="utf-8")))

out = io.open(os.path.join(os.environ.get("OUTDIR", "."), "rep9.txt"), "w", encoding="utf-8")

out.write("=== 利维坦 被发现 ===\n")
for L in leaves:
    if "Age of Rediscovery" in L["path"]:
        t = plain(L["s"])
        m = re.search(r".{0,60}被啃食过半.{0,140}", t)
        if m:
            out.write("%s|%s\n  CN: %s\n" % (L["file"], L["path"], m.group(0)))
            e = re.sub(r"<[^>]+>", " ", EN.get(L["file"], {}).get(L["path"], ""))
            m2 = re.search(r".{0,60}Leviathan carcass.{0,220}", e)
            if m2: out.write("  EN: %s\n" % m2.group(0))

out.write("\n=== 数字范围写法 ===\n")
c = collections.Counter(); ex = {}
for L in leaves:
    for m in re.finditer(r"\d[\d,]*\s*[-\u2013\u2014~\uff5e]\s*\d[\d,]*", L["s"]):
        pat = re.sub(r"\d", "N", m.group(0))
        c[pat] += 1
        ex.setdefault(pat, (L["file"], L["path"][-70:], m.group(0)))
for k, v in c.most_common(40):
    out.write("  %4d  %r  ex=%s\n" % (v, k, ex[k]))

out.close()
print("ok")
