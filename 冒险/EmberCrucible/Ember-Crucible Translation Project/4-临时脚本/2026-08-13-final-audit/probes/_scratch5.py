# -*- coding: utf-8 -*-
import io, re, sys, os, json, collections
R = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
sys.path.insert(0, os.path.join(R, "4-临时脚本", "2026-08-13-final-audit", "probes"))
from cn_corpus import load_all, plain

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

o = io.open(os.path.join(os.environ.get("OUTDIR", "."), "rep13.txt"), "w", encoding="utf-8")
leaves = load_all()

o.write("=== 调谐 / 同调 全库计数 ===\n")
a = b = 0; ex = []
for L in leaves:
    na = len(re.findall("调谐", L["s"])); nb = len(re.findall("同调", L["s"]))
    a += na; b += nb
    if na: ex.append((L["file"], L["path"][-70:], re.search(r".{0,40}调谐.{0,40}", L["s"]).group(0)))
o.write("  compendium: 调谐 %d / 同调 %d\n" % (a, b))
for e in ex[:20]: o.write("     %s | %s\n        %s\n" % e)

for repo in ["1-Ember汉化插件", "2-Crucible汉化插件"]:
    p = os.path.join(R, repo, "lang", "cn.json")
    if not os.path.exists(p): continue
    C = flat(json.load(io.open(p, encoding="utf-8")))
    a = sum(len(re.findall("调谐", v)) for v in C.values())
    b = sum(len(re.findall("同调", v)) for v in C.values())
    o.write("  %s lang: 调谐 %d / 同调 %d\n" % (repo, a, b))
    for k, v in C.items():
        if "调谐" in v: o.write("     %s = %s\n" % (k, v))

o.write("\n=== 恢复 / 复原 (Restoration) ===\n")
for repo in ["2-Crucible汉化插件"]:
    C = flat(json.load(io.open(os.path.join(R, repo, "lang", "cn.json"), encoding="utf-8")))
    E = flat(json.load(io.open(os.path.join(R, repo, "lang", "en.json"), encoding="utf-8")))
    for k in sorted(C):
        if re.search(r"restoration", E.get(k, ""), re.I) or "复原" in C[k]:
            o.write("  %-56s EN=%-24s CN=%s\n" % (k[-56:], E.get(k, "")[:24], C[k]))
cnt = collections.Counter()
for L in leaves:
    cnt["复原"] += len(re.findall("复原", L["s"]))
o.write("  compendium 复原 出现 %d 次\n" % cnt["复原"])

o.write("\n=== 目标范围 Self：自我 / 自身 / 仅自己 ===\n")
C = flat(json.load(io.open(os.path.join(R, "2-Crucible汉化插件", "lang", "cn.json"), encoding="utf-8")))
E = flat(json.load(io.open(os.path.join(R, "2-Crucible汉化插件", "lang", "en.json"), encoding="utf-8")))
for k in sorted(C):
    if k.startswith("ACTION.TARGET_SCOPES") or k.startswith("ACTION.TARGET_TYPES"):
        o.write("  %-44s EN=%-16s CN=%s\n" % (k, E.get(k, ""), C[k]))
for L in leaves:
    if L["file"] == "crucible.rules.json" and "Actions" in L["path"]:
        t = plain(L["s"])
        for m in re.finditer(r".{0,50}仅自己.{0,50}", t):
            o.write("  rules: %s\n" % m.group(0))
o.close()
print("ok")
