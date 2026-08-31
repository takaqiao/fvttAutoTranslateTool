# -*- coding: utf-8 -*-
import io, re, os, sys, json, collections
R = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
sys.path.insert(0, os.path.join(R, "4-临时脚本", "2026-08-13-final-audit", "probes"))
from cn_corpus import load_all, plain
leaves = load_all()
o = io.open(os.path.join(os.environ.get("OUTDIR", "."), "rep17.txt"), "w", encoding="utf-8")

o.write("=== 动态代币 上下文 ===\n")
for L in leaves:
    for m in re.finditer(r".{0,150}代币.{0,250}", plain(L["s"])):
        o.write("  %s | %s\n     %s\n\n" % (L["file"], L["path"][-70:], m.group(0)))

o.write("=== EN 对应 ===\n")
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
E = flat(json.load(io.open(os.path.join(R, "1-Ember汉化插件", "compendium", "en", "ember.adventure.json"), encoding="utf-8")))
for k, v in E.items():
    if "Patch 0.3.0" in k:
        for m in re.finditer(r".{0,120}Dynamic Token.{0,220}", re.sub(r"<[^>]+>", " ", v)):
            o.write("  %s\n\n" % m.group(0))

o.write("=== 令牌/指示物：GM 指南以外的正文分布 ===\n")
c = collections.Counter()
for L in leaves:
    n1 = len(re.findall("令牌", L["s"])); n2 = len(re.findall("指示物", L["s"]))
    if n1 or n2:
        g = "GM指南" if "Gamemaster's Guide" in L["path"] else ("场景行为" if ".behaviors." in L["path"] or ".regions." in L["path"] else "其他正文")
        c[(g, "令牌")] += n1; c[(g, "指示物")] += n2
for k, v in sorted(c.items()):
    o.write("  %s %s = %d\n" % (k[0], k[1], v))

o.write("\n=== 纯字母 与 汉字 之间空格 ===\n")
t = s = 0
tex = collections.Counter()
for L in leaves:
    x = plain(L["s"])
    for m in re.finditer(r"[A-Za-z][\u4e00-\u9fff]|[\u4e00-\u9fff][A-Za-z]", x):
        t += 1; tex[m.group(0)] += 1
    s += len(re.findall(r"[A-Za-z] [\u4e00-\u9fff]|[\u4e00-\u9fff] [A-Za-z]", x))
o.write("  紧贴 %d / 空格 %d\n" % (t, s))
for k, v in tex.most_common(20):
    o.write("     %5d  %s\n" % (v, k))
o.close()
print("ok")
