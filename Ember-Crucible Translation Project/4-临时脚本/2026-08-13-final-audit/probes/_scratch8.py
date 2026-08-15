# -*- coding: utf-8 -*-
import io, re, os, sys, json, collections
R = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
sys.path.insert(0, os.path.join(R, "4-临时脚本", "2026-08-13-final-audit", "probes"))
from cn_corpus import load_all, plain
leaves = load_all()
o = io.open(os.path.join(os.environ.get("OUTDIR", "."), "rep16.txt"), "w", encoding="utf-8")

o.write("=== 半数她年龄 ===\n")
for L in leaves:
    for m in re.finditer(r".{0,60}半数.{0,60}", L["s"]):
        o.write("  %s | %s\n     %s\n" % (L["file"], L["path"][-60:], m.group(0)))

o.write("\n=== 令牌 用法采样（compendium）===\n")
c = collections.Counter(); ex = {}
for L in leaves:
    for m in re.finditer(r".{0,12}令牌.{0,12}", plain(L["s"])):
        k = m.group(0)
        c[k] += 1
        ex.setdefault(k, (L["file"], L["path"][-60:]))
for k, v in c.most_common(30):
    o.write("  %4d  %s   @%s\n" % (v, k.replace("\n", " "), ex[k][1]))

o.write("\n=== 指示物 用法采样（compendium）===\n")
c2 = collections.Counter()
for L in leaves:
    for m in re.finditer(r".{0,12}指示物.{0,12}", plain(L["s"])):
        c2[m.group(0)] += 1
for k, v in c2.most_common(20):
    o.write("  %4d  %s\n" % (v, k.replace("\n", " ")))

o.write("\n=== 拉丁字母与汉字之间是否加空格 ===\n")
pat_tight = re.compile(r"[A-Za-z0-9][\u4e00-\u9fff]|[\u4e00-\u9fff][A-Za-z0-9]")
pat_space = re.compile(r"[A-Za-z0-9] [\u4e00-\u9fff]|[\u4e00-\u9fff] [A-Za-z0-9]")
t = s = 0
tex = collections.Counter()
for L in leaves:
    x = plain(L["s"])
    for m in pat_tight.finditer(x):
        t += 1; tex[m.group(0)] += 1
    s += len(pat_space.findall(x))
o.write("  紧贴 %d / 空格 %d\n" % (t, s))
for k, v in tex.most_common(25):
    o.write("     %5d  %s\n" % (v, k))
o.close()
print("ok")
