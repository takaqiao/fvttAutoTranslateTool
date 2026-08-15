# -*- coding: utf-8 -*-
"""数字—量词—单位 写法一致性 + 若干定点核查"""
import json, io, re, sys, os, collections
R = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
sys.path.insert(0, os.path.join(R, "4-临时脚本", "2026-08-13-final-audit", "probes"))
from cn_corpus import load_all, plain

leaves = load_all()
o = io.open(os.path.join(os.environ.get("OUTDIR", "."), "rep10.txt"), "w", encoding="utf-8")

UNITS = ["级", "点", "轮", "回合", "英尺", "英里", "格", "小时", "分钟", "天", "秒"]
o.write("=== 阿拉伯数字 + 单位：紧贴 vs 空格（全库，含孪生包）===\n")
for u in UNITS:
    tight = 0; sp = 0
    for L in leaves:
        t = L["s"]
        tight += len(re.findall(r"\d" + u, t))
        sp += len(re.findall(r"\d " + u, t))
    o.write("  %-4s 紧贴 %5d / 空格 %5d\n" % (u, tight, sp))

o.write("\n=== 汉字数字 + 单位（可能与阿拉伯数字并存）===\n")
for u in UNITS:
    c = collections.Counter(); ex = {}
    for L in leaves:
        for m in re.findall(r"(?<![\u4e00-\u9fff])([一二三四五六七八九十]{1,3})" + u, plain(L["s"])):
            c[m + u] += 1
            ex.setdefault(m + u, L["path"][-70:])
    if c:
        o.write("  %s: %s\n" % (u, ", ".join("%s×%d" % (k, v) for k, v in c.most_common(12))))

o.write("\n=== 疑似量词冗余：N 点 X点 / N单位 点 ===\n")
pats = [r"\d+\s*点[\u4e00-\u9fff]{1,6}点(?![\u4e00-\u9fff])", r"\d+[\u4e00-\u9fff]{1,4}\s+点"]
seen = collections.Counter(); ex = {}
for L in leaves:
    for p in pats:
        for m in re.finditer(p, L["s"]):
            seen[m.group(0)] += 1
            ex.setdefault(m.group(0), (L["file"], L["path"][-70:]))
for k, v in seen.most_common(40):
    o.write("  %4d  %s   ex=%s\n" % (v, k, ex[k]))

o.write("\n=== 英雄气概点 用法 ===\n")
c = collections.Counter()
for L in leaves:
    for m in re.findall(r".{0,8}英雄气概点?", plain(L["s"])):
        c[m[-8:]] += 1
for k, v in c.most_common(25):
    o.write("  %4d  %s\n" % (v, k))

o.write("\n=== 2专注 点 原文 ===\n")
for L in leaves:
    for m in re.finditer(r".{0,60}\d\s*专注\s*点.{0,40}", L["s"]):
        o.write("  [%s] %s\n     %s\n" % (L["file"], L["path"][-70:], m.group(0)))
o.close()
print("ok")
