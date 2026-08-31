# -*- coding: utf-8 -*-
import io, re, sys, os, json, collections
R = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
sys.path.insert(0, os.path.join(R, "4-临时脚本", "2026-08-13-final-audit", "probes"))
from cn_corpus import load_all, plain

leaves = load_all()
o = io.open(os.path.join(os.environ.get("OUTDIR", "."), "rep11.txt"), "w", encoding="utf-8")

o.write("=== 同一叶内 数字+单位 两种写法并存 ===\n")
for u in ["点", "英尺", "轮", "小时", "分钟", "级"]:
    hits = []
    seenpath = set()
    for L in leaves:
        t = plain(L["s"])
        a = re.findall(r"\d" + u, t); b = re.findall(r"\d " + u, t)
        if a and b:
            if L["path"] in seenpath: continue
            seenpath.add(L["path"])
            hits.append((L["file"], L["path"], a[:3], b[:3]))
    o.write("  %s：%d 叶同时含两种写法\n" % (u, len(hits)))
    for h in hits[:4]:
        o.write("     %s | %s | %s vs %s\n" % (h[0], h[1][-60:], h[2], h[3]))

o.write("\n=== exhaustion 等级写法 ===\n")
c = collections.Counter(); ex = {}
for L in leaves:
    for m in re.finditer(r".{0,14}(exhaustion|疲乏|力竭).{0,14}", plain(L["s"])):
        k = m.group(0)
        c[k] += 1
        ex.setdefault(k, L["path"][-60:])
for k, v in c.most_common(24):
    o.write("  %3d  %s   @%s\n" % (v, k.replace("\n", " "), ex[k]))

o.write("\n=== lang 文件 ===\n")
for repo in ["1-Ember汉化插件", "2-Crucible汉化插件"]:
    d = os.path.join(R, repo, "lang")
    if not os.path.isdir(d): continue
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        try:
            data = json.load(io.open(p, encoding="utf-8"))
        except Exception as e:
            o.write("  %s/%s  <parse error %s>\n" % (repo, fn, e)); continue
        flat = {}
        def w(ob, path):
            if isinstance(ob, dict):
                for k, v in ob.items(): w(v, path + [k])
            elif isinstance(ob, str):
                flat[".".join(path)] = ob
        w(data, [])
        o.write("  %s/%s  %d strings\n" % (repo, fn, len(flat)))
o.close()
print("ok")
