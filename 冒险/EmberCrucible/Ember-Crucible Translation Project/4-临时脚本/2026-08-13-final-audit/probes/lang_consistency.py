# -*- coding: utf-8 -*-
"""只读探针 #7：UI 语言文件（lang/cn.json）的中文自洽性。
两个仓库的 lang/en.json 与 lang/cn.json 是同键对照表，可直接做：
  A. 同一英文串 → 多个中文串
  B. 第二人称 你/您 混用
  C. 关键术语在 UI 内的分叉（同调 vs 调谐 等）
"""
import io, re, sys, os, json, collections
R = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"

def flat(d):
    out = {}
    def w(ob, p):
        if isinstance(ob, dict):
            for k, v in ob.items(): w(v, p + [str(k)])
        elif isinstance(ob, str): out[".".join(p)] = ob
    w(d, [])
    return out

o = io.open(os.environ.get("OUT", "lang_consistency.txt"), "w", encoding="utf-8")
for repo in ["1-Ember汉化插件", "2-Crucible汉化插件"]:
    ep = os.path.join(R, repo, "lang", "en.json")
    cp = os.path.join(R, repo, "lang", "cn.json")
    if not (os.path.exists(ep) and os.path.exists(cp)): continue
    E = flat(json.load(io.open(ep, encoding="utf-8")))
    C = flat(json.load(io.open(cp, encoding="utf-8")))
    o.write("\n########## %s  en=%d cn=%d\n" % (repo, len(E), len(C)))

    o.write("\n--- A. 同英文 → 多中文 ---\n")
    g = collections.defaultdict(dict)
    for k, ev in E.items():
        cv = C.get(k)
        if cv is None: continue
        if not re.search(r"[\u4e00-\u9fff]", cv): continue
        g[ev.strip()][cv] = g[ev.strip()].get(cv, []) + [k]
    n = 0
    for ev, m in sorted(g.items(), key=lambda x: -len(x[1])):
        if len(m) < 2: continue
        n += 1
        o.write('  EN "%s"\n' % ev[:120])
        for cv, ks in m.items():
            o.write("     -> %-30s  %s\n" % (cv[:30], ", ".join(k[-46:] for k in ks[:4])))
    o.write("  共 %d 组\n" % n)

    o.write("\n--- B. 你 / 您 ---\n")
    ni = [k for k, v in C.items() if "你" in v]
    nin = [k for k, v in C.items() if "您" in v]
    o.write("  含「你」%d 条；含「您」%d 条\n" % (len(ni), len(nin)))
    for k in nin:
        o.write("     您: %-56s %s\n" % (k[-56:], C[k]))
        o.write("        EN: %s\n" % E.get(k, "")[:160])

    o.write("\n--- C. attunement / 同调 vs 调谐 ---\n")
    for k, v in sorted(C.items()):
        if "调谐" in v or ("attunement" in E.get(k, "").lower() and "同调" not in v):
            o.write("     %-56s CN=%s\n        EN=%s\n" % (k[-56:], v, E.get(k, "")))
o.close()
print("ok")
