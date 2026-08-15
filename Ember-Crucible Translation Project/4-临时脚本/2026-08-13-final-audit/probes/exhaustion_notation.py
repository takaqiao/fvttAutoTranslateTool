# -*- coding: utf-8 -*-
"""只读探针 #6：把「N 级力竭」的中文写法与**同一叶同一位置的英文写法**配对，
证明中文侧的写法分裂不是对英文分裂的忠实反映。
另：导出两个 lang 文件的中文串供人工通读。
"""
import io, re, sys, os, json, collections
R = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

o = io.open(os.environ.get("OUT", "exhaustion.txt"), "w", encoding="utf-8")
pairs = []
for repo in ["1-Ember汉化插件", "2-Crucible汉化插件"]:
    en_d = os.path.join(R, repo, "compendium", "en")
    cn_d = os.path.join(R, repo, "compendium", "cn")
    for fn in sorted(os.listdir(en_d)):
        if not fn.endswith(".json") or fn == "_source.json": continue
        cp = os.path.join(cn_d, fn)
        if not os.path.exists(cp): continue
        E = flat(json.load(io.open(os.path.join(en_d, fn), encoding="utf-8")))
        C = flat(json.load(io.open(cp, encoding="utf-8")))
        for k, ev in E.items():
            if "exhaustion" not in ev.lower(): continue
            cv = C.get(k)
            if not cv: continue
            ens = re.findall(r"(?:\w+\s+levels?\s+of\s+|\b\d+\s+)?&(?:amp;)?[Rr]eference\[exhaustion\]", ev)
            cns = re.findall(r"(?:[一二三四五1-9]\s?级\s?)?&(?:amp;)?[Rr]eference\[exhaustion\]", cv)
            if not ens: continue
            pairs.append((fn, k, ens, cns))

tbl = collections.Counter()
ex = collections.defaultdict(list)
for fn, k, ens, cns in pairs:
    for i in range(min(len(ens), len(cns))):
        e = re.sub(r"\d+", "N", ens[i]).replace("&amp;", "&")
        c = re.sub(r"[1-9]", "N", re.sub(r"[一二三四五]", "H", cns[i])).replace("&amp;", "&")
        tbl[(e, c)] += 1
        ex[(e, c)].append((fn, k[-70:]))
o.write("=== EN 写法 → CN 写法 配对（含孪生包，故计数约为唯一叶两倍）===\n")
for (e, c), v in tbl.most_common():
    o.write("%4d  EN: %-46s  CN: %s\n" % (v, e, c))
    o.write("        e.g. %s | %s\n" % ex[(e, c)][0])

o.write("\n\n=== lang cn.json 中文串 ===\n")
for repo in ["1-Ember汉化插件", "2-Crucible汉化插件"]:
    p = os.path.join(R, repo, "lang", "cn.json")
    if not os.path.exists(p): continue
    C = flat(json.load(io.open(p, encoding="utf-8")))
    E = flat(json.load(io.open(os.path.join(R, repo, "lang", "en.json"), encoding="utf-8")))
    o.write("\n#### %s  (%d)\n" % (repo, len(C)))
    for k in sorted(C):
        if re.search(r"[\u4e00-\u9fff]", C[k]):
            o.write("  %-58s %s\n" % (k[-58:], C[k]))
o.close()
print("ok")
