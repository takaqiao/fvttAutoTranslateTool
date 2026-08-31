# -*- coding: utf-8 -*-
"""只读探针 #4：按体裁分层抽样，导出**纯中文**供人工朗读/精读。
抽样方法：先按叶去孪生包重复（同 path 只留一份），再按汉字数分 5 层（20/40/60/80 分位），
每层用固定随机种子等量抽取，保证长短都覆盖。
"""
import json, io, re, sys, os, random, collections

R = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
sys.path.insert(0, os.path.join(R, "4-临时脚本", "2026-08-13-final-audit", "probes"))
from cn_corpus import load_all, plain, HAN

SEED = int(os.environ.get("SEED", "20260813"))

def dedupe(leaves):
    seen = {}
    for L in leaves:
        if L["path"] not in seen:
            seen[L["path"]] = L
    return list(seen.values())

def strat_sample(items, n, keyfn):
    items = sorted(items, key=keyfn)
    if not items: return []
    k = 5
    out = []
    rnd = random.Random(SEED)
    step = max(1, len(items) // k)
    for i in range(k):
        chunk = items[i * step:(i + 1) * step] if i < k - 1 else items[(k - 1) * step:]
        if not chunk: continue
        out += rnd.sample(chunk, min(max(1, n // k), len(chunk)))
    return out

def readalouds(leaves):
    out = []
    for L in leaves:
        for m in re.finditer(r'<section class="block readaloud">(.*?)</section>', L["s"], re.S):
            t = plain(m.group(1))
            if len(HAN.findall(t)) >= 40:
                out.append((L["file"], L["path"], t))
    return out

def main():
    leaves = dedupe(load_all())
    mode = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    rows = []
    if mode == "readaloud":
        ra = readalouds(leaves)
        seen = set(); u = []
        for f, p, t in ra:
            if t in seen: continue
            seen.add(t); u.append((f, p, t))
        rows = strat_sample(u, n, lambda x: len(x[2]))
    elif mode == "bio":
        c = [(L["file"], L["path"], plain(L["s"])) for L in leaves if ".biography." in L["path"]]
        rows = strat_sample(c, n, lambda x: len(x[2]))
    elif mode == "itemdesc":
        c = [(L["file"], L["path"], plain(L["s"])) for L in leaves
             if L["path"].endswith(".description") and len(HAN.findall(plain(L["s"]))) >= 40]
        rows = strat_sample(c, n, lambda x: len(x[2]))
    elif mode == "rules":
        c = [(L["file"], L["path"], plain(L["s"])) for L in leaves
             if L["file"] in ("crucible.rules.json",) and len(HAN.findall(plain(L["s"]))) >= 100]
        rows = strat_sample(c, n, lambda x: len(x[2]))
    elif mode == "prose":
        c = []
        for L in leaves:
            if not L["path"].endswith((".text", ".contentGamemaster", ".contentOverview", ".exposition")):
                continue
            s = re.sub(r'<section class="block readaloud">.*?</section>', " ", L["s"], flags=re.S)
            t = plain(s)
            if len(HAN.findall(t)) >= 200:
                c.append((L["file"], L["path"], t))
        rows = strat_sample(c, n, lambda x: len(x[2]))
    lim = int(os.environ.get("LIM", "900"))
    o = io.open(os.environ.get("OUT", mode + ".txt"), "w", encoding="utf-8")
    o.write("mode=%s  样本 %d 条  seed=%d\n" % (mode, len(rows), SEED))
    for i, (f, p, t) in enumerate(rows):
        o.write("\n--- #%d  [%s] %s\n%s\n" % (i + 1, f, p[-80:], t[:lim]))
    o.close()
    print(mode, len(rows))

if __name__ == "__main__":
    main()
