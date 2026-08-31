# -*- coding: utf-8 -*-
"""只读探针：把两个汉化仓库 compendium/cn 下所有中文叶子抽出来，按体裁分层。
不写库。输出到 scratch 目录。
"""
import json, io, os, re, sys, collections

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
CN_DIRS = [
    os.path.join(ROOT, "1-Ember汉化插件", "compendium", "cn"),
    os.path.join(ROOT, "2-Crucible汉化插件", "compendium", "cn"),
]

HAN = re.compile(r"[\u4e00-\u9fff]")

def walk(obj, path, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, path + [str(k)], out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, path + ["[%d]" % i], out)
    elif isinstance(obj, str):
        if HAN.search(obj):
            out.append((".".join(path), obj))

def load_all():
    leaves = []
    for d in CN_DIRS:
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            p = os.path.join(d, fn)
            with io.open(p, encoding="utf-8") as f:
                data = json.load(f)
            sub = []
            walk(data, [], sub)
            for path, s in sub:
                leaves.append({"file": fn, "repo": os.path.basename(os.path.dirname(os.path.dirname(d))), "path": path, "s": s})
    return leaves

TAG = re.compile(r"<[^>]+>")
MACRO = re.compile(r"@\w+\[[^\]]*\](\{[^}]*\})?|\[\[[^\]]*\]\]|&\w+\[[^\]]*\]")

def plain(s):
    s = MACRO.sub(lambda m: (m.group(1) or "")[1:-1] if m.group(1) else " ", s)
    s = TAG.sub(" ", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def stratum(leaf):
    p = leaf["path"]
    if ".biography." in p:
        return "bio"
    if p.endswith(".description") or ".description." in p:
        return "itemdesc"
    if re.search(r"\.pages\.[^.]+\.(text|contentOverview|contentGamemaster)$", p) or p.endswith(".text"):
        return "journal"
    if p.endswith(".name") or p.endswith(".tokenName") or p.endswith(".label"):
        return "name"
    return "other"

if __name__ == "__main__":
    leaves = load_all()
    for L in leaves:
        L["plain"] = plain(L["s"])
        L["nhan"] = len(HAN.findall(L["plain"]))
        L["stratum"] = stratum(L)
    tot = collections.Counter()
    cnt = collections.Counter()
    for L in leaves:
        tot[L["stratum"]] += L["nhan"]
        cnt[L["stratum"]] += 1
    print("leaves:", len(leaves), "han chars:", sum(L["nhan"] for L in leaves))
    for k in sorted(tot, key=lambda x: -tot[x]):
        print("  %-10s leaves=%6d  hanchars=%9d" % (k, cnt[k], tot[k]))
    out = os.environ.get("OUT")
    if out:
        with io.open(out, "w", encoding="utf-8") as f:
            json.dump(leaves, f, ensure_ascii=False)
        print("wrote", out)
