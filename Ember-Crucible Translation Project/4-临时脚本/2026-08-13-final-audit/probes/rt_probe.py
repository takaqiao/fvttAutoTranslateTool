# -*- coding: utf-8 -*-
"""Probe: EN 'next/following round' -> CN rendering. Read-only."""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKIP_KEYS = {"_id", "path", "_variants", "_when"}
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [
    os.path.join(ROOT, "1-Ember汉化插件"),
    os.path.join(ROOT, "2-Crucible汉化插件"),
]

def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS: continue
            sub = cn.get(k) if isinstance(cn, dict) else None
            walk(v, sub, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        out.append({"path": ".".join(path), "en": en,
                    "cn": cn if isinstance(cn, str) else None})

def collect():
    rows = []
    for repo in REPOS:
        en_dir = os.path.join(repo, "compendium", "en")
        cn_dir = os.path.join(repo, "compendium", "cn")
        if not os.path.isdir(en_dir): continue
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith(".json") or fn.startswith("_"): continue
            en = json.load(open(os.path.join(en_dir, fn), encoding="utf-8"))
            cp = os.path.join(cn_dir, fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            sub = []
            walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], sub)
            for r in sub:
                r["pack"] = fn; r["repo"] = os.path.basename(repo)
            rows.extend(sub)
    return rows

RX = re.compile(r"\b(the\s+)?(next|following)\s+round\b", re.I)

def main():
    rows = collect()
    hits = [r for r in rows if RX.search(r["en"])]
    print(f"total leaves={len(rows)} en-hits={len(hits)}")
    out = []
    for r in hits:
        en_n = len(RX.findall(r["en"]))
        cn = r["cn"] or ""
        out.append({
            "repo": r["repo"], "pack": r["pack"], "path": r["path"],
            "en_hits": en_n,
            "cn_lun": len(re.findall(r"轮", cn)),
            "cn_huihe": len(re.findall(r"回合", cn)),
            "en": r["en"], "cn": cn,
        })
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rt_hits.json")
    json.dump({"n": len(out), "rows": out}, open(dst, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("->", dst)
    for o in out:
        print(f"{o['repo'][:8]} {o['pack']:<34} enhits={o['en_hits']} 轮={o['cn_lun']} 回合={o['cn_huihe']} {o['path'][:120]}")

main()
