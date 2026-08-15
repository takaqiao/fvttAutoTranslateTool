# -*- coding: utf-8 -*-
"""Broad probe: every EN use of 'round' as a time unit vs CN 轮/回合 count.

False-positive modes documented:
 - 'round' as adjective/noun of shape ('round table', 'a round shield') -> excluded by lexicon
 - CN 轮 also appears in 轮到/纺轮/车轮/轮廓/一轮明月 etc -> we count 轮 minus those tokens
 - block-level counting cannot align which 轮 maps to which 'round'; treat as a screen only
"""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SKIP_KEYS = {"_id", "path", "_variants", "_when"}
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [os.path.join(ROOT, "1-Ember汉化插件"), os.path.join(ROOT, "2-Crucible汉化插件")]

def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS: continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path+[str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None, path+[str(i)], out)
    elif isinstance(en, str):
        out.append({"path": ".".join(path), "en": en, "cn": cn if isinstance(cn, str) else None})

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
            for r in sub: r["pack"] = fn; r["repo"] = os.path.basename(repo)
            rows.extend(sub)
    return rows

# 'round' as a time unit
TIME_RX = re.compile(
    r"\b(?:\d+|one|two|three|four|five|a|an|each|every|per|next|following|that|this|"
    r"first|last|same|subsequent|additional|other|d\d+|extra)\s+round(?:s)?\b"
    r"|\bround(?:s)?\s+(?:later|of\s+combat|of\s+the\s+combat)\b"
    r"|\bon\s+(?:the\s+)?round\b|\bstart\s+of\s+(?:the\s+)?round\b|\bend\s+of\s+(?:the\s+)?round\b"
    r"|\brounds?\s+remain", re.I)
NEG = re.compile(r"round\s+(table|shield|room|window|tower|hole|stone|number)", re.I)

# CN 轮 that is NOT the combat round
CN_LUN_NEG = re.compile(r"轮到|轮廓|车轮|纺轮|轮子|轮回|轮换|轮盘|轮流|轮船|滚轮|齿轮|一轮明月|轮值|轮椅|轮胎|轮转|轮次序")

def main():
    rows = collect()
    flagged = []
    for r in rows:
        en = r["en"]
        hits = [m.group(0) for m in TIME_RX.finditer(en) if not NEG.search(m.group(0))]
        if not hits: continue
        cn = r["cn"] or ""
        lun_all = len(re.findall("轮", cn))
        lun_neg = sum(len(m.group(0)) - len(m.group(0).replace("轮","")) for m in CN_LUN_NEG.finditer(cn))
        lun = lun_all - lun_neg
        r2 = {"repo": r["repo"], "pack": r["pack"], "path": r["path"],
              "en_round_uses": len(hits), "en_samples": hits[:8],
              "cn_lun_net": lun, "cn_huihe": len(re.findall("回合", cn)),
              "deficit": len(hits) - lun}
        if r2["deficit"] > 0:
            flagged.append(r2)
    flagged.sort(key=lambda x: -x["deficit"])
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rt_broad.json")
    json.dump({"n": len(flagged), "rows": flagged}, open(dst,"w",encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("flagged (EN round-uses > CN net 轮):", len(flagged), "->", dst)
    for f in flagged:
        print(f"  d={f['deficit']} en={f['en_round_uses']} 轮={f['cn_lun_net']} 回合={f['cn_huihe']} "
              f"{f['pack'][:28]:<28} {f['path'][:95]}")
        print(f"      {f['en_samples']}")

main()
