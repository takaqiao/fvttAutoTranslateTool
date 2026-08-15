# -*- coding: utf-8 -*-
"""U1: one compact evidence block per DISTINCT target in findings[lo:hi].

Per target:
  EN doc name (ids table) / CN doc name (name-field index, strongest basis)
  full library census of CN labels for that target, split by the EN label used
  the finding indices that hit it
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
SWAP = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/uuid_swap.json"
IDS = os.path.join(P, "4-临时脚本", "2026-08-12-fix", "reports", "ember_ids.json")
SKIP_KEYS = {"_id", "path", "_variants", "_when"}
CJK = re.compile(r'[一-鿿]')
MARK = re.compile(r'@UUID\[([^\]]*)\](?:\{([^{}]*)\})?')


def iter_leaves():
    for repo in ("1-Ember汉化插件", "2-Crucible汉化插件"):
        en_dir = os.path.join(P, repo, "compendium", "en")
        cn_dir = os.path.join(P, repo, "compendium", "cn")
        if not os.path.isdir(en_dir):
            continue
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            en = json.load(open(os.path.join(en_dir, fn), encoding="utf-8"))
            cp = os.path.join(cn_dir, fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            stack = [(en.get("entries", {}), cn.get("entries", {}), [])]
            while stack:
                e, c, path = stack.pop()
                if isinstance(e, dict):
                    for k, v in e.items():
                        if k in SKIP_KEYS:
                            continue
                        stack.append((v, c.get(k) if isinstance(c, dict) else None, path + [str(k)]))
                elif isinstance(e, list):
                    for i, v in enumerate(e):
                        stack.append((v, c[i] if isinstance(c, list) and i < len(c) else None, path + [str(i)]))
                elif isinstance(e, str):
                    yield repo, fn, ".".join(path), e, (c if isinstance(c, str) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=76)
    a = ap.parse_args()
    swap = json.load(open(SWAP, encoding="utf-8"))["findings"]
    ids = json.load(open(IDS, encoding="utf-8"))
    want = {}
    for i in range(a.lo, a.hi):
        want.setdefault(swap[i]["target"], []).append(i)

    census = defaultdict(lambda: defaultdict(Counter))   # target -> en_label -> Counter(cn)
    cwhere = defaultdict(lambda: defaultdict(list))
    names = defaultdict(Counter)
    nwhere = defaultdict(list)
    en_names = {t: (ids.get(t.split('.')[-1]) or {}).get("name") for t in want}
    wanted_names = {v for v in en_names.values() if v}

    for repo, pack, path, e, c in iter_leaves():
        seg = path.split(".")[-1]
        if seg in ("name", "label") and e in wanted_names and c and CJK.search(c):
            names[e][c] += 1
            nwhere[e].append(f"{pack}:{path}")
        if not c or "@UUID[" not in e:
            continue
        cb = defaultdict(list)
        for m in MARK.finditer(c):
            cb[m.group(1)].append(m.group(2))
        seen = Counter()
        for m in MARK.finditer(e):
            t, el = m.group(1), m.group(2)
            if t not in want:
                seen[t] += 1
                continue
            k = seen[t]
            seen[t] += 1
            cls = cb.get(t) or []
            cl = cls[k] if k < len(cls) else None
            census[t][el][cl] += 1
            cwhere[t][el].append(f"{pack}:{path}")

    for t, idxs in want.items():
        rec = ids.get(t.split('.')[-1]) or {}
        en = en_names[t]
        print("=" * 96)
        print(f"TARGET {t}   findings {idxs}")
        print(f"  EN name : {en!r}   ({rec.get('type')} / {rec.get('via')})")
        if en and names.get(en):
            print(f"  CN name : {names[en].most_common()}")
            print(f"            @ {nwhere[en][:2]}")
        else:
            print(f"  CN name : <no CN name field found>")
        for el, cc in census[t].items():
            print(f"  EN label {{{el}}}:")
            for cl, n in cc.most_common():
                print(f"        {n:4d}  {cl!r}")
            for w in cwhere[t][el][:6]:
                print(f"          @ {w}")


if __name__ == "__main__":
    main()
