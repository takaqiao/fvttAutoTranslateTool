# -*- coding: utf-8 -*-
"""CN-hanzi / EN-plaintext ratio scan, restricted to a journal whitelist.

Library median in this repo is ~0.31.  >0.44 => candidate for invented content,
<0.20 => candidate omission.  The ratio is only a *candidate filter*; the verdict
always requires reading the English.

  python ratio_scan.py --repo <repo> --pack <pack.json> --journals <file with one name per line>
"""
import argparse, json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKIP_KEYS = {"_id", "path", "_variants", "_when"}
CJK = re.compile(r"[一-鿿]")
TAG = re.compile(r"<[^>]+>")
MARKUP_TARGET = re.compile(r"@[A-Za-z]+\[[^\]]*\]")
INLINE = re.compile(r"\[\[[^\]]*\]\]")
REFERENCE = re.compile(r"&(?:amp;)?[Rr]eference\[[^\]]*\]")


def plain(s):
    s = MARKUP_TARGET.sub(" ", s)
    s = INLINE.sub(" ", s)
    s = REFERENCE.sub(" ", s)
    s = TAG.sub(" ", s)
    s = s.replace("&amp;", "&").replace("&nbsp;", " ")
    # drop the {label} braces that survive after target removal
    s = re.sub(r"[{}]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--journals")
    ap.add_argument("--min-en", type=int, default=200)
    ap.add_argument("--hi", type=float, default=0.44)
    ap.add_argument("--lo", type=float, default=0.20)
    ap.add_argument("--out")
    a = ap.parse_args()

    names = None
    if a.journals:
        names = set(l.strip() for l in open(a.journals, encoding="utf-8") if l.strip())

    en = json.load(open(os.path.join(a.repo, "compendium", "en", a.pack), encoding="utf-8"))
    cn = json.load(open(os.path.join(a.repo, "compendium", "cn", a.pack), encoding="utf-8"))
    rows = []
    walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], rows)

    recs = []
    for p, e, c in rows:
        m = re.match(r"entries\.([^.]+)\.journals\.(.+?)\.pages\.", p)
        if not m:
            continue
        jname = m.group(2)
        if names is not None and jname not in names:
            continue
        pe = plain(e)
        if len(pe) < a.min_en:
            continue
        nz = len(CJK.findall(c))
        r = nz / len(pe)
        recs.append({"journal": jname, "path": p,
                     "batch_path": p[len("entries."):],
                     "en_chars": len(pe), "cn_hanzi": nz, "ratio": round(r, 3)})

    recs.sort(key=lambda x: -x["ratio"])
    hi = [r for r in recs if r["ratio"] > a.hi]
    lo = [r for r in recs if r["ratio"] < a.lo]
    med = sorted(r["ratio"] for r in recs)
    mid = med[len(med) // 2] if med else 0
    print(f"leaves considered: {len(recs)}  median ratio {mid}")
    print(f"HIGH (>{a.hi}): {len(hi)}     LOW (<{a.lo}): {len(lo)}")
    for r in hi:
        print(f"  HI {r['ratio']:.3f}  en={r['en_chars']:<6} cn={r['cn_hanzi']:<6} {r['path']}")
    for r in lo:
        print(f"  LO {r['ratio']:.3f}  en={r['en_chars']:<6} cn={r['cn_hanzi']:<6} {r['path']}")
    if a.out:
        json.dump({"all": recs, "hi": hi, "lo": lo},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
