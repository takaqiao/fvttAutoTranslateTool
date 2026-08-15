# -*- coding: utf-8 -*-
"""Extract the <sub data-system="..."> branch bodies from EN/CN pairs and flag
the ones whose Chinese body still has Latin prose (i.e. the branch was skipped).

Crucible worlds render ONLY the data-system="crucible" branch, so an untranslated
crucible branch is what the player actually reads.
"""
import argparse, json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SKIP_KEYS = {"_id", "path", "_variants", "_when"}
CJK = r"一-鿿"

RX_SUB = re.compile(r'<sub data-system="([^"]+)">(.*?)</sub>', re.S)
RX_ENRICH = re.compile(r'(?:@|&amp;|&)[A-Za-z]+\[[^\]]*\](?:\{([^}]*)\})?')
RX_INLINE = re.compile(r'\[\[.*?\]\]')
RX_TAG = re.compile(r'<[^>]*>')
RX_ENTITY = re.compile(r'&[A-Za-z#0-9]+;')
RX_WORD = re.compile(r'[A-Za-z][A-Za-z\'’-]{2,}')


def plain(s):
    s = RX_INLINE.sub(' ', s)
    s = RX_ENRICH.sub(lambda m: ' ' + (m.group(1) or '') + ' ', s)
    s = RX_TAG.sub(' ', s)
    s = RX_ENTITY.sub(' ', s)
    return s


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
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--pack", default="all")
    ap.add_argument("--system", default="crucible")
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = []
    for repo in a.repo:
        en_dir = os.path.join(repo, "compendium", "en")
        cn_dir = os.path.join(repo, "compendium", "cn")
        packs = (sorted(f for f in os.listdir(en_dir)
                        if f.endswith(".json") and not f.startswith("_"))
                 if a.pack == "all" else [p.strip() for p in a.pack.split(",")])
        for fn in packs:
            ep = os.path.join(en_dir, fn)
            if not os.path.isfile(ep):
                continue
            en = json.load(open(ep, encoding="utf-8"))
            cp = os.path.join(cn_dir, fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            sub = []
            walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], sub)
            for p, e, c in sub:
                rows.append((os.path.basename(repo), fn, p, e, c))

    findings = []
    tot = 0
    for repo, fn, p, e, c in rows:
        if not c:
            continue
        ens = [m for m in RX_SUB.finditer(e) if m.group(1) == a.system]
        cns = [m for m in RX_SUB.finditer(c) if m.group(1) == a.system]
        if not cns:
            continue
        bad = []
        for i, m in enumerate(cns):
            body = plain(m.group(2))
            words = [w for w in RX_WORD.findall(body)]
            if not words:
                continue
            # untranslated: has latin words and (no CJK at all, or matches EN body)
            enbody = plain(ens[i].group(2)) if i < len(ens) else ""
            has_cjk = bool(re.search('[' + CJK + ']', body))
            bad.append({"i": i, "cn_body": m.group(2), "en_body": ens[i].group(2) if i < len(ens) else "",
                        "has_cjk": has_cjk, "same_as_en": body.split() == enbody.split(),
                        "words": words})
        if bad:
            tot += len(bad)
            findings.append({"repo": repo, "pack": fn, "path": p,
                             "batch_path": p[len("entries."):] if p.startswith("entries.") else p,
                             "n": len(bad), "bad": bad})
    findings.sort(key=lambda r: -r["n"])
    print(f"leaves={len(findings)} bad_branches={tot}")
    for f in findings:
        print(f"{f['n']:3d} {f['pack']}::{f['path']}")
        for b in f["bad"][:6]:
            print(f"     cjk={b['has_cjk']} same={b['same_as_en']}  CN[{b['i']}]: {b['cn_body'][:180]}")
    if a.out:
        json.dump(findings, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("->", a.out)


if __name__ == "__main__":
    main()
