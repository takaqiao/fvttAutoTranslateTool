# -*- coding: utf-8 -*-
"""Find *runs* of untranslated English prose inside otherwise-Chinese leaves.

Strips markup, then looks for stretches of >=N consecutive Latin words with no
CJK in between. That is the "整句/整从句未译" class: the leaf has Chinese so
validate_translations (cn==en) and every coverage scan stay silent.
"""
import argparse, json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SKIP = {"_id", "path", "_variants", "_when"}
CJK = "一-鿿"

RX_REF_SPAN = re.compile(r'<span class="reference">.*?</span>', re.S)
RX_ENRICH = re.compile(r'(?:@|&amp;|&)[A-Za-z]+\[[^\]]*\](?:\{([^}]*)\})?')
RX_INLINE = re.compile(r'\[\[.*?\]\]')
RX_TAG = re.compile(r'<[^>]*>')
RX_ENTITY = re.compile(r'&[A-Za-z#0-9]+;')
RX_PRONOUN = re.compile(r'\b(?:he|she|they|it)\s*/\s*(?:him|her|them|its)\b', re.I)
RX_URL = re.compile(r'https?://\S+')
RX_RUN = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]*(?:[ ,.;:!?()\"“”'’\-–—/]+[A-Za-z][A-Za-z0-9'’\-]*)*")


def plain(s):
    s = RX_REF_SPAN.sub(' ', s)
    s = RX_INLINE.sub(' ', s)
    s = RX_ENRICH.sub(lambda m: ' ' + (m.group(1) or '') + ' ', s)
    s = RX_TAG.sub('\n', s)
    s = RX_ENTITY.sub(' ', s)
    s = RX_PRONOUN.sub(' ', s)
    s = RX_URL.sub(' ', s)
    return s


def walk(en, cn, p, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, p + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None, p + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(p), en, cn if isinstance(cn, str) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--pack", default="all")
    ap.add_argument("--words", type=int, default=4, help="min consecutive latin words")
    ap.add_argument("--out")
    a = ap.parse_args()

    found = []
    for repo in a.repo:
        ed = os.path.join(repo, "compendium", "en")
        cd = os.path.join(repo, "compendium", "cn")
        packs = (sorted(f for f in os.listdir(ed) if f.endswith(".json"))
                 if a.pack == "all" else [p.strip() for p in a.pack.split(",")])
        for fn in packs:
            ep = os.path.join(ed, fn)
            if not os.path.isfile(ep):
                continue
            en = json.load(open(ep, encoding="utf-8"))
            cp = os.path.join(cd, fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            rows = []
            walk(en.get("entries", {}), cn.get("entries", {}), [], rows)
            for p, e, c in rows:
                if not c or not re.search('[' + CJK + ']', c):
                    continue
                if 'pronunciation' in p:
                    continue
                t = plain(c)
                for m in RX_RUN.finditer(t):
                    seg = m.group(0)
                    nw = len(re.findall(r"[A-Za-z][A-Za-z'’\-]*", seg))
                    if nw < a.words:
                        continue
                    found.append({"repo": os.path.basename(repo), "pack": fn, "path": p,
                                  "batch_path": p, "words": nw, "run": seg.strip()})
    found.sort(key=lambda r: -r["words"])
    print(f"runs={len(found)}")
    for r in found:
        print(f"{r['words']:3d}  {r['pack']}::{r['path']}")
        print(f"      {r['run'][:220]}")
    if a.out:
        json.dump(found, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
