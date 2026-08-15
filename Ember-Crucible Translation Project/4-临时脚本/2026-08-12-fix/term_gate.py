# -*- coding: utf-8 -*-
"""English-gated term counting.

PROJECT.md hard rule: a bare Chinese word-frequency count both over- and
under-reports. Always split the count by what the *paired English* actually says.

  python term_gate.py --repo "<repoDir>" --cn 闪电 --en "\\bLightning\\b"
  python term_gate.py --repo "<repoDir>" --repo "<repoDir2>" --en "\\bElectricity\\b" --cn 电击,电能,电力
  python term_gate.py --repo "<repoDir>" --en "\\bBoon\\b" --show 5

Output per (cn-term):
  gated_hit   : rows where EN matches --en AND CN contains the term  (the real signal)
  cn_only     : rows where CN contains the term but EN does NOT match (usually a
                different English word -> NOT a residue; check before "fixing")
  en_only     : rows where EN matches but CN uses none of the given cn terms
                (candidate omission / competing rendering)
"""
import argparse, json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKIP_KEYS = {"_id", "path", "_variants", "_when"}


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
    ap.add_argument("--en", required=True, help="regex on English side")
    ap.add_argument("--cn", required=True, help="comma-separated Chinese renderings")
    ap.add_argument("--ignore-case", action="store_true")
    ap.add_argument("--show", type=int, default=3, help="sample rows per bucket")
    ap.add_argument("--trunc", type=int, default=260)
    a = ap.parse_args()

    rx = re.compile(a.en, re.I if a.ignore_case else 0)
    terms = [t.strip() for t in a.cn.split(",") if t.strip()]

    rows = []
    for repo in a.repo:
        en_dir = os.path.join(repo, "compendium", "en")
        cn_dir = os.path.join(repo, "compendium", "cn")
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            en = json.load(open(os.path.join(en_dir, fn), encoding="utf-8"))
            cp = os.path.join(cn_dir, fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            sub = []
            walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], sub)
            for p, e, c in sub:
                rows.append((os.path.basename(repo), fn, p, e, c))

    def clip(s):
        return s[: a.trunc] + ("…" if len(s) > a.trunc else "")

    print(f"scanned leaves: {len(rows)}   en-regex: {a.en!r}")
    en_match = [r for r in rows if rx.search(r[3])]
    print(f"rows whose ENGLISH matches: {len(en_match)}")
    print("-" * 78)

    for t in terms:
        gated, cn_only = [], []
        for r in rows:
            if t in (r[4] or ""):
                (gated if rx.search(r[3]) else cn_only).append(r)
        print(f"CN {t!r}:  gated_hit={len(gated)}   cn_only(different English)={len(cn_only)}")
        for r in cn_only[: a.show]:
            print(f"    [cn_only] {r[0]}/{r[1]}::{r[2]}")
            print(f"        EN: {clip(r[3])}")
            print(f"        CN: {clip(r[4])}")
    print("-" * 78)

    en_only = [r for r in en_match if r[4] and not any(t in r[4] for t in terms)]
    print(f"EN matches but CN uses none of {terms}: {len(en_only)}")
    for r in en_only[: a.show * 3]:
        print(f"    [en_only] {r[0]}/{r[1]}::{r[2]}")
        print(f"        EN: {clip(r[3])}")
        print(f"        CN: {clip(r[4])}")

    no_cn = [r for r in en_match if not r[4]]
    print(f"EN matches but NO Chinese at that path at all: {len(no_cn)}")
    for r in no_cn[: a.show]:
        print(f"    [no_cn] {r[0]}/{r[1]}::{r[2]}")
        print(f"        EN: {clip(r[3])}")


if __name__ == "__main__":
    main()
