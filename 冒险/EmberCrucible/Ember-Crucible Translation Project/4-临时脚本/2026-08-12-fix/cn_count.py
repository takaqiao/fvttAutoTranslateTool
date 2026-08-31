# -*- coding: utf-8 -*-
"""Count occurrences of Chinese strings across all CN leaf text (both repos)."""
import argparse, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SKIP_KEYS = {"_id", "path", "_variants", "_when"}


def leaves(o, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in SKIP_KEYS:
                continue
            leaves(v, out)
    elif isinstance(o, list):
        for v in o:
            leaves(v, out)
    elif isinstance(o, str):
        out.append(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("terms", nargs="+")
    a = ap.parse_args()
    texts = []
    for repo in a.repo:
        cn_dir = os.path.join(repo, "compendium", "cn")
        for fn in sorted(os.listdir(cn_dir)):
            if not fn.endswith(".json"):
                continue
            leaves(json.load(open(os.path.join(cn_dir, fn), encoding="utf-8")).get("entries", {}), texts)
    blob = "\n".join(texts)
    for t in a.terms:
        print(f"{blob.count(t):6d}  {t}")


if __name__ == "__main__":
    main()
