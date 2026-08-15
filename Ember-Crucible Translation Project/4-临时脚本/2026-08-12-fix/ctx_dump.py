# -*- coding: utf-8 -*-
"""Print EN/CN context windows around a regex match inside specific leaves."""
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
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--path", required=True, help="regex on path")
    ap.add_argument("--rx", help="regex to centre on (CN side); omit to print whole leaf")
    ap.add_argument("--before", type=int, default=200)
    ap.add_argument("--after", type=int, default=260)
    a = ap.parse_args()

    en = json.load(open(os.path.join(a.repo, "compendium", "en", a.pack), encoding="utf-8"))
    cp = os.path.join(a.repo, "compendium", "cn", a.pack)
    cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
    rows = []
    walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], rows)
    prx = re.compile(a.path)
    for p, e, c in rows:
        if not prx.search(p):
            continue
        print("#" * 10, p)
        if not a.rx:
            print("EN:", e)
            print("CN:", c)
            continue
        rx = re.compile(a.rx, re.S)
        for m in rx.finditer(c):
            print("--- CN ...", c[max(0, m.start() - a.before): m.end() + a.after], "...")
        for m in rx.finditer(e):
            print("=== EN ...", e[max(0, m.start() - a.before): m.end() + a.after], "...")


if __name__ == "__main__":
    main()
