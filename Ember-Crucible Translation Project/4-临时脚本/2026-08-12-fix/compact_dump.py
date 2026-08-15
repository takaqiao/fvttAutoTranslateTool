# -*- coding: utf-8 -*-
"""Compact EN/CN pair dump: one row per leaf, `path`\\n  EN: ...\\n  CN: ...`"""
import argparse, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKIP_KEYS = {"_id", "path", "_variants", "_when"}


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            sub = cn.get(k) if isinstance(cn, dict) else None
            walk(v, sub, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        p = ".".join(path)
        out.append((p[len("entries."):] if p.startswith("entries.") else p, en,
                    cn if isinstance(cn, str) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--root", default="entries")
    ap.add_argument("--grep-path")
    ap.add_argument("--grep-en")
    ap.add_argument("--grep-cn")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    en = json.load(open(os.path.join(a.repo, "compendium", "en", a.pack), encoding="utf-8"))
    cnp = os.path.join(a.repo, "compendium", "cn", a.pack)
    cn = json.load(open(cnp, encoding="utf-8")) if os.path.isfile(cnp) else {}
    rows = []
    walk(en.get(a.root, {}), cn.get(a.root, {}), [a.root], rows)
    if a.grep_path:
        rx = re.compile(a.grep_path); rows = [r for r in rows if rx.search(r[0])]
    if a.grep_en:
        rx = re.compile(a.grep_en); rows = [r for r in rows if rx.search(r[1])]
    if a.grep_cn:
        rx = re.compile(a.grep_cn); rows = [r for r in rows if r[2] and rx.search(r[2])]
    lines = []
    for p, e, c in rows:
        if a.max and len(e) > a.max:
            e = e[:a.max] + f"…<+{len(e)-a.max}>"
        if a.max and c and len(c) > a.max:
            c = c[:a.max] + f"…<+{len(c)-a.max}>"
        lines.append(f"### {p}\nEN| {e}\nCN| {c}")
    txt = "\n".join(lines) + f"\n--- {len(rows)} rows\n"
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt)
        print(f"{len(rows)} rows -> {a.out}")
    else:
        print(txt)


if __name__ == "__main__":
    main()
