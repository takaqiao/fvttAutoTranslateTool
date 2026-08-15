# -*- coding: utf-8 -*-
"""Build an apply_translations batch from surgical edits on the CURRENT Chinese.

Spec: JSON list of
  {"path": "<batch_path>", "reps": [["old","new"], ...]}   # substring replacements
  {"path": "<batch_path>", "set": "<full new value>"}      # whole-value replacement

Editing the existing Chinese (instead of retyping it) is what keeps the markup
multiset identical, which is exactly what apply_translations' third gate checks.
Every `old` must be present exactly once unless a count is given as a 3rd element.
"""
import argparse, json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def get_at(node, parts):
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p)
        elif isinstance(node, list):
            try:
                node = node[int(p)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def split_path(root, path):
    naive = path.split('.')
    if get_at(root, naive) is not None:
        return naive
    parts, node, rest = [], root, path
    while rest:
        if isinstance(node, dict):
            cands = [k for k in node if rest == k or rest.startswith(k + '.')]
            if cands:
                k = max(cands, key=len)
                parts.append(k)
                node = node[k]
                rest = rest[len(k) + 1:]
                continue
        head, _, rest = rest.partition('.')
        parts.append(head)
        if isinstance(node, list):
            try:
                node = node[int(head)]
            except (ValueError, IndexError):
                node = None
        elif isinstance(node, dict):
            node = node.get(head)
        else:
            node = None
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cn = json.load(open(os.path.join(a.repo, "compendium", "cn", a.pack), encoding="utf-8"))
    root = cn.get("entries", {})
    spec = json.load(open(a.spec, encoding="utf-8"))
    out, errs = {}, []
    for item in spec:
        p = item["path"]
        cur = get_at(root, split_path(root, p))
        if "set" in item:
            if not isinstance(cur, str):
                errs.append(f"{p}: no current CN (set anyway)")
            out[p] = item["set"]
            continue
        if not isinstance(cur, str):
            errs.append(f"{p}: NO CURRENT CN STRING")
            continue
        v = cur
        for rep in item["reps"]:
            old, new = rep[0], rep[1]
            want = rep[2] if len(rep) > 2 else 1
            n = v.count(old)
            if n != want:
                errs.append(f"{p}: {old!r} occurs {n}x, expected {want}x")
                continue
            v = v.replace(old, new)
        if v == cur:
            errs.append(f"{p}: NO CHANGE")
            continue
        out[p] = v
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"batch entries: {len(out)} -> {a.out}")
    for e in errs:
        print("  ERR " + e)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
