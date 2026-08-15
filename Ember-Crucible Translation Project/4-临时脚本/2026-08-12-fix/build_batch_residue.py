# -*- coding: utf-8 -*-
"""Build an apply_translations batch by applying *verified* surgical string
replacements to the existing Chinese leaves.  (residue-fix agent; do not reuse
the generic name `build_batch.py` -- another agent owns that filename.)

Rules file: JSON list of
  {"pack": "...", "path": "<batch_path>|*", "old": "...", "new": "...",
   "count": <expected number of replacements>,
   "en_contains": "<optional English gate>", "note": "..."}

`path` may be "*" to apply pack-wide (then `count` is the pack-wide total).
`en_contains` restricts a rule to leaves whose ENGLISH source contains that
string -- the mechanical form of PROJECT.md's "先查英文再落笔".
Every rule must hit exactly `count` times or the build fails loudly.
Nothing is written to compendium/cn -- output is a batch file only.
"""
import argparse, json, os, sys


def all_leaves(node, prefix, out):
    if isinstance(node, dict):
        for k, v in node.items():
            all_leaves(v, prefix + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            all_leaves(v, prefix + [str(i)], out)
    elif isinstance(node, str):
        out[".".join(prefix)] = node


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--rules", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pack", help="only build rules for this pack")
    a = ap.parse_args()

    rules = []
    for rf in a.rules:
        rules.extend(json.load(open(rf, encoding="utf-8")))
    packs = {}
    for r in rules:
        if a.pack and r["pack"] != a.pack:
            continue
        packs.setdefault(r["pack"], []).append(r)

    ok = True
    for pack, rs in packs.items():
        cn = json.load(open(os.path.join(a.repo, "compendium", "cn", pack), encoding="utf-8"))
        en = json.load(open(os.path.join(a.repo, "compendium", "en", pack), encoding="utf-8"))
        leaves, en_leaves = {}, {}
        all_leaves(cn.get("entries", {}), [], leaves)
        all_leaves(en.get("entries", {}), [], en_leaves)
        work = dict(leaves)
        touched = {}
        for r in rs:
            targets = [r["path"]] if r["path"] != "*" else list(work.keys())
            gate = r.get("en_contains")
            n = 0
            for p in targets:
                if p not in work:
                    print(f"!! MISSING PATH  {pack} :: {p}")
                    ok = False
                    continue
                if gate is not None and gate not in en_leaves.get(p, ""):
                    continue
                v = work[p]
                c = v.count(r["old"])
                if c:
                    work[p] = v.replace(r["old"], r["new"])
                    touched[p] = True
                    n += c
            if r["count"] == -1:
                # "-1" = twin-pack mode: the exact `old` string was already
                # proven on the primary pack; here just require >0 hits.
                if n == 0:
                    print(f"!! NO HIT  {pack} :: {r['old'][:70]!r}")
                    ok = False
                else:
                    print(f"ok  {n:3d}  {pack} :: {r.get('note') or r['old'][:60]!r}")
            elif n != r["count"]:
                print(f"!! COUNT MISMATCH  {pack} :: {r['path']} :: "
                      f"expected {r['count']} got {n}  old={r['old'][:80]!r}")
                ok = False
            else:
                print(f"ok  {n:3d}  {pack} :: {r.get('note') or r['old'][:60]!r}")
        batch = {p: work[p] for p in touched if work[p] != leaves[p]}
        outp = a.out if len(packs) == 1 else a.out.replace(".json", f".{pack}")
        json.dump(batch, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"--> {outp}  entries={len(batch)}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
