# -*- coding: utf-8 -*-
"""Detect and 3-way-merge conflicts across the repair batches.

`apply_translations.py` writes a whole leaf, so two batches that touch the same
path are last-write-wins -- the earlier one is silently rolled back. All 60
batches were generated in parallel from the SAME base, so ordering alone cannot
save us: the edits have to be composed.

For every path claimed by >1 batch:
  base = the value currently in compendium/cn
  A, B, ... = each batch's proposed full-leaf replacement
Compute each batch's edit as a diff against base, then replay all edits onto
base. Non-overlapping edits compose cleanly (AUTO). Overlapping ones are
reported for a human decision (CONFLICT) and are NOT written.

  python merge_batches.py --scan            # conflict census only
  python merge_batches.py --merge --out-dir merged/   # write merged batches
"""
import argparse, difflib, json, os, sys, collections

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
W = os.path.join(P, "4-临时脚本", "2026-08-12-fix")
REPOS = {"1-Ember汉化插件", "2-Crucible汉化插件"}


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def norm_repo(s):
    s = (s or "").replace("\\", "/").rstrip("/")
    return s.split("/")[-1]


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
    naive = path.split(".")
    if get_at(root, naive) is not None:
        return naive
    parts, node, rest = [], root, path
    while rest:
        if isinstance(node, dict):
            cands = [k for k in node if rest == k or rest.startswith(k + ".")]
            if cands:
                k = max(cands, key=len)
                parts.append(k)
                node = node[k]
                rest = rest[len(k) + 1:]
                continue
        head, _, rest = rest.partition(".")
        parts.append(head)
        node = get_at(node, [head])
    return parts


def collect(manifest):
    """-> {(repo, pack): {path: [(batchname, value), ...]}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for b in manifest:
        if b["kind"] != "translations":
            continue
        f = b["file"]
        if not os.path.isfile(f):
            print(f"  !! missing batch file: {f}")
            continue
        repo, pack = norm_repo(b["repo"]), b["pack"]
        if repo not in REPOS:
            continue
        data = load(f)
        items = data.get("items", data)
        name = os.path.basename(f)
        for path, val in items.items():
            if isinstance(val, str):
                out[(repo, pack)][path].append((name, val))
    return out


def three_way(base, variants):
    """Replay each variant's line-level edit onto base.

    Returns (merged, status). status is 'auto' when every edit applied to a
    region no other variant touched, 'conflict' otherwise.
    """
    if len(variants) == 1:
        return variants[0][1], "single"
    vals = {v for _, v in variants}
    if len(vals) == 1:
        return variants[0][1], "identical"
    if base is None:
        return None, "conflict"

    # Character-level opcode replay: batches rewrite prose, so line diffs are
    # too coarse -- one <p> is usually one line.
    edits = []          # (i1, i2, replacement, batchname)
    for name, val in variants:
        sm = difflib.SequenceMatcher(None, base, val, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                edits.append((i1, i2, val[j1:j2], name))
    edits.sort(key=lambda e: (e[0], e[1]))

    # overlap check on base spans
    for a, b in zip(edits, edits[1:]):
        if b[0] < a[1]:                       # spans intersect
            return None, "conflict"
        if b[0] == a[1] and a[0] == a[1] and b[0] == b[1]:
            return None, "conflict"           # two inserts at same point

    merged, cur = [], 0
    for i1, i2, rep, _ in edits:
        merged.append(base[cur:i1])
        merged.append(rep)
        cur = i2
    merged.append(base[cur:])
    return "".join(merged), "auto"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(W, "manifest.json"))
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(W, "merged"))
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--resolutions", default=os.path.join(W, "resolutions.json"))
    a = ap.parse_args()

    manifest = load(a.manifest)
    RES = load(a.resolutions) if os.path.isfile(a.resolutions) else {}
    groups = collect(manifest)

    tot_paths = tot_multi = tot_auto = tot_ident = tot_conflict = tot_resolved = 0
    conflicts = []
    merged_out = collections.defaultdict(dict)

    for (repo, pack), paths in sorted(groups.items()):
        cn_path = os.path.join(P, repo, "compendium", "cn", pack)
        cn = load(cn_path) if os.path.isfile(cn_path) else {"entries": {}}
        root = cn.get("entries", {})
        n_multi = n_auto = n_ident = n_conf = n_resolved = 0
        for path, variants in paths.items():
            tot_paths += 1
            if len(variants) > 1:
                n_multi += 1
                tot_multi += 1
            base = get_at(root, split_path(root, path))
            base = base if isinstance(base, str) else None
            merged, status = three_way(base, variants)
            if status == "identical":
                n_ident += 1
                tot_ident += 1
            elif status == "auto":
                n_auto += 1
                tot_auto += 1
            elif status == "conflict":
                n_conf += 1
                idx = str(len(conflicts))
                conflicts.append({
                    "repo": repo, "pack": pack, "path": path,
                    "base": base, "variants": [{"batch": n, "value": v} for n, v in variants],
                })
                r = RES.get(idx)
                if not r:
                    tot_conflict += 1
                    continue
                chosen = next((v for n, v in variants if n == r["take"]), None)
                if chosen is None:
                    raise SystemExit(f"resolution #{idx}: no variant named {r['take']!r} "
                                     f"(have {[n for n, _ in variants]})")
                for old, new in r.get("then", []):
                    if old not in chosen:
                        raise SystemExit(f"resolution #{idx}: replacement {old!r} not found "
                                         f"in chosen value -- stale resolution, re-check")
                    chosen = chosen.replace(old, new)
                merged = chosen
                n_resolved += 1
                tot_resolved += 1
            merged_out[(repo, pack)][path] = merged
        if n_multi:
            print(f"{repo:<22} {pack:<36} paths={len(paths):<5} multi={n_multi:<4} "
                  f"identical={n_ident:<4} auto-merged={n_auto:<4} conflict={n_conf} resolved={n_resolved}")

    print("-" * 100)
    print(f"total paths {tot_paths} | claimed by >1 batch {tot_multi} | "
          f"identical {tot_ident} | auto-merged {tot_auto} | resolved-by-hand {tot_resolved} | UNRESOLVED {tot_conflict}")

    if tot_conflict:
        cp = os.path.join(W, "reports", "batch_conflicts.json")
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(conflicts, f, ensure_ascii=False, indent=1)
        print(f"conflicts -> {cp}")
        for c in conflicts[: a.show]:
            print(f"\n  CONFLICT {c['repo']}/{c['pack']}::{c['path']}")
            for v in c["variants"]:
                print(f"    [{v['batch']}] {v['value'][:220]}")

    if a.merge:
        os.makedirs(a.out_dir, exist_ok=True)
        idx = []
        for (repo, pack), items in sorted(merged_out.items()):
            fn = os.path.join(a.out_dir, f"{repo}__{pack}")
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=1)
            idx.append({"file": fn, "repo": repo, "pack": pack, "entries": len(items)})
            print(f"merged {len(items):>5}  -> {fn}")
        with open(os.path.join(a.out_dir, "_index.json"), "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
