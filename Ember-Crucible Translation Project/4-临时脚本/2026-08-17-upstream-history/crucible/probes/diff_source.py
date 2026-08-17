"""Content-level diff of crucible _source between two git tags.

Identity key = (pack, document _id) parsed from the filename suffix.
Display name = first column-0 `name:` line inside the YAML.

PRE-VERIFICATION (must pass before any diff is reported):
  A. per-pack file counts at release-0.10.2 must equal the independently
     obtained `ls _source/<pack> | wc -l` truth table below.
  B. every file must yield a parseable 16-char id and a column-0 `name:`.
"""
import subprocess, sys, re, os, json, collections

SCRATCH = r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\289d7a82-7d7b-4b2d-ac68-1439487a5f75\scratchpad"
REPO = os.path.join(SCRATCH, "crucible-src")

# independently obtained via `ls _source/<d> | wc -l` on a checked-out release-0.10.2
TRUTH_1002 = {
    "adversary-equipment": 59, "adversary-talents": 95, "affixes": 157,
    "ancestry": 8, "archetype": 50, "background": 9, "equipment": 197,
    "macros": 5, "playtest": 1, "pregens": 18, "rules": 13, "spell": 21,
    "summons": 13, "talent": 377, "taxonomy": 38,
}
# independent second truth source: `documents` per pack recorded by our own
# extractor in english-baseline/crucible-0.10.2/_source.json (a completely
# separate artifact, produced from the installed system's LevelDB packs).
TRUTH_ENTRIES_1002 = {
    "adversary-equipment": 53, "adversary-talents": 76, "affixes": 135,
    "ancestry": 8, "archetype": 45, "background": 9, "equipment": 166,
    "macros": 5, "playtest": 1, "pregens": 16, "rules": 11, "spell": 14,
    "summons": 10, "talent": 325, "taxonomy": 28,
}
ID_RE = re.compile(r"_([A-Za-z0-9]{16})\.yml$")


def worktree(tag):
    wt = os.path.join(SCRATCH, "wt-" + tag)
    if not os.path.isdir(wt):
        subprocess.run(["git", "-C", REPO, "worktree", "add", "--detach", "--quiet", wt, tag],
                       check=True, capture_output=True, text=True)
    return wt


def parse(tag):
    """-> ({(pack,id): {file,name,text}}, bad_id[], bad_name[])"""
    wt = worktree(tag)
    root = os.path.join(wt, "_source")
    recs, bad_id, bad_name = {}, [], []
    for pack in sorted(os.listdir(root)):
        pdir = os.path.join(root, pack)
        if not os.path.isdir(pdir):
            continue
        for dirpath, _, files in os.walk(pdir):
            for fn in files:
                if not fn.endswith(".yml"):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root).replace("\\", "/")
                txt = open(full, encoding="utf-8", errors="replace").read()
                lines = txt.splitlines()
                # authoritative identity: the column-0 `_id:` field in the YAML
                ids = [l[5:].strip().strip("'\"") for l in lines if l.startswith("_id: ")]
                names = [l[6:].strip().strip("'\"") for l in lines if l.startswith("name: ")]
                if not ids:
                    bad_id.append(rel)
                    continue
                if not names:
                    bad_name.append(rel)
                keys = [l[6:].strip().strip("'\"") for l in lines if l.startswith("_key: ")]
                kind = "folder" if (keys and keys[0].startswith("!folders!")) else "entry"
                recs[(pack, ids[0])] = {
                    "file": rel, "name": names[0] if names else "(?)",
                    "kind": kind, "text": txt}
    return recs, bad_id, bad_name


def counts(recs):
    return dict(collections.Counter(p for (p, _) in recs))


if __name__ == "__main__":
    tag_a, tag_b, outdir = sys.argv[1], sys.argv[2], sys.argv[3]

    print("### PRE-VERIFICATION ###")
    ref, bad_id_r, bad_name_r = parse("release-0.10.2")
    cr = counts(ref)
    ok = True
    for k, v in sorted(TRUTH_1002.items()):
        got = cr.get(k, 0)
        if got != v:
            ok = False
        print(f"  A {'OK ' if got==v else 'MISMATCH'} {k}: got={got} truth={v}")
    tg, tt = sum(cr.values()), sum(TRUTH_1002.values())
    print(f"  A total: got={tg} truth={tt} -> {'OK' if tg==tt else 'MISMATCH'}")
    if tg != tt:
        ok = False
    print(f"  B unparseable id: {len(bad_id_r)} {bad_id_r[:5]}")
    print(f"  B missing col-0 name: {len(bad_name_r)} {bad_name_r[:5]}")
    if bad_id_r or bad_name_r:
        ok = False
    # C: entry-only counts must match our extractor's independent `documents`
    ce = collections.Counter(p for (p, i), r in ref.items() if r["kind"] == "entry")
    for k, v in sorted(TRUTH_ENTRIES_1002.items()):
        got = ce.get(k, 0)
        if got != v:
            ok = False
        print(f"  C {'OK ' if got==v else 'MISMATCH'} entries {k}: got={got} truth={v}")
    tge, tte = sum(ce.values()), sum(TRUTH_ENTRIES_1002.values())
    print(f"  C total entries: got={tge} truth={tte} -> {'OK' if tge==tte else 'MISMATCH'}")
    if tge != tte:
        ok = False
    if not ok:
        print("PRE-VERIFICATION FAILED -- aborting, no diff reported.")
        sys.exit(1)
    print("PRE-VERIFICATION PASSED\n")

    A = ref if tag_a == "release-0.10.2" else parse(tag_a)[0]
    B = ref if tag_b == "release-0.10.2" else parse(tag_b)[0]
    print(f"  side A {tag_a}: {len(A)} docs")
    print(f"  side B {tag_b}: {len(B)} docs")

    ka, kb = set(A), set(B)
    added, removed = sorted(kb - ka), sorted(ka - kb)
    common = ka & kb
    renamed = sorted(k for k in common if A[k]["name"] != B[k]["name"])
    changed = sorted(k for k in common if A[k]["text"] != B[k]["text"])

    res = {
        "tag_a": tag_a, "tag_b": tag_b, "count_a": len(A), "count_b": len(B),
        "added": [{"pack": p, "id": i, "name": B[(p, i)]["name"], "kind": B[(p, i)]["kind"]} for p, i in added],
        "removed": [{"pack": p, "id": i, "name": A[(p, i)]["name"], "kind": A[(p, i)]["kind"]} for p, i in removed],
        "renamed": [{"pack": p, "id": i, "from": A[(p, i)]["name"], "to": B[(p, i)]["name"], "kind": B[(p, i)]["kind"]} for p, i in renamed],
        "changed": [{"pack": p, "id": i, "name": B[(p, i)]["name"], "kind": B[(p, i)]["kind"]} for p, i in changed],
    }

    def by_kind(keys, side, kind):
        return [k for k in keys if side[k]["kind"] == kind]
    print("\n### ENTRY-ONLY (folders excluded) ###")
    for label, keys, side in (("added", added, B), ("removed", removed, A),
                              ("renamed", renamed, B), ("changed", changed, B)):
        e = by_kind(keys, side, "entry")
        f = by_kind(keys, side, "folder")
        print(f"  {label:8} entries={len(e):4} folders={len(f):3}")
        print(f"           by pack: {dict(collections.Counter(p for p,_ in e))}")
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f"diff_{tag_a}__{tag_b}.json")
    json.dump(res, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n### DIFF {tag_a} -> {tag_b} ###")
    print(f"  docs: {len(A)} -> {len(B)}")
    print(f"  added={len(added)} removed={len(removed)} renamed={len(renamed)} content-changed={len(changed)}")
    print("  added by pack:  ", dict(collections.Counter(p for p, _ in added)))
    print("  removed by pack:", dict(collections.Counter(p for p, _ in removed)))
    print("  renamed by pack:", dict(collections.Counter(p for p, _ in renamed)))
    print("  changed by pack:", dict(collections.Counter(p for p, _ in changed)))
    print(f"\n  written -> {fn}")
