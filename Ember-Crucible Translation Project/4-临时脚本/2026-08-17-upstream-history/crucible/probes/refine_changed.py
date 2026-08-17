"""Split byte-level 'changed' into metadata-only churn vs real content edits.

Every _source yml embeds a `_stats:` block (coreVersion / systemVersion /
modifiedTime / lastModifiedBy). Upstream runs bulk migration commits
(e.g. "Migrate all compendium YAML to system version 0.9.0") that rewrite
_stats in every file. Counting those as content changes would massively
overstate how much of the game actually changed.

Method: strip every `_stats:` block (and its indented children) plus
`ownership:` blocks, then re-compare.

PRE-VERIFICATION:
  D. the stripper must be a no-op on identical input, must actually remove
     _stats lines, and must leave a known content line intact.
"""
import os, re, sys, json, collections

SCRATCH = r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\289d7a82-7d7b-4b2d-ac68-1439487a5f75\scratchpad"
sys.path.insert(0, SCRATCH)
from diff_source import parse  # reuse the verified parser

BLOCK_RE = re.compile(r"^(\s*)(_stats|ownership):\s*$")


def strip_meta(txt):
    out, skip_indent = [], None
    for line in txt.splitlines():
        if skip_indent is not None:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if stripped and indent > skip_indent:
                continue
            skip_indent = None
        m = BLOCK_RE.match(line)
        if m:
            skip_indent = len(m.group(1))
            continue
        out.append(line)
    return "\n".join(out)


if __name__ == "__main__":
    tag_a, tag_b, outdir = sys.argv[1], sys.argv[2], sys.argv[3]

    # --- PRE-VERIFICATION D ---
    sample = """_id: abc
name: Cleave
system:
  description: hello
_stats:
  coreVersion: '14.363'
  systemVersion: 0.9.4
  modifiedTime: 123
_key: '!items!abc'
"""
    s = strip_meta(sample)
    print("### PRE-VERIFICATION D (stripper) ###")
    c1 = "_stats" not in s
    c2 = "coreVersion" not in s and "modifiedTime" not in s
    c3 = "name: Cleave" in s and "description: hello" in s and "_key: '!items!abc'" in s
    c4 = strip_meta(s) == s  # idempotent
    for lbl, c in (("removes _stats key", c1), ("removes _stats children", c2),
                   ("keeps content lines", c3), ("idempotent", c4)):
        print(f"  {'OK ' if c else 'FAIL'} {lbl}")
    if not all([c1, c2, c3, c4]):
        print("STRIPPER PRE-VERIFICATION FAILED -- aborting.")
        sys.exit(1)
    print("PRE-VERIFICATION D PASSED\n")

    A, _, _ = parse(tag_a)
    B, _, _ = parse(tag_b)
    common = [k for k in (set(A) & set(B)) if A[k]["kind"] == "entry"]

    byte_changed, real_changed, meta_only = [], [], []
    for k in sorted(common):
        if A[k]["text"] == B[k]["text"]:
            continue
        byte_changed.append(k)
        if strip_meta(A[k]["text"]) != strip_meta(B[k]["text"]):
            real_changed.append(k)
        else:
            meta_only.append(k)

    print(f"### {tag_a} -> {tag_b} (entries only, folders excluded) ###")
    print(f"  common entries      : {len(common)}")
    print(f"  byte-level changed  : {len(byte_changed)}")
    print(f"  REAL content changed: {len(real_changed)}")
    print(f"  metadata-only churn : {len(meta_only)}")
    print("\n  real-changed by pack:", dict(collections.Counter(p for p, _ in real_changed)))
    print("  meta-only  by pack:", dict(collections.Counter(p for p, _ in meta_only)))

    res = {
        "tag_a": tag_a, "tag_b": tag_b,
        "common_entries": len(common),
        "byte_changed": len(byte_changed),
        "real_changed": [{"pack": p, "id": i, "name": B[(p, i)]["name"]} for p, i in real_changed],
        "meta_only": [{"pack": p, "id": i, "name": B[(p, i)]["name"]} for p, i in meta_only],
    }
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f"refined_{tag_a}__{tag_b}.json")
    json.dump(res, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  written -> {fn}")
