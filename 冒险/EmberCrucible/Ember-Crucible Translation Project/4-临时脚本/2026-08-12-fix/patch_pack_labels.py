# -*- coding: utf-8 -*-
"""Patch the TOP-LEVEL `label` of Babele pack files.

`qa/apply_translations.py` batches address paths relative to `entries` (or
`(folders)`), so the pack's own `label` -- the name Foundry prints on the
compendium sidebar -- is unreachable from a batch. `pair_dump.py` /
`validate_translations.py` do not walk it either, which is why 7 of them were
still English after "coverage 100%".

Rules live in rules/pack_labels.json:  {"<repo>": {"<pack.json>": "<new label>"}}
Every rule is verified against the English pack (the rule must name a pack that
exists and whose current CN label is what the rule expects to replace).

  python patch_pack_labels.py --dry     # print the diff, write nothing
  python patch_pack_labels.py --write   # apply
"""
import argparse, json, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=os.path.join(HERE, "rules", "pack_labels.json"))
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if not (a.write or a.dry):
        a.dry = True

    rules = json.load(open(a.rules, encoding="utf-8"))
    changed = problems = 0
    for repo, packs in rules.items():
        for pack, new in packs.items():
            enp = os.path.join(a.root, repo, "compendium", "en", pack)
            cnp = os.path.join(a.root, repo, "compendium", "cn", pack)
            if not os.path.isfile(enp):
                print(f"  MISSING EN  {repo}/{pack}")
                problems += 1
                continue
            if not os.path.isfile(cnp):
                print(f"  MISSING CN  {repo}/{pack}")
                problems += 1
                continue
            en = json.load(open(enp, encoding="utf-8"))
            cn = json.load(open(cnp, encoding="utf-8"))
            cur = cn.get("label")
            enl = en.get("label")
            if cur == new:
                print(f"  same        {repo}/{pack}  {cur!r}")
                continue
            print(f"  {repo}/{pack}\n     EN   {enl!r}\n     from {cur!r}\n     to   {new!r}")
            changed += 1
            if a.write:
                cn["label"] = new
                with open(cnp, "w", encoding="utf-8") as f:
                    json.dump(cn, f, ensure_ascii=False, indent=2)
                    f.write("\n")
    print(f"\nlabels to change: {changed}   problems: {problems}")
    if a.dry:
        print("(dry run: nothing written)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
