"""check_pack_targets.py - does every translation file actually name a real pack?

Babele resolves a translation file purely by its FILENAME:

    translation-source-discovery.js:78
        baseName === encodeURI(`${collection}.json`)

so `<moduleId>.<packName>.json` must match an installed module's pack exactly.  A file
whose module was renamed, or that is not a pack at all, is loaded by nothing and fails
silently - the pack simply renders in English.  That is how
`footsteps-of-otari.footsteps-of-otari-Journals.json` went unnoticed after the module
became `footsteps-and-maps`.

This gate is zero-tolerance: there is no legitimate reason to ship a translation file
that targets nothing.  `labels.json` / `titles.json` are the sidebar indexes, not packs,
and are skipped.

Usage:
  python check_pack_targets.py --compendium <dir> --modules <Data/modules> [--json out.json]
"""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

INDEX_FILES = {"labels.json", "titles.json"}


def installed_packs(modules_root: Path):
    """{packageId: {packName: type}} for every installed module AND system.

    Systems matter: `pf2e.equipment-srd.json` targets the pf2e *system*, which lives in
    Data/systems, not Data/modules. Without it every `pf2e.*` file looks unresolvable.
    """
    out = {}
    manifests = list(modules_root.glob("*/module.json"))
    systems_root = modules_root.parent / "systems"
    manifests += list(systems_root.glob("*/system.json"))
    for manifest in sorted(manifests):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        package_id = data.get("id") or manifest.parent.name
        out[package_id] = {p.get("name"): p.get("type") for p in data.get("packs", []) if p.get("name")}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compendium", required=True, type=Path)
    parser.add_argument("--modules", required=True, type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    packs = installed_packs(args.modules)
    print(f"{len(packs)} installed modules, {sum(len(v) for v in packs.values())} packs\n")

    ok, problems, unknown_module, suspects = [], [], [], []
    for path in sorted(args.compendium.glob("*.json")):
        if path.name in INDEX_FILES:
            continue
        stem = path.stem
        module_id, sep, pack_name = stem.partition(".")
        if not sep:
            problems.append((path.name, "filename has no <moduleId>.<packName> split"))
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append((path.name, f"unparseable: {exc}"))
            continue
        if not isinstance(data.get("label"), str) or not data["label"].strip():
            problems.append((path.name, "missing a non-empty `label`"))
            continue
        if module_id not in packs:
            # Not installed is not automatically a defect - this repo ships translations
            # for modules the author may not have enabled. But a RENAMED module looks
            # exactly the same, and that is the failure that hid
            # footsteps-of-otari.* after the module became footsteps-and-maps. So when a
            # close-by module id IS installed, call it out.
            # A rename must look like one on BOTH halves: a similar package id AND a
            # similar pack name inside it. Package id alone is far too loose - it pairs
            # `pf2e` with `pf2_cn`, which share nothing.
            near = []
            for candidate in difflib.get_close_matches(module_id, packs.keys(), n=5, cutoff=0.6):
                pack_match = difflib.get_close_matches(pack_name, packs[candidate].keys(), n=1, cutoff=0.6)
                if pack_match:
                    near.append(f"{candidate}.{pack_match[0]}")
            if near:
                # A suspicion, not a verdict: the package may simply not be installed
                # here. Reported, but it does not fail the gate on its own.
                suspects.append((path.name,
                                 f"package '{module_id}' is not installed; possible rename of: "
                                 f"{', '.join(near)}"))
            else:
                unknown_module.append((path.name, module_id))
            continue
        if pack_name not in packs[module_id]:
            problems.append((path.name,
                             f"module '{module_id}' has no pack '{pack_name}'; "
                             f"it has: {', '.join(sorted(packs[module_id])) or '(none)'}"))
            continue
        ok.append(path.name)

    print(f"targets a real installed pack : {len(ok)}")
    print(f"package not installed locally  : {len(unknown_module)}  (not checkable here)")
    print(f"SUSPECTED RENAME              : {len(suspects)}  (review; not a failure by itself)")
    for name, why in suspects:
        print(f"  ? {name}\n      {why}")
    print(f"BROKEN                         : {len(problems)}")
    for name, why in problems:
        print(f"  ! {name}\n      {why}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"ok": ok, "unknown_module": unknown_module,
             "suspected_renames": [{"file": n, "why": w} for n, w in suspects],
             "problems": [{"file": n, "why": w} for n, w in problems]},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nreport -> {args.json}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
