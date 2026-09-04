"""pack_index.py - one id index for both link checkers.

`dump_pack_ids.mjs` reads LevelDB, so it can only run with Foundry closed, and the dump in
`reports/` covers the `pf2e` scope only. That made every link into one of the family's OWN
packs - `Compendium.abomination-vaults-addons.abomination-vaults-addons-scenes.Scene.…` -
come back as "pack not installed", which reads like a missing dependency and is not one.

`pack-keys.json` already carries every id of every family pack, and the gate refuses to run
without it, so the family half is free. Merging here rather than in each caller keeps the
two checkers from disagreeing about what "installed" means.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_pack_ids(pack_ids_path: Path | None, keys_path: Path | None) -> dict:
    """-> {"<scope>.<pack>": {"type": str|None, "ids": [...], "byName": {name: [id]}}}"""
    index: dict[str, dict] = {}
    if pack_ids_path and Path(pack_ids_path).exists():
        index.update(json.loads(Path(pack_ids_path).read_text(encoding="utf-8")))
    if not (keys_path and Path(keys_path).exists()):
        return index
    manifest = json.loads(Path(keys_path).read_text(encoding="utf-8"))
    for pack_key, pack in manifest.get("packs", {}).items():
        ids, by_name = set(), {}
        for node in pack["nodes"]:
            for entry in node["entries"]:
                doc_id, name = entry.get("_id"), entry.get("name")
                if not doc_id:
                    continue
                ids.add(doc_id)
                if name:
                    by_name.setdefault(name, []).append(doc_id)
        if not ids:
            continue
        # A dump straight from LevelDB is the better source when we have both.
        existing = index.get(pack_key)
        if existing:
            ids |= set(existing.get("ids") or [])
            for name, more in (existing.get("byName") or {}).items():
                by_name.setdefault(name, []).extend(more)
        index[pack_key] = {"type": pack.get("type"), "ids": sorted(ids), "byName": by_name}
    return index
