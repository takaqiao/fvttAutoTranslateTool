"""autofill_from_tm.py - fill untranslated leaves that the term memory can answer.

Two kinds of leaf are answerable mechanically, and together they are most of the
outstanding volume:

  names        an exact TM hit gives the Chinese; the bilingual tail is taken from the
               leaf's OWN English, not the TM's, so casing/spacing never drifts.

  SRD item     actors embed copies of core PF2e items (spells, weapons, feats).  Those
  descriptions carry `_stats.compendiumSource = Compendium.pf2e.<pack>.Item.<id>`, so the
               Chinese is whatever `pf2e_compendium_chn`'s <pack> says for that name.
               This is the evidence `fotrp_update.apply_term_memory` uses, and it is why
               a same-named item from a *different* pack cannot be mixed in.

Everything else is left for a human: this tool never guesses prose.

Usage:
  python autofill_from_tm.py --en-dir <dir> --cn-dir <dir> --keys <pack-keys.json>
      --raw-root <dir> --tm <tm.json> --compendium-dir <chn compendium> [--write]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CJK = re.compile(r"[㐀-鿿]")
WORD = re.compile(r"[A-Za-z]{2,}")
ENRICHER = re.compile(r"@[A-Za-z]+\[[^\]]*\](?:\{[^{}]*\})?|\[\[[^\]]*\]\](?:\{[^{}]*\})?")
TAG = re.compile(r"<[^>]+>")
LATIN_TAIL = re.compile(r"\s+[\x20-\x7E‘’–—]+$")
COMPENDIUM_SOURCE = re.compile(r"^Compendium\.pf2e\.([^.]+)\.")
NAME_KEYS = {"name", "tokenName", "prototypeToken"}
# pf2_cn is a UI-i18n file; build_3source_tm derives EN->ZH from CamelCase key segments,
# which is fine for a fallback gloss but produces whole sentences for a *name*
# (`No Breath` -> "该怪物无需呼吸，且免疫需要呼吸的效果。") and wrong compounds
# (`Difficult Terrain` -> "困难地形速度"). Never autofill a name from it.
NAME_SOURCES = {"wiki", "pf2e_compendium", "other"}
SENTENCE_PUNCT = re.compile(r"[。，；：！？]")
PARENS = re.compile(r"[（）()]")


def name_shape_ok(zh: str) -> bool:
    """A name is short, has no sentence punctuation, and carries no parenthetical."""
    return bool(zh) and len(zh) <= 24 and not SENTENCE_PUNCT.search(zh) and not PARENS.search(zh)

PROSE_UNDER_NOTES = {"label", "summary", "text", "description", "content",
                     "public", "private", "gamemaster", "caption", "subtitle"}
FROZEN_KEYS = {"command", "src", "width", "height"}


def visible(text):
    text = ENRICHER.sub(" ", text)
    text = TAG.sub(" ", text)
    return re.sub(r"\s+", " ", re.sub(r"&[a-zA-Z#0-9]+;", " ", text)).strip()


def chinese_only(name):
    if not isinstance(name, str) or not CJK.search(name):
        return None
    return LATIN_TAIL.sub("", name.strip()).strip() or None


def style_for(path, key):
    if key in NAME_KEYS:
        return "bilingual"
    if len(path) >= 2 and path[-2] == "notes" and key not in PROSE_UNDER_NOTES:
        return "bilingual"
    return "bilingual" if "folders" in path else "prose"


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def get_at(node, path):
    for seg in path:
        if not isinstance(node, dict) or seg not in node:
            return None
        node = node[seg]
    return node


def set_at(node, path, value):
    for seg in path[:-1]:
        node = node.setdefault(seg, {})
        if not isinstance(node, dict):
            return False
    node[path[-1]] = value
    return True


def index_compendium_sources(raw_root):
    """{_id: compendiumSource} over every raw document, including embedded ones."""
    index = {}

    def descend(doc):
        if isinstance(doc, dict):
            doc_id = doc.get("_id")
            source = (doc.get("_stats") or {}).get("compendiumSource")
            if doc_id and isinstance(source, str):
                index[doc_id] = source
            for value in doc.values():
                descend(value)
        elif isinstance(doc, list):
            for value in doc:
                descend(value)

    for path in Path(raw_root).rglob("*.json"):
        try:
            descend(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return index


def index_leaf_ids(pack_nodes):
    """{leaf-path-prefix: _id} so a leaf can be traced back to its document."""
    nodes = {n["path"]: n for n in pack_nodes}
    out = {}

    def descend(node_path, key_prefix):
        node = nodes.get(node_path)
        if node is None:
            return
        for entry in node["entries"]:
            prefix = key_prefix + (entry["key"],)
            if entry["_id"]:
                out[prefix] = entry["_id"]
            child_base = node_path + "." + entry["key"] + "."
            for child in nodes:
                if child.startswith(child_base) and "." not in child[len(child_base):]:
                    descend(child, prefix + (child[len(child_base):],))

    descend("entries", ("entries",))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--en-dir", required=True, type=Path)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--tm", required=True, type=Path)
    parser.add_argument("--compendium-dir", type=Path,
                        default=ROOT / "模组" / "pf2e_compendium_chn" / "compendium")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    spec = importlib.util.spec_from_file_location(
        "build_3source_tm", ROOT / "工具" / "翻译流程" / "scripts" / "build_3source_tm.py")
    tm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm_module)
    packs_index = tm_module.load_compendium_packs(args.compendium_dir)

    tm = json.loads(args.tm.read_text(encoding="utf-8"))
    keys_manifest = json.loads(args.keys.read_text(encoding="utf-8"))
    print("indexing compendiumSource over the raw dump ...")
    source_index = index_compendium_sources(args.raw_root)
    print(f"  {len(source_index)} documents carry a compendiumSource\n")

    grand = Counter()
    report = {}
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for en_path in sorted(args.en_dir.glob("*.json")):
        collection_id = en_path.stem
        pack = keys_manifest["packs"].get(collection_id)
        if not pack:
            continue
        en_data = json.loads(en_path.read_text(encoding="utf-8"))
        cn_path = args.cn_dir / en_path.name
        cn_data = json.loads(cn_path.read_text(encoding="utf-8")) if cn_path.exists() else {
            "label": en_data.get("label", collection_id), "entries": {}}
        cn_data.setdefault("entries", {})

        leaf_ids = index_leaf_ids(pack["nodes"])
        stats = Counter()
        samples = {"name": [], "description": [], "skipped": []}

        for path, english in walk(en_data.get("entries", {}), ("entries",)):
            key = path[-1]
            if key in FROZEN_KEYS:
                continue
            current = get_at(cn_data["entries"], path[1:])
            if isinstance(current, str) and CJK.search(current):
                continue
            if not WORD.search(visible(english)):
                continue

            style = style_for(path, key)
            if style == "bilingual":
                entry = tm.get(english)
                if entry and entry.get("source") not in NAME_SOURCES:
                    stats["skip-source"] += 1
                    continue
                zh = chinese_only(entry.get("name", "")) if entry else None
                if zh and not name_shape_ok(zh):
                    stats["skip-shape"] += 1
                    if len(samples.setdefault("skipped", [])) < 5:
                        samples["skipped"].append(f"{english} -> {zh}")
                    continue
                if zh:
                    # bilingual tail comes from THIS leaf's English, never the TM's
                    set_at(cn_data["entries"], path[1:], f"{zh} {english}")
                    stats["name"] += 1
                    if len(samples["name"]) < 3:
                        samples["name"].append(f"{english} -> {zh} {english}")
                continue

            if key != "description":
                continue
            doc_id = leaf_ids.get(path[:-1])
            source = source_index.get(doc_id) if doc_id else None
            match = COMPENDIUM_SOURCE.match(source or "")
            if not match:
                continue
            doc_name = get_at(en_data.get("entries", {}), path[1:-1] + ("name",))
            candidate = (packs_index.get(match.group(1)) or {}).get(doc_name)
            zh_desc = (candidate or {}).get("description")
            if zh_desc and (CJK.search(zh_desc) or "@Localize[" in zh_desc):
                set_at(cn_data["entries"], path[1:], zh_desc)
                stats["description"] += 1
                if len(samples["description"]) < 3:
                    samples["description"].append(f"{doc_name} <- pf2e.{match.group(1)}")

        grand.update(stats)
        report[collection_id] = {"stats": dict(stats), "samples": samples}
        filled = stats["name"] + stats["description"]
        print(f"[{'write' if (args.write and filled) else 'dry  '}] {collection_id[:56]:<56} "
              f"names={stats['name']:>5} descriptions={stats['description']:>5} "
              f"skipped(src={stats['skip-source']},shape={stats['skip-shape']})")

        if args.write and filled:
            backup = cn_path.parent.parent / "_backup" / f"autofill_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            if cn_path.exists():
                shutil.copy2(cn_path, backup / cn_path.name)
            cn_path.write_text(json.dumps(cn_data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    print(f"\nfilled: names={grand['name']} descriptions={grand['description']}")
    for collection_id, item in report.items():
        for kind in ("name", "description"):
            for sample in item["samples"][kind][:1]:
                print(f"    [{collection_id[:34]}] {kind}: {sample[:96]}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
