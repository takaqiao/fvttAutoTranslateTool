"""Build a PF2e zh-CN translation memory.

Approved authority order (HIGH → LOW):
1. PF2 Wiki (online review or an offline exported glossary)
2. pf2e_compendium / pf2_cn core translations
3. pf2e-compendium-extra-cn project memory
4. Other reviewed sources

Default output: 工具/翻译流程/tm_cache/tm_3source.json

Each entry shape:
{
  "<English key>": {
    "name": "<bilingual or zh>",        # primary translation
    "description": "<zh>",              # if available (only from compendium)
    "source": "pf2_cn" | "pf2e_compendium" | "wiki",
    "all_sources": {<source>: <name>}   # for human review on conflict
  }
}

Core compendium entries beat heuristic pf2_cn key derivations when both are on
the same authority tier.
"""
import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def default_paths(root=ROOT):
    """Return paths for the repository's current directory layout."""
    root = Path(root)
    env = os.environ.get
    return {
        # Was `root/pf2wiki-scraper/out/glossary_wiki.json` — that path died in the
        # 2026-08-31 reorg, and its contents were a noisy 2026-04-19 regex mine.
        # extract_wiki_terms.py now produces a layered table in the same shape.
        "wiki": Path(env("PF2_TM_WIKI") or (root / "工具" / "翻译流程" / "data" / "wiki_terms.json")),
        # Optional per-project overlay of hand-verified wiki terms (same tier as wiki).
        "wiki_project": Path(env("PF2_TM_WIKI_PROJECT") or (root / "工具" / "翻译流程" / "data" / "av_wiki_terms.json")),
        "pf2e_compendium": Path(env("PF2_TM_COMPENDIUM") or (root / "模组" / "pf2e_compendium_chn" / "compendium")),
        "pf2_cn": Path(env("PF2_TM_PF2CN") or (root / "模组" / "pf2_cn" / "zh_Hans")),
        # Tier-1 fallback. README lists it as the last-resort source; it was never wired in.
        "glossary": Path(env("PF2_TM_GLOSSARY") or (root / "术语表" / "glossary.json")),
        "output": root / "工具" / "翻译流程" / "tm_cache" / "tm_3source.json",
    }


PATHS = default_paths()
OUT = PATHS["output"]
GLOSSARY = PATHS["glossary"]
WIKI_PROJECT = PATHS["wiki_project"]
OUT.parent.mkdir(parents=True, exist_ok=True)
WIKI = PATHS["wiki"]
COMPENDIUM_DIR = PATHS["pf2e_compendium"]
PF2_CN_DIR = PATHS["pf2_cn"]

# User-approved authority tiers. Core compendium and pf2_cn intentionally share
# one tier; exact compendium entries beat heuristic i18n-key derivations on ties.
PRIORITY = {
    "other": 1,
    "pf2e_compendium_extra": 2,
    "pf2_cn": 3,
    "pf2e_compendium": 3,
    "wiki": 4,
}
TIE_BREAK = {"pf2_cn": 1, "pf2e_compendium": 2}


def load_wiki(path=None, label="wiki"):
    """Wiki glossary: {en: {zh, ...}} -> {en: {name: 'zh en', ...}}.

    Shape is shared by the retired glossary_wiki.json and by the current
    data/wiki_terms.json, so both load through here unchanged.
    """
    path = Path(path) if path is not None else WIKI
    out = {}
    if not path.exists():
        print(f"[{label}] missing: {path}")
        return out
    data = json.load(open(path, encoding="utf-8"))
    for en, val in data.items():
        if not isinstance(val, dict):
            continue
        zh = val.get("zh")
        if not zh:
            continue
        out[en] = {"name": f"{zh} {en}", "description": None}
    print(f"[{label}] loaded {len(out)} entries")
    return out


def load_glossary(path=None):
    """術語表/glossary.json: flat {en: zh}. Tier 1 ("other") - last-resort fallback.

    Values are synthesised into the bilingual `中文 English` form on purpose: apply_tm
    writes `name` straight through, and a bare-Chinese name would silently break the
    name-field convention everywhere it wins.
    """
    path = Path(path) if path is not None else GLOSSARY
    out = {}
    if not path.exists():
        print(f"[other] missing: {path}")
        return out
    data = json.load(open(path, encoding="utf-8"))
    for en, zh in data.items():
        if isinstance(zh, list):
            zh = zh[0] if zh else None
        if isinstance(zh, str) and zh.strip() and isinstance(en, str) and en.strip():
            out[en] = {"name": f"{zh.strip()} {en.strip()}", "description": None}
    print(f"[other] loaded {len(out)} entries")
    return out


def source_provenance(paths):
    """Record what each source actually was, so any TM can be traced to its inputs."""
    record = {}
    for key in ("wiki", "wiki_project", "pf2e_compendium", "pf2_cn", "glossary"):
        path = Path(paths[key])
        item = {"path": str(path), "exists": path.exists()}
        if path.is_dir():
            files = sorted(path.glob("*.json"))
            item["files"] = len(files)
            item["max_mtime"] = max((f.stat().st_mtime for f in files), default=0)
            manifest = path.parent / "module.json"
            if manifest.exists():
                try:
                    item["version"] = json.loads(manifest.read_text(encoding="utf-8")).get("version")
                except Exception:
                    pass
        elif path.exists():
            item["bytes"] = path.stat().st_size
            item["max_mtime"] = path.stat().st_mtime
        record[key] = item
    return record


def load_compendium_packs(directory=None):
    """Load zh-CN compendium entries grouped by their Foundry pack ID."""
    directory = Path(directory) if directory is not None else COMPENDIUM_DIR
    packs = {}
    files = sorted(p for p in directory.glob("pf2e.*.json"))
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[compendium] skip {fp.name}: {e}")
            continue
        ent = data.get("entries")
        if not isinstance(ent, dict):
            continue
        pack_id = fp.stem.removeprefix("pf2e.")
        pack_entries = packs.setdefault(pack_id, {})
        for en, val in ent.items():
            if not isinstance(val, dict):
                continue
            name = val.get("name")
            desc = val.get("description")
            if not name and not desc:
                continue
            pack_entries[en] = {"name": name, "description": desc}
    print(f"[pf2e_compendium] loaded {len(packs)} pack indexes from {len(files)} files")
    return packs


def load_compendium(directory=None):
    """Load a compatibility EN→ZH index across all pf2e.*.json files."""
    packs = load_compendium_packs(directory)
    out = {}
    for pack_entries in packs.values():
        for en, val in pack_entries.items():
            # Preserve historical first-file-wins behavior for callers that cannot
            # supply a source pack. Source-aware callers use load_compendium_packs.
            if en not in out:
                out[en] = val
    print(f"[pf2e_compendium] loaded {len(out)} entries in the flat index")
    return out


def load_pf2_cn(directory=None):
    """pf2_cn i18n: nested dict like {ACTOR: {CharacterSheetPF2e: {Tab: {...}}}}.
    Flatten leaf string values keyed by the *value* itself? No — these are translations
    keyed by i18n key. We need EN→ZH but pf2_cn doesn't store EN.

    Strategy: pf2_cn provides UI translations like 'PF2E.Saves.Fortitude' → '强韧'.
    We can't directly use these as EN→ZH for content lookup. So we extract any leaf
    where the *path* contains a recognizable English term + the value is Chinese.

    For now, treat pf2_cn as a low-priority fallback that only contributes pre-known
    UI fragments. We don't use it for content lookup — it's mostly system strings.

    Return empty dict (not used for content lookup); the workflow doc explains why.
    """
    # Walk the i18n tree and find leaves where the i18n key suggests the English form
    out = {}
    directory = Path(directory) if directory is not None else PF2_CN_DIR
    files = list(directory.glob("*.json"))
    for fp in files:
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception as e:
            print(f"[pf2_cn] skip {fp.name}: {e}")
            continue
        # Flatten the dict
        def walk(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    yield from walk(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, str):
                yield (path, obj)
        for key_path, value in walk(data):
            # Last segment as candidate English (if PascalCase)
            seg = key_path.split(".")[-1]
            # Only use if last segment looks like a CamelCase English word (3+ chars)
            if re.match(r'^[A-Z][a-zA-Z]{2,}$', seg):
                # Convert CamelCase to space-separated: 'CriticalSuccess' -> 'Critical Success'
                en = re.sub(r'([a-z])([A-Z])', r'\1 \2', seg)
                if en not in out and any('一' <= c <= '鿿' for c in value):
                    out[en] = {"name": value, "description": None}
    print(f"[pf2_cn] derived {len(out)} EN→ZH from i18n keys")
    return out


def merge_sources(wiki, compendium, pf2_cn):
    """Compatibility wrapper for the historical three-source API."""
    return merge_source_map(
        {
            "wiki": wiki,
            "pf2e_compendium": compendium,
            "pf2_cn": pf2_cn,
        }
    )


def merge_source_map(sources):
    """Merge EN→ZH sources using the approved authority hierarchy."""
    merged = {}
    description_priority = {}
    for src_name, src_dict in sources.items():
        priority = PRIORITY.get(src_name, 0)
        for en, val in src_dict.items():
            if en not in merged:
                merged[en] = {
                    "name": val.get("name"),
                    "description": val.get("description"),
                    "source": src_name,
                    "all_sources": {src_name: val.get("name")},
                }
                description_priority[en] = priority if val.get("description") else -1
            else:
                merged[en]["all_sources"][src_name] = val.get("name")
                current_source = merged[en]["source"]
                current_priority = PRIORITY.get(current_source, 0)
                wins_tie = (
                    priority == current_priority
                    and TIE_BREAK.get(src_name, 0) > TIE_BREAK.get(current_source, 0)
                )
                if val.get("name") and (priority > current_priority or wins_tie):
                    merged[en]["name"] = val.get("name")
                    merged[en]["source"] = src_name
                if val.get("description") and priority >= description_priority[en]:
                    merged[en]["description"] = val.get("description")
                    description_priority[en] = priority
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the priority-merged PF2e zh-CN translation memory.")
    parser.add_argument("--wiki", type=Path, default=PATHS["wiki"])
    parser.add_argument("--wiki-project", type=Path, default=PATHS["wiki_project"],
                        help="hand-verified per-project wiki terms; overlays the wiki tier")
    parser.add_argument("--compendium-dir", type=Path, default=PATHS["pf2e_compendium"])
    parser.add_argument("--pf2-cn-dir", type=Path, default=PATHS["pf2_cn"])
    parser.add_argument("--glossary", type=Path, default=PATHS["glossary"])
    parser.add_argument("--no-glossary", action="store_true",
                        help="drop the tier-1 術語表 fallback")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)

    # The project overlay sits inside the wiki tier rather than beside it, so PRIORITY
    # stays a four-tier ladder and no new tie-break case is introduced.
    wiki = load_wiki(args.wiki)
    project = load_wiki(args.wiki_project, label="wiki_project")
    wiki.update(project)

    sources = {
        "pf2_cn": load_pf2_cn(args.pf2_cn_dir),
        "pf2e_compendium": load_compendium(args.compendium_dir),
        "wiki": wiki,
    }
    if not args.no_glossary:
        sources["other"] = load_glossary(args.glossary)

    merged = merge_source_map(sources)

    by_src = {}
    for value in merged.values():
        by_src[value["source"]] = by_src.get(value["source"], 0) + 1
    print(f"\nMerged TM: {len(merged)} entries")
    for name in ("wiki", "pf2e_compendium", "pf2_cn", "other"):
        count = by_src.get(name, 0)
        print(f"  {name:<16} {count:>6} ({100 * count / max(len(merged), 1):.1f}%)")

    # Provenance goes to a sidecar, never into the map: consumers iterate the TM as
    # {english: entry}, and a `_provenance` key would read as a term.
    provenance = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "priority": PRIORITY,
        "tie_break": TIE_BREAK,
        "entries": len(merged),
        "by_source": by_src,
        "sources": source_provenance({
            "wiki": args.wiki, "wiki_project": args.wiki_project,
            "pf2e_compendium": args.compendium_dir, "pf2_cn": args.pf2_cn_dir,
            "glossary": args.glossary,
        }),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        shutil.copy2(args.output, args.output.with_suffix(".json.bak"))
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False)
    meta_path = args.output.with_name(args.output.stem + ".meta.json")
    meta_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nWritten to    {args.output}")
    print(f"Provenance -> {meta_path}")


if __name__ == "__main__":
    main()
