"""Build a 3-source PF2e zh-CN translation memory.

Sources (priority HIGH → LOW; higher level wins on conflict):
1. pf2_cn:             system/pf2_cn/zh_Hans/*.json (i18n UI strings) — HIGHEST
2. pf2e_compendium:    system/pf2e_compendium/zh-CN/pf2e.*.json (NON-extra: pf2e.* prefix only)
3. wiki:               pf2wiki-scraper/out/glossary_wiki.json — LOWEST (fallback only; scraper unstable)

Output: 翻译流程/tm_cache/tm_3source.json

Each entry shape:
{
  "<English key>": {
    "name": "<bilingual or zh>",        # primary translation
    "description": "<zh>",              # if available (only from compendium)
    "source": "pf2_cn" | "pf2e_compendium" | "wiki",
    "all_sources": {<source>: <name>}   # for human review on conflict
  }
}

Conflict resolution: pf2_cn wins; pf2_cn miss → fallback to compendium; both miss → fallback to wiki.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # back to fvtt root
OUT = ROOT / "翻译流程" / "tm_cache" / "tm_3source.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

WIKI = ROOT / "pf2wiki-scraper" / "out" / "glossary_wiki.json"
COMPENDIUM_DIR = ROOT / "system" / "pf2e_compendium" / "zh-CN"
PF2_CN_DIR = ROOT / "system" / "pf2_cn" / "zh_Hans"

# Source priority — higher number wins on conflict (pf2_cn highest, wiki fallback)
PRIORITY = {"pf2_cn": 3, "pf2e_compendium": 2, "wiki": 1}


def load_wiki():
    """Wiki glossary: {en: {zh, count, sources, ...}} — flatten to {en: {name: 'zh en', ...}}"""
    out = {}
    if not WIKI.exists():
        print(f"[wiki] missing: {WIKI}")
        return out
    data = json.load(open(WIKI, encoding="utf-8"))
    for en, val in data.items():
        if not isinstance(val, dict):
            continue
        zh = val.get("zh")
        if not zh:
            continue
        out[en] = {"name": f"{zh} {en}", "description": None}
    print(f"[wiki] loaded {len(out)} entries")
    return out


def load_compendium():
    """zh-CN compendium: 21 files of pf2e.*.json with entries dict."""
    out = {}
    files = sorted(p for p in COMPENDIUM_DIR.glob("pf2e.*.json"))
    for fp in files:
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception as e:
            print(f"[compendium] skip {fp.name}: {e}")
            continue
        ent = data.get("entries")
        if not isinstance(ent, dict):
            continue
        for en, val in ent.items():
            if not isinstance(val, dict):
                continue
            name = val.get("name")
            desc = val.get("description")
            if not name and not desc:
                continue
            # Don't overwrite if same en already from earlier file (same source level)
            if en not in out:
                out[en] = {"name": name, "description": desc}
    print(f"[pf2e_compendium] loaded {len(out)} entries from {len(files)} files")
    return out


def load_pf2_cn():
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
    files = list(PF2_CN_DIR.glob("*.json"))
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
    """Merge with priority: pf2_cn > pf2e_compendium > wiki (wiki is fallback only)."""
    merged = {}
    # Iterate from LOWEST to HIGHEST; later iterations overwrite earlier ones
    for src_name, src_dict in [("wiki", wiki), ("pf2e_compendium", compendium), ("pf2_cn", pf2_cn)]:
        for en, val in src_dict.items():
            if en not in merged:
                merged[en] = {
                    "name": val.get("name"),
                    "description": val.get("description"),
                    "source": src_name,
                    "all_sources": {src_name: val.get("name")},
                }
            else:
                # Record this source even if not winning
                merged[en]["all_sources"][src_name] = val.get("name")
                # Higher priority overrides
                if PRIORITY[src_name] > PRIORITY[merged[en]["source"]]:
                    merged[en]["name"] = val.get("name")
                    merged[en]["source"] = src_name
                # Description: take from compendium even if name comes from a source without description
                if val.get("description") and not merged[en].get("description"):
                    merged[en]["description"] = val.get("description")
    return merged


def main():
    wiki = load_wiki()
    compendium = load_compendium()
    pf2_cn = load_pf2_cn()
    merged = merge_sources(wiki, compendium, pf2_cn)
    # Stats
    by_src = {}
    for k, v in merged.items():
        s = v["source"]
        by_src[s] = by_src.get(s, 0) + 1
    print(f"\nMerged TM: {len(merged)} entries")
    for s in ["pf2_cn", "pf2e_compendium", "wiki"]:
        print(f"  {s}: {by_src.get(s, 0)} ({100*by_src.get(s, 0)/max(len(merged),1):.1f}%)")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    print(f"\nWritten to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
