"""extract_wiki_terms.py - build an EN->ZH term table from the offline PF2 wiki corpus.

Reads ``pf2wiki-scraper/out_v2/parsed/**`` directly rather than the built site, so it
does not depend on ``build_search_v2.py`` having been re-run: a content refresh alone
is enough to refresh the terms.

Three layers, highest authority first:

  1. data_page  ns=3500 ``Data:*.json`` pages render a jsonconfig table carrying literal
                ``中文`` / ``原文`` rows.  These are curated, bare terms - no
                disambiguating suffix - and they are the best source the wiki has.
  2. book_title ``《凄凉灯塔废墟》`` <-> ``Ruins of Gauntlight PZO90163``; strips the
                ``PZO#####`` / ``Player's Guide`` tail so the adventure name resolves.
  3. page_lead  ns=0 article bodies open with "<中文标题> <English Name> …".  Same rule
                as ``build_search_v2.py`` EN_NAME_RX.  Noisier, and titles here often
                carry a disambiguator (``毁木者（信仰）``), so layer 1 always wins.

Output shape deliberately matches the retired ``glossary_wiki.json``
(``{en: {"zh": ...}}``) so ``build_3source_tm.load_wiki`` needs no change.

Usage:
  python extract_wiki_terms.py [--corpus <parsed dir>] [--out <file>] [--report]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = ROOT / "其他项目" / "PF2离线百科" / "pf2wiki-scraper" / "out_v2" / "parsed"
DEFAULT_OUT = ROOT / "工具" / "翻译流程" / "data" / "wiki_terms.json"

JSONCONFIG_ROW = re.compile(
    r"<tr><th>(.*?)</th><td class=\"mw-jsonconfig-value\">(.*?)</td></tr>", re.S)
DATA_TITLE_PREFIX = re.compile(r"^(?:Data|数据):([A-Za-z]+)-")
# Same expression build_search_v2.py uses for the leading Latin run of an ns0 body.
EN_NAME_RX = re.compile(r"^[A-Za-z][A-Za-z0-9'’ \-]{0,58}[A-Za-z0-9'’]")
TAG = re.compile(r"<[^>]+>")
WHITESPACE = re.compile(r"\s+")
DATA_DOC = re.compile(r"^此数据的文档可以在.*?创建\s*")
BOOK_TITLE = re.compile(r"^《(?P<zh>[^》]+)》(?P<suffix>.*)$")
BOOK_EN_TAIL = re.compile(r"\s*(?:PZO\d+|Player['’]s Guide|Adventure Path)\s*$", re.I)

DATA_PREFIX_TYPE = {
    "Spells": "法术", "Feats": "专长", "Creatures": "生物", "Items": "物品",
    "Traits": "特征", "Conditions": "异常状态", "Backgrounds": "背景",
    "Actions": "动作", "Classes": "职业", "Ancestries": "族裔", "Heritages": "传承",
    "Deities": "信仰", "Archetypes": "变体", "Hazards": "陷阱", "Rules": "规则",
}
LAYER_RANK = {"data_page": 3, "book_title": 2, "page_lead": 1}

# Latin runs that are never a term. Mirrors pf2wiki-scraper/extract_terms.py.
NOISE_EN = {
    "AON", "DC", "PC", "PCs", "NPC", "NPCs", "HP", "XP", "AC", "CRB", "APG", "GM",
    "SRD", "OGL", "PF", "PF2", "PF2e", "TRPG", "Paizo", "Pathfinder", "Starfinder",
    "See", "See also", "The", "A", "An", "Of", "In", "And", "Or", "Is", "It",
}
# A disambiguator in the Chinese title means the bare form belongs to layer 1.
DISAMBIGUATOR = re.compile(r"[（(].{1,12}[)）]$|(?:背景|特征|信仰|职业|族裔|变体|法术|专长|生物|物品)$")


def plain_text(markup: str) -> str:
    """Cheap strip - fine for short jsonconfig cells."""
    text = TAG.sub(" ", markup or "")
    text = html.unescape(text)
    return WHITESPACE.sub(" ", text).strip()


def body_text(markup: str) -> str:
    """Reproduce build_search_v2.iter_parsed's body exactly.

    The leading license callout (`.well.quote-success` / `.quote-primary`) and the
    `规则导航` nav block sit *above* the article title in the rendered HTML, so a naive
    strip leaves the body starting with boilerplate and `body.startswith(title)` fails -
    which silently cost ~9,600 of the ~12,400 page_lead terms on the first run.
    """
    soup = BeautifulSoup(markup or "", "lxml")
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    for tag in soup.select(".well.quote-success, .quote-primary"):
        tag.decompose()
    for tag in soup.select("div.hidden-sm.hidden-xs"):
        if tag.get_text(strip=True).startswith("规则导航"):
            tag.decompose()
    text = WHITESPACE.sub(" ", soup.get_text(" ", strip=True))
    return DATA_DOC.sub("", text).strip()


def iter_pages(corpus: Path):
    for path in corpus.rglob("*.json"):
        try:
            yield json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue


def harvest_data_page(page):
    """ns=3500 Data:*.json -> (en, zh, type)"""
    title = page.get("title") or ""
    text = (page.get("parse") or {}).get("text") or ""
    if "mw-jsonconfig-value" not in text:
        return None
    rows = {}
    for key, value in JSONCONFIG_ROW.findall(text):
        key = plain_text(key)
        if key in ("中文", "原文", "data_type") and key not in rows:
            rows[key] = plain_text(value)
    en, zh = rows.get("原文"), rows.get("中文")
    if not en or not zh or not re.match(r"^[A-Za-z0-9]", en):
        return None
    kind = rows.get("data_type")
    if not kind:
        m = DATA_TITLE_PREFIX.match(title)
        kind = DATA_PREFIX_TYPE.get(m.group(1), m.group(1)) if m else None
    return en, zh, kind


def harvest_book_title(title, en_name):
    m = BOOK_TITLE.match(title or "")
    if not m or not en_name:
        return None
    zh = m.group("zh").strip()
    en = BOOK_EN_TAIL.sub("", en_name).strip()
    if len(en) < 3 or not zh:
        return None
    return en, zh


def harvest_page_lead(page):
    """ns=0 body opens with '<中文标题> <English Name> …'."""
    title = page.get("title") or ""
    text = (page.get("parse") or {}).get("text") or ""
    body = body_text(text)
    if not body.startswith(title):
        return None
    m = EN_NAME_RX.match(body[len(title):].lstrip())
    if not m:
        return None
    en = m.group(0).strip(" -")
    if len(en) < 2:
        return None
    return en, title


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-words", type=int, default=4)
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        parser.error(f"corpus not found: {args.corpus}")

    started = time.time()
    layers = {"data_page": {}, "book_title": {}, "page_lead": {}}
    provenance = {}
    conflicts = defaultdict(set)
    stats = Counter()
    captured_at = None

    for page in iter_pages(args.corpus):
        stats["pages"] += 1
        captured_at = captured_at or page.get("captured_at")
        ns = page.get("ns")
        title = page.get("title") or ""

        if ns == 3500:
            hit = harvest_data_page(page)
            if hit:
                en, zh, kind = hit
                if en in layers["data_page"] and layers["data_page"][en] != zh:
                    conflicts[en].add(layers["data_page"][en])
                    conflicts[en].add(zh)
                layers["data_page"].setdefault(en, zh)
                provenance.setdefault(("data_page", en), {"page": title, "type": kind})
                stats["data_page"] += 1
            continue

        if ns != 0:
            continue

        lead = harvest_page_lead(page)
        if not lead:
            continue
        en, zh = lead

        book = harvest_book_title(zh, en)
        if book:
            b_en, b_zh = book
            layers["book_title"].setdefault(b_en, b_zh)
            provenance.setdefault(("book_title", b_en), {"page": title})
            stats["book_title"] += 1

        if en in NOISE_EN or len(en.split()) > args.max_words:
            stats["dropped_noise"] += 1
            continue
        layers["page_lead"].setdefault(en, zh)
        provenance.setdefault(("page_lead", en), {"page": title})
        stats["page_lead"] += 1

    # Merge: data_page > book_title > page_lead. A page_lead entry whose Chinese carries
    # a disambiguator loses to any bare form a higher layer already supplied.
    merged = {}
    for layer in ("data_page", "book_title", "page_lead"):
        for en, zh in layers[layer].items():
            if en in merged:
                if merged[en]["zh"] != zh:
                    conflicts[en].add(merged[en]["zh"])
                    conflicts[en].add(zh)
                continue
            if layer == "page_lead" and DISAMBIGUATOR.search(zh) and en in layers["data_page"]:
                continue
            entry = {"zh": zh, "layer": layer}
            meta = provenance.get((layer, en)) or {}
            if meta.get("type"):
                entry["data_type"] = meta["type"]
            if meta.get("page"):
                entry["page"] = meta["page"]
                entry["source"] = "https://pf2.huijiwiki.com/wiki/" + meta["page"].replace(" ", "_")
            merged[en] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")

    meta_path = args.out.with_name(args.out.stem + ".meta.json")
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus": str(args.corpus),
        "corpus_captured_at": captured_at,
        "elapsed_seconds": round(time.time() - started, 1),
        "counts": {layer: len(values) for layer, values in layers.items()} | {"total": len(merged)},
        "conflicts": len(conflicts),
        "layer_precedence": ["data_page", "book_title", "page_lead"],
        "conflict_sample": {en: sorted(v) for en, v in list(conflicts.items())[:40]},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"pages scanned      {stats['pages']}")
    for layer in ("data_page", "book_title", "page_lead"):
        print(f"  {layer:<12} raw={stats[layer]:>6} distinct={len(layers[layer]):>6}")
    print(f"  dropped noise/long          {stats['dropped_noise']}")
    print(f"\nmerged terms       {len(merged)}  (conflicts recorded: {len(conflicts)})")
    print(f"elapsed            {time.time() - started:.1f}s")
    print(f"-> {args.out}")
    print(f"-> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
