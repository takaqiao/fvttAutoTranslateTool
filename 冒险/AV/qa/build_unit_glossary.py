"""build_unit_glossary.py - hand each translation unit the terms it is about to need.

Units are translated independently, which is what makes the work parallel and also what
makes it drift: two agents meet `Quick Alchemy` in two units and pick two Chinese names,
and nothing in the unit tells either of them that the corpus already settled the question.
`scan_name_consistency.py` finds that afterwards; this prevents it.

For each unit, the terms actually at stake are extracted from the ENGLISH side:

  * `{label}` tails of `@UUID[...]` / `@Check[...]` - these are exactly the entities the
    text links to, and the label is what the reader sees
  * `<strong>` / `<em>` runs - PF2e writes rules keywords that way
  * Title Case runs of up to four words - proper nouns and ability names

Each candidate is answered from two places, in this order:

  1. the corpus itself - the Chinese half of a `中文 English` name leaf already in the
     workspace. Intra-project agreement beats an external dictionary every time, and this
     is why names are translated before prose (emit_units.py orders them that way).
  2. the term memory, restricted to the sources whose values are real names. `pf2_cn` is
     a UI i18n file whose keys are CamelCase fragments, so it answers `Description` with a
     whole paragraph - never a usable term (autofill_from_tm.py makes the same exclusion).

A candidate nobody answers is dropped rather than guessed: an empty glossary line is
worse than none, because it reads as an endorsement.

Values are emitted as the Chinese alone. Both the corpus and the wiki store terms in the
`中文 English` name form, and a translator copying that into prose produces exactly the
bilingual body the whole pipeline then has to strip back out.

Usage:
  python build_unit_glossary.py --units-dir <dir> --cn-dir <dir> --tm <tm.json>
                                --out-dir <dir> [--max-terms 120]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

CJK = re.compile(r"[一-鿿]")
LABEL = re.compile(r"@[A-Za-z]+\[[^\]]*\]\{([^{}]+)\}|\[\[[^\]]*\]\]\{([^{}]+)\}")
EMPH = re.compile(r"<(?:strong|em)>([^<]{2,60})</(?:strong|em)>")
TITLE_RUN = re.compile(r"\b(?:[A-Z][a-z’'\-]+)(?:\s+(?:of|the|and|in|to)\s+)?"
                       r"(?:\s+[A-Z][a-z’'\-]+){0,3}\b")
NAME_KEYS = {"name", "tokenName", "prototypeToken"}
# Sources whose value for a key is a NAME. pf2_cn is UI i18n: its keys are CamelCase
# fragments and its values are whole sentences.
NAME_SOURCES = {"wiki", "pf2e_compendium", "pf2e_compendium_extra", "other"}
# Words that are Title Case in English prose without being terms.
STOP = {"You", "Your", "The", "This", "That", "If", "When", "While", "Each", "Any",
        "Critical Success", "Critical Failure", "Success", "Failure", "Trigger",
        "Requirements", "Effect", "Special", "Frequency", "Cost", "Duration", "Range",
        "Area", "Targets", "Saving Throw", "Activate", "Access", "Craft Requirements"}


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def corpus_names(cn_dir: Path):
    """English half -> Chinese half, from every `中文 English` name leaf already settled."""
    out = {}
    for path in sorted(cn_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for keys, value in walk(data.get("entries", {})):
            if keys[-1] not in NAME_KEYS or not CJK.search(value):
                continue
            match = re.search(r"^(.*?[一-鿿][^\x00-\x7F]*)\s+([\x20-\x7E]+)$", value.strip())
            if match:
                english, chinese = match.group(2).strip(), match.group(1).strip()
                if english and not CJK.search(english):
                    out.setdefault(english, chinese)
    return out


def chinese_only(value: str, english: str) -> str:
    """`恶心 Sickened` -> `恶心`. The bilingual form belongs in a name leaf, never in prose."""
    value = value.strip()
    if value.lower().endswith(" " + english.lower()):
        value = value[: -len(english) - 1].rstrip()
    return re.sub(r"\s+[\x20-\x7E]+$", "", value).strip() or value


def candidates(text: str):
    for match in LABEL.finditer(text):
        yield (match.group(1) or match.group(2)).strip()
    for match in EMPH.finditer(text):
        yield re.sub(r"\s+", " ", match.group(1)).strip()
    for match in TITLE_RUN.finditer(re.sub(r"<[^>]+>", " ", text)):
        yield match.group(0).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units-dir", type=Path, required=True)
    parser.add_argument("--cn-dir", type=Path, required=True)
    parser.add_argument("--tm", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-terms", type=int, default=120)
    args = parser.parse_args()

    tm = json.loads(args.tm.read_text(encoding="utf-8"))
    corpus = corpus_names(args.cn_dir)
    print(f"corpus names: {len(corpus)}   term memory: {len(tm)}")

    written, stats = 0, Counter()
    for unit_file in sorted(args.units_dir.glob("*/*.json")):
        unit = json.loads(unit_file.read_text(encoding="utf-8"))
        seen = Counter()
        for item in unit["items"]:
            for term in candidates(item["en"]):
                if term and term not in STOP and not CJK.search(term):
                    seen[term] += 1
            if item["style"] == "bilingual":
                seen[item["en"]] += 1

        glossary = []
        for term, count in seen.most_common():
            if term in corpus:
                glossary.append({"en": term, "zh": chinese_only(corpus[term], term),
                                 "from": "corpus"})
                stats["corpus"] += 1
            else:
                hit = tm.get(term)
                if hit and hit.get("source") in NAME_SOURCES and hit.get("name"):
                    value = hit["name"]
                    # A "name" longer than a name is a scraped paragraph, not a term.
                    if len(value) <= 24 and CJK.search(value):
                        glossary.append({"en": term, "zh": chinese_only(value, term),
                                         "from": hit["source"]})
                        stats[hit["source"]] += 1
            if len(glossary) >= args.max_terms:
                break

        out_file = args.out_dir / unit_file.parent.name / unit_file.name
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps({"pack": unit["pack"], "unit": unit["unit"],
                                        "terms": glossary}, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        written += 1

    print(f"wrote {written} glossaries -> {args.out_dir}")
    for key, count in stats.most_common():
        print(f"    {key:22s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
