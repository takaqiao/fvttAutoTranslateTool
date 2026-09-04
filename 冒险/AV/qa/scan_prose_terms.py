"""scan_prose_terms.py - one entity, one rendering, in PROSE as well as in names.

`scan_name_consistency.py` groups bilingual NAME leaves by their English half, so it can
only see a document that is named two ways. It is blind to the far more common defect:
a name that is stable on the sheet but drifts inside the text - the same NPC called
扎凯乌斯·夸格迈尔 in one of his letters and 撒该乌斯·奎格迈尔 in the next, or
奥塔里 becoming 奥塔瑞 halfway through a paragraph. Prose carries no key saying which
English term a Chinese phrase came from, so grouping cannot find it.

The English baseline supplies the missing key. Every Chinese leaf has an English leaf at
the same path, so:

  1. take the canonical Chinese for an English entity from the corpus's own bilingual name
     leaves (and the translation memory) - that is the rendering already agreed on;
  2. look only at the Chinese leaves whose English counterpart actually mentions that
     entity - which is what makes this evidence rather than a global text search;
  3. inside those, find substrings that are NEAR the canonical rendering but not equal.
     A near-miss of a name, in a paragraph that provably talks about that name, is a
     variant - not a coincidence.

Similarity is deliberately narrow (default 0.6 over same-length windows): 奥塔瑞 vs 奥塔里
scores high, while an unrelated word of the same length does not.

Usage:
  python scan_prose_terms.py --cn-dir <dir> --en-dir <dir> [--tm <tm.json>]
                             [--min-freq 3] [--min-sim 0.6] [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

CJK = re.compile(r"[一-鿿]")
LATIN_TAIL = re.compile(r"\s+[\x20-\x7E‘’–—]+$")
NAME_KEYS = {"name", "tokenName", "prototypeToken"}
TAG = re.compile(r"<[^>]+>")
BRACKET = re.compile(r"@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]")
# A proper noun in the English: capitalised words, optionally with of/the/'s between them.
PROPER = re.compile(r"\b[A-Z][a-z’'\-]+(?:\s+(?:of|the|and|de|von|van)\s+[A-Z][a-z’'\-]+|"
                    r"\s+[A-Z][a-z’'\-]+)*\b")
# Words that start a sentence and are not entities.
STOPWORDS = {
    "The", "This", "That", "These", "Those", "They", "There", "Then", "Their", "Its",
    "If", "When", "While", "After", "Before", "Once", "Each", "Every", "Any", "All",
    "You", "Your", "He", "She", "It", "His", "Her", "In", "On", "At", "As", "An", "A",
    "For", "From", "With", "Without", "By", "To", "Of", "And", "But", "Or", "So",
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "Success", "Failure", "Critical", "Critical Success", "Critical Failure",
    "Trigger", "Effect", "Requirements", "Frequency", "Range", "Area", "Duration",
    "Saving", "Special", "Note", "Level", "Price", "Bulk", "Hands", "Usage", "Activate",
    "Melee", "Ranged", "Damage", "Speed", "Perception", "Languages", "Skills", "Items",
    "AC", "HP", "DC", "XP", "GM", "NPC", "PC",
}


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def flat(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {".".join(p): v for p, v in walk(data.get("entries", {}))}


def visible(text):
    return TAG.sub(" ", BRACKET.sub(" ", text))


def chinese_head(value):
    """`焦皮地精 Charhide Goblin` -> `焦皮地精`."""
    if not value or not CJK.search(value):
        return None
    m = LATIN_TAIL.search(value.strip())
    head = value.strip()[: m.start()].strip() if m else value.strip()
    return head if head and CJK.search(head) else None


# A canonical rendering has to look like a NAME. The translation memory is polluted with
# statblock values pulled out of actor sheets (`Hit Points` -> `4生命值`) and with whole
# sentences (`Will` -> `当你意志豁免成功时…`); using those as the thing to match against
# produces pages of noise and no findings.
SENTENCE = re.compile(r"[，。；：！？、]")
MARKUP = re.compile(r"[{}\[\]<>@|]")
DIGIT = re.compile(r"[0-9]")


def name_shaped(zh, max_len=14):
    return (zh and len(zh) <= max_len and not SENTENCE.search(zh)
            and not MARKUP.search(zh) and not DIGIT.search(zh))


def real_variant(variant, canonical):
    """A window that merely clips the canonical is an artifact, not a second rendering.

    `沙伊坦` scanned at width 2 yields `伊坦`, which is 0.8 similar and completely
    uninteresting. A genuine variant differs in CONTENT: neither string contains the other.
    """
    if not variant or MARKUP.search(variant):
        return False
    if variant in canonical or canonical in variant:
        return False
    return True


def best_variant(zh_text, canonical, min_sim):
    """The closest substring of zh_text to `canonical` that is not `canonical` itself."""
    n = len(canonical)
    if n < 2:
        return None
    best, best_score = None, 0.0
    for width in (n - 1, n, n + 1):
        if width < 2:
            continue
        for i in range(len(zh_text) - width + 1):
            window = zh_text[i:i + width]
            if window == canonical or not CJK.search(window):
                continue
            if not real_variant(window, canonical):
                continue
            score = SequenceMatcher(None, canonical, window).ratio()
            if score > best_score:
                best, best_score = window, score
    return (best, best_score) if best_score >= min_sim else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--en-dir", required=True, type=Path)
    parser.add_argument("--tm", type=Path)
    parser.add_argument("--min-freq", type=int, default=3,
                        help="how many leaves must mention the entity before it is checked")
    parser.add_argument("--min-sim", type=float, default=0.65)
    parser.add_argument("--include-tm", action="store_true",
                        help="also accept canonical renderings that exist only in the TM. "
                             "Off by default: the TM holds rules keywords (Cast, Fortitude, "
                             "Huge) whose Chinese is ordinary prose, and matching prose "
                             "against prose produces noise, not findings.")
    parser.add_argument("--min-len", type=int, default=2, help="minimum Chinese name length")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    # --- canonical renderings, from the corpus's own bilingual names ---------
    canonical = {}
    corpus = {}
    for cn_path in sorted(args.cn_dir.glob("*.json")):
        en_path = args.en_dir / cn_path.name
        if not en_path.exists():
            continue
        cn, en = flat(cn_path), flat(en_path)
        corpus[cn_path.name] = (cn, en)
        for key, value in cn.items():
            if key.split(".")[-1] not in NAME_KEYS:
                continue
            head = chinese_head(value)
            m = LATIN_TAIL.search(value.strip())
            if head and m:
                english = value.strip()[m.start():].strip()
                if english and not CJK.search(english) and len(head) >= args.min_len:
                    canonical.setdefault(english, Counter())[head] += 1
    if args.include_tm and args.tm and args.tm.exists():
        tm = json.loads(args.tm.read_text(encoding="utf-8"))
        for english, entry in tm.items():
            head = chinese_head(entry.get("name", "")) if isinstance(entry, dict) else None
            if head and len(head) >= args.min_len:
                canonical.setdefault(english, Counter())[head] += 0  # candidate, no vote
    # settle on one canonical per English term
    canon = {}
    for english, counts in canonical.items():
        if not counts:
            continue
        best = counts.most_common(1)[0][0]
        if name_shaped(best):
            canon[english] = best
    print(f"canonical renderings available for {len(canon)} English terms")

    # --- which entities does the English actually talk about, and how often? --
    mentions = defaultdict(list)   # english term -> [(file, key)]
    for name, (cn, en) in corpus.items():
        for key, en_text in en.items():
            if key.split(".")[-1] in NAME_KEYS:
                continue
            vis = visible(en_text)
            for m in PROPER.finditer(vis):
                term = m.group(0).strip()
                if term in STOPWORDS or len(term) < 4:
                    continue
                if term in canon:
                    mentions[term].append((name, key))

    checked = {t: v for t, v in mentions.items() if len(v) >= args.min_freq}
    print(f"entities mentioned in >= {args.min_freq} prose leaves and having a canonical "
          f"rendering: {len(checked)}")

    findings = []
    for term, where in sorted(checked.items(), key=lambda kv: -len(kv[1])):
        want = canon[term]
        variants = Counter()
        examples = {}
        for name, key in where:
            cn, en = corpus[name]
            zh_text = visible(cn.get(key, ""))
            if not zh_text or want in zh_text:
                continue
            hit = best_variant(zh_text, want, args.min_sim)
            if hit:
                variants[hit[0]] += 1
                examples.setdefault(hit[0], (name, key, round(hit[1], 2)))
        if variants:
            findings.append({
                "english": term, "canonical": want, "mention_leaves": len(where),
                "variants": dict(variants),
                "examples": {k: {"file": v[0], "path": v[1], "similarity": v[2]}
                             for k, v in examples.items()},
            })

    print(f"\nentities whose prose carries a near-miss of the agreed rendering: {len(findings)}\n")
    for f in findings[:40]:
        print(f"  {f['english'][:34]:<34} 规范={f['canonical'][:14]:<14} "
              f"出现于 {f['mention_leaves']:>3} 叶   变体={f['variants']}")
        for var, ex in list(f["examples"].items())[:2]:
            print(f"        {var}  (相似 {ex['similarity']})  {ex['file'][:26]}:{ex['path'][-46:]}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(findings, ensure_ascii=False, indent=1),
                               encoding="utf-8")
        print(f"\nreport -> {args.report}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
