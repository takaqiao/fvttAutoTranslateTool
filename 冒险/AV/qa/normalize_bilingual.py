"""normalize_bilingual.py - strip the appended English half out of prose leaves.

The legacy AV-family translations store prose as ``CN_HTML`` immediately followed by
the *entire* English HTML block, so removing it also removes a duplicated
``<section>``/``<header>`` skeleton, not merely a text tail.

Exact string subtraction (the approach in Ember's ``normalize_adventure_translation.py``)
only lands on ~15% of AV journal pages: the appended English has drifted from the
current source (``class="box-text narrative-block"`` in the CN file vs
``class="box-text narrative"`` upstream), so a literal ``str.replace`` silently no-ops.

Primary strategy is therefore a **tag-sequence split**, which compares only tag
*names* and is immune to attribute drift:

  1. seq(s) = ordered [(is_close, tagname)] over every ``<...>``
  2. k = len(seq(cn)) - len(seq(en)); k <= 0 -> no room, skip
  3. seq(cn)[k:] == seq(en) -> the English block starts at the k-th tag
  4. accept only if ALL hold:
       - head has CJK and tail has none
       - head is independently HTML-balanced
       - similarity(strip(tail), strip(en)) >= --tail-ratio
       - head's enricher multiset is a subset of the original's

Fallbacks, each reported separately: newline-tail -> exact -> paragraph-align -> manual.

Two regressions inherited from Ember are re-asserted here as behaviour and covered by
``test_normalize_bilingual.py``:
  * a leaf whose path contains ``notes`` is NOT name-like when its own key is prose
    (that naive test once attached English tails to 287 prose leaves);
  * ``cn == en`` is preserved, never emptied (181 pure-markup leaves in EC).

Usage:
  python normalize_bilingual.py --cn-dir <dir> --en-dir <dir> [--write] [--report out.json]
  python normalize_bilingual.py --cn <file> --en <file> [--write]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

CJK = re.compile(r"[㐀-鿿]")
TAG = re.compile(r"<\s*(/?)\s*([A-Za-z][A-Za-z0-9]*)\b[^>]*>")
VOID_TAGS = {"br", "hr", "img", "input", "meta", "link", "source", "col", "area", "base", "wbr", "embed"}
ENRICHER = re.compile(r"@[A-Za-z]+\[[^\]]*\](?:\{[^{}]*\})?|\[\[[^\]]*\]\](?:\{[^{}]*\})?")

# Name-like leaves keep the bilingual tail; everything else becomes pure Chinese.
NAME_KEYS = {"name", "tokenName", "prototypeToken"}
# `notes` holds two different things: scenes.<s>.notes.<mapNoteText> is a textCollection
# (name-like), while ...outcomes.notes.label/.summary is prose read aloud to players.
PROSE_UNDER_NOTES = {"label", "summary", "text", "description", "content",
                     "public", "private", "gamemaster", "caption", "subtitle"}
LEAVE_ALONE = {"pronunciation"}
# Dropped entirely elsewhere in the pipeline; never treat as prose here.
FROZEN_KEYS = {"command", "src", "width", "height"}


def should_keep_bilingual(path: tuple[str, ...], key: str) -> bool:
    if key in NAME_KEYS:
        return True
    if len(path) >= 2 and path[-2] == "notes" and key not in PROSE_UNDER_NOTES:
        return True
    return "folders" in path


def seq(text: str):
    return [(bool(m.group(1)), m.group(2).lower()) for m in TAG.finditer(text)]


def tag_positions(text: str):
    return [m.start() for m in TAG.finditer(text)]


def strip_markup(text: str) -> str:
    text = ENRICHER.sub(" ", text)
    text = TAG.sub(" ", text)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def balanced(text: str) -> bool:
    stack = []
    for is_close, name in seq(text):
        if name in VOID_TAGS:
            continue
        if not is_close:
            stack.append(name)
        else:
            if not stack or stack[-1] != name:
                return False
            stack.pop()
    return not stack


def enrichers(text: str) -> Counter:
    return Counter(m.group(0) for m in ENRICHER.finditer(text))


def similar(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def has_english(text: str, min_words: int = 2) -> bool:
    """True when the *visible* text still carries an English word run."""
    return len(re.findall(r"[A-Za-z]{2,}", strip_markup(text))) >= min_words


def _accept(cn: str, head: str, tail: str, en, tail_ratio: float):
    """Shared acceptance test for every split strategy.

    `en` may be None: several packs have leaves whose key is not in the English
    baseline (upstream renamed the document).  Those are still strippable, so
    instead of the similarity check we require the discarded tail to actually be
    English - otherwise a split would be plain truncation.
    """
    if not CJK.search(head):
        return False, "head-no-cjk"
    if CJK.search(tail):
        return False, "tail-has-cjk"
    if not balanced(head) and balanced(cn):
        return False, "head-unbalanced"
    if en is None:
        if not has_english(tail):
            return False, "tail-not-english"
    else:
        ratio = similar(strip_markup(tail), strip_markup(en))
        if ratio < tail_ratio:
            return False, f"tail-ratio-{ratio:.2f}"
    if enrichers(head) - enrichers(cn):
        return False, "enricher-invented"
    return True, None


def split_tagseq(cn: str, en: str, tail_ratio: float):
    scn, sen = seq(cn), seq(en)
    k = len(scn) - len(sen)
    if k <= 0:
        return None, "no-room"
    if scn[k:] != sen:
        return None, "seq-mismatch"
    if k >= len(scn):
        # The English carries no tags at all, so `scn[k:] == sen` is the empty list matching
        # the empty list - a vacuous pass. There is no k-th tag to cut at; the appended
        # English is plain text and one of the other strategies has to find it.
        return None, "no-tag-anchor"
    pos = tag_positions(cn)[k]
    head, tail = cn[:pos], cn[pos:]
    ok, why = _accept(cn, head, tail, en, tail_ratio)
    return (head, None) if ok else (None, why)


def split_selfhalf(cn: str, en, tail_ratio: float):
    """CN_HTML + EN_HTML means seq(cn) is its own sequence twice over.

    Needs no English reference, which is what makes it work on leaves whose key
    upstream has renamed away.
    """
    s = seq(cn)
    if len(s) < 2 or len(s) % 2:
        return None, "not-doubled"
    half = len(s) // 2
    if s[:half] != s[half:]:
        return None, "halves-differ"
    pos = tag_positions(cn)[half]
    head, tail = cn[:pos], cn[pos:]
    # Deliberately pass en=None: the doubling is proof enough, and comparing the tail
    # to *current* English would reject leaves whose English upstream has since changed.
    ok, why = _accept(cn, head, tail, None, tail_ratio)
    return (head, None) if ok else (None, why)


def split_newline(cn: str, en, tail_ratio: float):
    best = None
    for m in re.finditer(r"\n", cn):
        head, tail = cn[: m.start()], cn[m.end():]
        if not CJK.search(head) or CJK.search(tail):
            continue
        ok, _ = _accept(cn, head, tail, en, tail_ratio)
        if ok:
            best = head
    return (best, None) if best is not None else (None, "no-newline-split")


def split_exact(cn: str, en: str, tail_ratio: float):
    """Ember's literal replacement list - still wins on short, non-HTML leaves."""
    for token in (f"\n{en}", f" {en}", f"（{en}）", f"({en})", f"【{en}】", f"[{en}]", f" / {en}", f"/{en}", en):
        if token and token in cn:
            head = cn.replace(token, "").strip()
            if head and CJK.search(head) and not CJK.search(en) and balanced(head):
                if not (enrichers(head) - enrichers(cn)):
                    return head, None
    return None, "no-exact-match"


BOUNDARY = re.compile(r"(?:</p>|</section>|</article>|</div>|<hr\s*/?>)", re.I)


def split_paragraph(cn: str, en: str, tail_ratio: float):
    best, best_ratio = None, 0.0
    for m in BOUNDARY.finditer(cn):
        head, tail = cn[: m.end()], cn[m.end():]
        if not head.strip() or not tail.strip():
            continue
        if not CJK.search(head) or CJK.search(tail):
            continue
        ratio = similar(strip_markup(tail), strip_markup(en))
        if ratio > best_ratio and balanced(head) and not (enrichers(head) - enrichers(cn)):
            best, best_ratio = head, ratio
    if best is not None and best_ratio >= 0.85:
        return best, None
    return None, f"no-paragraph-split-{best_ratio:.2f}"


# tagseq first (uses the EN reference, strongest), then the reference-free ones.
STRATEGIES = (("tagseq", split_tagseq), ("selfhalf", split_selfhalf),
              ("newline", split_newline), ("exact", split_exact),
              ("paragraph", split_paragraph))
# Strategies that do not need an English reference.
NO_REF_STRATEGIES = (("selfhalf", split_selfhalf), ("newline", split_newline))


def walk(cn_node, en_node, path, stats, leftovers, opts):
    if isinstance(cn_node, dict):
        return {
            key: walk(value, (en_node or {}).get(key) if isinstance(en_node, dict) else None,
                      path + (key,), stats, leftovers, opts)
            for key, value in cn_node.items()
        }
    if isinstance(cn_node, list):
        return [walk(v, None, path + (str(i),), stats, leftovers, opts) for i, v in enumerate(cn_node)]
    if not isinstance(cn_node, str):
        return cn_node

    key = path[-1] if path else ""
    cn = cn_node

    if key in LEAVE_ALONE or key in FROZEN_KEYS:
        stats["skipped-frozen"] += 1
        return cn
    if should_keep_bilingual(path, key):
        stats["skipped-name"] += 1
        return cn
    if not isinstance(en_node, str):
        if not (CJK.search(cn) and has_english(cn, 4)):
            stats["no-en-ref-clean"] += 1
            return cn
        reasons = []
        for name, fn in NO_REF_STRATEGIES:
            head, why = fn(cn, None, opts.tail_ratio)
            if head is not None:
                stats[name + "-noref"] += 1
                return head.strip()
            reasons.append(f"{name}:{why}")
        stats["no-en-ref-dirty"] += 1
        leftovers.append({"path": ".".join(path), "reason": "no-en-ref; " + "; ".join(reasons),
                          "cn": cn[:1500], "en": None})
        return cn

    en = en_node
    if cn == en:
        stats["identical-preserved"] += 1   # pure-markup leaves: identity is correct
        return cn
    if not CJK.search(cn):
        stats["untranslated"] += 1
        return cn
    if not has_english(en):
        stats["en-has-no-words"] += 1
        return cn
    if not has_english(cn):
        stats["already-clean"] += 1     # translator already produced pure Chinese
        return cn

    reasons = []
    for name, fn in STRATEGIES:
        head, why = fn(cn, en, opts.tail_ratio)
        if head is not None:
            stats[name] += 1
            if not balanced(head):
                stats["html-defect-carried"] += 1
                leftovers.append({"path": ".".join(path), "reason": "pre-existing-html-imbalance",
                                  "kind": "html", "cn": head[:800], "en": en[:800]})
            return head.strip()
        reasons.append(f"{name}:{why}")

    stats["manual"] += 1
    leftovers.append({
        "path": ".".join(path), "reason": "; ".join(reasons),
        "best_ratio": round(similar(strip_markup(cn), strip_markup(en)), 3),
        "cn": cn[:1500], "en": en[:1500],
    })
    return cn


def process(cn_path: Path, en_path: Path, opts, stats, leftovers):
    cn_data = json.loads(cn_path.read_text(encoding="utf-8"))
    en_data = json.loads(en_path.read_text(encoding="utf-8"))
    out = dict(cn_data)
    for section in ("entries", "folders"):
        cn_section = cn_data.get(section)
        en_section = en_data.get(section)
        if isinstance(cn_section, dict):
            out[section] = walk(cn_section, en_section if isinstance(en_section, dict) else {},
                                (section,), stats, leftovers, opts)
    return cn_data, out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", type=Path)
    parser.add_argument("--en-dir", type=Path)
    parser.add_argument("--cn", type=Path)
    parser.add_argument("--en", type=Path)
    parser.add_argument("--write", action="store_true", help="actually write; default is a dry run")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--leftovers", type=Path)
    parser.add_argument("--tail-ratio", type=float, default=0.90)
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args(argv)

    jobs = []
    if args.cn_dir and args.en_dir:
        for cn_path in sorted(args.cn_dir.glob("*.json")):
            en_path = args.en_dir / cn_path.name
            if en_path.exists():
                jobs.append((cn_path, en_path))
            else:
                print(f"[skip] no EN baseline for {cn_path.name}")
    if args.cn and args.en:
        jobs.append((args.cn, args.en))
    if not jobs:
        parser.error("need --cn-dir/--en-dir or --cn/--en")

    grand = Counter()
    report = {}
    all_leftovers = []
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for cn_path, en_path in jobs:
        stats = Counter()
        leftovers = []
        original, out = process(cn_path, en_path, args, stats, leftovers)
        changed = out != original
        for leaf in leftovers:
            leaf["file"] = cn_path.name
        all_leftovers.extend(leftovers)
        grand.update(stats)
        report[cn_path.name] = {"changed": changed, "stats": dict(stats), "leftovers": len(leftovers)}

        converted = sum(stats[k] for k, _ in STRATEGIES) + stats["selfhalf-noref"] + stats["newline-noref"]
        print(f"[{'write' if (args.write and changed) else 'dry  '}] {cn_path.name[:58]:<58} "
              f"stripped={converted:>5} (tagseq={stats['tagseq']} self={stats['selfhalf'] + stats['selfhalf-noref']} "
              f"nl={stats['newline'] + stats['newline-noref']} exact={stats['exact']} para={stats['paragraph']}) "
              f"manual={stats['manual']} noref={stats['no-en-ref-dirty']}")

        if args.write and changed:
            backup_root = args.backup_dir or (cn_path.parent.parent / "_backup" / stamp)
            backup_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup_root / cn_path.name)   # never json.load/dump a backup
            cn_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    converted = sum(grand[k] for k, _ in STRATEGIES) + grand["selfhalf-noref"] + grand["newline-noref"]
    total = converted + grand["manual"] + grand["no-en-ref-dirty"]
    print(f"\nstripped {converted}/{total} bilingual prose leaves "
          f"({converted / total:.1%})" if total else "\nnothing to strip")
    for key in ("tagseq", "selfhalf", "newline", "exact", "paragraph",
                "selfhalf-noref", "newline-noref", "manual", "no-en-ref-dirty",
                "no-en-ref-clean", "already-clean", "identical-preserved", "untranslated",
                "skipped-name", "skipped-frozen", "en-has-no-words", "html-defect-carried"):
        if grand[key]:
            print(f"    {key:<22} {grand[key]}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    if args.leftovers:
        args.leftovers.parent.mkdir(parents=True, exist_ok=True)
        args.leftovers.write_text(json.dumps(all_leftovers, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"leftovers ({len(all_leftovers)}) -> {args.leftovers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
