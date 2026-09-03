"""strip_english_suffix.py - remove an appended English block the tag-sequence split missed.

`normalize_bilingual.py` splits `中文 + English` by comparing HTML tag SEQUENCES. That
works for prose, and it took the corpus from 3,979 bilingual leaves to zero *by its own
measure*.  It cannot see a leaf whose two halves have the same tag sequence for another
reason, or one with almost no markup at all:

    <p>（专家）</p>\\n<p>(expert)</p>            <- stealthdetails, 2 tags either way
    <p>爪击</p><hr /><p>@Localize[..Rend]</p><p>Claw</p><hr /><p>@Localize[..Rend]</p>

Those survived, and the second one is worse than cosmetic: the appended English block
carries its own `@Localize` and `@Check`, so the leaf renders the same ability twice.

This uses a different and much stricter signal - the leaf must literally END with the
English baseline for the same path - so it cannot fire on a leaf that is merely long.
The comparison normalises only what a round-trip through an HTML serialiser changes
(`<hr />` vs `<hr>`, whitespace runs); nothing else is allowed to differ.

Usage:
  python strip_english_suffix.py --cn-dir <dir> --en-dir <dir> [--write] [--report out.json]
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

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("nb", HERE / "normalize_bilingual.py")
nb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb)

CJK = re.compile(r"[一-鿿]")
VOID_SELF_CLOSE = re.compile(r"<(br|hr|img|input|meta|link|col|source|area|base|wbr|embed)([^>]*?)\s*/>")
WS = re.compile(r"\s+")
# Minimum share of the leaf that must survive. A "translation" that is 90% appended
# English is not a translation, and silently keeping 10% of it would hide that.
MIN_KEEP_RATIO = 0.15
# A bilingual NAME is `中文 English` by design - it ends with the English on purpose and
# must never be stripped. Same for the short English tail of a folder label.
NAME_KEYS = {"name", "tokenName", "prototypeToken"}
# Below this, an English "suffix" is just a common word landing at the end by chance.
MIN_EN_CHARS = 24


def norm(text):
    text = VOID_SELF_CLOSE.sub(r"<\1\2>", text)
    return WS.sub(" ", text).strip()


def strip(cn, en):
    """Return (head, reason). head is None when nothing may safely be removed."""
    ncn, nen = norm(cn), norm(en)
    if not nen or ncn == nen:
        return None, "identical-or-empty-en"
    if not ncn.endswith(nen):
        return None, "no-en-suffix"

    # Map the normalised cut point back onto the raw string: walk the raw text from the
    # end, skipping the same number of *significant* characters the normalised tail has.
    keep_norm = ncn[: -len(nen)]
    if not keep_norm.strip():
        return None, "head-empty"
    # Find the raw offset by re-normalising growing prefixes; the leaf is short enough
    # (these are item descriptions, not the Audio Credits blob) for this to be cheap.
    keep_norm = keep_norm.strip()
    cut = None
    for i in range(len(cn), 0, -1):
        if norm(cn[:i]) == keep_norm:
            cut = i
            break
    if cut is None:
        return None, "cut-not-locatable"

    head = cn[:cut].rstrip()
    if not CJK.search(head):
        return None, "head-has-no-cjk"
    if CJK.search(cn[cut:]):
        return None, "tail-has-cjk"
    if len(head) < MIN_KEEP_RATIO * len(cn):
        return None, "head-too-small"
    if not nb.balanced(head):
        return None, "head-unbalanced"
    if nb.enrichers(head) - nb.enrichers(cn):
        return None, "enricher-invented"
    return head, "stripped"


def walk_fix(cn_node, en_node, path, stats, rows, filename):
    if isinstance(cn_node, dict):
        return {k: walk_fix(v, (en_node or {}).get(k) if isinstance(en_node, dict) else None,
                            path + (k,), stats, rows, filename)
                for k, v in cn_node.items()}
    if not isinstance(cn_node, str) or not isinstance(en_node, str):
        return cn_node
    # Reuse normalize_bilingual's own predicate rather than re-deciding what is a name:
    # name/tokenName/prototypeToken, anything under folders, and scene note keys are
    # bilingual BY DESIGN. Stripping them would be the 287-leaf accident again.
    if path and nb.should_keep_bilingual(path, path[-1]):
        stats["skip-bilingual-by-design"] += 1
        return cn_node
    if not CJK.search(cn_node) or len(norm(en_node)) < MIN_EN_CHARS:
        return cn_node
    head, reason = strip(cn_node, en_node)
    stats[reason] += 1
    if head is None:
        return cn_node
    rows.append({"file": filename, "path": ".".join(path[-3:]),
                 "removed": cn_node[len(head):][:200], "kept": head[:200]})
    return head


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--en-dir", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    grand, all_rows = Counter(), []
    for cn_path in sorted(args.cn_dir.glob("*.json")):
        en_path = args.en_dir / cn_path.name
        if not en_path.exists():
            continue
        cn_data = json.loads(cn_path.read_text(encoding="utf-8"))
        en_data = json.loads(en_path.read_text(encoding="utf-8"))
        stats, rows = Counter(), []
        entries = walk_fix(cn_data.get("entries", {}), en_data.get("entries", {}),
                           ("entries",), stats, rows, cn_path.name)
        grand.update(stats)
        all_rows.extend(rows)
        if not rows:
            continue
        print(f"[{'write' if args.write else 'dry  '}] {cn_path.name[:52]:<52} stripped {len(rows)}")
        if args.write:
            backup = cn_path.parent.parent / "_backup" / f"ensuffix_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup / cn_path.name)
            cn_data["entries"] = entries
            cn_path.write_text(json.dumps(cn_data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    print(f"\nstripped {grand['stripped']} leaves")
    for key in sorted(k for k in grand if k != "stripped" and k != "no-en-suffix"):
        print(f"    refused: {key:<24} {grand[key]}")
    print(f"    (no English suffix at all: {grand['no-en-suffix']})")
    for row in all_rows[:12]:
        print(f"\n  {row['file'][:30]}:{row['path'][:44]}")
        print(f"      kept   : {row['kept'][:110]}")
        print(f"      removed: {row['removed'][:110]}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"stats": dict(grand), "rows": all_rows},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
