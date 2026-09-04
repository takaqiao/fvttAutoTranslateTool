"""strip_original_marker.py - remove the `原文:` block an older translator appended.

`冒险/其他冒险/fvttproject/pf2e_translator.py` (the 2026-01 Gemini-based pass) wrote the
Chinese and then appended the source text behind a separator of its own making:

    …译文…<br><br><hr><b>原文:</b><br><p>The first encounter happens in…</p>

That is not the bilingual convention - it is a debugging aid that shipped. It also defeats
`strip_english_suffix.py`, because the appended part is usually only the FIRST paragraph of
the English, so the leaf does not end with the baseline and the suffix test cannot fire.

The marker is the tool's own annotation, so the split point is not inferred: everything
from `<hr><b>原文:</b>` onward goes. Three guards still apply, because a marker inside
genuine prose would otherwise truncate a page:

  * the head must still contain Chinese
  * the head must be HTML-balanced on its own
  * the head must keep every enricher the head had (nothing invented)

Usage:
  python strip_original_marker.py --cn-dir <dir> [--write] [--report out.json]
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
# The separator the old tool emitted. <br> runs and self-closing forms both occur.
MARKER = re.compile(r"(?:<br\s*/?>\s*)*<hr\s*/?>\s*<b>\s*原文\s*[:：]\s*</b>\s*(?:<br\s*/?>\s*)*",
                    re.IGNORECASE)


def strip_marker(text):
    m = MARKER.search(text)
    if not m:
        return None, "no-marker"
    head = text[: m.start()].rstrip()
    tail = text[m.end():]
    if not CJK.search(head):
        return None, "head-has-no-cjk"
    if not nb.balanced(head):
        return None, "head-unbalanced"
    if nb.enrichers(head) - nb.enrichers(text):
        return None, "enricher-invented"
    if CJK.search(tail) and len(CJK.findall(tail)) > 20:
        # The block after the marker is substantially Chinese, so it is not the appended
        # source; dropping it would delete a translation.
        return None, "tail-is-chinese"
    return head, "stripped"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    grand, rows = Counter(), []
    for cn_path in sorted(args.cn_dir.glob("*.json")):
        data = json.loads(cn_path.read_text(encoding="utf-8"))
        changed = 0

        def fix(node, path=()):
            nonlocal changed
            if isinstance(node, dict):
                return {k: fix(v, path + (k,)) for k, v in node.items()}
            if not isinstance(node, str) or "原文" not in node:
                return node
            head, why = strip_marker(node)
            grand[why] += 1
            if head is None:
                if why != "no-marker":
                    rows.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                                 "why": why, "text": node[:200]})
                return node
            changed += 1
            rows.append({"file": cn_path.name, "path": ".".join(path[-3:]), "why": "stripped",
                         "removed_chars": len(node) - len(head)})
            return head

        entries = fix(data.get("entries", {}), ("entries",))
        if not changed:
            continue
        print(f"[{'write' if args.write else 'dry  '}] {cn_path.name[:52]:<52} stripped {changed}")
        if args.write:
            backup = cn_path.parent.parent / "_backup" / f"origmarker_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup / cn_path.name)
            data["entries"] = entries
            cn_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    print(f"\n{dict(grand)}")
    for row in [r for r in rows if r["why"] != "stripped"][:10]:
        print(f"  [{row['why']:<18}] {row['file'][:28]}:{row['path'][-40:]}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"stats": dict(grand), "rows": rows},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
