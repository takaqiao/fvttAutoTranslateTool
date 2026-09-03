"""repair_html_prefix.py - restore opening tags a translation dropped.

A recurring legacy defect: the Chinese leaf begins part-way into the English structure,
keeping the closing tags but not the opening ones -

  EN  <section class="box-text investigation"><header><img …><h2>Exploration</h2>…</header><article>…
  CN                                                        <h2>探索</h2>…</header><article>…

so the leaf is HTML-imbalanced and Foundry renders it wrong.  The missing prefix is
exactly recoverable from the English: find the k that makes the English tag sequence
line up with the Chinese one, and re-attach everything before the k-th tag.

This is the mirror of normalize_bilingual's tag-sequence split, and it is only applied
when the result is provably better:

  * the English is itself balanced (otherwise there is nothing sound to copy)
  * the repaired leaf is balanced
  * not one CJK character changed
  * no enricher was invented

Usage:
  python repair_html_prefix.py --cn-dir <dir> --en-dir <dir> [--write] [--report out.json]
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
spec = importlib.util.spec_from_file_location("nb", HERE / "normalize_bilingual.py")
nb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb)

CJK_CHAR = re.compile(r"[㐀-鿿]")


def count_cjk(text):
    return len(CJK_CHAR.findall(text))


def repair(cn, en):
    """Return the repaired string, or (None, reason)."""
    if nb.balanced(cn):
        return None, "already-balanced"
    if not nb.balanced(en):
        return None, "en-also-unbalanced"
    scn, sen = nb.seq(cn), nb.seq(en)
    k = len(sen) - len(scn)
    if k <= 0:
        return None, "no-prefix-room"
    if sen[k:] != scn:
        return None, "seq-mismatch"
    prefix = en[: nb.tag_positions(en)[k]]
    fixed = prefix + cn
    if not nb.balanced(fixed):
        return None, "still-unbalanced"
    if count_cjk(fixed) != count_cjk(cn):
        return None, "cjk-changed"
    if nb.enrichers(fixed) - (nb.enrichers(cn) + nb.enrichers(en)):
        return None, "enricher-invented"
    return fixed, None


def walk(cn_node, en_node, path, stats, fixes, opts):
    if isinstance(cn_node, dict):
        return {k: walk(v, (en_node or {}).get(k) if isinstance(en_node, dict) else None,
                        path + (k,), stats, fixes, opts)
                for k, v in cn_node.items()}
    if not isinstance(cn_node, str) or "<" not in cn_node:
        return cn_node
    if nb.balanced(cn_node):
        return cn_node
    stats["imbalanced"] += 1
    if not isinstance(en_node, str):
        stats["no-en-ref"] += 1
        fixes.append({"path": ".".join(path), "status": "no-en-ref", "cn": cn_node[:300]})
        return cn_node
    fixed, why = repair(cn_node, en_node)
    if fixed is None:
        stats[f"skip-{why}"] += 1
        fixes.append({"path": ".".join(path), "status": why,
                      "cn": cn_node[:400], "en": en_node[:400]})
        return cn_node
    stats["repaired"] += 1
    fixes.append({"path": ".".join(path), "status": "repaired",
                  "prefix": fixed[: len(fixed) - len(cn_node)]})
    return fixed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--en-dir", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    grand = Counter()
    all_fixes = []
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for cn_path in sorted(args.cn_dir.glob("*.json")):
        en_path = args.en_dir / cn_path.name
        if not en_path.exists():
            continue
        cn_data = json.loads(cn_path.read_text(encoding="utf-8"))
        en_data = json.loads(en_path.read_text(encoding="utf-8"))
        stats = Counter()
        fixes = []
        out = dict(cn_data)
        out["entries"] = walk(cn_data.get("entries", {}), en_data.get("entries", {}),
                              ("entries",), stats, fixes, args)
        for fix in fixes:
            fix["file"] = cn_path.name
        all_fixes.extend(fixes)
        grand.update(stats)
        if not stats["imbalanced"]:
            continue
        print(f"[{'write' if (args.write and stats['repaired']) else 'dry  '}] {cn_path.name[:56]:<56} "
              f"imbalanced={stats['imbalanced']:>3} repaired={stats['repaired']:>3}")
        if args.write and stats["repaired"]:
            backup = cn_path.parent.parent / "_backup" / f"html_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup / cn_path.name)
            cn_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    print(f"\nimbalanced {grand['imbalanced']}, repaired {grand['repaired']}")
    for key in sorted(k for k in grand if k.startswith("skip-") or k == "no-en-ref"):
        print(f"    {key:<26} {grand[key]}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"stats": dict(grand), "fixes": all_fixes},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
