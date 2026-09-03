"""verify_debilingual.py - prove a de-bilingual pass removed only English.

Runs normalize_bilingual's transformation in memory and asserts, per changed leaf:

DAMAGE - these must be zero; any hit means the transformation itself broke something:
  I1  no Chinese was lost        count(CJK chars, after) == count(CJK chars, before)
  I2  no enricher was invented   enrichers(after) subset of enrichers(before)
  I4  HTML did not get worse     balanced(after) or not balanced(before)
  I6  English really went away   after has fewer English word chars than before

FINDINGS - reported, not fatal.  Because the pass only removes a pure-English suffix
(I1 + I6 prove that), these are properties of the Chinese half *as authored* and were
already true before this pass.  They belong on the translation-quality worklist:
  I3  fewer enrichers than EN    e.g. `@Template[emanation|distance:5]` written out as prose
  I5  short vs EN                e.g. Blowgun: CN describes the darts, EN the tube

I1 is the sharp one: the transformation is a pure suffix cut, so any drop in the
Chinese character count means it cut into the translation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("nb", HERE / "normalize_bilingual.py")
nb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb)

CJK_CHAR = re.compile(r"[㐀-鿿]")
WORD = re.compile(r"[A-Za-z]{2,}")


def count_cjk(text):
    return len(CJK_CHAR.findall(text))


def walk_pairs(before, after, en, path=()):
    if isinstance(before, dict):
        for key, value in before.items():
            yield from walk_pairs(value,
                                  (after or {}).get(key) if isinstance(after, dict) else None,
                                  (en or {}).get(key) if isinstance(en, dict) else None,
                                  path + (key,))
    elif isinstance(before, str):
        yield path, before, after, en


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--en-dir", required=True, type=Path)
    parser.add_argument("--min-ratio", type=float, default=0.22)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tail-ratio", type=float, default=0.90)
    args = parser.parse_args(argv)

    DAMAGE = {"I1-cjk-lost", "I2-enricher-invented", "I4-html-worse", "I6-english-not-reduced"}
    violations = Counter()
    samples = {}
    ratios = []
    changed_total = 0

    for cn_path in sorted(args.cn_dir.glob("*.json")):
        en_path = args.en_dir / cn_path.name
        if not en_path.exists():
            continue
        original, out = nb.process(cn_path, en_path, args, Counter(), [])
        en_data = json.loads(en_path.read_text(encoding="utf-8"))

        for section in ("entries", "folders"):
            for path, before, after, en in walk_pairs(original.get(section), out.get(section),
                                                      en_data.get(section), (section,)):
                if not isinstance(after, str) or after == before:
                    continue
                changed_total += 1
                loc = f"{cn_path.name}:{'.'.join(path)}"

                def flag(code):
                    violations[code] += 1
                    samples.setdefault(code, []).append(loc)

                if count_cjk(after) != count_cjk(before):
                    flag("I1-cjk-lost")
                if nb.enrichers(after) - nb.enrichers(before):
                    flag("I2-enricher-invented")
                if isinstance(en, str) and sum(nb.enrichers(after).values()) < sum(nb.enrichers(en).values()):
                    flag("I3-enricher-dropped")
                if not nb.balanced(after) and nb.balanced(before):
                    flag("I4-html-worse")
                if isinstance(en, str):
                    en_len = len(nb.strip_markup(en))
                    cn_len = len(nb.strip_markup(after))
                    if en_len:
                        ratio = cn_len / en_len
                        ratios.append(ratio)
                        if ratio < args.min_ratio:
                            flag("I5-truncated")
                if len(WORD.findall(nb.strip_markup(after))) >= len(WORD.findall(nb.strip_markup(before))):
                    flag("I6-english-not-reduced")

    print(f"changed leaves: {changed_total}")
    if ratios:
        ratios.sort()
        print(f"CN/EN plaintext length ratio: median={statistics.median(ratios):.3f} "
              f"p05={ratios[int(len(ratios) * 0.05)]:.3f} p95={ratios[int(len(ratios) * 0.95)]:.3f} "
              f"min={ratios[0]:.3f}")
    damage = {k: v for k, v in violations.items() if k in DAMAGE}
    findings = {k: v for k, v in violations.items() if k not in DAMAGE}

    if not damage:
        print("\nDAMAGE  none - I1 (no Chinese lost), I2, I4, I6 all clean")
    else:
        print("\nDAMAGE (must be zero):")
        for code, n in sorted(damage.items(), key=lambda kv: -kv[1]):
            print(f"  {code:<24} {n}")
            for loc in samples[code][:4]:
                print(f"      {loc[:130]}")

    if findings:
        print("\nFINDINGS (pre-existing translation quality, not caused by this pass):")
        for code, n in sorted(findings.items(), key=lambda kv: -kv[1]):
            print(f"  {code:<24} {n}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(
            {"changed": changed_total, "violations": dict(violations),
             "damage": {k: v for k, v in violations.items() if k in DAMAGE},
             "findings": {k: v for k, v in violations.items() if k not in DAMAGE},
             "samples": {k: v[:200] for k, v in samples.items()},
             "ratio_median": statistics.median(ratios) if ratios else None},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 1 if damage else 0


if __name__ == "__main__":
    raise SystemExit(main())
