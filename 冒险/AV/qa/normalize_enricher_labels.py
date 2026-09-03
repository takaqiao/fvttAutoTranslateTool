"""normalize_enricher_labels.py - translate the `{label}` tail of enrichers.

Labels are prose read by the player, so per the project standard they are pure Chinese
(the bilingual `中文 English` form belongs to `name` fields only).

Only the `{...}` tail is ever touched.  Everything inside `@X[...]` and `[[...]]` is a
machine reference and is copied byte for byte; the tool asserts before/after that the
bracket bodies and the enricher count are identical, and refuses to write otherwise.

Resolution ladder, first hit wins:
  1. overrides      hand-reviewed `_labels.json`
  2. reference      resolve the id inside the brackets against this pack's own
                    translated document name, then drop its bilingual English tail.
                    Self-consistent by construction, so it beats any dictionary.
  3. tm             exact hit in the merged translation memory
  4. tm+rank        `Stunned 1` -> `Stunned` + rank suffix (PF2e valued conditions)
  5. keep-latin     room codes (`C35`, `B17`) and similar identifiers - not English
  6. unresolved     reported for in-session translation

Usage:
  python normalize_enricher_labels.py --cn-dir <dir> --keys <pack-keys.json> \
      --tm <tm_3source.json> [--overrides _labels.json] [--write] [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

CJK = re.compile(r"[㐀-鿿]")
# group 1/2: @Kind[target]{label}   group 3/4: [[...]]{label}
ENRICHER_LABEL = re.compile(r"(@[A-Za-z]+\[[^\]]*\])\{([^{}]*)\}|(\[\[[^\]]*\]\])\{([^{}]*)\}")
BRACKET_BODY = re.compile(r"@[A-Za-z]+\[([^\]]*)\]|\[\[([^\]]*)\]\]")
ENRICHER_COUNT = re.compile(r"@[A-Za-z]+\[|\[\[")
# Room / area identifiers: A08, B31, I59, G04a - identifiers, not English prose.
ROOM_CODE = re.compile(r"^[A-Z]\d{1,2}[a-z]?$")
RANK_SUFFIX = re.compile(r"^(?P<base>.+?)\s+(?P<rank>\d+)$")
# Dice, modifiers and similar notation are not English prose either.
NOTATION = re.compile(r"^[+-]?\d+$|^\d*d\d+([+-]\d+)?$|^DC\s*\d+$", re.I)
FOUNDRY_ID = re.compile(r"[A-Za-z0-9]{16}")
GRADE_PREFIX = re.compile(r"^(?P<grade>Lesser|Moderate|Greater|Major|Minor|Superb|Expanded|True)\s+(?P<rest>.+)$")
# `10 feet`, `1d6 days`, `2d4 rounds` - a quantity plus a unit; only the unit is prose.
MEASURE = re.compile(r"^(?P<qty>[+-]?\d+(?:d\d+)?(?:[+-]\d+)?)\s+(?P<unit>[A-Za-z]+)$")
UNITS = {
    "feet": "尺", "foot": "尺", "ft": "尺", "mile": "哩", "miles": "哩",
    "round": "轮", "rounds": "轮", "minute": "分钟", "minutes": "分钟",
    "hour": "小时", "hours": "小时", "day": "天", "days": "天",
    "week": "周", "weeks": "周", "month": "月", "months": "月",
    "year": "年", "years": "年",
}


# English tails routinely carry typographic punctuation (`I09. Hunters’`), so an
# ASCII-only class leaves them half-stripped.
LATIN_TAIL = re.compile(r"\s+[\x20-\x7E‘’–—]+$")


def chinese_only(name: str) -> str | None:
    """`措手不及 Off-Guard` -> `措手不及`; return None when there is no Chinese."""
    if not isinstance(name, str) or not CJK.search(name):
        return None
    trimmed = LATIN_TAIL.sub("", name.strip())
    return trimmed.strip() or None


def build_reference_index(keys_manifest, collection_id, cn_data):
    """{foundry _id: translated-name} for this pack, from the key manifest + CN file."""
    pack = keys_manifest["packs"].get(collection_id)
    if not pack:
        return {}
    index = {}

    def descend(node_path, trans):
        node = nodes.get(node_path)
        if node is None or not isinstance(trans, dict):
            return
        for entry in node["entries"]:
            matched = next((c for c in entry["matchCandidates"] if c in trans), None)
            if matched is None:
                continue
            value = trans[matched]
            if isinstance(value, dict):
                zh = chinese_only(value.get("name", ""))
                if zh and entry["_id"]:
                    index[entry["_id"]] = zh
                prefix = node_path + "." + entry["key"] + "."
                for child in nodes:
                    if child.startswith(prefix) and "." not in child[len(prefix):]:
                        descend(child, value.get(child[len(prefix):]))

    nodes = {n["path"]: n for n in pack["nodes"]}
    descend("entries", cn_data.get("entries"))
    return index


def _case_and_number_variants(label):
    """Lowercase forms of the label, plus a de-pluralised one."""
    base = label.lower()
    out = [base]
    if base.endswith("ies") and len(base) > 4:
        out.append(base[:-3] + "y")
    if base.endswith("es") and len(base) > 3:
        out.append(base[:-2])
    if base.endswith("s") and len(base) > 2:
        out.append(base[:-1])
    seen, uniq = set(), []
    for v in out:
        if v and v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


# Case-insensitive view of the TM, built once per run by main().
tm_lower = {}


def resolve(raw_label, target, ref_index, tm, overrides, stats):
    label = raw_label.strip()
    if label in overrides:
        stats["overrides"] += 1
        return overrides[label]

    if ROOM_CODE.match(label) or NOTATION.match(label):
        stats["keep-latin"] += 1
        return label

    m = MEASURE.match(label)
    if m and m.group("unit").lower() in UNITS:
        stats["measure"] += 1
        return f"{m.group('qty')} {UNITS[m.group('unit').lower()]}"

    for candidate in reversed(FOUNDRY_ID.findall(target or "")):
        if candidate in ref_index:
            stats["reference"] += 1
            return ref_index[candidate]

    entry = tm.get(label)
    if entry:
        zh = chinese_only(entry.get("name", ""))
        if zh:
            stats["tm"] += 1
            return zh

    m = RANK_SUFFIX.match(label)
    if m:
        entry = tm.get(m.group("base"))
        if entry:
            zh = chinese_only(entry.get("name", ""))
            if zh:
                stats["tm-rank"] += 1
                return f"{zh} {m.group('rank')}"

    # PF2e item grades sort as a parenthetical: the compendium stores
    # `Repair Kit (Superb)` / `Cheetah's Elixir (Greater)`, prose writes them inverted.
    m = GRADE_PREFIX.match(label)
    if m:
        entry = tm.get(f"{m.group('rest')} ({m.group('grade')})")
        if entry:
            zh = chinese_only(entry.get("name", ""))
            if zh:
                stats["tm-grade"] += 1
                return zh

    # Prose writes creature and skill names in running case and often in the plural
    # (`leng spider`, `flumphs`, `athletics`); the TM is keyed in the compendium's
    # Title Case singular.
    for variant in _case_and_number_variants(label):
        entry = tm_lower.get(variant)
        if entry:
            zh = chinese_only(entry.get("name", ""))
            if zh:
                stats["tm-case"] += 1
                return zh

    # Index-style labels invert the head word: `Goblin, Charhide` is the compendium's
    # sort form of `Charhide Goblin`.
    if ", " in label:
        head, _, tail = label.partition(", ")
        entry = tm.get(f"{tail} {head}")
        if entry:
            zh = chinese_only(entry.get("name", ""))
            if zh:
                stats["tm-inverted"] += 1
                return zh

    return None


def process_text(text, ref_index, tm, overrides, stats, unresolved, where):
    def replace(match):
        target = match.group(1) or match.group(3)
        label = match.group(2) if match.group(2) is not None else match.group(4)
        if not label.strip() or CJK.search(label):
            stats["already-chinese" if label.strip() else "empty-label"] += 1
            return match.group(0)
        zh = resolve(label, target, ref_index, tm, overrides, stats)
        if zh is None:
            stats["unresolved"] += 1
            unresolved[label] += 1
            return match.group(0)
        return f"{target}{{{zh}}}"

    return ENRICHER_LABEL.sub(replace, text)


def walk(node, ref_index, tm, overrides, stats, unresolved, path=()):
    if isinstance(node, dict):
        return {k: walk(v, ref_index, tm, overrides, stats, unresolved, path + (k,)) for k, v in node.items()}
    if isinstance(node, list):
        return [walk(v, ref_index, tm, overrides, stats, unresolved, path + (str(i),)) for i, v in enumerate(node)]
    if not isinstance(node, str) or "{" not in node:
        return node
    return process_text(node, ref_index, tm, overrides, stats, unresolved, ".".join(path))


def assert_machine_parts_intact(before, after, where):
    """Bracket bodies and enricher counts must be byte-identical."""
    if ENRICHER_COUNT.findall(before) != ENRICHER_COUNT.findall(after):
        raise AssertionError(f"enricher count changed at {where}")
    if BRACKET_BODY.findall(before) != BRACKET_BODY.findall(after):
        raise AssertionError(f"bracket body changed at {where}")


def collect_strings(node, path=(), out=None):
    out = {} if out is None else out
    if isinstance(node, dict):
        for k, v in node.items():
            collect_strings(v, path + (k,), out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            collect_strings(v, path + (str(i),), out)
    elif isinstance(node, str):
        out[".".join(path)] = node
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--tm", required=True, type=Path)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--tree", type=Path,
                        help="also process an arbitrary JSON tree (e.g. a module's own "
                             "languages/cn.json), walking every leaf instead of `entries`")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    keys_manifest = json.loads(args.keys.read_text(encoding="utf-8"))
    tm = json.loads(args.tm.read_text(encoding="utf-8"))
    global tm_lower
    tm_lower = {}
    for english, entry in tm.items():
        tm_lower.setdefault(english.lower(), entry)
    overrides = {}
    if args.overrides and args.overrides.exists():
        overrides = json.loads(args.overrides.read_text(encoding="utf-8"))

    grand = Counter()
    unresolved_all = Counter()
    report = {}
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # Foundry ids are globally unique, so one merged index resolves cross-pack links.
    global_ref = {}
    loaded = {}
    for cn_path in sorted(args.cn_dir.glob("*.json")):
        loaded[cn_path] = json.loads(cn_path.read_text(encoding="utf-8"))
        global_ref.update(build_reference_index(keys_manifest, cn_path.stem, loaded[cn_path]))
    print(f"global reference index: {len(global_ref)} ids across {len(loaded)} packs\n")

    for cn_path in sorted(args.cn_dir.glob("*.json")):
        collection_id = cn_path.stem
        cn_data = loaded[cn_path]
        ref_index = global_ref

        stats = Counter()
        unresolved = Counter()
        before_strings = collect_strings(cn_data.get("entries"))
        out = dict(cn_data)
        out["entries"] = walk(cn_data.get("entries"), ref_index, tm, overrides, stats, unresolved, ("entries",))
        after_strings = collect_strings(out.get("entries"))
        for key, before in before_strings.items():
            assert_machine_parts_intact(before, after_strings.get(key, before), f"{cn_path.name}:{key}")

        grand.update(stats)
        unresolved_all.update(unresolved)
        changed = out != cn_data
        report[cn_path.name] = {"changed": changed, "stats": dict(stats),
                                "ref_index": len(ref_index), "unresolved": dict(unresolved)}
        print(f"[{'write' if (args.write and changed) else 'dry  '}] {cn_path.name[:56]:<56} "
              f"ref={stats['reference']:>4} tm={stats['tm']:>4} rank={stats['tm-rank']:>3} "
              f"ovr={stats['overrides']:>3} code={stats['keep-latin']:>4} unresolved={stats['unresolved']:>4}")

        if args.write and changed:
            backup = cn_path.parent.parent / "_backup" / f"labels_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup / cn_path.name)
            cn_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    if args.tree and args.tree.exists():
        tree = json.loads(args.tree.read_text(encoding="utf-8"))
        stats = Counter()
        unresolved = Counter()
        before = collect_strings(tree)
        out = walk(tree, global_ref, tm, overrides, stats, unresolved, ())
        after = collect_strings(out)
        for key, value in before.items():
            assert_machine_parts_intact(value, after.get(key, value), f"{args.tree.name}:{key}")
        grand.update(stats)
        unresolved_all.update(unresolved)
        changed = out != tree
        report[args.tree.name] = {"changed": changed, "stats": dict(stats),
                                  "unresolved": dict(unresolved)}
        print(f"[{'write' if (args.write and changed) else 'dry  '}] {args.tree.name[:56]:<56} "
              f"ref={stats['reference']:>4} tm={stats['tm']:>4} rank={stats['tm-rank']:>3} "
              f"ovr={stats['overrides']:>3} code={stats['keep-latin']:>4} unresolved={stats['unresolved']:>4}")
        if args.write and changed:
            backup = args.tree.parent.parent.parent / "_backup" / f"labels_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.tree, backup / args.tree.name)
            args.tree.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8", newline="\n")

    total = sum(grand[k] for k in ("reference", "tm", "tm-rank", "tm-inverted", "tm-grade",
                                   "tm-case", "measure", "overrides", "keep-latin",
                                   "unresolved"))
    print(f"\nlabels considered {total}")
    for key in ("reference", "tm", "tm-rank", "tm-inverted", "tm-grade", "measure", "overrides",
                "keep-latin", "unresolved", "already-chinese", "empty-label"):
        if grand[key]:
            print(f"    {key:<18} {grand[key]}")
    if unresolved_all:
        print(f"\ntop unresolved ({len(unresolved_all)} distinct):")
        for label, n in unresolved_all.most_common(30):
            print(f"    {n:>4}  {label[:60]}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(
            {"files": report, "unresolved": dict(unresolved_all.most_common())},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
