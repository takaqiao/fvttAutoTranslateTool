"""normalize_uuid_labels.py - one link target, one Chinese label.

`@UUID[...]{label}` and `@Compendium[...]{label}` name a specific document.  The
bracket body is machine data and is already guarded; the label is prose, and prose was
translated unit by unit, so the SAME condition ended up with two dozen renderings:

    Compendium.pf2e.conditionitems.Item.MIRkyAjyBeXivMa7
        力竭 1 (37) · 虚弱 1 (32) · 衰弱 1 (8) · 力竭（状态） 1 (2) · 力竭 Enfeebled 1 (5) ...

All of those are the same link. A player clicking two of them lands on the same sheet
and sees a third name on it.

The canonical name is not guessed: the English label at the SAME enricher in the
English baseline says which document it is, and the translation memory (wiki first)
says what that document is called in Chinese. That also survives a Remaster rename -
the baseline label moves from `Magic Missile` to `Force Barrage`, and the Chinese
follows - which a majority vote over the existing corpus would not.

Only targets that are actually inconsistent are touched. A target whose label is
already uniform is left alone even if the TM would word it differently: this fixes a
defect, it does not re-translate the corpus.

Usage:
  python normalize_uuid_labels.py --cn-dir <dir> --en-dir <dir> --tm <tm.json>
                                  [--rulings _uuid_labels.json] [--write] [--report r.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("rb", HERE / "repair_bracket_bodies.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)

CJK = re.compile(r"[一-鿿]")
LINK = re.compile(r"@(UUID|Compendium)\[([^\]]+)\]\{([^{}]*)\}")
# Trailing rank digits. The digits must NOT follow a letter, or a room code is
# taken for a rank and `区域 C15` becomes `区域 15` - a different, wrong area.
RANK = re.compile(r"[\s ]*(?<![A-Za-z0-9])([0-9]+)\s*$")
# a bilingual tail inside a label - labels are prose and must be pure Chinese
LATIN_TAIL = re.compile(r"[\s ]+[A-Za-z][A-Za-z''\- ]*$")
# parenthesised part-of-speech marker the compendium sometimes carries
POS_SUFFIX = re.compile(r"[（(](?:状态|特征|动作)[）)]\s*$")
# A label longer than this, or carrying sentence punctuation, is prose the translator
# wrote around the link, not the document's name.
MAX_LABEL = 22
SENTENCE = re.compile(r"[，。；：！？、]")


def base_of(label):
    """Strip rank digits, a bilingual English tail and a `（状态）` marker.

    The Latin tail is only stripped from a label that HAS a Chinese head. On a
    pure-English label there is no bilingual tail to remove, and stripping one turns
    `Shield Block` into `Shield` - which then resolves through the TM to 护盾术, the
    Shield *spell*, instead of 盾牌格挡, the Shield Block *feat*.
    """
    rank = None
    m = RANK.search(label)
    if m:
        rank = m.group(1)
        label = label[: m.start()]
    if CJK.search(label):
        label = LATIN_TAIL.sub("", label).strip()
    label = POS_SUFFIX.sub("", label).strip()
    return label.strip(), rank


def name_like(label):
    stem, _ = base_of(label)
    return bool(stem) and len(stem) <= MAX_LABEL and not SENTENCE.search(stem)


def collect(cn_dir, en_dir):
    """target -> (Counter of Chinese bases, Counter of English bases)."""
    zh = defaultdict(Counter)
    en = defaultdict(Counter)
    for cn_path in sorted(cn_dir.glob("*.json")):
        cn_data = json.loads(cn_path.read_text(encoding="utf-8")).get("entries", {})
        en_path = en_dir / cn_path.name
        en_flat = {}
        if en_path.exists():
            en_flat = {".".join(p): v for p, v in
                       rb.walk(json.loads(en_path.read_text(encoding="utf-8")).get("entries", {}))}
        for path, value in rb.walk(cn_data):
            hits = list(LINK.finditer(value))
            if not hits:
                continue
            ref = en_flat.get(".".join(path))
            en_hits = list(LINK.finditer(ref)) if ref else []
            aligned = len(en_hits) == len(hits) and all(
                a.group(2).strip() == b.group(2).strip() for a, b in zip(hits, en_hits))
            for i, m in enumerate(hits):
                target, label = m.group(2).strip(), m.group(3).strip()
                if label and CJK.search(label) and name_like(label):
                    zh[target][base_of(label)[0]] += 1
                if aligned:
                    en_label = en_hits[i].group(3).strip()
                    if en_label:
                        en[target][base_of(en_label)[0]] += 1
    return zh, en


def canonical_for(target, zh_counter, en_counter, tm, rulings):
    if target in rulings:
        return rulings[target], "ruling"
    en_name = en_counter.most_common(1)[0][0] if en_counter else None
    if en_name:
        entry = tm.get(en_name)
        if entry:
            head = rb_chinese_head(entry.get("name", ""))
            if head and head in zh_counter:
                return head, f"tm:{en_name}"
            if head:
                return head, f"tm-new:{en_name}"
    if zh_counter:
        return zh_counter.most_common(1)[0][0], "majority"
    return None, "unresolved"


# A TM value is `中文 English`, but the English half can carry parentheses and digits
# (`I型空间袋 Spacious Pouch (Type I)`), which the plain latin-tail pattern will not match.
TM_LATIN_TAIL = re.compile(r"[\s ]+[A-Za-z][ -~]*$")


def rb_chinese_head(value):
    """`力竭 Enfeebled` -> `力竭`;  `I型空间袋 Spacious Pouch (Type I)` -> `I型空间袋`."""
    if not value or not CJK.search(value):
        return None
    head = TM_LATIN_TAIL.sub("", value).strip()
    if CJK.search(head):
        return head or None
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--en-dir", required=True, type=Path)
    parser.add_argument("--tm", required=True, type=Path)
    parser.add_argument("--rulings", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    tm = json.loads(args.tm.read_text(encoding="utf-8"))
    rulings = {}
    if args.rulings and args.rulings.exists():
        rulings = {k: v for k, v in json.loads(args.rulings.read_text(encoding="utf-8")).items()
                   if not k.startswith("_")}

    zh, en = collect(args.cn_dir, args.en_dir)
    inconsistent = {t for t, c in zh.items() if len(c) > 1}
    print(f"link targets with a Chinese label: {len(zh)};  inconsistent: {len(inconsistent)}")

    decisions, unresolved = {}, []
    for target in sorted(inconsistent):
        name, how = canonical_for(target, zh[target], en.get(target, Counter()), tm, rulings)
        if name is None:
            unresolved.append(target)
            continue
        decisions[target] = {"canonical": name, "how": how,
                             "was": dict(zh[target]),
                             "english": (en.get(target) or Counter()).most_common(1)}
    by_how = Counter(d["how"].split(":")[0] for d in decisions.values())
    print(f"resolved {len(decisions)}  {dict(by_how)}   unresolved {len(unresolved)}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    grand, rows = Counter(), []
    for cn_path in sorted(args.cn_dir.glob("*.json")):
        data = json.loads(cn_path.read_text(encoding="utf-8"))
        changed = 0

        def fix(node):
            nonlocal changed
            if isinstance(node, dict):
                return {k: fix(v) for k, v in node.items()}
            if not isinstance(node, str) or "{" not in node:
                return node

            def sub(m):
                nonlocal changed
                target, label = m.group(2).strip(), m.group(3).strip()
                if not label or not CJK.search(label) or not name_like(label):
                    return m.group(0)
                stem, rank = base_of(label)
                if target in decisions:
                    # The term itself disagrees across the corpus: replace it.
                    canonical = decisions[target]["canonical"]
                else:
                    # The term is already uniform. Only the SHAPE may be wrong -
                    # `震慑（状态）`, `震慑2`, `震慑 Stunned 1` all carry the right term.
                    # Rebuild from the original label so a trailing qualifier that is
                    # NOT a bilingual echo (`奥塔里 Map`, `盔甲护腕 I`) survives.
                    canonical = label[: RANK.search(label).start()] if rank else label
                    canonical = POS_SUFFIX.sub("", canonical).strip()
                    tail = LATIN_TAIL.search(canonical)
                    if tail:
                        english = (en.get(target) or Counter()).most_common(1)
                        echo = english[0][0].lower() if english else None
                        if echo and tail.group(0).strip().lower() == echo:
                            canonical = canonical[: tail.start()].strip()
                    if not canonical or not CJK.search(canonical):
                        return m.group(0)
                new_label = f"{canonical} {rank}" if rank else canonical
                if new_label == label:
                    return m.group(0)
                changed += 1
                grand[f"{stem}->{canonical}"] += 1
                rows.append({"file": cn_path.name, "target": target,
                             "from": label, "to": new_label})
                return f"@{m.group(1)}[{m.group(2)}]{{{new_label}}}"

            return LINK.sub(sub, node)

        entries = fix(data.get("entries", {}))
        if not changed:
            continue
        print(f"[{'write' if args.write else 'dry  '}] {cn_path.name[:52]:<52} labels={changed}")
        if args.write:
            backup = cn_path.parent.parent / "_backup" / f"uuidlabels_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup / cn_path.name)
            data["entries"] = entries
            cn_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    print(f"\nlabels rewritten: {sum(grand.values())}")
    for key, n in grand.most_common(20):
        print(f"    {n:>4}  {key}")
    if unresolved:
        print(f"\nunresolved targets ({len(unresolved)}):")
        for t in unresolved[:15]:
            print(f"    {t}  {dict(zh[t])}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"decisions": decisions, "unresolved": unresolved,
                                           "rewrites": dict(grand), "rows": rows},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
