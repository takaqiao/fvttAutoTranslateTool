"""scan_latin_nouns.py - find (and optionally fix) English left inside Chinese prose.

After the de-bilingual and enricher-label passes, what remains is English *words inside
otherwise Chinese sentences*: a rules term the translator skipped (`Shield Block`), a
condition (`Deafened`), an NPC the glossary never covered (`Idriniliss`).

Three things are deliberately NOT defects and are never touched:

  intentional  game abbreviations and brands that Chinese PF2e text keeps in Latin -
               XP / DC / AC / HP / GM / NPC / PC / Paizo / Pathfinder / Foundry ...
  legal        OGL and copyright blocks must stay in English verbatim; a leaf carrying
               those markers is skipped whole
  markup       anything inside a tag, an attribute, or an `@X[...]` bracket - only the
               VISIBLE text is considered

Resolution order for a candidate: the pack's own translated document names first
(self-consistent), then the merged TM, then case/plural variants.  Anything unresolved
is reported, never guessed.

Usage:
  python scan_latin_nouns.py --cn-dir <dir> --keys <pack-keys.json> --tm <tm.json>
                             [--apply] [--report out.json]
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
_spec = importlib.util.spec_from_file_location("nl", HERE / "normalize_enricher_labels.py")
nl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nl)

CJK = re.compile(r"[㐀-鿿]")
ENRICHER = re.compile(r"@[A-Za-z]+\[[^\]]*\](?:\{[^{}]*\})?|\[\[[^\]]*\]\](?:\{[^{}]*\})?")
TAG = re.compile(r"<[^>]+>")
TOKEN = re.compile(r"\b[A-Z][A-Za-z'’\-]*(?:\s+[A-Z][A-Za-z'’\-]*)*\b")

NAME_KEYS = {"name", "tokenName", "prototypeToken"}
PROSE_UNDER_NOTES = {"label", "summary", "text", "description", "content",
                     "public", "private", "gamemaster", "caption", "subtitle"}

# Latin that Chinese PF2e text keeps as Latin.
INTENTIONAL = {
    "XP", "DC", "AC", "HP", "GM", "GMs", "NPC", "NPCs", "PC", "PCs", "TN", "LG", "NG", "CG",
    "LN", "CN", "LE", "NE", "CE", "HP BT", "BT", "AR", "SP", "RP", "MAP", "PF", "PF2", "PF2e",
    "Paizo", "Paizo Inc", "Pathfinder", "Starfinder", "Foundry", "Foundry VTT", "Inc",
    "Wizards", "Coast", "Wizards of the Coast", "OGL", "SRD", "COPYRIGHT NOTICE",
    "Open Game License", "Syrinscape", "Discord", "Patreon", "Ko-fi", "GitHub",
    # Credit handles and third-party tool names: transliterating them is wrong.
    "PipNee", "NoxAg", "KraDup", "Carman", "Cynemi", "Chasarooni", "Tianze",
    "DungeonDraft", "Affinity Designer", "Bing Image Creator", "Forgotten Adventures",
    "Faststone Image Viewer", "Toolbelt", "Token", "Photo", "This", "Extras", "Footsteps",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "The", "A Note", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
}
# A leaf containing any of these is legal text and is skipped entirely.
LEGAL_MARKERS = ("OPEN GAME LICENSE", "Open Game License", "COPYRIGHT NOTICE",
                 "Community Content Agreement", "All Rights Reserved", "OGL")


def visible(text):
    text = ENRICHER.sub("\x00", text)
    text = TAG.sub("\x00", text)
    return re.sub(r"&[a-zA-Z#0-9]+;", "\x00", text)


def is_prose(path, key):
    if key in NAME_KEYS or "folders" in path:
        return False
    if len(path) >= 2 and path[-2] == "notes" and key not in PROSE_UNDER_NOTES:
        return False
    return True


def build_name_index(cn_data):
    """{English document key: Chinese-only name} from the translation itself."""
    index = {}

    def descend(node):
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if isinstance(value, dict):
                zh = nl.chinese_only(value.get("name", "")) if isinstance(value.get("name"), str) else None
                if zh and re.search(r"[A-Za-z]", key):
                    index.setdefault(key, zh)
                descend(value)

    descend(cn_data.get("entries", {}))
    return index


# Substituting into running prose is far less forgiving than replacing a `{label}`,
# so this pass is deliberately narrow.
TRUSTED_SOURCES = {"wiki", "pf2e_compendium"}
BAD_VALUE = re.compile(r"[0-9]|[。，；：！？]")


def _usable(entry):
    """A TM entry fit to drop into a sentence."""
    if not entry or entry.get("source") not in TRUSTED_SOURCES:
        return None
    zh = nl.chinese_only(entry.get("name", ""))
    # `Hit Points` resolves to "4生命值" because a statblock hp value reached the TM;
    # anything carrying a digit or sentence punctuation is not a term.
    if not zh or BAD_VALUE.search(zh) or len(zh) > 12:
        return None
    return zh


def resolve(token, name_index, tm):
    # The pack's own names are NOT used here. They are self-consistent for a `{label}`
    # that points at that very document, but as a dictionary they are polluted by the
    # pack's own mistranslations - `Shield Block` maps to 护盾术格挡 (the *spell* Shield)
    # where the compendium correctly says 盾牌格挡.
    zh = _usable(tm.get(token))
    if zh:
        return zh, "tm"
    for variant in nl._case_and_number_variants(token):
        zh = _usable(nl.tm_lower.get(variant))
        if zh:
            return zh, "tm-case"
    return None, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--tm", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    tm = json.loads(args.tm.read_text(encoding="utf-8"))
    nl.tm_lower = {}
    for english, entry in tm.items():
        nl.tm_lower.setdefault(english.lower(), entry)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    grand = Counter()
    resolved_all, unresolved_all = Counter(), Counter()
    samples = {}

    for cn_path in sorted(args.cn_dir.glob("*.json")):
        cn_data = json.loads(cn_path.read_text(encoding="utf-8"))
        name_index = build_name_index(cn_data)
        changed = 0

        def fix(node, path=()):
            nonlocal changed
            if isinstance(node, dict):
                return {k: fix(v, path + (k,)) for k, v in node.items()}
            if not isinstance(node, str):
                return node
            key = path[-1] if path else ""
            if not is_prose(path, key) or not CJK.search(node):
                return node
            if any(marker in node for marker in LEGAL_MARKERS):
                grand["skip-legal"] += 1
                return node
            vis = visible(node)
            hits = {m.group(0) for m in TOKEN.finditer(vis)}
            out = node
            for token in sorted(hits, key=len, reverse=True):
                if token in INTENTIONAL or len(token) < args.min_len:
                    continue
                zh, how = resolve(token, name_index, tm)
                if zh is None:
                    unresolved_all[token] += 1
                    samples.setdefault(token, str(cn_path.name) + ":" + ".".join(path[-2:]))
                    continue
                # whole-word, and only outside markup: rebuild by scanning the raw text
                pattern = re.compile(r"(?<![A-Za-z'’])" + re.escape(token) + r"(?![A-Za-z'’])")
                pieces, last = [], 0
                for m in re.finditer(r"@[A-Za-z]+\[[^\]]*\](?:\{[^{}]*\})?|\[\[[^\]]*\]\](?:\{[^{}]*\})?|<[^>]+>", out):
                    pieces.append(pattern.sub(zh, out[last:m.start()]))
                    pieces.append(m.group(0))
                    last = m.end()
                pieces.append(pattern.sub(zh, out[last:]))
                new = "".join(pieces)
                if new != out:
                    out = new
                    resolved_all[token] += 1
                    grand[how] += 1
            if out != node:
                changed += 1
            return out

        new_entries = fix(cn_data.get("entries", {}), ("entries",))
        if changed:
            print(f"[{'apply' if args.apply else 'dry  '}] {cn_path.name[:56]:<56} leaves changed {changed}")
            if args.apply:
                cn_data["entries"] = new_entries
                backup = cn_path.parent.parent / "_backup" / f"latin_{stamp}"
                backup.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cn_path, backup / cn_path.name)
                cn_path.write_text(json.dumps(cn_data, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8", newline="\n")

    print(f"\nresolved   {sum(resolved_all.values())} occurrences / {len(resolved_all)} distinct  {dict(grand)}")
    for token, n in resolved_all.most_common(15):
        print(f"    {n:>4}  {token[:44]}")
    print(f"\nunresolved {sum(unresolved_all.values())} occurrences / {len(unresolved_all)} distinct")
    for token, n in unresolved_all.most_common(25):
        print(f"    {n:>4}  {token[:44]:<44} e.g. {samples.get(token, '')[:60]}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(
            {"resolved": dict(resolved_all), "unresolved": dict(unresolved_all),
             "how": dict(grand), "samples": samples}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
