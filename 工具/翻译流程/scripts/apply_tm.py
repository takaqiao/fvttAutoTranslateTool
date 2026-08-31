"""Apply 3-source TM with aggressive multi-strategy lookup.

Strategies (in order):
1. Direct lookup
2. Strip FVTT 4-char id suffix " (XXXX)"
3. Strip generic parenthetical clauses iteratively
4. Reorder "Greater X" -> "X (Greater)"
5. Strip leading runes (potency + property)
6. Apostrophe variant
7. Spell-trad / Lore / natural-attack synthesis (fallback)

Usage: python apply_tm.py <input.json> [output.json]
       (in-place if output omitted)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # back to fvtt root
TM_PATH = ROOT / "翻译流程" / "tm_cache" / "tm_3source.json"
if not TM_PATH.exists():
    print(f"TM not found at {TM_PATH}; run build_3source_tm.py first")
    sys.exit(1)
TM = json.load(open(TM_PATH, encoding="utf-8"))

POTENCY = re.compile(r'^\+\d+\s+', re.IGNORECASE)
PROPERTY_RUNE = re.compile(
    r'^((Major|Greater|Moderate|Lesser|Minor|True|Gerater)\s+)?'
    r'(Resilient|Striking|Disrupting|Astral|Hopeless|Bloodbane|Brilliant|Crushing|Dancing|'
    r'Anarchic|Axiomatic|Holy|Unholy|Frost|Flaming|Shock|Corrosive|Thundering|Fearsome|'
    r'Speed|Spell-?storing|Vorpal|Ghost\s+Touch|Returning|Wounding|Keen|Grievous|Cunning|'
    r'Energy-Resistant|Glamered|Invisibility|Slick|Shadow|Stanching|Winged)\s+', re.IGNORECASE
)


def strip_all_parens(s):
    """Iteratively remove " (anything)" suffixes from end of string."""
    cur = s
    while True:
        m = re.search(r'\s+\([^)]*\)\s*$', cur)
        if not m:
            break
        cur = cur[: m.start()].rstrip()
    return cur


def reorder_greater(s):
    """Try 'Greater X' -> 'X (Greater)' or 'Major X' -> 'X (Major)' etc."""
    m = re.match(r'^(Greater|Major|Lesser|Minor|Moderate|True)\s+(.+)$', s)
    if m:
        return f"{m.group(2)} ({m.group(1)})"
    return None


def lookup_aggressive(key):
    if key in TM:
        return TM[key], "exact"
    # Strip FVTT 4-char id
    cur = re.sub(r'\s+\([A-Za-z0-9]{4}\)$', '', key).rstrip()
    if cur != key and cur in TM:
        return TM[cur], "fvtt_id"
    # Strip generic parens
    no_paren = strip_all_parens(cur)
    if no_paren != cur and no_paren in TM:
        return TM[no_paren], "all_parens"
    # Try reordering "Greater X" -> "X (Greater)"
    reord = reorder_greater(cur)
    if reord and reord in TM:
        return TM[reord], "reorder"
    if reord:
        # Reorder + strip 4-char (no need)
        # Try with parens stripped
        np = strip_all_parens(reord)
        if np in TM and np != reord:
            return TM[np], "reorder_strip"
    # Strip leading runes
    rune_cur = no_paren
    runes = []
    for _ in range(8):
        m = POTENCY.match(rune_cur)
        if m:
            runes.append(m.group(0).strip())
            rune_cur = rune_cur[m.end():]
            continue
        m = PROPERTY_RUNE.match(rune_cur)
        if m:
            runes.append(m.group(0).strip())
            rune_cur = rune_cur[m.end():]
            continue
        break
    if rune_cur != no_paren and rune_cur in TM:
        return TM[rune_cur], "rune_strip"
    # apos variant
    if "'" in rune_cur:
        no_apos = rune_cur.replace("'", "")
        if no_apos in TM:
            return TM[no_apos], "apos"
    return None, "miss"


def parse_path(p):
    parts = []
    buf = ""
    i = 0
    while i < len(p):
        c = p[i]
        if c == '/':
            if buf:
                parts.append(("k", buf))
                buf = ""
        elif c == '[':
            if buf:
                parts.append(("k", buf))
                buf = ""
            j = p.index(']', i)
            parts.append(("i", int(p[i+1:j])))
            i = j
        else:
            buf += c
        i += 1
    if buf:
        parts.append(("k", buf))
    return parts


def get_item_info(p):
    """Return (lookup_key, mode) where mode controls how to apply TM result.

    Modes:
      'name'         — bilingual "中文 English" (top-level entry/item name)
      'description'  — pure Chinese description
      'token'        — Chinese-only token name (extracted from bilingual)
    """
    parts = parse_path(p)
    n = len(parts)
    # /entries/<key>/items/<X>/{name,description}
    for i in range(n - 2):
        if parts[i] == ("k", "items"):
            kind, key = parts[i+1]
            if kind == "k":
                last = parts[-1]
                if last == ("k", "name"):
                    return key, "name"
                if last == ("k", "description"):
                    return key, "description"
    # /entries/<key>/name  (top-level entry name)
    if n >= 3 and parts[0] == ("k", "entries") and parts[-1] == ("k", "name") and parts[-2][0] == "k":
        # Pattern check: just /entries/<X>/name (length 3) or /entries/<X>/items/<Y>/name (handled above)
        if n == 3:
            return parts[1][1], "name"
    # /entries/<key>/prototypeToken/name  → Chinese-only token
    if n >= 4 and parts[0] == ("k", "entries") and parts[-2] == ("k", "prototypeToken") and parts[-1] == ("k", "name"):
        # /entries/<X>/prototypeToken/name (length 4)
        if n == 4:
            return parts[1][1], "token"
        # /entries/<X>/items/<Y>/prototypeToken/name (length 6) — token of an item, use item key
        if n == 6 and parts[2] == ("k", "items"):
            return parts[3][1], "token"
    return None


def set_at(obj, p, value):
    parts = parse_path(p)
    cur = obj
    for kind, key in parts[:-1]:
        cur = cur[key]
    cur[parts[-1][1]] = value


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield (path, obj)


def extract_zh_only(bilingual_name):
    """From a bilingual TM name like '断头 Severed Head' return just '断头'.
    If the value is already pure Chinese, return as-is.
    """
    if not bilingual_name:
        return bilingual_name
    # Strip trailing English run (letters + spaces + apostrophes + parens)
    m = re.search(r'\s+[A-Za-z][A-Za-z\s\-\'\(\)]*$', bilingual_name)
    if m:
        return bilingual_name[: m.start()].rstrip()
    return bilingual_name


def apply_to(input_path, output_path=None):
    if output_path is None:
        output_path = input_path
    data = json.load(open(input_path, encoding="utf-8"))
    stats = {"name_hits": 0, "description_hits": 0, "token_hits": 0,
             "name_miss": 0, "description_miss": 0, "token_miss": 0,
             "skip_already_zh": 0}
    misses = []
    for path, value in list(walk(data)):
        if not isinstance(value, str):
            continue
        info = get_item_info(path)
        if not info:
            continue
        item_key, field = info
        if any('一' <= c <= '鿿' for c in value):
            stats["skip_already_zh"] += 1
            continue
        tm_entry, how = lookup_aggressive(item_key)
        if tm_entry is None:
            stats[f"{field}_miss"] += 1
            misses.append((path, item_key, value[:80]))
            continue
        if field == "name" and tm_entry.get("name"):
            set_at(data, path, tm_entry["name"])
            stats["name_hits"] += 1
        elif field == "description" and tm_entry.get("description"):
            set_at(data, path, tm_entry["description"])
            stats["description_hits"] += 1
        elif field == "token" and tm_entry.get("name"):
            zh = extract_zh_only(tm_entry["name"])
            if zh and any('一' <= c <= '鿿' for c in zh):
                set_at(data, path, zh)
                stats["token_hits"] += 1
            else:
                stats["token_miss"] += 1
                misses.append((path, item_key, value[:80]))
        else:
            stats[f"{field}_miss"] += 1
            misses.append((path, item_key, value[:80]))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return stats, misses


def main():
    if len(sys.argv) < 2:
        print("Usage: python apply_tm.py <input.json> [output.json]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path
    stats, misses = apply_to(input_path, output_path)
    print(f"=== {input_path} ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if misses:
        print(f"\nMisses: {len(misses)}; first 10:")
        for p, key, val in misses[:10]:
            print(f"  {key!r}")


if __name__ == "__main__":
    main()
