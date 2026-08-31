"""Step 1: Port OLD translations into NEW files.

For each path in OLD that also exists in NEW, copy OLD's value to NEW.
This is a pure TM merge: NEW values get overwritten by OLD's already-translated values
when paths match. New paths in NEW (Book 2 additions, etc.) remain English for now.

Outputs go to FotRP/需要翻译/NEW/_merged/<file>.json so OLD and NEW stay untouched.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
OLD_DIR = ROOT / "FotRP" / "需要翻译"
NEW_DIR = ROOT / "FotRP" / "需要翻译" / "NEW"
MERGED_DIR = NEW_DIR / "_merged"
MERGED_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "fist-of-the-ruby-phoenix-addons.fist-of-the-ruby-phoenix-addons-maps.json",
    "fist-of-the-ruby-phoenix-addons.fist-of-the-ruby-phoenix-addons.json",
    "fists-of-the-ruby-phoenix-map-remake-by-kalnix.fists-of-the-ruby-phoenix-map-remakes-by-kalnix.json",
    "kobold-dm-fists-of-the-ruby-phoenix-maps-remakes.fotrp-kdm-remakes.json",
    "pf2e.fists-of-the-ruby-phoenix-bestiary.json",
]


def collect_paths(obj, path=""):
    """Collect a flat dict {path -> string_leaf_value} for every string in the tree."""
    out = {}
    def walk(o, p):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{p}/{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{p}[{i}]")
        elif isinstance(o, str):
            out[p] = o
    walk(obj, path)
    return out


def set_path(obj, path, value):
    """Set a value at a parsed path like '/a/b[0]/c'."""
    parts = []
    buf = ""
    i = 0
    while i < len(path):
        c = path[i]
        if c == '/':
            if buf:
                parts.append(("k", buf))
                buf = ""
        elif c == '[':
            if buf:
                parts.append(("k", buf))
                buf = ""
            j = path.index(']', i)
            parts.append(("i", int(path[i+1:j])))
            i = j
        else:
            buf += c
        i += 1
    if buf:
        parts.append(("k", buf))
    cur = obj
    for kind, key in parts[:-1]:
        if kind == "k":
            cur = cur[key]
        else:
            cur = cur[key]
    last_kind, last_key = parts[-1]
    cur[last_key] = value


def main():
    summary = []
    for fname in FILES:
        old = json.load(open(OLD_DIR / fname, encoding="utf-8"))
        new = json.load(open(NEW_DIR / fname, encoding="utf-8"))
        old_paths = collect_paths(old)
        new_paths = collect_paths(new)
        common = set(old_paths) & set(new_paths)
        ported = 0
        skipped_same = 0
        for p in common:
            if old_paths[p] != new_paths[p]:
                set_path(new, p, old_paths[p])
                ported += 1
            else:
                skipped_same += 1
        # Persist merged file
        out_path = MERGED_DIR / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new, f, ensure_ascii=False, indent=2)
        summary.append({
            "file": fname,
            "old_paths": len(old_paths),
            "new_paths": len(new_paths),
            "common": len(common),
            "ported": ported,
            "already_same": skipped_same,
            "new_only": len(new_paths) - len(common),
            "out": str(out_path),
        })

    # Write summary report (utf-8 safe)
    with open(MERGED_DIR / "_step1_port_summary.txt", "w", encoding="utf-8") as f:
        for s in summary:
            f.write(f"{s['file']}\n")
            f.write(f"  ported={s['ported']}, same={s['already_same']}, new_only_remaining={s['new_only']}\n")
            f.write(f"  out: {s['out']}\n\n")
    print(f"Done. Summary in {MERGED_DIR / '_step1_port_summary.txt'}")


if __name__ == "__main__":
    main()
