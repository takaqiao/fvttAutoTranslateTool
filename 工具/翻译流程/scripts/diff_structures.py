"""Compare NEW vs old FotRP merged files structurally — find content that needs (re)translation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OLD_DIR = ROOT / "FotRP" / "需要翻译"
NEW_DIR = ROOT / "FotRP" / "需要翻译" / "NEW"

FILES = [
    "fist-of-the-ruby-phoenix-addons.fist-of-the-ruby-phoenix-addons-maps.json",
    "fist-of-the-ruby-phoenix-addons.fist-of-the-ruby-phoenix-addons.json",
    "fists-of-the-ruby-phoenix-map-remake-by-kalnix.fists-of-the-ruby-phoenix-map-remakes-by-kalnix.json",
    "kobold-dm-fists-of-the-ruby-phoenix-maps-remakes.fotrp-kdm-remakes.json",
    "pf2e.fists-of-the-ruby-phoenix-bestiary.json",
]


def walk(obj, path=""):
    """Yield (path, leaf_value) for every string leaf."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield (path, obj)


def collect_strings(obj):
    return dict(walk(obj))


def diff_file(fname):
    old = json.load(open(OLD_DIR / fname, encoding="utf-8"))
    new = json.load(open(NEW_DIR / fname, encoding="utf-8"))
    o_strs = collect_strings(old)
    n_strs = collect_strings(new)
    o_paths = set(o_strs)
    n_paths = set(n_strs)
    added_paths = n_paths - o_paths
    removed_paths = o_paths - n_paths
    common_paths = o_paths & n_paths
    changed = [p for p in common_paths if o_strs[p] != n_strs[p]]
    return {
        "added_paths": added_paths,
        "removed_paths": removed_paths,
        "changed_paths": changed,
        "old": o_strs,
        "new": n_strs,
    }


if __name__ == "__main__":
    for fname in FILES:
        print("=" * 78)
        print(fname)
        d = diff_file(fname)
        print(f"  Total NEW strings: {len(d['new'])}")
        print(f"  Total OLD strings: {len(d['old'])}")
        print(f"  Added paths (NEW only): {len(d['added_paths'])}")
        print(f"  Removed paths (OLD only): {len(d['removed_paths'])}")
        print(f"  Changed values (same path, different value): {len(d['changed_paths'])}")
