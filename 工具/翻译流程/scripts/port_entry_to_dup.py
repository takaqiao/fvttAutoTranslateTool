"""Port Book 2 actor translations to the (hmLe) duplicate entry."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
MERGED = ROOT / "FotRP" / "需要翻译" / "NEW" / "_merged"
F = MERGED / "fist-of-the-ruby-phoenix-addons.fist-of-the-ruby-phoenix-addons.json"

data = json.load(open(F, encoding="utf-8"))

src = data['entries']['Fists of the Ruby Phoenix: Addons (Book 2)']['actors']
dst = data['entries']['Fists of the Ruby Phoenix: Addons (Book 2) (hmLe)']['actors']


def walk_paths(obj, path=""):
    """Yield (path, leaf_value) for every string leaf."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_paths(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_paths(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield (path, obj)


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


def get_at(obj, p):
    parts = parse_path(p)
    cur = obj
    for kind, key in parts:
        if kind == "k":
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return None
        else:
            if isinstance(cur, list) and 0 <= key < len(cur):
                cur = cur[key]
            else:
                return None
    return cur


def set_at(obj, p, value):
    parts = parse_path(p)
    cur = obj
    for kind, key in parts[:-1]:
        cur = cur[key]
    cur[parts[-1][1]] = value


# For each actor present in both src and dst, port string-leaf values from src to dst
ported = 0
skipped = 0
shared_actor_keys = set(src.keys()) & set(dst.keys())
print(f"Shared actor keys: {len(shared_actor_keys)} (src={len(src)}, dst={len(dst)})")
for actor_key in shared_actor_keys:
    src_actor = src[actor_key]
    dst_actor = dst[actor_key]
    for path, src_val in walk_paths(src_actor):
        dst_val = get_at(dst_actor, path)
        if dst_val is None:
            continue
        if not isinstance(dst_val, str):
            continue
        # Only port if src has Chinese AND differs from dst
        if any('一' <= c <= '鿿' for c in src_val) and src_val != dst_val:
            set_at(dst_actor, path, src_val)
            ported += 1
        else:
            skipped += 1

with open(F, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Ported: {ported}, skipped: {skipped}")
print(f"Saved to {F.name}")
