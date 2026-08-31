# -*- coding: utf-8 -*-
"""S4 分片的只读查库工具：按点号路径取 EN/CN 叶子，或全库正则检索。

用法（只读，不写库）：
    python s4_lib.py get <repo1|2> <pack.json> <dotted path>
    python s4_lib.py grep <needle> [--cn|--en]
"""
import json, os, re, sys, glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPO = {"1": os.path.join(ROOT, "1-Ember汉化插件"), "2": os.path.join(ROOT, "2-Crucible汉化插件")}
_cache = {}


def load(repo, pack, side="cn"):
    key = (repo, pack, side)
    if key not in _cache:
        p = os.path.join(REPO[repo], "compendium", side, pack)
        with open(p, encoding="utf-8-sig") as f:
            _cache[key] = json.load(f)
    return _cache[key]


def get_at(node, parts):
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p)
        elif isinstance(node, list):
            try:
                node = node[int(p)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def split_path(root, path):
    naive = path.split('.')
    if get_at(root, naive) is not None:
        return naive
    parts, node, rest = [], root, path
    while rest:
        if isinstance(node, dict):
            cands = [k for k in node if rest == k or rest.startswith(k + '.')]
            if cands:
                k = max(cands, key=len)
                parts.append(k)
                node = node[k]
                rest = rest[len(k) + 1:]
                continue
        head, _, rest = rest.partition('.')
        parts.append(head)
        if isinstance(node, list):
            try:
                node = node[int(head)]
            except (ValueError, IndexError):
                node = None
        elif isinstance(node, dict):
            node = node.get(head)
        else:
            node = None
    return parts


def leaf(repo, pack, path, side="cn"):
    doc = load(repo, pack, side)
    parts = path.split('.')
    root = doc.get('folders', {}) if parts[0] == '(folders)' else doc.get('entries', {})
    if parts[0] == '(folders)':
        parts = parts[1:]
    return get_at(root, split_path(root, '.'.join(parts)))


def walk(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{prefix}.{i}")
    elif isinstance(node, str):
        yield prefix, node


def all_leaves(repo, pack, side="cn"):
    doc = load(repo, pack, side)
    for p, v in walk(doc.get('entries', {})):
        yield p, v
    for p, v in walk(doc.get('folders', {})):
        yield '(folders).' + p, v


def packs(repo):
    return [os.path.basename(p) for p in
            sorted(glob.glob(os.path.join(REPO[repo], "compendium", "cn", "*.json")))]


def search(needle, side="cn", regex=False):
    rx = re.compile(needle) if regex else None
    out = []
    for repo in ("1", "2"):
        for pack in packs(repo):
            en_path = os.path.join(REPO[repo], "compendium", "en", pack)
            if side == "en" and not os.path.exists(en_path):
                continue
            for p, v in all_leaves(repo, pack, side):
                if (rx.search(v) if rx else (needle in v)):
                    out.append((repo, pack, p, v))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1]
    if cmd == "get":
        repo, pack, path = sys.argv[2], sys.argv[3], sys.argv[4]
        print("EN:", repr(leaf(repo, pack, path, "en")))
        print("CN:", repr(leaf(repo, pack, path, "cn")))
    elif cmd == "grep":
        side = "en" if "--en" in sys.argv else "cn"
        for repo, pack, p, v in search(sys.argv[2], side, regex="--re" in sys.argv):
            print(f"{repo}|{pack}|{p}\n    {v[:200]}")
