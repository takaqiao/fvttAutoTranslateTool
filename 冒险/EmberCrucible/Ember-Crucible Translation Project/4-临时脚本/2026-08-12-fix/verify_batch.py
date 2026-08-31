# -*- coding: utf-8 -*-
"""独立复核批次：证明批次只做了「同叶内 @UUID 方括号内容换位」这一件事。

四条断言，任何一条不过就报错：
  1. 去掉 @UUID[...] 后的可见文字（含 {标签}）与库里现值逐字相同
  2. @UUID[...] 目标串的多重集与现值逐字相同
  3. 新值里每一个方括号内容都在旧值里原样出现过（没有新造/改写目标）
  4. 至少有一个位置的目标确实变了（不是空改动）
"""
import json, os, re, sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
U = re.compile(r"@UUID\[([^\]]*)\](\{([^}]*)\})?")
REPO = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件"


def get(node, parts):
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p)
        elif isinstance(node, list):
            node = node[int(p)] if p.isdigit() and int(p) < len(node) else None
        else:
            return None
    return node


def split_path(root, path):
    if get(root, path.split(".")) is not None:
        return path.split(".")
    parts, node, rest = [], root, path
    while rest:
        if isinstance(node, dict):
            c = [k for k in node if rest == k or rest.startswith(k + ".")]
            if c:
                k = max(c, key=len)
                parts.append(k); node = node[k]; rest = rest[len(k) + 1:]
                continue
        head, _, rest = rest.partition(".")
        parts.append(head)
        node = (node[int(head)] if isinstance(node, list) and head.isdigit()
                else node.get(head) if isinstance(node, dict) else None)
    return parts


ok = moved = 0
for pack in ("ember.crucible-adventure.json", "ember.adventure.json"):
    cn = json.load(open(os.path.join(REPO, "compendium", "cn", pack), encoding="utf-8"))
    batch = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "batches", "uuid-" + pack), encoding="utf-8"))
    for path, new in batch.items():
        root = cn["folders"] if path.startswith("(folders)") else cn["entries"]
        old = get(root, split_path(root, path))
        assert isinstance(old, str), f"{pack} {path}: 库里取不到现值"
        vis = lambda s: U.sub(lambda m: (m.group(3) or ""), s)
        assert vis(old) == vis(new), f"{pack} {path}: 可见文字变了"
        a, b = [m.group(1) for m in U.finditer(old)], [m.group(1) for m in U.finditer(new)]
        assert Counter(a) == Counter(b), f"{pack} {path}: 目标多重集变了"
        assert set(b) <= set(a), f"{pack} {path}: 出现了旧值里没有的目标"
        d = sum(1 for x, y in zip(a, b) if x != y)
        assert d, f"{pack} {path}: 空改动"
        ok += 1; moved += d
print(f"leaves={ok}  links_moved={moved}  ALL ASSERTIONS PASSED")
