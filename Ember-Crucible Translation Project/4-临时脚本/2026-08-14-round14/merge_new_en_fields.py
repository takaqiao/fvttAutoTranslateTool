"""把重抽出来的英文基准里**新增的路径**并进 compendium/en，绝不碰已有路径。

为什么不能整体覆盖：`compendium/en` 上打着 `LOCAL-PATCHES.md` 记的四条上游笔误补丁
（惯例是「补丁只打 compendium/en，english-baseline/ 快照保持上游原样」）。
整体覆盖会把它们静默回退，而那几条补丁是**任何译法都必被判 markup mismatch** 的那种阻断。

所以这里只做单向并入：scratch 里有、当前 en 里没有的路径才写进去。
已有路径**一律不动**，因此补丁在结构上不可能被回退。
先跑不带 --write 看清单。
"""
import argparse
import json
import os


def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, node


def get_parent(root, path):
    """按路径取到父容器与最后一段 key/index，沿途缺什么建什么。"""
    import re
    parts = re.findall(r"\[(\d+)\]|([^.\[\]]+)", path)
    keys = [int(i) if i else name for i, name in parts]
    cur = root
    for k, nxt in zip(keys[:-1], keys[1:]):
        if isinstance(k, int):
            while len(cur) <= k:
                cur.append([] if isinstance(nxt, int) else {})
            cur = cur[k]
        else:
            if k not in cur or not isinstance(cur[k], (dict, list)):
                cur[k] = [] if isinstance(nxt, int) else {}
            cur = cur[k]
    return cur, keys[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True, help="仓库的 compendium/en")
    ap.add_argument("--fresh", required=True, help="刚重抽出来的目录")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    total_new = 0
    total_changed = 0
    by_field = {}

    for fname in sorted(os.listdir(args.fresh)):
        if not fname.endswith(".json"):
            continue
        cur_path = os.path.join(args.current, fname)
        if not os.path.exists(cur_path):
            print(f"  {fname}: 当前 en 里没有这个包，跳过（本脚本只补字段，不新建包）")
            continue

        cur_doc = json.load(open(cur_path, encoding="utf-8-sig"))
        fresh_doc = json.load(open(os.path.join(args.fresh, fname), encoding="utf-8-sig"))

        cur_leaves = dict(walk(cur_doc))
        new_paths = []
        changed = []
        for p, v in walk(fresh_doc):
            if p not in cur_leaves:
                new_paths.append((p, v))
            elif cur_leaves[p] != v:
                changed.append(p)

        total_new += len(new_paths)
        total_changed += len(changed)
        for p, _ in new_paths:
            # 取路径里最后一个非索引段当字段名，用于分类统计
            seg = [s for s in p.replace("[", ".").replace("]", "").split(".") if not s.isdigit()]
            field = ".".join(seg[-2:]) if len(seg) >= 2 else p
            key = field.split(".")[-1]
            by_field[key] = by_field.get(key, 0) + 1

        if new_paths or changed:
            print(f"  {fname}: 新增 {len(new_paths)} 条 / 已有但值不同 {len(changed)} 条（**不动**）")

        if args.write and new_paths:
            for p, v in new_paths:
                parent, last = get_parent(cur_doc, p)
                if isinstance(last, int):
                    while len(parent) <= last:
                        parent.append(None)
                    parent[last] = v
                else:
                    parent[last] = v
            with open(cur_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(cur_doc, f, ensure_ascii=False, indent=1)
                f.write("\n")

    print(f"\n合计新增 {total_new} 条；已有但值不同 {total_changed} 条 —— 后者一律未改动"
          f"（其中就包含 LOCAL-PATCHES 的四条上游笔误补丁，本脚本结构上碰不到它们）。")
    print("新增路径按字段名分布：")
    for k, n in sorted(by_field.items(), key=lambda kv: -kv[1]):
        print(f"    {k:28s} {n}")
    if not args.write:
        print("\n(未加 --write，未改动任何文件)")


if __name__ == "__main__":
    main()
