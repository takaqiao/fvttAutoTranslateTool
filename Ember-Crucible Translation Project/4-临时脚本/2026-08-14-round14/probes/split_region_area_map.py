"""R4：把「区域地图」按英文原文拆成 Region Map=地区地图 / Area Map=区域地图。

为什么不能用 unify_terms.py 一把梭：全库 159 叶英文含 `Region Map`、268 叶含 `Area Map`，
其中 **46 叶两者同时出现**，而中文两边现在都是「区域地图」。按叶级英文闸做整叶替换
会把同一叶里的 Area Map 也一起改掉。

做法沿用本项目在 `@UUID` 标签上验证过的「**逐位对齐**」：
把英文里 `Region Map` / `Area Map` 的出现顺序，与中文里「区域地图」的出现顺序配对，
第 i 个中文对应第 i 个英文。只有**两侧计数相等**时才动手；不等的整叶跳过并列出来人看。

输出批次文件，交 apply_translations.py 过闸后落盘。
"""
import argparse
import json
import os
import re

# ⚠ **必须大小写不敏感。** 第一版只匹配首字母大写的 `Region Map` / `Area Map`，
# 而 Ember 英文正文里小写形态（`the region map`、`each new area map`）非常多 ——
# 于是英文侧计数偏小、大量叶子被误判成「计数不等，需人判」。
# 实测：72 条「不等」里绝大多数是这个假阳性，不是真的中文重复了术语。
# 这个错误直接导致主控一度判定「全库拆分做不了」——判据错了，结论就跟着错。
EN_TOKEN = re.compile(r"\b(Region|Area)[ -]Maps?\b", re.IGNORECASE)
CN_TOKEN = "区域地图"
CN_FOR = {"region": "地区地图", "area": "区域地图"}


def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    en_dir = os.path.join(args.repo, "compendium", "en")
    cn_dir = os.path.join(args.repo, "compendium", "cn")
    os.makedirs(args.out_dir, exist_ok=True)

    stats = {"changed": 0, "skipped_mismatch": 0, "untouched": 0}
    mismatches = []

    for fname in sorted(os.listdir(en_dir)):
        if not fname.endswith(".json"):
            continue
        cn_path = os.path.join(cn_dir, fname)
        if not os.path.exists(cn_path):
            continue
        en_doc = json.load(open(os.path.join(en_dir, fname), encoding="utf-8"))
        cn_map = dict(walk(json.load(open(cn_path, encoding="utf-8"))))

        batch = {}
        for path, en_val in walk(en_doc):
            kinds = [m.group(1).lower() for m in EN_TOKEN.finditer(en_val)]
            if not kinds:
                continue
            cn_val = cn_map.get(path)
            if not cn_val:
                continue
            n_cn = cn_val.count(CN_TOKEN)
            if n_cn == 0:
                stats["untouched"] += 1
                continue
            if n_cn != len(kinds):
                stats["skipped_mismatch"] += 1
                mismatches.append((fname, path, len(kinds), n_cn, kinds))
                continue
            # 逐位对齐替换
            out, cursor, idx = [], 0, 0
            while True:
                hit = cn_val.find(CN_TOKEN, cursor)
                if hit < 0:
                    out.append(cn_val[cursor:])
                    break
                out.append(cn_val[cursor:hit])
                out.append(CN_FOR[kinds[idx]])
                cursor = hit + len(CN_TOKEN)
                idx += 1
            new = "".join(out)
            if new != cn_val:
                batch[path] = new
                stats["changed"] += 1
            else:
                stats["untouched"] += 1

        if batch:
            out_path = os.path.join(args.out_dir, f"R4.{fname}")
            json.dump(batch, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"{fname:36s} {len(batch):4d} 叶 -> {out_path}")

    print(f"\n改写 {stats['changed']} 叶 / 计数不等跳过 {stats['skipped_mismatch']} 叶 / 无需改动 {stats['untouched']} 叶")
    if mismatches:
        print("\n=== 计数不等，需人看（英文出现数 vs 中文出现数）===")
        for fname, path, ne, nc, kinds in mismatches[:40]:
            print(f"  {fname} :: {path[:100]}  EN={ne}{kinds} CN={nc}")
        if len(mismatches) > 40:
            print(f"  …另 {len(mismatches) - 40} 条")


if __name__ == "__main__":
    main()
