"""R16 裁决1：Token -> 指示物。先扫描，逐位对齐前的判据收集。

英文闸 \bTokens?\b（IGNORECASE，参考 round14 教训）。
中文侧 Token 的现有三种译法：指示物 / 令牌 / 代币。
逐位对齐要把三种一起数，才能把「第 i 个中文」对上「第 i 个英文」。
"""
import json
import os
import re
import sys

EN_TOKEN = re.compile(r"\bTokens?\b", re.IGNORECASE)
CN_VARIANTS = ("指示物", "令牌", "代币")
CN_RE = re.compile("|".join(CN_VARIANTS))

REPOS = {
    "ember": r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件",
    "crucible": r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件",
}


def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def collect():
    rows = []
    for repo_key, repo in REPOS.items():
        en_dir = os.path.join(repo, "compendium", "en")
        cn_dir = os.path.join(repo, "compendium", "cn")
        for fname in sorted(os.listdir(en_dir)):
            if not fname.endswith(".json") or fname == "_source.json":
                continue
            cn_path = os.path.join(cn_dir, fname)
            if not os.path.exists(cn_path):
                continue
            en_map = dict(walk(load(os.path.join(en_dir, fname))))
            cn_map = dict(walk(load(cn_path)))
            for path, en_val in en_map.items():
                cn_val = cn_map.get(path)
                if not isinstance(cn_val, str) or not cn_val:
                    continue
                en_hits = EN_TOKEN.findall(en_val)
                cn_hits = CN_RE.findall(cn_val)
                if not en_hits and not cn_hits:
                    continue
                rows.append({
                    "repo": repo_key,
                    "pack": fname,
                    "path": path,
                    "en": en_val,
                    "cn": cn_val,
                    "en_n": len(en_hits),
                    "cn_hits": cn_hits,
                })
    return rows


if __name__ == "__main__":
    rows = collect()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token_rows.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    need = [r for r in rows if any(h in ("令牌", "代币") for h in r["cn_hits"])]
    print(f"总相关叶 {len(rows)}；中文含 令牌/代币 的叶 {len(need)}")
    from collections import Counter
    print("按仓/包：", Counter((r["repo"], r["pack"]) for r in need))
    eq = [r for r in need if r["en_n"] == len(r["cn_hits"])]
    ne = [r for r in need if r["en_n"] != len(r["cn_hits"])]
    noen = [r for r in need if r["en_n"] == 0]
    print(f"计数相等 {len(eq)} / 不等 {len(ne)}（其中英文零命中 {len(noen)}）")
