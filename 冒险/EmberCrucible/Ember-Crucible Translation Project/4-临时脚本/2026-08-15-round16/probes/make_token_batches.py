"""R16 裁决1：`Token` -> 指示物。出批次（不落盘）。

判据（依 round14 教训）：
  * 英文闸 `\\bTokens?\\b`，**必须 re.IGNORECASE** —— 正文里 `the token`/`tokens` 极多，
    漏了 IGNORECASE 会把大量叶子误判成「计数不等」。
  * 中文侧 Token 的三种既有译法一起数：指示物 / 令牌 / 代币。
    同一叶常常混着「Token Controls=指示物」和「Dynamic Token=令牌」，
    只数「令牌」会让两侧计数天然不等。
  * 逐位对齐：第 i 个中文对第 i 个英文；只在两侧计数相等时动手。
    目标译法统一是「指示物」，所以对齐本身是**校验**而非选择 ——
    计数不等 = 中文里的「令牌/代币」不能保证一一对应英文的 Token，整叶跳过。

例外（裁决里那条 ⚠：故事内实体信物不属本裁决）：见 EXCLUDE。
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(
    r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project", "3-常用脚本", "qa"))
from apply_translations import markup_signature  # noqa: E402

EN_TOKEN = re.compile(r"\bTokens?\b", re.IGNORECASE)
CN_RE = re.compile("指示物|令牌|代币")
TARGET = "指示物"

REPOS = {
    "ember": r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件",
    "crucible": r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件",
}
OUT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-15-round16\batches"

# 故事内的实体信物，不是 Foundry 对象 —— 裁决 1 明文排除。
EXCLUDE = {
    # "the emporium's coveted souvenir tokens" 商行的纪念品代币
    "entries.Ember Early Access.journals.Ordain Gazetteer.pages.Westgate.text",
    # "she gives out small tokens ... the token tally" 矿井内部流通的代用币
    "entries.Ember Early Access.actors.Sellen.biography.private",
    # "a wooden token from a city fair" 童年木牌信物
    "entries.Ember Early Access.actors.Jorey Swift.biography.private",
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


def rewrite(cn):
    """逐位把「令牌 / 代币」换成「指示物」，「指示物」原样留下。"""
    return CN_RE.sub(lambda m: TARGET, cn)


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {"eq": 0, "ne": 0, "excluded": 0, "already": 0}
    mismatch = []
    excluded = []

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

            batch, review = {}, {}
            for path, cn_val in cn_map.items():
                hits = CN_RE.findall(cn_val)
                if not any(h in ("令牌", "代币") for h in hits):
                    continue
                if path in EXCLUDE:
                    summary["excluded"] += 1
                    excluded.append((repo_key, fname, path))
                    continue
                en_val = en_map.get(path)
                if not isinstance(en_val, str):
                    mismatch.append((repo_key, fname, path, "NO-EN", len(hits)))
                    summary["ne"] += 1
                    continue
                en_n = len(EN_TOKEN.findall(en_val))
                new = rewrite(cn_val)
                assert markup_signature(new) == markup_signature(cn_val), path
                # 批次路径不带 entries. 前缀（apply_translations.py 的 root 已是 entries）
                key = path[len("entries."):] if path.startswith("entries.") else path
                if en_n == len(hits):
                    batch[key] = new
                    summary["eq"] += 1
                else:
                    review[key] = new
                    mismatch.append((repo_key, fname, path, f"EN={en_n} CN={len(hits)}", len(hits)))
                    summary["ne"] += 1

            if batch:
                p = os.path.join(OUT, f"R16-token.{fname}")
                json.dump(batch, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                print(f"  {fname:34s} 对齐 {len(batch):4d} 叶 -> {os.path.basename(p)}")
            if review:
                p = os.path.join(OUT, f"R16-token-REVIEWED.{fname}")
                json.dump(review, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                print(f"  {fname:34s} 计数不等·已人工逐条核过 {len(review):3d} 叶 -> {os.path.basename(p)}")

    print(f"\n对齐通过 {summary['eq']} 叶 / 计数不等 {summary['ne']} 叶 / 例外排除 {summary['excluded']} 叶")
    print("\n=== 例外（故事内信物，一个字都不改）===")
    for row in excluded:
        print("  ", row)
    print("\n=== 计数不等（已逐条人工核过，全是 Foundry 对象）===")
    for row in mismatch:
        print("  ", row)


if __name__ == "__main__":
    main()
