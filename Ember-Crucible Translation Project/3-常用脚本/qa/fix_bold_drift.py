# -*- coding: utf-8 -*-
"""修「译文比英文多包了 <strong>」这类加粗漂移。

没有单一规则能判断某个 <strong> 是不是多余的 —— 得先知道英文里哪些词该加粗。
办法是自举：

  1. 先扫**加粗数量本来就对得上**的条目，按出现顺序把 EN 粗体词与 CN 粗体词配对，
     统计出一张「英文粗体词 → 中文粗体词」的对应表（同一英文词取最高频的中文写法）；
  2. 对多包的条目：把该条英文里的粗体词经对应表映射，得到「本应加粗的中文词集合」；
     不在这个集合里的 CN 粗体词，就是译者自己加的，拆掉它的 <strong>；
  3. 拆完之后 <strong> 数量必须与英文相等，否则整条回滚。

只处理「译文多包」的方向。少包的方向（英文有粗体而译文没有）需要判断该粗哪个词，
不能自动做，只列出来。

用法：
  python fix_bold_drift.py --repo <repo> [--write] [--min-support N]
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

STRONG = re.compile(r"<strong>(.*?)</strong>", re.S)
TAG = re.compile(r"<\s*(/?)(strong|em|b|i)\b", re.I)


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def count_strong(s):
    return len(re.findall(r"<strong>", s))


def set_at(doc, path, value):
    node = doc
    parts = re.findall(r"[^.\[\]]+", path)
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node[p]
    if isinstance(node, list):
        node[int(parts[-1])] = value
    else:
        node[parts[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--min-support", type=int, default=2,
                    help="一个英文粗体词至少要在多少条对得上的译文里出现过，才采信它的中文对应")
    args = ap.parse_args()
    repo = Path(args.repo)

    packs = {}
    for f in sorted((repo / "compendium" / "en").glob("*.json")):
        cnp = repo / "compendium" / "cn" / f.name
        if cnp.exists():
            packs[f.name] = (dict(leaves(json.loads(f.read_text(encoding="utf-8-sig")))),
                             json.loads(cnp.read_text(encoding="utf-8-sig")))

    # ── 第一步：从数量对得上的条目里学出对应表 ────────────────
    votes = defaultdict(Counter)
    for name, (en, cn_doc) in packs.items():
        cn = dict(leaves(cn_doc))
        for p, s in en.items():
            t = cn.get(p)
            if not t:
                continue
            eb, cb = STRONG.findall(s), STRONG.findall(t)
            if len(eb) == len(cb) and eb:
                for a, b in zip(eb, cb):
                    votes[a.strip()][b.strip()] += 1
    table = {k: c.most_common(1)[0][0] for k, c in votes.items()
             if c.most_common(1)[0][1] >= args.min_support}
    print(f"学到 {len(table)} 条「英文粗体词 → 中文粗体词」对应（支持度 ≥ {args.min_support}）")

    # ── 第二步：拆掉多余的加粗 ──────────────────────────────
    fixed = defaultdict(dict)
    unresolved = []
    stats = Counter()
    for name, (en, cn_doc) in packs.items():
        cn = dict(leaves(cn_doc))
        for p, s in en.items():
            t = cn.get(p)
            if not t:
                continue
            ne, nc = count_strong(s), count_strong(t)
            if nc <= ne:
                if nc < ne:
                    unresolved.append((name, p, ne, nc))
                continue
            expected = {table[w.strip()] for w in STRONG.findall(s) if w.strip() in table}
            # 英文粗体词本身也可能原样出现在译文里（专有名词）
            expected |= {w.strip() for w in STRONG.findall(s)}
            # 只拆掉「超出的那几个」：不在预期集合的排前面，够数就停。
            # 拆多了会把本该加粗的词也拆掉，那比多包一层更糟。
            cands = STRONG.findall(t)
            cands.sort(key=lambda w: w.strip() in expected)
            new = t
            for w in cands[:nc - ne]:
                if w.strip() in expected:
                    break            # 已经没有「明显多余」的了，宁可留着不动
                frag = f"<strong>{w}</strong>"
                if new.count(frag) >= 1:
                    new = new.replace(frag, w, 1)
                    stats[w.strip()] += 1
            if count_strong(new) == ne and new != t:
                fixed[name][p] = new
            elif new != t:
                unresolved.append((name, p, ne, count_strong(new)))

    # ── 第三步：反向补加粗（英文有、译文没有）────────────────
    # 同一张对应表反过来用：英文加粗的词映射成中文，若该中文词在译文里恰好出现一次
    # 且当前没被加粗，就给它包上。数量同样必须正好落到英文的值，否则整条不动。
    added = Counter()
    for name, (en, cn_doc) in packs.items():
        cn = dict(leaves(cn_doc))
        for p, s in en.items():
            t = cn.get(p)
            if t is None or p in fixed.get(name, {}):
                continue
            ne, nc = count_strong(s), count_strong(t)
            if nc >= ne:
                continue
            already = {w.strip() for w in STRONG.findall(t)}
            new = t
            for w in STRONG.findall(s):
                if count_strong(new) >= ne:
                    break
                cw = table.get(w.strip())
                if not cw or cw in already:
                    continue
                # 只在「未被任何标签包裹、且全文恰好出现一次」时才动
                if new.count(cw) != 1 or f"<strong>{cw}</strong>" in new:
                    continue
                new = new.replace(cw, f"<strong>{cw}</strong>", 1)
                already.add(cw)
                added[cw] += 1
            if count_strong(new) == ne and new != t:
                fixed[name][p] = new

    total = sum(len(v) for v in fixed.values())
    if added:
        print(f"反向补加粗 {sum(added.values())} 处，top：")
        for w, n in added.most_common(8):
            print(f"    {n:>3}  {w}")
    print(f"可自动修 {total} 条；拆掉的多余加粗 top：")
    for w, n in stats.most_common(12):
        print(f"    {n:>3}  {w}")
    print(f"仍需人工：{len(unresolved)} 条（含英文有粗体而译文没有的方向）")

    if args.write:
        for name, items in fixed.items():
            cnp = repo / "compendium" / "cn" / name
            doc = json.loads(cnp.read_text(encoding="utf-8-sig"))
            for p, v in items.items():
                set_at(doc, p, v)
            cnp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"    写入 {name}（{len(items)} 条）")


if __name__ == "__main__":
    main()
