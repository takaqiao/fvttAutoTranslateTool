# -*- coding: utf-8 -*-
"""判定性检查：把每片叶解析成「元素骨架」逐位比对中英。

骨架 = 文档序下每个元素的  深度/标签名/除文本类属性外的全部属性(名与值)
文本类属性（会被翻译，不参与比对）：title alt aria-label data-tooltip
                                data-tooltip-text data-tooltip-html placeholder
id 单独统计（中文侧有意注入 1642 个锚点）。

骨架完全一致 => 结构层（含 class/style/data-* 的取值与位置、secret 块的覆盖范围、
表格单元格的归属）在中英之间没有任何漂移。这比「标签名多重集相等」强得多。

  K1  骨架不一致（列出首个差异位）
  K2  仅 id 不同（预期：中文多出注入锚点；若中文**少**了英文本来就有的 id 才是问题）
"""
import json, sys, collections
from pathlib import Path
from lxml import html as lhtml

TEXT_ATTRS = {"title", "alt", "aria-label", "data-tooltip", "data-tooltip-text",
              "data-tooltip-html", "placeholder", "label", "caption", "readaloud"}


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def skeleton(s):
    try:
        root = lhtml.fragment_fromstring(s, create_parent="root")
    except Exception:
        return None, None
    sk, ids = [], []

    def walk(el, d):
        for ch in el:
            if not isinstance(ch.tag, str):
                continue
            attrs = {k.lower(): v for k, v in ch.attrib.items()}
            i = attrs.pop("id", None)
            if i is not None:
                ids.append((d, ch.tag, i))
            attrs = {k: v for k, v in attrs.items() if k not in TEXT_ATTRS}
            sk.append((d, ch.tag, tuple(sorted(attrs.items()))))
            walk(ch, d + 1)
    walk(root, 0)
    return sk, ids


counts = collections.Counter()
rows = []
scanned = 0
for repo in sys.argv[1:]:
    repo = Path(repo)
    for f in sorted((repo / "compendium" / "en").glob("*.json")):
        cnp = repo / "compendium" / "cn" / f.name
        if not cnp.exists():
            continue
        en = dict(leaves(json.loads(f.read_text(encoding="utf-8-sig"))))
        cn = dict(leaves(json.loads(cnp.read_text(encoding="utf-8-sig"))))
        for p, s in cn.items():
            e = en.get(p, "")
            if "<" not in s and "<" not in e:
                continue
            scanned += 1
            ks, kids = skeleton(s)
            ke, eids = skeleton(e)
            if ks is None or ke is None:
                counts["parsefail"] += 1
                continue
            if ks != ke:
                counts["K1"] += 1
                d = next(((i, a, b) for i, (a, b) in enumerate(zip(ks, ke)) if a != b), None)
                rows.append(("K1", repo.name, f.name, p,
                             f"骨架长度 CN={len(ks)} EN={len(ke)}；首差 {d}", e[:250], s[:250]))
            # id：英文有而中文没有的
            lost = [x for x in eids if x not in kids]
            if lost:
                counts["K2"] += 1
                rows.append(("K2", repo.name, f.name, p, f"中文侧丢失英文原有 id: {lost[:5]}", "", ""))
            counts["cn_injected_ids"] += max(0, len(kids) - len(eids))

print("扫描含标签的叶:", scanned)
print(counts)
seen = collections.Counter()
for code, rn, pack, p, det, e, s in rows:
    seen[code] += 1
    if seen[code] > 25:
        continue
    print("-" * 96)
    print(f"[{code}] {rn} {pack} | {p}")
    print("    det:", str(det)[:400])
    if e:
        print("    EN :", e[:250])
        print("    CN :", s[:250])
