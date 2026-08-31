# -*- coding: utf-8 -*-
"""把 35 条骨架差异分成两类：**纯置换**（汉语语序，合法）与 **真增删/真改值**（缺陷）。

判据：忽略文档序，比较骨架元素的多重集。
  M0  多重集相等 -> 纯置换，只是位置变了（等同 EXCLUSIONS 里已豁免的 <strong> 语序翻转）
  M1  多重集不等 -> 中文侧真的多/少/改了某个元素或其属性值  <-- 这才是缺陷
另外单列：
  M2  data-uuid / data-id 的多重集不等（链接目标被增删）
  M3  fa-* 图标类的多重集不等（图标被换）
  M4  class 值的多重集不等
"""
import json, sys, re, collections
from pathlib import Path
from lxml import html as lhtml

TEXT_ATTRS = {"title", "alt", "aria-label", "data-tooltip", "data-tooltip-text",
              "data-tooltip-html", "placeholder", "label", "caption", "readaloud"}
FA = re.compile(r"\bfa-[a-z0-9-]+")


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def parts(s):
    try:
        root = lhtml.fragment_fromstring(s, create_parent="root")
    except Exception:
        return None
    sk, uuids, icons, classes = [], [], [], []
    for el in root.iter():
        if not isinstance(el.tag, str) or el.getparent() is None:
            continue
        a = {k.lower(): v for k, v in el.attrib.items()}
        a.pop("id", None)
        for k in ("data-uuid", "data-id"):
            if k in a:
                uuids.append(a[k])
        cls = a.get("class", "")
        if cls:
            classes.append(cls)
            icons += FA.findall(cls)
        a = {k: v for k, v in a.items() if k not in TEXT_ATTRS}
        sk.append((el.tag, tuple(sorted(a.items()))))
    return sk, uuids, icons, classes


counts = collections.Counter()
rows = []
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
            pc, pe = parts(s), parts(e)
            if pc is None or pe is None:
                continue
            if pc[0] == pe[0]:
                continue          # 连顺序都一样
            same = collections.Counter(pc[0]) == collections.Counter(pe[0])
            counts["M0_纯置换" if same else "M1_真差异"] += 1
            det = []
            for idx, name in ((1, "M2_uuid"), (2, "M3_icon"), (3, "M4_class")):
                a, b = collections.Counter(pc[idx]), collections.Counter(pe[idx])
                if a != b:
                    counts[name] += 1
                    diff = {k: (a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b) if a.get(k, 0) != b.get(k, 0)}
                    det.append(f"{name} CN/EN={diff}")
            if not same or det:
                rows.append(("M1" if not same else "M0", repo.name, f.name, p,
                             ("纯置换" if same else "多重集不等") + "; " + "; ".join(det),
                             e[:200], s[:200]))

print(counts)
for code, rn, pack, p, det, e, s in rows:
    print("-" * 96)
    print(f"[{code}] {rn} {pack} | {p}")
    print("    det:", str(det)[:500])
    print("    EN :", e[:220])
    print("    CN :", s[:220])
