# -*- coding: utf-8 -*-
"""span.reference 的**图标类**与其相邻文本的绑定是否在中英之间错位。

Ember 正文里大量出现 `<span class="reference fa-solid fa-xxx">​</span>​ 名词`
的图标+术语组合。图标类是语义的（fa-cog 机关 / fa-hand 徒手 / fa-mountain 山地…）。
骨架比对发现有叶子里两个不同图标换了位置；换位可能是汉语语序自然结果，
也可能是译者把图标接到了错的词上。这里把「图标类 -> 其后紧跟的可见文字」
配成对，中英逐对照。

  I1  同一图标类在中英对应到的**序数位置**不同，且两侧配到的文字不是互译关系
      （靠「英文词 -> 该叶中文里是否出现其既定译名」判断做不到，
        所以这里只输出配对表，由人判读）
"""
import json, re, sys, collections
from pathlib import Path
from lxml import html as lhtml

FA = re.compile(r"\bfa-(?!solid\b|regular\b|light\b|thin\b|duotone\b|brands\b)[a-z0-9-]+")


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def pairs(s):
    """[(图标类, 紧随其后的 60 字可见文本)]，按文档序。"""
    try:
        root = lhtml.fragment_fromstring(s, create_parent="root")
    except Exception:
        return None
    out = []
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        cls = el.get("class") or ""
        if "reference" not in cls:
            continue
        icons = FA.findall(cls)
        if not icons:
            continue
        tail = (el.tail or "")
        nxt = el.getnext()
        while len(tail.strip().strip("​")) < 3 and nxt is not None:
            tail += " " + ("".join(nxt.itertext())) + (nxt.tail or "")
            nxt = nxt.getnext()
        out.append((icons[0], re.sub(r"\s+", " ", tail).strip("​ \t")[:40]))
    return out


repo, pack, path = sys.argv[1], sys.argv[2], sys.argv[3]
d = {}
for side in ("en", "cn"):
    d[side] = dict(leaves(json.loads((Path(repo) / "compendium" / side / pack)
                                     .read_text(encoding="utf-8-sig"))))[path]
pe, pc = pairs(d["en"]), pairs(d["cn"])
print(f"EN {len(pe)} 个图标 / CN {len(pc)} 个图标")
print(f"{'#':>3}  {'EN 图标':<22} {'EN 后文':<44} | {'CN 图标':<22} CN 后文")
for i in range(max(len(pe), len(pc))):
    a = pe[i] if i < len(pe) else ("", "")
    b = pc[i] if i < len(pc) else ("", "")
    flag = "  <<<< 图标不同" if a[0] != b[0] else ""
    print(f"{i:>3}  {a[0]:<22} {a[1][:42]:<44} | {b[0]:<22} {b[1][:30]}{flag}")
