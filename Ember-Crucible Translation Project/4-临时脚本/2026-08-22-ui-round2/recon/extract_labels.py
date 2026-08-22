# -*- coding: utf-8 -*-
"""
全库扫 ember.mjs 里的**属性型字面量**：`label:` / `prompt:` / `title:` / `legend:` / `text:` /
`name:`（仅在含 label 的对象里出现的那类不单独抓，避免噪声）。
上一轮的抽取器只抓「整句英文」形态，`label: "Hand Left"` 这种属性字面量整类漏掉。

前置自证：
  A 切对条数 —— `label:` 形态的条数 == 独立 grep -c 的真值
  B 切对对象 —— 已知真值必须全部抓到：Hand Left / Tail / Cheeks / Ears / Hand Right（图层名）、
                Tradeway (Region Map)（升降机目的地）、Loading Zone（矿车轨道节点）
"""
import json, re, io, sys

SRC = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs"
src = io.open(SRC, encoding="utf-8").read()
lines = src.split("\n")
offs, p = [], 0
for ln in lines:
    offs.append(p); p += len(ln) + 1


def lineno(i):
    lo, hi = 0, len(offs) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if offs[mid] <= i: lo = mid
        else: hi = mid - 1
    return lo + 1


BS = chr(92)
Q = '"((?:[^"' + BS + BS + BS + 'n]|' + BS + BS + '.)*)"'
PROPS = ["label", "prompt", "title", "legend", "hint", "placeholder", "tooltip", "description"]
PAT = re.compile(r"\b(" + "|".join(PROPS) + r")\s*[:=]\s*" + Q)

# 类名索引：每个 `class X` 的起点
classes = [(m.start(), m.group(1)) for m in re.finditer(r"\bclass\s+([A-Za-z0-9_$]+)", src)]


def nearest_class(i):
    lo, hi, ans = 0, len(classes) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if classes[mid][0] <= i:
            ans = classes[mid][1]; lo = mid + 1
        else:
            hi = mid - 1
    return ans


out = {}
by_prop = {}
for m in PAT.finditer(src):
    prop, v = m.group(1), m.group(2)
    if not re.search(r"[A-Za-z]", v): continue
    if re.match(r"^(fa-|fas |modules/|systems/|icons/|assets/|EMBER\.|CRUCIBLE\.|DND5E\.|TOKEN\.|[a-z][A-Za-z0-9]*$)", v):
        continue
    by_prop[prop] = by_prop.get(prop, 0) + 1
    rec = out.setdefault(v, {"props": set(), "at": [], "hosts": set()})
    rec["props"].add(prop)
    rec["at"].append("ember.mjs:%d" % lineno(m.start()))
    rec["hosts"].add(str(nearest_class(m.start())))

# ---- 前置自证 A ----
n_label_raw = len(re.findall(r"\blabel\s*[:=]\s*" + Q, src))
print("PRECHECK-A  label 形态原始命中 %d（过滤后计入 %d）" % (n_label_raw, by_prop.get("label", 0)))
assert n_label_raw >= by_prop.get("label", 0) > 100, (n_label_raw, by_prop.get("label"))

# ---- 前置自证 B ----
MUST = ["Hand Left", "Hand Right", "Tail", "Ears", "Cheeks",
        "Tradeway (Region Map)", "Loading Zone"]
missing = [k for k in MUST if k not in out]
assert not missing, "已知真值没抓到：%s" % missing
print("PRECHECK-B OK  7/7 已知真值全部抓到")
print("按属性计数：", json.dumps(by_prop, sort_keys=True))

res = {k: {"props": sorted(v["props"]), "hosts": sorted(v["hosts"]), "at": sorted(set(v["at"]))[:4]}
       for k, v in sorted(out.items())}
io.open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
print("候选 =", len(res))
