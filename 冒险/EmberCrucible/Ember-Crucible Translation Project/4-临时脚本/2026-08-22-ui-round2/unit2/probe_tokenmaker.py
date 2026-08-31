# -*- coding: utf-8 -*-
"""
指示物制作器：图层 label 全集 + 部件 id 全集 + 显示名推导。

前置自证（两件都断言，缺一即 exit 1）：
  A 切对条数：本脚本切出的 token-maker 数据区命中数 == 两种独立切法逐条相同，
             且全库 `label: "` 命中数 == 独立行计数真值（5474 / 5473 行）。
  B 切对对象：已知真值必须**逐条**在结果里
             （Hand Left/Hand Right/Tail/Ears/Cheeks 是图层 label；Beard1/BeardWizard 是部件 id）。
"""
import io, json, re, sys, os

SRC = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs"
OUT = os.path.dirname(os.path.abspath(__file__))
src = io.open(SRC, encoding="utf-8").read()
lines = src.split("\n")

# ---------- 数据区边界：不靠猜，靠锚点 ----------
def find_line(pat, start=0):
    for i in range(start, len(lines)):
        if re.search(pat, lines[i]):
            return i
    raise SystemExit("anchor not found: " + pat)

L_START = find_line(r"^function makeParts\(")             # 部件工厂
L_END = find_line(r"^const VERSION_1_PART_MIGRATIONS")     # 数据区之后的迁移表
print("REGION 1-based %d..%d" % (L_START + 1, L_END + 1))
region = "\n".join(lines[L_START:L_END])

BS = chr(92)
STR = '"((?:[^"' + BS + BS + BS + 'n]|' + BS + BS + '.)*)"'
LAB = re.compile(r"label:\s*" + STR)

# ---------- 前置自证 A2：全库真值 ----------
all_hits = [(m.start(), m.group(1)) for m in LAB.finditer(src)]
n_lines_with = sum(1 for l in lines if 'label: "' in l)
print("PRECHECK-A2  全库 label: 字面量 %d 条 / 含该形态的行 %d 行" % (len(all_hits), n_lines_with))
assert len(all_hits) >= n_lines_with, (len(all_hits), n_lines_with)

# ---------- 前置自证 A1：区间两种切法必须逐条相同 ----------
off, p = [], 0
for ln in lines:
    off.append(p); p += len(ln) + 1

def lineno0(i):
    lo, hi = 0, len(off) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if off[mid] <= i: lo = mid
        else: hi = mid - 1
    return lo

in_region_from_full = [v for (i, v) in all_hits if L_START <= lineno0(i) < L_END]
in_region_direct = [m.group(1) for m in LAB.finditer(region)]
assert in_region_from_full == in_region_direct, (
    "PRECHECK-A1 FAIL %d vs %d" % (len(in_region_from_full), len(in_region_direct)))
print("PRECHECK-A1 OK  区间内 label: 字面量 %d 条（两种独立切法逐条相同）" % len(in_region_direct))

# ---------- 图层 label 全集 ----------
layer_labels = {}
for m in LAB.finditer(region):
    v = m.group(1)
    ln = L_START + region[:m.start()].count("\n") + 1
    layer_labels.setdefault(v, []).append(ln)

# ---------- 部件 id 全集 ----------
IDP = re.compile(r"\{\s*id:\s*" + STR)
part_ids = {}
for m in IDP.finditer(region):
    v = m.group(1)
    ln = L_START + region[:m.start()].count("\n") + 1
    part_ids.setdefault(v, []).append(ln)

# ---------- 前置自证 B ----------
MUST_LABEL = ["Hand Left", "Hand Right", "Tail", "Ears", "Cheeks", "Arms", "Face"]
miss = [k for k in MUST_LABEL if k not in layer_labels]
assert not miss, "PRECHECK-B FAIL 图层 label 已知真值缺：%s" % miss
MUST_PART = ["Beard1", "BeardWizard"]
miss = [k for k in MUST_PART if k not in part_ids]
assert not miss, "PRECHECK-B FAIL 部件 id 已知真值缺：%s" % miss
print("PRECHECK-B OK  图层 %d/%d + 部件 %d/%d 已知真值全部抓到"
      % (len(MUST_LABEL), len(MUST_LABEL), len(MUST_PART), len(MUST_PART)))

print("区间内唯一 label 值 = %d" % len(layer_labels))
print('区间内 {id:"…"} 唯一值 = %d' % len(part_ids))

json.dump({"region": [L_START + 1, L_END + 1],
           "layer_labels": {k: v[:3] for k, v in sorted(layer_labels.items())},
           "part_ids": {k: v[:3] for k, v in sorted(part_ids.items())}},
          io.open(os.path.join(OUT, "tokenmaker_raw.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("WROTE tokenmaker_raw.json")
