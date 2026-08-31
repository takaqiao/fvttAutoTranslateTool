#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抠出「矿车目的地」对话框里那些**地点名**（轨道节点 label）。

上屏点：ember.mjs `#presentDestinationDialog`（:127683 起）
  `label.append(input, " ", d.label)` —— d 来自 `ember.scene.getReachableDestinations(cart)`，
  最终取的是轨道 `nodes[x].label`。

前置自证（两件都断言）：
  (A) 切对条数 —— 抠出的节点 label 条数 == 人工从源码数出的真值；
  (B) 改对地方 —— 必含项目所有者点名的那几个，且不得含同文件里**不是**节点 label 的串。
"""
import re, os, sys, io, json, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

U = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember"
MJS = os.path.join(U, "scripts", "ember.mjs")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(MJS, encoding="utf-8").read()


def _match_brace(i):
    d, j = 0, i
    while j < len(SRC):
        if SRC[j] == "{":
            d += 1
        elif SRC[j] == "}":
            d -= 1
            if d == 0:
                return j
        j += 1
    return len(SRC) - 1


nodes = collections.OrderedDict()
blocks = 0
for m in re.finditer(r'^\s*nodes:\s*\{', SRC, re.M):
    blocks += 1
    i = SRC.find("{", m.end() - 1)
    j = _match_brace(i)
    seg, ln0 = SRC[i:j + 1], SRC.count("\n", 0, i) + 1
    for mm in re.finditer(r'(?<![\w$.])label:\s*"((?:[^"\\]|\\.)*)"', seg):
        nodes.setdefault(mm.group(1), []).append(ln0 + seg.count("\n", 0, mm.start()))

print(f"扫到 `nodes: {{` 块 {blocks} 个，节点 label {len(nodes)} 条")

# ---- 真值：用一条**独立**的判据数一遍（不复用上面的括号配平）----
# 轨道定义只有两处：`static #RED_TRACK = {` 与 `static #BLUE_TRACK = {`（YakoshtaMine 类内）。
# 在这两段的行区间里按「8 空格缩进的 label 行」数 —— 与括号配平是两条互不依赖的路子。
decls = [(m.group(1), SRC.count("\n", 0, m.start()) + 1)
         for m in re.finditer(r'^  static (#\w*TRACK) = \{', SRC, re.M)]
print(f"独立复算：找到轨道声明 {decls}")
if len(decls) != 2:
    print("  [A] FAIL 轨道声明不是 2 个，复算路子失效"); sys.exit(1)
all_lines = SRC.split("\n")
# 每段的结尾＝下一个行首为 `  }` 的行（打包器排版：类成员以 2 空格 `}` 收尾）
truth = collections.Counter()
for _, start in decls:
    k = start
    while k < len(all_lines) and all_lines[k] != "  };":
        m = re.match(r'^        label: "((?:[^"\\]|\\.)*)",?\s*$', all_lines[k])
        if m:
            truth[m.group(1)] += 1
        k += 1
print(f"独立复算（两段轨道内、8 空格缩进的 label 行）：{len(truth)} 条 / 出现 {sum(truth.values())} 次")

ok = True
print(f"[A 条数] 括号配平 {len(nodes)} == 缩进复算 {len(truth)} ? "
      + ("OK" if set(nodes) == set(truth) else f"FAIL 差集 {set(nodes) ^ set(truth)}"))
ok = ok and set(nodes) == set(truth)

MUST = ["Loading Zone", "Loading Zone (Broken)", "Ooze Farm", "Excavation Pit",
        "Southern Ore Pit", "Excavation Barricade", "Junction", "Supply Cache",
        "Waterfall (Broken)"]
MUST_NOT = ["Mine Cart Destination", "Forwards", "Backwards", "Unreachable",
            "Hair Roots", "Arms", "Close", "Confirm"]
miss = [x for x in MUST if x not in nodes]
leak = [x for x in MUST_NOT if x in nodes]
print(f"[B 必含] {len(MUST) - len(miss)}/{len(MUST)}" + ("" if not miss else f"  漏：{miss}"))
print(f"[B 必不含] {len(MUST_NOT) - len(leak)}/{len(MUST_NOT)}" + ("" if not leak else f"  串类：{leak}"))
ok = ok and not miss and not leak

print()
for k, v in sorted(nodes.items()):
    print(f"  {k!r}  @{v}")

json.dump({k: v for k, v in nodes.items()},
          open(os.path.join(HERE, "track_nodes.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n写出 track_nodes.json ；自证：" + ("OK" if ok else "FAIL"))
sys.exit(0 if ok else 1)
