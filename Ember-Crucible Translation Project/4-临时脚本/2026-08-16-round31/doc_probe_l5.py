# -*- coding: utf-8 -*-
"""文档这一路的独立复算（第三十一轮）：⑵ 号通则「反空转的数必须数规矩、不是数叶子」的现场证据。

做法：把发布规则集复制三份到本目录，各做**一处**掏空，用**现役判据**跑 `--rules <副本>`：
  A  15 条 `cn_absent` 的禁用词表**全部**清空
  B  `R-lantyr.cn_forbidden = []`
  C  `R-bare-cn-fields.fields = []`
然后**逐条对照 detail**：哪个数塌了、哪个数一个字节不变。

⚠ 前置自证（硬约束 4）：改之前先断言「我切到的条数 == 已知真值」——
   cn_absent 15 条、R-lantyr / R-bare-cn-fields 各 1 条，对不上当场退出。
⚠ 本脚本**只写自己目录里的副本**，一个字节都不碰发布规则集与判据文件。
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RULES = os.path.join(ROOT, "5-其他内容", "RESOLUTIONS.assertions.json")
JUDGE = os.path.join(ROOT, "3-常用脚本", "qa", "assert_resolutions.py")


def load():
    with open(RULES, encoding="utf-8") as fh:
        return json.load(fh)


def run(rules_path, tag):
    p = subprocess.run([sys.executable, JUDGE, "--verbose", "--rules", rules_path],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=ROOT)
    out = (p.stdout or "") + (p.stderr or "")
    with open(os.path.join(HERE, f"doc_probe_l5.{tag}.txt"), "w", encoding="utf-8") as fh:
        fh.write(out)
    last = [l for l in out.splitlines() if l.startswith("通过 ")]
    return p.returncode, (last[-1] if last else "（没打出合计行）"), out


def detail_of(out, rid):
    for l in out.splitlines():
        if f" {rid} " in l or l.strip().startswith(("ok    " + rid, "FAIL  " + rid)):
            if rid in l:
                return l.strip()
    return "（没找到这条）"


d = load()
A = d["assertions"]

# ── 前置自证 ─────────────────────────────────────────────
n_absent = sum(1 for r in A if r.get("kind") == "cn_absent")
n_lantyr = sum(1 for r in A if r["id"] == "R-lantyr")
n_bare = sum(1 for r in A if r["id"] == "R-bare-cn-fields")
print(f"[前置自证] 断言 {len(A)} 条 · cn_absent {n_absent} 条 · R-lantyr {n_lantyr} 条 · R-bare-cn-fields {n_bare} 条")
assert len(A) == 66, f"断言条数 {len(A)} ≠ 已知真值 66"
assert n_absent == 15, f"cn_absent {n_absent} ≠ 已知真值 15"
assert n_lantyr == 1 and n_bare == 1, "R-lantyr / R-bare-cn-fields 不是各 1 条"
# 禁用词的字段名也要自证：cn_absent 用的是哪个键、什么类型
#   ⚠ 第一版这里假设是 list 型的 `cn_absent`，前置自证当场把它拍掉了：
#     实际是**字符串**字段 `cn`（一条断言钉一个词）。任务书上的「禁用词表」是复数说法，
#     真实形状是 15 条断言各一个字符串 —— **这就是前置自证要干的活**。
#   ⚠ 第二版又被拍了一次：`cn` **不是同一种类型** —— 12 条是字符串、3 条是 list
#     （`R-token-foundry-ui` / `R-hex-tile` / `R-point-cape`，一条钉两个词）。
#     ⇒ 「15 条一次清空」在实现上要按类型分别置空，按单一类型写的探针会漏掉 3 条。
_t = {}
for r in A:
    if r.get("kind") == "cn_absent":
        _t.setdefault(type(r.get("cn")).__name__, []).append(r["id"])
print(f"[前置自证] cn_absent 的禁用词字段 `cn` 类型分布：" +
      "、".join(f"{k} {len(v)} 条" for k, v in sorted(_t.items())) +
      f"；list 型的是 {_t.get('list', [])}")
assert set(_t) <= {"str", "list"}, f"`cn` 出现了没预料到的类型：{set(_t)}"
assert sum(len(v) for v in _t.values()) == 15, "分类后条数对不上 15"
_words = sum(len(r["cn"]) if isinstance(r["cn"], list) else 1
             for r in A if r.get("kind") == "cn_absent")
print(f"[前置自证] 15 条 cn_absent 合计钉住 {_words} 个禁用词")
print("[前置自证] 通过\n")

base_rc, base_sum, base_out = run(RULES, "base")
print(f"基线（发布规则集）：{base_sum}　rc={base_rc}")
for rid in ("R-lantyr", "R-bare-cn-fields"):
    print(f"  基线 detail · {rid}：{detail_of(base_out, rid)}")
print()

cases = []

dA = load()
for r in dA["assertions"]:
    if r.get("kind") == "cn_absent":
        r["cn"] = [] if isinstance(r.get("cn"), list) else ""
cases.append(("A", "15 条 cn_absent 的禁用词（字段 `cn`，含 3 条 list 型）全部清空", dA, "R-soulbound-progression"))

# A2：同一个「清空」动作的**另一种写法** —— 一律置成空 list。
#     ⚠ A 里把字符串置成 "" 会让子串判据**命中一切**（空串是任何串的子串），
#       那是「炸成满屏违规」，不是「静默空转」；真正的空转形态是 A2 这种。
dA2 = load()
for r in dA2["assertions"]:
    if r.get("kind") == "cn_absent":
        r["cn"] = []
cases.append(("A2", "15 条 cn_absent 的禁用词一律置成空 list（真·空转形态）", dA2, "R-soulbound-progression"))

dB = load()
for r in dB["assertions"]:
    if r["id"] == "R-lantyr":
        r["cn_forbidden"] = []
cases.append(("B", "R-lantyr.cn_forbidden = []", dB, "R-lantyr"))

dC = load()
for r in dC["assertions"]:
    if r["id"] == "R-bare-cn-fields":
        r["fields"] = []
cases.append(("C", "R-bare-cn-fields.fields = []", dC, "R-bare-cn-fields"))

for tag, desc, doc, rid in cases:
    p = os.path.join(HERE, f"doc_probe_l5.{tag}.rules.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    rc, summ, out = run(p, tag)
    print(f"[{tag}] {desc}\n     → {summ}　rc={rc}")
    if rid:
        line = detail_of(out, rid)
        print(f"     detail：{line}")
