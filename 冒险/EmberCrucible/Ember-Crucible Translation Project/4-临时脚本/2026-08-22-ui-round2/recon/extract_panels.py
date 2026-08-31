# -*- coding: utf-8 -*-
"""
抽取 ember.mjs 里「互动面板」这一类的全部候选串。
区域三类：(1) `dialog: {...}` 配置块  (2) `_configureDialog(...) {...}` 方法体
         (3) DialogV2.{prompt,wait,input,confirm,query} 的实参
前置自证（见 __main__ 末尾）：
  A 切对条数：三类区域各自的条数 == 独立 grep 数出的已知真值
  B 切对对象：每个区域必须括号配平，且抽样区域的首尾字符必须是 {} / ()
"""
import json, re, sys, io

SRC = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs"
src = io.open(SRC, encoding="utf-8").read()
lines = src.split("\n")
offs = []
p = 0
for ln in lines:
    offs.append(p)
    p += len(ln) + 1


def lineno(idx):
    lo, hi = 0, len(offs) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if offs[mid] <= idx:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


BS = chr(92)      # 反斜杠
BT = chr(96)      # 反引号


def match_block(s, i, op, cl):
    """i 指向 op 的位置，返回闭合位置 index（含）。跳过字符串/注释/模板串。"""
    depth = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '"' or c == "'":
            q = c
            i += 1
            while i < n:
                if s[i] == BS:
                    i += 2
                    continue
                if s[i] == q:
                    break
                i += 1
        elif c == BT:
            i += 1
            while i < n:
                if s[i] == BS:
                    i += 2
                    continue
                if s[i] == BT:
                    break
                i += 1
        elif c == "/" and i + 1 < n and s[i + 1] == "/":
            while i < n and s[i] != "\n":
                i += 1
            continue
        elif c == "/" and i + 1 < n and s[i + 1] == "*":
            j = s.find("*/", i + 2)
            i = (j + 1) if j >= 0 else n
        elif c == op:
            depth += 1
        elif c == cl:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


regions = []   # (kind, start, end, open_char)

for m in re.finditer(r"\bdialog:\s*\{", src):
    b = src.index("{", m.start())
    e = match_block(src, b, "{", "}")
    if e < 0:
        sys.exit("dialog block unbalanced @%d" % lineno(b))
    regions.append(("dialog-config", b, e + 1, "{"))

for m in re.finditer(r"(?:async\s+)?_configureDialog\s*\([^)]*\)\s*\{", src):
    b = src.index("{", m.end() - 1)
    e = match_block(src, b, "{", "}")
    if e < 0:
        sys.exit("_configureDialog unbalanced @%d" % lineno(b))
    regions.append(("configureDialog", b, e + 1, "{"))

for m in re.finditer(r"DialogV2(?:\$\d+)?\s*\.\s*(prompt|wait|input|confirm|query)\s*\(", src):
    b = src.index("(", m.end() - 1)
    e = match_block(src, b, "(", ")")
    if e < 0:
        sys.exit("DialogV2 call unbalanced @%d" % lineno(b))
    regions.append(("DialogV2." + m.group(1), b, e + 1, "("))

# ---- 前置自证 A：切对条数 ----
EXPECT = {"dialog-config": 21, "configureDialog": 12,
          "DialogV2.confirm": 14, "DialogV2.input": 6,
          "DialogV2.prompt": 11, "DialogV2.wait": 6}
got = {}
for k, _s, _e, _o in regions:
    got[k] = got.get(k, 0) + 1
if got != EXPECT:
    sys.exit("PRECHECK-A FAIL 条数不符：got=%s expect=%s" % (got, EXPECT))
print("PRECHECK-A OK 区域条数 =", json.dumps(got, sort_keys=True))

# ---- 前置自证 B：切对对象（每个区域首尾是配对括号，且长度 > 1）----
CLOSE = {"{": "}", "(": ")"}
for k, s, e, o in regions:
    seg = src[s:e]
    if not (len(seg) > 1 and seg[0] == o and seg[-1] == CLOSE[o]):
        sys.exit("PRECHECK-B FAIL @%s ember.mjs:%d 首尾不是配对括号" % (k, lineno(s)))
print("PRECHECK-B OK 全部 %d 个区域首尾配对" % len(regions))

STR = re.compile('(?<![' + BS + 'w$])(?:"((?:[^"' + BS + BS + '\\n]|' + BS + BS + '.)*)"'
                 + "|'((?:[^'" + BS + BS + '\\n]|' + BS + BS + ".)*)')")
TPL = re.compile(BT + '((?:[^' + BT + BS + BS + ']|' + BS + BS + '.)*)' + BT, re.S)

SKIP = re.compile(r"^(fa-|fas |far |modules/|systems/|icons/|assets/|ember\.|EMBER\.|CRUCIBLE\.|DND5E\.|#|\.|--)")

out = {}
for kind, s, e, o in regions:
    seg = src[s:e]
    for m in STR.finditer(seg):
        v = m.group(1) if m.group(1) is not None else m.group(2)
        v = v.replace(BS + '"', '"').replace(BS + "'", "'").replace(BS + "n", "\n")
        if not re.search(r"[A-Za-z]", v):
            continue
        if SKIP.match(v):
            continue
        rec = out.setdefault(v, {"kinds": set(), "at": []})
        rec["kinds"].add(kind)
        rec["at"].append("ember.mjs:%d" % lineno(s + m.start()))
    for m in TPL.finditer(seg):
        v = m.group(1)
        if not re.search(r"[A-Za-z]", v):
            continue
        key = "TPL:" + v
        rec = out.setdefault(key, {"kinds": set(), "at": []})
        rec["kinds"].add(kind)
        rec["at"].append("ember.mjs:%d" % lineno(s + m.start()))

res = {k: {"kinds": sorted(v["kinds"]), "at": sorted(set(v["at"]))} for k, v in sorted(out.items())}
io.open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
print("候选串 =", len(res))
