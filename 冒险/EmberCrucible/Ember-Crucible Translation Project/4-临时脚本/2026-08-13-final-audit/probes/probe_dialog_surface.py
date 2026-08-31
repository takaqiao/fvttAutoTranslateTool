# -*- coding: utf-8 -*-
"""
probe_dialog_surface.py —— 母题的第二个机械化角度

母题：同一个产出点能吐出多种用户可见串，CN 层只收了其中一种。
对话框是这个形状最集中的地方：一次 DialogV2.confirm/prompt/input/wait 调用同时产出
  window.title + content + buttons[].label + ok.label/yes.label/no.label
EXACT 里收了 15 个 window.title，那么同一次调用的 content / buttons 收了没有？

做法：在 ember.mjs 里定位每个 DialogV2 调用（按 `window: {` 或 `.confirm(` 起手），
取该调用的字面量块（到配对的 `});`），抽出其中所有静态英文串，逐个问 CN 层能否吃。
只读。
"""
import re, os, io, json

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
HC = os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")
UPF = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"

src_hc = io.open(HC, encoding="utf-8").read()
def obj_keys(name):
    m = re.search(r"const %s\s*=\s*\{(.*?)\n\};" % name, src_hc, re.S)
    return set(re.findall(r'"((?:[^"\\]|\\.)*)"\s*:', m.group(1))) if m else set()
EXACT = obj_keys("EXACT") | obj_keys("RESULTS")
m = re.search(r"const PREFIXED\s*=\s*\[(.*?)\n\];", src_hc, re.S)
PREFIXES = re.findall(r'en:\s*"([^"]+)"', m.group(1))
m = re.search(r"const PATTERNS\s*=\s*\[(.*?)\n\];", src_hc, re.S)
PATS = [re.compile(r) for r in re.findall(r"re:\s*/(.+?)/,", m.group(1))]

def covered(s):
    s = s.strip()
    if not s: return True
    if s in EXACT: return True
    for p in PREFIXES:
        if s.startswith(p + ": "): return True
    inst = s.replace("{}", "X")
    if inst in EXACT: return True
    for p in PREFIXES:
        if inst.startswith(p + ": "): return True
    for rx in PATS:
        if rx.match(inst) or rx.match(s.replace("{}", "7")): return True
    return False

lines = io.open(UPF, encoding="utf-8").read().split("\n")
LIT = re.compile(r"`((?:[^`\\]|\\.)*)`|\"((?:[^\"\\]|\\.)*)\"|'((?:[^'\\]|\\.)*)'")
def skel(s): return re.sub(r"\$\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "{}", s)

# 找出每个含 `window: {` 或 DialogV2 方法名的调用块
starts = []
for i, l in enumerate(lines, 1):
    if re.search(r"DialogV2[\w$]*\.(confirm|prompt|input|wait|query)\s*\(", l) or re.search(r"new DialogV2", l):
        starts.append(i)

def block(i, maxlen=40):
    """从第 i 行开始按括号配平取块"""
    depth = 0; out = []; started = False
    for j in range(i, min(i + maxlen, len(lines)) + 1):
        l = lines[j - 1]
        out.append((j, l))
        for ch in l:
            if ch in "({[": depth += 1; started = True
            elif ch in ")}]": depth -= 1
        if started and depth <= 0:
            break
    return out

report = []
for s in starts:
    blk = block(s)
    title = None
    fields = []
    for j, l in blk:
        for m2 in re.finditer(r"(title|label|content|hint|placeholder)\s*:\s*", l):
            rest = l[m2.end():]
            mm = LIT.match(rest.lstrip())
            if not mm: continue
            v = mm.group(1) if mm.group(1) is not None else (mm.group(2) if mm.group(2) is not None else mm.group(3))
            v = skel(v)
            if not re.search(r"[A-Za-z]{3,}", v.replace("{}", " ")): continue
            if v.startswith("EMBER.") or v.startswith("CRUCIBLE."): continue
            if re.match(r"^fa-", v): continue
            fields.append((j, m2.group(1), v, covered(v)))
            if m2.group(1) == "title" and title is None: title = v
        # content 有时是多行模板串
    if not fields: continue
    title_cov = any(k == "title" and c for _, k, _, c in fields)
    unc = [(j, k, v) for j, k, v, c in fields if not c]
    if title_cov and unc:
        report.append({"start": s, "title": title,
                       "uncovered": [{"line": j, "key": k, "text": v} for j, k, v in unc]})

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dialog_surface.json")
json.dump(report, io.open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("DialogV2 call sites scanned: %d" % len(starts))
print("sites whose TITLE is covered but other surfaces are NOT: %d" % len(report))
for r in report:
    print("\n@%d  title=%r" % (r["start"], r["title"]))
    for u in r["uncovered"]:
        print("   MISS %-9s %6d  %s" % (u["key"], u["line"], u["text"][:120]))
