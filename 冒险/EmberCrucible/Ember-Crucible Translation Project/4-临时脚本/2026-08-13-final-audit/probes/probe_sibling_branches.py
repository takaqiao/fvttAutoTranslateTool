# -*- coding: utf-8 -*-
"""
probe_sibling_branches.py —— 母题的精确机械化

母题：PREFIXED 有 "Music Mood"，但产出它的那**一个表达式**（EmberSoundscape.enricherHTML）
还能产出另外两种 label，CN 层只吃到其中最罕见的一支。

判据（锚定式，噪声远低于「全库 sink 扫描」）：
  对 CN 替换层的每一条规则 R：
    1. 在上游源码里找到产出 R 的那一处/那几处代码（按 R 的静态字面量 grep）；
    2. 取其**所在的产出点**（同一个赋值语句 / 同一个三元 / 同一个枚举对象 / 同一个函数体内
       的同类 sink 赋值）；
    3. 枚举该产出点能产出的**全部** label 形态；
    4. 逐一问：CN 层能不能吃？吃不到的就是「同一处产出点的兄弟分支未覆盖」。

输出 = 每条 CN 规则的锚点 + 其兄弟分支的覆盖情况。人工只需看 SIBLING_UNCOVERED。

假阳性模式：
  - 同名字符串在别处巧合出现（如 "Common" 既是语言又是稀有度）→ 脚本给出所有命中点，需人工挑；
  - dnd5e 专属分支（项目已定先不管）→ 标注 system=dnd5e；
  - 兄弟分支本身走 i18n（同行有 localize）→ 标注。
只读。
"""
import re, os, io, json, collections

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
HC = os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")
UPDIR = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts"
FILES = ["ember.mjs", "dnd5e-async.mjs", "crucible-async.mjs"]

src_hc = io.open(HC, encoding="utf-8").read()

def obj_keys(name):
    m = re.search(r"const %s\s*=\s*\{(.*?)\n\};" % name, src_hc, re.S)
    return set(re.findall(r'"((?:[^"\\]|\\.)*)"\s*:', m.group(1))) if m else set()

EXACT = obj_keys("EXACT")
RESULTS = obj_keys("RESULTS")
m = re.search(r"const PREFIXED\s*=\s*\[(.*?)\n\];", src_hc, re.S)
PREFIXES = re.findall(r'en:\s*"([^"]+)"', m.group(1))
m = re.search(r"const PATTERNS\s*=\s*\[(.*?)\n\];", src_hc, re.S)
PATS = [re.compile(r) for r in re.findall(r"re:\s*/(.+?)/,", m.group(1))]

def cn_covers(s):
    """CN 层能否吃下这个具体串（{} 代表动态段，用 X 代入）"""
    s = s.strip()
    if s in EXACT or s in RESULTS:
        return True
    for p in PREFIXES:
        if s.startswith(p + ": "):
            return True
    inst = s.replace("{}", "X")
    if inst in EXACT or inst in RESULTS:
        return True
    for p in PREFIXES:
        if inst.startswith(p + ": "):
            return True
    for rx in PATS:
        if rx.match(inst) or rx.match(s.replace("{}", "7")):
            return True
    return False

TEXT = {f: io.open(os.path.join(UPDIR, f), encoding="utf-8").read().split("\n") for f in FILES}

# 只在「产出用户可见文本」的行上锚定
SINK = re.compile(r"innerHTML|innerText|textContent|\btitle\b|\blabel\b|tooltip|\bname\b|\btext\b|content|notify|notification|\bhint\b")

def skeleton(lit):
    return re.sub(r"\$\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "{}", lit)

LIT = re.compile(r"`((?:[^`\\]|\\.)*)`|\"((?:[^\"\\]|\\.)*)\"|'((?:[^'\\]|\\.)*)'")

def literals_in(line):
    out = []
    for m in LIT.finditer(line):
        out.append(m.group(1) if m.group(1) is not None else (m.group(2) if m.group(2) is not None else m.group(3)))
    return out

def anchors_for(token):
    """找到上游里出现该静态 token 的 sink 行"""
    hits = []
    for f, lines in TEXT.items():
        for i, line in enumerate(lines, 1):
            if token not in line:
                continue
            if not SINK.search(line):
                continue
            lits = [skeleton(l) for l in literals_in(line)]
            if not any(token in l for l in lits):
                continue
            hits.append((f, i, line.strip()))
    return hits

def window_siblings(f, ln, radius=6):
    """同一处产出点的兄弟字面量：取上下 radius 行里带 sink 的字面量"""
    lines = TEXT[f]
    lo, hi = max(1, ln - radius), min(len(lines), ln + radius)
    sibs = []
    for i in range(lo, hi + 1):
        line = lines[i - 1]
        if not SINK.search(line):
            continue
        for l in literals_in(line):
            s = skeleton(l)
            if re.search(r"[A-Z][a-z]{2,}", s.replace("{}", " ")):
                sibs.append((i, s, line.strip()))
    return sibs

def main():
    rules = sorted(EXACT) + [p + ": {}" for p in PREFIXES]
    report = []
    for rule in rules:
        token = rule.split(": {}")[0] if rule.endswith(": {}") else rule
        if len(token) < 4:
            continue
        hits = anchors_for(token)
        for f, ln, line in hits[:6]:
            sibs = window_siblings(f, ln)
            unc = []
            for si, s, sline in sibs:
                if cn_covers(s):
                    continue
                if re.match(r"^(?:[\w./-]+\.(?:hbs|html|css|mjs|js|json|webp|png|svg|ogg))$", s):
                    continue
                if re.match(r"^[A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)+$", s) or s.startswith("EMBER.") or s.startswith("CRUCIBLE."):
                    continue
                if "<" in s and ">" in s:
                    continue
                unc.append({"line": si, "shape": s, "src": sline[:200]})
            if unc:
                report.append({"rule": rule, "file": f, "anchor_line": ln,
                               "anchor_src": line[:200], "sibling_uncovered": unc})
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sibling_branches.json")
    json.dump(report, io.open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("rules=%d anchored-with-uncovered-siblings=%d" % (len(rules), len(report)))
    for r in report:
        print("\n== %s  @ %s:%d" % (r["rule"], r["file"], r["anchor_line"]))
        print("   anchor: %s" % r["anchor_src"])
        for u in r["sibling_uncovered"]:
            print("   MISS %6d  %s" % (u["line"], u["shape"]))

if __name__ == "__main__":
    main()
