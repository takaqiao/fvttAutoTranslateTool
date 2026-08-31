# -*- coding: utf-8 -*-
"""
probe_label_shape_coverage.py  —— 「覆盖表只覆盖了枚举的一个分支」这一类缺陷的机械判据

母题（已确认实例）：
  ember-hardcoded-cn.mjs 的 PREFIXED 里有 "Music Mood"，但 EmberSoundscape.enricherHTML
  实际能产出三种 label，PREFIXED 只吃到最罕见的那一种。

抽象成判据：
  **上游代码在运行时拼出的每一个「用户可见字面量形态」，都必须能被 CN 运行时替换层
  （EXACT / PREFIXED / PATTERNS / CALENDAR_*）匹配到；只覆盖其中一部分分支就是缺陷。**

做法：
  1. 从 ember-hardcoded-cn.mjs 里解析出 EXACT 键集、PREFIXED 前缀集、PATTERNS 正则集。
  2. 扫上游 ember.mjs / dnd5e-async.mjs / crucible-async.mjs / crucible-compiled.mjs，
     抓所有「赋给用户可见 sink」的模板字面量 / 字符串字面量，
     sink = innerHTML / innerText / textContent / .label= / label: / title: /
            tooltipText: / tooltip: / .name= / name: / hint: / placeholder:
  3. 把模板字面量压成 skeleton：`${...}` → \x00（DYN 占位）。
  4. 用 Python 重实现 translateText 的匹配顺序，问：这个 skeleton 的任意实例
     是否可能被匹配上？分三档：
        COVERED    —— 静态串在 EXACT 里 / 前缀在 PREFIXED 里 / 有 PATTERNS 正则能吃
        UNCOVERED  —— 含英文单词但没有任何规则能吃
        NOSTATIC   —— 静态部分没有英文单词（不关心）
  5. 只输出 UNCOVERED 且静态部分像人话（含 >=1 个长度>=3 的英文词、不像 CSS 类名 /
     i18n 键 / 文件路径 / HTML 标签）的。

已知假阳性模式（必须人工过一遍）：
  - i18n 键（`EMBER.XXX.YYY`）、CSS 选择器、文件路径、fa- 图标类、HTML 片段 → 已用启发式过滤，
    但过滤不干净；
  - dnd5e 分支专用的 label（项目已定「先不管」）；
  - 只在 GM 开发/调试界面出现的串；
  - 由 game.i18n.localize 包一层的（那条通道走 lang/cn.json，不归 hardcoded-cn 管）——
    脚本会检查同一行是否出现 localize/format，出现就降级为 I18N。
  - 上游会把它塞进 document.name 后由 babele 翻（极少数）。
只读，不写任何库文件。
"""
import re, os, sys, json, io

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
HC = os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")
UP = [
    r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs",
    r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\dnd5e-async.mjs",
    r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\crucible-async.mjs",
]

DYN = "\x00"

# ---------- 1. 解析 CN 替换层 ----------
def parse_cn_layer():
    src = io.open(HC, encoding="utf-8").read()

    def obj_keys(name):
        m = re.search(r"const %s\s*=\s*\{(.*?)\n\};" % name, src, re.S)
        if not m:
            return set()
        return set(re.findall(r'"((?:[^"\\]|\\.)*)"\s*:', m.group(1)))

    exact = obj_keys("EXACT")
    results = obj_keys("RESULTS")
    m = re.search(r"const PREFIXED\s*=\s*\[(.*?)\n\];", src, re.S)
    prefixes = set(re.findall(r'en:\s*"([^"]+)"', m.group(1))) if m else set()
    m = re.search(r"const PATTERNS\s*=\s*\[(.*?)\n\];", src, re.S)
    pats = re.findall(r"re:\s*/(.+?)/,", m.group(1)) if m else []
    return exact, results, prefixes, pats

EXACT, RESULTS, PREFIXES, PATTERNS_JS = parse_cn_layer()

def js_re_to_py(r):
    # 这几条都是简单正则，JS/Python 语法一致
    return re.compile(r)

PATTERNS = [js_re_to_py(r) for r in PATTERNS_JS]

# ---------- 2. 抓上游 sink ----------
SINKS = re.compile(
    r"(?:innerHTML|innerText|textContent|\.label|\.name|\.title|\.tooltip)\s*=\s*"
    r"|(?:label|title|tooltipText|tooltip|name|hint|placeholder|alias|flavor)\s*:\s*"
)

def scan_literals(path):
    """返回 (行号, 原文, skeleton, 整行) 列表"""
    out = []
    lines = io.open(path, encoding="utf-8").read().split("\n")
    for i, line in enumerate(lines, 1):
        for m in SINKS.finditer(line):
            rest = line[m.end():].lstrip()
            if not rest:
                continue
            q = rest[0]
            if q not in "`\"'":
                continue
            # 抓到配对的引号（模板串里 ${} 内部不会再有同种未转义引号的常见情形）
            j = 1
            depth = 0
            buf = []
            while j < len(rest):
                c = rest[j]
                if c == "\\":
                    buf.append(rest[j:j+2]); j += 2; continue
                if q == "`" and c == "$" and j+1 < len(rest) and rest[j+1] == "{":
                    depth += 1; buf.append("${"); j += 2; continue
                if depth and c == "}":
                    depth -= 1; buf.append("}"); j += 1; continue
                if c == q and depth == 0:
                    break
                buf.append(c); j += 1
            else:
                continue
            lit = "".join(buf)
            skel = re.sub(r"\$\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", DYN, lit)
            out.append((i, lit, skel, line.strip()))
    return out

# ---------- 3. 覆盖判定 ----------
WORD = re.compile(r"[A-Za-z]{3,}")

def classify(skel):
    static = skel.replace(DYN, "").strip()
    if not WORD.search(static):
        return "NOSTATIC"
    s = skel.strip()
    # 完全静态
    if DYN not in s:
        if s in EXACT or s in RESULTS:
            return "COVERED"
    # 前缀
    mm = re.match(r"^([A-Za-z][A-Za-z ']*): ", s)
    if mm and mm.group(1) in PREFIXES:
        return "COVERED"
    # PATTERNS：用 DYN→"X" 造一个实例试
    inst = s.replace(DYN, "X")
    if inst in EXACT or inst in RESULTS:
        return "COVERED"
    for p in PATTERNS:
        if p.match(inst):
            return "COVERED"
    # 数字型动态部分再试一次
    inst2 = s.replace(DYN, "7")
    for p in PATTERNS:
        if p.match(inst2):
            return "COVERED"
    return "UNCOVERED"

NOISE = re.compile(
    r"^(?:[\w./-]+\.(?:hbs|html|css|mjs|js|json|webp|png|jpg|svg|ogg|webm|wav))$"
    r"|^[A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)+$"          # i18n 键
    r"|^EMBER\.|^CRUCIBLE\.|^DND5E\."
    r"|^(?:fa-|fas |far |fa-solid|fa-regular)"
    r"|^[a-z][\w-]*(?: [a-z][\w-]*)*$"              # 纯小写 css class 串
    r"|^modules/|^systems/|^icons/|^assets/"
)

def looks_like_ui(skel):
    static = skel.replace(DYN, " ").strip()
    if NOISE.search(static):
        return False
    if "<" in skel and ">" in skel:
        return False
    if skel.count(DYN) and len(static) < 3:
        return False
    # 至少要有一个首字母大写的英文词，或一个包含空格的英文短语
    if not re.search(r"[A-Z][a-z]{2,}", static):
        return False
    return True

def main():
    rows = []
    for path in UP:
        if not os.path.exists(path):
            continue
        for ln, lit, skel, line in scan_literals(path):
            cls = classify(skel)
            if cls != "UNCOVERED":
                continue
            if not looks_like_ui(skel):
                continue
            i18n = bool(re.search(r"localize|i18n\.format|game\.i18n", line))
            rows.append({
                "file": os.path.basename(path), "line": ln,
                "skeleton": skel.replace(DYN, "{}"),
                "i18n_on_same_line": i18n,
                "src": line[:220],
            })
    rows.sort(key=lambda r: (r["i18n_on_same_line"], r["file"], r["line"]))
    print("EXACT=%d PREFIXES=%s PATTERNS=%d" % (len(EXACT), sorted(PREFIXES), len(PATTERNS)))
    print("UNCOVERED candidates: %d (i18n-shadowed: %d)"
          % (len(rows), sum(1 for r in rows if r["i18n_on_same_line"])))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_shape_uncovered.json")
    json.dump(rows, io.open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for r in rows:
        if r["i18n_on_same_line"]:
            continue
        print("%-18s %6d  %s" % (r["file"], r["line"], r["skeleton"]))

if __name__ == "__main__":
    main()
