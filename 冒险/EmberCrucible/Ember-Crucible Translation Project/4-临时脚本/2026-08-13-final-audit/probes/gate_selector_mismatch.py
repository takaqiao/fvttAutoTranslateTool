# -*- coding: utf-8 -*-
"""
gate_selector_mismatch.py  ——  「闸/选择器失配」类判据（只读）

种子实例：ember-hardcoded-cn.mjs 的 translateText 只有
  (a) EXACT 全串相等  (b) PREFIXED 「<英文前缀>: 」开头  (c) PATTERNS 5 条正则
三种匹配器；而上游 ember.mjs 里大量用户可见串是**模板字面量拼出来的**，
只要拼出来的形状不落在这三种匹配器的值域里，就永远翻不到 —— 闸放行了，
匹配器够不着。

本判据把这一类机械化为两个方向：

  D1（漏 / leak）：上游有一条**可见槽位**被模板字面量赋值（innerHTML / innerText /
      textContent / label = / .name = / dataset.tooltip* / title），把 ${...} 抠掉后
      得到一个「骨架」。若该骨架
        - 含插值（EXACT 原理上不可能匹配），且
        - 不以任何 PREFIXED 前缀开头，且
        - 任何一条 PATTERNS 正则都吃不下（用占位值实例化后仍不匹配）
      → 该串全生命周期都是英文。

  D2（死匹配器 / dead matcher）：插件某条 EXACT 键在上游源码里**从来不作为完整字符串
      字面量出现**（只作为模板字面量的固定前缀出现，或压根不出现）
      → 这条 EXACT 永远命中不了，等于零覆盖但有虚假信心。

  D3（半译 / partial）：骨架以某条 PREFIXED 前缀开头，但插值落在 leaf 位置，
      leaf 查表是精确匹配 → 前缀译了、leaf 不译，产出「中文前缀：English」。

假阳性模式（必须逐条人工核实）：
  * innerHTML 里的模板串可能只是内部 HTML 骨架（class/attr），不含可见英文词；
    脚本用「骨架去掉 HTML 标签后是否还剩 [A-Za-z]{3,}」粗筛，仍会放过一些。
  * 有些槽位只在 dnd5e 分支跑（game.system.id === "dnd5e"），crucible 世界看不到。
  * 有些槽位的插值本身来自 babele 已翻译的 compendium 数据，骨架里的固定部分
    才是要看的。
  * 上游可能根本没有内容调用（[[/xxx]] 在语料里 0 次）—— 所以 D1/D3 命中后
    必须再去语料计数，0 次的不报。
"""
import json
import os
import re
import sys
from collections import Counter

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER_MJS = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
PLUGIN = os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")

# ---------------------------------------------------------------- 1. 匹配器清单

def load_matchers():
    src = open(PLUGIN, encoding="utf-8").read()

    def obj_keys(name):
        m = re.search(r"const %s = \{(.*?)\n\};" % re.escape(name), src, re.S)
        if not m:
            return []
        return re.findall(r'"([^"]+)":', m.group(1))

    exact = obj_keys("EXACT")
    tables = {n: obj_keys(n) for n in
              ["ATTUNEMENTS", "LANGUAGES", "KNOWLEDGE", "MOODS", "RESULTS",
               "CALENDAR_MONTHS", "CALENDAR_DAYS", "CALENDAR_DAY_ABBR"]}
    prefixed = re.findall(r'\{ en: "([^"]+)", cn: "([^"]+)", table: (\w+) \}', src)
    patterns = re.findall(r"\{ re: (/[^/]+(?:\\/[^/]*)*/), cn:", src)
    return exact, tables, prefixed, patterns


# js 正则 -> python 正则（本文件用到的都很简单）
def js_re(p):
    body = p[1:p.rindex("/")]
    return re.compile(body)


# ---------------------------------------------------------------- 2. 上游可见槽位

SINKS = re.compile(
    r"(?P<sink>\.innerHTML|\.innerText|\.textContent|\btooltip|\btooltipText|\btooltipHtml"
    r"|\btitle|\blabel|\bname)\s*(?:\+?=|:)\s*`(?P<tpl>(?:[^`\\]|\\.)*)`"
)

TAG = re.compile(r"<[^>]*>")
INTERP = re.compile(r"\$\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")
PH = "‹›"   # ‹› —— 不能用 <#>，会被 TAG 正则当成标签一起吃掉（第一版的 bug）


def skeleton(tpl):
    """把 ${...} 换成 ‹›，并去掉 HTML 标签，得到可见骨架"""
    s = INTERP.sub(PH, tpl)
    s = TAG.sub("", s)
    return s.strip()


def scan_upstream():
    src = open(EMBER_MJS, encoding="utf-8").read()
    lines = src.split("\n")
    offs, acc = [], 0
    for l in lines:
        offs.append(acc)
        acc += len(l) + 1

    def lineno(pos):
        lo, hi = 0, len(offs) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offs[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    out = []
    for m in SINKS.finditer(src):
        tpl = m.group("tpl")
        if "${" not in tpl:
            continue
        sk = skeleton(tpl)
        # 骨架里必须还剩下三个字母以上的英文单词，否则是纯结构
        if not re.search(r"[A-Za-z]{3,}", sk):
            continue
        out.append({"line": lineno(m.start()), "sink": m.group("sink"),
                    "tpl": tpl, "skel": sk})
    return out, src


# ---------------------------------------------------------------- 3. 判据

def classify(skel, exact, prefixed, patterns):
    """返回 (verdict, detail)"""
    # 用占位值实例化骨架去试 PATTERNS
    probes = [skel.replace(PH, x) for x in ("X", "1", "12", "Foo Bar")]
    for en, cn, tbl in prefixed:
        if skel.startswith(en + ": "):
            leaf = skel[len(en) + 2:]
            if PH in leaf:
                return "PARTIAL_leaf_interp", f"{en}: -> leaf={leaf!r} table={tbl}"
            return "PREFIX_OK", en
    for p in patterns:
        rx = js_re(p)
        for pr in probes:
            if rx.match(pr.strip()):
                return "PATTERN_OK", p
    if skel in exact:
        return "EXACT_OK", skel
    return "UNREACHABLE", ""


def main():
    exact, tables, prefixed, patterns = load_matchers()
    print(f"[matchers] EXACT={len(exact)} PREFIXED={len(prefixed)} PATTERNS={len(patterns)}")
    for n, v in tables.items():
        print(f"           {n}={len(v)}")

    sinks, src = scan_upstream()
    print(f"[upstream] ember.mjs 含插值的可见槽位模板 {len(sinks)} 条")

    buckets = Counter()
    rows = []
    for s in sinks:
        v, d = classify(s["skel"], exact, prefixed, patterns)
        buckets[v] += 1
        s["verdict"], s["detail"] = v, d
        rows.append(s)
    print("[verdicts]", dict(buckets))

    # D2: 死 EXACT
    dead = []
    for k in exact:
        whole = (f'"{k}"' in src) or (f"'{k}'" in src) or (f"`{k}`" in src)
        as_prefix = f"`{k}: ${{" in src or f"`{k} ${{" in src
        if not whole:
            dead.append({"key": k, "appears_as_tpl_prefix": as_prefix,
                         "substring_anywhere": k in src})

    outdir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(outdir, "gsm_sinks.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    with open(os.path.join(outdir, "gsm_dead_exact.json"), "w", encoding="utf-8") as f:
        json.dump(dead, f, ensure_ascii=False, indent=1)
    print(f"[D2] EXACT 里 {len(dead)}/{len(exact)} 条在 ember.mjs 中不作为完整字符串字面量出现")
    for d in dead:
        print("     ", d)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
