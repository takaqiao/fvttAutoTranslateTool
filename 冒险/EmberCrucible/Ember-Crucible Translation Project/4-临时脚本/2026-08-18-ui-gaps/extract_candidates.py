#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从上游 ember.mjs + templates/** 抽「会上屏的英文串」候选集。

前置自证（阳性/阴性对照）在 selftest() 里，跑主流程前先跑，不过就当场退出。
"""
import json, re, sys, os

U = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember"
MJS = os.path.join(U, "scripts", "ember.mjs")
TPL = os.path.join(U, "templates")

DISPLAY_PROPS = ("label","title","hint","tooltip","placeholder","legend","header",
                 "caption","buttonLabel","text","heading","subtitle","summary","group")
DISPLAY_ATTRS = ("data-tooltip","data-tooltip-text","title","aria-label","placeholder","alt","label")

DQ = r'"((?:[^"\\]|\\.)*)"'
SQ = r"'((?:[^'\\]|\\.)*)'"

# (1) JS 对象里的显示属性： label: "Hair Roots"
RE_PROP = re.compile(r'(?<![\w$.])(' + "|".join(DISPLAY_PROPS) + r')\s*:\s*(?:' + DQ + '|' + SQ + ')')
# (2) 模板串/字符串里的 HTML 显示属性： data-tooltip="Show Tracks"
ATTR_ALT = "|".join(a.replace("-", r"\-") for a in DISPLAY_ATTRS)
RE_ATTR = re.compile(r'(' + ATTR_ALT + r')\s*=\s*\\?"([^"\\{}<>$]*)\\?"')
RE_ATTR_SQ = re.compile(r'(' + ATTR_ALT + r")\s*=\s*'([^'\\{}<>$]*)'")
# (3) ui.notifications.xxx("...")
RE_NOTIF = re.compile(r'ui\.notifications\.\w+\(\s*(?:' + DQ + r'|`([^`]*)`|' + SQ + ')')
# (4) HTML 文本节点（模板串内 >Text<）
RE_TEXTNODE = re.compile(r'>([^<>{}$`]{2,80})<')
# (5) .textContent = "..." / .innerText = "..."
RE_TEXTCONTENT = re.compile(r'\.(?:textContent|innerText)\s*=\s*(?:' + DQ + '|' + SQ + ')')

# 「像上屏英文」的判据
RE_IDENT = re.compile(r'^[a-z][A-Za-z0-9]*$')
RE_CONST = re.compile(r'^[A-Z0-9_]+$')
RE_PATHY = re.compile(r'[/\\]|\.(?:mjs|js|json|hbs|html|css|webp|png|svg|ogg|webm|woff2?)$')
RE_HEX   = re.compile(r'^#?[0-9a-fA-F]{3,8}$')
RE_HASLETTER = re.compile(r'[A-Za-z]')
RE_CJK   = re.compile(r'[\u4e00-\u9fff]')

def onscreen_like(s):
    s = s.strip()
    if not s: return False, "empty"
    if len(s) > 120: return False, "too-long"
    if RE_CJK.search(s): return False, "cjk"
    if not RE_HASLETTER.search(s): return False, "no-letter"
    if RE_PATHY.search(s): return False, "path-like"
    if RE_HEX.match(s): return False, "hex"
    if s.startswith(("EMBER.","CRUCIBLE.","DND5E.","ember.","crucible.")): return False, "i18n-key"
    if "." in s and " " not in s and re.match(r'^[\w.$-]+$', s): return False, "dotted-id"
    words = s.split()
    if len(words) == 1:
        w = words[0]
        if RE_IDENT.match(w): return False, "identifier"
        if RE_CONST.match(w): return False, "const-id"
        if "-" in w or "_" in w: return False, "slug"
        if not w[0].isupper(): return False, "lowercase-word"
        return True, "single-capitalized-word"
    if not re.match(r"^[A-Za-z0-9 ,.'\u2019\-\u2013\u2014:;!?()/%&+#\"]+$", s):
        return False, "non-english-chars"
    if all(RE_IDENT.match(w) for w in words): return False, "identifiers"
    return True, "phrase"

def _lineno_factory(src):
    starts = [0]
    for i, ch in enumerate(src):
        if ch == "\n": starts.append(i+1)
    def lineno(pos):
        lo, hi = 0, len(starts)-1
        while lo < hi:
            mid = (lo+hi+1)//2
            if starts[mid] <= pos: lo = mid
            else: hi = mid-1
        return lo+1
    return lineno

def scan_js(path, tag="ember.mjs"):
    src = open(path, encoding="utf-8").read()
    lineno = _lineno_factory(src)
    out = []
    for m in RE_PROP.finditer(src):
        val = m.group(2) if m.group(2) is not None else m.group(3)
        out.append((val, f"{tag}|prop:{m.group(1)}", lineno(m.start())))
    for m in RE_ATTR.finditer(src):
        out.append((m.group(2), f"{tag}|attr:{m.group(1)}", lineno(m.start())))
    for m in RE_ATTR_SQ.finditer(src):
        out.append((m.group(2), f"{tag}|attr:{m.group(1)}", lineno(m.start())))
    for m in RE_NOTIF.finditer(src):
        val = next((g for g in m.groups() if g is not None), None)
        if val: out.append((val, f"{tag}|notify", lineno(m.start())))
    for m in RE_TEXTNODE.finditer(src):
        out.append((m.group(1), f"{tag}|textnode", lineno(m.start())))
    for m in RE_TEXTCONTENT.finditer(src):
        val = m.group(1) if m.group(1) is not None else m.group(2)
        out.append((val, f"{tag}|textContent", lineno(m.start())))
    return out

def scan_templates(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith((".hbs", ".html")): continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, U).replace("\\", "/")
            src = open(p, encoding="utf-8").read()
            lineno = _lineno_factory(src)
            for m in re.finditer(r'>([^<>]*)<', src):
                t = re.sub(r'\{\{[^}]*\}\}', '', m.group(1)).strip()
                if t: out.append((t, f"tpl-text|{rel}", lineno(m.start())))
            for m in RE_ATTR.finditer(src):
                out.append((m.group(2), f"tpl-attr:{m.group(1)}|{rel}", lineno(m.start())))
            for m in RE_ATTR_SQ.finditer(src):
                out.append((m.group(2), f"tpl-attr:{m.group(1)}|{rel}", lineno(m.start())))
    return out

POSITIVE = [
    "Show Tracks", "Reset All", "Play Animation", "Anatomy", "Hair Roots",
    "Hair Highlights", "Randomize Layer", "Select Destination", "Colors", "Layers",
]
NEGATIVE = [
    "layerPrevious", "toggleAnimation", "emberAncestry", "buildNext", "stanceNext",
    "renderApplicationV2", "flexcol", "ember-header",
]

def selftest(kept, raw):
    ok = True
    print("-- 抽取器前置自证 --")
    for s in POSITIVE:
        hit = s in kept
        print(("  阳性 OK   " if hit else "  阳性 MISS ") + repr(s))
        ok = ok and hit
    for s in NEGATIVE:
        hit = s in kept
        print(("  阴性 OK   " if not hit else "  阴性 LEAK ") + repr(s))
        ok = ok and (not hit)
    print("  -- 阴性第二支：标识符即便被原始抽取捞到，也必须被 onscreen_like 挡下 --")
    n2 = 0
    for s in NEGATIVE:
        if s in raw:
            n2 += 1
            good, why = onscreen_like(s)
            print(("    OK   " if not good else "    LEAK ") + repr(s) + " -> " + why)
            ok = ok and (not good)
    print(f"    （原始抽取里实际出现的阴性样本 {n2}/{len(NEGATIVE)} 个）")
    return ok

if __name__ == "__main__":
    allrec = scan_js(MJS) + scan_templates(TPL)
    raw = {}
    for s, src, ln in allrec:
        raw.setdefault(s.strip(), []).append(f"{src}:{ln}")
    kept = {}
    for s, locs in raw.items():
        good, why = onscreen_like(s)
        if good: kept[s] = locs
    if not selftest(set(kept), set(raw)):
        print("\n[STOP] 前置自证不过，中止。"); sys.exit(1)
    print(f"\n原始抽取 {len(raw)} 个不同串 -> 过滤后候选 {len(kept)} 个")
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidates.json")
    json.dump({k: v[:6] for k, v in sorted(kept.items())}, open(outp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("写出 " + outp)
