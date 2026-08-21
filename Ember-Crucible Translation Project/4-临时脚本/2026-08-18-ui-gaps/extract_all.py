#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缺口全集的抽取器（第二版，宽召回）。

第一版按「显示属性 + HTML 属性」抽，实测**漏召回**：
  buildGroup("Forwards", …) / .textContent = cond ? "…" : "…" / prompt: "…"
这三种形态都够不着，而它们全是真上屏的串。
=> 第二版改抽 **ember.mjs 里全部字符串字面量（含模板串的静态段）+ templates/** 全部文本与属性**，
   再用 onscreen_like 过滤。宁可噪声大，也不许漏。

前置自证（两件都断言）：
  (A) 切对条数：原始字面量条数 / 过滤后条数如实报出，且过滤只减不增。
  (B) 改对地方：
      · 阳性 —— 32 条**已知真会上屏**的串（含第一版漏掉的那三种形态）必须全部在候选集里；
      · 阴性 —— 已知的纯代码标识符 / 资源路径 / i18n 键必须全部**不在**候选集里，
        且其中至少 8 条必须是**原始抽取里真的出现过、被过滤器挡下的**（不许空转对照）。
"""
import json, re, sys, os, io, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

U = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember"
MJS = os.path.join(U, "scripts", "ember.mjs")
TPL = os.path.join(U, "templates")
HERE = os.path.dirname(os.path.abspath(__file__))

RE_IDENT = re.compile(r'^[a-z][A-Za-z0-9]*$')
RE_CONST = re.compile(r'^[A-Z0-9_]+$')
RE_PATHY = re.compile(r'[/\\]|\.(?:mjs|js|json|hbs|html|css|webp|png|svg|ogg|webm|woff2?)$')
RE_HEX   = re.compile(r'^#?[0-9a-fA-F]{3,8}$')
RE_HASLETTER = re.compile(r'[A-Za-z]')
RE_CJK   = re.compile(r'[\u4e00-\u9fff]')
RE_CODEY = re.compile(r'[<>=]{1,3}|\{\{|\}\}|\breturn\b|\bfunction\b|\bconst\b|=>|\|\||&&|\+\+|;\s*$')

def onscreen_like(s):
    s = s.strip()
    if not s: return False, "empty"
    if len(s) > 200: return False, "too-long"
    if RE_CJK.search(s): return False, "cjk"
    if not RE_HASLETTER.search(s): return False, "no-letter"
    if RE_PATHY.search(s): return False, "path-like"
    if RE_HEX.match(s): return False, "hex"
    if s.startswith(("EMBER.", "CRUCIBLE.", "DND5E.", "ember.", "crucible.", "CONTROLS.",
                     "TOKEN.", "ACTION.", "SETTINGS.", "KEYBINDINGS.", "DOCUMENT.",
                     "FILES.", "SCENE.", "JOURNAL.", "TYPES.", "ACTOR.", "ITEM.", "EFFECT.")):
        return False, "i18n-key"
    if RE_CODEY.search(s): return False, "codey"
    if "." in s and " " not in s and re.match(r'^[\w.$-]+$', s): return False, "dotted-id"
    words = s.split()
    if len(words) == 1:
        w = words[0].rstrip(".,!?:;")
        if not w: return False, "punct-only"
        if RE_IDENT.match(w): return False, "identifier"
        if RE_CONST.match(w): return False, "const-id"
        if "-" in w or "_" in w:
            parts = re.split(r'[-_]', w)
            if not all(p[:1].isupper() for p in parts if p): return False, "slug"
            return True, "hyphenated-title"
        if not w[0].isupper(): return False, "lowercase-word"
        return True, "single-capitalized-word"
    if not re.match(r"^[A-Za-z0-9 ,.'\u2019\-\u2013\u2014:;!?()/%&+#\"\u00b0]+$", s):
        return False, "non-english-chars"
    if all(RE_IDENT.match(w) for w in words): return False, "identifiers"
    if not any(w[:1].isupper() for w in words): return False, "no-capital"
    return True, "phrase"

# ---- 字面量扫描：手写小状态机，正确处理 '' "" `` 与转义、注释 ----
def scan_literals(src):
    """返回 [(literal_text, lineno)]；模板串按 ${…} 切成静态段分别产出。"""
    out = []
    i, n = 0, len(src)
    line = 1
    while i < n:
        ch = src[i]
        if ch == "\n":
            line += 1; i += 1; continue
        # 行注释
        if ch == "/" and i+1 < n and src[i+1] == "/":
            while i < n and src[i] != "\n": i += 1
            continue
        # 块注释
        if ch == "/" and i+1 < n and src[i+1] == "*":
            j = src.find("*/", i+2)
            if j < 0: break
            line += src.count("\n", i, j); i = j+2; continue
        if ch in ("'", '"'):
            q = ch; j = i+1; buf = []
            while j < n:
                if src[j] == "\\":
                    if j+1 < n and src[j+1] == "\n": line += 1   # 反斜杠续行
                    buf.append(src[j:j+2]); j += 2; continue
                if src[j] == q: break
                if src[j] == "\n": break          # 未闭合，放弃
                buf.append(src[j]); j += 1
            if j < n and src[j] == q:
                out.append(("".join(buf), line))
                i = j+1; continue
            i += 1; continue
        if ch == "`":
            # ⚠ 每个**静态段**记自己的起始行，不是整条模板串的起始行 ——
            #   跨行模板串里后面的段会差好几十行，行号一错，后面按行号读上下文分档就全歪了。
            j = i+1; buf = []; seg_start = line
            while j < n:
                if src[j] == "\\":
                    if j+1 < n and src[j+1] == "\n":            # 反斜杠续行：也要跨行
                        out.append(("".join(buf), seg_start)); buf = []
                        line += 1; j += 2; seg_start = line; continue
                    buf.append(src[j:j+2]); j += 2; continue
                if src[j] == "`": break
                if src[j] == "$" and j+1 < n and src[j+1] == "{":
                    out.append(("".join(buf), seg_start)); buf = []
                    k = j+2; d = 1
                    while k < n and d:
                        if src[k] == "{": d += 1
                        elif src[k] == "}": d -= 1
                        elif src[k] == "\n": line += 1
                        k += 1
                    j = k; seg_start = line; continue
                if src[j] == "\n":
                    # 段内换行：把已积累的部分按当前段起始行收掉，新的一行开一段
                    out.append(("".join(buf), seg_start)); buf = []
                    line += 1; seg_start = line; j += 1; continue
                buf.append(src[j]); j += 1
            out.append(("".join(buf), seg_start))
            i = j+1; continue
        i += 1
    return out

def unescape(s):
    return (s.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "")
             .replace('\\"', '"').replace("\\'", "'").replace("\\`", "`").replace("\\\\", "\\"))

TAG_RE = re.compile(r'</?[a-zA-Z][^>]*>')

def literal_pieces(lit):
    """一条字面量可能是一整段 HTML；按标签切开，标签内的显示属性单独产出。"""
    pieces = []
    # ⚠ 属性这一路**无条件**跑：模板串按行切段之后，`<button` 与它的 aria-label 常常不在同一段
    #   （ember.mjs:129586 那个 `Add Reinforcements` 就是），要求段里同时有 `<` 和 `>` 会整类漏掉。
    for m in re.finditer(
            r'(data-tooltip|data-tooltip-text|title|aria-label|placeholder|alt|label)\s*=\s*"([^"]*)"', lit):
        pieces.append(m.group(2))
    if "<" in lit or ">" in lit:
        pieces.extend(TAG_RE.split(lit))
    else:
        pieces.append(lit)
    return pieces

def scan_mjs():
    src = open(MJS, encoding="utf-8").read()
    raw = collections.defaultdict(list)
    n_lit = 0
    for lit, ln in scan_literals(src):
        n_lit += 1
        for piece in literal_pieces(unescape(lit)):
            for sub in re.split(r'[\n\r]+', piece):
                sub = sub.strip()
                if sub: raw[sub].append(f"ember.mjs:{ln}")
    return raw, n_lit

def scan_templates():
    raw = collections.defaultdict(list)
    for dp, _, fns in os.walk(TPL):
        for fn in fns:
            if not fn.endswith((".hbs", ".html")): continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, U).replace("\\", "/")
            src = open(p, encoding="utf-8").read()
            def lineat(pos): return src.count("\n", 0, pos) + 1
            for m in re.finditer(r'>([^<>]*)<', src):
                t = re.sub(r'\{\{[^}]*\}\}', '', m.group(1)).strip()
                if t: raw[t].append(f"{rel}:{lineat(m.start())}")
            for m in re.finditer(r'(data-tooltip|data-tooltip-text|title|aria-label|placeholder|alt|label)\s*=\s*"([^"{}]*)"', src):
                t = m.group(2).strip()
                if t: raw[t].append(f"{rel}:{lineat(m.start())}")
    return raw

# ---------- 前置自证的对照组 ----------
POSITIVE = [
    # 第一版就抽到的
    "Show Tracks", "Reset All", "Play Animation", "Anatomy", "Hair Roots", "Hair Highlights",
    "Hair Base", "Hair Glow", "Hair Sparkle", "Randomize Layer", "Select Destination",
    "Colors", "Layers", "Mine Cart Destination", "Elevator Destination",
    "Ember: Teleport Destination", "Loading Zone", "Ooze Farm", "Southern Ore Pit",
    # ↓ 第一版**漏掉**的三种形态，正是本版要接住的
    "Forwards", "Backwards", "Unreachable",                       # 函数实参
    "Activate this mine cart with no passenger?",                 # 三元里的 textContent
    "Choose a destination.",                                      # prompt: 属性
    "No destinations are currently reachable. Adjust the track levers and try again.",
    "Close", "Confirm", "Metal Base", "Metal Accent", "Skin Base", "Fabric Accent",
    # 模板串按行切段后，标签与它的属性常常分居两行；这两条钉住那条路
    "Add Reinforcements",                                         # ember.mjs:129586 aria-label
    "Add Actor",                                                  # ember.mjs:123276 aria-label
    "Reinforcements", "Spawn Actors",                             # 同两处的 <legend> 文本
]
NEGATIVE_IDENT = [
    "layerPrevious", "toggleAnimation", "emberAncestry", "buildNext", "stanceNext",
    "renderApplicationV2", "flexcol", "ember-header", "hair1", "hair2", "metal1",
    "anatomy", "equipment", "destination", "blueCart", "forcedMovement",
]

def selftest_lineno():
    """[A2] 行号自证：每条**单行**字面量都必须真的出现在它自报的那一行上。

    这条不是装饰 —— 第一版跨行模板串全部记的是整条串的起始行，36053 条行号是错的，
    而后面的分档、写进表里的行号注释全靠它。0 条不符才算过。
    """
    src = open(MJS, encoding="utf-8").read()
    L = src.split("\n")
    bad, tot, worst = 0, 0, []
    for lit, ln in scan_literals(src):
        if not lit or "\n" in lit: continue
        tot += 1
        if 0 < ln <= len(L) and lit in L[ln - 1]: continue
        bad += 1
        if len(worst) < 5: worst.append((lit[:40], ln))
    print(f"[A2 行号] 单行字面量 {tot} 条，自报行号对不上的 {bad} 条 -> " + ("OK" if bad == 0 else "FAIL"))
    for w in worst: print("      ", w)
    return bad == 0


def selftest(kept, raw, n_lit):
    ok = True
    print("== 抽取器前置自证 ==")
    print(f"[A] ember.mjs 原始字面量 {n_lit} 条 -> 去重切片 {len(raw)} 个不同串 -> 过滤后候选 {len(kept)} 个")
    ok = ok and selftest_lineno()
    if not (len(kept) <= len(raw)):
        print("  [A] FAIL 过滤后反而变多"); ok = False
    else:
        print("  [A] OK  过滤只减不增")
    miss = [s for s in POSITIVE if s not in kept]
    print(f"[B阳性] {len(POSITIVE)-len(miss)}/{len(POSITIVE)} 命中" + ("" if not miss else f"  漏：{miss}"))
    ok = ok and not miss
    leak = [s for s in NEGATIVE_IDENT if s in kept]
    print(f"[B阴性] {len(NEGATIVE_IDENT)-len(leak)}/{len(NEGATIVE_IDENT)} 挡下" + ("" if not leak else f"  漏出：{leak}"))
    ok = ok and not leak
    seen_in_raw = [s for s in NEGATIVE_IDENT if s in raw]
    print(f"[B阴性-非空转] 其中 {len(seen_in_raw)} 条**确实出现在原始抽取里**、被过滤器挡下：{seen_in_raw[:12]}")
    if len(seen_in_raw) < 8:
        print("  [B阴性-非空转] FAIL 对照组空转（原始抽取里根本没出现，挡不挡都一样）"); ok = False
    else:
        print("  [B阴性-非空转] OK")
    return ok

if __name__ == "__main__":
    rawm, n_lit = scan_mjs()
    rawt = scan_templates()
    raw = collections.defaultdict(list)
    for d in (rawm, rawt):
        for k, v in d.items(): raw[k].extend(v)
    kept, dropped = {}, collections.Counter()
    for s, locs in raw.items():
        good, why = onscreen_like(s)
        if good: kept[s] = locs
        else: dropped[why] += 1
    if not selftest(set(kept), set(raw), n_lit):
        print("\n[STOP] 前置自证不过，中止。"); sys.exit(1)
    print("\n过滤掉的理由分布：", dropped.most_common())
    json.dump({k: v[:8] for k, v in sorted(kept.items())},
              open(os.path.join(HERE, "candidates.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n写出 candidates.json（{len(kept)} 条）")
