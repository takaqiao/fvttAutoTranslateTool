# -*- coding: utf-8 -*-
"""crucible 侧有没有「字面英文 + 无运行时替换层」的问题？
crucible-cn 仓库里只有 babele-register.js（babele）+ lang/cn.json（i18n），
**没有任何 ember-hardcoded-cn 那样的运行时 DOM 替换脚本**。
所以只要 crucible 本体存在字面英文界面串，就一定露白。
本脚本量化：crucible 的 ui.notifications / DialogV2 content / hbs 里字面 vs i18n key 的比例。只读。
"""
import os
import re

CRUCIBLE = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible"
SRC = os.path.join(CRUCIBLE, "crucible-compiled.mjs")
t = open(SRC, encoding="utf-8", errors="replace").read()

def is_key(s):
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(\.[A-Za-z0-9_]+)+", s))

# ui.notifications.X( <first arg> )
lit, key, var = [], [], 0
for m in re.finditer(r"ui\.notifications\.(info|warn|error|notify)\s*\(", t):
    seg = t[m.end(): m.end() + 300]
    sm = re.match(r"""\s*(['"`])((?:\\.|(?!\1)[^\\])*)\1""", seg)
    if not sm:
        var += 1
        continue
    v = sm.group(2)
    if is_key(v) or "_loc(" in v:
        key.append(v)
    else:
        lit.append((t.count("\n", 0, m.start()) + 1, v[:110]))
print(f"[crucible] ui.notifications 总数 {len(lit)+len(key)+var}  字面英文 {len(lit)}  i18n key {len(key)}  变量/表达式 {var}")
for x in lit[:40]:
    print("   LIT", x)

# _loc(`...`) 模板串里的 key 也算 i18n
# DialogV2 content 字面
n = 0
for m in re.finditer(r"DialogV2\$?\d*\.(prompt|confirm|wait|input|query)\s*\(", t):
    seg = t[m.start(): m.start() + 1500]
    cm = re.search(r"content\s*:\s*(['\"`])((?:\\.|(?!\1)[^\\])*)\1", seg)
    if not cm:
        continue
    vis = re.sub(r"<[^>]*>", " ", cm.group(2))
    vis = re.sub(r"\$\{[^}]*\}", " ", vis)
    words = [w for w in re.split(r"\s{2,}|\n", vis) if len(re.findall(r"[A-Za-z]{3,}", w)) >= 2]
    if words:
        n += 1
        print("   DLG", t.count("\n", 0, m.start()) + 1, words[:3])
print(f"[crucible] DialogV2 content 含字面英文的调用点: {n}")
