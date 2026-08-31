# -*- coding: utf-8 -*-
"""
把每个「互动面板」站点（dialog 配置块 / _configureDialog 方法体）连同它的
窗口标题、按钮 label、正文文本节点一起列出来，并标出每条走哪条通道。

依赖 panel_tn_classified.json（文本节点级）与 panel_classified.json（原串级）。

前置自证：
  A 切对条数 —— 站点数 == extract_panels.py 报的 dialog-config 21 + configureDialog 12 = 33
  B 切对对象 —— 已知真值：`Mine Cart Destination` 不在这 33 个站点里（它是 DialogV2.input 直调，
                不走 dialog 配置）；`Tar Pit` 必须出现在某个 _configureDialog 站点里。
"""
import json, re, io, sys
sys.path.insert(0, ".")
from extract_panels import src, lineno, match_block, BS, BT   # 复用同一份切法

regions = []
for m in re.finditer(r"\bdialog:\s*\{", src):
    b = src.index("{", m.start())
    regions.append(("dialog-config", b, match_block(src, b, "{", "}") + 1))
for m in re.finditer(r"(?:async\s+)?_configureDialog\s*\([^)]*\)\s*\{", src):
    b = src.index("{", m.end() - 1)
    regions.append(("configureDialog", b, match_block(src, b, "{", "}") + 1))
regions.sort(key=lambda r: r[1])

assert len(regions) == 33, len(regions)
print("PRECHECK-A OK  站点 33 个")

Q = r'"((?:[^"' + BS + BS + r']|' + BS + BS + r'.)*)"'
TITLE = re.compile(r'title\s*[:=]\s*' + Q)
LABEL = re.compile(r'(?:label|aria-label)\s*[:=]\s*' + Q)
CONTENT = re.compile(r'content\s*[:=]\s*' + Q)

tn = json.load(io.open("panel_tn_classified.json", encoding="utf-8"))
raw = json.load(io.open("panel_classified.json", encoding="utf-8"))


from to_textnodes import textnodes


def ch(s):
    """带标签的串按**文本节点**判（translateText 比的是整个文本节点，不是整串 HTML）。"""
    if "<" in s and ">" in s:
        parts = textnodes(s)
        if not parts:
            return "n/a"
        chans = [tn[p]["channel"] if p in tn else "?" for p in parts]
        return "GAP" if "GAP" in chans else ("+".join(sorted(set(chans))))
    if s in raw:
        return raw[s]["channel"]
    if s in tn:
        return tn[s]["channel"]
    return "?"


rows = []
seen_tarpit = False
for kind, s, e in regions:
    seg = src[s:e]
    ln = lineno(s)
    # 类名：往回找最近的 `class X`
    head = src[:s]
    cm = None
    for m in re.finditer(r"\bclass\s+([A-Za-z0-9_$]+)", head):
        cm = m.group(1)
    titles = TITLE.findall(seg)
    labels = LABEL.findall(seg)
    contents = CONTENT.findall(seg)
    if "Tar Pit" in titles:
        seen_tarpit = True
    rows.append({
        "kind": kind, "line": ln, "class": cm,
        "titles": [(t, ch(t)) for t in titles],
        "buttons": [(l, ch(l)) for l in labels],
        "content": [(c, ch(c)) for c in contents],
    })

assert seen_tarpit, "已知真值 Tar Pit 没出现在任何 _configureDialog 站点里"
assert "Mine Cart Destination" not in [t for r in rows for t, _c in r["titles"]], \
    "已知反例 Mine Cart Destination 不该出现在这 33 个站点里"
print("PRECHECK-B OK  Tar Pit 在站点里 / Mine Cart Destination 不在")

io.open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(rows, ensure_ascii=False, indent=1))
print()
for r in rows:
    if not (r["titles"] or r["buttons"] or r["content"]):
        continue
    print("ember.mjs:%-6d %-14s %s" % (r["line"], r["kind"], r["class"]))
    for t, c in r["titles"]:
        print("    title   [%-6s] %s" % (c, t))
    for l, c in r["buttons"]:
        print("    button  [%-6s] %s" % (c, l))
    for x, c in r["content"]:
        print("    content [%-6s] %s" % (c, x[:110]))
