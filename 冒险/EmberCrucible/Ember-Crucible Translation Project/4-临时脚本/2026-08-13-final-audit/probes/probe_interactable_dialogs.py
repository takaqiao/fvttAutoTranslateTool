# -*- coding: utf-8 -*-
"""
probe_interactable_dialogs.py

场景交互物（EmberInteractable 家族）的对话框配置：`static DEFAULT_CONFIG = {... dialog: {...} ...}`。
这些配置最终喂给 `_displayDialog` → `DialogV2.wait/prompt`（ember.mjs:62766-62771）。

闸的处理（ember-hardcoded-cn.mjs:453-465）：根元素 class 只有 "dialog"、类名 "DialogV2"
→ 走例外分支 → **只译 `.window-title`**。于是 `description` / `content` / 每个 `buttons[].label`
（以及 62794 的兜底 `ok: {label: "Interact"}`）永远是英文。

本脚本枚举全部 `dialog: {` 配置块，抽出 title / content / description / 各按钮 label，
并标出哪些串已在 EXACT 表里（＝已被译到，说明是标题）、哪些不在（＝结构性不可达）。
只读上游源码 + 只读本库 .mjs。
"""
import json
import re
from pathlib import Path

EMBER = Path(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs")
PATCH = Path(r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
             r"\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs")
OUT = Path(__file__).with_name("interactable_dialogs.json")

T = EMBER.read_text(encoding="utf-8", errors="replace")
P = PATCH.read_text(encoding="utf-8", errors="replace")

# 本库 EXACT 表里的英文键（够粗但足够判「有没有被译到」）
EXACT_KEYS = set(re.findall(r'^\s*"([^"]+)":\s*"[^"]*"', P, re.M))


def block(text, start):
    """从 '{' 起取平衡块"""
    i, d = start, 0
    while i < len(text):
        if text[i] == '"':
            i += 1
            while i < len(text) and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif text[i] == "{":
            d += 1
        elif text[i] == "}":
            d -= 1
            if d == 0:
                return text[start:i + 1]
        i += 1
    return text[start:]


rows = []
for m in re.finditer(r"\bdialog:\s*(\{|foundry\.utils\.mergeObject\()", T):
    st = T.index("{", m.end() - 1)
    b = block(T, st)
    ln = T.count("\n", 0, m.start()) + 1
    title = re.search(r'title:\s*"([^"]+)"', b)
    desc = re.search(r'description:\s*"([^"]+)"', b)
    cont = re.search(r'content:\s*"([^"]+)"', b)
    labels = re.findall(r'label:\s*"([^"]+)"', b)
    if not (title or desc or cont or labels):
        continue
    strs = []
    if title:
        strs.append(("title", title.group(1)))
    if desc:
        strs.append(("description", desc.group(1)))
    if cont:
        strs.append(("content", re.sub(r"<[^>]+>", "", cont.group(1)).strip()))
    for lb in labels:
        strs.append(("button", lb))
    rows.append({
        "line": ln,
        "strings": [{"slot": s, "en": v,
                     "in_EXACT": v in EXACT_KEYS,
                     # 只有 title 会被闸摸到；其它槽位结构性不可达
                     "gate_can_reach": s == "title"}
                    for s, v in strs]
    })

flat = [(r["line"], s["slot"], s["en"], s["in_EXACT"]) for r in rows for s in r["strings"]]
unreach = [f for f in flat if f[1] != "title"]
titles = [f for f in flat if f[1] == "title"]

OUT.write_text(json.dumps({"blocks": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"dialog 配置块          : {len(rows)}")
print(f"英文串合计             : {len(flat)}")
print(f"  其中 title（闸可达）  : {len(titles)}  已在 EXACT: {sum(1 for f in titles if f[3])}")
print(f"  非 title（结构不可达）: {len(unreach)}  已在 EXACT: {sum(1 for f in unreach if f[3])}")
print(f"  不同的英文串（非title）: {len({f[2] for f in unreach})}")
print("->", OUT)
for f in unreach:
    print(f"  ember.mjs:{f[0]:>7}  {f[1]:<11} {f[2]!r}")
