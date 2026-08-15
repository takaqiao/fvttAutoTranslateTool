# -*- coding: utf-8 -*-
"""
mood_selector.py —— 统计注入 PlaylistDirectory 的 #ember-mood 表单里会显示多少英文。

注入点 ember.mjs:15874-15903 `_createMoodSelector()`，由 ember.mjs:130018
`Hooks.on("renderPlaylistDirectory", …)` 触发，把 <form id="ember-mood"> 插到
`#playlists .currently-playing` 之前。
宿主 = PlaylistDirectory（classes = tab / sidebar-tab / directory / flexcol /
playlists-sidebar，见 Foundry client/applications/sidebar/sidebar-tab.mjs:26,87
与 document-directory.mjs），根元素 class 不含 "ember"、类名不以 Ember 开头，
=> ember-hardcoded-cn.mjs:453 的闸直接 return。

统计三类英文：
  a) 表单模板里的静态英文（header / data-tooltip）
  b) createSelectInput 的 blank 文案 "Ember Default"
  c) 选项文本：soundscape.label（作 optgroup）+ arrangements[].label（作 option）
"""
import re, os, json

EMB = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
src = open(EMB, encoding="utf-8").read()

# 找到所有 soundscape 定义：形如 { id: "...", type: "music"|"environment", label: "...", ... arrangements: {...} }
# 用 type: "music"/"environment" 作锚，向前找最近的 label:，向后在同一块里数 arrangements 的 label:
def balance_back(s, idx):
    """从 idx 向前找到包住它的 '{' 的位置"""
    d = 0
    for i in range(idx, max(0, idx - 60000), -1):
        if s[i] == "}": d += 1
        elif s[i] == "{":
            if d == 0: return i
            d -= 1
    return None

def balance_fwd(s, start, limit=60000):
    d = 0
    for i in range(start, min(len(s), start + limit)):
        if s[i] == "{": d += 1
        elif s[i] == "}":
            d -= 1
            if d == 0: return s[start:i+1]
    return s[start:start+limit]

soundscapes = []
for m in re.finditer(r'type:\s*"(music|environment)"', src):
    o = balance_back(src, m.start())
    if o is None: continue
    blk = balance_fwd(src, o)
    lab = re.search(r'label:\s*"([^"]+)"', blk)
    arr = re.search(r"arrangements:\s*\{", blk)
    arr_labels = []
    if arr:
        ab = balance_fwd(blk, arr.end()-1)
        arr_labels = re.findall(r'label:\s*"([^"]+)"', ab)
    soundscapes.append({"type": m.group(1), "label": lab.group(1) if lab else None,
                        "line": src[:o].count("\n")+1, "arrangements": arr_labels})

static = ["Ember Music", "Rearrange Music", "Ember Environment", "Ember Default"]
n_arr = sum(len(s["arrangements"]) for s in soundscapes)
groups = sorted({s["label"] for s in soundscapes if s["label"]})
print("静态英文串:", static, "（Ember Default 在 music/environment 两个 select 各出现一次）")
print(f"soundscape（optgroup 标签）: {len(groups)} 个 -> {groups}")
print(f"arrangement（option 文本）: {n_arr} 条")
print(f"合计会渲染出的英文文本节点: 3(静态) + 2(blank) + {len(groups)}(optgroup) + {n_arr}(option) = "
      f"{3 + 2 + len(groups) + n_arr}")
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mood_selector.json")
json.dump({"static": static, "soundscapes": soundscapes}, open(dst, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("->", dst)
