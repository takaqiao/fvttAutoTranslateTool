# -*- coding: utf-8 -*-
"""
p13b_dialog_pairs.py —— 判据的第二刀：DialogV2「标题被翻、正文/按钮翻不到」

替换层对非 Ember 应用只留了一个例外口子（ember-hardcoded-cn.mjs:459-465）：
    if ( id !== "DialogV2" && !/(^|\\s)dialog(\\s|$)/.test(cls) ) return;
    const title = root.querySelector(".window-title");
    ... translateText(title.textContent) ... return;   ← 只翻标题，随后 return
也就是说：对话框这一面只被翻了标题一行，正文（.dialog-content / form）与
自定义按钮（ok.label / buttons[].label）永远不会被 translateNode 走到。

本脚本把 ember.mjs / crucible-compiled.mjs 里的 DialogV2 调用切成块，抽出
window.title / content / ok.label / buttons label，与插件 EXACT 表求交：
标题命中 EXACT（＝会被翻成中文）而正文或按钮仍是英文的，就是「半中半英」实例。

假阳性模式：
  - content 里可能全是 ${_loc(...)} 插值（走 lang，已是中文）→ 脚本标 LOC-ONLY，不计入。
  - content 可能是变量名（content, 或 this._buildResults(...)）→ 标 DYNAMIC，人工看。
  - 少数 DialogV2 由 Ember 自己的子类渲染（类名以 Ember 开头）→ 那种走的是 /^Ember/ 闸的
    正常路径，会被 translateNode 全量遍历，脚本用 classes:["ember-…"] 粗略识别并标注。
只读。
"""
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
FVTT = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
SRC = {
    "ember.mjs": os.path.join(FVTT, r"modules\ember\scripts\ember.mjs"),
    "crucible-compiled.mjs": os.path.join(FVTT, r"systems\crucible\crucible-compiled.mjs"),
}
HC = os.path.join(ROOT, r"1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs")


def read(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


# 解析插件 EXACT 表的键
hc = read(HC)
exact_block = hc.split("const EXACT = {", 1)[1].split("\n};", 1)[0]
EXACT = set(re.findall(r'"([^"]+)":\s*"', exact_block))

CALL = re.compile(r"DialogV2\.(confirm|prompt|wait)\(\{")
ENGLISH_SENT = re.compile(r"[A-Za-z][A-Za-z' ,.\-]{12,}")
CJK = re.compile(r"[\u4e00-\u9fff]")


def block_at(t, i):
    """从 '{' 开始做括号配平，取出调用对象字面量。"""
    depth = 0
    for j in range(i, min(i + 6000, len(t))):
        c = t[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return t[i:j + 1]
    return t[i:i + 2000]


rows = []
for label, path in SRC.items():
    t = read(path)
    for m in CALL.finditer(t):
        start = t.index("{", m.end() - 1)
        blk = block_at(t, start)
        ln = t.count("\n", 0, m.start()) + 1
        tm = re.search(r"title:\s*(`[^`]*`|\"[^\"]*\"|'[^']*'|[A-Za-z_$][\w.$]*\([^)]*\))", blk)
        title = tm.group(1).strip("`\"'") if tm else None
        cm = re.search(r"content:\s*(`(?:[^`\\]|\\.)*`|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|[^,\n]+)", blk, re.S)
        content = cm.group(1) if cm else ""
        oks = re.findall(r"label:\s*[\"'`]([^\"'`]+)[\"'`]", blk)
        # content 里剥掉插值后剩下的英文
        stripped = re.sub(r"\$\{[^}]*\}", "", content)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        eng = [s.strip() for s in ENGLISH_SENT.findall(stripped) if len(s.strip()) > 12]
        rows.append({
            "file": label, "line": ln, "kind": m.group(1),
            "title": title,
            "title_in_EXACT": bool(title and title in EXACT),
            "content_english": eng[:4],
            "button_labels": [b for b in oks if not CJK.search(b)][:6],
            "ember_class": bool(re.search(r'classes:\s*\[[^\]]*ember', blk)),
        })

hits = [r for r in rows if r["title_in_EXACT"] and (r["content_english"] or r["button_labels"])]
print(f"DialogV2 调用总数 {len(rows)}；标题会被翻成中文而正文/按钮仍英文的 {len(hits)}")
for r in hits:
    print(json.dumps(r, ensure_ascii=False))
print("\n--- 标题不在 EXACT（整框全英，另一类）---")
for r in rows:
    if not r["title_in_EXACT"] and r["title"] and not r["title"].startswith("EMBER.") \
       and not CJK.search(r["title"]) and "_loc" not in (r["title"] or ""):
        print(f'{r["file"]}:{r["line"]}  title={r["title"]!r}  btn={r["button_labels"]}')
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p13b_dialog_pairs.json")
json.dump(rows, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n->", dst)
