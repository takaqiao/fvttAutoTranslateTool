# -*- coding: utf-8 -*-
"""
dialog_content2.py —— 补 dialog_strings.py 的漏：content 不是内联字面量、
而是先赋给局部变量 `content` 再传进 DialogV2 的情况。

做法：对每个 DialogV2.{prompt,confirm,wait,input} 调用点，若其实参里 content 是裸标识符，
就向上回溯 40 行找 `content = ` / `content += ` 的英文字面量（含模板串）。
假阳性：回溯窗口可能跨到别的函数；输出带行号供人工核对。
"""
import re, os, json

EMB = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
src = open(EMB, encoding="utf-8").read()
lines = src.splitlines()
LIT = re.compile(r'content\s*(?:\+?=)\s*(?:"((?:[^"\\]|\\.)*)"|`((?:[^`\\]|\\.)*)`)', re.S)

rows = []
for m in re.finditer(r"DialogV2[$\w]*\.(?:prompt|confirm|wait|input)\s*\(", src):
    ln = src[:m.start()].count("\n") + 1
    tail = "\n".join(lines[ln-1:ln+20])
    if not re.search(r"(^|[\s{,])content\s*[,}\n]", tail):
        continue                     # content 是内联字面量或压根没有
    back = "\n".join(lines[max(0, ln-45):ln-1])
    hits = [(a or b) for a, b in LIT.findall(back)]
    eng = [h for h in hits if re.search(r"[A-Za-z]{4}", h) and "_loc(" not in h]
    if eng:
        rows.append({"dialog_line": ln, "content_literals": [e.replace("\n", " ")[:220] for e in eng]})

print(f"变量式 content 且含英文字面量的 DialogV2 调用点: {len(rows)}")
for r in rows:
    print(" @", r["dialog_line"])
    for e in r["content_literals"]:
        print("     ", e)
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dialog_content2.json")
json.dump(rows, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("->", dst)
