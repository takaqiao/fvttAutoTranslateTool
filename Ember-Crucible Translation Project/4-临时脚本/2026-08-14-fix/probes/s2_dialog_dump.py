#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2 探针：把 ember.mjs 里所有 `dialog: {...}` 声明块与 DialogV2 调用块的
window.title / buttons[].label / ok.label / content 字面量抠出来，用于核对
ember-hardcoded-cn.mjs 的 EXACT / DIALOG 表覆盖率。只读，不写任何上游文件。
"""
import re, sys, json, io

SRC = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"

src = io.open(SRC, encoding="utf-8").read()


def block(text, start):
    """start 指向 '{'，返回配平后的 (块文本, 结束下标)"""
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c in "\"'`":
            q = c
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == q:
                    break
                i += 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1], i + 1
        i += 1
    return text[start:], n


def lineno(idx):
    return src.count("\n", 0, idx) + 1


STR = r'(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'|`((?:[^`\\]|\\.)*)`)'

out = []
for m in re.finditer(r"\bdialog:\s*\{", src):
    b, _ = block(src, m.end() - 1)
    out.append(("dialog-config", lineno(m.start()), b))
for m in re.finditer(r"DialogV2(?:\$\d+)?\.(?:wait|prompt|confirm|input)\(\{", src):
    b, _ = block(src, m.end() - 1)
    out.append(("dialogv2-call", lineno(m.start()), b))

titles, labels, contents = {}, {}, {}
for kind, ln, b in out:
    for mm in re.finditer(r"\btitle:\s*" + STR, b):
        v = next(x for x in mm.groups() if x is not None)
        titles.setdefault(v, []).append(ln)
    for mm in re.finditer(r"\blabel:\s*" + STR, b):
        v = next(x for x in mm.groups() if x is not None)
        labels.setdefault(v, []).append(ln)
    for mm in re.finditer(r"\bcontent:\s*" + STR, b):
        v = next(x for x in mm.groups() if x is not None)
        contents.setdefault(v, []).append(ln)

res = {"titles": titles, "labels": labels, "contents": contents}
sys.stdout.write(json.dumps(res, ensure_ascii=False, indent=1))
