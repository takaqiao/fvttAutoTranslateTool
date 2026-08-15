#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
probe: upstream_surface_gap
===========================

同一类缺陷的第二个形态。

已确认实例是「把 Document 的 flag API 用在 package 对象上」——
本质是**本模块对上游对象的形状/清单做了一个假设，假设错了却没有任何信号**。

这个探针查它的姊妹形态：
  ember 往 `crucible.CONFIG.*` / `CONFIG.*` 里塞了**裸英文 label**（不是 i18n key），
  而 `ember-hardcoded-cn.mjs::patchCrucibleConfig` 只写死遍历了 `["languages","knowledge"]` 两个键。
  同一层里被 ember 污染过、却不在这张清单里的键 = **永久漏译且无任何日志**
  （patchCrucibleConfig 仍然会打印「已改写 N 条」，N>0，看起来是成功的）。

做法：
  1. 从 ember.mjs 里抓出所有写进 `crucible.CONFIG.X` / `CONFIG.X` 的赋值
     （`Object.assign(crucible.CONFIG.X, {...})` 与 `crucible.CONFIG.X.k = {...}`）；
  2. 抓出其中 label/name 值形如**裸英文**（不含点号命名空间、首字母大写、有空格或纯词）
     的条目 —— i18n key 形如 `LANGUAGES.Common`，会被排除；
  3. 与 `ember-hardcoded-cn.mjs` 里 patchCrucibleConfig 实际遍历的键清单对表，
     报出「ember 污染了但补丁没覆盖」的键。

假阳性模式：
  - 有些 label 虽是裸英文，但 crucible 的 preLocalizeConfig 之后并不渲染到界面上
    （纯内部用）。**必须逐条回到 crucible 源码确认渲染点**，本探针不做这一步。
  - 正则只认单行 `label: "..."`，跨行/拼接写法会漏。
  - 只扫 ember.mjs，不扫 ember 的 .hbs 模板。

只读，不写库。
"""
import os
import re
import json

EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
PATCH = (r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
         r"\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "upstream_surface_gap.candidates.json")

# i18n key: 至少一个点号，且点号两侧都是标识符 —— 例如 LANGUAGES.Common / EMBER.CALENDAR.X
I18N_KEY = re.compile(r"^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)+$")
# 裸英文：以大写字母开头，只含字母/空格/撇号/连字符
BARE_EN = re.compile(r"^[A-Z][A-Za-z'’\- ]*$")

ASSIGN_TARGET = re.compile(
    r"(?:Object\.assign\(\s*((?:crucible\.)?CONFIG(?:\.[A-Za-z_$][\w$]*)+)\s*,"
    r"|((?:crucible\.)?CONFIG(?:\.[A-Za-z_$][\w$]*)+)\s*=\s*\{)"
)
LABEL_LINE = re.compile(r"""\b(label|name|abbreviation|group|tooltip)\s*:\s*["']([^"']+)["']""")


def brace_span(src, start):
    """从 start 处第一个 '{' 起，返回其配对 '}' 的索引（粗糙但够用）。"""
    i = src.find("{", start)
    if i < 0:
        return None, None
    depth = 0
    for j in range(i, min(len(src), i + 20000)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return i, j
    return i, None


def main():
    src = open(EMBER, encoding="utf-8", errors="replace").read()
    hits = []
    for m in ASSIGN_TARGET.finditer(src):
        target = m.group(1) or m.group(2)
        i, j = brace_span(src, m.start())
        if i is None or j is None:
            continue
        block = src[i:j + 1]
        line = src[:m.start()].count("\n") + 1
        bare = []
        for lm in LABEL_LINE.finditer(block):
            key, val = lm.group(1), lm.group(2)
            if I18N_KEY.match(val):
                continue          # i18n key，走 lang 通道，不归本补丁管
            if not BARE_EN.match(val):
                continue          # 路径/小写 id/带数字的，不是给人看的
            bare.append({"field": key, "text": val})
        if bare:
            hits.append({"target": target, "ember_mjs_line": line,
                         "bare_english": bare})

    patch_src = open(PATCH, encoding="utf-8").read()
    m = re.search(r"for \(const \[key, table\] of (\[\[.*?\]\])\)", patch_src, re.S)
    covered = re.findall(r'\["([a-zA-Z_$][\w$]*)"', m.group(1)) if m else []

    print("patchCrucibleConfig 实际遍历的 crucible.CONFIG 键 :", covered)
    print()
    print("%-42s %-9s %s" % ("ember 写入的上游 CONFIG 路径", "ember行", "裸英文 label"))
    print("-" * 100)
    gaps = []
    for h in sorted(hits, key=lambda x: x["target"]):
        leaf = h["target"].split(".")[-1]
        under_crucible = h["target"].startswith("crucible.CONFIG.")
        flag = ""
        if under_crucible:
            flag = "  <== 补丁未覆盖" if leaf not in covered else "  (已覆盖)"
            if leaf not in covered:
                gaps.append(h)
        txt = ", ".join("%s=%r" % (b["field"], b["text"]) for b in h["bare_english"][:6])
        print("%-42s %-9d %s%s" % (h["target"], h["ember_mjs_line"], txt[:120], flag))

    json.dump({"covered": covered, "hits": hits, "gaps": gaps},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT, "| crucible.CONFIG 下未覆盖的注入点:", len(gaps))


if __name__ == "__main__":
    main()
