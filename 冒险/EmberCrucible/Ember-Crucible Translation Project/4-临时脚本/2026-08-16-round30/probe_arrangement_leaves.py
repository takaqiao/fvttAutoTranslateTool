# -*- coding: utf-8 -*-
"""第三十轮 · 文档这一路自己的复算：ARRANGEMENT_LEAVES 到底有多少个 distinct 键。

用途：核对 §8 第二十九轮写的「721 − 212 = 509 个 distinct 键零常设译文判据」这个减法。

⚠ 本探针**落盘再跑**（硬约束：正则里的 \\b / \\s 不许经 shell heredoc 传，本项目栽过两次）。
⚠ 第一版写错过一次，值得记：`src.index("ARRANGEMENT_LEAVES")` 命中的是**注释里**的第一次出现，
  于是花括号配对配到了注释块上，算出 42 —— 假数、而且看不出是假的。
  ⇒ 锚点必须写成 `const <NAME> = {`。

口径：与面板 runner 的 countTableKeys 同口径（`Object.keys(t)`，只数顶层键）。
      `ARRANGEMENT_LEAVES = { ...ARRANGEMENTS, ...7 条来自 SOUNDSCAPE_GROUPS }`
      ⇒ 它的顶层键 = ARRANGEMENTS 的键 ∪ 那 7 条里在 SOUNDSCAPE_GROUPS 里存在的。
"""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = (r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
       r"\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs")

src = open(SRC, encoding="utf-8").read()

KEY_RE = re.compile(r'^\s{2}"((?:[^"\\]|\\.)*)"\s*:', re.M)


def block(name):
    anchor = "const %s = {" % name
    i = src.index(anchor)
    j = src.index("{", i)
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[j:k + 1]
    raise AssertionError("花括号没配上：%s" % name)


arr = KEY_RE.findall(block("ARRANGEMENTS"))
grp = KEY_RE.findall(block("SOUNDSCAPE_GROUPS"))
SPREAD_7 = ["Ancient Ruins", "Ankarist Theme", "Lyla Theme", "Marlstone Gala",
            "Ordain", "Sin Theme", "The Pit Trap"]

print("ARRANGEMENTS       raw=%d distinct=%d" % (len(arr), len(set(arr))))
print("SOUNDSCAPE_GROUPS  raw=%d distinct=%d" % (len(grp), len(set(grp))))
extra = [k for k in SPREAD_7 if k in set(grp)]
print("那 7 条里真在 SOUNDSCAPE_GROUPS 中的：%d 条 %s" % (len(extra), extra))
leaves = set(arr) | set(extra)
print("ARRANGEMENT_LEAVES distinct = %d" % len(leaves))
print("721 - %d = %d" % (len(leaves), 721 - len(leaves)))
