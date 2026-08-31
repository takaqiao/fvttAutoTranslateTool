# -*- coding: utf-8 -*-
"""数 ember.mjs 里所有会成为 DialogV2 正文的字面英文。

为什么只数正文：
  - `window.title` 走 ApplicationV2#title -> _loc(title)（core/client/applications/api/application.mjs:320）
  - 按钮 `label` 走 DialogV2 -> _loc(label)（core/.../api/dialog.mjs:249）
  两者都能用「英文原串当 lang key」在 lang/cn.json 里翻掉，属可达。
  只有 `content` / `description`（直接塞 innerHTML 的字符串或 HTMLElement）没有任何本地化通道，
  而 ember-hardcoded-cn.mjs:459-465 的 DialogV2 分支只翻 `.window-title` 就 return。
只读。
"""
import re

P = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
t = open(P, encoding="utf-8", errors="replace").read()
line = lambda i: t.count("\n", 0, i) + 1

def visible(s):
    s = re.sub(r"<[^>]*>", " ", s)
    s = re.sub(r"\$\{[^}]*\}", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(re.findall(r"[A-Za-z]{3,}", s)) >= 2 else ""

rows = []

# 1) 直接的 content: "..." / description: "..."（含模板串）
for m in re.finditer(r"""\b(content|description)\s*:\s*(['"`])((?:\\.|(?!\2)[^\\])*)\2""", t):
    v = visible(m.group(3))
    if v:
        rows.append(("literal-content", line(m.start()), v[:150]))

# 2) 以 HTMLElement 拼出来的正文：p.textContent = "..." 紧邻 content 变量
for m in re.finditer(r"""\.textContent\s*=\s*(['"`])((?:\\.|(?!\1)[^\\])*)\1""", t):
    v = visible(m.group(2))
    if v:
        rows.append(("textContent", line(m.start()), v[:150]))

# 3) 三元/拼接形式的 textContent
for m in re.finditer(r"""\.textContent\s*=\s*[\s\S]{0,200}?\?\s*`([^`]*)`\s*:\s*["'`]([^"'`]*)["'`]""", t):
    for g in (m.group(1), m.group(2)):
        v = visible(g)
        if v:
            rows.append(("textContent?:", line(m.start()), v[:150]))

seen = set()
out = []
for kind, ln, v in rows:
    k = (ln, v)
    if k in seen:
        continue
    seen.add(k)
    out.append((kind, ln, v))
out.sort(key=lambda r: r[1])
for r in out:
    print(f"{r[1]:>7}  {r[0]:<14} {r[2]}")
print("\n总计:", len(out))
