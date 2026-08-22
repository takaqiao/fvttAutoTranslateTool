# -*- coding: utf-8 -*-
"""把 611 条按现有排版拼成可以直接贴进 TOKEN_MAKER_PART_IDS 的文本块。

排版跟现有那 671 条一致：键按字典序、每行塞到 100 列上限为止、四空格缩进。
"""
import io, sys
sys.stdout.reconfigure(encoding='utf-8')
rows = [l.rstrip("\n").split("\t") for l in io.open("review.tsv", encoding="utf-8")]
rows.sort(key=lambda r: r[1])
items = ['"%s": "%s",' % (seg, cn) for _fam, seg, cn in rows]
LIM = 100
lines, cur = [], "  "
for it in items:
    add = (" " if cur.strip() else "") + it
    if len(cur) + len(add) > LIM and cur.strip():
        lines.append(cur.rstrip())
        cur = "  " + it
    else:
        cur += add
if cur.strip():
    lines.append(cur.rstrip())
lines[-1] = lines[-1].rstrip(",")
io.open("emit.txt", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("条数", len(items), "→", len(lines), "行")
print("\n".join(lines[:4]))
print("...")
print("\n".join(lines[-2:]))
