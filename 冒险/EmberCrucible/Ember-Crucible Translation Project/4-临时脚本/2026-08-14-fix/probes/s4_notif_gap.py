# -*- coding: utf-8 -*-
"""ui.notifications 第一实参的静态英文串 vs 兄弟分片 G1 已交的 lang 批次：算差集。

只读。用来判断本片的 `hook-gap|ui.notifications` /
`i18n-sink-bare-string|ui.notifications.notify-first-arg` 两条是否已被覆盖。
"""
import re, json, os, sys

EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
HERE = os.path.dirname(os.path.abspath(__file__))
G1 = os.path.join(HERE, "..", "lang", "G1.1.json")

CALL = re.compile(r'ui\.notifications\.(?:notify|info|warn|error)\(\s*(["\'])((?:[^\\]|\\.)*?)\1')


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    src = open(EMBER, encoding="utf-8").read()
    covered = set(json.load(open(G1, encoding="utf-8-sig")))
    hits = []
    for m in CALL.finditer(src):
        line = src.count("\n", 0, m.start()) + 1
        s = m.group(2).replace('\\"', '"').replace("\\'", "'")
        hits.append((line, s))
    gap = [(l, s) for l, s in hits if s not in covered]
    print(f"ui.notifications 静态串 {len(hits)}  G1 lang 批次键 {len(covered)}  未覆盖 {len(gap)}")
    for l, s in gap:
        print(f"  L{l}  {s!r}")
    print("\nG1 批次里没有对应静态串的键（应为 0，否则是拼写不一致）：")
    for k in covered - {s for _, s in hits}:
        print("  !", repr(k))


if __name__ == "__main__":
    main()
