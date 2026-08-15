# -*- coding: utf-8 -*-
"""
swap_aware_count.py —— 探针 E：扣掉 system-swap 块之后的真实可见计数

ember 的 finalizeEnrichedHTML（ember.mjs:23219）会在渲染时删掉
`sup.system-swap-inline > sub[data-system!=当前系统]` 与
`div.system-swap-block > div[data-system!=当前系统]`。
所以合集里的一处 `[[/xxx]]` 未必在 crucible 世界里可见。
本探针把 data-system="dnd5e" 的块整段剔除后再计数，得到 **crucible 世界实际渲染** 的处数。

只读，不写库。
"""
import json, os, re, collections

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
FILES = [
    os.path.join(ROOT, r"1-Ember汉化插件\compendium\cn\ember.crucible-adventure.json"),
    os.path.join(ROOT, r"1-Ember汉化插件\compendium\cn\ember.crucible-character.json"),
    os.path.join(ROOT, r"1-Ember汉化插件\compendium\cn\ember.crucible-adversary.json"),
    os.path.join(ROOT, r"1-Ember汉化插件\compendium\cn\ember.crucible-effects.json"),
    os.path.join(ROOT, r"1-Ember汉化插件\compendium\cn\ember.crucible-items.json"),
    os.path.join(ROOT, r"1-Ember汉化插件\compendium\cn\ember.crucible-affixes.json"),
]
FILES += [os.path.join(ROOT, r"2-Crucible汉化插件\compendium\cn", f)
          for f in os.listdir(os.path.join(ROOT, r"2-Crucible汉化插件\compendium\cn"))
          if f.endswith(".json")]

DND_BLOCK = re.compile(
    r'<(sub|div)[^>]*data-system="dnd5e"[^>]*>.*?</\1>', re.S | re.I)

TARGETS = {
    "[[/ancestry]]": re.compile(r"\[\[/ancestry "),
    "[[/culture]]": re.compile(r"\[\[/culture "),
    "[[/path]]": re.compile(r"\[\[/path "),
    "[[/talent]]": re.compile(r"\[\[/talent "),
    "@Spell[]": re.compile(r"@Spell\["),
    "[[/soundscape music]]": re.compile(r"\[\[/soundscape\s+\w+\s+(?!reset)\w+"),
    "[[/soundscape reset]]": re.compile(r"\[\[/soundscape\s+\w+\s+reset"),
    "[[/soundscape mood=]]": re.compile(r"\[\[/soundscape\s+mood="),
    "[[/attunement]]": re.compile(r"\[\[/attunement "),
    "[[/knowledge]]": re.compile(r"\[\[/knowledge "),
    "[[/language]]": re.compile(r"\[\[/language "),
}


def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)
    elif isinstance(o, str):
        yield o


def main():
    raw = collections.Counter()
    kept = collections.Counter()
    dropped_blocks = 0
    for p in FILES:
        if not os.path.exists(p):
            continue
        data = json.load(open(p, encoding="utf-8"))
        for s in walk(data):
            for name, rx in TARGETS.items():
                raw[name] += len(rx.findall(s))
            s2, n = DND_BLOCK.subn("", s)
            dropped_blocks += n
            for name, rx in TARGETS.items():
                kept[name] += len(rx.findall(s2))
    print(f"# crucible 侧文件 {len(FILES)}；剔除 data-system=dnd5e 块 {dropped_blocks} 段")
    print(f"{'pattern':26s} {'原始':>6s} {'crucible实际':>12s}")
    for name in TARGETS:
        print(f"{name:26s} {raw[name]:6d} {kept[name]:12d}")


if __name__ == "__main__":
    main()
