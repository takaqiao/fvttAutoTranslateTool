#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""往一份**临时副本**里注入 7 个已知的方位/序数错误，用来测 scan_orientation 的灵敏度。

绝不允许指向 `compendium/` 真目录：脚本会硬拒绝含 `1-Ember` / `2-Crucible` 的路径。

  python inject_orientation_faults.py --repo <临时副本目录>
"""
import argparse
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# (轴, 目标叶路径, 中文侧 old -> new)
FAULTS = [
    ('compass_ns', 'entries.Ember Early Access.journals.Arctus Plateau Gazetteer.pages.Talei.overview',
     '血林南缘', '血林北缘'),                       # EN: southern edge -> 中文改成北
    ('compass_ew', 'entries.Ember Early Access.journals.Ordain Gazetteer.pages.Westgate.text',
     '北侧的守望室', '南侧的守望室'),               # EN: northern Watchbox -> 南
    ('compass_ew_name', 'entries.Ember Early Access.scenes.Steed\'s Point.regions.Bridge - East.name',
     '桥梁 - 东', '桥梁 - 西侧'),                   # EN: Bridge - East -> 西
    ('lr', 'entries.Ember Early Access.journals.Chamber of Agaseros.pages.Antechamber.text',
     '左右两侧的墙中各嵌着一扇厚重的青铜门', '右侧的墙中嵌着一扇厚重的青铜门'),
    ('updown_spatial', 'entries.Ember Early Access.journals.Bastion Apex.pages.Lower Platforms.name',
     '下层平台', '上层平台'),                       # EN: Lower Platforms -> 上层
    ('ordinal', 'entries.Ember Early Access.journals.Spellbreaker Tower.categories.Level 3: Blue Block',
     '第3层：蓝区', '第五层：蓝区'),                # Level 3 -> 第五
    ('nameq', 'entries.Ember Early Access.journals.Vortest Tower.pages.Southern Sitting Room.name',
     '南会客室 Southern Sitting Room', '会客室 Southern Sitting Room'),  # 方位限定词整个丢掉
]


def get(node, parts):
    for p in parts[:-1]:
        node = node[p] if isinstance(node, dict) else node[int(p)]
    return node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--pack', default='ember.adventure.json')
    a = ap.parse_args()
    if re.search(r'1-Ember|2-Crucible', a.repo):
        sys.exit('拒绝：只能对临时副本注入，不能碰真仓库')

    p = os.path.join(a.repo, 'compendium', 'cn', a.pack)
    doc = json.load(open(p, encoding='utf-8-sig'))
    n = 0
    for axis, path, old, new in FAULTS:
        parts = path.split('.')
        assert parts[0] == 'entries'
        # key 里含 '.' 的情况：逐段贪婪匹配
        node, i, cur = doc, 1, []
        parts = parts[1:]
        node = doc['entries']
        while i <= len(parts):
            key = '.'.join(parts[:i]) if False else None
            break
        # 简单路径解析：本库这些叶子的 key 不含 '.'
        node = doc['entries']
        for k in parts[:-1]:
            node = node[k]
        leaf = parts[-1]
        cur = node[leaf]
        if old not in cur:
            print(f'  !! [{axis}] 未找到待改文本，跳过: {path}')
            continue
        node[leaf] = cur.replace(old, new, 1)
        n += 1
        print(f'  注入 [{axis}] {path}\n       {old!r} -> {new!r}')
    json.dump(doc, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'注入 {n}/{len(FAULTS)} 个错误 -> {p}')


if __name__ == '__main__':
    main()
