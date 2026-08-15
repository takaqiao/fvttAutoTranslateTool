#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""复核用：给定文档 id，从 ember_ids.json 取英文名，再现读 compendium/cn 找该英文键下的中文 name。"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
ids = json.load(open(f"{P}/4-临时脚本/2026-08-12-fix/reports/ember_ids.json", encoding='utf-8'))

packs = {}
for pk in ["ember.crucible-adventure.json", "ember.adventure.json", "ember.crucible-adversary.json",
           "ember.crucible-items.json", "ember.crucible-character.json"]:
    try:
        d = json.load(open(f"{P}/1-Ember汉化插件/compendium/cn/{pk}", encoding='utf-8'))
        packs[pk] = d.get('entries', d)
    except Exception:
        pass

def find_name(root, enname):
    """在整棵树里找 key==enname 且其值是 dict 且含 name 的节点，返回中文 name 列表（带路径）。"""
    hits = []
    def rec(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == enname and isinstance(v, dict) and 'name' in v and isinstance(v['name'], str):
                    hits.append(('.'.join(path + [k]), v['name']))
                rec(v, path + [k])
        elif isinstance(node, list):
            for i, v in enumerate(node):
                rec(v, path + [str(i)])
    rec(root, [])
    return hits

for target in sys.argv[1:]:
    doc = target.split('.')[-1] if '.' in target else target
    # 逐段找最后一个已知 id
    segs = [s for s in target.split('.') if s in ids]
    key = segs[-1] if segs else doc
    info = ids.get(key)
    print("=" * 90)
    print("TARGET:", target)
    if not info:
        print("  id 表里没有:", key)
        continue
    print(f"  EN name = «{info['name']}»   type={info.get('type')}  pack={info.get('pack')}  via={info.get('via')}")
    for pk, root in packs.items():
        for path, cnname in find_name(root, info['name']):
            print(f"    [{pk}] {path}\n        CN name = «{cnname}»")
