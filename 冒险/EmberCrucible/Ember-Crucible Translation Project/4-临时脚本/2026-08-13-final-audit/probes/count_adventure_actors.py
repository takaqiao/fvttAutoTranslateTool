#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量一下 degradeActorUpdatePayload 的影响面：冒险包里有多少 actor、其中多少带
embedded items / effects（en 基线按 mappings.mjs 抽取，actors.*.items.* 就是
Adventure -> actors -> items 这条链）。只读。"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for repo, packs in [('1-Ember汉化插件', ['ember.adventure.json', 'ember.crucible-adventure.json'])]:
    for pack in packs:
        p = os.path.join(ROOT, repo, 'compendium', 'en', pack)
        d = json.load(open(p, encoding='utf-8-sig'))
        ents = d.get('entries', d)
        tot_adv = len(ents)
        actors = 0
        with_items = 0
        n_items = 0
        with_effects = 0
        for name, adv in ents.items():
            acts = adv.get('actors') or {}
            actors += len(acts)
            for an, a in acts.items():
                its = a.get('items') or {}
                if its:
                    with_items += 1
                    n_items += len(its)
                if a.get('effects'):
                    with_effects += 1
        print(f'{repo}/{pack}: adventures={tot_adv} actors={actors} '
              f'带embedded items 的 actor={with_items}（items 叶条目 {n_items}）'
              f' 带 effects 的 actor={with_effects}')
