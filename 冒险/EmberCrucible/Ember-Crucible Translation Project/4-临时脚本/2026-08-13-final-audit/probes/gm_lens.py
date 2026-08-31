#!/usr/bin/env python3
"""GM-leak structural lens. READ ONLY."""
from __future__ import annotations
import json, os, re, sys

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPOS = {
    'ember': os.path.join(ROOT, '1-Ember汉化插件'),
    'crucible': os.path.join(ROOT, '2-Crucible汉化插件'),
}
CJK = re.compile(r'[\u4e00-\u9fff]')


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [k], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                 path + [str(i)], out)
    elif isinstance(en, str) and en.strip():
        out.append(('.'.join(path), en, cn if isinstance(cn, str) else None))


def pairs(repo):
    en_dir = os.path.join(repo, 'compendium', 'en')
    cn_dir = os.path.join(repo, 'compendium', 'cn')
    for pack in sorted(os.listdir(en_dir)):
        if not pack.endswith('.json') or pack.startswith('_'):
            continue
        cn_p = os.path.join(cn_dir, pack)
        if not os.path.exists(cn_p):
            continue
        en = json.load(open(os.path.join(en_dir, pack), encoding='utf-8')).get('entries', {})
        cn = json.load(open(cn_p, encoding='utf-8')).get('entries', {})
        o = []
        walk(en, cn, [], o)
        yield pack, o
