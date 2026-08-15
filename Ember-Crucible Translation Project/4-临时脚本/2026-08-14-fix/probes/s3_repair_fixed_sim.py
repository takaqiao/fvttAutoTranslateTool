# -*- coding: utf-8 -*-
"""Simulate the PROPOSED repair_bilingual_names.py (dry-run default + head-preserving
rebuild + drop the 'pure CJK with inner space' heuristic) over both repos."""
import json, os, re, sys
sys.stdout.reconfigure(encoding='utf-8')
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
CJK = re.compile(r'[\u4e00-\u9fff]')

def split_en_tail(cn_name, en_name):
    if cn_name.endswith(' ' + en_name): return cn_name[:-len(en_name) - 1].rstrip(), True
    if cn_name.endswith(en_name):       return cn_name[:-len(en_name)].rstrip(), True
    return cn_name, False

def is_broken(cn_name, en_name):
    if not isinstance(cn_name, str) or not isinstance(en_name, str): return False
    if cn_name == en_name: return False
    if not CJK.search(cn_name): return False
    head, has_tail = split_en_tail(cn_name, en_name)
    if has_tail:
        return f'{head} {en_name}' != cn_name          # only separator normalisation
    return bool(re.search(r'[A-Za-z]', cn_name))       # latin but no proper EN tail

def rebuild(cn_name, en_name):
    head, _ = split_en_tail(cn_name, en_name)
    return f'{head} {en_name}'.strip()

def scan(repo):
    cn_dir = os.path.join(PROJ, repo, 'compendium', 'cn'); en_dir = os.path.join(PROJ, repo, 'compendium', 'en')
    hits, refused = [], []
    def walk(cn, en):
        if isinstance(cn, dict) and isinstance(en, dict):
            if 'name' in cn and 'name' in en and is_broken(cn['name'], en['name']):
                new = rebuild(cn['name'], en['name'])
                if len(new) < len(cn['name']) or new == cn['name']:
                    refused.append((cn['name'], en['name'], new))
                else:
                    hits.append((cn['name'], en['name'], new))
            for k, v in cn.items():
                if k in en: walk(v, en[k])
        elif isinstance(cn, list) and isinstance(en, list):
            for a, b in zip(cn, en): walk(a, b)
    for fn in sorted(os.listdir(cn_dir)):
        if not fn.endswith('.json'): continue
        ep = os.path.join(en_dir, fn)
        if not os.path.exists(ep): continue
        walk(json.load(open(os.path.join(cn_dir, fn), encoding='utf-8')), json.load(open(ep, encoding='utf-8')))
    print(f'--- {repo}: proposed rewrites {len(hits)}   refused-by-guard {len(refused)}')
    for t in hits[:12]: print('   FIX  ', t)
scan('2-Crucible汉化插件'); scan('1-Ember汉化插件')
