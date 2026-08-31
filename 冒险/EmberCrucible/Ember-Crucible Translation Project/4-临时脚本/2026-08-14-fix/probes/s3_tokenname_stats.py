# -*- coding: utf-8 -*-
import json, os, glob, io

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PAIRS = []
for repo in ['1-Ember汉化插件', '2-Crucible汉化插件']:
    en_dir = os.path.join(ROOT, repo, 'compendium', 'en')
    cn_dir = os.path.join(ROOT, repo, 'compendium', 'cn')
    for fn in sorted(os.listdir(en_dir)):
        if not fn.endswith('.json') or fn == '_source.json':
            continue
        cn = os.path.join(cn_dir, fn)
        if os.path.exists(cn):
            PAIRS.append((os.path.join(en_dir, fn), cn))

assert PAIRS, 'no pairs'

def walk(entries, path, out):
    """Collect (path, entry-dict) for every dict that has a 'tokenName' or is an actor-ish node."""
    if isinstance(entries, dict):
        for k, v in entries.items():
            walk(v, path + [str(k)], out)
            if isinstance(v, dict) and ('tokenName' in v or 'name' in v):
                out.append(('/'.join(path + [str(k)]), v))

tot_en_tokenname = 0
tot_cn_tokenname = 0
have_name_no_tokenname = 0
both = 0
diff_from_name = 0
samples = []
for en_p, cn_p in PAIRS:
    assert en_p != cn_p
    en = json.load(io.open(en_p, encoding='utf-8'))
    cn = json.load(io.open(cn_p, encoding='utf-8'))
    eo, co = [], []
    walk(en.get('entries', {}), [], eo)
    walk(cn.get('entries', {}), [], co)
    cnmap = dict(co)
    for p, ev in eo:
        if 'tokenName' not in ev:
            continue
        tot_en_tokenname += 1
        cv = cnmap.get(p)
        if not isinstance(cv, dict):
            continue
        has_cn_token = isinstance(cv.get('tokenName'), str) and cv['tokenName'].strip()
        has_cn_name = isinstance(cv.get('name'), str) and cv['name'].strip()
        if has_cn_token:
            tot_cn_tokenname += 1
            both += 1
            if cv['tokenName'] != cv.get('name'):
                diff_from_name += 1
        elif has_cn_name:
            have_name_no_tokenname += 1
            if len(samples) < 15:
                samples.append((os.path.basename(en_p), p, ev.get('name'), ev.get('tokenName'), cv.get('name')))

print('en entries carrying tokenName        :', tot_en_tokenname)
print('cn entries carrying tokenName        :', tot_cn_tokenname)
print('cn has name but NO tokenName         :', have_name_no_tokenname)
print('cn tokenName != cn name              :', diff_from_name)
print()
for s in samples:
    print(s)
