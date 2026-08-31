# -*- coding: utf-8 -*-
"""Read-only reproduction of scripts/repair_bilingual_names.py, plus the effect of
the proposed 'no CJK may be lost' + 'no EN duplication' guards."""
import json, os, re, sys, importlib.util
sys.stdout.reconfigure(encoding='utf-8')

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SCRIPT = os.path.join(PROJ, '2-Crucible汉化插件', 'scripts', 'repair_bilingual_names.py')
spec = importlib.util.spec_from_file_location('rbn', SCRIPT)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

CJKre = re.compile(r'[\u4e00-\u9fff]')

def scan(repo):
    cn_dir = os.path.join(PROJ, repo, 'compendium', 'cn')
    en_dir = os.path.join(PROJ, repo, 'compendium', 'en')
    hits, lossy, dupe = [], [], []
    def walk(cn, en):
        if isinstance(cn, dict) and isinstance(en, dict):
            if 'name' in cn and 'name' in en and m.is_broken(cn['name'], en['name']):
                new = m.rebuild(cn['name'], en['name'])
                hits.append((cn['name'], en['name'], new))
                if len(CJKre.findall(new)) < len(CJKre.findall(cn['name'])):
                    lossy.append((cn['name'], en['name'], new))
                if en['name'] and new.count(en['name']) > 1:
                    dupe.append((cn['name'], en['name'], new))
            for k, v in cn.items():
                if k in en: walk(v, en[k])
        elif isinstance(cn, list) and isinstance(en, list):
            for a, b in zip(cn, en): walk(a, b)
    for fn in sorted(os.listdir(cn_dir)):
        if not fn.endswith('.json'): continue
        ep = os.path.join(en_dir, fn)
        if not os.path.exists(ep): continue
        walk(json.load(open(os.path.join(cn_dir, fn), encoding='utf-8')),
             json.load(open(ep, encoding='utf-8')))
    print(f'--- {repo}: would rewrite {len(hits)} names; '
          f'{len(lossy)} LOSE CJK chars; {len(dupe)} duplicate the EN name')
    for t in lossy[:6]: print('   LOSSY ', t)
    for t in dupe[:4]:  print('   DUPE  ', t)
    survivors = [h for h in hits if h not in lossy and h not in dupe]
    print(f'   survivors after both guards: {len(survivors)}')
    for t in survivors[:8]: print('   KEEP  ', t)

scan('2-Crucible汉化插件')
scan('1-Ember汉化插件')
