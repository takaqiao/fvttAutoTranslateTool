# -*- coding: utf-8 -*-
"""把 scan_content_coverage 报出的条目连英中原文一起打出来，供逐条人工判定。"""
import json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '3-常用脚本', 'qa'))
import importlib.util
QA = r'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/3-常用脚本/qa/scan_content_coverage.py'
spec = importlib.util.spec_from_file_location('scc', QA)
scc = importlib.util.module_from_spec(spec); spec.loader.exec_module(scc)

REPOS = {'crucible': r'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/2-Crucible汉化插件',
         'ember':    r'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件'}

def get(o, parts):
    for p in parts:
        if isinstance(o, dict): o = o.get(p)
        elif isinstance(o, list):
            try: o = o[int(p)]
            except Exception: return None
        else: return None
    return o

rep = sys.argv[1]; jf = sys.argv[2]
repo = REPOS[rep]
d = json.load(open(jf, encoding='utf-8'))
for it in d['items']:
    pack = it['pack']; path = it['path']
    en = json.load(open(os.path.join(repo,'compendium','en',pack), encoding='utf-8-sig'))['entries']
    cn = json.load(open(os.path.join(repo,'compendium','cn',pack), encoding='utf-8-sig'))['entries']
    parts = scc_split = path.split('.')
    e = get(en, parts); c = get(cn, parts)
    print('='*100)
    print(f'{rep} | {pack} | {path}')
    print(f'缺: {it["missing_numbers"]}')
    pe, pc = scc.plain(e), scc.plain(c, cn=True)
    for n in [x.split('\u00d7')[0] for x in it['missing_numbers']]:
        # 打出英文里出现该数字的每一处上下文
        for m in re.finditer(r'(?<!\d)'+re.escape(n)+r'(?!\d)', pe):
            print('   EN ctx:', repr(pe[max(0,m.start()-90):m.end()+90]))
        for m in re.finditer(r'(?<!\d)'+re.escape(n)+r'(?!\d)', pc):
            print('   CN ctx:', repr(pc[max(0,m.start()-60):m.end()+60]))
    print()
