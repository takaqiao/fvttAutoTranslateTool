# -*- coding: utf-8 -*-
import json,sys,re,collections
sys.stdout.reconfigure(encoding='utf-8')
d=json.load(open('Z5_remaining.json',encoding='utf-8'))
BI_FOLDER_PACKS={'ember.adventure.json','ember.character.json','ember.crucible-adventure.json',
                 'ember.crucible-adversary.json','ember.crucible-character.json'}
def conv(pack,path):
    parts=path.split('.'); last=parts[-1]
    if last=='tokenName': return 'CN'
    if last=='adjective': return 'CN'
    if '.levels.' in path: return 'CN'
    if '.regions.' in path: return 'CN'
    if '.tokens.' in path: return 'CN'
    if '.categories.' in path or path.startswith('categories.'):
        return 'BI' if pack=='crucible.rules.json' else 'CN'
    if '.folders.' in path or path.startswith('folders.'):
        return 'BI' if pack in BI_FOLDER_PACKS else 'CN'
    if path=='label': return 'BI'
    return 'BI'
def form(cn):
    return 'BI' if re.search(r'[A-Za-z]',cn) else 'CN'
def core(cn,en):
    s=cn
    # remove trailing english segment matching en, sep space or newline
    for sep in ('\n',' '):
        if s.endswith(sep+en): return s[:-(len(en)+1)]
    if s==en: return s
    # generic: strip trailing ascii run
    m=re.search(r'[\s\n]([\x20-\x7e]+)$',s)
    if m and re.search(r'[A-Za-z]',m.group(1)): return s[:m.start()]
    return s
res=[]
for i,g in enumerate(d):
    en=g['en']
    cores=set(); viol=[]
    for v in g['variants']:
        f=form(v['cn']); c=core(v['cn'],en); cores.add(c)
        for p in v['paths']:
            e=conv(p[1],p[2])
            if e!=f: viol.append((p[1],p[2],v['cn'],e))
    res.append(dict(i=i,en=en,n_leaf=g['n_leaf'],cores=sorted(cores),n_core=len(cores),viol=viol))
json.dump(res,open('classified.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
a=[r for r in res if r['n_core']==1 and not r['viol']]
b=[r for r in res if r['n_core']==1 and r['viol']]
c=[r for r in res if r['n_core']>1]
print('LEGIT (single core, no convention violation):',len(a))
print('CONVENTION VIOLATION (single core):',len(b))
print('MULTI-CORE (real divergence):',len(c))
print()
print('--- CONVENTION VIOLATIONS ---')
for r in b:
    print(f"#{r['i']} {r['en']!r} core={r['cores']}")
    for v in r['viol'][:8]: print('   ',v)
print()
print('--- MULTI-CORE ---')
for r in c:
    print(f"#{r['i']} {r['en']!r} leaves={r['n_leaf']} cores={r['cores']} viol={len(r['viol'])}")
