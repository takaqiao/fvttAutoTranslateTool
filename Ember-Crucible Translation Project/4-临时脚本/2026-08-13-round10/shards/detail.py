# -*- coding: utf-8 -*-
import json,sys
sys.stdout.reconfigure(encoding='utf-8')
d=json.load(open('Z5_remaining.json',encoding='utf-8'))
for a in sys.argv[1:]:
    g=d[int(a)]
    print('='*90)
    print('#%s EN=%r leaves=%d'%(a,g['en'],g['n_leaf']))
    for v in g['variants']:
        print('  CN=%r x%d'%(v['cn'],v['n']))
        for p in v['paths']: print('      %-32s %s'%(p[1],p[2]))
