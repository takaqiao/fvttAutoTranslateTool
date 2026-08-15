# -*- coding: utf-8 -*-
import sys,os,re
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'generic.py'),encoding='utf-8').read().split("out.sort")[0])
out.sort(key=lambda x:-x[0])
want=sys.argv[1]
for n,ne,nc,ncs,sk,rs in out:
    if want.lower() in sk.lower():
        print('#'*80); print('SKEL:',sk[:300]); print('leaves',n)
        seen=set()
        for r in rs:
            if r[4] in seen: continue
            seen.add(r[4])
            print('  --',r[1],r[2])
            print('     EN:',r[3][:300])
            print('     CN:',r[4][:300])
