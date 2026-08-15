import json,sys,re,os
# usage: v2_leaf.py <repo> <pack> <path> [uuid_frag] [win]
repo,pack,path=sys.argv[1],sys.argv[2],sys.argv[3]
frag=sys.argv[4] if len(sys.argv)>4 else None
win=int(sys.argv[5]) if len(sys.argv)>5 else 260
def get(root,p):
    cur=root.get('entries',root)
    for seg in p.split('.'):
        if isinstance(cur,dict) and seg in cur: cur=cur[seg]
        else: return None
    return cur
for side in ('en','cn'):
    d=json.load(open(os.path.join(repo,'compendium',side,pack),encoding='utf-8'))
    v=get(d,path)
    print('='*20,side.upper(),'='*20)
    if v is None: print('(missing)'); continue
    if frag:
        for m in re.finditer(re.escape(frag),v):
            s=max(0,m.start()-win); e=min(len(v),m.end()+win)
            print('...'+v[s:e]+'...'); print('  --')
    else:
        print(v)
