import json,sys,re,difflib,os
repo=sys.argv[1]; pack=sys.argv[2]; batch=sys.argv[3]
cn=json.load(open(os.path.join(repo,'compendium','cn',pack),encoding='utf-8'))
en=json.load(open(os.path.join(repo,'compendium','en',pack),encoding='utf-8'))
b=json.load(open(batch,encoding='utf-8'))
def get(root,path):
    cur=root.get('entries',root)
    for seg in path.split('.'):
        if isinstance(cur,dict) and seg in cur: cur=cur[seg]
        else: return None
    return cur
MARK=re.compile(r'@UUID\[[^\]]*\]|\[\[[^\]]*\]\]|@Embed\[[^\]]*\]|@Check\[[^\]]*\]|<[^>]+>')
tot=0
for p,new in b.items():
    old=get(cn,p)
    if old is None: print('!! NOT FOUND', p); continue
    if old==new: print('== IDENTICAL (no-op)', p); continue
    # marker multiset compare
    mo=sorted(MARK.findall(old)); mn=sorted(MARK.findall(new))
    md = 'MARKERS-DIFFER' if mo!=mn else 'markers-ok'
    sm=difflib.SequenceMatcher(None,old,new)
    ops=[o for o in sm.get_opcodes() if o[0]!='equal']
    chg=sum(max(o[2]-o[1],o[4]-o[3]) for o in ops)
    tot+=len(ops)
    print(f'--- {p}  [{md}] edits={len(ops)} chars={chg}/{len(old)}')
    for tag,i1,i2,j1,j2 in ops:
        print(f'     {tag}: OLD<<{old[i1:i2]}>> NEW<<{new[j1:j2]}>>  ctx:...{old[max(0,i1-45):i1]}|')
    if mo!=mn:
        so=set(mo);sn=set(mn)
        print('     MARKER ONLY-OLD:',[x for x in mo if x not in sn][:10])
        print('     MARKER ONLY-NEW:',[x for x in mn if x not in so][:10])
print('TOTAL EDIT HUNKS',tot)
