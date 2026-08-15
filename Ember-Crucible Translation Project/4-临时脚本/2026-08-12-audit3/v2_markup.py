import json,os,re,sys,collections
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
REPO=r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件"
BAT=r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches"
def get(root,p):
    cur=root.get('entries',root)
    for seg in p.split('.'):
        if isinstance(cur,dict) and seg in cur: cur=cur[seg]
        else: return None
    return cur
BR=re.compile(r'@\w+\[([^\]]*)\]')          # bracket interiors of all enrichers
INL=re.compile(r'\[\[([^\]]*)\]\]')          # inline command bodies
CLS=re.compile(r'class="([^"]*)"')
TAG=re.compile(r'<\s*/?\s*([a-zA-Z0-9]+)')
bad=0
for pack in ["ember.adventure.json","ember.crucible-adventure.json"]:
    cn=json.load(open(os.path.join(REPO,'compendium','cn',pack),encoding='utf-8'))
    en=json.load(open(os.path.join(REPO,'compendium','en',pack),encoding='utf-8'))
    b=json.load(open(os.path.join(BAT,f"U2__ember__{pack}"),encoding='utf-8'))
    order=sorted(b.items(),key=lambda kv:-len(kv[1]))
    for rank,(p,new) in enumerate(order):
        old=get(cn,p); ens=get(en,p) or ''
        for name,rx in (('BRACKET',BR),('INLINE',INL),('CLASS',CLS),('TAGS',TAG)):
            a=collections.Counter(rx.findall(old)); c=collections.Counter(rx.findall(new))
            if a!=c:
                bad+=1
                print(f'!! {name} DRIFT vs old cn: {pack} :: {p}')
                print('   only-old:',list((a-c).elements())[:8]); print('   only-new:',list((c-a).elements())[:8])
            # also compare bracket interiors against ENGLISH source (byte fidelity)
            if name=='BRACKET':
                e=collections.Counter(rx.findall(ens))
                if e!=c:
                    print(f'   ~ note: bracket set differs from EN ({pack} :: {p[:70]})')
                    print('     only-EN :',list((e-c).elements())[:5]); print('     only-CN :',list((c-e).elements())[:5])
        if rank<2:
            print(f'[longest #{rank+1}] {pack} :: {p}  len={len(new)}')
            print('   brackets preserved:',len(BR.findall(new)),' inline:',len(INL.findall(new)),' classes:',sorted(set(CLS.findall(new)))[:8])
print('\nDRIFT COUNT:',bad)
