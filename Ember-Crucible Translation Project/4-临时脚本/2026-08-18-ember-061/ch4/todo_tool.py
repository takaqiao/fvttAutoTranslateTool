# -*- coding: utf-8 -*-
"""待译项工具：编号 / 打印原文 / 术语提示 / 收取译文 / 校验。"""
import sys,os,json,re,collections,argparse
sys.path.insert(0,r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-18-ember-061\stage1")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
HERE=os.path.dirname(os.path.abspath(__file__))
J=lambda n: json.load(open(os.path.join(HERE,n),encoding='utf-8'))

def todo():
    t=J('prose_todo.json')
    t.sort(key=lambda x:(-len(x['en']), x['locs'][0][1]))
    return t

def terms_index():
    core=J('core_map.json'); gmd=J('gloss_md.json'); cand=J('gloss_map.json')
    idx={}
    for en,v in core.items(): idx[en]=v[0][0]
    for en,v in gmd.items():
        idx[en]=v[:-len(en)].rstrip() if v.endswith(en) else v
    for en,v in cand.items(): idx[en]=v[0]
    return idx

TAGSTRIP=re.compile(r'<[^>]+>')
_STRIP=re.compile(r'@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]|&(?:amp;)?[A-Za-z]+\[[^\]]*\]|<[^>]+>|&[a-z]+;')
def _readable(s):
    return re.sub(r'\s+',' ',_STRIP.sub(' ',s)).strip()
def hints(text, idx):
    plain=TAGSTRIP.sub(' ', text)
    found={}
    for m in re.finditer(r"\b[A-Z][A-Za-z'’]*(?:[ -][A-Z][A-Za-z'’]*)*\b", plain):
        w=m.group(0)
        for cand in (w, w.rstrip('s'), w+'s'):
            if cand in idx: found[cand]=idx[cand]; break
    return found

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['list','show','check','collect'])
    ap.add_argument('--ids', default='')
    ap.add_argument('--fill', default='')
    ap.add_argument('--out', default='')
    a=ap.parse_args()
    t=todo()
    if a.cmd=='list':
        for i,x in enumerate(t,1):
            print(f"{i:3d} {len(x['en']):7d} x{len(x['locs'])}  {x['locs'][0][1].replace('/entries/Ember Early Access/','')}")
        return
    ids=[]
    for part in a.ids.split(','):
        part=part.strip()
        if not part: continue
        if '-' in part:
            s,e=part.split('-'); ids+=list(range(int(s),int(e)+1))
        else: ids.append(int(part))
    if a.cmd=='show':
        idx=terms_index()
        for i in ids:
            x=t[i-1]
            print(f"<<<ITEM {i}>>>  {len(x['en'])} 字符 · {len(x['locs'])} 处")
            print(f"# 落点: {x['locs'][0][1].replace('/entries/Ember Early Access/','')}")
            h=hints(x['en'],idx)
            if h:
                print('# 术语（照抄，别另创）: '+' · '.join(f'{k}={v}' for k,v in sorted(h.items())))
            print(x['en'])
            print(f"<<<END {i}>>>\n")
        return
    if a.cmd in ('check','collect'):
        raw=open(a.fill,encoding='utf-8').read()
        blocks={}
        cur=None; buf=[]
        for line in raw.split('\n'):
            m=re.match(r'^<<<CN (\d+)>>>\s*$', line)
            e=re.match(r'^<<<END (\d+)>>>\s*$', line)
            if m:
                cur=int(m.group(1)); buf=[]
            elif e:
                assert cur==int(e.group(1)), f'块 {cur} 与结束标记 {e.group(1)} 不配对'
                blocks[cur]='\n'.join(buf); cur=None
            elif cur is not None: buf.append(line)
        prob=[]
        for i,cn in sorted(blocks.items()):
            en=t[i-1]['en']
            for nm,fn in (('TAG',tag_multiset),('ENH',uuid_targets),('ROLL',rolls)):
                x,y=collections.Counter(fn(cn)),collections.Counter(fn(en))
                if x!=y: prob.append(f'ITEM {i} {nm} 不齐 CN多={list(x-y)[:4]} EN多={list(y-x)[:4]}')
            if len(placeholders(cn))!=len(placeholders(en)):
                prob.append(f'ITEM {i} PH 个数 CN={len(placeholders(cn))} EN={len(placeholders(en))}')
            if not re.search(r'[\u4e00-\u9fff]',cn): prob.append(f'ITEM {i} 没有中文')
            # 外文杂串：只准 CJK / ASCII / 常用中文标点，抓 Cyrillic、Greek、假名之类的手滑
            bad=set(ch for ch in cn if not (ch.isascii() or '一'<=ch<='鿿' or '　'<=ch<='〿' or '＀'<=ch<='￯' or ch in '‘’“”—…·• –′″×°≠≤≥⬢​'))
            if bad: prob.append(f'ITEM {i} 混入非中文字符: {sorted(bad)}')
            # 比值只对「剥掉标签与记号之后的可读正文」算，否则纯记号叶必假警报
            se,sc=_readable(en),_readable(cn)
            if len(se)>=60:
                r=len(sc)/max(1,len(se))
                if r<0.18 or r>0.80: prob.append(f'ITEM {i} 中英字符比 {r:.2f} 异常（EN正文 {len(se)} → CN正文 {len(sc)}）')
        print(f'收到 {len(blocks)} 块: {sorted(blocks)}')
        for p in prob: print('  !!',p)
        if prob: return 1
        print('  校验全过')
        if a.cmd=='collect':
            out=[]
            for i,cn in sorted(blocks.items()):
                x=t[i-1]
                for f,p in x['locs']:
                    out.append({'pack':f,'ptr':p,'en':x['en'],'cn':cn,'src':'human'})
            json.dump(out,open(a.out,'w',encoding='utf-8'),ensure_ascii=False,indent=1)
            print(f'写出 {a.out}: {len(out)} 条落地项')
        return 0

if __name__=='__main__': sys.exit(main() or 0)
