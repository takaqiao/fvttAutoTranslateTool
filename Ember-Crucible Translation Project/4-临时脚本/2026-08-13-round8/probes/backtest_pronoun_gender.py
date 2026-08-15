#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scan_pronoun_gender.py 的**灵敏度回测**：往临时副本里注入已知的代词性别错误，
确认每一条臂都能把自己那一类报出来。

绝不碰真库 —— 全部操作在 `%TEMP%` 下的副本上做，脚本自己复制、自己清理。

用法:
  python backtest_pronoun_gender.py --root "<项目根>" [--keep]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scan_pronoun_gender as S           # noqa: E402

REPOS = ['1-Ember汉化插件', '2-Crucible汉化插件']


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def set_at(obj, path, value):
    seg = path.split('.')
    for k in seg[:-1]:
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    if isinstance(obj, list):
        obj[int(seg[-1])] = value
    else:
        obj[seg[-1]] = value


def get_at(obj, path):
    for k in path.split('.'):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    return obj


def pick_injections(rows, ents, confident):
    """在真库里挑出**本来正确**的叶，构造出该类错误的注入点。"""
    inj = []
    used = set()          # 同一片叶只能承载一个注入，否则会互相覆盖
    def take(item):
        k = (item['row']['repo'], item['row']['pack'], item['row']['path'])
        if k in used:
            return False
        used.add(k)
        inj.append(item)
        return True
    cn_of = {ents[n]: n for n in confident}

    # I1 / I2  G 臂：性别确凿的人物，名字后 60 字内的正确代词 -> 翻成反向
    for want, good, bad in (('F', '她', '他'), ('M', '他', '她')):
        for r in rows:
            if not r['cn'] or len(r['cn']) > 3000:
                continue
            c = S.strip_cn(r['cn'])
            for h, n in cn_of.items():
                if confident[n][0] != want or h not in c:
                    continue
                pos = c.index(h)
                rxg = S.CN_F if good == '她' else S.CN_M
                rxb = S.CN_M if good == '她' else S.CN_F
                m = rxg.search(c, pos)
                if not m or m.start() - pos > 50 or rxb.search(c):
                    continue
                if (want == 'M' and S.EN_FEM.search(S.strip_en(r['en']))) or \
                   (want == 'F' and S.EN_MASC.search(S.strip_en(r['en']))):
                    continue
                take({'id': f'I-G-{want}', 'arm': 'G', 'row': r,
                      'new': r['cn'].replace(good, bad),
                      'desc': f'把「{h}」（英文侧确凿为{"女" if want=="F" else "男"}）'
                              f'的正确代词「{good}」改成「{bad}」'})
                break
            if any(x['id'] == f'I-G-{want}' for x in inj):
                break

    # I3  B 臂：英文只有女性代词、中文只有「她」的叶 -> 她 全改「他」
    for r in rows:
        if not r['cn']:
            continue
        e, c = S.strip_en(r['en']), S.strip_cn(r['cn'])
        if S.EN_FEM.search(e) and not S.EN_MASC.search(e) \
                and S.CN_F.search(c) and not S.CN_M.search(c) and len(r['cn']) < 2500:
            if take({'id': 'I-B', 'arm': 'B', 'row': r,
                     'new': r['cn'].replace('她', '他'),
                     'desc': '英文只出现 she/her 的叶，中文「她」全改「他」'}):
                break

    # I4  D 臂：历史缺陷的形状 —— 中文有「女神」而代词写成「他」
    for r in rows:
        if not r['cn']:
            continue
        e, c = S.strip_en(r['en']), S.strip_cn(r['cn'])
        if '女神' in c and S.CN_F.search(c) and not S.CN_M.search(c) \
                and not S.EN_MASC.search(e) and len(r['cn']) < 2500:
            if take({'id': 'I-D', 'arm': 'D', 'row': r,
                     'new': r['cn'].replace('她', '他'),
                     'desc': '复刻历史缺陷形状：中文有「女神」，代词却写「他」'}):
                break

    # I5  E 臂：英文 Goddess、中文「女神」-> 抹掉性别标记写成「之神」
    for r in rows:
        if not r['cn']:
            continue
        e, c = S.strip_en(r['en']), S.strip_cn(r['cn'])
        if re.search(r'\bgoddess(es)?\b', e, re.I) and not re.search(r'\bgods?\b', e, re.I) \
                and '女神' in c and '之神' not in c and '男神' not in c:
            if take({'id': 'I-E', 'arm': 'E', 'row': r,
                     'new': r['cn'].replace('女神', '之神'),
                     'desc': '英文 Goddess，中文「女神」被改成中性的「之神」'}):
                break

    # I6  F 臂：英文只有 it/its 的叶，中文「它」改「他」
    for r in rows:
        if not r['cn']:
            continue
        e, c = S.strip_en(r['en']), S.strip_cn(r['cn'])
        if S.EN_INAN.search(e) and not (S.EN_FEM.search(e) or S.EN_MASC.search(e)
                                        or S.EN_NEUT.search(e)) \
                and S.CN_IT.search(c) and not S.CN_M.search(c) and len(r['cn']) < 1500:
            if take({'id': 'I-F', 'arm': 'F', 'row': r,
                     'new': re.sub(r'(?<!其)它(?!们)', '他', r['cn']),
                     'desc': '英文通篇 it/its 的叶，中文「它」改成「他」'}):
                break

    # I7  A 臂：同英文串多份译文，挑一份把「他」改「她」
    groups = {}
    for r in rows:
        if r['cn'] and (S.EN_FEM.search(r['en']) or S.EN_MASC.search(r['en'])
                        or S.EN_NEUT.search(r['en'])) and len(r['en']) >= 40:
            groups.setdefault(re.sub(r'\s+', ' ', r['en']).strip(), []).append(r)
    for k, g in groups.items():
        if len(g) < 3:
            continue
        sigs = {S.sig_cn(S.strip_cn(x['cn'])) for x in g}
        if sigs != {'他'}:
            continue
        if take({'id': 'I-A', 'arm': 'A', 'row': g[0],
                 'new': g[0]['cn'].replace('他', '她'),
                 'desc': f'同一条英文串有 {len(g)} 份译文且都写「他」，把其中一份改「她」'}):
            break
    return inj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()

    rows = []
    for rp in REPOS:
        rows.extend(S.collect(os.path.join(a.root, rp), rp))
    ents = S.build_entities(rows)
    # 复用扫描器的投票逻辑太长，这里只要「确凿」表，直接跑一次扫描器拿它的判定
    # —— 简化：用 name 表 + 全库句级投票的简版
    en_names = sorted(ents, key=len, reverse=True)
    RX_ALL = re.compile(r'\b(' + '|'.join(re.escape(n) for n in en_names) + r')\b')
    SENT = re.compile(r'(?<=[.!?;:])\s+|</p>|</li>|</dt>|</dd>')
    vote = {}
    for r in rows:
        for s in SENT.split(S.strip_en(r['en'])):
            hits = {h for h in RX_ALL.findall(s or '') if h in ents}
            if len(hits) != 1:
                continue
            n = next(iter(hits))
            v = vote.setdefault(n, [0, 0])
            v[0] += len(S.EN_FEM.findall(s))
            v[1] += len(S.EN_MASC.findall(s))
    confident = {}
    for n, (F, M) in vote.items():
        if (F >= 3 and M == 0) or (F + M >= 6 and F > M and F >= 6 * max(M, 1)):
            confident[n] = ('F', F, M, 0)
        elif (M >= 3 and F == 0) or (F + M >= 6 and M > F and M >= 6 * max(F, 1)):
            confident[n] = ('M', F, M, 0)

    inj = pick_injections(rows, ents, confident)
    print(f'构造出 {len(inj)} 个注入点：')
    for x in inj:
        print(f'  [{x["id"]}] {x["row"]["pack"]} :: {x["row"]["batch_path"][-60:]}')
        print(f'          {x["desc"]}')

    tmp = tempfile.mkdtemp(prefix='pgb_')
    try:
        for rp in REPOS:
            shutil.copytree(os.path.join(a.root, rp, 'compendium'),
                            os.path.join(tmp, rp, 'compendium'))
        for x in inj:
            r = x['row']
            p = os.path.join(tmp, r['repo'], 'compendium', 'cn', r['pack'])
            doc = load(p)
            rel = r['path'][len('entries.'):]
            cur = get_at(doc['entries'], rel)
            assert cur == r['cn'], f'{x["id"]} 定位不一致'
            set_at(doc['entries'], rel, x['new'])
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=1)

        out = os.path.join(tmp, 'inj.json')
        cmd = [sys.executable, os.path.join(HERE, 'scan_pronoun_gender.py'),
               '--out', out, '--show', '0']
        for rp in REPOS:
            cmd += ['--repo', os.path.join(tmp, rp)]
        subprocess.run(cmd, check=True, capture_output=True)
        got = json.load(open(out, encoding='utf-8'))
        hit_paths = {(f['pack'], f['batch_path'], f['arm']) for f in got['findings']}

        print('\n灵敏度结果：')
        ok = 0
        for x in inj:
            r = x['row']
            same_arm = (r['pack'], r['batch_path'], x['arm']) in hit_paths
            any_arm = any(p == r['pack'] and b == r['batch_path'] for p, b, _ in hit_paths)
            arms = sorted({arm for p, b, arm in hit_paths
                           if p == r['pack'] and b == r['batch_path']})
            mark = '报出(本臂)' if same_arm else ('报出(他臂 %s)' % ','.join(arms)
                                                  if any_arm else '**漏报**')
            ok += bool(any_arm)
            print(f'  [{x["id"]}] 目标臂 {x["arm"]} -> {mark}')
        print(f'\n注入 {len(inj)} 条，报出 {ok} 条')
    finally:
        if a.keep:
            print(f'副本保留在 {tmp}')
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
