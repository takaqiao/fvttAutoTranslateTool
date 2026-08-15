#!/usr/bin/env python3
"""复核 G3：把批次值与 compendium/cn 现值逐字符 diff，列出每一处改动。"""
import json, sys, io, os, re, difflib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = {
    'ember': os.path.join(P, '1-Ember汉化插件'),
    'crucible': os.path.join(P, '2-Crucible汉化插件'),
}
BD = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches"


def get(d, dotted):
    cur = d
    for part in dotted.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_pack(repo, pack, side):
    p = os.path.join(REPOS[repo], 'compendium', side, pack)
    return json.load(open(p, encoding='utf-8'))


def resolve(pack_data, batch_path):
    # batch path 去掉了 entries. 前缀；点号路径中段可能含点(名字里有点)，先直接试
    ent = pack_data.get('entries', pack_data)
    v = get(ent, batch_path)
    if v is not None:
        return v
    # 回退：贪心分段
    parts = batch_path.split('.')
    cur = ent
    i = 0
    while i < len(parts):
        for j in range(len(parts), i, -1):
            key = '.'.join(parts[i:j])
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
                i = j
                break
        else:
            return None
    return cur


def main():
    for fn in sorted(os.listdir(BD)):
        if not fn.startswith('G3__'):
            continue
        _, repo, packname = fn.split('__')
        pack = packname[:-5] + '.json' if packname.endswith('.json') else packname
        pack = packname
        batch = json.load(open(os.path.join(BD, fn), encoding='utf-8'))
        cn = load_pack(repo, pack, 'cn')
        try:
            en = load_pack(repo, pack, 'en')
        except Exception:
            en = None
        print('#' * 100)
        print('###', fn)
        for k, newv in batch.items():
            oldv = resolve(cn, k)
            env = resolve(en, k) if en else None
            print('-' * 90)
            print('PATH:', k)
            if oldv is None:
                print('  !! 现值不存在（新键？）')
                print('  NEW:', newv[:400])
                continue
            if oldv == newv:
                print('  !! 无变化（空跑）')
                continue
            sm = difflib.SequenceMatcher(None, oldv, newv, autojunk=False)
            ops = [o for o in sm.get_opcodes() if o[0] != 'equal']
            print('  改动块数:', len(ops), ' 长度', len(oldv), '->', len(newv))
            for tag, i1, i2, j1, j2 in ops:
                ctxa = oldv[max(0, i1 - 60):i1]
                ctxb = oldv[i2:i2 + 60]
                print('   [%s] ...%s  ⟪OLD: %r⟫ ⟪NEW: %r⟫  %s...' % (
                    tag, ctxa[-60:], oldv[i1:i2], newv[j1:j2], ctxb))
            if env:
                print('  EN(片段可用于人工核对) len=%d' % len(env))


if __name__ == '__main__':
    main()
