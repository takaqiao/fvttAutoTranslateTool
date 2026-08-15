# -*- coding: utf-8 -*-
"""人物一致性镜头 E：人物条里的**人名**跨页译法是否一致。

对每一处 `EnglishName (Alignment, Ancestry, pronouns)`：
  EN 名 = 紧邻左括号之前的连续「首字母大写词 / 引号别名」串
  CN 名 = 紧邻左括号之前的连续中文（含·「」“”）串
按 EN 名分组，同一 EN 名出现 >=2 次而 CN 名不止一种 -> 报。

对齐方式：**按叶内代词 token 的出现次序 zip**（EN/CN token 数相等的叶才参与），
比按 <dt> 下标稳，因为嵌套 <dl> 会让 dt 下标错位。
"""
from __future__ import annotations
import json, os, re, sys
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_all
from pc_align import plain, PRONS

PR = re.compile(PRONS, re.I)
OPEN = '([（'
EN_NAME = re.compile(r'((?:[“"\'’]?[A-Z][A-Za-z\'’À-ɏ.-]*[”"\'’]?\s+){0,4}'
                     r'[“"\'’]?[A-Z][A-Za-z\'’À-ɏ.-]*[”"\'’]?)\s*$')
CN_NAME = re.compile(r'([一-鿿·“”"\'’·\-A-Za-z]{2,24})\s*$')
STOP = {'The', 'A', 'An', 'And', 'Or', 'By', 'Of', 'To', 'As', 'Is', 'Are', 'Locals',
        'Members', 'Key', 'Named', 'Called', 'With', 'From', 'For', 'In', 'On', 'At',
        'This', 'That', 'These', 'Those', 'It', 'He', 'She', 'They', 'Treat', 'Refer'}


def head_en(s):
    m = EN_NAME.search(s)
    if not m:
        return None
    toks = [t for t in m.group(1).split()]
    while toks and toks[0].strip('“”"\'’') in STOP:
        toks.pop(0)
    return ' '.join(toks) if toks else None


def main():
    groups = defaultdict(list)
    n = 0
    for repo, pack, path, en, cn in load_all():
        if not cn:
            continue
        pe, pc = plain(en), plain(cn)
        me, mc = list(PR.finditer(pe)), list(PR.finditer(pc))
        if not me or len(me) != len(mc):
            continue
        for a, b in zip(me, mc):
            # 找到该代词所属括号的左括号
            we = pe[max(0, a.start() - 130):a.start()]
            wc = pc[max(0, b.start() - 90):b.start()]
            ie = max(we.rfind(c) for c in OPEN)
            ic = max(wc.rfind(c) for c in OPEN)
            if ie < 0 or ic < 0:
                continue
            ne, nc_ = head_en(we[:ie].rstrip()), None
            m2 = CN_NAME.search(wc[:ic].rstrip())
            if m2:
                nc_ = m2.group(1).strip()
            if not ne or not nc_:
                continue
            n += 1
            groups[ne].append({'cn': nc_, 'repo': repo, 'pack': pack, 'path': path,
                               'ctx_en': we[-90:], 'ctx_cn': wc[-50:]})

    out = []
    for ne, occ in sorted(groups.items()):
        variants = Counter(o['cn'] for o in occ)
        if len(variants) > 1:
            out.append({'en': ne, 'variants': variants.most_common(),
                        'occ': occ})
    print(f'抽到人名对 {n}，不同 EN 名 {len(groups)}，中文有分歧的 {len(out)}')
    for r in out:
        print(f"\n### {r['en']}  ->  {r['variants']}")
        for o in r['occ']:
            print(f"    [{o['cn']}] {o['pack'][:14]} {o['path'][-58:]}")
            print(f"        EN …{o['ctx_en']}")
            print(f"        CN …{o['ctx_cn']}")
    json.dump(out, open('pc_names.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
