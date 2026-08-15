"""Variant B of the role definition: self-keyed containers get `<container>.*`.

A path segment that is *identical to the English source* is an entity key, not a
field name (`folders.General` = "General", `scenes.X.levels.Lake Jinro` =
"Lake Jinro").  For those the role is the container segment, which separates
`levels` from `notes` and `folders` from `categories` -- the pools that variant A
(propagate_fix's role_of verbatim) still merges.
"""
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from g2_tm_roleblind import F, build, role_of, shape_after  # noqa: E402


def role2(path, src):
    segs = [s for s in path.split('.') if not s.isdigit()]
    if not segs:
        return path
    if segs[-1] == src and len(segs) >= 2:
        return segs[-2] + '.*'
    return segs[-1]


def shape2(path, src):
    segs = [p for p in path.split('.') if p in F.STRUCT or p.isdigit()]
    r = role2(path, src)
    if not segs or segs[-1] != r:
        segs.append(r)
    return '.'.join(segs)


def pick(counter, tgt):
    c = Counter(counter)
    c[tgt] -= 1
    if c[tgt] <= 0:
        del c[tgt]
    return c.most_common(1)[0][0] if c else None


def main():
    pairs = build()
    ts, tr = defaultdict(Counter), defaultdict(Counter)
    for _, path, src, tgt in pairs:
        ts[(shape2(path, src), src)][tgt] += 1
        tr[(role2(path, src), src)][tgt] += 1

    st, wrong = Counter(), []
    for pack, path, src, tgt in pairs:
        a = pick(ts[(shape2(path, src), src)], tgt)
        if a is None:
            a = pick(tr[(role2(path, src), src)], tgt)
        if a is None:
            st['none'] += 1
        elif a == tgt:
            st['ok'] += 1
        else:
            st['WRONG'] += 1
            wrong.append({'role': role2(path, src), 'path': path,
                          'en': src[:70], 'human': tgt[:70], 'tool': a[:70]})
    print(dict(st))
    for w in wrong:
        print(f'  [{w["role"]}] {w["path"][:85]}\n     人={w["human"]}  工具={w["tool"]}')
    with open(os.path.join(HERE, 'g2_tm_role2.json'), 'w', encoding='utf-8') as f:
        json.dump({'stats': dict(st), 'wrong': wrong}, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
