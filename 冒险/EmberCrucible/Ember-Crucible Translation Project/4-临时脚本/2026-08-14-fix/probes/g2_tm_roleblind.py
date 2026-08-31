"""G2 findings #3 / #5: leave-one-out check of fill_missing.py's TM lookup.

Replicates `fill_missing.py`'s index build and :133 lookup exactly, then for every
already-translated leaf drops that leaf's own vote and asks the tool what it would
write.  Compares BEFORE (current code) with AFTER (role-aware shape + role-keyed
fallback, no plain fallback).

Reported per variant:
  wrong   -- tool would write a Chinese string different from the human's
  recall  -- tool would produce any answer at all (upper bound on usefulness)
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

TM = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\3-常用脚本\tm'
sys.path.insert(0, TM)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fill_missing as F  # noqa: E402
from g2_lib import R1, R2  # noqa: E402

CJK = re.compile(r'[一-鿿]')


def role_of(path):
    """propagate_fix.py:39-48 的定义，逐字复制。"""
    for seg in reversed(path.split('.')):
        if not seg.isdigit():
            return seg
    return path


def shape_after(path):
    segs = [p for p in path.split('.') if p in F.STRUCT or p.isdigit()]
    r = role_of(path)
    if not segs or segs[-1] != r:
        segs.append(r)
    return '.'.join(segs)


def build():
    pairs = []
    for repo in (R1, R2):
        en_dir = os.path.join(repo, 'compendium', 'en')
        cn_dir = os.path.join(repo, 'compendium', 'cn')
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith('.json'):
                continue
            cnp = os.path.join(cn_dir, fn)
            if not os.path.exists(cnp):
                continue
            en = dict(F.walk(F.load(os.path.join(en_dir, fn))))
            cn = dict(F.walk(F.load(cnp)))
            for path, src in en.items():
                tgt = cn.get(path)
                if tgt and CJK.search(tgt):
                    pairs.append((fn, path, src, tgt))
    return pairs


def main():
    pairs = build()
    print(f'已译叶子 {len(pairs)}')

    tm_shape = defaultdict(Counter)
    tm_plain = defaultdict(Counter)
    tm_shape2 = defaultdict(Counter)
    tm_role = defaultdict(Counter)
    for _, path, src, tgt in pairs:
        tm_shape[(F.shape_of(path), src)][tgt] += 1
        tm_plain[src][tgt] += 1
        tm_shape2[(shape_after(path), src)][tgt] += 1
        tm_role[(role_of(path), src)][tgt] += 1

    def pick(counter, tgt):
        """扣掉本叶自己的一票后取多数派。"""
        c = Counter(counter)
        c[tgt] -= 1
        if c[tgt] <= 0:
            del c[tgt]
        if not c:
            return None
        return c.most_common(1)[0][0]

    stats = {'before': Counter(), 'after': Counter()}
    wrong_before, wrong_after = [], []
    for pack, path, src, tgt in pairs:
        # BEFORE: fill_missing.py:133
        b = pick(tm_shape[(F.shape_of(path), src)], tgt)
        if b is None:
            b = pick(tm_plain[src], tgt)
        # AFTER: role-aware shape, role-keyed fallback, no plain fallback
        a = pick(tm_shape2[(shape_after(path), src)], tgt)
        if a is None:
            a = pick(tm_role[(role_of(path), src)], tgt)

        for name, v, bucket in (('before', b, wrong_before), ('after', a, wrong_after)):
            if v is None:
                stats[name]['none'] += 1
            elif v == tgt:
                stats[name]['ok'] += 1
            else:
                if F.sig(v) != F.sig(src):
                    stats[name]['blocked_by_markup_gate'] += 1
                else:
                    stats[name]['WRONG'] += 1
                    bucket.append({'pack': pack, 'path': path, 'en': src[:80],
                                   'human': tgt[:80], 'tool': v[:80],
                                   'role': role_of(path)})

    for k in ('before', 'after'):
        print(f'{k:>7}: ' + '  '.join(f'{n}={stats[k][n]}' for n in
                                      ('ok', 'WRONG', 'blocked_by_markup_gate', 'none')))

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, 'g2_tm_roleblind.json'), 'w', encoding='utf-8') as f:
        json.dump({'stats': {k: dict(v) for k, v in stats.items()},
                   'wrong_before': wrong_before, 'wrong_after': wrong_after},
                  f, ensure_ascii=False, indent=1)
    print('\n仍错的样本（after）:')
    for w in wrong_after[:25]:
        print(f'  [{w["role"]}] {w["path"][:90]}\n     人={w["human"]}  工具={w["tool"]}')


if __name__ == '__main__':
    main()
