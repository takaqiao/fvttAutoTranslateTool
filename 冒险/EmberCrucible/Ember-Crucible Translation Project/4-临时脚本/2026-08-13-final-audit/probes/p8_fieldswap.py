#!/usr/bin/env python3
"""P8: sibling visibility-field contamination, sharpened.

P7's Jaccard was degenerate (empty anchor sets scored 1.0). Here the signature
is the ordered markup+digit fingerprint, which survives translation exactly
(the project's markup gate is green, so CN[F] must reproduce EN[F]'s tag
sequence). Two tests:

  T1 SWAP : sig(CN[F]) != sig(EN[F])  AND  sig(CN[F]) == sig(EN[G])
            (non-trivial signature only, len>=4, and sig(EN[F]) != sig(EN[G]))
  T2 DUP  : CN[F] == CN[G] verbatim while EN[F] != EN[G]
            -> the GM text and the player text became the same string
  T3 LEN  : CN[F] plain-text length wildly out of line with the per-object
            CN/EN ratio -> content shovelled between the two fields
"""
import sys, os, re, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gm_lens import REPOS, CJK

VIS_FIELDS = ['contentOverview', 'contentGamemaster', 'public', 'private',
              'overview', 'summary', 'exposition', 'text', 'description']
TAG = re.compile(r'<\s*(/?)([a-zA-Z][a-zA-Z0-9]*)([^>]*)>')
UUID = re.compile(r'@UUID\[([^\]]+)\]|@Embed\[([^\]]+)\]|\[\[([^\]]+)\]\]|src="([^"]+)"')
DIG = re.compile(r'\d+')


def sig(s):
    t = []
    for sl, n, at in TAG.findall(s):
        n = n.lower()
        m = re.search(r'class="([^"]*)"', at)
        t.append(('/' if sl else '') + n + ('.' + m.group(1) if m and not sl else ''))
    u = [x for g in UUID.findall(s) for x in g if x]
    plain = re.sub(r'<[^>]+>', '', s)
    return tuple(t), tuple(u), tuple(DIG.findall(plain))


def plainlen(s):
    return len(re.sub(r'\s+', '', re.sub(r'<[^>]+>', '', s)))


def collect(en, cn, path, out):
    if isinstance(en, dict):
        present = [k for k in en if k in VIS_FIELDS and isinstance(en[k], str) and en[k].strip()]
        if len(present) > 1 and isinstance(cn, dict):
            out.append(('.'.join(path), {k: en[k] for k in present},
                        {k: cn.get(k) for k in present}))
        for k, v in en.items():
            collect(v, cn.get(k) if isinstance(cn, dict) else None, path + [k], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            collect(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                    path + [str(i)], out)


groups = []
for rname, repo in REPOS.items():
    en_dir = os.path.join(repo, 'compendium', 'en')
    cn_dir = os.path.join(repo, 'compendium', 'cn')
    for pack in sorted(os.listdir(en_dir)):
        if not pack.endswith('.json') or pack.startswith('_'):
            continue
        cn_p = os.path.join(cn_dir, pack)
        if not os.path.exists(cn_p):
            continue
        en = json.load(open(os.path.join(en_dir, pack), encoding='utf-8')).get('entries', {})
        cn = json.load(open(cn_p, encoding='utf-8')).get('entries', {})
        o = []
        collect(en, cn, [], o)
        for p, ef, cf in o:
            groups.append((rname, pack, p, ef, cf))

print(f'objects with >1 visibility field: {len(groups)}')
t1 = t2 = t3 = 0
for rname, pack, p, ef, cf in groups:
    keys = list(ef)
    sigs = {k: sig(ef[k]) for k in keys}
    # T2 duplicate
    for i, k in enumerate(keys):
        for g in keys[i + 1:]:
            a, b = cf.get(k), cf.get(g)
            if a and b and a == b and ef[k] != ef[g]:
                t2 += 1
                print(f'\n[T2 DUP] {rname}/{pack} {p}: CN[{k}] == CN[{g}] but EN differ')
                print('   EN[%s]: %s' % (k, ef[k][:150]))
                print('   EN[%s]: %s' % (g, ef[g][:150]))
                print('   CN    : %s' % a[:150])
    # T1 swap
    for k in keys:
        c = cf.get(k)
        if not c or not CJK.search(c):
            continue
        sc = sig(c)
        if sc == sigs[k]:
            continue
        weight = len(sc[0]) + len(sc[1]) + len(sc[2])
        if weight < 4:
            continue
        for g in keys:
            if g == k or sigs[g] == sigs[k]:
                continue
            if sc == sigs[g]:
                t1 += 1
                print(f'\n[T1 SWAP] {rname}/{pack} {p}: CN[{k}] carries EN[{g}]\'s markup fingerprint')
                print('   EN[%s]: %s' % (k, ef[k][:200]))
                print('   EN[%s]: %s' % (g, ef[g][:200]))
                print('   CN[%s]: %s' % (k, c[:200]))
    # T3 length ratio outlier within the object
    tot_en = sum(plainlen(ef[k]) for k in keys)
    tot_cn = sum(plainlen(cf[k]) for k in keys if cf.get(k))
    if tot_en < 200 or not tot_cn:
        continue
    r = tot_cn / tot_en
    for k in keys:
        c = cf.get(k)
        if not c or plainlen(ef[k]) < 120:
            continue
        rk = plainlen(c) / plainlen(ef[k])
        if rk > 2.2 * r or rk < 0.42 * r:
            t3 += 1
            print(f'\n[T3 LEN] {rname}/{pack} {p} field={k}  obj-ratio={r:.2f} field-ratio={rk:.2f}'
                  f'  EN={plainlen(ef[k])} CN={plainlen(c)}')
            print('   EN:', ef[k][:160])
            print('   CN:', c[:160])
print(f'\nT1 swap={t1}  T2 dup={t2}  T3 len-outlier={t3}')
