#!/usr/bin/env python3
"""G4 的补漏筛：token 判据看不见「上游整段重写但没动数字/专名」。

这里换个正交角度 —— **块结构**。对 changed 桶每条算三份块签名：

    旧英文 / 新英文 / 中文

若 `中文 == 旧英文 != 新英文`，中文百分百停在旧版上（BLOCK 闸只比中文和**新**英文，
两边碰巧都变了同样多块时它不响）。

另出一份 `far`：三者块数两两都不同、或中文块数与新英文差 ≥2 的，人工看一眼。
"""
import argparse, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g4_sieve import load_json, leaves, norm, baseline_packs, TAG, CJK  # noqa

BLOCK = re.compile(r'<(p|li|h[1-6]|dt|dd|td|section)\b', re.I)


def sig(s):
    return tuple(m.group(1).lower() for m in BLOCK.finditer(s))


ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
ap.add_argument('--baseline', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()

en_dir = os.path.join(a.repo, 'compendium', 'en')
cn_dir = os.path.join(a.repo, 'compendium', 'cn')
same_as_old, far, total = [], [], 0
for pack, oldpath in baseline_packs(a.baseline).items():
    cur = os.path.join(en_dir, pack)
    if not os.path.exists(cur):
        continue
    o, n, c = {}, {}, {}
    leaves(load_json(oldpath).get('entries', {}), [], o)
    leaves(load_json(cur).get('entries', {}), [], n)
    cnp = os.path.join(cn_dir, pack)
    if os.path.exists(cnp):
        leaves(load_json(cnp).get('entries', {}), [], c)
    for path, new_en in n.items():
        old_en = o.get(path)
        if old_en is None or norm(old_en) == norm(new_en):
            continue
        if len(TAG.sub('', new_en)) < 40:
            continue
        cn = c.get(path)
        if not (cn and CJK.search(cn)):
            continue
        total += 1
        so, sn, sc = sig(old_en), sig(new_en), sig(cn)
        rec = {'pack': pack, 'path': path, 'n_old': len(so), 'n_new': len(sn),
               'n_cn': len(sc), 'old_en': old_en, 'new_en': new_en, 'cn': cn}
        if so != sn and sc == so:
            same_as_old.append(rec)
        elif abs(len(sc) - len(sn)) >= 2:
            far.append(rec)

far.sort(key=lambda r: -abs(r['n_cn'] - r['n_new']))
print(f'{a.repo}  changed {total} -> 中文块签名==旧英文 {len(same_as_old)} · 中文块数偏离新英文>=2 {len(far)}')
for r in same_as_old[:20]:
    print(f'  [==OLD] {r["n_old"]}/{r["n_new"]}/{r["n_cn"]}  {r["path"][-70:]}')
for r in far[:20]:
    print(f'  [ FAR ] {r["n_old"]}/{r["n_new"]}/{r["n_cn"]}  {r["path"][-70:]}')
json.dump({'total': total, 'same_as_old': same_as_old, 'far': far},
          open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
