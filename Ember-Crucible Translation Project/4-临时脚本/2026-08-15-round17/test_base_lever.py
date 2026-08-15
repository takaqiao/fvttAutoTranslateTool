#!/usr/bin/env python3
"""Is the base glossary a lever on `sameRoleMissingTail`? Run the real builder
with the base layer replaced by an EMPTY dict and compare the bucket counts.

`same_role_missing_tail` is derived from `conflicts_by_role`, which is derived
from `pairs`, which is populated ONLY inside `harvest()`. The base glossary is
read after that and merged in a later phase. If that reading is right, emptying
the base must leave the bucket at exactly 31.
"""
import io, json, os, sys

P = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
R = os.path.join(P, '4-临时脚本', '2026-08-15-round17')
BG = os.path.join(P, '3-常用脚本', 'tm', 'build_glossary.py')

empty = os.path.join(R, 'empty_base.json')
json.dump({}, io.open(empty, 'w', encoding='utf-8'))

src = io.open(BG, encoding='utf-8').read()
old = "base_path = r'C:\\Users\\Taka\\Desktop\\fvtt\\glossary_crucible_merged.json'"
assert old in src, 'base_path line not found -- builder changed shape'
src = src.replace(old, 'base_path = ' + repr(empty))

sys.argv = ['build_glossary.py', '--out-dir', os.path.join(R, 'nobase')]
exec(compile(src, BG, 'exec'), {'__name__': '__main__', '__file__': BG})

def bucket_counts(d):
    j = json.load(io.open(os.path.join(d, 'glossary_ec.disputes.json'), encoding='utf-8'))
    return {k: j[k]['count'] for k in
            ('crossRoleFormDifference', 'sameRoleMissingTail',
             'shippedTermInconsistency', 'unresolvedRoleTies')}

a = bucket_counts(os.path.join(R, 'base0'))
b = bucket_counts(os.path.join(R, 'nobase'))
io.open(os.path.join(R, 'base_lever_result.txt'), 'w', encoding='utf-8').write(
    f'with base (4788 terms): {a}\nwith EMPTY base        : {b}\nidentical: {a == b}\n')
print('with base :', a)
print('empty base:', b)
print('identical :', a == b)
