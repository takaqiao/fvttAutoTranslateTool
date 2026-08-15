#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build round-9 t1_2 batches from the shard + a hand-made decision table."""
import json, os, sys, collections

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
SHARD = os.path.join(ROOT, r'4-临时脚本\2026-08-13-round9\shards\t1_2.json')
OUT = os.path.join(ROOT, r'4-临时脚本\2026-08-13-round9\batches')
SLUG = 't1_2'

d = json.load(open(SHARD, encoding='utf-8'))


def V(g, i):
    return g['variants'][i]['cn']


def build(g, gid):
    """return (value, paths) or None to skip"""
    v = lambda i: V(g, i)

    # ---- plain "take variant i" ----
    plain = {
        0: 1, 1: 0, 2: 0, 9: 1, 12: 0, 57: 0, 63: 0, 64: 3, 66: 1, 68: 11,
        69: 11, 71: 4, 73: 8, 74: 4, 75: 3, 76: 1, 78: 2, 79: 0, 80: 4,
        81: 3, 82: 0, 83: 0, 84: 4, 85: 1, 86: 1, 87: 0, 88: 2, 89: 0,
        90: 2, 91: 1, 92: 3, 95: 3, 96: 3, 97: 3, 98: 3, 99: 3, 100: 2,
        101: 3, 102: 2, 103: 2, 104: 0, 105: 0, 106: 2, 107: 2, 108: 0,
        109: 1, 110: 2, 111: 2, 112: 2, 113: 2, 115: 1, 116: 0, 117: 1,
        118: 1, 119: 1, 120: 0, 122: 1, 123: 1, 124: 0, 125: 1, 126: 1,
        127: 0, 128: 0, 130: 0,
    }
    if gid in plain:
        return v(plain[gid]), None

    # ---- derived from a variant with a targeted correction ----
    if gid == 65:   # "living matter" -> 活体物质 (v1 said 生物材料)
        t = v(1)
        assert '负责生物材料' in t
        return t.replace('负责生物材料', '负责活体物质'), None
    if gid == 67:   # stray space before the comma
        t = v(6)
        assert '@Condition[restrained] ，' in t
        return t.replace('@Condition[restrained] ，', '@Condition[restrained]，'), None
    if gid == 70:
        t = v(8)
        assert '该生物因其处于<strong>易伤</strong>的某种伤害类型而失能。' in t
        t = t.replace('该生物因其处于<strong>易伤</strong>的某种伤害类型而失能。',
                      '该生物因一种它<strong>易伤</strong>的伤害类型而失能。')
        assert '执行“恢复”动作' in t
        return t.replace('执行“恢复”动作', '执行恢复动作'), None
    if gid == 72:   # "if your target is hit" mistranslated as "if it is your target"
        t = v(0)
        assert '不过如果你的目标是它，它会变为@Condition[restrained]' in t
        return t.replace('不过如果你的目标是它，它会变为@Condition[restrained]',
                         '不过若命中目标，目标会陷入@Condition[restrained]'), None
    if gid == 93:   # ability check -> 属性检定 (库内 204:14)
        t = v(3)
        assert '能力检定' in t
        return t.replace('能力检定', '属性检定'), None
    if gid == 94:   # <em>Wisdom Saving Throw</em> -> 感知豁免
        t = v(3)
        assert '<em>感知豁免检定</em>' in t
        return t.replace('<em>感知豁免检定</em>', '<em>感知豁免</em>'), None
    if gid == 121:  # 无异 -> 无兆 (parallels 吉兆/凶兆)
        t = v(0)
        assert '无异' in t
        return t.replace('无异', '无兆'), None

    # ---- fully hand-written ----
    if gid == 27:
        return ('<p>当被金属制成的近战武器击中时，软泥怪会尝试吞噬该武器本身，并可能将其破坏。</p>'
                '<p>此次攻击为<strong>无害</strong>，但若成功，用于此次攻击的武器的<strong>品质</strong>'
                '会降低一阶。若为大成功，或该武器已是<strong>粗糙</strong>，则该武器会彻底'
                '<strong>破碎</strong>。附魔武器可免受此效果影响。</p>'), None
    if gid == 77:
        return ('<p>将符文的精华凝聚成一道横扫或碎裂的弧形，以打击近距离内的多个目标。'
                '若与大地配合使用，这可能会沿弧线掷出一片碎石；而与念力配合使用，'
                '则可能会在多个敌人身上切开一道裂痕。</p>'
                '<p>扇形手势基于<strong>智力</strong>进行成长，目标为一个120度、6英尺的锥形区域，'
                '并在攻击成功时造成6点基础伤害。</p>'), None
    if gid == 114:
        return ('<p>你用主手武器佯攻，对目标的<strong>反射</strong>防御进行一次<strong>欺瞒</strong>攻击。'
                '你立即用你的<strong>副手</strong>武器进行一次<strong>打击</strong>。</p>'
                '<p>如果你的欺瞒成功，你的打击获得<strong>+2 恩惠骰</strong>，'
                '并额外造成<strong>+6 伤害</strong>。</p>'), None
    if gid == 129:
        return ('<p>当你用一个<strong>法术</strong>对一个你能看见的生物造成伤害时，'
                '你随后可以进行一次针对其<strong>强韧</strong>防御的<strong>奥术</strong>技能攻击，'
                '以窃取其部分生命精华。成功时，你恢复等同于该法术所造成伤害数值的'
                '<strong>生命值</strong>。</p>'), None

    # ---- partial: only the leaves that violate their field-role convention ----
    if gid == 4:
        return '奥拉 Aura', [['1-Ember汉化插件', 'ember.character.json', 'folders.Aura']]
    if gid == 31:
        return '先见反照 Prescient Reflection', [[
            '1-Ember汉化插件', 'ember.adventure.json',
            'entries.Ember Early Access.actors.Sajor Velex.items.Prescient Reflection.name']]
    if gid == 55:
        return '宝藏', [['1-Ember汉化插件', 'ember.crucible-items.json', 'folders.Treasure']]
    return None


def batch_path(p):
    if p.startswith('entries.'):
        return p[len('entries.'):]
    if p.startswith('folders.'):
        return '(folders).' + p[len('folders.'):]
    raise SystemExit('unhandled root: ' + p)


batches = collections.defaultdict(dict)
n_unified = n_skipped = n_leaves = 0
for gid, g in enumerate(d):
    r = build(g, gid)
    if r is None:
        n_skipped += 1
        continue
    value, only = r
    n_unified += 1
    for var in g['variants']:
        for p in var['paths']:
            if only is not None and list(p) not in [list(x) for x in only]:
                continue
            if var['cn'] == value:
                continue          # already correct
            batches[(p[0], p[1])][batch_path(p[2])] = value
            n_leaves += 1

os.makedirs(OUT, exist_ok=True)
files = []
for (repo, pack), items in sorted(batches.items()):
    tag = '1' if repo.startswith('1') else '2'
    fn = os.path.join(OUT, f'{SLUG}.{tag}.{pack}')
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
        f.write('\n')
    files.append((repo, pack, fn, len(items)))
    print(f'{len(items):5d}  {fn}')
print(f'\ngroups unified {n_unified} / skipped {n_skipped} / leaves {n_leaves}')
json.dump([[a, b, c, n] for a, b, c, n in files],
          open(os.path.join(OUT, f'_{SLUG}.manifest.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
