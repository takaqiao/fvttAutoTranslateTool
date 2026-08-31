# -*- coding: utf-8 -*-
"""F2 batch builder.

Every rule is (finding_tag, path, kind, arg, expected_hits).
kind 'sub'   -> literal replace, must hit exactly `expected` times
kind 're'    -> regex replace via callable, must hit exactly `expected` times
Rules are accumulated per (repo, pack, path); the final value of each leaf is
written once, so several findings touching one leaf can never clobber
each other.
"""
import json, os, re, sys
import f2_lib as L

OUT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-14-fix\batches"
EM = 'EM'; CR = 'CR'
EA = 'Ember Early Access.'
J = EA + 'journals.'
T = EA + 'tables.'

RULES = []          # (tag, repoTag, packs, path, kind, arg, expected)


def rule(tag, repotag, packs, path, kind, arg, expected):
    RULES.append((tag, repotag, packs, path, kind, arg, expected))


TWIN = ['ember.adventure.json', 'ember.crucible-adventure.json']
CRUC_ONLY = ['ember.crucible-adventure.json']

# ---------------------------------------------------------------- finding 1
rule('round_turn', EM, CRUC_ONLY,
     EA + 'actors.Kali Andrella.items.Control Water.description',
     'sub', ('接下来的一个回合中缓慢回填沟渠', '接下来一整轮中缓慢回填沟渠'), 1)

# --------------------------------------------------------------- finding 16
rule('typo', EM, CRUC_ONLY,
     EA + "actors.Vhismara's Claw.biography.private",
     'sub', ('这也许是真的，也也许只是', '这也许是真的，也许只是'), 1)

# ---------------------------------------------------------------- finding 7
for p, n in [(J + 'To Fall and Fall Again.pages.Savage Descent.text', 3),
             (T + 'Climbing Hazards.description', 1)]:
    rule('group_check', EM, TWIN, p, 'sub', ('群体检定', '团队检定'), n)
for p, n in [(J + 'Ancient Paths.pages.Missing Scouts.text', 2),
             (J + 'Crumbling Sanctuary.pages.The Blockaded Bridge.text', 4),
             (J + "Marlstone Manor.pages.Hephiss' Chambers.text", 1),
             (J + 'Kalion Stadium Underworks.pages.Area Overview.text', 1)]:
    rule('group_check', EM, TWIN, p, 'sub', ('团体检定', '团队检定'), n)

rule('group_check', CR, ['crucible.macros.json'], 'Configure Group Check.name',
     'sub', ('配置群组检定 Configure Group Check', '配置团队检定 Configure Group Check'), 1)
# crucible.rules Skill Checks: EN says "group" everywhere; UI lang says 团队.
rule('group_check', CR, ['crucible.rules.json'], 'Exploration.pages.Skill Checks.text',
     'sub', ('群体', '团队'), 18)
rule('group_check', CR, ['crucible.rules.json'], 'Exploration.pages.Skill Checks.text',
     'sub', ('群组检定', '团队检定'), 1)

# --------------------------------------------------------------- finding 15
for p in [J + 'Repurposed Quarry.pages.Gravel Chute.text',
          J + 'Spellbreaker Tower.pages.Sky Cells.text',
          J + 'Spellbreaker Tower.pages.Rickety Platform.text']:
    rule('hazard_name', EM, TWIN, p, 'sub', ('坠落危险', '坠落危害'), 1)

# ---------------------------------------------------------------- finding 8
rule('secondary_ability', CR, ['crucible.rules.json'],
     'Character Creation.pages.Ancestry.text', 'sub', ('次要属性', '副属性'), 1)
rule('secondary_ability', CR, ['crucible.rules.json'],
     'Character Creation.pages.Background and Ability Scores.text',
     'sub', ('次要属性', '副属性'), 2)

# ---------------------------------------------------------------- finding 9
rule('kinesis_grammar', CR, ['crucible.affixes.json'], 'Kinesis Potency.description',
     'sub', ('<p>使使用<strong>念力</strong>符文的法术的<strong>附魔加值</strong>，按此后缀的每<strong>阶</strong>提高 1。</p>',
             '<p>使用<strong>念力</strong>符文的法术，其<strong>附魔加值</strong>每有1个该后缀的<strong>阶</strong>便提高1。</p>'), 1)

# --------------------------------------------------------------- finding 17
rule('halfwidth_comma', CR, ['crucible.rules.json'], 'Equipment.pages.Weapons.text',
     'sub', ('<td><p>力量,敏捷</p></td>', '<td><p>力量、敏捷</p></td>'), 4)

# --------------------------------------------------------------- finding 10
# numeral before &Reference[exhaustion] -> Arabic, no space, macro flush.
CNNUM = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6',
         '七': '7', '八': '8', '九': '9', '十': '10'}
EXH = re.compile(r'(?<!第)([一二三四五六七八九十]|[0-9]+)\s*级\s*(&(?:amp;)?[Rr]eference\[[Ee]xhaustion\])')


def exh_fix(m):
    n = CNNUM.get(m.group(1), m.group(1))
    return f'{n}级{m.group(2)}'


EXH_LEAVES_TWIN = [
    (J + 'Yakoshta Mine.pages.Waterfall Bridge.text', 1),
    (J + 'The Winding Trail.pages.Clearing the Rubble.text', 1),
    (J + 'Chapter 1 Events.pages.Time for a Swim.text', 1),
    (J + "Chapter 2 Events.pages.The Giants' Span.text", 1),
    (J + 'To Fall and Fall Again.pages.Savage Descent.text', 5),
    (J + 'To Fall and Fall Again.pages.Tormented Threads.text', 1),
    (J + 'Oldcraft Lodge.pages.The Group of Wagons.text', 1),
    (J + 'Toothbreaker Hideout.pages.Kitchen.text', 1),
    (J + 'The Expedition Challenge.pages.Reclaiming the Concourse.text', 1),
    (J + 'The Expedition Challenge.pages.Amazing Brambles.text', 2),
    (J + 'Signal of Intent.pages.Well Enough Alone.text', 1),
    (J + 'Redwalk Ramble.pages.Main Plaza - West.text', 1),
    (T + 'Climbing Hazards.results.1-1.description', 1),
    (T + 'Climbing Hazards.results.2-2.description', 1),
    (T + 'Climbing Hazards.results.3-3.description', 1),
    (T + 'Climbing Hazards.results.4-4.description', 1),
    (T + 'Rushing Water Perils.results.1-1.description', 1),
    (T + 'Rushing Water Perils.results.2-2.description', 1),
    (T + 'Rushing Water Perils.results.3-3.description', 1),
    (T + 'Rushing Water Perils.results.4-4.description', 1),
]
for p, n in EXH_LEAVES_TWIN:
    rule('exhaustion_num', EM, TWIN, p, 're', (EXH, exh_fix), n)
for p, n in [(EA + 'actors.Harrower.items.Withering Touch.description.private', 1),
             (EA + 'actors.Sanguinary Warden.items.Ruby.description', 1),
             (EA + 'items.Rapture.description', 1),
             (T + 'Kaleidoscope Crystal Effects.results.1-1.description', 1)]:
    rule('exhaustion_num', EM, CRUC_ONLY, p, 're', (EXH, exh_fix), n)
rule('exhaustion_num', EM, ['ember.adventure.json'],
     EA + 'actors.Corrupted Cadrithor.items.Grave Mark.effects.Exhaustion.description',
     're', (EXH, exh_fix), 1)
rule('exhaustion_num', EM, ['ember.dnd5e-effects.json'], 'Kaleidoscopic Fatigue.description',
     're', (EXH, exh_fix), 1)
# second pass: the same set writes 「承受 1级力竭」 in 5 leaves and 「承受1级力竭」 in
# the other 20; library-wide 汉字+数字+级 is 394 no-space vs 21 with-space.
GAP = re.compile(r'([一-鿿])[ \t]+([0-9]+级&(?:amp;)?[Rr]eference\[[Ee]xhaustion\])')
for p in [J + 'Chapter 1 Events.pages.Time for a Swim.text',
          J + 'To Fall and Fall Again.pages.Tormented Threads.text',
          J + 'Oldcraft Lodge.pages.The Group of Wagons.text']:
    rule('exhaustion_num', EM, TWIN, p, 're', (GAP, r'\1\2'), 1)
rule('exhaustion_num', EM, CRUC_ONLY,
     EA + 'actors.Harrower.items.Withering Touch.description.private',
     're', (GAP, r'\1\2'), 1)
rule('exhaustion_num', EM, ['ember.dnd5e-effects.json'], 'Kaleidoscopic Fatigue.description',
     're', (GAP, r'\1\2'), 1)

# --------------------------------------------------------------- finding 18
SPACEP = re.compile(r'(\]\]|\})[ \t]+([，。：；、！？])')
SPACE_LEAVES_TWIN = [
    (J + 'Ancient Paths.pages.Missing Scouts.text', 1),
    (J + 'The Book Of Tales.pages.Down by the River.text', 1),
    (J + 'A Brush With Death.pages.Delivered from Evil.text', 2),
    (J + 'A Brush With Death.pages.Trampled Florescence.text', 1),
    (J + 'Ancestries.pages.Vrjnhar.text', 1),
    (J + 'Ancestries.pages.Signborn.text', 1),
    (J + 'To Fall and Fall Again.pages.Torn Veil.text', 1),
    (J + "Chamber of Agaseros.pages.Meri's Garden.text", 2),
    (J + 'Forgotten Cistern.pages.Sunken Entrance.text', 1),
    (J + 'Burial Grounds.pages.Crematorium Gate.text', 1),
    (J + 'The Expedition Challenge.pages.An Auspicious Acquaintance.text', 1),
    (J + 'The Expedition Challenge.pages.The Challenge Begins.text', 1),
    (J + 'Smoldering Cinders.pages.A Message from Sin.text', 1),
    (J + 'Diplomatic Impunity.pages.Parcels & Pirates.text', 3),
]
for p, n in SPACE_LEAVES_TWIN:
    rule('space_fw_punct', EM, TWIN, p, 're', (SPACEP, r'\1\2'), n)
for p, n in [(EA + 'actors.Sionia.items.Teleport.description', 1),
             (EA + 'actors.Eveis Brightstone.items.ssppC6vLljBySxHc.description', 1)]:
    rule('space_fw_punct', EM, CRUC_ONLY, p, 're', (SPACEP, r'\1\2'), n)


# ------------------------------------------------------------------ engine
REPO = {EM: L.EMBER, CR: L.CRUC}


def main():
    work = {}     # (repotag, pack) -> {path: value}
    stats = {}
    errors = []
    caches = {}
    for tag, rt, packs, path, kind, arg, expected in RULES:
        for pack in packs:
            key = (rt, pack)
            if key not in caches:
                caches[key] = L.cnmap(REPO[rt], pack)
            src = caches[key]
            if path not in src:
                errors.append(f'MISSING LEAF {rt}/{pack} :: {path}  ({tag})')
                continue
            bucket = work.setdefault(key, {})
            cur = bucket.get(path, src[path])
            if kind == 'sub':
                old, new = arg
                hits = cur.count(old)
                if hits != expected:
                    errors.append(f'COUNT {tag} {pack} :: {path} :: {old!r} '
                                  f'expected {expected} got {hits}')
                    continue
                nv = cur.replace(old, new)
            else:
                pat, rep = arg
                hits = len(pat.findall(cur))
                if hits != expected:
                    errors.append(f'COUNT {tag} {pack} :: {path} :: regex '
                                  f'expected {expected} got {hits}')
                    continue
                nv = pat.sub(rep, cur)
            if nv == cur:
                # normalisation rules legitimately hit leaves that are already
                # in the target form; only a literal `sub` rule that changes
                # nothing is a mistake in the rule table.
                if kind == 'sub':
                    errors.append(f'NOOP {tag} {pack} :: {path}')
                continue
            # invariants
            if nv.count('id="') != src[path].count('id="'):
                errors.append(f'ID-LOSS {tag} {pack} :: {path}')
                continue
            bucket[path] = nv
            stats[tag] = stats.get(tag, 0) + expected

    for e in errors:
        print('!!', e)
    print()
    for k, v in sorted(stats.items()):
        print(f'{k:20s} {v}')
    print()
    os.makedirs(OUT, exist_ok=True)
    slug = 'F2'
    written = []
    for (rt, pack), bucket in sorted(work.items()):
        n = '1' if rt == EM else '2'
        fp = os.path.join(OUT, f'{slug}.{n}.{pack}')
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(bucket, f, ensure_ascii=False, indent=1)
            f.write('\n')
        written.append((REPO[rt], pack, fp, len(bucket)))
        print(f'wrote {fp}  ({len(bucket)} leaves)')
    return errors, written


if __name__ == '__main__':
    errs, _ = main()
    sys.exit(1 if errs else 0)
