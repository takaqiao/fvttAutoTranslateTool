# -*- coding: utf-8 -*-
"""Build a realistic Babele cn fixture from the raw dumps, then N corrupted copies.

CLEAN fixture rules (what a correct translation looks like):
  * every T-FROZEN name kept byte-exact English
  * everything else given a Chinese name (bilingual tail, per the project convention)
  * the 12 skill-stunts items given exactly the lang/cn.json value (no tail)
  * the two char crit tables' result descriptions translated but keeping the
    ': ' + '<br />' skeleton and the frozen cells in lockstep with lang/cn.json
  * actor rTables/cTables not written at all (the alienRollTableRef converter
    resolves them) -- plus one actor that DOES write it explicitly, correctly
"""
import json, os, re, shutil, sys, copy

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJ = r"C:\Users\Taka\Desktop\fvtt\Alien-RPG Translation Project"
DUMPS = os.path.join(PROJ, "6-工作区", "raw-dumps")
REG = json.load(open(os.path.join(PROJ, "7-其他内容", "DO-NOT-TRANSLATE.json"), encoding='utf-8'))
TMP = os.path.join(os.environ['TEMP'], 'alien-dnt-fixture')

PACKS = {}
for n in ['system', 'starterset', 'corerules']:
    PACKS[n] = list(json.load(open(os.path.join(DUMPS, n + '.json'), encoding='utf-8')).values())[0]

REPO_OF = {'system': '1-系统汉化插件', 'starterset': '2-新手包汉化插件', 'corerules': '3-核心书汉化插件'}
PACKFILE = {'system': 'alienrpg.alien-rpg-system.json',
            'starterset': 'alien-evolved-starterset.alien-evolved-starter-set.json',
            'corerules': 'alien-evolved-corerules.alien-evolved-core-rules.json'}

S = REG['sections']

FROZEN_ADV = {e['string'] for e in S['name_lookups']['entries'] if e['role'] == 'Adventure document name'}
FROZEN_JOURNAL = {e['string'] for e in S['name_lookups']['entries'] if 'JournalEntry' in e['role']}
FROZEN_SCENE = {e['string'] for e in S['name_lookups']['entries'] if e['role'].startswith('Scene')}
FROZEN_TABLE = {e['string'] for e in S['rolltable_names']['entries']}
PREFIX_TABLES = set(S['rolltable_names']['prefix_filters'][0]['current_matches'])
FROZEN_FOLDER = {e['string'] for e in S['folder_names']['entries']}
FROZEN_ITEM = {d['actual_name'] for e in S['item_names']['entries'] for d in e['documents']}
SKILL = {e['item_name_en']: e['lang_key'] for e in S['exact_match_to_lang']['entries']}
CRIT_META = S['crit_parse_lockstep']['tables']
CASES = S['crit_parse_lockstep']['cases']

# ------------------------------------------------------------------ the lang file
LANG_CN = {
    "ALIENRPG": {
        "Yes": "是", "None": "无", "OneRound": "一轮", "OneTurn": "一回合",
        "OneShift": "一班", "OneDay": "一天", "Permanent": "永久", "Shift": "一班",
        "SkillheavyMach": "重型机械", "SkillcloseCbt": "近身格斗", "Skillstamina": "耐力",
        "SkillrangedCbt": "远程战斗", "Skillmobility": "机动", "Skillpiloting": "驾驶",
        "Skillcommand": "指挥", "Skillmanipulation": "交涉", "SkillmedicalAid": "医疗救助",
        "Skillobservation": "观察", "Skillsurvival": "生存", "Skillcomtech": "通信技术",
    }
}
FLAT = {}
for k, v in LANG_CN['ALIENRPG'].items():
    FLAT['ALIENRPG.' + k] = v


def cn(name, prefix='中'):
    return '%s%s %s' % (prefix, len(name), name)


# ------------------------------------------------------------------ crit tables
STRIP_RX = re.compile(r"(<b>)|(<p>|)(<strong>)|(</b>)|(</p>)|(</strong>)", re.I)
BR_RX = re.compile(r"<br />", re.I)
SPLIT_RX = re.compile(r"[:] |<br>", re.I)


def split_like_actor(d):
    return SPLIT_RX.split(BR_RX.sub("<br>", STRIP_RX.sub("", d)))


def translate_crit_desc(table_name, rng, desc):
    """Rebuild the 5-row card in Chinese, honouring the lockstep for the frozen cells."""
    arr = split_like_actor(desc)
    if len(arr) < 10:
        return None                                   # the two upstream-defective rows: leave alone
    meta = CRIT_META[table_name]['rows_with_significant_tokens'].get(rng, {})
    injury = '伤势' + arr[1].strip()
    fatal = arr[3]
    for c in CASES:
        if c['index'] == 3 and c['en_cell'] == meta.get('fatal_en'):
            fatal = FLAT[c['lang_key']] + c['suffix_literal']
    if fatal == arr[3] and arr[3].strip() == 'No':
        fatal = '否 '
    tl = arr[5]
    for c in CASES:
        if c['index'] == 5 and c['en_cell'] == meta.get('time_limit_en'):
            tl = FLAT[c['lang_key']] + c['suffix_literal']
    eff = ('效果' + arr[7].strip()) if arr[7].strip() else arr[7]
    heal = arr[9]
    if meta.get('healing_en') == 'Permanent':
        heal = FLAT['ALIENRPG.Permanent']
    elif meta.get('healing_en') == 'Shift':
        heal = 'Shift'                                # T-FROZEN bare literal
    elif heal.startswith('[['):
        heal = re.sub(r'^(\[\[[0-9]d[0-9]+\]\])\s*(.*)$', lambda m: m.group(1) + ' 天', heal)
    return ('<p><strong>INJURY: </strong>%s<br /><strong>FATAL: </strong>%s<br />'
            '<strong>TIME LIMIT: </strong>%s<br /><strong>EFFECTS: </strong>%s<br />'
            '<strong>HEALING TIME: </strong>%s</p>' % (injury, fatal, tl, eff, heal))


# ------------------------------------------------------------------ build clean
def build(pack):
    adv = PACKS[pack]
    node = {"name": adv['name'] if adv['name'] in FROZEN_ADV else cn(adv['name'])}

    node['folders'] = {}
    for f in adv['folders']:
        node['folders'][f['name']] = f['name'] if f['name'] in FROZEN_FOLDER else cn(f['name'])

    node['journals'] = {}
    for j in adv['journal']:
        node['journals'][j['name']] = {
            "name": j['name'] if j['name'] in FROZEN_JOURNAL else cn(j['name']),
            "pages": {p['name']: {"name": cn(p['name'])} for p in j.get('pages', [])},
        }

    node['scenes'] = {}
    for s in adv['scenes']:
        node['scenes'][s['name']] = {"name": s['name'] if s['name'] in FROZEN_SCENE else cn(s['name'])}

    node['macros'] = {m['name']: {"name": cn(m['name'])} for m in adv['macros']}

    node['tables'] = {}
    for t in adv['tables']:
        nm = t['name']
        if nm in FROZEN_TABLE or nm in PREFIX_TABLES:
            newname = nm
        else:
            newname = cn(nm)
        entry = {"name": newname, "description": cn(t.get('description') or '')}
        if nm in CRIT_META:
            res = {}
            for r in t['results']:
                rng = '%s-%s' % (r['range'][0], r['range'][1])
                d = translate_crit_desc(nm, rng, r.get('description') or '')
                if d:
                    res[rng] = {"description": d}
            entry['results'] = res
        node['tables'][nm] = entry

    node['items'] = {}
    for it in adv['items']:
        nm = it['name']
        if nm in FROZEN_ITEM:
            newname = nm
        elif nm in SKILL:
            newname = FLAT[SKILL[nm]]
        else:
            newname = cn(nm)
        node['items'][nm] = {"name": newname}

    node['actors'] = {}
    for i, a in enumerate(adv['actors']):
        ent = {"name": cn(a['name'])}
        if a.get('items'):
            ent['items'] = {}
            for it in a['items']:
                ent['items'][it['name']] = {
                    "name": it['name'] if it['name'] in FROZEN_ITEM else cn(it['name'])}
        sysd = a.get('system', {})
        # one actor per pack writes the ref explicitly, correctly (= the translated table name)
        if i == 0 and isinstance(sysd.get('rTables'), str) and sysd['rTables']:
            v = sysd['rTables']
            tgt = node['tables'].get(v)
            ent['rollTable'] = tgt['name'] if tgt else v
        if i == 0 and isinstance(sysd.get('cTables'), str) and sysd['cTables']:
            ent['critTable'] = sysd['cTables'] if sysd['cTables'] == 'None' else \
                (node['tables'].get(sysd['cTables'], {}).get('name') or sysd['cTables'])
        node['actors'][a['name']] = ent

    return {"label": PACKFILE[pack][:-5], "entries": {adv['name']: node}}


def write_tree(root, mutate=None):
    if os.path.isdir(root):
        shutil.rmtree(root)
    for pack, repo in REPO_OF.items():
        d = os.path.join(root, repo, 'compendium', 'cn')
        os.makedirs(d, exist_ok=True)
        doc = build(pack)
        if mutate:
            mutate(pack, doc)
        json.dump(doc, open(os.path.join(d, PACKFILE[pack]), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
    ld = os.path.join(root, '1-系统汉化插件', 'lang')
    os.makedirs(ld, exist_ok=True)
    json.dump(LANG_CN, open(os.path.join(ld, 'cn.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    shutil.copy(os.path.join(PROJ, '1-系统汉化插件', 'lang', 'en.json'),
                os.path.join(ld, 'en.json'))


def entry(doc):
    return list(doc['entries'].values())[0]


MUTATIONS = {
    # --- scan_name_lookup_traps
    'M1-frozen-adventure': lambda p, d: (
        d['entries'].__setitem__('Alien RPG System', d['entries'].pop('Alien RPG System'))
        or entry(d).__setitem__('name', '异形 RPG 系统')) if p == 'system' else None,
    'M2-frozen-journal': lambda p, d: entry(d)['journals']['MU/TH/ER Instructions.'].__setitem__(
        'name', '母亲操作说明。') if p == 'system' else None,
    'M3-frozen-scene': lambda p, d: entry(d)['scenes']['Alien Evolved Core Rules'].__setitem__(
        'name', '异形进化版核心规则') if p == 'corerules' else None,
    'M4-frozen-table': lambda p, d: entry(d)['tables']['Panic Table'].__setitem__(
        'name', '恐慌表') if p == 'system' else None,
    'M5-frozen-folder': lambda p, d: entry(d)['folders'].__setitem__(
        'Alien Creature Tables', '异形生物表') if p == 'system' else None,
    'M6-frozen-item-packmule': lambda p, d: entry(d)['items']['Pack Mule'].__setitem__(
        'name', '驮马 Pack Mule') if p == 'corerules' else None,
    'M7-frozen-item-takecontrol': lambda p, d: entry(d)['actors']['EV - ANDROID, COVERT']['items'][
        'Take Control'].__setitem__('name', '掌控 Take Control') if p == 'corerules' else None,
    'M8-prefix-lost': lambda p, d: entry(d)['tables']['Critical Injuries on Xenomorphs'].__setitem__(
        'name', '异形重伤表 Critical Injuries on Xenomorphs') if p == 'corerules' else None,
    'M9-exact-tail': lambda p, d: entry(d)['items']['Heavy Machinery'].__setitem__(
        'name', '重型机械 Heavy Machinery') if p == 'system' else None,
    'M10-table-key-renamed': lambda p, d: entry(d)['tables'].__setitem__(
        '异形 - 破胸体攻击', entry(d)['tables'].pop('EV - Chestburster Attacks')) if p == 'corerules' else None,
    'M11-sentinel-translated': lambda p, d: entry(d)['actors'][
        'EV - Adult Neomorph (Stage V)'].__setitem__('critTable', '无') if p == 'corerules' else None,
    'M12-explicit-ref-drift': lambda p, d: entry(d)['actors'][
        'EV - Adult Neomorph (Stage V)'].__setitem__('rollTable', '新形体攻击表') if p == 'corerules' else None,
    # --- scan_crit_lockstep
    'C1-yes-ascii-hyphen': lambda p, d: _crit_swap(
        d, p, 'Critical injuries', '52-52', '是, \u20131 ', '是, -1 '),
    'C2-timelimit-drift': lambda p, d: _crit_swap(
        d, p, 'Critical injuries', '44-44', '一天 ', '一日 '),
    'C3-permanent-drift': lambda p, d: _crit_swap(
        d, p, 'Critical injuries', '54-54', '永久', '永久性'),
    'C4-shift-translated': lambda p, d: _crit_swap(
        d, p, 'EV - Critical Injuries', '14-14', 'Shift', '一班'),
    'C5-fullwidth-colon': lambda p, d: _crit_raw(
        d, p, 'Critical injuries', '45-45', lambda s: s.replace('FATAL: ', 'FATAL\uff1a')),
    'C6-roll-shape': lambda p, d: _crit_raw(
        d, p, 'EV - Critical Injuries', '44-44', lambda s: s.replace('[[1d6]]', '（[[1d6]]）')),
    'C7-lang-padded': None,     # handled separately (mutates lang, not the packs)
}


def _crit_swap(doc, pack, table, rng, old, new):
    if pack not in ('corerules',):
        return
    e = entry(doc)['tables'].get(table)
    if not e or rng not in e.get('results', {}):
        raise SystemExit('fixture: %s %s missing' % (table, rng))
    d = e['results'][rng]['description']
    if old not in d:
        raise SystemExit('fixture: %r not in %r' % (old, d))
    e['results'][rng]['description'] = d.replace(old, new)


def _crit_raw(doc, pack, table, rng, fn):
    if pack not in ('corerules',):
        return
    e = entry(doc)['tables'][table]
    e['results'][rng]['description'] = fn(e['results'][rng]['description'])


if __name__ == '__main__':
    write_tree(os.path.join(TMP, 'clean'))
    print('clean ->', os.path.join(TMP, 'clean'))
    for name, mut in MUTATIONS.items():
        root = os.path.join(TMP, name)
        if mut is None:
            write_tree(root)
            lp = os.path.join(root, '1-系统汉化插件', 'lang', 'cn.json')
            l = json.load(open(lp, encoding='utf-8'))
            l['ALIENRPG']['Yes'] = '是 '            # trailing space
            l['ALIENRPG']['Permanent'] = None       # null value written explicitly
            json.dump(l, open(lp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        else:
            write_tree(root, mut)
        print('%-28s -> %s' % (name, root))
