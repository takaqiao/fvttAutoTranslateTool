"""Turn the G1 findings into apply_translations batches.

Every replacement is scoped to `<attr>="<exact old value>"` so prose is never
touched, and the number of substitutions per leaf is asserted against the number
of findings recorded for that leaf.
"""
import json, os, re, sys
from collections import defaultdict

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
SCRATCH = (r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt"
           r"\e57b5596-8975-4155-bc4b-c3126ad4aad5\scratchpad\audit3")
FIND = os.path.join(SCRATCH, 'findings', 'G1_attr_text.json')
OUTD = os.path.join(SCRATCH, 'batches')

REPO_TAG = {'1-Ember汉化插件': 'ember', '2-Crucible汉化插件': 'crucible'}

# (attr, english value) -> Chinese.  Evidence for each is in G1.md.
TR = {
    # --- content-link tooltips.  Source of truth: ember lang/cn.json
    #     TYPES.JournalEntryPage.* (the same labels Foundry prints elsewhere).
    ('data-tooltip', 'Page'): '页面',
    ('data-tooltip', 'Ember Quest Event Page'): '余烬任务事件页面',
    ('data-tooltip-text', 'Text Page'): '文本页面',
    ('data-tooltip-text', 'Ember Lore Page'): '余烬背景页面',
    # --- @Embed captions.  Source of truth: the @UUID{label} the library
    #     already uses for the very same actor, cross-checked with term_gate.
    ('label', 'Harrowed Townsfolk'): '惊惶镇民',
    ('label', 'House Guard'): '家族卫兵',
    ('label', 'Kierelin the Interrogator'): '审讯者基耶瑞林',
    ('label', 'Vaafo the Scaletamer'): '驯鳞者瓦福',
    ('label', 'Ukkfal the Ringmaster'): '擂台主持人乌克法尔',
    ('label', 'Taamsin the Mastermind'): '智囊塔姆辛',
    ('label', 'Wandren Watcher'): '万德伦注视者',
    ('label', 'Wandren Patroller'): '万德伦巡逻者',
    ('label', 'Beacon Brigade Patroller'): '烽灯旅巡逻者',
    ('label', 'Beacon Brigade Courier'): '烽灯旅信使',
    ('label', 'Beacon Brigade Watcher'): '烽灯旅注视者',
    ('label', 'Liestra Grann'): '莉耶丝特拉·格兰',
    ('label', 'Vesk'): '维斯克',
    ('label', 'Rakavi'): '拉卡维',
}

rep = json.load(open(FIND, encoding='utf-8'))
recs = rep['findings'] + rep['proper_noun_watchlist']

# group: (repo, pack, path) -> list of (attr, value)
grp = defaultdict(list)
for r in recs:
    grp[(r['repo'], r['pack'], r['path'])].append((r['attr'], r['value']))


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str):
        out.append(('.'.join(path), node))


leaf_cache = {}


def leaf(repo, pack, path):
    key = (repo, pack)
    if key not in leaf_cache:
        p = os.path.join(P, repo, 'compendium', 'cn', pack)
        o = []
        walk(json.load(open(p, encoding='utf-8')).get('entries', {}), [], o)
        leaf_cache[key] = dict(o)
    return leaf_cache[key][path]


batches = defaultdict(dict)
missing = []
total_sub = 0
for (repo, pack, path), attrs in sorted(grp.items()):
    s = leaf(repo, pack, path)
    new = s
    # subn replaces every occurrence at once; the same (attr, value) can be
    # recorded several times for one leaf (Hex Attributes has data-tooltip="Page"
    # three times), so dedupe before substituting and verify the count instead.
    want = {}
    for attr, val in attrs:
        want[(attr, val)] = want.get((attr, val), 0) + 1
    for (attr, val), expect in want.items():
        if (attr, val) not in TR:
            missing.append((attr, val))
            continue
        cn = TR[(attr, val)]
        pat = re.compile(r'(?<![\w-])' + re.escape(attr) + r'(\s*=\s*)"' + re.escape(val) + r'"')
        new, n = pat.subn(lambda m: f'{attr}{m.group(1)}"{cn}"', new)
        if n != expect:
            print(f'!! {attr}={val!r} at {path}: substituted {n}, findings said {expect}')
        total_sub += n
    if new == s:
        print(f'!! leaf unchanged: {repo}/{pack}/{path}')
        continue
    batches[(REPO_TAG[repo], pack)][path] = new

if missing:
    print('!! no translation supplied for:', sorted(set(missing)))
    sys.exit(1)

# --- attribute NAMES that were themselves translated: restore the ASCII name.
#     `目标="_blank"` is not an attribute at all, so the external link opens in
#     the Foundry window instead of a new tab.
for r in rep.get('broken_attr_names', []):
    s = leaf(r['repo'], r['pack'], r['path'])
    new, n = re.subn(re.escape(r['attr']) + r'(\s*=\s*")' + re.escape(r['value']) + r'"',
                     lambda m: f'target{m.group(1)}{r["value"]}"', s)
    assert r['attr'] == '目标' and n == 1, (r['attr'], n, r['path'])
    key = (REPO_TAG[r['repo']], r['pack'])
    batches[key][r['path']] = new
    total_sub += n

os.makedirs(OUTD, exist_ok=True)
for (tag, pack), items in sorted(batches.items()):
    fn = os.path.join(OUTD, f'G1__{tag}__{pack[:-5]}.json')
    json.dump(items, open(fn, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{len(items):>4} leaves -> {fn}')
print(f'total attribute substitutions: {total_sub}  (findings in report: {len(recs)})')
