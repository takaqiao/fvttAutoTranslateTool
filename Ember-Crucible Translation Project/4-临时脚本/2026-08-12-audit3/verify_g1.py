"""Verify the G1 batches: (1) each new leaf differs from the current CN only
inside attribute values, (2) re-running the G1 criterion over the patched text
yields zero findings, (3) show the target="_blank" drift leaves."""
import difflib, json, os, re, sys

sys.path.insert(0, r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\3-常用脚本\qa")
import scan_attr_text as S

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
SCRATCH = (r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt"
           r"\e57b5596-8975-4155-bc4b-c3126ad4aad5\scratchpad\audit3")
REPOS = [os.path.join(P, "1-Ember汉化插件"), os.path.join(P, "2-Crucible汉化插件")]
vocab = S.build_vocab(REPOS, 3)

cache = {}


def leaves(repo, pack):
    k = (repo, pack)
    if k not in cache:
        o = []
        S.walk(S.load_entries(os.path.join(repo, 'compendium', 'cn', pack)), [], o)
        cache[k] = dict(o)
    return cache[k]


bad = left = 0
for fn in sorted(os.listdir(os.path.join(SCRATCH, 'batches'))):
    if not fn.startswith('G1__'):
        continue
    _, tag, pack = fn[:-5].split('__')
    repo = os.path.join(P, "1-Ember汉化插件" if tag == 'ember' else "2-Crucible汉化插件")
    items = json.load(open(os.path.join(SCRATCH, 'batches', fn), encoding='utf-8'))
    lv = leaves(repo, pack + '.json')
    for path, new in items.items():
        old = lv[path]
        # (1) every differing region must lie inside a quoted attribute value
        spans = [(m.start(2), m.end(2)) for m in S.ATTR.finditer(old)]
        sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == 'equal':
                continue
            if old[i1:i2] == '目标' and new[j1:j2] == 'target':
                continue          # the deliberate attribute-NAME repair
            if not any(lo <= i1 and i2 <= hi for lo, hi in spans):
                bad += 1
                print(f'!! diff outside an attribute value: {pack} {path}')
                print(f'   old[{i1}:{i2}] = {old[i1:i2]!r}')
                print(f'   new[{j1}:{j2}] = {new[j1:j2]!r}')
        # (2) criterion must be clean after the patch
        for n, v, o in S.extract(new):
            if S.translatable(n, o) and S.classify(v, vocab) is not None:
                left += 1
                print(f'!! still English after patch: {pack} {path} {n}={v!r}')

print(f'diffs outside attribute values: {bad}')
print(f'visible-attribute English remaining after patch: {left}')

print()
print('===== target="_blank" drift (EN has it, CN dropped it) =====')
for repo in REPOS:
    en_d, cn_d = os.path.join(repo, 'compendium', 'en'), os.path.join(repo, 'compendium', 'cn')
    for f in S.packs_of(en_d):
        if not os.path.exists(os.path.join(cn_d, f)):
            continue
        eo, co = [], []
        S.walk(S.load_entries(os.path.join(en_d, f)), [], eo)
        S.walk(S.load_entries(os.path.join(cn_d, f)), [], co)
        cm = dict(co)
        for path, s in eo:
            c = cm.get(path)
            if not isinstance(c, str) or 'target=' not in s:
                continue
            for m in re.finditer(r'.{0,150}target="[^"]*".{0,40}', s):
                print(f'[{os.path.basename(repo)}/{f}] {path}')
                print(f'   EN {m.group(0)}')
            for m in re.finditer(r'<a[^>]*href="https?://[^"]*"[^>]*>.{0,60}', c):
                print(f'   CN {m.group(0)}')
