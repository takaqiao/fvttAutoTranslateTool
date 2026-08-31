"""Cross-validate our English extraction against Padhiver/Crucible-FR's.

Both were extracted from crucible 0.10.1 packs but by independent tooling, so
disagreements point at a real bug in one of them. Reports per pack:
  - entries only we have / only they have
  - translatable-string counts and char totals
  - field names we emit that they don't, and vice versa (top-level only)
"""
import json, os, re, sys
from collections import Counter

OURS = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\5-其他内容\english-baseline\crucible-0.10.1"
THEIRS = r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\9ed168a5-8f05-4b65-86dc-0ff0ecb4407e\scratchpad\repo_crucible_fr\compendium\en"
TAG = re.compile(r'<[^>]+>')


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def leaves(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from leaves(v)
    elif isinstance(o, list):
        for v in o:
            yield from leaves(v)
    elif isinstance(o, str) and o.strip():
        yield o


def stats(entries):
    n = ch = 0
    for s in leaves(entries):
        n += 1
        ch += len(TAG.sub(' ', s))
    return n, ch


print(f"{'pack':<34}{'ours':>7}{'theirs':>8}{'only-ours':>11}{'only-thrs':>11}"
      f"{'our str':>9}{'thr str':>9}{'our chars':>11}{'thr chars':>11}")

files = sorted(f for f in os.listdir(OURS) if f.endswith('.json') and not f.startswith('_'))
tot = [0, 0, 0, 0]
field_diff = {}

for fn in files:
    tp = os.path.join(THEIRS, fn)
    if not os.path.exists(tp):
        print(f"{fn:<34}{'--- not in Crucible-FR ---'}")
        continue
    a = load(os.path.join(OURS, fn))
    b = load(tp)
    ae, be = a.get('entries', {}), b.get('entries', {})
    only_a = sorted(set(ae) - set(be))
    only_b = sorted(set(be) - set(ae))
    an, ac = stats(ae)
    bn, bc = stats(be)
    tot = [tot[0] + an, tot[1] + bn, tot[2] + ac, tot[3] + bc]
    print(f"{fn:<34}{len(ae):>7}{len(be):>8}{len(only_a):>11}{len(only_b):>11}"
          f"{an:>9}{bn:>9}{ac:>11}{bc:>11}")
    # Key-naming difference (ours: name, theirs: id) is expected — not reported.

    # Crucible-FR keys entries by _id-ish slugs, we key by name. Re-key both
    # sides by the entry's own `name` value so they can actually be compared.
    def by_name(entries):
        out = {}
        for v in entries.values():
            if isinstance(v, dict) and isinstance(v.get('name'), str):
                out.setdefault(v['name'], v)
        return out

    na, nb = by_name(ae), by_name(be)
    shared = set(na) & set(nb)
    fa, fb = Counter(), Counter()
    char_gap = Counter()
    for k in shared:
        fa.update(na[k].keys())
        fb.update(nb[k].keys())
        for f in set(nb[k]) - set(na[k]):
            char_gap[f] += sum(len(TAG.sub(' ', s)) for s in leaves(nb[k][f]))
    missing = {k: v for k, v in fb.items() if k not in fa}
    extra = {k: v for k, v in fa.items() if k not in fb}
    if missing or extra or char_gap:
        d = field_diff.setdefault(fn, {})
        d['shared_entries'] = len(shared)
        if missing:
            d['fields_only_theirs'] = missing
        if extra:
            d['fields_only_ours'] = extra
        if char_gap:
            d['chars_we_miss_by_field'] = dict(char_gap.most_common(8))

print(f"{'TOTAL':<34}{'':>7}{'':>8}{'':>11}{'':>11}{tot[0]:>9}{tot[1]:>9}{tot[2]:>11}{tot[3]:>11}")

print("\n=== per-pack differences ===")
for fn, d in field_diff.items():
    print(f"\n-- {fn}")
    for k, v in d.items():
        if v:
            print(f"   {k}: {v}")
