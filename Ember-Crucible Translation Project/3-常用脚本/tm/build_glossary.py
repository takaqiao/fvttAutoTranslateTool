#!/usr/bin/env python3
"""Build the project glossary `glossary_ec.json`.

Layers, highest priority last:
  1. base        - glossary_crucible_merged.json (4602 已裁决译名)
  2. harvested   - EN->CN pairs mined from the existing shipped translations by
                   walking the English baseline and the CN file in parallel
  3. (report)    - English terms present in the current baselines with no CN at
                   all, written to glossary_ec.pending.json for translators

Outputs (into 5-其他内容/glossary/):
  glossary_ec.json             flat {en: cn}   <- consumed by QA / TM tooling
  glossary_ec.provenance.json  per-term source + conflicts
  glossary_ec.pending.json     English terms still needing a Chinese name

Usage:
  python build_glossary.py [--project <dir>]
"""
from __future__ import annotations
import argparse
import json
import os
import re
from collections import defaultdict

CJK = re.compile(r'[\u4e00-\u9fff]')
HTML = re.compile(r'<[^>]+>')
# A glossary term is a short, tag-free, non-sentence string: names and labels.
MAX_TERM_LEN = 60


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def is_term(s: str) -> bool:
    """Name-like enough to belong in a glossary."""
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s or len(s) > MAX_TERM_LEN:
        return False
    if HTML.search(s) or '\n' in s:
        return False
    # Skip sentences: a term rarely ends in punctuation or has many words.
    if s[-1] in '.!?:;,':
        return False
    return len(s.split()) <= 8


def walk_pairs(en, cn, out):
    """Walk two parallel structures, yielding (en_string, cn_string)."""
    if isinstance(en, dict) and isinstance(cn, dict):
        for k, v in en.items():
            if k in cn:
                walk_pairs(v, cn[k], out)
    elif isinstance(en, list) and isinstance(cn, list):
        for a, b in zip(en, cn):
            walk_pairs(a, b, out)
    elif isinstance(en, str) and isinstance(cn, str):
        out.append((en, cn))


def harvest(en_dir, cn_dir, label, pairs, pending):
    """Mine EN->CN term pairs from a baseline/translation directory pair."""
    if not (os.path.isdir(en_dir) and os.path.isdir(cn_dir)):
        return 0
    found = 0
    for fn in sorted(f for f in os.listdir(en_dir) if f.endswith('.json') and not f.startswith('_')):
        cn_path = os.path.join(cn_dir, fn)
        try:
            en_doc = load(os.path.join(en_dir, fn))
        except Exception:
            continue
        en_entries = en_doc.get('entries', {})

        # Every short EN string is a term candidate, translated or not.
        cand = []
        walk_pairs(en_entries, en_entries, cand)
        for s, _ in cand:
            if is_term(s):
                pending.setdefault(s, set()).add(f'{label}:{fn}')

        if not os.path.exists(cn_path):
            continue
        try:
            cn_doc = load(cn_path)
        except Exception as e:
            print(f'  ! unreadable CN file {fn}: {e}')
            continue

        got = []
        walk_pairs(en_entries, cn_doc.get('entries', {}), got)
        for en_s, cn_s in got:
            if not is_term(en_s) or not CJK.search(cn_s):
                continue
            pairs[en_s][cn_s] += 1
            found += 1
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')))
    a = ap.parse_args()
    P = a.project
    OUT = os.path.join(P, '5-其他内容', 'glossary')
    BASE_DIR = os.path.join(P, '5-其他内容', 'english-baseline')

    # ---- layer 1: base glossary -------------------------------------------
    base_path = r'C:\Users\Taka\Desktop\fvtt\glossary_crucible_merged.json'
    base_raw = load(base_path)
    base = {k: v for k, v in base_raw.items() if isinstance(v, str) and CJK.search(v)}
    print(f'base   : {len(base)} terms  ({os.path.basename(base_path)})')

    # ---- layer 2: harvest from shipped translations ------------------------
    pairs = defaultdict(lambda: defaultdict(int))
    pending_src: dict[str, set] = {}
    n = harvest(os.path.join(BASE_DIR, 'crucible-0.10.1'),
                os.path.join(P, '2-Crucible汉化插件', 'compendium', 'cn'),
                'crucible', pairs, pending_src)
    print(f'harvest: crucible {n} pair-hits')
    n = harvest(os.path.join(BASE_DIR, 'ember-0.6.0'),
                os.path.join(P, '1-Ember汉化插件', 'compendium', 'cn'),
                'ember', pairs, pending_src)
    print(f'harvest: ember    {n} pair-hits')

    harvested = {}
    conflicts = {}
    for en_s, cns in pairs.items():
        ranked = sorted(cns.items(), key=lambda kv: -kv[1])
        harvested[en_s] = ranked[0][0]
        if len(ranked) > 1:
            conflicts[en_s] = {cn: c for cn, c in ranked}
    print(f'harvest: {len(harvested)} distinct terms, {len(conflicts)} with >1 candidate')

    # ---- merge -------------------------------------------------------------
    glossary = {}
    provenance = {}
    base_vs_harvest = {}
    for en_s, cn_s in base.items():
        glossary[en_s] = cn_s
        provenance[en_s] = {'cn': cn_s, 'source': 'base'}
    disputes = {}
    for en_s, cn_s in harvested.items():
        if en_s in glossary:
            if glossary[en_s] != cn_s:
                b = glossary[en_s]
                # The house style is bilingual ("中文 English"). The base glossary
                # stores the bare Chinese. That is a FORMAT difference, not a
                # disagreement about the translation -> shipped wins silently.
                format_only = (cn_s.startswith(b) and en_s in cn_s) or (b in cn_s) or (cn_s in b)
                base_vs_harvest[en_s] = {'base': b, 'shipped': cn_s,
                                         'kind': 'format' if format_only else 'conflict'}
                if format_only:
                    glossary[en_s] = cn_s
                    provenance[en_s] = {'cn': cn_s, 'source': 'shipped-bilingual-form',
                                        'base_was': b}
                else:
                    # A real disagreement: two different Chinese renderings.
                    # Do NOT pick silently. Keep shipped (status quo, what players
                    # see today) but surface it for adjudication.
                    disputes[en_s] = {'base': b, 'shipped': cn_s}
                    glossary[en_s] = cn_s
                    provenance[en_s] = {'cn': cn_s, 'source': 'DISPUTED-shipped-kept',
                                        'base_was': b}
            continue
        glossary[en_s] = cn_s
        provenance[en_s] = {'cn': cn_s, 'source': 'shipped'}
        if en_s in conflicts:
            provenance[en_s]['candidates'] = conflicts[en_s]

    # ---- pending: English terms with no Chinese anywhere --------------------
    pending = {s: sorted(src) for s, src in pending_src.items() if s not in glossary}

    fmt = sum(1 for v in base_vs_harvest.values() if v['kind'] == 'format')
    print(f'\nglossary_ec : {len(glossary)} terms '
          f'({len(base)} base + {len(glossary) - len(base)} new from shipped)')
    print(f'base vs shipped, format-only (auto-resolved): {fmt}')
    print(f'base vs shipped, REAL disputes (need ruling) : {len(disputes)}')
    print(f'pending (no CN yet)                          : {len(pending)}')

    # Internal inconsistency: the same English proper noun rendered several ways
    # inside the shipped translations themselves. Split into the two very
    # different problems this hides.
    def bare(cn, en):
        """Strip the ' English' suffix of the bilingual house format."""
        return cn[:-len(en)].strip() if cn.endswith(en) else cn

    fmt_inconsistent, term_inconsistent = {}, {}
    for en_s, cands in conflicts.items():
        if len(cands) < 2:
            continue
        if len({bare(c, en_s) for c in cands}) == 1:
            fmt_inconsistent[en_s] = cands   # same Chinese, bilingual suffix applied unevenly
        else:
            term_inconsistent[en_s] = cands  # genuinely different Chinese
    print(f'shipped: bilingual format applied unevenly   : {len(fmt_inconsistent)}')
    print(f'shipped: same term translated differently    : {len(term_inconsistent)}')

    dump(os.path.join(OUT, 'glossary_ec.disputes.json'), {
        '_meta': {
            'count': len(disputes),
            'note': '基底术语表与已发布译文对同一英文给出了不同中文（非双语格式差异）。'
                    '当前一律先保留已发布值（玩家现在看到的），但每条都需要人工/上下文裁决。'
                    '裁决后写回 glossary_ec.json 并记入 PROJECT.md 第 8 节。',
        },
        'disputes': dict(sorted(disputes.items())),
        'shippedBilingualFormatInconsistency': {
            '_note': '同一专名的中文相同，但双语后缀（" English"）时加时不加。'
                     '属于格式问题，可脚本批量归一。',
            'count': len(fmt_inconsistent),
            'terms': dict(sorted(fmt_inconsistent.items())),
        },
        'shippedTermInconsistency': {
            '_note': '同一专名在已发布译文里被译成了不同的中文。必须裁决后统一。',
            'count': len(term_inconsistent),
            'terms': dict(sorted(term_inconsistent.items())),
        },
    })

    dump(os.path.join(OUT, 'glossary_ec.json'), dict(sorted(glossary.items())))
    dump(os.path.join(OUT, 'glossary_ec.provenance.json'), {
        '_meta': {
            'baseGlossary': base_path,
            'baseTerms': len(base),
            'totalTerms': len(glossary),
            'harvestedTerms': len(harvested),
            'baseVsShippedDisagreements': len(base_vs_harvest),
            'pendingTerms': len(pending),
            'policy': 'shipped translation wins over base glossary; '
                      'base glossary wins over nothing; PF2E master glossary NOT merged',
        },
        'baseVsShipped': dict(sorted(base_vs_harvest.items())),
        'multiCandidate': dict(sorted(conflicts.items())),
        'terms': dict(sorted(provenance.items())),
    })
    dump(os.path.join(OUT, 'glossary_ec.pending.json'), {
        '_meta': {'count': len(pending),
                  'note': '当前英文基准里出现、但项目术语表中还没有中文对应的名词。'
                          '翻译时遇到即补进 glossary_ec.json。'},
        'terms': dict(sorted(pending.items())),
    })
    print(f'\nwrote -> {OUT}')


main()
