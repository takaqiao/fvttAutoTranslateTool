#!/usr/bin/env python3
"""How much of the ORIGINAL Chinese prose survived a realign unit, verbatim?

  python prose_survival.py <单元目录> [--min 0.90] [--list]

This is the direct test of the batch's iron rule ("已有中文原样保留"). Block
counts cannot answer it:

* `diff_realign.py` splits on more tags than `measure_8c.py` counts, so its `+`
  runs ~2x the reported `missing_blocks` for purely mechanical reasons.
* An insert adjacent to a real edit collapses into one difflib `replace`, which
  inflates `~`. Batch 6 saw every one of 12 units flagged as "over-editing" by
  those counts -- uniformity that pointed at the metric, not the translators.

So instead of counting blocks, this counts **sentences of the original Chinese
that still appear verbatim in the new Chinese**.

READ THIS BEFORE TRUSTING THE NUMBER
------------------------------------
Low survival is NOT by itself evidence of rewriting. Batch 6 measured 0.70-0.87
across all 12 units, and spot-checking the worst pages showed the drops were
mostly legitimate:

1. **Markers changed.** A sentence carrying `[[/check 16 perception]]` scores as
   lost when the translator updates it to today's `[[/skillCheck …]]`, even
   though the prose is untouched. Much of the campaign's older Chinese was
   written against dnd5e-style commands.
2. **Upstream rewrote the page.** `Lantern Roads / Impromptu Jail` was flagged
   `missing_blocks: 1`, yet its old Chinese described eavesdropping dialogue that
   today's English no longer contains at all -- upstream replaced the scene.
   Re-translating it was the correct action, and survival there is ~0.03.

   That case also exposes a limit of `measure_8c.py` / `measure_stale_extra.py`:
   they compare block **counts**. When upstream swaps content while keeping a
   similar count, they see almost nothing. The markup gate is what catches it,
   because the old Chinese's `@UUID`/`[[…]]` no longer match today's English.

So use this to **pick candidates to look at**, never as a pass/fail gate. The
verdict requires today's English in hand, which is what the review agents have;
in batch 6 they reverted exactly 1 genuine rewrite across 493 pages.
"""
from __future__ import annotations
import json
import os
import re
import sys

TAG = re.compile(r'<[^>]+>')
SPLIT = re.compile(r'[。！？；\n]+')
CJK = re.compile(r'[一-鿿]')


def strip(html):
    """Tags out, whitespace collapsed. BOTH sides must go through this.

    Comparing stripped sentences against raw HTML is wrong: a block that merely
    gained an inline `<strong>` would look like vanished prose even though not a
    character of the translation changed. Batch 6 first measured 0.69-0.87
    survival that way, which was the metric misreading itself, not rewriting.
    """
    return re.sub(r'\s+', '', TAG.sub('\n', html))


def sentences(html):
    out = []
    for s in SPLIT.split(TAG.sub('\n', html)):
        s = re.sub(r'\s+', '', s)
        # short fragments match by chance; only score substantial prose
        if len(s) >= 12 and CJK.search(s):
            out.append(s)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    show = '--list' in sys.argv
    thr = 0.90
    if '--min' in sys.argv:
        thr = float(sys.argv[sys.argv.index('--min') + 1])
    if not args:
        raise SystemExit('usage: prose_survival.py <单元目录> [--min 0.90] [--list]')
    unit = args[0]

    index = json.load(open(os.path.join(unit, 'index.json'), encoding='utf-8'))
    original = json.load(open(os.path.join(unit, '_original_cn.json'), encoding='utf-8'))

    tot_old = tot_kept = 0
    flagged = []
    for page in index['pages']:
        i = str(page['id'])
        f = os.path.join(unit, 'pages', f'{i}.cn.html')
        if not os.path.exists(f):
            continue
        new = open(f, encoding='utf-8-sig', newline='').read()
        new_flat = strip(new)
        old = original.get(i, '')
        olds = sentences(old)
        if not olds:
            continue
        kept = sum(1 for s in olds if s in new_flat)
        tot_old += len(olds)
        tot_kept += kept
        rate = kept / len(olds)
        if rate < thr:
            flagged.append((rate, len(olds) - kept, page))

    rate = tot_kept / tot_old if tot_old else 1.0
    print(f'{index["unit"]}: 原中文句子 {tot_old}，原样保留 {tot_kept}  存活率 {rate:.3f}')
    if flagged:
        print(f'  低于 {thr:.2f} 的页 {len(flagged)}（丢失句数 vs 本页应删块数）:')
        for r, lost, page in sorted(flagged, key=lambda x: (x[0], -x[1]))[:20]:
            tail = '.'.join(page['path'].split('.')[-3:-1])
            print(f'    {r:.2f}  丢 {lost:>3} 句 / 应删 {page["extra_blocks"]:>3} 块   {tail}')
    if show:
        for r, lost, page in sorted(flagged, key=lambda x: (x[0], -x[1]))[:5]:
            i = str(page['id'])
            new_flat = strip(open(os.path.join(unit, 'pages', f'{i}.cn.html'),
                                  encoding='utf-8-sig').read())
            print(f'\n  === [{i}] {page["path"]}')
            for s in sentences(original.get(i, '')):
                if s not in new_flat:
                    print(f'      丢: {s[:110]}')


main()
