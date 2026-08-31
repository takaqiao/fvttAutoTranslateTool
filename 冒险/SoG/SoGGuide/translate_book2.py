"""Write the Chinese translation back into the docx, preserving everything else.

The walk here mirrors extract_book2.py exactly: top-level w:r and w:hyperlink children,
with consecutive identically-formatted runs merged into one chunk. Each chunk's translation
goes into the first run of its group; the remaining runs in the group are emptied.

Because hyperlinks are visited in document order rather than skipped, the translated anchor
text stays inside its w:hyperlink element. This is what Book 1's pipeline got wrong: it
assigned the whole paragraph to runs[0], so link text drifted to the end of the paragraph.

Runs carrying drawings hold no w:t and are never chunks, so images pass through untouched.
"""
import json
import os

from docx import Document
from docx.oxml.ns import qn
from docx.text.run import Run

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'Season of Ghosts GM Guide - Book 2.docx')
DST = os.path.join(HERE, 'Season of Ghosts GM Guide - Book 2_zh.docx')
SEGMENTS = os.path.join(HERE, 'book2_segments.json')
TRANSLATION = os.path.join(HERE, 'book2_zh.json')


def run_text(r):
    parts = []
    for e in r._element.iter():
        if e.tag == qn('w:t'):
            if e.text:
                parts.append(e.text)
        elif e.tag == qn('w:tab'):
            parts.append('\t')
        elif e.tag == qn('w:br'):
            parts.append('\n')
    return ''.join(parts)


def run_fmt_key(r):
    """Same signature extract_book2.py merges on, as a hashable tuple."""
    color = None
    c = r.font.color
    if c is not None and c.type is not None and c.rgb is not None:
        color = str(c.rgb)
    if color and color.lower() in ('191813', '000000'):
        color = None
    shade = None
    rpr = r._element.find(qn('w:rPr'))
    if rpr is not None:
        shd = rpr.find(qn('w:shd'))
        if shd is not None:
            fill = shd.get(qn('w:fill'))
            if fill and fill.lower() not in ('auto', 'ffffff'):
                shade = fill.lower()
    return (bool(r.bold), bool(r.italic), bool(r.font.strike), color, shade)


def chunk_run_groups(p):
    """Ordered list of run-lists, one per chunk, matching extract_book2.py's chunking."""
    groups = []
    last_key = None
    last_kind = None
    for child in p._element:
        if child.tag == qn('w:r'):
            r = Run(child, p)
            if not run_text(r):
                continue  # image-only or empty run: not a chunk
            key = run_fmt_key(r)
            if groups and last_kind == 'text' and last_key == key:
                groups[-1].append(r)
            else:
                groups.append([r])
                last_key, last_kind = key, 'text'
        elif child.tag == qn('w:hyperlink'):
            runs = [Run(e, p) for e in child.findall(qn('w:r'))]
            if not any(run_text(r) for r in runs):
                continue
            groups.append(runs)
            last_key, last_kind = None, 'link'
    return groups


def set_run_text(r, text):
    """Replace a run's text content, keeping its rPr. Never called on runs with drawings."""
    r.text = text


def main():
    segments = json.load(open(SEGMENTS, encoding='utf-8'))['segments']
    zh = json.load(open(TRANSLATION, encoding='utf-8'))
    expected = {s['i']: len(s['chunks']) for s in segments}

    doc = Document(SRC)
    written = 0
    problems = []

    for i, p in enumerate(doc.paragraphs):
        key = str(i)
        if key not in zh:
            continue
        groups = chunk_run_groups(p)
        translations = zh[key]

        if len(groups) != expected[i]:
            problems.append(f'para {i}: walk found {len(groups)} groups, '
                            f'extraction recorded {expected[i]}')
            continue
        if len(translations) != len(groups):
            problems.append(f'para {i}: {len(translations)} translations for '
                            f'{len(groups)} chunks')
            continue

        for runs, text in zip(groups, translations):
            set_run_text(runs[0], text)
            for extra in runs[1:]:
                set_run_text(extra, '')
        written += 1

    if problems:
        print(f'{len(problems)} PROBLEM(S):')
        for msg in problems[:20]:
            print('  ' + msg)
        raise SystemExit(1)

    doc.save(DST)
    print(f'wrote {DST}')
    print(f'  paragraphs translated: {written}/{len(zh)}')


if __name__ == '__main__':
    main()
