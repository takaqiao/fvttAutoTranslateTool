"""Convert Season of Ghosts GM Guide - Book 1_zh.docx to FVTT JournalEntry JSON.

Mapping rules:
- The "Title"-styled paragraph becomes the JournalEntry name.
- Each non-empty Heading 1/2/3 starts a new page with title.level matching the heading level.
- "normal" paragraphs accumulate as <p> HTML inside the current page's text.content.
- Empty paragraphs and empty headings are skipped (they exist as visual spacers in the docx).
- Inline hyperlinks are flattened by walking <w:t> in document order so text reads naturally.
"""
import json
import random
import string
import time
from html import escape

from docx import Document
from docx.oxml.ns import qn

SRC = 'Season of Ghosts GM Guide - Book 1_zh.docx'
DST = 'fvtt-JournalEntry-Season-of-Ghosts-Book1-zh.json'
TEMPLATE = 'fvtt-JournalEntry-test-Tp18xjy7qXQaZqMT.json'

ID_ALPHABET = string.ascii_letters + string.digits


def gen_id(rng: random.Random) -> str:
    return ''.join(rng.choice(ID_ALPHABET) for _ in range(16))


def paragraph_text(p) -> str:
    """Return the paragraph's text in document order, hyperlinks inlined."""
    parts = []
    for elem in p._element.iter():
        tag = elem.tag
        if tag == qn('w:t'):
            if elem.text:
                parts.append(elem.text)
        elif tag == qn('w:tab'):
            parts.append('\t')
        elif tag == qn('w:br'):
            parts.append('\n')
    return ''.join(parts).strip()


def heading_level(style_name: str) -> int | None:
    if style_name == 'Heading 1':
        return 1
    if style_name == 'Heading 2':
        return 2
    if style_name == 'Heading 3':
        return 3
    return None


def main():
    with open(TEMPLATE, 'r', encoding='utf-8') as f:
        tpl = json.load(f)

    stats = tpl['_stats']
    page_stats_template = tpl['pages'][0]['_stats']

    rng = random.Random(0x50C)  # deterministic IDs across runs
    now_ms = int(time.time() * 1000)

    doc = Document(SRC)

    journal_name = ''
    pages = []
    current = None  # {name, level, html_chunks}

    def flush():
        nonlocal current
        if current is None:
            return
        sort_index = (len(pages) + 1) * 100000
        page = {
            'sort': sort_index,
            'name': current['name'],
            'type': 'text',
            '_id': gen_id(rng),
            'system': {},
            'title': {
                'show': True,
                'level': current['level'],
            },
            'image': {},
            'text': {
                'format': 1,
                'content': ''.join(current['html_chunks']),
            },
            'video': {
                'controls': True,
                'volume': 0.5,
            },
            'src': None,
            'category': None,
            'flags': {},
            '_stats': {
                **page_stats_template,
                'createdTime': now_ms,
                'modifiedTime': now_ms,
            },
            'ownership': {
                'default': -1,
            },
        }
        pages.append(page)
        current = None

    for p in doc.paragraphs:
        style = p.style.name if p.style else ''
        text = paragraph_text(p)

        if style == 'Title':
            if text and not journal_name:
                journal_name = text
            continue

        level = heading_level(style)
        if level is not None:
            if not text:
                continue  # empty heading used as spacer
            flush()
            current = {'name': text, 'level': level, 'html_chunks': []}
            continue

        # normal paragraph (or anything else): treat as body
        if not text:
            continue
        if current is None:
            # body before any heading: hold under a synthetic intro page
            current = {'name': journal_name or 'Intro', 'level': 1, 'html_chunks': []}
        current['html_chunks'].append(f'<p>{escape(text)}</p>')

    flush()

    out = {
        'name': journal_name or 'Season of Ghosts Book 1',
        'pages': pages,
        'folder': None,
        'categories': [],
        'flags': {},
        '_stats': {
            **stats,
            'createdTime': now_ms,
            'modifiedTime': now_ms,
            'exportSource': {
                'worldId': 'sog',
                'uuid': f'JournalEntry.{gen_id(rng)}',
                'coreVersion': stats['coreVersion'],
                'systemId': stats['systemId'],
                'systemVersion': stats['systemVersion'],
            },
        },
        'ownership': {
            'default': 0,
        },
    }

    with open(DST, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f'wrote {DST}')
    print(f'  name: {journal_name}')
    print(f'  pages: {len(pages)}')


if __name__ == '__main__':
    main()
