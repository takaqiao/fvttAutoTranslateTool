"""Build the FVTT JournalEntry JSON for Book 2 from the segments + Chinese translation.

Differences from Book 1's builder, which emitted flat escaped <p> text:
- One page per Heading 1 (10 pages). Heading 2/3 become in-page <h2>/<h3>.
- Inline formatting survives: <strong>, <em>, <s>, and the author's semantic rating
  colours as <span style="background-color:...">.
- Bulleted lists are rebuilt as nested <ul>.
- Images are referenced from IMG_PREFIX.
- Every cross-reference in the document points at a heading on a *different* page, so a
  bare "#anchor" would be dead on arrival. They are emitted as Foundry relative UUID links
  (@UUID[.pageId]) instead, which resolve after import and jump to the right page.
"""
import json
import os
import random
import string
import time
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
SEGMENTS = os.path.join(HERE, 'book2_segments.json')
TRANSLATION = os.path.join(HERE, 'book2_zh.json')
TEMPLATE = os.path.join(HERE, 'fvtt-JournalEntry-test-Tp18xjy7qXQaZqMT.json')
DST = os.path.join(HERE, 'fvtt-JournalEntry-Season-of-Ghosts-Book2-zh.json')

# Where you will copy book2_images/ inside the Foundry Data directory.
IMG_PREFIX = 'assets/sog-gmguide-book2/'

# "Sidebar - Haunt Mechanics" is referenced four times but its Google-Docs bookmark did not
# survive the export. Its subject is the Suppress Haunt houserule, so it points at that H1.
EXTRA_ANCHORS = {'_4kpk55o53n30': 29}

ID_ALPHABET = string.ascii_letters + string.digits

SHADE_TEXT = '#1a1a1a'  # keep pastel-shaded runs legible in Foundry's dark journal theme


def gen_id(rng):
    return ''.join(rng.choice(ID_ALPHABET) for _ in range(16))


def slug(text):
    keep = [c if (c.isalnum() or c in '-_') else '-' for c in text.lower()]
    return 'h-' + ''.join(keep).strip('-')[:48]


def load_anchor_targets(segments):
    """anchor name -> paragraph index of the heading it marks."""
    anchors = dict(EXTRA_ANCHORS)
    from docx import Document
    from docx.oxml.ns import qn
    doc = Document(os.path.join(HERE, 'Season of Ghosts GM Guide - Book 2.docx'))
    for i, p in enumerate(doc.paragraphs):
        for bm in p._element.findall('.//' + qn('w:bookmarkStart')):
            name = bm.get(qn('w:name'))
            if name and name.startswith('_'):
                anchors.setdefault(name, i)
    return anchors


def assign_pages(segments):
    """Return (pages, para_to_page) where pages is a list of dicts with _id/name/start."""
    pages = []
    para_to_page = {}
    current = None
    for s in segments:
        if s['style'] == 'Title':
            continue
        if s['style'] == 'Heading 1':
            current = {'start': s['i'], 'segments': []}
            pages.append(current)
        if current is None:
            continue  # content before the first H1 (a banner image) is folded in later
        current['segments'].append(s)
        para_to_page[s['i']] = len(pages) - 1
    return pages, para_to_page


def render_chunks(chunks, translations, anchors, para_to_page, page_ids, heading_ids):
    out = []
    for chunk, zh in zip(chunks, translations):
        if not zh:
            continue
        fmt = chunk['fmt']
        # Keep padding outside the markup so link labels don't carry stray spaces.
        lead = zh[:len(zh) - len(zh.lstrip())]
        trail = zh[len(zh.rstrip()):]
        zh = zh.strip()
        if not zh:
            out.append(lead + trail)
            continue
        body = escape(zh)
        if fmt.get('strike'):
            body = f'<s>{body}</s>'
        if fmt.get('italic'):
            body = f'<em>{body}</em>'
        if fmt.get('bold'):
            body = f'<strong>{body}</strong>'
        if fmt.get('shade'):
            body = (f'<span style="background-color:#{fmt["shade"]};'
                    f'color:{SHADE_TEXT}">{body}</span>')
        if fmt.get('color'):
            body = f'<span style="color:#{fmt["color"]}">{body}</span>'

        if chunk['kind'] == 'link':
            href = chunk.get('href')
            anchor = chunk.get('anchor')
            if href and href.startswith('#'):
                anchor, href = href[1:], None
            if href:
                body = f'<a href="{escape(href, quote=True)}">{body}</a>'
            elif anchor:
                target = anchors.get(anchor)
                page_idx = para_to_page.get(target)
                if page_idx is not None:
                    label = zh.replace(']', '').replace('}', '')
                    frag = heading_ids.get(target)
                    ref = page_ids[page_idx] + (f'#{frag}' if frag else '')
                    body = f'@UUID[.{ref}]{{{label}}}'
        out.append(lead + body + trail)
    return ''.join(out)


class HtmlBuilder:
    """Accumulates page HTML, managing <ul> nesting across consecutive list paragraphs.

    A deeper list is opened *inside* the still-open parent <li>, which is where the HTML
    spec wants it — a <ul> as a direct child of another <ul> renders but is invalid.
    """

    def __init__(self):
        self.parts = []
        self.depth = 0        # number of open <ul>
        self.li_open = False  # an <li> at the current depth awaits its </li>

    def _close_li(self):
        if self.li_open:
            self.parts.append('</li>')
            self.li_open = False

    def _close_to(self, depth):
        while self.depth > depth:
            self._close_li()
            self.parts.append('</ul>')
            self.depth -= 1
            # unwinding lands back inside the <li> that wrapped the list we just closed
            self.li_open = self.depth > 0

    def block(self, html):
        self._close_to(0)
        self._close_li()
        self.parts.append(html)

    def item(self, level, html):
        want = level + 1
        if want > self.depth:
            while self.depth < want:
                if self.depth > 0 and not self.li_open:
                    # no parent <li> to nest under; synthesise one so the markup stays valid
                    self.parts.append('<li>')
                    self.li_open = True
                self.parts.append('<ul>')
                self.depth += 1
                self.li_open = False
        else:
            self._close_to(want)
            self._close_li()
        self.parts.append(f'<li>{html}')
        self.li_open = True

    def done(self):
        self._close_to(0)
        self._close_li()
        return ''.join(self.parts)


def main():
    data = json.load(open(SEGMENTS, encoding='utf-8'))
    segments = data['segments']
    images = data['images']
    zh = json.load(open(TRANSLATION, encoding='utf-8'))
    tpl = json.load(open(TEMPLATE, encoding='utf-8'))

    rng = random.Random(0x50C2)
    now_ms = int(time.time() * 1000)

    anchors = load_anchor_targets(segments)
    pages, para_to_page = assign_pages(segments)
    page_ids = [gen_id(rng) for _ in pages]

    heading_ids = {}
    for s in segments:
        if s['style'] in ('Heading 2', 'Heading 3') and str(s['i']) in zh:
            heading_ids[s['i']] = slug(''.join(zh[str(s['i'])]))

    journal_name = ''
    for s in segments:
        if s['style'] == 'Title':
            journal_name = ''.join(zh[str(s['i'])])
            break

    seg_by_index = {s['i']: s for s in segments}
    max_index = max(max(seg_by_index), max(int(k) for k in images)) if images else max(seg_by_index)

    # Content appearing before the first H1 (the banner image) is prepended to page 1.
    preamble = []
    out_pages = []
    builder = None
    page_no = -1

    def flush():
        if builder is None:
            return
        html = builder.done()
        if page_no == 0 and preamble:
            html = ''.join(preamble) + html
        out_pages.append({
            'sort': (len(out_pages) + 1) * 100000,
            'name': pages[page_no]['name'],
            'type': 'text',
            '_id': page_ids[page_no],
            'system': {},
            'title': {'show': True, 'level': 1},
            'image': {},
            'text': {'format': 1, 'content': html},
            'video': {'controls': True, 'volume': 0.5},
            'src': None,
            'category': None,
            'flags': {},
            '_stats': {**tpl['pages'][0]['_stats'],
                       'createdTime': now_ms, 'modifiedTime': now_ms},
            'ownership': {'default': -1},
        })

    for i in range(max_index + 1):
        s = seg_by_index.get(i)
        if s is not None and s['style'] != 'Title':
            translations = zh[str(i)]
            if s['style'] == 'Heading 1':
                flush()
                page_no += 1
                pages[page_no]['name'] = ''.join(translations)
                builder = HtmlBuilder()
            else:
                html = render_chunks(s['chunks'], translations, anchors,
                                     para_to_page, page_ids, heading_ids)
                if s['style'] in ('Heading 2', 'Heading 3'):
                    tag = 'h2' if s['style'] == 'Heading 2' else 'h3'
                    builder.block(f'<{tag} id="{heading_ids[i]}">{html}</{tag}>')
                elif s['list'] is not None:
                    builder.item(s['list'], html)
                else:
                    builder.block(f'<p>{html}</p>')

        for fn in images.get(str(i), []):
            tag = (f'<p><img src="{IMG_PREFIX}{fn}" alt="" '
                   f'style="max-width:100%;height:auto" /></p>')
            if builder is None:
                preamble.append(tag)
            else:
                builder.block(tag)

    flush()

    stats = tpl['_stats']
    out = {
        'name': journal_name,
        'pages': out_pages,
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
        'ownership': {'default': 0},
    }

    with open(DST, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f'wrote {DST}')
    print(f'  name: {journal_name}')
    print(f'  pages: {len(out_pages)}')
    for p in out_pages:
        print(f'    - {p["name"]}  ({len(p["text"]["content"])} chars)')


if __name__ == '__main__':
    main()
