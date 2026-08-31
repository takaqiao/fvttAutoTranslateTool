"""Long English residue scan for FotRP canonical translation files."""
import json, re
from pathlib import Path

BOOK_DIR = Path('FotRP/需要翻译/NEW')
ROOT_DIR = Path('FotRP/需要翻译')

book_files = sorted(BOOK_DIR.rglob('fvtt-*.json'))
root_files = sorted(p for p in ROOT_DIR.glob('*.json') if 'fvtt-' not in p.name)

STRIP_PATTERNS = [
    re.compile(r'@(UUID|Check|Damage|Template|Localize|Compendium|Actor|Item|JournalEntry|Scene)\[[^\]]+\](\{[^}]*\})?'),
    re.compile(r'\[\[/[^\]]+\]\](\{[^}]*\})?'),
    re.compile(r'<[^>]+>'),
    re.compile(r'&[a-z]+;'),
]
SKIP_PATH_TOKENS = ['/_id', '/_stats', '/src', '/img', '/icon', '/sort',
    '/flags/pdftofoundry', '/flags/babele', '/flags/core',
    '/ownership', '/sourceId', '/system/source',
    '/macro', '/command', '/mapping/', '/data/placeHolder', '/folderId']

def walk(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f'{path}/{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f'{path}[{i}]')
    elif isinstance(obj, str):
        yield path, obj

def strip_text(s):
    for p in STRIP_PATTERNS:
        s = p.sub(' ', s)
    return s

def has_chinese(s):
    return any('一' <= c <= '鿿' for c in s)

LONG_EN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9 ,.\-\(\)\"']{60,}")

print('=== Long English residue scan ===')
total_residues = 0
for f in book_files + root_files:
    data = json.loads(f.read_text(encoding='utf-8'))
    file_residues = []
    for path, s in walk(data):
        if any(tok in path for tok in SKIP_PATH_TOKENS):
            continue
        if path.endswith('/_id') or path.endswith('/sourceId') or path.endswith('/folder'):
            continue
        if '/en/' in path or path.endswith('/en'):
            continue
        stripped = strip_text(s)
        for m in LONG_EN_PATTERN.finditer(stripped):
            chunk = m.group(0).strip()
            if len(chunk) < 60:
                continue
            start, end = m.start(), m.end()
            nearby = stripped[max(0, start-30):min(len(stripped), end+30)]
            if has_chinese(nearby):
                continue
            file_residues.append((path[:100], chunk[:120]))
    if file_residues:
        print(f'\n{f.name}: {len(file_residues)} long-English residues')
        for path, chunk in file_residues[:5]:
            print(f'  {path}')
            print(f'    {chunk}')
        total_residues += len(file_residues)

print(f'\nTotal long-English residues: {total_residues}')
