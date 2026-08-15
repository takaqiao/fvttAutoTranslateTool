"""Probe: enumerate every DialogV2 config in ember.mjs and classify its
title / content / button-label strings as literal English, i18n key, or dynamic.

Read-only. Assumptions & false-positive modes:
  - Brace matching is naive about regex literals; verified by spot-checking output.
  - "[dynamic]" means the literal contains ${...} or _loc(, so part of it may still
    be hardcoded English (e.g. `Vantage Point: ${label}`) -- those are listed separately.
  - Only DEFAULT_CONFIG `dialog: {}` blocks and direct DialogV2.<m>({...}) call sites
    are scanned. Labels assigned imperatively inside _configureDialog/_displayDialog
    (e.g. "Ascend"/"Descend"/"Repair"/"Unseal") are NOT captured here -- counted by hand.
"""
import re

P = r'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs'
s = open(P, encoding='utf-8').read()

BS = chr(92)


def match_block(text, i):
    depth = 0
    j = i
    n = len(text)
    while j < n:
        c = text[j]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
        elif c in '"\'`':
            q = c
            j += 1
            while j < n:
                if text[j] == BS:
                    j += 2
                    continue
                if text[j] == q:
                    break
                j += 1
        j += 1
    return text[i:]


def lineno(idx):
    return s.count('\n', 0, idx) + 1


sites = []
for m in re.finditer(r'DialogV2\$?\d*\.(confirm|prompt|input|wait)\(\s*\{', s):
    sites.append(('call', m.start(), m.end() - 1))
for m in re.finditer(r'\bdialog:\s*\{', s):
    sites.append(('cfg', m.start(), m.end() - 1))
sites.sort(key=lambda x: x[1])

DQ = '"(?:[^"' + BS + BS + ']|' + BS + BS + '.)*"'
SQ = "'(?:[^'" + BS + BS + ']|' + BS + BS + ".)*'"
BQ = '`[^`]*`'
STR = re.compile(
    r'(title|content|label|description)\s*:\s*(' + BQ + '|' + DQ + '|' + SQ + ')'
)

I18N = re.compile(r'[A-Z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+\Z')

counts = {'content': 0, 'label': 0, 'title': 0, 'description': 0}
dyn = []
for kind, st, br in sites:
    blk = match_block(s, br)
    ln = lineno(st)
    rows = []
    for m in STR.finditer(blk):
        k, v = m.group(1), m.group(2)
        core = v[1:-1]
        if I18N.fullmatch(core):
            tag = 'i18nkey'
        elif '_loc(' in core:
            tag = 'dynamic-loc'
        elif '${' in core:
            tag = 'dynamic-EN'
            dyn.append((ln, k, core[:100]))
        else:
            tag = 'EN'
            counts[k] = counts.get(k, 0) + 1
        rows.append((k, core, tag))
    if rows:
        print('--- %s @%d' % (kind, ln))
        for k, core, tag in rows:
            print('    %-11s [%s] %r' % (k, tag, core[:130]))

print()
print('literal-English counts:', counts)
print('dynamic templates containing English:', len(dyn))
for ln, k, c in dyn:
    print('   @%d %s %r' % (ln, k, c))
