# -*- coding: utf-8 -*-
"""Check whether adding 7 dot-less top-level i18n keys can collide with anything
installed on this machine (Foundry core + all systems + all modules)."""
import json, os, re, sys, io
sys.stdout.reconfigure(encoding='utf-8')

KEYS = ["Special", "Reserved Action", "Slow Weaponry", "Broken", "Bulky Armor", "Elite", "Boss"]
FOUNDRY_APP = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
DATA = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"

def top_level_string_keys(path):
    try:
        d = json.load(io.open(path, encoding='utf-8-sig'))
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}
    return {k: v for k, v in d.items() if isinstance(v, str)}

print("== A. existing DOT-LESS top-level string keys that would be overwritten ==")
lang_files = []
for root in [os.path.join(FOUNDRY_APP, 'public', 'lang'), os.path.join(FOUNDRY_APP, 'lang')]:
    if os.path.isdir(root):
        for fn in os.listdir(root):
            if fn.endswith('.json'):
                lang_files.append(os.path.join(root, fn))
for base in ['systems', 'modules']:
    d = os.path.join(DATA, base)
    if not os.path.isdir(d):
        continue
    for pkg in os.listdir(d):
        ld = os.path.join(d, pkg, 'lang')
        if not os.path.isdir(ld):
            continue
        for fn in os.listdir(ld):
            if fn.endswith('.json'):
                lang_files.append(os.path.join(ld, fn))
print("   scanned lang files:", len(lang_files))
hits = 0
for p in lang_files:
    tl = top_level_string_keys(p)
    for k in KEYS:
        if k in tl:
            print("   HIT", os.path.relpath(p, DATA) if DATA in p else p, "|", k, "=>", tl[k][:60])
            hits += 1
if not hits:
    print("   none")

print()
print("== B. literal localize()/has()/format() calls with these exact strings ==")
pats = [re.compile(r'''(?:localize|has|format)\(\s*["'](%s)["']''' % re.escape(k)) for k in KEYS]
hb = [re.compile(r'''\{\{\s*(?:#?\w+\s+)?localize\s+["'](%s)["']''' % re.escape(k)) for k in KEYS]
scan_roots = [FOUNDRY_APP, os.path.join(DATA, 'systems'), os.path.join(DATA, 'modules')]
found = 0
files = 0
for root in scan_roots:
    for dp, dn, fnames in os.walk(root):
        dn[:] = [x for x in dn if x not in ('node_modules', '.git', 'assets', 'icons', 'ui', 'fonts', 'audio', 'media', 'packs', 'images', 'img')]
        for fn in fnames:
            if not fn.endswith(('.js', '.mjs', '.hbs', '.html')):
                continue
            p = os.path.join(dp, fn)
            try:
                txt = io.open(p, encoding='utf-8', errors='ignore').read()
            except Exception:
                continue
            files += 1
            for i, k in enumerate(KEYS):
                if k not in txt:
                    continue
                for rx in (pats[i], hb[i]):
                    for m in rx.finditer(txt):
                        print("   HIT", p.replace(DATA, '<Data>').replace(FOUNDRY_APP, '<app>'), '|', k, '|', txt[max(0,m.start()-60):m.end()+30].replace('\n',' '))
                        found += 1
print("   scanned code files:", files, " hits:", found)
