# -*- coding: utf-8 -*-
"""复检 out_reach_crucible.txt 里的 DEAD 候选：语料扩到
crucible + ember + Foundry 核心（client/common/public/templates），
逐键做**整键字面**搜索。只读。"""
import re
from pathlib import Path

HERE = Path(__file__).parent
keys = [l.strip() for l in (HERE / 'out_reach_crucible.txt').read_text(encoding='utf-8').splitlines()
        if re.fullmatch(r'\s{4}[A-Z][A-Za-z0-9_.]+', l)]
print('候选', len(keys))

EXTS = {'.mjs', '.js', '.hbs', '.html', '.json', '.ts'}
SKIP = {'assets', 'fonts', 'icons', 'ui', 'audio', 'node_modules', 'dist', 'lang'}


def corpus(roots):
    parts = []
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        for p in r.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in EXTS:
                continue
            rel = str(p).replace('\\', '/')
            if any('/%s/' % d in rel for d in SKIP):
                continue
            try:
                parts.append(p.read_text(encoding='utf-8', errors='ignore'))
            except Exception:
                pass
    return '\n'.join(parts)


C = corpus([
    r'C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible',
    r'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember',
    r'C:/Program Files/Foundry Virtual Tabletop/resources/app/client',
    r'C:/Program Files/Foundry Virtual Tabletop/resources/app/common',
    r'C:/Program Files/Foundry Virtual Tabletop/resources/app/public/scripts',
    r'C:/Program Files/Foundry Virtual Tabletop/resources/app/templates',
])
print('语料', len(C))
alive = [k for k in keys if k in C]
dead = [k for k in keys if k not in C]
print('复检仍死 %d   复检活了 %d %s' % (len(dead), len(alive), alive))
for k in dead:
    print('   ', k)
