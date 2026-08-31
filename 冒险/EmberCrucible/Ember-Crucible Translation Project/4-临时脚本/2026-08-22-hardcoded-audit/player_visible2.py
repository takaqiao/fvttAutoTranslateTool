# -*- coding: utf-8 -*-
"""玩家可见缺口 —— **修正版**（并入 12 个 patch* 的数据侧表之后重算）。"""
import io, re, sys, json, glob, collections, bisect
sys.stdout.reconfigure(encoding='utf-8')
RE_TOP = re.compile(r'^(?:export\s+)?(?:const|let|var|class|function|async function)\s+([A-Za-z_$][\w$]*)', re.M)
txt = io.open(r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs", encoding='utf-8').read()
tops = sorted((m.start(), m.group(1)) for m in RE_TOP.finditer(txt))
starts = [t[0] for t in tops]
span = {}
for i, (pos, name) in enumerate(tops):
    span.setdefault(name, (pos, tops[i+1][0] if i+1 < len(tops) else len(txt)))
SCENE_DEF = re.compile(r'managerClass\s*:|compositions\s*:|spritesheets\s*:')
VISTA_NEAR = re.compile(r'category\s*:|grounded\s*:|placements\s*:|spriteOptions|parallax\s*:')

f = [p for p in glob.glob('still.*.json') if 'JS' in p and len(json.load(io.open(p, encoding='utf-8'))) > 1000]
still = set(json.load(io.open(f[0], encoding='utf-8')))
owner = lambda p: tops[bisect.bisect_right(starts, p)-1][1] if starts else '?'

keep, drop = collections.defaultdict(list), collections.Counter()
seen = set()
for m in re.finditer(r'\b(label|title|hint|tooltip|placeholder|content|text|message|legend|caption|summary)\s*:\s*"([^"]*)"', txt):
    v = m.group(2)
    if v not in still or v in seen: continue
    seen.add(v)
    o = owner(m.start())
    ctx = txt[max(0, m.start()-220): m.end()+220]
    p, e = span.get(o, (0, 0))
    body = txt[p:e] if p else ''
    if VISTA_NEAR.search(ctx) or o.startswith(('SPRITES','NIGHT_TINT','VISTA_')) or o == 'EmberVistaConfiguration':
        drop['vista建图器'] += 1
    elif SCENE_DEF.search(body):
        drop['区域地图/场景定义'] += 1
    elif o.endswith('RegionBehavior') or o == 'registerProseMirrorBlocks':
        drop['GM 配置/编辑器'] += 1
    else:
        keep[o].append(v)
tot = sum(len(v) for v in keep.values())
print(f"未覆盖 {len(still)} 条 → 剔除 " + " · ".join(f"{k} {c}" for k, c in drop.items()) + f" ⇒ **候选玩家可见 {tot}**\n")
for o, vs in sorted(keep.items(), key=lambda kv: -len(kv[1])):
    print(f"  [{len(vs):>2}] {o}")
    print(f"       {' · '.join(sorted(vs))[:160]}")
io.open('worklist.json','w',encoding='utf-8').write(json.dumps({k: sorted(v) for k, v in keep.items()}, ensure_ascii=False, indent=1))
