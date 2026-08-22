# -*- coding: utf-8 -*-
"""从「其余」311 条里剔掉区域地图/Vista 的场景定义块，剩下的就是候选的玩家可见缺口。

剔除判据：该声明块内出现 `managerClass:` 或 `compositions:` 或 `spritesheets:`
⇒ 那是区域地图/Vista 的场景定义，它的 label 只在 GM 建图器里选，
   而层名在**玩家侧**由 Scene 文档那一份供给（已由 Babele 的 SCENE_LEVELS 翻译），
   代码里这份只是 `s.levels.get(id)?.name ?? cfg.compositions[id]?.label` 的**兜底**（ember.mjs:2376）。
"""
import io, re, sys, json, collections, bisect
sys.stdout.reconfigure(encoding='utf-8')
RE_TOP = re.compile(r'^(?:export\s+)?(?:const|let|var|class|function|async function)\s+([A-Za-z_$][\w$]*)', re.M)
txt = io.open(r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs", encoding='utf-8').read()
tops = sorted((m.start(), m.group(1)) for m in RE_TOP.finditer(txt))
starts = [t[0] for t in tops]
span = {}
for i, (pos, name) in enumerate(tops):
    end = tops[i + 1][0] if i + 1 < len(tops) else len(txt)
    span.setdefault(name, (pos, end))
SCENE_DEF = re.compile(r'managerClass\s*:|compositions\s*:|spritesheets\s*:')

rest = json.load(io.open('classify.json', encoding='utf-8'))["rest"]
keep, dropped = collections.defaultdict(list), collections.Counter()
for v, o in rest.items():
    p, e = span.get(o, (0, 0))
    if p and SCENE_DEF.search(txt[p:e]): dropped[o] += 1
    else: keep[o].append(v)
print(f"其余 {len(rest)} 条 → 剔掉区域地图/Vista 场景定义 {sum(dropped.values())} 条"
      f"（{len(dropped)} 个块）⇒ 候选玩家可见 {sum(len(v) for v in keep.values())} 条\n")
for o, vs in sorted(keep.items(), key=lambda kv: -len(kv[1])):
    print(f"  [{len(vs):>2}] {o}")
    print(f"       {' · '.join(sorted(vs))[:150]}")
