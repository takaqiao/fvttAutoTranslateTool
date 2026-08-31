# -*- coding: utf-8 -*-
"""380 个形态素，各自能不能从**已有定译**里查到 —— 决定这活里「真要新裁」的有多少。

三层来源，按本项目既定的术语优先级从严到宽：
  ① 发布中的 TOKEN_MAKER_PART_IDS（同一张表里已经定过的同名部件段）
  ② glossary_ec（项目主词表）
  ③ 都查不到 ⇒ 本轮要新裁
"""
import io, json, re, subprocess, sys
sys.stdout.reconfigure(encoding='utf-8')
M = json.load(io.open('recon/morphemes.json', encoding='utf-8'))
toks = M['tokens']

# ① 发布中的部件表（用 node 把真身导出来，别正则抠）
js = r'''
import fs from "node:fs"; import path from "node:path"; import {pathToFileURL} from "node:url";
const SRC = process.argv[2];
const s = fs.readFileSync(SRC, "utf8");
const h = path.join(process.cwd(), "recon", "_ms.mjs");
fs.writeFileSync(h, "globalThis.Hooks = globalThis.Hooks ?? { once(){}, on(){} };\n"
 + s.replace("import * as SELFCHECK from './ember-cn-selfcheck.mjs';",
   "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck(){}, keyLiveness(){} };")
 + "\nexport { TOKEN_MAKER_PART_IDS };\n", "utf8");
const M = await import(pathToFileURL(h).href); fs.unlinkSync(h);
process.stdout.write(JSON.stringify(M.TOKEN_MAKER_PART_IDS));
'''
io.open('recon/_dump.mjs', 'w', encoding='utf-8').write(js)
SRC = "../../1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs"
out = subprocess.run(['node', 'recon/_dump.mjs', SRC], capture_output=True, text=True, encoding='utf-8')
assert out.returncode == 0, out.stderr[:600]
PART = json.loads(out.stdout)
print('① 发布中的部件表', len(PART), '键')

# ② glossary
G = json.load(io.open('../../5-其他内容/glossary/glossary_ec.json', encoding='utf-8'))
def gval(v):
    if isinstance(v, str): return v
    if isinstance(v, dict):
        for k in ('cn', 'canonical', 'value', 'zh'):
            if isinstance(v.get(k), str): return v[k]
    return None
GL = {k: gval(v) for k, v in G.items()}
GL = {k: v for k, v in GL.items() if v}
print('② glossary_ec', len(GL), '条可用')

rows = []
for t, c in toks.items():
    src = None; cn = None
    if t in PART: src, cn = '部件表', PART[t]
    elif t in GL: src, cn = 'glossary', GL[t]
    rows.append((t, c, src, cn))
by = {}
for t, c, s, cn in rows: by.setdefault(s or '新裁', []).append((t, c, cn))
print()
for k in ('部件表', 'glossary', '新裁'):
    v = by.get(k, [])
    print(f'{k:<10} 形态素 {len(v):>4} 种，覆盖出现次数 {sum(c for _, c, _ in v):>5}')
print()
new = sorted(by.get('新裁', []), key=lambda x: -x[1])
print('要新裁的形态素（按出现次数）：')
print('  ', ' · '.join(f'{t}×{c}' for t, c, _ in new[:80]))
io.open('recon/morph_sources.json', 'w', encoding='utf-8').write(json.dumps(
    {'fromPart': {t: cn for t, c, cn in by.get('部件表', [])},
     'fromGlossary': {t: cn for t, c, cn in by.get('glossary', [])},
     'toDecide': {t: c for t, c, _ in new}}, ensure_ascii=False, indent=1))
