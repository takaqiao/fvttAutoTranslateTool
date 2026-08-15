# -*- coding: utf-8 -*-
"""lang 键的「上游代码还引用吗」探针（死翻译）

语料 = 上游包源码 + Foundry 核心源码（client / common / public / templates）。
判「可能可达」的规则（宁可放过，不可错杀）：
  a) 全键字面出现
  b) 任何 >=2 段的点号前缀出现（覆盖 `PREFIX.${dynamic}` 拼键）
  c) 形如 <P>.FIELDS.<path>.(label|hint)，且 <P> 出现在某个
     `LOCALIZATION_PREFIXES = [...]` 里（DataModel 自动本地化）
  d) TYPES.* / TYPES.<Doc>.<pkgId>.<subtype>（Foundry 核心按文档子类型自动查）
  e) 键名最后一段是复数形态 / 已知动态后缀

剩下的输出为 DEAD 候选，必须逐条人工核。
只读。
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

CORE = Path(r'C:/Program Files/Foundry Virtual Tabletop/resources/app')
CORE_ROOTS = [CORE / 'client', CORE / 'common', CORE / 'public' / 'scripts',
              CORE / 'templates', CORE / 'public' / 'lang']

def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def flat(o, pre=''):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items(): out.update(flat(v, f'{pre}.{k}' if pre else k))
    elif isinstance(o, str): out[pre] = o
    return out

EXTS = {'.mjs', '.js', '.hbs', '.html', '.json', '.ts'}
SKIP = {'assets', 'packs', 'fonts', 'icons', 'ui', 'audio', 'node_modules', 'dist'}

def corpus_of(roots):
    parts = []
    for root in roots:
        root = Path(root)
        if not root.exists(): continue
        for p in root.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in EXTS: continue
            rel = str(p).replace('\\', '/')
            if any(f'/{d}/' in rel for d in SKIP): continue
            if '/lang/' in rel and p.name != 'en.json': continue
            try: parts.append(p.read_text(encoding='utf-8', errors='ignore'))
            except Exception: pass
    return '\n'.join(parts)

def main():
    repo, pkg, label = sys.argv[1], sys.argv[2], sys.argv[3]
    cn = flat(load(Path(repo) / 'lang' / 'cn.json'))

    pkg_corpus = corpus_of([pkg])
    core_corpus = corpus_of(CORE_ROOTS)
    corpus = pkg_corpus + '\n' + core_corpus
    prefixes = set()
    for m in re.finditer(r'LOCALIZATION_PREFIXES\s*=\s*\[([^\]]*)\]', corpus):
        prefixes |= set(re.findall(r'["\'`]([^"\'`]+)["\'`]', m.group(1)))
    print(f'{label}: pkg语料 {len(pkg_corpus)} / 核心语料 {len(core_corpus)} 字符')
    print(f'  LOCALIZATION_PREFIXES 共 {len(prefixes)} 个: {sorted(prefixes)[:40]}')

    FORMS = {'zero','one','two','few','many','other'}
    dead = []
    for k in sorted(cn):
        if k in corpus: continue
        segs = k.split('.')
        if k.startswith('TYPES.'): continue
        if segs[-1] in FORMS and '.'.join(segs[:-1]) in corpus: continue
        # b) >=2 段前缀
        ok = False
        for i in range(len(segs) - 1, 1, -1):
            if '.'.join(segs[:i]) in corpus:
                ok = True; break
        if ok: continue
        # c) DataModel FIELDS
        if len(segs) >= 4 and segs[1] == 'FIELDS' and segs[-1] in ('label','hint') \
           and segs[0] in prefixes:
            continue
        dead.append(k)

    print(f'\n  DEAD 候选 {len(dead)} / 共 {len(cn)} 键')
    for k in dead:
        print(f'    {k}\n        = {cn[k]!r}')

if __name__ == '__main__':
    main()
