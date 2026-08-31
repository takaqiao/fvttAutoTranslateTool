# -*- coding: utf-8 -*-
"""lang/cn.json 的三类只读探测（本轮镜头：运行时可达性 + 占位符）

A) placeholder  —— 逐键比对 en/cn 的 {xxx} 占位符**多重集合**
B) reachable    —— 每个键在上游代码/模板里能不能被找到（全键字面 / 点号前缀）
C) plural       —— zero/one/two/few/many/other 六形态齐不齐、内容是否恒同

只读，不写任何库文件。
"""
from __future__ import annotations
import json, re, sys, os
from pathlib import Path
from collections import Counter

PH = re.compile(r'\{([^{}]*)\}')

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def flatten(obj, prefix=''):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f'{prefix}.{k}' if prefix else k))
    elif isinstance(obj, str):
        out[prefix] = obj
    return out

def phset(s):
    # 只把「像标识符」的当占位符；@UUID[...]{中文标签} 那种富文本标签排除
    return Counter(m for m in PH.findall(s) if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', m or ''))

def rich_braces(s):
    """非标识符的 {...}：可能是富文本标签，也可能是拼错的占位符"""
    return [m for m in PH.findall(s) if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', m or '')]


def read_corpus(roots, exts, skip_dirs):
    blobs = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            rel = str(p).replace('\\', '/')
            if any(f'/{d}/' in rel for d in skip_dirs):
                continue
            try:
                blobs.append(p.read_text(encoding='utf-8', errors='ignore'))
            except Exception:
                pass
    return '\n'.join(blobs)


def main():
    repo, pkg, label = sys.argv[1], sys.argv[2], sys.argv[3]
    en = flatten(load(Path(pkg) / 'lang' / 'en.json'))
    cn_raw = load(Path(repo) / 'lang' / 'cn.json')
    cn = flatten(cn_raw)

    print(f'===== {label} =====')
    print(f'en {len(en)} 键 / cn {len(cn)} 键')

    # ---- A 占位符 ----
    ph_bad = []
    for k, e in en.items():
        if k not in cn:
            continue
        a, b = phset(e), phset(cn[k])
        if a != b:
            ph_bad.append((k, sorted(a.elements()), sorted(b.elements()), e, cn[k]))
    print(f'\n[A] 占位符不一致 {len(ph_bad)}')
    for k, a, b, e, c in ph_bad:
        print(f'  {k}\n     en {a}  ::  {e}\n     cn {b}  ::  {c}')

    # 中文侧出现的「非标识符花括号」
    rb = [(k, rich_braces(v)) for k, v in cn.items() if rich_braces(v)]
    rb_en = {k: rich_braces(en[k]) for k in en if rich_braces(en[k])}
    print(f'\n[A2] cn 侧非标识符花括号 {len(rb)}（en 侧 {len(rb_en)}）')
    for k, m in rb:
        same = rb_en.get(k)
        flag = 'SAME_AS_EN' if same == m else f'EN={same}'
        print(f'  {k}  {m}  [{flag}]')

    # ---- C 复数形态 ----
    FORMS = ['zero', 'one', 'two', 'few', 'many', 'other']
    stems = {}
    for k in en:
        parts = k.rsplit('.', 1)
        if len(parts) == 2 and parts[1] in FORMS:
            stems.setdefault(parts[0], set()).add(parts[1])
    print(f'\n[C] 上游带复数形态的词条 {len(stems)}')
    for s, fs in sorted(stems.items()):
        cn_fs = {f for f in FORMS if f'{s}.{f}' in cn}
        miss = fs - cn_fs
        vals = {cn.get(f'{s}.{f}') for f in cn_fs}
        print(f'  {s}: en={sorted(fs)} cn={sorted(cn_fs)} 缺={sorted(miss) or "-"} cn取值数={len(vals)}')

    # ---- B 可达性 ----
    corpus = read_corpus(
        [pkg],
        {'.mjs', '.js', '.hbs', '.html', '.json'},
        {'lang', 'assets', 'packs', 'ui', 'fonts', 'icons', 'styles', 'audio'},
    )
    print(f'\n[B] 代码语料 {len(corpus)} 字符')

    def seg_prefixes(k):
        p = k.split('.')
        return ['.'.join(p[:i]) for i in range(len(p), 0, -1)]

    dead, dyn = [], []
    for k in sorted(cn):
        if k in corpus:
            continue
        hit = None
        for pre in seg_prefixes(k)[1:]:
            if len(pre.split('.')) < 2:
                break
            if pre in corpus:
                hit = pre
                break
        if hit:
            dyn.append((k, hit))
        else:
            dead.append(k)
    print(f'  全键字面命中不了、且 >=2 段前缀也命中不了 → 可疑死键 {len(dead)}')
    for k in dead:
        print(f'    DEAD? {k}   = {cn[k]!r}')
    print(f'  全键命中不了但有前缀命中（可能是动态拼键）{len(dyn)}')
    for k, h in dyn:
        print(f'    DYN   {k}   (prefix {h})')


if __name__ == '__main__':
    main()
