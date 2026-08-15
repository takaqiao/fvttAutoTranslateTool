#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
判据：**无差别补救 / 类型判据缺失，且挂在会真正落库的写入路径上**
（已确认实例：register.js 的 normalizeDescriptionValue 把物理物品的 {public,private}
 形状硬套到全部 item subtype，挂在 preUpdateItem 上）

把那一条抽象成两问，任何写入点都要能回答：

  Q1  这个写入点的**判据**（决定「要不要改 / 改成什么」的那个条件）是否
      覆盖了它能触达的**全部**目标？
      —— 反面：判据只对某一个 subtype / 某一种故障场景成立，却挂在无差别路径上。
  Q2  这个写入点写出去的东西**会不会落库**（或落进别人的持久数据）？
      —— 只改渲染出来的 DOM / 只改 CONFIG 内存对象 → 降一档；
         走 update/updateDocuments/createDocuments/setFlag → 数据破坏。

本脚本只做**候选枚举**，不下结论：
  A 面（代码）：把两个插件里所有「写」的语句抽出来，标注所在函数、挂载点、
                以及该函数体里**是否出现过类型/子类判据**（type / documentType /
                instanceof / schema / subtype / \.type ===）。没有判据 + 挂在
                无差别路径 = 候选。
  B 面（数据）：cn 包相对 en 基线的**值形状**逐叶比对。babele 2.9.1 的
                PrimitiveConverter 对静态字段有类型闸（primitive-converter.js:75-89：
                object/array → "structural"，number↔string → "type_mismatch"，
                两者都 return undefined＝整条跳过），所以形状不符的中文是**静默失效**
                而不是写坏；但**空字符串是合法 string，会被原样写进去**（map() 只挡
                undefined/null，field-mapping.js:61），等于用空白覆盖英文原文。
                这一档 prune_dead 看不见（它的 leaves() 对 "" 一样 yield 路径）。

假阳性模式（务必自己复核，不要直接采信本脚本的输出）：
  * A 面的「无判据」只看函数体文本，写在调用方的判据看不见 → 必然多报。
  * A 面认不出「只改 DOM」和「改文档」的区别，靠 sink 类型粗分，要人工确认。
  * B 面的 en 基线是 extract_en.mjs 按同一套 mapping 抽出来的，若抽取器本身漏抽，
    en 侧会缺路径，B 面就会把正常的中文报成「en 无此路径」。

只读，不写库。
"""
from __future__ import annotations
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REPOS = ['1-Ember汉化插件', '2-Crucible汉化插件']

# ----------------------------------------------------------------- A 面

WRITE_SINK = re.compile(
    r'(?P<sink>'
    r'\.update\s*\(|\.updateSource\s*\(|updateDocuments\s*\(|createDocuments\s*\('
    r'|updateEmbeddedDocuments\s*\(|createEmbeddedDocuments\s*\(|modifyBatch\s*\('
    r'|setFlag\s*\(|foundry\.utils\.setProperty\s*\(|foundry\.utils\.mergeObject\s*\('
    r'|\bdelete\s+\w+[\.\[]|setAttribute\s*\(|nodeValue\s*=[^=]|textContent\s*=[^=]'
    r'|innerHTML\s*=[^=]|\.label\s*=[^=]|\.name\s*=[^=]|\.abbreviation\s*=[^=]'
    r'|Object\.assign\s*\(|\.enricher\s*=[^=]'
    r')')

TYPE_GUARD = re.compile(
    r'\.type\s*===|\btype\s*===|documentType|instanceof\b|\.schema\b|subtype'
    r'|PHYSICAL|CruciblePhysicalItem|SYSTEM\.ITEM')

# 把「落库」和「只改内存/DOM」粗分开
PERSISTENT = ('.update(', 'updateDocuments(', 'createDocuments(', 'updateEmbeddedDocuments(',
              'createEmbeddedDocuments(', 'modifyBatch(', 'setFlag(', '.updateSource(')

FUNC_START = re.compile(r'^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)|^\s*const\s+(\w+)\s*=\s*(?:async\s*)?\(')
HOOK_LINE = re.compile(r'Hooks\.(?:on|once)\(\s*[\'"]([\w.]+)[\'"]')


def code_side():
    out = []
    for repo in REPOS:
        base = os.path.join(ROOT, repo)
        for dirpath, _dirs, files in os.walk(base):
            if '.git' in dirpath:
                continue
            for fn in files:
                if not fn.endswith(('.js', '.mjs')):
                    continue
                p = os.path.join(dirpath, fn)
                src = open(p, encoding='utf-8').read()
                lines = src.split('\n')
                # 函数边界（粗粒度：靠缩进 0 的 function 声明切段）
                bounds = []
                for i, ln in enumerate(lines):
                    m = FUNC_START.match(ln)
                    if m:
                        bounds.append((i, m.group(1) or m.group(2)))
                bounds.append((len(lines), None))

                def enclosing(idx):
                    name, start = '<top>', 0
                    for j in range(len(bounds) - 1):
                        if bounds[j][0] <= idx < bounds[j + 1][0]:
                            name, start = bounds[j][1], bounds[j][0]
                    return name, start, next((bounds[j + 1][0] for j in range(len(bounds) - 1)
                                              if bounds[j][0] <= idx < bounds[j + 1][0]), len(lines))

                # 钩子挂载点 -> 函数名
                mounts = {}
                for i, ln in enumerate(lines):
                    h = HOOK_LINE.search(ln)
                    if h:
                        blob = '\n'.join(lines[i:i + 4])
                        for fname in re.findall(r'(\w+)\s*\(', blob):
                            mounts.setdefault(fname, set()).add(h.group(1))

                for i, ln in enumerate(lines):
                    if ln.strip().startswith('*') or ln.strip().startswith('//'):
                        continue
                    m = WRITE_SINK.search(ln)
                    if not m:
                        continue
                    fname, s, e = enclosing(i)
                    body = '\n'.join(lines[s:e])
                    sink = m.group('sink').strip()
                    out.append({
                        'file': os.path.relpath(p, ROOT),
                        'line': i + 1,
                        'func': fname,
                        'sink': sink,
                        'persistent': any(k in ln for k in PERSISTENT),
                        'has_type_guard': bool(TYPE_GUARD.search(body)),
                        'mounted_on': sorted(mounts.get(fname, [])),
                        'code': ln.strip()[:120],
                    })
    return out


# ----------------------------------------------------------------- B 面

def walk(node, path=''):
    """产出 (path, kind, value)。kind ∈ {str, num, bool, null, obj, arr}"""
    if isinstance(node, dict):
        yield (path, 'obj', None)
        for k, v in node.items():
            yield from walk(v, f'{path}.{k}' if path else k)
    elif isinstance(node, list):
        yield (path, 'arr', len(node))
        for i, v in enumerate(node):
            yield from walk(v, f'{path}.{i}' if path else str(i))
    elif isinstance(node, str):
        yield (path, 'str', node)
    elif isinstance(node, bool):
        yield (path, 'bool', node)
    elif node is None:
        yield (path, 'null', None)
    else:
        yield (path, 'num', node)


def data_side():
    findings = {'empty_str': [], 'nonstring_scalar': [], 'shape_mismatch': [],
                'cn_only': 0, 'arr_len_diff': [], 'scanned_leaves': 0, 'scanned_files': 0}
    for repo in REPOS:
        cn_dir = os.path.join(ROOT, repo, 'compendium', 'cn')
        en_dir = os.path.join(ROOT, repo, 'compendium', 'en')
        if not os.path.isdir(cn_dir):
            continue
        for fn in sorted(os.listdir(cn_dir)):
            if not fn.endswith('.json'):
                continue
            ep = os.path.join(en_dir, fn)
            if not os.path.exists(ep):
                continue
            findings['scanned_files'] += 1
            en = {p: (k, v) for p, k, v in walk(json.load(open(ep, encoding='utf-8-sig')))}
            for p, k, v in walk(json.load(open(cn_dir + os.sep + fn, encoding='utf-8-sig'))):
                if k in ('str', 'num', 'bool', 'null'):
                    findings['scanned_leaves'] += 1
                ek = en.get(p, (None, None))[0]
                if ek is None:
                    if k in ('str', 'num', 'bool', 'null'):
                        findings['cn_only'] += 1
                    continue
                if k == 'str' and not v.strip():
                    findings['empty_str'].append(f'{repo}/{fn}::{p}')
                if k in ('num', 'bool', 'null'):
                    findings['nonstring_scalar'].append(f'{repo}/{fn}::{p} = {v!r}')
                if k != ek:
                    findings['shape_mismatch'].append(f'{repo}/{fn}::{p} cn={k} en={ek}')
                if k == 'arr' and ek == 'arr' and v != en[p][1]:
                    findings['arr_len_diff'].append(f'{repo}/{fn}::{p} cn={v} en={en[p][1]}')
    return findings


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('both', 'code'):
        rows = code_side()
        print(f'=== A 面：写入点 {len(rows)} 处 ===')
        for r in sorted(rows, key=lambda x: (not x['persistent'], x['has_type_guard'], x['file'], x['line'])):
            flag = 'PERSIST' if r['persistent'] else 'memory '
            guard = 'guard ' if r['has_type_guard'] else 'NOGUARD'
            print(f"  [{flag}][{guard}] {r['file']}:{r['line']} {r['func']}() "
                  f"mount={r['mounted_on'] or '-'}\n        {r['code']}")
    if which in ('both', 'data'):
        f = data_side()
        print(f"\n=== B 面：{f['scanned_files']} 个包 / {f['scanned_leaves']} 个叶子 ===")
        for key in ('empty_str', 'nonstring_scalar', 'shape_mismatch', 'arr_len_diff'):
            print(f'  {key}: {len(f[key])}')
            for s in f[key][:25]:
                print(f'      {s}')
        print(f"  cn 有 en 无（死键，prune_dead 覆盖）: {f['cn_only']}")
