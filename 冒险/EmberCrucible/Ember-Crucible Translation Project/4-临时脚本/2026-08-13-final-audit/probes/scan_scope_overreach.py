#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针：作用域外溢 / 破坏性写入 / 注释与实现不符（只读库）

抽象自已确认实例（register.js degradeActorUpdatePayload）：
  一段代码 (a) 通过某个**过窄的假设**（注释说是 X 专用 / fallback / 只在 Y 时）
  决定自己「是本项目的地盘」，(b) 实际触发面比假设宽得多，(c) 动作是**破坏性**的
  （删字段 / 改类型 / 覆盖别人的值 / 写库），(d) 失败或误伤时**没有任何提示**。

机械化判据（对本项目的 4 个运行时 JS 文件逐行）：
  R1 GLOBAL-WRITE  写入不属于本模块的全局命名空间：
                   game.i18n.translations / CONFIG.* / globalThis.* /
                   crucible.* / game.time.* / ui.* / <X>.documentClass
  R2 DESTRUCTIVE   delete / = [] / = {} / .filter( / 直接 = 覆盖已有字段 /
                   .textContent= / .nodeValue= / setAttribute(
  R3 UNGATED-HOOK  Hooks.on(<全局文档钩子>) 且 handler 体内没有
                   system.id / module.id / pack / type 之类的归属闸
  R4 TYPE-COERCE   把一个值**换成另一种类型**（string -> object 等）而调用点
                   没有类型判据（本项目 description 多态字段的老坑）
  R5 CLAIM-GAP     注释里出现 only / fallback / 只 / 仅 / 别去动 / 避免 /
                   Ember-specific 等收窄词，但同一函数体内没有对应的条件

假阳性模式（必须人工复核）：
  - R1 命中所有 CONFIG 读改，包括合法的「本模块自己塞进去的键」（__emberCn* 标记）
  - R3 会把带 try/catch 的防御性钩子也算进来
  - R5 纯靠词表，注释里提到「fallback」但确实实现了 fallback 的会误报
  → 本探针只产候选，结论一律以逐条阅读为准。
"""
from __future__ import annotations
import json
import os
import re
import sys

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'

TARGETS = [
    r'1-Ember汉化插件\register.js',
    r'1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs',
    r'1-Ember汉化插件\babele-mappings.js',
    r'2-Crucible汉化插件\babele-register.js',
    r'2-Crucible汉化插件\babele-mappings.js',
    r'3-常用脚本\extract\mappings.mjs',
    r'3-常用脚本\release\runtime-converters.js',
    r'3-常用脚本\release\generate_runtime.mjs',
]

# R1: 写入外部全局命名空间
GLOBAL_WRITE = re.compile(
    r'^\s*(?:'
    r'(game\.i18n\.translations[\w.\[\]"\']*)'
    r'|(CONFIG(?:\??\.)[\w.\[\]"\'?]*)'
    r'|(globalThis\.[\w.]+)'
    r'|(crucible(?:\??\.)[\w.]+)'
    r'|(game\.time\.[\w.]+)'
    r'|(ui\.[\w.]+)'
    r'|([\w$]+Class\.[\w]+)'
    r'|(entry\.\w+)|(hook\.\w+)|(cal\?\.\w+)|(v\.name)|(v\.abbreviation)|(entry\.label)'
    r')\s*(?:=[^=]|\+=)'
)

# R2: 破坏性动作
DESTRUCTIVE = re.compile(
    r'\bdelete\s+[\w.$\[\]\'"]+'
    r'|=\s*\[\s*\]\s*;'
    r'|=\s*\{\s*\}\s*;'
    r'|\.filter\('
    r'|\.textContent\s*='
    r'|\.nodeValue\s*='
    r'|setAttribute\('
    r'|\.splice\('
    r'|\.update\(|\.updateDocuments\(|updateEmbeddedDocuments\('
)

# R3: 全局文档钩子
DOC_HOOKS = re.compile(r"Hooks\.(on|once)\(\s*['\"](pre(Create|Update|Delete)\w+|render\w+)['\"]")

# 归属闸
GATE = re.compile(
    r"game\.system\.id|game\.modules\.get\(|\.pack\b|compendiumSource|"
    r"\btype\s*===|documentName|\.flags\?\.\[?['\"]?(ember|crucible)|"
    r"/ember/i|\^Ember|MODULE_ID\b"
)

# R4: 类型改写
TYPE_COERCE = re.compile(
    r"typeof\s+(\w[\w.]*)\s*===\s*'string'|typeof\s+(\w[\w.]*)\s*===\s*\"string\""
)

# R5: 收窄词
CLAIM = re.compile(
    r"\bonly\b|\bfallback\b|\bImport-specific\b|\bspecific\b|\bnever\b|"
    r"只(?!读不写)|仅|别去动|避免|不会|绝不"
)

FUNC = re.compile(r'^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)|^\s*const\s+(\w+)\s*=\s*(?:async\s*)?\(')


def func_at(lines, idx):
    for i in range(idx, -1, -1):
        m = FUNC.match(lines[i])
        if m:
            return (m.group(1) or m.group(2)), i + 1
    return '<top-level>', 0


def block_after(lines, idx, span=40):
    return '\n'.join(lines[idx:idx + span])


def main():
    out = {}
    for rel in TARGETS:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print('MISSING', p)
            continue
        src = open(p, encoding='utf-8').read()
        lines = src.split('\n')
        hits = []
        for i, ln in enumerate(lines):
            n = i + 1
            fn, fl = func_at(lines, i)
            if GLOBAL_WRITE.search(ln):
                hits.append(dict(rule='R1 GLOBAL-WRITE', line=n, fn=fn, text=ln.strip()[:160]))
            if DESTRUCTIVE.search(ln) and not ln.strip().startswith(('*', '//')):
                hits.append(dict(rule='R2 DESTRUCTIVE', line=n, fn=fn, text=ln.strip()[:160]))
            m = DOC_HOOKS.search(ln)
            if m:
                body = block_after(lines, i, 12)
                hits.append(dict(rule='R3 HOOK', line=n, fn=fn, text=ln.strip()[:160],
                                 gated=bool(GATE.search(body))))
            if TYPE_COERCE.search(ln):
                hits.append(dict(rule='R4 TYPE-COERCE', line=n, fn=fn, text=ln.strip()[:160]))
        # R5: 注释收窄词 vs 函数体
        for m in re.finditer(r'/\*\*(.*?)\*/', src, re.S):
            if not CLAIM.search(m.group(1)):
                continue
            ln = src[:m.start()].count('\n') + 1
            hits.append(dict(rule='R5 CLAIM', line=ln, fn='<jsdoc>',
                             text=re.sub(r'\s+', ' ', m.group(1))[:220]))
        out[rel] = hits
        print(f'\n########## {rel}  ({len(hits)} hits)')
        for h in hits:
            g = '' if 'gated' not in h else ('  [GATED]' if h['gated'] else '  [UNGATED]')
            print(f"  {h['rule']:<16} L{h['line']:<5} {h['fn']:<34}{g}  {h['text']}")

    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scope_overreach_candidates.json')
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n->', dst, sum(len(v) for v in out.values()), 'candidates')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
