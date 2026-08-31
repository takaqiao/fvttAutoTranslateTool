#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""探针：找「为窄故障场景写的补救被无条件套到全部目标上」这一类。

抽象出的四问判据（对插件运行时代码的每个**写/删/强转**点各问一遍）：
  Q1 触发闸  —— 这次写入前，有没有「这个目标真的坏了吗」的判据？
                （只有 `typeof x === 'string'` 这种**形状**判据不算 —— 形状本来就可能是合法的，
                  必须再问 Q3）
  Q2 目标面  —— 名字/注释说的是窄场景（legacy / malformed / 某个 id），实际遍历的是不是全集？
  Q3 前提    —— 它假定的「正确形状」在上游 schema 里成立吗？（要人工用 upstream 源码核）
  Q4 自噬    —— 这次无条件改写，会不会让它后面自己的兜底/catch 分支变成不可达？

本脚本机械化 Q1/Q2/Q4 的**候选定位**（Q3 必须人工核 upstream），输出：
  站点行号 · 操作 · 所在函数 · 上游 25 行内的守卫条件 · 是否遍历全集 · 是否为窄命名

只读。假阳性模式（必须知道）：
  - 纯正则，不建 AST：`if` 条件跨行时守卫可能抓不全；
  - 「遍历全集」只看同函数内有没有 `for (... of game.items / game.actors / updates / .map(`，
    对通过参数传进来的集合无能为力；
  - 「窄命名」只看函数名/注释里的 legacy|malformed|safe|degrade|sanitize|fallback|compat 等词。
  所以输出是**候选**，不是结论。
"""
import os, re, sys

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
TARGETS = [
    r'1-Ember汉化插件\register.js',
    r'1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs',
    r'1-Ember汉化插件\babele-mappings.js',
    r'2-Crucible汉化插件\babele-register.js',
    r'2-Crucible汉化插件\babele-mappings.js',
    r'3-常用脚本\release\runtime-converters.js',
    r'3-常用脚本\release\generate_runtime.mjs',
]

MUT = [
    ('delete',      re.compile(r'\bdelete\s+([A-Za-z_$][\w$.\[\]\'"]*)')),
    ('=[]',         re.compile(r'([A-Za-z_$][\w$.\[\]\'"]*)\s*=\s*\[\]\s*;')),
    ('={}',         re.compile(r'([A-Za-z_$][\w$.\[\]\'"]*)\s*=\s*\{\}\s*;')),
    ('assign',      re.compile(r'^\s*([A-Za-z_$][\w$]*(?:\.[\w$]+|\[[^\]]+\])+)\s*=\s*[^=]')),
    ('.update(',    re.compile(r'\.update(?:EmbeddedDocuments)?\(')),
    ('setProperty', re.compile(r'foundry\.utils\.setProperty\(')),
    ('mergeObject', re.compile(r'foundry\.utils\.mergeObject\(')),
    ('setFlag',     re.compile(r'\.setFlag\(')),
]
NARROW = re.compile(r'legacy|malformed|degrade|safe|sanitize|fallback|compat|repair|migrat|兜底|修复|坏|异常', re.I)
BROAD  = re.compile(r'for\s*\(\s*const\s+\w+\s+of\s+(game\.items|game\.actors|actor\.items|updates|Object\.values|enrichers|targets)|\.map\(|\.filter\(')
GUARD  = re.compile(r'^\s*(if|else if)\s*\(|^\s*if\s*\(.*\)\s*(return|continue)')

def func_of(lines, i):
    for j in range(i, -1, -1):
        m = re.match(r'\s*(?:async\s+)?function\s+([\w$]+)|\s*(?:export\s+)?(?:async\s+)?function\s+([\w$]+)', lines[j])
        if m:
            return m.group(1) or m.group(2)
        m = re.match(r"\s*Hooks\.(on|once)\('([\w.]+)'", lines[j])
        if m:
            return f'Hooks.{m.group(1)}({m.group(2)})'
    return '?'

def body_of(lines, i):
    """函数体粗略范围：从函数头到下一个顶格 } 之后。"""
    start = 0
    for j in range(i, -1, -1):
        if re.match(r'\s*(?:export\s+)?(?:async\s+)?function\s|\s*Hooks\.(on|once)\(', lines[j]):
            start = j; break
    end = len(lines)
    for j in range(i, len(lines)):
        if re.match(r'^\}', lines[j]):
            end = j; break
    return start, end

total = 0
for rel in TARGETS:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        print(f'!! 缺文件 {rel}'); continue
    lines = open(p, encoding='utf-8').read().split('\n')
    hits = []
    for i, ln in enumerate(lines):
        if ln.strip().startswith(('*', '//', '/*')):
            continue
        for op, rx in MUT:
            m = rx.search(ln)
            if not m:
                continue
            fn = func_of(lines, i)
            s, e = body_of(lines, i)
            body = '\n'.join(lines[s:e])
            guards = [lines[j].strip() for j in range(max(s, i - 25), i) if GUARD.match(lines[j])]
            hits.append(dict(line=i + 1, op=op, fn=fn,
                             narrow=bool(NARROW.search(fn) or NARROW.search('\n'.join(lines[max(0, s-12):s+1]))),
                             broad=bool(BROAD.search(body)),
                             guards=guards[-3:], text=ln.strip()[:96]))
            break
    print(f'\n================ {rel}   ({len(hits)} 个写/删点)')
    for h in hits:
        flag = ''
        if h['narrow'] and h['broad']:
            flag = '  <<< 窄命名 + 全集遍历'
        print(f"  L{h['line']:<5} {h['op']:<12} {h['fn']:<38} {h['text']}{flag}")
        for g in h['guards']:
            print(f"        guard: {g[:100]}")
    total += len(hits)
print(f'\n合计 {total} 个写/删点')
