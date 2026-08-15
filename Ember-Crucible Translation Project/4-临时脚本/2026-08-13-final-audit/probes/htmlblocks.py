#!/usr/bin/env python3
"""Minimal well-formed-HTML region finder for <section>/<div> with a class.

Returns list of (classname, start, end_of_open, close_start, close_end) using a
depth counter over the same tag name. The corpus is machine-generated Foundry
HTML; every <section>/<div> in it is explicitly closed (verified by balance
check in p4).
"""
import re

VOID = {'br', 'hr', 'img', 'input', 'meta', 'link'}


def regions(html, tag):
    """Yield (cls, inner_start, inner_end, outer_start, outer_end) for every <tag>."""
    open_re = re.compile(r'<' + tag + r'(\s[^>]*)?>', re.I)
    close_re = re.compile(r'</' + tag + r'\s*>', re.I)
    events = []
    for m in open_re.finditer(html):
        events.append((m.start(), 'o', m))
    for m in close_re.finditer(html):
        events.append((m.start(), 'c', m))
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'c' else 1))
    stack = []
    out = []
    for pos, kind, m in events:
        if kind == 'o':
            attrs = m.group(1) or ''
            cm = re.search(r'class="([^"]*)"', attrs)
            stack.append((cm.group(1) if cm else '', m.start(), m.end()))
        else:
            if not stack:
                continue
            cls, os_, oe = stack.pop()
            out.append((cls, oe, m.start(), os_, m.end()))
    out.sort(key=lambda r: r[3])
    return out


def balanced(html, tag):
    o = len(re.findall(r'<' + tag + r'(?:\s[^>]*)?>', html, re.I))
    c = len(re.findall(r'</' + tag + r'\s*>', html, re.I))
    return o == c, o, c
