#!/usr/bin/env python3
"""现状勘察：把两仓 compendium 里带 readaloud=/label= 的增强器全捞出来。

目的只有一个 —— 在动 scan_content_coverage 之前，先自己数一遍 48 段 / 16966 字符
这个前提是不是真的，以及这些参数值里面有没有嵌套标记（[[/...]]、<p>、@UUID[...]）。
"""
import json
import os
import re
import sys

P = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPOS = [os.path.join(P, '1-Ember汉化插件'), os.path.join(P, '2-Crucible汉化插件')]

AT_ENR = re.compile(r"@([A-Za-z]+)\[([^\]]*)\](?:\{([^{}]*)\})?")
ENR_PARAM = re.compile(r'\b([A-Za-z][\w-]*)\s*=\s*"([^"]*)"')


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [k], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                 path + [str(i)], out)
    elif isinstance(en, str) and en.strip():
        out.append(('.'.join(path), en, cn if isinstance(cn, str) else None))


def main():
    stats = {}
    param_names = {}
    nested = {'[[': 0, '<': 0, '@': 0, ']': 0}
    leaves = set()
    samples = []
    for repo in REPOS:
        en_dir = os.path.join(repo, 'compendium', 'en')
        cn_dir = os.path.join(repo, 'compendium', 'cn')
        for pack in sorted(os.listdir(en_dir)):
            if not pack.endswith('.json'):
                continue
            cp = os.path.join(cn_dir, pack)
            if not os.path.exists(cp):
                continue
            o = []
            walk(json.load(open(os.path.join(en_dir, pack), encoding='utf-8')).get('entries', {}),
                 json.load(open(cp, encoding='utf-8')).get('entries', {}), [], o)
            for path, e, c in o:
                for side, val in (('en', e), ('cn', c)):
                    if not val:
                        continue
                    for m in AT_ENR.finditer(val):
                        for pm in ENR_PARAM.finditer(m.group(2)):
                            n = pm.group(1).lower()
                            v = pm.group(2)
                            param_names[n] = param_names.get(n, 0) + 1
                            if n not in ('readaloud', 'label'):
                                continue
                            k = (side, n)
                            s = stats.setdefault(k, {'n': 0, 'chars': 0, 'max': 0})
                            s['n'] += 1
                            s['chars'] += len(v)
                            s['max'] = max(s['max'], len(v))
                            if side == 'en' and n == 'readaloud':
                                leaves.add((os.path.basename(repo), pack, path))
                                if '[[' in v:
                                    nested['[['] += 1
                                if '<' in v:
                                    nested['<'] += 1
                                if '@' in v:
                                    nested['@'] += 1
                                if len(samples) < 3:
                                    samples.append((pack, path, v[:400]))
                        # 方括号里有没有 `]`（会把 [^\]]* 截断）
                        pass
    print('参数名频次（两侧合计）:', dict(sorted(param_names.items(), key=lambda t: -t[1])))
    for k in sorted(stats):
        print(k, stats[k])
    print('EN readaloud 所在叶数:', len(leaves))
    print('EN readaloud 值里的嵌套:', nested)
    for s in samples:
        print('---', s[0], s[1])
        print(s[2])


if __name__ == '__main__':
    main()
