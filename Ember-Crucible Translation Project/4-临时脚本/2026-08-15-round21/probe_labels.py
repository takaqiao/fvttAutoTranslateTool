#!/usr/bin/env python3
"""勘察二：label= 的值长什么样；以及 `readaloud="` 的**裸出现次数**是否等于正则捞到的数。

后者是反空转的关键 —— 若某个值里含 `]`，`@X\[[^\]]*\]` 会在那里截断，
参数被静默漏掉，而两个数字一比就能看出来。
"""
import json
import os
import re

P = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPOS = [os.path.join(P, '1-Ember汉化插件'), os.path.join(P, '2-Crucible汉化插件')]
AT_ENR = re.compile(r"@([A-Za-z]+)\[([^\]]*)\](?:\{([^{}]*)\})?")
ENR_PARAM = re.compile(r'\b([A-Za-z][\w-]*)\s*=\s*"([^"]*)"')

raw = {'readaloud': 0, 'label': 0}
via = {'readaloud': 0, 'label': 0}
labels = {'en': [], 'cn': []}

for repo in REPOS:
    for side, sub in (('en', 'en'), ('cn', 'cn')):
        d = os.path.join(repo, 'compendium', sub)
        for pack in sorted(os.listdir(d)):
            if not pack.endswith('.json'):
                continue
            txt = open(os.path.join(d, pack), encoding='utf-8').read()
            for n in raw:
                raw[n] += len(re.findall(n + r'\s*=\s*\\?"', txt))
            data = json.loads(txt)
            for m in AT_ENR.finditer(json.dumps(data, ensure_ascii=False)):
                pass
            # 逐叶走一遍才是真口径
            def rec(x):
                if isinstance(x, dict):
                    for v in x.values():
                        rec(v)
                elif isinstance(x, list):
                    for v in x:
                        rec(v)
                elif isinstance(x, str):
                    for m in AT_ENR.finditer(x):
                        for pm in ENR_PARAM.finditer(m.group(2)):
                            n = pm.group(1).lower()
                            if n in via:
                                via[n] += 1
                            if n == 'label':
                                labels[side].append(pm.group(2))
            rec(data)

print('裸出现（源文件字符串里 `xxx="`）:', raw)
print('正则捞到:', via)
print('EN label 值:', sorted(set(labels['en'])))
print('CN label 值:', sorted(set(labels['cn'])))
print('EN label 里含数字的:', [x for x in labels['en'] if re.search(r'\d', x)])
