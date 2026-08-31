# -*- coding: utf-8 -*-
"""探针：把仓库里两个「无作用域批量改写器」按**原样**在当前库上空跑一遍，
数它们现在会改坏多少条 —— 与种子同类（批量写 + 只看形状 + 不看归属），
只是集合从「世界文档」换成了「翻译库全部字符串」。

被查对象（都在**发布仓库**里，不是临时脚本）：
  1. 2-Crucible汉化插件/scripts/fix_word_leaks.py
  2. 2-Crucible汉化插件/scripts/repair_bilingual_names.py

做法：原样 import 它们的核心函数（不改一个字符），喂当前 compendium/cn，
对比前后。**不写回任何文件。**

假阳性模式：
  - 「会改」不等于「改错」。所以输出把每条的**英文原文**一起打出来，
    由人判断这次替换有没有英文依据；统计里也把「英文侧确实有这个词」和
    「英文侧根本没有」分开计数。
  - repair_bilingual_names 的 is_broken 有一档是「纯中文但含空格 → 判为重复翻译」，
    对含空格的正常中文名是误判，这一档单列。
"""
import importlib.util
import io
import json
import os
import re
import sys
from collections import Counter

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPO = os.path.join(ROOT, '2-Crucible汉化插件')
EMBER = os.path.join(ROOT, '1-Ember汉化插件')


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def leaves(o, path=(), key=None):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, path + (k,), k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, path + (str(i),), key)
    elif isinstance(o, str):
        yield path, key, o


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def run_wordleaks():
    src = os.path.join(REPO, 'scripts', 'fix_word_leaks.py')
    # 直接 exec 到独立命名空间，避免它的 main() 被触发
    ns = {'__name__': 'wl_probe', '__file__': src}
    exec(compile(open(src, encoding='utf-8').read(), src, 'exec'), ns)
    fix_string, SKIP_KEYS = ns['fix_string'], ns['SKIP_KEYS']
    TRANS = ns['TRANS']

    print('=' * 70)
    print('fix_word_leaks.py —— 现在跑一遍会改多少条')
    for label, repo in (('crucible-cn', REPO), ('ember_cn', EMBER)):
        cn_dir = os.path.join(repo, 'compendium', 'cn')
        en_dir = os.path.join(repo, 'compendium', 'en')
        if not os.path.isdir(cn_dir):
            continue
        n_changed = 0
        no_en_basis = 0
        by_word = Counter()
        samples = []
        for fn in sorted(os.listdir(cn_dir)):
            if not fn.endswith('.json'):
                continue
            cn = load(os.path.join(cn_dir, fn))
            ep = os.path.join(en_dir, fn)
            en_map = {p: v for p, _k, v in leaves(load(ep))} if os.path.exists(ep) else {}
            for path, key, val in leaves(cn):
                if key in SKIP_KEYS:
                    continue
                new = fix_string(val)
                if new == val:
                    continue
                n_changed += 1
                en_val = en_map.get('.'.join(path)) if False else en_map.get(path, '')
                words = [w for w in TRANS
                         if re.search(r'(?<![A-Za-z])' + re.escape(w) + r'(?![A-Za-z])', val)]
                for w in words:
                    by_word[w] += 1
                basis = any(en_val and re.search(r'(?<![A-Za-z])' + re.escape(w) + r'(?![A-Za-z])', en_val)
                            for w in words)
                if not basis:
                    no_en_basis += 1
                    if len(samples) < 14:
                        samples.append((fn, '.'.join(path), words, val, en_val))
        print(f'\n-- {label}: 会被改写的叶子 {n_changed} 条，其中**英文原文里根本没有那个词**的 {no_en_basis} 条')
        print(f'   命中词频 top: {by_word.most_common(12)}')
        for fn, p, words, val, en_val in samples:
            print(f'\n   {fn} :: {p}   〔{words}〕')
            print(f'     CN: {val[:150]}')
            print(f'     EN: {(en_val or "(英文侧无此路径)")[:150]}')


def run_bilingual():
    src = os.path.join(REPO, 'scripts', 'repair_bilingual_names.py')
    ns = {'__name__': 'rb_probe', '__file__': src}
    exec(compile(open(src, encoding='utf-8').read(), src, 'exec'), ns)
    is_broken, rebuild = ns['is_broken'], ns['rebuild']

    print('\n' + '=' * 70)
    print('repair_bilingual_names.py —— 现在跑一遍会重写多少个 name')
    for label, repo in (('crucible-cn', REPO), ('ember_cn', EMBER)):
        cn_dir = os.path.join(repo, 'compendium', 'cn')
        en_dir = os.path.join(repo, 'compendium', 'en')
        if not os.path.isdir(cn_dir):
            continue
        hits = []

        def walk(cn, en):
            if isinstance(cn, dict) and isinstance(en, dict):
                if 'name' in cn and 'name' in en and is_broken(cn['name'], en['name']):
                    hits.append((cn['name'], en['name'], rebuild(cn['name'], en['name'])))
                for k, v in cn.items():
                    if k in en:
                        walk(v, en[k])
            elif isinstance(cn, list) and isinstance(en, list):
                for a, b in zip(cn, en):
                    walk(a, b)

        for fn in sorted(os.listdir(cn_dir)):
            if not fn.endswith('.json'):
                continue
            ep = os.path.join(en_dir, fn)
            if not os.path.exists(ep):
                continue
            walk(load(os.path.join(cn_dir, fn)), load(ep))
        lossy = [(o, e, n) for o, e, n in hits if len(n) < len(o)]
        print(f'\n-- {label}: 会被重写的 name {len(hits)} 个，其中**变短（丢字）**的 {len(lossy)} 个')
        for o, e, n in hits[:20]:
            flag = '  ← 丢字' if len(n) < len(o) else ''
            print(f'   EN {e!r}\n     旧 {o!r}\n     新 {n!r}{flag}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    run_wordleaks()
    run_bilingual()
