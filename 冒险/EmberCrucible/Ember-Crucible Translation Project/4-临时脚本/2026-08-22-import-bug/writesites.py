# -*- coding: utf-8 -*-
"""把两个插件里**所有对已有对象属性的赋值**列全，看清楚我们到底改了哪些字段。

判据：形如 `X.prop = …` 或 `X[…] = …` 的赋值（排除 const/let/var 声明、比较、箭头函数）。
目的是回答「会不会影响正常代码」—— 只要写的全是展示字段，就碰不到逻辑。
"""
import io, re, sys, collections
sys.stdout.reconfigure(encoding='utf-8')
FILES = {
    'ember': '1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs',
    'crucible': '2-Crucible汉化插件/crucible-hardcoded-cn.mjs',
    'crucible/lang-reclaim': '2-Crucible汉化插件/lang-reclaim.js',
    'ember/babele-mappings': '1-Ember汉化插件/babele-mappings.js',
}
# `something.prop =` （不是 == / === / =>）
RE = re.compile(r'(?<![=!<>])\b([A-Za-z_$][\w$]*(?:\??\.[A-Za-z_$][\w$]*)*)\s*=\s*(?![=>])')
SKIP = re.compile(r'^\s*(?://|\*|/\*)')
for tag, p in FILES.items():
    props = collections.Counter()
    other = []
    for i, line in enumerate(io.open(p, encoding='utf-8', newline=''), 1):
        if SKIP.match(line): continue
        s = line.strip()
        if re.match(r'^(const|let|var|export|import|function|async|for|if|while|return)\b', s): continue
        for m in RE.finditer(line):
            t = m.group(1)
            if '.' not in t: continue
            props[t.split('.')[-1]] += 1
    # setAttribute / defineProperty 另算
    txt = io.open(p, encoding='utf-8').read()
    attrs = re.findall(r'setAttribute\(\s*"([^"]+)"', txt)
    dyn = len(re.findall(r'setAttribute\(\s*(?:rule\.)?attr\b', txt))
    dp = re.findall(r'Object\.defineProperty\([^,]+,\s*([^,]+),', txt)
    print(f'== {tag}')
    print(f'   属性赋值：{dict(props) if props else "（无）"}')
    print(f'   setAttribute 字面量：{sorted(set(attrs)) or "（无）"}' + (f' · 变量形式 {dyn} 处（取白名单）' if dyn else ''))
    if dp: print(f'   defineProperty：{len(dp)} 处')
    print()
