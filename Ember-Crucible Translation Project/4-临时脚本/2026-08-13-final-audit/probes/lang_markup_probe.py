# -*- coding: utf-8 -*-
"""lang/cn.json 的「结构漂移」探针（现有 qa 判据只扫 compendium，lang 是盲区）

逐键比对 en / cn 的：
  1. HTML 标签多重集（<strong> <p> <em> <br> <ul> <li> …）
  2. 富文本增强器：@UUID[...]{...} / @Embed / &Reference / [[ ]] 内联骰
  3. 方括号 [...] 内容（本项目铁律：方括号内照抄）
  4. 首尾空白、换行 \n、不间断空格
  5. 纯符号/数字型值（EN 全是符号，CN 却改了）

只读。
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
from collections import Counter

TAG = re.compile(r'</?([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>')
UUID = re.compile(r'@[A-Za-z]+\[[^\]]*\]')
BRACKET = re.compile(r'\[([^\[\]]*)\]')
INLINE_ROLL = re.compile(r'\[\[[^\]]*\]\]')
ENT = re.compile(r'&[a-zA-Z]+;|&#\d+;')

def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def flat(o, pre=''):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items(): out.update(flat(v, f'{pre}.{k}' if pre else k))
    elif isinstance(o, str): out[pre] = o
    return out

def sig(s):
    return {
        'tags': Counter(m.lower() for m in TAG.findall(s)),
        'uuid': Counter(UUID.findall(s)),
        'brackets': Counter(BRACKET.findall(s)),
        'rolls': Counter(INLINE_ROLL.findall(s)),
        'ents': Counter(ENT.findall(s)),
        'nl': s.count('\n'),
        'lead': len(s) - len(s.lstrip()),
        'trail': len(s) - len(s.rstrip()),
        'nbsp': s.count('\u00a0'),
    }

def main():
    for repo, pkg, label in [
        (r'1-Ember汉化插件', r'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember', 'EMBER'),
        (r'2-Crucible汉化插件', r'C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible', 'CRUCIBLE'),
    ]:
        en = flat(load(Path(pkg) / 'lang' / 'en.json'))
        cn = flat(load(Path(repo) / 'lang' / 'cn.json'))
        print(f'\n===== {label}  en={len(en)} cn={len(cn)} =====')
        n = 0
        for k in sorted(en):
            if k not in cn:
                continue
            a, b = sig(en[k]), sig(cn[k])
            diffs = [f for f in a if a[f] != b[f]]
            if diffs:
                n += 1
                print(f'  [{",".join(diffs)}] {k}')
                print(f'     EN {en[k]!r}')
                print(f'     CN {cn[k]!r}')
        print(f'  -> 结构漂移 {n} 处')

        # 纯符号型英文值（不含字母）：CN 不该改
        sym = [k for k in en if k in cn and en[k].strip()
               and not re.search(r'[A-Za-z]', en[k]) and en[k] != cn[k]]
        print(f'  纯符号型 EN 但 CN 改过：{len(sym)}')
        for k in sym:
            print(f'     {k}  EN={en[k]!r}  CN={cn[k]!r}')

        # CN 与 EN 完全相同（未译）且含拉丁字母
        same = [k for k in en if k in cn and en[k] == cn[k] and re.search(r'[A-Za-z]', en[k])]
        keepp = Path(repo) / 'lang' / 'lang_keep_english.json'
        keep = set(load(keepp)) if keepp.exists() else set()
        same_nk = [k for k in same if k not in keep]
        print(f'  CN==EN 且含字母：{len(same)}（白名单外 {len(same_nk)}）')
        for k in same_nk:
            print(f'     {k}  = {en[k]!r}')

if __name__ == '__main__':
    main()
