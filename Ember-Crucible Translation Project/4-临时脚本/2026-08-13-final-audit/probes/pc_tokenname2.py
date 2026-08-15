# -*- coding: utf-8 -*-
"""人物一致性镜头 H：`scan_token_name.py` **明确排除**的那一档 —— 英文 name != tokenName。

scan_token_name 的 docstring 写明：「英文侧本来就不同的（`Kalasak the Cutter` 的 token
叫 `Kalasak`）是作者有意的短称，**不在判据内**」。
但「有意的短称」只保证**长度**可以不同，不保证**音译用字**可以不同。
本探针专查这一档里中文音译对不上的：

  EN name / EN tokenName 有包含关系（一方是另一方的子串，即真·短称）
  -> 中文 tokenName 的每个 2 字以上音译片段都应能在中文 name 里找到（或反之）。
"""
from __future__ import annotations
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_all

BILINGUAL_TAIL = re.compile(r'\s+[^一-鿿　-〿＀-￯]+$')
HAN = re.compile(r'[一-鿿]+')


def head(s):
    return BILINGUAL_TAIL.sub('', s).strip() or s.strip()


def main():
    rows = load_all()
    byactor = {}
    for repo, pack, path, en, cn in rows:
        m = re.match(r'^([^.]+)\.actors\.([^.]+)\.(name|tokenName)$', path)
        if m:
            byactor.setdefault((repo, pack, m.group(2)), {})[m.group(3)] = (en, cn)
    same, diff, flagged = 0, 0, []
    for key, d in sorted(byactor.items()):
        if 'name' not in d or 'tokenName' not in d:
            continue
        (en_n, cn_n), (en_t, cn_t) = d['name'], d['tokenName']
        if not cn_n or not cn_t:
            continue
        if en_n.strip() == en_t.strip():
            same += 1
            continue
        diff += 1
        hn, ht = head(cn_n), head(cn_t)
        # 中文 tokenName 的汉字片段是否都能在中文 name 里找到（或反过来）
        segs_t = [s for s in HAN.findall(ht) if len(s) >= 2]
        segs_n = [s for s in HAN.findall(hn) if len(s) >= 2]
        ok = all(s in hn for s in segs_t) or all(s in ht for s in segs_n)
        if not ok:
            flagged.append({'repo': key[0], 'pack': key[1], 'actor': key[2],
                            'en_name': en_n, 'en_token': en_t,
                            'cn_name': cn_n, 'cn_token': cn_t})
    print(f'英文 name==tokenName 的 actor {same}（scan_token_name 的辖区，已报 0）')
    print(f'英文 name!=tokenName 的 actor {diff}（scan_token_name 明确排除的一档）')
    print(f'  其中中文音译对不上：{len(flagged)}')
    for f in flagged:
        print(f"\n  {f['repo']}/{f['pack'][:16]} :: {f['actor']}")
        print(f"     EN name  = {f['en_name']!r}   EN token = {f['en_token']!r}")
        print(f"     CN name  = {f['cn_name']!r}")
        print(f"     CN token = {f['cn_token']!r}")
    json.dump(flagged, open('pc_tokenname2.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
