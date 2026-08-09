#!/usr/bin/env python3
"""不靠长度、靠「跨语言不变量」找出中文没跟上英文的条目。

  python scan_content_coverage.py --repo <repo> [--pack <name>] [--out <json>] [--top N]

为什么需要它
------------
现有三种检查各有各的盲区，而且盲区**重叠**：

* `validate_translations.py` —— 路径上有中文就算已译，看不见内容对不对。
* `measure_8c.py` / `measure_stale_extra.py` —— 只比 `<p>`/`<li>` **块数**。
  上游「换掉内容但块数不变」时它们完全沉默（实证：`Lantern Roads/Impromptu Jail`
  标着「应补 1 块」，实际整段场景被换成了完全不同的文字）。
* `scan_markup_drift.py` 的 `TRUNCATED` —— 判据是「中文纯文本 < 英文的 0.22 倍」。
  上游把英文**改写**但长度相当时，比值仍落在正常区间，它一声不响。

于是「中文是照更早的英文写的」这一类，只有在**恰好变短**或**恰好动了标记**时才会被发现。
本项目有大量译文是**移植/继承**来的（孪生包 1.4 万条整条复制自 crucible 侧、
`Arcturel Tradeway` 28 页从改名前的旧路径搬来、v1.0.15 的存量），
这些都没有针对今天的英文逐句核对过 —— 正是这个盲区的高危区。

判据：跨语言不变量
------------------
1. **数字**。英文正文里的阿拉伯数字（DC、尺数、伤害、轮数、次数）在中文里必须还在。
   中文译文照惯例保留阿拉伯数字（「DC 15」「4 英尺」「持续 3 轮」），所以这条极稳。
2. **有定译的专名**。`glossary_ec.json` 里有中文定译的英文词，若英文正文出现了它，
   中文里就应当出现对应的中文（或该专名的其它已知写法）。

两者都**先剥掉标记再比**：`@UUID[...]`、`[[/...]]`、`&Reference[...]`、HTML 标签里的
数字属于机关参数，闸门已经在管，重复计入只会制造噪声。

误报来源（读结果时要知道）
--------------------------
* 英文用英文数词（"four"、"a dozen"）而中文写「四」—— 本脚本只看阿拉伯数字，不会误判；
  反过来英文写 `4` 中文写「四」会被报出来，属**真实**的风格不一致，值得看。
* 同一数字在英文里出现多次、中文合并成一次表述 —— 本脚本按**集合**比对，不按次数，已规避。
"""
from __future__ import annotations
import argparse
import json
import os
import re

CJK = re.compile(r'[一-鿿]')
TAG = re.compile(r'<[^>]+>')
MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]|&(?:amp;)?[Rr]eference\[[^\]]*\]')
NUM = re.compile(r'(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])')


def plain(s: str) -> str:
    """剥掉标记与标签，只留给人读的正文。"""
    return TAG.sub(' ', MARKUP.sub(' ', s))


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
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--pack')
    ap.add_argument('--out')
    ap.add_argument('--min-en', type=int, default=120,
                    help='英文纯文本短于此长度的条目不查（名字、标签噪声大）')
    ap.add_argument('--top', type=int, default=25)
    a = ap.parse_args()

    P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gloss = {}
    gp = os.path.join(P, '5-其他内容', 'glossary', 'glossary_ec.json')
    if os.path.exists(gp):
        for k, v in json.load(open(gp, encoding='utf-8')).items():
            zh = v.split(' ')[0].strip() if isinstance(v, str) else ''
            # 只用「多字中文 + 多字英文」的条目当锚点：单字中文误报率太高
            if len(k) >= 5 and len(zh) >= 2 and CJK.search(zh) and not re.search(r'[A-Za-z]', zh):
                gloss[k] = zh

    en_dir = os.path.join(a.repo, 'compendium', 'en')
    cn_dir = os.path.join(a.repo, 'compendium', 'cn')
    packs = [a.pack] if a.pack else sorted(
        f for f in os.listdir(en_dir)
        if f.endswith('.json') and os.path.exists(os.path.join(cn_dir, f)))

    rows = []
    checked = 0
    for pack in packs:
        o = []
        walk(json.load(open(os.path.join(en_dir, pack), encoding='utf-8')).get('entries', {}),
             json.load(open(os.path.join(cn_dir, pack), encoding='utf-8')).get('entries', {}),
             [], o)
        for path, e, c in o:
            if not (c and CJK.search(c)):
                continue
            pe, pc = plain(e), plain(c)
            if len(pe) < a.min_en:
                continue
            checked += 1
            miss_num = sorted(set(NUM.findall(pe)) - set(NUM.findall(pc)),
                              key=lambda x: -len(x))
            miss_term = [f'{k}→{v}' for k, v in gloss.items()
                         if re.search(r'\b' + re.escape(k) + r'\b', pe) and v not in pc]
            if miss_num or miss_term:
                rows.append({'pack': pack, 'path': path,
                             'missing_numbers': miss_num,
                             'missing_terms': miss_term[:6],
                             'en_len': len(pe), 'cn_len': len(pc),
                             'ratio': round(len(pc) / max(len(pe), 1), 2)})

    rows.sort(key=lambda r: -(len(r['missing_numbers']) * 2 + len(r['missing_terms'])))
    print(f'查了 {checked} 条已译且英文正文 ≥ {a.min_en} 字符的条目')
    print(f'  其中中文丢了英文里的数字或定译专名：**{len(rows)}** 条')
    hard = [r for r in rows if r['missing_numbers']]
    print(f'  丢数字的（信号最强，多半是漏译整句规则）：{len(hard)} 条')
    print(f'\n前 {a.top} 条：')
    for r in rows[:a.top]:
        print(f'  [{r["ratio"]}] {r["path"][-72:]}')
        if r['missing_numbers']:
            print(f'      缺数字: {r["missing_numbers"][:10]}')
        if r['missing_terms']:
            print(f'      缺专名: {r["missing_terms"][:4]}')
    if a.out:
        json.dump({'checked': checked, 'flagged': len(rows),
                   'with_missing_numbers': len(hard), 'items': rows},
                  open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
