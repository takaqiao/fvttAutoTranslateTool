# -*- coding: utf-8 -*-
"""探针：`fix_word_leaks.py` 这类**无英文闸的批量词替换**在库里留下的残迹。

被查工具：`2-Crucible汉化插件/scripts/fix_word_leaks.py`
  - 遍历 compendium/cn/*.json 全部字符串 + lang/cn.json；
  - 准入条件只有两条：**字符串里有 CJK** + 键不在 {name,label,title,mapping}；
  - 然后拿一张 46 条的英译中表做全局正则替换，**一次都没有读过 compendium/en**。
  与种子同类：批量写入 + 只看当前形状（这串里有没有这个英文词）+ 不看归属
  （英文原文到底写的是不是这个词、这个词是不是短语的一半）。

本探针找两种可判定的残迹：
  A. **半截译**：译文里出现 `中文替换词` 紧跟着一段英文（如 `致命 Success`），
     因为替换用的是单词边界，`Critical Success` 只有前半个词在表里。
  B. **无英文依据**：译文叶子含 `中文替换词`，而**同路径的英文叶子里根本没有那个英文词**
     —— 说明这个中文词不是从那个英文词译过来的，替换是误伤（或至少无依据）。
     B 有大量合法情况（中文词本来就是别的英文词的正常译名），所以 B 只作候选，必须逐条看英文。

假阳性模式：
  - A 类：译名约定本来就是「中文 English」双语并列，凡 name/label/title 键都要排除
    （fix_word_leaks 自己也跳过这些键，所以本探针同样跳过，两边定义域一致）。
  - B 类：如上，噪声很大，只用来定位，不用来下结论。
只读，不写库。
"""
import json
import os
import re
import sys

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPOS = [
    ('crucible-cn', os.path.join(ROOT, '2-Crucible汉化插件')),
    ('ember_cn', os.path.join(ROOT, '1-Ember汉化插件')),
]

TRANS = {
    'Tier': '等级', 'TODO': '待办', 'WIP': '待完成', 'feet': '英尺', 'Feet': '英尺',
    'Hand': '只手', 'Number': '数量', 'Capacity': '负载', 'Configure': '配置',
    'Group': '群组', 'Open': '打开', 'Standard': '标准', 'Critical': '致命',
    'Dominated': '支配', 'Bomb': '炸弹', 'Glyphs': '符印', 'Toss': '投掷',
    'Push': '推开', 'Shoddy': '劣质', 'occupants': '名居住者', 'horns': '角',
    'Displace': '位移', 'grotesque': '狰狞', 'practitioners': '施法者',
    'Bandit': '强盗', 'conjured': '召唤而出', 'compendium': '合集包',
    'Compendium': '合集包', 'ActiveEffect': '主动效果', 'slug': '标识符',
    'Ward': '守护', 'Blossom': '绽放', 'Token': '令牌', 'Fine': '精良',
    'Horrific': '骇人', 'Wrestler': '摔角手', 'Hellguard': '地狱守卫',
    'Inflection': '屈折', 'Inflections': '屈折', 'Gesture': '手势',
    'pack': '包', 'Superior': '卓越', 'Glyphweaver': '符印编织师',
}
SKIP_KEYS = {'name', 'label', 'title', 'mapping'}
CJK = re.compile(r'[\u4e00-\u9fff]')


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


def strip_markup(s):
    """把 fix_word_leaks 自己屏蔽掉的那几类标记去掉，避免拿链接/属性当正文。"""
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'@[A-Za-z]+\[[^\]]+\](?:\{[^}]*\})?', ' ', s)
    s = re.sub(r'\[\[[^\]]*\]\]', ' ', s)
    s = re.sub(r'Compendium\.[\w\-.]+', ' ', s)
    s = re.sub(r'https?://\S+', ' ', s)
    return s


def main():
    grand_a = grand_b = 0
    for label, repo in REPOS:
        cn_dir = os.path.join(repo, 'compendium', 'cn')
        en_dir = os.path.join(repo, 'compendium', 'en')
        if not os.path.isdir(cn_dir):
            continue
        print(f'===== {label}')
        for fn in sorted(os.listdir(cn_dir)):
            if not fn.endswith('.json'):
                continue
            cn = load(os.path.join(cn_dir, fn))
            ep = os.path.join(en_dir, fn)
            en_map = {}
            if os.path.exists(ep):
                en_map = {p: v for p, _k, v in leaves(load(ep))}
            for path, key, val in leaves(cn):
                if key in SKIP_KEYS or not CJK.search(val):
                    continue
                body = strip_markup(val)
                en_val = strip_markup(en_map.get(path, ''))
                for en_w, cn_w in TRANS.items():
                    if cn_w not in body:
                        continue
                    # A: 中文替换词后面紧跟一段英文（允许一个空格）
                    for m in re.finditer(re.escape(cn_w) + r'[ \u00a0]?([A-Za-z][A-Za-z\'-]+)', body):
                        i = m.start()
                        print(f'  [A] {fn} :: {".".join(path)}')
                        print(f'      CN …{body[max(0,i-30):i+50]}…')
                        if en_val:
                            j = en_val.find(en_w)
                            print(f'      EN …{en_val[max(0,j-30):j+50] if j>=0 else en_val[:80]}…')
                        grand_a += 1
                        break
                    # B: 英文侧根本没有这个词
                    if en_val and not re.search(r'(?<![A-Za-z])' + re.escape(en_w) + r'(?![A-Za-z])', en_val):
                        grand_b += 1
    print(f'\nA 类（半截译，硬证据）合计 {grand_a}')
    print(f'B 类（无英文依据，仅候选）合计 {grand_b}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
