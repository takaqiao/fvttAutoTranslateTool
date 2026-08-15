# -*- coding: utf-8 -*-
"""G1 · 阵营缩写写法统一：把「EN 缩写 -> CN 中文全称」的 31 处改回保留字母。
只读探针：打印每个叶子里 CN 中文全称的出现位置与配对的 EN 缩写，供人工核对。"""
import json, os, re, sys
sys.stdout.reconfigure(encoding='utf-8')

ZH = {'lawful good': '守序善良', 'neutral good': '中立善良', 'chaotic good': '混乱善良',
      'lawful neutral': '守序中立', 'neutral': '中立', 'chaotic neutral': '混乱中立',
      'lawful evil': '守序邪恶', 'neutral evil': '中立邪恶', 'chaotic evil': '混乱邪恶',
      'unaligned': '无阵营'}
ABBR = {'LG': 'lawful good', 'NG': 'neutral good', 'CG': 'chaotic good',
        'LN': 'lawful neutral', 'N': 'neutral', 'TN': 'neutral', 'CN': 'chaotic neutral',
        'LE': 'lawful evil', 'NE': 'neutral evil', 'CE': 'chaotic evil',
        'U': 'unaligned', 'NN': 'neutral'}
ZH_ALT = sorted(set(ZH.values()), key=len, reverse=True)
AB_ALT = sorted(ABBR, key=len, reverse=True)
FULL_ALT = [f.replace(' ', r'\s+') for f in sorted(ZH, key=len, reverse=True)]

EN_TAG = re.compile(r'[(（]\s*(' + '|'.join(AB_ALT + FULL_ALT) + r')\s*[,，]', re.I)
CN_TAG = re.compile(r'[(（]\s*(' + '|'.join(ZH_ALT) + r')\s*[,，]')

LEAVES = [
    "Ember Early Access.journals.Ordain Gazetteer.pages.Westgate.text",
    "Ember Early Access.journals.Ordain Gazetteer.pages.Scholar's Nook.text",
    "Ember Early Access.journals.Organizations.pages.Burnished Hand.text",
    "Ember Early Access.journals.Organizations.pages.House Bastilla.text",
    "Ember Early Access.journals.Organizations.pages.House Cevher.text",
    "Ember Early Access.journals.Glitter in the Dark.pages.A Troubled Tradeway.text",
    "Ember Early Access.journals.Glitter in the Dark.pages.Scene of the Crime.text",
    "Ember Early Access.journals.Smoldering Cinders.pages.Lost in the Stacks.text",
]
REPO = '1-Ember汉化插件'


def get(root, path):
    node = root
    for p in path.split('.'):
        node = node[p]
    return node


ZH_REV = {v: k for k, v in ZH.items()}


def norm(tok):
    t = re.sub(r'\s+', ' ', tok.strip())
    if t in ZH_REV:
        return ZH_REV[t]
    if t.upper() in ABBR:
        return ABBR[t.upper()]
    return t.lower() if t.lower() in ZH else None


def main():
    out = {}
    for pack in ('ember.adventure.json', 'ember.crucible-adventure.json'):
        en = json.load(open(os.path.join(REPO, 'compendium', 'en', pack), encoding='utf-8'))['entries']
        cn = json.load(open(os.path.join(REPO, 'compendium', 'cn', pack), encoding='utf-8'))['entries']
        for path in LEAVES:
            e, c = get(en, path), get(cn, path)
            ems = list(EN_TAG.finditer(e))
            cms = list(CN_TAG.finditer(c))
            print(f'\n== {pack} :: {path[-52:]}  EN {len(ems)} / CN(中文全称) {len(cms)}')
            print('   EN 全部:', [m.group(1) for m in ems])
            print('   CN 全部:', [m.group(1) for m in cms])
            ok = len(ems) == len(cms) and all(
                norm(a.group(1)) == norm(b.group(1)) for a, b in zip(ems, cms))
            print('   一一对齐且语义相同:', ok)
            if not ok:
                out.setdefault('BAD', []).append((pack, path))
                continue
            pairs = [(a, b) for a, b in zip(ems, cms)
                     if a.group(1).upper() in ABBR and a.group(1).upper() == a.group(1)]
            print('   需改回字母的:', [(a.group(1), b.group(1)) for a, b in pairs])
            new = c
            for a, b in reversed(pairs):
                new = new[:b.start(1)] + a.group(1) + new[b.end(1):]
            out.setdefault(pack, {})[path] = new
            print('   本叶改动数:', len(pairs), ' 长度差:', len(new)-len(c))
    json.dump(out, open(os.path.join('4-临时脚本', '2026-08-14-fix', 'probes', 'g1_align.out.json'),
                        'w', encoding='utf-8'), ensure_ascii=False, indent=1)


main()
