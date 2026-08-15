# -*- coding: utf-8 -*-
"""G1 · 生成两个孪生冒险包的回写批次（Thayloc Courser 统一 / tokenName 音译分裂 / 阵营缩写写法）。"""
import json, os, re, sys
sys.stdout.reconfigure(encoding='utf-8')

REPO = '1-Ember汉化插件'
OUT = os.path.join('4-临时脚本', '2026-08-14-fix', 'batches')
ALIGN = json.load(open(os.path.join('4-临时脚本', '2026-08-14-fix', 'probes', 'g1_align.out.json'),
                       encoding='utf-8'))
os.makedirs(OUT, exist_ok=True)

A = 'Ember Early Access'
PACKS = ['ember.adventure.json', 'ember.crucible-adventure.json']


def get(root, path):
    node = root
    for p in path.split('|'):
        node = node[p]
    return node


def sub1(s, old, new, where):
    """必须恰好命中一次，否则报错停下。"""
    n = s.count(old)
    assert n == 1, f'{where}: {old!r} 命中 {n} 次'
    return s.replace(old, new)


for pack in PACKS:
    cn = json.load(open(os.path.join(REPO, 'compendium', 'cn', pack), encoding='utf-8'))['entries']
    batch = {}

    # ---- 1. Thayloc Courser：疾奔兽（人型战斗牧师被译成野兽）统一到全书 42 处已用的「赛洛克弓手」
    a = cn[A]['actors']['Thayloc Courser']
    batch[f'{A}.actors.Thayloc Courser.name'] = sub1(
        a['name'], '赛洛克疾奔兽', '赛洛克弓手', 'Thayloc.name')
    batch[f'{A}.actors.Thayloc Courser.tokenName'] = sub1(
        a['tokenName'], '赛洛克疾奔兽', '赛洛克弓手', 'Thayloc.tokenName')
    p = cn[A]['journals']['Unfinished Business']['pages']['Overview']['text']
    batch[f'{A}.journals.Unfinished Business.pages.Overview.text'] = sub1(
        p, '{赛洛克疾奔兽}', '{赛洛克弓手}', 'Overview.text')
    p = cn[A]['journals']["Gamemaster's Guide"]['pages']['Patch 0.4.5']['text']
    batch[f"{A}.journals.Gamemaster's Guide.pages.Patch 0.4.5.text"] = sub1(
        p, '以及赛洛克疾奔兽。', '以及赛洛克弓手。', 'Patch045.text')
    # 裸英文 Courser(s) 残留（只有 crucible-adventure 有 biography.private）
    priv = a.get('biography', {}).get('private')
    if priv:
        new = sub1(priv, '的仆从与牧师，Coursers 与整个信仰体系',
                   '的仆从与牧师，赛洛克弓手与整个信仰体系', 'bio.private#1')
        new = sub1(new, '个别 Courser 拥有自己的信众群体',
                   '个别赛洛克弓手拥有自己的信众群体', 'bio.private#2')
        new = sub1(new, '</p><p>Coursers 与 @UUID',
                   '</p><p>赛洛克弓手与 @UUID', 'bio.private#3')
        assert 'Courser' not in new
        batch[f'{A}.actors.Thayloc Courser.biography.private'] = new

    # ---- 2. tokenName 音译/术语分裂（复核裁定的 5 个 actor 中，Amelia Naxan 归 G2 处理）
    batch[f'{A}.actors.Funar Cevher.tokenName'] = '富纳尔·杰夫赫尔'
    batch[f'{A}.actors.Brynna Verocorrt.tokenName'] = '布琳娜·维罗科尔特'
    batch[f'{A}.actors.Wind Raider Boarder.tokenName'] = '风袭劫掠者登舰兵'
    batch[f'{A}.actors.Pallid Ultra Drake.name'] = sub1(
        cn[A]['actors']['Pallid Ultra Drake']['name'], '苍白极巨龙', '苍白究极龙兽', 'PallidUltraDrake.name')

    # ---- 3. 阵营缩写：EN 缩写一侧统一保留字母（全库 84 处已如此）
    for path, new in ALIGN[pack].items():
        batch[path] = new

    out = os.path.join(OUT, f'G1.1.{pack}')
    json.dump(batch, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{out}: {len(batch)} 条')
