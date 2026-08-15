#!/usr/bin/env python3
"""从**当前库真实状态**（Y1-B 批次已落之后）生成 Y2 升报 4 条真缺陷的整叶批次。

每条替换都写死 expect 次数；数不对就抛错不出批次（反空转 + 防误伤同叶其它术语）。
"""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPO = os.path.join(ROOT, '1-Ember汉化插件')
OUT = os.path.join(ROOT, '4-临时脚本', '2026-08-15-round18', 'batches')

E1 = ('E1', 'Ember Early Access.actors.Sadri Zhalimorne.biography.private',
      [('阿克图里安高原西部地区', '阿克图斯高原西部地区', 1)])
E2 = ('E2', 'Ember Early Access.actors.Constructed Companion.biography.private',
      [('被视为名匠瓦索洛缪·切斯', '被视为著名阿克图里安工匠瓦索洛缪·切斯', 1)])
# ⚠ 分两条替换：带前导空格的 3 处把空格一起吃掉（否则出现「介绍 阶位 1 魂印」的双词间空格），
#   @UUID 标签内的 1 处没有前导空格，单独一条。两条相加必须正好等于全叶的 4 处。
E3 = ('E3', 'Ember Early Access.journals.Unfinished Business.pages.Shine On.text',
      [('{1 级魂印}', '{阶位 1 魂印}', 1),
       (' 1 级魂印', '阶位 1 魂印', 3)])
E4 = ('E4', 'Ember Early Access.journals.Unfinished Business.pages.The Old Flame.text',
      [('Rank 1的角色', '阶位 1 的角色', 1),
       ('Rank 2或更高的角色', '阶位 2 或更高的角色', 1)])

PLAN = {
    'ember.adventure.json': [E3, E4],
    'ember.crucible-adventure.json': [E1, E2, E3, E4],
}


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def get_at(node, parts):
    for p in parts:
        if isinstance(node, dict): node = node.get(p)
        elif isinstance(node, list):
            try: node = node[int(p)]
            except (ValueError, IndexError): return None
        else: return None
    return node


def split_path(root, path):
    naive = path.split('.')
    if get_at(root, naive) is not None: return naive
    parts, node, rest = [], root, path
    while rest:
        if isinstance(node, dict):
            cands = [k for k in node if rest == k or rest.startswith(k + '.')]
            if cands:
                k = max(cands, key=len); parts.append(k); node = node[k]; rest = rest[len(k)+1:]; continue
        head, _, rest = rest.partition('.')
        parts.append(head); node = get_at(node, [head])
    return parts


total_repl = 0
for pack, cases in PLAN.items():
    en = load(os.path.join(REPO, 'compendium', 'en', pack))
    cn = load(os.path.join(REPO, 'compendium', 'cn', pack))
    batch = {}
    for tag, path, repls in cases:
        parts = split_path(en['entries'], path)
        cur = get_at(cn['entries'], parts)
        if not isinstance(cur, str):
            raise SystemExit(f'{pack} {path}: 库里没有中文叶')
        new = cur
        for old, rep, expect in repls:
            got = new.count(old)
            if got != expect:
                raise SystemExit(f'{pack} {tag} {path}: {old!r} 实测 {got} 次，期望 {expect} 次 —— 不出批次')
            new = new.replace(old, rep)
            total_repl += expect
            print(f'  {pack} {tag}: {old!r} -> {rep!r} ×{expect}')
        if new == cur:
            raise SystemExit(f'{pack} {tag}: 无改动')
        batch[path] = new
    p = os.path.join(OUT, f'r18-escalations.{pack}')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f'写出 {p}  keys={len(batch)}')
print(f'总替换处数={total_repl}')
