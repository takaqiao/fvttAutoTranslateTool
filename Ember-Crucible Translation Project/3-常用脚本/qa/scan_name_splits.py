#!/usr/bin/env python3
"""同一个英文 `name` 在库里有**两套以上中文名**。

判据：把所有 `*.name` 叶子按**英文值**聚合，比较各自的中文头（去掉双语并列的英文尾巴）。
同一个英文名出现两种中文头 —— 玩家会在两个地方看到同一样东西叫两个名字。

为什么单独做一个闸
------------------
* `scan_label_vs_name` 比的是「`@UUID{标签}` ↔ 目标的 name」，两个 name 之间不比；
* `scan_token_name` 只比同一个 actor 的 name ↔ tokenName；
* `unify_terms` 要先有人给出规则表才能跑，而这类分裂正是**规则表本身没有的那些**。

⚠ **多数派常常是错的那一边**（2026-08-13e 实测）
------------------------------------------------
`Signborn Lineage` 星兆血统 3 : 印记裔血统 1 —— 但 §8 已裁 `Signborn`＝印记裔；
`Kivahr Lineage` 基瓦赫 4 : 基瓦尔 1 —— 而祖裔页的 name 就是「基瓦尔 Kivahr」。
所以本闸**只报分裂、不给建议**，方向必须逐条按依据阶梯判：
同名条目的 name > 同卷已译页 > 全库多数 > `glossary_ec.json`。

⚠ **有些分裂是合法的**：`Shield` 在 crucible 里既是法术（护盾术）又是装备（盾牌）；
`Swarm` 作 archetype 是「群集」、作具体生物（Insect Swarm）是「虫群」（§8 已裁）。
判据看不出词义，这类要人排除。

**已留证的合法分裂写进 `KNOWN_SPLITS`**（下方），本闸把它们从「同名不同译」里摘出来
单列一节，不再混在待裁清单里 —— 但**仍然打印**（数量 + 当前各变体计数），
因为「摘掉了就看不见」正是本项目反复踩的坑：一旦上游改了名，
被摘掉的那条会连同它的证据一起悄悄失效。
`--no-known` 可以关掉摘除，回到旧行为。
⚠ 加条目前必须先在 `5-其他内容/EXCLUSIONS.same_en_split.json` 里留证，
（2026-08-15 第十七轮从 `4-临时脚本/2026-08-13-round12/findings/` 挪来 —— 那里被
 `.gitignore` 的 `4-临时脚本/**/*.json` 挡着，换台机器 clone 后判据输入会直接消失）
这里只放**指针**，不放论证。

用法：
  python scan_name_splits.py --repo <repo> [--repo <另一个>] [--out <json>] [--show 40]
                             [--no-known]
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CJK = re.compile(r'[一-鿿]')
BILINGUAL_TAIL = re.compile(r'\s+[^一-鿿　-〿＀-￯]+$')

# 英文名 -> {允许的中文头: 该译名对应的义项}。证据在 EXCLUSIONS.json，这里只留指针。
# 一个组只有在**中文头集合完全落在表内**时才算「已留证」；冒出新的第三种译法照样报。
KNOWN_SPLITS = {
    # EXCLUSIONS.json G1 / G3 / G5 三条都记着它，2026-08-15 第十六轮又被推翻过一次：
    # 把 actors.Arcturian 的 name/tokenName 改成「阿克图里安」会让 scan_label_vs_name
    # 从 2 处涨到 20 处（18 处 `{Arcturians}` 标签），实测过两次，不要再试。
    # 英文侧本身就是这个二分：定语／文化标签 472 次、指人的名词 223 次（单 86 + 复 137）。
    'Arcturian': {'阿克图里安': '定语／文化标签（Arcturian dwellings / Arcturian Wirrun）',
                  '阿克图里安人': '指人的名词，单复数（actors.Arcturian 本体与 tokenName）'},
    # EXCLUSIONS.json G1：带 Imperceptible Barrier 效果的是法术，裸的是装备，23:11 无一例外。
    'Shield': {'护盾术': 'crucible 法术', '盾牌': 'crucible 装备'},
    # EXCLUSIONS.json：ember.character.json 是 dnd5e 侧包，与 crucible 永不同载。
    'Luminous': {},
    'Spirited': {},
    # EXCLUSIONS.json P-name-split-color-commentary：英文一语双关，两处所指无关，
    # 互相取代都会丢义。2026-08-16d 主控终段补裁 T-3 正式登记为合法分裂。
    'Color Commentary': {
        '彩色解说': 'journals.Local Color.pages.Color Commentary（颜料／地方色彩双关）',
        '精彩解说': 'tables.Helkas Drake Moments 6-6（体育解说本义）'},
}


def walk(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f'{path}.{k}' if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f'{path}.{i}')
    elif isinstance(obj, str) and obj:
        yield path, obj


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def head(s):
    return BILINGUAL_TAIL.sub('', s).strip() or s.strip()


def known(en, variants):
    """这一组是不是 `KNOWN_SPLITS` 里已留证的合法分裂。

    空 mapping = 整条英文名整组豁免（Luminous / Spirited 那种「两个包永不同载」）；
    非空 mapping = **中文头必须全部落在表内**，冒出第三种译法照样报出来 ——
    否则一条豁免会连带把它旁边的新缺陷一起盖掉。
    """
    allow = KNOWN_SPLITS.get(en)
    if allow is None:
        return False
    return not allow or set(variants) <= set(allow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--out')
    ap.add_argument('--show', type=int, default=40)
    ap.add_argument('--no-known', action='store_true',
                    help='不摘除 KNOWN_SPLITS，回到旧行为')
    a = ap.parse_args()

    by_en = collections.defaultdict(lambda: collections.defaultdict(list))
    for repo in a.repo:
        en_dir = os.path.join(repo, 'compendium', 'en')
        for pack in sorted(os.listdir(en_dir)):
            if not pack.endswith('.json') or pack == '_source.json':
                continue
            cn_p = os.path.join(repo, 'compendium', 'cn', pack)
            if not os.path.exists(cn_p):
                continue
            en = dict(walk(load(os.path.join(en_dir, pack)).get('entries', {})))
            cn = dict(walk(load(cn_p).get('entries', {})))
            for path, v in en.items():
                if not path.endswith('.name'):
                    continue
                c = cn.get(path)
                if c and CJK.search(c):
                    by_en[v][head(c)].append({'repo': repo, 'pack': pack, 'path': path,
                                              'full_cn': c})

    all_splits = {k: v for k, v in by_en.items() if len(v) > 1}
    archived = {} if a.no_known else {
        k: v for k, v in all_splits.items() if known(k, v)}
    splits = {k: v for k, v in all_splits.items() if k not in archived}
    findings = []
    for en, variants in sorted(splits.items(),
                               key=lambda kv: -sum(len(x) for x in kv[1].values())):
        findings.append({
            'english': en,
            'total': sum(len(x) for x in variants.values()),
            'variants': {z: {'count': len(ps), 'paths': [
                f"{p['repo'][:1]}/{p['pack']}::{p['path']}" for p in ps[:6]]}
                for z, ps in sorted(variants.items(), key=lambda kv: -len(kv[1]))},
        })

    archived_rows = [{
        'english': en,
        'total': sum(len(x) for x in variants.values()),
        'variants': {z: len(ps) for z, ps in sorted(
            variants.items(), key=lambda kv: -len(kv[1]))},
        'why': 'EXCLUSIONS.json 已留证（见 KNOWN_SPLITS 注释）',
    } for en, variants in sorted(archived.items())]

    print(f'有中文 name 的唯一英文名 {len(by_en)} 个')
    print(f'同名不同译 {len(all_splits)} 个 —— 已留证合法分裂 {len(archived)} 个，'
          f'**待裁 {len(splits)} 个**\n')
    for f in findings[:a.show]:
        v = ' | '.join(f'{z}×{d["count"]}' for z, d in f['variants'].items())
        print(f'  {f["english"][:34]:36s} {f["total"]:3d} 处  {v}')
    if archived_rows:
        print('\n已留证的合法分裂（不是缺陷，但仍打印 —— 上游改名时这里会先变）：')
        for r in archived_rows:
            v = ' | '.join(f'{z}×{n}' for z, n in r['variants'].items())
            print(f'  {r["english"][:34]:36s} {r["total"]:3d} 处  {v}')
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump({'unique_en_names': len(by_en), 'splits': len(splits),
                       'archived_known': archived_rows,
                       'findings': findings}, fh, ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
