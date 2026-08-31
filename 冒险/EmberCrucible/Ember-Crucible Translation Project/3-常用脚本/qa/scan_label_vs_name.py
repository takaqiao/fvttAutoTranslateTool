#!/usr/bin/env python3
"""`@UUID[目标]{标签}` 的**中文标签**与目标文档的**中文 name** 不一致。

判据（严格版，只报英文侧本来就同名的）：

    英文标签 == 目标文档的英文 name      ← 作者在英文里就是拿文档名当标签用
    中文标签 != 目标文档的中文 name      ← 那中文侧就不该不同名

英文侧本来就不同名的场合非常多（作者有意换称呼、用代词式简称、`#锚点` 指向文档的一小节），
那些**不是缺陷**，所以第一条是硬条件。这也是本项目「先查英文再判中文」的机械化。

为什么单独做一个闸
------------------
* `scan_markup_targets.py` 只看方括号**内部**有没有被译坏，不看标签。
* `scan_uuid_swap.py` 的判据是「该目标的中文标签 != 全库多数标签」——
  **多数派本身可能就是错的那一边**（2026-08-13 实测有 6 组如此），而且它的 `en_label`
  在「同一目标在同一叶出现多次」时取的是第一个，不可直接采信（见 PROJECT.md 第 8 节）。
  本闸拿 `name` 字段仲裁，是依据阶梯里最强的一层，不投票。
* 2026-08-13 的 35 本 journal 逐句复核里，对抗式复核 agent 补出的 188 条漏项中，
  这一类占了最大头 —— 而它**完全可以机械化**，不必靠人读。

目标 id 只存在于 LevelDB packs 里（译文文件是按 name 建键的），所以要先拿
`4-临时脚本/2026-08-12-fix/dump_ids.mjs` 导出 id -> 英文 name 的表。

用法：
  node dump_ids.mjs --package <foundry包目录> --out ids.json
  python scan_label_vs_name.py --repo <repo> [--repo <另一个>] --ids ids.json [--out x.json]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# @UUID[...]{标签} —— 只认带标签的；不带标签的渲染出来是文档名，本来就不会不一致
UUID_RX = re.compile(r'@UUID\[([^\]]+)\]\{([^}]*)\}')
TAG_RX = re.compile(r'<[^>]+>')
CJK_RX = re.compile(r'[一-鿿]')
# 双语并列的英文尾巴：「秘藏书架 The Secret Shelf」-> 取中文头。
# 字符类必须收进重音字母（`瓦伦 Varún`）与标点（`软泥怪爆炸！ Ooze Go Boom!`），
# 否则那些条目会被当成「中文名不同」报出来 —— 实测这是本闸最大的一类假阳性。
BILINGUAL_TAIL = re.compile(
    r'\s+[^一-鿿　-〿＀-￯]+$')


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


def head(name):
    """中文 name 去掉双语并列的英文尾巴。「科拉克 Cor'ak」-> 科拉克"""
    s = BILINGUAL_TAIL.sub('', name).strip()
    return s or name.strip()


def target_id(target):
    """`JournalEntry.a.JournalEntryPage.b#锚点` -> ('b', 有无锚点)。取**最后**一个 id。"""
    anchor = '#' in target
    t = target.split('#', 1)[0]
    parts = [p for p in t.split('.') if p]
    return (parts[-1] if parts else None), anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--ids', required=True, help='dump_ids.mjs 的输出：id -> {name}')
    ap.add_argument('--out')
    ap.add_argument('--show', type=int, default=30)
    ap.add_argument('--include-anchors', action='store_true',
                    help='连 `#锚点` 形式一起报（默认跳过：锚点指的是文档的一小节，'
                         '标签用小节名是对的，不是缺陷）')
    a = ap.parse_args()

    ids = load(a.ids)
    en_name_of = {k: (v.get('name') if isinstance(v, dict) else v) for k, v in ids.items()}

    # 中文 name：从各仓库译文里按「路径以 .name 结尾」收集，并按英文 name 建索引
    cn_name_by_en = {}
    for repo in a.repo:
        cn_dir = os.path.join(repo, 'compendium', 'cn')
        en_dir = os.path.join(repo, 'compendium', 'en')
        for pack in sorted(os.listdir(cn_dir)):
            if not pack.endswith('.json'):
                continue
            ep = os.path.join(en_dir, pack)
            if not os.path.exists(ep):
                continue
            cn = dict(walk(load(os.path.join(cn_dir, pack)).get('entries', {})))
            en = dict(walk(load(ep).get('entries', {})))
            for path, cv in cn.items():
                if not path.endswith('.name'):
                    continue
                ev = en.get(path)
                if ev and CJK_RX.search(cv):
                    # 同一个英文 name 可能对应多个条目；取多数写法
                    cn_name_by_en.setdefault(ev, Counter())[head(cv)] += 1

    findings, stats = [], Counter()
    for repo in a.repo:
        cn_dir = os.path.join(repo, 'compendium', 'cn')
        for pack in sorted(os.listdir(cn_dir)):
            if not pack.endswith('.json'):
                continue
            en_p = os.path.join(repo, 'compendium', 'en', pack)
            if not os.path.exists(en_p):
                continue
            cn = dict(walk(load(os.path.join(cn_dir, pack)).get('entries', {})))
            en = dict(walk(load(en_p).get('entries', {})))
            for path, cv in cn.items():
                ev = en.get(path)
                if not ev:
                    continue
                # 对齐方式：**按同一 target 的出现序号**配对，而不是叶内全局序号。
                #
                # 同一叶对同一目标用两个不同英文标签是常态，所以不能按目标聚合取「第一个」
                # （PROJECT.md 第 8 节 2026-08-13 记的坑）；但按全局序号对齐又要求
                # 两侧链接总数相等 —— 中文侧多一个或少一个链接（`[[/item …]]` 这类
                # 由 by-design 差异造成）就整叶跳过，实测因此漏掉 120 叶。
                # 按 target 分组后再逐位配对，两种失败模式都避开了。
                cn_by_t, en_by_t = {}, {}
                for t, l in UUID_RX.findall(cv):
                    cn_by_t.setdefault(t, []).append(l)
                for t, l in UUID_RX.findall(ev):
                    en_by_t.setdefault(t, []).append(l)
                pairs = []
                for t, els in en_by_t.items():
                    cls = cn_by_t.get(t)
                    if cls is None:
                        stats['目标不一致（另有闸负责）'] += 1
                        continue
                    if len(cls) != len(els):
                        stats['同目标出现次数不等（跳过）'] += 1
                        continue
                    pairs.extend((t, cl, t, el) for cl, el in zip(cls, els))
                for (ct, cl, et, el) in pairs:
                    tid, anchor = target_id(et)
                    if anchor and not a.include_anchors:
                        stats['锚点形式（跳过）'] += 1
                        continue
                    en_name = en_name_of.get(tid)
                    if not en_name:
                        # 绝大多数是 **dnd5e 系统自己的合集**（`Compendium.dnd5e.spells24.Item.…`）——
                        # 那些条目的译名来自 dnd5e 的汉化模块，**本项目不负责**，
                        # 不是覆盖洞。分开计数，否则这一档看着像 1900 多条漏检。
                        if re.search(r'(?:^|\.)dnd5e\.', et):
                            stats['外部合集（dnd5e 系统），本项目不负责'] += 1
                        elif et.startswith('.'):
                            # `@UUID[.xxxx]` 是相对引用（相对当前文档），id 表里没有
                            stats['相对引用（同文档内），id 表不覆盖'] += 1
                        else:
                            stats['目标 id 不在 id 表里（真覆盖洞）'] += 1
                        continue
                    # 硬条件一：英文标签本来就等于目标的英文 name
                    if TAG_RX.sub('', el).strip() != en_name.strip():
                        stats['英文标签本来就与文档名不同（by design）'] += 1
                        continue
                    cands = cn_name_by_en.get(en_name)
                    if not cands:
                        stats['目标没有中文 name'] += 1
                        continue
                    want = cands.most_common(1)[0][0]
                    got = head(TAG_RX.sub('', cl).strip())
                    if got == want:
                        stats['一致'] += 1
                        continue
                    # 同一个**英文** name 可能属于两个不同实体（实测 `Arcturian`：
                    # 文化页 name＝阿克图里安、泛用 NPC actor name＝阿克图里安人）。
                    # 中文 name 是按英文名聚合取多数的，于是少数派实体的正确标签会被
                    # 多数派污染成「不一致」—— 2026-08-13b 实测一组 40 处里有 16 处是这么来的。
                    # 只要标签命中该英文名下的**任一**中文 name，就不算缺陷。
                    if got in cands:
                        stats['多义英文名下命中其它候选（不算缺陷）'] += 1
                        continue
                    stats['**不一致**'] += 1
                    findings.append({
                        'repo': repo, 'pack': pack, 'path': path,
                        'batch_path': path,
                        'target': et, 'en_label': el,
                        'cn_label': cl, 'target_cn_name': want,
                        'target_cn_variants': dict(cands),
                    })

    print('统计：')
    for k, v in stats.most_common():
        print(f'  {k:34s} {v}')
    print(f'\n**中文标签 != 目标文档中文 name** 共 {len(findings)} 处')
    for f in findings[:a.show]:
        print(f'  {f["pack"][:26]:28s} {f["path"].split(".pages.")[-1][:34]:36s} '
              f'{f["cn_label"][:16]:18s} -> {f["target_cn_name"][:16]}')
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump({'stats': dict(stats), 'findings': findings}, fh,
                      ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
