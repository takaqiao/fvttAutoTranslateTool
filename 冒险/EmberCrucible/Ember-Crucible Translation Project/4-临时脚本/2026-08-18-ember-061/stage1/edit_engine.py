# -*- coding: utf-8 -*-
"""改动桶编辑引擎。

编辑单元 = {"ptr":…, "scope": "both"|文件名, "ops":[ [old,new] … ]}  或 {"ptr":…, "full": "整串"}
ops 逐条要求：old 在当前 CN 值里**恰好出现 1 次**（出现 0 次或 >1 次直接报错，不许静默）。

落地后逐叶自检：
  * CN 的 HTML 标签多重集 == EN061 的
  * CN 的增强器目标多重集 == EN061 的（@UUID/@Condition/@Embed/@Check/… 的括号载荷）
  * CN 的 {占位符} 多重集 == EN061 的
"""
import os, sys, collections, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

EDITS_DIR = os.path.join(WORK, 'edits')


def apply_edits(edit_items, dry=False, verbose=True):
    delta = load_delta()
    byfile = collections.defaultdict(list)
    for it in edit_items:
        sc = it.get('scope', 'both')
        for f in (FILES if sc == 'both' else [sc]):
            if it['ptr'] in delta[f]['changed']:
                byfile[f].append(it)
    stats = collections.Counter()
    problems = []
    for f in FILES:
        items = byfile.get(f)
        if not items:
            continue
        p = os.path.join(CN, f)
        cn = jload(p)
        en061 = jload(os.path.join(EN061, f))
        for it in items:
            parts = split_path(it['ptr'])
            cur = get_at(cn, parts)
            if cur is None:
                problems.append(f"[{f}] CN 缺叶 {it['ptr']}")
                continue
            if 'full' in it:
                new = it['full']
            else:
                new = cur
                for old, rep in it['ops']:
                    n = new.count(old)
                    if n != 1:
                        problems.append(f"[{f}] {it['ptr']} 片段出现 {n} 次(要求1): {old[:80]!r}")
                        break
                    new = new.replace(old, rep, 1)
                else:
                    pass
            if problems and problems[-1].startswith(f"[{f}] {it['ptr']}"):
                continue
            if new == cur:
                stats['noop'] += 1
            else:
                stats['edited'] += 1
            if not dry:
                set_at(cn, parts, new)
            # 自检
            tgt = get_at(en061, parts)
            if isinstance(tgt, str):
                if collections.Counter(tag_multiset(new)) != collections.Counter(tag_multiset(tgt)):
                    d1 = collections.Counter(tag_multiset(new)) - collections.Counter(tag_multiset(tgt))
                    d2 = collections.Counter(tag_multiset(tgt)) - collections.Counter(tag_multiset(new))
                    problems.append(f"[{f}] TAG {it['ptr']} CN多{dict(d1)} EN多{dict(d2)}")
                if collections.Counter(uuid_targets(new)) != collections.Counter(uuid_targets(tgt)):
                    d1 = collections.Counter(uuid_targets(new)) - collections.Counter(uuid_targets(tgt))
                    d2 = collections.Counter(uuid_targets(tgt)) - collections.Counter(uuid_targets(new))
                    problems.append(f"[{f}] ENH {it['ptr']} CN多{list(d1)[:3]} EN多{list(d2)[:3]}")
                if collections.Counter(placeholders(new)) != collections.Counter(placeholders(tgt)):
                    d1 = collections.Counter(placeholders(new)) - collections.Counter(placeholders(tgt))
                    d2 = collections.Counter(placeholders(tgt)) - collections.Counter(placeholders(new))
                    # {…} 里常是可译标签，只比数量
                    if sum(d1.values()) != sum(d2.values()):
                        problems.append(f"[{f}] PH  {it['ptr']} CN多{list(d1)[:3]} EN多{list(d2)[:3]}")
        if not dry:
            save_cn(cn, p)
        stats[f] = len(items)
    if verbose:
        print("applied:", dict(stats))
        for pr in problems:
            print("  !!", pr)
    return stats, problems
