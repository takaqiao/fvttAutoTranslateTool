# -*- coding: utf-8 -*-
r"""
scan_roleblind_tm_fallback.py
—— 「对整个异质集合施加同一处变换，没有按成员做类型/角色判据」在**回填工具**上的实例。

判据抽象
--------
已确认实例：register.js 面对「个别 actor 数据可能畸形」，对**整批 actor**做同一处
破坏性降级，没有按成员判类型。
本探针查的是同一形状的另一处：回填工具面对「个别叶子在按角色建的 TM 里查不到」，
**退化成按英文原文查全库多数派**，于是把一个异质集合（items.name / actions.name /
effects.name / tokenName / adjective / label …各有各的书写约定）当成同质集合处理。

  3-常用脚本/tm/fill_missing.py:133   cands = tm_shape.get((shape_of(path), src)) or tm_plain.get(src)
  3-常用脚本/tm/fill_twin.py:140      hit   = shaped.get((shape, key))          or plain.get(key)

同库的 qa/propagate_fix.py:39-48 与 :86-93 明确写了这条闸门为什么必须存在
（`name` 是双语并列「辉耀 Luminary」，`adjective` 是裸中文「辉耀」，跨角色传播
会拼出「辉耀 Luminary长剑」），并实测「crucible 侧 10 条候选里 7 条正是跨角色配对」。
而 qa/apply_translations.py 的三道闸（英文在位 / 含中文 / 标记签名）**没有角色判据**，
所以退化路径产出的跨角色译文能一路落库。

本探针做的事（留一法，read-only）
--------------------------------
对两个仓库里**已经有人翻过**的每一条叶子 L=(路径, 英文, 中文)：
  1. 按 fill_missing 的 shape_of() 建 tm_shape、按英文建 tm_plain（把 L 自己的票扣掉）；
  2. 若 tm_shape[(shape,en)] 扣掉自己后为空 → 该叶子在真实缺口场景下会走**退化路径**；
  3. 退化路径取 tm_plain[en] 的多数派，与人写的 L.cn 比对；
  4. 通过 apply_translations.markup_signature 闸后仍不同的，即为「会被写错的叶子」。

假阳性模式
----------
* 同一英文的两种中文可能只是**同角色内的正常异写**（例如同一物品名两个包各写一次），
  这类不算跨角色缺陷 —— 脚本因此单独统计「多数派来自**不同 shape**」的子集，
  并单独统计「双语并列 vs 裸中文」这一已被本项目实证过的格式冲突。
* 留一法高估的一面：真实缺口叶子的 shape 不一定在 tm_shape 里出现过（更容易退化）；
  低估的一面：真实缺口是**上游新内容**，其英文更可能在 TM 里只有一种译法。
  两个方向都记在下面的输出里，不做单点断言。

只读。不写任何仓库文件。
"""
from __future__ import annotations
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]
CJK = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")

_spec = importlib.util.spec_from_file_location(
    "_apply", os.path.join(P, "3-常用脚本", "qa", "apply_translations.py"))
_apply = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_apply)
markup_signature = _apply.markup_signature

# fill_missing.py:40-42 原样
STRUCT = {'entries', 'actors', 'items', 'actions', 'effects', 'pages', 'journals',
          'results', 'folders', 'biography', 'description', 'name', 'text',
          'label', 'public', 'private', 'appearance', 'condition', 'tokenName'}


def shape_of(path):
    return '.'.join(p for p in path.split('.') if p in STRUCT or p.isdigit())


def walk(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f'{p}.{k}' if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f'{p}.{i}' if p else str(i))
    elif isinstance(o, str) and o.strip():
        yield p, o


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def bilingual(s):
    """本项目 name 的双语并列约定：中文后跟拉丁字母。"""
    return bool(CJK.search(s) and LATIN.search(s))


def main():
    rows = []          # (repo, pack, path, en, cn)
    for repo in REPOS:
        en_dir = os.path.join(P, repo, 'compendium', 'en')
        cn_dir = os.path.join(P, repo, 'compendium', 'cn')
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith('.json'):
                continue
            cnp = os.path.join(cn_dir, fn)
            if not os.path.exists(cnp):
                continue
            en = dict(walk(load(os.path.join(en_dir, fn))))
            cn = dict(walk(load(cnp)))
            for path, src in en.items():
                tgt = cn.get(path)
                if tgt and CJK.search(tgt):
                    rows.append((repo, fn, path, src, tgt))

    print(f"扫描 {len(REPOS)} 个仓库，已译叶子 {len(rows)} 条")

    tm_shape = defaultdict(Counter)
    tm_plain = defaultdict(Counter)
    shapes_of_en = defaultdict(set)
    for _repo, _pack, path, src, tgt in rows:
        s = shape_of(path)
        tm_shape[(s, src)][tgt] += 1
        tm_plain[src][tgt] += 1
        shapes_of_en[src].add(s)

    print(f"  按 (shape, 英文) 建的 TM 键 {len(tm_shape)}")
    print(f"  仅按英文建的 TM 键        {len(tm_plain)}")

    # --- A. 静态口径：同一英文有多种中文，且分布在不同 shape 上 -------------
    cross = []
    for src, cnt in tm_plain.items():
        if len(cnt) < 2:
            continue
        shapes = defaultdict(set)
        for _repo, _pack, path, s2, tgt in rows:
            if s2 == src:
                shapes[tgt].add(shape_of(path))
        allsh = set()
        for v in shapes.values():
            allsh |= v
        if len(allsh) < 2:
            continue
        # 只有当不同译文确实落在不同 shape 上，才算跨角色分歧
        pairs = list(shapes.items())
        crossed = any(shapes[a] != shapes[b] for a, _x in pairs for b, _y in pairs if a != b)
        if crossed:
            cross.append((src, dict((k, sorted(v)) for k, v in shapes.items())))
    print(f"\nA. 同一英文有≥2 种中文且分布在不同 shape 上的英文条目：{len(cross)}")
    fmt = [c for c in cross
           if any(bilingual(k) for k in c[1]) and any(not bilingual(k) for k in c[1])]
    print(f"   其中「双语并列 vs 裸中文」格式冲突：{len(fmt)}")
    for src, m in fmt[:8]:
        print(f"     EN {src[:60]!r}")
        for cn, sh in m.items():
            print(f"        {cn[:48]!r:<52} @ {sh}")

    # --- B. 留一法：这条叶子若缺失，退化路径会写成什么 ----------------------
    degraded = wrong = wrong_cross = wrong_fmt = markup_dropped = 0
    samples = []
    for repo, pack, path, src, tgt in rows:
        s = shape_of(path)
        c_shape = Counter(tm_shape[(s, src)])
        c_shape[tgt] -= 1
        if c_shape[tgt] <= 0:
            del c_shape[tgt]
        if c_shape:
            continue                      # 形状键还有别人，不会退化
        c_plain = Counter(tm_plain[src])
        c_plain[tgt] -= 1
        if c_plain[tgt] <= 0:
            del c_plain[tgt]
        if not c_plain:
            continue                      # TM 里没有别的候选，工具会留给人翻
        degraded += 1
        best = c_plain.most_common(1)[0][0]
        if best == tgt:
            continue
        if markup_signature(best) != markup_signature(src):
            markup_dropped += 1           # 标记闸会拦下，不会落库
            continue
        wrong += 1
        other_shapes = {shape_of(p2) for _r, _pk, p2, s2, t2 in rows
                        if s2 == src and t2 == best}
        if s not in other_shapes:
            wrong_cross += 1
        if bilingual(best) != bilingual(tgt):
            wrong_fmt += 1
            if len(samples) < 12:
                samples.append((repo, pack, path, s, src, tgt, best, sorted(other_shapes)))

    print(f"\nB. 留一法（{len(rows)} 条已译叶子）")
    print(f"   会走「仅按英文」退化路径的            {degraded}")
    print(f"   退化后写出的中文 ≠ 人写的中文         {wrong + markup_dropped}"
          f"（其中 {markup_dropped} 条被标记闸拦下，实际落库 {wrong}）")
    print(f"   落库且**多数派来自别的 shape**        {wrong_cross}")
    print(f"   落库且**双语并列/裸中文格式互换**     {wrong_fmt}")
    print("\n   样本（格式互换，即 propagate_fix.py:86-93 记载的那种错）：")
    for repo, pack, path, s, src, tgt, best, osh in samples:
        print(f"\n     {repo}/{pack}")
        print(f"       path   {path[:110]}")
        print(f"       shape  {s}")
        print(f"       EN     {src[:70]!r}")
        print(f"       人写    {tgt[:60]!r}")
        print(f"       退化写  {best[:60]!r}   ← 来自 shape {osh}")


if __name__ == "__main__":
    main()
