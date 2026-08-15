# -*- coding: utf-8 -*-
r"""
scan_blind_substring_write.py
—— 「对整个异质集合施加同一处变换，不按成员判类型」在**字符串内位置**上的实例。

判据抽象
--------
已确认实例：面对「个别 actor 可能畸形」，对整批 actor 施加同一处破坏性变换。
这里是同一形状：面对「这条译文的 <strong> 数量比英文少」，
`qa/fix_bold_drift.py` 第三步对**整条字符串**做 `str.replace(词, "<strong>词</strong>", 1)`，
把「这条字符串里的所有位置」当成同质集合 —— 没有区分
  (a) 正文文本      ← 唯一该动的
  (b) HTML 属性值   `data-tooltip="决心"` / `alt="…"`
  (c) Foundry 标记的可见 label   `@UUID[Item.x]{决心}` / `[[/check ...]]{决心}`
  (d) 标记的机械参数 `@Embed[Actor.x label=决心]`
唯一的护栏是 `new.count(cw) != 1`（全文恰好一次），它管的是**次数**不是**位置**。

而且 `fix_bold_drift.py --write` 直接 `cnp.write_text()` 落库，
**不过 apply_translations.py 的三道闸**；它自己的自检是 `count_strong(new) == ne`，
也就是说：注进属性/标记里的那个 `<strong>` 恰恰**满足**这条自检，
并且让 `scan_markup_drift` 的 `<strong>` 计数从「少一个」变成「刚好相等」——
损伤被判据判成了修复。

本探针做两件事
--------------
D1 现存损伤：全库 cn 叶子里，`<strong>`/`</strong>` 落在 HTML 属性值内、
   或落在 `@X[...]{...}` / `[[...]]{...}` 的 label 内的条数（英文侧同位置没有）。
D2 下次运行的命中面：完整复刻 fix_bold_drift 的第一步（学对应表）与第三步（反向补粗），
   只统计**注入点落在 (b)(c)(d) 区域**的条数，不写任何文件。

假阳性模式
----------
* D1：英文侧本来就在 label 里写了 <strong> 的，属于合法 —— 脚本逐条对照英文，
  只报英文侧同一位置没有的。
* D2：`--min-support` 默认 2，与工具默认一致；换参数会改变命中面，脚本把参数打出来。
* D2 统计的是「若这些条目今天被判定为少加粗」的注入点，不代表工具一定会重跑。

只读。不写任何仓库文件。
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]

STRONG = re.compile(r"<strong>(.*?)</strong>", re.S)
ATTR = re.compile(r'=\s*"([^"]*)"')
LABEL = re.compile(r"(?:@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\])\{([^}]*)\}")
BARE_PARAM = re.compile(r"\b(?:label|readaloud|caption)\s*=\s*(?!['\"])([^\s\]<>\"']+)")


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def protected_spans(s):
    """返回 (start, end) 区间列表：属性值 / 标记 label / 裸参数值。"""
    spans = []
    for m in ATTR.finditer(s):
        spans.append((m.start(1), m.end(1)))
    for m in LABEL.finditer(s):
        spans.append((m.start(1), m.end(1)))
    for m in BARE_PARAM.finditer(s):
        spans.append((m.start(1), m.end(1)))
    return spans


def in_protected(pos, spans):
    return any(a <= pos < b for a, b in spans)


def collect():
    packs = {}
    for repo in REPOS:
        en_dir = os.path.join(P, repo, "compendium", "en")
        cn_dir = os.path.join(P, repo, "compendium", "cn")
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith(".json"):
                continue
            cnp = os.path.join(cn_dir, fn)
            if not os.path.exists(cnp):
                continue
            en = dict(leaves(load(os.path.join(en_dir, fn))))
            cn = dict(leaves(load(cnp)))
            packs[f"{repo}/{fn}"] = (en, cn)
    return packs


def main():
    packs = collect()
    n_leaves = sum(len(cn) for _en, cn in packs.values())
    print(f"扫描 {len(packs)} 个包 / cn 叶子 {n_leaves}")

    # ---------------- D1 现存损伤 ----------------
    hits = []
    for pack, (en, cn) in packs.items():
        for p, t in cn.items():
            if "<strong" not in t:
                continue
            spans = protected_spans(t)
            for m in re.finditer(r"</?strong>", t):
                if in_protected(m.start(), spans):
                    s = en.get(p, "")
                    espans = protected_spans(s)
                    en_bad = any(in_protected(x.start(), espans)
                                 for x in re.finditer(r"</?strong>", s))
                    hits.append((pack, p, t[max(0, m.start() - 60):m.start() + 60], en_bad))
                    break
    print(f"\nD1 现存：<strong> 落在属性值 / 标记 label / 裸参数里的 cn 叶子 {len(hits)}")
    for pack, p, ctx, en_bad in hits[:10]:
        print(f"  {pack} :: {p[:90]}  英文侧同样如此={en_bad}")
        print(f"     …{ctx}…")

    # ---------------- D2 下次运行的命中面 ----------------
    MIN_SUPPORT = 2
    votes = defaultdict(Counter)
    for pack, (en, cn) in packs.items():
        for p, s in en.items():
            t = cn.get(p)
            if not t:
                continue
            eb, cb = STRONG.findall(s), STRONG.findall(t)
            if len(eb) == len(cb) and eb:
                for a, b in zip(eb, cb):
                    votes[a.strip()][b.strip()] += 1
    table = {k: c.most_common(1)[0][0] for k, c in votes.items()
             if c.most_common(1)[0][1] >= MIN_SUPPORT}
    print(f"\nD2 复刻 fix_bold_drift 第一步：学到对应表 {len(table)} 条"
          f"（--min-support {MIN_SUPPORT}）")

    inject_total = inject_bad = 0
    bad_samples = []
    for pack, (en, cn) in packs.items():
        for p, s in en.items():
            t = cn.get(p)
            if t is None:
                continue
            ne, nc = len(re.findall(r"<strong>", s)), len(re.findall(r"<strong>", t))
            if nc >= ne:
                continue
            already = {w.strip() for w in STRONG.findall(t)}
            new = t
            for w in STRONG.findall(s):
                if len(re.findall(r"<strong>", new)) >= ne:
                    break
                cw = table.get(w.strip())
                if not cw or cw in already:
                    continue
                if new.count(cw) != 1 or f"<strong>{cw}</strong>" in new:
                    continue
                pos = new.find(cw)
                spans = protected_spans(new)
                inject_total += 1
                if in_protected(pos, spans):
                    inject_bad += 1
                    if len(bad_samples) < 10:
                        bad_samples.append((pack, p, cw,
                                            new[max(0, pos - 70):pos + 70]))
                new = new.replace(cw, f"<strong>{cw}</strong>", 1)
                already.add(cw)
    print(f"   下次 --write 会做的 <strong> 注入总数 {inject_total}")
    print(f"   其中注入点落在属性值 / 标记 label / 裸参数里的 {inject_bad}")
    for pack, p, cw, ctx in bad_samples:
        print(f"\n   {pack} :: {p[:95]}")
        print(f"     要包粗的词 {cw!r}，落点上下文：")
        print(f"     …{ctx}…")


if __name__ == "__main__":
    main()
