#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全库「同一条英文串有 N 种中文」—— 玩家在不同地方看到同一段文字的两种译法。

为什么需要这一档（2026-08-13 第九轮新增）：
  - `sync_twin_packs.py` 只比两个孪生包的**同一条路径**；
  - `propagate_fix.py` 只推 `--since <commit>` 之后改过的叶子；
  - `scan_name_splits.py` 只比 `name` 字段。
  **同一条英文串出现在不同路径上**（例：`Gesture: Influence.description` 挂在 23 个 actor 身上）
  从来没有任何判据比过。首测：1514 条英文串有 2 种以上中文，涉及 8308 叶。

这不只是观感。首测抽样立刻看到三类真缺陷：
  - 定译被译错：`Presence` 在两份副本里成了「风采」「风范」（定译是「存在」）
  - 裸英文残留：六份副本写「Influence手势」
  - **多数派是错的那一边**：`Light Weapon Training / lunge` 的 16 份副本写
    「使用一把具有灵巧属性的武器」，英文是 `scales using Dexterity`（＝以敏捷成长），
    16 : 7 的多数派整个译错。**所以本脚本只报分叉、不给建议**，方向必须人判。

────────────────────────────────────────────────────────────────────────────
2026-08-13 第十二轮：加双语尾巴归一（本文件唯一一次判据变更）
────────────────────────────────────────────────────────────────────────────
本库的既定约定是**同一个词按字段类型写两种形态**：

    actors.X.name        = "螯蛛艾斯 Cheliceraeth"   ← 双语并列
    actors.X.tokenName   = "螯蛛艾斯"                ← 裸中文
    tables.…results.name = "螯蛛艾斯"                ← 裸中文
    scenes.…levels.<k>   = "螯蛛艾斯"                ← 裸中文
    <pack>.label         = "血统\\nAncestries"        ← 换行分隔的双语

`scan_token_name.py` 就是按这个约定判的（它剥掉双语尾巴后比对，现报 0）。
本脚本原先逐字节比中文，于是把这条约定整个当成缺陷报出来：
**479 组 / 2566 叶里有 463 组纯粹是「一处带英文尾巴、一处不带」**，
`裸英文残留` 旗标因此在 467 组上无差别地亮着，等于没有信号。

归一规则（**保守**，只剥「与该叶英文值逐字相同」的尾巴）：
    cn.endswith(en)  且剥掉后剩下的头部非空且含汉字  ->  取头部
  尾巴允许被 () （） [] 【】 包裹，尾巴前允许有空白/换行/连接号/冒号/逗号。
  **不做部分匹配**：`EN="Gesture: Aspect"` 而中文尾巴只有 `Aspect` 的不剥
  —— 那种情况说明双语尾巴本身不完整，属于另一档判据的事。

归一**不会**掩盖真缺陷：被归一收掉的组，其两个变体的中文头部**逐字节相同**，
差别只有那条英文尾巴的有无。

⚠ 交接说明订正（2026-08-14 第十四轮）：这里原先写着「某字段该不该带双语尾巴」是
`scan_bare_english_names` / `scan_token_name` 的辖区。**实查两个接盘方都看不见这一
维度**：`scan_bare_english_names.py:185` 的主扫描循环第一行就是
`if path.endswith('.name') or not CJK.search(v): continue`，把所有 `.name` 路径整个
排除掉了（同文件 :138-162 对 `.name` 的用法是**反向**的 —— 从 EN 侧建「英文专名→中
文译名」词典）；`scan_token_name.py` 只管 tokenName 该不该**剥掉**尾巴，不管 name
该不该**带**尾巴。所以在 2026-08-14 之前，「应带双语尾巴的 name 缺了尾巴」全库**无
判据负责**（已复验的漏网例：`crucible.rules.json` 的 `entries.Combat.pages` 共 12 页，
11 页是「先攻与回合顺序 Initiative and Turn Order」这种双语，唯
`Engagement and Flanking` 的 name = 「交战与夹击」，没有英文尾巴）。
该维度现由 `qa/scan_missing_bilingual_tail.py` 负责，本判据仍然只报分叉。

回测（1-Ember汉化插件 + 2-Crucible汉化插件，2026-08-13 round12 基线）：
    --raw（旧判据）  479 组 / 2566 叶，其中「带旗标」467
    默认（新判据）   16 组 /  138 叶，其中「带旗标」5
  `--selftest` 会往索引里注入一个已知真分裂（`Monstrosities` 怪物 / 畸怪）
  和一个纯约定分裂（`概览 X` / `概览`），断言前者仍报出、后者被收掉。

⚠ 有些分叉是合法的，不要机械统一（第十二轮已逐条裁定，见 round12 记录）：
  - `Shield`：带 `Imperceptible Barrier` 效果的是法术→护盾术，裸的是装备→盾牌
  - `Light`：actor 身上的戏法→光，`crucible.equipment` 的负重分级折叠→轻型
  - `Arcturian`：文化/血统→阿克图里安，actor 本体→阿克图里安人
  - `Luminous` / `Spirited`：`ember.character.json` 是 dnd5e 侧包，与 crucible 永不同载
  - `Hallows`：组织→幽圣所，城区/场景层→圣堂区
  - 英文 ≤20 字符的短串风险最高，逐条查语境

用法：
  python scan_same_en_split.py --repo 1-Ember汉化插件 --repo 2-Crucible汉化插件 \
         [--min-en-len N] [--flagged-only] [--raw] [--selftest] --out <json>
"""
import argparse
import collections
import json
import os
import re
import sys

# 定译表：英文 -> 可接受的中文写法。用于给分叉组打「术语分歧」旗标。
# 只在英文串里确实出现该词时才检查（英文闸），否则会大面积误伤。
CANON = {
    "Presence": ["存在"], "Fortitude": ["强韧"], "Toughness": ["坚韧"],
    "Wisdom": ["感知"], "Willpower": ["意志"], "Boon": ["恩惠骰"], "Bane": ["祸骰"],
    "Electricity": ["电击"], "Lightning": ["闪电"], "Cold": ["寒冷"],
    "Accurate": ["精准"], "Stride": ["步幅"], "Arrow": ["箭矢"],
    "Attunement": ["同调"], "essence": ["精华"], "Kinesis": ["念力"], "Tier": ["阶"],
    "Confused": ["混乱"], "Restrained": ["受缚"], "Guarded": ["戒备"],
}

TAG = re.compile(r"<[^>]*>")
ENRICHER = re.compile(r"@\w+\[[^\]]*\]|\[\[[^\]]*\]\]")
LATIN = re.compile(r"[A-Za-z]{4,}")
CJK = re.compile(r"[㐀-鿿]")

# 双语尾巴可以被这些成对符号包起来：「概览（Overview）」
WRAPS = [("", ""), ("(", ")"), ("（", "）"), ("[", "]"), ("【", "】")]
# 尾巴前允许出现的分隔符（rstrip 用字符集，不是范围）
SEPS = " \t\r\n　-—–~·:：,，、;；/|(（[【"


def strip_bilingual_tail(cn, en):
    """剥掉与该叶英文值逐字相同的双语尾巴：「概览 Overview」-> 「概览」。

    只在剥完还剩下含汉字的头部时才剥；`cn` 本身就等于 `en`（整条没译）时不动，
    这样未翻译残留仍然会与已翻译的变体分裂并被报出来。
    """
    s = (cn or "").strip()
    e = (en or "").strip()
    if not e or s == e:
        return s
    for lb, rb in WRAPS:
        pat = lb + e + rb
        if len(pat) < len(s) and s.endswith(pat):
            head = s[:-len(pat)].rstrip(SEPS)
            if head and CJK.search(head):
                return head
    return s


def leaves(obj, path, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            leaves(v, (path + "." + str(k)) if path else str(k), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            leaves(v, path + "[" + str(i) + "]", out)
    elif isinstance(obj, str):
        out[path] = obj


def visible(s):
    """剥掉标记与 enricher，只留玩家看得见的裸文字。"""
    return TAG.sub("", ENRICHER.sub("", s))


def collect(repos):
    by_en = collections.defaultdict(lambda: collections.defaultdict(list))
    for repo in repos:
        cdir = os.path.join(repo, "compendium", "cn")
        edir = os.path.join(repo, "compendium", "en")
        if not os.path.isdir(cdir):
            print("跳过（无 compendium/cn）:", repo)
            continue
        for fn in sorted(os.listdir(cdir)):
            if not fn.endswith(".json"):
                continue
            ep = os.path.join(edir, fn)
            if not os.path.exists(ep):
                continue
            cn_leaves, en_leaves = {}, {}
            leaves(json.load(open(os.path.join(cdir, fn), encoding="utf-8-sig")), "", cn_leaves)
            leaves(json.load(open(ep, encoding="utf-8-sig")), "", en_leaves)
            for path, en_val in en_leaves.items():
                cn_val = cn_leaves.get(path)
                if cn_val:
                    by_en[en_val][cn_val].append([repo, fn, path])
    return by_en


# ── 自测注入 ────────────────────────────────────────────────────────────────
# 归一必须**只**收掉「纯双语尾巴」的分叉，真分叉一条都不能少报。
SELFTEST_TRUE_SPLIT = ("Monstrosities", "怪物", "畸怪")
SELFTEST_CONVENTION = ("__SELFTEST_Overview__", "概览 __SELFTEST_Overview__", "概览")


def inject_selftest(by_en):
    en, a, b = SELFTEST_TRUE_SPLIT
    by_en[en][a].append(["__selftest__", "__selftest__.json", "injected.trueSplit.a"])
    by_en[en][b].append(["__selftest__", "__selftest__.json", "injected.trueSplit.b"])
    en, a, b = SELFTEST_CONVENTION
    by_en[en][a].append(["__selftest__", "__selftest__.json", "injected.convention.a"])
    by_en[en][b].append(["__selftest__", "__selftest__.json", "injected.convention.b"])


def check_selftest(groups, raw):
    reported = {g["en"] for g in groups}
    ok = True
    if SELFTEST_TRUE_SPLIT[0] not in reported:
        print("  ✗ 真分叉 %r（怪物/畸怪）被归一吞掉了 —— 判据已改瞎" % SELFTEST_TRUE_SPLIT[0])
        ok = False
    else:
        print("  ✓ 真分叉 %r 仍然报出" % SELFTEST_TRUE_SPLIT[0])
    conv_reported = SELFTEST_CONVENTION[0] in reported
    if raw:
        print("  ✓ --raw 下约定分叉 %r 报出（旧行为）" % SELFTEST_CONVENTION[0]
              if conv_reported else
              "  ✗ --raw 下约定分叉应当报出却没有")
        ok = ok and conv_reported
    else:
        print("  ✓ 约定分叉「概览 X / 概览」被归一收掉" if not conv_reported else
              "  ✗ 约定分叉「概览 X / 概览」没被收掉 —— 归一没生效")
        ok = ok and not conv_reported
    return ok


def flags_for(en_val, variants):
    out = []
    for term, oks in CANON.items():
        # 英文闸：英文串里确实出现该词才检查
        if not re.search(r"\b" + re.escape(term) + r"\b", en_val, re.I):
            continue
        hit = [any(o in cn for o in oks) for cn, _ in variants]
        if any(hit) and not all(hit):
            out.append("术语分歧:" + term)
    # 裸英文：某变体的可见文字里有拉丁词，而不是所有变体都有
    has_latin = [bool(LATIN.search(visible(cn))) for cn, _ in variants]
    if any(has_latin) and not all(has_latin):
        out.append("裸英文残留")
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--min-en-len", type=int, default=0,
                    help="英文短于此长度的不报（短标签合法分叉率高）")
    ap.add_argument("--flagged-only", action="store_true",
                    help="只报带旗标（术语分歧/裸英文）的组")
    ap.add_argument("--raw", action="store_true",
                    help="关掉双语尾巴归一，回到 2026-08-13 之前的逐字节比对（回测用）")
    ap.add_argument("--selftest", action="store_true",
                    help="注入一条已知真分叉和一条纯约定分叉，断言归一没把判据改瞎")
    ap.add_argument("--out")
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    by_en = collect(args.repo)
    if args.selftest:
        inject_selftest(by_en)

    groups = []
    for en_val, d in by_en.items():
        if len(en_val) < args.min_en_len:
            continue
        # 归一：把只差一条「与英文值相同的尾巴」的写法并成一个变体。
        # 组内挑一个原始写法做代表（叶多的那个），原始写法全留在 raw_variants 里。
        merged = collections.OrderedDict()
        for cn, paths in sorted(d.items(), key=lambda t: -len(t[1])):
            key = cn if args.raw else strip_bilingual_tail(cn, en_val)
            slot = merged.setdefault(key, {"cn": key, "paths": [], "raw": []})
            slot["paths"].extend(paths)
            if cn != key:
                slot["raw"].append(cn)
        if len(merged) < 2:
            continue
        variants = sorted(merged.values(), key=lambda v: -len(v["paths"]))
        pairs = [(v["cn"], v["paths"]) for v in variants]
        fl = flags_for(en_val, pairs)
        if args.flagged_only and not fl:
            continue
        groups.append({
            "en": en_val,
            "en_len": len(en_val),
            "n_leaf": sum(len(v["paths"]) for v in variants),
            "n_variant": len(variants),
            "flags": fl,
            "variants": [{"cn": v["cn"], "n": len(v["paths"]),
                          "raw_forms": sorted(set(v["raw"])), "paths": v["paths"]}
                         for v in variants],
        })

    # 优先级：带旗标 > 3 种以上中文 > 叶多 > 英文长
    groups.sort(key=lambda g: (-len(g["flags"]), -(g["n_variant"] >= 3), -g["n_leaf"], -g["en_len"]))

    total_leaf = sum(g["n_leaf"] for g in groups)
    print("模式                        %s" % ("--raw 逐字节（旧判据）" if args.raw else "双语尾巴归一（默认）"))
    print("英文唯一串（有中文的）      %d" % len(by_en))
    print("**同一英文串有 2 种以上中文** %d 组 / %d 叶" % (len(groups), total_leaf))
    print("  带旗标            %d" % sum(1 for g in groups if g["flags"]))
    print("    术语分歧        %d" % sum(1 for g in groups if any(f.startswith("术语分歧") for f in g["flags"])))
    print("    裸英文残留      %d" % sum(1 for g in groups if "裸英文残留" in g["flags"]))
    print("  3 种以上中文      %d" % sum(1 for g in groups if g["n_variant"] >= 3))
    print()
    for lo, hi, lab in [(0, 20, "≤20 (短标签, 合法分叉率高)"), (20, 80, "21-80"),
                        (80, 300, "81-300"), (300, 10 ** 9, ">300 (长正文, 几乎必是缺陷)")]:
        sel = [g for g in groups if lo < g["en_len"] <= hi]
        print("  英文 %-28s 组 %5d  叶 %6d" % (lab, len(sel), sum(g["n_leaf"] for g in sel)))

    if args.show:
        print("\n前 %d 组（按优先级）：" % args.show)
        for g in groups[:args.show]:
            print("  叶%3d 变体%2d %s  EN=%r" % (g["n_leaf"], g["n_variant"], g["flags"], g["en"][:64]))
            for v in g["variants"]:
                print("        %3d  %r%s" % (v["n"], v["cn"][:56],
                                             ("  (原文形态 %s)" % v["raw_forms"][:2]) if v["raw_forms"] else ""))

    rc = 0
    if args.selftest:
        print("\n自测：")
        rc = 0 if check_selftest(groups, args.raw) else 2

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(groups, fh, ensure_ascii=False, indent=1)
        print("\n-> " + args.out)
    return rc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
