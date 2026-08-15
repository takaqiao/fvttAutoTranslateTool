# -*- coding: utf-8 -*-
"""**逐处变异回测**：把「阿克图瑞尔|阿克图里安」落在增强器内的那 278 处逐个改错，
每处重跑**全部读库断言**，统计有多少处会让某条断言变红。

为什么必须是这个形态（第十八轮的教训）
--------------------------------------
上一轮 `R-arcturel-arcturian-blocks` 的 why 里写着「标签另有 X / Y / Z 三条闸看着」——
一做逐处变异回测就塌：一条一处都盖不到、一条根本不在断言套里。
**「另有 X 闸看着」是可证伪断言，写之前必须逐处变异回测。** 本脚本就是那个回测。

判定口径
--------
对每一处，跑两遍断言：原文一遍、改错一遍，取**新增的违规行**。
两遍相减而不是「看改错后有没有红」，是为了把配置级噪声（min_hits / 死豁免 / 版本矩阵之类
与本处无关、两遍都在的行）自动抵消掉 —— 否则统计的是噪声不是覆盖。

`--old` 用第十八轮的 52 条（剔掉本轮新增的 7 条 `enricher_slot_gate`）复现基线；
默认两套都跑，直接给出「无闸 189 → X」。

⚠ 变异只发生在**内存里的 ctx.pairs**，真库一个字节都不碰（跑完校验 sha256）。
⚠ 只跑**读 ctx.pairs 的断言**。`exclusions_closed` / `glossary_value` / `version_matrix`
   从磁盘或子进程取数据，内存变异它们本来就看不见 —— 把它们算进来只会稀释口径。
   这三条与本回测要回答的问题（「标签里的译名有没有闸」）无关，此处显式列出而不是默默跳过。
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
QA = P + "/3-常用脚本/qa"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, QA)
import assert_resolutions as A          # noqa: E402

RES = P + "/5-其他内容/RESOLUTIONS.assertions.json"
NEW_KIND = "enricher_slot_gate"
DISK_KINDS = {"exclusions_closed", "glossary_value", "version_matrix"}
SWAP = {"阿克图瑞尔": "阿克图里安", "阿克图里安": "阿克图瑞尔"}
TERM = re.compile("|".join(SWAP))
ENR = [re.compile(r"@[A-Za-z]+\[[^\]]*\](?:\{[^{}]*\})?"), re.compile(r"\[\[[^\]]*\]\]")]


def sha_tree():
    h = hashlib.sha256()
    for repo in ("1-Ember汉化插件", "2-Crucible汉化插件"):
        for sub in ("en", "cn"):
            d = os.path.join(P, repo, "compendium", sub)
            for f in sorted(os.listdir(d)):
                with open(os.path.join(d, f), "rb") as fh:
                    h.update(f.encode()); h.update(fh.read())
    return h.hexdigest()


class Sub:
    """只含指定几叶的假 ctx（lang 通道原样透传）。"""

    def __init__(self, rows, ctx):
        self._rows = rows
        self._ctx = ctx

    def all_pairs(self, scope=None):
        for r in self._rows:
            if scope and r[0] not in scope:
                continue
            yield r

    def all_lang(self, scope=None):
        return self._ctx.all_lang(scope)


def run_rules(rules, ctx):
    """跑一遍，返回违规行集合（去掉不稳定的长文本，只留可比的键）。"""
    out = set()
    for rule in rules:
        fn = A.KINDS.get(rule["kind"])
        if not fn:
            continue
        try:
            bad, _ = fn(rule, ctx)
        except Exception as exc:                    # noqa: BLE001
            out.add((rule["id"], "!!", repr(exc)[:80]))
            continue
        for repo, pack, path, why in bad:
            out.add((rule["id"], repo, str(path), why[:120]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="每处都用**整库 ctx** 重跑（最保守，慢）；默认用单叶子集 ctx 跑，"
                         "再抽样与整库口径对照")
    ap.add_argument("--sample", type=int, default=16, help="单叶口径 vs 整库口径的对照样本数")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--stride", type=int, default=1,
                    help="每隔 N 处取一处 —— 抽样对照要**摊开取**，只取前 N 处会全落在同一本 journal 上")
    ap.add_argument("--out", default="backtest.json")
    a = ap.parse_args()

    before = sha_tree()
    rules_all = json.load(open(RES, encoding="utf-8"))["assertions"]
    rules_mem = [r for r in rules_all if r["kind"] not in DISK_KINDS]
    rules_old = [r for r in rules_mem if r["kind"] != NEW_KIND]
    print(f"断言总数 {len(rules_all)}；读 ctx.pairs 的 {len(rules_mem)}；"
          f"其中本轮新增的 {NEW_KIND} {len(rules_mem) - len(rules_old)} 条")
    print(f"磁盘/子进程型（本回测不跑，内存变异它们看不见）：{sorted(DISK_KINDS)}\n")

    repos = {"ember": os.path.join(P, "1-Ember汉化插件"),
             "crucible": os.path.join(P, "2-Crucible汉化插件")}
    ctx = A.Ctx(repos, json.load(open(RES, encoding="utf-8")).get("meta"))

    # ---- 枚举 278 处 ----
    sites = []
    for repo in repos:
        for idx, (pack, path, ev, cv) in enumerate(ctx.pairs[repo]):
            if not TERM.search(cv):
                continue
            spans = [(m.start(), m.end()) for p in ENR for m in p.finditer(cv)]
            for m in TERM.finditer(cv):
                if any(s <= m.start() < e for s, e in spans):
                    sites.append((repo, idx, pack, path, m.start(), m.group()))
    print(f"落在增强器内的出现处：{len(sites)}")
    if a.stride > 1:
        sites = sites[::a.stride]
    if a.limit:
        sites = sites[:a.limit]

    def mutate(repo, idx, off, term):
        pack, path, ev, cv = ctx.pairs[repo][idx]
        nv = cv[:off] + SWAP[term] + cv[off + len(term):]
        return (pack, path, ev, cv), (pack, path, ev, nv)

    rows = []
    t0 = time.time()
    # 整库口径下基线是**常量**（每处跑完都还原），只算一次；放在循环里会白跑一倍时间。
    full_base_old = run_rules(rules_old, ctx) if a.full else None
    full_base_new = run_rules(rules_mem, ctx) if a.full else None
    if a.full:
        print(f"整库基线算完（{time.time() - t0:.0f}s）：旧套 {len(full_base_old)} 行 / "
              f"新套 {len(full_base_new)} 行")
    for n, (repo, idx, pack, path, off, term) in enumerate(sites):
        orig, mut = mutate(repo, idx, off, term)
        if a.full:
            base_old, base_new = full_base_old, full_base_new
            ctx.pairs[repo][idx] = mut
            got_old = run_rules(rules_old, ctx)
            got_new = run_rules(rules_mem, ctx)
            ctx.pairs[repo][idx] = orig
        else:
            sub_o = Sub([(repo,) + orig], ctx)
            sub_m = Sub([(repo,) + mut], ctx)
            base_old, base_new = run_rules(rules_old, sub_o), run_rules(rules_mem, sub_o)
            got_old, got_new = run_rules(rules_old, sub_m), run_rules(rules_mem, sub_m)
        d_old, d_new = got_old - base_old, got_new - base_new
        rows.append({"repo": repo, "pack": pack, "path": path, "off": off, "term": term,
                     "old_hit": sorted({r[0] for r in d_old}),
                     "new_hit": sorted({r[0] for r in d_new})})
        if (n + 1) % 40 == 0:
            print(f"  …{n + 1}/{len(sites)}  ({time.time() - t0:.0f}s)")

    json.dump(rows, open(os.path.join(HERE, a.out), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    from collections import Counter
    old_cov = [r for r in rows if r["old_hit"]]
    new_cov = [r for r in rows if r["new_hit"]]
    print(f"\n{'=' * 70}")
    print(f"逐处变异 {len(rows)} 处（口径：{'整库 ctx' if a.full else '单叶子集 ctx'}）")
    print(f"  第十八轮 52 条：有闸 {len(old_cov)} · **无闸 {len(rows) - len(old_cov)}**")
    print(f"  本轮  59 条：有闸 {len(new_cov)} · **无闸 {len(rows) - len(new_cov)}**")
    print("\n旧套按断言分布：", Counter(x for r in old_cov for x in r["old_hit"]).most_common())
    print("新套按断言分布：", Counter(x for r in new_cov for x in r["new_hit"]).most_common())
    left = [r for r in rows if not r["new_hit"]]
    print(f"\n新套仍然无闸的 {len(left)} 处：")
    for r in left[:40]:
        print(f"  [{r['repo']}/{r['pack']}] {r['path'][:88]} @{r['off']} {r['term']}")

    after = sha_tree()
    print(f"\n真库指纹：{'同（一个字节都没碰）' if before == after else '**变了 —— 不该有**'}")
    return 0 if before == after else 1


main()
