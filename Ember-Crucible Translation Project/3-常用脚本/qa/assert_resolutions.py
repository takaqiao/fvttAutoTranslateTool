#!/usr/bin/env python3
"""把 PROJECT.md 第 8 节的裁决**编译成可执行断言**，跑在 5.4 全套后面。

  python assert_resolutions.py [--rules <json>] [--repo <repo> ...] [--verbose]

为什么需要这个
--------------
第 8 节现在有上百条裁决，**全是散文**。没有任何机制阻止下一轮悄悄推翻某一条 ——
十四轮里已经出过好几次险：`scan_uuid_swap` 改判据降噪时消掉过一条真缺陷；
`Scout` 的处理差点与已归档豁免冲突；给 crucible 的 CI 断言照抄进 ember 差点让下一次发版失败；
`Trident's Point` 的「本地区+地图」差点被术语替换改坏。**这些都是靠人当场想起来才没出事。**

散文裁决 → 机器断言，一致性就从「靠记忆维持」变成「靠闸维持」。

断言类型
--------
`term_gated`      英文侧命中 `en` 的叶，中文必须含 `cn_required`（且不得含 `cn_forbidden`）
`cn_absent`       某个中文串在全库不得出现（已清零的错译不许回潮）
`distinct_terms`  一组术语的中文必须两两不同（防撞名）
`lang_parity`     两仓 `lang/cn.json` 的键数必须等于英文侧键数
`anchor_ids`      标题上的显式 `id=` 数量不得低于阈值（撑着锚点链接）
`no_bilingual_tail` 指定字段的中文不得带「中文 English」双语尾巴
`exclusions_closed` `same_en_split` 的分组必须全部在已归档豁免表内
`leaf_literal`    指定叶必须/不得包含某个字面串（用于登记「绝不能动」的假阳性）
`glossary_value`  词表的 base 层与产物层都必须是某个值（防「词表把错误洗成权威」）

每条断言都带 `decision`（对应第 8 节哪一天的裁决）与 `why`，失败时一并打印 ——
让下一个人看到的不是「断言 R-xx 挂了」，而是「你违反了 2026-08-13j 定的那条，理由是……」。

已知局限（**别把它当成全覆盖**，这是本项目方法教训 1 的又一处应用）
-----------------------------------------------------------------
1. `term_gated` 是**叶级**的：只要求该叶中文里**出现过**定译。一叶里提到该术语 5 次、
   只错 1 次的情况**它抓不到**。回测实测：把一叶里 3 处「邪术师」全改成「术士」才会响，
   只改 1 处不响。要抓叶内部分错译，得做逐位对齐（成本高得多），本轮没做。
2. `cn_absent` 反过来会**误伤合法用法**：某个废弃写法如果在别处是正当中文，会假阳性。
   目前 6 条都验过全库为 0，加新条目前务必先数一遍。必要时用 `except_paths` 登记豁免。
3. 断言只覆盖第 8 节里**能机械表达**的那些裁决。像「改动面小的那边优先」「name 与正文
   冲突时多数该改 name」这类**方法性**裁决，本质上表达不成断言，仍然只能靠人读第 8 节。

自检与回测
----------
`--selftest` 跑判据自身的正反例（双语尾巴那条最容易写错，第一版就把「圣堂区路人 A」
这种编号后缀全报成了尾巴）。
`--root <另一棵树>` 用于灵敏度回测：往副本里注入违规，确认断言真的会响 ——
**只测特异度（全绿）是不够的**，那样「所有断言都返回空」也能过。
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # 项目根
DEFAULT_RULES = os.path.join(ROOT, "5-其他内容", "RESOLUTIONS.assertions.json")
REPOS = {"ember": "1-Ember汉化插件", "crucible": "2-Crucible汉化插件"}

# 双语尾巴：中文 …… 空白 + 拉丁串结尾。
#
# ⚠ **必须排除单字母编号。** 第一版写成 `[A-Z][A-Za-z'\-]*`，把「圣堂区路人 A」
# 「菌丝旷野底图 B」这类**编号后缀**全报成双语尾巴（14 处全是假阳性）——
# 而那正是既定约定要求的写法（英文侧就是 `Hallows Passerby A`，字母是编号本体，
# 前面留半角空格是库内 name 的惯例）。
#
# 判据取与 `scan_same_en_split` 的归一规则同源的定义：尾巴是**英文名本身**被附在中文后面，
# 所以要求结尾那串拉丁**至少有 2 个字母**。单个字母（或数字）是编号，不是名字。
_TAIL = re.compile(r"[一-鿿].*?\s+([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*)$")


def has_bilingual_tail(text):
    m = _TAIL.match(text.strip())
    if not m:
        return False
    letters = re.sub(r"[^A-Za-z]", "", m.group(1))
    return len(letters) >= 2


def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def load_pack_pairs(repo_dir):
    """产出 (pack, path, en, cn)；只收两侧都有的叶。"""
    en_dir = os.path.join(repo_dir, "compendium", "en")
    cn_dir = os.path.join(repo_dir, "compendium", "cn")
    if not os.path.isdir(en_dir):
        return
    for fname in sorted(os.listdir(en_dir)):
        if not fname.endswith(".json") or fname == "_source.json":
            continue
        cn_path = os.path.join(cn_dir, fname)
        if not os.path.exists(cn_path):
            continue
        en = dict(walk(json.load(open(os.path.join(en_dir, fname), encoding="utf-8-sig"))))
        cn = dict(walk(json.load(open(cn_path, encoding="utf-8-sig"))))
        for p, ev in en.items():
            cv = cn.get(p)
            if cv is not None:
                yield fname, p, ev, cv


class Ctx:
    """一次性把两个仓库读进来，所有断言共用（否则每条断言重读一遍太慢）。"""

    def __init__(self, repos):
        self.repos = repos
        self.pairs = {}
        for name, d in repos.items():
            self.pairs[name] = list(load_pack_pairs(d))

    def all_pairs(self, scope):
        for name in (scope or self.repos.keys()):
            for row in self.pairs.get(name, []):
                yield (name,) + row


# ----------------------------------------------------------------- 断言实现

def a_term_gated(rule, ctx):
    en_re = re.compile(rule["en"])
    req = rule["cn_required"]
    forbid = rule.get("cn_forbidden", [])
    bad = []
    hits = 0
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        if not en_re.search(ev):
            continue
        hits += 1
        if req not in cv:
            bad.append((repo, pack, path, f"英文命中但中文无「{req}」"))
            continue
        for f in forbid:
            if f in cv:
                bad.append((repo, pack, path, f"中文同时出现禁用写法「{f}」"))
                break
    return bad, f"英文闸命中 {hits} 叶"


def a_cn_absent(rule, ctx):
    needle = rule["cn"]
    allow = set(rule.get("except_paths", []))
    bad = []
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        if needle in cv and path.removeprefix("entries.") not in allow:
            bad.append((repo, pack, path, f"出现了已废弃的写法「{needle}」"))
    return bad, f"扫描全库"


def a_distinct_terms(rule, ctx):
    """一组 {en, cn} 的中文必须两两不同。纯配置自检，不读库。"""
    seen = {}
    bad = []
    for item in rule["terms"]:
        cn = item["cn"]
        if cn in seen:
            bad.append(("-", "-", item["en"], f"与「{seen[cn]}」共用中文「{cn}」"))
        seen[cn] = item["en"]
    return bad, f"{len(rule['terms'])} 个术语"


def a_lang_parity(rule, ctx):
    bad = []
    detail = []
    for name, pkg in rule["packages"].items():
        repo = ctx.repos.get(name)
        if not repo:
            continue
        cn_p = os.path.join(repo, "lang", "cn.json")
        en_p = os.path.join(os.path.expandvars(pkg), "lang", "en.json")
        if not (os.path.exists(cn_p) and os.path.exists(en_p)):
            bad.append((name, "-", cn_p, "lang 文件找不到，无法核对"))
            continue
        cn = dict(walk(json.load(open(cn_p, encoding="utf-8-sig"))))
        en = dict(walk(json.load(open(en_p, encoding="utf-8-sig"))))
        detail.append(f"{name}: cn {len(cn)} / en {len(en)}")
        if len(cn) != len(en):
            bad.append((name, "-", "lang/cn.json", f"键数 {len(cn)} != 英文侧 {len(en)}"))
    return bad, " | ".join(detail)


def a_anchor_ids(rule, ctx):
    pat = re.compile(r"<h[1-6][^>]*\sid=", re.IGNORECASE)
    n = 0
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        n += len(pat.findall(cv))
    if n < rule["min"]:
        return [("-", "-", "-", f"标题显式 id 只有 {n} 个，低于阈值 {rule['min']}")], f"实测 {n}"
    return [], f"实测 {n} 个（阈值 {rule['min']}）"


def a_no_bilingual_tail(rule, ctx):
    fields = tuple(rule["fields"])
    bad = []
    n = 0
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        seg = path.replace("[", ".").split(".")
        # 字段名可能是最后一段（tokenName），也可能是倒数第二段（encounterTokens.<名>）
        if not (seg[-1] in fields or (len(seg) >= 2 and seg[-2] in fields)):
            continue
        n += 1
        if has_bilingual_tail(cv):
            bad.append((repo, pack, path, f"带双语尾巴：{cv[:40]!r}"))
    return bad, f"检查 {n} 个该约定下的叶"


def a_exclusions_closed(rule, ctx):
    exc_p = os.path.join(ROOT, rule["exclusions"])
    rep_p = os.path.join(ROOT, rule["report"])
    if not os.path.exists(rep_p):
        return [("-", "-", rep_p, "找不到 same_en_split 报告，先跑一次该扫描")], "跳过"
    exc = json.load(open(exc_p, encoding="utf-8"))
    blob = " ".join(f"{e.get('what','')} {e.get('why','')}" for e in exc)
    cur = json.load(open(rep_p, encoding="utf-8"))
    bad = []
    for g in cur:
        en = g.get("en", "")
        key = en if len(en) < 40 else (re.search(r"Adelyne|lookup @name", en) or [None])[0]
        key = key if isinstance(key, str) else (en[:30] if not key else key.group(0) if hasattr(key, "group") else en[:30])
        if key not in blob:
            bad.append(("-", "-", en[:60], "该分叉组不在已归档豁免表里 —— 要么是新缺陷，要么要补一条豁免"))
    return bad, f"当前 {len(cur)} 组 / 归档 {len(exc)} 条"


def a_glossary_value(rule, ctx):
    """词表的 base 层与产物层**都**必须是这个值。

    只查产物是不够的：词表是构建产物，`build_glossary.py` 会用 base + harvest 重建，
    只改产物下一次构建就退回去（2026-08-13j 的裁决）。所以两层一起查。
    这一条守的是本项目最隐蔽的机制 —— **词表把错误洗成权威**：
    包里改对了，词表停在旧值，下一轮有人拿词表当依据反向回灌。
    """
    bad = []
    checked = 0
    for layer, rel in rule["files"].items():
        path = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
        if not os.path.exists(path):
            # base 层在项目根之外（fvtt/），跑在别的树上时可能不存在，跳过而不是假失败
            continue
        d = json.load(open(path, encoding="utf-8"))
        for key, want in rule["entries"].items():
            if key not in d:
                if rule.get("require_present", True):
                    bad.append((layer, os.path.basename(path), key, f"词表里缺这个锚点键（应为「{want}」）"))
                continue
            checked += 1
            got = d[key]
            # 产物层允许带双语尾巴（既定书写约定），只比中文头部
            head = got.split(" ")[0] if isinstance(got, str) else got
            if want not in (got, head):
                bad.append((layer, os.path.basename(path), key, f"是「{got}」，应为「{want}」"))
    return bad, f"两层合计核对 {checked} 条"


def a_leaf_literal(rule, ctx):
    bad = []
    checked = 0
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        if path.removeprefix("entries.") != rule["path"] or pack != rule["pack"]:
            continue
        checked += 1
        for must in rule.get("must_contain", []):
            if must not in cv:
                bad.append((repo, pack, path, f"必须包含「{must}」但没有"))
        for never in rule.get("must_not_contain", []):
            if never in cv:
                bad.append((repo, pack, path, f"不得包含「{never}」但出现了"))
    if not checked:
        bad.append(("-", rule["pack"], rule["path"], "这一叶找不到了（上游改名？）—— 断言失效，需要人看"))
    return bad, f"命中 {checked} 叶"


KINDS = {
    "term_gated": a_term_gated,
    "cn_absent": a_cn_absent,
    "distinct_terms": a_distinct_terms,
    "lang_parity": a_lang_parity,
    "anchor_ids": a_anchor_ids,
    "no_bilingual_tail": a_no_bilingual_tail,
    "exclusions_closed": a_exclusions_closed,
    "leaf_literal": a_leaf_literal,
    "glossary_value": a_glossary_value,
}


# ----------------------------------------------------------------- 自检

SELFTEST = [
    # (值, 是否应判为双语尾巴)  —— 正例来自库内真实写法，反例来自本项目既定约定
    ("螯蛛艾斯 Cheliceraeth", True),
    ("血鸟 Gore Bird", True),
    ("赛洛克弓手 Thayloc Courser", True),
    ("圣堂区路人 A", False),        # 单字母编号，英文侧就是 Hallows Passerby A
    ("菌丝旷野底图 B", False),      # 同上
    ("软泥池 2", False),            # 数字编号
    ("环境音效（2）", False),        # 全角括注
    ("斥候", False),                # 纯裸中文
    ("莫伊雷语", False),
    ("瀑布外部东北", False),
]


def run_selftest():
    bad = 0
    for value, want in SELFTEST:
        got = has_bilingual_tail(value)
        flag = "ok  " if got == want else "FAIL"
        if got != want:
            bad += 1
        print(f"  {flag} {value!r:32s} 期望 {want} 实得 {got}")
    print(f"\n自检：{len(SELFTEST) - bad} / {len(SELFTEST)} 通过")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=DEFAULT_RULES)
    ap.add_argument("--repo", action="append", help="限定仓库（默认两个都跑）")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--max-show", type=int, default=6)
    ap.add_argument("--selftest", action="store_true", help="只跑判据自身的正反例回测")
    ap.add_argument("--root", help="改用另一棵项目树（灵敏度回测用：往副本里注入违规，确认断言真的会响）")
    a = ap.parse_args()

    if a.selftest:
        return run_selftest()

    rules = json.load(open(a.rules, encoding="utf-8"))
    global ROOT
    if a.root:
        ROOT = os.path.abspath(a.root)
    repos = {}
    for name, rel in REPOS.items():
        d = os.path.join(ROOT, rel)
        if a.repo and rel not in a.repo and name not in a.repo:
            continue
        if os.path.isdir(d):
            repos[name] = d
    ctx = Ctx(repos)
    total_leaves = sum(len(v) for v in ctx.pairs.values())
    print(f"读入 {len(repos)} 个仓库 / {total_leaves} 对中英叶\n")

    failed = passed = skipped = 0
    for rule in rules["assertions"]:
        fn = KINDS.get(rule["kind"])
        if not fn:
            print(f"  ?? {rule['id']}: 未知断言类型 {rule['kind']}")
            skipped += 1
            continue
        try:
            bad, detail = fn(rule, ctx)
        except Exception as exc:                       # 断言自己炸了也要说清楚，不能静默
            print(f"  !! {rule['id']}: 断言执行出错 {exc!r}")
            failed += 1
            continue
        if bad:
            failed += 1
            print(f"  FAIL  {rule['id']}  —— {rule['title']}")
            print(f"        裁决 {rule['decision']}：{rule['why']}")
            print(f"        {detail}，违反 {len(bad)} 处：")
            for repo, pack, path, why in bad[:a.max_show]:
                print(f"          [{repo}/{pack}] {str(path)[:78]}")
                print(f"            {why}")
            if len(bad) > a.max_show:
                print(f"          …另 {len(bad) - a.max_show} 处")
        else:
            passed += 1
            if a.verbose:
                print(f"  ok    {rule['id']}  {rule['title']}  （{detail}）")

    print(f"\n{'=' * 62}")
    print(f"通过 {passed} / 失败 {failed} / 跳过 {skipped}")
    if failed:
        print("\n⚠ 失败的每一条都对应第 8 节的一条既定裁决。")
        print("  正确的处理是：要么改回来，要么**显式推翻那条裁决并同时改断言**——")
        print("  不要只改断言让它变绿，那正是这套东西要防的事。")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
