# -*- coding: utf-8 -*-
"""扫译文与英文之间的**标记**差异（不是文字差异）。

apply_translations.py 只在回写时把关，库里那些历史遗留的译文从来没被检查过。
这类问题在 diff 里看不出来，但玩家会看到：链接失效、少一段、莫名其妙的加粗。

分四类报：
  LINK      @UUID / @Action / @Condition / [[/…]] 的**目标**对不上 —— 链接是坏的，必须修
  BLOCK     <p>/<li>/<h3>/<table> 等块级标签数量对不上 —— 多一段或少一段正文
  INLINE    <strong>/<em> 数量对不上 —— 加粗漂移，观感问题
  PLACEHOLDER  {xxx} 对不上 —— UI 里会直接露出花括号
  TRUNCATED 译文纯文本长度明显短于英文（中文一般是英文的 0.4–0.7 倍）—— 整段没译

⚠️ TRUNCATED 是 `validate_translations.py` 看不见的一类缺陷：
覆盖率是**按路径**算的，一条路径只要有中文值就算 100%，
哪怕译文把英文十段里的六段直接丢了。

用法：
  python scan_markup_drift.py --repo <repo> [--out <report.json>] [--kind LINK,BLOCK]
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]')
INLINE_CMD = re.compile(r'\[\[[^\]]*\]\]')
PLACEHOLDER = re.compile(r'\{[A-Za-z_][A-Za-z0-9_.\-]*\}')
# @UUID[目标]{标签} / [[/cmd]]{标签} 的标签是给玩家看的文字，本来就该翻译，
# 不能当成占位符去比对，否则每条 @ref[actor.name]{creature} 都会被误报
LABEL_AFTER_MARKUP = re.compile(r'(?:@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\])\s*\{[^}]*\}')


def strip_labels(s):
    return LABEL_AFTER_MARKUP.sub(lambda m: m.group(0).split("{")[0], s)
TAGNAME = re.compile(r'<\s*(/?)([a-zA-Z][a-zA-Z0-9]*)')
BLOCK_TAGS = {"p", "li", "ul", "ol", "table", "tr", "td", "th", "h1", "h2", "h3", "h4",
              "section", "figure", "figcaption", "blockquote", "div", "tbody", "thead", "tfoot"}
INLINE_TAGS = {"strong", "em", "b", "i", "u", "span", "code"}


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


EMPTY_P = re.compile(r"<p>\s*</p>")


def tag_counter(s, which):
    # 英文原文里散着一些空的 <p></p>（排版留白），译文普遍没有照抄。
    # 那不是漏译，先剔掉再数，免得淹没真正的段落差异。
    s = EMPTY_P.sub("", s)
    return Counter(f"{slash}{name.lower()}" for slash, name in TAGNAME.findall(s)
                   if name.lower() in which)


def diff(a, b):
    return {k: (a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b) if a.get(k, 0) != b.get(k, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out")
    ap.add_argument("--kind", help="只报这几类，逗号分隔：LINK,BLOCK,INLINE,PLACEHOLDER")
    args = ap.parse_args()
    kinds = set(args.kind.split(",")) if args.kind else {"LINK", "BLOCK", "INLINE", "PLACEHOLDER", "TRUNCATED"}

    repo = Path(args.repo)
    findings = []
    for f in sorted((repo / "compendium" / "en").glob("*.json")):
        cnp = repo / "compendium" / "cn" / f.name
        if not cnp.exists():
            continue
        en = dict(leaves(json.loads(f.read_text(encoding="utf-8-sig"))))
        cn = dict(leaves(json.loads(cnp.read_text(encoding="utf-8-sig"))))
        for p, s in en.items():
            t = cn.get(p)
            if not t:
                continue
            checks = {
                "LINK": diff(Counter(MARKUP.findall(s)) + Counter(INLINE_CMD.findall(s)),
                             Counter(MARKUP.findall(t)) + Counter(INLINE_CMD.findall(t))),
                "BLOCK": diff(tag_counter(s, BLOCK_TAGS), tag_counter(t, BLOCK_TAGS)),
                "INLINE": diff(tag_counter(s, INLINE_TAGS), tag_counter(t, INLINE_TAGS)),
                "PLACEHOLDER": diff(Counter(PLACEHOLDER.findall(strip_labels(s))),
                                    Counter(PLACEHOLDER.findall(strip_labels(t)))),
            }
            # TRUNCATED：按纯文本长度比判断整段漏译
            plain_en = len(re.sub(r"<[^>]+>", "", s))
            plain_cn = len(re.sub(r"<[^>]+>", "", t))
            # 本库译文/英文的纯文本长度比中位数是 0.31（中文本来就紧凑）。
            # 低于 0.22 ≈ 比正常译文短三成以上，基本都是整段漏译。
            if "TRUNCATED" in kinds and plain_en > 200 and plain_cn < 0.22 * plain_en:
                findings.append({"kind": "TRUNCATED", "pack": f.name, "path": p,
                                 "diff": {"纯文本长度 EN/CN": (plain_en, plain_cn),
                                          "长度比": round(plain_cn / plain_en, 2),
                                          "按 0.31 倍中位估算约缺中文字": int(plain_en * 0.31 - plain_cn)},
                                 "en": s[:300], "cn": t[:300]})
            for kind, d in checks.items():
                if d and kind in kinds:
                    findings.append({"kind": kind, "pack": f.name, "path": p, "diff": d,
                                     "en": s[:300], "cn": t[:300]})

    by_kind = Counter(x["kind"] for x in findings)
    for k in ("LINK", "BLOCK", "INLINE", "PLACEHOLDER", "TRUNCATED"):
        if k in kinds:
            print(f"  {k:<12} {by_kind.get(k, 0)} 条")
    by_pack = Counter(x["pack"] for x in findings)
    for pack, n in by_pack.most_common():
        print(f"      {pack:<36} {n}")
    if args.out:
        Path(args.out).write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → {args.out}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
