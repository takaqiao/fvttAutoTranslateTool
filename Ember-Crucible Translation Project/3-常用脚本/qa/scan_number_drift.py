# -*- coding: utf-8 -*-
"""数字漂移闸：上游把**数字**改了，中文还停在旧数字上。

    python scan_number_drift.py --repo <repo> --baseline <旧英文基准目录> [--out <json>]
                                [--md <md>] [--show 30] [--include-missing]

为什么单独设这一闸（§8 2026-08-13k 判据的结构盲区之二）
--------------------------------------------------------
`scan_en_drift.py` 只回答「这一条英文动过没有」，接着用的分诊判据是
**「新英文独有的 token 在不在中文里」**。那条判据看得见「上游加了词」，
但两类改动它结构上就看不见：

* **删除型** —— 上游删掉一个词（真缺陷：`Ability, Skill, and Talent` → `Ability and
  Talent`，中文还写着「技能」）。新英文没有任何独有 token，判据全程沉默。
  → 那一类归 `scan_dropped_terms.py`。
* **数字型** —— `30` 改成 `20`、`/2` 改成 `/4`、`2` 改成 `3`。数字不是「新词」，
  分词后也常被当噪声滤掉；而它恰恰是**玩家照着算、算错就出错**的那一档。
  → 本脚本。

判据
----
对**基准与当前英文里路径相同、且有中文**的每一条：

1. 先抹掉机械部分（`@X[…]` 方括号内部、`[[…]]`、HTML 标签与属性、图片路径），
   只留玩家读到的散文 —— 标记内部的数字归 `scan_markup_drift.py` 的 LINK 档管，
   在这里数会两边重复报。
2. 把英文数词折算成阿拉伯数字（`thirty`→30、`one hundred`→100）。
   **这是本闸最重要的一条降噪**：上一轮 4 条告警里有 2 条就是上游把 `30/50/100`
   改写成 `thirty/fifty/one hundred`，中文照旧写阿拉伯数字，语义完全等价。
3. 取**集合**（不是多重集）差：`gone = 旧 - 新`、`added = 新 - 旧`。
   用集合是因为「同一个数字在文中出现三次、上游删掉其中一次」不是数字改动。
4. 报 `STALE_NUMBER`：`gone` 里有数字**仍出现在中文里**，且 `added` 里至少有一个
   数字**没出现在中文里**。两个条件都要满足才报 —— 只有前者时，中文很可能是在
   讲另一件事时用到了同一个数字。
5. `--include-missing` 才报的弱档 `MISSING_NUMBER`：`gone` 为空（纯新增数字），
   而新增的数字一个都没进中文。这一档假阳性高（英文补一句带数字的话、中文合并
   进上文很常见），默认不报。

已知假阳性 / 已做的规避
----------------------
* **英文数词 ↔ 阿拉伯数字**：已折算（第 2 步）。仍可能漏的写法：`a dozen`、
  `half`、序数词 `third`、罗马数字 `IV`。这些当前一律不识别，不会误报，只会漏。
* **中文汉字数字**：中文写「三」而英文写 `3` 时，判 `added` 是否进了中文会误判。
  已做保守折算（`三十`→30、`一百`→100、孤立的 `二`–`九`），但**故意不折算孤立的
  「一」「两」**（「一个」「两次」在中文散文里太常见，折算会凭空造出数字 1/2）。
  代价是「英文 1→2、中文写『一』改成『两』」这一类会漏报。
* **标记内部的数字**（`@Check[dc:14]`、`[[/r 2d6]]`、`ember-0.6.0.webp`）：已整段抹掉，
  本闸不管；那是 `scan_markup_drift.py` LINK 档的定义域。
* **表格里的整列数字重排**：中文跟着重排时两边集合相同，不报；只改一格才报，正确。
* 中文里的**年份 / 版本号 / 页码**：抹不掉，若与 `gone` 撞上会误报，逐条人核即可。

回测（2026-08-15，`4-临时脚本/2026-08-15-round16/qa/backtest_number_drift.py`）
--------------------------------------------------------------------------
双向 6/6 PASS。灵敏度 3/3：`30→20` 数值改小、`Armor/2→Armor/4` 分数形式、
`2→3` 点数序号，中文一律停在旧数字，全部报出。
特异度 3/3 静默：`30/50/100`→`thirty/fifty/one hundred`（数词折算）、
同一数字出现两次删掉一次（用集合而非多重集）、只有 `@Check[dc:14→18]` 与
图片路径里的数字变了（机械部分已抹）。

当前库基线（2026-08-15）
------------------------
* crucible（基准 crucible-0.9.1-legacy）：比对 3740 条，英文数字变过 43 条，
  **STALE_NUMBER 0**。
* ember（基准 ember-cn-v1.0.15-shipped-en）：比对 9214 条，英文数字变过 345 条，
  **STALE_NUMBER 0**。
* 弱档 `--include-missing` 共 22 条，抽核后**全部**是同一个良性形态：
  英文写 `one` / `two`，中文写「一次」「两次」，而本脚本故意不折算孤立的
  「一」「两」。这印证了该档默认关闭是对的。
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CJK = re.compile(r"[一-鿿]")

# --------------------------------------------------------------------- 机械部分
# 方括号机械：`@UUID[...]`、`&Reference[...]`、`[[/check ...]]`。尾随 `{标签}` 是
# 给玩家看的散文，留着（标签里的数字确实该跟着改）。
MARKUP = re.compile(r"@[A-Za-z]+\[[^\]]*\]|&(?:amp;)?[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]")
HTML_TAG = re.compile(r"<[^>]+>")          # 标签连同 class/id/src 一起没了
ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")


def strip_machinery(s: str) -> str:
    return ENTITY.sub(" ", HTML_TAG.sub(" ", MARKUP.sub(" ", s)))


# --------------------------------------------------------------------- 英文数词
_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
          "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000}
_ALL_WORDS = {**_UNITS, **_TENS, **_SCALES}
_WORD_SEQ = re.compile(
    r"\b(?:" + "|".join(sorted(_ALL_WORDS, key=len, reverse=True)) + r")"
    r"(?:[-\s]+(?:and[-\s]+)?(?:" + "|".join(sorted(_ALL_WORDS, key=len, reverse=True)) + r"))*\b",
    re.I)


def _phrase_value(phrase: str) -> int | None:
    """`one hundred twenty-five` → 125。看不懂就返回 None（宁可漏，别乱造数字）。"""
    total, cur, seen = 0, 0, False
    for w in re.split(r"[-\s]+", phrase.lower()):
        if w == "and":
            continue
        if w in _UNITS:
            cur += _UNITS[w]
            seen = True
        elif w in _TENS:
            cur += _TENS[w]
            seen = True
        elif w in _SCALES:
            cur = (cur or 1) * _SCALES[w]
            if _SCALES[w] >= 1000:
                total += cur
                cur = 0
            seen = True
        else:
            return None
    return (total + cur) if seen else None


def en_words_to_digits(s: str) -> str:
    """把英文数词折算成阿拉伯数字，好和中文里的写法对齐。"""
    def sub(m):
        v = _phrase_value(m.group(0))
        return f" {v} " if v is not None else m.group(0)
    return _WORD_SEQ.sub(sub, s)


# --------------------------------------------------------------------- 汉字数字
_CJK_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
_CJK_UNIT = {"十": 10, "百": 100, "千": 1000}
# 孤立的「一」「两」在中文散文里到处都是（一个 / 两次），折算它们会凭空造出 1 和 2。
_CJK_SOLO_SKIP = {"一", "两"}
_CJK_RUN = re.compile(r"[零一二两三四五六七八九十百千]+")


def cjk_numbers(s: str) -> set[str]:
    """中文里写成汉字的数字（保守版，只认得懂的形式）。"""
    out = set()
    for run in _CJK_RUN.findall(s):
        if len(run) == 1:
            if run in _CJK_SOLO_SKIP:
                continue
            if run in _CJK_DIGIT:
                out.add(str(_CJK_DIGIT[run]))
            elif run in _CJK_UNIT:
                out.add(str(_CJK_UNIT[run]))
            continue
        total, cur, ok = 0, 0, True
        for ch in run:
            if ch in _CJK_DIGIT:
                cur = _CJK_DIGIT[ch]
            elif ch in _CJK_UNIT:
                total += (cur or 1) * _CJK_UNIT[ch]
                cur = 0
            else:
                ok = False
                break
        if ok:
            out.add(str(total + cur))
    return out


NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers(s: str) -> set[str]:
    """散文里的阿拉伯数字。`3.0` 与 `3` 视为不同；`30` 与 `030` 归一。"""
    out = set()
    for tok in NUM.findall(s):
        if "." in tok:
            out.add(str(float(tok)))
        else:
            out.add(str(int(tok)))
    return out


# --------------------------------------------------------------------- 载入
def load_json(path):
    """旧基准里混着 BOM 和一处尾逗号（当年那个损坏的 -en.json），都兜住。"""
    raw = open(path, encoding="utf-8-sig").read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))


def leaves(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out[".".join(path)] = node


def baseline_packs(bdir):
    """与 scan_en_drift.py 同一套映射：`_repaired.json` 先到先得，别让损坏那份盖上来。"""
    out = {}
    for f in sorted(os.listdir(bdir)):
        if not f.endswith(".json") or f == "_source.json":
            continue
        key = ("ember.crucible-adventure.json" if f == "_repaired.json"
               else f.replace("-en.json", ".json"))
        out.setdefault(key, os.path.join(bdir, f))
    return out


# --------------------------------------------------------------------- 主流程
def scan(repo, baseline, include_missing=False):
    en_dir = os.path.join(repo, "compendium", "en")
    cn_dir = os.path.join(repo, "compendium", "cn")
    findings = []
    stats = collections.Counter()

    for pack, oldpath in baseline_packs(baseline).items():
        cur_p = os.path.join(en_dir, pack)
        if not os.path.exists(cur_p):
            continue
        o, n, c = {}, {}, {}
        leaves(load_json(oldpath).get("entries", {}), [], o)
        leaves(load_json(cur_p).get("entries", {}), [], n)
        cnp = os.path.join(cn_dir, pack)
        if os.path.exists(cnp):
            leaves(load_json(cnp).get("entries", {}), [], c)

        for path, new_en in n.items():
            old_en = o.get(path)
            cn = c.get(path)
            if old_en is None or not cn or not CJK.search(cn):
                continue
            stats["pairs"] += 1
            old_p = en_words_to_digits(strip_machinery(old_en))
            new_p = en_words_to_digits(strip_machinery(new_en))
            old_nums, new_nums = numbers(old_p), numbers(new_p)
            if old_nums == new_nums:
                continue
            stats["number_changed"] += 1
            gone, added = old_nums - new_nums, new_nums - old_nums
            cn_plain = strip_machinery(cn)
            cn_strict = numbers(cn_plain)                 # 只认阿拉伯数字
            cn_loose = cn_strict | cjk_numbers(cn_plain)  # 再加保守折算的汉字数字

            kept = sorted(gone & cn_strict, key=float)
            not_followed = sorted(added - cn_loose, key=float)
            if kept and not_followed:
                verdict = "STALE_NUMBER"
            elif include_missing and not gone and added and not (added & cn_loose):
                verdict = "MISSING_NUMBER"
                kept, not_followed = [], sorted(added, key=float)
            else:
                stats["benign"] += 1
                continue
            stats[verdict] += 1
            findings.append({
                "verdict": verdict, "repo": os.path.basename(repo), "pack": pack,
                "path": path,
                "en_gone": sorted(gone, key=float), "en_added": sorted(added, key=float),
                "cn_still_has_old": kept, "cn_missing_new": not_followed,
                "old_en": old_en[:300], "new_en": new_en[:300], "cn": cn[:300],
            })
    return findings, stats


def main():
    ap = argparse.ArgumentParser(description="数字漂移闸：英文数字变了、中文没跟上")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out")
    ap.add_argument("--md")
    ap.add_argument("--show", type=int, default=30)
    ap.add_argument("--include-missing", action="store_true",
                    help="连弱档 MISSING_NUMBER 一起报（假阳性高，默认关）")
    a = ap.parse_args()

    findings, stats = scan(a.repo, a.baseline, a.include_missing)
    print(f"{os.path.basename(a.repo)}  基准 {os.path.basename(a.baseline)}")
    print(f"  两边都有且有中文的条目 {stats['pairs']}  ·  英文数字变过 {stats['number_changed']}")
    print(f"  **STALE_NUMBER {stats['STALE_NUMBER']}**"
          f"  MISSING_NUMBER {stats['MISSING_NUMBER']}  良性 {stats['benign']}")
    for f in findings[:a.show]:
        print(f"  [{f['verdict']}] {f['pack']} :: {f['path'][-70:]}")
        print(f"     英文 去掉 {f['en_gone']} → 换成 {f['en_added']}；"
              f"中文还留着 {f['cn_still_has_old']}，没跟上 {f['cn_missing_new']}")
    if a.out:
        json.dump({"stats": dict(stats), "findings": findings},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  -> {a.out}")
    if a.md:
        with open(a.md, "w", encoding="utf-8") as fh:
            fh.write(f"# 数字漂移（scan_number_drift.py）— {os.path.basename(a.repo)}\n\n")
            fh.write(f"- 比对条目 {stats['pairs']}，英文数字变过 {stats['number_changed']}，"
                     f"**告警 {len(findings)}**\n\n")
            for f in findings:
                fh.write(f"## `{f['pack']}` `{f['path']}`  ({f['verdict']})\n\n"
                         f"- 英文 `{f['en_gone']}` → `{f['en_added']}`\n"
                         f"- 中文还留着 `{f['cn_still_has_old']}`，没跟上 `{f['cn_missing_new']}`\n"
                         f"- 旧英文：{f['old_en']}\n- 新英文：{f['new_en']}\n- 中文：{f['cn']}\n\n")
        print(f"  -> {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
