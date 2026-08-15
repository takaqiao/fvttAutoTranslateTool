# -*- coding: utf-8 -*-
"""中文排版与全库既有约定不符 —— round8 新判据。

设计原则（本项目铁律的落地）
--------------------------
1. **先查英文再判中文**。库里 90% 的「排版异常」是忠实照抄英文原文的结果
   （`"Phrase 3.` 少一个引号是**英文侧**就少的；`M’abb` 的弯撇号是**英文侧**就弯的；
   `a) b) c)` 的半角括号是英文枚举标号）。所以每条规则要么自带英文闸，
   要么其判定对象是「英文侧本来完全相同、中文侧却分裂成两种写法」的构造。
2. **多数派即约定**，不凭空定标准。每条规则都先普查全库分布，把比例写进 stats，
   少数派只有在「与英文侧同构的多数派直接冲突」时才报。
3. **区分标记区与正文区**。`class="block gamemaster"` 里的半角引号是功能性的，
   `@Embed[... readaloud="中文"]` 引号内又是可译正文。见 `split_regions()`。

六条规则
--------
T1  标记功能区里出现全角/中文字符（**功能性**，最高优先级）
      `@UUID［...］` 用全角方括号会让整个 enricher 失效。
      同时做负向检查：全角 ＠［］｛｝、`[[`/`]]` 与 `<`/`>` 计数与英文侧不等。
T2  中文标签后用半角冒号 `:`（全库 `：` 12844 : 261 ≈ 98%）
T3  中文正文里用半角 `"` 当引号（2026-08-06 决议「引号一律 “”」；
      英文正文含 `"` 的 1485 叶里 1413 叶已转成 “”，只剩 6 叶没转）
T4  「」直角引号（全库 “ 5522 : 「 5）
T5  数字区间用 `—`（全角破折号）当连接号（全库 `-`106 `–`46 `到`30 `至`30 `—`2；
      且英文侧写的是 `1-2`）
T6  成对标点不配平，**且英文侧对应符号是配平的**

用法：
  python scan_cn_typography.py --repo <仓库目录> [--repo <另一个>] [--out x.json] [--show 40]
  仓库目录可以是绝对路径（回测注入用），也可以是项目根下的目录名。
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
SKIP_KEYS = {"_id", "path", "_variants", "_when"}
CJK = "一-鿿㐀-䶿"

# ---------------------------------------------------------------- 遍历
def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            sub = cn.get(k) if isinstance(cn, dict) else None
            walk(v, sub, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        p = ".".join(path)
        out.append({"path": p,
                    "batch_path": p[len("entries."):] if p.startswith("entries.") else p,
                    "en": en, "cn": cn if isinstance(cn, str) else None})


def load_repo(repo):
    d = repo if os.path.isdir(repo) else os.path.join(PROJECT_ROOT, repo)
    en_dir, cn_dir = os.path.join(d, "compendium", "en"), os.path.join(d, "compendium", "cn")
    rows = []
    for fn in sorted(os.listdir(en_dir)):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        en = json.load(open(os.path.join(en_dir, fn), encoding="utf-8-sig"))
        cp = os.path.join(cn_dir, fn)
        cn = json.load(open(cp, encoding="utf-8-sig")) if os.path.isfile(cp) else {}
        sub = []
        # 注意：**整份文档**都要走，不能只走 entries。Crucible 侧 `folders` / `label`
        # 挂在顶层（`folders: {"Melee (1h)": "近战（单手）"}`），pair_dump.py 与既有
        # qa 脚本只走 entries，这批叶子对所有既有判据都是盲区。
        walk(en, cn, [], sub)
        for r in sub:
            r["pack"], r["repo"] = fn, os.path.basename(d.rstrip("\\/"))
        rows.extend(sub)
    return rows


# ------------------------------------------------- 标记区 / 正文区 切分
TAG = re.compile(r"<[^<>]*>")
ROLL = re.compile(r"\[\[[^\[\]]*\]\]")
ENR = re.compile(r"@\w+\[[^\[\]]*\]")
# enricher / 标签内**可译**的属性值：这些引号里的东西是正文，不是功能区
ATTRQ = re.compile(r'(?:readaloud|label|name|alt|title|data-tooltip)\s*=\s*"([^"]*)"')
QUOTED = re.compile(r'"[^"]*"|\'[^\']*\'')


def split_regions(s):
    """-> (功能区字符串, 正文字符串)。正文里保留 `{标签}` 与可译属性值。"""
    marks = []
    for rx in (ROLL, ENR, TAG):
        for m in rx.finditer(s):
            marks.append((m.start(), m.end(), m.group(0)))
    marks.sort()
    merged = []
    for a, b, t in marks:
        if merged and a < merged[-1][1]:
            continue
        merged.append((a, b, t))
    prose, func, pos = [], [], 0
    for a, b, t in merged:
        prose.append(s[pos:a])
        for q in ATTRQ.finditer(t):
            prose.append(q.group(1))
        func.append(QUOTED.sub('""', t))     # 引号内容当作可译，剥掉
        pos = b
    prose.append(s[pos:])
    return "".join(func), "\n".join(p for p in prose if p.strip())


# ---------------------------------------------------------------- 规则
FULLWIDTH_IN_MARKUP = re.compile(
    rf"[{CJK}！-～　-〿‘’“”…—·]")
LABEL_HALF_COLON = re.compile(rf"([{CJK}]{{2,10}}):")
LABEL_FULL_COLON = re.compile(rf"([{CJK}]{{2,10}})：")
HALF_DQ_ON_CJK = re.compile(rf'"(?=[{CJK}])|(?<=[{CJK}])"')
CORNER = re.compile(r"[「」]")
RANGE = re.compile(r"(?<![\d.])(\d{1,4})\s*([-–—~～])\s*(\d{1,4})(?![\d.])")
# 枚举标号 a) / 1) —— 不是括号的一半，配平时要先剔掉；只认「单个字母/数字 + 收括号」
ENUM = re.compile(r"(?<![\w一-鿿’'\-])([a-zA-Z0-9])\s*([)）])(?=[\s一-鿿]|$)")
PAIRS = [("（", "）", "全角括号"), ("“", "”", "全角引号"),
         ("《", "》", "书名号"), ("「", "」", "直角引号"), ("【", "】", "方头括号")]


def ctx(s, i, w=60):
    return s[max(0, i - w): i + w].replace("\n", "⏎")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--out")
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--only", help="只跑某条规则，如 T2")
    a = ap.parse_args()

    rows = []
    for repo in a.repo:
        rows.extend(load_repo(repo))
    rows = [r for r in rows if r["cn"]]

    st = collections.Counter()
    findings = []

    def add(rule, r, en_x, cn_x, why, sev, sug=""):
        findings.append({"rule": rule, "repo": r["repo"], "pack": r["pack"],
                         "path": r["path"], "batch_path": r["batch_path"],
                         "en": en_x, "cn": cn_x, "why": why,
                         "severity": sev, "suggest": sug})

    st["扫描叶数"] = len(rows)
    st["中文总字符"] = sum(len(r["cn"]) for r in rows)

    for r in rows:
        cn_func, cn_prose = split_regions(r["cn"])
        en_func, en_prose = split_regions(r["en"])
        st["标记token数"] += len(TAG.findall(r["cn"])) + len(ENR.findall(r["cn"])) + len(ROLL.findall(r["cn"]))

        # ---- T1 标记功能区里的全角/中文字符（功能性）
        if not a.only or a.only == "T1":
            m = FULLWIDTH_IN_MARKUP.search(cn_func)
            if m:
                st["T1命中"] += 1
                add("T1", r, "", ctx(cn_func, m.start()),
                    f"标记功能区出现全角字符 {m.group(0)!r}，会让 enricher/标签失效", "阻断")
            # 负向：不可能出现在正常内容里的全角标记符
            for ch in "＠［］｛｝＜＞＝":
                if ch in r["cn"]:
                    st["T1全角标记符"] += 1
                    add("T1", r, "", ctx(r["cn"], r["cn"].index(ch)),
                        f"出现全角标记符 {ch!r}", "阻断")
            for o, c in (("[[", "]]"), ("<", ">")):
                if r["cn"].count(o) != r["en"].count(o) or r["cn"].count(c) != r["en"].count(c):
                    st["T1标记数与英文不等"] += 1
                    add("T1", r, f"EN {o}x{r['en'].count(o)} {c}x{r['en'].count(c)}",
                        f"CN {o}x{r['cn'].count(o)} {c}x{r['cn'].count(c)}",
                        "标记符号数量与英文侧不等", "阻断")

        # ---- T2 中文标签后的半角冒号
        if not a.only or a.only == "T2":
            for m in LABEL_FULL_COLON.finditer(cn_prose):
                st["T2全角冒号"] += 1
            hits = list(LABEL_HALF_COLON.finditer(cn_prose))
            if hits:
                st["T2半角冒号"] += len(hits)
                st["T2命中叶"] += 1
                add("T2", r, "", " ｜ ".join(f"…{ctx(cn_prose, h.start(), 24)}…" for h in hits[:6]),
                    f"{len(hits)} 处中文标签后用半角 `:`；全库 `：` 是压倒性多数", "观感",
                    "把这些 `:` 改成 `：`")

        # ---- T3 半角双引号贴着中文
        if not a.only or a.only == "T3":
            hits = list(HALF_DQ_ON_CJK.finditer(cn_prose))
            if '"' in en_prose:
                st["T3英文正文含半角引号的叶"] += 1
                if "“" in cn_prose:
                    st["T3已转成“”的叶"] += 1
            if hits:
                st["T3命中叶"] += 1
                add("T3", r, (en_prose[:0] or "") + ("英文正文用 ASCII \" 引号" if '"' in en_prose else "英文侧无对应引号"),
                    " ｜ ".join(f"…{ctx(cn_prose, h.start(), 34)}…" for h in hits[:4]),
                    "中文正文用半角 \" 当引号，与 2026-08-06「引号一律 “”」决议不符", "一般",
                    "改成 “…”")

        # ---- T4 直角引号
        if not a.only or a.only == "T4":
            hits = list(CORNER.finditer(cn_prose))
            if hits:
                st["T4命中叶"] += 1
                st["T4「」字符"] += len(hits)
                add("T4", r, "", " ｜ ".join(f"…{ctx(cn_prose, h.start(), 26)}…" for h in hits[:4]),
                    "用「」而非 “”，与全库多数及既定决议不符", "一般", "改成 “…”")

        # ---- T5 数字区间连接号
        if not a.only or a.only == "T5":
            for m in RANGE.finditer(cn_prose):
                st["T5区间_" + m.group(2)] += 1
                if m.group(2) == "—":       # em dash
                    en_has = re.search(rf"{m.group(1)}\s*[-–]\s*{m.group(3)}", en_prose)
                    st["T5命中"] += 1
                    add("T5", r, en_has.group(0) if en_has else "(英文侧未直接找到)",
                        ctx(cn_prose, m.start(), 30),
                        "数字区间用了破折号 —（U+2014）；库内区间连接号是 - / – / 到 / 至，"
                        "— 只此一处（含孪生包）", "一般",
                        m.group(0).replace("—", "–"))

        # ---- T6 成对标点配平（英文闸）
        # 不做「枚举标号剔除」这种启发式（`层级 2）` 会被误剔）。改为**只比较中英两侧的
        # 净差值**：英文原文本身就用 `a) b) c)` 枚举、或本身就少一个引号（`"Phrase 3.`），
        # 那么中文照抄出来的净差值与英文相同 —— 相同即放过，只有中文侧**额外**丢了一半才报。
        if not a.only or a.only == "T6":
            cp, ep = cn_prose, en_prose
            cn_par = (cp.count("（") + cp.count("(")) - (cp.count("）") + cp.count(")"))
            en_par = ep.count("(") - ep.count(")")
            if cn_par != en_par:
                # 中文侧配平、英文侧不配平 = 上游英文原文漏了一半括号，译文顺手补齐了。
                # 这不是中文侧的缺陷，单独记数不报。
                if cn_par == 0 and en_par != 0:
                    st["T6英文上游少半个括号_中文已补齐"] += 1
                else:
                    st["T6命中"] += 1
                    add("T6", r, f"EN 净差 {en_par}（(x{ep.count('(')} )x{ep.count(')')}）",
                        f"CN 净差 {cn_par}（（x{cp.count('（')} ）x{cp.count('）')} "
                        f"(x{cp.count('(')} )x{cp.count(')')}）",
                        "中文侧括号不配平（英文侧配平）", "一般")
            if cp.count("“") != cp.count("”"):
                if ep.count('"') % 2 == 1:
                    st["T6英文侧也不配平_跳过"] += 1
                else:
                    st["T6命中"] += 1
                    add("T6", r, f"EN \"x{ep.count(chr(34))}（偶数=配平）",
                        f"“x{cp.count('“')} ”x{cp.count('”')}", "中文引号不配平", "一般")
            for o, c, nm in PAIRS[2:]:
                if cp.count(o) != cp.count(c):
                    st["T6命中"] += 1
                    add("T6", r, "", f"{nm} {o}x{cp.count(o)} {c}x{cp.count(c)}",
                        f"{nm}不配平", "一般")

    # ------------------------------------------------------------ 输出
    print("统计：")
    for k, v in st.most_common():
        print(f"  {k:34s} {v}")
    by = collections.Counter(f["rule"] for f in findings)
    print("\n命中：", dict(by), " 合计", len(findings))
    for f in findings[:a.show]:
        print(f"  [{f['rule']}] {f['pack'][:24]:26s} {f['batch_path'][:58]:60s}")
        print(f"        {f['cn'][:200]}")
    if a.out:
        json.dump({"stats": dict(st), "findings": findings},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
