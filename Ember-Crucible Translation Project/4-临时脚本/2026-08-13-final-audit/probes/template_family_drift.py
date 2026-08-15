# -*- coding: utf-8 -*-
"""只读探针 #3：模板族译文碎片化。

动机：既有的 scan_same_en_split 只在**英文全串完全相同**时才报。
本项目大量条目是「同一句英文模板 + 换一个专名」（词缀、天赋、Bane/Resistance/Spellcraft 族），
英文串因专名不同而互不相等 → 既有判据永远看不到它们，
但中文侧同一模板被译成了七八种句式，玩家在同一个列表里逐条看时会明显觉得不是一套东西。

做法：
1. 把 EN 归一化：所有首字母大写的词 → <X>，数字 → <N>。同 key 的算一族。
2. 对每族 >=4 个成员、EN 归一后长度 >=60 字符的，两两算中文 difflib 相似度。
3. 平均相似度 < 阈值的族报出来，并列出中文侧的句式变体。

假阳性模式：
- 英文模板相同但语义上确实要求不同译法（例：rune / gesture / inflection 三种子类）。
  → 报告里保留原文供人工判断，不自动定性。
- 归一化把不同模板误并成一族（例：两句都以 Increase ... by <N> 开头）。
  → 报告输出 EN 样例，可核。
"""
import re, io, os, sys, json, difflib, collections, itertools

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
PAIRS = [
    (os.path.join(ROOT, "1-Ember汉化插件", "compendium", "en"), os.path.join(ROOT, "1-Ember汉化插件", "compendium", "cn")),
    (os.path.join(ROOT, "2-Crucible汉化插件", "compendium", "en"), os.path.join(ROOT, "2-Crucible汉化插件", "compendium", "cn")),
]
HAN = re.compile(r"[\u4e00-\u9fff]")
TAG = re.compile(r"<[^>]+>")

def walk(obj, path, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, path + [str(k)], out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, path + ["[%d]" % i], out)
    elif isinstance(obj, str):
        out[".".join(path)] = obj

def strip_html(s):
    s = TAG.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def norm_en(s):
    s = strip_html(s)
    s = re.sub(r"@\w+\[[^\]]*\]", "<M>", s)
    s = re.sub(r"\[\[[^\]]*\]\]", "<M>", s)
    s = re.sub(r"\d+", "<N>", s)
    # 首字母大写的词（含连字符/撇号）→ <X>；句首也一并抹掉，避免因句首词不同而分族
    s = re.sub(r"\b[A-Z][A-Za-z'\-]*\b", "<X>", s)
    s = re.sub(r"(<X>[ ]?)+", "<X> ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def main():
    groups = collections.defaultdict(list)
    for endir, cndir in PAIRS:
        for fn in sorted(os.listdir(endir)):
            if not fn.endswith(".json") or fn == "_source.json":
                continue
            cnp = os.path.join(cndir, fn)
            if not os.path.exists(cnp):
                continue
            eo, co = {}, {}
            walk(json.load(io.open(os.path.join(endir, fn), encoding="utf-8")), [], eo)
            walk(json.load(io.open(cnp, encoding="utf-8")), [], co)
            for k, ev in eo.items():
                cv = co.get(k)
                if not cv or not HAN.search(cv):
                    continue
                if len(strip_html(ev)) < 60:
                    continue
                groups[norm_en(ev)].append((fn, k, strip_html(ev), strip_html(cv)))

    rows = []
    for gkey, members in groups.items():
        # 同一叶在孪生包各一份 → 按 (path 尾段) 去重，避免孪生包放大
        seen = {}
        for fn, k, ev, cv in members:
            tail = k
            if tail not in seen:
                seen[tail] = (fn, k, ev, cv)
        uniq = list(seen.values())
        if len(uniq) < 4:
            continue
        cns = [u[3] for u in uniq]
        ens = [u[2] for u in uniq]
        if len(set(ens)) == 1:
            continue  # 英文完全相同 → 既有 scan_same_en_split 已覆盖
        sims = []
        for a, b in itertools.combinations(range(len(cns)), 2):
            sims.append(difflib.SequenceMatcher(None, cns[a], cns[b]).ratio())
            if len(sims) > 400:
                break
        mean = sum(sims) / len(sims)
        rows.append((mean, len(uniq), gkey, uniq))

    rows.sort()
    out = io.open(os.environ.get("OUT", "template_drift.txt"), "w", encoding="utf-8")
    out.write("模板族总数(>=4 成员且 EN 不全同): %d\n" % len(rows))
    for mean, n, gkey, uniq in rows:
        if mean >= 0.90:
            continue
        out.write("\n" + "=" * 90 + "\n")
        out.write("平均中文相似度 %.3f  成员 %d\n模板: %s\n" % (mean, n, gkey[:300]))
        for fn, k, ev, cv in uniq[:60]:
            out.write("  [%s] %s\n     EN: %s\n     CN: %s\n" % (fn, k, ev[:260], cv[:260]))
    out.close()
    print("groups>=4:", len(rows), "flagged<0.90:", sum(1 for r in rows if r[0] < 0.90))

if __name__ == "__main__":
    main()
