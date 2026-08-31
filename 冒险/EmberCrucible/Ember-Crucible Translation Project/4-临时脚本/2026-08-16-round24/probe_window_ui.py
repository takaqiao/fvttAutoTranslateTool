# -*- coding: utf-8 -*-
"""
第二十四轮 · ① EMBER_WINDOW_UI 键活性复核探针

面板 D 档报的「上游查无此串」只以 **.mjs** 为语料（见 ember-cn-selfcheck.mjs:577-586：
只抓 modules/ember/scripts/ember.mjs + systems/<sys>/<sys>-compiled.mjs）。
本探针把语料扩到上游**真实产出这些串的地方**，逐条定性。

语料分层：
  A  ember.mjs                       ← 判据现在唯一抓的 ember 侧语料
  B  crucible-compiled.mjs           ← 判据在 crucible 世界下会抓
  C  modules/ember/scripts/*.mjs 其余（crucible-async / dnd5e-async）← 判据**没抓**
  D  modules/ember/templates/**/*.hbs                                ← 判据**没抓**
  E  modules/ember/lang/en.json                                      ← 判据**没抓**
  F  systems/crucible/templates/**  + crucible lang/en.json          ← 判据**没抓**

落盘再跑（禁止 heredoc）。
"""
import json, os, re, sys, io

ROOT_EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
ROOT_CRUC  = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible"
HARDCODED  = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"
OUTDIR     = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-16-round24"


def rd(p):
    with io.open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def walk(root, exts):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in exts:
                out.append(os.path.join(dp, fn))
    return out


# ---------- 1. 抽出 EMBER_WINDOW_UI 的键 ----------
src = rd(HARDCODED)
m = re.search(r"const EMBER_WINDOW_UI = \{(.*?)\n\};", src, re.S)
assert m, "没找到 EMBER_WINDOW_UI"
body = m.group(1)


def strip_line_comments(s):
    """去掉 // 行注释，但不动字符串内部的 //。逐字符状态机。"""
    out = []
    i, n = 0, len(s)
    instr = False
    while i < n:
        c = s[i]
        if instr:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(s[i + 1]); i += 2; continue
            if c == '"':
                instr = False
            i += 1; continue
        if c == '"':
            instr = True; out.append(c); i += 1; continue
        if c == "/" and i + 1 < n and s[i + 1] == "/":
            j = s.find("\n", i)
            if j < 0:
                break
            i = j; continue
        out.append(c); i += 1
    return "".join(out)


clean = strip_line_comments(body)
# 键 = 位于 { 或 , 之后（跳过空白/换行）的 "..." 且后面紧跟 :
keys = []
for km in re.finditer(r'"((?:[^"\\]|\\.)*)"\s*:', clean):
    k = km.group(1)
    prev = clean[:km.start()].rstrip()
    if prev and prev[-1] not in "{,":
        continue
    k = k.encode().decode("unicode_escape") if "\\" in k else k
    if k not in keys:
        keys.append(k)

# ---------- 2. 建语料 ----------
corp = {}
corp["A_ember.mjs"] = rd(os.path.join(ROOT_EMBER, "scripts", "ember.mjs"))
corp["B_crucible-compiled.mjs"] = rd(os.path.join(ROOT_CRUC, "crucible-compiled.mjs"))

other_mjs = {}
for p in walk(os.path.join(ROOT_EMBER, "scripts"), {".mjs", ".js"}):
    if os.path.basename(p) == "ember.mjs":
        continue
    other_mjs[p] = rd(p)
corp["C_ember_other_scripts"] = "\n".join(other_mjs.values())

hbs = {}
for p in walk(os.path.join(ROOT_EMBER, "templates"), {".hbs", ".html", ".handlebars"}):
    hbs[p] = rd(p)
corp["D_ember_templates"] = "\n".join(hbs.values())

lang_en = rd(os.path.join(ROOT_EMBER, "lang", "en.json"))
corp["E_ember_lang_en"] = lang_en

cruc_tpl = {}
for p in walk(os.path.join(ROOT_CRUC, "templates"), {".hbs", ".html", ".handlebars"}):
    cruc_tpl[p] = rd(p)
cruc_lang = ""
for p in walk(os.path.join(ROOT_CRUC, "lang"), {".json"}):
    cruc_lang += rd(p)
corp["F_crucible_templates_lang"] = "\n".join(cruc_tpl.values()) + "\n" + cruc_lang

# 判据当前实际用的语料（crucible 世界）
judge_corpus = corp["A_ember.mjs"] + "\n" + corp["B_crucible-compiled.mjs"]

# ---------- 3. 逐键定位 ----------
def locate(text_map, key):
    """返回 [(文件, 行号, 行内容片段)]"""
    hits = []
    for p, t in text_map.items():
        idx = t.find(key)
        if idx < 0:
            continue
        line = t.count("\n", 0, idx) + 1
        ls = t.rfind("\n", 0, idx) + 1
        le = t.find("\n", idx)
        if le < 0:
            le = len(t)
        hits.append((p, line, t[ls:le].strip()[:300]))
    return hits


# lang/en.json 值层命中
lang_obj = json.loads(lang_en)
def flat(o, pre=""):
    for k, v in o.items():
        kk = pre + "." + k if pre else k
        if isinstance(v, dict):
            for x in flat(v, kk):
                yield x
        else:
            yield kk, v
lang_pairs = list(flat(lang_obj))

records = []
for k in keys:
    inA = k in corp["A_ember.mjs"]
    inB = k in corp["B_crucible-compiled.mjs"]
    inC = k in corp["C_ember_other_scripts"]
    inD = k in corp["D_ember_templates"]
    inE = k in corp["E_ember_lang_en"]
    inF = k in corp["F_crucible_templates_lang"]
    judged_missing = k not in judge_corpus

    rec = {
        "key": k,
        "judge_reports_missing": judged_missing,
        "in": {"ember.mjs": inA, "crucible-compiled.mjs": inB,
               "ember_other_scripts": inC, "ember_templates": inD,
               "ember_lang_en": inE, "crucible_templates_lang": inF},
        "hbs_hits": [],
        "other_mjs_hits": [],
        "lang_keys_with_this_value": [vk for vk, vv in lang_pairs if vv == k],
    }
    if inD:
        for p, ln, txt in locate(hbs, k):
            rec["hbs_hits"].append({
                "file": os.path.relpath(p, ROOT_EMBER).replace("\\", "/"),
                "line": ln, "text": txt})
    if inC:
        for p, ln, txt in locate(other_mjs, k):
            rec["other_mjs_hits"].append({
                "file": os.path.relpath(p, ROOT_EMBER).replace("\\", "/"),
                "line": ln, "text": txt})
    if inF:
        for p, ln, txt in locate(cruc_tpl, k):
            rec["crucible_hbs_hits"] = rec.get("crucible_hbs_hits", []) + [{
                "file": os.path.relpath(p, ROOT_CRUC).replace("\\", "/"),
                "line": ln, "text": txt}]
    records.append(rec)

miss = [r for r in records if r["judge_reports_missing"]]
print("EMBER_WINDOW_UI 键总数 =", len(keys))
print("判据（只抓 .mjs）会报缺 =", len(miss))
print()
buckets = {}
for r in miss:
    tags = [n for n, v in r["in"].items() if v] or ["NOWHERE"]
    buckets.setdefault("+".join(tags), []).append(r["key"])
for b, ks in sorted(buckets.items(), key=lambda x: -len(x[1])):
    print("[%-45s] %2d" % (b, len(ks)))
    for k in ks:
        print("      -", k[:90])
print()

with io.open(os.path.join(OUTDIR, "probe_window_ui_raw.json"), "w", encoding="utf-8") as f:
    json.dump({"total_keys": len(keys), "keys": keys, "records": records},
              f, ensure_ascii=False, indent=2)
print("raw ->", os.path.join(OUTDIR, "probe_window_ui_raw.json"))
