# -*- coding: utf-8 -*-
"""
探针：闸/选择器失配（enricher / 运行时替换表）

判据（可机械化）：
  对每一个注册进 CONFIG.TextEditor.enrichers 的富文本增强器：
    1. 静态抽出它**实际写进 DOM 的英文串**（innerHTML / innerText / textContent /
       dataset.tooltip / setAttribute("aria-label"|"title") 的模板字面量），
       排除全部走 _loc(...) 的（那些归 i18n 管）。
    2. 判断该增强器是否被 ember 插件的包装闸命中
       （闸 = /attunement|language|knowledge|soundscape|ancestry|culture|path|
              eventState|outcome|Advantage|Critical|date/i 对 String(entry.pattern)）。
    3. 对每一个英文串形状，判断 translateText()（EXACT / PREFIXED / PATTERNS）
       能否命中。
    4. 统计该增强器语法在已发布中文语料里的出现次数（= 玩家实际会看到多少处）。

  命中 = 落在闸外，或落在闸内但形状不被任何表命中，且语料计数 > 0 →候选缺陷。

已知假阳性模式：
  - 静态抽取只看模板字面量，若英文串由变量拼出（如 label 变量在别处赋值）会漏（假阴）。
  - 若某形状实际由世界数据（page.name / index.name，babele 已翻）填充，
    表面看是英文模板但运行时是中文 → 假阳，必须人工核实每个插值来源。
  - 语料计数按 [[/xxx 或 @Xxx[ 直接数，未排除代码块 / 未渲染上下文。
只读，不写库。
"""
import json
import os
import re
import sys

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER_UP = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts"
CRUC_UP = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\crucible-compiled.mjs"

GATE = re.compile(
    r"attunement|language|knowledge|soundscape|ancestry|culture|path|"
    r"eventState|outcome|Advantage|Critical|date", re.I)

# 从 ember-hardcoded-cn.mjs 里抽 EXACT / PREFIXED 前缀 / PATTERNS
HC = os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")


def load_tables():
    src = open(HC, encoding="utf-8").read()
    exact = set(re.findall(r'^\s*"([^"]+)":\s*"', src, re.M))
    prefixes = re.findall(r'\{\s*en:\s*"([^"]+)"', src)
    patterns = re.findall(r'\{\s*re:\s*/\^([^/]+)/', src)
    return exact, prefixes, patterns


def covered(shape, exact, prefixes, patterns):
    """shape 是带 {} 占位的模板串；判断 translateText 能否命中它。"""
    if shape in exact:
        return "EXACT"
    for p in prefixes:
        if shape.startswith(p + ": "):
            return "PREFIXED:" + p
    for p in patterns:
        # 把模板占位换成宽松通配再拿正则试
        probe = shape.replace("{}", "X")
        try:
            if re.match(p.replace("(.+)", "(.+)"), probe):
                return "PATTERN:" + p
        except re.error:
            pass
    return None


# ---- 1. 抽增强器注册表 -------------------------------------------------
ENTRY_RE = re.compile(
    r"\{\s*(?:(?://[^\n]*\n)\s*)?(?:id:\s*\"(?P<id>[^\"]+)\",\s*)?"
    r"pattern:\s*(?P<pat>/(?:\\.|[^/\\])+/[gimsuy]*),\s*"
    r"enricher:\s*(?P<fn>[\w$.]+)")


def enrichers_from(path, label):
    src = open(path, encoding="utf-8").read()
    out = []
    for m in ENTRY_RE.finditer(src):
        out.append({"src_file": label, "id": m.group("id"),
                    "pattern": m.group("pat"), "fn": m.group("fn")})
    return out, src


# ---- 2. 抽函数体里的英文输出 ------------------------------------------
SINK_RE = re.compile(
    r"(?:innerHTML|innerText|textContent)\s*\+?=\s*(?P<v>`[^`]*`|\"[^\"]*\")"
    r"|dataset\.tooltip\s*=\s*(?P<v2>`[^`]*`|\"[^\"]*\")"
    r"|setAttribute\(\s*\"(?:aria-label|title)\"\s*,\s*(?P<v3>`[^`]*`|\"[^\"]*\")"
    r"|tooltipText:\s*(?P<v4>`[^`]*`|\"[^\"]*\")"
    r"|label\s*=\s*(?P<v5>`[^`]*`|\"[^\"]*\")")

ASCII_WORD = re.compile(r"[A-Za-z]{3,}")


def fn_body(src, name):
    """取 `function name(` 到下一个顶层 `\n}` 的片段（粗糙但够用）。"""
    base = name.split(".")[0]
    m = re.search(r"(?:async\s+)?function\s+" + re.escape(base) + r"\s*\(", src)
    if not m:
        # 静态方法 EmberSoundscape.enricherHTML 之类
        parts = name.split(".")
        if len(parts) == 2:
            m = re.search(r"\n\s*(?:static\s+)?" + re.escape(parts[1]) + r"\s*\(", src)
        if not m:
            return None
    i = src.index("{", m.end() - 1)
    depth = 0
    for j in range(i, min(len(src), i + 20000)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i:j + 1]
    return src[i:i + 20000]


def shapes(body):
    out = []
    for m in SINK_RE.finditer(body):
        v = next(g for g in m.groups() if g is not None)
        lit = v[1:-1]
        if "_loc(" in lit or "game.i18n" in lit:
            continue
        norm = re.sub(r"\$\{[^}]*\}", "{}", lit)
        stripped = re.sub(r"<[^>]+>", "", norm)
        if not ASCII_WORD.search(stripped):
            continue
        out.append(stripped.strip())
    return out


# ---- 3. 语料计数 -------------------------------------------------------
def corpus_count(syntax_regex):
    total = {}
    for repo, sub in (("ember", os.path.join(ROOT, "1-Ember汉化插件", "compendium", "cn")),
                      ("crucible", os.path.join(ROOT, "2-Crucible汉化插件", "compendium", "cn"))):
        n = 0
        for f in os.listdir(sub):
            if not f.endswith(".json"):
                continue
            txt = open(os.path.join(sub, f), encoding="utf-8").read()
            n += len(syntax_regex.findall(txt))
        total[repo] = n
    return total


def syntax_probe(pattern_src):
    """把 JS 正则字面量转成一个够用的 python 正则，用来数语料。"""
    body = pattern_src.rsplit("/", 1)[0][1:]
    # 只取开头的固定前缀，避免 JS/py 正则方言差异
    m = re.match(r"((?:\\\[){0,2}(?:\\/)?[@\w]+)", body)
    if not m:
        return None
    lit = m.group(1).replace("\\[", "[").replace("\\/", "/")
    return re.compile(re.escape(lit))


def main():
    exact, prefixes, patterns = load_tables()
    rows = []
    for path, label in ((os.path.join(EMBER_UP, "ember.mjs"), "ember.mjs"),
                        (CRUC_UP, "crucible-compiled.mjs")):
        entries, src = enrichers_from(path, label)
        for e in entries:
            gated = bool(GATE.search(e["pattern"]))
            body = fn_body(src, e["fn"])
            sh = shapes(body) if body else []
            probe = syntax_probe(e["pattern"])
            cnt = corpus_count(probe) if probe else {}
            miss = [s for s in sh if not covered(s, exact, prefixes, patterns)]
            rows.append({
                "file": label, "id": e["id"], "fn": e["fn"],
                "pattern": e["pattern"], "wrapped_by_gate": gated,
                "hardcoded_en_shapes": sh, "uncovered_shapes": miss,
                "corpus": cnt,
            })
    json.dump(rows, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
