# -*- coding: utf-8 -*-
"""
抽取上游 ember 0.6.1 里**会上屏**的英文串。

口径（写死在这里，改了就得重跑前置自证）：
  A. templates/**.hbs/.html
     A1 文本节点：先剥 <!-- -->、<script>/<style>、handlebars {{...}}，再剥标签，
        剩下的文本按行切，取含字母的片段。
     A2 属性字面量：data-tooltip / data-tooltip-text / data-tooltip-html / title /
        aria-label / placeholder / alt / label —— 值里含 {{ 的**整条跳过**（那是数据不是字面量）。
  B. scripts/*.mjs（ember.mjs / crucible-async.mjs / dnd5e-async.mjs）
     B1 上屏键的字符串字面量：label / title / hint / placeholder / tooltip / prompt /
        legend / caption / header / buttonLabel / emptyLabel / text
     B2 `.textContent = "..."` 一族
     B3 ui.notifications.{notify,info,warn,error}("...")
     B4 DialogV2 的 content 内联 HTML（剥标签后取文本）

前置自证（--selfcheck）：阳性对照 = 一组已知会上屏的串必须被抽到且出处对得上；
阴性对照 = 一组纯代码标识符 / 路径 / 选择器必须**不**被抽到。
"""
import json, re, sys, os, argparse

UI_ATTRS = ["data-tooltip", "data-tooltip-text", "data-tooltip-html", "title",
            "aria-label", "placeholder", "alt", "label"]
UI_KEYS = ["label", "title", "hint", "placeholder", "tooltip", "prompt", "legend",
           "caption", "header", "buttonLabel", "emptyLabel", "text"]

_IDENT = re.compile(r'^[A-Za-z_$][A-Za-z0-9_$]*$')
_PATH = re.compile(r'[/\\]')
_SELECTOR = re.compile(r'^[.#\[]|[{}<>]|::')
_HASLETTER = re.compile(r'[A-Za-z]')
_FA = re.compile(r'^fa[srlbd]?[- ]')
_DOTTED = re.compile(r'^[A-Za-z0-9_$]+(\.[A-Za-z0-9_$]+)+$')
_HEX = re.compile(r'^#[0-9a-fA-F]{3,8}$')


def looks_ui(s):
    s = s.strip()
    if not s or not _HASLETTER.search(s):
        return False
    if len(s) > 400:
        return False
    if _PATH.search(s):
        return False
    if _FA.match(s):
        return False
    if _HEX.match(s):
        return False
    if _DOTTED.match(s):
        return False
    if _SELECTOR.search(s):
        return False
    if _IDENT.match(s):
        if not s[0].isupper():
            return False
        if s.isupper() and len(s) > 3:
            return False
    return True


_HB = re.compile(r'\{\{[^}]*\}\}', re.S)
_CMT = re.compile(r'<!--.*?-->', re.S)
_SCR = re.compile(r'<(script|style)\b.*?</\1>', re.S | re.I)
_TAG = re.compile(r'<[^>]*>', re.S)


def scan_template(path, text, out):
    for m in re.finditer(r'<[^>]*>', text, re.S):
        tag = m.group(0)
        for attr in UI_ATTRS:
            for am in re.finditer(r'\b' + re.escape(attr) + r'\s*=\s*"([^"]*)"', tag):
                v = am.group(1)
                if '{{' in v:
                    continue
                if looks_ui(v):
                    out.append({"s": v.strip(), "src": path, "how": "attr:" + attr})
    t = _CMT.sub(' ', text)
    t = _SCR.sub(' ', t)
    t = _HB.sub('\x00', t)
    t = _TAG.sub('\x00', t)
    for chunk in t.split('\x00'):
        for line in chunk.splitlines():
            v = line.strip()
            if looks_ui(v):
                out.append({"s": v, "src": path, "how": "text"})


_STR = r'"((?:[^"\\\n]|\\.)*)"'


def _unesc(s):
    return s.replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')


def scan_script(path, text, out):
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        for k in UI_KEYS:
            for m in re.finditer(r'(?<![A-Za-z0-9_$])' + k + r'\s*:\s*' + _STR, line):
                v = _unesc(m.group(1))
                if looks_ui(v):
                    out.append({"s": v.strip(), "src": path + ":" + str(i), "how": "key:" + k})
        for m in re.finditer(r'\.(textContent|innerText|placeholder|title|label|ariaLabel)\s*=\s*' + _STR, line):
            v = _unesc(m.group(2))
            if looks_ui(v):
                out.append({"s": v.strip(), "src": path + ":" + str(i), "how": "assign:" + m.group(1)})
        for m in re.finditer(r'ui\.notifications\.(?:notify|info|warn|error)\(\s*' + _STR, line):
            v = _unesc(m.group(1))
            if looks_ui(v):
                out.append({"s": v.strip(), "src": path + ":" + str(i), "how": "notify"})
        for m in re.finditer(r'content\s*:\s*(?:`([^`]*)`|' + _STR + r')', line):
            raw = m.group(1) if m.group(1) is not None else _unesc(m.group(2) or '')
            if not raw:
                continue
            raw = re.sub(r'\$\{[^}]*\}', '\x00', raw)
            for chunk in _TAG.sub('\x00', raw).split('\x00'):
                v = chunk.strip()
                if looks_ui(v):
                    out.append({"s": v, "src": path + ":" + str(i), "how": "content"})


POSITIVE = [
    ("Ember Token Maker", "ember.mjs"),
    ("Show Tracks", "ember.mjs"),
    ("Mine Cart Destination", "ember.mjs"),
    ("Loading Zone", "ember.mjs"),
    ("Hair Roots", "ember.mjs"),
    ("Reset All", "body.hbs"),
    ("Layers", "layers.hbs"),
    ("Previous Option", "layers.hbs"),
    ("Character Name", "header.hbs"),
    ("Event Flowchart", "flowchart-view.hbs"),
]
NEGATIVE = [
    "modules/ember/templates/applications/token-maker/layers.hbs",
    "fa-solid fa-caret-left",
    "token-maker-layers flexcol scrollable",
    "buildPrevious",
    "#660000",
    "ember.trapTrigger",
    "emberVistaArctur",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    out = []
    tdir = os.path.join(a.root, "templates")
    for dirpath, _, files in os.walk(tdir):
        for f in sorted(files):
            if not f.endswith((".hbs", ".html")):
                continue
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, a.root).replace("\\", "/")
            with open(p, encoding="utf-8") as fh:
                scan_template(rel, fh.read(), out)
    for f in ["ember.mjs", "crucible-async.mjs", "dnd5e-async.mjs"]:
        p = os.path.join(a.root, "scripts", f)
        with open(p, encoding="utf-8") as fh:
            scan_script("scripts/" + f, fh.read(), out)

    agg = {}
    for r in out:
        e = agg.setdefault(r["s"], {"s": r["s"], "srcs": [], "hows": set(), "n": 0})
        e["n"] += 1
        if len(e["srcs"]) < 4:
            e["srcs"].append(r["src"])
        e["hows"].add(r["how"])
    rows = [{"s": v["s"], "srcs": v["srcs"], "hows": sorted(v["hows"]), "n": v["n"]}
            for v in agg.values()]
    rows.sort(key=lambda r: r["s"])

    if a.selfcheck:
        seen = {r["s"]: r for r in rows}
        bad = []
        for s, srchint in POSITIVE:
            if s not in seen:
                bad.append("阳性对照没抽到：%r" % s)
            elif not any(srchint in x for x in seen[s]["srcs"]):
                bad.append("阳性对照出处不符：%r 期望含 %r，实得 %s" % (s, srchint, seen[s]["srcs"]))
        for s in NEGATIVE:
            if s in seen:
                bad.append("阽性对照被抽到了：%r ← %s" % (s, seen[s]["srcs"]))
        if bad:
            print("前置自证失败：")
            for b in bad:
                print("  " + b)
            sys.exit(3)
        print("前置自证通过：阳性 %d/%d，阴性 %d/%d" % (len(POSITIVE), len(POSITIVE), len(NEGATIVE), len(NEGATIVE)))

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    print("候选 %d 条 → %s" % (len(rows), a.out))


main()
