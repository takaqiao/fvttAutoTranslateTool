# -*- coding: utf-8 -*-
"""
探针 gate_reach —— 把「闸/选择器失配导致整块界面不进入替换」抽象成机械判据。

判据（复现 ember-hardcoded-cn.mjs:445-471 patchRenderedApplications 的闸）：
  对每一个「会显示 ember 硬编码英文」的 Application：
      cls = 根元素 className（= ApplicationV2 的 options.classes 拼起来 + 主题类）
      id  = app.constructor.name
      放行 ⟺  /ember/i.test(cls)  ||  /^Ember/.test(id)
      否则只有 DialogV2/带 dialog 类的窗口能翻到 .window-title 一行。
  凡「模板里有裸英文 / JS 里拼英文」而闸不放行的 App，都是同一类缺陷。

做法：
  1. 从 ember.mjs 里抓所有 `template: "modules/ember/templates/..."`，
     向上找最近的 `class X extends Y`（顶层，缩进 0），得到宿主类。
  2. 在该 class 体内找 `classes: [...]`（DEFAULT_OPTIONS），得到根元素类名。
     找不到就沿 extends 链向上找一次。
  3. 跑闸，输出 PASS/BLOCKED。
  4. 对 BLOCKED 的宿主，列出其模板里的裸英文字面量（文本节点 + 可翻属性）。

只读。假阳性模式：
  - 静态解析拿不到运行时才加的 classes（如 crucible/dnd5e 注入的 PARTS，
    宿主根本不是 ember 自己的类）—— 这类要人工补。
  - `classes:` 可能出现在非 DEFAULT_OPTIONS 的地方 → 取 class 体内第一个 `classes: [`。
  - hbs 里的英文可能被 {{#if}} 包住永不显示 → 需人工看。
"""
import os, re, sys, io, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
SRC = os.path.join(EMBER, "scripts", "ember.mjs")
TPLROOT = os.path.join(EMBER, "templates")

lines = open(SRC, encoding="utf-8").read().split("\n")

# --- 1. 顶层 class 索引 ---
class_at = []  # (lineno, name, parent)
for i, l in enumerate(lines):
    m = re.match(r"^\s{0,2}(?:export\s+)?class\s+([A-Za-z0-9_$]+)(?:\s+extends\s+([A-Za-z0-9_$.()]+))?", l)
    if m:
        class_at.append((i, m.group(1), m.group(2) or ""))

def owner(lineno):
    best = None
    for i, n, p in class_at:
        if i <= lineno:
            best = (i, n, p)
        else:
            break
    return best

# --- 2. class 体范围（粗略：到下一个顶层 class 或文件末） ---
def body_range(start):
    for i, n, p in class_at:
        if i > start:
            return (start, i)
    return (start, len(lines))

def classes_of(start, end):
    txt = "\n".join(lines[start:end])
    m = re.search(r"classes:\s*\[([^\]]*)\]", txt)
    if not m:
        return None
    return [x.strip().strip('"\'') for x in m.group(1).split(",") if x.strip()]

GATE_CLS = re.compile(r"ember", re.I)

# --- 3. 收集 template -> 宿主 ---
hosts = {}
for i, l in enumerate(lines):
    for m in re.finditer(r'"(modules/ember/templates/[^"]+\.(?:hbs|html))"', l):
        tpl = m.group(1)
        o = owner(i)
        if not o:
            continue
        s, name, parent = o
        e = body_range(s)[1]
        cl = classes_of(s, e)
        hosts.setdefault((name, parent, tuple(cl) if cl else None), set()).add(tpl)

# --- 4. hbs 裸英文 ---
TAG = re.compile(r"<[^>]*>", re.S)
HB = re.compile(r"\{\{[^}]*\}\}", re.S)
ATTR = re.compile(r'(aria-label|data-tooltip|data-tooltip-text|data-tooltip-html|title|placeholder|alt)\s*=\s*"([^"]*)"')

def literals(tplpath):
    p = os.path.join(EMBER, tplpath.replace("modules/ember/", "").replace("/", os.sep))
    if not os.path.exists(p):
        return None
    s = open(p, encoding="utf-8").read()
    out = []
    for m in ATTR.finditer(s):
        v = m.group(2).strip()
        if v and "{{" not in v and re.search(r"[A-Za-z]{3}", v):
            out.append(("@" + m.group(1), v))
    body = HB.sub("\n", ATTR.sub("", TAG.sub("\n", s)))
    for t in body.split("\n"):
        t = t.strip()
        if len(t) < 3 or not re.search(r"[A-Za-z]{3}", t):
            continue
        if re.fullmatch(r"[\w\-./]+", t) and "." in t:
            continue
        out.append(("text", t))
    return out

rows = []
for (name, parent, cl), tpls in sorted(hosts.items()):
    clsstr = " ".join(cl) if cl else ""
    passes = bool(GATE_CLS.search(clsstr)) or name.startswith("Ember")
    rows.append(dict(app=name, parent=parent, classes=cl, gate="PASS" if passes else "BLOCKED",
                     templates=sorted(tpls)))

print(f"从 ember.mjs 抓到 {len(hosts)} 个宿主类 / {sum(len(v) for v in hosts.values())} 个模板引用\n")
for r in sorted(rows, key=lambda r: (r["gate"], r["app"])):
    print(f'[{r["gate"]:7s}] {r["app"]}  extends {r["parent"]}  classes={r["classes"]}')
    for t in r["templates"]:
        print(f"           {t}")
print("\n" + "=" * 70)
print("BLOCKED 宿主的模板裸英文：")
for r in rows:
    if r["gate"] != "BLOCKED":
        continue
    for t in r["templates"]:
        lit = literals(t)
        if lit is None:
            print(f"  [缺文件] {t}")
            continue
        if not lit:
            continue
        print(f"\n  -- {r['app']} :: {t}")
        seen = set()
        for kind, v in lit:
            if (kind, v) in seen:
                continue
            seen.add((kind, v))
            print(f"       {kind:22s} {v}")
