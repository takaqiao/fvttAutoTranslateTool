# -*- coding: utf-8 -*-
"""
探针 2：模板里的硬编码英文 × patchRenderedApplications 的宿主闸

判据：
  1. 扫 ember 模块 66 个 .hbs，抽出**不走 {{localize}} / i18n 键**的英文可见文本：
     元素内文本节点、aria-label="..."、title="..."、alt="..."、label="..."、placeholder="..."。
  2. 对每个模板，判断它被注册进哪一个 PARTS（宿主 Application 类名）。
  3. 宿主闸 = /^Ember/.test(类名) || /ember/i.test(根元素 class)。
     宿主是 crucible / dnd5e 的类（Ember 只是往里 splice 了一个 tab）→ 闸外，
     patchRenderedApplications 直接 return，整块面板永不被翻。
  4. 闸内的模板，再看每条英文串是否落在 EXACT 里。

已知假阳性：
  - 纯 GM 开发工具面板（token-maker / vista-config）的英文，项目可能有意不译。
  - 文本节点里可能混 handlebars 表达式，抽取按行做粗过滤。
只读，不写库。
"""
import os
import re
import sys
import json

EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
HC = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"

EXACT = set(re.findall(r'^\s*"([^"]+)":\s*"', open(HC, encoding="utf-8").read(), re.M))

TEXT_RE = re.compile(r">([^<>{}]{3,120})<")
ATTR_RE = re.compile(r'\b(aria-label|title|alt|label|placeholder)="([^"{}]{3,120})"')
HAS_WORD = re.compile(r"[A-Za-z]{3,}")

# 模板路径 -> 宿主类名（从 ember.mjs 的 PARTS 注册里抓）
src = open(os.path.join(EMBER, "scripts", "ember.mjs"), encoding="utf-8").read()


def host_of(rel):
    """找 template: "modules/ember/<rel>" 出现处，往上回溯最近的 class 声明。"""
    needle = 'modules/ember/' + rel.replace("\\", "/")
    i = src.find(needle)
    if i < 0:
        return None
    head = src[:i]
    ms = list(re.finditer(r"\nclass (\w+)|cls\s*=\s*([\w.]+)\.?", head))
    # 最近的 class 声明
    cm = None
    for m in re.finditer(r"\nclass (\w+)", head):
        cm = m
    # 若在 addAttunementTab 这类注入函数里，优先取 cls = crucible…/dnd5e…
    tail = head[-1500:]
    inj = re.findall(r"cls\s*=\s*([\w.]+);|const \{\s*(\w+)\s*\} = dnd5e\.applications\.actor", tail)
    if inj:
        for a, b in inj:
            if a or b:
                return "INJECTED:" + (a or b)
    return cm.group(1) if cm else None


rows = []
for dirpath, _dirs, files in os.walk(os.path.join(EMBER, "templates")):
    for fn in files:
        if not fn.endswith((".hbs", ".html")):
            continue
        full = os.path.join(dirpath, fn)
        rel = os.path.relpath(full, EMBER).replace("\\", "/")
        body = open(full, encoding="utf-8").read()
        strings = []
        for m in TEXT_RE.finditer(body):
            t = m.group(1).strip()
            if t and HAS_WORD.search(t) and "{{" not in t:
                strings.append(("text", t))
        for m in ATTR_RE.finditer(body):
            t = m.group(2).strip()
            if HAS_WORD.search(t) and "." not in t.split()[0][:1]:
                strings.append((m.group(1), t))
        if not strings:
            continue
        host = host_of(rel)
        gated = bool(host and (host.startswith("Ember") or host.startswith("Ember")))
        rows.append({
            "template": rel,
            "host": host,
            "host_gate_pass": gated,
            "strings": [{"where": k, "text": t, "in_EXACT": t in EXACT} for k, t in strings],
        })

json.dump(rows, sys.stdout, ensure_ascii=False, indent=1)
