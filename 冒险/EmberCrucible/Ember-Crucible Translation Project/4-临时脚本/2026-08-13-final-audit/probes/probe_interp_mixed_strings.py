# -*- coding: utf-8 -*-
r"""
probe_interp_mixed_strings.py —「补丁覆盖面不足」的**插值混合串**一支（只读）

为什么单独立一支：
  本轮既有的模板扫描（p_hbs_batch / gate_reach / probe_hbs_literals）都是先把
  `{{...}}` 整段删掉再判断，于是
      data-tooltip="Location: {{location.label}}"
      <figcaption>An example {{ancestry.name}} character.</figcaption>
  这类**英文字面量 + 插值**的串，要么被整条丢掉，要么只剩下 "Location" 这样
  的残渣，从而在报告里表现为「这个文件只有 1 条」。实测 p_hbs_batch.json 里
  hex-hud.hbs 只报了 1 条（Event Probabilities），而该文件真实有 5 条。

同时它们在**修法**上也自成一类：这类串永远不可能整串命中 EXACT
（尾巴是运行期才知道的），必须进 PREFIXED（`前缀: ` 形式）或 PATTERNS（正则），
所以即使有人把它们塞进 EXACT 也不会生效 —— 与已确认实例「只翻 .window-title」
同一个病根：补丁的形状与上游发射串的形状对不上。

判据：
  在 ember 的 .hbs 里找同时满足下面两条的串
    (a) 含 `{{...}}` 插值；
    (b) 去掉插值后仍有 >=1 个英文单词（>=3 字母，且不是 i18n 键 / 布尔属性名）。
  再按补丁的三张表判定可达性：
    EXACT      —— 整串相等才行 → 含插值必不命中
    PREFIXED   —— 需要 `<英文前缀>: ` 且前缀在表里
    PATTERNS   —— 需要有对应正则
输出每条的「所在文件 / 属性或文本 / 现有表能否覆盖」。

假阳性模式：
  - `alt="..."` 只在图片加载失败时可见，权重低，单独标注；
  - `{{#if}}`/`{{else}}` 之类块级 helper 不是插值，已剔除；
  - 有的串在 GM 专用界面（vista 配置 / token maker），玩家看不到；
  - 少数串的英文部分是标点或单位（如 `%`），已过滤。
"""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
CN = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"

src = open(CN, encoding="utf-8").read()
PREFIXES = set(re.findall(r'\{\s*en:\s*"([^"]+)"', src))
PATTERNS = re.findall(r"re:\s*(/[^,]+/),", src)

HB_BLOCK = re.compile(r"\{\{[#/][^}]*\}\}")
HB = re.compile(r"\{\{[^}]*\}\}")
TAG = re.compile(r"<[^>]+>")
ATTR = re.compile(r'\b(aria-label|data-tooltip|data-tooltip-text|title|placeholder|alt)\s*=\s*"([^"]*)"')
WORD = re.compile(r"[A-Za-z]{3,}")

rows = []
for dirpath, _d, files in os.walk(os.path.join(EMBER, "templates")):
    for f in sorted(files):
        if not f.endswith(".hbs"):
            continue
        rel = os.path.relpath(os.path.join(dirpath, f), EMBER).replace("\\", "/")
        text = open(os.path.join(dirpath, f), encoding="utf-8").read()

        cands = []
        for attr, val in ATTR.findall(text):
            if "{{" in val:
                cands.append(("attr:" + attr, val))
        body = HB_BLOCK.sub("", text)
        for chunk in TAG.split(body):
            for line in chunk.splitlines():
                if "{{" in line and HB.sub("", line).strip():
                    cands.append(("text", line.strip()))

        for kind, raw in cands:
            lit = HB.sub("\x00", raw)
            words = WORD.findall(lit.replace("\x00", " "))
            words = [w for w in words if not re.match(r"^(localize|value|placeholder|cssClass|class|data|true|false)$", w)]
            if not words:
                continue
            if re.match(r"^[A-Z][A-Z0-9_]*(\.[A-Za-z0-9_]+)+$", raw.strip()):
                continue
            m = re.match(r"^([A-Z][A-Za-z ]{1,20}): ", lit)
            prefix = m.group(1) if m else None
            rows.append({"tpl": rel, "kind": kind, "raw": raw.strip()[:100],
                         "prefix": prefix,
                         "prefixInTable": bool(prefix) and prefix in PREFIXES})

print(f"含插值且带英文字面量的串：{len(rows)} 条")
print(f"补丁现有前缀表 PREFIXED = {sorted(PREFIXES)}")
print(f"补丁现有 PATTERNS = {PATTERNS}")
print()
cur = None
for r in rows:
    if r["tpl"] != cur:
        cur = r["tpl"]
        print(f"-- {cur}")
    flag = "前缀式" if r["prefix"] else "内嵌式"
    cov = "已在PREFIXED" if r["prefixInTable"] else "无法命中任何表"
    print(f"   [{r['kind']:<18}] {flag} {cov:<14} {r['raw']}")
