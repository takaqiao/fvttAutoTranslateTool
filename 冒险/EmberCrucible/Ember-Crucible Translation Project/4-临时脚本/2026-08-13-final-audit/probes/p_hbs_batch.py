# -*- coding: utf-8 -*-
r"""
探针 P-HBS：ember 模板里的裸英文字面量 vs EXACT —— 找「同一个模板里一半有键一半没键」。

只看 ember 自己的 templates/**/*.hbs（crucible 的模板由 crucible-cn 的 lang 负责）。
抽出：
   - 标签文本节点（>xxx<）里不含 {{ }} 的纯英文
   - placeholder="..." / aria-label="..." / data-tooltip="..." / title="..." / alt="..."
按**文件**分组，算 EXACT 覆盖率；0 < 覆盖 < 总数 → PARTIAL。

假阳性模式：
  FP1 有些文本被外层 {{#if}} 挡着、实际场景里不出现。
  FP2 单字母 / 缩写 / 单位不需要翻。
  FP3 placeholder 这类属性即使补了键也不生效 —— translateNode 的属性白名单里没有
      placeholder（只有 data-tooltip / data-tooltip-text / data-tooltip-html / title /
      aria-label），要连引擎一起改，本探针只负责指出「键」这一层。
"""
import io, os, re, sys, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER_UP = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
HC = os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")
src = open(HC, encoding="utf-8").read()
EXACT = set(re.findall(r'"([^"]+)":\s*"', re.search(r"const EXACT = \{(.*?)\n\};", src, re.S).group(1)))

TXT = re.compile(r">([^<>{}\n]{2,80})<")
ATTR = re.compile(r'\b(placeholder|aria-label|data-tooltip|title|alt)="([^"{}]{2,80})"')
ENG = re.compile(r"^[A-Z][A-Za-z0-9'’&/\-]*(?: [A-Za-z0-9'’&/\-()?.,:]+)*[?!.]?$")

rows = []
for dp, dn, fn in os.walk(os.path.join(EMBER_UP, "templates")):
    for f in sorted(fn):
        if not f.endswith(".hbs"):
            continue
        p = os.path.join(dp, f)
        c = open(p, encoding="utf-8").read()
        rel = os.path.relpath(p, EMBER_UP)
        items = []
        for m in TXT.finditer(c):
            s = m.group(1).strip()
            if not s or not ENG.match(s) or len(s) < 3:
                continue
            items.append(("text", c.count("\n", 0, m.start()) + 1, s))
        for m in ATTR.finditer(c):
            s = m.group(2).strip()
            if not ENG.match(s):
                continue
            items.append((m.group(1), c.count("\n", 0, m.start()) + 1, s))
        if not items:
            continue
        cov = [i for i in items if i[2] in EXACT]
        rows.append({"file": rel, "n": len(items), "ncov": len(cov),
                     "covered": sorted({i[2] for i in cov}),
                     "uncovered": sorted({i[2] for i in items if i[2] not in EXACT}),
                     "detail": [{"kind": a, "line": b, "s": d} for a, b, d in items]})

part = [r for r in rows if 0 < r["ncov"] < r["n"]]
none = [r for r in rows if r["ncov"] == 0]
full = [r for r in rows if r["ncov"] == r["n"]]
print(f"ember 模板 {len(rows)} 个含裸英文；PARTIAL {len(part)}  全无键 {len(none)}  全有键 {len(full)}")
print("\n=== PARTIAL（同一模板里一半有键一半没键） ===")
for r in part:
    print(f'  {r["file"]}   {r["ncov"]}/{r["n"]}')
    print(f'     有键: {r["covered"]}')
    print(f'     缺键: {r["uncovered"]}')

print("\n=== 全无键、但文件名说明它是 Ember 自己的界面 ===")
for r in sorted(none, key=lambda x: -x["n"])[:20]:
    print(f'  {r["file"]}  n={r["n"]}  {r["uncovered"][:8]}')

outp = os.path.join(ROOT, "4-临时脚本", "2026-08-13-final-audit", "findings", "p_hbs_batch.json")
json.dump(rows, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nwrote", outp)
