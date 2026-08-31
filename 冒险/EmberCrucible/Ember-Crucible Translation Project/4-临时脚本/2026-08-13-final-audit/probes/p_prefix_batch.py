# -*- coding: utf-8 -*-
r"""
探针 P-PREFIX：种子那一类的「前缀表」切面。

种子是「同一批 UI 字符串只补了一半」。本探针只盯一种最容易半补的批次：
**上游用模板字面量拼出来的 `前缀: 动态值` 标签**。
汉化侧 translateText 只有两条路能吃掉这种串：
   1) PREFIXED（en 必须精确等于前缀，且后面跟 ": "）
   2) PATTERNS 里那五条正则
EXACT 里放一个**光秃秃的前缀词**（如 "Ancestry"）对 `Ancestry: Keth` **一点用都没有**
—— trim 后整串不等于 "Ancestry"。

做法：
  从 ember 上游源码里正则抓出所有形如
      innerHTML = `Xxx: ${...}`   /  innerText = `Xxx: ${...}`
      label = `Xxx: ${...}`       /  tooltipText: `Xxx ...`
  的模板字面量，取出「冒号前的静态前缀」，与 PREFIXED / PATTERNS 比对。

假阳性模式：
  FP1 有些前缀本身就是动态的（`${dataset.channel.capitalize()}: ...`），
      静态文本抓不到，要人工把它展开成可能的取值集合。
  FP2 有些拼出来的串根本不上屏（日志、异常消息、通知）——要看它进的是不是 DOM。
  FP3 有些串外层还有别的通道能翻（例如整段进了 lang key）。
"""
import io, os, re, sys, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER_UP = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
HC = os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")
src = open(HC, encoding="utf-8").read()

PREFIXED = re.findall(r'\{\s*en:\s*"([^"]+)"', src)
print("汉化 PREFIXED 前缀：", PREFIXED)
PATTERNS = re.findall(r"re:\s*(/[^,]+/)", src)
print("汉化 PATTERNS：", PATTERNS)
EXACT_BLOCK = re.search(r"const EXACT = \{(.*?)\n\};", src, re.S).group(1)
EXACT = set(re.findall(r'"([^"]+)":\s*"', EXACT_BLOCK))

files = []
d = os.path.join(EMBER_UP, "scripts")
for f in sorted(os.listdir(d)):
    if f.endswith(".mjs"):
        files.append((f, open(os.path.join(d, f), encoding="utf-8").read()))

# 抓「进 DOM 的模板字面量」
SINKS = r"(?:innerHTML|innerText|textContent|outerHTML|label|name|tooltipText|tooltip|title|content)"
TL = re.compile(r"%s\s*(?:=|\+=|:)\s*`([^`]{2,160})`" % SINKS)

found = {}
for fn, c in files:
    for m in TL.finditer(c):
        s = m.group(1)
        line = c.count("\n", 0, m.start()) + 1
        # 只要含 ${ 的动态串，且冒号前是静态文本
        if "${" not in s:
            continue
        mm = re.match(r"^([A-Za-z][A-Za-z .'\-]{0,40}):\s*\$\{", s)
        if not mm:
            continue
        pre = mm.group(1).strip()
        found.setdefault(pre, []).append((fn, line, s))

print("\n=== ember 上游用 `<前缀>: ${...}` 拼出来的标签 ===")
rows = []
for pre, occ in sorted(found.items()):
    ok = "PREFIXED" if pre in PREFIXED else ("EXACT(无效)" if pre in EXACT else "缺")
    print(f'  {pre!r:26s} {ok:12s} x{len(occ)}   e.g. {occ[0][0]}:{occ[0][1]}  `{occ[0][2][:70]}`')
    rows.append({"prefix": pre, "status": ok, "n": len(occ),
                 "occ": [{"file": a, "line": b, "src": c} for a, b, c in occ]})

# 动态前缀（`${x}: ...`）单列
print("\n=== 前缀本身是动态表达式的（需人工展开取值集合） ===")
DYN = re.compile(r"%s\s*(?:=|\+=|:)\s*`(\$\{[^`]{2,80})`" % SINKS)
dyn = []
for fn, c in files:
    for m in DYN.finditer(c):
        s = m.group(1)
        if ":" not in s:
            continue
        line = c.count("\n", 0, m.start()) + 1
        dyn.append((fn, line, s))
        print(f"  {fn}:{line}  `{s[:90]}`")

outp = os.path.join(ROOT, "4-临时脚本", "2026-08-13-final-audit", "findings", "p_prefix_batch.json")
json.dump({"prefixed_cn": PREFIXED, "rows": rows,
           "dynamic": [{"file": a, "line": b, "src": c} for a, b, c in dyn]},
          open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nwrote", outp)
