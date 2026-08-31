# -*- coding: utf-8 -*-
"""模板（.hbs）里的**上屏硬编码串**：文本节点 + 展示类属性。

判据（每一条都保守，宁可漏不可滥，漏的那部分在报告里单列）：
  · 文本节点：去掉 `{{...}}` / `<!-- -->` / `<style>`/`<script>` 之后，剩下的可见文字
  · 属性：placeholder / title / data-tooltip / data-tooltip-text / aria-label / alt / label
  · 剔除：纯 `{{...}}` 插值、纯符号/数字、长度 < 2、明显是类名或路径的
  · `{{localize "X"}}` / `{{#if}}` 之类**本来就走 i18n 或不上屏**，不算硬编码
"""
import io, os, re, sys, json, collections
sys.stdout.reconfigure(encoding='utf-8')

ROOTS = {
    "ember": r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/templates",
    "crucible": r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/templates",
}
ATTRS = ["placeholder", "title", "data-tooltip", "data-tooltip-text", "aria-label", "alt", "label"]
RE_MUSTACHE = re.compile(r'\{\{[^}]*\}\}')
RE_COMMENT = re.compile(r'<!--.*?-->', re.S)
RE_BLOCK = re.compile(r'<(script|style)\b.*?</\1>', re.S | re.I)
RE_TAG = re.compile(r'<[^>]+>', re.S)
RE_ATTR = re.compile(r'\b(' + '|'.join(a.replace('-', r'\-') for a in ATTRS) + r')\s*=\s*"([^"]*)"')
RE_LETTER = re.compile(r'[A-Za-z]')
RE_PATHY = re.compile(r'^[a-z0-9._/-]+$')          # 类名 / 路径 / 变量名那一类
RE_ENTITY = re.compile(r'&[a-z]+;|&#\d+;')

def clean_text(t):
    t = RE_ENTITY.sub(' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def keep(s):
    if len(s) < 2: return False
    if not RE_LETTER.search(s): return False
    if RE_PATHY.match(s): return False                 # 全小写+点斜杠 = 标识符
    if s.startswith('{{') or s.endswith('}}'): return False
    return True

out = {}
for repo, root in ROOTS.items():
    files = []
    for dirpath, _d, fnames in os.walk(root):
        for f in fnames:
            if f.endswith('.hbs'): files.append(os.path.join(dirpath, f))
    texts, attrs = collections.Counter(), collections.Counter()
    where = collections.defaultdict(set)
    for p in sorted(files):
        raw = io.open(p, encoding='utf-8').read()
        rel = os.path.relpath(p, root).replace(os.sep, '/')
        body = RE_BLOCK.sub(' ', RE_COMMENT.sub(' ', raw))
        # 属性先抽（下一步会把标签整个删掉）
        for m in RE_ATTR.finditer(body):
            v = clean_text(RE_MUSTACHE.sub(' ', m.group(2)))
            if keep(v): attrs[v] += 1; where[v].add(rel)
        # 文本节点
        stripped = RE_TAG.sub('\n', body)
        for chunk in stripped.split('\n'):
            v = clean_text(RE_MUSTACHE.sub(' ', chunk))
            if keep(v): texts[v] += 1; where[v].add(rel)
    out[repo] = {"files": len(files), "texts": texts, "attrs": attrs, "where": where}
    tot = set(texts) | set(attrs)
    print(f"{repo:<10} 模板 {len(files):>3} 份 ｜ 文本节点串 {len(texts):>4} 唯一 / {sum(texts.values()):>4} 次"
          f" ｜ 属性串 {len(attrs):>3} 唯一 / {sum(attrs.values()):>3} 次 ｜ 合计唯一 {len(tot)}")

io.open('hbs_scan.json', 'w', encoding='utf-8').write(json.dumps(
    {r: {"files": d["files"],
         "texts": dict(d["texts"].most_common()),
         "attrs": dict(d["attrs"].most_common()),
         "where": {k: sorted(v) for k, v in d["where"].items()}}
     for r, d in out.items()}, ensure_ascii=False, indent=1))
print("\n→ hbs_scan.json")
for repo in ROOTS:
    d = out[repo]
    print(f"\n=== {repo} 文本节点 前 25 ===")
    for s, c in d["texts"].most_common(25):
        print(f"  {c:>3}  {s[:88]}")
