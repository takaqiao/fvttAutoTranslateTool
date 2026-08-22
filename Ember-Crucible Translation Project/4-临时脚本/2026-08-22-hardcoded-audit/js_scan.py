# -*- coding: utf-8 -*-
"""JS 里的**上屏硬编码串**。

只认「值一定会上屏」的那几个字段名，以及 `ui.notifications.*` 的直接实参。
候选分三类：
  key      —— 长得像 i18n 键（`CRUCIBLE.Foo.Bar`）⇒ 走 i18n，**不算硬编码**
  literal  —— 像人话的英文 ⇒ **硬编码**
  other    —— 标识符 / 路径 / 枚举 / 空串 ⇒ 不上屏或不该翻

⚠ 判据边界：只扫**字面量**。走变量、模板插值、或从数据里取的值一律扫不到 ——
  本表是**下界**，不是全集。够不到的那部分在报告里单列。

⚠ 本文件刻意不写 `[^"\]` 这种双反斜杠字符类：本项目的 heredoc 会把 `\` 吃成 `\`，
  正则当场炸（已经栽过三次）。需要反斜杠的地方一律用 chr(92) 现拼。
"""
import io, re, sys, json, collections
sys.stdout.reconfigure(encoding='utf-8')

BS = chr(92)
STR = '"((?:[^"' + BS + BS + ']|' + BS + BS + '.)*)"'     # JS 双引号字符串（含转义）

SRC = {
    "ember": r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs",
    "crucible": r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/crucible-compiled.mjs",
}
FIELDS = ["label", "title", "hint", "tooltip", "placeholder", "content", "text", "message",
          "legend", "caption", "summary"]

RE_FIELD = re.compile(r'\b(' + '|'.join(FIELDS) + r')' + r'\s*:\s*' + STR)
RE_NOTIF = re.compile(r'ui' + BS + r'.notifications' + BS + r'.(?:info|warn|warning|error|notify)' + BS + r'(' + BS + r's*' + STR)
RE_LOCALIZE = re.compile(r'(?:localize|format|has)' + BS + r'(' + BS + r's*' + STR)
RE_KEYISH = re.compile(r'^[A-Za-z][A-Za-z0-9]*(?:' + BS + r'.[A-Za-z0-9_]+)+$')
RE_LETTER = re.compile(r'[A-Za-z]')
RE_IDENT = re.compile(r'^[a-z][A-Za-z0-9]*$')
RE_PATHY = re.compile(r'^[A-Za-z0-9._/#-]+$')

def classify(v):
    s = v.strip()
    if not s or not RE_LETTER.search(s): return "other"
    if ' ' not in s and RE_KEYISH.match(s): return "key"
    if RE_IDENT.match(s): return "other"
    if s.startswith("fa-") or s.startswith("modules/") or s.startswith("systems/") or s.startswith("icons/"):
        return "other"
    if ' ' not in s and RE_PATHY.match(s) and ('/' in s or '#' in s): return "other"
    return "literal"

report = {}
for repo, path in SRC.items():
    txt = io.open(path, encoding='utf-8').read()
    starts = [0]
    for i, ch in enumerate(txt):
        if ch == '\n': starts.append(i)
    def lineno(pos, _s=starts):
        lo, hi = 0, len(_s) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _s[mid] <= pos: lo = mid
            else: hi = mid - 1
        return lo + 1
    keys = set(m.group(1) for m in RE_LOCALIZE.finditer(txt))
    buckets = collections.defaultdict(collections.Counter)
    sites = collections.defaultdict(list)
    for m in RE_FIELD.finditer(txt):
        fld, v = m.group(1), m.group(2)
        c = classify(v)
        if c == "literal" and v in keys: c = "key"
        buckets[c][v] += 1
        if c == "literal" and len(sites[v]) < 3: sites[v].append(f"{fld}@{lineno(m.start())}")
    notif = collections.Counter()
    for m in RE_NOTIF.finditer(txt):
        v = m.group(1)
        if classify(v) == "literal" and v not in keys: notif[v] += 1
    report[repo] = {"literal": buckets["literal"], "key": buckets["key"],
                    "other": buckets["other"], "notif": notif, "sites": sites}
    print(f"{repo:<10} 展示字段：硬编码 {len(buckets['literal']):>4} 唯一 / {sum(buckets['literal'].values()):>5} 次"
          f" ｜ i18n 键 {len(buckets['key']):>4} ｜ 非上屏 {len(buckets['other']):>4}"
          f" ｜ notifications 硬编码 {len(notif):>3} 唯一 ｜ localize() 用过的键 {len(keys)}")

io.open('js_scan.json', 'w', encoding='utf-8').write(json.dumps(
    {r: {"literal": dict(d["literal"].most_common()),
         "notif": dict(d["notif"].most_common()),
         "keyCount": len(d["key"]), "otherCount": len(d["other"]),
         "sites": {k: v for k, v in d["sites"].items()}} for r, d in report.items()},
    ensure_ascii=False, indent=1))
print("\n→ js_scan.json")
for repo in SRC:
    print(f"\n=== {repo} 硬编码展示串 前 18 ===")
    for s, c in report[repo]["literal"].most_common(18):
        print(f"  {c:>3}  {s[:80]}")
