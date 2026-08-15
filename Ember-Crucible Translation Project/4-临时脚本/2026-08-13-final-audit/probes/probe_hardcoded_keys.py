# -*- coding: utf-8 -*-
"""
探针：ember-hardcoded-cn.mjs 的替换表键 —— 在上游当前版本里还找得到吗？

只读。语料：
  ember 0.6.0   scripts/*.mjs + templates/**/*.hbs + lang/en.json
  crucible 0.10.1 crucible-compiled.mjs + module/**/*.mjs + templates/**/*.hbs + lang/en.json

假阳性模式：
  - 上游可能用模板拼接（`Award ${x}`），字面量搜不到 → 报「未命中」但实际可能存在。
    所以对未命中项要人工看上下文，不能直接判死键。
  - 命中不等于「以完整文本节点出现」，EXACT 表要求整段 trim 后全等。
"""
import json, os, re, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
CRUC = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible"

def collect(root, exts):
    out = []
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if os.path.splitext(f)[1].lower() in exts:
                p = os.path.join(dp, f)
                try:
                    out.append((p, open(p, encoding="utf-8").read()))
                except Exception:
                    pass
    return out

corpus = collect(EMBER, {".mjs", ".js", ".hbs", ".json"}) + collect(CRUC, {".mjs", ".js", ".hbs", ".json"})
print(f"语料文件数 {len(corpus)}  总字节 {sum(len(c) for _, c in corpus)}")

MOD = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"
src = open(MOD, encoding="utf-8").read()

# 从 mjs 里提取每张表的英文键（简单解析：形如 "Key": "值"）
def table(name):
    m = re.search(r"const %s = \{(.*?)\n\};" % name, src, re.S)
    if not m:
        return []
    return re.findall(r'"([^"]+)":\s*"', m.group(1))

TABLES = ["ATTUNEMENTS", "LANGUAGES", "KNOWLEDGE", "MOODS", "EXACT", "RESULTS",
          "CALENDAR_MONTHS", "CALENDAR_DAYS", "CALENDAR_DAY_ABBR"]

report = {}
for t in TABLES:
    keys = table(t)
    rows = []
    for k in keys:
        hits = []
        for p, c in corpus:
            n = c.count(k)
            if n:
                hits.append((os.path.relpath(p, os.path.dirname(EMBER) if p.startswith(EMBER) else os.path.dirname(CRUC)), n))
        rows.append({"key": k, "nfiles": len(hits), "hits": hits[:6]})
    report[t] = rows
    miss = [r["key"] for r in rows if r["nfiles"] == 0]
    print(f"\n### {t}  keys={len(keys)}  未命中={len(miss)}")
    for k in miss:
        print("   MISS:", k)

out = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-13-final-audit\probes\keys_report.json"
json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nwrote", out)
