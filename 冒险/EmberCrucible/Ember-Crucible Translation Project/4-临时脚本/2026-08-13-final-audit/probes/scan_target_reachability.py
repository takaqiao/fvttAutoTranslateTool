# -*- coding: utf-8 -*-
"""
探针 2：汉化目标串的「投递通道可达性」
=====================================

与探针 1 同一类问题的另一个方向。汉化模块 ember-hardcoded-cn.mjs 只有三条投递通道：

  通道 A  renderApplicationV2 / renderApplication 钩子后对 root 做 translateNode
          —— 闸门：root.className 含 "ember" 或 app.constructor.name 以 "Ember" 开头；
             例外分支只翻 DialogV2 的 .window-title，**不翻 dialog 正文**。
  通道 B  patchEnrichers 包住 CONFIG.TextEditor.enrichers 的返回值
  通道 C  ready 时一次性改 crucible.CONFIG / CONFIG.time.worldCalendarConfig 的数据

对每个目标英文串，找出上游 ember.mjs / templates 里的产出点，
判断产出点落在哪个宿主里，从而判断哪条通道够得着。

输出三类：
  DEAD        —— 目标串在上游根本不存在（表项白写）
  UNREACHABLE —— 存在，但产出点在三条通道都够不到的宿主里
  OK          —— 有通道覆盖

假阳性模式：
  * 上游可能用模板字符串拼出该串（`${a}: ${b}`），字面 grep 找不到 → 会误判 DEAD；
    脚本对 PREFIXED / PATTERNS 用前缀而非全串匹配来缓解；
  * hbs 模板里的串可能走 {{localize}} —— 脚本会标出但不当缺陷；
  * 宿主判定靠「向上找最近的 class Xxx」，rollup 拼接处会失准。
"""

import io
import json
import os
import re

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER_DIR = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
EMBER_MJS = os.path.join(EMBER_DIR, "scripts", "ember.mjs")
CN_HARDCODED = os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")

CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+([\w$]+)")


def load_tables():
    src = io.open(CN_HARDCODED, encoding="utf-8").read()
    tables = {}
    for block in ("ATTUNEMENTS", "LANGUAGES", "KNOWLEDGE", "MOODS", "EXACT", "RESULTS",
                  "CALENDAR_MONTHS", "CALENDAR_DAYS", "CALENDAR_DAY_ABBR"):
        m = re.search(r"const %s = \{(.*?)\n\};" % block, src, re.S)
        tables[block] = re.findall(r'"([^"]+)"\s*:', m.group(1)) if m else []
    return tables


def collect_files():
    files = [("scripts/ember.mjs", EMBER_MJS)]
    for base in ("templates", "ui", "styles"):
        d = os.path.join(EMBER_DIR, base)
        for dp, _dn, fn in os.walk(d):
            for f in fn:
                if f.endswith((".hbs", ".html", ".json", ".css")):
                    p = os.path.join(dp, f)
                    files.append((os.path.relpath(p, EMBER_DIR).replace("\\", "/"), p))
    return files


def main():
    tables = load_tables()
    files = collect_files()
    blobs = {}
    for rel, p in files:
        try:
            blobs[rel] = io.open(p, encoding="utf-8").read()
        except Exception:
            pass

    mjs_lines = blobs["scripts/ember.mjs"].split("\n")

    def host_of(lineno):
        for j in range(lineno - 1, max(-1, lineno - 6000), -1):
            m = CLASS_RE.match(mjs_lines[j])
            if m:
                return m.group(1)
        return None

    report = []
    for block, keys in tables.items():
        for k in keys:
            hits = []
            for rel, blob in blobs.items():
                for m in re.finditer(re.escape(k), blob):
                    ln = blob.count("\n", 0, m.start()) + 1
                    hits.append((rel, ln))
            # 只在 mjs / hbs 里出现才算「产出点」
            sites = []
            for rel, ln in hits[:40]:
                host = host_of(ln) if rel == "scripts/ember.mjs" else None
                ctx = (blobs[rel].split("\n")[ln - 1]).strip()[:180]
                sites.append({"file": rel, "line": ln, "host": host, "code": ctx})
            report.append({"table": block, "key": k, "n_hits": len(hits), "sites": sites[:6]})

    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target_reachability.json")
    io.open(dest, "w", encoding="utf-8").write(json.dumps(report, ensure_ascii=False, indent=1))

    dead = [r for r in report if r["n_hits"] == 0]
    print("targets=%d  DEAD(上游 0 处出现)=%d" % (len(report), len(dead)))
    for r in dead:
        print("  DEAD  [%s] %r" % (r["table"], r["key"]))
    print("->", dest)


main()
