# -*- coding: utf-8 -*-
"""
第二十四轮 · ① stage2：对 48 条逐条定性所需的三个判断

1. 模板里这串是**裸字面量**还是 `{{localize "..."}}` / `{{ i18n }}`？
   —— 裸字面量 ⇒ DOM 注入（现状）是对的通道；
   —— localize 键 ⇒ 该走 lang 覆盖，键形态不对。
2. 表注释里写的 `file.hbs:NN` 与当前上游实际行号是否一致（上游挪没挪窝）。
3. NOWHERE 那条到底是不是空白折叠。
"""
import io, json, os, re

ROOT_EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
OUTDIR = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-16-round24"
HARDCODED = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"


def rd(p):
    with io.open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


raw = json.load(io.open(os.path.join(OUTDIR, "probe_window_ui_raw.json"), encoding="utf-8"))
miss = [r for r in raw["records"] if r["judge_reports_missing"]]

# ---- 1/2. 每条 hbs 命中：裸串还是 localize？行号是多少？ ----
print("=" * 100)
print("A. 48 条在 templates 里的形态（裸串 vs localize）")
print("=" * 100)
localize_keys = []
bare = []
for r in miss:
    for h in r["hbs_hits"]:
        t = h["text"]
        is_loc = bool(re.search(r'\{\{\s*(?:localize|#?i18n)\s+"' + re.escape(r["key"]) + r'"', t))
        tag = "LOCALIZE" if is_loc else "bare"
        (localize_keys if is_loc else bare).append((r["key"], h["file"], h["line"]))
        print("  %-8s %-55s %s:%s" % (tag, r["key"][:53], h["file"], h["line"]))
print()
print("  裸串命中 %d 处 / localize 命中 %d 处" % (len(bare), len(localize_keys)))
if localize_keys:
    print("  ⚠ 以下键在模板里是 localize 参数，通道应该是 lang 覆盖：")
    for k, f, l in localize_keys:
        print("     -", k, "@", f, l)
print()

# ---- 2. 注释行号 vs 实际行号 ----
print("=" * 100)
print("B. 表注释登记的 hbs:行号  vs  当前上游实际行号")
print("=" * 100)
src = rd(HARDCODED)
m = re.search(r"const EMBER_WINDOW_UI = \{(.*?)\n\};", src, re.S)
body = m.group(1)
# 抓形如  // xxx.hbs:12  或 // creation/class.hbs:42
claimed = {}
for line in body.split("\n"):
    cm = re.search(r"//\s*([\w/\-]+\.hbs):(\d+)", line)
    if not cm:
        continue
    km = re.search(r'^\s*"((?:[^"\\]|\\.)*)"\s*:', line)
    if km:
        claimed[km.group(1)] = (cm.group(1), int(cm.group(2)))

drift = []
for r in miss:
    k = r["key"]
    if k not in claimed:
        continue
    cf, cl = claimed[k]
    actual = [(h["file"], h["line"]) for h in r["hbs_hits"]]
    ok = any(af.endswith(cf) and al == cl for af, al in actual)
    near = any(af.endswith(cf) for af, al in actual)
    if not ok:
        drift.append((k, cf, cl, actual, near))
        print("  DRIFT %-50s 注释=%s:%s  实际=%s" % (k[:48], cf, cl, actual))
print("  行号漂移条数 =", len(drift), "（注：只影响注释准确性，不影响键是否命中）")
print()

# ---- 3. NOWHERE 那条 ----
print("=" * 100)
print("C. NOWHERE 那条：空白折叠验证")
print("=" * 100)
nowhere = [r for r in miss if not any(r["in"].values())]
vcs = rd(os.path.join(ROOT_EMBER, "templates", "applications", "vista-config-scene.hbs"))
for r in nowhere:
    k = r["key"]
    print("  键:", k[:80], "...")
    folded = re.sub(r"\s+", " ", vcs)
    print("  折叠上游全文后能否找到该键:", k in folded)
    idx = folded.find(k)
    if idx >= 0:
        print("  折叠后上下文:", repr(folded[max(0, idx - 60):idx + len(k) + 60]))
    # 原文里定位
    head = k[:40]
    for i, line in enumerate(vcs.split("\n"), 1):
        if head[:25] in line:
            print("  原文行 %d: %r" % (i, line))
print()

# ---- 4. Aster Progression / Soulbound Progression ----
print("=" * 100)
print("D. Aster Progression（判据没抓的 .mjs）")
print("=" * 100)
for r in miss:
    if r["other_mjs_hits"]:
        print(" ", r["key"])
        for h in r["other_mjs_hits"]:
            print("    %s:%s  %s" % (h["file"], h["line"], h["text"][:160]))
# 对照：Soulbound Progression 判据没报（说明它在 ember.mjs 里），确认
for r in raw["records"]:
    if r["key"] in ("Soulbound Progression", "Aster Progression"):
        print("  [对照] %-24s 判据报缺=%s  in=%s" % (r["key"], r["judge_reports_missing"],
              {k: v for k, v in r["in"].items() if v}))
print()

# ---- 5. 28 条「判据找得到」的里面有没有其实只是短词偶然命中 ----
print("=" * 100)
print("E. 判据找得到的 28 条（本轮不定性，仅列出以便判断是否偶然命中）")
print("=" * 100)
found = [r for r in raw["records"] if not r["judge_reports_missing"]]
for r in found:
    print("  %-28s len=%-3d in=%s" % (r["key"][:26], len(r["key"]),
          ",".join(k for k, v in r["in"].items() if v)))
