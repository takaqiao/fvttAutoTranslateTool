# -*- coding: utf-8 -*-
r"""
探针 P-DIALOG：种子那一类的「对话框标题/按钮」切面。

ember-hardcoded-cn.mjs 的 EXACT 里有一整节「对话框标题」，注释写「Ember 的十五个确认框」。
本探针把上游所有 DialogV2 的 window.title / buttons[].label / ok.label / yes.label
全抓出来，与 EXACT 比对，看这一批是不是也只补了一半。

抓的形态：
   window: {title: "..."}              window: {title: `...`}
   dialog.window ||= {title: `...`}
   buttons: [{... label: "..."}]       ok: {label: "..."}   yes: {label: "..."}
   DialogV2.confirm({... })            DialogV2.prompt / .input / .wait

假阳性模式：
   FP1 title 走 i18n key 或 _loc()，抓到的是 key 名不是英文。
   FP2 有些 DialogV2 只有 GM 在开发调试时能开到。
   FP3 buttons 用 Foundry 默认（Yes/No/Confirm/Cancel）时由核心 i18n 翻，不需要键。
   FP4 正则抓 `window: {title:` 时若 title 在对象里换行写就抓不到 → 漏报，不是误报。
"""
import io, os, re, sys, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER_UP = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
HC = os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")
src = open(HC, encoding="utf-8").read()
EXACT = set(re.findall(r'"([^"]+)":\s*"', re.search(r"const EXACT = \{(.*?)\n\};", src, re.S).group(1)))
PREFIXED = re.findall(r'\{\s*en:\s*"([^"]+)"', src)
PATS = [re.compile(r"^Result of (.+)$"), re.compile(r"^Award Attunement: (.+)$"),
        re.compile(r"^Revoke Attunement: (.+)$"), re.compile(r"^Activate Attunement: (.+)$"),
        re.compile(r"^Day (\d+)\b(.*)$")]

# Foundry / crucible 核心自带、由核心 i18n 负责的按钮词
CORE_BUTTONS = {"Yes", "No", "OK", "Ok", "Cancel", "Confirm", "Close", "Save", "Submit", "Apply", "Delete"}


def cov(s):
    s = s.strip()
    if s in EXACT:
        return "EXACT"
    for p in PREFIXED:
        if s.startswith(p + ": "):
            return "PREFIXED"
    for r in PATS:
        if r.match(s):
            return "PATTERN"
    if s in CORE_BUTTONS:
        return "CORE-i18n"
    return None


files = []
d = os.path.join(EMBER_UP, "scripts")
for f in sorted(os.listdir(d)):
    if f.endswith(".mjs"):
        files.append((f, open(os.path.join(d, f), encoding="utf-8").read()))

TITLE = re.compile(r"window(?:\s*\|\|=|\s*:|\s*=)\s*\{\s*title:\s*[\"`']([^\"`']{2,90})[\"`']")
BTN = re.compile(r"(?:\bok|\byes|\bno|\bbuttons?)\s*:\s*(?:\[)?\s*\{[^}]{0,120}?\blabel:\s*[\"`']([^\"`']{1,60})[\"`']")
BTN2 = re.compile(r"\{\s*(?:action|value)?[^}]{0,60}?label:\s*[\"`']([^\"`']{1,60})[\"`'][^}]{0,80}?icon:")

titles, btns = [], []
for fn, c in files:
    for m in TITLE.finditer(c):
        titles.append((fn, c.count("\n", 0, m.start()) + 1, m.group(1)))
    for m in BTN.finditer(c):
        btns.append((fn, c.count("\n", 0, m.start()) + 1, m.group(1)))

print(f"=== DialogV2 window.title：抓到 {len(titles)} 条 ===")
miss_t = []
for fn, ln, t in sorted(titles, key=lambda x: (x[0], x[1])):
    st = cov(t)
    if "${" in t:
        st = st or "DYN"
    flag = st or "缺键"
    if not st:
        miss_t.append((fn, ln, t))
    print(f"  {flag:10s} {fn}:{ln:<7d} {t!r}")

print(f"\n=== 对话框按钮 label：抓到 {len(btns)} 条 ===")
miss_b = []
seen = set()
for fn, ln, t in sorted(btns, key=lambda x: (x[0], x[1])):
    st = cov(t)
    flag = st or "缺键"
    if not st:
        miss_b.append((fn, ln, t))
    if (t,) in seen:
        continue
    seen.add((t,))
    print(f"  {flag:10s} {fn}:{ln:<7d} {t!r}")

print(f"\n小结：标题 {len(titles)} 条中 {len(titles)-len(miss_t)} 条有键、{len(miss_t)} 条缺键")
print(f"      按钮 {len(btns)} 条中 {len(btns)-len(miss_b)} 条有键、{len(miss_b)} 条缺键")

outp = os.path.join(ROOT, "4-临时脚本", "2026-08-13-final-audit", "findings", "p_dialog_batch.json")
json.dump({"titles": [{"f": a, "l": b, "t": c, "cov": cov(c)} for a, b, c in titles],
           "buttons": [{"f": a, "l": b, "t": c, "cov": cov(c)} for a, b, c in btns]},
          open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", outp)
