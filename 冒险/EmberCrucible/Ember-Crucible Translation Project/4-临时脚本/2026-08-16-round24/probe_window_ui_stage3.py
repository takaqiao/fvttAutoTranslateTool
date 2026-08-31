# -*- coding: utf-8 -*-
"""
第二十四轮 · ① stage3：REMOVED 排除

「串在 .hbs 里」还不够 —— 上游可能留下了孤儿模板文件却已不再注册/渲染它。
只有该 .hbs **仍被 ember.mjs 引用**，串才真的会出现在界面上。
逐个模板核：
  a) 模板路径是否出现在 ember.mjs（PARTS / template: / loadTemplates）
  b) 宿主 Application 类名是否以 Ember 开头（主闸 /^Ember/ 是否放行）
"""
import io, json, os, re

ROOT_EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
OUTDIR = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-16-round24"


def rd(p):
    with io.open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


ember = rd(os.path.join(ROOT_EMBER, "scripts", "ember.mjs"))
raw = json.load(io.open(os.path.join(OUTDIR, "probe_window_ui_raw.json"), encoding="utf-8"))
miss = [r for r in raw["records"] if r["judge_reports_missing"]]

tpls = sorted({h["file"] for r in miss for h in r["hbs_hits"]})
print("=" * 100)
print("涉及的模板文件 %d 个 —— 每个都核「是否仍被 ember.mjs 引用」" % len(tpls))
print("=" * 100)
orphans = []
for t in tpls:
    short = t.split("/")[-1]
    full = "modules/ember/" + t
    hits = []
    for pat, name in [(re.escape(full), "全路径"),
                      (re.escape(t), "相对路径"),
                      (re.escape(short), "文件名")]:
        for m in re.finditer(pat, ember):
            ln = ember.count("\n", 0, m.start()) + 1
            ls = ember.rfind("\n", 0, m.start()) + 1
            le = ember.find("\n", m.start())
            hits.append((name, ln, ember[ls:le if le > 0 else len(ember)].strip()[:170]))
            break
        if hits:
            break
    if hits:
        n, ln, txt = hits[0]
        print("  OK   %-52s ember.mjs:%-7d %s" % (t, ln, txt))
    else:
        orphans.append(t)
        print("  ORPHAN?? %-48s 在 ember.mjs 里找不到任何引用" % t)
print()
print("孤儿模板数 =", len(orphans))
print()

# 宿主类名（主闸 /^Ember/）
print("=" * 100)
print("宿主类：模板所属窗口的类名（主闸 /^Ember/ 放行判据）")
print("=" * 100)
for cls in ["EmberQuestEventPageSheet", "EmberPageSheet", "EmberHexHUD",
            "EmberHeroCreationSheet", "EmberDynamicTokenConfig", "EmberRandomTokenConfig",
            "EmberVistaConfiguration", "EmberCalendar"]:
    ms = [m for m in re.finditer(r"class\s+" + cls + r"\b", ember)]
    print("  %-28s %s" % (cls, ("定义于 ember.mjs:%d" % (ember.count("\n", 0, ms[0].start()) + 1))
                          if ms else "未找到"))
print()

# 逐模板 -> 它出现在哪个 PARTS 块附近（给出窗口归属证据）
print("=" * 100)
print("模板 -> 最近的上文 class 定义（窗口归属）")
print("=" * 100)
for t in tpls:
    idx = ember.find("modules/ember/" + t)
    if idx < 0:
        idx = ember.find(t)
    if idx < 0:
        print("  %-52s -- 无引用" % t); continue
    seg = ember[:idx]
    cm = None
    for m in re.finditer(r"class\s+([A-Za-z0-9_$]+)", seg):
        cm = m
    ln = ember.count("\n", 0, idx) + 1
    print("  %-52s @%-7d <- class %s" % (t, ln, cm.group(1) if cm else "?"))
