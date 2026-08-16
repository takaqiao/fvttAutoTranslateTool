# -*- coding: utf-8 -*-
"""
第二十四轮 · ① stage4：验证「给判据补什么才能不再误报」

候选修法逐个叠加，看 48 条能消掉多少。**不验证就写建议 = 又一个空转形态。**
  修法 1  语料加 modules/ember/templates/**（.hbs/.html）
  修法 2  语料加 modules/ember/scripts/*.mjs 全部（不止 ember.mjs）
  修法 3  匹配前把语料与键都做空白折叠（\\s+ -> 单空格）
"""
import io, json, os, re

ROOT_EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
ROOT_CRUC = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible"
OUTDIR = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-16-round24"


def rd(p):
    with io.open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def walk(root, exts):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in exts:
                out.append(os.path.join(dp, fn))
    return out


raw = json.load(io.open(os.path.join(OUTDIR, "probe_window_ui_raw.json"), encoding="utf-8"))
miss_keys = [r["key"] for r in raw["records"] if r["judge_reports_missing"]]

base = rd(os.path.join(ROOT_EMBER, "scripts", "ember.mjs")) + "\n" + \
       rd(os.path.join(ROOT_CRUC, "crucible-compiled.mjs"))
tpl = "\n".join(rd(p) for p in walk(os.path.join(ROOT_EMBER, "templates"),
                                    {".hbs", ".html", ".handlebars"}))
allmjs = "\n".join(rd(p) for p in walk(os.path.join(ROOT_EMBER, "scripts"), {".mjs", ".js"})
                   if os.path.basename(p) != "ember.mjs")


def fold(s):
    return re.sub(r"\s+", " ", s)


scenarios = [
    ("现状（判据只抓 .mjs 两份）", base, False),
    ("+修法1 templates", base + "\n" + tpl, False),
    ("+修法1+2 templates & 全部 ember scripts", base + "\n" + tpl + "\n" + allmjs, False),
    ("+修法1+2+3 再加空白折叠", base + "\n" + tpl + "\n" + allmjs, True),
]
print("=" * 96)
print("48 条误报在各修法下的残留数")
print("=" * 96)
for name, corpus, folding in scenarios:
    c = fold(corpus) if folding else corpus
    left = [k for k in miss_keys if (fold(k) if folding else k) not in c]
    print("  %-46s 残留 %2d  %s" % (name, len(left), [x[:45] for x in left]))
print()

# 反向副作用：修法3 会不会把本来该报的东西也吞掉？
print("=" * 96)
print("副作用检查：空白折叠会不会造出假阴性（拿一批肯定不存在的串试）")
print("=" * 96)
fakes = ["Ember Flowchart Wizard", "Outcome Identifier", "Rotate Sideways",
         "Reset Everything", "Place  Assets  Now"]
c3 = fold(base + "\n" + tpl + "\n" + allmjs)
for f in fakes:
    print("  %-30s 折叠后仍查无此串 = %s" % (f, fold(f) not in c3))
