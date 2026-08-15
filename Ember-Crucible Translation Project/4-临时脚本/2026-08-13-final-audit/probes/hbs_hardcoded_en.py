# -*- coding: utf-8 -*-
"""
hbs_hardcoded_en.py —— 「闸/选择器失配」子类 D5：闸挡在门外的模板

patchRenderedApplications 的闸：
    /ember/i.test(root.className) || /^Ember/.test(app.constructor.name)
Ember 自己的 28 个应用类**全部**叫 Ember*，所以闸对它们是通的。
但 Ember 还把自己的模板**注入到别人的 sheet**（crucible HeroSheet / dnd5e
CharacterActorSheet）——那些 app 的 constructor.name 是 HeroSheet /
CharacterActorSheet，root class 是 ["crucible","actor",...]，两个条件都不满足，
闸把整块挡在门外。

本脚本列出 ember/templates/ 下每个 .hbs 里的裸英文文本节点与裸英文属性
（aria-label / data-tooltip / placeholder / alt），并标出该模板属于哪个宿主。
只读。
"""
import os
import re
import sys

TPL = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\templates"

# 宿主判定：哪些模板被注入到非 Ember* 的应用里
FOREIGN_HOSTS = {
    r"crucible\tab-attunement.hbs": "crucible HeroSheet (classes=[crucible,actor,...])",
    r"applications\dnd5e\actor\tabs\attunement.hbs": "dnd5e CharacterActorSheet",
    r"applications\dnd5e\actor\knowledge-config.hbs": "dnd5e 应用",
    r"applications\advancement\knowledge\details.hbs": "dnd5e Advancement",
}

HANDLEBARS = re.compile(r"\{\{[^}]*\}\}")
TAGRE = re.compile(r"<[^>]+>")
ATTR = re.compile(r'(aria-label|data-tooltip|placeholder|alt|title)\s*=\s*"([^"]*)"')


def text_nodes(s):
    """去掉 handlebars 表达式与标签后残留的可见文本"""
    s2 = HANDLEBARS.sub("\x00", s)
    s2 = TAGRE.sub("\n", s2)
    out = []
    for line in s2.split("\n"):
        t = line.replace("\x00", "").strip()
        if len(t) >= 3 and re.search(r"[A-Za-z]{3,}", t):
            out.append(t)
    return out


def main():
    tot = 0
    for root, _ds, fs in os.walk(TPL):
        for f in sorted(fs):
            if not f.endswith((".hbs", ".html")):
                continue
            fp = os.path.join(root, f)
            rel = fp.replace(TPL + "\\", "")
            s = open(fp, encoding="utf-8").read()
            hits = text_nodes(s)
            attrs = [(k, v) for k, v in ATTR.findall(s)
                     if re.search(r"[A-Za-z]{3,}", HANDLEBARS.sub("", v))]
            if not hits and not attrs:
                continue
            host = FOREIGN_HOSTS.get(rel)
            mark = f"  ←← 宿主非 Ember*：{host}" if host else ""
            print(f"\n### {rel}{mark}")
            for h in hits:
                print(f"    text | {h[:120]}")
                tot += 1
            for k, v in attrs:
                print(f"    @{k} | {v[:120]}")
                tot += 1
    print(f"\n合计 {tot} 条裸英文")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
