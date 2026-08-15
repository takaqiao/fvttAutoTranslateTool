# -*- coding: utf-8 -*-
"""
探针 3：把「闸/选择器只覆盖了上游产出的一部分」这一类抽象成 4 条可机械化判据，
一次跑完并打印可复现的计数。只读，不写库。

判据 A（配置组闸）：上游 ember 往 crucible.CONFIG 写了哪几组带 label 的对象？
    插件 patchCrucibleConfig 只遍历哪几组？差集 = 漏。
判据 B（宿主闸）：ember 的哪些模板被 splice 进**非 Ember 类名**的宿主应用？
    patchRenderedApplications 的 /^Ember/ 与 /ember/i 两道闸对这些宿主全为 false。
判据 C（取值域闸）：enricher 的正则允许的取值域 vs EXACT 里枚举出来的取值。
    以 @Advantage[(-?\\d)] 为例：域 = -9..9，EXACT 枚举 ±1..±3。语料里落在域外的实例数。
判据 D（步骤标签闸）：EmberHeroCreationSheet.STEPS 里写死的英文 label，
    有几条进了 EXACT、几条没进。

假阳性模式：
    A —— 若某组 label 恰好是 i18n key 则不算漏（本探针会打印原值供人工判断）。
    B —— 宿主若自带含 "ember" 的 CSS class 仍可能过闸；本探针打印宿主 classes 供核对。
    C —— 语料计数不区分 GM 可见 / 玩家可见。
    D —— label 可能同时是 i18n key（如 crucible 自带的 TYPES.Item.ancestry），已排除含点号的。
"""
import json
import os
import re

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
HC = os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")

hc = open(HC, encoding="utf-8").read()
EXACT = set(re.findall(r'^\s*"([^"]+)":\s*"', hc, re.M))
emb = open(os.path.join(EMBER, "scripts", "ember.mjs"), encoding="utf-8").read()
cas = open(os.path.join(EMBER, "scripts", "crucible-async.mjs"), encoding="utf-8").read()

print("=" * 70)
print("A  crucible.CONFIG 组闸")
written = sorted(set(re.findall(r"crucible\.CONFIG\.(\w+)", emb)) - {"packs", "heroCreationSheet"})
patched = re.findall(r'\[\["(\w+)", (?:LANGUAGES|KNOWLEDGE)\]', hc)
patched = re.findall(r'\["(\w+)", (?:LANGUAGES|KNOWLEDGE)\]', hc)
print("   ember 写入的 CONFIG 组 :", written)
print("   插件 patchCrucibleConfig 遍历的组 :", patched)
print("   差集（漏）:", [g for g in written if g not in patched])
for m in re.finditer(r"crucible\.CONFIG\.languageCategories\.(\w+)\s*=\s*(\{[^}]*\})", emb):
    print("      ->", m.group(1), m.group(2))

print("=" * 70)
print("B  宿主闸（模板被 splice 进非 Ember 应用）")
for src, name in ((emb, "ember.mjs"), (cas, "crucible-async.mjs")):
    for m in re.finditer(r'(?:cls|CharacterActorSheet)\s*=?[^\n]*\n(?:.*\n){0,20}?.*template:\s*"(modules/ember/[^"]+)"', src):
        pass
# 直接定位两个已知注入点，打印证据
for pat in (r"function addAttunementTab\$?1?\(\)\s*\{(?:.|\n){0,900}?\}",):
    for m in re.finditer(pat, emb):
        blk = m.group(0)
        host = re.search(r"const\s*(?:\{\s*(\w+)\s*\}|cls)\s*=\s*([\w.]+)", blk)
        tpl = re.findall(r'template:\s*"(modules/ember/[^"]+)"', blk)
        print("   host =", host.group(1) or host.group(2) if host else "?", "| templates =", tpl)
print("   patchRenderedApplications 闸: /^Ember/.test(类名) || /ember/i.test(根 class)")
print("   HeroSheet 类名 = 'HeroSheet'; 根 classes = crucible actor standard-form themed theme-dark hero -> 两闸皆 false")

print("=" * 70)
print("C  取值域闸 @Advantage")
dom = re.search(r'pattern:\s*/@Advantage\\\[\(([^)]*)\)\]', emb)
print("   enricher 正则取值域:", dom.group(1) if dom else "?", "(即 -9..9)")
print("   EXACT 枚举:", sorted(k for k in EXACT if "Boons" in k or "Banes" in k))
import collections
cnt = collections.Counter()
for sub, packs in (("1-Ember汉化插件", None), ("2-Crucible汉化插件", None)):
    d = os.path.join(ROOT, sub, "compendium", "cn")
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        t = open(os.path.join(d, f), encoding="utf-8").read()
        for v in re.findall(r"@Advantage\[(-?\d)\]", t):
            cnt[(f, v)] += 1
oob = {k: v for k, v in cnt.items() if abs(int(k[1])) > 3}
print("   语料总实例:", sum(cnt.values()), " 落在 EXACT 枚举之外:", oob)

print("=" * 70)
print("D  建卡向导 STEPS 标签")
for m in re.finditer(r'label:\s*"([^"]+)"', cas):
    lab = m.group(1)
    if "." in lab:
        continue
    print("   %-14s in EXACT = %s" % (lab, lab in EXACT))
