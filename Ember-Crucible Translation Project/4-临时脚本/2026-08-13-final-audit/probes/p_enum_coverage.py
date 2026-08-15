# -*- coding: utf-8 -*-
r"""
探针 P-ENUM：三张「叶子表」（LANGUAGES / KNOWLEDGE / ATTUNEMENTS）对上游枚举的覆盖率。

种子那一类在这里的形态：上游一批同级枚举项，汉化表只收了其中一部分。
第十一轮已经这样补过三条语言（Moiré/Borel/Kost），说明这条判据在本库确实会中。

上游枚举来源（三处，两个系统各一套 + 共享一套）：
  A crucible 侧  crucible.CONFIG.knowledge / .languages   ← ember 在 crucible-async 初始化时合并
  B dnd5e   侧  dnd5e-async.mjs 的 KNOWLEDGE_TYPES / LANGUAGES.children
  C 共享     ember.CONST.ATTUNEMENTS（定义在 dnd5e-async.mjs，两个系统都 import）

再用**实际用量**校准：统计汉化仓库 compendium/cn 里 `[[/knowledge x]]` 等标签的 id 分布，
只有真的被引用的 id 才算「会上屏」。

假阳性模式：
  FP1 上游枚举里有的 id 从来没被内容引用过 → 缺键但不会上屏，只算隐患。
  FP2 crucible 本体的知识/语言由 crucible-cn 的 lang key 翻，ember 的 patchCrucibleConfig
      只在 label 命中表时才改；若 crucible 侧 label 已是 i18n key，则不需要本表。
"""
import io, os, re, sys, json, collections

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER_UP = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
CRUC_UP = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible"
HC = os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")
src = open(HC, encoding="utf-8").read()


def tk(name):
    m = re.search(r"const %s = \{(.*?)\n\};" % name, src, re.S)
    return set(re.findall(r'"([^"]+)":\s*"', m.group(1))) if m else set()


CN = {n: tk(n) for n in ["LANGUAGES", "KNOWLEDGE", "ATTUNEMENTS", "MOODS"]}

dn = open(os.path.join(EMBER_UP, "scripts", "dnd5e-async.mjs"), encoding="utf-8").read()
em = open(os.path.join(EMBER_UP, "scripts", "ember.mjs"), encoding="utf-8").read()
cr = open(os.path.join(CRUC_UP, "crucible-compiled.mjs"), encoding="utf-8").read()


def obj(text, name):
    m = re.search(r"const %s = \{" % name, text)
    if not m:
        return ""
    i = m.end() - 1
    d, j, q = 0, i, None
    while j < len(text):
        ch = text[j]
        if q:
            if ch == "\\":
                j += 2
                continue
            if ch == q:
                q = None
        elif ch in "\"'`":
            q = ch
        elif ch == "{":
            d += 1
        elif ch == "}":
            d -= 1
            if d == 0:
                return text[i:j + 1]
        j += 1
    return ""


# --- A) dnd5e KNOWLEDGE_TYPES
kt = obj(dn, "KNOWLEDGE_TYPES")
kt_ids = dict(re.findall(r'(\w+):\s*\{label:\s*"([^"]+)"\}', kt))
# --- B) dnd5e LANGUAGES.children
lg = obj(dn, "LANGUAGES")
lg_children = dict(re.findall(r'(\w+):\s*"([^"]+)"', lg))
lg_groups = re.findall(r'label:\s*"([^"]+)"', lg)
# --- C) ATTUNEMENTS
at = obj(dn, "ATTUNEMENTS")
at_ids = dict(re.findall(r'(\w+):\s*\{id:\s*"\w+",\s*identifier:\s*"\w+",\s*label:\s*"([^"]+)"\}', at))
# --- D) crucible 本体 knowledge / languages（看 label 是不是 i18n key）
ck = obj(cr, "KNOWLEDGE")
cl = obj(cr, "LANGUAGES")
# --- E) ember 合并进 crucible.CONFIG 的
em_k = dict(re.findall(r'(\w+):\s*\{label:\s*"([^"]+)",\s*skill:', em))
em_l = dict(re.findall(r'(\w+):\s*\{label:\s*"([^"]+)",\s*category:', em))

print("=== dnd5e KNOWLEDGE_TYPES 共", len(kt_ids), "项 ===")
miss = {i: l for i, l in kt_ids.items() if l not in CN["KNOWLEDGE"]}
print("   CN KNOWLEDGE 表未收：", miss)
print("=== dnd5e LANGUAGES.children 共", len(lg_children), "项 ===")
missl = {i: l for i, l in lg_children.items() if l not in CN["LANGUAGES"]}
print("   CN LANGUAGES 表未收：", missl)
print("   语言分组 label：", lg_groups)
print("      分组 label 未收：", [g for g in lg_groups if g not in CN["LANGUAGES"]])
print("=== ATTUNEMENTS 共", len(at_ids), "项 ===")
print("   CN ATTUNEMENTS 未收 label：", {i: l for i, l in at_ids.items() if l not in CN["ATTUNEMENTS"]})
print("=== ember 合并进 crucible.CONFIG.knowledge 的：", em_k)
print("      未收：", {i: l for i, l in em_k.items() if l not in CN["KNOWLEDGE"]})
print("=== ember 合并进 crucible.CONFIG.languages 的：", len(em_l), "项")
print("      未收：", {i: l for i, l in em_l.items() if l not in CN["LANGUAGES"]})

# --- 实际用量
print("\n=== 汉化仓库 compendium/cn 里实际引用的 id ===")
use = collections.defaultdict(collections.Counter)
base = os.path.join(ROOT, "1-Ember汉化插件", "compendium", "cn")
for f in sorted(os.listdir(base)):
    if not f.endswith(".json"):
        continue
    s = open(os.path.join(base, f), encoding="utf-8").read()
    for k in ["knowledge", "language", "attunement", "ancestry", "culture", "path"]:
        for m in re.findall(r"\[\[/%s (\w+)" % k, s):
            use[k][m] += 1

for k in ["knowledge", "language", "attunement"]:
    print(f"\n  [[/{k}]] 用到 {len(use[k])} 个 id，总 {sum(use[k].values())} 次")
    tbl = {"knowledge": (kt_ids, CN["KNOWLEDGE"]), "language": (lg_children, CN["LANGUAGES"]),
           "attunement": (at_ids, CN["ATTUNEMENTS"])}[k]
    ids, cnset = tbl
    bad = []
    for i, n in use[k].most_common():
        lab = ids.get(i)
        if lab is None:
            bad.append((i, n, "上游枚举里没有这个 id"))
        elif lab not in cnset:
            bad.append((i, n, f'label "{lab}" 不在 CN 表里'))
    if bad:
        for i, n, why in bad:
            print(f"    ✘ {i} x{n}  {why}")
    else:
        print("    全部命中")

outp = os.path.join(ROOT, "4-临时脚本", "2026-08-13-final-audit", "findings", "p_enum_coverage.json")
json.dump({"kt": kt_ids, "lg": lg_children, "at": at_ids, "groups": lg_groups,
           "use": {k: dict(v) for k, v in use.items()},
           "cn": {k: sorted(v) for k, v in CN.items()}},
          open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nwrote", outp)
