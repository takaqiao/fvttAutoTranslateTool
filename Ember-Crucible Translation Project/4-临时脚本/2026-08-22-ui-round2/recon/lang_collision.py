# -*- coding: utf-8 -*-
"""
查「第三方核心汉化模块把整个命名空间顶掉」这一形态。

机理（与第三十四轮 A 的 `EMBER.*` 事故同型，只是跨模块）：
  Foundry 装载语言文件 = 逐份 expandObject 后按「核心 → 系统 → 各模块」顺序 mergeObject。
  mergeObject 只在**两侧都是对象**时递归，否则直接覆盖。
  ⇒ 后装载的一份里如果有一个**顶层裸键**（值是字符串）与先前建好的**命名空间对象**同名，
     那个命名空间会被整块换成一个字符串，其下全部键失效。
  ⇒ 失效之后 `game.i18n.localize()` 走 `_fallback`（英文），所以屏幕上看到的是**英文原文**，
     不是裸键 —— 这正是「按钮是中文、token 上的字是英文」那种局部英文的形状。

前置自证：
  A 切对条数 —— foundry_chn/cn.json 顶层键 177、其中值为字符串的条数必须与独立数出的一致；
  B 切对对象 —— 已知真值 `TOKEN` 必须被判为「裸键且撞上 crucible 的命名空间」，
                而已知反例 `SETTINGS`（在 foundry_chn 里是对象）必须**不**被判为裸键。
"""
import json, io, sys

FOUNDRY_CHN = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/foundry_chn/cn.json"
CRUCIBLE_EN = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/lang/en.json"
EMBER_EN = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/lang/en.json"
CRUCIBLE_CN = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/2-Crucible汉化插件/lang/cn.json"
EMBER_CN = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/lang/cn.json"

load = lambda p: json.load(io.open(p, encoding="utf-8"))

chn = load(FOUNDRY_CHN)
bare = {k: v for k, v in chn.items() if isinstance(v, str)}

# ---- 前置自证 A ----
assert len(chn) == 177, len(chn)
assert len(bare) == 98, ("裸键条数", len(bare))
print("PRECHECK-A OK  foundry_chn 顶层 %d 键 / 其中裸字符串键 %d" % (len(chn), len(bare)))

# ---- 前置自证 B ----
assert "TOKEN" in bare, "已知真值 TOKEN 不是裸键"
assert "SETTINGS" not in bare, "已知反例 SETTINGS 被误判成裸键"
print("PRECHECK-B OK  已知真值 TOKEN ∈ 裸键、已知反例 SETTINGS ∉ 裸键")


def namespaces(d):
    return {k for k, v in d.items() if isinstance(v, dict)}


def flat_namespaces(d):
    """扁平点号键文件（我们两份 cn.json）里出现过的顶层命名空间"""
    out = set()
    for k, v in d.items():
        if isinstance(v, dict):
            out.add(k)
        elif "." in k:
            out.add(k.split(".", 1)[0])
    return out


cru_en, emb_en = namespaces(load(CRUCIBLE_EN)), namespaces(load(EMBER_EN))
cru_cn, emb_cn = flat_namespaces(load(CRUCIBLE_CN)), flat_namespaces(load(EMBER_CN))

rows = []
for ns in sorted(bare):
    hit = []
    if ns in cru_en: hit.append("crucible/en")
    if ns in emb_en: hit.append("ember/en")
    if ns in cru_cn: hit.append("crucible-cn")
    if ns in emb_cn: hit.append("ember-cn")
    if hit:
        rows.append((ns, bare[ns], hit))

print()
print("=== 撞上我们两个包命名空间的裸键：%d 条 ===" % len(rows))
for ns, v, hit in rows:
    n_cru = sum(1 for k in load(CRUCIBLE_CN) if k == ns or k.startswith(ns + "."))
    n_emb = sum(1 for k in load(EMBER_CN) if k == ns or k.startswith(ns + "."))
    print("  %-24s = %-10s  撞: %-40s  我们的键: crucible-cn %d / ember-cn %d"
          % (ns, v, ",".join(hit), n_cru, n_emb))

json.dump({ns: {"value": v, "collides_with": hit} for ns, v, hit in rows},
          io.open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
