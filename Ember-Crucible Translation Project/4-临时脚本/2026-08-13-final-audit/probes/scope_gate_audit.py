#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
探针：无作用域判据地改/删「别人的东西」。

把种子缺陷（register.js 的 degradeActorUpdatePayload 在 Adventure 导入时无条件
删掉每个 actor 更新负载的 items/effects）抽象成一条判据：

    本模块在一个 **全局的、不属于自己的** 写入点上动手，
    却没有任何「这是不是我该管的东西」的判据。

「全局写入点」在 Foundry 里只有有限的几种，本探针逐类枚举：

  A. 文档钩子（preUpdateX / preCreateX / preDeleteX）—— 对世界里**任何来源**的
     该类文档都会触发。判据 = 有没有检查 doc.type / doc.pack / game.system.id。
  B. 猴补丁（给 CONFIG.*.documentClass 的静态方法、系统 API、CONFIG 表赋值）。
  C. ready/setup 里对 game.items / game.actors / game.* 集合的批量 .update()。
  D. 语言文件：一个 module 的 lang/*.json 会被并进**全局** game.i18n.translations，
     声明了不属于自己的键 = 静默覆盖别人的翻译。
  E. babele registerMapping —— babele 文档写明「Register one **global** document
     mapping layer」，套到世界里所有被翻译的合集上，不分模块。

A/B/C 用正则在源码上定位候选，D/E 用真实数据做归属判定（可量化）。
只读，不写库。
"""
import json
import os
import re
import sys

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER = os.path.join(ROOT, "1-Ember汉化插件")
CRUC = os.path.join(ROOT, "2-Crucible汉化插件")
FDATA = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
FAPP = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"

SHIPPED_JS = [
    os.path.join(EMBER, "register.js"),
    os.path.join(EMBER, "scripts", "ember-hardcoded-cn.mjs"),
    os.path.join(EMBER, "babele-mappings.js"),
    os.path.join(CRUC, "babele-register.js"),
    os.path.join(CRUC, "babele-mappings.js"),
]

# ---------------------------------------------------------------- A/B/C
GLOBAL_WRITE_POINTS = [
    ("A-hook", re.compile(r"""Hooks\.(?:on|once)\(\s*['"](pre(?:Update|Create|Delete)\w+)['"]""")),
    ("B-monkey", re.compile(r"""(\w+)\.(updateDocuments|createDocuments|deleteDocuments)\s*=""")),
    ("B-config", re.compile(r"""(?:CONFIG|crucible\?\.CONFIG|globalThis\.crucible\?\.CONFIG|cfg)\b[^\n;]{0,80}=\s*""")),
    ("C-bulk", re.compile(r"""for\s*\(\s*const\s+\w+\s+of\s+game\.(items|actors|journal|scenes|tables|macros|cards|playlists)""")),
    ("D-i18n", re.compile(r"""game\.i18n\.translations\.\w+\s*=""")),
    ("E-mapping", re.compile(r"""registerMapping\(""")),
]

# 作用域判据：出现在同一函数体附近就算「有闸」
SCOPE_TOKENS = re.compile(
    r"""\.type\s*[=!]==|game\.system\.id|\.pack\b|documentName|"""
    r"""modules\.get\(|\.flags\?\.\[?['"]?ember|compendiumSource|PHYSICAL_ITEM_TYPES"""
)


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def fn_body_around(src, idx, span=1400):
    return src[max(0, idx - span): idx + span]


def scan_code():
    rows = []
    for path in SHIPPED_JS:
        if not os.path.exists(path):
            continue
        src = read(path)
        lines = src.split("\n")
        for kind, rx in GLOBAL_WRITE_POINTS:
            for m in rx.finditer(src):
                ln = src[:m.start()].count("\n") + 1
                ctx = fn_body_around(src, m.start())
                gated = bool(SCOPE_TOKENS.search(ctx))
                rows.append({
                    "kind": kind,
                    "file": os.path.relpath(path, ROOT),
                    "line": ln,
                    "text": lines[ln - 1].strip()[:130],
                    "scope_gate_nearby": gated,
                })
    return rows


# ---------------------------------------------------------------- D
def flat_keys(obj, prefix=""):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out |= flat_keys(v, p)
            else:
                out.add(p)
    return out


def load_keys(path):
    try:
        with open(path, encoding="utf-8") as f:
            return flat_keys(json.load(f))
    except Exception as e:
        print(f"  ! cannot read {path}: {e}", file=sys.stderr)
        return set()


def scan_lang():
    core = load_keys(os.path.join(FAPP, "public", "lang", "en.json"))
    sysk = load_keys(os.path.join(FDATA, "systems", "crucible", "lang", "en.json"))
    embk = set()
    embdir = os.path.join(FDATA, "modules", "ember", "lang")
    if os.path.isdir(embdir):
        for f in os.listdir(embdir):
            if f.endswith("en.json"):
                embk |= load_keys(os.path.join(embdir, f))
    out = {}
    for label, p in [("crucible-cn", os.path.join(CRUC, "lang", "cn.json")),
                     ("ember_cn", os.path.join(EMBER, "lang", "cn.json"))]:
        mine = load_keys(p)
        out[label] = {
            "total": len(mine),
            "owned_by_core_only": sorted(k for k in mine if k in core and k not in sysk and k not in embk),
            "owned_by_crucible": len(mine & sysk),
            "owned_by_ember": len(mine & embk),
            "owned_by_nobody": sorted(k for k in mine if k not in core and k not in sysk and k not in embk)[:60],
            "owned_by_nobody_n": len([k for k in mine if k not in core and k not in sysk and k not in embk]),
        }
    return out


# ---------------------------------------------------------------- shape
CRUCIBLE_STRING_DESC_TYPES = {
    "talent", "spell", "ancestry", "background", "taxonomy", "archetype",
}
CRUCIBLE_OBJECT_DESC_TYPES = {
    "accessory", "armor", "consumable", "loot", "schematic", "tool", "weapon",
}


def count_items_by_type():
    """量化 normalizeDescriptionValue 的爆炸半径：
    统计两个汉化仓库 compendium/cn 里、以及上游 crucible/ember 包里，
    system.description 为字符串（HTMLField）的 item 类型有多少个实例。"""
    stats = {"string_desc": 0, "object_desc": 0, "by_type": {}}
    packs = []
    for base in [os.path.join(FDATA, "systems", "crucible", "packs"),
                 os.path.join(FDATA, "modules", "ember", "packs")]:
        if os.path.isdir(base):
            packs.append(base)
    # 上游是 LevelDB，读不了；改用汉化仓库的 compendium/cn json 里的 key 形态推断。
    for repo in (EMBER, CRUC):
        d = os.path.join(repo, "compendium", "cn")
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            try:
                data = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            except Exception:
                continue
            walk_entries(data.get("entries", {}), stats, fn)
    return stats


def walk_entries(entries, stats, fn):
    it = entries.values() if isinstance(entries, dict) else entries
    for e in it:
        if not isinstance(e, dict):
            continue
        d = e.get("description")
        if isinstance(d, str):
            stats["string_desc"] += 1
            stats["by_type"].setdefault(fn, [0, 0])[0] += 1
        elif isinstance(d, dict):
            stats["object_desc"] += 1
            stats["by_type"].setdefault(fn, [0, 0])[1] += 1
        for k in ("items", "effects", "pages", "actors", "results"):
            sub = e.get(k)
            if isinstance(sub, (dict, list)):
                walk_entries(sub, stats, fn)


# ---------------------------------------------------------------- 上游事实表
# 由 crucible-compiled.mjs 逐条核实（crucible 0.10.1）：
#   CONFIG.Item.dataModels 共 13 型（:47344-47357）
#   description 是 SchemaField{public,private} 的只有 CruciblePhysicalItem 的 7 个子类
#   （:44144 accessory / :44163 armor / :44303 consumable / :44561 loot /
#     :44621 schematic / :44921 tool / :44958 weapon，基类 :22357，字段 :22372）
#   其余 6 型的 description 是**裸 HTMLField**（字符串）：
#     talent :29219 · taxonomy :42103 · archetype :42258 · ancestry :42762 ·
#     background :42936 · spell :44691
# 而 HTMLField extends StringField（fields.mjs:3949），StringField._cast = String(value)
# （fields.mjs:1705），DataField.clean 必过 _cast（fields.mjs:246）。
# 且 client-backend.mjs:248-253 在 preUpdateX 钩子**之后**再 clean 一次
#   （注释原文 "We need to clean again because data may have changed in preUpdate"）。
# ⇒ 把 {public,private} 塞给这 6 型 = 落库成字符串 "[object Object]"，不报错。
CRUCIBLE_ITEM_DESC_SHAPE = {
    "accessory": "object", "armor": "object", "consumable": "object",
    "loot": "object", "schematic": "object", "tool": "object", "weapon": "object",
    "talent": "STRING", "taxonomy": "STRING", "archetype": "STRING",
    "ancestry": "STRING", "background": "STRING", "spell": "STRING",
}

FALSE_POSITIVE_NOTES = """
本探针已知的假阳性 / 假阴性：
  * A-hook 三行被标成 GATED 是**假阳性**：SCOPE_TOKENS 在 ±1400 字符窗口里
    命中的是同文件 442 行的 `game.modules.get('babele')`，与钩子体内的作用域
    判据无关。人工核实：register.js:430-440 三个钩子体内没有任何 doc.type /
    pack / system.id 判据。
  * E-mapping 被标成 GATED 同样是窗口污染（命中 `game.modules.get`）。
    babele 的 registerMapping 按其自身文档是 "one **global** document mapping
    layer"（script/core/babele.js:225-234），本来就没有 per-module 作用域，
    不是本项目能加闸的地方——两个仓库的 DOCUMENT_MAPPINGS 逐字节相同，
    合并幂等，不构成缺陷。
  * D 段用「键是否出现在上游 en.json」判归属；上游若有动态生成的键会漏判。
    实测两个 cn.json 的 1842 / 486 个键 100% 落在各自目标包的 en.json 里，
    没有一个越界到 core 或对方包，这一支是 no-signal。
"""

if __name__ == "__main__":
    print("=" * 70)
    print("A/B/C  全局写入点 × 作用域判据")
    print("=" * 70)
    for r in scan_code():
        flag = "GATED " if r["scope_gate_nearby"] else "UNGATED"
        print(f"[{flag}] {r['kind']:10} {r['file']}:{r['line']}  {r['text']}")

    print()
    print("=" * 70)
    print("D  lang/cn.json 键归属")
    print("=" * 70)
    for label, v in scan_lang().items():
        print(f"-- {label}: {v['total']} keys; crucible-owned={v['owned_by_crucible']} "
              f"ember-owned={v['owned_by_ember']} core-only={len(v['owned_by_core_only'])} "
              f"unowned={v['owned_by_nobody_n']}")
        if v["owned_by_core_only"]:
            print("   CORE-ONLY KEYS (本模块无权改):")
            for k in v["owned_by_core_only"][:80]:
                print("     ", k)
        if v["owned_by_nobody"]:
            print("   UNOWNED sample:", v["owned_by_nobody"][:20])

    print()
    print("=" * 70)
    print("shape  description 字符串/对象两种形态在译文库里的占比")
    print("=" * 70)
    s = count_items_by_type()
    print(f"  string-shaped description leaves : {s['string_desc']}")
    print(f"  object-shaped description leaves : {s['object_desc']}")
    print()
    print("  crucible 0.10.1 的 13 个 item 型 description 形态：")
    for t, shape in sorted(CRUCIBLE_ITEM_DESC_SHAPE.items(), key=lambda kv: (kv[1] != "STRING", kv[0])):
        print(f"    {t:12} {shape}")
    print(FALSE_POSITIVE_NOTES)
