# -*- coding: utf-8 -*-
r"""
probe: i18n_autoloc_sink_gap —— 「上游注入面枚举不全」的**第五形态**（只读）
==============================================================================

已确认实例（本轮兄弟探针 i18n_literal_gap 报的那条）的形状：

    上游把一个**没有键的裸英文串**送进了 i18n 通道。
    · lang 判据拿 en.json 当全集 → 看不见（en.json 里根本没这个键）
    · 硬编码 EXACT 表判据以为「凡是走 localize 的都归 lang 管」→ 主动剔除
    两个覆盖模型之间的缝。

那条实例的注入点是**语法上就挨着 localize 的**：`{{localize "Anchor"}}`、`_loc("Save Changes")`。
所以正则一抓就中。

本探针问的是这一类的**另一半**：
    Foundry 有一批**声明式自动本地化面** —— 你只是写了个对象字面量属性，
    core 在渲染时会替你调 `_loc()`。这些串**同样在 i18n 通道上**，
    但它们**语法上离 localize 十万八千里**，所以：
        · i18n_literal_gap 的正则（要求字面量紧跟 localize 括号）抓不到；
        · i18n_undeclared_key 只看形如 A.B.C 的键，裸英文也抓不到；
        · 硬编码 EXACT 表判据同样会以为「这是 i18n / 这是数据」而放过。
    ——同一条缝，另一半。

判据（三步，全机械）：

  step 1  **汇（sink）发现**：不预设名单，从三处源码里实测「哪些属性名会被喂给 localize」
          扫 core(client+common) / crucible-compiled.mjs / ember.mjs / 两边模板，
          抓  _loc(x.P) / _lf(x.P) / game.i18n.localize(x.P) / {{localize x.P}}
          得到属性名集合 SINKS（附每个属性名的 core 证据行，便于复核）。

  step 2  **源（source）发现**：在 crucible-compiled.mjs / ember.mjs 里抓对象字面量
          `P: "字符串"`（P ∈ SINKS）。

  step 3  **差集**：该字符串
              · 不在 core en.json / crucible en.json / ember en.json 的拍平键集里 →
                `_loc()` 查不到键 → 原样吐英文；
              · 不形如 A.B.C（形如键的另属 i18n_undeclared_key 那一形态）；
              · 不在本项目 cn.json 的顶层键里 / 不在 ember-hardcoded-cn.mjs 的替换表里
                → 本项目也没兜住。
          三条全中 = 界面上永远英文，且现有任何判据都看不见。

输出按「注入面」分组，并给出上下文片段供人工判定该属性是否真的落在自动本地化面上。

假阳性模式（本脚本不做判定，必须逐条回源核实）：
  A. 属性名撞名：`label` 到处都是，crucible 自己有大量**纯数据** label
     （CONFIG 表、伤害类型…）由模板 `{{x.label}}` 直接吐出、根本不过 _loc；
     这类属于「硬编码」老判据的地盘，**不是本类**，必须剔。
  B. 有些 `name:`/`title:` 是内部标识符、CSS 类、事件名、文件名，不上屏。
  C. 死代码 / 仅开发者可达路径。
  D. 值可能在运行时被别处覆盖成真键。
  E. 反过来：**漏报**——属性名从未在任何源码里以 `x.P` 形态喂给 localize
     （例如只在模板里写成 `{{localize (concat ...)}}`）就进不了 SINKS。

只读，不写库。
"""
import io
import json
import os
import re
import sys

CORE = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
FVTT = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "i18n_autoloc_sink_gap.json")

CRUCIBLE_JS = os.path.join(FVTT, "systems", "crucible", "crucible-compiled.mjs")
EMBER_JS = os.path.join(FVTT, "modules", "ember", "scripts", "ember.mjs")

EN_JSONS = [
    os.path.join(CORE, "public", "lang", "en.json"),
    os.path.join(FVTT, "systems", "crucible", "lang", "en.json"),
    os.path.join(FVTT, "modules", "ember", "lang", "en.json"),
]
CN_JSONS = [
    os.path.join(ROOT, "2-Crucible汉化插件", "lang", "cn.json"),
    os.path.join(ROOT, "1-Ember汉化插件", "lang", "cn.json"),
]
CN_JS = [
    os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs"),
    os.path.join(ROOT, "1-Ember汉化插件", "register.js"),
    os.path.join(ROOT, "2-Crucible汉化插件", "babele-register.js"),
]


def flat(o, p=""):
    s = set()
    if isinstance(o, dict):
        for k, v in o.items():
            q = f"{p}.{k}" if p else k
            s.add(q)
            s |= flat(v, q)
    return s


def read(p):
    return io.open(p, encoding="utf-8", errors="replace").read()


def walk(d, exts):
    if os.path.isfile(d):
        yield d
        return
    for b, _dn, fns in os.walk(d):
        for fn in fns:
            if os.path.splitext(fn)[1] in exts:
                yield os.path.join(b, fn)


# ---------------------------------------------------------------- step 1: sinks
SINK_JS = re.compile(
    r"(?:game\.i18n\.(?:localize|format)|\b_loc|\b_lf|\b_lformat)\s*\(\s*"
    r"([A-Za-z_$][\w$]*(?:\.[\w$]+)+)")
SINK_HBS = re.compile(r"\{\{\s*localize\s+([A-Za-z_$][\w$]*(?:\.[\w$]+)+)")

SINK_SOURCES = [
    ("core", os.path.join(CORE, "client"), {".mjs", ".js"}),
    ("core", os.path.join(CORE, "common"), {".mjs", ".js"}),
    ("crucible", CRUCIBLE_JS, {".mjs"}),
    ("crucible", os.path.join(FVTT, "systems", "crucible", "templates"), {".hbs", ".html"}),
    ("ember", EMBER_JS, {".mjs"}),
    ("ember", os.path.join(FVTT, "modules", "ember", "templates"), {".hbs", ".html"}),
]


def discover_sinks():
    sinks = {}
    for who, d, exts in SINK_SOURCES:
        for path in walk(d, exts):
            s = read(path)
            rel = os.path.relpath(path, CORE if who == "core" else FVTT)
            for rx in (SINK_JS, SINK_HBS):
                for m in rx.finditer(s):
                    prop = m.group(1).split(".")[-1]
                    if not re.match(r"^[A-Za-z_$][\w$]*$", prop):
                        continue
                    e = sinks.setdefault(prop, {"n": 0, "who": set(), "ev": []})
                    e["n"] += 1
                    e["who"].add(who)
                    if len(e["ev"]) < 3:
                        e["ev"].append(f"{rel}:{s[:m.start()].count(chr(10)) + 1}  {m.group(1)}")
    return sinks


# -------------------------------------------------------------- step 2: sources
def prop_literals(path, props):
    """抓对象字面量 `P: "str"` / `P: 'str'`（含 ||= / = 赋值形态）。"""
    s = read(path)
    alt = "|".join(sorted(re.escape(p) for p in props))
    rx = re.compile(r"(?<![\w$.])(" + alt + r")\s*(?::|=|\|\|=|\?\?=)\s*"
                    r"([\"'])((?:[^\"'\\\n]|\\.)*)\2")
    out = []
    for m in rx.finditer(s):
        ln = s[:m.start()].count("\n") + 1
        a = max(0, m.start() - 170)
        b = min(len(s), m.end() + 90)
        out.append({"prop": m.group(1), "text": m.group(3), "line": ln,
                    "ctx": s[a:b].replace("\n", " ⏎ ")})
    return out


KEYISH = re.compile(r"^[A-Za-z_$][\w$]*(\.[\w$\-]+)+$")


def main():
    en_keys = set()
    for p in EN_JSONS:
        en_keys |= flat(json.load(io.open(p, encoding="utf-8")))
    cn_keys = set()
    for p in CN_JSONS:
        cn_keys |= flat(json.load(io.open(p, encoding="utf-8")))
    cn_lit = set()
    for p in CN_JS:
        if not os.path.exists(p):
            continue
        s = read(p)
        for m in re.finditer(r"[\"'`]((?:[^\"'`\\]|\\.){2,120}?)[\"'`]\s*:", s):
            cn_lit.add(m.group(1))
        for m in re.finditer(r"game\.i18n\.translations(?:\.([\w$]+)|\[[\"']([^\"']+)[\"']\])", s):
            cn_keys.add(m.group(1) or m.group(2))

    sinks = discover_sinks()
    props = set(sinks)
    print(f"[step1] 实测出的自动本地化属性名 SINKS = {len(props)} 个")
    for p in sorted(sinks, key=lambda k: -sinks[k]["n"])[:40]:
        e = sinks[p]
        print(f"   {p:<18} x{e['n']:<4} {','.join(sorted(e['who'])):<20} {e['ev'][0]}")

    result = {"sinks": {k: {"n": v["n"], "who": sorted(v["who"]), "ev": v["ev"]}
                        for k, v in sorted(sinks.items())},
              "en_keys": len(en_keys), "cn_keys": len(cn_keys), "cand": {}}

    for name, path in (("crucible", CRUCIBLE_JS), ("ember", EMBER_JS)):
        hits = prop_literals(path, props)
        gaps = []
        for h in hits:
            t = h["text"].strip()
            if not t or len(t) > 90:
                continue
            if t in en_keys or t in cn_keys or t in cn_lit:
                continue
            if KEYISH.match(t):
                continue
            if "${" in t:
                continue
            if not re.search(r"[A-Za-z]{2,}", t):
                continue
            # 必须像给人看的英文：有大写开头的词或含空格
            if not re.search(r"[A-Z]", t) and " " not in t:
                continue
            gaps.append(h)
        result["cand"][name] = gaps
        print("=" * 90)
        print(f"[{name}] 属性字面量命中 {len(hits)} → 无键裸英文候选 {len(gaps)}")
        by = {}
        for g in gaps:
            by.setdefault(g["prop"], []).append(g)
        for p, gs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            print(f"  -- {p}  ({len(gs)})")
            for g in gs[:400]:
                print(f"     L{g['line']:<7} {g['text']!r}")

    json.dump(result, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
