# -*- coding: utf-8 -*-
r"""
probe: i18n_dynamic_key_family —— 「上游注入面枚举不全」这一类里**被所有兄弟探针明文排除**的那一半
==========================================================================================

同一类，不同一半
----------------
已报实例是「裸英文进了 i18n 通道，en.json 里没这个键」。
本探针问的是**同一条缝的动态形态**：

    上游用**模板串拼键**：`_loc(`PREFIX.${x}`)`。
    en.json 声明了 PREFIX.a / PREFIX.b / PREFIX.c，
    但 x 在运行时还能取到 d —— `PREFIX.d` 没人声明过。
    Foundry 的 localize 查不到就**原样返回键名**，屏幕上直接出现 `PREFIX.d`。

为什么至今无人看见（这一条的成因，与已报那条同构）
  · `i18n_undeclared_key.py` 的假阳性说明第二条白纸黑字写：
      「模板字符串里带 ${} 的动态键无法静态判定，**已排除**」
  · `i18n_literal_gap.py` 的 gap 过滤里 `if KEYISH.match(t): continue`，
      而拼出来的键连字面量都不是，正则连碰都碰不到；
  · `i18n_sink_gap` / `i18n_slot_gap` / `i18n_registration_sink_gap` 三支都只吃**字面量**；
  · 本项目的 lang 判据是「cn.json 的键集 == en.json 的键集」，
      而这类东西**en.json 自己就缺**，两边一样缺 → 三数相等，判据永远绿。
  一条规则对「静态串」成立、对「拼接串」恰好相反 —— 与已报那条「有键的走 lang / 无键的两边落空」
  是同一种缝。

判据（四步，可机械化 + 可人工复核）
  A. 抓所有**拼键点**：`_loc`/`_lf`/`game.i18n.localize|format|has` 的第一参含 `${}` 的模板串，
     或 `"字面量" + 变量` 的加法拼接；解析出静态前缀 PREFIX 与静态后缀 SUFFIX。
  B. 对每个 PREFIX，从三张 en.json（core / crucible / ember）里取出**已声明族** `PREFIX.*`。
  C. 推断 `${x}` 的**运行时取值域**：本脚本做保守推断 ——
     若 x 形如 `A.B.id` / `this.type` / `xxx.key`，就在同一份源码里找同名枚举/CONFIG 的键集；
     另外把「同一 PREFIX 在别处以**字面量**形式出现过的键」并进来。
     推断不出来的标 `domain=?`，交人工。
  D. 差集：域里有、族里无 → 该值一旦出现，屏幕上就是裸键名。

假阳性模式（脚本不判，人工核）
  · 取值域推断可能过宽（枚举里有些成员走不到这个分支）；
  · 有些分支是死代码 / 仅 GM 调试；
  · plural 选择器（`plurals.select()`）的域取决于目标语言的复数规则，
    英语只有 one/other，中文只有 other —— **英语跑得通不代表中文跑得通**，这一支要单列；
  · 键可能由第三方模块声明。

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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n_dynamic_key_family.json")

SRC = {
    "crucible": [os.path.join(FVTT, "systems", "crucible", "crucible-compiled.mjs")],
    "ember": [os.path.join(FVTT, "modules", "ember", "scripts", "ember.mjs")],
}
TPL = {
    "crucible": os.path.join(FVTT, "systems", "crucible", "templates"),
    "ember": os.path.join(FVTT, "modules", "ember", "templates"),
}
EN = {"core": os.path.join(CORE, "public", "lang", "en.json"),
      "crucible": os.path.join(FVTT, "systems", "crucible", "lang", "en.json"),
      "ember": os.path.join(FVTT, "modules", "ember", "lang", "en.json")}
CN = {"crucible": os.path.join(ROOT, "2-Crucible汉化插件", "lang", "cn.json"),
      "ember": os.path.join(ROOT, "1-Ember汉化插件", "lang", "cn.json")}


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


# `_loc(`A.B.${x}`)` 或 game.i18n.localize(`...`) / format / has
TMPL = re.compile(
    r"(?:game\.i18n\.(?:localize|format|has)|\b_loc|\b_lf|\b_lformat)\s*\(\s*`([^`]*\$\{[^`]*)`")
# `_loc("A.B." + x)`
CONCAT = re.compile(
    r"(?:game\.i18n\.(?:localize|format|has)|\b_loc|\b_lf|\b_lformat)\s*\(\s*"
    r"([\"'])([A-Za-z_$][\w$.]*\.)\1\s*\+\s*([A-Za-z_$][\w$.\[\]]*)")


def parse_tmpl(t):
    """把 `A.B.${x}.C` 拆成 (prefix, expr, suffix)。只处理恰好一个 ${}."""
    parts = re.findall(r"\$\{([^}]*)\}", t)
    if len(parts) != 1:
        return None
    i = t.index("${")
    j = t.index("}", i)
    return t[:i], parts[0].strip(), t[j + 1:]


def main():
    enk = {k: flat(json.load(io.open(v, encoding="utf-8"))) for k, v in EN.items()}
    allen = set().union(*enk.values())
    cnk = {k: flat(json.load(io.open(v, encoding="utf-8"))) for k, v in CN.items()}
    allcn = set().union(*cnk.values())

    result = {}
    for who, files in SRC.items():
        sites = []
        for path in files:
            s = read(path)
            for m in TMPL.finditer(s):
                p = parse_tmpl(m.group(1))
                if not p:
                    sites.append({"raw": m.group(1), "prefix": None, "expr": None, "suffix": None,
                                  "line": s[:m.start()].count("\n") + 1,
                                  "ctx": s[max(0, m.start() - 150):m.end() + 90].replace("\n", " ⏎ ")})
                    continue
                pre, expr, suf = p
                sites.append({"raw": m.group(1), "prefix": pre, "expr": expr, "suffix": suf,
                              "line": s[:m.start()].count("\n") + 1,
                              "ctx": s[max(0, m.start() - 150):m.end() + 90].replace("\n", " ⏎ ")})
            for m in CONCAT.finditer(s):
                sites.append({"raw": m.group(2) + "${" + m.group(3) + "}", "prefix": m.group(2),
                              "expr": m.group(3), "suffix": "",
                              "line": s[:m.start()].count("\n") + 1,
                              "ctx": s[max(0, m.start() - 150):m.end() + 90].replace("\n", " ⏎ ")})
        # 模板里的 (concat …) 拼键
        for b, _dn, fns in os.walk(TPL[who]):
            for fn in fns:
                if not fn.endswith((".hbs", ".html")):
                    continue
                fp = os.path.join(b, fn)
                t = read(fp)
                for m in re.finditer(r"\{\{\s*localize\s*\(\s*concat\s+([^)]*)\)", t):
                    sites.append({"raw": "(concat " + m.group(1).strip() + ")", "prefix": None,
                                  "expr": m.group(1).strip(), "suffix": None,
                                  "line": t[:m.start()].count("\n") + 1,
                                  "ctx": os.path.relpath(fp, FVTT)})

        # 归并到 PREFIX 家族
        fams = {}
        for st in sites:
            pre = st["prefix"]
            if not pre:
                fams.setdefault("<unparsed>", []).append(st)
                continue
            fams.setdefault(pre.rstrip("."), []).append(st)

        print("=" * 100)
        print(f"[{who}] 动态拼键点 {len(sites)} 个，落在 {len(fams)} 个前缀家族上")
        rows = []
        for pre, sts in sorted(fams.items()):
            if pre == "<unparsed>":
                continue
            decl = sorted(k[len(pre) + 1:] for k in allen if k.startswith(pre + ".")
                          and "." not in k[len(pre) + 1:])
            decl_deep = sorted(k for k in allen if k.startswith(pre + "."))
            cnhas = sorted(k for k in allcn if k.startswith(pre + "."))
            rows.append({"prefix": pre, "declared": decl, "declared_all": len(decl_deep),
                         "cn": len(cnhas),
                         "sites": [{"line": x["line"], "expr": x["expr"], "suffix": x["suffix"],
                                    "raw": x["raw"], "ctx": x["ctx"]} for x in sts]})
            print(f"  ── {pre}")
            print(f"     en.json 已声明直接子键 {len(decl)}: {decl[:14]}{' …' if len(decl) > 14 else ''}")
            for x in sts:
                print(f"     site L{x['line']:<7} ${{{x['expr']}}}  suffix={x['suffix']!r}")
        if "<unparsed>" in fams:
            print(f"  ── <多 ${{}} 或无法解析> {len(fams['<unparsed>'])} 处")
            for x in fams["<unparsed>"][:40]:
                print(f"     L{x['line']:<7} {x['raw']!r}")
        result[who] = rows
    json.dump(result, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
