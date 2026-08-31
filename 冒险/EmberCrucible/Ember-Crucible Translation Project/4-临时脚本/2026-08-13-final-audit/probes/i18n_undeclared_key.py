# -*- coding: utf-8 -*-
"""
probe: i18n_undeclared_key —— 「上游注入面枚举不全」的第四形态（只读）

上一形态问的是「上游把**裸英文**喂给 i18n」；这一形态问反面：
    上游把一个**形如键的字符串**喂给 i18n，但这个键**三张 en.json 里都没声明**
    （core / crucible / ember）。
Foundry 的 localize 查不到就原样返回，于是界面上直接显示 `SCENE.Preload` 这种
**裸键名**。这类东西：
  - babele 够不到（不是合集内容）；
  - 本项目的 lang 判据是拿上游 en.json 当全集对的，en.json 里根本没有这个键，
    所以 cn.json 缺它是「合规」的 —— 判据永远看不见；
  - 而它其实是本项目**最容易修**的一类：cn.json 加一条键即可。

只读，不写库。

假阳性模式：
  - 键可能由**另一个已装模块**声明（例如 dnd5e 系统的 DND5E.*）；本脚本只查
    core + crucible + ember 三张表，扫 dnd5e 分支代码时会误报，故按 DND5E. 前缀剔除；
  - 模板字符串里带 ${} 的动态键无法静态判定，已排除；
  - 某些键只在 CONFIG.debug 或开发者路径上出现。
"""
import io
import json
import os
import re
import sys

FVTT = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
CORE = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n_undeclared_key.json")


def flat(o, p=""):
    s = set()
    if isinstance(o, dict):
        for k, v in o.items():
            q = p + "." + k if p else k
            s.add(q)
            s |= flat(v, q)
    return s


KNOWN = set()
for p in (os.path.join(CORE, "public", "lang", "en.json"),
          os.path.join(FVTT, "systems", "crucible", "lang", "en.json"),
          os.path.join(FVTT, "modules", "ember", "lang", "en.json")):
    KNOWN |= flat(json.load(io.open(p, encoding="utf-8")))

Q = "[\"']"
CALL = re.compile(r"(?:game\.i18n\.(?:localize|format|has)|\b_loc|\b_lformat)\s*\(\s*" + Q + r"([^\"']+)" + Q)
HBS = re.compile(r"\{\{\s*localize\s+" + Q + r"([^\"']+)" + Q)
TIP = re.compile(r'data-tooltip="([A-Za-z_$][\w$]*(?:\.[\w$]+)+)"')
KEYISH = re.compile(r"^[A-Za-z_$][\w$]*(\.[\w$-]+)+$")

ROOTS = {
    "ember": [os.path.join(FVTT, "modules", "ember", "scripts"),
              os.path.join(FVTT, "modules", "ember", "templates")],
    "crucible": [os.path.join(FVTT, "systems", "crucible", "module"),
                 os.path.join(FVTT, "systems", "crucible", "templates")],
}


def main():
    result = {}
    for name, ds in ROOTS.items():
        miss = {}
        for d in ds:
            for b, _dn, fns in os.walk(d):
                for fn in fns:
                    if not fn.endswith((".mjs", ".js", ".hbs", ".html")):
                        continue
                    path = os.path.join(b, fn)
                    s = io.open(path, encoding="utf-8", errors="replace").read()
                    rel = os.path.relpath(path, FVTT)
                    for rx in (CALL, HBS, TIP):
                        for m in rx.finditer(s):
                            t = m.group(1).strip()
                            if "${" in t or not KEYISH.match(t):
                                continue
                            if t.startswith("DND5E."):
                                continue          # dnd5e 系统声明，本项目不负责
                            if t in KNOWN:
                                continue
                            miss.setdefault(t, []).append((rel, s[:m.start()].count("\n") + 1))
        result[name] = {k: v for k, v in sorted(miss.items())}
        print("=" * 84)
        print(f"[{name}] 引用了但 core/crucible/ember 三张 en.json 都没声明的 i18n 键：{len(miss)}")
        for k, v in sorted(miss.items()):
            print(f"  {k:<54} x{len(v):<3} {v[0][0]}:{v[0][1]}")
    json.dump(result, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
