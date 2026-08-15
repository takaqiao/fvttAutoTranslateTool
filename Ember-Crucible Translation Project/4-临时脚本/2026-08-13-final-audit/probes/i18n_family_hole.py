# -*- coding: utf-8 -*-
r"""
probe: i18n_family_hole —— 「上游注入面枚举不全」在**键族**上的形态（只读）

前四支探针都在问「这个串有没有键」。这一支反过来问：

    上游的 en.json 里，形如 `PREFIX.<成员>.<字段>` 的键族，
    是不是**每个成员都齐了字段**？
    少掉的那一格，代码照样按 `PREFIX.${id}.字段` 去取 —— 取不到就把键名原样吐到屏幕上。

为什么这是同一类、又为什么至今无人看见
  · 与已报实例同源：覆盖模型（cn.json 按 en.json 逐键翻）**以 en.json 为全集**，
    而这一格 **en.json 自己就没有** → cn.json 缺它是「合规」的 →
    「lang 四项且拍平三数相等」永远绿；
  · 兄弟探针 `i18n_undeclared_key.py` 的假阳性说明第二条明写
    「模板字符串里带 ${} 的动态键无法静态判定，**已排除**」，
    而这一格恰恰只能由 `${}` 拼出来 —— 静态判据碰不到它；
  · 它在英文端也是坏的（英文客户端同样显示裸键名），
    所以「对照英文」这条人工手法也发现不了 —— 两边一样坏。
    但本项目**能修**：cn.json 里补一条键，中文端就正常了
    （en.json 没有的键，cn.json 照样能声明；本项目已用过这招）。

判据
  A. 把 en.json 拍平成 `PREFIX.成员.字段`（取倒数第二段为成员、最后一段为字段）；
  B. 对每个 PREFIX，统计各字段在成员间的出现率；
  C. 出现率 ≥ THRESH（默认 0.6）且成员数 ≥ 3 的字段，若某成员缺 → 报为「族洞」；
  D. 交叉验证：该 PREFIX 是否真的被 `${}` 动态索引过（在上游 JS 里查
     `PREFIX.${...}` / `PREFIX.` + 变量）。有动态索引 = 高危；没有 = 可能只是上游没写。

假阳性模式
  · 有些字段本来就只对部分成员有意义（例如只有远程武器才有 range 提示）；
  · 有些成员是分组节点（GROUPS/TABLE 之类），不是真成员；
  · 缺的那一格可能根本走不到（代码里有 if 挡着）—— 必须回源核可达性。

只读，不写库。
"""
import io
import json
import os
import re
import sys
from collections import defaultdict

CORE = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
FVTT = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n_family_hole.json")

PKGS = {
    "crucible": {"en": [os.path.join(FVTT, "systems", "crucible", "lang", "en.json"),
                        os.path.join(CORE, "public", "lang", "en.json")],
                 "js": os.path.join(FVTT, "systems", "crucible", "crucible-compiled.mjs"),
                 "cn": os.path.join(ROOT, "2-Crucible汉化插件", "lang", "cn.json")},
    "ember": {"en": [os.path.join(FVTT, "modules", "ember", "lang", "en.json"),
                     os.path.join(CORE, "public", "lang", "en.json")],
              "js": os.path.join(FVTT, "modules", "ember", "scripts", "ember.mjs"),
              "cn": os.path.join(ROOT, "1-Ember汉化插件", "lang", "cn.json")},
}
THRESH = 0.6
MINMEM = 3
SKIP_MEMBER = re.compile(r"^(GROUPS?|TABLE|FIELDS|TABS|SECTIONS|TYPES|LABELS)$")


def flat(o, p=""):
    d = {}
    for k, v in o.items():
        q = f"{p}.{k}" if p else k
        if isinstance(v, dict):
            d.update(flat(v, q))
        else:
            d[q] = v
    return d


def main():
    result = {}
    for who, cfg in PKGS.items():
        en = {}
        for p in cfg["en"]:
            en.update(flat(json.load(io.open(p, encoding="utf-8"))))
        cn = flat(json.load(io.open(cfg["cn"], encoding="utf-8")))
        js = io.open(cfg["js"], encoding="utf-8", errors="replace").read()

        # PREFIX -> member -> set(field)
        fam = defaultdict(lambda: defaultdict(set))
        for k in en:
            parts = k.split(".")
            if len(parts) < 3:
                continue
            pre, mem, fld = ".".join(parts[:-2]), parts[-2], parts[-1]
            if SKIP_MEMBER.match(mem):
                continue
            fam[pre][mem].add(fld)

        holes = []
        for pre, mems in sorted(fam.items()):
            if len(mems) < MINMEM:
                continue
            cnt = defaultdict(int)
            for f in mems.values():
                for x in f:
                    cnt[x] += 1
            n = len(mems)
            for fld, c in cnt.items():
                if c / n < THRESH or c == n:
                    continue
                missing = sorted(m for m, f in mems.items() if fld not in f)
                dyn = bool(re.search(re.escape(pre) + r"\.\$\{", js)) or \
                    bool(re.search(re.escape(pre) + r'\." *\+', js))
                holes.append({"prefix": pre, "field": fld, "have": c, "total": n,
                              "missing": missing, "dynamic_indexed": dyn,
                              "cn_has_any": sum(1 for k in cn if k.startswith(pre + "."))})
        holes.sort(key=lambda h: (not h["dynamic_indexed"], -h["have"] / h["total"]))
        result[who] = holes
        print("=" * 100)
        print(f"[{who}] 键族 {len(fam)} 个，族洞 {len(holes)} 处（其中被 ${{}} 动态索引过的 "
              f"{sum(1 for h in holes if h['dynamic_indexed'])} 处 ← 高危）")
        for h in holes:
            flag = "★动态索引" if h["dynamic_indexed"] else "         "
            print(f"  {flag} {h['prefix']}.<member>.{h['field']}  "
                  f"{h['have']}/{h['total']} 缺: {h['missing']}")
    json.dump(result, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
