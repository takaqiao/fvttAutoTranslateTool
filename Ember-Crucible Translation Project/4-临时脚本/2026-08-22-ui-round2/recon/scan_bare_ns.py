# -*- coding: utf-8 -*-
"""
扫全部已装模块 + 系统的 `cn` 语言文件，找「顶层裸键 = 别人的命名空间」这一形态，
并按 Foundry 的装载顺序（核心 → 系统 → 模块，模块按目录名排序）判谁最后写、谁赢。

前置自证：
  A 切对条数 —— 扫到的 cn 语言文件数 == 独立数出的真值（打印出来逐个列，人工可核）；
  B 切对对象 —— 已知真值 foundry_chn 的 `TOKEN` 必须出现在结果里且被判为「赢家」；
                已知反例：crucible 系统自己的 en.json 不参与（我们只扫 cn）。
"""
import json, io, os, sys

MODULES = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules"
SYSTEMS = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems"

def lang_files(root, kind):
    out = []
    for name in sorted(os.listdir(root)):
        mj = os.path.join(root, name, "module.json" if kind == "module" else "system.json")
        if not os.path.isfile(mj):
            continue
        try:
            m = json.load(io.open(mj, encoding="utf-8"))
        except Exception:
            continue
        for l in m.get("languages", []) or []:
            if l.get("lang") in ("cn", "zh-CN", "zh-Hans"):
                p = os.path.join(root, name, l.get("path", ""))
                if os.path.isfile(p):
                    out.append((name, p))
    return out

files = lang_files(SYSTEMS, "system") + lang_files(MODULES, "module")
print("扫到 %d 份 cn 语言文件：" % len(files))
for n, p in files:
    print("   ", n, "->", os.path.relpath(p, os.path.dirname(os.path.dirname(p))))

# 逐份装载、逐份合并，复刻 mergeObject 的「两边都是对象才递归，否则覆盖」
def merge(target, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            merge(target[k], v)
        else:
            target[k] = v

def expand(d):
    out = {}
    for k, v in d.items():
        if "." in k:
            parts = k.split(".")
            cur = out
            for p in parts[:-1]:
                if not isinstance(cur.get(p), dict):
                    cur[p] = {}
                cur = cur[p]
            cur[parts[-1]] = v
        else:
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                merge(out[k], v)
            else:
                out[k] = v
    return out

state = {}
events = []          # (顺序, 来源, 命名空间, 被顶掉的键数)
for order, (name, p) in enumerate(files):
    try:
        raw = json.load(io.open(p, encoding="utf-8"))
    except Exception as e:
        print("  !! 读不了", p, e)
        continue
    exp = expand(raw)
    for k, v in exp.items():
        if isinstance(state.get(k), dict) and not isinstance(v, dict):
            # 数一下这一刀砍掉多少条叶子
            def leaves(o):
                if isinstance(o, dict):
                    return sum(leaves(x) for x in o.values())
                return 1
            events.append((order, name, k, leaves(state[k]), repr(v)[:30]))
    merge(state, exp)

print()
print("=== 命名空间被顶掉的事件（按装载顺序）：%d 起 ===" % len(events))
for order, name, ns, n, v in events:
    print("  #%-2d %-28s 顶掉命名空间 %-16s（其下 %4d 条叶子）用值 %s" % (order, name, ns, n, v))

# ---- 前置自证 B ----
assert any(n == "foundry_chn" and ns == "TOKEN" for _o, n, ns, _c, _v in events), \
    "已知真值 foundry_chn/TOKEN 没被抓到"
print()
print("PRECHECK-B OK  已知真值 foundry_chn 顶掉 TOKEN 被抓到")
print("最终 state.TOKEN 是：", type(state.get("TOKEN")).__name__, repr(state.get("TOKEN"))[:40])

json.dump([{"order": o, "module": n, "namespace": ns, "leaves_lost": c, "value": v}
           for o, n, ns, c, v in events],
          io.open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
