# -*- coding: utf-8 -*-
"""
scan_api_guard_assumptions.py  —  「判据建立在错误的 API 假设上，静默失效」的机械化扫描

判据（抽象自 register.js 的 game.world.getFlag/setFlag 幂等闸）：
  插件运行时 JS 里每一处**对外部宿主 API 的存在性判据**，都必须能在钉死的上游源码里
  找到对应成员。找不到 ⇒ 判据恒为假 / 恒为真，而写法（?. / typeof=== / if(!x) return /
  ?? {} / 空 catch）保证它**不抛异常、不打日志**，于是静默失效。

扫描对象（只读）：
  1-Ember汉化插件/register.js
  1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs
  1-Ember汉化插件/babele-mappings.js
  2-Crucible汉化插件/babele-register.js
  2-Crucible汉化插件/babele-mappings.js
  3-常用脚本/release/{generate_runtime.mjs,runtime-converters.js}
  3-常用脚本/extract/mappings.mjs

上游语料（只读，钉死版本）：
  Foundry 14.365.0  C:/Program Files/Foundry Virtual Tabletop/resources/app/{client,common}
  crucible 0.10.1   Data/systems/crucible/crucible-compiled.mjs
  ember 0.6.0       Data/modules/ember/scripts/*.mjs
  babele 2.9.1      Data/modules/babele/script/**

输出：每个「守卫点」一行，标注它引用的最末成员名在各语料里的命中数。
命中数全 0 = 强候选（该成员在任何上游里都不存在）。

已知假阳性模式（必须人工复核，不可直接采信）：
  * 成员名是本仓库自定义的（如 __emberSafePatched / __emberCnWrapped）→ 上游必然 0，不是缺陷
  * 成员名太通用（name / label / id / value）→ 上游必然大量命中，掩盖真实缺陷
  * 成员挂在 mixin / Proxy / getter 上，源码里以字符串形式定义 → grep 可能漏
  * 反过来：成员名在上游存在但**挂在别的类上**（正是 getFlag 那一条的形态）→
    grep 会命中，判据放过。所以对「存在但可疑」的还要做 owner 归属核查（第二段输出）。
"""
import os, re, sys, json, io

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
FOUNDRY = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
FDATA = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"

TARGETS = [
    os.path.join(ROOT, "1-Ember汉化插件", "register.js"),
    os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs"),
    os.path.join(ROOT, "1-Ember汉化插件", "babele-mappings.js"),
    os.path.join(ROOT, "2-Crucible汉化插件", "babele-register.js"),
    os.path.join(ROOT, "2-Crucible汉化插件", "babele-mappings.js"),
    os.path.join(ROOT, "3-常用脚本", "release", "generate_runtime.mjs"),
    os.path.join(ROOT, "3-常用脚本", "release", "runtime-converters.js"),
    os.path.join(ROOT, "3-常用脚本", "extract", "mappings.mjs"),
]

CORPORA = {
    "foundry": [os.path.join(FOUNDRY, "client"), os.path.join(FOUNDRY, "common")],
    "crucible": [os.path.join(FDATA, "systems", "crucible", "crucible-compiled.mjs")],
    "ember": [os.path.join(FDATA, "modules", "ember", "scripts")],
    "babele": [os.path.join(FDATA, "modules", "babele", "script")],
}

# 宿主根：只有从这些根出发的路径才算「外部 API」
HOST_ROOTS = ("game", "CONFIG", "ui", "foundry", "globalThis", "canvas", "Hooks",
              "crucible", "babele", "ember", "document", "CONST")

# 守卫写法
GUARD_PAT = re.compile(
    r"(\?\.|typeof\s+[\w.$?\[\]]+\s*===?\s*[\"']function[\"']|"
    r"\?\?\s*[\{\[]|catch\s*\{\s*/?|instanceof\s+Function)"
)

PATH_PAT = re.compile(
    r"\b(" + "|".join(HOST_ROOTS) + r")((?:\s*\??\.\s*[A-Za-z_$][\w$]*|\s*\??\[\s*[\"'][^\"']+[\"']\s*\])+)"
)

def load_corpus(paths):
    buf = []
    for p in paths:
        if os.path.isfile(p):
            buf.append(open(p, encoding="utf-8", errors="replace").read())
        else:
            for dp, _dn, fn in os.walk(p):
                for f in fn:
                    if f.endswith((".mjs", ".js")):
                        buf.append(open(os.path.join(dp, f), encoding="utf-8", errors="replace").read())
    return "\n".join(buf)

def main():
    corpora = {k: load_corpus(v) for k, v in CORPORA.items()}
    sizes = {k: len(v) for k, v in corpora.items()}
    print("# corpus sizes (chars):", json.dumps(sizes))

    rows = []
    for tgt in TARGETS:
        if not os.path.exists(tgt):
            print("MISSING TARGET", tgt); continue
        src = open(tgt, encoding="utf-8").read().splitlines()
        for i, line in enumerate(src, 1):
            if not GUARD_PAT.search(line):
                continue
            for m in PATH_PAT.finditer(line):
                root = m.group(1)
                tail = re.findall(r"[A-Za-z_$][\w$]*", m.group(2))
                if not tail:
                    continue
                path = root + "." + ".".join(tail)
                leaf = tail[-1]
                counts = {}
                for k, txt in corpora.items():
                    # 成员定义或使用：`leaf(`  `leaf:`  `leaf =`  `.leaf`  `"leaf"`
                    counts[k] = len(re.findall(r"[\.\"'\s]" + re.escape(leaf) + r"\b", txt))
                rows.append({
                    "file": os.path.relpath(tgt, ROOT).replace("\\", "/"),
                    "line": i, "path": path, "leaf": leaf,
                    "counts": counts,
                    "total": sum(counts.values()),
                    "src": line.strip()[:160],
                })

    # 去重（同一 path 在同一文件多行，保留全部但排序）
    rows.sort(key=lambda r: (r["total"], r["file"], r["line"]))
    print("\n## 全部守卫点（按上游命中数升序；0 命中在最前）\n")
    for r in rows:
        c = r["counts"]
        print(f'{r["total"]:>6}  f={c["foundry"]:<5} c={c["crucible"]:<5} e={c["ember"]:<5} b={c["babele"]:<5} '
              f'| {r["file"]}:{r["line"]} | {r["path"]}')
        print(f'        {r["src"]}')
    print(f"\n# total guard sites: {len(rows)}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
