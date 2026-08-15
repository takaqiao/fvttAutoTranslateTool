# -*- coding: utf-8 -*-
"""
probe_manifest_css_pack.py —— 只读探针：module.json / CSS / 打包 / 依赖声明

复现本轮四条「铁」结论所需的全部证据，一次跑完打印出来。不写任何库文件。

用法：
    python probe_manifest_css_pack.py

假阳性说明：
  - A/B 两段只是 grep 上游源码，不依赖任何启发式；若 Foundry 版本变了，行号会漂，
    但被匹配的那句 `type !== "module"` 是语义锚点，改了才说明上游行为变了。
  - D 段的字体 CJK 判据用「文件体积」代理：任何含 CJK 的字体 >= 2 MB。
    这是充分不必要条件 —— 小体积 100% 无 CJK，大体积不一定有。此处只用前者。
"""
import json, os, re, sys

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FOUNDRY_APP = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
FOUNDRY_DATA = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"

EMBER_REPO = os.path.join(PROJ, "1-Ember\u6c49\u5316\u63d2\u4ef6")
CRUC_REPO = os.path.join(PROJ, "2-Crucible\u6c49\u5316\u63d2\u4ef6")


def out(*a):
    print(*a)


def grep(path, pattern, label):
    if not os.path.exists(path):
        out("  [MISSING] %s" % path)
        return 0
    n = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if re.search(pattern, line):
                out("  %s:%d  %s" % (label, i, line.strip()[:150]))
                n += 1
    return n


def main():
    out("=" * 78)
    out("A. Foundry v14 \u5bf9 relationships.requires \u91cc type != 'module' \u7684\u6761\u76ee\u5168\u90e8\u8df3\u8fc7")
    out("=" * 78)
    hits = 0
    hits += grep(os.path.join(FOUNDRY_APP, "common", "packages", "base-package.mjs"),
                 r'type\s*!==\s*"module"', "base-package.mjs")
    hits += grep(os.path.join(FOUNDRY_APP, "client", "applications", "sidebar", "apps", "module-management.mjs"),
                 r'type\s*!==\s*"module"', "module-management.mjs")
    hits += grep(os.path.join(FOUNDRY_APP, "client", "packages", "client-package.mjs"),
                 r'type\s*!==\s*"module"', "client-package.mjs")
    out("  -> \u5171 %d \u5904\u8df3\u8fc7\u70b9\uff08\u671f\u671b >= 4\uff09" % hits)

    out("")
    out("  \u552f\u4e00\u771f\u6b63\u6821\u9a8c\u7cfb\u7edf\u7248\u672c\u7684\u901a\u9053\uff1arelationships.systems")
    grep(os.path.join(FOUNDRY_APP, "common", "packages", "base-package.mjs"),
         r"_testSupportedSystems", "base-package.mjs")
    grep(os.path.join(FOUNDRY_APP, "client", "applications", "sidebar", "apps", "module-management.mjs"),
         r"evaluateSystemCompatibility|relationships\.systems", "module-management.mjs")

    out("")
    out("=" * 78)
    out("B. \u6a21\u5757 CSS \u7684 @layer \u5f52\u5c5e\uff08view.mjs + main.hbs\uff09")
    out("=" * 78)
    grep(os.path.join(FOUNDRY_APP, "dist", "server", "views", "view.mjs"),
         r'void 0===s\?"modules"', "view.mjs(minified)")
    grep(os.path.join(FOUNDRY_APP, "templates", "views", "layouts", "main.hbs"),
         r"@import", "main.hbs")
    grep(os.path.join(FOUNDRY_APP, "common", "packages", "base-package.mjs"),
         r"_migrateStyles|deprecated since v13", "base-package.mjs")

    out("")
    out("=" * 78)
    out("C. \u4e24\u4e2a module.json \u7684\u5173\u952e\u5b57\u6bb5")
    out("=" * 78)
    for label, repo in (("ember_cn_unofficial", EMBER_REPO), ("crucible-cn", CRUC_REPO)):
        p = os.path.join(repo, "module.json")
        d = json.load(open(p, encoding="utf-8"))
        out("  --- %s ---" % label)
        for k in ("id", "version", "compatibility", "relationships", "styles", "bugs", "changelog", "title"):
            out("    %-14s %s" % (k, json.dumps(d.get(k, "<absent>"), ensure_ascii=False)))
    for label, p in (("crucible(system)", os.path.join(FOUNDRY_DATA, "systems", "crucible", "system.json")),
                     ("ember(module)", os.path.join(FOUNDRY_DATA, "modules", "ember", "module.json")),
                     ("babele", os.path.join(FOUNDRY_DATA, "modules", "babele", "module.json"))):
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        out("  --- \u4e0a\u6e38 %s ---" % label)
        out("    version        %s" % d.get("version"))
        out("    compatibility  %s" % json.dumps(d.get("compatibility"), ensure_ascii=False))
        out("    relationships  %s" % json.dumps(d.get("relationships", "<absent>"), ensure_ascii=False)[:300])
        out("    styles         %s" % json.dumps(d.get("styles", "<absent>"), ensure_ascii=False))

    out("")
    out("=" * 78)
    out("D. CSS\uff1a\u884c\u9ad8\u3001\u6b7b\u9009\u62e9\u5668\u3001CJK \u56de\u9000")
    out("=" * 78)
    ember_css = os.path.join(FOUNDRY_DATA, "modules", "ember", "styles", "ember.css")
    out("  \u4e0a\u6e38\u5728\u6b63\u6587\u5bb9\u5668\u4e0a\u81ea\u5df1\u5199\u4e86 line-height\uff08\u7ee7\u627f\u6c38\u8fdc\u8f93\u7ed9\u81ea\u8eab\u58f0\u660e\uff09\uff1a")
    grep(ember_css, r"journal-page-content \{|line-height: var\(--font-size-20\)", "ember.css")
    out("  \u672c\u5e93\u628a line-height \u5199\u5728\u4e86\u5916\u5c42\u5bb9\u5668\u4e0a\uff1a")
    grep(os.path.join(EMBER_REPO, "styles", "ember-cn.css"), r"journal-entry-content|line-height", "ember-cn.css")
    out("  ember-content \u4f5c\u4e3a\u72ec\u7acb class \u5728 ember 0.6.0 \u91cc\u7684\u51fa\u73b0\u6b21\u6570\uff1a")
    n = 0
    for root, _dirs, files in os.walk(os.path.join(FOUNDRY_DATA, "modules", "ember")):
        for fn in files:
            if not fn.endswith((".mjs", ".hbs", ".css", ".less")):
                continue
            try:
                s = open(os.path.join(root, fn), encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            n += len(re.findall(r"ember-content(?![-\w])", s))
    out("    -> %d \u6b21\uff080 \u5373\u6b7b\u9009\u62e9\u5668\uff09" % n)

    out("  Crucible \u5b57\u4f53\u6808\u4e0e\u5b57\u4f53\u6587\u4ef6\u4f53\u79ef\uff08<2MB \u5fc5\u65e0 CJK\uff09\uff1a")
    grep(os.path.join(FOUNDRY_DATA, "systems", "crucible", "styles", "crucible.css"),
         r"--font-(h[123]|body|quote|sans):", "crucible.css")
    fdir = os.path.join(FOUNDRY_DATA, "systems", "crucible", "fonts")
    for root, _dirs, files in os.walk(fdir):
        for fn in sorted(files):
            if fn.endswith((".ttf", ".otf", ".woff2")):
                out("    %-46s %8.1f KB" % (fn, os.path.getsize(os.path.join(root, fn)) / 1024))
    out("  crucible-cn \u58f0\u660e\u7684 styles: %s" %
        json.dumps(json.load(open(os.path.join(CRUC_REPO, "module.json"), encoding="utf-8")).get("styles", "<absent>")))


if __name__ == "__main__":
    sys.exit(main())
