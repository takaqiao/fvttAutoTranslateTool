# 模拟 .github/workflows/release.yml 的 zip 排除规则，列出实际会进 module.zip 的文件
import os, fnmatch, sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件"
PATS = [".git/*", ".github/*", ".gitignore", "release/*",
        "lang/lang_keep_english.json", "lang/en.json", "compendium/en/*",
        "*.py", "scripts/__pycache__/*", "__pycache__/*", "*.pyc",
        "*.zip", "*.bak", ".DS_Store", "Thumbs.db"]

total = 0
rows = []
for root, dirs, files in os.walk(ROOT):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), ROOT).replace(os.sep, "/")
        if any(fnmatch.fnmatchcase(rel, p) for p in PATS):
            continue
        size = os.path.getsize(os.path.join(ROOT, rel))
        total += size
        rows.append((size, rel))
for s, r in sorted(rows, reverse=True):
    print(f"{s:>12,}  {r}")
print(f"{total:>12,}  TOTAL (未压缩)")
