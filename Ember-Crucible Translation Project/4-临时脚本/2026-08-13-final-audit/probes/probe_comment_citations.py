# -*- coding: utf-8 -*-
"""
探针：注释里对**上游**的事实断言，逐条回上游核对
=================================================

把已确认的那条实例抽象成判据：

  「本库代码里的一句注释，断言了一个关于**上游源码**的事实
   （某文件某行是什么 / 某 API 怎么写 / 某常量有几项），
   而后续的补丁与译名决策以这句话为依据 ——
   一旦这句话是错的，补丁就可能是死的，决策就建立在假前提上。」

可机械化的部分：
  C1  形如 `xxx.mjs:12345` / `xxx.mjs:123/456` 的**行号引用**
      → 直接去上游那一行读，看内容与注释描述是否对得上。
  C2  形如 `X.Y.Z` 的**上游 API 路径**出现在注释里（不在代码里）
      → 去上游 grep 这个标识符是否存在。
  C3  注释里的**数量断言**（「十五个确认框」「31 条」「11 轮」…）
      → 人工核。

只读。不写库。

假阳性模式（必须知道）：
  - 行号是**当时**版本的行号，上游升过版就会整体漂移 —— 命中失败不等于注释错，
    要看该行附近 ±40 行有没有注释描述的东西。本探针输出的是**行内容**，判定交给人。
  - 有些引用指的是本库自己的文件（不是上游），要按文件名归属过滤。
  - 注释描述的是「语义」，字面对不上不代表事实错。
"""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
CRUC = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible"
BABELE = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\babele"
CORE = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"

# 我们自己的代码面（只扫代码，不扫 PROJECT.md 之类的文档）
TARGETS = [
    r"1-Ember汉化插件\register.js",
    r"1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs",
    r"1-Ember汉化插件\babele-mappings.js",
    r"1-Ember汉化插件\styles\ember-cn.css",
    r"2-Crucible汉化插件\babele-register.js",
    r"2-Crucible汉化插件\babele-mappings.js",
    r"3-常用脚本\extract\mappings.mjs",
    r"3-常用脚本\release\runtime-converters.js",
    r"3-常用脚本\release\generate_runtime.mjs",
]

# 上游文件名 → 实际路径（多处同名时全给出）
def index_upstream():
    idx = {}
    for base in (EMBER, CRUC, BABELE, CORE):
        for dp, dn, fn in os.walk(base):
            if "node_modules" in dp and "app\\node_modules" in dp:
                continue
            for f in fn:
                if os.path.splitext(f)[1].lower() in {".mjs", ".js", ".hbs", ".css", ".less", ".json"}:
                    idx.setdefault(f.lower(), []).append(os.path.join(dp, f))
    return idx


CITE = re.compile(r"([A-Za-z0-9_\-\.]+\.(?:mjs|js|css|less|hbs))\s*[:：]\s*(\d{2,6})(?:\s*[/、,，]\s*(\d{2,6}))*")
NUMS = re.compile(r"(\d{2,6})")


def read_lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().split("\n")
    except Exception:
        return []


def main():
    idx = index_upstream()
    print(f"上游索引文件数 {sum(len(v) for v in idx.values())}")
    total_c = 0
    for rel in TARGETS:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"!! 缺文件 {rel}")
            continue
        src = open(path, encoding="utf-8").read()
        lines = src.split("\n")
        for i, line in enumerate(lines, 1):
            for m in CITE.finditer(line):
                fname = m.group(1)
                nums = NUMS.findall(m.group(0)[len(fname):])
                cands = idx.get(fname.lower(), [])
                total_c += 1
                print("=" * 100)
                print(f"[{rel}:{i}] 引用 {fname}:{','.join(nums)}")
                print(f"    注释行: {line.strip()[:200]}")
                if not cands:
                    print("    !! 上游找不到同名文件")
                    continue
                for c in cands[:2]:
                    ul = read_lines(c)
                    print(f"    -> {c}  (共 {len(ul)} 行)")
                    for n in nums:
                        n = int(n)
                        if n <= len(ul):
                            print(f"       L{n}: {ul[n-1].strip()[:220]}")
                        else:
                            print(f"       L{n}: <超出文件行数>")
    print("=" * 100)
    print(f"共 {total_c} 处行号引用")


if __name__ == "__main__":
    main()
