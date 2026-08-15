# -*- coding: utf-8 -*-
"""
enricher_census.py  —— 探针 A：全库富文本增强器用量普查

判据（把「[[/date]] tooltip 全英文」抽象成机械判据）：
  1. 枚举 CN 合集里所有 **增强器调用**（`[[/verb ...]]` 与 `@Verb[...]`）
  2. 按 verb 归类计数
  3. 对每个 verb，人工去上游源码找它的 **输出面**（可见文本 / tooltip / aria-label /
     title / 模板 partial），检查 ember-hardcoded-cn.mjs 的 EXACT/PREFIXED/PATTERNS
     能否命中
  4. 命不中 → 该 verb 的那个面就是英文残留

只读，不写库。
"""
import json, os, re, sys, collections

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = {
    "ember": os.path.join(ROOT, "1-Ember汉化插件"),
    "crucible": os.path.join(ROOT, "2-Crucible汉化插件"),
}

PAT_BRACKET = re.compile(r"\[\[/(\w+)\b([^\]]*)\]\]")
PAT_AT = re.compile(r"@(\w+)\[([^\]]*)\]")


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(o, str):
        yield path, o


def main():
    verbs = collections.Counter()
    atverbs = collections.Counter()
    samples = collections.defaultdict(list)
    at_samples = collections.defaultdict(list)
    per_pack = collections.defaultdict(collections.Counter)
    nfiles = 0
    nstr = 0
    for repo, base in REPOS.items():
        for side in ("cn",):
            d = os.path.join(base, "compendium", side)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".json"):
                    continue
                nfiles += 1
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    data = json.load(f)
                for p, s in walk(data):
                    nstr += 1
                    for m in PAT_BRACKET.finditer(s):
                        v = m.group(1)
                        verbs[v] += 1
                        per_pack[v][fn] += 1
                        if len(samples[v]) < 4:
                            samples[v].append((fn, p[:110], m.group(0)[:90]))
                    for m in PAT_AT.finditer(s):
                        v = m.group(1)
                        atverbs[v] += 1
                        per_pack["@" + v][fn] += 1
                        if len(at_samples[v]) < 4:
                            at_samples[v].append((fn, p[:110], m.group(0)[:90]))
    print(f"# scanned files={nfiles} strings={nstr}")
    print("\n## [[/verb ...]] 计数")
    for v, c in verbs.most_common():
        print(f"{c:6d}  [[/{v}]]   packs={dict(per_pack[v])}")
        for s in samples[v]:
            print(f"          e.g. {s[0]} {s[1]} :: {s[2]}")
    print("\n## @Verb[...] 计数")
    for v, c in atverbs.most_common():
        print(f"{c:6d}  @{v}[]   packs={dict(per_pack['@'+v])}")
        for s in at_samples[v]:
            print(f"          e.g. {s[0]} {s[1]} :: {s[2]}")


if __name__ == "__main__":
    main()
