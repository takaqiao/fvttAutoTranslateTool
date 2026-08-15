# -*- coding: utf-8 -*-
"""
config_keylist_gap.py —— 「闸/选择器失配」子类 D6：改写器的 key 列表漏了兄弟键

patchCrucibleConfig 的选择器是一张**写死的 key 列表**：
    for (const [key, table] of [["languages", LANGUAGES], ["knowledge", KNOWLEDGE]])
凡是 ember 往 crucible.CONFIG 里塞的、但不在这两个 key 下的裸英文 label，
这个改写器原理上够不到。

判据：扫 ember.mjs 里所有对 `crucible.CONFIG.<key>` 的写入（直接赋值或
Object.assign），抽出其中形如 `label: "英文"` 的裸串，标出 <key> 是否在
patchCrucibleConfig 的 key 列表里。同样的写法也适用于 CONFIG.DND5E.*。

假阳性：有些 label 其实是 i18n key（"XXX.Yyy" 形状），crucible 的
localizeConfigObject 会翻掉，不算缺陷 —— 脚本按含点且无空格过滤掉。
只读。
"""
import re
import sys

EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
PLUGIN = (r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
          r"\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs")

WRITE = re.compile(r"crucible\.CONFIG\.(\w+)")
LABEL = re.compile(r'label:\s*"([^"]+)"')


def patched_keys():
    src = open(PLUGIN, encoding="utf-8").read()
    m = re.search(r'for \(const \[key, table\] of \[(.*?)\]\)', src, re.S)
    return set(re.findall(r'\["(\w+)"', m.group(1))) if m else set()


def main():
    keys = patched_keys()
    print("patchCrucibleConfig 的 key 列表 =", sorted(keys))
    L = open(EMBER, encoding="utf-8").read().split("\n")
    hits = {}
    for i, l in enumerate(L):
        for m in WRITE.finditer(l):
            k = m.group(1)
            # 同行 + 之后 40 行内的 label: "..."（Object.assign 块）
            blk = "\n".join(L[i:i + 40])
            for lm in LABEL.finditer(blk):
                v = lm.group(1)
                if re.fullmatch(r"[\w.]+", v) and "." in v:   # i18n key，跳过
                    continue
                hits.setdefault(k, set()).add(v)
    for k in sorted(hits):
        tag = "已覆盖" if k in keys else ">>> key 列表里没有，改写器够不到"
        print(f"\ncrucible.CONFIG.{k}  [{tag}]  {len(hits[k])} 条裸英文 label")
        for v in sorted(hits[k])[:40]:
            print("     ", v)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
