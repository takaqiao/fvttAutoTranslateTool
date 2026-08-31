# -*- coding: utf-8 -*-
"""
从 bestiary entries 抽 English→Chinese 名称映射, 然后扫 journals 查不一致.

bestiary key 是 English (canonical), value.name 是 "中文 English" 格式.
"""
import json, re, os
from collections import defaultdict, Counter

ROOT = r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW"
bestiary_path = os.path.join(ROOT, "pf2e.fists-of-the-ruby-phoenix-bestiary.json")

with open(bestiary_path, encoding="utf-8") as f:
    bestiary = json.load(f)

# 1. 从 actor.name 抽 中文部分
# 格式: "中文 English Name"  (English 是 ASCII 段)
# 用正则: 把末尾的 ASCII 段(+空格+括号) 当作 English
actor_chinese_map = {}  # english_key → chinese_label
for en_key, actor in bestiary.get("entries", {}).items():
    if not isinstance(actor, dict): continue
    full_name = actor.get("name", "")
    # full_name = "中文 English Name"; 切到第一段 ASCII (含 () 数字 空格 大写) 开头
    # 简化: 匹配末尾 [A-Za-z\s\(\)\d\-\'’]+  (English Name)
    m = re.search(r"([A-Z][A-Za-z\(\)\d\s\-\'’]*)\s*$", full_name)
    if m:
        en_part = m.group(1).strip()
        zh_part = full_name[:m.start()].strip()
        # 跳过 zh_part 为空的 (说明 actor 还没翻译 或 是 'Bul-Gae' 这种没有 zh_label 的)
        if zh_part:
            actor_chinese_map[en_key] = (zh_part, en_part)

print(f"Bestiary entries with bilingual name: {len(actor_chinese_map)}")

# 抽取每个 actor 内 items 的双语对照 (items.name = "中文 English ability" 通常)
# items 里有 actions, spells, equipment, conditions 等
item_chinese_map = defaultdict(set)  # english → set of chinese renderings
for en_key, actor in bestiary.get("entries", {}).items():
    if not isinstance(actor, dict): continue
    items = actor.get("items", {})
    if isinstance(items, dict):
        for it_key, it in items.items():
            if not isinstance(it, dict): continue
            it_name = it.get("name", "")
            m = re.search(r"([A-Z][A-Za-z\(\)\d\s\-\'’,]*)\s*$", it_name)
            if m:
                en_part = m.group(1).strip()
                zh_part = it_name[:m.start()].strip()
                if zh_part and en_part and len(en_part) >= 3:
                    item_chinese_map[en_part].add(zh_part)
    elif isinstance(items, list):
        for it in items:
            if not isinstance(it, dict): continue
            it_name = it.get("name", "")
            m = re.search(r"([A-Z][A-Za-z\(\)\d\s\-\'’,]*)\s*$", it_name)
            if m:
                en_part = m.group(1).strip()
                zh_part = it_name[:m.start()].strip()
                if zh_part and en_part and len(en_part) >= 3:
                    item_chinese_map[en_part].add(zh_part)

print(f"Item-level English terms: {len(item_chinese_map)}")

# 输出
with open("bestiary_name_map.txt", "w", encoding="utf-8") as out:
    out.write("# Bestiary Actor 中-英 映射\n\n")
    for en_key, (zh, en_part) in sorted(actor_chinese_map.items()):
        out.write(f"  {en_key:50} → 中: {zh}  | EN片段: {en_part}\n")
    out.write("\n\n# Item-level 多重渲染候选 (跨 actor 同一 ability)\n\n")
    multi = {k: v for k, v in item_chinese_map.items() if len(v) > 1}
    out.write(f"找到 {len(multi)} 个多渲染 item term:\n\n")
    for en, zhs in sorted(multi.items(), key=lambda x: -len(x[1])):
        out.write(f"  {en:40} → {zhs}\n")

print("Output: bestiary_name_map.txt")
print(f"  Actor map: {len(actor_chinese_map)}")
print(f"  Multi-rendering items: {sum(1 for v in item_chinese_map.values() if len(v) > 1)}")
