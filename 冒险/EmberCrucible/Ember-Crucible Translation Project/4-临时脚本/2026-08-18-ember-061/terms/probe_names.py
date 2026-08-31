# -*- coding: utf-8 -*-
"""前置自证 + 抽取 0.6.1 新增名称叶。

两条断言（本项目要求「切对条数」与「切对地方」都要断言）：
  A1 切出来的名称叶条数 == 1122（任务给的已知真值）
  A2 切出来的确实是「名称位」而非正文位：
      - 每条叶的路径末段 ∈ {name, label}
      - 无一条值里含 HTML 标签 / @UUID / @Condition / @Embed（正文特征）
      - 补丁说明里点名的新专名（Ruby Grove / Kadra-Zann / Nimaelle / Vespinoth / ...）确实落在切片里
"""
import json, re, sys, io, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DELTA = r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\289d7a82-7d7b-4b2d-ac68-1439487a5f75\scratchpad\delta_em_060_to_061.json"
OUT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-18-ember-061\terms"

d = json.load(open(DELTA, encoding='utf-8'))

# --- 桶计数复核（基准自证：与任务给的三桶数字逐包相等）---
EXPECT_BUCKETS = {
    'ember.adventure.json': (804, 40, 129),
    'ember.crucible-adventure.json': (1255, 274, 204),
    'ember.crucible-adversary.json': (2, 0, 1),
}
bad = []
for pk, (ea, eg, ec) in EXPECT_BUCKETS.items():
    got = (len(d[pk]['added']), len(d[pk]['gone']), len(d[pk]['changed']))
    if got != (ea, eg, ec):
        bad.append((pk, got, (ea, eg, ec)))
print("A0 三桶复核:", "PASS" if not bad else f"FAIL {bad}")

NAME_TAIL = {'name', 'label'}
rows = []
for pk, pv in d.items():
    for k, v in pv['added'].items():
        tail = k.split('/')[-1]
        if tail in NAME_TAIL:
            rows.append({'pack': pk, 'path': k, 'tail': tail, 'en': v})

print(f"A1 名称叶条数 = {len(rows)}  (真值 1122) ->", "PASS" if len(rows) == 1122 else "FAIL")

# A2-a 路径末段
assert all(r['tail'] in NAME_TAIL for r in rows)
# A2-b 无正文特征
BODY = re.compile(r'<[a-zA-Z/!]|@UUID\[|@Condition\[|@Embed\[')
dirty = [r for r in rows if isinstance(r['en'], str) and BODY.search(r['en'])]
print(f"A2-b 含正文特征(HTML/@UUID/@Condition/@Embed)的名称叶 = {len(dirty)} ->",
      "PASS" if not dirty else "FAIL")
for r in dirty[:5]:
    print("    脏:", r['path'], '->', repr(r['en'])[:120])
# A2-c 非字符串
nonstr = [r for r in rows if not isinstance(r['en'], str)]
print(f"A2-c 非字符串值 = {len(nonstr)} ->", "PASS" if not nonstr else "FAIL")

# A2-d 补丁说明点名的新专名必须命中（证明切在了「新名字」这个地方）
# 第一版按补丁说明的拼写写，5 条未命中；查清后（见 GLOSSARY-061.md §1）改成包体真实拼写。
# 未命中的那 5 条不是切错了，是上游补丁说明与包体对不上 —— 订正记录留在这里：
#   Kryban→Kyrban · Kadra-Zann→Kadra Zann · Amersap Queen→Amerasp Queen ·
#   Casir Cats→Casir Cat · Sanguinary Warden 是 0.6.0 就有的旧 actor（不产生新叶）·
#   Caryx Savannah / Elenain Delta 只在补丁说明正文里（不产生新叶）
SENTINELS = ["Ruby Grove", "Kadra Zann", "Talei", "Nimaelle", "Maevren", "Kyrban",
             "Verno Kreed", "Vespinoth", "Grayling", "Shadebranch", "Amerasp Grove",
             "Amerasp Queen", "Casir Cat", "Proctus Caylas", "Doomsayer Shadewright"]
vals = set(r['en'] for r in rows)
allvals_sub = "\n".join(sorted(vals))
miss = [s for s in SENTINELS if s not in allvals_sub]
print(f"A2-d 补丁点名新专名命中 {len(SENTINELS)-len(miss)}/{len(SENTINELS)} ->",
      "PASS" if not miss else f"MISS {miss}")

# 值的长度分布（名称位应当短）
L = sorted(len(r['en']) for r in rows)
print(f"A2-e 名称长度: 中位 {L[len(L)//2]} 最长 {L[-1]} 总字符 {sum(L)}")
longest = sorted(rows, key=lambda r: -len(r['en']))[:5]
for r in longest:
    print("    最长:", len(r['en']), repr(r['en'])[:100])

uniq = sorted(vals)
print(f"去重后唯一名称 N = {len(uniq)}")

json.dump(rows, open(OUT + r"\names_raw.json", 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
json.dump(uniq, open(OUT + r"\names_uniq.json", 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print("written names_raw.json / names_uniq.json")
