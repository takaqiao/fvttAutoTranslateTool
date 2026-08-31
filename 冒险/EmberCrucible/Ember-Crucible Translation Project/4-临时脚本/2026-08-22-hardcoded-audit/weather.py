# -*- coding: utf-8 -*-
"""抽出天气配置里的全部 label（类型名 + 强度名），连同它们所属的天气类型。"""
import io, re, sys, json
sys.stdout.reconfigure(encoding='utf-8')
txt = io.open(r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs", encoding='utf-8').read()
i = txt.index('var weather$1 = {') if 'var weather$1 = {' in txt else txt.index('const weather$1 = {')
j = txt.index('\n};', i)
blk = txt[i:j]
print(f"weather$1 区段 {blk.count(chr(10))} 行")
cur = None
out = []
for line in blk.split('\n'):
    m = re.match(r'\s{2}(\w+):\s*\{', line)
    if m: cur = m.group(1)
    for lm in re.finditer(r'label:\s*"([^"]+)"', line):
        depth = len(line) - len(line.lstrip())
        out.append((cur, '类型' if depth <= 4 else '强度', lm.group(1)))
seen, uniq = set(), []
for c, k, v in out:
    if v in seen: continue
    seen.add(v); uniq.append((c, k, v))
print(f"label 共 {len(out)} 处 / 唯一 {len(uniq)}\n")
for c, k, v in uniq: print(f"  {str(c):<16}{k}  {v}")
io.open('weather.json','w',encoding='utf-8').write(json.dumps(uniq, ensure_ascii=False, indent=1))
