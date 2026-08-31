# -*- coding: utf-8 -*-
"""对单个叶子打印 fits / loose / 状态，用来核查判定为什么这样。"""
import json, sys, re
from collections import Counter, defaultdict
sys.path.insert(0, r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-12-fix")
from uuid_fix2 import load_pairs, links, key_of, Authority

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
repo = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件"
base = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-12-fix\reports"
auth = Authority(json.load(open(base + r"\ember_ids.json", encoding="utf-8")),
                 json.load(open(base + r"\name_map.json", encoding="utf-8")))
pat = re.compile(sys.argv[1])
pack = sys.argv[2] if len(sys.argv) > 2 else "ember.crucible-adventure.json"

for r in load_pairs(repo, pack):
    if not r["cn"] or not pat.search(r["path"]):
        continue
    el = [(t, l) for t, l, _, _ in links(r["en"]) if l]
    cl = [(t, l) for t, l, _, _ in links(r["cn"]) if l]
    ck = [key_of(t) for t, _ in cl]
    en_by_key = defaultdict(list)
    for t, l in el:
        en_by_key[key_of(t)].append(l)
    print("=" * 90); print(r["path"])
    for j, (t, l) in enumerate(cl):
        s, w = set(), set()
        for k in Counter(ck):
            for elab in (en_by_key.get(k) or [None]):
                if auth.fit(l, k, elab, strong=True):
                    s.add(k)
                if auth.fit(l, k, elab):
                    w.add(k)
        st = "MIS" if (s and ck[j] not in s and ck[j] not in w) else ("FIT" if ck[j] in w else "UNK")
        print(f"  [{j:>2}] {st} 「{l}」 on {ck[j]}({auth.en_name(ck[j])}) "
              f"strong={[auth.en_name(x) for x in s]} loose={[auth.en_name(x) for x in w]} "
              f"enlabels={en_by_key.get(ck[j])} forms={sorted(auth.forms(ck[j], (en_by_key.get(ck[j]) or [None])[0]))[:6]}")
