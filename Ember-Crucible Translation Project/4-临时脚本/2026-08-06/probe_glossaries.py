import json, os, re
CJK = re.compile(r'[一-鿿]')
CANDS = {
 'crucible_merged  (4/16)': r"C:\Users\Taka\Desktop\fvtt\glossary_crucible_merged.json",
 'adaptive_crucible(4/16)': r"C:\Users\Taka\Desktop\fvtt\glossary_adaptive_crucible.json",
 'crusible         (3/30)': r"C:\Users\Taka\Desktop\fvtt\glossary_crusible.json",
 'crucible备份       (3/22)': r"C:\Users\Taka\Desktop\fvtt\crucible备份\glossary_crusible.json",
 'ember+crucible术语表(3/22)': r"C:\Users\Taka\Desktop\ember+crucible术语表.json",
 'PF2E master      (5/9) ': r"C:\Users\Taka\Desktop\fvtt\glossary.json",
}
loaded = {}
for k, p in CANDS.items():
    if not os.path.exists(p):
        print(f"{k}  MISSING"); continue
    d = json.load(open(p, encoding='utf-8'))
    loaded[k] = d
    if isinstance(d, dict):
        vals = list(d.values())[:3]
        print(f"{k}  type=dict  entries={len(d)}  sample_key={list(d)[:2]}")
        print(f"{'':26}sample_val={json.dumps(vals[:1], ensure_ascii=False)[:220]}")
    elif isinstance(d, list):
        print(f"{k}  type=list  entries={len(d)}  sample={json.dumps(d[:1], ensure_ascii=False)[:260]}")
    print()

def to_pairs(d):
    """Normalize any glossary shape to {en: cn}."""
    out = {}
    if isinstance(d, dict):
        # shape A: {"en": "cn"}   shape B: {"en": {...,"cn"/"zh"/"translation":...}}
        for k, v in d.items():
            if isinstance(v, str):
                out[k] = v
            elif isinstance(v, dict):
                for f in ('cn', 'zh', 'zh_CN', 'translation', 'target', 'value', 'name_cn', 'cn_name'):
                    if isinstance(v.get(f), str):
                        out[k] = v[f]; break
    elif isinstance(d, list):
        for it in d:
            if not isinstance(it, dict): continue
            e = it.get('en') or it.get('source') or it.get('term') or it.get('name')
            c = it.get('cn') or it.get('zh') or it.get('translation') or it.get('target')
            if isinstance(e, str) and isinstance(c, str): out[e] = c
    return {k: v for k, v in out.items() if CJK.search(v)}

print("=" * 78)
P = {k: to_pairs(d) for k, d in loaded.items()}
for k, p in P.items():
    print(f"{k}  usable EN->CN pairs = {len(p)}")

print("\n--- pairwise overlap / conflict (crucible-family only) ---")
fam = [k for k in P if 'PF2E' not in k]
for i in range(len(fam)):
    for j in range(i + 1, len(fam)):
        a, b = P[fam[i]], P[fam[j]]
        common = set(a) & set(b)
        conf = [k for k in common if a[k] != b[k]]
        print(f"{fam[i]} vs {fam[j]}: common={len(common):>5} conflict={len(conf):>4}  e.g.{[(k, a[k], b[k]) for k in conf[:3]]}")

union = {}
for k in fam:
    for e, c in P[k].items():
        union.setdefault(e, set()).add(c)
print(f"\nUNION of crucible-family: {len(union)} EN terms, "
      f"{sum(1 for v in union.values() if len(v) > 1)} with conflicting CN")
