import json, re, os
SP = r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\9ed168a5-8f05-4b65-86dc-0ff0ecb4407e\scratchpad"
CNDIR = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember_cn_unofficial\compendium\cn"
CJK = re.compile(r'[一-鿿]')
TAG = re.compile(r'<[^>]+>')

def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}.{k}" if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{p}[{i}]")
    elif isinstance(o, str):
        yield (p, o)

def flat(o, p=""):
    """flatten a lang json (nested dict of strings)"""
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, f"{p}.{k}" if p else k))
    elif isinstance(o, str):
        out[p] = o
    return out

print("=" * 78)
print("A. EMBER crucible-adventure coverage AFTER simulating top-key rename")
print("=" * 78)
en = json.load(open(SP + r"\en_ember_060\ember.crucible-adventure.json", encoding="utf-8"))["entries"]
cn = json.load(open(CNDIR + r"\ember.crucible-adventure.json", encoding="utf-8"))["entries"]
e = en["Ember Early Access"]; c = cn["Ember Beta Two"]
cl = {p: s for p, s in leaves(c)}
tot = cov = todo = todo_ch = 0
for p, s in leaves(e):
    if not s.strip(): continue
    tot += 1
    t = cl.get(p)
    if t and CJK.search(t): cov += 1
    else:
        todo += 1; todo_ch += len(TAG.sub(' ', s))
print(f"  strings={tot}  covered={cov} ({100*cov/tot:.1f}%)  todo={todo}  todo_chars={todo_ch}")

print()
print("  -- ember.adventure (dnd5e twin) reusing crucible CN by identical path --")
en2 = json.load(open(SP + r"\en_ember_060\ember.adventure.json", encoding="utf-8"))["entries"]
e2 = en2["Ember Early Access"]
tot2 = cov2 = 0
for p, s in leaves(e2):
    if not s.strip(): continue
    tot2 += 1
    t = cl.get(p)
    if t and CJK.search(t): cov2 += 1
print(f"  strings={tot2}  path-identical CN hit={cov2} ({100*cov2/tot2:.1f}%)")

# also try value-level TM reuse: EN-string -> CN-string map built from crucible pair
tm = {}
el = {p: s for p, s in leaves(e)}
for p, s in el.items():
    t = cl.get(p)
    if t and CJK.search(t): tm[" ".join(s.split())] = t
hit = 0
for p, s in leaves(e2):
    if not s.strip(): continue
    if " ".join(s.split()) in tm: hit += 1
print(f"  value-level TM hit={hit} ({100*hit/tot2:.1f}%)   TM size={len(tm)}")

print()
print("=" * 78)
print("B. CRUCIBLE system lang/en.json (0.10.1) vs crucible-cn lang/cn.json")
print("=" * 78)
sen = flat(json.load(open(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\lang\en.json", encoding="utf-8")))
scn = flat(json.load(open(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\crucible-cn\lang\cn.json", encoding="utf-8")))
sold = flat(json.load(open(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\crucible-cn\lang\en.json", encoding="utf-8")))
new = [k for k in sen if k not in scn]
stale = [k for k in scn if k not in sen]
drift = [k for k in sen if k in sold and sen[k] != sold[k] and k in scn]
untr = [k for k in sen if k in scn and not CJK.search(scn[k]) and scn[k] == sen[k]]
print(f"  EN keys={len(sen)}  CN keys={len(scn)}  old-EN keys={len(sold)}")
print(f"  NEW (missing in cn.json)     = {len(new)}")
print(f"  STALE (cn has, en dropped)   = {len(stale)}")
print(f"  DRIFT (EN text changed)      = {len(drift)}")
print(f"  UNTRANSLATED (cn == en)      = {len(untr)}")
print("  sample NEW:", new[:20])
print("  sample DRIFT:", drift[:12])

print()
print("=" * 78)
print("C. EMBER module lang/en.json vs ember_cn workdir lang/cn.json")
print("=" * 78)
een = flat(json.load(open(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\lang\en.json", encoding="utf-8")))
ecn_p = r"C:\Users\Taka\Desktop\fvtt\ember_cn_unofficial\lang\cn.json"
eold_p = r"C:\Users\Taka\Desktop\fvtt\ember_cn_unofficial\lang\en.json"
ecn = flat(json.load(open(ecn_p, encoding="utf-8")))
eold = flat(json.load(open(eold_p, encoding="utf-8")))
new = [k for k in een if k not in ecn]
stale = [k for k in ecn if k not in een]
drift = [k for k in een if k in eold and een[k] != eold[k] and k in ecn]
untr = [k for k in een if k in ecn and not CJK.search(ecn[k]) and ecn[k] == een[k]]
print(f"  EN keys={len(een)}  CN keys={len(ecn)}  old-EN keys={len(eold)}")
print(f"  NEW={len(new)}  STALE={len(stale)}  DRIFT={len(drift)}  UNTRANSLATED={len(untr)}")
print("  sample NEW:", new[:20])
print("  NOTE: installed ember_cn_unofficial ships NO lang folder ->",
      os.path.exists(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember_cn_unofficial\lang"))
