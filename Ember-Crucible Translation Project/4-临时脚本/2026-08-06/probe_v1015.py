import json, re, os
SP = r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\9ed168a5-8f05-4b65-86dc-0ff0ecb4407e\scratchpad"
R = SP + r"\repo_ember_cn"
CJK = re.compile(r'[一-鿿]')

cn = json.load(open(R + r"\compendium\cn\ember.crucible-adventure.json", encoding="utf-8"))
print("CN top keys:", list(cn["entries"].keys()))
print("\nCN mapping:")
print(json.dumps(cn.get("mapping"), ensure_ascii=False, indent=1)[:2500])

c = cn["entries"][list(cn["entries"])[0]]
print("\nCN adventure sections:", {k: (len(v) if isinstance(v, dict) else type(v).__name__) for k, v in c.items()})

# sample a journal page translation entry to see which fields are carried
j = c.get("journals", {})
shown = 0
for jn, jv in j.items():
    for pn, pv in (jv.get("pages") or {}).items():
        if isinstance(pv, dict) and set(pv) - {"name", "text"}:
            print(f"\nsample page entry [{jn} / {pn}] fields:", sorted(pv.keys()))
            for k, v in pv.items():
                s = str(v)
                print(f"   {k:<14} {s[:110]!r}")
            shown += 1
            break
    if shown >= 3:
        break

# how many page entries carry each field
fieldcount = {}
pages_total = 0
for jn, jv in j.items():
    for pn, pv in (jv.get("pages") or {}).items():
        pages_total += 1
        if isinstance(pv, dict):
            for k in pv:
                fieldcount[k] = fieldcount.get(k, 0) + 1
print(f"\nCN page entries total={pages_total}")
for k, v in sorted(fieldcount.items(), key=lambda x: -x[1]):
    print(f"   {k:<16} {v}")

print("\n--- lang/cn.json vs ember lang/en.json (v1.0.15 base) ---")
def flat(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, f"{p}.{k}" if p else k))
    elif isinstance(o, str):
        out[p] = o
    return out
een = flat(json.load(open(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\lang\en.json", encoding="utf-8")))
ecn = flat(json.load(open(R + r"\lang\cn.json", encoding="utf-8")))
eold = flat(json.load(open(R + r"\lang\en.json", encoding="utf-8")))
new = [k for k in een if k not in ecn]
drift = [k for k in een if k in eold and een[k] != eold[k] and k in ecn]
untr = [k for k in een if k in ecn and not CJK.search(ecn[k]) and ecn[k] == een[k]]
print(f"EN={len(een)} CN={len(ecn)} oldEN={len(eold)}  NEW={len(new)} DRIFT={len(drift)} UNTRANSLATED={len(untr)}")
print("sample NEW:", new[:15])
