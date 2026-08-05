import json, re
SP = r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\9ed168a5-8f05-4b65-86dc-0ff0ecb4407e\scratchpad"
CNDIR = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember_cn_unofficial\compendium\cn"
CJK = re.compile(r'[一-鿿]')

en = json.load(open(SP + r"\en_ember_060\ember.crucible-adventure.json", encoding="utf-8"))
cn = json.load(open(CNDIR + r"\ember.crucible-adventure.json", encoding="utf-8"))
print("EN top keys:", list(en["entries"].keys()))
print("CN top keys:", list(cn["entries"].keys()))
e = en["entries"][list(en["entries"])[0]]
c = cn["entries"][list(cn["entries"])[0]]
for sec in ["journals", "actors", "items", "scenes", "macros", "folders"]:
    E = e.get(sec, {}) or {}
    C = c.get(sec, {}) or {}
    common = set(E) & set(C)
    print(f"{sec:<10} EN={len(E):>5} CN={len(C):>5} common={len(common):>5} only_en={len(set(E)-set(C)):>5} only_cn={len(set(C)-set(E)):>5}")

# sample of journal names present in EN but not CN
E = e.get("journals", {}) or {}
C = c.get("journals", {}) or {}
print("\nsample EN-only journals:", list(set(E) - set(C))[:15])
print("sample CN-only journals:", list(set(C) - set(E))[:15])

print("\n--- ember.adventure (dnd5e) ---")
en2 = json.load(open(SP + r"\en_ember_060\ember.adventure.json", encoding="utf-8"))
e2 = en2["entries"][list(en2["entries"])[0]]
print("EN top key:", list(en2["entries"].keys()))
for sec in ["journals", "actors", "items", "scenes", "macros", "folders"]:
    print(f"  {sec:<10} {len(e2.get(sec, {}) or {})}")

print("\n--- crucible-character key overlap ---")
en3 = json.load(open(SP + r"\en_ember_060\ember.crucible-character.json", encoding="utf-8"))["entries"]
cn3 = json.load(open(CNDIR + r"\ember.crucible-character.json", encoding="utf-8"))["entries"]
print("only_en:", sorted(set(en3) - set(cn3))[:30])
print("only_cn:", sorted(set(cn3) - set(en3))[:30])
