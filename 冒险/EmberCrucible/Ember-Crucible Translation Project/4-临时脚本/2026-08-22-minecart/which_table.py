import re, sys
sys.stdout.reconfigure(encoding='utf-8')
P = "1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs"
lines = open(P, encoding='utf-8').read().split("\n")
tabs, cur = {}, None
start = re.compile(r'^const ([A-Z_0-9]+)\s*=\s*\{')
for ln in lines:
    m = start.match(ln)
    if m:
        cur = m.group(1); tabs[cur] = set(); continue
    if cur and ln.startswith("};"):
        cur = None; continue
    if cur:
        for k in re.findall('"([^"]*)"\s*:', ln):
            tabs[cur].add(k)
want = ["DIALOG_UI","EXACT","DIALOG_TITLES","EMBER_WINDOW_UI","TOKEN_MAKER_UI"]
for n in want:
    print("%-18s %d 键" % (n, len(tabs.get(n, ()))))
print()
for k in ["Mine Cart Destination","Activate this mine cart with no passenger?",
          "Forwards","Backwards","Unreachable","Close","Confirm"]:
    w = [n for n in want if k in tabs.get(n, ())]
    print("%-46s %s" % (k, " + ".join(w) if w else "❌ 不在以上任何表"))
