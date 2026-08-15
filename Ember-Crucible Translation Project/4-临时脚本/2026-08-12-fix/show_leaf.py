# -*- coding: utf-8 -*-
"""渲染指定叶子的英文/中文，链接写成 «标签→目标文档»，用来人工核对。"""
import json, sys, re
sys.path.insert(0, r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-12-fix")
from uuid_fix2 import load_pairs, links, key_of

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
repo = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件"
base = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-12-fix\reports"
I = json.load(open(base + r"\ember_ids.json", encoding="utf-8"))
U = re.compile(r"@UUID\[([^\]]*)\](\{([^}]*)\})?")
T = re.compile(r"<[^>]+>")
pat = re.compile(sys.argv[1])
pack = sys.argv[2] if len(sys.argv) > 2 else "ember.crucible-adventure.json"
lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
hi = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9


def doc(t):
    r = I.get(key_of(t).split("#")[0])
    return (r["name"] if r else "?") + ("#" + t.split("#")[1] if "#" in t else "")


def ren(s):
    n = [0]
    def f(m):
        n[0] += 1
        return "«%d:%s→%s»" % (n[0] - 1, m.group(3) or "", doc(m.group(1)))
    return T.sub(" ", U.sub(f, s))


for r in load_pairs(repo, pack):
    if not r["cn"] or not pat.search(r["path"]):
        continue
    print("=" * 90)
    print(r["path"])
    print("EN:", ren(r["en"])[lo:hi])
    print()
    print("CN:", ren(r["cn"])[lo:hi])
