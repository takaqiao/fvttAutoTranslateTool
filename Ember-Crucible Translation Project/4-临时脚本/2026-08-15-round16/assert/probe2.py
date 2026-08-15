# -*- coding: utf-8 -*-
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
exec(open(r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-15-round16\assert\probe_gates.py", encoding="utf-8").read().split("ALL = []")[0])
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
ALL = []
for name, rel in REPOS.items():
    for row in pairs(os.path.join(ROOT, rel)):
        ALL.append((name,) + row)

def cnhits(needle, label=""):
    rows = [(r, p, path, cv) for r, p, path, ev, cv in ALL if needle in cv]
    print(f"\n### 全库中文含「{needle}」：{len(rows)} 叶 {label}")
    for r, p, path, cv in rows[:20]:
        i = cv.find(needle)
        print("   ", r, p, "|", path[:72], "|", cv[max(0,i-30):i+30].replace("\n"," "))
    return rows

cnhits("令牌")
cnhits("代币")
cnhits("六角格")
cnhits("兰蒂尔")
cnhits("月神殿")
cnhits("法珠被摧毁")
cnhits("法珠已毁")
cnhits("黑曜古")
cnhits("旋风的")
cnhits("晶片女神")

print("\n\n===== 逐条违规细看 =====")
def show(pat, req, only_pack=None, n=6):
    r = re.compile(pat, re.I)
    out=[]
    for repo, pack, path, ev, cv in ALL:
        if r.search(ev) and req not in cv:
            out.append((repo,pack,path,ev,cv))
    print(f"\n--- {pat} 缺「{req}」共 {len(out)}")
    for repo,pack,path,ev,cv in out[:n]:
        m=r.search(ev)
        print(f"  [{repo}/{pack}] {path}")
        print(f"    EN … {ev[max(0,m.start()-90):m.start()+90]}")
        print(f"    CN … {cv[:160]}")
    return out

show(r"\bCora\b","科拉")
show(r"\bAura of Life\b","灵气")
show(r"\bShard God\b","碎片之神",n=2)
show(r"\bShard Gods\b","碎片诸神",n=10)
show(r"\bTokens?\b","指示物",n=10)
