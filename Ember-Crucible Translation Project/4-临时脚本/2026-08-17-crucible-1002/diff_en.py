"""crucible 0.10.1 基准 vs 0.10.2 实抽：逐叶差量。

⚠ 前置自证：先断言两侧包数与已知真值一致，再往下跑（本项目在「探针量错对象」上栽过四次）。
"""
import json, os, sys, io
sys.stdout.reconfigure(encoding='utf-8')

OLD = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/2-Crucible汉化插件/compendium/en"
NEW = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/289d7a82-7d7b-4b2d-ac68-1439487a5f75/scratchpad/en1002"


def leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(o, str):
        yield path, o


def load(d, fn):
    p = os.path.join(d, fn)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


old_files = {f for f in os.listdir(OLD) if f.endswith(".json") and f != "_source.json"}
new_files = {f for f in os.listdir(NEW) if f.endswith(".json") and f != "_source.json"}

# ── 前置自证 ──
src_old = json.load(open(os.path.join(OLD, "_source.json"), encoding="utf-8"))
src_new = json.load(open(os.path.join(NEW, "_source.json"), encoding="utf-8"))
assert src_old["packageVersion"] == "0.10.1", src_old["packageVersion"]
assert src_new["packageVersion"] == "0.10.2", src_new["packageVersion"]
assert len(old_files) == 15, len(old_files)
assert len(new_files) == 15, len(new_files)
print(f"前置自证 ok：基准 {src_old['packageVersion']}（{len(old_files)} 包） "
      f"vs 实抽 {src_new['packageVersion']}（{len(new_files)} 包）\n")

print(f"包集合差异：仅旧有 {sorted(old_files-new_files)} · 仅新有 {sorted(new_files-old_files)}\n")

tot_new = tot_gone = tot_chg = 0
rows = []
detail = []
for fn in sorted(old_files | new_files):
    a, b = load(OLD, fn), load(NEW, fn)
    la = dict(leaves(a)) if a else {}
    lb = dict(leaves(b)) if b else {}
    added = {k: v for k, v in lb.items() if k not in la}
    gone = {k: v for k, v in la.items() if k not in lb}
    chg = {k: (la[k], lb[k]) for k in la.keys() & lb.keys() if la[k] != lb[k]}
    tot_new += len(added); tot_gone += len(gone); tot_chg += len(chg)
    if added or gone or chg:
        rows.append((fn, len(la), len(lb), len(added), len(gone), len(chg),
                     sum(len(v) for v in added.values()),
                     sum(len(b2) for _, b2 in chg.values())))
        detail.append((fn, added, gone, chg))

print("%-38s %7s %7s %6s %6s %6s %9s" % ("pack", "旧叶", "新叶", "新增", "消失", "改动", "新增字符"))
for r in rows:
    print("%-38s %7d %7d %6d %6d %6d %9d" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
print("%-38s %7s %7s %6d %6d %6d" % ("合计", "", "", tot_new, tot_gone, tot_chg))

print("\n" + "=" * 70)
print("新增叶抽样（最多 25 条，只看有实际英文的）")
n = 0
for fn, added, gone, chg in detail:
    for k, v in added.items():
        if len(v.strip()) < 2:
            continue
        n += 1
        if n > 25:
            break
        print(f"  [{fn.replace('crucible.','').replace('.json','')}] {k[-58:]}")
        print(f"      {v[:110]!r}")
    if n > 25:
        break

print("\n" + "=" * 70)
print("英文改动抽样（最多 15 条）")
n = 0
for fn, added, gone, chg in detail:
    for k, (o, nv) in chg.items():
        n += 1
        if n > 15:
            break
        print(f"  [{fn.replace('crucible.','').replace('.json','')}] {k[-58:]}")
        print(f"      旧 {o[:90]!r}")
        print(f"      新 {nv[:90]!r}")
    if n > 15:
        break

json.dump({fn: {"added": a, "gone": g, "changed": {k: list(v) for k, v in c.items()}}
           for fn, a, g, c in detail},
          open(os.path.join(os.path.dirname(NEW), "delta_1001_to_1002.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"\n完整差量已落盘：delta_1001_to_1002.json")
