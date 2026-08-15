# -*- coding: utf-8 -*-
"""灵敏度回测 v3：在**归一化后的**串上外科式注入（norm 幂等，所以判据看到的文本一致）。
副本在 scratchpad，绝不碰真库 compendium/。"""
import os, re, sys, importlib.util
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SB = os.environ.get("EC_BACKTEST_COPY", "")   # 指向 compendium/ 的**副本**目录，内含 ember/ 与 crucible/ 两个子目录
assert SB, "先把两个仓库的 compendium 拷到临时目录，再把路径放进 EC_BACKTEST_COPY"
PROBE = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-13-round8\probes\scan_relation_direction.py"
spec = importlib.util.spec_from_file_location("m", PROBE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
AXES, MODES = {"dir", "mag", "unit", "unit_rt"}, {"leaf", "anchor"}

rows = m.load_pairs(os.path.join(SB, "ember")) + m.load_pairs(os.path.join(SB, "crucible"))
clean = [r for r in rows if r["cn"] and not m.check_pair(r, AXES, MODES)]
print(f"copy leaves={len(rows)}  currently-clean={len(clean)}")
assert m.norm(m.norm(clean[0]["en"])) == m.norm(clean[0]["en"]), "norm 必须幂等"


def run(label, en_rx, olds, new, side, min_en=0, max_en=10**9, want=None, win=18):
    """side='post' 在锚点数字之后找 olds，'pre' 在之前找。"""
    tried = 0
    for r in clean:
        en, cn = m.norm(r["en"]), m.norm(r["cn"])
        if not (min_en <= len(en) <= max_en):
            continue
        uniq = m.unique_numbers(en, cn)
        for mt in re.finditer(en_rx, en, re.I):
            n = mt.group(1)
            if n not in uniq or uniq[n][0] != mt.start(1):
                continue
            es, ee, cs, ce = uniq[n]
            lo, hi = (ce, ce + win) if side == "post" else (max(0, cs - win), cs)
            seg = cn[lo:hi]
            for o in olds:
                k = seg.find(o)
                if k < 0:
                    continue
                p = lo + k
                bad = cn[:p] + new + cn[p + len(o):]
                tried += 1
                row2 = dict(r, en=en, cn=bad)
                hits = m.check_pair(row2, AXES, MODES)
                tag = ",".join(sorted({f"{h['axis']}/{h['mode']}" for h in hits}))
                if want and not any(h["axis"] == want for h in hits):
                    continue    # 换个叶子再试，别拿一次失败下结论
                return dict(label=label, path=r["path"], pack=r["pack"], enlen=len(en),
                            verdict=("CAUGHT " + tag) if hits else "MISSED", tried=tried,
                            en_ctx=en[max(0, es - 60):ee + 45].strip(),
                            cn_ctx=bad[max(0, cs - 40):ce + 30].strip())
    return dict(label=label, path=None, pack="", enlen=0,
                verdict=("MISSED (试了 %d 个叶子)" % tried) if tried else "NO TARGET",
                tried=tried, en_ctx="", cn_ctx="")


CASES = [
    # 短叶（leaf 模式覆盖）
    ("dir  短叶 至少->至多", r"\bat\s+least\s+(\d+)", ["至少", "最少"], "至多", "pre", 0, 880, "dir"),
    # 长叶（只有 anchor 模式够得着）
    ("dir  长叶 至少->至多", r"\bat\s+least\s+(\d+)", ["至少", "最少"], "至多", "pre", 1500, 10**9, "dir"),
    ("dir  长叶 或以上->或以下", r"\b(\d+)\s+or\s+more\b", ["或以上", "或更多", "及以上", "以上"],
     "或以下", "post", 1200, 10**9, "dir"),
    ("dir  长叶 up to -> 至少", r"\bup\s+to\s+(\d+)", ["最多", "至多", "不超过"], "至少", "pre",
     1200, 10**9, "dir"),
    # 单位
    ("unit 英尺->英里", r"\b(\d+)\s*(?:feet|foot|ft\.?)\b", ["英尺", "呎", "尺"], "英里", "post",
     0, 10**9, "unit"),
    ("unit 小时->天", r"\b(\d+)\s+hours?\b", ["小时"], "天", "post", 0, 10**9, "unit"),
    ("unit 分钟->磅", r"\b(\d+)\s+minutes?\b", ["分钟"], "磅", "post", 0, 10**9, "unit"),
    ("unit 长叶 轮->回合", r"\b(\d+)\s+rounds?\b", ["轮"], "回合", "post", 3000, 10**9, "unit"),
    ("unit 天->年", r"\b(\d+)\s+days?\b", ["天", "日"], "年", "post", 0, 10**9, "unit"),
]

res = [run(*c) for c in CASES]
print()
for d in res:
    print(f"{d['verdict']:<26} | {d['label']}")
    if d["path"]:
        print(f"{'':26} | enLen={d['enlen']} {d['pack']} {d['path'][-64:]}")
        print(f"{'':26} |   EN …{d['en_ctx']}…")
        print(f"{'':26} |   CN …{d['cn_ctx']}…  <-- 已注错")
ok = sum(1 for d in res if d["verdict"].startswith("CAUGHT"))
print(f"\nSENSITIVITY: {ok}/{len(res)}")

# ---------------------------------------------------------------------------
# 阴性对照（合法改写必须静默）—— 与上面的阳性注入合起来才算一次完整回测。
NEG = [
    ("[阴] 英尺 -> 呎（同一单位的另一种写法）", r"\b(\d+)\s*(?:feet|foot)\b", ["英尺"], "呎", "post"),
    ("[阴] 小时 -> 分钟（同量纲换算）", r"\b(\d+)\s+hours?\b", ["小时"], "分钟", "post"),
    ("[阴] 轮 -> 轮次（同一单位的另一种写法）", r"\b(\d+)\s+rounds?\b", ["轮"], "轮次", "post"),
    ("[阴] 删掉「至少」（中文改用别的说法表达同一关系）",
     r"\bat\s+least\s+(\d+)", ["至少"], "", "pre"),
]
print("\n--- 阴性对照 ---")
nok = True
for label, rx, olds, new, side in NEG:
    d = run(label, rx, olds, new, side)
    silent = not d["verdict"].startswith("CAUGHT")
    nok &= silent
    print(f"{'OK  静默' if silent else 'FAIL 误报'} | {label}"
          + ("" if silent else f"  -> {d['verdict']} @ {d['path']}"))
print("\nSPECIFICITY(阴性对照):", "PASSED" if nok else "FAILED")
