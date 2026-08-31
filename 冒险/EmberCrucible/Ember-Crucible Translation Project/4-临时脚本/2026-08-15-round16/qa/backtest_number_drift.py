# -*- coding: utf-8 -*-
"""scan_number_drift.py 的双向回测。

灵敏度：把 3 个**已知形态**的真缺陷注入真库副本（数值改小 / 分数形式 / 点数序号），
中文一律停在旧数字上，看闸报不报。
特异度：把 3 个**已知良性**形态一并注入（英文数词改写、同一数字删掉重复的一处、
标记内部数字变动），看闸会不会误报。

用法：python backtest_number_drift.py --work <临时目录>
"""
import argparse, json, os, re, shutil, subprocess, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
QA = P + "/3-常用脚本/qa"


def jload(p):
    raw = open(p, encoding="utf-8-sig").read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))


def jdump(o, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(o, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def setpath(root, path, val):
    node = root
    parts = path.split(".")
    for k in parts[:-1]:
        node = node[k]
    node[parts[-1]] = val


def getpath(root, path):
    node = root
    for k in path.split("."):
        node = node[k]
    return node


CASES = [
    # (id, 期望, 说明, 旧英文, 新英文, 中文)
    ("TP1-数值改小", "REPORT", "Able Crewman 型：游泳速度 30 改成 20，中文仍写 30",
     "<p>The creature has a swim speed of 30 feet and can hold its breath for 15 minutes.</p>",
     "<p>The creature has a swim speed of 20 feet and can hold its breath for 15 minutes.</p>",
     "<p>该生物拥有 30 尺的游泳速度，并能屏息 15 分钟。</p>"),
    ("TP2-分数形式", "REPORT", "Defenses 型：减伤由 /2 改成 /4，中文仍写 /2",
     "<p>Physical damage is reduced by Armor/2 before it is applied.</p>",
     "<p>Physical damage is reduced by Armor/4 before it is applied.</p>",
     "<p>物理伤害在结算前先减去 护甲/2。</p>"),
    ("TP3-点数序号", "REPORT", "Level Advancement 型：天赋点 2 改成 3，中文仍写 2",
     "<p>Each level grants 2 Talent points which may be spent freely.</p>",
     "<p>Each level grants 3 Talent points which may be spent freely.</p>",
     "<p>每一等级给予 2 点天赋点数，可自由分配。</p>"),
    ("FP1-英文数词改写", "SILENT", "上游把 30/50/100 改写成 thirty/fifty/one hundred，中文用阿拉伯数字等价",
     "<p>The hallway runs 30 feet east, 50 feet north, and 100 feet down.</p>",
     "<p>The hallway runs thirty feet east, fifty feet north, and one hundred feet down.</p>",
     "<p>走廊向东延伸 30 尺、向北 50 尺、向下 100 尺。</p>"),
    ("FP2-删掉重复的一处", "SILENT", "同一个数字文中出现两次，上游删掉其中一次，中文照旧",
     "<p>Roll 4 dice, then roll 4 dice again to confirm the 12 point threshold.</p>",
     "<p>Roll 4 dice to confirm the 12 point threshold.</p>",
     "<p>投 4 枚骰子以确认 12 点的阈值。</p>"),
    ("FP3-标记内部数字", "SILENT", "只有 @Check/图片路径里的数字变了，散文没变",
     '<p>Make a @Check[skill:awareness dc:14] test.</p><img src="icons/rune-3.webp">',
     '<p>Make a @Check[skill:awareness dc:18] test.</p><img src="icons/rune-7.webp">',
     "<p>进行一次 @Check[skill:awareness dc:14]{察觉}检定。</p>"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = a.work
    shutil.rmtree(work, ignore_errors=True)
    base_dir = os.path.join(work, "baseline")
    repo = os.path.join(work, "repo")
    os.makedirs(base_dir, exist_ok=True)

    pack = "crucible.rules.json"
    src_en = jload(f"{P}/2-Crucible汉化插件/compendium/en/{pack}")
    src_cn = jload(f"{P}/2-Crucible汉化插件/compendium/cn/{pack}")
    old_base = jload(f"{P}/5-其他内容/english-baseline/crucible-0.9.1-legacy/{pack}")

    # 找一条基准/当前/中文三边都在的 text 叶子当宿主，逐个注入。
    host = "Character Creation.pages.Overview.text"
    for probe in (host,):
        getpath(old_base["entries"], probe); getpath(src_en["entries"], probe); getpath(src_cn["entries"], probe)

    results = []
    for cid, expect, desc, oe, ne, cn in CASES:
        b, e, c = json.loads(json.dumps(old_base)), json.loads(json.dumps(src_en)), json.loads(json.dumps(src_cn))
        setpath(b["entries"], host, oe)
        setpath(e["entries"], host, ne)
        setpath(c["entries"], host, cn)
        jdump(b, os.path.join(base_dir, pack))
        jdump(e, os.path.join(repo, "compendium", "en", pack))
        jdump(c, os.path.join(repo, "compendium", "cn", pack))
        out = os.path.join(work, "r.json")
        subprocess.run([sys.executable, f"{QA}/scan_number_drift.py", "--repo", repo,
                        "--baseline", base_dir, "--out", out],
                       capture_output=True, check=True)
        fs = json.load(open(out, encoding="utf-8"))["findings"]
        hit = [f for f in fs if f["path"] == host]
        got = "REPORT" if hit else "SILENT"
        results.append((cid, expect, got, "PASS" if got == expect else "**FAIL**", desc,
                        hit[0] if hit else None))

    for cid, expect, got, ok, desc, h in results:
        print(f"{ok:8} {cid:16} 期望={expect:7} 实得={got:7}  {desc}")
        if h:
            print(f"           去 {h['en_gone']} → 来 {h['en_added']} | 中文留着 {h['cn_still_has_old']}"
                  f" 没跟上 {h['cn_missing_new']}")
    bad = [r for r in results if r[3] != "PASS"]
    print(f"\n{len(results) - len(bad)}/{len(results)} PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
