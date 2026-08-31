# -*- coding: utf-8 -*-
"""scan_marker_followup.py 的双向回测。

灵敏度：注入 3 个已知形态的真缺陷（上游换 @UUID 目标 / 上游改标题锚点 id /
上游新增 @Condition），中文一律停在旧版，看闸报不报。
特异度：注入 3 个已知良性形态（只有 {标签} 改写、readaloud 参数值改写、
`[[/item …]]` 按中文名解析），看闸会不会误报。

用法：python backtest_marker_followup.py --work <临时目录>
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


CASES = [
    ("TP1-换UUID目标", "MARKER_STALE", "上游把 @UUID 指到了新文档，中文还指着旧的",
     '<p>See @UUID[JournalEntry.abc.JournalEntryPage.OLDPAGE0000]{Overview} for details.</p>',
     '<p>See @UUID[JournalEntry.abc.JournalEntryPage.NEWPAGE1111]{Overview} for details.</p>',
     '<p>详见 @UUID[JournalEntry.abc.JournalEntryPage.OLDPAGE0000]{概览}。</p>'),
    ("TP2-改锚点id", "MARKER_STALE", "上游把标题锚点 id 由 overview 改成 basics，中文还写 overview",
     '<h3 id="overview">Overview</h3><p>The basics of play.</p>',
     '<h3 id="basics">Overview</h3><p>The basics of play.</p>',
     '<h3 id="overview">概览</h3><p>游戏的基础。</p>'),
    ("TP3-新增Condition", "MARKER_MISSING", "上游补了 @Condition[exhaustion]，中文没有这条标记",
     '<p>The creature falls prone and cannot act.</p>',
     '<p>The creature falls prone, becomes @Condition[exhaustion], and cannot act.</p>',
     '<p>该生物陷入倒地且无法行动。</p>'),
    ("FP1-只改标签", "SILENT", "英文只把 {标签} 改写了，目标没动，中文照译标签",
     '<p>See @UUID[JournalEntry.abc.JournalEntryPage.SAME00000000]{The Overview} now.</p>',
     '<p>See @UUID[JournalEntry.abc.JournalEntryPage.SAME00000000]{Overview Page} now.</p>',
     '<p>现在请看 @UUID[JournalEntry.abc.JournalEntryPage.SAME00000000]{概览页}。</p>'),
    ("FP2-readaloud参数", "SILENT", "@Embed 的 readaloud 参数值是散文，中文当然是中文",
     '<p>@Embed[Actor.xyz readaloud="A cold wind blows."] enters.</p>',
     '<p>@Embed[Actor.xyz readaloud="A freezing wind blows in."] enters.</p>',
     '<p>@Embed[Actor.xyz readaloud="一阵刺骨的寒风吹了进来。"] 登场。</p>'),
    ("FP3-item按中文解析", "SILENT", "[[/item …]] 由 dnd5e 按中文物品名解析，两边本来就不同",
     '<p>Use [[/item Warhammer]] against the door, dealing damage.</p>',
     '<p>Use [[/item Warhammer]] on the door, dealing extra damage.</p>',
     '<p>用 [[/item 战锤]] 砸门，造成伤害。</p>'),
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
    host = "Character Creation.pages.Overview.text"

    results = []
    for cid, expect, desc, oe, ne, cn in CASES:
        b, e, c = (json.loads(json.dumps(x)) for x in (old_base, src_en, src_cn))
        setpath(b["entries"], host, oe)
        setpath(e["entries"], host, ne)
        setpath(c["entries"], host, cn)
        jdump(b, os.path.join(base_dir, pack))
        jdump(e, os.path.join(repo, "compendium", "en", pack))
        jdump(c, os.path.join(repo, "compendium", "cn", pack))
        out = os.path.join(work, "r.json")
        subprocess.run([sys.executable, f"{QA}/scan_marker_followup.py", "--repo", repo,
                        "--baseline", base_dir, "--out", out], capture_output=True, check=True)
        fs = [f for f in json.load(open(out, encoding="utf-8"))["findings"] if f["path"] == host]
        got = ",".join(sorted({f["verdict"] for f in fs})) or "SILENT"
        ok = "PASS" if (expect in got if expect != "SILENT" else got == "SILENT") else "**FAIL**"
        results.append((cid, expect, got, ok, desc, fs))

    for cid, expect, got, ok, desc, fs in results:
        print(f"{ok:8} {cid:16} 期望={expect:14} 实得={got:28}  {desc}")
        for f in fs:
            print(f"           {f['verdict']}: {f['markers']}")
    bad = [r for r in results if r[3] != "PASS"]
    print(f"\n{len(results) - len(bad)}/{len(results)} PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
