# -*- coding: utf-8 -*-
"""`scan_dropped_terms.py` 逐位对齐过滤器的**双向**回测。

为什么必须双向（本项目为「只测特异度」吃过亏，`R-catwalk` 空转四个发布版）：
一个只会返回「没问题」的过滤器在特异度上是满分。所以本回测**先往库里注入已知形态的
真缺陷**，确认新过滤器仍报得出，再看它把哪些良性形态压了下去。

每个用例都跑**两遍** —— 带过滤器与 `--no-block-filter` —— 于是「是不是过滤器干的」
有对照，而不是靠猜。判定键在**目标词干**上，不看整叶是否有告警：
注入文本里的 `relic`/`paladin` 之类在词表里也有译名，会自己带出无关命中。

灵敏度 7 例（必须仍然 REPORT）
------------------------------
TP1 三项并列删中项，**同块内另一句还留着同一个词的小写形式**
    ← 这是历史上那条真缺陷的原形，专门用来卡「块内该词归零才报」这种过度收紧。
      它的局部三元组是 (旧2, 新1, 中2)，与 ① 类假阳性**数值上完全同形**，
      所以块级之下再加任何数值闸都必然连它一起杀掉 —— 这就是本过滤器停在块粒度的理由。
TP2 整句删掉一个术语（块内该词归零）
TP3 公式里换掉一个属性名
TP4 **整块删除、中文没跟**：上游删掉整个 `<p>`，中文那个 `<p>` 还在
    → 中文块数 ≠ 新英文块数，走 `shape_mismatch_blockcount` 退回整叶口径，仍然报
TP5 `a Skill Check` → `a check`（块内该词归零，ember 侧真缺陷的原形）
TP6 **别的块里有同名词的正当用法**：缺陷在第 2 块，第 1/3 块两侧都正当地用着同一个词
    → 测锚点没把候选块选歪
TP7 **删除点正好落在块边界**：被删内容在 `<p>` 末尾，新侧的下一个词已属下一块
    → 测 `delete` 取 `{前一词的块, 后一词的块}` 两个候选这条规则；只取后一词会漏报

特异度 5 例（必须 SILENT）
--------------------------
FP1 裸词升级成 `@Condition[…]`（旧闸已有）
FP2 明文换成不带标签的 `@UUID`（旧闸已有）
FP3 整段重写，词级相似度低于门槛（旧闸已有）
FP4 **整块删除、中文已跟进**（新过滤器的主目标，第十六轮 ≈18 条）：
    上游删掉整个 `<p>`，中文那个 `<p>` 早就不在；整叶计数被别的块里的正当用法撑着
    → 带过滤器 SILENT，`--no-block-filter` 仍 REPORT
FP5 **同义改写、中文命中全在同叶别处**（第十六轮最大头 ≈46 条）：
    `The Bickering Priests guard` → `Two sentries watch`，中文的「牧师」来自另一块
    → 带过滤器 SILENT，`--no-block-filter` 仍 REPORT

⚠ 注入用副本树：真库一个字节都不碰，跑完校验 sha256。

用法：python backtest_block_filter.py --work <临时目录>
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
QA = P + "/3-常用脚本/qa"
PACK = "crucible.rules.json"
HOST = "Character Creation.pages.Overview.text"


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


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# (id, 目标词干, 带过滤器期望, 不带过滤器期望, 说明, 旧英文, 新英文, 中文)
CASES = [
    ("TP1-并列删中项", "skill", "REPORT", "REPORT",
     "Ability, Skill, and Talent → Ability and Talent；同块内另一句还留着小写 skill",
     "<p>Increases in level grant additional Ability, Skill, and Talent points to spend. "
     "Players should prepare anecdotes explaining the skill advancements their character unlocked.</p>",
     "<p>Increases in level grant additional Ability and Talent points to spend. "
     "Players should prepare anecdotes explaining the skill advancements their character unlocked.</p>",
     "<p>等级提升会给予额外的能力、技能和天赋点数可供分配。玩家应准备一些轶事，"
     "说明其角色解锁了哪些技能晋升。</p>"),

    ("TP2-整句删术语", "bane", "REPORT", "REPORT",
     "上游删掉「Attempts to Disarm you are made with +2 Bane dice」整句，中文还写着祸骰",
     "<p>Your grasp is firm and unyielding while wielding a two-handed weapon. "
     "Attempts to Disarm you are made with +2 Bane dice.</p>",
     "<p>Your grasp is firm and unyielding while wielding a two-handed weapon.</p>",
     "<p>在持用双手武器时，你的握持稳固而坚定不移。试图对你进行缴械的尝试会承受 +2 祸骰。</p>"),

    ("TP3-换属性名", "intellect", "REPORT", "REPORT",
     "公式里的 Intellect 被上游删掉，中文仍写「智力」",
     "<p>Your spell power is derived from your Intellect and your Presence score combined.</p>",
     "<p>Your spell power is derived from your Presence score alone.</p>",
     "<p>你的法术强度由你的智力与风度属性共同决定。</p>"),

    ("TP4-整块删中文没跟", "talent", "REPORT", "REPORT",
     "上游删掉整个 <p>，中文那个 <p> 还在 → 块数不等，退回整叶口径仍然报",
     "<p>The vault holds three sealed caskets.</p>"
     "<p>Each casket grants a Talent to whoever opens it.</p>",
     "<p>The vault holds three sealed caskets.</p>",
     "<p>宝库中存放着三个密封的匣子。</p><p>每个匣子都会赋予开启者一项天赋。</p>"),

    ("TP5-SkillCheck降级", "skill", "REPORT", "REPORT",
     "a Skill Check → a check（块内该词归零），中文仍写「技能检定」",
     "<p>Crossing the ledge requires a Skill Check against the terrain.</p>"
     "<p>The group rests afterward.</p>",
     "<p>Crossing the ledge requires a check against the terrain.</p>"
     "<p>The group rests afterward.</p>",
     "<p>跨越岩架需要一次技能检定以对抗地形。</p><p>随后众人休息。</p>"),

    ("TP6-别块有正当同名", "skill", "REPORT", "REPORT",
     "缺陷在第 2 块；第 1/3 块两侧都正当地用着 Skill/技能 —— 测锚点没把候选块选歪",
     "<p>Skill training is covered in chapter one.</p>"
     "<p>The amulet grants a Skill bonus to whoever wears it.</p>"
     "<p>Skill points accrue at every level.</p>",
     "<p>Skill training is covered in chapter one.</p>"
     "<p>The amulet grants a bonus to whoever wears it.</p>"
     "<p>Skill points accrue at every level.</p>",
     "<p>技能训练在第一章中介绍。</p><p>该护符会为佩戴者提供技能加值。</p>"
     "<p>技能点数随每个等级累积。</p>"),

    ("TP7-删除点在块边界", "bane", "REPORT", "REPORT",
     "被删内容在 <p> 末尾，新侧下一个词已属下一块 → 测 delete 取前后两块这条规则",
     "<p>The bearer gains a bonus die and a Bane die.</p><p>Rest restores the die.</p>",
     "<p>The bearer gains a bonus die.</p><p>Rest restores the die.</p>",
     "<p>持有者获得一枚加值骰和一枚祸骰。</p><p>休息可恢复该骰。</p>"),

    ("FP1-升级成enricher", "boon", "SILENT", "SILENT",
     "Flanked → @Condition[flanked]，方括号里那个词照样是玩家看到的字",
     "<p>This creature doubles the number of Boons awarded from attacking a Flanked enemy nearby.</p>",
     "<p>This creature doubles the number of Boons awarded from attacking a @Condition[flanked] enemy nearby.</p>",
     "<p>这种生物从攻击附近处于夹击状态的敌人中获得的恩惠骰数量翻倍。</p>"),

    ("FP2-换成裸UUID", "movement", "SILENT", "SILENT",
     "Incorporeal Movement → 不带标签的 @UUID，Foundry 自己渲染目标名",
     "<p>Throughout the combat Tethra will rely upon Incorporeal Movement and her fly speed to navigate.</p>",
     "<p>Throughout the combat Tethra will rely upon @UUID[Actor.zzz.Item.testItem0001] and her fly speed to navigate.</p>",
     "<p>在整个战斗过程中，泰斯拉会依赖@UUID[Actor.zzz.Item.testItem0001]{虚体移动}和她的飞行速度穿梭。</p>"),

    ("FP3-整段重写", "shent", "SILENT", "SILENT",
     "上游把整段推倒重写，词级相似度低于门槛",
     "<p>The ancient hall stretches north, lined with Shent statues and a crumbling Vista mural.</p>",
     "<p>Water drips from a shattered ceiling into a black pool; nothing else remains of what stood here.</p>",
     "<p>古老的厅堂向北延伸，两侧立着申特雕像，还有一幅剥落的远景壁画。</p>"),

    # ↓↓ 新过滤器的两个主目标：不带过滤器仍会报，带上就该静默 ↓↓
    ("FP4-整块删中文已跟", "talent", "SILENT", "REPORT",
     "上游删掉整个 <p>，中文那个 <p> 早就不在；整叶的「天赋」×2 全来自保留下来的第 3 块",
     "<p>Each casket grants a Talent to whoever opens it.</p>"
     "<p>Consult the index before spending points.</p>"
     "<p>A Talent may be swapped during rest.</p>",
     "<p>Consult the index before spending points.</p>"
     "<p>A Talent may be swapped during rest.</p>",
     "<p>在消耗点数之前，请查阅索引。</p>"
     "<p>天赋可在休息时更换，更换天赋不需要任何花费。</p>"),

    ("FP5-同义改写别块撞名", "priest", "SILENT", "REPORT",
     "The Bickering Priests guard → Two sentries watch；中文的「牧师」全在第 2 块的正当用法里",
     "<p>The Bickering Priests guard the north gate.</p>"
     "<p>If a Priest or a warrior joins the group, the Priest can calm them, and the Priest "
     "should lead the talks.</p>",
     "<p>Two sentries watch the north gate.</p>"
     "<p>If a Priest or a warrior joins the group, the Priest can calm them, and the Priest "
     "should lead the talks.</p>",
     # 中文侧「牧师」要有 4 次才够撑起整叶口径的告警（旧英文 4 / 新英文 3）——
     # 这正是「中文爱重复名词、英文用代词」的真实形态，不是为了凑数。
     "<p>两名哨兵看守着北门。</p>"
     "<p>若牧师或战士加入队伍，该牧师可以安抚他们，并且牧师应当主导谈判，"
     "牧师的判断值得信赖。</p>"),
]

BINDINGS = {"packages": [{"pkg": "test"}],
            "ids": {"testItem0001": [{"name": "Incorporeal Movement", "pkg": "test",
                                      "pack": "t", "kind": "items"}]},
            "notes": [], "results": []}


def run(repo, base_dir, bind, out, block_filter):
    cmd = [sys.executable, f"{QA}/scan_dropped_terms.py", "--repo", repo,
           "--baseline", base_dir, "--bindings", bind, "--out", out]
    if not block_filter:
        cmd.append("--no-block-filter")
    r = subprocess.run(cmd, capture_output=True, check=True, encoding="utf-8", errors="replace")
    return json.load(open(out, encoding="utf-8")), r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = a.work

    # ---- 真库 + 被测脚本的指纹（跑完要一字节不差）----
    watched = [f"{P}/{PACK.join(['2-Crucible汉化插件/compendium/en/', ''])}",
               f"{P}/2-Crucible汉化插件/compendium/cn/{PACK}",
               f"{P}/5-其他内容/english-baseline/crucible-0.9.1-legacy/{PACK}",
               f"{QA}/scan_dropped_terms.py"]
    watched[0] = f"{P}/2-Crucible汉化插件/compendium/en/{PACK}"
    before = {p: sha(p) for p in watched}

    shutil.rmtree(work, ignore_errors=True)
    base_dir = os.path.join(work, "baseline")
    repo = os.path.join(work, "repo")
    os.makedirs(base_dir, exist_ok=True)
    bind = os.path.join(work, "bindings.json")
    jdump(BINDINGS, bind)

    src_en = jload(f"{P}/2-Crucible汉化插件/compendium/en/{PACK}")
    src_cn = jload(f"{P}/2-Crucible汉化插件/compendium/cn/{PACK}")
    old_base = jload(f"{P}/5-其他内容/english-baseline/crucible-0.9.1-legacy/{PACK}")

    rows = []
    for cid, stem_t, exp_on, exp_off, desc, oe, ne, cn in CASES:
        b, e, c = (json.loads(json.dumps(x)) for x in (old_base, src_en, src_cn))
        setpath(b["entries"], HOST, oe)
        setpath(e["entries"], HOST, ne)
        setpath(c["entries"], HOST, cn)
        jdump(b, os.path.join(base_dir, PACK))
        jdump(e, os.path.join(repo, "compendium", "en", PACK))
        jdump(c, os.path.join(repo, "compendium", "cn", PACK))

        got, local = {}, None
        for mode, flag in (("on", True), ("off", False)):
            rep, _ = run(repo, base_dir, bind, os.path.join(work, f"r_{mode}.json"), flag)
            hits = [h for f in rep["findings"] if f["path"] == HOST
                    for h in f["dropped"] if h["en"].lower().startswith(stem_t[:5])]
            got[mode] = "REPORT" if hits else "SILENT"
            if mode == "on" and hits:
                local = hits[0].get("local")
            if mode == "off" and not local and hits:
                local = "（带过滤器时已静默）"
        ok = "PASS" if (got["on"] == exp_on and got["off"] == exp_off) else "**FAIL**"
        rows.append((ok, cid, stem_t, exp_on, got["on"], exp_off, got["off"], desc, local))

    print("用例                目标词    带过滤器(期望/实得)   不带(期望/实得)   判定")
    print("-" * 96)
    for ok, cid, t, eon, gon, eoff, goff, desc, local in rows:
        print(f"{cid:<20}{t:<10}{eon:>7}/{gon:<8}{eoff:>7}/{goff:<8}{ok}")
        print(f"    {desc}")
        if local:
            print(f"    局部三元组 {local}")
    bad = [r for r in rows if r[0] != "PASS"]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} PASS"
          f"  （灵敏度 {sum(1 for r in rows if r[1].startswith('TP') and r[0] == 'PASS')}/"
          f"{sum(1 for r in rows if r[1].startswith('TP'))}"
          f" · 特异度 {sum(1 for r in rows if r[1].startswith('FP') and r[0] == 'PASS')}/"
          f"{sum(1 for r in rows if r[1].startswith('FP'))}）")

    after = {p: sha(p) for p in watched}
    changed = [p for p in watched if before[p] != after[p]]
    print(f"\n真库/脚本指纹校验：{len(watched)} 个文件，改动 {len(changed)} 个"
          + ("" if not changed else "  ← **不该有**"))
    for p in watched:
        print(f"  {'同' if before[p] == after[p] else '**变**'}  {before[p][:16]}  {p}")
    return 1 if (bad or changed) else 0


if __name__ == "__main__":
    sys.exit(main())
