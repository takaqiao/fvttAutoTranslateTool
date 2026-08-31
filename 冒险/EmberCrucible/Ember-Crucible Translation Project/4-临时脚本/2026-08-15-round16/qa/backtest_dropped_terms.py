# -*- coding: utf-8 -*-
"""scan_dropped_terms.py 的双向回测。

灵敏度：注入 3 个已知形态的真缺陷 ——
  TP1 三项并列删中间一项（＝上一轮 crucible.rules 那条真缺陷的形态，
      而且同叶别处还留着同一个词的小写形式，专门用来卡「只看有无」的旧判据）
  TP2 整句删掉一个术语
  TP3 公式里换掉一个属性名
特异度：注入 3 个已知良性形态 ——
  FP1 上游把裸词升级成语义 enricher（`Flanked` → `@Condition[flanked]`）
  FP2 上游把明文换成**不带标签**的 `@UUID`（Foundry 渲染目标名，中文必须写标签）
  FP3 整段重写（词级相似度低于门槛，逐词数次数没有意义）

用法：python backtest_dropped_terms.py --work <临时目录>
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
    ("TP1-并列删中项", "REPORT", "Ability, Skill, and Talent → Ability and Talent；同叶别处还有小写 skill",
     "<p>Increases in level grant additional Ability, Skill, and Talent points to spend. "
     "Players should prepare anecdotes explaining the skill advancements their character unlocked.</p>",
     "<p>Increases in level grant additional Ability and Talent points to spend. "
     "Players should prepare anecdotes explaining the skill advancements their character unlocked.</p>",
     "<p>等级提升会给予额外的能力、技能和天赋点数可供分配。玩家应准备一些轶事，"
     "说明其角色解锁了哪些技能晋升。</p>"),
    ("TP2-整句删术语", "REPORT", "上游删掉「Attempts to Disarm you are made with +2 Banes」整句，中文还写着祸骰",
     "<p>Your grasp is firm and unyielding while wielding a two-handed weapon. "
     "Attempts to Disarm you are made with +2 Bane dice.</p>",
     "<p>Your grasp is firm and unyielding while wielding a two-handed weapon.</p>",
     "<p>在持用双手武器时，你的握持稳固而坚定不移。试图对你进行缴械的尝试会承受 +2 祸骰。</p>"),
    ("TP3-换属性名", "REPORT", "公式里的 Intellect 被上游删掉，中文仍写「智力」",
     "<p>Your spell power is derived from your Intellect and your Presence score combined.</p>",
     "<p>Your spell power is derived from your Presence score alone.</p>",
     "<p>你的法术强度由你的智力与风度属性共同决定。</p>"),
    ("FP1-升级成enricher", "SILENT", "Flanked → @Condition[flanked]，方括号里那个词照样是玩家看到的字",
     "<p>This creature doubles the number of Boons awarded from attacking a Flanked enemy nearby.</p>",
     "<p>This creature doubles the number of Boons awarded from attacking a @Condition[flanked] enemy nearby.</p>",
     "<p>这种生物从攻击附近处于夹击状态的敌人中获得的恩惠骰数量翻倍。</p>"),
    ("FP2-换成裸UUID", "SILENT", "Incorporeal Movement → 不带标签的 @UUID，Foundry 自己渲染目标名",
     "<p>Throughout the combat Tethra will rely upon Incorporeal Movement and her fly speed to navigate.</p>",
     "<p>Throughout the combat Tethra will rely upon @UUID[Actor.zzz.Item.testItem0001] and her fly speed to navigate.</p>",
     "<p>在整个战斗过程中，泰斯拉会依赖@UUID[Actor.zzz.Item.testItem0001]{虚体移动}和她的飞行速度穿梭。</p>"),
    ("FP3-整段重写", "SILENT", "上游把整段推倒重写，词级相似度低于门槛",
     "<p>The ancient hall stretches north, lined with Shent statues and a crumbling Vista mural.</p>",
     "<p>Water drips from a shattered ceiling into a black pool; nothing else remains of what stood here.</p>",
     "<p>古老的厅堂向北延伸，两侧立着申特雕像，还有一幅剥落的远景壁画。</p>"),
]

BINDINGS = {"packages": [{"pkg": "test"}],
            "ids": {"testItem0001": [{"name": "Incorporeal Movement", "pkg": "test",
                                      "pack": "t", "kind": "items"}]},
            "notes": [], "results": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = a.work
    shutil.rmtree(work, ignore_errors=True)
    base_dir = os.path.join(work, "baseline")
    repo = os.path.join(work, "repo")
    os.makedirs(base_dir, exist_ok=True)
    bind = os.path.join(work, "bindings.json")
    jdump(BINDINGS, bind)

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
        subprocess.run([sys.executable, f"{QA}/scan_dropped_terms.py", "--repo", repo,
                        "--baseline", base_dir, "--bindings", bind, "--out", out],
                       capture_output=True, check=True)
        fs = [f for f in json.load(open(out, encoding="utf-8"))["findings"] if f["path"] == host]
        got = "REPORT" if fs else "SILENT"
        results.append((cid, expect, got, "PASS" if got == expect else "**FAIL**", desc, fs))

    for cid, expect, got, ok, desc, fs in results:
        print(f"{ok:8} {cid:16} 期望={expect:7} 实得={got:7}  {desc}")
        for f in fs:
            for h in f["dropped"]:
                print(f"           {h['en']} {h['en_old_n']}→{h['en_new_n']} 次；"
                      f"中文 {h['cn_term']} 仍有 {h['cn_count']} 次 (sim {h['similarity']})")
    bad = [r for r in results if r[3] != "PASS"]
    print(f"\n{len(results) - len(bad)}/{len(results)} PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
