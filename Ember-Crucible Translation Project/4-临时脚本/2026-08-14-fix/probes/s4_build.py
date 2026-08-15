# -*- coding: utf-8 -*-
"""S4 · 生成回写批次。只读 compendium，只写 4-临时脚本/.../batches/。

每条替换都带「预期命中数」断言：命中数不符就 abort，绝不静默少改/多改。
"""
import sys, os, re, json
import s4_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "batches")

# (repo, pack, path, [(pattern, repl, n_expected, is_regex)])
EDITS = []

TWIN = ["ember.crucible-adventure.json", "ember.adventure.json"]


def add_twin(path, subs):
    for pack in TWIN:
        EDITS.append(("1", pack, path, subs))


def add(repo, pack, path, subs):
    EDITS.append((repo, pack, path, subs))


P = "Ember Early Access."

# ---- npc-name-split | Amelia Naxan -------------------------------------
add_twin(P + "actors.Amelia Naxan.tokenName", [("娜珂珊女士阿米莉亚", "阿梅莉亚·纳克桑女士", 1, False)])
add("1", "ember.crucible-adventure.json", P + "actors.Amelia Naxan.biography.private",
    [("阿米莉亚·纳克珊女士", "阿梅莉亚·纳克桑女士", 1, False)])

# ---- tokenname-translit-split | 术语分裂三组 ---------------------------
add_twin(P + "actors.Arcturian Sea Captain.tokenName", [("海船船长", "海船队长", 1, False)])
add_twin(P + "actors.Otherhood Raider.tokenName", [("袭击者", "劫掠者", 1, False)])
add_twin(P + "actors.Mutagist Scout.tokenName", [("侦察兵", "斥候", 1, False)])

# ---- npc-translit-split | 11 组音译分裂 --------------------------------
add_twin(P + "journals.Ordain Gazetteer.pages.Foxlairs.text",
         [("纳西拉·杰索斯", "纳希拉·杰索斯", 1, False)])
add_twin(P + "journals.Arctus Plateau Gazetteer.pages.Skybrush.text",
         [("盖德隆·塔斯", "格德隆·塔斯", 2, False),
          ("格蒂", "古蒂", 2, False)])
add_twin(P + "journals.Ordain Gazetteer.pages.Sunhaven.text",
         [("奥基娅·提尔瓦", "奥基娅·提尔沃", 2, False)])
add_twin(P + "journals.Arctus Plateau Gazetteer.pages.Storsa's Strand.text",
         [("希耶尔", "西耶尔", 4, False)])
add_twin(P + "journals.Ordain Gazetteer.pages.Lantern Roads.text",
         [("彭妮·特拉奇尼", "佩妮·特拉奇尼", 2, False)])
add_twin(P + "journals.Diplomatic Impunity.pages.Criminal Records.text",
         [("加斯特恩·法维奥斯", "加斯特恩·法维约斯", 1, False)])
add_twin(P + "journals.Signal of Intent.pages.A Watchful Spire.text",
         [("科代恩·科勒", "科代恩·科尔", 1, False)])
add_twin(P + "journals.Arcturel Tradeway.pages.Arcturian Automatons.text",
         [("沃索洛缪·切斯", "瓦索洛缪·切斯", 1, False)])
add_twin(P + "journals.Spreading Sickness.pages.Midnight Meeting.summary",
         [("布琳娜·维罗科特", "布琳娜·维罗科尔特", 1, False)])
add("1", "ember.crucible-adventure.json", P + "actors.Arcos Sarinland.biography.private",
    [(r"阿科斯·萨林兰(?!德)", "阿科斯·萨林兰德", 1, True)])
add("1", "ember.crucible-adventure.json", P + "actors.Fernis Ossa.biography.private",
    [(r"阿科斯·萨林兰(?!德)", "阿科斯·萨林兰德", 1, True)])

# ---- cross_layer_term_split | 月相 UI 枚举 -----------------------------
LUNAR = [("<strong>盈月</strong>", "<strong>渐盈</strong>", 1, False),
         ("<strong>亏月</strong>", "<strong>渐亏</strong>", 1, False)]
# Moon Ring 是 crucible 侧独有：dnd5e 孪生包 ember.adventure 无此叶（已实测 EN/CN 均无），
# 故此条不适用孪生规则。
add("1", "ember.crucible-adventure.json", P + "items.Moon Ring.effects.Lunar Shield.description", LUNAR)
add("1", "ember.crucible-affixes.json", "Lunar Shield.description", LUNAR)

# ---- cross_layer_term_split | 规则页 vs lang UI ------------------------
add("2", "crucible.rules.json", "Character Creation.pages.Finishing Touches.text",
    [("在 传记 标签页中", "在 生平 标签页中", 1, False),
     ("<strong>代称</strong>", "<strong>代词</strong>", 1, False),
     ("<strong>年代</strong>", "<strong>年龄</strong>", 1, False),
     ("<strong>重量</strong>", "<strong>体重</strong>", 1, False),
     ("<strong>缩放价格</strong>", "<strong>按比例定价</strong>", 1, False),
     ("<strong>公开（Public）传记</strong>", "<strong>公开传记</strong>", 1, False),
     ("<strong>私密（Private）传记</strong>", "<strong>私人传记</strong>", 1, False)])
add("2", "crucible.rules.json", "Combat.pages.Actions.text",
    [("<h4>墙</h4>", "<h4>墙壁</h4>", 1, False),
     ("墙类型动作", "墙壁类型动作", 1, False),
     ("用于描述墙的宽度", "用于描述墙壁的宽度", 1, False),
     ("用于描述墙的中心", "用于描述墙壁的中心", 1, False)])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = {}
    errs = []
    for repo, pack, path, subs in EDITS:
        cn = L.leaf(repo, pack, path, "cn")
        en = L.leaf(repo, pack, path, "en")
        if not isinstance(cn, str):
            errs.append(f"CN 叶不存在: {repo}|{pack}|{path}")
            continue
        if not isinstance(en, str):
            errs.append(f"EN 叶不存在: {repo}|{pack}|{path}")
            continue
        new = cn
        for pat, repl, n, rx in subs:
            hits = len(re.findall(pat, new)) if rx else new.count(pat)
            if hits != n:
                errs.append(f"命中数不符 {repo}|{pack}|{path}: {pat!r} 期望 {n} 实得 {hits}")
                break
            new = re.sub(pat, repl, new) if rx else new.replace(pat, repl)
        else:
            if new == cn:
                errs.append(f"无变化 {repo}|{pack}|{path}")
                continue
            out.setdefault((repo, pack), {})[path] = new
    for e in errs:
        print("  !", e)
    os.makedirs(OUT, exist_ok=True)
    for (repo, pack), items in sorted(out.items()):
        fn = os.path.join(OUT, f"S4.{repo}.{pack}")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"  写出 {os.path.basename(fn)}  {len(items)} 条")
    print(f"\n错误 {len(errs)} / 批次条目 {sum(len(v) for v in out.values())}")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
