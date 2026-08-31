"""Y2 灵敏度回测：往**副本树**里注入六处违规，确认三条新断言真的会变红，再还原。

只测特异度（全绿）是不够的 —— 那样「所有断言都返回空」也能过。本脚本每一处注入都
对准一条判据的**一个方向**，并且尽量挑「叶级判据看不见、只有块级看得见」的位置。

跑法：
  python inject_backtest.py --tree <副本树> --mode inject
  python assert_resolutions.py --root <副本树>        # 期望：三条全红
  python inject_backtest.py --tree <副本树> --mode restore
"""
import argparse
import json
import os
import sys

PACKS = [("1-Ember汉化插件", "ember.adventure.json"),
         ("1-Ember汉化插件", "ember.crucible-adventure.json")]

# (标签, 叶路径后缀, 原串, 改成, 打哪条判据)
SHOTS = [
    ("arct-叶内串行", "Arctus Plateau Gazetteer.pages.Brevin.text",
     "和大多数阿克图里安人一样", "和大多数阿克图瑞尔人一样",
     "R-arcturel-arcturian-blocks（sequence：块内 I 被写成 E）"),
    ("arct-块内漏译", "Arcturel Tradeway.pages.Silver Beam Foyer.text",
     "缺少阿克图瑞尔标志性的传统", "缺少标志性的传统",
     "R-arcturel-arcturian-blocks（sequence：块内少一个 E）"),
    ("shard-女神并进神", "Kadísos Gazetteer.pages.Region Overview.text",
     "碎片女神斯科里斯", "碎片之神斯科里斯",
     "R-shard-god-blocks（count_ge：F 类计数掉到 0）"),
    # ⚠ 这一枪必须打在**中英计数正好相等**的块上。第一版打的是 `Deities.pages.Nite.text`
    #   块51（EN 2 处 / CN 3 处，中文代词还原多出一处），删掉一处后仍是 2≥2，判据按设计不响
    #   —— 那不是判据漏了，正是「可多不可少」这条边界本身。换成 EN 2 / CN 2 的块才测得到。
    ("shard-块内漏译", "Introduction.pages.Setting Introduction.text",
     "有些碎片诸神真正古老", "有些神真正古老",
     "R-shard-god-blocks（count_ge：G 类中文少于英文）"),
    ("rank-正向闸", "Gamemaster's Guide.pages.Attunement Mechanics.text",
     "同样可以拥有同调阶位", "同样可以拥有同调层级",
     "R-rank-sense-blocks（块内全机制义、中文没了「阶位」）"),
    ("rank-反向闸", "Organizations.pages.Flame Guard.text",
     "其内部依照经验与技能设有多个等级", "其内部依照经验与技能设有多个阶位",
     "R-rank-sense-blocks（块内全普通名词义、中文却用了「阶位」）"),
]


def walk_set(node, parts, old, new, hits):
    """按路径尾段找到叶并替换（只替换第一处）。"""
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if isinstance(v, str):
                if k == parts[-1] and old in v:
                    node[k] = v.replace(old, new, 1)
                    hits.append(k)
            else:
                walk_set(v, parts, old, new, hits)
    elif isinstance(node, list):
        for v in node:
            walk_set(v, parts, old, new, hits)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--mode", choices=["inject", "restore"], required=True)
    a = ap.parse_args()

    total = 0
    for repo, pack in PACKS:
        p = os.path.join(a.tree, repo, "compendium", "cn", pack)
        if not os.path.exists(p):
            continue
        doc = json.load(open(p, encoding="utf-8-sig"))
        raw = json.dumps(doc, ensure_ascii=False)
        for tag, path, old, new, which in SHOTS:
            src, dst = (old, new) if a.mode == "inject" else (new, old)
            n = raw.count(src)
            if n == 0:
                print(f"  ⚠ {pack:30s} {tag:16s} 找不到「{src[:16]}」—— 这一枪没打出去")
                continue
            raw = raw.replace(src, dst, 1)
            total += 1
            print(f"  {pack:30s} {tag:16s} 改 1 处（全文共 {n} 处）→ {which}")
        json.dump(json.loads(raw), open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n{a.mode}：共改 {total} 处")


if __name__ == "__main__":
    main()
