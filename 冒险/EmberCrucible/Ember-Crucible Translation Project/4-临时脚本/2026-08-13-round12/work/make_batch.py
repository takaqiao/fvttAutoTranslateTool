# -*- coding: utf-8 -*-
"""G12 / K5 batch builder: targeted, assertion-checked replacements on CN leaves."""
import json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPO = "1-Ember汉化插件"
PACKS = ["ember.adventure.json", "ember.crucible-adventure.json"]
OUT = os.path.join(BASE, "4-临时脚本", "2026-08-13-round12", "batches")
PFX = "Ember Early Access."

# key = batch_path (relative to entries.) ; value = list of (old, new, expected_count)
EDITS = {}

def E(path, *subs):
    EDITS.setdefault(PFX + path, []).extend(subs)

J = "journals."

# ---------------- Kadísos Gazetteer ----------------
E(J + "Kadísos Gazetteer.pages.The Cauldron.text",
  ("误认为是余烬之心本身的呼吸", "误认为是余烬本身的呼吸", 1),
  ("伟大的城市雷贾赫", "伟大的城市雷贾尔", 1))

E(J + "Kadísos Gazetteer.pages.The Cauldron.overview",
  ("这座沸腾的泻湖", "这座沸腾的潟湖", 1))

E(J + "Kadísos Gazetteer.pages.Reaver Ocean.exposition",
  ("仿佛与余烬之心本身的心跳相互回响", "仿佛与余烬本身跳动的心脏相互回响", 1))

E(J + "Kadísos Gazetteer.pages.Region Overview.text",
  ("其中最密集的分布区域位于北岸的@UUID[.5jydYxd6imXEAjqT]{坩埚湖}周边。",
   "其中最集中的一片位于@UUID[.5jydYxd6imXEAjqT]{坩埚湖}的北岸一带。", 1),
  ("而因其靠近大塔，被派驻此地曾被视为理想任命",
   "而因其靠近那座宏伟高塔，被派驻此地曾被视为理想任命", 1))

E(J + "Kadísos Gazetteer.pages.Jade Mountains.text",
  ("大多数人会选择沿着历经岁月由自然元素塑造出的天然通路前进。只要天气不与人作对，这些通路便能在这片危险的环境中提供些许安全感。",
   "大多数人会选择沿着历经岁月由自然元素塑造出的天然通道前进。只要天气不与人作对，这些通道便能在这片危险的环境中提供些许安全感。", 1))

E(J + "Kadísos Gazetteer.pages.Volcanic Bluffs.overview",
  ("岩石峭壁为下方的海洋提供了令人惊叹的景色，也是种类繁多、生物体型大小不一的生物的栖息地，",
   "从岩石峭壁上可以俯瞰下方海洋的壮丽景色，这里也是各种大大小小的生物的栖息地，", 1))

E(J + "Kadísos Gazetteer.pages.Tidal Pools.text",
  ("覆满史莱姆的湿滑岩石", "覆满黏滑藻泥的岩石", 1))

# ---------------- Steed's Point ----------------
E(J + "Steed's Point.pages.Kali's Cottage.text",
  ("莫赖娅", "莫赖亚", 6),
  ('<h2 class="divider">秘密通道</h2>',
   '<h2 class="divider" id="secret-passage-entrance">秘密通道</h2>', 1),
  ('<h2 class="divider">与卡莉的交谈</h2>',
   '<h2 class="divider" id="a-conversation-with-kali">与卡莉的交谈</h2>', 1),
  ('<h2 class="divider">再次来访时</h2>',
   '<h2 class="divider" id="return-visit">再次来访时</h2>', 1),
  ("如果玩家是通过斯蒂德角东北角的@UUID[.jCEAfUatFe6LoEnE#secret-passage-entrance]{秘密通道入口}前来，请见下方内容。",
   "如果玩家是通过斯蒂德角东北角的秘密通道前来，请见下方的 @UUID[.jCEAfUatFe6LoEnE#secret-passage-entrance]{秘密通道入口}。", 1))

E(J + "Steed's Point.pages.Ruined Home.text",
  ("卡莉·安德蕾拉的女儿姓氏好像是“Fox什么来着”",
   "卡莉·安德蕾拉的女儿姓氏好像是“福克斯……什么来着”", 1))

E(J + "Steed's Point.pages.Dilapidated Shed.text",
  ('<sub data-system="crucible">完全掩蔽</sub>', '<sub data-system="crucible">完全掩护</sub>', 1))

E(J + "Steed's Point.pages.The Old Barn.text",
  ("任何位于树篱 5 尺内的角色", "任何位于树篱 5 英尺内的角色", 2))

E(J + "Steed's Point.pages.The Ersatz Bridge.text",
  ("<strong>学识</strong>", "<strong>知识</strong>", 1),
  ("能够让人无须担心危险或阻碍地穿过这片旷野", "能够让人无须担心危险或阻碍地跨过这段距离", 1))

E(J + "Steed's Point.pages.Vine Path.text",
  ("<strong>学识</strong>", "<strong>知识</strong>", 1),
  ("穿越村庄南部旷野的一条合理路线", "穿越村庄南部开阔地带的一条合理路线", 1))

E(J + "Steed's Point.pages.Old Damaged Bridge.text",
  ("就会注意到桥的中段已经开始垮塌", "就会注意到桥的中段已经开始变得不牢固", 1))

E(J + "Steed's Point.pages.Overgrown Trapdoor.text",
  ("任何使用盗贼工具并成功通过一次", "任何使用开锁工具并成功通过一次", 1),
  ("或此处这扇杂草丛生的活板门现身", "或此处这扇蔓生的活板门现身", 1))

# ---------------- Yakoshta Mine ----------------
for p in ["Area Overview", "Blue Track", "Red Track"]:
    E(J + "Yakoshta Mine.pages.%s.text" % p,
      ("拉杆上标有三个象征", "拉杆上标有三个符号", 1))

E(J + "Yakoshta Mine.pages.Loading Zone.text",
  ("其中一面墙上画着一个银色象征", "其中一面墙上画着一个银色符号", 1))

E(J + "Yakoshta Mine.pages.Ooze Pool.text",
  ("象征", "符号", 13))

E(J + "Yakoshta Mine.pages.Glowing Ore Pit.text",
  ("@UUID[Actor.mziAGiva2J2ZTaAQ.Item.oozeCorrodeWeapo]{腐蚀武器}能力",
   "@UUID[Actor.mziAGiva2J2ZTaAQ.Item.oozeCorrodeWeapo]能力", 1))

E(J + "Yakoshta Mine.pages.Cavern Bridge.text",
  ("下方的深渊上横跨着一座长长的绳索桥", "下方的裂谷上横跨着一座长长的绳索桥", 1),
  ("与通往@UUID[.xm50UssrvERyHUig]{发光矿坑}的上层观景处",
   "与@UUID[.xm50UssrvERyHUig]{发光矿坑}的上层高处平台", 1))

E(J + "Yakoshta Mine.pages.Mine Cart Loading Station.text",
  ("没入你面前峡谷微光之前", "没入你面前裂谷微光之前", 1))

E(J + "Yakoshta Mine.pages.Waterfall Bridge.text",
  ("横跨裂隙两侧的绳索", "横跨裂谷两侧的绳索", 1),
  ("吊在下方裂隙上方的绳索上", "吊在下方裂谷上方的绳索上", 1),
  ("入口桥因最近的地震而部分已摧毁。", "入口桥已被最近的地震部分损毁。", 1),
  ("会立刻从那堆岩石后方压低声音高声喊道", "会立刻从那堆岩石后方压着嗓子大声喊道", 1))

E(J + "Yakoshta Mine.pages.Excavation Pit.text",
  ("动作使敌人陷入迷乱", "动作使敌人陷入迷失方向", 1))

E(J + "Yakoshta Mine.pages.Old Ore Pit.text",
  ("路口轮盘", "枢纽轮盘", 4),
  ("所以也许只有真需要的时候再用？", "所以也许只有真需要的时候再用。", 1))

E(J + "Yakoshta Mine.pages.Ooze Go Boom!.text",
  ("路口轮盘", "枢纽轮盘", 2))

E("items.Yakoshta Junction Wheel.name",
  ("雅科什塔路口轮盘 Yakoshta Junction Wheel", "雅科什塔枢纽轮盘 Yakoshta Junction Wheel", 1))

E(J + "Yakoshta Mine.pages.Elevator.text",
  ("使用一套撬锁工具进行", "使用一套开锁工具进行", 1))

E(J + "Yakoshta Mine.pages.Supply Cache.text",
  ("使用盗贼工具并成功通过", "使用开锁工具并成功通过", 1),
  ("使用一套开锁器并成功通过", "使用一套开锁工具并成功通过", 1))


def get(d, dotted):
    cur = d
    for k in dotted.split("."):
        cur = cur[k]
    return cur


def main():
    problems = []
    for pack in PACKS:
        cnp = os.path.join(BASE, REPO, "compendium", "cn", pack)
        cn = json.load(open(cnp, encoding="utf-8"))["entries"]
        batch = {}
        for bp, subs in EDITS.items():
            try:
                old = get(cn, bp)
            except KeyError:
                problems.append(f"{pack}: MISSING key {bp}")
                continue
            new = old
            for a, b, n in subs:
                got = new.count(a)
                if got != n:
                    problems.append(f"{pack}: {bp}: pattern {a[:40]!r} count {got} != {n}")
                    continue
                new = new.replace(a, b)
            if new == old:
                problems.append(f"{pack}: {bp}: NO CHANGE")
                continue
            # id-preservation self check
            if new.count('id="') < old.count('id="'):
                problems.append(f"{pack}: {bp}: id= count dropped "
                                f"{old.count('id=\"')} -> {new.count('id=\"')}")
            batch[bp] = new
        os.makedirs(OUT, exist_ok=True)
        n = 1 if True else 1
        fp = os.path.join(OUT, f"G12.1.{pack}")
        json.dump(batch, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{pack}: {len(batch)} leaves -> {fp}")
    if problems:
        print("\n!!! PROBLEMS")
        for p in problems:
            print("  ", p)
    else:
        print("\nall assertions passed")


main()
