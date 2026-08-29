#!/usr/bin/env python3
"""把系统自带 Adventure 包切成互不重叠的翻译单元。

  python prep_sys_units.py [--out <dir>]      切单元
  python prep_sys_units.py --collect [--out <dir>]   回收并装配成 compendium/cn

为什么切在 h1 边界
------------------
那一页 66,228 字是**一整个字符串**，六个 agent 不可能各写一份完整页再合并
（EC 教训：整页重写会把已校对的译文洗掉，且把工作量低估 5.6 倍）。
h1 的字节偏移是**无歧义的**，所以按 h1 切段、每段单独翻、最后按序拼回去，
拼接结果必须与原页**长度对齐、段序一致**——`--collect` 会断言这一点。

为什么不用 page-file 的「改现有中文」格式
----------------------------------------
那个格式是给**重对齐**用的（已有中文，要局部手术）。这里 compendium/cn 是空的，
是全新翻译，没有「未动过的字节」可保。等 Phase 5 跟版时再用那个格式。

不重叠是**结构保证的**，不是靠自觉：每段由 [start,end) 字节区间定义，
区间由脚本算出并写进 index.json，agent 拿到的就是切好的片段。
"""
import argparse
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
HUB = os.path.join(ROOT, "1-系统汉化插件")
EN = os.path.join(HUB, "compendium", "en", "alienrpg.alien-rpg-system.json")
CN = os.path.join(HUB, "compendium", "cn", "alienrpg.alien-rpg-system.json")
ADV = "Alien RPG System"
JKEY = "MU/TH/ER Instructions."

# 期刊分段：按 h1 序号归组。名字只用于人看，切分靠序号。
JOURNAL_UNITS = [
    ("SYS-J1", [0, 1, 2, 3], "欢迎/说明/更新日志/Actors 总述",
     "含 4.1.11→4.1.13 更新日志。⚠ 更新日志里全是模块名与版本号，属名从严。"),
    ("SYS-J2", [4], "Characters —— 角色卡逐字段走查",
     "⚑ 全项目的**术语锚点**。这一段把角色卡每一个字段走一遍，"
     "每个标签都必须与 lang/cn.json 逐字符一致。先做完它，别的才能开。"),
    ("SYS-J3", [5, 6, 7, 8], "Synthetics / Creatures / Spaceships / Vehicles 卡",
     "同为卡片走查，标签同样要对齐 lang。飞船那段有大量部件名。"),
    ("SYS-J4", [9, 10, 11, 12, 13], "Territories / Colony / Items 类型 / Tokens",
     "Items/Item Types 段列出全部物品类型，要与 lang 的 TYPES.Item.* 对齐。"),
    ("SYS-J5", [14, 15, 16, 17, 18, 19], "掷骰 / 生物 / 战斗轮 / 表格 / 系统设置 / 宏",
     "⚠ 「Tables」段与「Macros」段会提到 T-FROZEN 的表名与文件夹名，照抄英文，别翻。"),
    ("SYS-J6", [20], "推荐附加模块",
     "⚠ 通篇是**第三方模块名**，模块名一律保留英文，只翻描述。"),
]


def jload(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(o, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def h1_bounds(text):
    marks = [(m.start(), re.sub(r"<[^>]+>", "", m.group(1)).strip())
             for m in re.finditer(r"<h1[^>]*>(.*?)</h1>", text, re.S)]
    starts = [m[0] for m in marks]
    return marks, starts + [len(text)]


def cmd_prep(out):
    en = jload(EN)
    adv = en["entries"][ADV]
    page = adv["journals"][JKEY]["pages"][JKEY]
    text = page["text"]
    marks, bounds = h1_bounds(text)

    os.makedirs(os.path.join(out, "pages"), exist_ok=True)
    index, covered = {}, []

    # 序言（第一个 h1 之前）挂在 SYS-J1 上
    for uid, idxs, title, note in JOURNAL_UNITS:
        lo = bounds[idxs[0]] if idxs[0] > 0 else 0
        hi = bounds[idxs[-1] + 1]
        seg = text[lo:hi]
        covered.append((lo, hi, uid))
        p = os.path.join(out, "pages", uid + ".en.html")
        io.open(p, "w", encoding="utf-8", newline="\n").write(seg)
        index[uid] = {
            "kind": "journal_section",
            "title": title,
            "note": note,
            "h1_indexes": idxs,
            "h1_titles": [marks[i][1] for i in idxs],
            "byte_range": [lo, hi],
            "raw_chars": len(seg),
            "visible_chars": len(re.sub(r"<[^>]+>", "", seg)),
            "en_file": "pages/%s.en.html" % uid,
            "cn_file": "pages/%s.cn.html" % uid,
        }

    # 覆盖性断言：分段必须**恰好**铺满整页，不重叠不留缝
    covered.sort()
    pos = 0
    for lo, hi, uid in covered:
        if lo != pos:
            sys.exit("COVERAGE GAP/OVERLAP at %d..%d (unit %s expects %d)" % (pos, lo, uid, pos))
        pos = hi
    if pos != len(text):
        sys.exit("COVERAGE TAIL: covered %d of %d" % (pos, len(text)))

    # 非期刊单元
    index["SYS-T1"] = {
        "kind": "tables", "title": "三张表：Panic / Stress Response / Panic Response",
        "note": "⚑ 这三张表在 corerules 与 starterset 里有**逐字节相同**的副本。"
                "在这里翻好，Phase 4/5 用 tm/fill_twin.py 灌过去，别翻第二遍。"
                "⚠ 表名本身是 T-FROZEN（rollTableData.mjs 按英文名查表），只翻 results。",
        "payload": {k: v for k, v in adv["tables"].items()},
        "raw_chars": sum(len(json.dumps(v, ensure_ascii=False)) for v in adv["tables"].values()),
    }
    index["SYS-I1"] = {
        "kind": "items", "title": "26 个物品：12 skill-stunts + 14 specialty",
        "note": "⚑ 12 个 skill-stunts 的 **name 是 T-EXACT**，必须与 lang/cn.json 的 "
                "ALIENRPG.Skill<key> 逐字节相等，且**不许带英文尾巴**。"
                "⚠ 所有 notes 字段都是上游迁移留下的字面量 '[object Object]'，**原样保留，不要翻**。",
        "payload": adv["items"],
        "raw_chars": sum(len(json.dumps(v, ensure_ascii=False)) for v in adv["items"].values()),
    }
    index["SYS-U1"] = {
        "kind": "shell", "title": "Adventure 名/描述 + 7 个文件夹名 + 4 个宏名",
        "note": "⚠ Adventure 名 'Alien RPG System' 与文件夹 'Alien Tables' / "
                "'Alien Creature Tables' / 'Alien Mother Tables' 是 T-FROZEN。"
                "'Alien Sub-Tables' **没有被代码引用，可以翻**。宏只有 name，command 已在 mapping 里排除。",
        "payload": {"name": adv["name"], "description": adv["description"],
                    "folders": adv["folders"], "macros": adv["macros"]},
        "raw_chars": len(adv["description"]) + sum(len(v) for v in adv["folders"].values()),
    }

    jdump(os.path.join(out, "index.json"), index)
    total = sum(u["raw_chars"] for u in index.values())
    print("prepped %d units into %s" % (len(index), out))
    for uid in index:
        u = index[uid]
        print("  %-8s %-46s raw=%6d" % (uid, u["title"][:46], u["raw_chars"]))
    print("  page coverage: %d/%d bytes, no gaps, no overlaps" % (pos, len(text)))
    print("  total raw: %d" % total)
    return 0


def cmd_collect(out):
    en = jload(EN)
    adv_en = en["entries"][ADV]
    text = adv_en["journals"][JKEY]["pages"][JKEY]["text"]
    index = jload(os.path.join(out, "index.json"))

    fail, parts = [], []
    for uid, idxs, title, note in JOURNAL_UNITS:
        u = index[uid]
        p = os.path.join(out, u["cn_file"])
        if not os.path.exists(p):
            fail.append("MISSING %s (%s)" % (uid, u["cn_file"]))
            continue
        seg = io.open(p, encoding="utf-8").read()
        en_seg = io.open(os.path.join(out, u["en_file"]), encoding="utf-8").read()
        if seg == en_seg:
            fail.append("UNTOUCHED %s — 与英文逐字节相同，agent 没干活" % uid)
        parts.append((u["byte_range"][0], seg))
    if fail:
        print("COLLECT FAILED:")
        for f in fail:
            print("  " + f)
        return 1

    parts.sort()
    page_cn = "".join(s for _, s in parts)

    out_doc = {"label": en.get("label"), "folders": {}, "entries": {}}
    # shell
    shell = jload(os.path.join(out, "SYS-U1.cn.json"))

    # 日志条目名与它唯一那一页的页名。
    #
    # 这两串英文都是 "MU/TH/ER Instructions."，曾经是 T-FROZEN —— 系统有 6 处
    # `game.journal.getName("MU/TH/ER Instructions.")`，其中 4 处裸解引用。
    # 现在由 1-系统汉化插件/scripts/alienrpg-hardcoded-cn.mjs 的**通道 F 译名回退
    # 垫片**兜住（installNameFallback / NAME_FALLBACKS.journal），所以可以译。
    #
    # ⚑ LOCKSTEP：SYS-J-name.cn.json 的 journal_name 必须与那个文件里
    #   `NAME_FALLBACKS.journal[0].cn` **逐字节相等**。垫片一旦被删，
    #   这个文件也必须删掉（回落到英文原串），否则首次开世界就炸。
    #   7-其他内容/DO-NOT-TRANSLATE.json 与
    #   4-常用脚本/qa/adversarial_hardcoded_patch.mjs 的 S 组都盯着这条依赖。
    jname_path = os.path.join(out, "SYS-J-name.cn.json")
    jname = jload(jname_path) if os.path.exists(jname_path) else {}
    journal_name = jname.get("journal_name", adv_en["journals"][JKEY]["name"])
    page_name = jname.get("page_name", adv_en["journals"][JKEY]["pages"][JKEY]["name"])

    adv_cn = {
        "name": shell["name"],
        "description": shell["description"],
        "folders": shell["folders"],
        "macros": shell["macros"],
        "journals": {JKEY: {"name": journal_name,
                            "pages": {JKEY: {
                                "name": page_name,
                                "text": page_cn}}}},
        "tables": jload(os.path.join(out, "SYS-T1.cn.json")),
        "items": jload(os.path.join(out, "SYS-I1.cn.json")),
    }
    out_doc["entries"][ADV] = adv_cn
    jdump(CN, out_doc)
    print("wrote %s" % CN)
    print("  page: %d chars (en %d)" % (len(page_cn), len(text)))
    print("  journal name: %r   page name: %r" % (journal_name, page_name))
    if journal_name != adv_en["journals"][JKEY]["name"]:
        print("  ⚠ 日志名已译 —— 依赖 alienrpg-hardcoded-cn.mjs 的通道 F 译名回退垫片")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("ALIEN_PARALLEL_ROOT")
                    or os.path.join(ROOT, "6-工作区", "phase2"))
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    return cmd_collect(a.out) if a.collect else cmd_prep(a.out)


if __name__ == "__main__":
    sys.exit(main())
