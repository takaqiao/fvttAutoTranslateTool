#!/usr/bin/env python3
"""把一个内容包（starterset / corerules）切成互不重叠的翻译单元。

  python prep_pack_units.py --pack starterset [--out <dir>]
  python prep_pack_units.py --pack starterset --collect [--out <dir>]

与 prep_sys_units.py 的区别
--------------------------
系统包只有一页正文，按 h1 字节区间切就够了。内容包有多页 + 大量非期刊文档，
所以这里切两种单元：

  journal_section  某一页的 [start,end) 字节区间（同样按 h1 边界，同样断言铺满）
  collection       tables / actors / items / scenes / folders 的一个子集（按 key 划分）

两种单元都由**脚本算出的显式成员表**定义，agent 拿到的是切好的片段。
不重叠是结构保证的，不是靠任务书叮嘱——Phase 1 那次四个 agent 各自判断边界，
结果 162 键无人认领、57 键多方认领。

⚠ 重伤表（crit tables）是特例，见 CRIT_TABLES 与 §重伤表的两条铁律。
"""
import argparse
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

PACKS = {
    "starterset": {
        "repo": "2-新手包汉化插件",
        "file": "alien-evolved-starterset.alien-evolved-starter-set.json",
        "adventure": "Alien Evolved Starter Set",
    },
    "corerules": {
        "repo": "3-核心书汉化插件",
        "file": "alien-evolved-corerules.alien-evolved-core-rules.json",
        "adventure": "Alien Evolved Core Rules",
    },
}

# ── 重伤表：机器解析，两条铁律 ─────────────────────────────────────────────
# actor.mjs 把结果正文剥标签后按 /[:] |<br \/>/gi 切，再读**固定下标**。
# 切出来偶数位是字段名、奇数位是值。于是：
#
#   铁律一  分隔符必须是 **ASCII 冒号 + 空格**。写成全角「：」整个 split 形状就变了，
#           下标全部错位。字段名可以译（伤势: / 致命: / 时限: / 效果: / 恢复时间:），
#           但那个 `: ` 一个字节都不能动。
#   铁律二  被 switch 读的**值**（FATAL / TIME LIMIT / HEALING TIME 三格）在
#           lang 的 8 个键翻过来之前，必须保持英文。INJURY 名与 EFFECTS 正文可以译。
#
# 这四张表在 starterset 与 corerules 里都有，**两个包必须同时翻**，
# 所以 Phase 4 只译名字与效果，值与 lang 键留到 Phase 5 一起翻。
CRIT_TABLES = [
    "EV - Critical Injuries", "Critical injuries",
    "Critical Injuries on Xenomorphs", "Critical Injuries on Synthetics",
    "Sub Table Critical Injuries on Xenomorphs",
]

# 与系统包逐字节相同、已在 Phase 2 译过的表 —— 用 tm/fill_twin.py 灌，别重译
ALREADY_TRANSLATED_ELSEWHERE = ["Panic Response Table", "Stress Response Table", "Panic Table"]


def jload(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(o, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def vis(t):
    return len(re.sub(r"<[^>]+>", "", t))


def slice_page(text, target_chars):
    """按 h1（不够就 h2）边界把一页切成若干 [start,end)，每块尽量接近 target。"""
    for pat in (r"<h1[^>]*>", r"<h2[^>]*>"):
        starts = [m.start() for m in re.finditer(pat, text)]
        if len(starts) >= 2:
            break
    if not starts or starts[0] != 0:
        starts = [0] + starts
    bounds = starts + [len(text)]
    chunks, lo = [], 0
    for i in range(1, len(bounds)):
        if bounds[i] - lo >= target_chars or i == len(bounds) - 1:
            chunks.append((lo, bounds[i]))
            lo = bounds[i]
    if lo < len(text):
        chunks[-1] = (chunks[-1][0], len(text))
    return chunks


def cmd_prep(pack, out, target):
    cfg = PACKS[pack]
    en = jload(os.path.join(ROOT, cfg["repo"], "compendium", "en", cfg["file"]))
    adv = en["entries"][cfg["adventure"]]
    os.makedirs(os.path.join(out, "pages"), exist_ok=True)
    index = {}

    # ---- journal sections -------------------------------------------------
    n = 0
    for jname, j in adv.get("journals", {}).items():
        for pname, page in (j.get("pages") or {}).items():
            text = page.get("text") or ""
            if not text.strip():
                continue                      # 纯图片页：只有页名要译，进 shell 单元
            chunks = slice_page(text, target) if len(text) > target * 1.5 else [(0, len(text))]
            pos = 0
            for ci, (lo, hi) in enumerate(chunks):
                if lo != pos:
                    sys.exit("SLICE GAP in %r at %d (expected %d)" % (pname, lo, pos))
                pos = hi
                n += 1
                uid = "%s-J%02d" % (pack[:2].upper(), n)
                seg = text[lo:hi]
                io.open(os.path.join(out, "pages", uid + ".en.html"), "w",
                        encoding="utf-8", newline="\n").write(seg)
                index[uid] = {
                    "kind": "journal_section",
                    "journal": jname, "page": pname,
                    "chunk": "%d/%d" % (ci + 1, len(chunks)),
                    "byte_range": [lo, hi],
                    "raw_chars": len(seg), "visible_chars": vis(seg),
                    "en_file": "pages/%s.en.html" % uid,
                    "cn_file": "pages/%s.cn.html" % uid,
                }
            if pos != len(text):
                sys.exit("SLICE TAIL in %r: covered %d of %d" % (pname, pos, len(text)))

    # ---- collections ------------------------------------------------------
    tables = adv.get("tables", {})
    crit = {k: v for k, v in tables.items() if k in CRIT_TABLES}
    reuse = {k: v for k, v in tables.items() if k in ALREADY_TRANSLATED_ELSEWHERE}
    rest = {k: v for k, v in tables.items()
            if k not in CRIT_TABLES and k not in ALREADY_TRANSLATED_ELSEWHERE}

    def add(uid, kind, title, note, payload):
        if not payload:
            return
        index[uid] = {"kind": kind, "title": title, "note": note, "payload": payload,
                      "raw_chars": len(json.dumps(payload, ensure_ascii=False))}

    P = pack[:2].upper()
    add(P + "-T-CRIT", "tables", "重伤表（%d 张）" % len(crit),
        "⚠⚠ 机器解析表，两条铁律：① 字段名后的分隔符必须是 **ASCII 冒号+空格**，"
        "写成全角「：」会让 actor.mjs 的 split 形状改变、固定下标全部错位；"
        "② FATAL / TIME LIMIT / HEALING TIME 三格的**值**保持英文，"
        "等 Phase 5 与 lang 的 8 个键一起翻。INJURY 名与 EFFECTS 正文可以译。"
        "这几张表 corerules 里也有，两个包必须同时动。", crit)
    add(P + "-T-REUSE", "tables_reuse", "已在别处译过的表（%d 张）" % len(reuse),
        "⚑ 与系统包逐字节相同，Phase 2 已译。**不要重译**——"
        "用 tm/fill_twin.py 从 1-系统汉化插件 灌过来，然后逐字节比对。", reuse)
    add(P + "-T-REST", "tables", "其余表（%d 张）" % len(rest), "", rest)
    add(P + "-A-CAST", "actors", "预设角色",
        "⚠ system.general.relOne/relTwo 存的是**其他角色的姓氏**，必须跟着译名走。",
        {k: v for k, v in adv.get("actors", {}).items() if not k.startswith("EV -")})
    add(P + "-A-CRE", "actors", "生物 actor",
        "⚠ system.rTables / cTables 存的是**字面表名**，是 T-FROZEN。"
        "⚑ 生命周期六词（抱脸虫/破胸体/卵/工蜂…）目前无中文先例，见 PROJECT.md §7.2.0。",
        {k: v for k, v in adv.get("actors", {}).items() if k.startswith("EV -")})
    add(P + "-I", "items", "物品（%d 件）" % len(adv.get("items", {})), "", adv.get("items", {}))
    add(P + "-U", "shell", "Adventure 名/描述 + 文件夹 + 场景 + 纯图片页名",
        "⚠ Adventure 名与部分文件夹名是 T-FROZEN，逐条查册子。",
        {"name": adv.get("name"), "description": adv.get("description"),
         "folders": adv.get("folders", {}), "scenes": adv.get("scenes", {}),
         "image_only_pages": {jn: [pn for pn, p in (j.get("pages") or {}).items()
                                   if not (p.get("text") or "").strip()]
                              for jn, j in adv.get("journals", {}).items()
                              if any(not (p.get("text") or "").strip()
                                     for p in (j.get("pages") or {}).values())}})

    jdump(os.path.join(out, "index.json"), index)
    total = sum(u["raw_chars"] for u in index.values())
    print("pack=%s  units=%d  total_raw=%d" % (pack, len(index), total))
    for uid in sorted(index):
        u = index[uid]
        label = u.get("title") or ("%s / %s %s" % (u.get("journal", ""), u.get("page", ""), u.get("chunk", "")))
        print("  %-12s %-52s raw=%7d" % (uid, label[:52], u["raw_chars"]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, choices=sorted(PACKS))
    ap.add_argument("--out")
    ap.add_argument("--target", type=int, default=30000, help="每个期刊单元的目标原始字符数")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "6-工作区", "phase4-" + a.pack)
    if a.collect:
        sys.exit("--collect 尚未实现；先跑 prep")
    return cmd_prep(a.pack, out, a.target)


if __name__ == "__main__":
    sys.exit(main())
