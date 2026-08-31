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


def _heading_starts(text, levels=("1", "2", "3")):
    """收集指定层级标题的起始偏移，去重后升序。"""
    starts = set()
    for lv in levels:
        starts.update(m.start() for m in re.finditer(r"<h%s[^>]*>" % lv, text))
    return sorted(starts)


def slice_page(text, target_chars):
    """按标题边界把一页切成若干 [start,end)，每块尽量接近 target。

    ⚠ 尾块问题：只用 h1（或只用 h2）时，最后一块会把**所有余量**吞掉。
    starterset 的 Hope's Last Day 第 3 块实测 43,325 字（目标 26,000），
    因为那一段只有 2 个 h2 —— 但它有 10 个 h3。
    所以策略是：先用最粗的层级切；**任何仍然超过 1.6×target 的块，
    再用更细的层级在块内二次切**。二次切只在块内进行，所以铺满性不受影响。
    """
    def cut(lo, hi, levels):
        seg = text[lo:hi]
        starts = [lo + s for s in _heading_starts(seg, levels) if s > 0]
        if not starts:
            return [(lo, hi)]
        out, cur = [], lo
        for s in starts:
            if s - cur >= target_chars:
                out.append((cur, s))
                cur = s
        out.append((cur, hi))
        return out

    chunks = cut(0, len(text), ("1",)) if _heading_starts(text, ("1",)) else [(0, len(text))]
    # 二次切：h2，再 h3
    for levels in (("1", "2"), ("1", "2", "3")):
        refined = []
        for lo, hi in chunks:
            if hi - lo > target_chars * 1.6:
                refined.extend(cut(lo, hi, levels))
            else:
                refined.append((lo, hi))
        chunks = refined

    # 铺满性自检：块必须首尾相接、覆盖全页
    pos = 0
    for lo, hi in chunks:
        if lo != pos:
            sys.exit("slice_page: gap/overlap at %d (expected %d)" % (lo, pos))
        pos = hi
    if pos != len(text):
        sys.exit("slice_page: tail %d of %d" % (pos, len(text)))
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
    return cmd_collect(a.pack, out) if a.collect else cmd_prep(a.pack, out, a.target)




# ---------------------------------------------------------------- collect --
def cmd_collect(pack, out):
    """把单元产出装配成 compendium/cn/<pack>.json。

    与 prep_sys_units.py 的 collect 同构，但多两件事：
      · 期刊分段要按 byte_range 排序拼回**各自的页**（内容包有多页，系统包只有一页）；
      · 日志名与页名来自 ST-NAMES.cn.json（切单元时漏掉的那一层，见该文件 _why_it_was_missed）。
    """
    cfg = PACKS[pack]
    en_path = os.path.join(ROOT, cfg["repo"], "compendium", "en", cfg["file"])
    cn_path = os.path.join(ROOT, cfg["repo"], "compendium", "cn", cfg["file"])
    en = jload(en_path)
    adv_en = en["entries"][cfg["adventure"]]
    index = jload(os.path.join(out, "index.json"))
    P = pack[:2].upper()

    fail = []

    def need(name):
        p = os.path.join(out, name)
        if not os.path.exists(p):
            fail.append("MISSING %s" % name)
            return None
        return jload(p)

    names = need("ST-NAMES.cn.json") if pack == "starterset" else {}
    shell = need("%s-U.cn.json" % P)
    items = need("%s-I.cn.json" % P)
    cast = need("%s-A-CAST.cn.json" % P)
    cre = need("%s-A-CRE.cn.json" % P)
    t_crit = need("%s-T-CRIT.cn.json" % P)
    t_rest = need("%s-T-REST.cn.json" % P)
    t_reuse = need("%s-T-REUSE.cn.json" % P)

    # 期刊：按 (journal, page) 归组，组内按 byte_range 排序拼接
    pages = {}
    for uid, u in index.items():
        if u.get("kind") != "journal_section":
            continue
        p = os.path.join(out, u["cn_file"])
        if not os.path.exists(p):
            fail.append("MISSING %s (%s)" % (uid, u["cn_file"]))
            continue
        seg = io.open(p, encoding="utf-8").read()
        en_seg = io.open(os.path.join(out, u["en_file"]), encoding="utf-8").read()
        if seg == en_seg:
            fail.append("UNTOUCHED %s — 与英文逐字节相同，agent 没干活" % uid)
        pages.setdefault((u["journal"], u["page"]), []).append((u["byte_range"][0], seg))

    if fail:
        print("COLLECT FAILED:")
        for f in fail:
            print("  " + f)
        return 1

    jn_map = (names or {}).get("journals", {})
    pn_map = (names or {}).get("pages", {})

    journals = {}
    for jname, j in adv_en.get("journals", {}).items():
        out_pages = {}
        for pname, page in (j.get("pages") or {}).items():
            parts = pages.get((jname, pname))
            entry = {"name": pn_map.get(pname, page.get("name", pname))}
            if parts:
                parts.sort()
                entry["text"] = "".join(s for _, s in parts)
            else:
                # 纯图片页：只有名字，取 shell 里的 image_only_pages
                iop = (shell or {}).get("image_only_pages", {}).get(jname, {})
                if isinstance(iop, dict) and pname in iop:
                    entry["name"] = iop[pname]
            out_pages[pname] = entry
        journals[jname] = {"name": jn_map.get(jname, j.get("name", jname)), "pages": out_pages}

    tables = {}
    for src in (t_crit, t_rest, t_reuse):
        if src:
            tables.update(src)

    actors = {}
    for src in (cast, cre):
        if src:
            actors.update(src)

    adv_cn = {
        "name": (shell or {}).get("name", adv_en.get("name")),
        "description": (shell or {}).get("description", adv_en.get("description")),
        "folders": (shell or {}).get("folders", {}),
        "scenes": (shell or {}).get("scenes", {}),
        "journals": journals,
        "tables": tables,
        "items": items or {},
        "actors": actors,
    }
    jdump(cn_path, {"label": en.get("label"), "folders": {}, "entries": {cfg["adventure"]: adv_cn}})

    print("wrote %s" % cn_path)
    for coll in ("journals", "tables", "items", "actors", "folders", "scenes"):
        print("  %-9s en=%-4d cn=%d" % (coll, len(adv_en.get(coll, {})), len(adv_cn.get(coll, {}))))
    for (jn, pn), parts in sorted(pages.items()):
        src = adv_en["journals"][jn]["pages"][pn].get("text") or ""
        got = "".join(s for _, s in sorted(parts))
        print("  page %-34s en=%7d cn=%7d" % (pn[:34], len(src), len(got)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
