# -*- coding: utf-8 -*-
"""名一致性闸：用 id 绑定目标文档、标签另存的两类对象。

背景（PROJECT.md 第 1 节盲区表第 6 项）
--------------------------------------
`scan_markup_targets.py` 只看 `@UUID[目标]{标签}` 的方括号，而 Foundry 里还有两类
「标签和目标分开存」的东西，它一个都管不到：

* **场景针脚 Note** —— `text` 是画在地图上的字，真正打开哪份文档由 `entryId` /
  `pageId` 决定。Babele 把 `text` 当 `textCollection` 翻译，页名由另一处翻译。
* **RollTable 结果 TableResult** —— `name` 是掷骰结果那一行显示的字，实际抽到哪份
  文档由 `documentUuid`（Foundry ≤ v12 是 `documentCollection` + `documentId`）决定。

两边各自都是合法中文、覆盖率 100%、没有任何标记，所以既有的每一项检查都全盲。
玩家侧的症状是：点了写着「A」的针脚，弹出来的窗口标题是「B」。

判据
----
**先查英文再判中文。** 对每一条针脚 / 结果行：

* `NOT_BOUND`    —— 结果行 `type=text`、或针脚根本没写 `entryId`/`pageId`（纯地图标签），
  压根没有目标文档，不是本闸的定义域
* `NO_LABEL`     —— 英文侧标签为空（Foundry 直接显示目标文档名），没有可译的东西
* `BY_DESIGN`    —— **英文侧标签本来就 ≠ 目标文档英文名**（`Corpse Loot (Arctus Plateau)`
  指向 `Corpse Loot`）。上游有意起的别名，中文跟着不同是对的，**不算缺陷**
* `OUT_OF_SCOPE` —— 目标落在本项目不负责的包里（dnd5e 的 items/equipment24 等），
  它的中文来自别的模块，无从核对；**目标包压根没导 id 表的也归这一档**（见下）
* `UNCERTAIN`    —— id 悬空（上游删了目标文档）、目标没有中文名、或标签本身没译
* `OK`           —— 英文同名，中文也同名
* `BROKEN`       —— **英文同名，目标的中文名存在，但标签的中文与它不同** ← 唯一算缺陷的一档

id 绑定不在 Babele 译文文件里
-----------------------------
`compendium/en|cn/*.json` 按**名字**建键、且只存可译字段，`entryId` / `documentUuid`
只存在于 LevelDB packs。所以本脚本吃一份由同目录 `dump_bindings.mjs` 导出的绑定表：

    node qa/dump_bindings.mjs --package <ember 模块> --package <crucible 系统> \
                              --package <dnd5e 系统> --out bindings.json

⚠ **三个包目录一次全导**（2026-08-15 修好的机制缺陷）
--------------------------------------------------
只导 ember 一个包时，凡目标在别的包里的行都解析不出来，会被当成「上游删了目标文档 /
悬空 id」塞进 `UNCERTAIN`。实测这一个原因就占了 199 条 UNCERTAIN 里的 188 条
（114 条指向 `Compendium.dnd5e.*`、74 条指向 `Compendium.crucible.*`），
而它们 188/188 全都能解析 —— 判据没错，是**输入缺了包**。

现在脚本知道 `bindings.json` 里到底导了哪些包（`dump_bindings.mjs` 会写 `packages`
字段）：目标 uuid 指向的包**不在**已导列表里时判 `OUT_OF_SCOPE`，理由写
「目标在本包之外（<包名>）」；只有目标包已导、id 仍然找不到，才判 `UNCERTAIN`
的「上游悬空」。三包全导后本库实测 `table-result` 的 UNCERTAIN 由 189 降到 1
（`Compendium.crucible.equipment.Item.prybar0000000000`，crucible 0.10.1 里确实
没有这份文档，是上游真悬空）。

没有绑定表时会退化成 `--fallback` 模式（用「英文标签与某个文档英文名逐字节相同」
当绑定的代理判据，也就是两个一次性探针原来的做法）—— 那是启发式，会漏掉
「英文标签就是别名」的绑定，只当没有 Foundry 安装时的兜底。

另：**`textCollection` 的键是按 `text` 建的，不是 `_id`**（PROJECT.md 第 8 节
2026-08-12 那条）。本脚本一律按 `text` 取针脚译文，别改成 `_id`。

用法
----
    python scan_name_binding.py --repo <repo> [--repo <另一个>] \
           --bindings bind_ember.json [--bindings bind_crucible.json] \
           [--out <json>] [--md <md>] [--show 40] [--fallback]
"""
import argparse
import collections
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 译文文件里「一组同类文档」的键名。键既可能是文档英文名，也可能是 _id（见第 8 节
# 2026-08-06「同名文档合并到名字键，标量冲突才用 _id 键」），两种都建索引。
COLLECTION_KEYS = {
    "entries", "journals", "pages", "tables", "items", "actors",
    "scenes", "macros", "effects", "playlists",
}

# LevelDB 里的 kind（顶层用 collection 名，嵌套时逐层拼字段名）→ 译文文件里的角色。
KIND_TO_ROLE = {
    "journal": "journals", "journals": "journals", "pages": "pages",
    "tables": "tables", "items": "items", "actors": "actors",
    "scenes": "scenes", "macros": "macros", "effects": "effects",
}


# --------------------------------------------------------------------------- 译文侧

def load_packs(repos):
    """{'<pkg>.<pack>': {'en':…, 'cn':…, 'repo':…, 'file':…}}"""
    packs = {}
    for repo in repos:
        en_dir = os.path.join(repo, "compendium", "en")
        cn_dir = os.path.join(repo, "compendium", "cn")
        if not os.path.isdir(en_dir):
            continue
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            en = json.load(open(os.path.join(en_dir, fn), encoding="utf-8"))
            cn_path = os.path.join(cn_dir, fn)
            cn = json.load(open(cn_path, encoding="utf-8")) if os.path.isfile(cn_path) else {}
            packs[fn[:-5]] = {"en": en, "cn": cn, "repo": repo, "file": fn}
    return packs


def build_name_index(en, cn):
    """{role: {english name: Counter(chinese name)}}，同时按键和按 `name` 字段建。"""
    idx = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))

    def walk(e, c, role):
        if not isinstance(e, dict):
            return
        for k, ev in e.items():
            cv = c.get(k) if isinstance(c, dict) else None
            if k in COLLECTION_KEYS and isinstance(ev, dict):
                for name_key, node in ev.items():
                    if not isinstance(node, dict):
                        continue
                    cnode = (cv or {}).get(name_key) if isinstance(cv, dict) else None
                    cn_name = (cnode or {}).get("name") if isinstance(cnode, dict) else None
                    en_name = node.get("name") if isinstance(node.get("name"), str) else None
                    if isinstance(cn_name, str) and cn_name:
                        idx[k][name_key][cn_name] += 1
                        if en_name:
                            idx[k][en_name][cn_name] += 1
                    walk(node, cnode if isinstance(cnode, dict) else {}, k)
            elif isinstance(ev, dict):
                walk(ev, cv if isinstance(cv, dict) else {}, role)

    walk(en, cn, None)
    return idx


# --------------------------------------------------------------------------- 绑定侧

def load_bindings(paths):
    """返回 (ids, notes, results, dumped_pkgs)。

    `dumped_pkgs` 是「这份绑定表里到底有哪些包的 id」。新版 `dump_bindings.mjs` 直接写
    `packages` 字段；旧的输出没有，就从每条 id 记录的 `pkg` 反推（一样准，只是慢一点）。
    没有它就分不清「上游删了目标」和「这个包我压根没导」—— 那正是 187 条误报的来源。
    """
    ids = collections.defaultdict(list)
    notes, results, pkgs = [], [], set()
    for p in paths:
        b = json.load(open(p, encoding="utf-8"))
        for entry in (b.get("packages") or []):
            if isinstance(entry, dict) and entry.get("pkg"):
                pkgs.add(entry["pkg"])
        for k, v in (b.get("ids") or {}).items():
            v = v if isinstance(v, list) else [v]
            ids[k].extend(v)
            if not b.get("packages"):
                pkgs.update(o.get("pkg") for o in v if isinstance(o, dict) and o.get("pkg"))
        notes.extend(b.get("notes") or [])
        results.extend(b.get("results") or [])
    pkgs.discard(None)
    return ids, notes, results, pkgs


def resolve(ids, doc_id, prefer_pkg=None, prefer_pack=None):
    """同 id 可能在多个包里出现（两本 adventure 互为副本、内嵌拷贝沿用源 id）。
    先在指定的包里找，再退到同 package，最后才全局取第一个。"""
    bucket = ids.get(doc_id) or []
    if not bucket:
        return None
    for o in bucket:
        if o.get("pkg") == prefer_pkg and o.get("pack") == prefer_pack:
            return o
    for o in bucket:
        if o.get("pkg") == prefer_pkg:
            return o
    return bucket[0]


def parse_uuid(uuid, own_packkey):
    """`Compendium.<pkg>.<pack>.<Type>.<id>` / `<Type>.<id>` → (packkey, id)。

    世界相对 uuid（`Item.xxx` / `RollTable.xxx`）在合集 Adventure 里指的是同一本
    冒险内部的文档，所以退回调用方自己的包。"""
    parts = uuid.split(".")
    if parts[0] == "Compendium":
        return ".".join(parts[1:3]), parts[-1]
    return own_packkey, parts[-1]


def library_lookup(indexes, roles, en_name):
    """退化模式用：全库按英文名找中文名。

    只在**全库唯一**时才认；英文碰巧同名（`Dagger` 在 dnd5e 与 crucible 各有一件）
    时返回歧义，交给 UNCERTAIN。两个一次性探针只在同一个包里找，所以跨包的目标
    （表结果指向 `crucible.equipment`）它们一条都发现不了。"""
    cand = collections.Counter()
    for idx in indexes.values():
        for role in roles:
            cand.update((idx.get(role) or {}).get(en_name) or {})
    return cand


def role_of(kind):
    """LevelDB 的 kind → 译文文件里的角色。顶层文档（kind 不含点）落在 `entries`。"""
    if not kind:
        return None
    if "." not in kind:
        return "entries"
    return KIND_TO_ROLE.get(kind.rsplit(".", 1)[1])


# --------------------------------------------------------------------------- 判据

def judge(en_label, cn_label, target, cand, ambiguous=False, in_scope=True,
          missing_pkg=None):
    """返回 (verdict, target_cn_names:dict, reason)。`cand` 是目标文档的中文名候选集。

    `missing_pkg` —— 目标 uuid 指向的包**没有**出现在 `--bindings` 里。这不是缺陷、
    也不是悬空 id，只是我们没导那个包的 id 表，所以归 `OUT_OF_SCOPE` 而不是
    `UNCERTAIN`（2026-08-15 之前一律写成「上游删了目标文档」，187 条全是误报）。
    """
    if target is None:
        if missing_pkg:
            return ("OUT_OF_SCOPE", {},
                    f"目标在本包之外（{missing_pkg}）：--bindings 里没有该包的 id 表，无从核对")
        return "UNCERTAIN", {}, "目标 id 解析不出来（目标包已导，仍找不到 → 上游悬空 id）"
    tgt_en = target.get("name")
    if not isinstance(tgt_en, str) or not tgt_en:
        return "UNCERTAIN", {}, "目标文档没有英文名"
    if en_label.strip() != tgt_en.strip():
        return "BY_DESIGN", {}, f"英文侧本来就不同名：标签 {en_label!r} vs 目标 {tgt_en!r}"
    if not in_scope:
        return "OUT_OF_SCOPE", {}, "目标在本项目不负责的包里（中文来自别的模块）"
    cand = dict(cand or {})
    if not cand:
        return "UNCERTAIN", {}, "目标文档没有中文名（该条未译）"
    if not cn_label:
        return "UNCERTAIN", cand, "标签本身没有中文"
    if cn_label in cand:
        return "OK", cand, ""
    if ambiguous:
        return "UNCERTAIN", cand, "目标不唯一（同一标签绑了多个目标 / 英文碰巧同名）"
    return "BROKEN", cand, "英文同名，中文不同名"


# --------------------------------------------------------------------------- 主流程

def scan(packs, ids, notes, results, use_fallback, dumped_pkgs=frozenset()):
    findings = []
    stats = collections.Counter()
    indexes = {pk: build_name_index(v["en"], v["cn"]) for pk, v in packs.items()}

    # 绑定行按 (packkey, adventure, scene, text) / (packkey, adventure, table, range) 归组
    note_bind = collections.defaultdict(list)
    for n in notes:
        pk = f"{n.get('pkg')}.{n.get('pack')}"
        note_bind[(pk, n.get("adventure"), n.get("scene"), n.get("text") or "")].append(n)
    res_bind = collections.defaultdict(list)
    for r in results:
        pk = f"{r.get('pkg')}.{r.get('pack')}"
        res_bind[(pk, r.get("adventure"), r.get("table"), r.get("range"))].append(r)

    for packkey, pv in sorted(packs.items()):
        en, cn = pv["en"], pv["cn"]
        idx = indexes[packkey]
        for ent_name, ev in (en.get("entries") or {}).items():
            cv = (cn.get("entries") or {}).get(ent_name) or {}

            # ---------------- 场景针脚 ----------------
            for sname, sv in (ev.get("scenes") or {}).items():
                csv_ = ((cv.get("scenes") or {}).get(sname) or {})
                for text, en_label in (sv.get("notes") or {}).items():
                    if not isinstance(en_label, str):
                        continue
                    stats["notes_seen"] += 1
                    cn_label = (csv_.get("notes") or {}).get(text)
                    rows = note_bind.get((packkey, ent_name, sname, text)) or \
                           note_bind.get((packkey, None, sname, text)) or []
                    # 完全没写 entryId/pageId 的针脚是纯地图标签，没有目标可比
                    bound = [r for r in rows if r.get("entryId") or r.get("pageId")]
                    if rows and not bound:
                        stats["note_NOT_BOUND"] += 1
                        continue
                    targets, role, dangling = [], "pages", False
                    for row in bound:
                        t = None
                        if row.get("pageId"):
                            t = resolve(ids, row["pageId"], row.get("pkg"), row.get("pack"))
                            if t is None:
                                dangling = True
                        if t is None and row.get("entryId"):
                            t = resolve(ids, row["entryId"], row.get("pkg"), row.get("pack"))
                            if t is not None:
                                role = role_of(t.get("kind")) or "journals"
                        elif t is not None:
                            role = role_of(t.get("kind")) or "pages"
                        targets.append(t)
                    have = [t for t in targets if t]
                    ambiguous = len({t.get("name") for t in have}) > 1
                    target = have[0] if have else None
                    if target is not None:
                        # 目标未必在本包里（针脚可以指向别的合集的日志）。先查目标所在
                        # 包的索引，再并上本包的 —— 只查本包会把跨包目标误判成
                        # 「目标没有中文名」，把本该是 BROKEN 的藏进 UNCERTAIN。
                        tgt_pk = f"{target.get('pkg')}.{target.get('pack')}"
                        cand0 = collections.Counter()
                        cand0.update((indexes.get(tgt_pk, {}).get(role) or {})
                                     .get(target.get("name") or "") or {})
                        cand0.update((idx.get(role) or {}).get(target.get("name") or "") or {})
                    elif use_fallback:   # 无绑定：退化成「英文标签 == 某页英文名」
                        cand0 = library_lookup(indexes, ("pages",), en_label.strip())
                        if cand0:
                            target = {"name": en_label.strip(), "kind": "fallback", "via": "en-name"}
                            ambiguous = len(cand0) > 1
                    else:
                        cand0 = {}
                    verdict, cand, reason = judge(en_label, cn_label, target, cand0, ambiguous)
                    if dangling and verdict not in ("BROKEN",):
                        verdict, reason = "UNCERTAIN", "针脚的 pageId 悬空（上游删了那一页），会落到条目首页"
                    stats[f"note_{verdict}"] += 1
                    if verdict in ("BROKEN", "UNCERTAIN"):
                        findings.append({
                            "kind": "scene-note", "verdict": verdict, "reason": reason,
                            "repo": os.path.basename(pv["repo"]), "pack": pv["file"],
                            "batch_path": f"{ent_name}.scenes.{sname}.notes.{text}",
                            "en_label": en_label, "cn_label": cn_label,
                            "target_en": (target or {}).get("name"),
                            "target_role": role, "target_cn": cand,
                            "bound_by": "id" if bound else ("en-name" if have else None),
                        })

            # ---------------- 表结果 ----------------
            for tname, tv in (ev.get("tables") or {}).items():
                ctv = ((cv.get("tables") or {}).get(tname) or {})
                for rng, rv in (tv.get("results") or {}).items():
                    if not isinstance(rv, dict):
                        continue
                    stats["results_seen"] += 1
                    en_label = rv.get("name")
                    cn_label = ((ctv.get("results") or {}).get(rng) or {}).get("name")
                    rows = res_bind.get((packkey, ent_name, tname, rng)) or \
                           res_bind.get((packkey, None, tname, rng)) or []
                    bound = [r for r in rows if r.get("documentUuid") or r.get("documentId")]
                    if rows and not bound:
                        stats["result_NOT_BOUND"] += 1
                        continue
                    if not isinstance(en_label, str) or not en_label.strip():
                        stats["result_NO_LABEL"] += 1
                        continue
                    targets, role, missing_pkgs = [], None, []
                    for r in bound:
                        uuid = r.get("documentUuid")
                        if uuid:
                            tpack, tid = parse_uuid(uuid, packkey)
                        else:
                            tpack, tid = (r.get("documentCollection") or packkey), r.get("documentId")
                        t = resolve(ids, tid, *(tpack.split(".", 1) + [None])[:2]) if tid else None
                        if t:
                            role = role_of(t.get("kind"))
                            t = dict(t, packkey=tpack)
                        elif tid:
                            # 解析不出来时先问一句：目标那个包我们到底导没导？
                            tpkg = tpack.split(".", 1)[0]
                            if dumped_pkgs and tpkg not in dumped_pkgs:
                                missing_pkgs.append(tpkg)
                        targets.append(t)
                    have = [t for t in targets if t]
                    ambiguous = len({t.get("name") for t in have}) > 1
                    target = have[0] if have else None
                    tpk = (target or {}).get("packkey")
                    in_scope = (tpk in packs) if tpk else True
                    if target is not None:
                        tgt_idx = indexes.get(tpk, idx)
                        cand0 = (tgt_idx.get(role or "entries") or {}).get(target.get("name") or "") or {}
                    elif use_fallback:   # 无绑定：退化成「英文标签 == 某文档英文名」
                        cand0 = library_lookup(indexes, ("entries", "items", "tables"),
                                               en_label.strip())
                        if cand0:
                            target = {"name": en_label.strip(), "kind": "fallback",
                                      "via": "en-name"}
                            ambiguous = len(cand0) > 1
                    else:
                        cand0 = {}
                    verdict, cand, reason = judge(
                        en_label, cn_label, target, cand0, ambiguous, in_scope,
                        missing_pkg=(sorted(set(missing_pkgs))[0] if missing_pkgs else None))
                    stats[f"result_{verdict}"] += 1
                    if verdict in ("BROKEN", "UNCERTAIN"):
                        findings.append({
                            "kind": "table-result", "verdict": verdict, "reason": reason,
                            "repo": os.path.basename(pv["repo"]), "pack": pv["file"],
                            "batch_path": f"{ent_name}.tables.{tname}.results.{rng}.name",
                            "en_label": en_label, "cn_label": cn_label,
                            "target_en": (target or {}).get("name"),
                            "target_pack": (target or {}).get("packkey"),
                            "target_role": role, "target_cn": cand,
                            "bound_by": "id" if bound else ("en-name" if have else None),
                        })
    return findings, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--bindings", action="append", default=[],
                    help="dump_bindings.mjs 的输出，可重复")
    ap.add_argument("--fallback", action="store_true",
                    help="没有绑定行时退化成「英文标签 == 某文档英文名」的代理判据（启发式）")
    ap.add_argument("--out")
    ap.add_argument("--md")
    ap.add_argument("--show", type=int, default=30)
    a = ap.parse_args()

    packs = load_packs(a.repo)
    ids, notes, results, dumped_pkgs = load_bindings(a.bindings)
    if not a.bindings:
        print("⚠ 没给 --bindings：id 绑定只存在于 LevelDB packs，先跑同目录的 dump_bindings.mjs。\n"
              "  退化模式用「英文标签 == 某文档英文名」当绑定的代理判据，实测对 16 条真缺陷"
              "多报 10 条、漏报 3 条，只当兜底看。")
    elif not {"ember", "crucible", "dnd5e"} <= dumped_pkgs:
        missing = ", ".join(sorted({"ember", "crucible", "dnd5e"} - dumped_pkgs))
        print(f"⚠ 绑定表里缺这些包的 id：{missing}。指向它们的目标只能判 OUT_OF_SCOPE，"
              f"核不了名。一次全导：dump_bindings.mjs --package … --package … --package …")
    findings, stats = scan(packs, ids, notes, results, a.fallback or not a.bindings,
                           dumped_pkgs)

    broken = [f for f in findings if f["verdict"] == "BROKEN"]
    uncertain = [f for f in findings if f["verdict"] == "UNCERTAIN"]
    print(f"packs={len(packs)}  bindings: notes={len(notes)} results={len(results)}")
    print(f"scene-note   seen={stats['notes_seen']}  NOT_BOUND={stats['note_NOT_BOUND']} "
          f"OK={stats['note_OK']} BY_DESIGN={stats['note_BY_DESIGN']} "
          f"OUT_OF_SCOPE={stats['note_OUT_OF_SCOPE']} UNCERTAIN={stats['note_UNCERTAIN']} "
          f"BROKEN={stats['note_BROKEN']}")
    print(f"table-result seen={stats['results_seen']}  NOT_BOUND={stats['result_NOT_BOUND']} "
          f"NO_LABEL={stats['result_NO_LABEL']} OK={stats['result_OK']} "
          f"BY_DESIGN={stats['result_BY_DESIGN']} OUT_OF_SCOPE={stats['result_OUT_OF_SCOPE']} "
          f"UNCERTAIN={stats['result_UNCERTAIN']} BROKEN={stats['result_BROKEN']}")
    print(f"\nBROKEN={len(broken)}  UNCERTAIN={len(uncertain)}")
    for f in broken[:a.show]:
        print(f"  [{f['kind']}] {f['pack']} :: {f['batch_path']}")
        print(f"     EN   {f['en_label']!r}   ->  target EN {f['target_en']!r}")
        print(f"     CN   {f['cn_label']!r}   ->  target CN {f['target_cn']}")

    if a.out:
        json.dump({"stats": dict(stats), "findings": findings},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"-> {a.out}")
    if a.md:
        with open(a.md, "w", encoding="utf-8") as fh:
            fh.write("# 名绑定一致性（scan_name_binding.py）\n\n")
            fh.write(f"- 场景针脚 {stats['notes_seen']} 条：NOT_BOUND {stats['note_NOT_BOUND']} / "
                     f"OK {stats['note_OK']} / BY_DESIGN {stats['note_BY_DESIGN']} / "
                     f"UNCERTAIN {stats['note_UNCERTAIN']} / **BROKEN {stats['note_BROKEN']}**\n")
            fh.write(f"- 表结果 {stats['results_seen']} 条：NOT_BOUND {stats['result_NOT_BOUND']} / "
                     f"NO_LABEL {stats['result_NO_LABEL']} / OK {stats['result_OK']} / "
                     f"BY_DESIGN {stats['result_BY_DESIGN']} / "
                     f"OUT_OF_SCOPE {stats['result_OUT_OF_SCOPE']} / "
                     f"UNCERTAIN {stats['result_UNCERTAIN']} / "
                     f"**BROKEN {stats['result_BROKEN']}**\n\n")
            for title, group in (("BROKEN", broken), ("UNCERTAIN", uncertain)):
                fh.write(f"## {title}（{len(group)}）\n\n")
                for f in group:
                    fh.write(f"- `{f['pack']}` `{f['batch_path']}`  \n"
                             f"  EN `{f['en_label']}` → 目标 EN `{f['target_en']}`  \n"
                             f"  CN `{f['cn_label']}` → 目标 CN `{f['target_cn']}`  \n"
                             f"  {f['reason']}\n")
                fh.write("\n")
        print(f"-> {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
