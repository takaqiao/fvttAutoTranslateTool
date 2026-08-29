# -*- coding: utf-8 -*-
"""硬冻结字符串闸：把 `DO-NOT-TRANSLATE.json` 里的每一条拿到 `compendium/cn/*.json` 上核。

为什么单独一道闸
----------------
这三个包里有一类字符串，**代码用 `===` 跟它比**：Adventure 名、欢迎日志名、场景名、
RollTable 名、Folder 名、`PACK MULE`/`TAKE CONTROL`、以及 `None` 哨兵。翻了它们不会
报错、不会掉覆盖率、不会留下任何标记 —— 只是某个功能从此不响。实测最狠的两条：

* `Alien Creature Tables` / `Alien Mother Tables` 这两个 Folder 名一翻，
  `rollTableData.mjs` 的 `folder.contents` 对 `undefined` 取属性抛 TypeError，
  **整张异种生物卡片渲染不出来**（不是下拉框空了，是白窗）。
* 生物 actor 的 `system.rTables` 和它指向的 RollTable 名一旦不同步，
  `actor.mjs` 里 `game.tables.contents.find(...)` 返回 undefined，
  下一行 `table.roll({roll})` 直接抛未捕获 TypeError，攻击按钮死掉。

判据（只有这三档会让脚本退出非零）
----------------------------------
* `FROZEN_TRANSLATED`  —— 登记表里标 T-FROZEN 的英文键在译文里存在，且 `name` 被改了
* `EXACT_MISMATCH`     —— 12 个 skill-stunts 物品名 ≠ `lang/cn.json` 对应
                          `ALIENRPG.Skill<key>` 的值（T-EXACT，不许带英文尾巴）
* `TABLE_REF_DANGLING` —— actor 的 `rTables`/`cTables` 指向的 RollTable 英文键
                          在整个译文集里找不到，或者被显式改成了第三种写法
* `MACRO_COMMAND_TRANSLATED` —— 译文里的 macro 节点上出现了 `command` 键
                          （2026-08-29 补：Babele 默认就翻 Macro.command，
                          而三个宏的源码里带着 `t.folder.name === 'Alien Mother Tables'`
                          和 `game.tables.getName("EV - 48a. LS - DANGER EVENT DETAIL")`）

`FROZEN_TRANSLATED` 里还多了一档 `name_substring`（2026-08-29 补）：按子串判的物品名
——`i.name.includes(" RPG ")` 决定武器弹药重量算 0.5 还是 0.25 kg，译名必须仍满足
登记表 `name_substring_tests` 里三条测试中的至少一条。

其余一律只统计不判缺陷。`compendium/cn` 空的时候报 `checked=0` 并退出 0，
所以从第一天就能挂进流水线。

用法
----
    python scan_name_lookup_traps.py \
        [--register <DO-NOT-TRANSLATE.json>] \
        [--repo <汉化插件目录> ...] \
        [--lang <lang/cn.json>] [--out <json>] [--show 40]

不给 `--repo` 时默认用项目里的三个插件目录；不给 `--lang` 时默认
`1-系统汉化插件/lang/cn.json`，缺文件就跳过 T-EXACT 那一档（记 SKIPPED，不算缺陷）。

退出码：0 干净 / 1 有缺陷 / 2 用不了（登记表读不到之类）
"""
import argparse
import collections
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))

DEFAULT_REGISTER = os.path.join(PROJ, "7-其他内容", "DO-NOT-TRANSLATE.json")
DEFAULT_REPOS = [
    os.path.join(PROJ, "1-系统汉化插件"),
    os.path.join(PROJ, "2-新手包汉化插件"),
    os.path.join(PROJ, "3-核心书汉化插件"),
]
DEFAULT_LANG = os.path.join(PROJ, "1-系统汉化插件", "lang", "cn.json")
DEFAULT_LANG_EN = os.path.join(PROJ, "1-系统汉化插件", "lang", "en.json")

# 译文文件里「一组同类文档」的键名 -> 角色。值是 {英文名: 译文节点}，
# 唯独 folders 是 {英文名: 中文字符串}（Babele 的 folders 转换器存标量）。
COLLECTION_ROLES = {
    "entries": "adventure",
    "journals": "journal",
    "pages": "page",
    "tables": "table",
    "results": "result",
    "items": "item",
    "actors": "actor",
    "scenes": "scene",
    "macros": "macro",
    "playlists": "playlist",
    "effects": "effect",
}
SCALAR_COLLECTIONS = {"folders": "folder"}

# actor 上「表名引用」这个字段在译文文件里可能叫的几种名字。Babele mapping 可以给
# 字段起别名（survey 提的 alienRollTableRef 就把 system.rTables 起名叫 rollTable），
# 所以这里把所有可能的写法都当同一个字段看。
RTABLE_ALIASES = {"rTables", "rollTable", "system.rTables", "rtables"}
CTABLE_ALIASES = {"cTables", "critTable", "system.cTables", "ctables"}


# --------------------------------------------------------------------------- 载入

def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def flatten_lang(d, prefix=""):
    out = {}
    for k, v in d.items():
        n = prefix + "." + k if prefix else k
        if isinstance(v, dict):
            out.update(flatten_lang(v, n))
        else:
            out[n] = v
    return out


def load_cn_files(repos, pack_identity):
    """[(repo, filename, pack, dict)]，只读 compendium/cn 下的 *.json（跳过下划线开头的）。

    `pack` 是登记表 `measured_against.pack_identity` 里那三个逻辑包名之一，靠 Babele
    译文文件名 `<moduleId>.<packName>.json` 认出来 —— **不靠目录名**，这样脚本在
    %TEMP% 的测试装置里和在项目里表现一致。认不出来的文件 pack=None。
    """
    by_file = {v["babele_file"]: k for k, v in pack_identity.items()}
    files = []
    for repo in repos:
        cn_dir = os.path.join(repo, "compendium", "cn")
        if not os.path.isdir(cn_dir):
            continue
        for fn in sorted(os.listdir(cn_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            pack = by_file.get(fn)
            try:
                files.append((repo, fn, pack, load_json(os.path.join(cn_dir, fn))))
            except Exception as exc:                       # 坏 JSON 也算缺陷，别静默跳过
                files.append((repo, fn, pack, {"__parse_error__": str(exc)}))
    return files


# --------------------------------------------------------------------------- 建索引

class Index:
    """role -> 英文名 -> [ {repo, file, path, cn_name, node} ]

    同一个英文名可能在多个包里各有一份（三个包共用 `Panic Response Table`），
    所以每条都存列表，判据逐条走。
    """

    def __init__(self):
        self.by_role = collections.defaultdict(lambda: collections.defaultdict(list))
        self.parse_errors = []
        self.nodes = 0

    def add(self, role, en_name, rec):
        self.by_role[role][en_name].append(rec)
        self.nodes += 1

    def get(self, role, en_name):
        return self.by_role.get(role, {}).get(en_name, [])

    def any_role(self, en_name, roles):
        out = []
        for r in roles:
            out.extend(self.get(r, en_name))
        return out

    def has_role(self, role):
        return bool(self.by_role.get(role))

    def in_pack(self, role, en_name, pack):
        return [r for r in self.get(role, en_name) if r["pack"] == pack]

    def packs_with_role(self, role):
        return {r["pack"] for recs in self.by_role.get(role, {}).values() for r in recs}


def build_index(cn_files):
    idx = Index()
    for repo, fn, pack, doc in cn_files:
        if "__parse_error__" in doc:
            idx.parse_errors.append((repo, fn, doc["__parse_error__"]))
            continue

        def walk(node, path):
            if not isinstance(node, dict):
                return
            for key, val in node.items():
                if key in SCALAR_COLLECTIONS and isinstance(val, dict):
                    role = SCALAR_COLLECTIONS[key]
                    for en_name, cn_val in val.items():
                        idx.add(role, en_name, {
                            "repo": os.path.basename(repo), "file": fn, "pack": pack,
                            "path": "%s.%s.%s" % (path, key, en_name),
                            "cn_name": cn_val if isinstance(cn_val, str) else None,
                            "node": None,
                        })
                    continue
                if key in COLLECTION_ROLES and isinstance(val, dict):
                    role = COLLECTION_ROLES[key]
                    for en_name, sub in val.items():
                        if not isinstance(sub, dict):
                            continue
                        idx.add(role, en_name, {
                            "repo": os.path.basename(repo), "file": fn, "pack": pack,
                            "path": "%s.%s.%s" % (path, key, en_name),
                            "cn_name": sub.get("name") if isinstance(sub.get("name"), str) else None,
                            "node": sub,
                        })
                        walk(sub, "%s.%s.%s" % (path, key, en_name))
                    continue
                if isinstance(val, dict):
                    walk(val, "%s.%s" % (path, key))

        walk(doc, fn[:-5])
    return idx


# --------------------------------------------------------------------------- 判据

def check_frozen(register, idx, findings, stats):
    """T-FROZEN：英文键在译文里存在时，中文 name 必须与英文逐字节相同。"""
    sec = register["sections"]

    def one(kind, en_string, roles, note, case_insensitive=False, source=""):
        recs = idx.any_role(en_string, roles)
        if not recs:
            stats["FROZEN_absent(未译到，跳过)"] += 1
            return
        for r in recs:
            stats["FROZEN_seen"] += 1
            cn = r["cn_name"]
            if cn is None:
                stats["FROZEN_no_name_field(只有子字段被译，键名没动)"] += 1
                continue
            ok = (cn.upper() == en_string.upper()) if case_insensitive else (cn == en_string)
            if ok and case_insensitive and cn != en_string:
                stats["FROZEN_case_changed(大小写变了但比较是大写不敏感的)"] += 1
            if ok:
                stats["FROZEN_ok"] += 1
                continue
            findings.append({
                "verdict": "FROZEN_TRANSLATED", "kind": kind,
                "repo": r["repo"], "file": r["file"], "path": r["path"],
                "en": en_string, "cn": cn,
                "source": source, "note": note,
            })
            stats["**FROZEN_TRANSLATED**"] += 1

    for e in sec["name_lookups"]["entries"]:
        roles = {"Adventure document name": ["adventure"],
                 "JournalEntry name shown after import": ["journal"],
                 "JournalEntry name used by the release-notes updater": ["journal"],
                 "Scene name activated after import": ["scene"]}.get(e["role"], ["adventure", "journal", "scene"])
        one("name_lookup:" + e["id"], e["string"], roles, e["breaks_if_translated"],
            source=e["declared_at"])

    for e in sec["rolltable_names"]["entries"]:
        one("rolltable_name", e["string"], ["table"], e["breaks_if_translated"], source=e["call_site"])

    for e in sec["folder_names"]["entries"]:
        one("folder_name", e["string"], ["folder"], e["breaks_if_translated"], source=e["call_site"])

    for e in sec["item_names"]["entries"]:
        for d in e["documents"]:
            one("item_name:" + e["string_compared"], d["actual_name"], ["item"],
                e["breaks_if_translated"], case_insensitive=e.get("case_insensitive", False),
                source=e["call_sites"][0])

    # 「Critical Injuries」前缀过滤：任何被 cTables 下拉框吃掉的表，中文名也得以同样
    # 的前缀开头，否则 cTableget() 的 startsWith 过滤把它踢出下拉框。
    for pf in sec["rolltable_names"].get("prefix_filters", []):
        for nm in pf["current_matches"]:
            for r in idx.get("table", nm):
                stats["PREFIX_seen"] += 1
                cn = r["cn_name"]
                if cn is None:
                    continue
                if cn.startswith(pf["prefix"]):
                    stats["PREFIX_ok"] += 1
                else:
                    findings.append({
                        "verdict": "FROZEN_TRANSLATED", "kind": "rolltable_prefix",
                        "repo": r["repo"], "file": r["file"], "path": r["path"],
                        "en": nm, "cn": cn, "source": pf["call_site"],
                        "note": "cTableget() 只收 name.startsWith(%r) 的表；译名丢了这个前缀，"
                                "这张表就从异种生物卡的重伤表下拉框里消失。%s"
                                % (pf["prefix"], pf["breaks_if_translated"]),
                    })
                    stats["**FROZEN_TRANSLATED**"] += 1


def check_macro_commands(register, idx, findings, stats):
    """Macro.command 是可执行 JS，Babele 默认就翻它（default-mappings.js:163
    把 `"command": "command"` 写进了 Macro 的默认 mapping）。

    三个宏的源码里带着 `t.folder.name === 'Alien Mother Tables'`、
    `t.folder.name === 'Alien Creature Tables'` 和
    `game.tables.getName("EV - 48a. LS - DANGER EVENT DETAIL")`。抽取器今天**不**
    输出 `command` —— 这一条是承重的，不是碰巧。所以判据很简单：
    `compendium/cn` 的任何 macro 节点上出现 `command` 键，就是缺陷。
    """
    sec = register["sections"].get("macro_commands")
    if not sec:
        return
    frozen_by_macro = {m["macro"]: m for m in sec.get("macros_carrying_frozen_literals", [])}
    for en_name, recs in sorted(idx.by_role.get("macro", {}).items()):
        for r in recs:
            node = r["node"] or {}
            if "command" not in node:
                stats["MACRO_command_absent(正确)"] += 1
                continue
            stats["MACRO_command_present"] += 1
            cmd = node["command"]
            meta = frozen_by_macro.get(en_name)
            lost = []
            if meta and isinstance(cmd, str):
                lost = [lit for lit in meta["frozen_literals"] if lit not in cmd]
            findings.append({
                "verdict": "MACRO_COMMAND_TRANSLATED", "kind": "macro_command",
                "repo": r["repo"], "file": r["file"], "path": "%s.command" % r["path"],
                "en": en_name, "cn": (cmd[:120] if isinstance(cmd, str) else cmd),
                "source": "modules/babele/script/mapping/default-mappings.js:163",
                "note": "%s%s" % (
                    sec["rule"],
                    ("  这个宏里已经丢了这些冻结字面量：%r。%s" % (lost, sec["breaks_if_translated"]))
                    if lost else "  （目前字面量都还在，但这个键本来就不该出现在译文里。）"),
            })
            stats["**MACRO_COMMAND_TRANSLATED**"] += 1


def check_name_substrings(register, idx, findings, stats):
    """按子串判的物品名：`i.name.includes(" RPG ") || startsWith("RPG") || endsWith("RPG")`
    决定武器每发弹药算 0.5 kg 还是 0.25 kg（character-sheet.mjs / synthetic-sheet.mjs）。
    数据那条腿（`system.attributes.class.value === "RPG"`）在三个包里一个都不匹配，
    所以名字是**唯一**的活路径。译名必须仍然满足三条测试中的至少一条。
    """
    sec = register["sections"].get("name_substring_tests")
    if not sec:
        return
    OPS = {
        "includes": lambda s, lit: lit in s,
        "startsWith": lambda s, lit: s.startswith(lit),
        "endsWith": lambda s, lit: s.endswith(lit),
    }
    for e in sec["entries"]:
        tests = e.get("tests") or []
        if not tests:
            stats["SUBSTR_skipped(登记表里这条没写机器可读的 tests)"] += 1
            continue
        for m in e.get("current_matches", []):
            for r in idx.get("item", m["name"]):
                if r["pack"] is not None and r["pack"] != m["pack"]:
                    continue
                cn = r["cn_name"]
                if cn is None:
                    stats["SUBSTR_no_name_field"] += 1
                    continue
                stats["SUBSTR_seen"] += 1
                if any(OPS[t["op"]](cn, t["literal"]) for t in tests if t["op"] in OPS):
                    stats["SUBSTR_ok"] += 1
                    continue
                findings.append({
                    "verdict": "FROZEN_TRANSLATED", "kind": "name_substring",
                    "repo": r["repo"], "file": r["file"], "path": r["path"],
                    "en": m["name"], "cn": cn,
                    "source": e["call_sites"][0],
                    "note": "译名三条测试一条都不满足（%s）。%s  %s"
                            % (" / ".join("%s %r" % (t["op"], t["literal"]) for t in tests),
                               e["breaks_if_translated"], e["rule"]),
                })
                stats["**FROZEN_TRANSLATED**"] += 1


def check_exact_to_lang(register, idx, resolve, findings, stats):
    """T-EXACT：12 个 skill-stunts 物品名必须逐字节等于 localize(ALIENRPG.Skill<key>)。

    `resolve` 复刻 Foundry 的 `Localization#localize`：cn 值是字符串就用它，否则回退
    en.json，再不行才返回 key 本身。**缺键不算缺陷** —— 缺键就是「还没译」，此时
    localize 返回英文，物品名也还是英文，两边一致。只有「物品译了 / 键没译」或者
    「两边都译了但对不上」才是缺陷。
    """
    sec = register["sections"]["exact_match_to_lang"]
    if resolve is None:
        stats["EXACT_skipped(没有 lang 文件)"] += len(sec["entries"])
        return
    for e in sec["entries"]:
        key = e["lang_key"]
        want, origin = resolve(key)
        if want is None:
            findings.append({
                "verdict": "EXACT_MISMATCH", "kind": "lang_key_unresolvable",
                "repo": "-", "file": "lang/cn.json", "path": key,
                "en": e["item_name_en"], "cn": None,
                "source": e["config_entry"],
                "note": "cn 与 en 两侧都拿不到字符串，localize() 会把键名本身当技能名渲染出来，"
                        "而 skill-stunts 物品名不可能叫 %r。" % key,
            })
            stats["**EXACT_MISMATCH**"] += 1
            continue
        if origin == "cn" and want != want.strip():
            findings.append({
                "verdict": "EXACT_MISMATCH", "kind": "lang_value_padded",
                "repo": "-", "file": "lang/cn.json", "path": key,
                "en": e["item_name_en"], "cn": want,
                "source": e["config_entry"],
                "note": "lang 值首尾带空白。skill.description 会原样带着它去 "
                        "game.items.getName()，物品名不可能带这种尾巴。",
            })
            stats["**EXACT_MISMATCH**"] += 1
            continue
        stats["EXACT_lang_from_" + origin] += 1
        recs = idx.get("item", e["item_name_en"])
        if not recs:
            stats["EXACT_absent(物品还没译，跳过)"] += 1
            continue
        for r in recs:
            stats["EXACT_seen"] += 1
            cn = r["cn_name"]
            if cn is None:
                stats["EXACT_no_name_field"] += 1
                continue
            if cn == want:
                stats["EXACT_ok"] += 1
                continue
            findings.append({
                "verdict": "EXACT_MISMATCH", "kind": "skill_stunts_item",
                "repo": r["repo"], "file": r["file"], "path": r["path"],
                "en": e["item_name_en"], "cn": cn,
                "source": e["config_entry"],
                "note": "必须逐字节等于 localize(%s) = %r（取自 %s.json）。差一个字（哪怕"
                        "只是补了英文尾巴），_stuntBtn 的 game.items.getName() 就返回 "
                        "undefined，catch 吞掉异常后面板显示 <h2>No Stunts Entered</h2>。"
                        % (key, want, origin),
            })
            stats["**EXACT_MISMATCH**"] += 1


def check_table_refs(register, idx, findings, stats):
    """actor 的 rTables/cTables 必须还能在同一份译文集里找到那张表。"""
    sec = register["sections"]["actor_table_refs"]
    tables_present = idx.has_role("table")
    for ref in sec["refs"]:
        # **按包取 actor**。4 个 creature actor（Chestburster / Drone / Facehugger /
        # Ovomorph）在新手包和核心书里各有一份，`_id` 还相同。不按包取的话，同一处
        # 缺陷会照着 ref 的份数重复报，而且一旦两个包给同名 actor 填了不同的表引用，
        # 就会拿甲包的节点去核乙包的 ref —— 那是纯粹的误报。
        actor_recs = [r for r in idx.get("actor", ref["actor"])
                      if r["pack"] is None or r["pack"] == ref["pack"]]
        field_short = ref["field"].split(".")[-1]
        aliases = RTABLE_ALIASES if field_short == "rTables" else CTABLE_ALIASES

        # (a) actor 节点里显式写死了这个字段的情况
        for r in actor_recs:
            node = r["node"] or {}
            for alias in aliases:
                if alias not in node:
                    continue
                stats["REF_explicit_seen"] += 1
                got = node[alias]
                if ref["is_sentinel"]:
                    if got == "None":
                        stats["REF_ok"] += 1
                    else:
                        findings.append({
                            "verdict": "TABLE_REF_DANGLING", "kind": "sentinel_translated",
                            "repo": r["repo"], "file": r["file"],
                            "path": "%s.%s" % (r["path"], alias),
                            "en": "None", "cn": got,
                            "source": register["sections"]["item_names"]["none_sentinel"]["compared_at"][0],
                            "note": register["sections"]["item_names"]["none_sentinel"]["breaks_if_translated"],
                        })
                        stats["**TABLE_REF_DANGLING**"] += 1
                    continue
                table_recs = idx.get("table", ref["value"])
                allowed = {ref["value"]} | {t["cn_name"] for t in table_recs if t["cn_name"]}
                if got in allowed:
                    stats["REF_ok"] += 1
                else:
                    findings.append({
                        "verdict": "TABLE_REF_DANGLING", "kind": "explicit_ref_mismatch",
                        "repo": r["repo"], "file": r["file"],
                        "path": "%s.%s" % (r["path"], alias),
                        "en": ref["value"], "cn": got,
                        "source": sec["consumers"][1]["at"],
                        "note": "字段里写的既不是英文表名也不是那张表的译名（允许集合 %r）。%s"
                                % (sorted(allowed), sec["breaks_if_desynced"]),
                    })
                    stats["**TABLE_REF_DANGLING**"] += 1

        # (b) 引用的表在译文集里到底还在不在。
        #
        # **逐包判**，不是全局判。'EV - Chestburster Attacks' 在新手包和核心书里各有
        # 一份；只把核心书那份的键改名、全局索引里仍然找得到 —— 但只装核心书的用户
        # 世界里就没有这张表了。所以判据是：登记表说这张表**属于**哪几个包
        # （`table_lives_in`），那几个包的译文文件里就都得留着这个英文键。
        if ref["is_sentinel"]:
            stats["REF_sentinel(跳过)"] += 1
            continue
        if not tables_present:
            stats["REF_skipped(译文集里还没有任何 tables，跳过)"] += 1
            continue
        owners = ref.get("table_lives_in") or []
        translated_packs = idx.packs_with_role("table")
        checked_any = False
        for owner in owners:
            if owner not in translated_packs:
                stats["REF_skipped(%s 包还没译 tables)" % owner] += 1
                continue
            checked_any = True
            if idx.in_pack("table", ref["value"], owner):
                stats["REF_resolvable"] += 1
                continue
            findings.append({
                "verdict": "TABLE_REF_DANGLING", "kind": "table_key_missing",
                "repo": "-", "file": "%s (%s)" % (owner, ref["pack"]),
                "path": "%s / %s / %s" % (ref["pack"], ref["actor"], ref["field"]),
                "en": ref["value"], "cn": None,
                "source": sec["consumers"][1]["at"],
                "note": "%s 包的译文里已经有 tables 键了，却没有 %r —— 只装这个包的用户"
                        "世界里就不存在这张表，%s 的 %s 解析不出来。%s"
                        % (owner, ref["value"], ref["actor"], ref["field"],
                           sec["breaks_if_desynced"]),
            })
            stats["**TABLE_REF_DANGLING**"] += 1
        if not checked_any and owners:
            continue
        if not owners:
            stats["REF_no_owner_recorded(登记表里这条没记表在哪个包)"] += 1


# --------------------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", default=DEFAULT_REGISTER)
    ap.add_argument("--repo", action="append")
    ap.add_argument("--lang", default=DEFAULT_LANG)
    ap.add_argument("--lang-en", default=DEFAULT_LANG_EN,
                    help="英文基准，用来复刻 Foundry 的 en 回退（默认 1-系统汉化插件/lang/en.json）")
    ap.add_argument("--out")
    ap.add_argument("--show", type=int, default=40)
    a = ap.parse_args()

    try:
        register = load_json(a.register)
    except Exception as exc:
        print("✗ 读不到登记表 %s：%s" % (a.register, exc))
        return 2
    if register.get("schema_version") != 1:
        print("✗ 登记表 schema_version=%r，本脚本只认 1" % register.get("schema_version"))
        return 2

    repos = a.repo or DEFAULT_REPOS
    lang_cn = flatten_lang(load_json(a.lang)) if (a.lang and os.path.isfile(a.lang)) else None
    lang_en = flatten_lang(load_json(a.lang_en)) if (a.lang_en and os.path.isfile(a.lang_en)) else None
    if lang_cn is None and lang_en is None:
        print("⚠ 既没有 %s 也没有 %s：T-EXACT 那一档跳过（不算缺陷）。" % (a.lang, a.lang_en))
        resolve = None
    else:
        def resolve(key):
            """复刻 foundry.mjs 的 Localization#localize：cn -> en -> key。"""
            v = (lang_cn or {}).get(key)
            if isinstance(v, str):
                return v, "cn"
            v = (lang_en or {}).get(key)
            if isinstance(v, str):
                return v, "en"
            return None, "missing"

    pack_identity = register.get("measured_against", {}).get("pack_identity")
    if not pack_identity:
        print("✗ 登记表里没有 measured_against.pack_identity，逐包判据没法建立")
        return 2
    cn_files = load_cn_files(repos, pack_identity)
    unknown = [(os.path.basename(r), f) for r, f, p, _ in cn_files if p is None]
    if unknown:
        print("⚠ 这些 cn 文件的名字对不上 pack_identity 里任何一个 babele_file，"
              "逐包判据对它们退化成全局判：%s" % unknown)
    idx = build_index(cn_files)

    findings = []
    stats = collections.Counter()
    if idx.parse_errors:
        for repo, fn, err in idx.parse_errors:
            findings.append({"verdict": "FROZEN_TRANSLATED", "kind": "unparseable_cn_file",
                             "repo": os.path.basename(repo), "file": fn, "path": "-",
                             "en": "-", "cn": "-", "source": "-",
                             "note": "JSON 读不动：%s" % err})
            stats["**FROZEN_TRANSLATED**"] += 1

    check_frozen(register, idx, findings, stats)
    check_macro_commands(register, idx, findings, stats)
    check_name_substrings(register, idx, findings, stats)
    check_exact_to_lang(register, idx, resolve, findings, stats)
    check_table_refs(register, idx, findings, stats)

    checked = (stats["FROZEN_seen"] + stats["PREFIX_seen"] + stats["EXACT_seen"]
               + stats["REF_explicit_seen"] + stats["REF_resolvable"]
               + stats["SUBSTR_seen"] + stats["MACRO_command_present"]
               + stats["MACRO_command_absent(正确)"])

    print("登记表  %s（schema %d，生成于 %s）"
          % (os.path.relpath(a.register, PROJ), register["schema_version"], register["generated_on"]))
    print("译文集  %d 个 cn 文件 / %d 个命名节点"
          % (len(cn_files), idx.nodes))
    for role in sorted(idx.by_role):
        print("        %-10s %d" % (role, len(idx.by_role[role])))
    print()
    print("统计：")
    for k, v in sorted(stats.items()):
        print("   %-52s %d" % (k, v))
    print()
    print("checked=%d  violations=%d" % (checked, len(findings)))

    if not findings:
        if checked == 0:
            print("✓ compendium/cn 里还没有任何可核的东西，checked=0，退出 0。")
        else:
            print("✓ 没有缺陷。")
    else:
        by_verdict = collections.Counter(f["verdict"] for f in findings)
        print("按判据：" + "  ".join("%s=%d" % kv for kv in sorted(by_verdict.items())))
        for f in findings[:a.show]:
            print()
            print("  ✗ [%s / %s] %s :: %s" % (f["verdict"], f["kind"], f["file"], f["path"]))
            print("     EN %r" % f["en"])
            print("     CN %r" % f["cn"])
            print("     由 %s 比较" % f["source"])
            print("     %s" % f["note"])
        if len(findings) > a.show:
            print("\n  …… 还有 %d 条，用 --out 全量导出。" % (len(findings) - a.show))

    if a.out:
        json.dump({"register": os.path.abspath(a.register),
                   "repos": [os.path.abspath(r) for r in repos],
                   "checked": checked, "stats": dict(stats), "findings": findings},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("-> %s" % a.out)

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
