#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_extract_leaves.py — 独立复算 `extract/extract_en.mjs` 的抽取结果。

为什么要有这个脚本
------------------
抽取器读的是 **LevelDB 包目录**（`!items!`/`!actors.items!`/`!tables.results!`
这些兄弟桶，靠 `attachEmbedded()` 重新拼成文档树）；本脚本读的是
`6-工作区/raw-dumps/{system,starterset,corerules}.json` —— 每个 pack 一份
**已经嵌套好的 Adventure 文档**。两份序列化格式不同，拼装代码也不同。

因此本脚本用**自己的一套实现**（自己的 `_variants` 归一化、自己的 `_when`
求值、自己的转换器、自己的 key 分配）重放同一份 mapping **数据**，
再和抽取器的产物逐叶比对。两边一致，才说明：

  · LevelDB 遍历 + 嵌套桶重挂 没有丢文档；
  · key 分配（含同名合并）两边一致；
  · 转换器实现两边一致。

mapping 数据本身仍然只有一份真相源（`extract/mappings.mjs`），本脚本通过
`node -e` 把 `effectiveMappings()` 导成 JSON 来读，不做第二份抄写。

用法
----
    export PYTHONIOENCODING=utf-8
    python 4-常用脚本/qa/verify_extract_leaves.py
    python 4-常用脚本/qa/verify_extract_leaves.py --json out.json

退出码：0 = 全部一致；1 = 有差异。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPPINGS_MJS = os.path.join(PROJECT_ROOT, "4-常用脚本", "extract", "mappings.mjs")
RAW_DUMPS = os.path.join(PROJECT_ROOT, "6-工作区", "raw-dumps")

# pack 槽位：raw dump 名 -> (target, 模块目录, 输出文件名, packageId, packageVersion)
SLOTS = [
    ("system", "alienrpg", "1-系统汉化插件",
     "alienrpg.alien-rpg-system.json", "alienrpg", "4.1.13"),
    ("starterset", "starterset", "2-新手包汉化插件",
     "alien-evolved-starterset.alien-evolved-starter-set.json",
     "alien-evolved-starterset", "1.0.2"),
    ("corerules", "corerules", "3-核心书汉化插件",
     "alien-evolved-corerules.alien-evolved-core-rules.json",
     "alien-evolved-corerules", "1.0.2"),
]

# Adventure 文档里嵌套文档所在的字段 -> documentType（本脚本自己的树，
# 与抽取器的 BUCKET_FOR / attachEmbedded 是两套独立代码）
NESTED = {
    "actors": "Actor",
    "items": "Item",
    "journal": "JournalEntry",
    "scenes": "Scene",
    "tables": "RollTable",
    "macros": "Macro",
    "playlists": "Playlist",
    "cards": "Cards",
}


# --------------------------------------------------------------------------- #
# mapping 数据（唯一真相源仍是 mappings.mjs）
# --------------------------------------------------------------------------- #
def load_mapping_data(target: str) -> dict:
    """用 node 把 effectiveMappings(target) / FIELD_CENSUS 导成 JSON。"""
    url = "file:///" + MAPPINGS_MJS.replace("\\", "/").replace(" ", "%20")
    # 目录名是中文，quote 掉非 ASCII
    from urllib.parse import quote
    url = "file:///" + quote(MAPPINGS_MJS.replace("\\", "/"), safe="/:")
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    script = (
        "import(%s).then(m=>{"
        "require('fs').writeFileSync(%s, JSON.stringify({"
        "effective:m.effectiveMappings(%s),census:m.FIELD_CENSUS,"
        "targets:m.ALIEN_TARGETS,converters:Object.keys(m.EXTRACT_CONVERTERS||{})"
        "}));"
        "}).catch(e=>{console.error(e&&e.stack||e);process.exit(1)})"
        % (json.dumps(url), json.dumps(tmp.replace("\\", "/")), json.dumps(target))
    )
    r = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit("node 无法导出 mappings.mjs")
    with open(tmp, encoding="utf-8") as fh:
        data = json.load(fh)
    os.unlink(tmp)
    return data


# --------------------------------------------------------------------------- #
# 独立实现：_variants 归一化 / _when 求值 / activeFields
# --------------------------------------------------------------------------- #
def get_path(obj, dotted):
    cur = obj
    for k in dotted.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, list):
            try:
                cur = cur[int(k)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def path_exists(obj, dotted):
    """`_when.exists` 判的是 `typeof value !== 'undefined'`，不是真值。"""
    cur = obj
    for k in dotted.split("."):
        if isinstance(cur, dict):
            if k not in cur:
                return False
            cur = cur[k]
        elif isinstance(cur, list):
            try:
                cur = cur[int(k)]
            except (ValueError, IndexError):
                return False
        else:
            return False
    return True


def normalize(definition):
    definition = definition or {}
    variants = []
    for v in definition.get("_variants") or []:
        v = dict(v or {})
        when = v.pop("_when", None)
        if v:
            variants.append({"when": when, "mapping": v})
    base = {k: v for k, v in definition.items() if k != "_variants"}
    return {"base": base, "variants": variants}


def when_matches(cond, data):
    if not cond:
        return False
    if isinstance(cond.get("all"), list):
        return all(when_matches(c, data) for c in cond["all"])
    if isinstance(cond.get("any"), list):
        return any(when_matches(c, data) for c in cond["any"])
    p = cond.get("path")
    if not isinstance(p, str) or not p:
        return False
    checks = []
    if "equals" in cond:
        checks.append(get_path(data, p) == cond["equals"])
    if isinstance(cond.get("in"), list):
        checks.append(get_path(data, p) in cond["in"])
    if "exists" in cond:
        checks.append(path_exists(data, p) == bool(cond["exists"]))
    return bool(checks) and all(checks)


def active_fields(norm, doc):
    eff = {}
    def push(obj):
        for k, v in obj.items():
            if k.startswith("_"):
                continue
            eff.pop(k, None)
            eff[k] = v
    push(norm["base"])
    for var in norm["variants"]:
        if when_matches(var["when"], doc):
            push(var["mapping"])
    return eff


# --------------------------------------------------------------------------- #
# 独立实现：key 分配（含同名合并）
# --------------------------------------------------------------------------- #
DEFAULT_EXPORT_TOKENS = ["name", "_id", "id"]


def is_str(v):
    return isinstance(v, str) and v.strip() != ""


def identity_candidate(doc, token):
    if token == "range":
        rng = (doc or {}).get("range") or []
        if len(rng) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in rng):
            return "%d-%d" % (rng[0], rng[1])
        return None
    if token == "sourceId":
        raw = get_path(doc, "flags.core.sourceId") or get_path(doc, "_stats.compendiumSource")
        if not is_str(raw):
            return None
        tail = raw.split(".")[-1]
        return tail if is_str(tail) else None
    v = (doc or {}).get(token)
    return v if is_str(v) else None


def merge_entries(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, str) or isinstance(b, str):
        return a if a == b else False
    if isinstance(a, list) or isinstance(b, list):
        return a if json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) else False
    out = dict(a)
    for k, v in b.items():
        if k not in out:
            out[k] = v
            continue
        m = merge_entries(out[k], v)
        if m is False:
            return False
        out[k] = m
    return out


class KeyAllocator:
    """`extract_en.mjs:makeKeyAllocator` 的独立重写。

    `loss` 记的是**同名合并吃掉的叶子**：union 后两边同键同值的那一份消失了。
    这不是丢内容（值一模一样），但必须记账，否则「产物比原文少 2 叶」
    就只能靠猜。
    """

    def __init__(self, loss=None):
        self.used = {}
        self.loss = loss if loss is not None else [0, 0]

    def __call__(self, doc, identity, entry, fallback_prefix="entry"):
        tokens = (identity or {}).get("export") or DEFAULT_EXPORT_TOKENS
        cands = []
        for t in tokens:
            c = identity_candidate(doc, t)
            if c is not None and c not in cands:
                cands.append(c)
        for c in cands:
            if c not in self.used:
                self.used[c] = entry
                return c, None
            before = self.used[c]
            merged = merge_entries(before, entry)
            if merged is not False:
                bn, bc = count_leaves(before)
                en, ec = count_leaves(entry)
                mn, mc = count_leaves(merged)
                self.loss[0] += bn + en - mn
                self.loss[1] += bc + ec - mc
                self.used[c] = merged
                return c, merged
        i = len(self.used)
        fb = "%s-%d" % (fallback_prefix, i)
        while fb in self.used:
            i += 1
            fb = "%s-%d" % (fallback_prefix, i)
        self.used[fb] = entry
        return fb, None


# --------------------------------------------------------------------------- #
# 独立实现：文档抽取
# --------------------------------------------------------------------------- #
def to_array(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("contents"), list):
            return value["contents"]
        return list(value.values())
    return []


class Extractor:
    def __init__(self, normalized, loss=None):
        self.norm = normalized
        self.unknown = Counter()
        self.loss = loss if loss is not None else [0, 0]

    def doc(self, d, dtype):
        norm = self.norm.get(dtype)
        if not norm or not isinstance(d, dict):
            return None
        out = {}
        for field, spec in active_fields(norm, d).items():
            if isinstance(spec, str):
                v = get_path(d, spec)
                if is_str(v):
                    out[field] = v
                continue
            if not isinstance(spec, dict):
                continue
            value = get_path(d, spec.get("path", field))
            conv = spec.get("converter")

            # 自定义 converter 的 extract 半边（EXTRACT_CONVERTERS）
            if conv == "alienRollTableRef":
                if is_str(value) and value != "None":
                    out[field] = value
                continue

            if conv in ("name", "crucibleTokenName", "alienTokenName"):
                if is_str(value):
                    out[field] = value
            elif conv == "nameCollection":
                m = {}
                for it in to_array(value):
                    n = (it or {}).get("name") if isinstance(it, dict) else None
                    if is_str(n):
                        m.setdefault(n, n)
                if m:
                    out[field] = m
            elif conv == "textCollection":
                m = {}
                for it in to_array(value):
                    t = (it or {}).get("text") if isinstance(it, dict) else None
                    if is_str(t):
                        m.setdefault(t, t)
                if m:
                    out[field] = m
            elif conv == "structured":
                m = {}
                for it in to_array(value):
                    if not isinstance(it, dict):
                        continue
                    k = it.get(spec.get("key", "id"))
                    if not is_str(k):
                        continue
                    sub = {}
                    for sf, sp in (spec.get("mapping") or {}).items():
                        sv = get_path(it, sp)
                        if is_str(sv):
                            sub[sf] = sv
                    if sub:
                        m.setdefault(k, sub)
                if m:
                    out[field] = m
            elif conv == "document":
                child_type = spec.get("documentType")
                identity = (self.norm.get(child_type) or {}).get("base", {}).get("_identity")
                alloc = KeyAllocator(self.loss)
                m = {}
                for child in to_array(value):
                    e = self.doc(child, child_type)
                    if not e:
                        continue
                    key, merged = alloc(child, identity, e, "embedded")
                    m[key] = merged if merged is not None else e
                if m:
                    out[field] = m
            else:
                self.unknown["%s.%s:%s" % (dtype, field, conv or "(no converter)")] += 1
                if is_str(value):
                    out[field] = value
        return out or None


# --------------------------------------------------------------------------- #
# 叶子计数 / 语料普查
# --------------------------------------------------------------------------- #
def count_leaves(obj):
    """字符串叶子数与总字符数。"""
    if isinstance(obj, str):
        return 1, len(obj)
    if isinstance(obj, dict):
        n = c = 0
        for v in obj.values():
            a, b = count_leaves(v)
            n += a
            c += b
        return n, c
    if isinstance(obj, list):
        n = c = 0
        for v in obj:
            a, b = count_leaves(v)
            n += a
            c += b
        return n, c
    return 0, 0


def leaf_paths(obj, prefix=""):
    """dotted-path -> 字符串，用于逐叶 diff。"""
    if isinstance(obj, str):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaf_paths(v, "%s/%s" % (prefix, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaf_paths(v, "%s/[%d]" % (prefix, i))


def census_walk(adventure, counts):
    """独立的文档普查：按 documentType（Item/Actor 再按 subtype）计数。"""
    counts["Adventure"] += 1
    for f in adventure.get("folders") or []:
        counts["Folder"] += 1
    for field, dtype in NESTED.items():
        for d in adventure.get(field) or []:
            counts[dtype] += 1
            if dtype in ("Item", "Actor"):
                counts["%s.%s" % (dtype, d.get("type"))] += 1
            if dtype == "Actor":
                for it in d.get("items") or []:
                    counts["Item"] += 1
                    counts["Item.%s" % it.get("type")] += 1
                    for _ in it.get("effects") or []:
                        counts["ActiveEffect"] += 1
                for _ in d.get("effects") or []:
                    counts["ActiveEffect"] += 1
            if dtype == "Item":
                for _ in d.get("effects") or []:
                    counts["ActiveEffect"] += 1
            if dtype == "JournalEntry":
                for _ in d.get("pages") or []:
                    counts["JournalEntryPage"] += 1
            if dtype == "RollTable":
                for _ in d.get("results") or []:
                    counts["TableResult"] += 1
            if dtype == "Scene":
                for r in d.get("regions") or []:
                    counts["Region"] += 1
                    for _ in r.get("behaviors") or []:
                        counts["RegionBehavior"] += 1
            if dtype == "Playlist":
                for _ in d.get("sounds") or []:
                    counts["PlaylistSound"] += 1
            if dtype == "Cards":
                for _ in d.get("cards") or []:
                    counts["Card"] += 1


# --------------------------------------------------------------------------- #
# FIELD_CENSUS 复算 + 掉叶账
# --------------------------------------------------------------------------- #
class CensusWalker:
    """按 mapping 的 **源路径** 重新计数，用于和 FIELD_CENSUS 对账。

    与 `Extractor` 的区别：这里不做去重、不做 key 分配、不丢哨兵，统计的是
    「原始文档里这条路径上有多少个非空字符串」。抽取器的产物比它少多少，
    必须能被下面三条**明账**解释干净，否则就是丢内容。
    """

    def __init__(self, normalized):
        self.norm = normalized
        self.n = Counter()
        self.chars = Counter()
        self.sentinel_leaves = 0
        self.sentinel_chars = 0
        self.dedupe_leaves = Counter()
        self.dedupe_chars = Counter()

    def walk(self, d, dtype):
        norm = self.norm.get(dtype)
        if not norm or not isinstance(d, dict):
            return
        for field, spec in active_fields(norm, d).items():
            if isinstance(spec, str):
                v = get_path(d, spec)
                if is_str(v):
                    self.n[(dtype, spec)] += 1
                    self.chars[(dtype, spec)] += len(v)
                continue
            if not isinstance(spec, dict):
                continue
            p = spec.get("path", field)
            value = get_path(d, p)
            conv = spec.get("converter")
            if conv == "document":
                for c in to_array(value):
                    self.walk(c, spec.get("documentType"))
            elif conv in ("nameCollection", "textCollection"):
                key = "name" if conv == "nameCollection" else "text"
                seen = set()
                for it in to_array(value):
                    s = (it or {}).get(key) if isinstance(it, dict) else None
                    if not is_str(s):
                        continue
                    self.n[(dtype, "%s[].%s" % (p, key))] += 1
                    self.chars[(dtype, "%s[].%s" % (p, key))] += len(s)
                    if s in seen:
                        self.dedupe_leaves[(dtype, field)] += 1
                        self.dedupe_chars[(dtype, field)] += len(s)
                    seen.add(s)
            else:
                if is_str(value):
                    self.n[(dtype, p)] += 1
                    self.chars[(dtype, p)] += len(value)
                    if conv == "alienRollTableRef" and value == "None":
                        self.sentinel_leaves += 1
                        self.sentinel_chars += len(value)


def reconcile(census_rows, walker, emitted_leaves, emitted_chars, merge_leaves, merge_chars):
    """把 FIELD_CENSUS、原始路径计数、抽取器产物三方对账。"""
    # FIELD_CENSUS 用 (doc, path, types) 三元组；同一 (doc,path) 可能有多行
    # （例如 Actor/system.notes 有 NOTES_RENDERED_TYPES 与 territory 两行），
    # 所以按 (doc,path) 汇总后再比。
    declared = defaultdict(lambda: [0, 0])
    for r in census_rows:
        k = (r["doc"], r["path"])
        declared[k][0] += r.get("n", 0)
        declared[k][1] += r.get("chars", 0)

    walked = {k: (walker.n[k], walker.chars[k]) for k in walker.n}
    unlisted = sorted(k for k in walked if k not in declared)
    drifted = []
    for k, (dn, dc) in sorted(declared.items()):
        wn, wc = walked.get(k, (0, 0))
        if (wn, wc) != (dn, dc):
            drifted.append((k, dn, dc, wn, wc))

    raw_leaves = sum(walker.n.values())
    raw_chars = sum(walker.chars.values())
    dedupe_l = sum(walker.dedupe_leaves.values())
    dedupe_c = sum(walker.dedupe_chars.values())
    accounted = raw_leaves - walker.sentinel_leaves - dedupe_l - merge_leaves
    accounted_c = raw_chars - walker.sentinel_chars - dedupe_c - merge_chars
    return {
        "censusDeclared": {"leaves": sum(v[0] for v in declared.values()),
                           "chars": sum(v[1] for v in declared.values())},
        "rawPathWalk": {"leaves": raw_leaves, "chars": raw_chars},
        "mappedButNotInCensus": [{"doc": k[0], "path": k[1],
                                  "n": walked[k][0], "chars": walked[k][1]} for k in unlisted],
        "censusDrift": [{"doc": k[0], "path": k[1], "censusN": dn, "censusChars": dc,
                         "walkN": wn, "walkChars": wc} for k, dn, dc, wn, wc in drifted],
        "drops": {
            "sentinelNone": {"leaves": walker.sentinel_leaves, "chars": walker.sentinel_chars},
            "collectionDedupe": {"leaves": dedupe_l, "chars": dedupe_c,
                                 "byField": {"%s.%s" % k: v
                                             for k, v in walker.dedupe_leaves.items()}},
            "keyCollapse": {"leaves": merge_leaves, "chars": merge_chars},
        },
        "predicted": {"leaves": accounted, "chars": accounted_c},
        "emitted": {"leaves": emitted_leaves, "chars": emitted_chars},
        "balances": accounted == emitted_leaves and accounted_c == emitted_chars,
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="把完整结果写到这个文件")
    ap.add_argument("--show", type=int, default=8, help="每个 pack 最多打印几条差异")
    args = ap.parse_args()

    report = {"packs": [], "census": {}, "ok": True}
    total_counts = Counter()
    census_walker = None
    census_rows = []
    merge_loss = [0, 0]
    print("=" * 78)
    print("独立复算 extract_en.mjs —— raw-dumps(嵌套 Adventure) vs compendium/en(LevelDB 抽取)")
    print("=" * 78)

    for dump_name, target, slot, out_name, pkg_id, pkg_ver in SLOTS:
        dump_path = os.path.join(RAW_DUMPS, "%s.json" % dump_name)
        out_path = os.path.join(PROJECT_ROOT, slot, "compendium", "en", out_name)
        src_path = os.path.join(PROJECT_ROOT, slot, "compendium", "en", "_source.json")

        missing = [p for p in (dump_path, out_path) if not os.path.exists(p)]
        if missing:
            print("\n!! %s: 缺文件 %s" % (slot, missing))
            report["ok"] = False
            report["packs"].append({"slot": slot, "error": "missing", "missing": missing})
            continue

        with open(dump_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        adventures = [v for k, v in raw.items() if k.startswith("!adventures!")]

        md = load_mapping_data(target)
        normalized = {dt: normalize(df) for dt, df in md["effective"].items()}

        loss = [0, 0]
        ex = Extractor(normalized, loss)
        # FIELD_CENSUS 的数字是**三个包合计**，所以对账用的 walker 跨包累计，
        # 最后统一结算；每包单独比会假报漂移。
        cw = census_walker or CensusWalker(normalized)
        census_walker = cw
        census_rows = md["census"]
        identity = (normalized.get("Adventure") or {}).get("base", {}).get("_identity")
        alloc = KeyAllocator(loss)
        entries = {}
        for adv in adventures:
            census_walk(adv, total_counts)
            cw.walk(adv, "Adventure")
            e = ex.doc(adv, "Adventure")
            if not e:
                continue
            key, merged = alloc(adv, identity, e, "entry")
            entries[key] = merged if merged is not None else e

        mine = {"entries": entries}
        with open(out_path, encoding="utf-8") as fh:
            theirs_full = json.load(fh)
        theirs = {"entries": theirs_full.get("entries", {})}

        n_mine, c_mine = count_leaves(mine)
        n_theirs, c_theirs = count_leaves(theirs)

        lp_mine = dict(leaf_paths(mine))
        lp_theirs = dict(leaf_paths(theirs))
        only_mine = sorted(set(lp_mine) - set(lp_theirs))
        only_theirs = sorted(set(lp_theirs) - set(lp_mine))
        differing = sorted(k for k in set(lp_mine) & set(lp_theirs) if lp_mine[k] != lp_theirs[k])

        ok = not (only_mine or only_theirs or differing)
        report["ok"] = report["ok"] and ok

        src = {}
        if os.path.exists(src_path):
            with open(src_path, encoding="utf-8") as fh:
                src = json.load(fh)

        print("\n--- %s  (%s v%s, target=%s)" % (slot, pkg_id, pkg_ver, target))
        print("    raw dump         : %s  (%d Adventure 文档)"
              % (os.path.basename(dump_path), len(adventures)))
        print("    独立走查叶子      : %7d leaves / %9d chars" % (n_mine, c_mine))
        print("    抽取器产物叶子    : %7d leaves / %9d chars" % (n_theirs, c_theirs))
        print("    差异              : only-raw %d · only-extract %d · value-mismatch %d"
              % (len(only_mine), len(only_theirs), len(differing)))
        print("    verdict           : %s" % ("MATCH" if ok else "MISMATCH"))
        if src:
            ver_ok = src.get("packageVersion") == pkg_ver and src.get("packageId") == pkg_id
            print("    _source.json      : %s v%s  mapping=%s  extractedAt=%s  %s"
                  % (src.get("packageId"), src.get("packageVersion"),
                     src.get("mappingSource"), src.get("extractedAt"),
                     "OK" if ver_ok else "!! 版本/ID 与预期不符"))
            report["ok"] = report["ok"] and ver_ok
        for label, lst in (("only-in-raw-walk", only_mine),
                           ("only-in-extractor", only_theirs),
                           ("value-mismatch", differing)):
            for k in lst[: args.show]:
                print("      [%s] %s" % (label, k[:150]))
            if len(lst) > args.show:
                print("      ... 另有 %d 条" % (len(lst) - args.show))

        merge_loss[0] += loss[0]
        merge_loss[1] += loss[1]

        report["packs"].append({
            "slot": slot, "packageId": pkg_id, "packageVersion": pkg_ver, "target": target,
            "keyCollapseLoss": {"leaves": loss[0], "chars": loss[1]},
            "adventureDocs": len(adventures),
            "rawWalk": {"leaves": n_mine, "chars": c_mine},
            "extractor": {"leaves": n_theirs, "chars": c_theirs},
            "onlyInRawWalk": only_mine, "onlyInExtractor": only_theirs,
            "valueMismatch": differing,
            "unknownConvertersRawWalk": dict(ex.unknown),
            "unknownConvertersExtractor": src.get("unknownConverters", {}),
            "match": ok,
        })

    print("\n" + "-" * 78)
    print("语料普查（独立走查三份 raw dump）")
    for k in sorted(total_counts):
        print("    %-28s %6d" % (k, total_counts[k]))
    report["census"] = dict(total_counts)

    tot_mine = sum(p.get("rawWalk", {}).get("leaves", 0) for p in report["packs"])
    tot_theirs = sum(p.get("extractor", {}).get("leaves", 0) for p in report["packs"])
    tot_c_mine = sum(p.get("rawWalk", {}).get("chars", 0) for p in report["packs"])
    tot_c_theirs = sum(p.get("extractor", {}).get("chars", 0) for p in report["packs"])

    if census_walker is not None:
        rec = reconcile(census_rows, census_walker, tot_theirs, tot_c_theirs,
                        merge_loss[0], merge_loss[1])
        report["reconciliation"] = rec
        print("\n" + "-" * 78)
        print("掉叶账（三包合计；FIELD_CENSUS 的数字本来就是三包合计）")
        print("    FIELD_CENSUS 声明        %6d leaves / %9d chars"
              % (rec["censusDeclared"]["leaves"], rec["censusDeclared"]["chars"]))
        print("    原始路径实测             %6d leaves / %9d chars"
              % (rec["rawPathWalk"]["leaves"], rec["rawPathWalk"]["chars"]))
        for u in rec["mappedButNotInCensus"]:
            print("      [FIELD_CENSUS 缺行] %s / %s  -> n=%d chars=%d"
                  % (u["doc"], u["path"], u["n"], u["chars"]))
        for dft in rec["censusDrift"]:
            print("      [FIELD_CENSUS 漂移] %s / %s  census n=%d chars=%d | walk n=%d chars=%d"
                  % (dft["doc"], dft["path"], dft["censusN"], dft["censusChars"],
                     dft["walkN"], dft["walkChars"]))
        d = rec["drops"]
        print("    - None 哨兵丢弃          %6d leaves / %9d chars   (alienRollTableRef)"
              % (d["sentinelNone"]["leaves"], d["sentinelNone"]["chars"]))
        print("    - 集合去重               %6d leaves / %9d chars   %s"
              % (d["collectionDedupe"]["leaves"], d["collectionDedupe"]["chars"],
                 d["collectionDedupe"]["byField"]))
        print("    - 同名合并 union         %6d leaves / %9d chars"
              % (d["keyCollapse"]["leaves"], d["keyCollapse"]["chars"]))
        print("    = 预测产物               %6d leaves / %9d chars"
              % (rec["predicted"]["leaves"], rec["predicted"]["chars"]))
        print("      实际产物               %6d leaves / %9d chars   -> %s"
              % (rec["emitted"]["leaves"], rec["emitted"]["chars"],
                 "账平" if rec["balances"] else "!! 账不平"))
        if not rec["balances"]:
            report["ok"] = False
    print("\n" + "=" * 78)
    print("TOTAL  raw-walk %d leaves / %d chars   vs   extractor %d leaves / %d chars"
          % (tot_mine, tot_c_mine, tot_theirs, tot_c_theirs))
    print("RESULT %s" % ("ALL MATCH" if report["ok"] else "MISMATCH — 见上"))
    print("=" * 78)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print("wrote %s" % args.json)

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
