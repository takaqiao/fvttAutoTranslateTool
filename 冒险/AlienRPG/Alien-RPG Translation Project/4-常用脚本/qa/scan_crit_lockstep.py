# -*- coding: utf-8 -*-
"""重伤表锁步闸：8 个 lang 键的中文值（连同尾随空格）必须与译好的重伤表逐字节对上。

背景
----
`actor.mjs` 拿到抽出来的重伤结果之后干的事（登记表 `crit_parse_lockstep.pipeline` 里
有逐行出处）：

    cleanText  = messG.replace(/(<b>)|(<p>|)(<strong>)|(<\\/b>)|(<\\/p>)|(<\\/strong>)/gi, "")
    factorFour = cleanText.replace(/<br \\/>/gi, "<br>")
    testArray  = factorFour.split(/[:] |<br>/gi)

然后**按固定下标**读格子：`[3]` = FATAL、`[5]` = TIME LIMIT、`[7]` = EFFECTS、
`[9]` = HEALING TIME，并且拿 `game.i18n.localize(...) + 字面后缀` 去 `===`。

所以译文这边有三件事必须同时成立，少一件就静默出错：

1. **结构**：五个标签后面的「冒号 + 半角空格」和行间的 `<br />` 都得在，否则数组长度
   不是 10，下标全体错位 —— FATAL 会从 TIME LIMIT 那格读出来，重伤被错误应用且不报错。
   注意中文全角冒号 `\\uff1a` 不匹配 `/[:] /`。
2. **字面后缀**：`Yes` 那三个 case 的后缀是 `" "` / `", \\u20131 "` / `", \\u20132 "`，
   中间那个横杠是 **EN DASH U+2013，不是 ASCII 减号**。五个 `One*` case 的后缀都是
   一个半角空格。`Permanent` 那格在字符串末尾，**没有**尾随空格。
3. **`Shift` 是裸字面量，没有 i18n 键**（`actor.mjs` 的 `testArray[9] === "Shift"`）。
   `EV - Critical Injuries` 有两行的 HEALING TIME 格就是这个词。把它翻了，控制流走到
   `testArray[9].match(/^\\[\\[([0-9]d[0-9]+)]/)[1]`，`.match()` 返回 null、`null[1]`
   抛**未捕获** TypeError —— 整次进化版重伤掷骰当场死掉，聊天卡都发不出来。
   `ALIENRPG.Shift` 是**输出**键：那个键填中文，表格里的格子留英文。

判据（这几档退出非零）
----------------------
* `LANG_BAD`      —— 8 个键里有的不是字符串、或首尾带空白
* `SPLIT_SHAPE`   —— 译好的重伤行切出来段数不对（登记表 `rows_with_nonstandard_split` /
                     `measured_split_length_by_range` 里逐行记了上游本来就不标准的例外），
                     或者某条分支要读的下标落到数组外面
* `CELL_MISMATCH` —— `[3]`/`[5]`/`[9]` 的中文格子 ≠ 「lang 值 + 登记的后缀」
* `SHIFT_TRANSLATED` —— `[9]` 那两行的 `Shift` 被翻了
* `ROLL_SHAPE`    —— `[9]` 里 `[[NdM]]` 的形状被破坏（前面多了字符 / 括号被全角化）

4. **另外三条分支**（2026-08-29 补）。`actor.mjs` 里除了 `case "character"` 之外还有
   `case "synthetic": case "creature":`（actor.mjs:2101，读 `[0]`/`[1]`）和
   `case "spacecraft":`（actor.mjs:2134，读 `[0]`/`[2]`/`[5]`）两处同样的定下标解析，
   splitter 是 `/[:] |<br \\/>/gi` —— **没有** `<br />` -> `<br>` 归一化那一步，所以
   `<br>` 在这三条分支上根本不切。覆盖 `Critical Injuries on Synthetics` /
   `Critical Injuries on Xenomorphs` / `Spaceship Minor Component Damage` /
   `Spaceship Major Component Damage` 四张表。飞船那两张的行形状是
   `<strong>名: </strong><br /> 正文 <br /><br /><strong>Repair Roll: </strong> 值`，
   **那个双 `<br /><br />` 是承重的**：并成一个，下标 5 就掉到数组外面，
   `system.header.repairroll` 变 undefined，物品照样建出来，一声不吭。

`compendium/cn` 里还没有重伤表时只跑 lang 那一档；两边都没有就 `checked=0` 退出 0。

用法
----
    python scan_crit_lockstep.py [--register <DO-NOT-TRANSLATE.json>]
        [--repo <汉化插件目录> ...] [--lang <lang/cn.json>] [--out <json>]

退出码：0 干净 / 1 有缺陷 / 2 用不了
"""
import argparse
import collections
import json
import os
import re
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

# actor.mjs 的三步，逐字节照抄（JS 的 /gi 对应 Python 的 re.I，全局由 sub/split 负责）
STRIP_RX = re.compile(r"(<b>)|(<p>|)(<strong>)|(</b>)|(</p>)|(</strong>)", re.I)
BR_RX = re.compile(r"<br />", re.I)
SPLIT_RX = re.compile(r"[:] |<br>", re.I)
ROLL_RX = re.compile(r"^\[\[([0-9]d[0-9]+)]")

# 另外三条分支（synthetic / creature / spacecraft）用的是**另一个** splitter：
# actor.mjs:2101 和 actor.mjs:2134 都是 `factorFour.split(/[:] |<br \/>/gi)`，
# 而且**没有** `<br />` -> `<br>` 那一步。所以这三条分支上 `<br>` 不切、只有字面的
# `<br />` 才切 —— 把 `<br />` 「整理」成 `<br>` 会静默地把格子并起来。
SPLIT_OTHER_RX = re.compile(r"[:] |<br />", re.I)


def split_like_actor(desc):
    return SPLIT_RX.split(BR_RX.sub("<br>", STRIP_RX.sub("", desc)))


def split_like_other_branch(desc):
    return SPLIT_OTHER_RX.split(STRIP_RX.sub("", desc))


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


def find_translated_tables(repos, wanted_names):
    """在 compendium/cn 的任意嵌套深度上找 `tables` 块里这几张表。

    返回 [(repo, file, path, en_table_name, node)]。
    """
    hits = []
    for repo in repos:
        cn_dir = os.path.join(repo, "compendium", "cn")
        if not os.path.isdir(cn_dir):
            continue
        for fn in sorted(os.listdir(cn_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            try:
                doc = load_json(os.path.join(cn_dir, fn))
            except Exception as exc:
                hits.append((os.path.basename(repo), fn, "-", None, {"__parse_error__": str(exc)}))
                continue

            def walk(node, path):
                if not isinstance(node, dict):
                    return
                for k, v in node.items():
                    if k == "tables" and isinstance(v, dict):
                        for en_name, sub in v.items():
                            if en_name in wanted_names and isinstance(sub, dict):
                                hits.append((os.path.basename(repo), fn,
                                             "%s.tables.%s" % (path, en_name), en_name, sub))
                    if isinstance(v, dict):
                        walk(v, "%s.%s" % (path, k))

            walk(doc, fn[:-5])
    return hits


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
    crit = register.get("sections", {}).get("crit_parse_lockstep")
    if not crit:
        print("✗ 登记表里没有 sections.crit_parse_lockstep")
        return 2

    repos = a.repo or DEFAULT_REPOS
    findings = []
    stats = collections.Counter()

    # ------------------------------------------------------------ 1. lang 卫生
    #
    # `resolve` 复刻 foundry.mjs 的 Localization#localize：cn 值是字符串就用它，否则
    # 回退 en.json，再不行返回 None（渲染出来就是键名本身）。**缺键不是缺陷** ——
    # 缺键就是「这一档还没译」，此时 localize 返回英文、表格也还是英文，两边一致。
    lang_cn = flatten_lang(load_json(a.lang)) if (a.lang and os.path.isfile(a.lang)) else None
    lang_en = flatten_lang(load_json(a.lang_en)) if (a.lang_en and os.path.isfile(a.lang_en)) else None
    if lang_cn is None and lang_en is None:
        print("⚠ 既没有 %s 也没有 %s：lang 与格子比对那两档跳过。" % (a.lang, a.lang_en))

    def resolve(key):
        v = (lang_cn or {}).get(key)
        if isinstance(v, str):
            return v, "cn"
        v = (lang_en or {}).get(key)
        if isinstance(v, str):
            return v, "en"
        return None, "missing"

    lang = None if (lang_cn is None and lang_en is None) else resolve

    # 受控的是 8 个键，不是 7 个。7 个来自 `cases[*].lang_key`（那几个键的值参与 `===`
    # 比较），第 8 个是 `ALIENRPG.Shift` —— 它挂在裸字面量那条 case 的 `output_key` 上，
    # 只做**输出**（actor.mjs:1891 把 'Shift' 换成它），不参与比较。但它照样要过卫生检查：
    # 首尾带空白就会把「Shift」渲染成带空格的伤情时长，写成非字符串则静默回退英文，
    # 表面上「已经译了」实际上一个字都没变。
    keys = sorted({c["lang_key"] for c in crit["cases"] if c.get("lang_key")}
                  | {c["output_key"]["key"] for c in crit["cases"]
                     if isinstance(c.get("output_key"), dict) and c["output_key"].get("key")})
    if lang is not None:
        for k in keys:
            raw = (lang_cn or {}).get(k, "__ABSENT__")
            v, origin = resolve(k)
            stats["LANG_seen"] += 1
            if raw == "__ABSENT__":
                stats["LANG_absent_in_cn(回退英文，跳过)"] += 1
                continue
            if not isinstance(raw, str) or raw == "":
                findings.append({
                    "verdict": "LANG_BAD", "where": "lang/cn.json", "key": k,
                    "got": repr(raw), "want": "非空字符串（或者干脆不要这个键）",
                    "note": "键写了却不是字符串。Localization#localize 只认字符串，会静默回退到 "
                            "en.json —— 于是 case 比较的是英文而表格里可能是中文，永远匹配不上，"
                            "而且从 JSON 上看这个键「明明已经译了」。要么给字符串，要么删键。",
                })
                stats["**LANG_BAD**"] += 1
            elif raw != raw.strip():
                findings.append({
                    "verdict": "LANG_BAD", "where": "lang/cn.json", "key": k,
                    "got": repr(raw), "want": repr(raw.strip()),
                    "note": "首尾带空白。代码是 localize(key) + 字面后缀，多出来的空白"
                            "会把 case 变成双空格，表格里不可能出现。",
                })
                stats["**LANG_BAD**"] += 1
            else:
                stats["LANG_ok"] += 1

    # ------------------------------------------------------------ 2. 表格锁步
    tables_meta = crit["tables"]
    hits = find_translated_tables(repos, set(tables_meta))
    for repo, fn, path, en_name, node in hits:
        if "__parse_error__" in node:
            findings.append({"verdict": "SPLIT_SHAPE", "where": "%s/%s" % (repo, fn),
                             "key": "-", "got": node["__parse_error__"], "want": "可解析的 JSON",
                             "note": "重伤表所在的译文文件读不动。"})
            stats["**SPLIT_SHAPE**"] += 1
            continue

        meta = tables_meta[en_name]
        sig = meta["rows_with_significant_tokens"]
        # 上游本来就切不出 10 段的行：登记表 `rows_with_nonstandard_split` 里逐行记了
        # 每个包实测的段数。**不能**只容忍 want_n-1 —— 'EV - Critical Injuries' 的
        # 11-11 行上游是 11 段（HEALING TIME 后面多一个 <br />），比标准**多**一段。
        nonstd = meta.get("rows_with_nonstandard_split") or {}
        allowed_len = {}
        for _rng, _info in nonstd.items():
            allowed_len[_rng] = {v for k, v in _info.items() if isinstance(v, int)}
        # 向后兼容旧字段：只记了「哪些行在包之间不一样」而没记段数时，容忍少一段。
        for _rng in (meta.get("rows_that_differ_between_packs") or {}):
            allowed_len.setdefault(_rng, set()).add(meta["expected_split_length"] - 1)
        results = node.get("results")
        if not isinstance(results, dict):
            stats["TABLE_no_results(表被译了但没译 results，跳过)"] += 1
            continue
        stats["TABLE_seen"] += 1

        for rng, row in results.items():
            if not isinstance(row, dict):
                continue
            desc = row.get("description")
            if not isinstance(desc, str) or desc == "":
                stats["ROW_no_description(跳过)"] += 1
                continue
            stats["ROW_seen"] += 1
            arr = split_like_actor(desc)
            want_n = meta["expected_split_length"]
            if len(arr) != want_n:
                if len(arr) in allowed_len.get(rng, ()):
                    stats["ROW_nonstandard_but_upstream(容忍 %d 段)" % len(arr)] += 1
                else:
                    findings.append({
                        "verdict": "SPLIT_SHAPE", "where": "%s/%s :: %s [%s]" % (repo, fn, path, rng),
                        "key": "split length", "got": len(arr),
                        "want": "%d%s" % (want_n,
                                          ("（或上游实测的 %s）" % sorted(allowed_len[rng]))
                                          if rng in allowed_len else ""),
                        "note": "%s  实际切片：%r" % (crit["structural_invariant"], arr),
                    })
                    stats["**SPLIT_SHAPE**"] += 1
                    continue

            expect = sig.get(rng)
            if expect is None:
                stats["ROW_no_frozen_token(这一行英文侧没有受控 token)"] += 1
                # 仍然要查 [[NdM]] 形状
                cell9 = arr[9] if len(arr) > 9 else None
                if isinstance(cell9, str) and "[[" in cell9 and not ROLL_RX.match(cell9):
                    findings.append({
                        "verdict": "ROLL_SHAPE", "where": "%s/%s :: %s [%s]" % (repo, fn, path, rng),
                        "key": "testArray[9]", "got": cell9,
                        "want": "以 [[NdM]] 开头（%s）" % crit["healing_time_roll_shape"]["regex_1"],
                        "note": crit["healing_time_roll_shape"]["rule"],
                    })
                    stats["**ROLL_SHAPE**"] += 1
                continue

            # -- [3] FATAL
            f_en = expect.get("fatal_en")
            case3 = next((c for c in crit["cases"] if c["index"] == 3 and c["en_cell"] == f_en), None)
            if case3 and lang is not None:
                base, _origin = resolve(case3["lang_key"])
                if isinstance(base, str):
                    want = base + case3["suffix_literal"]
                    got = arr[3]
                    stats["CELL3_seen"] += 1
                    if got == want:
                        stats["CELL3_ok"] += 1
                    else:
                        findings.append({
                            "verdict": "CELL_MISMATCH",
                            "where": "%s/%s :: %s [%s] testArray[3]" % (repo, fn, path, rng),
                            "key": case3["lang_key"], "got": got, "want": want,
                            "note": "%s  ← %s%s" % (case3["effect"], case3["at"],
                                                    "  " + case3["WARNING"] if case3.get("WARNING") else ""),
                        })
                        stats["**CELL_MISMATCH**"] += 1

            # -- [5] TIME LIMIT
            t_en = expect.get("time_limit_en")
            case5 = next((c for c in crit["cases"] if c["index"] == 5 and c["en_cell"] == t_en), None)
            if case5 and lang is not None:
                base, _origin = resolve(case5["lang_key"])
                if isinstance(base, str):
                    want = base + case5["suffix_literal"]
                    got = arr[5]
                    stats["CELL5_seen"] += 1
                    if got == want:
                        stats["CELL5_ok"] += 1
                    else:
                        findings.append({
                            "verdict": "CELL_MISMATCH",
                            "where": "%s/%s :: %s [%s] testArray[5]" % (repo, fn, path, rng),
                            "key": case5["lang_key"], "got": got, "want": want,
                            "note": "%s  ← %s" % (case5["effect"], case5["at"]),
                        })
                        stats["**CELL_MISMATCH**"] += 1

            # -- [9] HEALING TIME
            h_en = expect.get("healing_en")
            got9 = arr[9] if len(arr) > 9 else None
            if h_en == "Shift":
                stats["CELL9_shift_seen"] += 1
                lit = next(c for c in crit["cases"] if c.get("literal") == "Shift")
                if got9 == "Shift":
                    stats["CELL9_shift_ok"] += 1
                else:
                    findings.append({
                        "verdict": "SHIFT_TRANSLATED",
                        "where": "%s/%s :: %s [%s] testArray[9]" % (repo, fn, path, rng),
                        "key": "(裸字面量，无 i18n 键)", "got": got9, "want": "Shift",
                        "note": lit["WARNING"],
                    })
                    stats["**SHIFT_TRANSLATED**"] += 1
            elif h_en == "Permanent" and lang is not None:
                case9 = next(c for c in crit["cases"] if c["index"] == 9 and c.get("lang_key") == "ALIENRPG.Permanent")
                base, _origin = resolve("ALIENRPG.Permanent")
                if isinstance(base, str):
                    want = base + case9["suffix_literal"]
                    stats["CELL9_seen"] += 1
                    if got9 == want:
                        stats["CELL9_ok"] += 1
                    else:
                        findings.append({
                            "verdict": "CELL_MISMATCH",
                            "where": "%s/%s :: %s [%s] testArray[9]" % (repo, fn, path, rng),
                            "key": "ALIENRPG.Permanent", "got": got9, "want": want,
                            "note": "这一格在字符串末尾，**没有**尾随空格。对不上就掉进 "
                                    "[[NdM]] 解析分支，.match() 返回 null 后 null[1] 抛未捕获 "
                                    "TypeError，整次重伤掷骰中断。← " + case9["at"],
                        })
                        stats["**CELL_MISMATCH**"] += 1
            elif isinstance(h_en, str) and h_en.startswith("[["):
                stats["CELL9_roll_seen"] += 1
                if isinstance(got9, str) and ROLL_RX.match(got9):
                    stats["CELL9_roll_ok"] += 1
                else:
                    findings.append({
                        "verdict": "ROLL_SHAPE",
                        "where": "%s/%s :: %s [%s] testArray[9]" % (repo, fn, path, rng),
                        "key": "testArray[9]", "got": got9,
                        "want": "以 %s]] 开头" % h_en.split("]]")[0],
                        "note": crit["healing_time_roll_shape"]["rule"],
                    })
                    stats["**ROLL_SHAPE**"] += 1

    # ------------------------------------------------------------ 3. 另外三条分支的下标锁步
    #
    # `case "synthetic"` / `case "creature"` 读 testArray[0] 与 [1]；
    # `case "spacecraft"` 读 testArray[0]、[2]、[5]。两者都用 /[:] |<br \/>/gi 切，
    # 不做 <br /> 归一化。切少一段，下标就落到数组外面：`system.header.repairroll`
    # 变成 undefined，重伤物品照样被创建，**什么都不报**。
    other = crit.get("other_branches") or {}
    other_meta = other.get("tables") or {}
    if other_meta:
        for repo, fn, path, en_name, node in find_translated_tables(repos, set(other_meta)):
            if "__parse_error__" in node:
                findings.append({"verdict": "SPLIT_SHAPE", "where": "%s/%s" % (repo, fn),
                                 "key": "-", "got": node["__parse_error__"], "want": "可解析的 JSON",
                                 "note": "重伤表所在的译文文件读不动。"})
                stats["**SPLIT_SHAPE**"] += 1
                continue
            meta = other_meta[en_name]
            want_n = meta["expected_split_length"]
            idxs = meta.get("indices_read") or []
            # 每个 range 允许的段数 = 登记表在各个包里**实测**到的段数的并集
            allowed = {}
            for _pk, rows in (meta.get("measured_split_length_by_range") or {}).items():
                for _rng, _n in rows.items():
                    allowed.setdefault(_rng, set()).add(_n)
            results = node.get("results")
            if not isinstance(results, dict):
                stats["OTHER_no_results(表被译了但没译 results，跳过)"] += 1
                continue
            stats["OTHER_TABLE_seen"] += 1
            for rng, row in results.items():
                if not isinstance(row, dict):
                    continue
                desc = row.get("description")
                if not isinstance(desc, str) or desc == "":
                    stats["OTHER_ROW_no_description(跳过)"] += 1
                    continue
                stats["OTHER_ROW_seen"] += 1
                arr = split_like_other_branch(desc)
                # 允许两种：上游**实测**的段数（忠实翻译）和标准段数（顺手把上游的
                # markup 缺陷修好）。别的都不行。
                ok_lens = (allowed.get(rng) or set()) | {want_n}
                if len(arr) not in ok_lens:
                    findings.append({
                        "verdict": "SPLIT_SHAPE",
                        "where": "%s/%s :: %s [%s]" % (repo, fn, path, rng),
                        "key": "split length (%s 分支)" % meta.get("branch", "?"),
                        "got": len(arr),
                        "want": "%s（上游实测）" % sorted(ok_lens),
                        "note": "%s  这条分支用 /[:] |<br \\/>/gi 切，读下标 %s。%s  实际切片：%r"
                                % (other.get("rule", ""), idxs,
                                   next((b.get("breaks_if_desynced", "") for b in other.get("branches", [])
                                         if en_name in (b.get("tables") or [])), ""),
                                   arr),
                    })
                    stats["**SPLIT_SHAPE**"] += 1
                    continue
                if len(arr) != want_n:
                    # 上游本来就不标准的行（登记表 upstream_deviations 里记了原因）。
                    # 忠实翻译理应保持原样，所以**不**再拿下标去核 —— 否则一份完全
                    # 正确的译文会被这条上游缺陷判死。
                    stats["OTHER_ROW_nonstandard_but_upstream(容忍 %d 段)" % len(arr)] += 1
                    continue
                for i in idxs:
                    if i >= len(arr):
                        findings.append({
                            "verdict": "SPLIT_SHAPE",
                            "where": "%s/%s :: %s [%s] testArray[%d]" % (repo, fn, path, rng, i),
                            "key": "index out of range", "got": len(arr), "want": "> %d" % i,
                            "note": "这条分支要读 testArray[%d]，译文切出来只有 %d 段。" % (i, len(arr)),
                        })
                        stats["**SPLIT_SHAPE**"] += 1
                    else:
                        stats["OTHER_index_ok"] += 1

    checked = (stats["LANG_seen"] + stats["CELL3_seen"] + stats["CELL5_seen"]
               + stats["CELL9_seen"] + stats["CELL9_shift_seen"] + stats["CELL9_roll_seen"]
               + stats["OTHER_ROW_seen"])

    print("登记表  %s" % os.path.relpath(a.register, PROJ))
    print("受控键  %d 个：%s" % (len(keys), ", ".join(keys)))
    print("        （其中 ALIENRPG.Shift 只做输出，不参与 ===，但同样过卫生检查）")
    print("        + 1 个裸字面量 'Shift'（%s）"
          % next(c for c in crit["cases"] if c.get("literal") == "Shift")["at"])
    print("重伤表  在 compendium/cn 里找到 %d 份（%s）"
          % (len(hits), ", ".join(sorted({h[3] for h in hits if h[3]})) or "无"))
    if other_meta:
        print("另三支  %s 分支的 %d 张表也在核："
              % ("/".join(sorted({m.get("branch", "?") for m in other_meta.values()})), len(other_meta)))
        for tn in sorted(other_meta):
            m = other_meta[tn]
            print("        %-34s 期望 %d 段，读下标 %s"
                  % (tn, m["expected_split_length"], m.get("indices_read")))
    print()
    print("统计：")
    for k, v in sorted(stats.items()):
        print("   %-52s %d" % (k, v))
    print()
    print("checked=%d  violations=%d" % (checked, len(findings)))

    if not findings:
        if checked == 0:
            print("✓ 没有可核的东西（lang 与 compendium/cn 都还是空的），checked=0，退出 0。")
        else:
            print("✓ 8 个键与译好的重伤表逐字节对齐。")
    else:
        by = collections.Counter(f["verdict"] for f in findings)
        print("按判据：" + "  ".join("%s=%d" % kv for kv in sorted(by.items())))
        for f in findings[:a.show]:
            print()
            print("  ✗ [%s] %s" % (f["verdict"], f["where"]))
            print("     键   %s" % f["key"])
            print("     实际 %r" % f["got"])
            print("     应为 %r" % f["want"])
            print("     %s" % f["note"])
        if len(findings) > a.show:
            print("\n  …… 还有 %d 条，用 --out 全量导出。" % (len(findings) - a.show))

    if a.out:
        json.dump({"register": os.path.abspath(a.register),
                   "repos": [os.path.abspath(r) for r in repos],
                   "lang": os.path.abspath(a.lang) if a.lang else None,
                   "checked": checked, "stats": dict(stats), "findings": findings},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("-> %s" % a.out)

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
