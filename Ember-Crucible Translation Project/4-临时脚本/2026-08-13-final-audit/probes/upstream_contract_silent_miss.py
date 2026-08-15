#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
probe: upstream_contract_silent_miss
====================================

把已确认那条（`game.world.getFlag?.()` —— World 是 DataModel 没有 getFlag，
可选调用把「方法不存在」静默变成「护栏 false」）抽象成一条可机械化判据：

  判据 P：
    (1) 代码从**上游对象**（Foundry v14 / crucible 0.10.1 / ember 0.6.0 / babele 2.9.1
        拥有的对象）上读一个成员，或往一个**上游注册表**里查一个目标；
    (2) 该成员 / 该目标在**钉死的上游版本**里其实**不存在**、或该注册表在结构上
        **不可能**装下那个目标；
    (3) 取不到的后果被**吞掉**：`?.` 可选取值 / `?.()` 可选调用 / `?? {}`/`?? []` 空兜底 /
        `if (!x) return;` 无日志早退 / `catch {}` 空捕获；
    (4) 于是控制流**静默走错分支**，且没有任何日志、异常、可观测信号。

  三条都满足 = 同类缺陷。只满足 (1)(3) 而 (2) 不成立 = 正常防御，不报。

本探针只做 (1)(3) 的机械抽取 + 上游成员名的存在性初筛，**不做判定**；
(2) 必须逐条回上游源码人工确认（本文件末尾的 RESOLVED 表记录人工结论）。

扫描面：两个汉化插件里所有会在 Foundry 运行时里执行的 JS/MJS。
  1-Ember汉化插件/register.js
  1-Ember汉化插件/babele-mappings.js
  1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs
  2-Crucible汉化插件/babele-register.js
  2-Crucible汉化插件/babele-mappings.js
（3-常用脚本/ 下的是 node 侧生成器，不在 Foundry 里跑，单列不判）

上游源码根：
  Foundry v14 build 365  C:\Program Files\Foundry Virtual Tabletop\resources\app\{client,common}
  crucible 0.10.1        %LOCALAPPDATA%\FoundryVTT\Data\systems\crucible\crucible-compiled.mjs
  ember 0.6.0            %LOCALAPPDATA%\FoundryVTT\Data\modules\ember\scripts\ember.mjs
  babele 2.9.1           %LOCALAPPDATA%\FoundryVTT\Data\modules\babele\script\babele.js

假阳性模式（必须知道，报之前要排掉）：
  FP1 成员是本模块自己造的私有标记（__emberSafePatched / __emberCnWrapped）→ 与上游无关。
  FP2 成员在上游确实存在 → 只是正常防御。
  FP3 空兜底的集合本来就允许为空（game.items 在空世界里就是空）→ 正常。
  FP4 成员名同名但属于别的类型（getFlag 在 Document 上有、在 DataModel 上没有）
      —— 所以「grep 到名字」**不等于**「这个对象上有」，必须看归属类型。
      这一条正是已确认那条的成因，也是本探针最大的误判来源，故初筛结果一律标 NEEDS_TYPE_CHECK。
  FP5 正则漏检：动态成员名 obj[key]、跨行链式写法、解构后再用。
      故另出「上游根对象出现次数」总表兜底，人工对读。

只读，不写库。
"""
import io
import os
import re
import json
import sys

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
OUT = os.path.join(ROOT, r"4-临时脚本\2026-08-13-final-audit\probes")

RUNTIME_FILES = [
    r"1-Ember汉化插件\register.js",
    r"1-Ember汉化插件\babele-mappings.js",
    r"1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs",
    r"2-Crucible汉化插件\babele-register.js",
    r"2-Crucible汉化插件\babele-mappings.js",
]

UPSTREAM_ROOTS = {
    "game", "CONFIG", "ui", "foundry", "canvas", "Hooks", "crucible", "ember",
    "babele", "CONST", "document", "Node", "globalThis", "HTMLElement",
}

# 本模块自己造的成员（FP1），不参与上游存在性判定
OWN_MEMBERS = {
    "__emberSafePatched", "__emberCnWrapped", "__emberCnEnrichers",
    "__emberCnConfig", "__emberCnCalendar", "__emberCnRender",
}

CHAIN = re.compile(
    r"\b(" + "|".join(sorted(UPSTREAM_ROOTS)) + r")"
    r"((?:\s*\??\.\s*[A-Za-z_$][\w$]*)+)"
)

SWALLOW_PATTERNS = [
    ("opt_call", re.compile(r"\?\.\(")),
    ("opt_get", re.compile(r"\?\.")),
    ("nullish_empty", re.compile(r"\?\?\s*(\{\}|\[\]|''|\"\")")),
    ("nullish_any", re.compile(r"\?\?")),
    ("silent_return", re.compile(r"^\s*if\s*\(.*\)\s*return[^;]*;")),
    ("filter_bool", re.compile(r"\.filter\(Boolean\)")),
]

EMPTY_CATCH = re.compile(r"catch\s*(?:\([^)]*\))?\s*\{\s*(?:/\*[^*]*\*/|//[^\n]*)?\s*\}")


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def empty_catch_line_ranges(src):
    """返回被 `catch {}`（空体或只有注释）覆盖的 try 块行号区间，用于标 (3) 的 catch 吞法。"""
    ranges = []
    for m in EMPTY_CATCH.finditer(src):
        # 往前找配对的 try
        head = src[:m.start()]
        idx = head.rfind("try")
        if idx < 0:
            continue
        start_line = head[:idx].count("\n") + 1
        end_line = src[:m.end()].count("\n") + 1
        ranges.append((start_line, end_line))
    return ranges


def scan_file(path, rel):
    src = read(path)
    lines = src.split("\n")
    catch_ranges = empty_catch_line_ranges(src)
    rows = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*"):
            continue
        for m in CHAIN.finditer(line):
            root = m.group(1)
            tail = [t.strip() for t in re.split(r"\??\.", m.group(2)) if t.strip()]
            if not tail:
                continue
            chain = root + "." + ".".join(tail)
            terminal = tail[-1]
            swallows = [k for k, p in SWALLOW_PATTERNS if p.search(line)]
            in_empty_catch = any(a <= i <= b for a, b in catch_ranges)
            if in_empty_catch:
                swallows.append("empty_catch")
            rows.append({
                "file": rel,
                "line": i,
                "chain": chain,
                "root": root,
                "terminal": terminal,
                "own_member": terminal in OWN_MEMBERS,
                "swallowed": sorted(set(swallows)),
                "text": stripped[:180],
            })
    return rows


# ---------------------------------------------------------------- #
# 判据 P 的第二个检测器：钩子回调「写进上游传来的参数对象」                       #
#                                                                    #
# 与 (1)(2)(3) 同型：上游把某个对象交给回调，回调往里写；但上游在**调用回调之前**   #
# 就已经把它 deepClone / 转成别的对象，回调之后读的是另一份 —— 写进去的东西被丢弃， #
# 没有异常、没有日志。                                                  #
# v14 的三条契约（client/data/client-backend.mjs 逐行核过）：              #
#   preUpdate<Type>(doc, changes, ...)  -> 之后 doc.updateSource(changes)   写入生效
#   preCreate<Type>(doc, data,  ...)    -> 之前 new cls(deepClone(data))，  写入被丢弃
#                                          之后 operation.data = documents
#   preDelete<Type>(doc, options, ...)  -> 只有 options 生效
HOOK_ARG_CONTRACT = {
    "preUpdateActor": ("changes", "LIVE  — updateSource(changes) 在钩子之后"),
    "preUpdateItem": ("changes", "LIVE  — 同上"),
    "preCreateItem": ("data", "DEAD  — 文档已由 deepClone(data) 建好，operation.data=documents"),
    "preCreateActor": ("data", "DEAD  — 同上"),
    "preCreateActiveEffect": ("data", "DEAD  — 同上"),
}
HOOK_ON = re.compile(r"Hooks\.(?:on|once)\(\s*['\"]([A-Za-z0-9_.]+)['\"]")


def scan_hook_arg_writes(path, rel):
    src = read(path)
    lines = src.split("\n")
    rows = []
    for i, line in enumerate(lines, 1):
        m = HOOK_ON.search(line)
        if not m:
            continue
        name = m.group(1)
        body = "\n".join(lines[i - 1:i + 6])
        contract = HOOK_ARG_CONTRACT.get(name)
        rows.append({
            "file": rel, "line": i, "hook": name,
            "contract": contract[1] if contract else "n/a — 非 pre* 文档钩子，不适用",
            "mutating_arg": contract[0] if contract else None,
            "body": body.strip()[:220],
        })
    return rows


def main():
    all_rows = []
    for rel in RUNTIME_FILES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print("MISSING", p)
            continue
        all_rows.extend(scan_file(p, rel))

    # 只有被吞掉的才是本判据的候选（(3) 成立）
    cands = [r for r in all_rows if r["swallowed"] and not r["own_member"]]

    # 按 chain 归并，便于逐条回上游查
    by_chain = {}
    for r in cands:
        by_chain.setdefault(r["chain"], []).append(r)

    out = {
        "total_upstream_accesses": len(all_rows),
        "swallowed_candidates": len(cands),
        "distinct_chains": len(by_chain),
        "chains": {k: [{"file": x["file"], "line": x["line"],
                        "swallowed": x["swallowed"], "text": x["text"]} for x in v]
                   for k, v in sorted(by_chain.items())},
        "note": "每条 chain 都是 NEEDS_TYPE_CHECK —— grep 到名字不等于该类型上有（FP4）。",
    }
    hook_rows = []
    for rel in RUNTIME_FILES:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            hook_rows.extend(scan_hook_arg_writes(p, rel))
    out["hook_arg_writes"] = hook_rows
    print("\n-- hook registrations --")
    for r in hook_rows:
        print("  ", r["file"], r["line"], r["hook"], "|", r["contract"])

    dest = os.path.join(OUT, "upstream_contract_silent_miss.json")
    with io.open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("total upstream member accesses:", len(all_rows))
    print("swallowed candidates:", len(cands), "distinct chains:", len(by_chain))
    for k in sorted(by_chain):
        print("  ", k, "->", ",".join(str(x["line"]) for x in by_chain[k]))
    print("wrote", dest)


RESOLVED = r"""
逐条回上游核对的结论（2026-08-14，Foundry v14 build 365 / crucible 0.10.1 /
ember 0.6.0 / babele 2.9.1，全部读源码，非猜测）：

== 检测器一：16 条被吞掉的上游成员链 ==
CONFIG.TextEditor.enrichers        存在 client/config.mjs:2832；条目形状 {id,pattern,enricher,
                                   onRender,replaceParent} 与 text-editor.mjs:264 的解构一致；
                                   enrichHTML 每次调用现读数组（text-editor.mjs:145）→ 包装生效。OK
CONFIG.time.worldCalendarConfig    存在 client/config.mjs:1972；ember 在 ember.mjs:129024 换成
                                   EMBER_CALENDAR_CONFIG，形状 months.values[].name 对得上。OK
Node.ELEMENT_NODE / document.querySelector                                            OK
crucible.CONFIG / globalThis.crucible.CONFIG   crucible-compiled.mjs:47243 设。OK
  └ 但 crucible 自带 languages/knowledge 的 label 是 i18n 键（"LANGUAGES.Common" /
    "KNOWLEDGE.Alchemy"，见 1007 / 586），只有 ember 在 126683/126693 新增的那批才是裸英文；
    patchCrucibleConfig 的 `table[entry.label]` 只会命中 ember 那批 —— 与注释一致，不是缺陷。
    （注：crucible.CONFIG.languageCategories 里 ember 新增的两条 label 没有任何补丁覆盖，
      属**覆盖缺口**不属本判据，另案。）
game.items / game.actors / actor.items / game.user.*   OK（空世界为空是合法的，FP3）
game.modules.get                    OK
game.time.calendar                  存在 client/helpers/time.mjs:127-128。OK
globalThis.CONFIG.Actor.documentClass + 静态 updateDocuments   OK（setup 时 crucible 已在 init 设好）
globalThis.crucible.api.methods.syncOwnedItems  存在 crucible-compiled.mjs:47231。
                                   syncTalents/syncIconicSpells/defineBatchOperations/
                                   foundry.documents.modifyBatch 也全部存在，且 register.js:232-251
                                   与 crucible 自己的 48190-48211 写法一致。OK
globalThis.crucible.api.hooks.action.causticPhial  存在（HOOKS$6.causticPhial，8964）；
                                   hooks 命名空间虽 Object.freeze，但 freeze 是浅的，
                                   HOOKS$6.causticPhial 可写 → hook.prepare = wrapped 生效。
                                   CrucibleAction._initialize 每次实例化都重读注册表
                                   （19023 → #prepareHooks 19047），而 #prepareActions 在每次
                                   actor 数据准备时重建 action（41694/42030）→ setup 期打的补丁
                                   会被后续准备取到。唯一没覆盖到的是 game.mjs:730
                                   initializeDocuments() 那一遍（早于 740 的 setup 钩子），
                                   但 ACTION_HOOKS.prepare 没有 throws，_callActionHooks
                                   (20234) 会 catch 并 console.error —— **有信号、不致命**，
                                   不满足判据 (4)，故**不报**。
ui.windows                          存在 client/ui.mjs:21，但只有 AppV1 popOut 会写进去
                                   （appv1/api/application-v1.mjs:415）。EmberCalendarNavigation
                                   是 HandlebarsApplicationMixin(ApplicationV2)（ember.mjs:24382，
                                   id "ember-calendar"），只会进 foundry.applications.instances。
                                   → **命中判据 P**，见 FINDING B。

== 检测器二：钩子参数写入 ==
preUpdateActor / preUpdateItem      LIVE。client-backend.mjs:248 Hooks.call 之后
                                   250 doc.updateSource(changes, {clean:true}) —— 改 changes 生效。
preCreateItem                       DEAD。client-backend.mjs:93 先 new cls(deepClone(createData))，
                                   103/104 才调 _preCreate / Hooks.call('preCreateItem', doc,
                                   createData, ...)，121 又把 operation.data 整个换成 documents。
                                   → 往 createData 上写的东西全被丢弃，无异常无日志。
                                   → **命中判据 P**，见 FINDING A。

== 顺带核过、结论为「没问题」的上游契约 ==
Hooks 'babele.init'                 babele.js:33 Hooks.callAll('babele.init', babele)。OK
babele.register/registerConverters/registerMapping   babele.js:166/197/291。OK
'renderApplicationV2'               application.mjs:592 hookName "render" + 1725 加 "{}" +
                                   1727 遍历 inheritanceChain → 每个 AppV2 都会发
                                   renderApplicationV2，hookArgs = [element, context, options]。OK
'i18nInit' 里改 game.i18n.translations   localization.mjs:235 先赋值、104 才 callAll。OK
isAdventureImportInvocation()       Adventure.importContent 存在（adventure.mjs:166）且**直接**
                                   调 cls.updateDocuments（第 187 行），栈深 3 帧，远低于 V8
                                   默认 stackTraceLimit=10；ember/crucible 都没有覆盖
                                   CONFIG.Adventure.documentClass，帧名就是 "Adventure.importContent"。
                                   第二个析取项 EmberAdventureImporter._processSubmitData 虽然
                                   EmberAdventureImporter 自己没定义该方法，但 V8 的 TypeName 取
                                   接收者构造函数名，继承自 AdventureImporter（adventure-importer.mjs:237）
                                   的调用照样打印成这个串 → 冗余但不失效。**不报**。
isKnownUpdateDiffError()            "getFailure"（common/data/fields.mjs:2849）与
                                   "One of original or other are not Objects"
                                   （common/utils/helpers.mjs:1129）在 v14 里都真实存在。OK
"""

if __name__ == "__main__":
    main()
