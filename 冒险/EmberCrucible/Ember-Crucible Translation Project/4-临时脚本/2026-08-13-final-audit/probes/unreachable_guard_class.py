# -*- coding: utf-8 -*-
"""
probe: unreachable_guard_class  —— 「守卫/兜底恒不可达」这一类的全库扩查
========================================================================

已确认实例（不重复报）
----------------------
register.js 的两个 migrate* 把 Document 独有的 getFlag/setFlag 用在 game.world
（DataModel）上，并用 `?.()` 吞掉失败 → 「只跑一次」闸与「失败告警」双双失效。

抽象出的机械判据
----------------
一个「守卫 / 兜底 / 补偿补丁」G，若满足下面任一条，就是同一类缺陷：

  S1  UPSTREAM-MEMBER-MISS
      G 读的成员在宿主对象的**实际类型**上不存在（用 `?.` / `??` 兜住后静默为 undefined）。
  S2  SWALLOWED-CALL
      G 的副作用调用写成 `X?.m?.(...)`：方法不存在时不抛错，包住它的 catch 恒不触发。
  S3  DEAD-GATE / DEAD-CATCH
      G 的条件恒为假，或 try 块里没有任何能抛的语句。
  S4  UNREAD-TARGET（本轮新增的子签名）
      G 确实改了数据、也确实打印了「已改写 N 条」，但**宿主从不读那个字段** ——
      「成功计数」与「实际效果」脱钩，等价于「失败告警恒不触发」。
  S5  WRONG-REGISTRY
      G 的兜底去一个**不可能装着目标对象**的注册表里找目标
      （典型：v14 里用 `ui.windows`（只装 ApplicationV1）去找 ApplicationV2）。

判据怎么跑
----------
第一步机械枚举（本脚本）：抽出全部
  · `<宿主>.<成员>` 链（宿主 = game/CONFIG/ui/canvas/foundry/crucible/babele/Hooks/document）
  · 全部 `X?.(` 可选调用
  · 全部 `X ?? {}` / `X ?? []` 兜底
第二步人工核实（必需）：每条候选到上游源码里确认成员是否存在、是否被读。
上游：
  Foundry 14.365.0  C:\\Program Files\\Foundry Virtual Tabletop\\resources\\app
  crucible 0.10.1   %LOCALAPPDATA%\\FoundryVTT\\Data\\systems\\crucible
  ember 0.6.0       %LOCALAPPDATA%\\FoundryVTT\\Data\\modules\\ember
  babele 2.9.1      %LOCALAPPDATA%\\FoundryVTT\\Data\\modules\\babele

假阳性模式（务必知道）
----------------------
 1. S1 只靠 grep 成员名判定「不存在」；Proxy / defineProperty / 动态键定义的成员会误判。
    → 每条必须找到上游的**定义点**才算数（本轮 31 条链全部找到了定义点）。
 2. S2 里有一类合法：目标本来就是**可选依赖**，作者有意让它缺失时静默跳过。
    区分标准：缺失时项目自身的功能是否失效。
 3. S4 的「无读者」结论靠反向 grep，只对**已发行的这一个上游版本**成立；
    上游一升级就要重跑（本轮的 grep 命令写在下面 VERIFIED 表里，可原样复现）。
 4. S3 的「恒假」判断是词法的；赋值右侧若能抛（frozen 对象在严格模式下）catch 就不是死的。
    → 本轮专门查了 crucible 只 deepFreeze 了 SYSTEM，`crucible.CONFIG` 没冻。

只读，不写库。
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"

CODE_DIRS = [
    os.path.join(ROOT, "1-Ember汉化插件"),
    os.path.join(ROOT, "2-Crucible汉化插件"),
    os.path.join(ROOT, "3-常用脚本", "release"),
    os.path.join(ROOT, "3-常用脚本", "extract"),
    os.path.join(ROOT, "3-常用脚本", "qa"),
]
SKIP_PARTS = ("compendium", "node_modules", "__pycache__", "lang")

HOSTS = r"(?:globalThis\.crucible|globalThis|game|CONFIG|ui|canvas|foundry|crucible|babele|Hooks|document|window)"
CHAIN = re.compile(HOSTS + r"(?:\s*\??\.\s*[A-Za-z_$][\w$]*)+")
OPT_CALL = re.compile(r"([A-Za-z_$][\w$.?\[\]'\"]*)\?\.\(")
NULLISH = re.compile(r"([A-Za-z_$][\w$.?\[\]'\"]*)\s*\?\?\s*(\{\}|\[\])")

# ---------------------------------------------------------------- #
# 第二步的人工核实结果。verdict:
#   OK    = 上游确实有这个成员/这条路径确实可达
#   DEAD  = 恒不可达（→ finding）
#   NOOP  = 可达但无消费者（S4）
VERIFIED = [
    # chain / 位置                                        verdict  上游依据
    ("game.world.getFlag/setFlag  register.js:64,127,353,403",
     "DEAD", "World→BaseWorld→BasePackage extends DataModel；getFlag 只在 common/abstract/document.mjs:947。"
             "**已由主控记录，本探针不重复报**"),
    ("globalThis.crucible.api.methods.syncOwnedItems  register.js:214",
     "OK", "crucible-compiled.mjs:47232 crucible.api.methods.syncOwnedItems"),
    ("globalThis.crucible.api.hooks.action.causticPhial  register.js:410",
     "OK", "crucible-compiled.mjs:8964 HOOKS$6.causticPhial；hooks 命名空间被 Object.freeze"
           "（14044）但**只冻最外层**，HOOKS$6.causticPhial 仍可写"),
    ("globalThis.CONFIG.Actor.documentClass  register.js:310",
     "OK", "setup 时已就绪；Adventure#importContent 用 getDocumentClass('Actor') 取的就是它"),
    ("isAdventureImportInvocation() 第 1 支 'Adventure.importContent'  register.js:205",
     "OK", "v14 client/documents/adventure.mjs:195 在 importContent 自己的帧里直接调用 "
           "cls.updateDocuments；V8 保留直接调用者帧（v8_async_stack_repro.mjs 实测 true）。"
           "ember/crucible 都没有子类化 Adventure（grep CONFIG.Adventure 零命中）"),
    ("isAdventureImportInvocation() 第 2 支 'EmberAdventureImporter._processSubmitData'  register.js:205",
     "DEAD", "① EmberAdventureImporter（ember.mjs:24012）根本没有 _processSubmitData，只覆写 "
             "_preImport/_onImport；② 即便基类那帧，也被中间的 await 抹掉（repro 实测 false）"),
    ("CONFIG.TextEditor.enrichers  ember-hardcoded-cn.mjs:361",
     "OK", "client/config.mjs:2832 enrichers:[]；text-editor.mjs:145 每次 enrich 现读 config 对象，"
           "包装 entry.enricher 生效"),
    ("crucible.CONFIG.languages/.knowledge 的 entry.label  ember-hardcoded-cn.mjs:394-402",
     "OK", "crucible i18nInit(47669) 的 preLocalizeConfig 把本体 31+2 条 label 就地本地化成中文；"
           "ember initialize()(126682) 塞进来的 4+23 条是裸英文，正是这张表要改的那些。"
           "crucible.CONFIG 没被 freeze（只有 SYSTEM$1 在 47810 deepFreeze）"),
    ("CONFIG.time.worldCalendarConfig / game.time.calendar  ember-hardcoded-cn.mjs:418",
     "NOOP", "两个对象都存在、都改得动（n≈26）；但**没有任何消费者读月名/星期名** —— 见下 grep"),
    ("ui.windows（patchCalendarNames 的重画兜底）  ember-hardcoded-cn.mjs:433",
     "DEAD", "client/ui.mjs:21 windows={} 只被 client/appv1/api/application-v1.mjs:415 写入；"
             "EmberCalendarNavigation(ember.mjs:24382) 是 ApplicationV2"),
    ("document.querySelector('#ember-calendar').dispatchEvent(new Event('change'))  :436",
     "DEAD", "new Event('change') 默认 bubbles:false；ApplicationV2._attachFrameListeners(1874) 只挂 "
             "pointerdown/click/auxclick；ember 全部 addEventListener('change') 都挂在表单控件上"),
    ("Hooks.on('renderApplicationV2')  ember-hardcoded-cn.mjs:472",
     "OK", "application.mjs:591 hookName:'render' + parentClassHooks 默认 true，"
           "#callHooks(1724) 沿继承链逐个 callAll → renderApplicationV2 必发；"
           "hookArgs=[this.#element,…]，element 是 frame，能 querySelector('.window-title')"),
    ("game.modules.get('babele')?.active  register.js:443 / babele-register.js:36",
     "DEAD", "babele.init 只由 babele 自己的 Hooks.once('init') 发（script/babele.js:33）；"
             "且两个 module.json 的 relationships.requires 都声明了 babele。恒真守卫，零后果"),
    ("mapping 选项 fallbackPolicy:'owner-package-before-generic'",
     "OK", "babele script/converter/document-converter.js:366 _fallbackPolicySteps 认这个值"),
    ("effectiveMappings 的浅层展开 vs babele 的 mergeObject 递归",
     "OK", "_mapmerge_check.mjs 实跑：两种合并对现有 layer 逐类型 JSON 等价，无分歧"),
]

MONTH_GREPS = [
    # S4 的反向证据，可原样复现
    (r'grep -c "months" %EMBER%\scripts\ember.mjs', "1（只有 3626 行的配置字面量本身）"),
    (r'grep -n "days.values" %EMBER%\scripts\ember.mjs', "0"),
    (r'grep -n "\.name" %FOUNDRY%\client\data\calendar.mjs', "0"),
    (r'grep -rn "months" %FOUNDRY%\client\applications\ %FOUNDRY%\templates\\', "0"),
    (r'core 唯一读 months 的是 calendar.mjs:384 formatTimestamp，取的是 month.ordinal（数字）', "—"),
    (r'ember 的日期串 formatEmberDate(ember.mjs:4063) 读 calendar.seasons.values[].name → _loc()',
     "→ 走的是 EMBER.CALENDAR.SEASONS.* 这套 i18n 键，不是月名"),
]


def js_files():
    out = []
    for d in CODE_DIRS:
        for dp, dns, fns in os.walk(d):
            low = dp.lower().replace("\\", "/")
            if any("/" + p in low or low.endswith("/" + p) for p in SKIP_PARTS):
                continue
            dns[:] = [x for x in dns if x not in ("node_modules", "__pycache__", "compendium")]
            for fn in fns:
                if fn.endswith((".js", ".mjs")):
                    out.append(os.path.join(dp, fn))
    return out


def strip_comments(src):
    src = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), src, flags=re.S)
    out = []
    for line in src.split("\n"):
        i = line.find("//")
        out.append(line[:i] if i >= 0 else line)
    return "\n".join(out)


def main():
    chains, optcalls, nullish = {}, [], []
    files = js_files()
    for f in files:
        src = strip_comments(io.open(f, encoding="utf-8").read())
        for i, line in enumerate(src.split("\n"), 1):
            for m in CHAIN.finditer(line):
                chains.setdefault(re.sub(r"\s+", "", m.group(0)).replace("?.", "."), []).append((f, i))
            for m in OPT_CALL.finditer(line):
                optcalls.append((f, i, line.strip()))
            for m in NULLISH.finditer(line):
                nullish.append((f, i, line.strip()))

    print("### 扫描范围：%d 个 js/mjs（两个插件运行时 + release/extract/qa 工装）" % len(files))
    for f in files:
        print("   ", os.path.relpath(f, ROOT))
    print("\n### S1 候选：不同的上游成员链 %d 条" % len(chains))
    for c in sorted(chains):
        print("  %-58s %s" % (c, "; ".join("%s:%d" % (os.path.basename(f), n) for f, n in chains[c][:5])))
    print("\n### S2 候选：可选调用 `X?.(` %d 处" % len(optcalls))
    for f, n, l in optcalls:
        print("  %s:%d  %s" % (os.path.basename(f), n, l[:140]))
    print("\n### S3 候选：`X ?? {}` / `X ?? []` %d 处" % len(nullish))
    for f, n, l in nullish:
        print("  %s:%d  %s" % (os.path.basename(f), n, l[:140]))

    print("\n### 人工核实结论（逐条到上游找定义点）")
    for what, verdict, why in VERIFIED:
        print("  [%-4s] %s\n          %s" % (verdict, what, why))

    print("\n### S4（历法表无读者）的反向证据，可原样复现")
    for cmd, res in MONTH_GREPS:
        print("  %s\n      → %s" % (cmd, res))


if __name__ == "__main__":
    main()
