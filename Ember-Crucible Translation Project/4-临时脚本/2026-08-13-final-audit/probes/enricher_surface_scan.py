# -*- coding: utf-8 -*-
"""
enricher_surface_scan.py —— 探针 C：增强器「输出面」硬编码英文扫描

把「[[/date]] 的 tooltip 全英文」抽象成的机械判据：

    对每一个**注册进 CONFIG.TextEditor.enrichers 的增强器**，
    取它的 enricher 回调 + onRender 回调 + 这两者直接调用的处理函数，
    枚举所有写进「用户可见输出面」的字符串：
        innerHTML= / innerText= / textContent= / label= /
        dataset.tooltip / dataset.tooltipText / dataset.tooltipHtml /
        setAttribute("aria-label"|"title"|"data-tooltip"…) /
        anchorOptions.name / Object.assign(x.dataset, {... tooltip: ...})
    如果该字符串**不是** _loc()/game.i18n 调用、含 ASCII 字母、
    且 ember-hardcoded-cn.mjs 的 EXACT/PREFIXED/PATTERNS/子表都命不中 →
    判为「未纳入替换表的增强器输出面」。

假阳性模式（必须人工复核）：
  - 值来自文档 name（babele 已译）→ 模板里看着是英文变量名，实际输出中文
  - 值来自 SYSTEM.*.label，这些在 i18nInit 时已被本地化过
  - 分支实际不会触发（如 game.system.id === "dnd5e" 分支在 crucible 世界里死代码）
所以本脚本只产候选，结论一律回源码逐条读。

只读，不写库。
"""
import re, os, sys, json

EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
CRUCIBLE = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\crucible-compiled.mjs"
PLUGIN = (r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
          r"\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs")

SURFACE = re.compile(
    r"(innerHTML|innerText|textContent|\.label|dataset\.tooltip\w*|"
    r"tooltipText|tooltipHtml|aria-label|setAttribute\(|toggleAttribute\(|"
    r"anchorOptions\.name|\bname:\s|\btitle:\s|\btooltip:\s)")

# 含 ASCII 单词的字符串/模板串字面量
LIT = re.compile(r"`([^`]*)`|\"([^\"\\]{2,})\"|'([^'\\]{2,})'")
HAS_WORD = re.compile(r"[A-Za-z]{3,}")
LOC = re.compile(r"_loc\(|game\.i18n|localize\(|format\(")


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def fn_body(src, start_idx):
    """从 function 关键字处起，按大括号配平截出函数体。"""
    i = src.find("{", start_idx)
    if i < 0:
        return ""
    depth, j = 0, i
    while j < len(src):
        c = src[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start_idx:j + 1]
        j += 1
    return src[start_idx:start_idx + 4000]


def plugin_tables():
    """读出插件替换表里所有英文键，用来判断「已覆盖」。"""
    s = read(PLUGIN)
    keys = set(re.findall(r'"([^"]+)":\s*"', s))
    prefixes = set(re.findall(r'\{\s*en:\s*"([^"]+)"', s))
    return keys, prefixes


def scan(path, label, fn_names):
    src = read(path)
    keys, prefixes = plugin_tables()
    out = []
    for name in fn_names:
        m = re.search(r"^(?:async )?function %s\b" % re.escape(name), src, re.M)
        if not m:
            out.append((name, "NOT-FOUND", ""))
            continue
        body = fn_body(src, m.start())
        line0 = src[:m.start()].count("\n") + 1
        for ln_off, line in enumerate(body.split("\n")):
            if not SURFACE.search(line):
                continue
            for gm in LIT.finditer(line):
                lit = next(g for g in gm.groups() if g is not None)
                if not HAS_WORD.search(lit):
                    continue
                if LOC.search(line) and "`" not in gm.group(0):
                    continue
                # 已覆盖判定
                bare = re.sub(r"\$\{[^}]*\}", "", lit).strip()
                covered = bare in keys or any(bare.startswith(p + ":") for p in prefixes)
                out.append((f"{label}:{name}@{line0+ln_off}", lit.strip(),
                            "COVERED" if covered else "UNCOVERED"))
    return out


EMBER_FNS = ["enrichAdvantage", "enrichCriticalResult", "enrichAncestry", "enrichCulture",
             "enrichLanguage$2", "enrichPath", "enrichAttunement", "enrichEventState",
             "enrichEventOutcome", "onRenderPassiveCheck$2", "enrichLanguage",
             "onRenderPassiveCheck", "onHoverGroupCheck", "enrichKnowledge",
             "enrichLanguage$1", "onRenderPassiveCheck$1", "finalizeEnrichedHTML",
             "onClickAttunementAward"]

CRUCIBLE_FNS = ["enrichDND5ESkill", "enrichAward", "renderAward", "enrichCounterspell",
                "renderCounterspell", "enrichMilestone", "renderMilestone", "enrichHazard",
                "renderHazard", "enrichCondition", "enrichAction", "enrichSpell",
                "enrichRule", "enrichSkillCheck", "createSkillCheckElement",
                "enrichKnowledge", "enrichTalent", "enrichLanguage", "renderSkillCheck",
                "enrichRef", "enrichLoot", "renderLoot", "enrichScroll",
                "displayPassiveCheck", "displayKnowledgeCheck", "displayTalentCheck",
                "displayLanguageCheck", "onClickSkillCheck", "onClickGroupCheck"]


def main():
    rows = scan(EMBER, "ember", EMBER_FNS) + scan(CRUCIBLE, "crucible", CRUCIBLE_FNS)
    for where, lit, verdict in rows:
        if verdict == "COVERED":
            continue
        print(f"{verdict:10s} {where:46s} {lit[:110]!r}")


if __name__ == "__main__":
    main()
