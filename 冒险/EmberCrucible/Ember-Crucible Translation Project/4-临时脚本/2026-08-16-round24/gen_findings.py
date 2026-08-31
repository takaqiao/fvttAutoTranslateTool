# -*- coding: utf-8 -*-
"""生成 findings/window-ui.json —— 全部证据取自 probe_window_ui_raw.json（探针落盘产物）"""
import io, json, os, re

OUTDIR = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-16-round24"
FIND = os.path.join(OUTDIR, "findings")
ROOT_EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"

raw = json.load(io.open(os.path.join(OUTDIR, "probe_window_ui_raw.json"), encoding="utf-8"))
miss = [r for r in raw["records"] if r["judge_reports_missing"]]

# 模板 -> 宿主窗口（stage3 已核实全部仍被 ember.mjs 注册，行号见 registered_at）
HOST = {
  "templates/journal/pages/flowchart-view.hbs":        ("EmberQuestFlowchartPageSheet", 37252),
  "templates/journal/partials/tab-development.hbs":    ("EmberPageSheet 系 PARTS", 35679),
  "templates/journal/partials/tab-event-outcomes.hbs": ("EmberEventPageSheet", 36458),
  "templates/journal/partials/tab-event-hooks.hbs":    ("EmberEventPageSheet", 36462),
  "templates/journal/partials/hook.hbs":               ("HOOK_PARTIAL（各 Ember*PageSheet 共用）", 36377),
  "templates/applications/hex-hud.hbs":                ("EmberHexHUD", 25438),
  "templates/applications/calendar-visual.hbs":        ("EmberCalendarNavigation", 24428),
  "templates/applications/creation/header.hbs":        ("EmberCharacterCreationSheet", 121833),
  "templates/applications/creation/class.hbs":         ("EmberCharacterCreationSheet", 121841),
  "templates/applications/creation/path.hbs":          ("EmberCharacterCreationSheet", 121853),
  "templates/applications/token-maker/body.hbs":       ("EmberDynamicTokenConfig", 50559),
  "templates/applications/token-maker/layers.hbs":     ("EmberDynamicTokenConfig", 50555),
  "templates/applications/token-randomization.hbs":    ("EmberRandomTokenConfig", 51514),
  "templates/applications/vista-config-assets.hbs":    ("EmberVistaConfiguration", 33877),
  "templates/applications/vista-config-placement.hbs": ("EmberVistaConfiguration", 33882),
  "templates/applications/vista-config-scene.hbs":     ("EmberVistaConfiguration", 33887),
}

items = []
for r in miss:
    k = r["key"]
    if r["hbs_hits"]:
        h = r["hbs_hits"][0]
        host, reg = HOST.get(h["file"], ("?", 0))
        items.append({
            "key": k,
            "verdict": "FALSE_POSITIVE",
            "blind_spot": "语料不全 · 上游产出这串的地方是 .hbs 模板，判据只抓 .mjs",
            "upstream_now": "逐字未变，仍是 %s:%d 的**裸字面量**（不是 {{localize}} 参数）" % (h["file"], h["line"]),
            "evidence": {
                "file": "modules/ember/" + h["file"],
                "line": h["line"],
                "line_text": h["text"],
                "template_still_registered": True,
                "registered_at": "ember.mjs:%d" % reg,
                "host_application": host,
                "gate": "宿主类名以 Ember 开头，主闸 /^Ember/ 放行",
            },
            "action": "不动键。DOM 注入仍是唯一可行通道（上游没给 i18n 键）。",
        })
    elif r["other_mjs_hits"]:
        h = r["other_mjs_hits"][0]
        items.append({
            "key": k,
            "verdict": "FALSE_POSITIVE",
            "blind_spot": "语料不全 · 串在 modules/ember/scripts/ 下**另一个** .mjs 里，"
                          "判据只抓 ember.mjs 这一个文件",
            "upstream_now": "逐字未变，仍是 %s:%d 的 label 字面量" % (h["file"], h["line"]),
            "evidence": {
                "file": "modules/ember/" + h["file"],
                "line": h["line"],
                "line_text": h["text"],
                "note": "同组的 `Soulbound Progression` 在 crucible-async.mjs:233 定义，"
                        "但因为**碰巧**也出现在 ember.mjs 里而没被报缺 —— "
                        "两条同源同命的键一报一不报，正是语料缺口的实证，不是它俩状态不同。",
            },
            "action": "不动键。",
        })
    else:
        items.append({
            "key": k,
            "verdict": "FALSE_POSITIVE",
            "blind_spot": "空白差异 · 上游模板里这句跨两行且带缩进，"
                          "判据用原文 String.includes() 匹配单行键必然落空",
            "upstream_now": "逐字未变，vista-config-scene.hbs:6-7 的 <p class=\"hint\">，"
                            "换行点在 “that will / be used”",
            "evidence": {
                "file": "modules/ember/templates/applications/vista-config-scene.hbs",
                "line": 6,
                "line_text": '<p class="hint">Create a new custom vista composition, starting from '
                             'your currently viewed one, that will',
                "line_7_text": "be used as the starting point from which you can make further changes.</p>",
                "proof": "把上游全文做 \\s+ -> 单空格折叠后，该键即可命中（stage2 已跑，结果 True）",
                "note": "本表注释早已写明「键写成折叠空白后的单行形态，靠 translateNode 的回退命中」—— "
                        "运行时用的是 DOM textContent，浏览器已折叠空白，所以键是**对的**；"
                        "错的是判据拿未折叠的源文件比。",
            },
            "action": "不动键。",
        })

report = {
    "轮次": "第二十四轮 · ① EMBER_WINDOW_UI",
    "生成时间": "2026-08-16",
    "上游版本": {"ember": "0.6.0", "crucible": "见 systems/crucible/system.json"},
    "结论": {
        "一句话": "48 条**全部是 FALSE_POSITIVE**，成因是判据语料不全，"
                  "不是上游改措辞、也不是上游整块重写。REWORDED 0 / REMOVED 0 / STILL_LIVE 0。",
        "为什么比例这么高（48/76）": (
            "不是「整块搬家」，恰恰相反：EMBER_WINDOW_UI 这张表按设计就是**为模板裸串而建的**。"
            "表里 76 条中 46 条的出处注释本来就写着 `xxx.hbs:NN`，"
            "而判据（ember-cn-selfcheck.mjs:577-586）的语料只有 "
            "`modules/ember/scripts/ember.mjs` + `systems/<sys>/<sys>-compiled.mjs` 两份 .mjs。"
            "**表的性质与判据的语料从一开始就不匹配**，所以这张表的误报率天然最高 —— "
            "48/76 这个比例是判据缺口的度量，不是上游改动的度量。"),
        "三个先决问题的回答": {
            "1_是否集中在少数窗口": (
                "是。48 条散落在 16 个模板、8 个窗口：任务事件页（流程图/结果/钩子/开发）13 条、"
                "指示物制作器 + 随机化配置 16 条、远景配置 11 条、创角向导 7 条、"
                "六角格 HUD 与日历各 1 条。但『集中』不代表『被重写』—— "
                "集中只是因为这些窗口本来就是模板驱动的。"),
            "2_是否挪进了模板或_i18n_键": (
                "挪进模板：**本来就在模板里**，不是这次挪的（表注释里的 hbs:行号与当前上游"
                "**逐条完全一致，0 处漂移**，见 stage2 B 段）。"
                "挪进 i18n 键：**没有**。46 处模板命中里 `{{localize \"…\"}}` 形态 **0 处**，"
                "全是裸字面量；modules/ember/lang/en.json 里也没有任何一条以这些串为值。"
                "⇒ 通道不用改，DOM 注入仍是唯一可行通道。"),
            "3_窗口是否在_0.6.0_里已不存在": (
                "不存在的 0 个。16 个模板**全部**仍被 ember.mjs 以 `template: \"modules/ember/…\"` 注册"
                "（孤儿模板数 = 0，stage3），宿主类名全部以 Ember 开头、主闸 /^Ember/ 放行。"),
        },
        "分类计数": {"FALSE_POSITIVE": 48, "REWORDED": 0, "REMOVED": 0, "STILL_LIVE": 0},
        "误报成因分三种": {
            "语料缺 templates/（46 条）": "上游产出串的地方是 .hbs，判据不抓",
            "语料缺 ember/scripts 下其余 .mjs（1 条：Aster Progression）":
                "判据只抓 ember.mjs 一个文件，crucible-async.mjs / dnd5e-async.mjs 不在语料里",
            "空白折叠差异（1 条：Create a new custom vista composition…）":
                "模板里跨行 + 缩进，原文 includes() 匹配不到单行键",
        },
    },
    "判据该怎么补（已实测，未验证不写建议）": {
        "修法1": "语料加 modules/ember/templates/** 的 .hbs/.html —— 48 → 2",
        "修法2": "语料加 modules/ember/scripts/ 下**全部** .mjs（不止 ember.mjs）—— 2 → 1",
        "修法3": "匹配前把语料与键都做 \\s+ → 单空格折叠 —— 1 → 0",
        "三修法叠加后残留": 0,
        "副作用实测": "拿 5 个构造的不存在串（Ember Flowchart Wizard / Outcome Identifier / "
                      "Rotate Sideways / Reset Everything / Place  Assets  Now）过三修法后的语料，"
                      "5/5 仍正确报「查无此串」—— 折叠没有造出假阴性。",
        "还要补的一条元信息": (
            "面板 D 档现在只按表分组报数，不区分「该表的串本来该在哪种文件里」。"
            "建议给表加 `corpus` 标注（`mjs` / `hbs` / `both`），"
            "像已有的 `kind: composed|data|absent-by-design` 那样 —— "
            "语料对不上时报 skip（无从查起）而不是 warn（疑似失效）。"
            "否则下一次有人补了 templates 语料，别的 `.json`/`.hbs` 来源的表还会重演同一场误报。"),
        "顺带记一笔": (
            "本轮之所以能一眼看穿，是因为表注释里早写着 hbs 行号。"
            "判据没有去读这些注释 —— 硬编码表**自己就带着出处**，"
            "让判据按注释里的 `文件:行号` 去核（而不是全语料 grep），"
            "既能定位更准，还能顺带查出行号漂移。"),
    },
    "旁证 · 跑主闸时撞出的一条判据自身的洞（不属本工作面，已上报）": {
        "断言": "R-selfcheck-twin（第二十三轮新增，kind: twin_files）",
        "现象": "同一份 assert_resolutions.py，**换个 cwd 结论就变**："
                "从项目根跑 = 通过 61 / 失败 0；从 `3-常用脚本/qa/` 跑 = 通过 60 / 失败 1，"
                "报「配对文件缺失（a 在=False b 在=False）」—— 而两份文件都**实际存在**。",
        "根因": "assert_resolutions.py:145 `REPOS = {\"ember\": \"1-Ember汉化插件\", "
                "\"crucible\": \"2-Crucible汉化插件\"}`，于是 `ctx.repos` 的**键是 "
                "`ember`/`crucible`**；而规则 `pairs` 里写的 `repo_a` 是**目录名** "
                "`1-Ember汉化插件`。a_twin_files:1721 的 "
                "`ctx.repos.get(pair[\"repo_a\"], pair[\"repo_a\"])` 于是**永远取不到**，"
                "静默退化成裸目录名这个**相对路径**，按 cwd 解析。",
        "为什么这是空转而不只是路径 bug": (
            "该 handler 的注释（:1719-1720）明写「路径按仓解析，不用全局 root —— "
            "这样 `--root <副本>` 的灵敏度回测才能作用到副本树上」。实测**恰好相反**："
            "把 twin 复制到副本树、只在副本的 b 侧注入一行漂移，再 `--root 副本` 跑，"
            "断言 **detail 报「比对 1 对文件」、violations = 0**（probe_twin_sensitivity2.py）。"
            "即：它满足了 min_pairs≥1 的空转闸、报告自己比过了、结论是绿的，"
            "但比的是**真实树**不是副本树 —— 这条断言**做不了灵敏度回测**，"
            "它现在的绿是 cwd 恰好等于项目根的巧合。"
            "min_pairs 那道防空转闸在这里没救得了它，因为它确实『比成了 1 对』。"),
        "建议修法（未动手 —— 该文件不是本工作面独占）": (
            "① `pairs` 的 repo 字段改用 REPOS 的键（`ember` / `crucible`），"
            "或让 a_twin_files 同时接受目录名（反查 REPOS.values()）；"
            "② 把 `.get(k, k)` 的静默兜底删掉 —— **取不到就判失败**，"
            "「规则里写的仓名不在 REPOS 里」本身就该是硬错误，而不是退化成相对路径；"
            "③ 给自检加一条用例：往 `--root` 副本注入 twin 漂移，断言必须响。"),
        "证据脚本": "4-临时脚本/2026-08-16-round24/probe_twin_sensitivity.py"
                    " · probe_twin_sensitivity2.py",
        "本工作面的主闸状态": "从项目根跑：通过 61 / 失败 0；--selftest：6 组 70/70 全绿，exit=0。",
    },
    "逐条明细": items,
    "探针": [
        "4-临时脚本/2026-08-16-round24/probe_window_ui.py（抽键 + 分层语料对差）",
        "4-临时脚本/2026-08-16-round24/probe_window_ui_stage2.py（裸串 vs localize / 行号漂移 / 空白折叠）",
        "4-临时脚本/2026-08-16-round24/probe_window_ui_stage3.py（孤儿模板排除 = REMOVED 排除）",
        "4-临时脚本/2026-08-16-round24/probe_window_ui_stage4.py（修法验证 48→0 + 假阴性副作用）",
        "4-临时脚本/2026-08-16-round24/probe_window_ui_raw.json（原始逐键命中矩阵）",
    ],
    "本轮未改动任何 .mjs": True,
}

os.makedirs(FIND, exist_ok=True)
p = os.path.join(FIND, "window-ui.json")
with io.open(p, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("wrote", p, "items =", len(items))
vs = {}
for i in items:
    vs[i["verdict"]] = vs.get(i["verdict"], 0) + 1
print(vs)
