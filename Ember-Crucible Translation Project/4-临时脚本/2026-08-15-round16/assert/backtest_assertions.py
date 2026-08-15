# -*- coding: utf-8 -*-
"""灵敏度回测：往副本树里逐条注入违规，确认每条断言**真的会响**。

为什么非做不可：只测特异度（库是干净的、断言全绿）是**不够的** ——
「所有断言都返回空」也能全绿。本项目已经两次栽在这上面
（`R-catwalk` 正则被 JSON 转义吃掉、`distinct_terms` 压根不读库）。

跑法：python backtest_assertions.py
"""
import json, os, re, shutil, subprocess, sys, tempfile
sys.stdout.reconfigure(encoding="utf-8")

SRC = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
SCRATCH = r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\289d7a82-7d7b-4b2d-ac68-1439487a5f75\scratchpad"
COPY = os.path.join(SCRATCH, "assert-backtest-root")
RULES = os.path.join(SRC, "5-其他内容", "RESOLUTIONS.assertions.json")
SCRIPT = os.path.join(SRC, "3-常用脚本", "qa", "assert_resolutions.py")
EMBER, CRUC = "1-Ember汉化插件", "2-Crucible汉化插件"


def build_copy():
    if os.path.isdir(COPY):
        shutil.rmtree(COPY)
    for rel in (EMBER, CRUC):
        for sub in ("compendium", "lang"):
            s = os.path.join(SRC, rel, sub)
            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(COPY, rel, sub))
    print(f"副本树建好：{COPY}")


def run_rule(rule_id):
    """只跑一条规则（写一个单规则 rules 文件），返回 (通过?, 输出)。"""
    full = json.load(open(RULES, encoding="utf-8"))
    one = [r for r in full["assertions"] if r["id"] == rule_id]
    assert one, f"找不到规则 {rule_id}"
    tmp = os.path.join(SCRATCH, f"rule_{rule_id}.json")
    json.dump({"meta": full["meta"], "assertions": one}, open(tmp, "w", encoding="utf-8"),
              ensure_ascii=False)
    out = subprocess.run([sys.executable, SCRIPT, "--rules", tmp, "--root", COPY],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    return ("失败 0" in out.stdout), out.stdout


def patch(rel, sub, name, old, new, count=1):
    p = os.path.join(COPY, rel, sub, name)
    raw = open(p, encoding="utf-8").read()
    if old not in raw:
        return None, f"注入失败：副本里找不到 {old!r}"
    open(p, "w", encoding="utf-8").write(raw.replace(old, new, count))
    return (p, raw), None


# ---------------------------------------------------------------- 带闸的注入
#
# ⚠ 第一版用「整包文本里的第一处 old」做注入，`R-cyclonic` 与 `R-orb-destroyed` 两条
# 因此假报「抓不到」：包里第一处「气旋」在一段 cyclone（阵风/气旋）的法术描述里、
# 第一处「法珠已摧毁」在一段 `when the Orb is destroyed` 的传记里 ——
# **两处的英文都不命中该规则的闸**，注入自然打空。
# 这不是断言的毛病，恰恰说明闸很窄；但回测harness 必须按闸挑注入点，否则它测的是别的东西。

def _walk(n, p=""):
    if isinstance(n, dict):
        for k, v in n.items():
            yield from _walk(v, f"{p}.{k}" if p else k)
    elif isinstance(n, list):
        for i, v in enumerate(n):
            yield from _walk(v, f"{p}[{i}]")
    elif isinstance(n, str):
        yield p, n


def _set_at(node, target, newval, p=""):
    """按 `_walk` 生成的路径串定位并改写一个叶（不解析路径，避免键名自带点号的歧义）。"""
    if isinstance(node, dict):
        for k, v in node.items():
            np = f"{p}.{k}" if p else k
            if isinstance(v, str):
                if np == target:
                    node[k] = newval
                    return True
            elif _set_at(v, target, newval, np):
                return True
    elif isinstance(node, list):
        for i, v in enumerate(node):
            np = f"{p}[{i}]"
            if isinstance(v, str):
                if np == target:
                    node[i] = newval
                    return True
            elif _set_at(v, target, newval, np):
                return True
    return False


def patch_gated(rel, pack, en_gate, old, new, case_sensitive=False):
    """只往「英文确实命中该规则的闸」的那一叶里注入。"""
    en_p = os.path.join(COPY, rel, "compendium", "en", pack)
    cn_p = os.path.join(COPY, rel, "compendium", "cn", pack)
    raw = open(cn_p, encoding="utf-8").read()
    en = dict(_walk(json.load(open(en_p, encoding="utf-8-sig"))))
    doc = json.load(open(cn_p, encoding="utf-8-sig"))
    cn = dict(_walk(doc))
    rx = re.compile(en_gate, 0 if case_sensitive else re.I)
    for path, cv in cn.items():
        ev = en.get(path)
        if ev is None or old not in cv or not rx.search(ev):
            continue
        assert _set_at(doc, path, cv.replace(old, new, 1)), path
        json.dump(doc, open(cn_p, "w", encoding="utf-8"), ensure_ascii=False)
        return (cn_p, raw), None
    return None, f"注入失败：{pack} 里没有「英文命中 {en_gate} 且中文含 {old!r}」的叶"


def restore(state):
    p, raw = state
    open(p, "w", encoding="utf-8").write(raw)


# (规则 id, 说明, 文件三元组, 原串, 注入串[, 英文闸])
# 带第 6 项的走 patch_gated（按闸挑注入点）；不带的是 lang 键，直接按字面替换。
CASES = [
    ("R-region-area-map", "把 EMBER.CALENDAR.REGION 改回历史错值「区域地图」——四个版本没人发现的那一处",
     (EMBER, "lang", "cn.json"), '"EMBER.CALENDAR.REGION": "地区地图"', '"EMBER.CALENDAR.REGION": "区域地图"'),
    ("R-round-turn", "把 crucible 的 COMBAT.INITIATIVE.Round 从「轮」改成「回合」",
     (CRUC, "lang", "cn.json"), '"COMBAT.INITIATIVE.Round": "先攻 - 轮 {round}"',
     '"COMBAT.INITIATIVE.Round": "先攻 - 回合 {round}"'),
    ("R-aura-three-way", "把 ember 的 SPELL.INFLECTIONS.Aura 从「奥拉」统一成「灵气」",
     (EMBER, "lang", "cn.json"), '"SPELL.INFLECTIONS.Aura": "奥拉"', '"SPELL.INFLECTIONS.Aura": "灵气"'),
    ("R-token-foundry-ui", "把一处「指示物」改回「令牌」",
     (EMBER, "compendium/cn", "ember.adventure.json"), "指示物", "令牌", r"\bTokens?\b"),
    ("R-hex-tile", "把一处「六边格」改回「六角格」",
     (EMBER, "compendium/cn", "ember.adventure.json"), "六边格", "六角格", r"\bhex(es)?\b"),
    ("R-lantyr", "把一处「兰提尔」改回「兰蒂尔」",
     (EMBER, "compendium/cn", "ember.adventure.json"), "兰提尔", "兰蒂尔", r"\bLantyr"),
    ("R-point-cape", "把场景针脚「岬 Point」改回音译「波因特 Point」",
     (EMBER, "compendium/cn", "ember.adventure.json"), "岬 Point", "波因特 Point", r"\bPoints?\b"),
    ("R-cyclonic", "把「气旋」改成与 Whirlwind 撞名的「旋风的」",
     (EMBER, "compendium/cn", "ember.crucible-adventure.json"), "气旋", "旋风的", r"\bCyclonic\b"),
    ("R-orb-destroyed", "把「法珠已摧毁」改成 mappings.mjs 那个库内零支持的「法珠已毁」",
     (EMBER, "compendium/cn", "ember.crucible-adventure.json"), "法珠已摧毁", "法珠已毁",
     r"\bOrb Destroyed\b"),
    ("R-obsidian-antiquary", "把「黑曜石古物」改回「黑曜古物」",
     (EMBER, "compendium/cn", "ember.adventure.json"), "黑曜石古物", "黑曜古物",
     r"\bObsidian Antiquar\w*"),
    ("R-temple-lunarium", "把「神殿月辉宫」改回「月神殿」",
     (EMBER, "compendium/cn", "ember.crucible-adventure.json"), "神殿月辉宫", "月神殿",
     r"\bTemple Lunarium\b"),
    ("R-corla-cora", "把「科尔拉」改回与人物 Cora 撞名的「科拉」",
     (EMBER, "compendium/cn", "ember.adventure.json"), "科尔拉", "科拉", r"\bCorla\b"),
]


def main():
    build_copy()
    print("\n=== 基线（不注入）：这几条应当全绿 ===")
    base_bad = []
    for rid in sorted({c[0] for c in CASES}):
        ok, out = run_rule(rid)
        print(f"  {'ok  ' if ok else 'FAIL'} {rid}")
        if not ok:
            base_bad.append(rid)
            print("      " + "\n      ".join(out.strip().splitlines()[-6:]))

    print("\n=== 注入回测：每条都必须由绿变红 ===")
    fails = []
    for case in CASES:
        rid, note, (rel, sub, name), old, new = case[:5]
        gate = case[5] if len(case) > 5 else None
        if gate:
            state, err = patch_gated(rel, name, gate, old, new)
        else:
            state, err = patch(rel, sub, name, old, new)
        if err:
            print(f"  ??   {rid}  {err}")
            fails.append(rid)
            continue
        try:
            ok, out = run_rule(rid)
        finally:
            restore(state)
        flipped = not ok
        print(f"  {'ok  ' if flipped else 'FAIL'} {rid}  —— {note}")
        if not flipped:
            fails.append(rid)
            print("      注入后仍然全绿 —— 这条断言抓不到它要守的东西！")

    print("\n=== 结构性回测：distinct_terms 缺 scan / 英文闸打空 ===")
    full = json.load(open(RULES, encoding="utf-8"))
    probe = [dict(r) for r in full["assertions"] if r["id"] == "R-region-area-map"]
    probe[0] = {k: v for k, v in probe[0].items() if k != "scan"}
    tmp = os.path.join(SCRATCH, "rule_noscan.json")
    json.dump({"meta": full["meta"], "assertions": probe}, open(tmp, "w", encoding="utf-8"),
              ensure_ascii=False)
    out = subprocess.run([sys.executable, SCRIPT, "--rules", tmp, "--root", COPY],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = "失败 0" not in out.stdout
    print(f"  {'ok  ' if ok else 'FAIL'} 去掉 scan 的 distinct_terms 必须判失败（不给「只做配置自检」留后门）")
    if not ok:
        fails.append("no-scan-guard")

    probe = [dict(r) for r in full["assertions"] if r["id"] == "R-hex-tile"]
    probe[0]["en"] = r"\bZZZNotAWordZZZ\b"
    tmp = os.path.join(SCRATCH, "rule_idle.json")
    json.dump({"meta": full["meta"], "assertions": probe}, open(tmp, "w", encoding="utf-8"),
              ensure_ascii=False)
    out = subprocess.run([sys.executable, SCRIPT, "--rules", tmp, "--root", COPY],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = "失败 0" not in out.stdout
    print(f"  {'ok  ' if ok else 'FAIL'} 英文闸命中 0 的 cn_absent 必须判失败（min_hits 反空转）")
    if not ok:
        fails.append("min-hits-guard")

    print("\n" + "=" * 62)
    if base_bad:
        print(f"⚠ 基线本就不绿：{base_bad}")
    print(f"回测结论：{len(CASES) + 2 - len(fails)} / {len(CASES) + 2} 条断言确认可被触发")
    if fails:
        print(f"⚠ 抓不到的：{fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
