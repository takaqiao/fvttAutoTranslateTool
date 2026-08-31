# -*- coding: utf-8 -*-
"""
把 panel_candidates.json 里的候选串切成「DOM 文本节点」粒度 —— translateText 比的是
**整个 trim 过的文本节点**（ember-hardcoded-cn.mjs:2572 起），所以带 HTML 的整串永远查不到。

切法：
  · 去掉 `TPL:` 前缀（模板串）
  · 按 `<...>` 标签切开，取标签之间的文字
  · `${...}` 原样保留在文本里（它和左右文字同属一个文本节点），但记下「含变量」标记
  · 折叠内部空白（translateNode 的回退分支就是这么做的）
前置自证：
  A 切对条数 —— 三条已知真值：
      `<p>This lever has already been activated.</p>`      -> 恰好 1 个文本节点
      tar-pit-spawn 那个 fieldset 模板                      -> 文本节点里必须含 `Spawn Actors`
      `<p>Transport everyone inside the tower to ${destination}?</p>` -> 1 个、且标记为含变量
  B 切对对象 —— 切出来的文本节点里不许再含 `<` 或 `>`
"""
import json, re, sys, io

TAG = re.compile(r"<[^<>]*>", re.S)
VAR = re.compile(r"\$\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.S)


def textnodes(s):
    if s.startswith("TPL:"):
        s = s[4:]
    parts = TAG.split(s)
    out = []
    for p in parts:
        flat = re.sub(r"\s+", " ", p).strip()
        if not flat:
            continue
        out.append(flat)
    return out


def has_var(s):
    return bool(VAR.search(s))


if __name__ == "__main__":
    src = json.load(io.open(sys.argv[1], encoding="utf-8"))

    # ---- 前置自证 A ----
    a1 = textnodes("<p>This lever has already been activated.</p>")
    assert a1 == ["This lever has already been activated."], a1
    a2 = textnodes("TPL:${config.content ?? \"\"}\n <fieldset class=\"tar-pit-spawn\">\n <legend>\n Spawn Actors\n <button type=\"button\">x</button>\n </legend>\n </fieldset>")
    assert "Spawn Actors" in a2, a2
    a3 = textnodes("TPL:<p>Transport everyone inside the tower to ${destination}?</p>")
    assert a3 == ["Transport everyone inside the tower to ${destination}?"] and has_var(a3[0]), a3
    print("PRECHECK-A OK  3/3 已知真值一致")

    out = {}
    for s, meta in src.items():
        for t in textnodes(s):
            # ---- 前置自证 B ----
            if "<" in t or ">" in t:
                sys.exit("PRECHECK-B FAIL 文本节点里还有标签：%r" % t)
            if not re.search(r"[A-Za-z]", t):
                continue
            rec = out.setdefault(t, {"kinds": set(), "at": [], "var": has_var(t), "from": []})
            rec["kinds"].update(meta["kinds"])
            rec["at"].extend(meta["at"])
            rec["from"].append(s[:60])
    print("PRECHECK-B OK  %d 个文本节点均无标签" % len(out))

    res = {k: {"kinds": sorted(v["kinds"]), "at": sorted(set(v["at"])), "var": v["var"]}
           for k, v in sorted(out.items())}
    io.open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
    print("文本节点候选 =", len(res))
