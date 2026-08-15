# -*- coding: utf-8 -*-
"""反空转护栏的**灵敏度**探针：把 7 条新断言的英文正则按**已经发生过的那种事故形态**弄坏，
确认 `min_gated` 会把它们吵红，而不是静静地全绿。

形态 (a) 复现的是 `R-catwalk` 的真实事故：`"\\bcatwalk"` 在 JSON 里写成单反斜杠，
被 JSON 当成退格符 `\\x08` 吃掉，正则一个都匹配不到，断言一路报绿而库里实有 10 叶违规。
形态 (b) 是「上游改了措辞」：把英文正则换成一个库里根本不存在的词。

⚠ 只改内存里的规则副本，`RESOLUTIONS.assertions.json` 一个字节都不碰。
"""
import json
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = P + "/5-其他内容/RESOLUTIONS.assertions.json"
BS = "\x08"                       # 退格符：JSON 把 "\b" 解成它，正则里就是「匹配退格符」


def run(rules, tag):
    p = os.path.join(HERE, f"idle_probe.{tag}.json")
    json.dump({"meta": {}, "assertions": rules}, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    r = subprocess.run([sys.executable, P + "/3-常用脚本/qa/assert_resolutions.py",
                        "--rules", p, "--max-show", "1"],
                       capture_output=True, encoding="utf-8", errors="replace")
    tail = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("通过 ")]
    fails = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip().startswith("FAIL")]
    print(f"  {tag}: {tail[-1] if tail else '(没拿到统计行)'}")
    for f in fails[:8]:
        print(f"      {f}")
    return tail[-1] if tail else ""


def main():
    d = json.load(open(RES, encoding="utf-8"))
    new = [r for r in d["assertions"] if r["kind"] == "enricher_slot_gate"]
    print(f"新断言 {len(new)} 条：{[r['id'] for r in new]}\n")

    print("对照组（原样跑）：")
    run(json.loads(json.dumps(new)), "control")

    print("\n形态 (a) 判据被 JSON 转义吃掉（\\b -> 退格符），英文侧一个都匹配不到：")
    a = json.loads(json.dumps(new))
    for r in a:
        r["id"] += "-BROKEN-A"
        r.pop("forbid_absent", None)          # 反向闸会自己报红，那不是本探针要测的东西
        r.pop("cn_only_leaf_fallback", None)
        for t in r["en_tokens"]:
            t["re"] = t["re"].replace("\\b", BS)
    run(a, "broken_a")

    print("\n形态 (b) 上游改了措辞：英文正则换成库里根本不存在的词：")
    b = json.loads(json.dumps(new))
    for r in b:
        r["id"] += "-BROKEN-B"
        r.pop("forbid_absent", None)
        r.pop("cn_only_leaf_fallback", None)
        for i, t in enumerate(r["en_tokens"]):
            t["re"] = f"\\bZzzNoSuchWord{i}\\b"
    run(b, "broken_b")


main()
