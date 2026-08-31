# -*- coding: utf-8 -*-
"""
生成**两份判据侧文件的补丁副本**（这两份文件不归本轮作者，验到底后升报）。

改什么、为什么，逐条：
 ① `R-patterns-translate-cases`（PATTERNS 加了 1 条 ⇒ 三处必须同时动，少一处主闸红）
    · positive +1：新正则的正例（逐字符相符）
    · negative +1：**属于它自己的**近似反例（含它的字面骨架、但不在串首 ⇒ 锚定正则吃不到）
    · recorded.patterns_size 28→29 · recorded.positive 81→82 · recorded.negative 66→67
 ② `R-selfcheck-d-liveness`（登记表加了 671 键 + 1 条正则 ⇒ 覆盖侧的数全往上走）
    · min 的 8 个值抬到**现跑实测值**（面板 runner 现跑，不是手算）
    · max 的 5 个值**一个不动**：missDistinct 4 / rawMiss 7 / fetchFail 3 /
      uncheckedRaw 186 / uncheckedDistinct 164 实测**原地未动** —— 这正是本轮
      把部件表按 **id 建键 + literal 登记**（而不是显示名 + composed）换来的。
 ③ `assert_resolutions.py` 的 `PAYLOAD_FLOORS` 对应 13 个记录值同步。
"""
import io, json, os, re, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RULES = os.path.join(ROOT, "5-其他内容", "RESOLUTIONS.assertions.json")
PY = os.path.join(ROOT, "3-常用脚本", "qa", "assert_resolutions.py")
PANEL = os.path.join(HERE, "..", "verify", "panel_after.json")

POS = ["Transport everyone inside the tower to Golden Flats?", "将塔内所有人传送至 Golden Flats？"]
NEG = "Please Transport everyone inside the tower to the docks?"

def die(m): sys.stderr.write("make_patch: " + m + "\n"); sys.exit(2)

counts = json.load(io.open(PANEL, encoding="utf-8"))["counts"]
MIN_NEW = {"checkedDistinct": counts["checkedDistinct"], "rawChecked": counts["rawChecked"],
           "registeredRaw": counts["registeredRaw"], "registeredDistinct": counts["registeredDistinct"],
           "tableRegexEntries": counts["tableRegexEntries"], "tableRows": counts["tableRows"],
           "tablesFedIn": counts["tablesFedIn"], "fetchOk": counts["fetchOk"]}
MAX_KEEP = {"missDistinct": 4, "rawMiss": 7, "fetchFail": 3, "uncheckedRaw": 186, "uncheckedDistinct": 164}
for k, v in MAX_KEEP.items():
    if counts[k] != v:
        die(f"前置 FAIL 面板 max 侧 {k} 实测 {counts[k]} ≠ 基线 {v} —— 本轮的前提（miss/没核两侧原地不动）不成立")
print("前置 OK  面板 max 侧 5 个数实测与基线逐条相同：" + " · ".join(f"{k} {v}" for k, v in MAX_KEEP.items()))

rules = json.load(io.open(RULES, encoding="utf-8"))
hitP = hitS = 0
for a in rules["assertions"]:
    if a["id"] == "R-patterns-translate-cases":
        hitP += 1
        if POS[0] in [p[0] for p in a["positive"]]: die("正例已存在，补丁重复")
        if NEG in a["negative"]: die("反例已存在，补丁重复")
        a["positive"].append(POS)
        a["negative"].append(NEG)
        a["recorded"]["patterns_size"] += 1
        a["recorded"]["positive"] = len(a["positive"])
        a["recorded"]["negative"] = len(a["negative"])
        a["recorded"]["_why_30"] = (
            "第三十轮（2026-08-22 第三轮 UI 缺口）：`ember-hardcoded-cn.mjs` 的 PATTERNS 新增 1 条 —— "
            "沃特斯特塔传送装置的确认正文 `^Transport everyone inside the tower to (.+)\\?$`"
            "（ember.mjs:126652，变量位是 `canvas.scene.levels.get(id)?.name`，是场景层名这个开放集合，"
            "枚举不了所以只能走正则）。⇒ patterns_size 28→29 · positive 81→82 · negative 66→67。"
            "反例取 `Please Transport everyone inside the tower to the docks?`：含这条正则的字面骨架、"
            "但骨架**不在串首**，锚定的 `^…$` 结构上吃不到它 —— 这正是 (N1) 要的那种「属于它自己的近似反例」。")
        P = a["recorded"]
        print(f"① R-patterns-translate-cases：positive {len(a['positive'])} · negative {len(a['negative'])} · "
              f"patterns_size {P['patterns_size']}")
    if a["id"] == "R-selfcheck-d-liveness":
        hitS += 1
        old = dict(a["min"])
        a["min"].update(MIN_NEW)
        print("② R-selfcheck-d-liveness.min：" + " · ".join(
            f"{k} {old[k]}→{v}" for k, v in MIN_NEW.items() if old[k] != v))
        if a["max"] != MAX_KEEP | {k: a["max"][k] for k in a["max"] if k not in MAX_KEEP}:
            pass
        print("   max 五个值不动：" + json.dumps(a["max"], ensure_ascii=False))
if hitP != 1 or hitS != 1: die(f"规则里命中 {hitP} / {hitS} 条，各期望 1 条")

out_rules = os.path.join(HERE, "RESOLUTIONS.assertions.PATCHED.json")
json.dump(rules, io.open(out_rules, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("已写 " + out_rules)

# ── PAYLOAD_FLOORS ──
src = io.open(PY, encoding="utf-8").read()
def bump(block_key, pairs):
    global src
    i = src.index('    "%s": {' % block_key)
    j = src.index("},\n", i) + 1
    blk = src[i:j]
    for k, (kind, new) in pairs.items():
        pat = '"%s": ("%s", ' % (k, kind)
        p = blk.index(pat)
        q = blk.index(")", p)
        old = blk[p + len(pat):q]
        blk = blk[:p + len(pat)] + str(new) + blk[q:]
        print(f"   PAYLOAD_FLOORS[{block_key}][{k}] {old} → {new}")
    src = src[:i] + blk + src[j:]

bump("R-patterns-translate-cases", {
    "negative": ("list", 67), "positive": ("list", 82),
    # `recorded` 块多了一条 `_why_30`（本轮改动的留痕），它的**键数**也是一道地板：
    # 前置 C「68 条 / 329 道地板必须逐道等于从规则集现推的值」会点名 ('list',10) vs 现推 ('list',11)。
    "recorded": ("list", 11),
    "recorded.negative": ("eq", 67), "recorded.positive": ("eq", 82),
    "recorded.patterns_size": ("eq", 29)})
bump("R-selfcheck-d-liveness", {("min.%s" % k): ("eq", v) for k, v in MIN_NEW.items()})
out_py = os.path.join(HERE, "assert_resolutions.PATCHED.py")
io.open(out_py, "w", encoding="utf-8", newline="").write(src)
print("已写 " + out_py)
