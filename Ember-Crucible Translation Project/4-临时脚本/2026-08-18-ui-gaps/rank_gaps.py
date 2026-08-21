#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给缺口串排「上屏可见性」。

方法：拿每条串的**每一处**出处（file:line），读那一行 + 前后若干行，按落在哪种通道给一个档，
再取所有出处里**最高**的那一档。分档只看通道，不看内容 —— 内容判断留给人。

  T1  控件/窗口/对话框：window.title、按钮 label、data-tooltip / aria-label / title= / placeholder、
      表单字段 label:/hint:、legend/textContent、ui.notifications、模板 .hbs 的文本与显示属性
  T2  数据/资产 label：落在精灵图 / 部件 / 远景素材注册表里的 label:（会上屏，但属内容层，成千上万条）
  T0  不上屏：throw new Error / console.* / warn( / JSDoc
  T?  通道认不出来（默认档，需人工看）

前置自证（两件都断言）：
  (A) 切对条数 —— 各档之和 == gaps.json 条数；
  (B) 改对地方 —— 已知样本必须落进已知档，错一条即 FAIL。
"""
import json, re, os, sys, io, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

U = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember"
HERE = os.path.dirname(os.path.abspath(__file__))
MJS_LINES = open(os.path.join(U, "scripts", "ember.mjs"), encoding="utf-8").read().split("\n")

_tpl = {}
def lines_of(f):
    if f == "ember.mjs":
        return MJS_LINES
    if f not in _tpl:
        _tpl[f] = open(os.path.join(U, f), encoding="utf-8").read().split("\n")
    return _tpl[f]

T0_RE = re.compile(r'throw new \w*Error|console\.(?:warn|error|log|debug)|#logOnce|'
                   r'^\s*\*|@param|@returns|@type|@property')
T1_RE = re.compile(r'data-tooltip|aria-label|placeholder\s*=|\btitle\s*=|\balt\s*=|'
                   r'\btitle:\s*["`]|\bhint:\s*["`]|\bprompt:\s*["`]|legend|'
                   r'\.textContent|\.innerText|\.innerHTML|ui\.notifications|'
                   r'\bok:\s*\{|\bbuttons\b|\bcontrols:|\bwindow:\s*\{|DialogV2|'
                   r'\.append\(|createElement')
LABEL_RE = re.compile(r'(?<![\w$.])label:\s*"')
T2_HOST = re.compile(r'^(SPRITES|NIGHT_TINT|VISTA_SPRITE|LAYERS|ANATOMY|EQUIPMENT|COLORS|'
                     r'BUILDS|STANCES|TEMPLATE|L|ABYSSAL|UNDEAD_\w+|SIGNARA_\w+|'
                     r'HUMAN|KETH|HULGRUN|CONSTRUCT|DRAKON|ALTYRA|NIRAE|CORAK|KISKA|KIVAHR|'
                     r'ASHKA|FEJ|ZEPH|THORNLING|SIGNBORN|VRJNHAR|WIRRUN|jurtak|party|'
                     r'undeadMonster|cosmos|make\w*Parts|EmberDynamicToken)(\$[\w]+)?$')
# 远景构图（贴图/装饰素材注册表）——宿主是小驼峰的地点 id 或 *AreaMap/*Vista*
T2_HOST2 = re.compile(r'^([a-z]\w+|\w*AreaMap|\w*Composition|Ember\w*Vista\w*)$')

RANK = {"T1": 3, "T2": 2, "T?": 1, "T0": 0}

gaps = json.load(open(os.path.join(HERE, "gaps.json"), encoding="utf-8"))
byhost = json.load(open(os.path.join(HERE, "gaps_by_host.json"), encoding="utf-8"))
host_of = {}
for h, v in byhost.items():
    for s in v:
        host_of[s] = h


def tier_at(s, loc):
    f, ln = loc.rsplit(":", 1)
    ln = int(ln)
    if not f.endswith(".mjs"):
        return "T1"                      # 模板里的文本/显示属性，天然上屏
    L = lines_of(f)
    win = "\n".join(L[max(0, ln - 4):ln + 2])
    line = L[ln - 1] if 0 < ln <= len(L) else ""
    if T1_RE.search(win):
        return "T1"
    hn = host_of.get(s, "").split("::", 1)[-1]
    if LABEL_RE.search(line) or LABEL_RE.search(win):
        return "T2" if (T2_HOST.match(hn) or T2_HOST2.match(hn)) else "T1"
    # 部件 id：不是 label，但会被 getLayerChoicesV2 在运行时派生成 partLabel 上屏
    # （ember.mjs:64457 `partId.split("/").at(-1).replace(/(?<!^)([A-Z1-9])/g, " $1")`）。
    if re.search(r'(?<![\w$.])id:\s*"', line) and (T2_HOST.match(hn) or T2_HOST2.match(hn)):
        return "T2"
    if T0_RE.search(win):
        return "T0"
    return "T?"


tier = {}
for s, locs in gaps.items():
    best = "T0"
    for loc in locs:
        try:
            t = tier_at(s, loc)
        except Exception:
            continue
        if RANK[t] > RANK[best]:
            best = t
    tier[s] = best

cnt = collections.Counter(tier.values())
print("[自证A] 分档：", dict(cnt), "合计", sum(cnt.values()), "/ gaps", len(gaps))
assert sum(cnt.values()) == len(gaps)

KNOWN = {
    "Hair Roots": "T2", "Hair Base": "T2",
    "Add Reinforcements": "T1",
    "Spawn Actors": "T1",
    "Trigger Text": "T1",
    "Automatically pause the game when the trap is triggered?": "T1",
    "Bloody1": "T2", "Adult1": "T2", "BraidsMany": "T2",
    "Reinforcements": "T1",
}
print("[自证B] 抽样分档：")
ok = True
for s, want in KNOWN.items():
    got = tier.get(s, "<不在缺口集>")
    hit = got == want
    print(f"   {'OK  ' if hit else 'FAIL'}  {s!r} -> {got}（期望 {want}）宿主 {host_of.get(s)} @ {gaps.get(s, [''])[:2]}")
    ok = ok and hit

out = collections.defaultdict(dict)
for s, t in tier.items():
    out[t][s] = {"host": host_of.get(s, "?"), "at": gaps[s][:3]}
json.dump(out, open(os.path.join(HERE, "gaps_tiered.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n写出 gaps_tiered.json ；自证：" + ("OK" if ok else "FAIL"))
sys.exit(0 if ok else 1)
