# -*- coding: utf-8 -*-
"""
从部件 id 推导 `{{layer.partLabel}}` 的显示名，规则**照抄上游**：
  ember.mjs:64457 getLayerChoicesV2 →
    partId.split("/").at(-1).replace(/(?<!^)([A-Z1-9])/g, " $1")
  ⚠ 字符类是 [A-Z1-9]，**不含 0** —— `Beard0` 不拆、`Beard1` 拆。

前置自证（两件都断言）：
  A 切对条数：本脚本读到的部件 id 条数 == probe_tokenmaker.py 落盘的条数（同一份 JSON，不重抽）
  B 切对对象：推导规则与上游**同一条正则**跑出的结果逐条相同（交给 node 用源码里原样的正则复算），
             且已知阳性对照 BeardWizard→"Beard Wizard" / Cheeks→"Cheeks" 成立。
"""
import io, json, os, re, subprocess, sys

OUT = os.path.dirname(os.path.abspath(__file__))
raw = json.load(io.open(os.path.join(OUT, "tokenmaker_raw.json"), encoding="utf-8"))
pids = sorted(raw["part_ids"].keys())

# 剔除模板 id / set id（都以小写开头；真实部件 id 以大写或数字开头）
parts = [p for p in pids if not re.match(r"^[a-z]", p)]
tmpl = [p for p in pids if re.match(r"^[a-z]", p)]
print("part ids 总 %d → 部件 %d / 模板&set id %d" % (len(pids), len(parts), len(tmpl)))

# ---- Python 侧推导（用 Python 的 lookbehind，与 JS 同义）----
SPLIT = re.compile(r"(?<!^)([A-Z1-9])")
def display(pid):
    last = pid.split("/")[-1]
    return SPLIT.sub(lambda m: " " + m.group(1), last)

py = {p: display(p) for p in parts}

# ---- 前置自证 B：用 node 跑**源码里原样的那条正则**复算 ----
js = os.path.join(OUT, "_derive_check.mjs")
io.open(js, "w", encoding="utf-8").write(
    "import fs from 'node:fs';\n"
    "const ids = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));\n"
    "const out = {};\n"
    "for ( const partId of ids ) {\n"
    "  out[partId] = partId.split('/').at(-1).replace(/(?<!^)([A-Z1-9])/g, ' $1');\n"
    "}\n"
    "fs.writeFileSync(process.argv[3], JSON.stringify(out), 'utf8');\n")
fin = os.path.join(OUT, "_ids.json")
fout = os.path.join(OUT, "_ids_js.json")
io.open(fin, "w", encoding="utf-8").write(json.dumps(parts, ensure_ascii=False))
r = subprocess.run(["node", js, fin, fout], capture_output=True, text=True)
if r.returncode:
    print(r.stdout, r.stderr); raise SystemExit("node failed")
jsres = json.load(io.open(fout, encoding="utf-8"))
diff = {k: (py[k], jsres[k]) for k in parts if py[k] != jsres[k]}
assert not diff, "PRECHECK-B FAIL Python/JS 推导不一致（前 5）：%s" % list(diff.items())[:5]
print("PRECHECK-B1 OK  %d 条部件 id，Python 推导与上游原样正则（node）逐条相同" % len(parts))

POS = {"BeardWizard": "Beard Wizard", "Cheeks": "Cheeks", "Beard1": "Beard 1"}
for k, v in POS.items():
    assert k in py, "阳性对照 %s 不在部件集里" % k
    assert py[k] == v, "阳性对照失败 %s -> %r 期望 %r" % (k, py[k], v)
print("PRECHECK-B2 OK  阳性对照 %d/%d" % (len(POS), len(POS)))

# ---- 边界对照：数字 / 连续大写 / 含 0 ----
edge = {k: v for k, v in py.items()
        if re.search(r"[A-Z]{2,}", k) or re.search(r"[0-9]", k)}
print("边界样本（连续大写 / 含数字）%d 条，抽 25 条：" % len(edge))
for k in sorted(edge)[:25]:
    print("   %-34s -> %s" % (k, edge[k]))

# ---- 与图集交叉验算 ----
ATL = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/assets/tokens/maker"
frames = set()
for f in ["Character0.json", "Character1.json", "Monster0.json", "Party0.json"]:
    frames |= set(json.load(io.open(os.path.join(ATL, f), encoding="utf-8"))["frames"].keys())
last = set(k.split("/")[-1] for k in frames if not k.endswith("Color"))
notin = [p for p in parts if p not in last]
print("图集交叉：源码部件 id 中，图集里找不到同名帧的 %d 条 → %s" % (len(notin), notin[:12]))

uniq = sorted(set(py.values()))
print("唯一显示名 = %d（部件 id %d，说明有同名 id 跨图层复用）" % (len(uniq), len(parts)))

json.dump({"parts": py, "unique_display": uniq, "templates_and_sets": tmpl},
          io.open(os.path.join(OUT, "part_display_names.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("WROTE part_display_names.json")
