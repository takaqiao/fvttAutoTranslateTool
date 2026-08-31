# -*- coding: utf-8 -*-
"""round24 ③ 探针：把候选串在「完整上游语料」里逐一核实。
落盘再跑（不是 heredoc），避免反斜杠被吃掉。"""
import json, os, re, sys

BASE_EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
SRC = {
    "ember.mjs": os.path.join(BASE_EMBER, "scripts", "ember.mjs"),
    "crucible-async.mjs": os.path.join(BASE_EMBER, "scripts", "crucible-async.mjs"),
    "dnd5e-async.mjs": os.path.join(BASE_EMBER, "scripts", "dnd5e-async.mjs"),
    "ember/lang/en.json": os.path.join(BASE_EMBER, "lang", "en.json"),
    "crucible-compiled.mjs": r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\crucible-compiled.mjs",
}

TEXT = {}
LINES = {}
for k, p in SRC.items():
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        t = f.read()
    TEXT[k] = t
    LINES[k] = t.splitlines()

# 只有两个 .mjs 是「面板判据」实际抓取的语料
PANEL_CORPUS = ["ember.mjs", "crucible-compiled.mjs"]

TARGETS = [
    "The Abyss", "Abyss", "Akon", "Aura", "Cora", "Heart of Ember", "Heart",
    "Luxarum", "Mayis", "Orbis", "Primordis", "Ragen", "Signara", "Ember",
    "Make Active", "blurStrength", "Blur Strength",
    "Arcturian", "Human", "Kavir", "Keth", "Kivahr", "Lumek", "Oaken", "Wirrun",
    "Bejak", "Kessian", "Ordani", "Waerd",
]

out = {}
for t in TARGETS:
    rec = {"panel_corpus_total": 0, "per_file": {}}
    for k in SRC:
        n = TEXT[k].count(t)
        rec["per_file"][k] = n
        if k in PANEL_CORPUS:
            rec["panel_corpus_total"] += n
    out[t] = rec

print(json.dumps(out, ensure_ascii=False, indent=1))
