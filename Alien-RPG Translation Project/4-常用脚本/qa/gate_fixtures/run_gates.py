# -*- coding: utf-8 -*-
import os, re, subprocess, sys, json
sys.stdout.reconfigure(encoding='utf-8')
PROJ = r"C:/Users/Taka/Desktop/fvtt/Alien-RPG Translation Project"
QA   = os.path.join(PROJ, "4-常用脚本", "qa")
TMP  = os.environ.get("ALIEN_GATE_FIXTURES") or os.path.join(
    os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp", "alien-gate-fixtures")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

def run(script, root):
    cmd = [sys.executable, os.path.join(QA, script),
           "--repo", os.path.join(root, "r1"),
           "--repo", os.path.join(root, "r2"),
           "--repo", os.path.join(root, "r3"),
           "--lang", os.path.join(root, "r1", "lang", "cn.json"),
           "--lang-en", os.path.join(root, "r1", "lang", "en.json"),
           "--show", "3"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=ENV)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"checked=(\d+)\s+violations=(\d+)", out)
    ck, vi = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
    vd = re.search(r"按判据：(.*)", out)
    return p.returncode, ck, vi, (vd.group(1).strip() if vd else ""), out

def _k(s):
    m = re.search(r"\d+", s)
    return (s[0], int(m.group()) if m else 0)
targets = ["GOOD"] + sorted([d for d in os.listdir(TMP) if d != "GOOD" and os.path.isdir(os.path.join(TMP, d))], key=_k)
only = sys.argv[1] if len(sys.argv) > 1 else None
print("%-32s | %-34s | %s" % ("fixture", "scan_name_lookup_traps.py", "scan_crit_lockstep.py"))
print("-" * 118)
rows = []
for t in targets:
    if only and only not in t:
        continue
    root = os.path.join(TMP, t)
    a = run("scan_name_lookup_traps.py", root)
    b = run("scan_crit_lockstep.py", root)
    rows.append((t, a, b))
    print("%-32s | exit=%d checked=%-5d viol=%-4d %-8s | exit=%d checked=%-5d viol=%-4d %s"
          % (t, a[0], a[1], a[2], a[3][:8], b[0], b[1], b[2], b[3][:34]))
json.dump([{"fixture": t, "traps": {"exit": a[0], "checked": a[1], "viol": a[2], "verdicts": a[3]},
            "crit": {"exit": b[0], "checked": b[1], "viol": b[2], "verdicts": b[3]}} for t, a, b in rows],
          open(os.path.join(TMP, "_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
if only:
    for t, a, b in rows:
        print("\n===== %s :: traps =====\n%s" % (t, a[4]))
        print("\n===== %s :: crit =====\n%s" % (t, b[4]))
