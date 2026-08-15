# -*- coding: utf-8 -*-
"""Drive qa/apply_translations.py over every merged batch.

Default is dry. Pass --write to actually land them.
Every batch rewrites existing Chinese, so --force is always on.
"""
import argparse, json, os, re, subprocess, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
W = os.path.join(P, "4-临时脚本", "2026-08-12-fix")
APPLY = os.path.join(P, "3-常用脚本", "qa", "apply_translations.py")

NUM = re.compile(r"REJECTED\s+(\S+)\s+(\d+)")
APPLIED = re.compile(r"applied\s+(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=os.path.join(W, "merged"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    files = sorted(f for f in os.listdir(a.merged)
                   if f.endswith(".json") and not f.startswith("_"))
    tot_applied = tot_rej = 0
    bad = []
    for fn in files:
        repo, pack = fn.split("__", 1)
        cmd = [sys.executable, APPLY, "--repo", os.path.join(P, repo),
               "--pack", pack, "--batch", os.path.join(a.merged, fn), "--force"]
        if not a.write:
            cmd.append("--dry")
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
        out = (r.stdout or "") + (r.stderr or "")
        applied = int(APPLIED.search(out).group(1)) if APPLIED.search(out) else -1
        rej = {k: int(v) for k, v in NUM.findall(out)}
        nrej = sum(rej.values())
        tot_applied += max(applied, 0)
        tot_rej += nrej
        flag = "OK " if nrej == 0 and r.returncode == 0 else "!! "
        print(f"{flag}{repo:<22}{pack:<36} applied={applied:<5} rejected={nrej} {rej if nrej else ''}")
        if nrej or r.returncode != 0:
            bad.append((fn, out[-1500:]))

    print("-" * 96)
    print(f"{'WROTE' if a.write else 'DRY'}: applied {tot_applied} | rejected {tot_rej}")
    for fn, out in bad:
        print("\n### " + fn)
        print(out)
    return 1 if tot_rej else 0


if __name__ == "__main__":
    sys.exit(main())
