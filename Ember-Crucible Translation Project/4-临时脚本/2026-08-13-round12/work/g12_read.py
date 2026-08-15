# -*- coding: utf-8 -*-
"""Block-aligned EN/CN reader for G12 (K5)."""
import json, re, sys, io, argparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BOUND = re.compile(r"(?<=</p>)|(?<=</li>)|(?<=</h1>)|(?<=</h2>)|(?<=</h3>)|(?<=</h4>)|(?<=</h5>)|(?<=</h6>)|(?<=</tr>)|(?<=</dd>)|(?<=</dt>)|(?<=</blockquote>)|(?<=</table>)|(?<=</section>)|(?<=</div>)|(?<=</ul>)|(?<=</ol>)|(?<=</dl>)|(?<=<br />)|(?<=<br>)")

def blocks(s):
    if not s:
        return []
    parts = [p for p in BOUND.split(s) if p and p.strip()]
    return parts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--from", dest="a", type=int, default=0)
    ap.add_argument("--to", dest="b", type=int, default=10**9)
    ap.add_argument("--grep-path")
    a = ap.parse_args()
    rows = json.load(open(a.src, encoding="utf-8"))["rows"]
    if a.grep_path:
        rx = re.compile(a.grep_path)
        rows = [r for r in rows if rx.search(r["path"])]
    for i, r in enumerate(rows):
        if i < a.a or i >= a.b:
            continue
        en = r["en"]; cn = r["cn"] or ""
        eb = blocks(en); cb = blocks(cn)
        print("=" * 100)
        print(f"### [{i}] {r['batch_path']}   enBlocks={len(eb)} cnBlocks={len(cb)}"
              + ("   *** BLOCK COUNT MISMATCH ***" if len(eb) != len(cb) else ""))
        if len(eb) == len(cb):
            for j, (e, c) in enumerate(zip(eb, cb)):
                print(f"--{j}--")
                print("E| " + e.strip())
                print("C| " + c.strip())
        else:
            print("~~~EN~~~"); print(en)
            print("~~~CN~~~"); print(cn)

main()
