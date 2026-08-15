# -*- coding: utf-8 -*-
"""G8 helpers: load Deities pairs, split into blocks, run targeted scans."""
import json, re, sys, os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "G8.deities.adv.json")

BLOCK_RE = re.compile(
    r"<(?:p|h[1-6]|li|dt|dd|td|th|blockquote|figcaption)\b[^>]*>.*?</(?:p|h[1-6]|li|dt|dd|td|th|blockquote|figcaption)>",
    re.S | re.I)


def load():
    rows = json.load(open(SRC, encoding="utf-8"))["rows"]
    out = []
    for r in rows:
        k = r["batch_path"].split("journals.Deities.", 1)[1]
        r["k"] = k
        out.append(r)
    return out


def blocks(html):
    if not html:
        return []
    return BLOCK_RE.findall(html)


def strip(t):
    t = re.sub(r"<[^>]+>", "", t)
    t = t.replace("&nbsp;", " ").replace("&amp;", "&").replace("&mdash;", "—")
    t = t.replace("&rsquo;", "'").replace("&lsquo;", "'").replace("&ldquo;", '"').replace("&rdquo;", '"')
    return re.sub(r"\s+", " ", t).strip()


def pagename(k):
    m = re.match(r"pages\.(.+?)\.(\w+)$", k)
    return (m.group(1), m.group(2)) if m else (None, k)


if __name__ == "__main__":
    cmd = sys.argv[1]
    rows = load()
    if cmd == "page":
        want = sys.argv[2]
        fields = sys.argv[3].split(",") if len(sys.argv) > 3 else None
        for r in rows:
            p, f = pagename(r["k"])
            if p != want:
                continue
            if fields and f not in fields:
                continue
            eb, cb = blocks(r["en"]), blocks(r["cn"] or "")
            print(f"\n######## {r['k']}   enBlocks={len(eb)} cnBlocks={len(cb)}")
            if not eb:
                print("EN:", r["en"])
                print("CN:", r["cn"])
                continue
            for i in range(max(len(eb), len(cb))):
                e = eb[i] if i < len(eb) else "<<<MISSING>>>"
                c = cb[i] if i < len(cb) else "<<<MISSING>>>"
                print(f"--{i}--E| {e}")
                print(f"--{i}--C| {c}")
