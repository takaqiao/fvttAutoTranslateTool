# -*- coding: utf-8 -*-
"""Show every untranslated-label occurrence for a given English label, with EN+CN context."""
import argparse, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CJK = re.compile(r'[一-鿿]')
MARK = re.compile(r'(@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\])(\{([^{}]*)\})?')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--en", action="append", default=[])
    ap.add_argument("--path")
    ap.add_argument("--ctx", type=int, default=260)
    ap.add_argument("--repo")
    ap.add_argument("--pack")
    a = ap.parse_args()

    en_pack = None
    if a.repo and a.pack:
        en_pack = json.load(open(os.path.join(a.repo, "compendium", "en", a.pack), encoding="utf-8"))

    def en_leaf(path):
        node = en_pack
        parts = path.split(".")
        # tolerant walk (keys may contain dots)
        cur = node
        rest = path
        while rest and isinstance(cur, (dict, list)):
            if isinstance(cur, dict):
                cands = [k for k in cur if rest == k or rest.startswith(k + ".")]
                if not cands:
                    return None
                k = max(cands, key=len)
                cur = cur[k]
                rest = rest[len(k) + 1:]
            else:
                head, _, rest = rest.partition(".")
                try:
                    cur = cur[int(head)]
                except Exception:
                    return None
        return cur if isinstance(cur, str) else None

    for it in json.load(open(a.plan, encoding="utf-8")):
        if a.path and not re.search(a.path, it["path"]):
            continue
        for l in it["labels"]:
            if a.en and (l["en"] or "") not in a.en:
                continue
            print("PATH", it["batch_path"])
            print("  IDX", l["idx"], "TARGET", l["target"])
            print("  EN-LABEL", repr(l["en"]), " CN-LABEL", repr(l["cn"]))
            c = it["cn_full"]
            m = list(MARK.finditer(c))[l["idx"]]
            s, e = max(0, m.start() - a.ctx), min(len(c), m.end() + a.ctx)
            print("  CN ...", c[s:e].replace("\n", " "), "...")
            if en_pack is not None:
                es = en_leaf(it["path"])
                if es:
                    em = list(MARK.finditer(es))
                    if l["idx"] < len(em):
                        mm = em[l["idx"]]
                        s2, e2 = max(0, mm.start() - a.ctx), min(len(es), mm.end() + a.ctx)
                        print("  EN ...", es[s2:e2].replace("\n", " "), "...")
                    else:
                        print("  EN (index out of range, markup count", len(em), ")")
            print()


if __name__ == "__main__":
    main()
