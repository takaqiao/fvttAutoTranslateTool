# -*- coding: utf-8 -*-
"""
tb_shape_mismatch.py -- the DATA-level manifestation of the same class.

If some tool in the pipeline wrote a translation without honouring the
polymorphic shape of the source field, the CN file will disagree with the EN
file about whether a given leaf is a string or an object. That is exactly the
"no type discriminator" defect, only frozen into data instead of code.

Walks compendium/en/<pack>.json and compendium/cn/<pack>.json in lockstep and
reports every leaf path where:
    type(EN leaf) != type(CN leaf)     (str vs dict vs list)

False positives:
  * A CN file legitimately omits a leaf (untranslated) -> skipped, not a
    mismatch.
  * `structured`/`nameCollection` sub-objects keyed by id can legitimately
    differ in KEY SET; only shape at a shared path is compared.

Read-only.
"""
import json
import os
from collections import Counter

BASE = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]


def shape(v):
    if isinstance(v, str):
        return "str"
    if isinstance(v, dict):
        return "dict"
    if isinstance(v, list):
        return "list"
    return type(v).__name__


def compare(en, cn, path, out):
    if isinstance(en, dict) and isinstance(cn, dict):
        for k in en:
            if k in cn:
                compare(en[k], cn[k], path + [k], out)
        return
    if isinstance(en, list) and isinstance(cn, list):
        for i in range(min(len(en), len(cn))):
            compare(en[i], cn[i], path + ["[%d]" % i], out)
        return
    if shape(en) != shape(cn):
        out.append((".".join(path), shape(en), shape(cn)))


def main():
    total = Counter()
    hits = []
    packs = 0
    for repo in REPOS:
        en_dir = os.path.join(BASE, repo, "compendium", "en")
        cn_dir = os.path.join(BASE, repo, "compendium", "cn")
        if not os.path.isdir(en_dir):
            continue
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith(".json"):
                continue
            cn_path = os.path.join(cn_dir, fn)
            if not os.path.exists(cn_path):
                continue
            packs += 1
            en = json.load(open(os.path.join(en_dir, fn), encoding="utf-8"))
            cn = json.load(open(cn_path, encoding="utf-8"))
            out = []
            compare(en.get("entries", {}), cn.get("entries", {}), [], out)
            for p, se, sc in out:
                leaf = p.split(".")[-1]
                total["%s:%s->%s" % (leaf, se, sc)] += 1
                hits.append({"pack": "%s/%s" % (repo, fn), "path": p, "en": se, "cn": sc})

    print("packs compared: %d" % packs)
    print("shape mismatches: %d" % len(hits))
    for k, v in total.most_common(40):
        print("   %-44s %d" % (k, v))
    for h in hits[:25]:
        print("   %s :: %s  EN=%s CN=%s" % (h["pack"], h["path"][:150], h["en"], h["cn"]))

    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(hits, open(os.path.join(here, "tb_shape_mismatch.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
