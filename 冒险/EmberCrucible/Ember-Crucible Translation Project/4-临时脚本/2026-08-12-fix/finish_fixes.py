# -*- coding: utf-8 -*-
"""Three residual fixes the batch pipeline could not express.

1. uuid-swap leftovers: scan_uuid_swap still reports 12 BROKEN (6 unique x the
   twin packs). Each carries a `suggested` label with strong majority support,
   so the repair is mechanical: inside that one leaf, retarget that one label.
2. terrain desync: mappings.mjs no longer declares `terrain` translatable, and
   the 104 Chinese values were purged -- but compendium/en still carries the
   field, so fill_missing now reports 104 phantom gaps. Drop it from the English
   baseline too, which is what re-running extract_en.mjs would produce.
3. one real dead key in crucible.playtest: `items.吞噬思维 Devour Thoughts.…`
   keys the item by its TRANSLATION, so Babele can never match it. Verified the
   correctly-keyed `items.Devour Thoughts` sibling already holds a complete and
   better translation, so deletion loses nothing.

   NOT touched: ember's 8 dead keys under `_legacyActions`. The `_` prefix is a
   deliberate parking convention (validate_translations.py:97 skips it,
   migrate_cn_schema.mjs parks content there "so a later pass can rescue it by
   hand"). prune_dead --write would delete them; that is why it is not run.
"""
import argparse, json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
W = os.path.join(P, "4-临时脚本", "2026-08-12-fix")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def dump(p, o):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=2)
        f.write("\n")


def get_at(node, parts):
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p)
        elif isinstance(node, list):
            try:
                node = node[int(p)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def split_path(root, path):
    naive = path.split(".")
    if get_at(root, naive) is not None:
        return naive
    parts, node, rest = [], root, path
    while rest:
        if isinstance(node, dict):
            cands = [k for k in node if rest == k or rest.startswith(k + ".")]
            if cands:
                k = max(cands, key=len)
                parts.append(k)
                node = node[k]
                rest = rest[len(k) + 1:]
                continue
        head, _, rest = rest.partition(".")
        parts.append(head)
        node = get_at(node, [head])
    return parts


def set_at(root, parts, value):
    node = root
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node[p]
    if isinstance(node, list):
        node[int(parts[-1])] = value
    else:
        node[parts[-1]] = value


def fix_uuid(write):
    rep = load(os.path.join(W, "reports", "uuid_swap_after.json"))
    broken = [f for f in rep["findings"] if f.get("verdict") == "BROKEN"]
    by_pack = {}
    for f in broken:
        by_pack.setdefault((f["repo"], f["pack"]), []).append(f)

    total = 0
    for (repo, pack), fs in sorted(by_pack.items()):
        cn_path = os.path.join(P, repo, "compendium", "cn", pack)
        cn = load(cn_path)
        root = cn["entries"]
        n = 0
        for f in fs:
            if f.get("suggested_is_english") or not f.get("suggested"):
                print(f"  SKIP (no Chinese suggestion) {f['key']} {f['en_label']}")
                continue
            parts = split_path(root, f["path"][len("entries."):])
            cur = get_at(root, parts)
            if not isinstance(cur, str):
                print(f"  !! path not found: {f['path']}")
                continue
            # retarget ONLY this link: match the bracketed target then its label
            pat = re.compile(r"(@UUID\[[^\]]*" + re.escape(f["key"]) + r"[^\]]*\]\{)"
                             + re.escape(f["cn_label"]) + r"(\})")
            new, k = pat.subn(r"\1" + f["suggested"].replace("\\", "\\\\") + r"\2", cur)
            if k == 0:
                print(f"  !! label not matched: {f['key']} {f['cn_label']!r} in {f['path']}")
                continue
            print(f"  {repo}/{pack}\n     {f['en_label']}: {f['cn_label']} -> {f['suggested']}  ({k}x)")
            set_at(root, parts, new)
            n += k
        if write and n:
            dump(cn_path, cn)
        total += n
    print(f"uuid labels retargeted: {total}{'' if write else '  (dry)'}")


def fix_terrain_en(write):
    n = 0
    for pack in ("ember.crucible-adventure.json", "ember.adventure.json"):
        p = os.path.join(P, "1-Ember汉化插件", "compendium", "en", pack)
        d = load(p)
        k = 0

        def strip(node):
            nonlocal k
            if isinstance(node, dict):
                if isinstance(node.get("terrain"), str):
                    del node["terrain"]
                    k += 1
                for v in node.values():
                    strip(v)
            elif isinstance(node, list):
                for v in node:
                    strip(v)

        strip(d.get("entries", {}))
        print(f"  {pack}: removed {k} terrain keys from English baseline")
        if write and k:
            dump(p, d)
        n += k
    print(f"english baseline terrain keys removed: {n}{'' if write else '  (dry)'}")


def fix_dead_key(write):
    p = os.path.join(P, "2-Crucible汉化插件", "compendium", "cn", "crucible.playtest.json")
    d = load(p)
    items = d["entries"]["Playtest 1 - The Ring of Valor"]["actors"]["Harbinger of Madness"]["items"]
    bad, good = "吞噬思维 Devour Thoughts", "Devour Thoughts"
    if bad not in items:
        print("  dead key already gone")
        return
    live = get_at(items, [good, "actions", "devourThoughts", "effects", "0", "name"])
    if not isinstance(live, str) or not live.strip():
        print("  !! live sibling has no value -- REFUSING to delete")
        return
    print(f"  deleting items[{bad!r}] (holds {get_at(items,[bad,'actions','devourThoughts','effects','0','name'])!r});"
          f" live sibling keeps {live!r}")
    if write:
        del items[bad]
        dump(p, d)
    print(f"dead key removed: 1{'' if write else '  (dry)'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    print("== 1. uuid-swap leftovers ==");  fix_uuid(a.write)
    print("\n== 2. terrain english baseline ==");  fix_terrain_en(a.write)
    print("\n== 3. dead key ==");  fix_dead_key(a.write)
