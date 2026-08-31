# -*- coding: utf-8 -*-
"""G3: build the fix batches for upstream-renamed proper nouns.

Each edit is a surgical replacement inside one `@UUID[...id...]{label}` (or one
name leaf); everything else in the leaf stays byte-identical.
"""
import json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
OUT = (r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/"
       r"e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches")

# target id -> correct CN label (= the CN part of the target's current name)
LABEL = {
    "kY9aP3qlK3nJDWhX": "钱币袋",        # Bag of Bronze Coins -> Bag of Coins
    "tFyeCSvVP5TJ5Bib": "花瓣石",        # Gem-Encased Petals  -> Petalzon
    "TKinFp8DoVau7DPY": "虹彩蛋白石",     # Signaran Opal       -> Opalix
    "y6lXtFPEDoPA71c6": "尖晶石",        # Big Spinel          -> Spinel
    "ZrQEbleUzvMVCfSu": "辉光石",        # Lambent Gem         -> Lamberite
    "sqMiu46682JrLQki": "特拉奇尼酒馆",   # Terracini's Restaurant -> Terracini's Tavern
    "A0x1R29zmbLYX0QT": "特拉奇尼阳台",   # Terracini's Too     -> Terracini Balcony
}

RX_UUID = re.compile(r"(@UUID\[([^\]]*)\])\{([^}]*)\}")


def relabel(s):
    n = [0]

    def sub(m):
        tid = m.group(2).split(".")[-1]
        want = LABEL.get(tid)
        if want and m.group(3) != want:
            n[0] += 1
            return m.group(1) + "{" + want + "}"
        return m.group(0)

    return RX_UUID.sub(sub, s), n[0]


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str):
        out.append((".".join(path), node))


def main():
    os.makedirs(OUT, exist_ok=True)
    plan = [("ember", "1-Ember汉化插件",
             ["ember.adventure.json", "ember.crucible-adventure.json"])]
    for repokey, repodir, packs in plan:
        for pack in packs:
            cn = json.load(open(os.path.join(P, repodir, "compendium", "cn", pack),
                                encoding="utf-8"))
            rows = []
            walk(cn.get("entries", {}), [], rows)
            batch, log = {}, []
            for p, s in rows:
                new, n = relabel(s)
                # the duplicated bilingual tail (detector A)
                if p.endswith("Altar of  Aura.name") and "Altar of Aura Altar of" in new:
                    new = "奥拉祭坛 Altar of Aura"
                    n += 1
                if n and new != s:
                    batch[p] = new
                    log.append((p, n))
            fn = os.path.join(OUT, f"G3__{repokey}__{pack}")
            json.dump(batch, open(fn, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            print(f"{fn}  leaves={len(batch)}")
            for p, n in log:
                print(f"    {n}x  {p}")

    # crucible: the translated HTML attribute name  目标="_blank"
    pack = "crucible.rules.json"
    cn = json.load(open(os.path.join(P, "2-Crucible汉化插件", "compendium", "cn", pack),
                        encoding="utf-8"))
    rows = []
    walk(cn.get("entries", {}), [], rows)
    batch = {}
    for p, s in rows:
        if '目标="' in s:
            batch[p] = s.replace('目标="', 'target="')
    fn = os.path.join(OUT, f"G3__crucible__{pack}")
    json.dump(batch, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{fn}  leaves={len(batch)}")
    for p in batch:
        print(f"    {p}")


if __name__ == "__main__":
    main()
