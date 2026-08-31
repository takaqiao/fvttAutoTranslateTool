# -*- coding: utf-8 -*-
"""Build U2 batches: replace one @UUID label per decision, keeping the whole leaf intact.

Rule: inside the leaf, every `@X[<target ending in KEY>]{OLD}` becomes `@X[<same bracket>]{NEW}`.
The bracket body is copied byte-for-byte; only the {label} changes. Occurrences of the same
key carrying a *different* label are left alone (verified per finding by u2_align.py).
"""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
OUT = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches"
SW = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/uuid_swap.json"
MARK = re.compile(r'(@[A-Za-z]+)\[([^\]]*)\]\{([^{}]*)\}')

# finding index -> (new label, expected replacements in that leaf, suffix appended after the link)
DEC = {
    79: ("书籍", 1, ""), 80: ("库阿尔塔神殿钥匙", 1, ""),
    81: ("尸体战利品（阿克图斯高原）", 1, ""), 82: ("万德伦注视者", 1, ""),
    83: ("十九人", 1, ""), 84: ("陷坑保镖", 1, ""), 85: ("星之盾", 1, ""),
    86: ("明暗野兽", 1, ""), 87: ("明暗野兽", 1, ""),
    88: ("阿梅莉亚·纳克桑", 2, ""), 90: ("阿梅莉亚·纳克桑", 1, ""),
    91: ("崩塌圣所", 1, ""), 94: ("突变学派卡夫托尔·布伦克", 1, ""),
    96: ("尸体战利品（阿克图斯高原）", 1, ""), 97: ("崩塌圣所", 1, ""),
    99: ("攀爬者工具包", 1, ""),
    104: ("奥尔达尼市民", 1, ""), 105: ("深渊恐魔", 1, ""),
    106: ("阿梅莉亚·纳克桑", 1, ""), 107: ("深渊恐魔", 1, ""),
    108: ("点燃的肯瑞斯火炬", 1, ""), 111: ("奥尔达尼市民", 1, ""),
    112: ("采石场奔道", 1, ""), 115: ("符文标记绶带", 1, ""),
    116: ("微粒追逐战", 1, ""), 122: ("大破裂", 1, ""),
    124: ("厨师工具", 1, ""), 125: ("扑克牌", 1, ""), 131: ("开锁工具", 1, ""),
    132: ("碎片诸神", 1, ""), 133: ("阿克图里安呼吸器", 2, ""),
    135: ("遭憎魔法", 1, ""), 139: ("奥拉", 1, ""),
    141: ("识别施法", 1, "天赋"), 142: ("碎片诸神", 1, ""),
    147: ("埃维索的便笺", 1, ""), 150: ("奥拉", 1, ""),
}
# 89 duplicates 88 (same leaf, both occurrences), 134 duplicates 133, so they are folded in.

def resolve(base, p):
    segs = p.split("."); cur = base; i = 0
    while i < len(segs):
        if isinstance(cur, list):
            cur = cur[int(segs[i])]; i += 1; continue
        for j in range(len(segs), i, -1):
            k = ".".join(segs[i:j])
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]; i = j; break
        else:
            return None
    return cur

def main():
    F = json.load(open(SW, encoding="utf-8"))["findings"]
    cncache = {}
    batches = {}
    log = []
    for i in sorted(DEC):
        x = F[i]
        new, expect, suffix = DEC[i]
        repo, pack, path, key, old = x["repo"], x["pack"], x["path"], x["key"], x["cn_label"]
        ck = (repo, pack)
        if ck not in cncache:
            cncache[ck] = json.load(open(os.path.join(ROOT, repo, "compendium", "cn", pack), encoding="utf-8"))
        bkey = (repo, pack)
        cur = batches.setdefault(bkey, {})
        bp = path[len("entries."):]
        src = cur.get(bp)
        if src is None:
            src = resolve(cncache[ck], path)
            assert isinstance(src, str), (i, path)
        n = [0]
        def sub(m):
            tgt = m.group(2).split()[0].split("#")[0] if m.group(2) else ""
            if tgt.split(".")[-1] == key and m.group(3) == old:
                n[0] += 1
                return f"{m.group(1)}[{m.group(2)}]{{{new}}}" + suffix
            return m.group(0)
        out = MARK.sub(sub, src)
        assert n[0] == expect, f"finding {i}: replaced {n[0]}, expected {expect}"
        cur[bp] = out
        log.append((i, repo, pack, bp, old, new, n[0]))
    os.makedirs(OUT, exist_ok=True)
    for (repo, pack), d in batches.items():
        tag = "ember" if repo.startswith("1-") else "crucible"
        fn = os.path.join(OUT, f"U2__{tag}__{pack}")
        json.dump(d, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("wrote", fn, len(d), "leaves")
    for r in log:
        print(r[0], r[3][:70], "|", r[4], "->", r[5], f"x{r[6]}")
main()
