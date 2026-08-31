# -*- coding: utf-8 -*-
"""U3: turn per-finding label verdicts into whole-leaf batch values.

Each finding names one `@UUID[...]{label}` occurrence inside a leaf, identified
by its ordinal `i` among the leaf's links.  The tokenizer here is a copy of
scan_uuid_swap's `links_in`, so `i` addresses the same occurrence.  Only the
`{label}` of that occurrence changes; every other byte of the leaf (targets,
inline commands, HTML classes) is carried over untouched.
"""
import json, os, re, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
SC = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3"
OUT = SC + "/batches"
LINK = re.compile(r'@([A-Za-z]+)\[([^\]\n]*)\]\{([^}\n]*)\}')
CMD = re.compile(r'\[\[([^\]\n]*)\]\]\{([^}\n]*)\}')

# finding index -> new CN label.  Everything absent is a deliberate 不改.
NEW = {
    151: "深渊恐魔", 152: "十九人", 153: "萨克茨", 155: "十九人", 158: "萨克茨",
    159: "十九人", 160: "瑟洛克", 161: "碎片诸神", 162: "奥拉", 163: "深渊恐魔",
    164: "卡西娅", 165: "卡西娅", 166: "萨克茨", 167: "卡西娅",
    171: "马尔石", 172: "画廊", 173: "画廊",
    175: "华服", 176: "旅行服", 177: "与卡夫托尔战斗",
    179: "一再坠落", 180: "辉耀", 181: "回避侦测术", 182: "制图工具",
    183: "网", 184: "突进强攻", 185: "总管",
    189: "奥拉", 190: "星之盾", 194: "书籍", 195: "库阿尔塔神殿钥匙",
    196: "尸体战利品（阿克图斯高原）", 197: "明暗野兽", 198: "十九人",
    199: "水妖精", 200: "陷坑保镖", 201: "星之盾",
    204: "阿梅莉亚·纳克桑", 205: "阿梅莉亚·纳克桑", 206: "物件定位术",
    207: "阿梅莉亚·纳克桑", 208: "崩塌圣所", 211: "突变学派卡夫托尔·布伦克",
    214: "尸体战利品（阿克图斯高原）", 215: "杰克罗卡的账本", 216: "拉斯特·索恩",
    217: "崩塌圣所", 219: "攀爬者工具包",
    222: "水妖精", 223: "水访客", 224: "霍伦多尔", 225: "角色创建",
}
REPO_TAG = {"1-Ember汉化插件": "ember", "2-Crucible汉化插件": "crucible"}


def links_in(s):
    out = []
    for m in LINK.finditer(s):
        out.append({'at': m.start(), 'end': m.end(), 'lab_s': m.start(3),
                    'lab_e': m.end(3), 'label': m.group(3)})
    for m in CMD.finditer(s):
        out.append({'at': m.start(), 'end': m.end(), 'lab_s': m.start(2),
                    'lab_e': m.end(2), 'label': m.group(2)})
    out.sort(key=lambda d: d['at'])
    return out


def flat(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            flat(v, path + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            flat(v, path + [str(i)], out)
    elif isinstance(node, str):
        out['.'.join(path)] = node


d = json.load(open(SC + "/uuid_swap.json", encoding="utf-8"))
packs, edits = {}, defaultdict(lambda: defaultdict(list))
for idx, lab in NEW.items():
    f = d["findings"][idx]
    edits[(f["repo"], f["pack"])][f["path"]].append((f["i"], lab, idx, f["cn_label"]))

os.makedirs(OUT, exist_ok=True)
summary = []
for (repo, pack), leaves in sorted(edits.items()):
    doc = json.load(open(os.path.join(P, repo, "compendium", "cn", pack), encoding="utf-8"))
    fl = {}
    flat(doc, [], fl)
    batch = {}
    for path, items in sorted(leaves.items()):
        s = fl[path]
        ls = links_in(s)
        new = s
        for i, lab, idx, old in sorted(items, reverse=True):   # right-to-left
            L = ls[i]
            assert L['label'] == old, (idx, L['label'], old)
            new = new[:L['lab_s']] + lab + new[L['lab_e']:]
            summary.append((idx, repo, pack, path, old, lab))
        assert new != s
        bp = path[len('entries.'):] if path.startswith('entries.') else path
        batch[bp] = new
    fn = f"{OUT}/U3__{REPO_TAG[repo]}__{pack[:-5]}.json"
    json.dump(batch, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"-> {fn}   leaves={len(batch)}")

for r in sorted(summary):
    print(f"  {r[0]:3} {r[4]!r} -> {r[5]!r}   {r[3][:70]}")
print("edits:", len(summary))
