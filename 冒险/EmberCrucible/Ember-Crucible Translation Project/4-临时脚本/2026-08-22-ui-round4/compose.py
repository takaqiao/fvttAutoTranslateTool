# -*- coding: utf-8 -*-
"""按形态素词典拼出 611 条装备族显示名的初稿。

⚠⚠ **这是初稿，不是成品。** 拼串只负责把「每个词该怎么译」落成一行，
   拼出来的中文顺不顺、地不地道，必须**逐条人工过一遍**（见 review.tsv）。
   本项目已经因为「机器生成的东西没人逐条看」栽过，不再犯。

组合规则（三段）：
  · **序号后缀**（尾部连续的数字/字母，如 `Flag1A` 的 `1A`）——原样贴在最后，空格分隔
  · **姿态后缀**（Backward/Forward/Neutral/Sitting/Resting/Action/Spell/Down）
    ——紧跟中心词、不加空格，与发布中的表同款（`AbyssalBackward`＝深渊后撤）
  · **中心词殿后、修饰倒序前置**：这些 id 是「族名在前、修饰按离中心词由近及远排」，
    正好是英文形容词序的**反向**，所以倒过来读才顺
    （`AxeBattleSteelShoddy` → Shoddy Steel Battle Axe → 粗糙钢战斧）
"""
import io, json, re, sys, importlib.util
sys.stdout.reconfigure(encoding='utf-8')

spec = importlib.util.spec_from_file_location('m', 'morph.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
spec2 = importlib.util.spec_from_file_location('ov', 'overrides.py')
OV = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(OV)
SPLIT = re.compile(r'(?<!^)(?=[A-Z1-9])')
POSE = ["Backward", "Forward", "Neutral", "Sitting", "Resting", "Action", "Spell", "Down"]

def compose(seg, fam=""):
    toks = SPLIT.split(seg)
    # 尾部序号
    tail = []
    while toks and toks[-1] in m.SUFFIX:
        tail.insert(0, toks.pop())
    # 姿态
    pose = ""
    if toks and toks[-1] in POSE:
        pose = m.MODS[toks.pop()]
    if not toks:
        # 整条只有一个姿态词（`Resting`）—— 第一版在这里返回空串，等于**静默丢了一条**。
        return (pose or "".join(tail)), toks
    # 中心词 = **第一个**落在 HEADS 里的段；一个都没有就退回第一段。
    # ⚠ 第一版取的是「最后一个」，实测被附件抢走中心词：`BackpackLeatherBedroll` 的
    #   中心词被判成 Bedroll（铺盖卷）、`TogaClothRightNecklace` 被判成 Necklace、
    #   `SkullBoneLion` 被判成 Lion。这些 id 的排法是**族名在最前**，取第一个才对。
    hi = min((i for i, t in enumerate(toks) if t in m.HEADS), default=None)
    if hi is None:
        # **一个中心词都没有**（`RippedClothBackward` / `ClothSimpleShort` 这类，中心词
        # 由所在图层兜底补）。这一支不能照搬「倒序」：倒序会把材质推到最前，
        # 拼出「布撕裂裤」这种颠三倒四的东西。中文里**材质紧贴中心词**，
        # 所以先整体倒序、再把材质段稳定地挪到最后。
        order = list(reversed(toks))
        mats = [t for t in order if t in m.MATERIAL]
        head = None
        cn_head = ""
        cn_mods = "".join((m.MODS.get(t) or m.HEADS.get(t) or ("〔" + t + "〕"))
                          for t in [t for t in order if t not in m.MATERIAL] + mats)
    else:
        head = toks[hi]
        mods = toks[:hi] + toks[hi + 1:]
        cn_head = m.HEADS.get(head) or m.MODS.get(head) or ("〔" + head + "〕")
        cn_mods = "".join((m.MODS.get(t) or m.HEADS.get(t) or ("〔" + t + "〕")) for t in reversed(mods))
    s = cn_mods + cn_head + pose
    if tail:
        s += " " + "".join(tail)
    return s, toks

def compose_fam(seg, fam):
    """带图层兜底的组合：这一族里省掉中心词的，把图层默认中心词补回去。"""
    toks = SPLIT.split(seg)
    core = [t for t in toks if t not in m.SUFFIX and t not in POSE]
    has_head = any(t in m.HEADS for t in core)
    s, _ = compose(seg)
    skip = OV.TAIL_SKIP_SUFFIX.get(fam, ())
    base = s.split(" ")[0]
    if not has_head and fam in OV.TAIL and not base.endswith(skip):
        # 补在中心词该在的位置：姿态与序号后缀都得让到它后面
        for pose_cn in [m.MODS[p] for p in POSE]:
            if s.endswith(pose_cn):
                return s[:-len(pose_cn)] + OV.TAIL[fam] + pose_cn
        if " " in s:
            head_part, _, tl = s.rpartition(" ")
            return head_part + OV.TAIL[fam] + " " + tl
        return s + OV.TAIL[fam]
    return s

G = json.load(io.open('recon/gap.json', encoding='utf-8'))
seg2fam = {}
for fam, v in G['groups'].items():
    for s in v:
        seg2fam.setdefault(s, fam)
M = json.load(io.open('recon/morphemes.json', encoding='utf-8'))

rows = []
for seg in sorted(M['words']):
    fam = seg2fam.get(seg, '?')
    cn = OV.OVERRIDES.get(seg) or compose_fam(seg, fam)
    rows.append((fam, seg, cn))
rows.sort()
with io.open('review.tsv', 'w', encoding='utf-8', newline='\n') as f:
    for fam, seg, cn in rows:
        f.write(f"{fam}\t{seg}\t{cn}\n")
unresolved = [r for r in rows if '〔' in r[2]]
print(f"拼出 {len(rows)} 条 → review.tsv")
print(f"含未解析形态素的：{len(unresolved)}")
for r in unresolved[:10]:
    print("   ", r)
# 数字那 80 条单独出一份（上屏就是数字，不进表）
io.open('recon/numeric.json', 'w', encoding='utf-8').write(json.dumps(M['numeric'], ensure_ascii=False))
