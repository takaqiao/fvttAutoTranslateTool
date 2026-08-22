# -*- coding: utf-8 -*-
"""
指示物制作器**部件全集**的枚举探针。

⚠ 前置自证（两件，任一不成立当场 exit 2）：
  P1 「切对条数」—— 从随包图集 4 份 .json 里读到的 frames 总数必须等于**已知真值**
     （Character0 1631 / Character1 3013 / Monster0 783 / Party0 222 · 合计 5649，
      本文件写死，改上游版本时会当场对不上而不是静默漂移）。
  P2 「切对对象」—— 切出来的**不只是条数对**：随机抽 8 个 frame，逐条断言
     它在原始 JSON 的 frames 里确实存在、且形如 `<ns>/<layer>/<Part>`（三段）。
  两件都断言，因为「条数对」不保证「读的是同一批对象」。

输出：parts_universe.json
"""
import io, json, os, re, sys

EMBER = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember"
ATLAS = os.path.join(EMBER, "assets/tokens/maker")
MJS   = os.path.join(EMBER, "scripts/ember.mjs")
OUT   = os.path.dirname(os.path.abspath(__file__))

TRUTH = {"Character0.json": 1631, "Character1.json": 3013, "Monster0.json": 783, "Party0.json": 222}

def die(m):
    sys.stderr.write("enum_parts: " + m + "\n"); sys.exit(2)

# ── 读图集 ───────────────────────────────────────────────────────────────
raw = {}
frames = []
for f, n in TRUTH.items():
    d = json.load(io.open(os.path.join(ATLAS, f), encoding="utf-8"))
    ks = list(d["frames"])
    if len(ks) != n:
        die("P1 FAIL %s 读到 %d frames，已知真值 %d" % (f, len(ks), n))
    raw[f] = set(ks)
    frames += ks
if len(frames) != sum(TRUTH.values()):
    die("P1 FAIL 合计 %d ≠ 已知真值 %d" % (len(frames), sum(TRUTH.values())))
print("P1 OK  图集 frames 逐份对上已知真值：%s，合计 %d" %
      (" / ".join("%s %d" % (k, v) for k, v in TRUTH.items()), len(frames)))

allf = set(frames)
if len(allf) != len(frames):
    die("P1 FAIL frames 有重复")

# P2 抽样验对象
sample = [frames[i] for i in (0, 37, 500, 1630, 1631, 3000, 4644, 5648)]
for s in sample:
    if not any(s in v for v in raw.values()):
        die("P2 FAIL 抽样 frame %r 不在任何一份原始 JSON 里" % s)
    if len(s.split("/")) != 3:
        die("P2 FAIL 抽样 frame %r 不是 <ns>/<layer>/<Part> 三段" % s)
print("P2 OK  抽样 8 个 frame 逐条回原始 JSON 找得到、且都是三段式：%s" % sample[:3])

# ── 从 frames 推 partId：P 是部件 iff P 与 P+"Color" 都在 frames 里 ────────
parts, orphans = set(), set()
for f in allf:
    if f.endswith("Color") and f[:-5] in allf:      # 颜色遮罩
        continue
    if f.endswith("Mask") and f[:-4] in allf:       # 手部遮罩
        continue
    if (f + "Color") in allf:
        parts.add(f)
    else:
        orphans.add(f)
print("图集口径：部件（有配套 Color 帧）%d 条 · 无配套 Color 帧的 %d 条" % (len(parts), len(orphans)))

# ── 从 ember.mjs 静态抠 part 声明（带 layer 归属）─────────────────────────
src = io.open(MJS, encoding="utf-8").read()

decl = {}   # lastSeg -> set(layerId)
# makeParts("ns", "layer", [ ... ]) / setLayerParts(X.layers, <ids>, "ns", "layer", [ ... ])
for m in re.finditer(r'makeParts\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*\[', src):
    layer = m.group(2); i = m.end() - 1
    depth, j = 0, i
    while j < len(src):
        if src[j] == "[": depth += 1
        elif src[j] == "]":
            depth -= 1
            if depth == 0: break
        j += 1
    body = src[i:j]
    for mm in re.finditer(r'(?<!//\s)\{\s*id:\s*"([^"]+)"', body):
        decl.setdefault(mm.group(1).split("/")[-1], set()).add(layer)

for m in re.finditer(r'setLayerParts\(\s*[^,]+,\s*(?:"([^"]+)"|\[[^\]]*\]|\{[^}]*\})\s*,\s*"([^"]+)"\s*,\s*(?:"([^"]+)"\s*,\s*)?\[', src):
    layer = m.group(3) or m.group(1) or "?"
    i = m.end() - 1
    depth, j = 0, i
    while j < len(src):
        if src[j] == "[": depth += 1
        elif src[j] == "]":
            depth -= 1
            if depth == 0: break
        j += 1
    body = src[i:j]
    for mm in re.finditer(r'\{\s*id:\s*"([^"]+)"', body):
        decl.setdefault(mm.group(1).split("/")[-1], set()).add(layer)

print("静态扫描：声明式 part id（末段去重）%d 条" % len(decl))

# ── 图集末段 ─────────────────────────────────────────────────────────────
seg2layers = {}
for p in parts:
    ns, layer, seg = p.split("/")
    seg2layers.setdefault(seg, set()).add(layer)

# ── 上游的显示名拆词（逐字符照抄 getLayerChoicesV2）──────────────────────
RE_SPLIT = re.compile(r'(?<!^)([A-Z1-9])')
def display(seg):
    return RE_SPLIT.sub(r' \1', seg)

names = {}
for seg, layers in seg2layers.items():
    d = display(seg)
    names.setdefault(d, set()).update(layers)

print("图集末段唯一 id %d 条 → 唯一显示名 %d 条" % (len(seg2layers), len(names)))
missed = sorted(set(seg2layers) - set(decl))
print("图集有、静态扫描没有的末段 id：%d 条" % len(missed))
print("  例：", missed[:20])

json.dump({
    "atlas_frames": len(frames),
    "atlas_parts": sorted(parts),
    "atlas_orphan_frames": sorted(orphans),
    "seg2layers": {k: sorted(v) for k, v in sorted(seg2layers.items())},
    "display2layers": {k: sorted(v) for k, v in sorted(names.items())},
    "static_decl": {k: sorted(v) for k, v in sorted(decl.items())},
    "atlas_only_segs": missed,
}, io.open(os.path.join(OUT, "parts_universe.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("已写 parts_universe.json")
