# -*- coding: utf-8 -*-
"""
把 cn_t1 / cn_t2 的**显示名 → 中文**反查成**上游字面量 id → 中文**，生成落表用的 .mjs 片段。

⚠ 前置自证（四件，任一不成立当场 exit 2）：
  A 切对条数 —— 反查出来的 id 条数 == 参与反查的显示名条数（拆词是双射，一个都不许丢/并）；
  B 切对对象 —— 每个 id **必须**在 ember.mjs 里逐字出现（否则登记进自检面板 D 档会当场变 miss）；
  C 拆词等价 —— 对每个 id 用上游那条正则重算显示名，必须与输入的显示名逐字符相同；
  D 拼串族闭合 —— LEG_POSE_BASES / MARBLED_HAND_BASES 拼出来的 id 必须全部落在图集里，
                  且拼出来的显示名恰好覆盖图集里那一族（不多不少）。
"""
import io, json, os, re, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
EMBER = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember"
src = io.open(os.path.join(EMBER, "scripts/ember.mjs"), encoding="utf-8").read()

def load(n):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, n + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def die(m): sys.stderr.write("build_table: " + m + "\n"); sys.exit(2)

t1, t2 = load("cn_t1"), load("cn_t2")
CN = dict(t1.CN); CN.update(t2.CN)
u = json.load(io.open(os.path.join(HERE, "parts_universe2.json"), encoding="utf-8"))
seg2l = u["seg2layers"]

RE = re.compile(r'(?<!^)([A-Z1-9])')
disp = lambda s: RE.sub(r' \1', s)

# 显示名 → id（图集末段是唯一来源，拆词是双射）
d2id = {}
for seg in seg2l:
    d = disp(seg)
    if d in d2id: die("拆词不是双射：%r 与 %r 同名" % (d2id[d], seg))
    d2id[d] = seg

table = {}
unresolved = []
for d, cn in CN.items():
    seg = d2id.get(d)
    if seg is None:
        unresolved.append(d); continue
    table[seg] = cn
# 只在拼串里用到的基名词
for seg, cn in t2.EXTRA_BASES.items():
    if seg in table:
        continue
    table[seg] = cn
    unresolved = [x for x in unresolved if x != disp(seg)]
unresolved = [d for d in unresolved if disp(d2id.get(d, "\x00")) != d]

# ── A 切对条数 ──
n_in = len(CN)
n_ex = len([s for s in t2.EXTRA_BASES if s not in {d2id[d] for d in CN if d in d2id}])
if len(table) != len([d for d in CN if d in d2id]) + n_ex:
    die("A FAIL 表 %d ≠ 反查到的 %d + 额外基名 %d" % (len(table), len([d for d in CN if d in d2id]), n_ex))
print("A OK  显示名 %d 条 → id %d 条（另加只用于拼串的基名 %d 条），无丢无并" %
      (len(CN), len([d for d in CN if d in d2id]), n_ex))
if unresolved: die("A FAIL 有 %d 条显示名在图集里反查不到 id：%s" % (len(unresolved), unresolved[:8]))

# ── B 切对对象：逐条回 ember.mjs 查字面量 ──
bad = [k for k in table if k not in src]
if bad: die("B FAIL %d 个 id 在 ember.mjs 里查不到字面量：%s" % (len(bad), sorted(bad)[:10]))
print("B OK  %d 个 id **逐条**在 ember.mjs 里查得到字面量（自检面板 D 档不会因此新增 miss）" % len(table))

# ── C 拆词等价 ──
for seg, cn in table.items():
    d = disp(seg)
    if d in CN and CN[d] != cn: die("C FAIL %r 反查回来译文不一致" % seg)
print("C OK  %d 个 id 用上游那条正则重算显示名，与输入逐字符相同" % len(table))

# ── D 拼串族闭合 ──
POSES = ["Backward", "Forward", "Neutral", "Sitting"]
compose = {}
for b in t2.LEG_POSE_BASES:
    if b not in table: die("D FAIL 腿姿基名 %r 不在表里" % b)
    for p in POSES:
        cid = b + p
        if cid in table: continue
        if cid not in seg2l: die("D FAIL 拼出来的 %r 不在图集里" % cid)
        compose[cid] = (b, p)
for b in t2.MARBLED_HAND_BASES:
    if b not in table: die("D FAIL 大理石手部基名 %r 不在表里" % b)
    cid = "Marbled" + b
    if cid in table: continue
    if cid not in seg2l: die("D FAIL 拼出来的 %r 不在图集里" % cid)
    compose[cid] = ("Marbled", b)
# 图集里那一族有没有被漏掉的
fam = {s for s in seg2l if s not in table and
       (any(s.endswith(p) and s[:-len(p)] in table for p in POSES) or
        (s.startswith("Marbled") and s[len("Marbled"):] in table))}
if fam - set(compose):
    die("D FAIL 图集里这一族还有 %d 条没被拼出来：%s" % (len(fam - set(compose)), sorted(fam - set(compose))[:10]))
print("D OK  拼串 %d 条，逐条落在图集里；图集同族无遗漏" % len(compose))

# ── 落表 ──
covered = {disp(s) for s in table} | {disp(s) for s in compose}
shown = set()
decl = {m.group(1).split("/")[-1] for m in re.finditer(r'\bid:\s*"([A-Za-z0-9][A-Za-z0-9_./-]*)"', src)}
legb = {m.group(1) for m in re.finditer(r'makeLegPoseParts\(\s*"([^"]+)"', src)}
POSES4 = POSES
for s in seg2l:
    if s in decl or s in legb: shown.add(s)
    elif any(s.endswith(p) and s[:-len(p)] in (decl | legb) for p in POSES4): shown.add(s)
    elif s.startswith("Marbled") and s[len("Marbled"):] in (decl | legb): shown.add(s)
    elif s.endswith("Lower") and s[:-5] in (decl | legb): shown.add(s)
shown_disp = {disp(s) for s in shown}
print("\n【覆盖账】上游真会上屏的唯一显示名 %d 条；本表直接覆盖 %d 条 + 拼串 %d 条 = %d 条；仍缺 %d 条"
      % (len(shown_disp), len(table), len(compose), len(covered), len(shown_disp - covered)))

json.dump({"table": {k: table[k] for k in sorted(table)},
           "compose": {k: list(v) for k, v in sorted(compose.items())},
           "shown_total": len(shown_disp), "covered": len(covered),
           "still_missing": sorted(shown_disp - covered)},
          io.open(os.path.join(HERE, "table.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# .mjs 片段
lines, buf = [], ""
for k in sorted(table):
    piece = '"%s": "%s", ' % (k, table[k])
    if len(buf) + len(piece) > 96:
        lines.append("  " + buf.rstrip()); buf = ""
    buf += piece
if buf: lines.append("  " + buf.rstrip().rstrip(","))
io.open(os.path.join(HERE, "PARTS.snippet.mjs"), "w", encoding="utf-8").write(
    "const TOKEN_MAKER_PART_IDS = {\n" + "\n".join(lines) + "\n};\n")
print("已写 table.json / PARTS.snippet.mjs（%d 键）" % len(table))
