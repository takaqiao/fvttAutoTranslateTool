# -*- coding: utf-8 -*-
"""按 Foundry v14 的真实契约解析 @Embed[...] 的配置串，比对中英解析结果。

契约（foundry.mjs 逐字抄来）：
  匹配   /@Embed\\[(?<config>[^\\]]+)](?:{(?<label>[^}]+)})?/gi
  分词   raw.match(/(?:[^\\s"]+|"[^"]*")+/g)
  拆键值 const [key, value] = part.split("=")      <-- 只取前两段！值里再有 = 就被丢掉
  取值   true/false -> 布尔; Number.isNumeric -> 数字; 否则去掉首尾引号

所以译文里只要在 label/readaloud/caption 的值里出现：
  ASCII '='      -> 值被截断
  ASCII '"'      -> 分词错位
  未加引号+空格  -> 后半截掉进 values[]，参数只剩前半
  纯数字         -> 变成 Number
  ASCII ']'      -> 整个 @Embed 提前结束
都会静默坏掉，而标记闸（只比 @Embed[…] 整串是否相等，且已把 key="value" 抹平）看不见。

  E1 中英解析出的**参数键集合**不同
  E2 中英 values[]（无键裸值，uuid 就在里面）不同
  E3 CN 侧值里含 ASCII = / " / ]（会触发上面几条）
  E4 CN 值被 Number.isNumeric 判为数字而 EN 不是
  E5 {label} 里含 ASCII }（label 提前截断）
"""
import json, re, sys, collections
from pathlib import Path

EMBED = re.compile(r"@Embed\[([^\]]+)\](?:\{([^}]+)\})?", re.I)
TOKEN = re.compile(r'(?:[^\s"]+|"[^"]*")+')
NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$", re.I)


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def parse_config(raw):
    """逐字复刻 TextEditor._parseEmbedConfig。"""
    cfg, values = {}, []
    for part in TOKEN.findall(raw):
        if not part:
            continue
        bits = part.split("=")
        key = bits[0]
        value = bits[1] if len(bits) > 1 else None      # destructuring 只取前两个
        if value is None:
            values.append(re.sub(r'(^"|"$)', "", key))
        elif value.lower() in ("true", "false"):
            cfg[key] = value.lower() == "true"
        elif NUMERIC.match(value):
            cfg[key] = float(value)
        else:
            cfg[key] = re.sub(r'(^"|"$)', "", value)
    return cfg, values


counts = collections.Counter()
rows = []
n_embed = 0
for repo in sys.argv[1:]:
    repo = Path(repo)
    for f in sorted((repo / "compendium" / "en").glob("*.json")):
        cnp = repo / "compendium" / "cn" / f.name
        if not cnp.exists():
            continue
        en = dict(leaves(json.loads(f.read_text(encoding="utf-8-sig"))))
        cn = dict(leaves(json.loads(cnp.read_text(encoding="utf-8-sig"))))
        for p, s in cn.items():
            e = en.get(p, "")
            if "@Embed[" not in s and "@Embed[" not in e:
                continue
            cm, em = EMBED.findall(s), EMBED.findall(e)
            n_embed += len(cm)
            if len(cm) != len(em):
                counts["E0"] += 1
                rows.append(("E0", repo.name, f.name, p, f"@Embed 个数 CN={len(cm)} EN={len(em)}", "", ""))
                continue
            for (craw, clab), (eraw, elab) in zip(cm, em):
                ccfg, cval = parse_config(craw)
                ecfg, eval_ = parse_config(eraw)
                if set(ccfg) != set(ecfg):
                    counts["E1"] += 1
                    rows.append(("E1", repo.name, f.name, p,
                                 f"键集 CN={sorted(ccfg)} EN={sorted(ecfg)}", eraw[:200], craw[:200]))
                if cval != eval_:
                    counts["E2"] += 1
                    rows.append(("E2", repo.name, f.name, p,
                                 f"values CN={cval} EN={eval_}", eraw[:200], craw[:200]))
                for k, v in ccfg.items():
                    if isinstance(v, str) and any(ch in v for ch in ('=', '"', ']')):
                        counts["E3"] += 1
                        rows.append(("E3", repo.name, f.name, p,
                                     f"{k} 的值里含 = 或 \" 或 ]: {v[:120]!r}", eraw[:150], craw[:150]))
                    if isinstance(v, float) and not isinstance(ecfg.get(k), float):
                        counts["E4"] += 1
                        rows.append(("E4", repo.name, f.name, p,
                                     f"{k} 在中文侧被判为数字 {v}", eraw[:150], craw[:150]))
                if clab and "}" in clab:
                    counts["E5"] += 1
                    rows.append(("E5", repo.name, f.name, p, f"label 含 }}: {clab[:80]!r}", "", ""))

print(f"扫描 @Embed 实例: {n_embed}")
print(counts)
seen = collections.Counter()
for code, rn, pack, p, det, e, s in rows:
    seen[code] += 1
    if seen[code] > 20:
        continue
    print("-" * 96)
    print(f"[{code}] {rn} {pack} | {p}")
    print("    det:", str(det)[:400])
    if e:
        print("    EN :", e[:250])
        print("    CN :", s[:250])
