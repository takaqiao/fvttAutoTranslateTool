# -*- coding: utf-8 -*-
"""标记闸彻底看不见的一层：**属性 -> 内容的绑定**。

Ember 大量使用双系统切换块：
  <sup class="system-swap-inline"><sub data-system="dnd5e">A</sub>
                                  <sub data-system="crucible">B</sub></sup>
两个 <sub> 的标签名与个数完全相同，多重集判据、树形判据都看不出 A 与 B 被对调。
但对调之后，Crucible 玩家会读到 dnd5e 的规则数值 —— 是硬错。

  S1  同一 data-system 值下的「机械内容签名」中英不一致
      （机械内容＝该节点内的 UUID/数字/骰式/[[…]] 指令，翻译不该动这些）
  S2  CN 少了/多了某个 data-system 值的节点
  S3  重复属性 <p class="a" class="b">（HTML 里非法，浏览器取第一个）
  S4  @Xxx[目标] 的目标里混进非 ASCII 字符（中文标点/全角空格混进标记内部）
"""
import json, re, sys, collections
from pathlib import Path
from lxml import html as lhtml

MECH = re.compile(
    r"[A-Za-z0-9]{16}"                 # Foundry id
    r"|\d+d\d+(?:[+-]\d+)?"            # 骰式
    r"|\[\[[^\]]*\]\]"                 # 内联指令
    r"|\d+(?:\.\d+)?"                  # 数字
)
DUP_ATTR = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)((?:\s+[a-zA-Z-]+(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+))?)+)\s*/?>")
TARGET = re.compile(r"(@[A-Za-z]+|&(?:amp;)?[A-Za-z]+)\[([^\[\]]*)\]")


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def swap_sig(s):
    """返回 {data-system 值: [该节点内机械内容签名, ...]}，按文档序。"""
    try:
        root = lhtml.fragment_fromstring(s, create_parent="root")
    except Exception:
        return None
    out = collections.defaultdict(list)
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        v = el.get("data-system")
        if v is None:
            continue
        inner = lhtml.tostring(el, encoding="unicode")
        out[v].append(tuple(MECH.findall(inner)))
    return dict(out)


def dup_attrs(s):
    out = []
    for m in DUP_ATTR.finditer(s):
        names = re.findall(r"(?:^|\s)([a-zA-Z-]+)\s*=", m.group(2))
        c = collections.Counter(n.lower() for n in names)
        d = [n for n, k in c.items() if k > 1]
        if d:
            out.append((d, m.group(0)[:120]))
    return out


counts = collections.Counter()
rows = []
n_swap = 0
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
            if "data-system" in s or "data-system" in e:
                n_swap += 1
                a, b = swap_sig(s), swap_sig(e)
                if a is None or b is None:
                    continue
                if set(a) != set(b):
                    counts["S2"] += 1
                    rows.append(("S2", repo.name, f.name, p,
                                 f"CN keys={sorted(a)} EN keys={sorted(b)}", e[:200], s[:200]))
                else:
                    for k in a:
                        if a[k] != b[k]:
                            counts["S1"] += 1
                            rows.append(("S1", repo.name, f.name, p,
                                         f"data-system={k} 机械签名 CN={a[k]} EN={b[k]}",
                                         e[:300], s[:300]))
                            break
            if "<" in s:
                d = dup_attrs(s)
                if d and not dup_attrs(e):
                    counts["S3"] += 1
                    rows.append(("S3", repo.name, f.name, p, d[:2], e[:150], s[:150]))
            for m in TARGET.finditer(s):
                tgt = m.group(2)
                bad = [ch for ch in tgt if ord(ch) > 0x7F]
                if bad:
                    counts["S4"] += 1
                    rows.append(("S4", repo.name, f.name, p,
                                 f"目标含非ASCII {bad[:5]!r}: {m.group(0)[:120]}", e[:150], s[:150]))
                    break

print(f"含 data-system 的叶: {n_swap}")
print(counts)
seen = collections.Counter()
for code, rn, pack, p, det, e, s in rows:
    seen[code] += 1
    if seen[code] > 20:
        continue
    print("-" * 96)
    print(f"[{code}] {rn} {pack} | {p}")
    print("    det:", str(det)[:400])
    print("    EN :", e[:300])
    print("    CN :", s[:300])
