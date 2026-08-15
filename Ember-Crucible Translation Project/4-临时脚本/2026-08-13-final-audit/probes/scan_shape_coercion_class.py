# -*- coding: utf-8 -*-
"""
举一反三探针：**形状/类型不匹配导致的静默强制转换**（"多态字段被单态代码强改类型"这一类）

判据抽象
--------
只要一处 *写入方* 产生的值，其 JS 运行时**形状**（string / object / array）与
*读取方（上游 schema 或上游 fallback 数据）* 声明的形状不同，且中间层
（DataField._cast / mergeObject / Localization merge）**不会报错、只会静默转换或整块顶掉**，
就属于同一类缺陷。

三条可机械化的子判据（本探针实现 S1 / S2；S3 走静态枚举，见脚本尾部清单）：

S1  lang/cn.json 的节点形状 vs 同路径 fallback（本仓 lang/en.json + 上游 crucible/ember lang）
    - cn 是 str、fallback 是 dict  → Localization 用 mergeObject 合并时整棵子树被字符串顶掉
    - cn 是 dict、fallback 是 str  → localize() 拿到对象

S2  compendium/cn/<pack>.json 的每个字段值形状 vs compendium/en/<pack>.json 同键同字段
    英文基准是从 LevelDB 按 schema 抽出来的，所以它的形状 == 源文档形状。
    babele 的默认 converter 直接 setProperty(译文)，形状不同就等于换类型。

假阳性模式（必须人工核对）
-------------------------
* S1: 上游 fallback 里同一路径既有 "FOO" 又有 "FOO.BAR"（Foundry 允许 flat key
  和嵌套并存），这时 str/dict 并存是合法的 —— 脚本会把两侧都打印出来供判断。
* S2: 我方有意用 converter（crucibleDescription / crucibleNested / crucibleActions /
  structured / nameCollection）吸收形状差异的字段，形状不同**不一定**是缺陷。
  脚本会标注该字段走的是哪个 converter，由人判断该 converter 是否真的能吸收。
* S2: 英文基准可能是旧版抽取的，值缺失（None）不算形状不匹配，单独计数。
"""
import json, os, sys, io, collections

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
FOUNDRY = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
REPOS = {
    "ember": os.path.join(ROOT, "1-Ember汉化插件"),
    "crucible": os.path.join(ROOT, "2-Crucible汉化插件"),
}

def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def kind(v):
    if isinstance(v, str): return "str"
    if isinstance(v, dict): return "dict"
    if isinstance(v, list): return "list"
    if v is None: return "null"
    if isinstance(v, bool): return "bool"
    if isinstance(v, (int, float)): return "num"
    return type(v).__name__

# ---------------------------------------------------------------- S1
def walk(node, prefix, out):
    """把嵌套 lang 对象拍平成 {点路径: 值}，同时保留中间节点。"""
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{prefix}.{k}" if prefix else k
            out[p] = v
            walk(v, p, out)

def s1_lang():
    print("=" * 78)
    print("S1  lang JSON 节点形状 vs fallback")
    print("=" * 78)
    upstream = {}
    for name, p in [
        ("crucible", os.path.join(FOUNDRY, "systems", "crucible", "lang", "en.json")),
    ]:
        if os.path.exists(p):
            flat = {}
            walk(load(p), "", flat)
            upstream[name] = flat
            print(f"  上游 {name} lang/en.json 展开 {len(flat)} 个节点")

    total_hits = 0
    for repo, base in REPOS.items():
        cn_p = os.path.join(base, "lang", "cn.json")
        en_p = os.path.join(base, "lang", "en.json")
        if not os.path.exists(cn_p):
            continue
        cn, en = {}, {}
        walk(load(cn_p), "", cn)
        if os.path.exists(en_p):
            walk(load(en_p), "", en)
        print(f"\n[{repo}] cn 节点 {len(cn)} / 本仓 en 节点 {len(en)}")

        # 与本仓 en 比
        for p, v in sorted(cn.items()):
            for src_name, src in [("self-en", en)] + list(upstream.items()):
                if p not in src:
                    continue
                a, b = kind(v), kind(src[p])
                if a != b:
                    total_hits += 1
                    print(f"  SHAPE {repo} {p}: cn={a} vs {src_name}={b}")
                    print(f"        cn  = {json.dumps(v, ensure_ascii=False)[:160]}")
                    print(f"        src = {json.dumps(src[p], ensure_ascii=False)[:160]}")
                break  # 只跟第一个能找到该路径的来源比
    print(f"\nS1 命中 {total_hits}")
    return total_hits

# ---------------------------------------------------------------- S2
# mappings.mjs 里声明了 converter 的键（converter 有可能吸收形状差异）
CONVERTER_KEYS = {
    "description": "crucibleDescription",
    "actions": "crucibleActions",
    "biography": "crucibleNested", "ancestry": "crucibleNested",
    "background": "crucibleNested", "archetype": "crucibleNested",
    "taxonomy": "crucibleNested",
    "outcomes": "structured",
    "effects": "document", "items": "document", "pages": "document",
    "journals": "document", "scenes": "document", "macros": "document",
    "playlists": "document", "tables": "document", "actors": "document",
    "results": "document", "sounds": "document", "regions": "document",
    "behaviors": "document", "changes": "structured",
    "folders": "nameCollection", "categories": "nameCollection",
    "levels": "nameCollection", "tokens": "nameCollection",
    "drawings": "textCollection", "notes": "textCollection",
}

NODES = [0]

def compare_entry(path, cn, en, hits, stats, converter=None):
    """递归比较译文与英文基准同路径的形状。"""
    NODES[0] += 1
    ka, kb = kind(cn), kind(en)
    if ka != kb:
        stats[(ka, kb)] += 1
        hits.append((path, ka, kb, cn, en, converter))
        return                       # 形状已不同，不再深入
    if ka == "dict":
        for k, v in cn.items():
            if k not in en:
                continue
            compare_entry(f"{path}.{k}", v, en[k],
                          hits, stats, CONVERTER_KEYS.get(k, converter))
    elif ka == "list":
        for i, v in enumerate(cn):
            if i < len(en):
                compare_entry(f"{path}[{i}]", v, en[i], hits, stats, converter)

def s2_compendium():
    print()
    print("=" * 78)
    print("S2  compendium/cn 字段形状 vs compendium/en 同键同字段")
    print("=" * 78)
    all_hits = []
    stats = collections.Counter()
    files = 0
    entries_cmp = 0
    for repo, base in REPOS.items():
        cn_dir = os.path.join(base, "compendium", "cn")
        en_dir = os.path.join(base, "compendium", "en")
        for fn in sorted(os.listdir(cn_dir)):
            if not fn.endswith(".json"):
                continue
            en_p = os.path.join(en_dir, fn)
            if not os.path.exists(en_p):
                print(f"  [skip] {repo}/{fn} 无英文基准")
                continue
            files += 1
            cn = load(os.path.join(cn_dir, fn))
            en = load(en_p)
            ce, ee = cn.get("entries"), en.get("entries")
            if not isinstance(ce, dict) or not isinstance(ee, dict):
                print(f"  [warn] {repo}/{fn} entries 不是 dict: {kind(ce)}/{kind(ee)}")
                continue
            for key, cval in ce.items():
                if key not in ee:
                    continue
                entries_cmp += 1
                compare_entry(f"{repo}/{fn}::{key}", cval, ee[key], all_hits, stats)
            # folders / label 也比
            for top in ("label", "folders"):
                if top in cn and top in en:
                    compare_entry(f"{repo}/{fn}::<{top}>", cn[top], en[top], all_hits, stats)
    print(f"  比对 {files} 个包 / {entries_cmp} 个顶层条目 / 递归比对节点 {NODES[0]} 个")
    print(f"  形状分布 (cn_kind -> en_kind): {dict(stats)}")
    print(f"  命中 {len(all_hits)}")
    by_conv = collections.Counter(h[5] for h in all_hits)
    print(f"  按 converter 归类: {dict(by_conv)}")
    # 按 (converter, 形状对, 末级字段名) 聚类打样本
    groups = collections.defaultdict(list)
    for path, ka, kb, cv, ev, conv in all_hits:
        leaf = path.split("::")[-1]
        field = leaf.split(".")[-1] if "." in leaf else "<root>"
        groups[(conv, ka, kb, field)].append((path, cv, ev))
    print(f"  聚类 {len(groups)} 组：")
    for g, items in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"\n  --- converter={g[0]} cn={g[1]} en={g[2]} field={g[3]}  x{len(items)}")
        for path, cv, ev in items[:3]:
            print(f"      {path}")
            print(f"        cn = {json.dumps(cv, ensure_ascii=False)[:200]}")
            print(f"        en = {json.dumps(ev, ensure_ascii=False)[:200]}")
    return all_hits

if __name__ == "__main__":
    s1_lang()
    s2_compendium()
