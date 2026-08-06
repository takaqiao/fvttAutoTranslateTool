"""Wave 2 work files. Two kinds of unit:
  - journal  : one work dir per journal (same shape as wave 1)
  - bucket   : non-journal slices (tables / scenes / actor-embedded items), split
               into chunks small enough for one agent
Reads the residual list so items Babele resolves for free are never handed out.
"""
import json, os, re, sys, collections

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
# 单元工作目录是会话级临时件，不进仓库。用环境变量指过去：
#   $env:EMBER_PARALLEL_ROOT = "<scratchpad>\parallel"
ROOT = os.environ.get("EMBER_PARALLEL_ROOT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "parallel")
RES = os.path.join(P, "5-其他内容", "reports", "ember", "todo", "_residual_after_fallback.json")
CNP = os.path.join(P, "1-Ember汉化插件", "compendium", "cn", "ember.crucible-adventure.json")
CJK = re.compile(r"[一-鿿]")

items = json.load(open(RES, encoding="utf-8"))["packs"]["ember.crucible-adventure"]
cn = json.load(open(CNP, encoding="utf-8"))
CJ = cn["entries"]["Ember Early Access"]["journals"]


def slug(s):
    out = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    # 全中文的名字会被剥成空串，os.path.join(ROOT, "") 就等于 ROOT 本身 ——
    # todo.json 会被写进并行根目录，而 agent 按 ROOT\<名字> 去找，什么都找不到。
    # 踩过一次（「小卷合集」），这里必须兜底。
    if not out:
        raise SystemExit(f"单元名 {s!r} 生成不出 ASCII 目录名，请改用带拉丁字母的名字")
    return out


def journal_of(p):
    parts = p.split(".")
    if "journals" in parts:
        i = parts.index("journals")
        return parts[i + 1] if i + 1 < len(parts) else None
    return None


def bucket_of(p):
    parts = p.split(".")
    if "journals" in parts:
        return None
    return parts[1] if len(parts) > 1 else parts[0]


manifest = []
mode = sys.argv[1]

if mode == "journals":
    for jn in sys.argv[2:]:
        sel = [it for it in items if journal_of(it["path"]) == jn]
        if not sel:
            print("!! nothing for", jn); continue
        d = os.path.join(ROOT, slug(jn)); os.makedirs(d, exist_ok=True)
        json.dump({"journal": jn, "items": sel}, open(os.path.join(d, "todo.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        done = {pn: {"name": p.get("name"), "text": p.get("text")}
                for pn, p in (CJ.get(jn, {}).get("pages") or {}).items()
                if isinstance(p.get("text"), str) and CJK.search(p["text"])}
        json.dump(done, open(os.path.join(d, "already_translated.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        c = sum(i["chars"] for i in sel)
        manifest.append({"unit": jn, "dir": d, "items": len(sel), "chars": c})
        print(f"{jn:<30}{len(sel):>5} items {c:>8} chars  已译同卷页 {len(done)}")

elif mode == "merge":
    # 多个小卷合成一个单元。已译页锚点按卷分组，别混在一起
    name = sys.argv[2]
    jns = sys.argv[3:]
    sel = [it for it in items if journal_of(it["path"]) in jns]
    d = os.path.join(ROOT, slug(name)); os.makedirs(d, exist_ok=True)
    json.dump({"journal": f"{name}（{len(jns)} 个小卷）", "items": sel},
              open(os.path.join(d, "todo.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    done = {}
    for jn in jns:
        for pn, p in (CJ.get(jn, {}).get("pages") or {}).items():
            if isinstance(p.get("text"), str) and CJK.search(p["text"]):
                done[f"{jn} :: {pn}"] = {"name": p.get("name"), "text": p["text"]}
    json.dump(done, open(os.path.join(d, "already_translated.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    c = sum(i["chars"] for i in sel)
    manifest.append({"unit": name, "dir": d, "items": len(sel), "chars": c})
    print(f"{name:<26}{len(sel):>5} items {c:>8} chars  已译锚点页 {len(done)}")

elif mode == "split":
    # 单卷太大，按页切块；同一页的所有字段留在同一块里，别把一页劈成两半
    jn, chunk_chars = sys.argv[2], int(sys.argv[3])
    sel = [it for it in items if journal_of(it["path"]) == jn]
    by_page = collections.OrderedDict()
    for it in sorted(sel, key=lambda x: x["path"]):
        parts = it["path"].split(".")
        page = parts[parts.index("pages") + 1] if "pages" in parts else "(其他)"
        by_page.setdefault(page, []).append(it)
    chunks, cur, acc = [], [], 0
    for page, rows in by_page.items():
        c = sum(r["chars"] for r in rows)
        if cur and acc + c > chunk_chars:
            chunks.append(cur); cur, acc = [], 0
        cur += rows; acc += c
    if cur:
        chunks.append(cur)
    done = {pn: {"name": p.get("name"), "text": p.get("text")}
            for pn, p in (CJ.get(jn, {}).get("pages") or {}).items()
            if isinstance(p.get("text"), str) and CJK.search(p["text"])}
    for i, ch in enumerate(chunks, 1):
        d = os.path.join(ROOT, f"{slug(jn)}-{i}"); os.makedirs(d, exist_ok=True)
        json.dump({"journal": f"{jn}（第 {i}/{len(chunks)} 块）", "items": ch},
                  open(os.path.join(d, "todo.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(done, open(os.path.join(d, "already_translated.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        c = sum(x["chars"] for x in ch)
        manifest.append({"unit": f"{jn}-{i}", "dir": d, "items": len(ch), "chars": c})
        print(f"{jn}-{i:<20}{len(ch):>5} items {c:>8} chars")

elif mode == "bucket":
    name, chunk_chars = sys.argv[2], int(sys.argv[3])
    sel = [it for it in items if bucket_of(it["path"]) == name]
    sel.sort(key=lambda it: it["path"])
    chunks, cur, acc = [], [], 0
    for it in sel:
        cur.append(it); acc += it["chars"]
        if acc >= chunk_chars:
            chunks.append(cur); cur, acc = [], 0
    if cur:
        chunks.append(cur)
    for i, ch in enumerate(chunks, 1):
        d = os.path.join(ROOT, f"{name}-{i}"); os.makedirs(d, exist_ok=True)
        json.dump({"journal": f"{name} 第 {i}/{len(chunks)} 批", "items": ch},
                  open(os.path.join(d, "todo.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump({}, open(os.path.join(d, "already_translated.json"), "w", encoding="utf-8"))
        c = sum(x["chars"] for x in ch)
        manifest.append({"unit": f"{name}-{i}", "dir": d, "items": len(ch), "chars": c})
        print(f"{name}-{i:<24}{len(ch):>5} items {c:>8} chars")

out = os.path.join(ROOT, f"manifest.wave2.{mode}.json")
json.dump(manifest, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n->", out)
