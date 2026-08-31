# -*- coding: utf-8 -*-
r"""
探针 P-INTERACTABLE：把 ember 每个 interactable 的 `dialog: {…}` 整块抓出来，
逐块列「标题有没有键 / 按钮有没有键」。

为什么这是种子那一类：标题和按钮写在**同一个对象字面量**里、往往就是相邻行，
是不折不扣的「同一批字符串」。汉化 EXACT 补了其中一部分标题，按钮一条没补。

假阳性模式：
  FP1 花括号配平是字符级的，字符串里若有未转义的 { } 会切错块（已用引号跳过缓解）。
  FP2 少数 dialog 在 _configureDialog 里被整块替换，静态块里的按钮实际不会出现。
  FP3 有的 interactable 只在特定场景里存在，玩家不一定开得到。
"""
import io, os, re, sys, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER_UP = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
HC = os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")
src = open(HC, encoding="utf-8").read()
EXACT = set(re.findall(r'"([^"]+)":\s*"', re.search(r"const EXACT = \{(.*?)\n\};", src, re.S).group(1)))

c = open(os.path.join(EMBER_UP, "scripts", "ember.mjs"), encoding="utf-8").read()


def block(text, i):
    """从 text[i] == '{' 开始返回配平的块"""
    depth, j, q = 0, i, None
    while j < len(text):
        ch = text[j]
        if q:
            if ch == "\\":
                j += 2
                continue
            if ch == q:
                q = None
        elif ch in "\"'`":
            q = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
        j += 1
    return text[i:i + 4000]


rows = []
for m in re.finditer(r"\bdialog:\s*\{", c):
    b = block(c, m.end() - 1)
    line = c.count("\n", 0, m.start()) + 1
    title = re.search(r"title:\s*[\"`']([^\"`']+)[\"`']", b)
    labels = re.findall(r"label:\s*[\"`']([^\"`']+)[\"`']", b)
    # 找所属类名（往上找最近的 class 定义）
    pre = c[:m.start()]
    k = pre.rfind("\nclass ")
    cls = re.match(r"\nclass (\w+)", pre[k:]).group(1) if k >= 0 else "?"
    rows.append({"line": line, "cls": cls,
                 "title": title.group(1) if title else None,
                 "title_cov": bool(title and title.group(1) in EXACT),
                 "labels": labels,
                 "labels_cov": [l for l in labels if l in EXACT]})

print(f"抓到 interactable dialog 块 {len(rows)} 个\n")
tt = tc = bt = bc = 0
part = []
for r in rows:
    if r["title"]:
        tt += 1
        tc += r["title_cov"]
    bt += len(r["labels"])
    bc += len(r["labels_cov"])
    st = "标题✔" if r["title_cov"] else ("标题✘" if r["title"] else "标题-")
    sb = f'按钮 {len(r["labels_cov"])}/{len(r["labels"])}'
    mark = "  <== PARTIAL" if (r["title_cov"] and r["labels"] and not r["labels_cov"]) else ""
    if mark:
        part.append(r)
    print(f'  {r["cls"]:28s} :{r["line"]:<7d} {st}  {sb:12s} {r["title"]!r}')
    if r["labels"]:
        print(f'        按钮: {r["labels"]}{mark}')

print(f"\n合计：标题 {tc}/{tt} 有键；按钮 {bc}/{bt} 有键")
print(f"PARTIAL（标题有键、按钮全无键）的 dialog 块：{len(part)} 个")
for r in part:
    print(f'   {r["cls"]} :{r["line"]}  "{r["title"]}" -> {r["labels"]}')

outp = os.path.join(ROOT, "4-临时脚本", "2026-08-13-final-audit", "findings", "p_interactable_dialog.json")
json.dump(rows, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", outp)
