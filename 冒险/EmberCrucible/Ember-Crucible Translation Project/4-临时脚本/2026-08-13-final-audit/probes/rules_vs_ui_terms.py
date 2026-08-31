# -*- coding: utf-8 -*-
"""只读探针 #8：规则日志里的**术语标签** vs 界面 lang 文件里的**同一术语**。

玩家先读规则页，再在动作卡/物品卡的下拉框里选同一个词。
这两层分别由 compendium/*.json 和 lang/cn.json 提供，历轮扫描从未把它们放在一起比过。

做法：
 1. 从 crucible.rules.json 的 EN/CN 同位置抽出「短标签」：<h4>、<h3>、表格首列 <td><p>…</p></td>、<strong>。
 2. 从 lang/en.json + lang/cn.json 抽出所有英文短串（<=24 字符）的中文。
 3. 同一英文短串在两层给出不同中文 → 报出。

假阳性模式：
 - 同形异义（例：Light 光/轻型、Aura 奥拉/灵气）。报告保留出处供判断。
 - 规则页的标签有时是英文长式（"Self Only" vs UI "Self"），此时英文本就不同，已被 key 天然排除。
 - 中文双语并列格式（"护盾术 Shield"）只出现在 name 字段，本探针不取 name。
"""
import io, re, os, json, collections

R = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
TAG = re.compile(r"<[^>]+>")

def flat(d):
    out = {}
    def w(ob, p):
        if isinstance(ob, dict):
            for k, v in ob.items(): w(v, p + [str(k)])
        elif isinstance(ob, list):
            for i, v in enumerate(ob): w(v, p + ["[%d]" % i])
        elif isinstance(ob, str): out[".".join(p)] = ob
    w(d, [])
    return out

def labels(html):
    out = []
    for pat in [r"<h[34][^>]*>(.*?)</h[34]>", r"<td><p>(.*?)</p></td>", r"<strong>(.*?)</strong>"]:
        for m in re.findall(pat, html, re.S):
            t = TAG.sub("", m).strip()
            if t and len(t) <= 24:
                out.append(t)
    return out

def main():
    o = io.open(os.environ.get("OUT", "rules_vs_ui.txt"), "w", encoding="utf-8")
    for repo, files in [("2-Crucible汉化插件", ["crucible.rules.json"]),
                        ("1-Ember汉化插件", ["ember.crucible-adventure.json"])]:
        try:
            LE = flat(json.load(io.open(os.path.join(R, repo, "lang", "en.json"), encoding="utf-8")))
            LC = flat(json.load(io.open(os.path.join(R, repo, "lang", "cn.json"), encoding="utf-8")))
        except Exception:
            continue
        ui = collections.defaultdict(set)
        for k, ev in LE.items():
            cv = LC.get(k)
            if cv and len(ev) <= 24 and re.search(r"[\u4e00-\u9fff]", cv):
                ui[ev.strip()].add(cv.strip())
        doc = collections.defaultdict(lambda: collections.defaultdict(list))
        for fn in files:
            E = flat(json.load(io.open(os.path.join(R, repo, "compendium", "en", fn), encoding="utf-8")))
            C = flat(json.load(io.open(os.path.join(R, repo, "compendium", "cn", fn), encoding="utf-8")))
            for k, ev in E.items():
                cv = C.get(k)
                if not cv: continue
                le, lc = labels(ev), labels(cv)
                if len(le) != len(lc): continue   # 位置对不齐就跳过，宁缺毋滥
                for a, b in zip(le, lc):
                    if re.search(r"[\u4e00-\u9fff]", b) and re.search(r"^[A-Za-z][A-Za-z '\-/&]*$", a):
                        doc[a][b].append(k[-56:])
        o.write("\n########## %s\n" % repo)
        n = 0
        for en_s, cns in sorted(doc.items()):
            if en_s not in ui: continue
            docset = set(cns.keys())
            uiset = ui[en_s]
            if docset & uiset: continue        # 有交集就算一致
            n += 1
            o.write("  EN %-22s  规则页: %-24s  UI: %s\n" % (
                en_s, " / ".join(sorted(docset))[:24], " / ".join(sorted(uiset))))
            for b, ks in cns.items():
                o.write("        %s  @ %s\n" % (b, ks[0]))
        o.write("  共 %d 组\n" % n)
    o.close()
    print("ok")

if __name__ == "__main__":
    main()
