# -*- coding: utf-8 -*-
"""把 fix2 的每条改动渲染成可通读的对照：英文句 vs 中文句，
链接写成 «标签→目标文档»，方便逐条确认「改完仍然通顺且语义正确」。"""
import json, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
UUID_RX = re.compile(r"@UUID\[([^\]]*)\](\{([^}]*)\})?")
TAG = re.compile(r"<[^>]+>")


def key_of(t):
    core = t.split()[0] if t.split() else t
    if "#" in core:
        h, f = core.split("#", 1)
        return h.split(".")[-1] + "#" + f
    return core.split(".")[-1]


def main():
    rep, ids, only = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else None)
    d = json.load(open(rep, encoding="utf-8"))
    I = json.load(open(ids, encoding="utf-8"))

    def doc(t):
        r = I.get(key_of(t).split("#")[0])
        return (r["name"] if r else "?") + ("#" + t.split("#")[1].split("]")[0] if "#" in t else "")

    def render(s, mark=()):
        def rep_(m):
            t, l = m.group(1), m.group(3) or ""
            star = "*" if key_of(t) in mark else ""
            return f"«{star}{l}→{doc(t)}»"
        return TAG.sub(" ", UUID_RX.sub(rep_, s))

    seen = set()
    for f in d["fixed"]:
        if only and not f["pack"].startswith(only):
            continue
        if f["path"] in seen:
            continue
        seen.add(f["path"])
        print("=" * 110)
        print(f["path"])
        en = render(f.get("en", ""))
        print("EN:", " ".join(x["label"] + "→" + str(x["doc"]) for x in f["en_links"]))
        print("CN(before):", render(f["cn_before"]))
        print("CN(after) :", render(f["new_cn"]))
        print()


if __name__ == "__main__":
    main()
