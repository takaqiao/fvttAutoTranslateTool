# -*- coding: utf-8 -*-
"""`fix_gazetteer_dt.py` 幂等护栏的双向回测。

  python 4-临时脚本/2026-08-15-round16/qa/backtest_gazetteer_dt.py      # 从项目根跑

为什么要有这份回测
------------------
`fix_gazetteer_dt.py` 在 `PROJECT.md §5.4「发版前必跑」` 清单里，会被下一个会话
无脑全跑。它的旧 `NOOK` 是链式替换表、天生不幂等：库修好之后再跑一遍，它会拿
修好的值当输入**再错位一次**（2026-08-15 实测正确库上仍报「错位修复 3」）。
本回测把「正确库上必须静默」钉成可执行断言。

  特异度：当前（已修好的）库上跑 → 0 叶、不产任何批次文件。
  灵敏度：把 `Scholar's Nook` 的一个 `<dt>` 名人为改错 → 必须精确产出把它改回去的
          批次，且**只动那一个 `<dt>`**；把批次落盘后再跑一遍 → 回到 0 叶（幂等）。

灵敏度那一侧特意用了**原缺陷的形态**（整体错位一格：把中文 7 个 landmark 名整体
前移，末位补上历史误写「抄写员之巢穴」），而不是随便涂一个字 —— 旧版链式表能修的
就是这个形态，新版必须至少不比它弱。
"""
from __future__ import annotations
import importlib.util
import json
import os
import shutil
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
TOOL = os.path.join(ROOT, "4-临时脚本", "2026-08-12-audit3", "fix_gazetteer_dt.py")
REPO = "1-Ember汉化插件"
PACK = "ember.crucible-adventure.json"
PATH = "Ember Early Access.journals.Ordain Gazetteer.pages.Scholar's Nook.text"

# 缺陷当初的样子：整体前移一格，末位是历史误写
BROKEN = ["旧城区图书馆", "秘藏书架", "墨泉书店", "抄写员之巢",
          "余墨书店", "光谱藏书馆", "抄写员之巢穴"]
GOOD = ["旧城区图书馆", "曲脊巷", "秘藏书架", "盖德里克宅邸",
        "墨泉书店", "斯佩克特拉藏书馆", "抄写员之巢"]


def load_tool():
    spec = importlib.util.spec_from_file_location("fix_gazetteer_dt", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def get_leaf(repo, side, pack, path):
    d = json.load(open(os.path.join(repo, "compendium", side, pack), encoding="utf-8"))
    node = d["entries"]
    for k in path.split("."):
        node = node[k]
    return node


def set_leaf(repo, side, pack, path, val):
    p = os.path.join(repo, "compendium", side, pack)
    d = json.load(open(p, encoding="utf-8"))
    node = d["entries"]
    keys = path.split(".")
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = val
    json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def dt_names(m, text):
    return [m.plain(x.group(1)) for x in m.DT.finditer(text)]


def set_dt(m, text, names):
    """把前 len(names) 个 `<dt>` 的内容**按位置**换成给定名字（构造损坏样本用）。"""
    spans = list(m.DT.finditer(text))[:len(names)]
    buf, last = [], 0
    for sp, nm in zip(spans, names):
        buf.append(text[last:sp.start(1)])
        buf.append(f"<p>{nm}</p>")
        last = sp.end(1)
    buf.append(text[last:])
    return "".join(buf)


def clone_repo(dst):
    """只复制两个 pack 的 cn/en，够本脚本用。"""
    for side in ("cn", "en"):
        os.makedirs(os.path.join(dst, "compendium", side), exist_ok=True)
        for pack in ("ember.crucible-adventure.json", "ember.adventure.json"):
            shutil.copy(os.path.join(ROOT, REPO, "compendium", side, pack),
                        os.path.join(dst, "compendium", side, pack))


def main():
    os.chdir(ROOT)
    m = load_tool()
    fails = []

    # ---- 特异度：正确库上必须静默、且不落任何文件 -----------------------------
    res = m.build(os.path.join(ROOT, REPO))
    total = sum(len(o) for _, o, _, _ in res)
    nook = sum(s["nook"] for _, _, s, _ in res)
    ok = (total == 0 and nook == 0)
    print(f"[特异度] 正确库：叶 {total}，错位修复 {nook}  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("正确库上产出了改动（不幂等）")
    with tempfile.TemporaryDirectory() as td:
        rc = os.system(f'python "{TOOL}" --out-dir "{td}" >nul 2>&1'
                       if os.name == "nt" else
                       f'python "{TOOL}" --out-dir "{td}" >/dev/null 2>&1')
        made = os.path.exists(os.path.join(td, "batches"))
        ok2 = (rc == 0 and not made)
        print(f"[特异度] 正确库上 main()：退出码 {rc}，批次目录 {'建了' if made else '没建'}"
              f"  -> {'PASS' if ok2 else 'FAIL'}")
        if not ok2:
            fails.append("正确库上仍产出批次文件")

    # ---- 灵敏度：人为改错一个/一批 <dt> 必须能修回来 --------------------------
    with tempfile.TemporaryDirectory() as td:
        rp = os.path.join(td, "repo")
        clone_repo(rp)
        good_cn = get_leaf(rp, "cn", PACK, PATH)
        # ⚠ 构造损坏样本必须**按位置**写，不能 `replace` 链 ——
        # `曲脊巷→秘藏书架` 写完之后库里就有两个「秘藏书架」，下一条 `秘藏书架→墨泉书店`
        # 会打到刚写出来的那一个。（这正是旧版 NOOK 表的病，别在回测里重犯一遍。）
        broken = set_dt(m, good_cn, BROKEN)
        assert dt_names(m, broken)[:7] == BROKEN, "构造损坏样本失败"
        set_leaf(rp, "cn", PACK, PATH, broken)

        res = m.build(rp)
        out = dict(res[0][1])
        got = dt_names(m, out.get(PATH, ""))[:7] if PATH in out else None
        ok3 = (got == GOOD)
        print(f"[灵敏度] 整体错位一格：批次 {len(out)} 叶，修回 {got}"
              f"  -> {'PASS' if ok3 else 'FAIL'}")
        if not ok3:
            fails.append("错位样本没能修回正确顺序")
        # 只动 <dt>，正文（<dd>）一个字不许动
        ok4 = (m.DT.sub("", out[PATH]) == m.DT.sub("", broken)) if PATH in out else False
        print(f"[灵敏度] 只改 <dt>、<dd> 正文原样  -> {'PASS' if ok4 else 'FAIL'}")
        if not ok4:
            fails.append("批次动到了 <dt> 以外的正文")

        # 落盘后再跑 → 必须回到 0（幂等收敛）
        set_leaf(rp, "cn", PACK, PATH, out[PATH])
        res2 = m.build(rp)
        ok5 = sum(len(o) for _, o, _, _ in res2) == 0
        print(f"[灵敏度] 落盘后重跑收敛到 0 叶  -> {'PASS' if ok5 else 'FAIL'}")
        if not ok5:
            fails.append("修完再跑仍有改动，未收敛")

        # 单点损坏（不是整体错位）也要能定位
        clone_repo(rp)
        one = good_cn.replace("<dt><p>盖德里克宅邸</p></dt>", "<dt><p>盖德里克庄园</p></dt>", 1)
        set_leaf(rp, "cn", PACK, PATH, one)
        out2 = dict(m.build(rp)[0][1])
        ok6 = PATH in out2 and dt_names(m, out2[PATH])[:7] == GOOD
        print(f"[灵敏度] 单个 <dt> 改错也能修回  -> {'PASS' if ok6 else 'FAIL'}")
        if not ok6:
            fails.append("单点损坏没修回")

    print()
    if fails:
        for f in fails:
            print("FAIL:", f)
        return 1
    print("双向回测 6/6 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
