# -*- coding: utf-8 -*-
"""lang/cn.json 缺口计算（与 compendium 相互独立的一条工作线）

三方 diff：
  旧版英文 = <repo>/lang/en.json          （上次发版时存档的基准，进 git）
  新版英文 = <foundry包>/lang/en.json      （当前安装的系统/模块）
  译文     = <repo>/lang/cn.json

输出四类：
  NEW          新版有、旧版没有 → 要翻
  DRIFT        两版都有但英文原文变了 → 要重翻
  UNTRANSLATED cn 与 en 完全相同（且含英文字母、无中文）→ 漏翻
  STALE        cn 里有、新版英文里没有 → 上游已删除，可清

用法：
  python lang_gap.py --repo <repo目录> --package <foundry包目录> --out <报告目录>
  python lang_gap.py ... --sync-baseline     # 顺带把 <repo>/lang/en.json 更新成新版
"""
import argparse
import json
import re
from pathlib import Path

CJK = re.compile(r'[一-鿿㐀-䶿]')
LATIN = re.compile(r'[A-Za-z]')


def flatten(obj, prefix=""):
    """lang json 是纯嵌套字典 → 扁平点分路径"""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, str):
        out[prefix] = obj
    return out


def foundry_lookup(tr, key):
    """复刻 `foundry.utils.getProperty`：先整键命中，否则按点逐级下探。

    **不要用 flatten() 判断译文有没有生效。** flatten 会把
    `"EMBER.CALENDAR": {"WORLD": "世界地图"}` 展开成 `EMBER.CALENDAR.WORLD`，
    看着齐全，但 Foundry 查的是「整键」或「逐级下探」，这种顶层带点、值是嵌套对象
    的混合形态**两条路都断**。ember 侧 486 键里 372 个就这么死掉，而校验一路报「缺口 0」。
    校验必须复刻被验证系统的查找语义。
    """
    if key in tr:
        v = tr[key]
        return v if isinstance(v, str) else None
    node = tr
    for p in key.split('.'):
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node if isinstance(node, str) else None


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def load_keep_english(path):
    """读 lang/lang_keep_english.json，返回 (keep_english:set, never_write:dict)。

    两种形态都接受：
      · 对象  {"_README": [...], "keep_english": [...], "never_write": {key: 理由}}
        —— 当前形态。keep_english = 我们**必须写上英文串**去压过上游中文；
           never_write  = 我们**一个字都不写**，让 localize() 回落英文 fallback。
      · 裸数组 ["KEY", ...] —— 历史形态，全部当 keep_english。
    其它形态 raise。**不要**退回 `set(json.load(f))`：对裸数组它拿到元素，
    对对象它静默拿到键名 `_README`/`keep_english`/`never_write`，白名单当场清空
    而没有任何报错。
    """
    if not Path(path).exists():
        return set(), {}
    doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(doc, list):
        return set(doc), {}
    if isinstance(doc, dict):
        keep = doc.get("keep_english", [])
        never = doc.get("never_write", {})
        if not isinstance(keep, list) or not isinstance(never, dict):
            raise ValueError(f"{path}: keep_english 必须是数组、never_write 必须是对象")
        return set(keep), dict(never)
    raise ValueError(f"{path}: 形态不认识（既不是数组也不是对象）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="汉化插件仓库目录")
    ap.add_argument("--package", required=True, help="Foundry 里的系统/模块目录（新版英文来源）")
    ap.add_argument("--out", required=True, help="报告输出目录")
    ap.add_argument("--lang-file", default="en.json", help="包内英文文件名，默认 en.json")
    ap.add_argument("--sync-baseline", action="store_true",
                    help="把 <repo>/lang/en.json 覆盖为新版英文（翻译完成后再做）")
    args = ap.parse_args()

    repo = Path(args.repo)
    new_en = flatten(load(Path(args.package) / "lang" / args.lang_file))
    old_en_path = repo / "lang" / "en.json"
    old_en = flatten(load(old_en_path)) if old_en_path.exists() else {}
    cn_raw = load(repo / "lang" / "cn.json")
    cn = flatten(cn_raw)

    keep_path = repo / "lang" / "lang_keep_english.json"
    keep_english, never_write = load_keep_english(keep_path)

    report = {"new": {}, "drift": {}, "untranslated": {}, "stale": [],
              "unreachable": [], "never_write": {}}

    # 有中文、但 Foundry 按自己的查找语义**取不到** → 键形态错了，等同于没译。
    report["unreachable"] = sorted(k for k in new_en if k in cn and not foundry_lookup(cn_raw, k))

    for k, en in new_en.items():
        # never_write：我们**故意**不写这个键，靠 localize() 回落英文 fallback
        # （localization.mjs:437 -> :441）。它永远「缺席」，但那不是缺口。
        if k in never_write:
            report["never_write"][k] = {"en": en, "why": never_write[k],
                                        "present_in_cn": k in cn}
            continue
        if k not in cn:
            report["new"][k] = en
            continue
        before = old_en.get(k)
        if before is not None and before != en:
            report["drift"][k] = {"en_now": en, "en_before": before, "cn": cn[k]}
            continue
        v = cn[k]
        if v == en and LATIN.search(en) and not CJK.search(v) and k not in keep_english:
            report["untranslated"][k] = en

    report["stale"] = sorted(set(cn) - set(new_en))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "lang_gap.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"新版英文 {len(new_en)} 键 / 旧版基准 {len(old_en)} 键 / 译文 {len(cn)} 键")
    print(f"  NEW          {len(report['new'])}")
    print(f"  DRIFT        {len(report['drift'])}")
    print(f"  UNTRANSLATED {len(report['untranslated'])}")
    print(f"  STALE        {len(report['stale'])}")
    print(f"  UNREACHABLE  {len(report['unreachable'])}   ← 有中文但 Foundry 查不到（键形态错）")
    print(f"  NEVER-WRITE  {len(report['never_write'])}   ← 故意不写，回落英文 fallback")
    for k, v in report["never_write"].items():
        if v["present_in_cn"]:
            print(f"    ⚠ {k} 竟然出现在 cn.json 里 —— 按 lang_keep_english.json 的约定应当删掉")
    print(f"  → {outdir / 'lang_gap.json'}")

    if args.sync_baseline:
        src = Path(args.package) / "lang" / args.lang_file
        old_en_path.write_text(src.read_text(encoding="utf-8-sig"), encoding="utf-8")
        print(f"  基准已更新：{old_en_path}")


if __name__ == "__main__":
    main()
