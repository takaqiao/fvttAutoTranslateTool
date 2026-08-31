# -*- coding: utf-8 -*-
"""探针：i18n 全局命名空间越界

判据（与「无作用域判据地改他人的东西」同一类，只是写入点换成 game.i18n.translations）：
  Foundry 把所有已启用包的 lang 文件**合并进同一个扁平命名空间**，后加载的整键覆盖先加载的。
  于是「本模块的 cn.json 里定义了某个键」= 「对全世界宣告这个键归我」。
  合法的只有两种键：
    (a) 本模块要翻的那个包（crucible 系统 / ember 模块）自己在 en.json 里声明过的键；
    (b) 本模块自己新造、且带自有前缀的键。
  凡是 **本包英文里没有、Foundry 核心里却有** 的键，就是在覆盖核心 UI —— 无判据、静默。

用法：
  python i18n_scope_gate.py --cn <repo/lang/cn.json> --owner <包目录/lang/en.json> \
      --core "<Foundry core en.json>" [--owner2 ...]
只读。
"""
from __future__ import annotations
import argparse, json, sys


def flat(obj, p=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flat(v, f"{p}.{k}" if p else k))
    elif isinstance(obj, str):
        out[p] = obj
    return out


def lookup(tr, key):
    """复刻 foundry.utils.getProperty：整键优先，再点号下探。"""
    if key in tr:
        return tr[key] if isinstance(tr[key], str) else None
    node = tr
    for seg in key.split('.'):
        if not isinstance(node, dict) or seg not in node:
            return None
        node = node[seg]
    return node if isinstance(node, str) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cn", required=True)
    ap.add_argument("--owner", required=True, action="append",
                    help="本模块负责翻译的那个包的 lang/en.json（可多次）")
    ap.add_argument("--core", required=True)
    a = ap.parse_args()

    cn_raw = json.load(open(a.cn, encoding="utf-8-sig"))
    cn = flat(cn_raw)
    owners = [json.load(open(p, encoding="utf-8-sig")) for p in a.owner]
    core = json.load(open(a.core, encoding="utf-8-sig"))

    owned, orphan, clobber = [], [], []
    for k in cn:
        if any(lookup(o, k) is not None for o in owners):
            owned.append(k)
        elif lookup(core, k) is not None:
            clobber.append(k)
        else:
            orphan.append(k)

    print(f"cn 键 {len(cn)}  |  本包拥有 {len(owned)}  |  "
          f"覆盖核心 {len(clobber)}  |  两边都没有(孤儿) {len(orphan)}")
    if clobber:
        print("\n--- 覆盖 Foundry 核心键（无作用域的全局写） ---")
        for k in sorted(clobber):
            print(f"  {k}\n      core = {lookup(core,k)!r}\n      ours = {cn[k]!r}")
    if orphan:
        print(f"\n--- 孤儿键（前 40 / 共 {len(orphan)}） ---")
        for k in sorted(orphan)[:40]:
            print(f"  {k} = {cn[k]!r}")


if __name__ == "__main__":
    main()
