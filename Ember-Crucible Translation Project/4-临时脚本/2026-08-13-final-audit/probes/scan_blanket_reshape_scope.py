# -*- coding: utf-8 -*-
r"""
scan_blanket_reshape_scope.py
—— 「对整个异质集合施加同一处破坏性变换，没有按成员做类型/用途判据」这一类的
   **消费端**判据探针。

抽象自已确认实例（register.js degradeActorUpdatePayload 对所有 actor 无差别删
items/effects）：面对「个别成员可能畸形」，代码对**整个异质集合**施加同一处变换，
没有按成员分辨「这个成员的消费方需要的是什么形状」。

已有的 scan_untyped_transform.py 从**生产端**（JS 函数体里有没有类型闸门）找候选；
本探针从**消费端**找：把集合里每个成员**实际被谁读、以什么形状读**列出来，
凡是「成员被以形状 A 消费、而我们统一变换成了形状 B」的，就是同一类缺陷。

本探针实现的两条子判据
----------------------
C1  lang/*.json 的「子树消费者」：
    Foundry 的 Localization.localizeSchema（foundry.mjs:204368-204378）读的是
        foundry.utils.getProperty(game.i18n.translations, `${PREFIX}.FIELDS`)
    并把结果 Object.assign 进 rules —— 它要的是一棵**对象子树**，不是叶子字符串。
    而 getProperty（foundry.mjs:2389）只有两条路：整键命中 / 按点逐级下探。
    所以只要 lang 文件被**整体拍平**成点号叶子键，这些子树消费者一律拿到 undefined。
    判据：对每个上游 DataModel 的 LOCALIZATION_PREFIXES，检查我方 lang/cn.json 里
    `getProperty(cn, PREFIX + ".FIELDS")` 是否返回 dict。

C2  「叶子查得到 ≠ 子树查得到」的自检：复刻 flatten_lang.py 的 foundry_lookup()，
    证明它只覆盖 localize() 的叶子语义，对 localizeSchema() 的子树语义无感 ——
    也就是这条缺陷为什么能骗过既有校验。

假阳性模式
----------
* 若上游 schema 字段自己写死了 label（`new StringField({label: "..."})`），
  localizeSchema 的 `this.label ||= ...` 不会覆盖，那条本来就不靠 lang —— 脚本会
  把「上游 en.json 里有、schema 里也写死」的情况另计（需人工抽查）。
* 若某 PREFIX 在我方 lang 里压根没有 FIELDS 条目，那不是本缺陷（是没译），单独计数。
* Foundry 允许 flat key 与嵌套并存；只有**两条路都断**才算死。脚本按 getProperty
  原样复刻，不用自己的一套。

只读。不写任何仓库文件。
"""
from __future__ import annotations
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
FOUNDRY = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
CRUCIBLE = os.path.join(FOUNDRY, "systems", "crucible")
EMBER = os.path.join(FOUNDRY, "modules", "ember")

REPOS = {
    "crucible-cn": (os.path.join(ROOT, "2-Crucible汉化插件"),
                    os.path.join(CRUCIBLE, "lang", "en.json")),
    "ember_cn_unofficial": (os.path.join(ROOT, "1-Ember汉化插件"),
                            os.path.join(EMBER, "lang", "en.json")),
}


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def get_property(obj, key):
    """逐字复刻 foundry.mjs:2389 getProperty。"""
    if not key or obj is None:
        return None
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    target = obj
    for p in key.split("."):
        if target is None:
            return None
        if not isinstance(target, (dict, list)):
            return None
        if isinstance(target, dict) and p in target:
            target = target[p]
        else:
            return None
    return target


def foundry_lookup(tr, key):
    """flatten_lang.py 自带的校验函数（叶子语义）。"""
    if key in tr:
        v = tr[key]
        return v if isinstance(v, str) else None
    node = tr
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node if isinstance(node, str) else None


PREFIX_RE = re.compile(r"LOCALIZATION_PREFIXES\s*=\s*\[([^\]]*)\]")
STR_RE = re.compile(r"""['"]([^'"]+)['"]""")


def collect_prefixes(paths):
    """从上游源码里抓所有 static LOCALIZATION_PREFIXES。"""
    out = {}
    for base in paths:
        for dirpath, _dirs, files in os.walk(base):
            for fn in files:
                if not fn.endswith((".mjs", ".js")):
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    txt = open(fp, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                for m in PREFIX_RE.finditer(txt):
                    for s in STR_RE.findall(m.group(1)):
                        out.setdefault(s, []).append(os.path.relpath(fp, base))
    return out


def main():
    print("=" * 78)
    print("C1  lang/cn.json 的子树消费者（Localization.localizeSchema）")
    print("=" * 78)

    prefixes = collect_prefixes([os.path.join(CRUCIBLE, "module"),
                                 os.path.join(EMBER, "scripts")])
    print(f"上游源码里抓到 LOCALIZATION_PREFIXES 共 {len(prefixes)} 个不同前缀")

    grand = 0
    for repo_id, (repo, en_path) in REPOS.items():
        cn_path = os.path.join(repo, "lang", "cn.json")
        cn = load(cn_path)
        en = load(en_path) if os.path.exists(en_path) else {}
        nested = sum(1 for v in cn.values() if isinstance(v, dict))
        print(f"\n[{repo_id}] {cn_path}")
        print(f"  顶层键 {len(cn)}，其中值为对象的 {nested}  "
              f"（0 = 完全拍平）")
        print(f"  上游 en.json 顶层键 {len(en)}，其中值为对象的 "
              f"{sum(1 for v in en.values() if isinstance(v, dict))}")

        dead = alive = untranslated = 0
        rows = []
        for prefix in sorted(prefixes):
            key = f"{prefix}.FIELDS"
            cn_sub = get_property(cn, key)
            en_sub = get_property(en, key)
            # 我方有没有为这个前缀写过 FIELDS 译文（哪怕是拍平后的点号键）
            flat_leaves = [k for k in cn if k.startswith(key + ".")]
            if not flat_leaves and not isinstance(cn_sub, dict):
                untranslated += 1
                continue
            if isinstance(cn_sub, dict):
                alive += 1
                rows.append((prefix, "OK-subtree", len(flat_leaves), en_sub is not None))
            else:
                dead += 1
                rows.append((prefix, "DEAD-flattened", len(flat_leaves), en_sub is not None))
                grand += len(flat_leaves)
        for prefix, state, n, upstream in rows:
            print(f"    {state:<16} {prefix:<28} 拍平后的点号叶子 {n:>4}"
                  f"   上游同前缀 FIELDS 存在={upstream}")
        print(f"  子树可用 {alive} / 子树已死 {dead} / 未译（不计） {untranslated}")

    print(f"\nC1 合计：因整体拍平而**永远读不到**的 FIELDS 叶子 {grand} 条")

    print()
    print("=" * 78)
    print("C2  为什么既有校验发现不了：flatten_lang.py 的 foundry_lookup 只有叶子语义")
    print("=" * 78)
    for repo_id, (repo, en_path) in REPOS.items():
        cn = load(os.path.join(repo, "lang", "cn.json"))
        en = load(en_path) if os.path.exists(en_path) else {}

        def flat_keys(o, p=""):
            for k, v in o.items():
                kk = f"{p}.{k}" if p else k
                if isinstance(v, dict):
                    yield from flat_keys(v, kk)
                else:
                    yield kk

        en_leaves = list(flat_keys(en))
        leaf_ok = sum(1 for k in en_leaves if foundry_lookup(cn, k))
        sub_keys = [f"{pfx}.FIELDS" for pfx in prefixes]
        sub_ok = sum(1 for k in sub_keys if isinstance(get_property(cn, k), dict))
        print(f"[{repo_id}] 英文叶子 {len(en_leaves)}；flatten_lang 口径「查得到」"
              f" {leaf_ok}  →  它报的是绿的")
        print(f"          而子树口径（localizeSchema 真正用的）：{sub_ok}/"
              f"{len(sub_keys)} 个 PREFIX.FIELDS 能拿到对象")


if __name__ == "__main__":
    main()
