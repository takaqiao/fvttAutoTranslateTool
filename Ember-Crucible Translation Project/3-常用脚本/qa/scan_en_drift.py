#!/usr/bin/env python3
"""拿**旧版英文基准**和**当前英文**逐条比，精确标出「英文变过、而中文可能没跟上」的条目。

  python scan_en_drift.py --repo <repo> --baseline <旧基准目录> [--out <json>] [--top N]

为什么这条路优于所有启发式
--------------------------
此前找「中文停在旧版英文上」靠的都是间接信号，各有盲区：

* `measure_8c.py` / `measure_stale_extra.py` 只比 `<p>`/`<li>` **块数** ——
  上游换内容但块数不变时完全沉默。
* `scan_markup_drift.py` 的 `TRUNCATED` 只在中文短到 **0.22 倍**以下才响 ——
  上游把英文改写而长度相当时同样沉默。
* 标记签名只有在**标记**跟着变时才响。

而旧版英文是**直接证据**：`EN_old != EN_new` 就说明这一条上游动过；若该条中文是在
旧英文时期写的，它十有八九没跟上。这不是猜，是比对。

本项目尤其需要它，因为库里有大量**移植/继承**来的译文：孪生包 1.4 万条整条复制自
crucible 侧、`Arcturel Tradeway` 28 页从改名前的旧路径搬来、v1.0.15 的存量译文。
这些都没有针对今天的英文逐句核对过。

手上现有的基准
--------------
    5-其他内容/english-baseline/crucible-0.9.1-legacy      ← crucible 旧版
    5-其他内容/english-baseline/ember-cn-v1.0.15-shipped-en ← ember 旧版（随 v1.0.15 发布）

输出分三档，按该管的先后排：

* `CHANGED`   —— 英文变了、中文存在。**要复核**，这是本工具的主产物。
* `NEW`       —— 旧基准里没有这条（上游新增）。若中文也有，多半是后来补译的，风险低。
* `GONE`      —— 旧基准有、今天没有（上游删了）。中文若还在就是死文本。

⚠ 基准缺包 = 静默不扫（第十九轮 Y5 的同型盲区，第二十轮补上）
--------------------------------------------------------------
本闸只扫**基准目录里存在**的包：下面 `for pack, oldpath in old.items()` 的定义域就是
基准目录，基准里没有的包连循环都进不去，而报告长得和「扫过、干净」一模一样。
实测 2026-08-15：`--repo 1-Ember汉化插件 --baseline ember-cn-v1.0.15-shipped-en`
→ changed 922 条**全部**落在 crucible-adventure(906) + crucible-character(16)，
`ember.adventure.json` 一条也没有 —— 因为那份基准只有 **3** 个包
（`_repaired.json` + adversary + character），仓里却有 10 个 en 包 / 9 个有中文。
本闸是 §5.0 整节与跨版本复核的**主判据**，这个洞比 Y5 更要紧。

现在每次运行都印「本次扫 N 个包 / 仓里共 M 个包 / 缺 K 个」以及缺掉的中文叶数；
`--strict-coverage` 让缺包直接以退出码 3 结束。

**这个洞补不了「过去」，只补得了「将来」**：基准是历史快照，那 6 个包发版当时就没被
捕获。所以第十九轮起改为**升级前先截全包快照**
（`3-常用脚本/qa/capture_baseline.py`）：
`english-baseline/ember-0.6.0-preupgrade-2026-08-15/`（10 包）、
`crucible-0.10.1-preupgrade-2026-08-15/`（15 包）。
拿这两份跑**今天**恒为 0 变动（它们就是当前英文），那是构造性的、不算「闸绿了」；
它们的用途是下一次上游升级之后当「旧英文」。
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

# 覆盖行里有 ⚠ / **；GBK 控制台会 UnicodeEncodeError 把整个脚本打断（实测 2026-08-15）。
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CJK = re.compile(r'[一-鿿]')
TAG = re.compile(r'<[^>]+>')


def load_json(path):
    """旧基准里混着 BOM 和一处尾逗号（当年那个损坏的 -en.json），都兜住。"""
    raw = open(path, encoding='utf-8-sig').read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r',(\s*[}\]])', r'\1', raw)
        return json.loads(fixed)


def leaves(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out[('.'.join(path))] = node


def norm(s):
    """比对前只抹掉空白差异 —— 缩进/换行的变动不算内容变了。"""
    return re.sub(r'\s+', ' ', s).strip()


def baseline_packs(bdir):
    """旧基准的文件名 -> 当前包名。v1.0.15 那份用了 `-en` 后缀，战役正文在 _repaired.json。"""
    out = {}
    # sorted() 让 `_repaired.json` 排在 `ember.*-en.json` 前面，配合 setdefault =
    # **先到先得**。这很重要：v1.0.15 发布包里的 `ember.crucible-adventure-en.json`
    # 本身就是损坏 JSON（PROJECT 第 3.5 节记过），`_repaired.json` 才是修好的那份。
    # 用赋值而非 setdefault 会让损坏的那个把修好的覆盖掉。
    for f in sorted(os.listdir(bdir)):
        if not f.endswith('.json') or f == '_source.json':
            continue
        key = ('ember.crucible-adventure.json' if f == '_repaired.json'
               else f.replace('-en.json', '.json'))
        out.setdefault(key, os.path.join(bdir, f))
    return out


# ----------------------------------------------------------- 基准覆盖了几个包
def cjk_leaf_count(path):
    """一个 cn 包里**含中文**的叶数 —— 本闸能报「中文没跟上」的定义域就是这些叶。"""
    if not os.path.exists(path):
        return 0
    d = {}
    try:
        leaves(load_json(path).get('entries', {}), [], d)
    except Exception:
        return 0
    return sum(1 for v in d.values() if CJK.search(v))


def coverage(repo, baseline):
    """扫了几个包 / 仓里共几个包 / 基准缺几个包（连带缺掉多少条中文叶）。

    ⚠ **反空转第 (d) 型的探针**。前三种空转形态（判据写坏 / 压根不读库 / 读旧报告
    快照）的防法对它全部无效：判据没问题、库也读了、报告是当场生成的 —— 只是基准
    目录里**压根没有那个包**，`for pack, oldpath in old.items()` 根本不会遍历到它，
    报告上一片绿，实情是没扫。

    与 `scan_number_drift.coverage()` / `scan_dropped_terms.coverage()` /
    `scan_marker_followup.coverage()` 同一套语义（第十九轮实装，第二十轮照搬到本闸
    与 `scan_renamed_terms.py`）。所以这三个数必须每次跟着结果一起印，不能只印
    changed / stale 数。
    """
    en_dir = os.path.join(repo, 'compendium', 'en')
    cn_dir = os.path.join(repo, 'compendium', 'cn')
    bmap = baseline_packs(baseline)
    repo_packs = sorted(f for f in os.listdir(en_dir)
                        if f.endswith('.json') and f != '_source.json')
    scanned, missing = [], []
    for f in repo_packs:
        p = bmap.get(f)
        if p and os.path.exists(p):
            scanned.append(f)
        else:
            cnp = os.path.join(cn_dir, f)
            missing.append({'pack': f, 'cn_present': os.path.exists(cnp),
                            'cn_cjk_leaves': cjk_leaf_count(cnp)})
    return {
        'baseline': os.path.abspath(baseline),
        'repo_en_packs': len(repo_packs),
        'scanned_packs': scanned,
        'missing_packs': missing,
        'cn_cjk_leaves_uncovered': sum(m['cn_cjk_leaves'] for m in missing),
        # 基准里有、仓里已经没有的包（上游删包 / 改名）。不影响本次结果，但要看得见。
        'baseline_only_packs': [f for f in sorted(bmap) if f not in repo_packs],
    }


def print_coverage(cov):
    print(f"  包覆盖：本次扫 {len(cov['scanned_packs'])} 个包 / 仓里 en 共 "
          f"{cov['repo_en_packs']} 个包 / 基准缺 {len(cov['missing_packs'])} 个")
    if cov['missing_packs']:
        print('  ⚠ 下列包**基准里没有，本闸一条也没看** —— 它们的 0 变动不是「干净」，是「没扫」：')
        for m in cov['missing_packs']:
            tail = '' if m['cn_present'] else '（无 cn 文件）'
            print(f"       {m['pack']}  含中文的叶 {m['cn_cjk_leaves']}{tail}")
        print(f"     合计未进闸的中文叶 {cov['cn_cjk_leaves_uncovered']}")
        print('     历史快照补不回来（那些包发版当时就没捕获）；将来靠 '
              '3-常用脚本/qa/capture_baseline.py 在**升级前**截全包基准。')
    if cov['baseline_only_packs']:
        print(f"  ⚠ 基准里有、仓里 en 已没有的包 {len(cov['baseline_only_packs'])} 个："
              f"{'、'.join(cov['baseline_only_packs'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--out')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--min-len', type=int, default=40,
                    help='英文纯文本短于此长度的不报（名字/标签改动噪声大）')
    ap.add_argument('--strict-coverage', action='store_true',
                    help='基准缺包时以退出码 3 结束（默认只告警：回测脚本会拿单包小仓跑，'
                         '默认非零会把它们全打红）')
    a = ap.parse_args()

    en_dir = os.path.join(a.repo, 'compendium', 'en')
    cn_dir = os.path.join(a.repo, 'compendium', 'cn')
    cov = coverage(a.repo, a.baseline)
    old = baseline_packs(a.baseline)
    per_pack = {}

    changed, new, gone = [], 0, []
    compared = 0
    for pack, oldpath in old.items():
        cur_p = os.path.join(en_dir, pack)
        if not os.path.exists(cur_p):
            continue
        o, n, c = {}, {}, {}
        leaves(load_json(oldpath).get('entries', {}), [], o)
        leaves(load_json(cur_p).get('entries', {}), [], n)
        cnp = os.path.join(cn_dir, pack)
        if os.path.exists(cnp):
            leaves(load_json(cnp).get('entries', {}), [], c)
        pp = per_pack.setdefault(pack, {'baseline_leaves': len(o),
                                        'cur_en_leaves': len(n),
                                        'cn_leaves': len(c),
                                        'pairs': 0, 'changed': 0})
        for path, new_en in n.items():
            old_en = o.get(path)
            if old_en is None:
                new += 1
                continue
            compared += 1
            pp['pairs'] += 1
            if norm(old_en) == norm(new_en):
                continue
            cn = c.get(path)
            plain_new = TAG.sub('', new_en)
            if len(plain_new) < a.min_len:
                continue
            lo, ln = len(TAG.sub('', old_en)), len(plain_new)
            lc = len(TAG.sub('', cn)) if cn else 0
            # 中文更贴合哪一版英文？本库译文/英文纯文本长度比中位数 0.31。
            # 若中文长度按旧英文算更接近 0.31，说明它八成还停在旧版上 —— 这是把
            # 「早已重译好」的条目从「真没跟上」里分出来的关键判别，否则 921 条无从下手。
            fits_old = abs(lc / max(lo, 1) - 0.31) < abs(lc / max(ln, 1) - 0.31)
            pp['changed'] += 1
            changed.append({
                'pack': pack, 'path': path,
                'has_cn': bool(cn and CJK.search(cn)),
                'fits_old': bool(cn and fits_old),
                'en_len_old': lo, 'en_len_new': ln, 'cn_len': lc,
                'ratio_old': round(lc / max(lo, 1), 2), 'ratio_new': round(lc / max(ln, 1), 2),
                'delta': ln - lo,
                'old_en': old_en[:220], 'new_en': new_en[:220],
                'cn': (cn or '')[:220],
            })
        for path in o:
            if path not in n and c.get(path) and CJK.search(c[path]):
                gone.append({'pack': pack, 'path': path})

    suspect = [r for r in changed if r['has_cn']]
    stale = [r for r in suspect if r['fits_old']]
    suspect.sort(key=lambda r: -abs(r['delta']))
    stale.sort(key=lambda r: -abs(r['delta']))
    print(f'基准 {os.path.basename(a.baseline)} vs 当前英文')
    print(f"  baseline = {cov['baseline']}")
    print_coverage(cov)
    for pk, d in per_pack.items():
        print(f"     · {pk}  基准叶 {d['baseline_leaves']}  当前英文叶 {d['cur_en_leaves']}"
              f"  中文叶 {d['cn_leaves']}  进闸对 {d['pairs']}  英文变过 {d['changed']}")
    print(f'  两边都有的条目 {compared}')
    print(f'  **英文变过** {len(changed)}  其中中文已存在: {len(suspect)}')
    print(f'    └ 中文长度更贴合**旧**英文（八成没跟上，优先复核）: **{len(stale)}**')
    print(f'  上游新增 {new} · 上游删除但中文还在（死文本）: {len(gone)}')
    print(f'\n最可疑的 {a.top} 条（中文更贴合旧英文，按英文变动量排）：')
    for r in stale[:a.top]:
        print(f'  EN {r["en_len_old"]}→{r["en_len_new"]}  '
              f'中文/旧={r["ratio_old"]} 中文/新={r["ratio_new"]}  {r["path"][-56:]}')
    # 报告里必须带覆盖口径：只写「changed N」的报告，事后分不清是「全库 N」还是「只扫了 3 包的 N」。
    meta = {'tool': os.path.basename(__file__), 'repo': os.path.abspath(a.repo),
            'coverage': cov, 'per_pack': per_pack, 'argv': sys.argv}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        json.dump({'meta': meta,
                   'compared': compared, 'changed': len(changed),
                   'suspect': len(suspect), 'stale': len(stale),
                   'new': new, 'gone': gone,
                   'items': stale, 'all_changed_with_cn': suspect}, open(a.out, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')
    if a.strict_coverage and cov['missing_packs']:
        print(f"✗ --strict-coverage：基准缺 {len(cov['missing_packs'])} 个包，"
              f'本次结果不构成全库结论。')
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
