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
"""
from __future__ import annotations
import argparse
import json
import os
import re

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--out')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--min-len', type=int, default=40,
                    help='英文纯文本短于此长度的不报（名字/标签改动噪声大）')
    a = ap.parse_args()

    en_dir = os.path.join(a.repo, 'compendium', 'en')
    cn_dir = os.path.join(a.repo, 'compendium', 'cn')
    old = baseline_packs(a.baseline)

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
        for path, new_en in n.items():
            old_en = o.get(path)
            if old_en is None:
                new += 1
                continue
            compared += 1
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
    print(f'  两边都有的条目 {compared}')
    print(f'  **英文变过** {len(changed)}  其中中文已存在: {len(suspect)}')
    print(f'    └ 中文长度更贴合**旧**英文（八成没跟上，优先复核）: **{len(stale)}**')
    print(f'  上游新增 {new} · 上游删除但中文还在（死文本）: {len(gone)}')
    print(f'\n最可疑的 {a.top} 条（中文更贴合旧英文，按英文变动量排）：')
    for r in stale[:a.top]:
        print(f'  EN {r["en_len_old"]}→{r["en_len_new"]}  '
              f'中文/旧={r["ratio_old"]} 中文/新={r["ratio_new"]}  {r["path"][-56:]}')
    if a.out:
        json.dump({'compared': compared, 'changed': len(changed),
                   'suspect': len(suspect), 'stale': len(stale),
                   'new': new, 'gone': gone,
                   'items': stale, 'all_changed_with_cn': suspect}, open(a.out, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
