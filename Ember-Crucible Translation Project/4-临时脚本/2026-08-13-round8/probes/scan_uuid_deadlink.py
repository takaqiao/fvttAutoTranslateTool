#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""死链判据：`@UUID[...]` / `@Embed[...]` 引用的目标（文档 id **或页内锚点**）在实际 packs / 译文里不存在。

为什么现有判据看不见这一类
--------------------------
「方括号内照抄不译」是硬规矩，`scan_markup_targets.py` 能证明括号里没被译成中文，
`scan_markup_drift.py` 的 LINK 段能证明中英两侧的标记串逐字节相同 ——
**但两者都没有问过「这个目标真的存在吗」**。目标可以照抄照抄错，上游改版也会删文档；
更隐蔽的是 `#锚点`：锚点本身照抄了，可它指的是**标题文字的 slug**，
标题一译成中文，Foundry 现算出来的 slug 就变了，锚点当场失效。
这两种坏法在所有既有闸门下都是绿的：叶子有中文（覆盖率 100%）、
标记签名一致（无漂移）、括号里没有 CJK（目标干净）。

判法
----
A 段·文档存在性
  `dump_uuid_index.mjs` 从 LevelDB packs 导出每个 `_id` 的 Document 类与父 id。
  中文侧每个目标都拿它解析：id 要存在、Document 类要对得上、`A.x.B.y` 里
  子文档得真的挂在父文档下面。

B 段·锚点存在性
  `#slug` 由 Foundry `JournalEntryPage.slugifyHeading()` 从**标题文字**现算
  （`client/documents/journal-entry-page.mjs`：`slug = heading.id || slugifyHeading(heading)`），
  只有 `type === "text"` 的页面才有 toc（其余类型 `toc` 直接返回 `{}`，锚点在任何语言下都是空转）。
  所以：英文标题能算出该 slug、中文标题算不出 → 中文侧锚点死了。
  本脚本内的 slugify 是 Foundry `String#slugify` + `slugifyHeading` 的逐行复刻
  （含它的 522 条 CHAR_MAP，`&`→`and` 这种映射不复刻就会误判）。

英文侧对照（判据可信的关键）
  同一条路径的**英文叶**用同一套规则再解析一遍：
    - 中文死、且**同一个字符串**也在英文叶里 → `UPSTREAM`（上游自己的坏链，我们照抄无过，记 LOCAL-PATCHES）
    - 中文死、该字符串**不在**英文叶里     → `OURS`（我们抄错/改坏了）
    - 锚点在英文标题里算得出、中文标题算不出 → `OURS`（翻译把锚点的落点弄没了）
  只有 `OURS` 是本项目的缺陷。

用法：
  node dump_uuid_index.mjs --package <foundry 包目录> --out idx_x.json      # 每个包各跑一次
  python scan_uuid_deadlink.py --repo 1-Ember汉化插件 --repo 2-Crucible汉化插件 \
      --index idx_ember.json --index idx_crucible.json --index idx_dnd5e.json \
      --out report.json
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- markup ----

# 只取「括号体以文档引用开头」的 enricher。
# `@ref[actor.name]` / `@Condition[prone]` / `@Spell[flame.arrow]` / `@Advantage[2]`
# 是 slug / 模板类 enricher，不是 UUID，故意排除。
TARGET_RE = re.compile(r'@(UUID|Embed|embed|Action)\[([^\]]*)\]')

# 本库出现过的 UUID 文档词。`Level` 是 Ember 自己的 Scene 子文档。
DOC_TOKENS = {
    'Actor', 'Item', 'JournalEntry', 'JournalEntryPage', 'Scene', 'RollTable',
    'TableResult', 'Macro', 'Folder', 'ActiveEffect', 'Playlist', 'PlaylistSound',
    'Cards', 'Card', 'Combat', 'Combatant', 'AmbientLight', 'AmbientSound',
    'Note', 'Region', 'RegionBehavior', 'Token', 'Tile', 'Drawing', 'Wall',
    'MeasuredTemplate', 'Level', 'JournalEntryCategory', 'Adventure', 'Activity',
}

# --------------------------------------------------------------- slugify ----

with open(os.path.join(HERE, 'foundry_char_map.json'), encoding='utf-8') as _f:
    CHAR_MAP = json.load(_f)          # 由 Foundry common/primitives/string.mjs 原样导出

_WS_DASH = re.compile(r'[\s-]+')


def foundry_slugify(text: str) -> str:
    """复刻 Foundry `String#slugify()`（replacement='-', strict=false, lowercase=true）。"""
    s = ''.join(CHAR_MAP.get(c, c) for c in text).strip()
    s = s.lower()
    return _WS_DASH.sub('-', s)


def slugify_heading(text: str) -> str:
    """复刻 `JournalEntryPage.slugifyHeading()`：slugify → 去引号 → 截 64。"""
    return foundry_slugify(text).replace('"', '').replace("'", '')[:64]


_HEADING = re.compile(r'<h([1-6])\b[^>]*>(.*?)</h\1\s*>', re.S | re.I)
_TAG = re.compile(r'<[^>]+>')
_ENTITY = {'&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
           '&quot;': '"', '&#39;': "'", '&rsquo;': '’', '&mdash;': '—',
           '&ndash;': '–', '&hellip;': '…'}


def _unescape(s: str) -> str:
    for k, v in _ENTITY.items():
        s = s.replace(k, v)
    return s


def heading_slugs(html: str) -> dict:
    """{slug: 标题原文}，模拟 `buildTOC` 对 text.content 的遍历。

    带 `data-notoc` 的标题 Foundry 会跳过（`!("noToc" in element.dataset)`）。
    """
    out = {}
    for m in _HEADING.finditer(html or ''):
        if re.search(r'data-no-?toc', m.group(0), re.I):
            continue
        text = _unescape(_TAG.sub('', m.group(2)))
        slug = slugify_heading(text)
        if slug and slug not in out:
            out[slug] = text.strip()
    return out


# ------------------------------------------------------------- pack index ---

def load_indices(paths):
    ids = defaultdict(list)
    packs = {}
    known = set()
    for p in paths:
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
        pkg = d['package']['id']
        known.add(pkg)
        for pack, classes in d['packs'].items():
            packs[(pkg, pack)] = classes
        for i, entries in d['ids'].items():
            ids[i].extend(entries)
    return ids, packs, known


def leaves(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    elif isinstance(node, str):
        out['.'.join(path)] = node


def extract_targets(text):
    for m in TARGET_RE.finditer(text):
        kind, body = m.group(1), m.group(2)
        if kind == 'UUID':
            tgt = body.strip()
        else:
            tgt = body.strip().split(' ')[0] if body.strip() else ''
            if kind == 'Action' and not tgt.startswith('Compendium.'):
                continue            # `@Action[default defend]` 是 slug 不是 uuid
        if tgt:
            yield m.group(0), tgt


# ------------------------------------------------------- babele page index --

def index_pages(obj, path, acc):
    """(journal 英文名, page 英文名) -> babele 文件里的路径。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == 'pages' and isinstance(v, dict) and path:
                for pn in v:
                    acc.setdefault((path[-1], pn), []).append(path + ['pages', pn])
            index_pages(v, path + [k], acc)


def dig(obj, path):
    for p in path:
        if isinstance(obj, dict) and p in obj:
            obj = obj[p]
        else:
            return None
    return obj


# ------------------------------------------------------------- resolution ---

class Resolver:
    def __init__(self, ids, packs, known, pagebooks):
        self.ids, self.packs, self.known = ids, packs, known
        self.pagebooks = pagebooks       # [(repo, fn, en_json, cn_json, {(j,p): [path]})]

    def _entries(self, doc_id, pkg=None, pack=None):
        out = self.ids.get(doc_id, [])
        if pkg is not None:
            out = [e for e in out if e['pkg'] == pkg and e['pack'] == pack]
        return out

    # -- A 段：文档存在性 ---------------------------------------------------
    def resolve_doc(self, target):
        """-> (status, detail, page_id|None)。status: OK / UNVERIFIABLE / 失败码。"""
        tgt = target.split('#', 1)[0]
        if not tgt:
            return 'OK', 'anchor-only', None
        if ' ' in tgt or '[' in tgt or ']' in tgt or '}' in tgt:
            return 'MALFORMED', '目标里混进了空格或括号', None
        parts = tgt.split('.')

        pkg = pack = None
        if parts[0] == 'Compendium':
            if len(parts) < 4:
                return 'MALFORMED', 'Compendium uuid 段数不足', None
            pkg, pack, parts = parts[1], parts[2], parts[3:]
            if pkg not in self.known:
                return 'UNVERIFIABLE', f'包 "{pkg}" 未索引（未安装）', None
            if (pkg, pack) not in self.packs:
                return 'NO_PACK', f'不存在合集 "{pkg}.{pack}"', None
            if len(parts) == 1:                      # 旧式 Compendium.pkg.pack.<id>
                return (('OK', 'legacy', parts[0]) if self._entries(parts[0], pkg, pack)
                        else ('DEAD_ID', f'{parts[0]} 不在 {pkg}.{pack} 里', None))
        elif parts[0] == '':
            rest = [p for p in parts if p]
            if not rest:
                # `@UUID[.#anchor]` —— Foundry `_resolveRelativeUuid` 对光杆 "." 返回
                # `parseUuid(relative.uuid)`，也就是**当前这一页**。是合法的同页锚点链接。
                return 'SELF_ANCHOR', '同页锚点', None
            # 相对 uuid（`.<id>`）运行时按所在文档解析，静态只能验存在性
            return (('OK', 'relative', rest[-1]) if self._entries(rest[-1])
                    else ('DEAD_ID', f'相对目标 {rest[-1]} 在任何包里都不存在', None))

        if len(parts) % 2:
            return 'MALFORMED', f'"{tgt}" 的段数是奇数', None
        chain = [(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]
        for tok, _ in chain:
            if tok not in DOC_TOKENS:
                return 'MALFORMED', f'不认识的文档词 "{tok}"', None

        prev = None
        for depth, (tok, doc_id) in enumerate(chain):
            entries = self._entries(doc_id, pkg, pack) if (depth == 0 and pkg) \
                else self._entries(doc_id)
            if not entries:
                return ('DEAD_ID' if depth == len(chain) - 1 else 'DEAD_MID',
                        f'{tok}.{doc_id} 在任何包里都不存在', None)
            if not any(e['doc'] == tok for e in entries):
                return ('WRONG_TYPE',
                        f'{doc_id} 实际是 {sorted({e["doc"] for e in entries})}，uuid 写的是 {tok}',
                        None)
            if prev is not None and not any(e['doc'] == tok and e['parent'] == prev
                                            for e in entries):
                return ('WRONG_PARENT',
                        f'{tok}.{doc_id} 并不在 {prev} 下面'
                        f'（实际父文档 {sorted({str(e["parent"]) for e in entries})}）',
                        None)
            prev = doc_id
        return 'OK', '', chain[-1][1]

    # -- B 段：锚点存在性 ---------------------------------------------------
    def resolve_anchor(self, page_id, anchor, ref_pack):
        """-> (status, detail, extra)。status: OK / ANCHOR_INERT / ANCHOR_DEAD_EN /
        ANCHOR_DEAD_CN / ANCHOR_UNRESOLVED。

        `ref_pack` 是**发出链接的那个译文文件名**。孪生包（`ember.adventure` 与
        `ember.crucible-adventure`）里 1488 个页面同名同 id，不锁定来源包就会拿
        另一个包的中文去判这个包的锚点。
        """
        ents = [e for e in self.ids.get(page_id, []) if e['doc'] == 'JournalEntryPage']
        if not ents:
            return 'ANCHOR_UNRESOLVED', f'{page_id} 不是 JournalEntryPage', {}
        same = [x for x in ents if f'{x["pkg"]}.{x["pack"]}.json' == ref_pack]
        e = same[0] if same else ents[0]
        want = f'{e["pkg"]}.{e["pack"]}.json'
        if e['type'] != 'text':
            # Foundry: `get toc() { if (this.type !== "text") return {}; }`
            # 非 text 页面根本没有 toc，锚点在中英两侧都是空转 —— 不是我们弄坏的
            return 'ANCHOR_INERT', f'页面类型是 {e["type"]}，Foundry 不给它建 toc', {}
        jents = self.ids.get(e['parent'], [])
        jname = jents[0]['name'] if jents else None
        pname = e['name']
        books = [b for b in self.pagebooks if b[1] == want] or self.pagebooks
        for repo, fn, en, cn, pages in books:
            path = pages.get((jname, pname))
            if not path:
                continue
            en_node, cn_node = dig(en, path[0]), dig(cn, path[0])
            if not isinstance(en_node, dict):
                continue
            en_body = en_node.get('text') or ''
            cn_body = (cn_node or {}).get('text') or ''
            if not cn_body:
                # 该页正文没有译文 -> Babele 回退到英文 -> 锚点照样能命中，不是缺陷
                return ('ANCHOR_UNRESOLVED',
                        f'{jname} / {pname} 的正文没有中文，运行时回退英文', {})
            en_slugs = heading_slugs(en_body)
            cn_slugs = heading_slugs(cn_body)
            info = {'page_pack': fn, 'page_repo': repo,
                    'page_path': '.'.join(path[0]),
                    'journal': jname, 'page': pname,
                    'en_heading': en_slugs.get(anchor),
                    'cn_slugs': list(cn_slugs)[:12]}
            if anchor not in en_slugs:
                return 'ANCHOR_DEAD_EN', f'英文标题里也算不出 "{anchor}"', info
            if anchor in cn_slugs:
                return 'OK', '', info
            return ('ANCHOR_DEAD_CN',
                    f'英文标题「{en_slugs[anchor]}」的 slug 是 "{anchor}"，'
                    f'中文标题算出来的 slug 里没有它 —— 点链接只会打开页面，不再跳到该小节',
                    info)
        return 'ANCHOR_UNRESOLVED', f'译文文件里找不到 {jname} / {pname}', {}

    def resolve_self_anchor(self, ref_pack, leaf_path, anchor):
        """`@UUID[.#anchor]`：锚点落在**这条叶子所属的那一页**上。"""
        for repo, fn, en, cn, pages in self.pagebooks:
            if fn != ref_pack:
                continue
            for (jname, pname), paths in pages.items():
                for path in paths:
                    prefix = '.'.join(path)
                    if not leaf_path.startswith(prefix + '.'):
                        continue
                    en_node, cn_node = dig(en, path), dig(cn, path)
                    if not isinstance(en_node, dict):
                        continue
                    en_body = en_node.get('text') or ''
                    cn_body = (cn_node or {}).get('text') or ''
                    if not cn_body:
                        return 'ANCHOR_UNRESOLVED', f'{jname} / {pname} 正文无中文', {}
                    en_slugs = heading_slugs(en_body)
                    cn_slugs = heading_slugs(cn_body)
                    info = {'page_pack': fn, 'page_repo': repo, 'page_path': prefix,
                            'journal': jname, 'page': pname,
                            'en_heading': en_slugs.get(anchor),
                            'cn_slugs': list(cn_slugs)[:12]}
                    if anchor not in en_slugs:
                        return 'ANCHOR_DEAD_EN', f'英文标题里也算不出 "{anchor}"', info
                    if anchor in cn_slugs:
                        return 'OK', '', info
                    return ('ANCHOR_DEAD_CN',
                            f'同页锚点：英文标题「{en_slugs[anchor]}」的 slug 是 "{anchor}"，'
                            f'中文标题算出来的 slug 里没有它', info)
        return 'ANCHOR_UNRESOLVED', f'定位不到 {leaf_path} 所属的页面', {}


# ------------------------------------------------------------------ scan ----

def scan_repo(repo, resolver):
    cn_dir = os.path.join(repo, 'compendium', 'cn')
    en_dir = os.path.join(repo, 'compendium', 'en')
    name = os.path.basename(os.path.normpath(repo))
    rows, stats = [], Counter()
    for fn in sorted(f for f in os.listdir(cn_dir) if f.endswith('.json')):
        with open(os.path.join(cn_dir, fn), encoding='utf-8') as f:
            cn = {}
            leaves(json.load(f), [], cn)
        en = {}
        p = os.path.join(en_dir, fn)
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as f:
                leaves(json.load(f), [], en)
        for path, cn_text in cn.items():
            cn_targets = list(extract_targets(cn_text))
            if not cn_targets:
                continue
            en_text = en.get(path, '')
            en_set = {t for _, t in extract_targets(en_text)}
            for raw, tgt in cn_targets:
                stats['targets'] += 1
                status, detail, page_id = resolver.resolve_doc(tgt)
                extra = {}
                if status == 'OK' and '#' in tgt and page_id:
                    anchor = tgt.split('#', 1)[1]
                    status, detail, extra = resolver.resolve_anchor(page_id, anchor, fn)
                elif status == 'SELF_ANCHOR':
                    if '#' not in tgt:
                        status, detail = 'MALFORMED', '光杆 "." 目标，没有锚点'
                    else:
                        status, detail, extra = resolver.resolve_self_anchor(
                            fn, path, tgt.split('#', 1)[1])
                stats[status] += 1
                if status in ('OK', 'UNVERIFIABLE'):
                    continue
                if status == 'ANCHOR_DEAD_CN':
                    side = 'OURS'          # 英文侧活、中文侧死，定义上就是我们弄的
                elif status in ('ANCHOR_INERT', 'ANCHOR_DEAD_EN', 'ANCHOR_UNRESOLVED'):
                    side = 'UPSTREAM'
                else:
                    side = 'UPSTREAM' if tgt in en_set else 'OURS'
                rows.append({
                    'repo': name, 'pack': fn, 'path': path,
                    'batch_path': path[len('entries.'):] if path.startswith('entries.') else path,
                    'status': status, 'side': side, 'detail': detail,
                    'target': tgt, 'markup': raw[:200],
                    'en_has_same_target': tgt in en_set,
                    'cn_excerpt': cn_text[:500], 'en_excerpt': en_text[:500],
                    **extra,
                })
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--index', action='append', required=True)
    ap.add_argument('--out')
    ap.add_argument('--root', default=os.getcwd())
    a = ap.parse_args()

    ids, packs, known = load_indices(a.index)

    pagebooks = []
    repos = [r if os.path.isabs(r) else os.path.join(a.root, r) for r in a.repo]
    for repo in repos:
        en_dir = os.path.join(repo, 'compendium', 'en')
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith('.json') or fn == '_source.json':
                continue
            with open(os.path.join(en_dir, fn), encoding='utf-8') as f:
                en = json.load(f)
            cnp = os.path.join(repo, 'compendium', 'cn', fn)
            cn = {}
            if os.path.isfile(cnp):
                with open(cnp, encoding='utf-8') as f:
                    cn = json.load(f)
            acc = {}
            index_pages(en, [], acc)
            if acc:
                pagebooks.append((os.path.basename(os.path.normpath(repo)), fn, en, cn, acc))

    resolver = Resolver(ids, packs, known, pagebooks)

    all_rows, total = [], Counter()
    for repo in repos:
        rows, stats = scan_repo(repo, resolver)
        all_rows += rows
        total.update(stats)
        print(f'{os.path.basename(os.path.normpath(repo))}: targets={stats["targets"]} '
              f'ok={stats["OK"]} bad={len(rows)}')

    ours = [r for r in all_rows if r['side'] == 'OURS']
    up = [r for r in all_rows if r['side'] == 'UPSTREAM']
    print(f'\nTOTAL targets={total["targets"]}  OURS={len(ours)}  UPSTREAM={len(up)}')
    print('by status:', dict(Counter(f'{r["side"]}/{r["status"]}' for r in all_rows)))

    uniq = {}
    for r in ours:
        uniq.setdefault((r['status'], r['target']), []).append(r)
    print(f'unique OURS targets: {len(uniq)}')
    for (st, tgt), rs in list(uniq.items())[:40]:
        print(f'  [{st} x{len(rs)}] {tgt}\n      {rs[0]["detail"]}')

    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump({'summary': dict(total), 'ours': ours, 'upstream': up},
                      f, ensure_ascii=False, indent=1)
        print(f'-> {a.out}')
    return 1 if ours else 0


sys.exit(main())
