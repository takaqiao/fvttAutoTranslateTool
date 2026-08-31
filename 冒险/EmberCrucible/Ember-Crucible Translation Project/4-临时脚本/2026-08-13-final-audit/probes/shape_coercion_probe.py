#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""形状强转 / 失效开关 —— 全库判据探针（只读，不写任何库文件）

被举一反三的那一类缺陷有三个可机械化的特征：

  S1  写调用（落库或落盘）之前，只按 **值的形状**（typeof / isinstance / Array.isArray）
      做判断，没有按 **文档类型（item.type / documentType）或上游 schema 字段类型**
      做判断  ->  同一段代码会把不该转的东西也转了。
  S2  「只跑一次 / 别覆盖已有内容」的开关，其**存放位置或判定对象**根本不成立
      （非 Document 上 setFlag、只查叶子不查中间节点、guard 与它保护的写在同一侧等）。
  S3  写调用的**触发条件**是自动的（ready / 钩子 / 导入流程），不需要用户点任何东西。

三段扫描：

  A) 运行时写面：两个插件仓里所有会落到 Foundry 世界数据的调用点，
     逐点标注「有没有类型判据 / 有没有开关 / 开关成不成立」。
  B) 工具写面：3-常用脚本 里所有会覆写库文件的脚本，同样标注。
  C) 数据侧形状普查：EN 基线 与 CN 译文 在同一路径上的 JSON 类型是否一致。
     这是 S1 在数据侧的镜像 —— 某处 EN 是 dict、CN 是 str（或反过来）时，
     任何按点号路径写值的工具（apply_translations.set_at）都会把整节
     用 {} 顶掉，而它的「不覆盖已有中文」护栏只看叶子、看不见这一层。

用法:
  python shape_coercion_probe.py --root "<项目根>" [--json <out.json>]

已知假阳性模式（读结论时必须知道）：
  - A/B 两段是**正则定位 + 上下文窗口**，不是 AST。函数边界靠缩进/花括号近似，
    「有没有类型判据」是在写调用**上方 40 行**里找 `\.type\b` / `documentType` /
    `isinstance` 这类词，函数很长时会漏判为「无判据」（假阳性偏多，需人工过）。
  - C 段只比较 EN 基线里存在的路径。CN 侧多出来的键由 prune_dead 负责，不在这里报。
  - C 段把 EN=str / CN=None 视作「未翻译」，不算形状不一致。
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

# ---------------------------------------------------------------- A/B 段

# 会落到 Foundry 世界数据（持久化）的调用
PERSIST_JS = re.compile(
    r'\.update\s*\(|\.updateDocuments\s*\(|\.updateEmbeddedDocuments\s*\(|'
    r'\.createDocuments\s*\(|\.createEmbeddedDocuments\s*\(|\.deleteEmbeddedDocuments\s*\(|'
    r'\.setFlag\s*\(|\.updateSource\s*\(|modifyBatch\s*\(|\.delete\s*\(\s*\)')
# 会覆写库文件的调用
PERSIST_TOOL = re.compile(r'json\.dump\s*\(|writeFileSync\s*\(|\bdump\s*\(\s*\w*path|open\s*\([^)]*[\'"]w[\'"]')
# 形状强转：判断只看形状
SHAPE_ONLY = re.compile(
    r"typeof\s+\w[\w.?\[\]]*\s*[!=]==?\s*['\"](string|object|number|function)['\"]|"
    r"Array\.isArray\s*\(|isinstance\s*\(\s*\w+\s*,\s*\(?\s*(str|dict|list)")
# 真正的类型判据：按文档类型/字段名/schema 分支
TYPE_JUDGE = re.compile(
    r"\.type\s*[!=]==?|\btype\s*===|documentType|documentName|\bsubtype\b|"
    r"CONFIG\.\w+\.documentClass|_source\b|schema\b|fields\.\w+Field|"
    r"pack\.metadata|\.collection\b")
# 「只跑一次 / 别覆盖」开关
GUARD = re.compile(r'getFlag|setFlag|__\w*[Pp]atched|already|Migrated|migrated|'
                   r'\bonce\b|force|skipped_existing|--write|\bdry\b')

CODE_EXT = {'.js', '.mjs', '.py'}


def scan_code(root):
    hits = []
    targets = []
    for sub in ['1-Ember汉化插件', '2-Crucible汉化插件', '3-常用脚本']:
        d = os.path.join(root, sub)
        for dp, dns, fns in os.walk(d):
            dns[:] = [x for x in dns if x not in ('node_modules', '__pycache__', '.git',
                                                  'compendium', 'lang', 'packs')]
            for fn in fns:
                if os.path.splitext(fn)[1] in CODE_EXT:
                    targets.append(os.path.join(dp, fn))
    for p in targets:
        try:
            lines = open(p, encoding='utf-8').read().splitlines()
        except Exception:
            continue
        is_tool = os.sep + '3-常用脚本' + os.sep in p
        pat = PERSIST_TOOL if is_tool else PERSIST_JS
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s.startswith('*') or s.startswith('//') or s.startswith('#'):
                continue
            if not pat.search(ln):
                continue
            lo, hi = max(0, i - 40), min(len(lines), i + 6)
            ctx = '\n'.join(lines[lo:hi])
            hits.append({
                'file': os.path.relpath(p, root).replace('\\', '/'),
                'line': i + 1,
                'call': s[:110],
                'shape_only': bool(SHAPE_ONLY.search(ctx)),
                'type_judge': bool(TYPE_JUDGE.search(ctx)),
                'guard': sorted(set(GUARD.findall(ctx)))[:4],
                'kind': 'tool-write' if is_tool else 'world-write',
            })
    return hits


# ---------------------------------------------------------------- C 段

def jtype(v):
    if v is None:
        return 'null'
    if isinstance(v, str):
        return 'str'
    if isinstance(v, dict):
        return 'dict'
    if isinstance(v, list):
        return 'list'
    return type(v).__name__


def walk_shapes(en, cn, path, out, missing_ok=True):
    """比较 EN / CN 在同一路径上的 JSON 类型。"""
    te, tc = jtype(en), jtype(cn)
    if tc == 'null':
        return                      # 没翻译，不是形状问题
    if te != tc:
        out.append({'path': '.'.join(path), 'en': te, 'cn': tc,
                    'en_sample': (en if isinstance(en, str) else json.dumps(
                        en, ensure_ascii=False))[:90],
                    'cn_sample': (cn if isinstance(cn, str) else json.dumps(
                        cn, ensure_ascii=False))[:90]})
        return
    if te == 'dict':
        for k, v in en.items():
            walk_shapes(v, cn.get(k), path + [k], out)
    elif te == 'list':
        for i, v in enumerate(en):
            walk_shapes(v, cn[i] if i < len(cn) else None, path + [str(i)], out)


def scan_data(root):
    res = {}
    for repo in ['1-Ember汉化插件', '2-Crucible汉化插件']:
        en_dir = os.path.join(root, repo, 'compendium', 'en')
        cn_dir = os.path.join(root, repo, 'compendium', 'cn')
        if not os.path.isdir(en_dir):
            continue
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith('.json'):
                continue
            cp = os.path.join(cn_dir, fn)
            if not os.path.exists(cp):
                continue
            en = json.load(open(os.path.join(en_dir, fn), encoding='utf-8-sig'))
            cn = json.load(open(cp, encoding='utf-8-sig'))
            out = []
            for section in ('entries', 'folders'):
                walk_shapes(en.get(section, {}), cn.get(section, {}), [section], out)
            if out:
                res[f'{repo}/{fn}'] = out
    return res


# ---------------------------------------------------------------- D 段

# 间接写面：注册在写流程上的钩子 / 猴补丁。它们本身不含写调用，
# 但会**改写别人即将落库的负载**，A 段的写调用正则看不见它们。
HOOK_REG = re.compile(r"Hooks\.(on|once)\s*\(\s*['\"](pre(?:Create|Update|Delete)\w+)['\"]")
PATCH_WRITE_API = re.compile(
    r"(\w+)\.(updateDocuments|createDocuments|deleteDocuments|update|updateSource)\s*=\s")


def scan_indirect(root):
    out = []
    for sub in ['1-Ember汉化插件', '2-Crucible汉化插件']:
        d = os.path.join(root, sub)
        for dp, dns, fns in os.walk(d):
            dns[:] = [x for x in dns if x not in ('.git', 'compendium', 'lang', 'node_modules')]
            for fn in fns:
                if os.path.splitext(fn)[1] not in ('.js', '.mjs'):
                    continue
                p = os.path.join(dp, fn)
                lines = open(p, encoding='utf-8').read().splitlines()
                for i, ln in enumerate(lines):
                    for m in HOOK_REG.finditer(ln):
                        body = '\n'.join(lines[i:i + 4])
                        out.append({'file': os.path.relpath(p, root).replace('\\', '/'),
                                    'line': i + 1, 'what': f'hook {m.group(2)}',
                                    'mutates_payload': bool(re.search(
                                        r'sanitize|normalize|=\s|delete\s', body)),
                                    'src': ln.strip()[:110]})
                    for m in PATCH_WRITE_API.finditer(ln):
                        if ln.strip().startswith('*') or ln.strip().startswith('//'):
                            continue
                        out.append({'file': os.path.relpath(p, root).replace('\\', '/'),
                                    'line': i + 1, 'what': f'monkeypatch {m.group(2)}',
                                    'mutates_payload': True, 'src': ln.strip()[:110]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--json')
    a = ap.parse_args()

    ind = scan_indirect(a.root)
    print(f'== D 段 间接写面（改写他人落库负载）{len(ind)} 处 ==')
    for h in ind:
        print(f"  {h['file']}:{h['line']}  {h['what']}  改负载={h['mutates_payload']}")
        print(f"        {h['src']}")
    print()

    code = scan_code(a.root)
    print(f'== A/B 写调用点 {len(code)} 处 ==')
    for h in code:
        flags = []
        if h['shape_only'] and not h['type_judge']:
            flags.append('仅形状判据')
        if not h['guard']:
            flags.append('无开关')
        print(f"  [{h['kind']}] {h['file']}:{h['line']}  {'/'.join(flags) or '-'}")
        print(f"        {h['call']}")
        if h['guard']:
            print(f"        guard: {h['guard']}")

    data = scan_data(a.root)
    n = sum(len(v) for v in data.values())
    print(f'\n== C 段 EN/CN 形状不一致 {n} 处，分布 {len(data)} 个包 ==')
    for k, v in data.items():
        print(f'  {k}: {len(v)}')
        for x in v[:8]:
            print(f'    {x["path"]}  EN={x["en"]} CN={x["cn"]}')
            print(f'      EN {x["en_sample"]}')
            print(f'      CN {x["cn_sample"]}')

    if a.json:
        json.dump({'code': code, 'data': data}, open(a.json, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        print(f'\n-> {a.json}')


if __name__ == '__main__':
    sys.exit(main())
