#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针：子类型盲的防御式改写（只读库）

把种子实例（register.js 的 prepareSafeActorUpdatesForImport 无条件删 items/effects）
抽象成一条可机械化的判据：

  一个「修复 / 防御 / 归一化」变换 T，
  (a) **写**数据（delete / 改类型 / 置空容器 / 整表 map 替换 / 全集合循环 update），
  (b) 挂在一个**会收到全部子类型**的调用点上（全局 Hook、猴补丁、game.items/game.actors 全扫），
  (c) 而闸只测「形状/类型」（typeof / Array.isArray / 存在性），**不测子类型或归属**，
  → 只对少数病态输入正确的修复被套到全部输入上，正常数据被静默改写或丢弃，中间层不报错。

本脚本做两件事：
  P1 静态：把两个仓库里所有**运行时会加载**的 JS/MJS（module.json 的 esmodules 及其 import）
     按函数切开，标出每个写点与它最近的闸，按闸的种类分档：
       NO_GATE / SHAPE_GATE / SUBTYPE_GATE / OWNERSHIP_GATE
     SHAPE_GATE + 调用点为全局钩子/猴补丁/全集合循环 = 候选。
  P2 动态取证：从 Foundry 的 LevelDB 真包里读出每个 Item 子类型的
     `system.description` 实际形状与条数，回答「候选闸放行了多少**正常**数据」。

只读。不写库。

已知假阳性模式：
  · 正则切函数，箭头函数与嵌套函数会归到外层，闸的归属可能偏一层——所以 P1 的输出
    只能当**候选清单**，每条都要手工回到源码确认（本轮 10 条候选全部手工核过）。
  · 「写点」按 delete / 赋值到对象属性 / .map( 判定，纯局部变量赋值会误报。
  · P2 只能证明**包内**数据的形状；世界里用户手改出来的形状不在扫描范围内，
    所以 P2 给的是「误伤下界」，不是上界。
"""
from __future__ import annotations
import json, os, re, sys

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPOS = [os.path.join(ROOT, '1-Ember汉化插件'), os.path.join(ROOT, '2-Crucible汉化插件')]

WRITE = re.compile(r'\bdelete\s+\w+[.\[]|^\s*\w+(?:\.\w+|\[[^\]]+\])+\s*=[^=]|\.map\(|\.update\(|\.updateEmbeddedDocuments\(|setProperty\(|mergeObject\(', re.M)  # re.M 必须有：第一版漏了 re.M，^ 只在串首匹配，赋值型写点整批假阴
SHAPE_GATE = re.compile(r'typeof\s|Array\.isArray|instanceof\s|!==\s*undefined|\?\?')
SUBTYPE_GATE = re.compile(r'\.type\b|documentName|PHYSICAL_ITEM_TYPES|dataModels|documentType')
OWNER_GATE = re.compile(r'compendiumSource|packageId|\.pack\b|modules\.get|game\.system\.id')
BROAD_SINK = re.compile(r"Hooks\.on\(\s*['\"](?:preUpdate|preCreate|preDelete|render)|game\.items|game\.actors|documentClass\s*[.=]|CONFIG\.")

def fn_blocks(src):
    """粗切：从 function / const x = ( ... ) => 开始到下一个顶层 function 之前。"""
    lines = src.split('\n')
    starts = [i for i, l in enumerate(lines)
              if re.match(r'^(async\s+)?function\s+\w+|^(export\s+)?(async\s+)?function\s+\w+|^Hooks\.(on|once)\(', l)]
    starts.append(len(lines))
    for a, b in zip(starts, starts[1:]):
        name = re.sub(r'^(export\s+)?(async\s+)?', '', lines[a]).split('(')[0].strip() or f'@{a+1}'
        yield name, a + 1, '\n'.join(lines[a:b])

def runtime_files(repo):
    mj = json.load(open(os.path.join(repo, 'module.json'), encoding='utf-8'))
    out = []
    for e in mj.get('esmodules', []):
        p = os.path.join(repo, e.replace('/', os.sep))
        if os.path.exists(p):
            out.append(p)
    # 被 esmodule import 的同仓文件
    for p in list(out):
        for m in re.finditer(r"from\s+'\./([^']+)'", open(p, encoding='utf-8').read()):
            q = os.path.join(os.path.dirname(p), m.group(1))
            if os.path.exists(q) and q not in out:
                out.append(q)
    return out

def main():
    cands = []
    scanned = []
    for repo in REPOS:
        for path in runtime_files(repo):
            src = open(path, encoding='utf-8').read()
            scanned.append((path, len(src.split('\n'))))
            for name, ln, blk in fn_blocks(src):
                if not WRITE.search(blk):
                    continue
                gate = ('OWNERSHIP' if OWNER_GATE.search(blk) else
                        'SUBTYPE' if SUBTYPE_GATE.search(blk) else
                        'SHAPE' if SHAPE_GATE.search(blk) else 'NONE')
                broad = bool(BROAD_SINK.search(blk))
                if gate in ('SHAPE', 'NONE'):
                    cands.append({'file': os.path.relpath(path, ROOT), 'fn': name, 'line': ln,
                                  'gate': gate, 'broad_sink_in_block': broad,
                                  'writes': sorted(set(WRITE.findall(blk)))[:4]})
    print(json.dumps({'scanned': [{'file': os.path.relpath(p, ROOT), 'lines': n} for p, n in scanned],
                      'candidates': cands, 'n_candidates': len(cands)},
                     ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()
