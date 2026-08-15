# -*- coding: utf-8 -*-
"""探针：运行时「无作用域判据的全局写入点」

种子缺陷的抽象形式
------------------
在一个**对全世界生效**的写入/注册点上动手，却不带任何「这是不是我该管的东西」的
判据（不看 type / 不看包 / 不看是不是本模块的内容），出错静默。

本探针把「写入点」机械化为 7 类，并在**同一函数体内**找是否存在 scope predicate。

写入点（WRITE SITES）
  W1 Hooks.on('preUpdate*'|'preCreate*'|'render*', ...)   —— 全局文档/渲染钩子
  W2 <X>.<method> = <wrapped>                             —— 猴补丁
  W3 for (... of game.items|game.actors|game.folders...)  + .update()/.setFlag()
  W4 就地改写别人的对象：CONFIG.* / crucible.CONFIG / game.time.calendar /
     game.i18n.translations / CONFIG.TextEditor.enrichers
  W5 babele.registerMapping / registerConverters          —— babele 的注册层是全局层
  W6 _packs-folders 里的**通用英文名**                     —— babele allTranslations()
     会把所有 _packs-folders 合并成一张裸名字典，然后改名**整个世界**的 Folder
  W7 module.json styles 里的 CSS 选择器不带自有类名

作用域判据（SCOPE PREDICATES）
  .type / documentName / item.type / actor.type      —— 类型闸
  pack / collection / compendiumSource / metadata.id —— 包闸
  game.system.id / MODULE_ID / modules.get(          —— 归属闸
  正则/字符串里含本项目自有前缀（ember / crucible / EMBER. / CRUCIBLE.）

假阳性模式（必须人工复核）
  * W4 命中很多，但「改自己刚建的对象」也会被算进去 —— 要看被改的对象是谁的。
  * 归属闸可能出现在**调用点**而不是函数体内（如 Hooks.once('ready') 里的
    game.modules.get('ember')?.active），本探针按「函数体 + 直接调用点」两级找。
  * W6 只有 5 个键，通用与否要人判：Actors/Items 是通用，Adversary Options 不是。

用法：python scope_gate_runtime.py --repo <仓库> [--repo <仓库2>]   只读。
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

HOOK = re.compile(r"Hooks\.(on|once)\(\s*['\"]((?:pre(?:Update|Create|Delete)|render)\w*)['\"]")
MONKEY = re.compile(r"^\s*(?!//)([\w$.?\[\]'\"]+)\.(\w+)\s*=\s*(wrapped|function|async|\(|\w+\s*;)", re.M)
WORLD_ITER = re.compile(r"for\s*\(\s*const\s+\w+\s+of\s+(game\.(?:items|actors|folders|journal|scenes|macros|tables|users|combats)\b[^)]*)\)")
DB_WRITE = re.compile(r"\.(update|updateEmbeddedDocuments|createEmbeddedDocuments|deleteEmbeddedDocuments|setFlag|updateDocuments)\(")
FOREIGN_MUT = re.compile(
    r"(CONFIG\.[\w.$?\[\]]*|globalThis\.crucible[\w.$?]*|game\.time\.calendar[\w.$?]*"
    r"|game\.i18n\.translations[\w.$?]*|ui\.\w+)\s*(\[[^\]]+\])?\s*(=[^=]|\.\w+\s*=[^=])")
BABELE_REG = re.compile(r"babele\.(registerMapping|registerConverters)\(")

SCOPE = re.compile(
    r"\.type\b|documentName|\bpack\b|collection|compendiumSource|metadata\.id"
    r"|game\.system\.id|MODULE_ID|modules\.get\(|/ember/i|\bEMBER\.|CRUCIBLE\.|crucible\.")

FUNC = re.compile(r"^(?:async\s+)?function\s+(\w+)|^\s*(\w+)\s*\([^)]*\)\s*\{", re.M)


def functions(src: str):
    """粗切函数块：从 `function name(` 到下一个顶层 `function`/文件尾。够用即可。"""
    marks = [(m.start(), m.group(1) or m.group(2)) for m in FUNC.finditer(src) if (m.group(1) or m.group(2))]
    marks.append((len(src), None))
    out = []
    for i in range(len(marks) - 1):
        out.append((marks[i][1], marks[i][0], marks[i + 1][0], src[marks[i][0]:marks[i + 1][0]]))
    return out


def lineno(src, pos):
    return src.count("\n", 0, pos) + 1


def audit_js(path: Path):
    src = path.read_text(encoding="utf-8")
    funcs = functions(src)

    def owner(pos):
        for name, a, b, body in funcs:
            if a <= pos < b:
                return name, body
        return None, src

    rows = []

    def add(kind, pos, snippet):
        fname, body = owner(pos)
        gated = bool(SCOPE.search(body))
        rows.append({"kind": kind, "file": path.name, "line": lineno(src, pos),
                     "func": fname, "scope_predicate": gated, "snippet": snippet.strip()[:150]})

    for m in HOOK.finditer(src):
        add("W1 global-hook", m.start(), src[m.start():m.start() + 90])
    for m in MONKEY.finditer(src):
        tgt = m.group(1)
        if tgt.startswith(("const", "let", "var")) or "." not in tgt:
            continue
        add("W2 monkeypatch", m.start(), m.group(0))
    for m in WORLD_ITER.finditer(src):
        blk = src[m.start():m.start() + 2500]
        if DB_WRITE.search(blk):
            add("W3 world-sweep-write", m.start(), m.group(0))
    for m in FOREIGN_MUT.finditer(src):
        add("W4 foreign-object-mutation", m.start(), src[m.start():m.start() + 80])
    for m in BABELE_REG.finditer(src):
        add("W5 babele-global-layer", m.start(), m.group(0))
    return rows


GENERIC = {"actors", "items", "actor", "item", "npcs", "characters", "journals", "scenes",
           "macros", "tables", "playlists", "cards", "effects", "spells", "equipment",
           "monsters", "creatures", "misc", "other", "assets", "adventures", "rules"}


def audit_packs_folders(repo: Path):
    rows = []
    for f in sorted((repo / "compendium" / "cn").glob("*_packs-folders.json")):
        d = json.loads(f.read_text(encoding="utf-8-sig"))
        for en, cn in (d.get("entries") or {}).items():
            rows.append({"kind": "W6 global-folder-name", "file": f.name, "line": 0,
                         "func": "babele FolderTranslations.translateSystemPackFolders",
                         "scope_predicate": en.strip().lower() not in GENERIC,
                         "snippet": f"{en!r} -> {cn!r}"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    a = ap.parse_args()
    all_rows = []
    for r in a.repo:
        repo = Path(r)
        for js in sorted(list(repo.glob("*.js")) + list(repo.glob("scripts/*.mjs")) + list(repo.glob("*.mjs"))):
            all_rows += [dict(row, repo=repo.name) for row in audit_js(js)]
        all_rows += [dict(row, repo=repo.name) for row in audit_packs_folders(repo)]

    ungated = [r for r in all_rows if not r["scope_predicate"]]
    print(f"写入点合计 {len(all_rows)}  |  无作用域判据 {len(ungated)}\n")
    for r in all_rows:
        flag = "  " if r["scope_predicate"] else "!!"
        print(f"{flag} [{r['kind']:<26}] {r['repo']}/{r['file']}:{r['line']}  fn={r['func']}")
        print(f"      {r['snippet']}")


if __name__ == "__main__":
    main()
