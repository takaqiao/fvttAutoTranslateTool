# -*- coding: utf-8 -*-
"""
scan_untyped_transform.py  —— 「无类型/归属判据的批量改写」判据探针

抽象自已确认实例：
  register.js 的 preUpdateItem 钩子对**所有**物品做同一处 string→object 改写，
  没有 itemData.type 判据 —— 而上游 `system.description` 是**多态**的
  （6 类是 HTMLField 字符串，装备类才是 {public,private}）。

判据 P（本探针实现的机械形式）：
  在运行时 JS 里找出所有「会改写外部状态」的语句，取其**所在函数**，
  再看这个函数体里有没有出现**归属/类型闸门**。
    改写动作 W ∈ { .update( / updateDocuments / updateEmbeddedDocuments /
                   setProperty( / mergeObject( 非 inplace:false /
                   setAttribute( / nodeValue = / .label = / .name = /
                   delete X / X[k] = }
    闸门 G ∈ { .type ===  |  .type !==  |  instanceof  |  documentName
               |  .id === '…'  |  === '<字面量>'  |  Object.hasOwn
               |  .startsWith(  |  in <表>  |  <表>[x] 存在性判断 }
  报出：作用于**集合**（for/of、.map、.filter、childNodes、Object.values）
  且函数体内 **没有类型/归属闸门** 的改写点。

  另外单独标出「用子串正则近似判断归属」的闸门（gate-by-substring），
  因为它是判据 P 的 (b) 形态：`/ember/i.test(className)` 会命中 "member"。

假阳性模式（必须人工核实，脚本只给候选）：
  1. 闸门写在**调用方**而不是本函数里（本探针只看函数体）；
  2. 集合本来就是同质的（例如只有一种类型的数组），此时无闸门是对的；
  3. 纯读取式 map（返回新对象、不落库）也会被算成改写点；
  4. DOM 改写在受控子树内本来就该无差别（例如只翻自己渲染的节点）。

只读，不写任何库文件。
"""
import re
import sys
import json
from pathlib import Path

ROOT = Path(r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project")

TARGETS = [
    ROOT / "1-Ember汉化插件" / "register.js",
    ROOT / "1-Ember汉化插件" / "scripts" / "ember-hardcoded-cn.mjs",
    ROOT / "1-Ember汉化插件" / "babele-mappings.js",
    ROOT / "2-Crucible汉化插件" / "babele-register.js",
    ROOT / "2-Crucible汉化插件" / "babele-mappings.js",
    ROOT / "3-常用脚本" / "release" / "runtime-converters.js",
    ROOT / "3-常用脚本" / "release" / "generate_runtime.mjs",
    ROOT / "3-常用脚本" / "extract" / "mappings.mjs",
    ROOT / "3-常用脚本" / "extract" / "extract_en.mjs",
]

WRITE = re.compile(
    r"""(?x)
    \.update\(|\.updateDocuments\(|updateEmbeddedDocuments\(|createDocuments\(|
    setProperty\(|mergeObject\(|setAttribute\(|
    \bdelete\s+[\w.]+|
    \.[A-Za-z_]\w*\s*=[^=]|          # 任意属性赋值（含 changes.items = / hook.prepare =）
    \w+\[\w+\]\s*=[^=]
    """
)

GATE_TYPE = re.compile(
    r"""(?x)
    \.type\s*[!=]==|documentName|instanceof\s|
    \.id\s*===|===\s*['"][A-Za-z]|
    \.startsWith\(|\bin\s+[A-Z_]{3,}\b|
    Object\.hasOwn|hasOwnProperty
    """
)

GATE_SUBSTR = re.compile(r"/[^/\n]*/i?\.test\(|\.test\(|\.includes\(|\.match\(")

COLLECTION = re.compile(
    r"for\s*\(\s*const\s+\w+\s+of\b|\.map\(|\.filter\(|\.forEach\(|childNodes|Object\.values\(|Object\.entries\("
)

FUNC = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)|^\s*(\w+)\s*[:=]\s*(?:async\s*)?\(")


def functions(text):
    """粗切函数体：按顶层 `function name(` 起，到下一个同级 function/文件尾。"""
    lines = text.splitlines()
    marks = []
    for i, ln in enumerate(lines):
        m = re.match(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)", ln)
        if m:
            marks.append((i, m.group(1)))
        m2 = re.match(r"^Hooks\.(on|once)\(['\"]([\w.]+)['\"]", ln)
        if m2:
            marks.append((i, f"Hooks.{m2.group(1)}({m2.group(2)})"))
    marks.append((len(lines), "<eof>"))
    out = []
    for (a, name), (b, _) in zip(marks, marks[1:]):
        out.append((name, a + 1, b, "\n".join(lines[a:b])))
    return out


def main():
    report = []
    for p in TARGETS:
        if not p.exists():
            print(f"!! missing {p}")
            continue
        text = p.read_text(encoding="utf-8")
        for name, start, end, body in functions(text):
            if not WRITE.search(body):
                continue
            has_type_gate = bool(GATE_TYPE.search(body))
            has_substr_gate = bool(GATE_SUBSTR.search(body))
            over_collection = bool(COLLECTION.search(body))
            writes = sorted({m.group(0).strip() for m in WRITE.finditer(body)})
            flag = []
            if over_collection and not has_type_gate:
                flag.append("NO-TYPE-GATE-OVER-COLLECTION")
            if has_substr_gate:
                flag.append("GATE-BY-SUBSTRING")
            if not flag:
                continue
            report.append(
                {
                    "file": str(p.relative_to(ROOT)),
                    "func": name,
                    "lines": f"{start}-{end}",
                    "flags": flag,
                    "writes": writes[:8],
                }
            )
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\n候选 {len(report)} 处，扫描文件 {len(TARGETS)} 个", file=sys.stderr)


if __name__ == "__main__":
    main()
