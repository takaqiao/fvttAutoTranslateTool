# -*- coding: utf-8 -*-
"""确认 tree_shape 的 block_paths / lxml_strict_err / secret_shape 真的在工作。"""
import sys, json, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import tree_shape as T  # noqa

A = '<ul><li><p>甲</p></li></ul><p>乙</p>'
B = '<ul><li><p>甲</p></li><p>乙</p></ul>'   # 多重集相同，树形不同
print("A paths:", T.block_paths(A))
print("B paths:", T.block_paths(B))
print("多重集相同?", collections.Counter(T.block_paths(A)) == collections.Counter(T.block_paths(B)))
print("序列相同?", T.block_paths(A) == T.block_paths(B))
assert T.block_paths(A) != T.block_paths(B), "块级树形对比是死的！"

C = '<section class="secret"><p>秘密</p></section><p>公开</p>'
D = '<p>秘密</p><section class="secret"><p>公开</p></section>'
print("C secret:", T.secret_shape(C))
print("D secret:", T.secret_shape(D))
assert T.secret_shape(C) != T.secret_shape(D), "secret 对比是死的！"

print("lxml strict 好:", T.lxml_strict_err("<p>ok</p>"))
print("lxml strict 坏:", T.lxml_strict_err("<p>x</div>"))
print("非ASCII标签:", T.NONASCII_TAG.findall("<强调>你好</强调>"))
print("属性裸<:", T.ATTR_RAW_LT.findall('<p title="a<b">x</p>'))

# 真实语料抽样：确认 block_paths 在真数据上非空
repo = Path(sys.argv[1])
f = repo / "compendium" / "cn" / "ember.adventure.json"
d = dict(T.leaves(json.loads(f.read_text(encoding="utf-8-sig"))))
n_nonempty = 0
sample = None
for p, s in d.items():
    if "<p>" in s:
        bp = T.block_paths(s)
        if bp:
            n_nonempty += 1
            if sample is None:
                sample = (p, bp[:12], len(bp))
print(f"真实语料里 block_paths 非空的叶: {n_nonempty}")
print("样例:", sample)
assert n_nonempty > 1000, "在真实语料上几乎没解析出块级节点，probe 可疑"
print("SELFTEST2 OK")
