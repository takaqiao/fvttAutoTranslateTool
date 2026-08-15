# -*- coding: utf-8 -*-
"""给 html_wellformed.py 种人工缺陷，确认它不是个死探针。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from html_wellformed import analyze, raw_amp_count, bracket_sig  # noqa

CASES = [
    ("P1", "<p><strong>你好</p></strong>"),
    ("P1", "<ul><li><em>甲</li></em></ul>"),
    ("P2", "<p>你好</p></div>"),
    ("P3", "<div><p>你好</p>"),
    ("P3", "<p>你好<strong>加粗</p>"),
    ("P4", "<p><div>块级</div></p>"),
    ("P4", "<ul>裸文本<li>甲</li></ul>"),
    ("P4", "<table>裸文本<tr><td>甲</td></tr></table>"),
    ("P5", "<强调>你好</强调>"),
    ("P5", "<Insert Name>"),
    ("P6", '<p class="tip 你好>断引号</p>'),
    ("P6", '<a href="a&b">链接</a>'),
    ("P6", '<p title="a<b">x</p>'),
    ("P8", "<p class=“tip”>全角引号</p>"),
    ("P8", "@UUID［Actor.x］{名字}"),
    ("P8", "@UUID[Actor.x]｛名字｝"),
]

ok = True
for want, s in CASES:
    errs, _, _ = analyze(s)
    codes = {c for c, _ in errs}
    hit = want in codes
    print(f"{'PASS' if hit else 'FAIL'} [{want}] {s!r} -> {sorted(codes)} {[d for _,d in errs][:2]}")
    ok &= hit

# 阴性对照：正常写法不该报
CLEAN = [
    "<p>你好<strong>加粗</strong>结束</p>",
    "<ul><li>甲</li><li>乙</li></ul>",
    "<p>甲<p>乙",                    # p 可省略结束标签
    "<table><tr><td>甲</td></tr></table>",
    "<p>&amp;Reference[blinded]</p>",
    "<p>@UUID[Actor.x]{名字}</p>",
    "<p>伤害 [[/r 2d6]] 点</p>",
    '<h3 id="the-hallows">圣堂区</h3>',
    "<p>价格 5 &lt; 10</p>",
    '<img src="a.webp">',
    '<p><em>斜体</em>与<strong>粗体</strong></p>',
    '<section class="secret"><p>秘密</p></section>',
]
for s in CLEAN:
    errs, _, _ = analyze(s)
    if errs:
        ok = False
        print(f"FALSE-POSITIVE {s!r} -> {errs}")
    else:
        print(f"clean OK {s!r}")

print("SELFTEST", "OK" if ok else "BROKEN")
