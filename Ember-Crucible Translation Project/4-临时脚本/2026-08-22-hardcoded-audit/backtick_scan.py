# -*- coding: utf-8 -*-
"""量一下**我这套扫描漏掉了多少**：反引号模板串里的可见文案。

只统计规模，不逐条定性 —— 目的是让报告里的「下界」有个数，而不是含糊说「还有一些」。
"""
import io, re, sys, collections
sys.stdout.reconfigure(encoding='utf-8')
BS = chr(92)
SRC = {
    "ember": r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs",
    "crucible": r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/crucible-compiled.mjs",
}
# 反引号串（不跨过反引号本身），再要求里面有 <p>/<div>/<span> 或至少两个英文单词
RE_TICK = re.compile(r'`([^`]{8,4000})`', re.S)
RE_TAGTEXT = re.compile(r'>([^<>{}]{3,})<')
RE_TWOWORDS = re.compile(r'[A-Z][a-z]+ +[a-z]{2,}')
RE_LETTER = re.compile(r'[A-Za-z]')

for repo, path in SRC.items():
    txt = io.open(path, encoding='utf-8').read()
    html, plain = collections.Counter(), collections.Counter()
    for m in RE_TICK.finditer(txt):
        body = m.group(1)
        if '<' in body and '>' in body:
            for t in RE_TAGTEXT.findall(body):
                s = re.sub(r'\s+', ' ', t).strip()
                if len(s) >= 3 and RE_LETTER.search(s) and not s.startswith('$'):
                    html[s] += 1
        elif RE_TWOWORDS.search(body) and '${' not in body[:2]:
            s = re.sub(r'\s+', ' ', body).strip()
            if len(s) < 200: plain[s] += 1
    print(f"{repo:<10} 反引号里的 HTML 文本节点 {len(html):>4} 唯一 / {sum(html.values()):>4} 次"
          f" ｜ 反引号纯文案 {len(plain):>4} 唯一")
    for s, c in html.most_common(6):
        print(f"           · {s[:70]}")
