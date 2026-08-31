# -*- coding: utf-8 -*-
"""只读探针 #5：数字—单位写法一致性（在**去标记后的纯中文**上统计，避免 HTML 标签造成的假象）。

统计两件事：
 A. 阿拉伯数字与单位之间「有无空格」的分布 —— 若两种写法都占相当比例，说明全库无约定。
 B. 同一量在同一语境下「阿拉伯数字」与「汉字数字」并存的情况。

假阳性说明：
 - 「点」也用于「点燃/地点/一点点」等词，已要求前面紧跟数字。
 - 「天」也用于「天花板/天赋」，同上限定。
 - 汉字数字统计里「一次/一名」这类量词天然用汉字，故只统计与阿拉伯数字**同时存在**的单位。
 - 孪生包（ember.adventure / ember.crucible-adventure）互为副本，绝对数约为唯一叶的两倍。
"""
import io, re, sys, os, collections
R = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cn_corpus import load_all, plain

UNITS = ["级", "点", "轮", "回合", "英尺", "英里", "格", "小时", "分钟", "天", "秒", "码"]

def main():
    leaves = load_all()
    texts = [plain(L["s"]) for L in leaves]
    paths = [(L["file"], L["path"]) for L in leaves]
    o = io.open(os.environ.get("OUT", "num_unit_style.txt"), "w", encoding="utf-8")
    o.write("=== A. 阿拉伯数字 + 单位（纯中文文本上统计）===\n")
    o.write("%-6s %8s %8s  少数派占比\n" % ("单位", "紧贴", "空格"))
    for u in UNITS:
        t = sum(len(re.findall(r"\d" + u, x)) for x in texts)
        s = sum(len(re.findall(r"\d " + u, x)) for x in texts)
        if t + s == 0: continue
        o.write("%-6s %8d %8d   %.1f%%\n" % (u, t, s, 100.0 * min(t, s) / (t + s)))
    o.write("\n=== B. 同单位下 汉字数字 vs 阿拉伯数字 ===\n")
    for u in UNITS:
        han = collections.Counter(); ex = {}
        for x, p in zip(texts, paths):
            for m in re.finditer(r"(?<![\u4e00-\u9fff])([一二三四五六七八九十]{1,3})" + u, x):
                han[m.group(0)] += 1
                ex.setdefault(m.group(0), (p[0], p[1][-60:], x[max(0, m.start()-25):m.end()+25]))
        ara = sum(len(re.findall(r"\d\s?" + u, x)) for x in texts)
        if han and ara:
            o.write("\n-- %s：阿拉伯 %d 处；汉字 %s\n" % (u, ara, ", ".join("%s×%d" % (k, v) for k, v in han.most_common())))
            for k, v in han.most_common(8):
                o.write("     %s  @%s :: …%s…\n" % (k, ex[k][1], ex[k][2]))
    o.close()
    print("ok")

if __name__ == "__main__":
    main()
