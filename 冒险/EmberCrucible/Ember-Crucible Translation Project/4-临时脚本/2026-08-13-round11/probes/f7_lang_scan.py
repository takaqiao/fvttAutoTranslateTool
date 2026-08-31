# -*- coding: utf-8 -*-
"""Scan lang/cn.json and scripts/*.mjs for the dead renderings F7 is retiring."""
import json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
FILES = [
    os.path.join(ROOT, "1-Ember\u6c49\u5316\u63d2\u4ef6", "lang", "cn.json"),
    os.path.join(ROOT, "2-Crucible\u6c49\u5316\u63d2\u4ef6", "lang", "cn.json"),
    os.path.join(ROOT, "1-Ember\u6c49\u5316\u63d2\u4ef6", "scripts", "ember-hardcoded-cn.mjs"),
    os.path.join(ROOT, "1-Ember\u6c49\u5316\u63d2\u4ef6", "babele-mappings.js"),
    os.path.join(ROOT, "2-Crucible\u6c49\u5316\u63d2\u4ef6", "babele-mappings.js"),
]
DEAD = ["乌头", "海门", "希格纳", "法序议会", "奥术巨龙", "残酷巨龙", "卡达娜",
        "赫尔卡斯·格林", "离火之家", "碎齿", "碎牙者", "突变派", "嬗变师", "变异师",
        "变异学者", "突变剂师", "荆棘裔", "星兆", "维伦血统", "基瓦赫", "阿克登",
        "螯蛛以太兽", "印记者", "卡斯奇利安", "烬界", "阿纳克雷努姆", "阿纳克拉埃努姆",
        "鲁玛林", "大地", "地球", "龙语", "卢玛"]


def flat(o, path, out):
    if isinstance(o, dict):
        for k, v in o.items():
            flat(v, path + "." + str(k) if path else str(k), out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            flat(v, "%s[%d]" % (path, i), out)
    elif isinstance(o, str):
        out.append((path, o))


for f in FILES:
    if not os.path.isfile(f):
        print("MISSING %s" % f)
        continue
    print("#### %s" % f.replace(ROOT + os.sep, ""))
    if f.endswith(".json"):
        leaves = []
        flat(json.load(open(f, encoding="utf-8")), "", leaves)
        for p, v in leaves:
            hit = [t for t in DEAD if t in v]
            if hit:
                print("   %-60s %r   <%s>" % (p, v, ",".join(hit)))
    else:
        for n, line in enumerate(open(f, encoding="utf-8"), 1):
            hit = [t for t in DEAD if t in line]
            if hit:
                print("   L%-5d %s   <%s>" % (n, line.strip()[:150], ",".join(hit)))
    print()
