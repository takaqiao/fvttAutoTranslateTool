# -*- coding: utf-8 -*-
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
ROOT=r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
SRC={"ember":r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember",
     "crucible":r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible"}
CN={"ember":"1-Ember汉化插件","crucible":"2-Crucible汉化插件"}
def walk(n,p=""):
    if isinstance(n,dict):
        for k,v in n.items(): yield from walk(v,f"{p}.{k}" if p else k)
    elif isinstance(n,list):
        for i,v in enumerate(n): yield from walk(v,f"{p}[{i}]")
    elif isinstance(n,str): yield p,n
for needle in ["魂缚进程","海门","异缘会","马尔斯通","证据值","感知力","受苦的苍白","因卡罗池","快剪手",
               "令牌","代币","六角格","兰蒂尔","波因特","法珠被摧毁","法珠已毁","黑曜古","晶片"]:
    tot=[]
    for k in CN:
        cn=dict(walk(json.load(open(os.path.join(ROOT,CN[k],"lang","cn.json"),encoding="utf-8-sig"))))
        for key,v in cn.items():
            if needle in v: tot.append((k,key,v[:60]))
    print(f"{needle}: lang {len(tot)}", tot[:4])
