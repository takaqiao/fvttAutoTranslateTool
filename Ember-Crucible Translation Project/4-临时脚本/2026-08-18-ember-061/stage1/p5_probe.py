# -*- coding: utf-8 -*-
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

f = sys.argv[1]
pref = sys.argv[2]
src = sys.argv[3] if len(sys.argv) > 3 else 'en061'
root = {'en061': os.path.join(EN061, f), 'en060': os.path.join(EN060, f), 'cn': os.path.join(CN, f)}[src]
d = jload(root)
lim = int(os.environ.get('LIM', '600'))
for parts, v in walk_leaves(d):
    p = '/'.join(parts)
    if pref.lower() in p.lower():
        print(f"{p}\n   {repr(v)[:lim]}")
