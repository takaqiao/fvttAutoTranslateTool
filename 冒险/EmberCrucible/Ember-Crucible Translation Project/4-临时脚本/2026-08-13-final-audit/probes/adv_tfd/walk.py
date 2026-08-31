# -*- coding: utf-8 -*-
import json, io, os, sys, re
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'

def load(rel):
    return json.load(io.open(os.path.join(ROOT, rel), encoding='utf-8'))

def walk(node, path):
    """yield (path, key, value) for every string leaf"""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str):
                yield (path, k, v)
            else:
                yield from walk(v, path + '/' + str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, str):
                yield (path, str(i), v)
            else:
                yield from walk(v, path + '/' + str(i))
