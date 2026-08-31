# -*- coding: utf-8 -*-
import io, json, sys, importlib.util
sys.stdout.reconfigure(encoding='utf-8')
spec = importlib.util.spec_from_file_location('m', 'morph.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
M = json.load(io.open('recon/morphemes.json', encoding='utf-8'))
known = set(m.HEADS) | set(m.MODS) | set(m.SUFFIX)
miss = [t for t in M['tokens'] if t not in known]
dup = sorted(set(m.HEADS) & set(m.MODS))
print('词典覆盖', len(M['tokens']) - len(miss), '/', len(M['tokens']))
print('未覆盖', len(miss), ':', ' · '.join(miss))
print('既是中心词又是修饰的（组合时要有明确优先级）:', ' · '.join(dup) or '无')
