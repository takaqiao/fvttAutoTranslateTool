"""G2 shard helpers: flatten en/cn packs into dotted-path leaf maps.

Path form matches apply_translations.py: `(folders).X…` for the folders root,
otherwise the path is rooted at `entries`, i.e. `entries.<...>` is written as
`entries.<...>`?  -- no: apply_translations splits on '.' and roots at
`en['entries']` unless the first segment is `(folders)`.  So the batch key for
`en['entries']['A']['name']` is `A.name`.
"""
import json
import os

R1 = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件'
R2 = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件'


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def walk(node, prefix, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, prefix + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, prefix + [str(i)], out)
    elif isinstance(node, str):
        out['.'.join(prefix)] = node


def pack_leaves(repo, pack):
    """Return (en_map, cn_map) keyed by batch path."""
    en = load(os.path.join(repo, 'compendium', 'en', pack))
    cn_p = os.path.join(repo, 'compendium', 'cn', pack)
    cn = load(cn_p) if os.path.exists(cn_p) else {}
    eo, co = {}, {}
    walk(en.get('entries', {}), [], eo)
    walk(cn.get('entries', {}), [], co)
    f1, f2 = {}, {}
    walk(en.get('folders', {}), [], f1)
    walk(cn.get('folders', {}), [], f2)
    for k, v in f1.items():
        eo['(folders).' + k] = v
    for k, v in f2.items():
        co['(folders).' + k] = v
    return eo, co


def all_packs():
    for repo in (R1, R2):
        d = os.path.join(repo, 'compendium', 'en')
        for p in sorted(os.listdir(d)):
            if p.endswith('.json'):
                yield repo, p
