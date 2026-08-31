# -*- coding: utf-8 -*-
"""Shared helpers for the ember 0.6.0 -> 0.6.1 stage-1 follow-up."""
import json, os, sys, io, re

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
SCRATCH = r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\289d7a82-7d7b-4b2d-ac68-1439487a5f75\scratchpad"
CN = os.path.join(P, "1-Ember汉化插件", "compendium", "cn")
EN060 = os.path.join(P, "1-Ember汉化插件", "compendium", "en")
EN061 = os.path.join(SCRATCH, "em061")
DELTA = os.path.join(SCRATCH, "delta_em_060_to_061.json")
WORK = os.path.join(P, "4-临时脚本", "2026-08-18-ember-061", "stage1")

FILES = ["ember.adventure.json", "ember.crucible-adventure.json", "ember.crucible-adversary.json"]


def jload(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def jdump(obj, p):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def save_cn(obj, p):
    """按 CN 文件既有式样写回：UTF-8 无 BOM / indent=2 / CRLF / 末尾换行。"""
    s = json.dumps(obj, ensure_ascii=False, indent=2) + '\n'
    with open(p, 'wb') as f:
        f.write(s.replace('\n', '\r\n').encode('utf-8'))


def load_delta():
    return jload(DELTA)


_IDX = re.compile(r'^(.*?)((?:\[\d+\])+)$')


def split_path(ptr):
    """'/entries/X/items/Y/actions/z/effects[0]/name' ->
    ['entries','X','items','Y','actions','z','effects',0,'name']
    数组下标切成 int 段。"""
    assert ptr.startswith('/'), ptr
    out = []
    for seg in ptr[1:].split('/'):
        m = _IDX.match(seg)
        if m and m.group(1) != '':
            out.append(m.group(1))
            for n in re.findall(r'\[(\d+)\]', m.group(2)):
                out.append(int(n))
        else:
            out.append(seg)
    return out


_MISS = object()


def _walk(root, parts):
    cur = root
    for seg in parts:
        if isinstance(seg, int):
            if isinstance(cur, list) and 0 <= seg < len(cur):
                cur = cur[seg]
            else:
                return _MISS
        else:
            if isinstance(cur, dict) and seg in cur:
                cur = cur[seg]
            else:
                return _MISS
    return cur


def get_at(root, parts):
    v = _walk(root, parts)
    return None if v is _MISS else v


def has_at(root, parts):
    return _walk(root, parts) is not _MISS


def has_leaf(root, parts):
    """路径存在且落点是标量叶（字符串/数字），不是子树。"""
    v = _walk(root, parts)
    return v is not _MISS and not isinstance(v, (dict, list))


def set_at(root, parts, val):
    cur = root
    for i, seg in enumerate(parts[:-1]):
        nxt = parts[i + 1]
        if isinstance(seg, int):
            while len(cur) <= seg:
                cur.append([] if isinstance(nxt, int) else {})
            if cur[seg] is None:
                cur[seg] = [] if isinstance(nxt, int) else {}
            cur = cur[seg]
        else:
            if seg not in cur or not isinstance(cur[seg], (dict, list)):
                cur[seg] = [] if isinstance(nxt, int) else {}
            cur = cur[seg]
    cur[parts[-1]] = val


def del_at(root, parts):
    """Delete leaf, then prune now-empty ancestor containers."""
    stack = []
    cur = root
    for seg in parts[:-1]:
        nxt = _walk(cur, [seg])
        if nxt is _MISS:
            return False
        stack.append((cur, seg))
        cur = nxt
    last = parts[-1]
    if _walk(cur, [last]) is _MISS:
        return False
    del cur[last]
    for parent, seg in reversed(stack):
        v = parent[seg]
        if isinstance(v, (dict, list)) and len(v) == 0:
            del parent[seg]
        else:
            break
    return True


# --- enhancer / tag multiset extraction (judgment-critical) ---
ENH_RE = re.compile(r'@(\w+)\[')
TAG_RE = re.compile(r'</?([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>')


def enh_multiset(s):
    """Full bracket-balanced enhancer tokens incl. optional {label}."""
    out = []
    if not isinstance(s, str):
        return out
    for m in ENH_RE.finditer(s):
        i = m.end() - 1  # at '['
        depth = 0
        j = i
        while j < len(s):
            if s[j] == '[':
                depth += 1
            elif s[j] == ']':
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        tok = s[m.start():j]
        if j < len(s) and s[j] == '{':
            k = j
            d2 = 0
            while k < len(s):
                if s[k] == '{':
                    d2 += 1
                elif s[k] == '}':
                    d2 -= 1
                    if d2 == 0:
                        k += 1
                        break
                k += 1
            tok = s[m.start():k]
        out.append(tok)
    return out


def uuid_targets(s):
    """Just the bracket payload (target), ignoring the {label} which is translatable."""
    out = []
    if not isinstance(s, str):
        return out
    for m in ENH_RE.finditer(s):
        i = m.end() - 1
        depth = 0
        j = i
        lim = min(len(s), i + 4000)
        while j < lim:
            if s[j] == '[':
                depth += 1
            elif s[j] == ']':
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        else:
            j = m.end()  # 括号没闭合（上游 bug）：只取记号头，别把整段吞进来
        tok = s[m.start():j]
        # @Embed[... label="译文" readaloud="译文"] 里的参数值是可译的，比对时剥掉
        tok = re.sub(r'\s+(?:label|readaloud|title|caption)\s*=\s*(\"[^\"]*\"|[^\]\s]+)', ' PARAM', tok)
        out.append(tok)
    return out


ROLL_RE = re.compile(r'\[\[[^\]]*\]\]|&(?:amp;)?[Rr]eference\[[^\]]*\]')


def rolls(s):
    return ROLL_RE.findall(s) if isinstance(s, str) else []


def tag_multiset(s):
    if not isinstance(s, str):
        return []
    return TAG_RE.findall(s)


def placeholders(s):
    if not isinstance(s, str):
        return []
    return re.findall(r'\{[^{}]*\}', s)


def walk_leaves(obj, prefix=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_leaves(v, prefix + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_leaves(v, prefix + (str(i),))
    else:
        yield prefix, obj
