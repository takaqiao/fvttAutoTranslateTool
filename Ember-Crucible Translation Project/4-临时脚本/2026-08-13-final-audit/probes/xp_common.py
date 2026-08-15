#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨页一致性探针的共用件：叶子遍历 / 块切分 / 数字规范化。

设计要点（与 3-常用脚本/qa/scan_content_coverage.py 的差别）
------------------------------------------------------------
* scan_content_coverage 只在**单叶内**比英中数字多重集。本文件提供**块级对齐**，
  使得同一段英文出现在不同 journal / 不同 page 时可以互相比对中文。
* 块切分用与 scan_content_coverage 相同的 BLOCK_TAG 集合，保证「块」的定义一致。
* 对齐方式：EN 与 CN 按块级标签切成同样长度的原始片段列表后**按下标 zip**。
  markup drift 已归零（tag 多重集逐叶相等）是这条对齐成立的前提；
  长度不等的叶子直接跳过并计数（假阴性来源，见各探针输出的 skipped 统计）。
"""
from __future__ import annotations
import json
import os
import re

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPOS = {
    'ember': os.path.join(ROOT, '1-Ember汉化插件'),
    'crucible': os.path.join(ROOT, '2-Crucible汉化插件'),
}

CJK = re.compile(r'[\u4e00-\u9fff]')
TAG = re.compile(r'<[^>]+>')
BLOCK_TAG = re.compile(
    r'<\s*/?\s*(?:p|div|br|hr|table|thead|tbody|tfoot|tr|td|th|caption|'
    r'ul|ol|li|dl|dt|dd|h[1-6]|section|article|aside|header|footer|'
    r'blockquote|pre|figure|figcaption|form|fieldset)\b[^>]*>', re.I)
ENTITY = re.compile(r'&#?\w{1,8};')
NUM = re.compile(r'(?<!\d)(\d+(?:\.\d+)?)(?!\d)')

# @UUID[...]{标签} / [[/宏 参数]] / &Reference[...] —— 机关参数，逐字保留，不参与文字比对
MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\](?:\{[^}]*\})?|\[\[[^\]]*\]\]'
                    r'|&(?:amp;)?[Rr]eference\[[^\]]*\](?:\{[^}]*\})?')

CN_DIGIT = {'零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
            '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
CN_NUM = re.compile(r'[零〇一二三四五六七八九十百]+')

EN_WORD_NUM = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
    'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100,
    'thousand': 1000, 'dozen': 12, 'score': 20, 'first': 1, 'second': 2,
    'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6, 'seventh': 7, 'eighth': 8,
    'ninth': 9, 'tenth': 10, 'twelfth': 12, 'twentieth': 20, 'single': 1,
    'once': 1, 'twice': 2, 'thrice': 3, 'pair': 2, 'couple': 2, 'both': 2,
    'half': 2, 'quarter': 4, 'decade': 10, 'century': 100, 'triple': 3,
    'double': 2, 'trio': 3, 'duo': 2, 'dual': 2, 'sole': 1, 'lone': 1,
}
EN_WORD_RE = re.compile(r'\b(' + '|'.join(EN_WORD_NUM) + r')s?\b', re.I)


def cn_num_value(t: str):
    """中文数词串 -> 值（够用即可；认不出返回 None）。"""
    if t == '十':
        return 10
    if len(t) == 1:
        return CN_DIGIT.get(t)
    if t[0] == '十' and len(t) == 2:
        return 10 + CN_DIGIT.get(t[1], 0) if t[1] in CN_DIGIT else None
    if len(t) == 2 and t[1] == '十':
        return CN_DIGIT.get(t[0], 0) * 10 if t[0] in CN_DIGIT else None
    if len(t) == 3 and t[1] == '十':
        if t[0] in CN_DIGIT and t[2] in CN_DIGIT:
            return CN_DIGIT[t[0]] * 10 + CN_DIGIT[t[2]]
        return None
    if len(t) == 2 and t[1] == '百':
        return CN_DIGIT.get(t[0], 0) * 100 if t[0] in CN_DIGIT else None
    if t == '百':
        return 100
    return None


def plain(s: str) -> str:
    """剥掉 enricher / HTML 标签 / 实体，保留块边界为空格。"""
    out = MARKUP.sub(' ', s)
    out = ENTITY.sub(' ', out)
    out = TAG.sub(' ', out)
    return re.sub(r'\s+', ' ', out).strip()


def split_blocks(s: str):
    """按块级标签切成原始片段列表（不做长度过滤，保证 EN/CN 下标可对齐）。"""
    return BLOCK_TAG.split(s)


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [k], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                 path + [str(i)], out)
    elif isinstance(en, str) and en.strip():
        out.append(('.'.join(path), en, cn if isinstance(cn, str) else None))


def load_pairs(repo_key: str):
    """-> [(pack, path, en, cn)]"""
    repo = REPOS[repo_key]
    en_dir = os.path.join(repo, 'compendium', 'en')
    cn_dir = os.path.join(repo, 'compendium', 'cn')
    rows = []
    for pack in sorted(os.listdir(en_dir)):
        if not pack.endswith('.json') or pack.startswith('_'):
            continue
        cp = os.path.join(cn_dir, pack)
        if not os.path.exists(cp):
            continue
        e = json.load(open(os.path.join(en_dir, pack), encoding='utf-8'))
        c = json.load(open(cp, encoding='utf-8'))
        o = []
        walk(e.get('entries', {}), c.get('entries', {}), [], o)
        for path, en, cn in o:
            rows.append((pack, path, en, cn))
    return rows


def load_all():
    rows = []
    for k in REPOS:
        for pack, path, en, cn in load_pairs(k):
            rows.append((k, pack, path, en, cn))
    return rows
