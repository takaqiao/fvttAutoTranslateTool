#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速汉化当前文件夹下所有文件名（不翻译扩展名），中文插入到英文前。
使用环境变量 OPENAI_API_KEY，模型 gpt-5.2。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Dict

from openai import OpenAI

BATCH_SIZE = 50  # 超过该数量会分批请求
MODEL = "gpt-5.2"

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def starts_with_cjk(text: str) -> bool:
    text = text.lstrip()
    return bool(text) and bool(CJK_RE.match(text[0]))


def split_name(filename: str) -> tuple[str, str]:
    p = Path(filename)
    return p.stem, p.suffix


def unique_name(target: Path, reserved: set[str]) -> Path:
    if target.name.lower() not in reserved:
        return target

    base = target.stem
    ext = target.suffix
    i = 1
    while True:
        cand = target.with_name(f"{base} ({i}){ext}")
        if cand.name.lower() not in reserved:
            return cand
        i += 1


def chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def request_translations(client: OpenAI, names: List[str]) -> List[str]:
    system = (
        "你是TRPG翻译助手，优先采用DND/PF2E术语。"
        "将英文标题翻译成简体中文，保留专有名词的常见译名。"
        "仅返回JSON数组，顺序与输入一致，不要附加多余文本。"
    )
    user = {
        "type": "text",
        "text": json.dumps(names, ensure_ascii=False),
    }

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [user]},
        ],
    )

    text = resp.output_text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"模型返回无法解析为JSON: {text}") from e

    if not isinstance(data, list) or len(data) != len(names):
        raise RuntimeError("模型返回的数组长度与输入不一致")

    return [str(x) for x in data]


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未设置环境变量 OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    root = Path.cwd()
    files = [p for p in root.iterdir() if p.is_file()]

    # 收集待翻译文件名
    targets: Dict[Path, str] = {}
    names: List[str] = []
    for p in files:
        base, _ = split_name(p.name)
        if not base:
            continue
        # 已含中文且以中文开头则跳过，避免重复插入
        if contains_cjk(base) and starts_with_cjk(base):
            continue
        targets[p] = base
        names.append(base)

    if not names:
        print("没有需要处理的文件名。")
        return

    # 翻译
    translations: Dict[str, str] = {}
    for batch in chunk_list(names, BATCH_SIZE):
        zh_list = request_translations(client, batch)
        for en, zh in zip(batch, zh_list):
            translations[en] = zh.strip()

    # 预占用现有文件名，避免冲突
    reserved = {p.name.lower() for p in files}

    # 执行重命名
    for p, en in targets.items():
        zh = translations.get(en, "").strip()
        if not zh:
            continue
        base, ext = split_name(p.name)
        new_name = f"{zh} {base}{ext}"
        target_path = p.with_name(new_name)
        target_path = unique_name(target_path, reserved)
        if target_path.name.lower() == p.name.lower():
            continue
        reserved.add(target_path.name.lower())
        p.rename(target_path)

    print("完成。")


if __name__ == "__main__":
    main()
