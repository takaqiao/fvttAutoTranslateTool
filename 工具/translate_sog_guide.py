"""
Translate SoGGuide DOCX files to Chinese using glossary terms.
Preserves hyperlinks and formatting.
"""

import json
import os
import re
import time
import sys
from docx import Document
from docx.oxml.ns import qn
from openai import OpenAI

# --- Config ---
GLOSSARY_SOG_PATH = "C:/Users/Taka/Desktop/fvtt/glossary_sog.json"
GLOSSARY_PATH = "C:/Users/Taka/Desktop/fvtt/glossary.json"
REF_PATH = "C:/Users/Taka/Desktop/fvttpublish/pf2e-compendium-extra/compendium/cn/pf2e-season-of-ghosts.adventures.json"
INPUT_DIR = "C:/Users/Taka/Desktop/fvtt/SoGGuide"
OUTPUT_DIR = "C:/Users/Taka/Desktop/fvtt/SoGGuide"

BATCH_SIZE = 12  # paragraphs per API call
MODEL = "gpt-4o"

client = OpenAI()


def load_glossaries():
    """Load and merge glossaries. glossary_sog takes priority."""
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(GLOSSARY_SOG_PATH, "r", encoding="utf-8") as f:
        sog = json.load(f)

    merged = {}
    for k, v in base.items():
        clean_key = re.sub(r"\(.*?\)$", "", k).strip()
        if isinstance(v, list):
            v = v[0]
        merged[clean_key] = v

    for k, v in sog.items():
        if isinstance(v, list):
            v = v[0]
        merged[k] = v

    return merged


def load_reference():
    """Load reference translations from completed SoG adventure module."""
    with open(REF_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    ref = {}
    entry = data["entries"]["Season of Ghosts"]

    for section in ["actors", "journals", "scenes", "macros", "items"]:
        items = entry.get(section, {})
        for k, v in items.items():
            if isinstance(v, dict) and "name" in v:
                ref[k] = v["name"]
            elif isinstance(v, str):
                ref[k] = v

    folders = entry.get("folders", {})
    for k, v in folders.items():
        if isinstance(v, str):
            ref[k] = v

    return ref


def build_glossary_prompt(glossary, reference, text_batch):
    """Build a compact glossary subset relevant to the text batch."""
    combined_text = " ".join(text_batch).lower()

    relevant = {}
    for en, zh in glossary.items():
        if en.lower() in combined_text:
            relevant[en] = zh

    for en, zh in reference.items():
        if en.lower() in combined_text:
            relevant[en] = zh

    sorted_terms = sorted(relevant.items(), key=lambda x: -len(x[0]))

    if len(sorted_terms) > 200:
        sorted_terms = sorted_terms[:200]

    return sorted_terms


def extract_paragraph_segments(para):
    """Extract text segments from a paragraph, marking hyperlinks.

    Returns list of segments:
    [
        {"type": "text", "text": "...", "t_elements": [...]},
        {"type": "link", "text": "...", "t_elements": [...], "link_idx": 0},
        ...
    ]
    """
    segments = []
    link_counter = 0

    for child in para._element:
        if child.tag == qn("w:r"):
            t_elems = child.findall(qn("w:t"))
            text = "".join(t.text or "" for t in t_elems)
            if text:
                segments.append({
                    "type": "text",
                    "text": text,
                    "t_elements": t_elems,
                })
        elif child.tag == qn("w:hyperlink"):
            t_elems = []
            for run in child.findall(qn("w:r")):
                t_elems.extend(run.findall(qn("w:t")))
            text = "".join(t.text or "" for t in t_elems)
            if text:
                segments.append({
                    "type": "link",
                    "text": text,
                    "t_elements": t_elems,
                    "link_idx": link_counter,
                })
            link_counter += 1

    return segments


def build_marked_text(segments):
    """Build paragraph text with hyperlink markers.

    E.g. "Visit the <<0:PF2E Discord>> for help"
    """
    parts = []
    for seg in segments:
        if seg["type"] == "link":
            parts.append(f"<<{seg['link_idx']}:{seg['text']}>>")
        else:
            parts.append(seg["text"])
    return "".join(parts)


def parse_translated_segments(translated_text, segments):
    """Parse translated text back into segments, respecting link markers.

    Returns a list of (segment_index, new_text) pairs.
    """
    # Find all <<idx:text>> markers in translated text
    # Pattern: <<number:translated_link_text>>
    pattern = r"<<(\d+):(.*?)>>"

    result = {}
    link_segments = {seg["link_idx"]: i for i, seg in enumerate(segments) if seg["type"] == "link"}

    # Split by link markers
    parts = re.split(pattern, translated_text)
    # parts = [text_before, link_idx, link_text, text_between, link_idx, link_text, ...]

    text_parts = []
    i = 0
    while i < len(parts):
        if i + 2 < len(parts):
            # Check if next part is a link index
            try:
                link_idx = int(parts[i + 1])
                # Current part is text before the link
                text_parts.append(("text", parts[i]))
                text_parts.append(("link", link_idx, parts[i + 2]))
                i += 3
                continue
            except (ValueError, IndexError):
                pass
        text_parts.append(("text", parts[i]))
        i += 1

    # Now assign text to segments
    # Collect all non-link text parts
    non_link_text = "".join(p[1] for p in text_parts if p[0] == "text")

    # Assign link texts
    for p in text_parts:
        if p[0] == "link":
            link_idx = p[1]
            if link_idx in link_segments:
                seg_idx = link_segments[link_idx]
                result[seg_idx] = p[2]

    # Assign non-link text to text segments
    text_seg_indices = [i for i, seg in enumerate(segments) if seg["type"] == "text"]
    if text_seg_indices:
        # Put all non-link text in the first text segment
        result[text_seg_indices[0]] = non_link_text
        for idx in text_seg_indices[1:]:
            result[idx] = ""

    return result


def apply_segment_translations(segments, translations):
    """Apply translated text back to the DOCX t elements."""
    for seg_idx, new_text in translations.items():
        seg = segments[seg_idx]
        t_elements = seg["t_elements"]
        if t_elements:
            t_elements[0].text = new_text
            # Preserve space attribute
            if new_text and (new_text[0] == " " or new_text[-1] == " "):
                t_elements[0].set(qn("xml:space"), "preserve")
            for t in t_elements[1:]:
                t.text = ""


def translate_batch(marked_texts, glossary_terms):
    """Translate a batch of marked texts using OpenAI."""
    if not marked_texts:
        return marked_texts

    glossary_lines = []
    for en, zh in glossary_terms:
        glossary_lines.append(f"  {en} = {zh}")
    glossary_section = "\n".join(glossary_lines)

    numbered = []
    for i, t in enumerate(marked_texts):
        numbered.append(f"[{i}] {t}")
    text_section = "\n".join(numbered)

    prompt = f"""你是一名专业的TTRPG翻译。请将以下Pathfinder 2e《肆季鬼志》(Season of Ghosts) GM指南的英文段落翻译成简体中文。

## 术语表（必须严格使用以下译名）：
{glossary_section}

## 翻译要求：
1. 严格使用术语表中的译名，不要自行翻译术语表中已有的词汇
2. 保持原文的语气和风格（这是一份GM指南/FAQ，语气比较随意口语化）
3. 保留所有英文专有名词的原文（如人名、地名等术语表中没有的），格式为"中文翻译 英文原名"或直接保留英文
4. 如果段落是纯URL或纯标点，原样返回
5. **重要**：文本中的 <<数字:文字>> 标记是超链接，翻译时必须保留这种格式！只翻译冒号后面的文字，保留 << >> 和数字。例如 <<0:PF2E Discord>> 保持为 <<0:PF2E Discord>>，<<1:Travel Speed>> 翻译为 <<1:移动速度>>
6. 每个翻译结果前面加上对应的编号 [数字]

## 待翻译段落：
{text_section}

请逐段翻译，每段前标注编号如 [0] [1] 等。"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8000,
            )
            result_text = response.choices[0].message.content
            break
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print("  Failed after retries, returning originals")
                return marked_texts

    # Parse results
    translations = {}
    pattern = r"\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)"
    matches = re.findall(pattern, result_text, re.DOTALL)
    for idx_str, trans in matches:
        idx = int(idx_str)
        translations[idx] = trans.strip()

    result = []
    for i in range(len(marked_texts)):
        if i in translations:
            result.append(translations[i])
        else:
            result.append(marked_texts[i])

    return result


def translate_docx(input_path, output_path, glossary, reference):
    """Translate a DOCX file."""
    print(f"\n{'='*60}")
    print(f"Translating: {input_path}")
    print(f"Output: {output_path}")

    doc = Document(input_path)

    # Collect all paragraphs with translatable text
    para_data = []
    for i, para in enumerate(doc.paragraphs):
        segments = extract_paragraph_segments(para)
        if segments:
            full_text = "".join(s["text"] for s in segments)
            if full_text.strip():
                marked = build_marked_text(segments)
                para_data.append({
                    "index": i,
                    "segments": segments,
                    "marked_text": marked,
                    "full_text": full_text,
                })

    print(f"Total paragraphs with text: {len(para_data)}")

    # Translate in batches
    total_batches = (len(para_data) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(para_data))
        batch = para_data[start:end]

        batch_texts = [d["marked_text"] for d in batch]

        # Skip pure URLs
        skip_indices = set()
        for j, t in enumerate(batch_texts):
            raw = batch[j]["full_text"].strip()
            if re.match(r"^https?://\S+$", raw):
                skip_indices.add(j)
            elif re.match(r"^Last updated:.*$", raw):
                skip_indices.add(j)

        translatable_texts = [t for j, t in enumerate(batch_texts) if j not in skip_indices]
        translatable_batch_indices = [j for j in range(len(batch_texts)) if j not in skip_indices]

        if translatable_texts:
            plain_texts = [batch[j]["full_text"] for j in translatable_batch_indices]
            glossary_terms = build_glossary_prompt(glossary, reference, plain_texts)
            print(f"  Batch {batch_num+1}/{total_batches}: translating {len(translatable_texts)} paragraphs ({len(glossary_terms)} terms)")

            translated = translate_batch(translatable_texts, glossary_terms)

            # Apply translations
            for j, trans in zip(translatable_batch_indices, translated):
                segments = batch[j]["segments"]
                seg_translations = parse_translated_segments(trans, segments)
                apply_segment_translations(segments, seg_translations)
        else:
            print(f"  Batch {batch_num+1}/{total_batches}: skipped")

    # Also handle tables if any
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    segments = extract_paragraph_segments(para)
                    if segments:
                        full_text = "".join(s["text"] for s in segments)
                        if full_text.strip() and not re.match(r"^https?://\S+$", full_text.strip()):
                            marked = build_marked_text(segments)
                            glossary_terms = build_glossary_prompt(glossary, reference, [full_text])
                            translated = translate_batch([marked], glossary_terms)
                            if translated:
                                seg_translations = parse_translated_segments(translated[0], segments)
                                apply_segment_translations(segments, seg_translations)

    doc.save(output_path)
    print(f"Saved: {output_path}")


def main():
    print("Loading glossaries...")
    glossary = load_glossaries()
    print(f"  Merged glossary: {len(glossary)} terms")

    print("Loading reference translations...")
    reference = load_reference()
    print(f"  Reference: {len(reference)} entries")

    docx_files = [
        "Season of Ghosts GM Guide - Landing Page.docx",
        "Season of Ghosts GM Guide - Book 1.docx",
    ]

    for filename in docx_files:
        input_path = os.path.join(INPUT_DIR, filename)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(OUTPUT_DIR, f"{name}_zh{ext}")
        translate_docx(input_path, output_path, glossary, reference)

    print("\nAll files translated!")


if __name__ == "__main__":
    main()
