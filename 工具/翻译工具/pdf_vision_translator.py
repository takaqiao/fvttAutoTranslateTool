import base64
import concurrent.futures
import hashlib
import json
import os
import re
import time
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


CONFIG_PATH = Path("pdf_vision_config.json")
DEFAULT_CONFIG = {
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",
    "model": "gpt-5.2",
    "input_pdf_path": "",
    "output_dir": "pdf_output",
    "glossary_path": "glossary.json",
    "prompt_profile": "default",
    "max_workers": 4,
    "target_rpm": 240,
    "max_retries": 3,
    "render_dpi": 180,
    "neighbor_vision_enabled": True,
    "neighbor_vision_dpi": 110,
    "neighbor_context_chars": 1200,
    "current_ref_chars": 12000,
    "page_start": 1,
    "page_end": None,
    "cache_enabled": True,
    "cache_path": "pdf_vision_cache.json",
    "log_path": "pdf_vision_run.log",
    "output_markdown": True,
    "output_json": True,
    "output_text_reference": True,
    "batch_size": 40,
    "top_n_slowest": 20
}


_log_lock = Lock()
_cache_lock = Lock()
_stats_lock = Lock()


def _write_log(log_path: Path, msg: str):
    ts = time.strftime("%H:%M:%S", time.localtime())
    line = f"[{ts}] {msg}\n"
    with _log_lock:
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass


def _snip(text: str, limit: int = 100) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            user_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(user_cfg, dict):
                cfg.update(user_cfg)
        except Exception:
            pass
    else:
        try:
            CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"已生成默认配置：{CONFIG_PATH}")
        except Exception:
            pass
    return cfg


def _hash_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _strip_code_like_tokens(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"@UUID\[[^\]]+\]", " ", text)
    text = re.sub(r"@Compendium\[[^\]]+\]", " ", text)
    text = re.sub(r"@Localize\[[^\]]+\]", " ", text)
    text = re.sub(r"\[\[.*?\]\]", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def _contains_english(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", _strip_code_like_tokens(text)))


def _sanitize_ws(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[ \t\x0b\x0c\r]+", " ", text).replace("\u00a0", " ").strip()


def _load_cache(path: Path, enabled: bool) -> dict:
    if not enabled or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(path: Path, cache: dict, enabled: bool):
    if not enabled:
        return
    try:
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _load_glossary(path: Path):
    if not path.exists():
        return {}, [], 0
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}, [], 0
    if not isinstance(data, dict):
        return {}, [], 0

    buckets = {}
    terms_for_replace = []
    count = 0

    for en_raw, cn_raw in data.items():
        en = str(en_raw).strip()
        if not en:
            continue
        if isinstance(cn_raw, list):
            cns = [str(x).strip() for x in cn_raw if str(x).strip()]
            cn = cns[0] if cns else ""
        else:
            cn = str(cn_raw).strip()
            cns = [cn] if cn else []
        if not cns:
            continue

        letters = re.sub(r"[^A-Za-z]", "", en)
        if letters:
            pattern = r"(?<![A-Za-z])" + r"\s*".join(list(letters)) + r"(?![A-Za-z])"
            first = letters[0].lower()
            length = len(letters)
            length_bucket = (length // 5) * 5
            buckets.setdefault(first, {}).setdefault(length_bucket, []).append((
                length,
                re.compile(pattern, flags=re.IGNORECASE),
                cns,
                en,
            ))

        escaped_en = re.escape(en)
        replace_pattern = re.compile(rf"(?<![A-Za-z]){escaped_en}(?![A-Za-z])", flags=re.IGNORECASE)
        terms_for_replace.append((len(en), replace_pattern, cn))
        count += 1

    for first in list(buckets.keys()):
        for length_bucket in list(buckets[first].keys()):
            buckets[first][length_bucket].sort(key=lambda x: x[0], reverse=True)

    terms_for_replace.sort(key=lambda x: x[0], reverse=True)
    return buckets, terms_for_replace, count


def _find_glossary_matches(text: str, glossary_buckets: dict):
    if not text or not glossary_buckets:
        return []
    letters = {w[0].lower() for w in re.findall(r"[A-Za-z]+", text)}
    if not letters:
        return []
    matched = []
    for first in letters:
        length_map = glossary_buckets.get(first)
        if not length_map:
            continue
        for bucket_terms in length_map.values():
            for _, pattern, cns, en in bucket_terms:
                if pattern.search(text):
                    matched.append((en, cns[0]))
    dedup = {}
    for en, cn in matched:
        dedup[en.lower()] = (en, cn)
    out = list(dedup.values())
    out.sort(key=lambda x: len(x[0]), reverse=True)
    return out


def _apply_glossary_post_translation(text: str, terms_for_replace):
    if not text or not terms_for_replace:
        return text
    out = text
    for _, pattern, cn in terms_for_replace:
        out = pattern.sub(cn, out)
    return out


def _parse_json_block(raw: str) -> dict:
    if not raw:
        return {}
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, flags=re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = max(int(rpm), 1)
        self.capacity = self.rpm
        self.tokens = self.capacity
        self.last_refill = time.time()
        self.min_interval = 60.0 / self.rpm
        self.last_request = 0.0
        self.lock = Lock()

    def wait(self):
        with self.lock:
            while True:
                now = time.time()
                elapsed = now - self.last_refill
                if elapsed >= 60.0:
                    refill = int(elapsed // 60.0) * self.capacity
                    self.tokens = min(self.capacity, self.tokens + refill)
                    self.last_refill = now

                if self.tokens <= 0:
                    sleep_time = max(0.5, 60.0 - (now - self.last_refill))
                    time.sleep(sleep_time)
                    continue

                spacing = self.last_request + self.min_interval - now
                if spacing > 0:
                    time.sleep(spacing)
                    continue

                self.tokens -= 1
                self.last_request = time.time()
                break


class PdfVisionTranslator:
    def __init__(self, cfg: dict, glossary_buckets: dict, glossary_replace_terms: list, log_path: Path):
        api_key = cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("未设置 openai_api_key 且环境变量 OPENAI_API_KEY 为空")
        self.client = OpenAI(api_key=api_key, base_url=cfg.get("openai_base_url", "https://api.openai.com/v1"))
        self.model = cfg.get("model", "gpt-5.2")
        self.max_retries = int(cfg.get("max_retries", 3))
        self.prompt_profile = str(cfg.get("prompt_profile", "default")).strip().lower()
        self.neighbor_context_chars = max(int(cfg.get("neighbor_context_chars", 1200)), 200)
        self.current_ref_chars = max(int(cfg.get("current_ref_chars", 12000)), 2000)
        self.glossary_buckets = glossary_buckets
        self.glossary_replace_terms = glossary_replace_terms
        self.log_path = log_path

    def _build_instruction(self, glossary_matches):
        if self.prompt_profile == "dscryb":
            profile = (
                "你是资深TRPG场景文本翻译。"
                "将英文翻译为简体中文，保持沉浸感、画面感和自然中文节奏。"
            )
        else:
            profile = (
                "你是资深PF2E冒险文本翻译。"
                "将英文翻译为简体中文，术语风格保持一致，规则词汇准确。"
            )

        glossary_hint = ""
        if glossary_matches:
            pairs = [f"{en} -> {cn}" for en, cn in glossary_matches[:60]]
            glossary_hint = "术语优先参考：" + "；".join(pairs)

        return (
            profile
            + glossary_hint
            + "只翻译当前页，不要翻译上一页或下一页。"
            "你会收到当前页图片、当前页文本抽取、上一页摘要、下一页摘要。"
            "在启用时还会收到上一页/下一页图片，它们仅用于上下文理解。"
            "上一页和下一页仅用于理解语境，不可并入输出。"
            "必须返回严格JSON对象，字段为："
            "translated_page（当前页中文译文，保留段落结构），"
            "vision_extracted_text（你从当前页视觉识别得到的英文/原文文本，尽量完整）。"
            "不要返回任何额外字段，不要解释。"
        )

    def translate_page(
        self,
        page_num: int,
        image_b64: str,
        curr_ref_text: str,
        prev_ref_text: str,
        next_ref_text: str,
        prev_image_b64: str = "",
        next_image_b64: str = "",
    ):
        glossary_matches = _find_glossary_matches(curr_ref_text, self.glossary_buckets)
        instruction = self._build_instruction(glossary_matches)

        prev_snip = _snip(prev_ref_text, self.neighbor_context_chars)
        curr_snip = _snip(curr_ref_text, self.current_ref_chars)
        next_snip = _snip(next_ref_text, self.neighbor_context_chars)

        user_payload = (
            f"当前页码: {page_num}\n"
            f"上一页文本摘要(仅上下文):\n{prev_snip}\n\n"
            f"当前页文本抽取(用于查漏补缺):\n{curr_snip}\n\n"
            f"下一页文本摘要(仅上下文):\n{next_snip}\n"
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                content = [
                    {"type": "input_text", "text": user_payload},
                ]
                if prev_image_b64:
                    content.append({"type": "input_text", "text": "上一页图片（仅上下文参考，不可翻译进输出）"})
                    content.append({"type": "input_image", "image_url": f"data:image/png;base64,{prev_image_b64}"})
                content.append({"type": "input_text", "text": "当前页图片（只翻译这一页）"})
                content.append({"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"})
                if next_image_b64:
                    content.append({"type": "input_text", "text": "下一页图片（仅上下文参考，不可翻译进输出）"})
                    content.append({"type": "input_image", "image_url": f"data:image/png;base64,{next_image_b64}"})

                resp = self.client.responses.create(
                    model=self.model,
                    instructions=instruction,
                    input=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                )
                text = (resp.output_text or "").strip()
                obj = _parse_json_block(text)
                translated = _sanitize_ws(str(obj.get("translated_page", "")))
                vision_text = _sanitize_ws(str(obj.get("vision_extracted_text", "")))

                if not translated:
                    raise RuntimeError("模型未返回 translated_page")

                translated = _apply_glossary_post_translation(translated, self.glossary_replace_terms)
                return translated, vision_text
            except Exception as e:
                if attempt == self.max_retries:
                    _write_log(self.log_path, f"ERROR page={page_num} translate failed | {e}")
                    return "", ""
                _write_log(self.log_path, f"RETRY page={page_num} {attempt}/{self.max_retries} | {e}")
                time.sleep(attempt)
        return "", ""


def _render_page_to_png_b64(page, dpi: int) -> str:
    scale = max(float(dpi), 72.0) / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")
    return base64.b64encode(png_bytes).decode("ascii")


def _extract_page_text(page) -> str:
    txt = page.get_text("text") or ""
    txt = txt.replace("\u00ad", "")
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def main():
    cfg = _load_config()
    log_path = Path(cfg.get("log_path", "pdf_vision_run.log"))

    if fitz is None:
        print("❌ 依赖缺失：请先安装 PyMuPDF（pip install pymupdf）")
        return

    pdf_path = Path(str(cfg.get("input_pdf_path", "")).strip())
    if not str(pdf_path):
        print(f"❌ 请先在 {CONFIG_PATH} 里填写 input_pdf_path")
        return
    if not pdf_path.exists():
        print(f"❌ PDF 文件不存在：{pdf_path}")
        return

    output_dir = Path(cfg.get("output_dir", "pdf_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_enabled = bool(cfg.get("cache_enabled", True))
    cache_path = Path(cfg.get("cache_path", "pdf_vision_cache.json"))
    cache = _load_cache(cache_path, cache_enabled)

    glossary_path = Path(cfg.get("glossary_path", "glossary.json"))
    glossary_buckets, glossary_replace_terms, glossary_count = _load_glossary(glossary_path)
    print(f"术语表加载：{glossary_count} 条")

    limiter = RateLimiter(int(cfg.get("target_rpm", 240)))
    translator = PdfVisionTranslator(cfg, glossary_buckets, glossary_replace_terms, log_path)
    neighbor_vision_enabled = bool(cfg.get("neighbor_vision_enabled", True))
    neighbor_vision_dpi = max(int(cfg.get("neighbor_vision_dpi", 110)), 72)
    render_dpi = max(int(cfg.get("render_dpi", 180)), 72)
    batch_size = int(cfg.get("batch_size", 40) or 0)

    if render_dpi > 360:
        print(f"⚠️ 当前 render_dpi={render_dpi}，可能显著增加耗时和失败率，建议 280~360")
    if neighbor_vision_enabled and neighbor_vision_dpi > render_dpi:
        print("⚠️ neighbor_vision_dpi 高于 render_dpi，通常无收益且更耗 token")

    page_start = max(int(cfg.get("page_start", 1) or 1), 1)
    page_end_cfg = cfg.get("page_end", None)

    _write_log(log_path, f"START pdf={pdf_path} glossary={glossary_count} cache={len(cache)}")

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        page_end = total_pages if page_end_cfg in (None, "", 0) else min(int(page_end_cfg), total_pages)
        if page_start > page_end:
            print(f"❌ page_start({page_start}) > page_end({page_end})")
            return

        selected_pages = list(range(page_start, page_end + 1))
        effective_batch_size = batch_size if batch_size > 0 else len(selected_pages)
        batches = [
            selected_pages[i:i + effective_batch_size]
            for i in range(0, len(selected_pages), effective_batch_size)
        ]

        print(
            f"PDF页数：{total_pages}，处理范围：{page_start}-{page_end}，"
            f"分批：{len(batches)} 批（batch_size={effective_batch_size}）"
        )

        ref_text_cache = {}

        def get_ref_text(page_num: int) -> str:
            if page_num < 1 or page_num > total_pages:
                return ""
            if page_num not in ref_text_cache:
                ref_text_cache[page_num] = _extract_page_text(doc[page_num - 1])
            return ref_text_cache[page_num]

        for p in selected_pages:
            get_ref_text(p)

        max_workers = max(int(cfg.get("max_workers", 4)), 1)
        top_n_slowest = max(int(cfg.get("top_n_slowest", 20)), 0)

        results = {}
        stats = {"translated": 0, "cached": 0, "failed": 0, "skipped": 0}
        slowest = []

        def worker(item: dict):
            page = item["page"]
            curr_ref = item["ref_text"]
            curr_hash = _hash_text(curr_ref)
            prev_ref = item.get("prev_ref", "")
            next_ref = item.get("next_ref", "")
            prev_image_b64 = item.get("prev_image_b64", "")
            next_image_b64 = item.get("next_image_b64", "")

            key = f"{pdf_path.name}|p{page}"
            cached = cache.get(key, {}) if cache_enabled else {}
            if cache_enabled and isinstance(cached, dict) and cached.get("hash") == curr_hash:
                with _stats_lock:
                    stats["cached"] += 1
                return page, {
                    "page": page,
                    "translated": cached.get("translated", ""),
                    "vision_extracted_text": cached.get("vision_extracted_text", ""),
                    "text_reference": curr_ref,
                    "cached": True,
                }

            if not item["image_b64"] and not _contains_english(curr_ref):
                with _stats_lock:
                    stats["skipped"] += 1
                return page, {
                    "page": page,
                    "translated": "",
                    "vision_extracted_text": "",
                    "text_reference": curr_ref,
                    "cached": False,
                }

            limiter.wait()
            t0 = time.time()

            translated, vision_text = translator.translate_page(
                page_num=page,
                image_b64=item["image_b64"],
                curr_ref_text=curr_ref,
                prev_ref_text=prev_ref,
                next_ref_text=next_ref,
                prev_image_b64=prev_image_b64,
                next_image_b64=next_image_b64,
            )
            elapsed = time.time() - t0

            with _stats_lock:
                slowest.append((elapsed, page))

            if not translated and _contains_english(curr_ref):
                with _stats_lock:
                    stats["failed"] += 1
                return page, {
                    "page": page,
                    "translated": "",
                    "vision_extracted_text": vision_text,
                    "text_reference": curr_ref,
                    "cached": False,
                }

            with _stats_lock:
                stats["translated"] += 1

            if cache_enabled:
                with _cache_lock:
                    cache[key] = {
                        "hash": curr_hash,
                        "translated": translated,
                        "vision_extracted_text": vision_text,
                        "updated_at": int(time.time()),
                    }

            return page, {
                "page": page,
                "translated": translated,
                "vision_extracted_text": vision_text,
                "text_reference": curr_ref,
                "cached": False,
            }

        for batch_idx, batch_pages in enumerate(batches, start=1):
            pages_data = []
            current_image_map = {}
            neighbor_image_map = {}

            for p in batch_pages:
                try:
                    current_image_map[p] = _render_page_to_png_b64(doc[p - 1], render_dpi)
                except Exception:
                    current_image_map[p] = ""

            if neighbor_vision_enabled:
                neighbor_pages = set()
                for p in batch_pages:
                    if p - 1 >= 1:
                        neighbor_pages.add(p - 1)
                    if p + 1 <= total_pages:
                        neighbor_pages.add(p + 1)
                for p in sorted(neighbor_pages):
                    if p in current_image_map:
                        continue
                    try:
                        neighbor_image_map[p] = _render_page_to_png_b64(doc[p - 1], neighbor_vision_dpi)
                    except Exception:
                        neighbor_image_map[p] = ""

            for p in batch_pages:
                prev_p = p - 1 if p - 1 >= 1 else None
                next_p = p + 1 if p + 1 <= total_pages else None

                prev_img = ""
                next_img = ""
                if neighbor_vision_enabled:
                    if prev_p is not None:
                        prev_img = current_image_map.get(prev_p) or neighbor_image_map.get(prev_p, "")
                    if next_p is not None:
                        next_img = current_image_map.get(next_p) or neighbor_image_map.get(next_p, "")

                pages_data.append({
                    "page": p,
                    "ref_text": get_ref_text(p),
                    "prev_ref": get_ref_text(prev_p) if prev_p else "",
                    "next_ref": get_ref_text(next_p) if next_p else "",
                    "image_b64": current_image_map.get(p, ""),
                    "prev_image_b64": prev_img,
                    "next_image_b64": next_img,
                })

            desc = f"PDF翻译 B{batch_idx}/{len(batches)}"
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(worker, item) for item in pages_data]
                for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc, dynamic_ncols=True):
                    try:
                        page, value = fut.result()
                        results[page] = value
                    except Exception as e:
                        _write_log(log_path, f"ERROR future failed | {e}")

            _save_cache(cache_path, cache, cache_enabled)
            _write_log(log_path, f"BATCH_DONE {batch_idx}/{len(batches)} pages={len(batch_pages)}")

    ordered = [results[p] for p in selected_pages if p in results]

    stem = pdf_path.stem
    if bool(cfg.get("output_json", True)):
        out_json = output_dir / f"{stem}.translated.pages.json"
        out_json.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已输出：{out_json}")

    if bool(cfg.get("output_markdown", True)):
        out_md = output_dir / f"{stem}.translated.md"
        chunks = [f"# {stem} 翻译稿\n"]
        for item in ordered:
            page = item["page"]
            tr = (item.get("translated") or "").strip()
            ref = (item.get("text_reference") or "").strip()
            vis = (item.get("vision_extracted_text") or "").strip()

            chunks.append(f"\n## Page {page}\n")
            chunks.append((tr if tr else "（本页无可翻译内容或翻译失败）") + "\n")
            chunks.append("\n<details><summary>Text Reference (PDF抽取)</summary>\n\n")
            chunks.append((ref if ref else "（无文本抽取）") + "\n")
            chunks.append("\n</details>\n")
            chunks.append("\n<details><summary>Vision Extracted Text (视觉识别)</summary>\n\n")
            chunks.append((vis if vis else "（无视觉识别文本）") + "\n")
            chunks.append("\n</details>\n")

        out_md.write_text("".join(chunks), encoding="utf-8")
        print(f"已输出：{out_md}")

    if bool(cfg.get("output_text_reference", True)):
        out_ref = output_dir / f"{stem}.text_reference.txt"
        ref_chunks = []
        for item in ordered:
            ref_chunks.append(f"\n===== Page {item['page']} =====\n")
            ref_chunks.append((item.get("text_reference") or "") + "\n")
        out_ref.write_text("".join(ref_chunks), encoding="utf-8")
        print(f"已输出：{out_ref}")

    _save_cache(cache_path, cache, cache_enabled)

    slowest.sort(key=lambda x: x[0], reverse=True)
    if top_n_slowest > 0 and slowest:
        print("最慢页面：")
        for sec, page in slowest[:top_n_slowest]:
            print(f"  {sec:.2f}s | Page {page}")

    print(
        "完成："
        f"翻译={stats['translated']} "
        f"缓存命中={stats['cached']} "
        f"跳过={stats['skipped']} "
        f"失败={stats['failed']}"
    )
    _write_log(
        log_path,
        "DONE "
        f"translated={stats['translated']} cached={stats['cached']} skipped={stats['skipped']} failed={stats['failed']}",
    )


if __name__ == "__main__":
    main()
