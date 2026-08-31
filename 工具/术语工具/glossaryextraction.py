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


CONFIG_PATH = Path("glossary_extract_config2.json")
DEFAULT_CONFIG = {
	"openai_api_key": "",
	"openai_base_url": "https://api.openai.com/v1",
	"model": "gpt-5.2",
	"pdf_root": "术语工具",
	"recursive": True,
	"pdf_glob": "*.pdf",
	"include_keywords": ["pf2e", "pathfinder", "adventure", "vaults", "otari", "冒险"],
	"exclude_keywords": [],
	"page_start": 1,
	"page_end": None,
	"pages_per_batch": 18,
	"max_workers": 64,
	"target_rpm": 9900,
	"max_retries": 3,
	"max_chars_per_batch": 32000,
	"cache_enabled": True,
	"cache_path": "glossary_extract_cache.json",
	"log_path": "glossary_extract.log",
	"output_json": "glossary_from_pdf_fog.json",
	"output_candidates_json": "glossary_candidates_from_pdf.json",
	"output_conflicts_json": "glossary_conflicts_from_pdf.json",
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


def _snip(text: str, limit: int = 120) -> str:
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


def _load_cache(path: Path, enabled: bool) -> dict:
	if not enabled or not path.exists():
		return {"batches": {}}
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
		if not isinstance(data, dict):
			return {"batches": {}}
		if not isinstance(data.get("batches"), dict):
			data["batches"] = {}
		return data
	except Exception:
		return {"batches": {}}


def _save_cache(path: Path, cache: dict, enabled: bool):
	if not enabled:
		return
	try:
		path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
	except Exception:
		pass


def _hash_text(text: str) -> str:
	return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _contains_zh(text: str) -> bool:
	return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _contains_en(text: str) -> bool:
	return bool(re.search(r"[A-Za-z]", text or ""))


def _normalize_ws(text: str) -> str:
	return re.sub(r"\s+", " ", (text or "")).strip()


def _norm_en(text: str) -> str:
	t = _normalize_ws(text)
	t = re.sub(r"\s*([,;/:&\-])\s*", r"\1", t)
	return t


def _norm_zh(text: str) -> str:
	return _normalize_ws(text)


def _find_pdfs(cfg: dict) -> list[Path]:
	root = Path(str(cfg.get("pdf_root", ".")).strip() or ".")
	recursive = bool(cfg.get("recursive", True))
	glob_pat = str(cfg.get("pdf_glob", "*.pdf")).strip() or "*.pdf"
	include_keywords = [str(x).lower() for x in cfg.get("include_keywords", []) if str(x).strip()]
	exclude_keywords = [str(x).lower() for x in cfg.get("exclude_keywords", []) if str(x).strip()]

	iterator = root.rglob(glob_pat) if recursive else root.glob(glob_pat)
	found = []
	for p in iterator:
		if not p.is_file() or p.suffix.lower() != ".pdf":
			continue
		tag = str(p).lower()
		if include_keywords and not any(k in tag for k in include_keywords):
			continue
		if exclude_keywords and any(k in tag for k in exclude_keywords):
			continue
		found.append(p)
	found.sort()
	return found


def _extract_pdf_pages_text(pdf_path: Path, page_start: int, page_end: int | None):
	out = []
	with fitz.open(pdf_path) as doc:
		total = len(doc)
		start = max(1, int(page_start))
		end = total if page_end is None else min(total, int(page_end))
		if end < start:
			return out, total
		for page_no in range(start, end + 1):
			page = doc[page_no - 1]
			text = page.get_text("text") or ""
			text = text.replace("\x00", "")
			out.append((page_no, text))
	return out, total


def _build_batches(pages_text: list[tuple[int, str]], pages_per_batch: int, max_chars_per_batch: int):
	batches = []
	pages_per_batch = max(1, int(pages_per_batch))
	max_chars_per_batch = max(2000, int(max_chars_per_batch))

	cur_pages = []
	cur_parts = []
	cur_chars = 0

	for page_no, text in pages_text:
		part = f"\n\n=== Page {page_no} ===\n{text}"
		part_len = len(part)

		need_split = (
			cur_pages
			and (len(cur_pages) >= pages_per_batch or cur_chars + part_len > max_chars_per_batch)
		)
		if need_split:
			batches.append((cur_pages[:], "".join(cur_parts)))
			cur_pages = []
			cur_parts = []
			cur_chars = 0

		cur_pages.append(page_no)
		cur_parts.append(part)
		cur_chars += part_len

	if cur_pages:
		batches.append((cur_pages, "".join(cur_parts)))
	return batches


def _regex_extract_pairs(text: str) -> list[tuple[str, str]]:
	if not text:
		return []
	pattern = re.compile(
		r"(?P<zh>[\u4e00-\u9fff][\u4e00-\u9fffA-Za-z0-9·・\-—\s]{0,80}?)\s*[（(]\s*"
		r"(?P<en>[A-Za-z][A-Za-z0-9'’\-\s,&/:]{0,100})\s*[）)]"
	)
	pairs = []
	for match in pattern.finditer(text):
		zh = _norm_zh(match.group("zh"))
		en = _norm_en(match.group("en"))
		if not zh or not en:
			continue
		if not _contains_zh(zh) or not _contains_en(en):
			continue
		pairs.append((en, zh))
	return pairs


def _parse_json_array(raw: str):
	if not raw:
		return []
	text = raw.strip()
	fence_match = re.search(r"```(?:json)?\s*(\[[\s\S]*\])\s*```", text, flags=re.IGNORECASE)
	if fence_match:
		text = fence_match.group(1).strip()
	else:
		start = text.find("[")
		end = text.rfind("]")
		if start != -1 and end != -1 and end > start:
			text = text[start:end + 1]
	try:
		arr = json.loads(text)
		return arr if isinstance(arr, list) else []
	except Exception:
		return []


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


class PairExtractor:
	def __init__(self, cfg: dict):
		api_key = cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")
		if not api_key:
			raise RuntimeError("未设置 openai_api_key 且环境变量 OPENAI_API_KEY 为空")
		self.client = OpenAI(api_key=api_key, base_url=cfg.get("openai_base_url", "https://api.openai.com/v1"))
		self.model = cfg.get("model", "gpt-5.2")
		self.max_retries = max(1, int(cfg.get("max_retries", 3)))

	def extract_pairs_from_batch(self, batch_text: str, source_tag: str):
		instructions = (
			"你是PF2E术语抽取助手。"
			"只提取文本中明确出现的‘中文（英文）’或‘中文(英文)’术语对。"
			"不要猜测，不要改写，不要补充。"
			"输出严格 JSON 数组，每项格式："
			"{\"english\":\"...\",\"chinese\":\"...\"}。"
			"若没有可提取项，输出 []。"
		)

		for attempt in range(1, self.max_retries + 1):
			try:
				resp = self.client.responses.create(
					model=self.model,
					instructions=instructions,
					input=batch_text,
				)
				raw = (resp.output_text or "").strip()
				arr = _parse_json_array(raw)
				pairs = []
				for row in arr:
					if not isinstance(row, dict):
						continue
					en = _norm_en(str(row.get("english", "")))
					zh = _norm_zh(str(row.get("chinese", "")))
					if not en or not zh:
						continue
					if not _contains_en(en) or not _contains_zh(zh):
						continue
					pairs.append((en, zh))
				return pairs
			except Exception as e:
				if attempt == self.max_retries:
					raise RuntimeError(f"AI 提取失败: {source_tag} | {e}") from e
				time.sleep(attempt)
		return []


def _merge_pairs(all_pairs: list[tuple[str, str]]):
	bucket = {}
	for en_raw, zh_raw in all_pairs:
		en = _norm_en(en_raw)
		zh = _norm_zh(zh_raw)
		if not en or not zh:
			continue
		k = en.lower()
		entry = bucket.setdefault(k, {"english": en, "zh_variants": {}, "count": 0})
		if len(en) > len(entry["english"]):
			entry["english"] = en
		entry["zh_variants"][zh] = entry["zh_variants"].get(zh, 0) + 1
		entry["count"] += 1

	glossary = {}
	candidates = []
	rows = sorted(bucket.values(), key=lambda x: x["count"], reverse=True)
	for row in rows:
		variants_sorted = sorted(row["zh_variants"].items(), key=lambda item: item[1], reverse=True)
		preferred_zh = variants_sorted[0][0] if variants_sorted else ""
		english = row["english"]
		if preferred_zh:
			glossary[english] = preferred_zh
		candidates.append(
			{
				"english": english,
				"preferred_zh": preferred_zh,
				"zh_variants": {k: v for k, v in variants_sorted},
				"occurrences": row["count"],
			}
		)
	return glossary, candidates


def _load_existing_glossary(path: Path) -> dict:
	if not path.exists():
		return {}
	try:
		data = json.loads(path.read_text(encoding="utf-8-sig"))
		if isinstance(data, dict):
			return data
	except Exception:
		pass
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
		if isinstance(data, dict):
			return data
	except Exception:
		pass
	return {}


def _append_glossary_with_conflicts(existing: dict, incoming: dict):
	merged = dict(existing)
	conflicts = []
	added = 0

	lower_existing = {str(k).strip().lower(): str(k) for k in merged.keys() if str(k).strip()}

	for en_raw, zh_raw in incoming.items():
		en = _norm_en(str(en_raw))
		zh = _norm_zh(str(zh_raw))
		if not en or not zh:
			continue

		if en in merged:
			old_zh = _norm_zh(str(merged.get(en, "")))
			if old_zh != zh:
				conflicts.append(
					{
						"english": en,
						"existing_zh": old_zh,
						"new_zh": zh,
						"reason": "exact_key_conflict",
					}
				)
			continue

		lk = en.lower()
		if lk in lower_existing:
			existing_key = lower_existing[lk]
			old_zh = _norm_zh(str(merged.get(existing_key, "")))
			if old_zh != zh:
				conflicts.append(
					{
						"english": en,
						"existing_key": existing_key,
						"existing_zh": old_zh,
						"new_zh": zh,
						"reason": "case_insensitive_conflict",
					}
				)
			continue

		merged[en] = zh
		lower_existing[lk] = en
		added += 1

	return merged, conflicts, added


def main():
	cfg = _load_config()
	log_path = Path(cfg.get("log_path", "glossary_extract.log"))

	if fitz is None:
		print("❌ 依赖缺失：请先安装 PyMuPDF（pip install pymupdf）")
		return

	pdfs = _find_pdfs(cfg)
	if not pdfs:
		print("❌ 未找到符合条件的 PDF，请检查 glossary_extract_config.json 的路径与关键词设置")
		return

	print(f"检测到 PDF：{len(pdfs)} 个")
	_write_log(log_path, f"START pdf_count={len(pdfs)}")

	cache_enabled = bool(cfg.get("cache_enabled", True))
	cache_path = Path(cfg.get("cache_path", "glossary_extract_cache.json"))
	cache = _load_cache(cache_path, cache_enabled)

	extractor = PairExtractor(cfg)
	limiter = RateLimiter(int(cfg.get("target_rpm", 180)))

	page_start = int(cfg.get("page_start", 1))
	page_end = cfg.get("page_end")
	pages_per_batch = int(cfg.get("pages_per_batch", 18))
	max_chars_per_batch = int(cfg.get("max_chars_per_batch", 32000))
	max_workers = max(1, int(cfg.get("max_workers", 4)))

	batch_jobs = []
	stats = {
		"pdf_count": len(pdfs),
		"batch_count": 0,
		"cache_hit": 0,
		"ai_ok": 0,
		"ai_fail": 0,
		"regex_only": 0,
	}

	for pdf_path in pdfs:
		try:
			pages_text, total_pages = _extract_pdf_pages_text(pdf_path, page_start, page_end)
		except Exception as e:
			_write_log(log_path, f"ERROR read_pdf {pdf_path} | {e}")
			print(f"⚠️ 读取失败：{pdf_path} | {e}")
			continue

		if not pages_text:
			_write_log(log_path, f"SKIP empty_text {pdf_path}")
			continue

		batches = _build_batches(pages_text, pages_per_batch, max_chars_per_batch)
		_write_log(log_path, f"PDF {pdf_path} total_pages={total_pages} batches={len(batches)}")

		for pages, text in batches:
			if not pages:
				continue
			batch_key = f"{pdf_path.resolve()}|{pages[0]}-{pages[-1]}"
			batch_hash = _hash_text(text)
			batch_jobs.append((pdf_path, pages, text, batch_key, batch_hash))

	if not batch_jobs:
		print("❌ 没有可处理的文本批次")
		return

	stats["batch_count"] = len(batch_jobs)
	all_pairs = []

	def worker(job):
		pdf_path, pages, text, batch_key, batch_hash = job
		source_tag = f"{pdf_path.name} p{pages[0]}-{pages[-1]}"

		with _cache_lock:
			item = cache["batches"].get(batch_key)
			if (
				cache_enabled
				and isinstance(item, dict)
				and item.get("hash") == batch_hash
				and isinstance(item.get("pairs"), list)
			):
				cached_pairs = []
				for pair in item["pairs"]:
					if not isinstance(pair, list) or len(pair) != 2:
						continue
					cached_pairs.append((_norm_en(str(pair[0])), _norm_zh(str(pair[1]))))
				with _stats_lock:
					stats["cache_hit"] += 1
				return cached_pairs

		regex_pairs = _regex_extract_pairs(text)

		try:
			limiter.wait()
			ai_pairs = extractor.extract_pairs_from_batch(text, source_tag)
			pairs = ai_pairs if ai_pairs else regex_pairs
			with _stats_lock:
				stats["ai_ok"] += 1
				if not ai_pairs and regex_pairs:
					stats["regex_only"] += 1
		except Exception as e:
			_write_log(log_path, f"ERROR ai_batch {source_tag} | {e} | text={_snip(text)}")
			pairs = regex_pairs
			with _stats_lock:
				stats["ai_fail"] += 1
				if regex_pairs:
					stats["regex_only"] += 1

		if cache_enabled:
			with _cache_lock:
				cache["batches"][batch_key] = {
					"hash": batch_hash,
					"pairs": [[en, zh] for en, zh in pairs],
				}
		return pairs

	with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
		iterator = executor.map(worker, batch_jobs)
		for pairs in tqdm(iterator, total=len(batch_jobs), desc="术语提取", dynamic_ncols=True):
			if not pairs:
				continue
			all_pairs.extend(pairs)

	glossary, candidates = _merge_pairs(all_pairs)
	output_json = Path(cfg.get("output_json", "glossary_from_pdf.json"))
	output_candidates_json = Path(cfg.get("output_candidates_json", "glossary_candidates_from_pdf.json"))
	output_conflicts_json = Path(cfg.get("output_conflicts_json", "glossary_conflicts_from_pdf.json"))

	existing_glossary = _load_existing_glossary(output_json)
	final_glossary, conflicts, added_count = _append_glossary_with_conflicts(existing_glossary, glossary)

	output_json.write_text(json.dumps(final_glossary, ensure_ascii=False, indent=2), encoding="utf-8")
	payload = {
		"meta": {
			"pdf_count": stats["pdf_count"],
			"batch_count": stats["batch_count"],
			"cache_hit": stats["cache_hit"],
			"ai_ok": stats["ai_ok"],
			"ai_fail": stats["ai_fail"],
			"regex_only": stats["regex_only"],
			"total_pairs": len(all_pairs),
			"detected_terms": len(glossary),
			"existing_terms": len(existing_glossary),
			"added_terms": added_count,
			"conflicts": len(conflicts),
			"total_terms": len(final_glossary),
			"generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
		},
		"terms": candidates,
	}
	output_candidates_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
	conflict_payload = {
		"meta": {
			"existing_terms": len(existing_glossary),
			"detected_terms": len(glossary),
			"added_terms": added_count,
			"conflicts": len(conflicts),
			"generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
		},
		"conflicts": conflicts,
	}
	output_conflicts_json.write_text(json.dumps(conflict_payload, ensure_ascii=False, indent=2), encoding="utf-8")
	_save_cache(cache_path, cache, cache_enabled)

	_write_log(
		log_path,
		(
			f"DONE terms={len(final_glossary)} added={added_count} conflicts={len(conflicts)} pairs={len(all_pairs)} "
			f"cache_hit={stats['cache_hit']} ai_ok={stats['ai_ok']} ai_fail={stats['ai_fail']}"
		),
	)

	print("✅ 完成")
	print(f"术语表输出：{output_json} | 条目数：{len(final_glossary)}（新增 {added_count}）")
	print(f"候选明细：{output_candidates_json}")
	print(f"冲突明细：{output_conflicts_json} | 冲突数：{len(conflicts)}（冲突已保留旧值）")
	print(
		"统计："
		f"批次={stats['batch_count']} "
		f"缓存命中={stats['cache_hit']} "
		f"AI成功={stats['ai_ok']} "
		f"AI失败={stats['ai_fail']} "
		f"仅正则回退={stats['regex_only']}"
	)


if __name__ == "__main__":
	main()
