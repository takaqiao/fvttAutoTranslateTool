import json
import re
from pathlib import Path

# ====== 配置区（按需修改）======
BASE_DIR = Path(r"C:\Users\Taka\Desktop\fvtt")
INPUT_FILE = "kctg-2e.kctg-2e.json"
OUTPUT_FILE = "kctg-2e.kctg-2e.json"  # 同名则原地覆盖
# ============================

def _clean_name(value: str) -> str:
	value = value.replace("\r\n", "\n")
	value = re.sub(r"\s*\n\s*", " ", value)
	value = re.sub(r"\s{2,}", " ", value).strip()
	return value

def _walk(node):
	if isinstance(node, dict):
		for k, v in node.items():
			if k == "name" and isinstance(v, str) and "\n" in v:
				node[k] = _clean_name(v)
			else:
				_walk(v)
	elif isinstance(node, list):
		for item in node:
			_walk(item)

def main() -> None:
	input_path = BASE_DIR / INPUT_FILE
	output_path = BASE_DIR / OUTPUT_FILE

	data = json.loads(input_path.read_text(encoding="utf-8"))
	_walk(data)
	output_path.write_text(
		json.dumps(data, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

if __name__ == "__main__":
	main()
