import json
import os
import time
import re
import hashlib
import concurrent.futures
import difflib
from html import escape
import pandas as pd
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from threading import Lock
from tqdm import tqdm

# 需要安装: pip install google-genai openai pandas openpyxl
from google import genai
from google.genai import types
from openai import OpenAI
try:
    from opencc import OpenCC
    _opencc_t2s = OpenCC("t2s")
except Exception:
    _opencc_t2s = None

# ================= 配置区域 =================

# API 配置 (优先从环境变量读取，否则使用默认值)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") 

# 1. [核心开关] 同步模式选择
# "TARGET_MASTER": 以目标文件(CN)结构为准。不新增Key，只翻译现有的。结构绝对安全。(解决读不出来的问题)
# "SOURCE_MASTER": 以源文件(EN)结构为准。会自动补全缺失的Key。可能导致结构改变。
SYNC_MODE = "TARGET_MASTER" 

# 2. 核心文件配置 (自动转换为Path对象)
SOURCE_EN_JSON_PATH = Path("pf2e-beginner-box-en.json")  # 仅用于参考原文
TARGET_JSON_PATH = Path("pf2e-beginner-box.adventures.json")  # 既是结构模板，也是输出目标

# 3. 模型优先级 (自动降级)
MODEL_PRIORITY_LIST = [
    ("openai", "gpt-5.2"),          # 优先级1
    ("openai", "gpt-5-mini"),       # 优先级2
    ("google", "gemini-3-flash-preview"), # 优先级3
]

# 4. 性能配置
MAX_WORKERS = 16    
TARGET_RPM = 450   
MAX_RETRIES = 5     
MAX_AUDIT_ROUNDS = 1  # 校对最多轮数
# 校对次数上限（同一条目达到次数后不再校对）
MAX_AUDIT_PASSES = 2

# 4.5 稳定模式（优先保证格式不炸）
# - HTML 文本走安全分段翻译
# - 若检测到占位符泄漏或HTML结构异常，回退为原文/草稿
SAFE_MODE = True

# 4.6 测试模式（不调用AI，仅验证格式一致性）
TEST_MODE = False
TEST_MODE_WRITE = False  # True 时输出到 TEST_OUTPUT_PATH
TEST_OUTPUT_PATH = Path("pf2e-beginner-box.adventures.test.json")
# 是否模拟完整处理流程（不调用AI，但会走清洗/规范化等流程）
# 注意：该模式不会落盘，仅统计可能改动的条目数量
TEST_MODE_SIMULATE_PIPELINE = False
SIMULATE_AI = False

# 5. [核心开关] 暴力防漏模式 (仅在翻译内容判断时生效)
BRUTE_FORCE_MODE = False

# 5.1 后处理：强制简体中文（需要 opencc 可用）
FORCE_SIMPLIFIED_POST = True

# 5.2 差异报告输出
DIFF_REPORT_ENABLED = True
DIFF_REPORT_PATH = Path("translation_diff_report.html")

# 8. 输出风格配置
# - FULL_BILINGUAL_MODE: 全字段双语（自动补全英文）
# - BILINGUAL_KEYS: 输出“中文 英文”
# - CN_ONLY_KEYS: 仅输出中文（当 FULL_BILINGUAL_MODE=True 时也会补英文）
# - LONG_TEXT_KEYS: 长文本（当 FULL_BILINGUAL_MODE=True 时追加原文块）
FULL_BILINGUAL_MODE = True
BILINGUAL_KEYS = {"name", "label", "navName", "header", "tooltip"}
CN_ONLY_KEYS = {"tokenName", "caption"}
LONG_TEXT_KEYS = {"description", "text", "content", "gm_notes", "gm_description", "publicnotes", "publicNotes"}

# 9. 容器翻译策略
# - TRANSLATE_MACROS: 宏名称是否翻译
TRANSLATE_MACROS = True
SKIP_CONTAINERS = set()

# 6. 术语表与日志 (自动转换为Path对象)
GLOBAL_GLOSSARY_PATH = Path("术语译名对照表.csv") 
LOCAL_GLOSSARY_EXPORT_PATH = Path("术语表_本地提取.csv")
REPORT_XLSX_PATH = Path("翻译审查报告.xlsx")
PROCESS_LOG_PATH = Path("运行日志.txt")
MISSED_LOG_PATH = Path("失败漏翻记录.txt")
HISTORY_FILE_PATH = Path("translation_history.json")
AUDIT_HISTORY_PATH = Path("audit_history.json")
BACKUP_DIR = Path("backups")

# 7. 日志输出
PRINT_LOG_TO_TERMINAL = True  # 同步输出到终端，便于实时观察
USE_TQDM_WRITE = True         # 使用 tqdm.write 避免打断进度条

# 目标字段白名单
TARGET_KEYS = {
    "name", "description", "text", "label", "caption", "value", 
    "unidentifiedName", "tokenName", "publicnotes", "publicNotes",
    "gm_notes", "gm_description", "header", "content", "items", 
    "navName", "tooltip", "preAuthored"
}
SPECIAL_CONTAINERS = {
    "notes", "folders", "journal", "journals", "scenes", 
    "actors", "items", "pages", "entries", "flags", "system"
}

# ===========================================

# 初始化客户端（延迟到配置加载后）
google_client = None
openai_client = None

log_lock = Lock()
report_data = {"New": [], "Fixed": [], "Kept": [], "TermAdjusted": [], "FormatAnomaly": []}
process_log_buffer = []
missed_log_buffer = [] 
history_cache = set()
new_history_entries = set()
audit_history = {}
_opencc_warned = False

CONFIG_PATH = Path("translator_config.json")

DEFAULT_CONFIG = {
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "OPENAI_BASE_URL": OPENAI_BASE_URL,
    "SYNC_MODE": SYNC_MODE,
    "SOURCE_EN_JSON_PATH": str(SOURCE_EN_JSON_PATH),
    "TARGET_JSON_PATH": str(TARGET_JSON_PATH),
    "MODEL_PRIORITY_LIST": MODEL_PRIORITY_LIST,
    "MAX_WORKERS": MAX_WORKERS,
    "TARGET_RPM": TARGET_RPM,
    "MAX_RETRIES": MAX_RETRIES,
    "MAX_AUDIT_ROUNDS": MAX_AUDIT_ROUNDS,
    "MAX_AUDIT_PASSES": MAX_AUDIT_PASSES,
    "SAFE_MODE": SAFE_MODE,
    "TEST_MODE": TEST_MODE,
    "TEST_MODE_WRITE": TEST_MODE_WRITE,
    "TEST_OUTPUT_PATH": str(TEST_OUTPUT_PATH),
    "TEST_MODE_SIMULATE_PIPELINE": TEST_MODE_SIMULATE_PIPELINE,
    "SIMULATE_AI": SIMULATE_AI,
    "BRUTE_FORCE_MODE": BRUTE_FORCE_MODE,
    "FULL_BILINGUAL_MODE": FULL_BILINGUAL_MODE,
    "BILINGUAL_KEYS": list(BILINGUAL_KEYS),
    "CN_ONLY_KEYS": list(CN_ONLY_KEYS),
    "LONG_TEXT_KEYS": list(LONG_TEXT_KEYS),
    "TRANSLATE_MACROS": TRANSLATE_MACROS,
    "SKIP_CONTAINERS": list(SKIP_CONTAINERS),
    "GLOBAL_GLOSSARY_PATH": str(GLOBAL_GLOSSARY_PATH),
    "LOCAL_GLOSSARY_EXPORT_PATH": str(LOCAL_GLOSSARY_EXPORT_PATH),
    "REPORT_XLSX_PATH": str(REPORT_XLSX_PATH),
    "PROCESS_LOG_PATH": str(PROCESS_LOG_PATH),
    "MISSED_LOG_PATH": str(MISSED_LOG_PATH),
    "HISTORY_FILE_PATH": str(HISTORY_FILE_PATH),
    "AUDIT_HISTORY_PATH": str(AUDIT_HISTORY_PATH),
    "BACKUP_DIR": str(BACKUP_DIR),
    "PRINT_LOG_TO_TERMINAL": PRINT_LOG_TO_TERMINAL,
    "USE_TQDM_WRITE": USE_TQDM_WRITE,
    "TARGET_KEYS": list(TARGET_KEYS),
    "SPECIAL_CONTAINERS": list(SPECIAL_CONTAINERS),
    "FORCE_SIMPLIFIED_POST": FORCE_SIMPLIFIED_POST,
    "DIFF_REPORT_ENABLED": DIFF_REPORT_ENABLED,
    "DIFF_REPORT_PATH": str(DIFF_REPORT_PATH),
}

def load_config():
    cfg = deepcopy(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            if isinstance(user_cfg, dict):
                cfg.update(user_cfg)
        except Exception as e:
            print(f"⚠️ 配置文件读取失败: {e}")
    return cfg

def _coerce_set(val):
    if isinstance(val, set):
        return val
    if isinstance(val, list):
        return set([str(x).strip() for x in val if str(x).strip()])
    if isinstance(val, str):
        parts = re.split(r"[\n,]", val)
        return set([p.strip() for p in parts if p.strip()])
    return set()

def apply_config(cfg):
    global GOOGLE_API_KEY, OPENAI_API_KEY, OPENAI_BASE_URL
    global SYNC_MODE, SOURCE_EN_JSON_PATH, TARGET_JSON_PATH
    global MODEL_PRIORITY_LIST, MAX_WORKERS, TARGET_RPM, MAX_RETRIES
    global MAX_AUDIT_ROUNDS, MAX_AUDIT_PASSES, SAFE_MODE
    global TEST_MODE, TEST_MODE_WRITE, TEST_OUTPUT_PATH
    global TEST_MODE_SIMULATE_PIPELINE, SIMULATE_AI, BRUTE_FORCE_MODE
    global FULL_BILINGUAL_MODE, BILINGUAL_KEYS, CN_ONLY_KEYS, LONG_TEXT_KEYS
    global TRANSLATE_MACROS, SKIP_CONTAINERS
    global GLOBAL_GLOSSARY_PATH, LOCAL_GLOSSARY_EXPORT_PATH, REPORT_XLSX_PATH
    global PROCESS_LOG_PATH, MISSED_LOG_PATH, HISTORY_FILE_PATH, AUDIT_HISTORY_PATH, BACKUP_DIR
    global PRINT_LOG_TO_TERMINAL, USE_TQDM_WRITE, TARGET_KEYS, SPECIAL_CONTAINERS
    global FORCE_SIMPLIFIED_POST, DIFF_REPORT_ENABLED, DIFF_REPORT_PATH

    GOOGLE_API_KEY = cfg.get("GOOGLE_API_KEY", GOOGLE_API_KEY)
    OPENAI_API_KEY = cfg.get("OPENAI_API_KEY", OPENAI_API_KEY)
    OPENAI_BASE_URL = cfg.get("OPENAI_BASE_URL", OPENAI_BASE_URL)
    SYNC_MODE = cfg.get("SYNC_MODE", SYNC_MODE)
    SOURCE_EN_JSON_PATH = Path(cfg.get("SOURCE_EN_JSON_PATH", str(SOURCE_EN_JSON_PATH)))
    TARGET_JSON_PATH = Path(cfg.get("TARGET_JSON_PATH", str(TARGET_JSON_PATH)))
    model_list = cfg.get("MODEL_PRIORITY_LIST", MODEL_PRIORITY_LIST)
    if isinstance(model_list, list):
        MODEL_PRIORITY_LIST = [tuple(x) for x in model_list]
    MAX_WORKERS = int(cfg.get("MAX_WORKERS", MAX_WORKERS))
    TARGET_RPM = int(cfg.get("TARGET_RPM", TARGET_RPM))
    MAX_RETRIES = int(cfg.get("MAX_RETRIES", MAX_RETRIES))
    MAX_AUDIT_ROUNDS = int(cfg.get("MAX_AUDIT_ROUNDS", MAX_AUDIT_ROUNDS))
    MAX_AUDIT_PASSES = int(cfg.get("MAX_AUDIT_PASSES", MAX_AUDIT_PASSES))
    SAFE_MODE = bool(cfg.get("SAFE_MODE", SAFE_MODE))
    TEST_MODE = bool(cfg.get("TEST_MODE", TEST_MODE))
    TEST_MODE_WRITE = bool(cfg.get("TEST_MODE_WRITE", TEST_MODE_WRITE))
    TEST_OUTPUT_PATH = Path(cfg.get("TEST_OUTPUT_PATH", str(TEST_OUTPUT_PATH)))
    TEST_MODE_SIMULATE_PIPELINE = bool(cfg.get("TEST_MODE_SIMULATE_PIPELINE", TEST_MODE_SIMULATE_PIPELINE))
    SIMULATE_AI = bool(cfg.get("SIMULATE_AI", SIMULATE_AI))
    BRUTE_FORCE_MODE = bool(cfg.get("BRUTE_FORCE_MODE", BRUTE_FORCE_MODE))
    FULL_BILINGUAL_MODE = bool(cfg.get("FULL_BILINGUAL_MODE", FULL_BILINGUAL_MODE))
    BILINGUAL_KEYS = _coerce_set(cfg.get("BILINGUAL_KEYS", BILINGUAL_KEYS))
    CN_ONLY_KEYS = _coerce_set(cfg.get("CN_ONLY_KEYS", CN_ONLY_KEYS))
    LONG_TEXT_KEYS = _coerce_set(cfg.get("LONG_TEXT_KEYS", LONG_TEXT_KEYS))
    TRANSLATE_MACROS = bool(cfg.get("TRANSLATE_MACROS", TRANSLATE_MACROS))
    SKIP_CONTAINERS = _coerce_set(cfg.get("SKIP_CONTAINERS", SKIP_CONTAINERS))
    GLOBAL_GLOSSARY_PATH = Path(cfg.get("GLOBAL_GLOSSARY_PATH", str(GLOBAL_GLOSSARY_PATH)))
    LOCAL_GLOSSARY_EXPORT_PATH = Path(cfg.get("LOCAL_GLOSSARY_EXPORT_PATH", str(LOCAL_GLOSSARY_EXPORT_PATH)))
    REPORT_XLSX_PATH = Path(cfg.get("REPORT_XLSX_PATH", str(REPORT_XLSX_PATH)))
    PROCESS_LOG_PATH = Path(cfg.get("PROCESS_LOG_PATH", str(PROCESS_LOG_PATH)))
    MISSED_LOG_PATH = Path(cfg.get("MISSED_LOG_PATH", str(MISSED_LOG_PATH)))
    HISTORY_FILE_PATH = Path(cfg.get("HISTORY_FILE_PATH", str(HISTORY_FILE_PATH)))
    AUDIT_HISTORY_PATH = Path(cfg.get("AUDIT_HISTORY_PATH", str(AUDIT_HISTORY_PATH)))
    BACKUP_DIR = Path(cfg.get("BACKUP_DIR", str(BACKUP_DIR)))
    PRINT_LOG_TO_TERMINAL = bool(cfg.get("PRINT_LOG_TO_TERMINAL", PRINT_LOG_TO_TERMINAL))
    USE_TQDM_WRITE = bool(cfg.get("USE_TQDM_WRITE", USE_TQDM_WRITE))
    TARGET_KEYS = _coerce_set(cfg.get("TARGET_KEYS", TARGET_KEYS))
    SPECIAL_CONTAINERS = _coerce_set(cfg.get("SPECIAL_CONTAINERS", SPECIAL_CONTAINERS))
    FORCE_SIMPLIFIED_POST = bool(cfg.get("FORCE_SIMPLIFIED_POST", FORCE_SIMPLIFIED_POST))
    DIFF_REPORT_ENABLED = bool(cfg.get("DIFF_REPORT_ENABLED", DIFF_REPORT_ENABLED))
    DIFF_REPORT_PATH = Path(cfg.get("DIFF_REPORT_PATH", str(DIFF_REPORT_PATH)))

def init_clients():
    global google_client, openai_client
    google_client = None
    openai_client = None
    if any(p == "google" for p, m in MODEL_PRIORITY_LIST) and GOOGLE_API_KEY:
        google_client = genai.Client(api_key=GOOGLE_API_KEY)
    if any(p == "openai" for p, m in MODEL_PRIORITY_LIST) and OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

GOOGLE_SAFETY = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
]

def write_process_log(msg):
    """线程安全的日志写入"""
    with log_lock:
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        line = f"[{timestamp}] {msg}"
        process_log_buffer.append(line)
        if PRINT_LOG_TO_TERMINAL:
            if USE_TQDM_WRITE:
                tqdm.write(line)
            else:
                print(line)
        # 每100条日志自动刷盘一次
        if len(process_log_buffer) >= 100:
            _flush_process_log()

def _flush_process_log():
    """刷新进程日志到文件"""
    if process_log_buffer and PROCESS_LOG_PATH:
        try:
            with PROCESS_LOG_PATH.open('a', encoding='utf-8') as f:
                f.write("\n".join(process_log_buffer) + "\n")
            process_log_buffer.clear()
        except Exception as e:
            print(f"⚠️ 日志写入失败: {e}")

def write_missed_log(path, text, reason):
    """记录漏翻条目"""
    with log_lock:
        missed_log_buffer.append(f"【{reason}】Path: {path}\nText: {text[:50]}...\n{'-'*30}")

# === AI 调用接口 ===
def call_ai_with_fallback(sys_prompt, user_prompt, path_str):
    """带回退机制的AI调用
    
    优先级顺序：
    1. OpenAI GPT-5.2
    2. OpenAI GPT-5-mini  
    3. Google Gemini-3-flash-preview
    """
    if SIMULATE_AI:
        m = re.search(r"```\n(.*?)\n```", user_prompt, flags=re.S)
        if m:
            return m.group(1).strip()
        return user_prompt
    last_error = None
    for provider, model_id in MODEL_PRIORITY_LIST:
        if provider == "google" and not google_client:
            continue
        if provider == "openai" and not openai_client:
            continue
        
        for attempt in range(MAX_RETRIES):
            try:
                write_process_log(f"🧠 调用模型: {model_id} | 第{attempt+1}次 | {path_str}")
                if provider == "google":
                    response = google_client.models.generate_content(
                        model=model_id,
                        contents=f"{sys_prompt}\n{user_prompt}",
                        config=types.GenerateContentConfig(temperature=0.1, safety_settings=GOOGLE_SAFETY)
                    )
                    if not response.text:
                        raise ValueError("Empty Google Response")
                    write_process_log(f"✅ 模型完成: {model_id} | {path_str}")
                    return response.text
                elif provider == "openai":
                    if "gpt-5" in model_id or "o1" in model_id or "o3" in model_id:
                        # 使用 Responses API
                        response = openai_client.responses.create(
                            model=model_id,
                            instructions=sys_prompt,
                            input=user_prompt,
                            reasoning={"effort": "none"}
                        )
                        write_process_log(f"✅ 模型完成: {model_id} | {path_str}")
                        return response.output_text
                    else:
                        # 使用 Chat Completions
                        response = openai_client.chat.completions.create(
                            model=model_id,
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.1
                        )
                        write_process_log(f"✅ 模型完成: {model_id} | {path_str}")
                        return response.choices[0].message.content
            except Exception as e:
                last_error = e
                write_process_log(f"⚠️ 模型失败: {model_id} | {path_str} | {e}")
                # 遇到速率限制立即重试下一个模型
                if "429" in str(e) or "Resource Unavailable" in str(e):
                    break
                # 其他错误等待后重试
                time.sleep(1 * (attempt + 1))
        
        write_process_log(f"⚠️ 模型 {model_id} 失败: {last_error} -> 尝试下一顺位")
    
    # 所有模型都失败
    raise last_error

# === 基础工具 ===
def backup_existing_files():
    """备份现有文件到备份目录"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    files = [TARGET_JSON_PATH, REPORT_XLSX_PATH, LOCAL_GLOSSARY_EXPORT_PATH, HISTORY_FILE_PATH]
    for file_path in files:
        if file_path.exists():
            try:
                backup_path = BACKUP_DIR / f"{timestamp}_{file_path.name}.bak"
                backup_path.write_bytes(file_path.read_bytes())
                write_process_log(f"✅ 备份完成: {file_path.name} -> {backup_path}")
            except Exception as e:
                write_process_log(f"备份失败: {file_path.name} - {e}")

def get_content_hash(en, cn):
    return hashlib.md5(f"{en or ''}::{cn or ''}".encode('utf-8')).hexdigest()

def load_history():
    """从历史文件加载已处理项目的哈希值"""
    if not HISTORY_FILE_PATH.exists():
        return set()
    try:
        with HISTORY_FILE_PATH.open('r', encoding='utf-8') as f:
            return set(json.load(f))
    except Exception as e:
        write_process_log(f"⚠️ 加载历史文件失败: {e}")
        return set()

def save_history():
    """将缓存和新条目保存到历史文件"""
    try:
        with HISTORY_FILE_PATH.open('w', encoding='utf-8') as f:
            json.dump(list(history_cache.union(new_history_entries)), f)
    except Exception as e:
        write_process_log(f"⚠️ 保存历史文件失败: {e}")

def load_audit_history():
    """加载校对次数历史"""
    if not AUDIT_HISTORY_PATH.exists():
        return {}
    try:
        with AUDIT_HISTORY_PATH.open('r', encoding='utf-8') as f:
            data = json.load(f)
            # 兼容旧格式：int
            if isinstance(data, dict):
                for k, v in list(data.items()):
                    if isinstance(v, int):
                        data[k] = {"passes": v, "kept": False}
            return data
    except Exception as e:
        write_process_log(f"⚠️ 加载校对历史失败: {e}")
        return {}

def save_audit_history():
    """保存校对次数历史"""
    try:
        with AUDIT_HISTORY_PATH.open('w', encoding='utf-8') as f:
            json.dump(audit_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        write_process_log(f"⚠️ 保存校对历史失败: {e}")

def can_audit(task_hash):
    if not task_hash:
        return True
    entry = audit_history.get(task_hash, {"passes": 0, "kept": False})
    if isinstance(entry, int):
        entry = {"passes": entry, "kept": False}
    if entry.get("kept"):
        return False
    return entry.get("passes", 0) < MAX_AUDIT_PASSES

def record_audit(task_hash, status=None):
    if not task_hash:
        return
    entry = audit_history.get(task_hash, {"passes": 0, "kept": False})
    if isinstance(entry, int):
        entry = {"passes": entry, "kept": False}
    entry["passes"] = entry.get("passes", 0) + 1
    if status == "Kept":
        entry["kept"] = True
    audit_history[task_hash] = entry

class RateLimiter:
    def __init__(self, rpm):
        self.interval = 60.0 / rpm
        self.last_dispatch_time = 0
    def wait_for_slot(self):
        now = time.time()
        wait = self.last_dispatch_time + self.interval - now
        if wait > 0: time.sleep(wait)
        self.last_dispatch_time = time.time()

class CodeProtector:
    def __init__(self, mask_html=True):
        self.patterns = [
            re.compile(r'(@[a-zA-Z0-9]+\[[^\]]*\])'),
            re.compile(r'(\[\[.*?\]\])'),
            re.compile(r'(&[a-zA-Z0-9#]+;)'),
        ]
        if mask_html:
            self.patterns.insert(2, re.compile(r'(<[^>]+>)'))
    def mask(self, text):
        if not text: return text, {}
        ph, ctr = {}, 0
        def repl(m):
            nonlocal ctr
            k = f"__CODE_{ctr}__"
            ph[k] = m.group(1)
            ctr += 1
            return k
        for p in self.patterns: text = p.sub(repl, text)
        return text, ph
    def unmask(self, text, ph):
        if not text: return text
        for k, v in ph.items():
            text = text.replace(k, v)
            if k not in text:
                text = re.sub(k.replace('_', r'\s*_\s*'), v, text)
            # 容错：修复被模型破坏的占位符（如 ___1__ / __CODE1__ 等）
            if k not in text:
                m = re.search(r'__CODE_(\d+)__', k)
                if m:
                    idx = re.escape(m.group(1))
                    text = re.sub(r'_{2,}CODE_?\s*' + idx + r'_{1,}', v, text, flags=re.IGNORECASE)
                    text = re.sub(r'_{2,}\s*' + idx + r'_{1,}', v, text)
                    text = re.sub(r'__\s*CODE_?\s*' + idx + r'\b', v, text, flags=re.IGNORECASE)
                    text = re.sub(r'\bCODE_?\s*' + idx + r'__\b', v, text, flags=re.IGNORECASE)
                    text = re.sub(r'\bCODE_?\s*' + idx + r'\b', v, text, flags=re.IGNORECASE)
        return text

class GlossaryManager:
    """术语表管理器
    
    负责加载术语表、匹配和注入术语到文本
    支持多种编码的CSV文件
    """
    
    def __init__(self, global_csv, local_csv=None):
        """初始化术语表
        
        Args:
            global_csv: 全局术语表路径
            local_csv: 本地术语表路径（可选）
        """
        self.term_map = {}  # 术语映射 {英文: {target, org, re}}
        self.sorted_keys = []
        self.load_glossary(global_csv)
        if local_csv:
            local_csv = Path(local_csv)
            if local_csv.exists():
                self.load_glossary(local_csv)
        # 按长度降序排序，防止短词先匹配
        self.sorted_keys = sorted(self.term_map.keys(), key=lambda x: len(x), reverse=True)
        print(f"术语库加载: {len(self.sorted_keys)} 条")

    def load_glossary(self, path):
        """加载CSV术语表，支持多种编码"""
        path = Path(path)
        if not path.exists():
            return
        df = None
        for enc in ['utf-8', 'utf-8-sig', 'gbk']:
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except Exception:
                continue
        if df is None:
            return
        for r in df.to_dict('records'):
            cn, en = str(r.get('Target', '')).strip(), str(r.get('Source', '')).strip()
            if cn and en and en.lower() != 'nan':
                flags = 0 if any(c.isupper() for c in en) else re.IGNORECASE
                self.term_map[en] = {"target": cn, "org": en, "re": re.compile(r'\b' + re.escape(en) + r'\b', flags)}

    def pre_inject_text(self, text, path_str):
        """在文本中注入术语标记，供AI识别
        
        Args:
            text: 待处理文本
            path_str: JSON路径（用于日志）
        
        Returns:
            (注入后的文本, 注入的术语列表)
        """
        if not text:
            return text, []
        
        inj, ph, idx = [], {}, 0
        # 快速过滤：只检查文本中实际包含的词汇
        tokens = set(re.findall(r'[a-z]+', text.lower()))
        cands = [k for k in self.sorted_keys if (k.lower() in tokens) or (" " in k and k.lower() in text.lower())]
        
        for k in cands:
            d = self.term_map[k]
            matches = list(d["re"].finditer(text))
            if matches:
                def repl(m):
                    nonlocal idx
                    mt = m.group(0)
                    # 保持原有的大小写样式
                    if mt == d["org"] or (d["org"].islower() and mt.istitle()):
                        inj.append((d["org"], d["target"]))
                        k_ph = f"__Tm_{idx}__"
                        ph[k_ph] = f"⟪{d['target']}|原文:{mt}⟫"
                        idx += 1
                        return k_ph
                    return mt
                text = d["re"].sub(repl, text)
        
        # 将占位符替换为标记
        for k, v in ph.items():
            text = text.replace(k, v)
        
        return text, inj

def smart_format_bilingual(cn, en):
    """兼容旧逻辑的双语格式化（仅用于术语提取/兜底）"""
    if not cn:
        return en
    if not en:
        return cn
    cn = re.sub(r'⟪(.*?)\|原文:.*?⟫', r'\1', cn)
    return cn

def extract_last_key(path_str):
    """从 JSON 路径中提取最后一个 key（忽略数组下标）"""
    if not path_str:
        return ""
    # 去掉数组下标
    cleaned = re.sub(r"\[\d+\]", "", path_str)
    return cleaned.split(".")[-1] if "." in cleaned else cleaned

def strip_codes_for_lang_detect(text):
    if not text:
        return ""
    text = re.sub(r'@UUID\[[^\]]+\]', ' ', text)
    text = re.sub(r'@Compendium\[[^\]]+\]', ' ', text)
    text = re.sub(r'\[\[.*?\]\]', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
    return text

def contains_english(text):
    return bool(re.search(r'[A-Za-z]', strip_codes_for_lang_detect(text)))

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text or ''))

def get_value_style(path_str, key):
    path_segments = path_str.split('.')
    if any(seg in SKIP_CONTAINERS for seg in path_segments):
        return "skip"
    if "macros" in path_segments and not TRANSLATE_MACROS:
        return "skip"
    if "notes" in path_segments:
        return "cn_only"
    if "folders" in path_segments:
        return "bilingual"
    if key in CN_ONLY_KEYS:
        return "cn_only"
    if key in BILINGUAL_KEYS:
        return "bilingual"
    if key in LONG_TEXT_KEYS:
        return "cn_only"
    return "cn_only"

def strip_original_block(text):
    if not text:
        return text
    text = re.sub(r'<br><br><hr><b>(原文|Original):</b><br>.*$', '', text, flags=re.DOTALL)
    return text

def normalize_bilingual_short(cn_text, en_text):
    if not cn_text:
        return en_text or cn_text
    if not en_text:
        return cn_text
    if contains_html_tags(cn_text) or (isinstance(en_text, str) and contains_html_tags(en_text)):
        clean_cn = cn_text
        return f"{clean_cn} {en_text}" if clean_cn else en_text
    clean_cn = strip_english_tokens(cn_text)
    clean_cn = collapse_duplicate_cn_prefix(clean_cn)
    clean_cn = collapse_duplicate_numeric_suffix(clean_cn)
    return f"{clean_cn} {en_text}" if clean_cn else en_text

def strip_trailing_english(text, min_len=30, min_words=4):
    if not text:
        return text
    prot = CodeProtector()
    masked, ph = prot.mask(text)
    m = re.search(r"(\s+[A-Za-z][A-Za-z0-9'’\-\s]*)$", masked)
    if m:
        tail = m.group(1)
        words = re.findall(r"[A-Za-z][A-Za-z0-9'’\-]*", tail)
        if len(words) >= min_words or len(tail) >= min_len:
            masked = masked[:m.start()].strip()
    return prot.unmask(masked, ph)

def contains_html_tags(text):
    if not isinstance(text, str):
        return False
    return bool(re.search(r'<[^>]+>', text or ''))

def extract_english_portion(text):
    if not isinstance(text, str) or not text:
        return ""
    # 仅移除中文字符，保留英文、数字、符号、占位符
    cleaned = re.sub(r'[\u4e00-\u9fff]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def is_cn_only_token(token):
    return bool(re.search(r'[\u4e00-\u9fff]', token)) and not re.search(r'[A-Za-z]', token)

def cleanup_noise_parens(text):
    if not isinstance(text, str) or not text:
        return text
    text = re.sub(r'\b\d+\s*\(\)\b', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

def cleanup_short_cn_duplicates(text, max_len=120):
    if not isinstance(text, str) or not text:
        return text
    if contains_html_tags(text):
        return text
    if len(text) > max_len:
        return text
    tokens = text.split()
    if len(tokens) < 2:
        return text
    out = []
    i = 0
    while i < len(tokens):
        cur = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt and is_cn_only_token(cur) and is_cn_only_token(nxt):
            if cur == nxt:
                out.append(cur)
                i += 2
                continue
            if cur in nxt:
                out.append(nxt)
                i += 2
                continue
            if nxt in cur:
                out.append(cur)
                i += 2
                continue
            if len(cur) <= 6 and len(nxt) <= 6 and set(cur) == set(nxt):
                out.append(cur)
                i += 2
                continue
        out.append(cur)
        i += 1
    return " ".join(out)

def has_placeholder_leak(text):
    if not isinstance(text, str):
        return False
    return bool(re.search(r'__CODE_\d+__|__CODE_\d+\b|\bCODE_\d+__|\bCODE_\d+\b|_{2,}\d+_{1,}', text))

def is_garbled_html(text):
    if not text:
        return False
    if re.search(r'<\s*>', text) or re.search(r'</\s*>', text):
        return True
    if re.search(r'<\s*=', text) or re.search(r'</\s*=', text):
        return True
    if text.count('<') != text.count('>'):
        return True
    return False

def extract_html_tags(text):
    if not text:
        return []
    return re.findall(r'<[^>]+>', text)

def _normalize_html_tag_for_compare(tag):
    if not tag:
        return tag
    m = re.match(r'<\s*(/?)\s*([^\s>/]+)([^>]*)>', tag)
    if not m:
        return tag.strip()
    slash, name, attr_blob = m.groups()
    name = name.lower()
    if slash:
        return f"</{name}>"
    attrs = []
    for am in re.finditer(r'([a-zA-Z_:][\w:.-]*)(?:\s*=\s*("[^"]*"|\'[^\']*\'|[^\s"\'>]+))?', attr_blob):
        a_name = am.group(1).lower()
        if a_name == "title":
            continue
        a_val = am.group(2) or ""
        if a_val.startswith(('"', "'")) and a_val.endswith(('"', "'")):
            a_val = a_val[1:-1]
        attrs.append((a_name, a_val))
    attrs.sort(key=lambda x: x[0])
    if attrs:
        attr_str = " ".join([f"{k}={v}" if v != "" else k for k, v in attrs])
        return f"<{name} {attr_str}>"
    return f"<{name}>"

def extract_html_tags_for_compare(text):
    return [_normalize_html_tag_for_compare(t) for t in extract_html_tags(text)]

def repair_html_tags_by_source(src_text, out_text):
    if not isinstance(src_text, str) or not isinstance(out_text, str):
        return out_text
    if not contains_html_tags(src_text) or not contains_html_tags(out_text):
        return out_text
    src_tags = extract_html_tags(src_text)
    out_tags = extract_html_tags(out_text)
    if len(src_tags) != len(out_tags):
        return out_text
    src_parts = re.split(r'(<[^>]+>)', src_text)
    out_parts = re.split(r'(<[^>]+>)', out_text)
    if len(src_parts) != len(out_parts):
        return out_text
    rebuilt = []
    for sp, op in zip(src_parts, out_parts):
        if sp.startswith('<') and sp.endswith('>'):
            rebuilt.append(sp)
        else:
            rebuilt.append(op)
    return "".join(rebuilt)

def repair_placeholders_by_source(src_text, out_text):
    if not isinstance(src_text, str) or not isinstance(out_text, str):
        return out_text
    if not contains_html_tags(src_text):
        return out_text
    if not has_placeholder_leak(out_text):
        return out_text
    en_tags = extract_html_tags(src_text)
    cn_tags = extract_html_tags(out_text)
    missing = []
    i = j = 0
    while i < len(en_tags) and j < len(cn_tags):
        if en_tags[i] == cn_tags[j]:
            i += 1
            j += 1
        else:
            missing.append(en_tags[i])
            i += 1
    while i < len(en_tags):
        missing.append(en_tags[i])
        i += 1

    def repl(m):
        return missing.pop(0) if missing else ""

    text = re.sub(r'__CODE_\d+.*?CODE_\d+__', repl, out_text, flags=re.DOTALL)
    text = re.sub(r'__CODE_\d+__', repl, text)
    text = re.sub(r'\bCODE_\d+__', repl, text)
    text = re.sub(r'__CODE_\d+\b', repl, text)
    text = re.sub(r'\bCODE_\d+\b', repl, text)
    return text

def rebuild_html_by_source_structure(src_text, out_text):
    if not isinstance(src_text, str) or not isinstance(out_text, str):
        return out_text
    if not contains_html_tags(src_text):
        return out_text
    src_parts = re.split(r'(<[^>]+>)', src_text)
    out_parts = re.split(r'(<[^>]+>)', out_text)
    src_text_parts = [p for p in src_parts if not (p.startswith('<') and p.endswith('>'))]
    out_text_parts = [p for p in out_parts if not (p.startswith('<') and p.endswith('>'))]
    if len(src_text_parts) != len(out_text_parts):
        if len(out_text_parts) < len(src_text_parts):
            out_text_parts = out_text_parts + [""] * (len(src_text_parts) - len(out_text_parts))
        else:
            out_text_parts = out_text_parts[:len(src_text_parts) - 1] + ["".join(out_text_parts[len(src_text_parts) - 1:])]
    rebuilt = []
    text_iter = iter(out_text_parts)
    for part in src_parts:
        if part.startswith('<') and part.endswith('>'):
            rebuilt.append(part)
        else:
            rebuilt.append(next(text_iter, ""))
    return "".join(rebuilt)

def is_html_structure_changed(src_text, out_text):
    if not src_text:
        return False
    src_tags = extract_html_tags_for_compare(src_text)
    out_tags = extract_html_tags_for_compare(out_text)
    return src_tags != out_tags

def log_format_anomaly(path, original, translated, reason):
    row = {
        "JSON Path": path,
        "Reason": reason,
        "Original": original,
        "Translation": translated
    }
    with log_lock:
        report_data["FormatAnomaly"].append(row)

def scan_format_anomalies(cn_node, en_node=None, path_str="root", seen=None):
    if seen is None:
        seen = set()
    if isinstance(cn_node, dict):
        for k, v in cn_node.items():
            cur_path = f"{path_str}.{k}"
            en_v = en_node.get(k) if isinstance(en_node, dict) else None
            if isinstance(v, str):
                if isinstance(en_v, str) and contains_html_tags(en_v):
                    if (not contains_html_tags(v)) or is_garbled_html(v) or is_html_structure_changed(en_v, v):
                        if cur_path not in seen:
                            seen.add(cur_path)
                            log_format_anomaly(cur_path, en_v, v, "HTML structure changed")
            elif isinstance(v, (dict, list)):
                scan_format_anomalies(v, en_v, cur_path, seen)
    elif isinstance(cn_node, list):
        for i, v in enumerate(cn_node):
            cur_path = f"{path_str}[{i}]"
            en_v = en_node[i] if isinstance(en_node, list) and i < len(en_node) else None
            if isinstance(v, str):
                if isinstance(en_v, str) and contains_html_tags(en_v):
                    if (not contains_html_tags(v)) or is_garbled_html(v) or is_html_structure_changed(en_v, v):
                        if cur_path not in seen:
                            seen.add(cur_path)
                            log_format_anomaly(cur_path, en_v, v, "HTML structure changed")
            elif isinstance(v, (dict, list)):
                scan_format_anomalies(v, en_v, cur_path, seen)

def validate_format_consistency(cn_node, en_node=None):
    """对比原文与译文的格式一致性"""
    results = {
        "html_missing": 0,
        "html_changed": 0,
        "html_garbled": 0,
        "placeholder_leak": 0,
        "uuid_mismatch": 0,
        "compendium_mismatch": 0,
        "roll_mismatch": 0,
    }

    def extract_uuids(text):
        return re.findall(r'@UUID\[[^\]]+\]', text) if isinstance(text, str) else []

    def extract_compendium(text):
        return re.findall(r'@Compendium\[[^\]]+\]', text) if isinstance(text, str) else []

    def extract_rolls(text):
        return re.findall(r'\[\[/r[^\]]+\]\]', text) if isinstance(text, str) else []

    def walk(cn_n, en_n):
        if isinstance(cn_n, dict):
            for k, v in cn_n.items():
                en_v = en_n.get(k) if isinstance(en_n, dict) else None
                walk(v, en_v)
        elif isinstance(cn_n, list):
            for i, v in enumerate(cn_n):
                en_v = en_n[i] if isinstance(en_n, list) and i < len(en_n) else None
                walk(v, en_v)
        else:
            if isinstance(en_n, str) and "<" in en_n and ">" in en_n:
                en_tags = extract_html_tags_for_compare(en_n)
                cn_tags = extract_html_tags_for_compare(cn_n) if isinstance(cn_n, str) else []
                if not cn_tags:
                    results["html_missing"] += 1
                elif en_tags != cn_tags:
                    results["html_changed"] += 1
                if isinstance(cn_n, str) and is_garbled_html(cn_n):
                    results["html_garbled"] += 1
            if isinstance(cn_n, str) and has_placeholder_leak(cn_n):
                results["placeholder_leak"] += 1
            if isinstance(en_n, str) and isinstance(cn_n, str):
                if sorted(extract_uuids(en_n)) != sorted(extract_uuids(cn_n)):
                    results["uuid_mismatch"] += 1
                if sorted(extract_compendium(en_n)) != sorted(extract_compendium(cn_n)):
                    results["compendium_mismatch"] += 1
                if sorted(extract_rolls(en_n)) != sorted(extract_rolls(cn_n)):
                    results["roll_mismatch"] += 1

    walk(cn_node, en_node)
    return results

def _diff_inline(a, b):
    if a == b:
        return escape(a or "")
    sm = difflib.SequenceMatcher(None, a or "", b or "")
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(escape((a or "")[i1:i2]))
        elif tag == "delete":
            parts.append(f"<del>{escape((a or "")[i1:i2])}</del>")
        elif tag == "insert":
            parts.append(f"<ins>{escape((b or "")[j1:j2])}</ins>")
        else:
            parts.append(f"<del>{escape((a or "")[i1:i2])}</del><ins>{escape((b or "")[j1:j2])}</ins>")
    return "".join(parts)

def generate_diff_report(old_data, new_data, output_path):
    """生成变更差异报告（路径 + A→B + 高亮差异）"""
    changes = []

    def walk(old_n, new_n, path_str="root"):
        if isinstance(new_n, dict):
            for k, v in new_n.items():
                old_v = old_n.get(k) if isinstance(old_n, dict) else None
                walk(old_v, v, f"{path_str}.{k}")
        elif isinstance(new_n, list):
            for i, v in enumerate(new_n):
                old_v = old_n[i] if isinstance(old_n, list) and i < len(old_n) else None
                walk(old_v, v, f"{path_str}[{i}]")
        else:
            if isinstance(old_n, str) and isinstance(new_n, str) and old_n != new_n:
                changes.append({
                    "path": path_str,
                    "before": old_n,
                    "after": new_n,
                    "diff": _diff_inline(old_n, new_n)
                })

    walk(old_data, new_data)
    if not changes:
        return 0

    rows = []
    for c in changes:
        rows.append(
            "<tr>"
            f"<td class='path'>{escape(c['path'])}</td>"
            f"<td class='before'>{escape(c['before'])}</td>"
            f"<td class='after'>{escape(c['after'])}</td>"
            f"<td class='diff'>{c['diff']}</td>"
            "</tr>"
        )

    html = """
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>翻译差异报告</title>
<style>
body{font-family:Segoe UI,Arial,Helvetica,sans-serif;margin:20px;}
table{border-collapse:collapse;width:100%;table-layout:fixed;}
th,td{border:1px solid #ddd;padding:8px;vertical-align:top;word-break:break-word;}
th{background:#f5f5f5;}
del{background:#ffecec;color:#b00020;text-decoration:line-through;}
ins{background:#eaffea;color:#0a7a0a;text-decoration:none;}
.path{width:22%;}
.before{width:26%;}
.after{width:26%;}
.diff{width:26%;}
</style>
</head>
<body>
<h1>翻译差异报告</h1>
<p>共 {count} 条变更</p>
<table>
<thead><tr><th>路径</th><th>原值 A</th><th>新值 B</th><th>差异</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>
""".format(count=len(changes), rows="\n".join(rows))

    try:
        output_path = Path(output_path)
        output_path.write_text(html, encoding="utf-8")
        return len(changes)
    except Exception as e:
        write_process_log(f"⚠️ 差异报告写入失败: {e}")
        return 0

def translate_html_segments(en_text, glossary_mgr, path_str):
    """将HTML文本拆分为文本段落翻译，再按原标签拼接"""
    if not en_text:
        return en_text
    if SIMULATE_AI:
        return en_text
    parts = re.split(r'(<[^>]+>)', en_text)
    text_segs = [p for p in parts if p and not p.startswith('<')]
    if not text_segs:
        return en_text

    seg_payload = []
    seg_meta = []
    for i, seg in enumerate(text_segs):
        prot = CodeProtector(mask_html=False)
        masked, code_ph = prot.mask(seg)
        injected, terms = glossary_mgr.pre_inject_text(masked, path_str)
        seg_payload.append(f"⟦SEG{i}⟧{injected}⟦/SEG{i}⟧")
        seg_meta.append((code_ph, terms, prot))

    sys_prompt = (
        "You are a professional Pathfinder 2e translator. "
        "Translate ONLY the text between segment markers. "
        "Keep all segment markers EXACTLY as given. "
        "Keep all codes like @UUID[...] and [[...]] unchanged. "
        "Output ONLY Simplified Chinese (简体中文 only). Do NOT use Traditional Chinese. "
        "No notes, no comments."
    )
    prompt = "\n".join(seg_payload)

    try:
        res_text = call_ai_with_fallback(sys_prompt, prompt, path_str)
        trans = clean_response_text(res_text)
        seg_map = {}
        for m in re.finditer(r'⟦SEG(\d+)⟧(.*?)⟦/SEG\1⟧', trans, flags=re.S):
            idx = int(m.group(1))
            seg_map[idx] = m.group(2)
        if len(seg_map) != len(text_segs):
            return en_text

        out_parts = []
        seg_idx = 0
        for p in parts:
            if p.startswith('<') and p.endswith('>'):
                out_parts.append(p)
            else:
                if p == "":
                    out_parts.append(p)
                    continue
                code_ph, terms, prot = seg_meta[seg_idx]
                seg_idx += 1
                seg_text = seg_map.get(seg_idx - 1, p)
                seg_text = prot.unmask(seg_text, code_ph)
                seg_text = cleanup_injection_tags(seg_text)
                seg_text = collapse_duplicate_cn_prefix(seg_text)
                seg_text = collapse_duplicate_numeric_suffix(seg_text)
                out_parts.append(seg_text)

        return "".join(out_parts)
    except Exception as e:
        write_process_log(f"⚠️ HTML分段翻译失败: {path_str} | {e}")
        return en_text

def normalize_output_text(cn_text, en_text, path_str):
    if not cn_text:
        return cn_text
    key = extract_last_key(path_str)
    style = get_value_style(path_str, key)
    if style == "skip":
        return cn_text

    cn_text = cleanup_injection_tags(cn_text)
    cn_text = cleanup_noise_parens(cn_text)
    cn_text = cleanup_short_cn_duplicates(cn_text)
    has_html = contains_html_tags(cn_text) or (isinstance(en_text, str) and contains_html_tags(en_text))
    if has_html and isinstance(en_text, str) and contains_html_tags(en_text):
        cn_text = repair_placeholders_by_source(en_text, cn_text)
        fixed_html = cn_text
        if is_html_structure_changed(en_text, fixed_html):
            if len(extract_html_tags(en_text)) == len(extract_html_tags(fixed_html)):
                fixed_html = repair_html_tags_by_source(en_text, fixed_html)
            else:
                fixed_html = rebuild_html_by_source_structure(en_text, fixed_html)
                fixed_html = repair_html_tags_by_source(en_text, fixed_html)
        else:
            fixed_html = repair_html_tags_by_source(en_text, fixed_html)
        if fixed_html != cn_text and not is_html_structure_changed(en_text, fixed_html):
            cn_text = fixed_html
        return enforce_simplified(cn_text.strip())
    if style == "cn_only":
        cn_text = strip_original_block(cn_text)
        if FULL_BILINGUAL_MODE and en_text and not has_html:
            en_for_append = en_text
            if isinstance(en_for_append, str) and contains_chinese(en_for_append):
                en_for_append = extract_english_portion(en_for_append)
            if isinstance(en_for_append, str) and en_for_append in cn_text:
                return cn_text.strip()
            if isinstance(en_for_append, str) and en_for_append:
                return normalize_bilingual_short(cn_text, en_for_append).strip()
        if contains_chinese(cn_text) and contains_english(cn_text):
            cn_text = strip_trailing_english(cn_text)
        return enforce_simplified(cn_text.strip())

    cn_text = strip_original_block(cn_text)
    if en_text and not has_html:
        en_for_append = en_text
        if isinstance(en_for_append, str) and contains_chinese(en_for_append):
            en_for_append = extract_english_portion(en_for_append)
        if isinstance(en_for_append, str) and en_for_append in cn_text:
            return cn_text.strip()
        if isinstance(en_for_append, str) and en_for_append:
            return normalize_bilingual_short(cn_text, en_for_append).strip()
    return enforce_simplified(cn_text.strip())

def enforce_simplified(text):
    """强制简体中文后处理（需要 opencc 可用）"""
    global _opencc_warned
    if not FORCE_SIMPLIFIED_POST:
        return text
    if not isinstance(text, str) or not text:
        return text
    if _opencc_t2s is None:
        if not _opencc_warned:
            write_process_log("⚠️ opencc 未安装，跳过简繁转换后处理")
            _opencc_warned = True
        return text
    try:
        return _opencc_t2s.convert(text)
    except Exception:
        return text

def normalize_output_inplace(cn_node, en_node=None, path_str="root"):
    """全量规范化输出格式（不触发AI）"""
    fixed = 0
    if isinstance(cn_node, dict):
        for k, v in cn_node.items():
            cur_path = f"{path_str}.{k}"
            en_v = None
            if isinstance(en_node, dict):
                en_v = en_node.get(k)
            if isinstance(v, str):
                new_v = normalize_output_text(v, en_v, cur_path)
                if new_v != v:
                    cn_node[k] = new_v
                    fixed += 1
            elif isinstance(v, (dict, list)):
                fixed += normalize_output_inplace(v, en_v, cur_path)
    elif isinstance(cn_node, list):
        for i, v in enumerate(cn_node):
            cur_path = f"{path_str}[{i}]"
            en_v = None
            if isinstance(en_node, list) and i < len(en_node):
                en_v = en_node[i]
            if isinstance(v, str):
                new_v = normalize_output_text(v, en_v, cur_path)
                if new_v != v:
                    cn_node[i] = new_v
                    fixed += 1
            elif isinstance(v, (dict, list)):
                fixed += normalize_output_inplace(v, en_v, cur_path)
    return fixed

def extract_local_glossary(en_data, cn_data, output_path):
    """从翻译数据中提取本地术语表"""
    print("正在扫描本地术语...")
    write_process_log("开始提取本地术语表")
    extracted = []
    
    def traverse(en_node, cn_node):
        """递归遍历数据结构，提取已翻译项"""
        if isinstance(en_node, dict) and isinstance(cn_node, dict):
            for k, v in en_node.items():
                if k in cn_node:
                    traverse(v, cn_node[k])
        elif isinstance(en_node, list) and isinstance(cn_node, list):
            for i in range(min(len(en_node), len(cn_node))):
                traverse(en_node[i], cn_node[i])
        elif isinstance(en_node, str) and isinstance(cn_node, str):
            if len(en_node) < 60 and len(cn_node) > 0 and en_node != cn_node:
                clean_cn = smart_format_bilingual(cn_node, "")
                clean_cn = strip_english_tokens(clean_cn)
                clean_cn = collapse_duplicate_numeric_suffix(clean_cn)
                if clean_cn and re.search(r'[\u4e00-\u9fff]', clean_cn):
                    extracted.append({'Source': en_node, 'Target': clean_cn})
    
    traverse(en_data, cn_data)
    if extracted:
        try:
            df = pd.DataFrame(extracted).drop_duplicates(subset=['Source'])
            output_path = Path(output_path)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            write_process_log(f"本地术语表导出完成: {output_path} | 条目: {len(df)}")
        except Exception as e:
            write_process_log(f"⚠️ 导出术语表失败: {e}")
    else:
        write_process_log("本地术语表为空：未导出")

def clean_response_text(text):
    """清理AI响应文本"""
    if not text:
        return ""
    text = re.sub(r'^```[a-zA-Z]*\n', '', text)
    text = re.sub(r'\n```$', '', text)
    text = re.sub(r'^(Here is|Below is|以下是).*?(\n|$)', '', text, flags=re.IGNORECASE).strip()
    # 移除模型输出的注释行（例如 // ...）
    text = re.sub(r'^\s*//.*$', '', text, flags=re.MULTILINE)
    return text.strip()

def cleanup_injection_tags(text):
    """清理术语注入标签"""
    if not text:
        return text
    # 移除注入标签，保留翻译
    text = re.sub(r'⟪(.*?)\|原文:.*?⟫', r'\1', text)
    # 清理残留的“|原文:”碎片（含缺失左括号的情况）
    text = re.sub(r'\s*\d+\|原文:[^⟫]*', '', text)
    text = re.sub(r'\s*\|原文:[^⟫]*', '', text)
    # 清理残留注入符号与代码块标记
    text = text.replace('⟪', '').replace('⟫', '')
    text = text.replace('```', '')
    return text

def collapse_duplicate_cn_prefix(text):
    """清理短文本开头的重复中文词组

    例: "属性值 属性值 Ability Scores" -> "属性值 Ability Scores"
    """
    if not text:
        return text
    parts = text.split()
    if len(parts) < 2:
        return text
    if parts[0] == parts[1] and re.search(r'[\u4e00-\u9fff]', parts[0]):
        i = 1
        while i < len(parts) and parts[i] == parts[0]:
            i += 1
        return " ".join([parts[0]] + parts[i:])
    return text

def collapse_duplicate_numeric_suffix(text):
    """清理末尾重复数字（如: 01 01 / 2 2）"""
    if not text:
        return text
    parts = text.split()
    while len(parts) >= 2 and parts[-1] == parts[-2] and re.fullmatch(r"\d+", parts[-1]):
        parts.pop()
    return " ".join(parts)

def strip_english_tokens(text):
    """移除文本中的英文词，保留中文与数字

    用于本地术语提取，避免术语表携带英文尾巴。
    """
    if not text:
        return text
    cleaned = re.sub(r"[A-Za-z][A-Za-z0-9'\-]*", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else text

def normalize_for_compare(text):
    return re.sub(r'\s', '', text or '')

def detect_adjusted_terms(output_text, injected_terms):
    """检测术语是否被改动（输出中缺失术语译名）"""
    adjusted = []
    if not output_text or not injected_terms:
        return adjusted
    for en, target in injected_terms:
        if target and target not in output_text:
            adjusted.append((en, target))
    return adjusted

protector = CodeProtector()

def process_single_item(task_type, en_text, cn_draft, glossary_mgr, path_str, audit_mode=None, task_hash=None):
    """处理单个翻译项
    
    Args:
        task_type: 'NEW' 或 'AUDIT'
        en_text: 英文原文
        cn_draft: 中文初稿（审核模式）
        glossary_mgr: 术语表管理器
        path_str: JSON路径
    
    Returns:
        (翻译结果, 状态)
    """
    # 基础检查
    if not en_text or len(en_text) < 2:
        return en_text, None
    if not re.search(r'[a-zA-Z]', en_text):
        return en_text, None

    write_process_log(f"🧩 处理任务: {task_type} | {path_str}")

    # 稳定模式：HTML 走安全分段翻译，避免标签被改写
    if SAFE_MODE and contains_html_tags(en_text):
        try:
            html_trans = translate_html_segments(en_text, glossary_mgr, path_str)
            html_trans = cleanup_injection_tags(html_trans)
            html_trans = collapse_duplicate_cn_prefix(html_trans)
            html_trans = collapse_duplicate_numeric_suffix(html_trans)
            final_trans = normalize_output_text(html_trans, en_text, path_str)

            if is_garbled_html(final_trans) or not contains_html_tags(final_trans) or is_html_structure_changed(en_text, final_trans):
                log_format_anomaly(path_str, en_text, final_trans, "HTML structure changed")
                final_trans = cn_draft if cn_draft else en_text
            if has_placeholder_leak(final_trans):
                log_format_anomaly(path_str, en_text, final_trans, "Placeholder leaked")
                final_trans = cn_draft if cn_draft else en_text

            status = "Fixed" if (cn_draft and final_trans != cn_draft) else "New"
            adjusted_terms = []
            log_report(status, path_str, en_text, final_trans, [], adjusted_terms)
            write_process_log(f"✅ 任务完成: {status} | {path_str}")
            if task_type == "AUDIT":
                record_audit(task_hash, status)
            return final_trans, status
        except Exception as e:
            write_process_log(f"⚠️ HTML安全翻译失败: {path_str} | {e}")
            return (cn_draft if cn_draft else en_text), None
    
    # 代码保护和术语注入（强制屏蔽HTML标签，避免模型改动标签）
    prot = CodeProtector(mask_html=True)
    masked, code_ph = prot.mask(en_text)
    injected, terms = glossary_mgr.pre_inject_text(masked, path_str)
    
    # 构建审校初稿
    clean_draft_txt = cn_draft or ""
    
    # 构建提示词
    sys_prompt = (
        "You are a professional Pathfinder 2e translator. "
        "Output ONLY Simplified Chinese (简体中文 only). Do NOT use Traditional Chinese. "
        "Keep HTML/Foundry codes unchanged. "
        "Do NOT alter any placeholders like __CODE_0__. "
        "Do NOT output notes, tags like ⟪⟫ or |原文:, "
        "do NOT add markdown fences or comments, and do NOT repeat words. "
        "Keep numbers and units exactly as in the source."
    )
    audit_prompt = (
        "You are a professional Pathfinder 2e editor. "
        "Output ONLY Simplified Chinese (简体中文 only). Do NOT use Traditional Chinese. "
        "Keep the Draft format unchanged. If the Draft contains English (e.g., after <hr>原文), "
        "preserve the English segment as-is. Only fix Chinese wording/grammar. "
        "Glossary terms in the Draft should be kept unless they are contextually incorrect; "
        "if incorrect, replace with a better Chinese term. "
        "Do NOT add/remove HTML or Foundry codes. Output the corrected Draft only."
    )
    if task_type == "AUDIT":
        prompt = (
            f"Original:\n```\n{injected}\n```\nDraft:\n```\n{clean_draft_txt}\n```\n"
            f"Task: Review draft. If correct, output it. If wrong, correct it."
        )
    else:
        prompt = f"Translate:\n```\n{injected}\n```"

    try:
        # 译文/校对
        if task_type == "AUDIT":
            draft = clean_draft_txt
            changed = False
            final_trans = draft
            for _ in range(MAX_AUDIT_ROUNDS):
                res_text = call_ai_with_fallback(audit_prompt, (
                    f"Original:\n```\n{injected}\n```\n"
                    f"Draft:\n```\n{draft}\n```\n"
                    f"Task: Review draft. If correct, output it. If wrong, correct it."
                ), path_str)
                trans = clean_response_text(res_text)
                final_trans = prot.unmask(trans, code_ph)
                final_trans = cleanup_injection_tags(final_trans)
                final_trans = collapse_duplicate_cn_prefix(final_trans)
                final_trans = collapse_duplicate_numeric_suffix(final_trans)
                final_trans = normalize_output_text(final_trans, en_text, path_str)

                if contains_html_tags(en_text):
                    final_trans = repair_placeholders_by_source(en_text, final_trans)
                    if is_garbled_html(final_trans) or not contains_html_tags(final_trans) or is_html_structure_changed(en_text, final_trans):
                        rebuilt = rebuild_html_by_source_structure(en_text, final_trans)
                        repaired = repair_html_tags_by_source(en_text, rebuilt)
                        if repaired != final_trans and not is_html_structure_changed(en_text, repaired):
                            final_trans = repaired
                        else:
                            log_format_anomaly(path_str, en_text, final_trans, "HTML structure changed")
                if has_placeholder_leak(final_trans):
                    log_format_anomaly(path_str, en_text, final_trans, "Placeholder leaked")

                if normalize_for_compare(final_trans) == normalize_for_compare(draft):
                    break
                changed = True
                draft = final_trans

            status = "Fixed" if changed else "Kept"
            adjusted_terms = detect_adjusted_terms(final_trans, terms)
            log_report(status, path_str, en_text, final_trans, terms, adjusted_terms)
            write_process_log(f"✅ 任务完成: {status} | {path_str}")
            record_audit(task_hash, status)
            return final_trans, status

        # NEW 翻译
        res_text = call_ai_with_fallback(sys_prompt, prompt, path_str)
        trans = clean_response_text(res_text)
        final_trans = prot.unmask(trans, code_ph)
        final_trans = cleanup_injection_tags(final_trans)
        final_trans = collapse_duplicate_cn_prefix(final_trans)
        final_trans = collapse_duplicate_numeric_suffix(final_trans)
        final_trans = normalize_output_text(final_trans, en_text, path_str)

        if contains_html_tags(en_text):
            final_trans = repair_placeholders_by_source(en_text, final_trans)
            if is_garbled_html(final_trans) or not contains_html_tags(final_trans) or is_html_structure_changed(en_text, final_trans):
                rebuilt = rebuild_html_by_source_structure(en_text, final_trans)
                repaired = repair_html_tags_by_source(en_text, rebuilt)
                if repaired != final_trans and not is_html_structure_changed(en_text, repaired):
                    final_trans = repaired
                else:
                    log_format_anomaly(path_str, en_text, final_trans, "HTML structure changed")
        if has_placeholder_leak(final_trans):
            log_format_anomaly(path_str, en_text, final_trans, "Placeholder leaked")

        adjusted_terms = detect_adjusted_terms(final_trans, terms)
        log_report("New", path_str, en_text, final_trans, terms, adjusted_terms)
        write_process_log(f"✅ 任务完成: New | {path_str}")
        return final_trans, "New"

    except Exception as e:
        write_process_log(f"❌ 所有模型失败 {path_str}: {e}")
        write_missed_log(path_str, en_text, "All Models Failed")
        fallback = cn_draft if cn_draft else f"【FAIL】{en_text}"
        return fallback, None

def log_report(status, path, original, translated, injected_terms, adjusted_terms=None):
    term_str = " | ".join([f"{e}->{c}" for e, c in injected_terms])
    adjusted_str = " | ".join([f"{e}->{c}" for e, c in (adjusted_terms or [])])
    row = {
        "JSON Path": path,
        "Involved Terms": term_str,
        "Adjusted Terms": adjusted_str,
        "Original": original,
        "Translation": translated
    }
    with log_lock:
        if status in report_data:
            report_data[status].append(row)
        if adjusted_terms:
            report_data["TermAdjusted"].append(row)

# === V32 核心：任务收集分流 ===

def collect_tasks_source_master(en_data, cn_data, path_str="root"):
    """
    [旧逻辑] 以 Source (英文) 为主。
    如果 Source 有但 Target 没有，会新增（可能破坏 Target 结构）。
    """
    tasks = []
    
    def get_cn(d, k):
        if isinstance(d, dict): return d.get(k)
        if isinstance(d, list) and isinstance(k, int) and k < len(d): return d[k]
        return None

    if isinstance(en_data, dict): iter_items = en_data.items()
    elif isinstance(en_data, list): iter_items = enumerate(en_data)
    else: return []

    for k, v in iter_items:
        cur_path = f"{path_str}.{k}" if isinstance(en_data, dict) else f"{path_str}[{k}]"
        cn_v = get_cn(cn_data, k)
        
        # 判断逻辑
        should_translate = False
        if isinstance(v, str) and len(v) > 1:
            has_en = contains_english(v)
            is_file = v.lower().endswith(('.png', '.webp', '.jpg', '.mp3', '.ogg', '.m4a', '.webm'))
            is_target_key = isinstance(en_data, dict) and k in TARGET_KEYS
            is_in_container = any(c in path_str.split('.') for c in SPECIAL_CONTAINERS)
            if has_en and not is_file:
                if BRUTE_FORCE_MODE:
                    should_translate = True
                else:
                    if is_target_key or is_in_container:
                        should_translate = True

        if should_translate:
            tt = 'NEW'
            if isinstance(cn_v, str) and contains_english(cn_v) and contains_chinese(cn_v):
                tt = 'AUDIT'
            if isinstance(v, str) and isinstance(cn_v, str) and contains_html_tags(v):
                if (not contains_html_tags(cn_v)) or is_garbled_html(cn_v) or is_html_structure_changed(v, cn_v):
                    tt = 'NEW'
            if isinstance(cn_v, str) and has_placeholder_leak(cn_v):
                tt = 'NEW'

            task_hash = get_content_hash(v, cn_v if isinstance(cn_v, str) else "")
            if tt == 'AUDIT' and not can_audit(task_hash):
                continue
            if tt == 'NEW' and cn_v and get_content_hash(v, cn_v) in history_cache:
                continue
            tasks.append({'type': tt, 'mode': None, 'ref': en_data, 'k': k, 'en_v': v, 'cn_v': cn_v if tt=='AUDIT' else None, 'path': cur_path, 'hash': task_hash})
            
        elif isinstance(v, (dict, list)):
            new_cn = cn_v if isinstance(cn_v, (dict, list)) else {}
            tasks.extend(collect_tasks_source_master(v, new_cn, cur_path))
            
    return tasks

def collect_tasks_target_master(cn_data, en_data, path_str="root"):
    """
    [V32 新逻辑] 以 Target (中文) 为主。
    只遍历 Target 的结构。如果 Target 里有英文，就翻译。
    完全忽略 Source 中多出来的结构（Dump掉）。
    """
    tasks = []
    
    # 辅助函数：尝试在 Source 数据里找到对应的路径，以获取最纯正的原文（用于参考）
    def get_en_counterpart(source_node, key):
        if source_node is None: return None
        if isinstance(source_node, dict): return source_node.get(key)
        if isinstance(source_node, list) and isinstance(key, int) and key < len(source_node): return source_node[key]
        return None

    if isinstance(cn_data, dict): iter_items = cn_data.items()
    elif isinstance(cn_data, list): iter_items = enumerate(cn_data)
    else: return []

    for k, v in iter_items:
        cur_path = f"{path_str}.{k}" if isinstance(cn_data, dict) else f"{path_str}[{k}]"
        
        # 尝试去 Source 里找对应的原文
        # 如果结构不匹配（Target结构浅，Source结构深），en_v 可能是 None
        en_v = get_en_counterpart(en_data, k)
        
        # 判断逻辑
        should_translate = False
        if isinstance(v, str) and len(v) > 1:
            style = get_value_style(cur_path, k)
            if style != "skip":
                has_en = contains_english(v)
                is_file = v.lower().endswith(('.png', '.webp', '.jpg', '.mp3', '.ogg', '.m4a', '.webm'))
                if has_en and not is_file:
                    should_translate = True

        if should_translate:
            # 如果 Source 里找不到对应的 en_v (因为结构不同)，我们就把当前 Target 里的 v 当作原文
            original_text = en_v if (en_v and isinstance(en_v, str)) else v
            if isinstance(original_text, str) and isinstance(v, str) and en_v is None:
                if contains_chinese(original_text) and contains_english(original_text) and not contains_html_tags(original_text):
                    eng_only = extract_english_portion(original_text)
                    if eng_only:
                        original_text = eng_only
            task_hash = get_content_hash(original_text, v)

            task_type = 'AUDIT' if (contains_chinese(v) and contains_english(v)) else 'NEW'
            if isinstance(original_text, str) and isinstance(v, str) and contains_html_tags(original_text):
                if (not contains_html_tags(v)) or is_garbled_html(v) or is_html_structure_changed(original_text, v):
                    task_type = 'NEW'
                if SAFE_MODE:
                    task_type = 'NEW'
            if isinstance(v, str) and has_placeholder_leak(v):
                task_type = 'NEW'
            if task_type == 'AUDIT' and not can_audit(task_hash):
                continue
            if task_type == 'NEW' and get_content_hash(original_text, v) in history_cache:
                continue
            
            # 注意：这里的 ref 是 cn_data，因为我们要回写到 Target
            tasks.append({
                'type': task_type,
                'mode': None,
                'ref': cn_data, 
                'k': k,
                'en_v': original_text, # 送给 AI 的参考原文
                'cn_v': v,             # 送给 AI 的现有译文（用于校对）
                'path': cur_path,
                'hash': task_hash
            })
            
        elif isinstance(v, (dict, list)):
            # 递归时，en_v 可能是 None (如果结构对不上)，这没关系，继续往下传 None 即可
            tasks.extend(collect_tasks_target_master(v, en_v, cur_path))
            
    return tasks

def check_environment():
    """检查运行环境和API配置"""
    issues = []
    
    # 检查API密钥
    if not GOOGLE_API_KEY and not OPENAI_API_KEY:
        issues.append("❌ 未配置任何API密钥。请设置环境变量: GOOGLE_API_KEY 或 OPENAI_API_KEY")
    
    if GOOGLE_API_KEY and not google_client:
        issues.append("⚠️ Google API密钥配置失败，跳过Google服务")
    
    if OPENAI_API_KEY and not openai_client:
        issues.append("⚠️ OpenAI API密钥配置失败，跳过OpenAI服务")
    
    # 检查数据文件
    if not SOURCE_EN_JSON_PATH.exists():
        issues.append(f"❌ 源文件不存在: {SOURCE_EN_JSON_PATH}")
    
    if SYNC_MODE == "TARGET_MASTER" and not TARGET_JSON_PATH.exists():
        issues.append(f"❌ TARGET_MASTER模式需要目标文件: {TARGET_JSON_PATH}")
    
    # 输出检查结果
    for issue in issues:
        print(issue)
    
    # 如果有关键错误则返回False
    return len([i for i in issues if i.startswith("❌")]) == 0

def main():
    """主函数：协调整个翻译流程"""
    cfg = load_config()
    apply_config(cfg)
    init_clients()
    print(f"PF2e 汉化脚本 V32 (同步模式: {SYNC_MODE})")
    print(f"源文件: {SOURCE_EN_JSON_PATH.name if SOURCE_EN_JSON_PATH.exists() else '(不存在)'}")
    print(f"目标文件: {TARGET_JSON_PATH.name if TARGET_JSON_PATH.exists() else '(不存在)'}")
    print("")
    
    # 环境检查
    if not check_environment():
        return

    if TEST_MODE:
        global SIMULATE_AI
        SIMULATE_AI = True

    # 备份现有文件
    backup_existing_files()
    global history_cache
    history_cache = load_history()
    global audit_history
    audit_history = load_audit_history()
    print(f"🧠 已加载缓存: {len(history_cache)} 条记录")
    write_process_log(f"缓存条目: {len(history_cache)}")
    write_process_log(f"校对历史条目: {len(audit_history)}")

    # 加载源文件
    with SOURCE_EN_JSON_PATH.open('r', encoding='utf-8-sig') as f:
        en_data = json.load(f)
    
    # 加载或初始化目标文件
    cn_data = {}
    if TARGET_JSON_PATH.exists():
        try:
            print("🔄 加载目标文件...")
            with TARGET_JSON_PATH.open('r', encoding='utf-8-sig') as f:
                cn_data = json.load(f)
            write_process_log(f"目标文件加载成功: {TARGET_JSON_PATH}")
        except Exception as e:
            print(f"❌ 加载目标文件失败: {e}")
            if SYNC_MODE == "TARGET_MASTER":
                print("⛔ 在 TARGET_MASTER 模式下，目标文件必须存在且有效！")
                return
    else:
        if SYNC_MODE == "TARGET_MASTER":
            print("⛔ 错误：TARGET_MASTER 模式需要目标文件存在 (作为结构模板)。")
            return
        # Source Master 模式可以从空开始
        cn_data = {}

    original_cn_snapshot = deepcopy(cn_data)

    extract_local_glossary(en_data, cn_data, LOCAL_GLOSSARY_EXPORT_PATH)
    glossary = GlossaryManager(GLOBAL_GLOSSARY_PATH, LOCAL_GLOSSARY_EXPORT_PATH)
    write_process_log(f"术语库加载完成: {len(glossary.sorted_keys)} 条")

    if TEST_MODE:
        print("🧪 测试模式：不调用AI，仅做格式一致性检查...")
        test_cn = deepcopy(cn_data)

        if TEST_MODE_SIMULATE_PIPELINE:
            print("🧪 测试模式：模拟处理流程（不落盘）...")
            all_tasks = collect_tasks_target_master(test_cn, en_data) if SYNC_MODE == "TARGET_MASTER" else collect_tasks_source_master(en_data, test_cn)
            rl = RateLimiter(TARGET_RPM)
            changed_paths = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
                fut_map = {}
                for t in tqdm(all_tasks, desc="分发(测试-模拟)", position=0, leave=True, dynamic_ncols=True):
                    rl.wait_for_slot()
                    future = exe.submit(process_single_item, t['type'], t['en_v'], t['cn_v'], glossary, t['path'], t.get('mode'), t.get('hash'))
                    fut_map[future] = t
                for f in tqdm(concurrent.futures.as_completed(fut_map), total=len(all_tasks), desc="回收(测试-模拟)", position=1, leave=True, dynamic_ncols=True):
                    task = fut_map[f]
                    try:
                        res, _ = f.result(timeout=30)
                        original_v = task['ref'][task['k']]
                        if isinstance(original_v, str) and isinstance(res, str) and original_v != res:
                            changed_paths.append(task['path'])
                    except Exception as e:
                        write_process_log(f"❌ 测试任务失败: {task['path']} - {e}")

            print(f"🧪 模拟流程产生改动条目数: {len(changed_paths)}")
            if changed_paths:
                sample = changed_paths[:20]
                print("🧪 改动样例(最多20条):")
                for p in sample:
                    print(f"   {p}")

        # 测试模式下也执行与正式流程一致的输出规范化
        fixed_cnt = normalize_output_inplace(test_cn, en_data)
        if fixed_cnt:
            write_process_log(f"🧹 测试输出规范化: {fixed_cnt} 项")

        if TEST_MODE_WRITE:
            with TEST_OUTPUT_PATH.open('w', encoding='utf-8') as f:
                json.dump(test_cn, f, ensure_ascii=False, indent=2)
            write_process_log(f"测试输出写入: {TEST_OUTPUT_PATH}")

        scan_format_anomalies(test_cn, en_data)
        stats = validate_format_consistency(test_cn, en_data)
        print("\n" + "="*50)
        print("🧪 测试模式格式检查:")
        for k, v in stats.items():
            print(f"   {k}: {v}")
        print("="*50)
        return
    
    print("构建任务队列...")
    
    # === V32 分流逻辑 ===
    if SYNC_MODE == "TARGET_MASTER":
        # 遍历 Target，忽略 Source 多余结构
        all_tasks = collect_tasks_target_master(cn_data, en_data)
    else:
        # 遍历 Source，强制补全 Target
        all_tasks = collect_tasks_source_master(en_data, cn_data)
        
    print(f"待处理任务: {len(all_tasks)}")
    # 统计任务类型
    type_counts = {"NEW": 0, "AUDIT": 0}
    for t in all_tasks:
        if t["type"] in type_counts:
            type_counts[t["type"]] += 1
    write_process_log(f"任务统计: NEW={type_counts['NEW']}, AUDIT={type_counts['AUDIT']}")
    
    if not all_tasks: 
        print("🎉 没有需要更新的内容！")
        return

    rl = RateLimiter(TARGET_RPM)
    print("🚀 引擎启动...")
    write_process_log(f"线程池启动: workers={MAX_WORKERS}, RPM={TARGET_RPM}")
    
    # 使用线程池并发处理任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        fut_map = {}
        # 提交任务
        for t in tqdm(all_tasks, desc="分发", position=0, leave=True, dynamic_ncols=True):
            rl.wait_for_slot()
            future = exe.submit(process_single_item, t['type'], t['en_v'], t['cn_v'], glossary, t['path'], t.get('mode'), t.get('hash'))
            fut_map[future] = t
        
        # 收集结果
        for f in tqdm(concurrent.futures.as_completed(fut_map), total=len(all_tasks), desc="回收", position=1, leave=True, dynamic_ncols=True):
            task = fut_map[f]
            try:
                res, st = f.result(timeout=30)  # 单个任务超时30秒
                task['ref'][task['k']] = res
                # 如果翻译未改变，加入缓存以加快下次运行
                if st == "Kept":
                    with log_lock:
                        new_history_entries.add(get_content_hash(task['en_v'], res))
            except concurrent.futures.TimeoutError:
                write_process_log(f"❌ 任务超时: {task['path']}")
            except Exception as e:
                write_process_log(f"❌ 任务失败: {task['path']} - {e}")

    print("正在保存最终结果...")
    # Target Master 模式：保持Target结构，只更新值
    # Source Master 模式：使用Source结构，强制补全Target
    output_obj = cn_data if SYNC_MODE == "TARGET_MASTER" else en_data

    # 规范化输出风格（短字段双语/中文、去除原文块）
    if SYNC_MODE == "TARGET_MASTER":
        fixed_cnt = normalize_output_inplace(cn_data, en_data)
        if fixed_cnt:
            write_process_log(f"🧹 输出规范化: {fixed_cnt} 项")
    
    # 保存翻译结果
    with TARGET_JSON_PATH.open('w', encoding='utf-8') as f:
        json.dump(output_obj, f, ensure_ascii=False, indent=2)
    write_process_log(f"写入目标文件: {TARGET_JSON_PATH}")

    # 差异报告
    if DIFF_REPORT_ENABLED:
        changed_cnt = generate_diff_report(original_cn_snapshot, cn_data, DIFF_REPORT_PATH)
        if changed_cnt:
            write_process_log(f"差异报告生成: {DIFF_REPORT_PATH} | 变更: {changed_cnt}")

    # 全量格式检查（不做任何修改，仅记录）
    scan_format_anomalies(cn_data, en_data)
    
    # 保存遗漏日志
    if missed_log_buffer:
        with MISSED_LOG_PATH.open('w', encoding='utf-8') as f:
            f.write("\n".join(missed_log_buffer))
        print(f"⚠️ 警告：有 {len(missed_log_buffer)} 条内容漏翻")
    
    # 保存审查报告（支持重试）
    max_retry = 3
    for attempt in range(max_retry):
        try:
            with pd.ExcelWriter(REPORT_XLSX_PATH) as w:
                pd.DataFrame(report_data["New"]).to_excel(w, sheet_name="New", index=False)
                pd.DataFrame(report_data["Fixed"]).to_excel(w, sheet_name="Fixed", index=False)
                pd.DataFrame(report_data["Kept"]).to_excel(w, sheet_name="Kept", index=False)
                pd.DataFrame(report_data["TermAdjusted"]).to_excel(w, sheet_name="TermAdjusted", index=False)
                pd.DataFrame(report_data["FormatAnomaly"]).to_excel(w, sheet_name="FormatAnomaly", index=False)
            break
        except PermissionError:
            if attempt < max_retry - 1:
                input(f"❌ 请关闭 {REPORT_XLSX_PATH} 后回车...")
            else:
                print(f"⚠️ 无法保存报告，文件被占用")
        except Exception as e:
            write_process_log(f"⚠️ 保存报告失败: {e}")
            break

    save_history()
    write_process_log("历史缓存已保存")
    save_audit_history()
    write_process_log("校对历史已保存")
    # 刷新日志缓冲区
    _flush_process_log()
    write_process_log("日志已刷新")
    
    # 生成总结报告
    print("\n" + "="*50)
    print("📊 本次运行统计:")
    print(f"   新增翻译: {len(report_data['New'])} 项")
    print(f"   修复翻译: {len(report_data['Fixed'])} 项")
    print(f"   保持不变: {len(report_data['Kept'])} 项")
    if missed_log_buffer:
        print(f"   ⚠️ 漏翻项目: {len(missed_log_buffer)} 项")
    print("="*50)
    print("🎉 全部完成。")

if __name__ == "__main__":
    main()