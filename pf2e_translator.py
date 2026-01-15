import json
import os
import time
import re
import hashlib
import concurrent.futures
import pandas as pd
from pathlib import Path
from datetime import datetime
from threading import Lock
from tqdm import tqdm

# 需要安装: pip install google-genai openai pandas openpyxl
from google import genai
from google.genai import types
from openai import OpenAI

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

# 5. [核心开关] 暴力防漏模式 (仅在翻译内容判断时生效)
BRUTE_FORCE_MODE = True 

# 6. 术语表与日志 (自动转换为Path对象)
GLOBAL_GLOSSARY_PATH = Path("术语译名对照表.csv") 
LOCAL_GLOSSARY_EXPORT_PATH = Path("术语表_本地提取.csv")
REPORT_XLSX_PATH = Path("翻译审查报告.xlsx")
PROCESS_LOG_PATH = Path("运行日志.txt")
MISSED_LOG_PATH = Path("失败漏翻记录.txt")
HISTORY_FILE_PATH = Path("translation_history.json")
BACKUP_DIR = Path("backups")

# 7. 日志输出
PRINT_LOG_TO_TERMINAL = True  # 同步输出到终端，便于实时观察

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

# 初始化客户端
google_client = None
openai_client = None

if any(p == "google" for p, m in MODEL_PRIORITY_LIST) and GOOGLE_API_KEY:
    google_client = genai.Client(api_key=GOOGLE_API_KEY)

if any(p == "openai" for p, m in MODEL_PRIORITY_LIST) and OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

log_lock = Lock()
report_data = {"New": [], "Fixed": [], "Kept": []}
process_log_buffer = []
missed_log_buffer = [] 
history_cache = set()
new_history_entries = set()

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
    def __init__(self):
        self.patterns = [
            re.compile(r'(@[a-zA-Z0-9]+\[[^\]]*\])'),
            re.compile(r'(\[\[.*?\]\])'),
            re.compile(r'(<[^>]+>)'),
            re.compile(r'(&[a-zA-Z0-9#]+;)'),
        ]
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
            if k not in text: text = re.sub(k.replace('_', r'\s*_\s*'), v, text)
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
    """智能格式化双语文本
    
    Args:
        cn: 中文文本
        en: 英文原文
    
    Returns:
        格式化后的文本（按需添加原文）
    """
    if not cn:
        return en
    # 清理注入标签
    cn = re.sub(r'⟪(.*?)\|原文:.*?⟫', r'\1', cn)
    # 检查中文是否已包含英文内容
    clean_en = re.sub(r'[\s\W]', '', en).lower()
    clean_cn = re.sub(r'[\s\W]', '', cn).lower()
    if clean_en in clean_cn:
        return cn
    # 短文本：如果译文里已包含英文，避免再次追加原文导致重复
    if en and len(en) <= 80 and re.search(r'[A-Za-z]', cn):
        return cn
    # 短文本：如果译文已包含原文中的任一关键英文词，也视为已含原文
    if en and len(en) <= 80:
        for w in re.findall(r"[A-Za-z][A-Za-z']{2,}", en):
            if re.search(rf"\b{re.escape(w)}\b", cn, flags=re.IGNORECASE):
                return cn
    # 长文本用换行分隔，短文本用空格
    sep = "<br><br><hr><b>原文:</b><br>" if (len(en) > 80 or "<p>" in en) else " "
    return f"{cn}{sep}{en}"

def extract_local_glossary(en_data, cn_data, output_path):
    """从翻译数据中提取本地术语表"""
    print("正在扫描本地术语...")
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
                if clean_cn and re.search(r'[\u4e00-\u9fff]', clean_cn):
                    extracted.append({'Source': en_node, 'Target': clean_cn})
    
    traverse(en_data, cn_data)
    if extracted:
        try:
            df = pd.DataFrame(extracted).drop_duplicates(subset=['Source'])
            output_path = Path(output_path)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            write_process_log(f"⚠️ 导出术语表失败: {e}")

def clean_response_text(text):
    """清理AI响应文本"""
    if not text:
        return ""
    text = re.sub(r'^```[a-zA-Z]*\n', '', text)
    text = re.sub(r'\n```$', '', text)
    text = re.sub(r'^(Here is|Below is|以下是).*?(\n|$)', '', text, flags=re.IGNORECASE).strip()
    return text.strip()

def cleanup_injection_tags(text):
    """清理术语注入标签"""
    if not text:
        return text
    # 移除注入标签，保留翻译
    text = re.sub(r'⟪(.*?)\|原文:.*?⟫', r'\1', text)
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

protector = CodeProtector()

def process_single_item(task_type, en_text, cn_draft, glossary_mgr, path_str):
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
    
    # 代码保护和术语注入
    prot = CodeProtector()
    masked, code_ph = prot.mask(en_text)
    injected, terms = glossary_mgr.pre_inject_text(masked, path_str)
    
    # 清理初稿
    clean_draft_txt = cn_draft
    if cn_draft and "<hr>" in cn_draft:
        clean_draft_txt = cn_draft.split("<hr>")[0].strip()
    
    # 构建提示词
    sys_prompt = "You are a professional Pathfinder 2e translator. Output ONLY Chinese. Keep HTML/Codes."
    if task_type == "AUDIT":
        prompt = (
            f"Original:\n```\n{injected}\n```\nDraft:\n```\n{clean_draft_txt}\n```\n"
            f"Task: Review draft. If correct, output it. If wrong, correct it."
        )
    else:
        prompt = f"Translate:\n```\n{injected}\n```"

    try:
        res_text = call_ai_with_fallback(sys_prompt, prompt, path_str)
        trans = clean_response_text(res_text)
        final_trans = prot.unmask(trans, code_ph)
        final_trans = cleanup_injection_tags(final_trans)
        final_trans = collapse_duplicate_cn_prefix(final_trans)

        # 确定状态
        status = "New"
        if task_type == "AUDIT":
            # 对比前后去空格
            if re.sub(r'\s', '', final_trans) == re.sub(r'\s', '', clean_draft_txt):
                status = "Kept"
            else:
                status = "Fixed"

        log_report(status, path_str, en_text, final_trans, terms)
        write_process_log(f"✅ 任务完成: {status} | {path_str}")
        return smart_format_bilingual(final_trans, en_text), status

    except Exception as e:
        write_process_log(f"❌ 所有模型失败 {path_str}: {e}")
        write_missed_log(path_str, en_text, "All Models Failed")
        fallback = smart_format_bilingual(cn_draft, en_text) if cn_draft else f"【FAIL】{en_text}"
        return fallback, None

def log_report(status, path, original, translated, injected_terms):
    term_str = " | ".join([f"{e}->{c}" for e, c in injected_terms])
    row = {"JSON Path": path, "Involved Terms": term_str, "Original": original, "Translation": translated}
    with log_lock:
        if status in report_data:
            report_data[status].append(row)

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
            has_letters = bool(re.search(r'[a-zA-Z]', v))
            if has_letters:
                is_target_key = False
                if isinstance(en_data, dict) and k in TARGET_KEYS: is_target_key = True
                elif any(c in path_str.split('.') for c in SPECIAL_CONTAINERS): is_target_key = True
                is_file = v.lower().endswith(('.png', '.webp', '.jpg', '.mp3', '.ogg', '.m4a', '.webm'))
                has_space = " " in v
                if BRUTE_FORCE_MODE:
                    if not is_file and (is_target_key or has_space): should_translate = True
                else:
                    if is_target_key: should_translate = True

        if should_translate:
            if cn_v and get_content_hash(v, cn_v) in history_cache: continue
            tt = 'AUDIT' if (cn_v and isinstance(cn_v, str) and len(cn_v) > 0 and cn_v != v) else 'NEW'
            tasks.append({'type': tt, 'ref': en_data, 'k': k, 'en_v': v, 'cn_v': cn_v if tt=='AUDIT' else None, 'path': cur_path})
            
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
        # 我们检查 v (Target里的值)。如果它是字符串，且包含英文，我们就得翻。
        # 不管它是不是在白名单里，只要在 Target 文件里出现了，通常都意味着要保留。
        if isinstance(v, str) and len(v) > 1:
            has_letters = bool(re.search(r'[a-zA-Z]', v))
            if has_letters:
                # 依然应用一些基础过滤，防止翻译 ID 或 路径
                is_file = v.lower().endswith(('.png', '.webp', '.jpg', '.mp3', '.ogg', '.m4a', '.webm'))
                is_target_key = False
                if isinstance(cn_data, dict) and k in TARGET_KEYS: is_target_key = True
                
                # 在 TargetMaster 模式下，如果 key 在白名单里，或者看起来像句子（有空格），就翻
                if not is_file and (is_target_key or " " in v):
                     should_translate = True

        if should_translate:
            # 如果 Source 里找不到对应的 en_v (因为结构不同)，我们就把当前 Target 里的 v 当作原文
            original_text = en_v if (en_v and isinstance(en_v, str)) else v
            
            # 检查缓存
            if get_content_hash(original_text, v) in history_cache: continue
            
            # 如果 v 已经是中文了，或者含中文，我们标记为 AUDIT；如果是纯英文，标记为 NEW
            is_translated = bool(re.search(r'[\u4e00-\u9fff]', v))
            task_type = 'AUDIT' if is_translated else 'NEW'
            
            # 注意：这里的 ref 是 cn_data，因为我们要回写到 Target
            tasks.append({
                'type': task_type,
                'ref': cn_data, 
                'k': k,
                'en_v': original_text, # 送给 AI 的参考原文
                'cn_v': v,             # 送给 AI 的现有译文（用于校对）
                'path': cur_path
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
    print(f"PF2e 汉化脚本 V32 (同步模式: {SYNC_MODE})")
    print(f"源文件: {SOURCE_EN_JSON_PATH.name if SOURCE_EN_JSON_PATH.exists() else '(不存在)'}")
    print(f"目标文件: {TARGET_JSON_PATH.name if TARGET_JSON_PATH.exists() else '(不存在)'}")
    print("")
    
    # 环境检查
    if not check_environment():
        return

    # 备份现有文件
    backup_existing_files()
    global history_cache
    history_cache = load_history()
    print(f"🧠 已加载缓存: {len(history_cache)} 条记录")
    write_process_log(f"缓存条目: {len(history_cache)}")

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

    extract_local_glossary(en_data, cn_data, LOCAL_GLOSSARY_EXPORT_PATH)
    glossary = GlossaryManager(GLOBAL_GLOSSARY_PATH, LOCAL_GLOSSARY_EXPORT_PATH)
    write_process_log(f"术语库加载完成: {len(glossary.sorted_keys)} 条")
    
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
        for t in tqdm(all_tasks, desc="分发"):
            rl.wait_for_slot()
            future = exe.submit(process_single_item, t['type'], t['en_v'], t['cn_v'], glossary, t['path'])
            fut_map[future] = t
        
        # 收集结果
        for f in tqdm(concurrent.futures.as_completed(fut_map), total=len(all_tasks), desc="回收"):
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
    
    # 保存翻译结果
    with TARGET_JSON_PATH.open('w', encoding='utf-8') as f:
        json.dump(output_obj, f, ensure_ascii=False, indent=2)
    write_process_log(f"写入目标文件: {TARGET_JSON_PATH}")
    
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