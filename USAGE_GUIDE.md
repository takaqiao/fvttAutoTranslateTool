# PF2e 汉化脚本使用指南

## 快速开始

### 环境配置

#### 1. 安装依赖
```bash
pip install google-genai openai pandas openpyxl tqdm
```

#### 2. 配置API密钥

**方式一：PowerShell (Windows)**
```powershell
# 临时设置（仅当前会话）
$env:GOOGLE_API_KEY = "your_google_api_key"
$env:OPENAI_API_KEY = "your_openai_api_key"

# 永久设置（推荐）
[Environment]::SetEnvironmentVariable("GOOGLE_API_KEY", "your_google_api_key", "User")
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your_openai_api_key", "User")
```

**方式二：Bash (Linux/Mac)**
```bash
export GOOGLE_API_KEY="your_google_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# 或添加到 ~/.bashrc 或 ~/.zshrc 以持久化
echo 'export GOOGLE_API_KEY="..."' >> ~/.bashrc
source ~/.bashrc
```

#### 3. 准备数据文件

脚本需要以下文件在同一目录：
- `pf2e-beginner-box-en.json` - 英文源文件（必需）
- `pf2e-beginner-box.adventures.json` - 中文目标文件（可选）
- `术语译名对照表.csv` - 术语表（可选）

### 运行脚本

```bash
python pf2e_translator.py
```

## 配置项说明

### 同步模式选择

在 `pf2e_translator.py` 中修改 `SYNC_MODE`：

```python
# TARGET_MASTER: 以目标文件(CN)结构为准
# 优点：结构绝对安全，不会新增未预期的字段
# 缺点：只能更新现有字段，不能补全新字段
SYNC_MODE = "TARGET_MASTER"

# SOURCE_MASTER: 以源文件(EN)结构为准
# 优点：可自动补全缺失的字段
# 缺点：可能导致结构改变，不建议用于生产环境
```

### 模型优先级

```python
MODEL_PRIORITY_LIST = [
    ("openai", "gpt-5.2"),                    # 优先级1
    ("openai", "gpt-5-mini"),                 # 优先级2
    ("google", "gemini-3-flash-preview"),     # 优先级3
]
```

脚本会按顺序尝试调用模型，遇到错误自动降级。

### 性能参数

```python
MAX_WORKERS = 16       # 并发线程数（建议：8-16）
TARGET_RPM = 450       # API调用速率限制（请求/分钟）
MAX_RETRIES = 5        # 单个模型的重试次数
BRUTE_FORCE_MODE = True # 激进的翻译判断模式
```

## 输出文件说明

运行完成后生成的文件：

| 文件名 | 说明 | 何时生成 |
|--------|------|---------|
| `pf2e-beginner-box.adventures.json` | 翻译结果 | 总是 |
| `翻译审查报告.xlsx` | 审查报告（按状态分表） | 总是 |
| `术语表_本地提取.csv` | 从翻译中提取的术语 | 总是 |
| `translation_history.json` | 翻译历史缓存 | 总是 |
| `失败漏翻记录.txt` | 失败的翻译项 | 有漏翻时 |
| `backups/` | 每次运行的备份 | 总是 |

### 审查报告详解

Excel报告包含三个Sheet：

1. **New** - 新增翻译的项目
   - JSON Path: 项目在JSON中的路径
   - Original: 英文原文
   - Translation: 中文翻译
   - Involved Terms: 涉及的术语映射

2. **Fixed** - 修正过的翻译
   - 原有翻译有误，已修正的项目

3. **Kept** - 保持不变
   - 翻译已经正确，无需修改的项目

## 术语表管理

### CSV格式

```csv
Source,Target
Pathfinder,龙与地下城：探险者 (PF)
Spell,法术
Feat,专长
```

### 术语自动提取

脚本运行时会在 `术语表_本地提取.csv` 中提取已翻译的术语。可以：
1. 复制提取的术语到 `术语译名对照表.csv`
2. 下次运行时自动应用这些术语

## 故障排除

### 问题：API密钥不被识别

**解决**：
1. 确认环境变量已正确设置
2. 重启 PowerShell 或 IDE
3. 运行 `echo $env:GOOGLE_API_KEY` 验证

```powershell
# 检查环境变量是否正确设置
echo $env:GOOGLE_API_KEY
echo $env:OPENAI_API_KEY
```

### 问题：网络错误或超时

**解决**：
1. 检查网络连接
2. 调整 `MAX_WORKERS` 为较小值（如 4）
3. 增加 `MAX_RETRIES` 数值
4. 尝试使用代理或VPN

```python
# 减小并发，提高稳定性
MAX_WORKERS = 4
MAX_RETRIES = 10
```

### 问题：API调用限制

**解决**：
- 降低 `TARGET_RPM` 参数
- 等待一段时间后重新运行

```python
TARGET_RPM = 200  # 从450降低到200请求/分钟
```

### 问题：内存占用过高

**解决**：
1. 减少 `MAX_WORKERS`
2. 在日志缓冲达到阈值时自动刷盘

```python
MAX_WORKERS = 4  # 减少并发线程
```

## 日志文件位置

所有日志和临时文件都在脚本同一目录：

```
pf2e/
├── pf2e_translator.py
├── pf2e-beginner-box-en.json
├── pf2e-beginner-box.adventures.json
├── 术语译名对照表.csv
├── 翻译审查报告.xlsx           ← 主要输出
├── 术语表_本地提取.csv          ← 术语自动提取
├── translation_history.json     ← 缓存
├── 失败漏翻记录.txt             ← 错误记录
├── 运行日志.txt                 ← 详细日志
└── backups/                     ← 自动备份目录
    ├── 20260115_120000_...json.bak
    └── ...
```

## 高级用法

### 多次运行优化

首次运行会更慢（需要翻译大量内容），后续运行会更快（利用缓存）：

```bash
# 第一次运行：完整翻译（可能耗时数小时）
python pf2e_translator.py

# 后续运行：仅处理新增内容（数分钟）
python pf2e_translator.py
```

### 批量处理多个文件

创建包装脚本 `batch_translate.py`：

```python
import subprocess
from pathlib import Path

files = [
    ("pf2e-beginner-box-en.json", "pf2e-beginner-box.adventures.json"),
    # 添加更多文件对
]

for source, target in files:
    print(f"处理 {target}...")
    # 修改pf2e_translator.py中的路径变量后运行
    subprocess.run(["python", "pf2e_translator.py"])
```

## 性能优化建议

| 场景 | 建议配置 |
|------|---------|
| 快速测试 | MAX_WORKERS=4, TARGET_RPM=200 |
| 标准运行 | MAX_WORKERS=16, TARGET_RPM=450 |
| 高速运行 | MAX_WORKERS=32, TARGET_RPM=800 |
| 低配机器 | MAX_WORKERS=2, TARGET_RPM=100 |

---
**更多帮助**：检查脚本中的注释和函数docstring

