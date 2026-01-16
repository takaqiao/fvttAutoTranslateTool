# Foundry VTT 模组智能汉化脚本

面向 Foundry VTT（Babele JSON）模组汉化的自动化脚本，支持结构安全同步、术语一致性、自动审校与格式保护，适合长期维护。

## 1. 一句话特性

- 结构不乱：可选以目标结构为准或以源结构补全。
- 术语统一：全局术语 + 本地术语自动注入。
- 格式稳定：保护 HTML、`@UUID`、`[[/r ...]]` 等标记。
- 审校可控：多轮审校 + 缓存减少重复翻译。

## 2. 快速使用

1. 配置环境变量：`OPENAI_API_KEY` 或 `GOOGLE_API_KEY`。
2. 确认文件路径：`SOURCE_EN_JSON_PATH` / `TARGET_JSON_PATH`。
3. 运行：`python pf2e_translator.py`。

## 3. 核心配置（只看这些）

- `SYNC_MODE`
  - `TARGET_MASTER`：只改已有键，结构最安全。
  - `SOURCE_MASTER`：按英文结构补全，可能增键。
- `MODEL_PRIORITY_LIST`：模型优先级与自动降级。
- `MAX_WORKERS` / `TARGET_RPM`：并发与速率控制。
- `SAFE_MODE`：HTML 分段翻译 + 结构保护。
- `TEST_MODE`：不调用 AI，仅格式检查。
- `TEST_MODE_SIMULATE_PIPELINE`：模拟完整流程（不落盘）。

## 4. 输出与文件

- 目标文件：`pf2e-beginner-box.adventures.json`
- 测试输出：`pf2e-beginner-box.adventures.test.json`
- 审查报告：`翻译审查报告.xlsx`
- 运行日志：`运行日志.txt`
- 历史缓存：`translation_history.json`
- 校对历史：`audit_history.json`
- 备份目录：`backups/`

## 5. 术语表

- 全局：`术语译名对照表.csv`
- 本地：自动生成 `术语表_本地提取.csv`
- 优先级：本地 > 全局 > 模型输出

## 6. 工具

`glossary_cleaner.py`：清洗术语表并输出冲突/剔除结果。

## 7. 致谢

核心术语表来源：**[开拓者2版中文维基 (灰机Wiki)](https://pf2.huijiwiki.com/wiki/%E6%9C%AF%E8%AF%AD%E7%B4%A2%E5%BC%95)**。