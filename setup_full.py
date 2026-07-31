#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整模式一键部署预检：python setup_full.py

按链路顺序逐项检查，缺什么打印精确的解决命令；全部就绪则自动构建向量索引。
四个环节：
  1. Python 依赖（torch / transformers / faiss / python-docx）
  2. config/config.py（不存在则从 example 自动生成并填好 3 处）
  3. Jina v4 模型权重（约 7.5GB，需手动下载一次）
  4. 向量索引（调用现有 rebuild_index.py 构建）

完成后运行:  python API_KIT/web_server.py
前端顶栏显示「完整系统 · Jina v4」即部署成功。
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "config.py"
CONFIG_EXAMPLE = ROOT / "config" / "config.example.py"
MODEL_DIR = ROOT / "models" / "jina-embeddings-v4"
DOCS_DIR = ROOT / "docs"


def step(n, title):
    print(f"\n{'=' * 60}\n[{n}/4] {title}\n{'=' * 60}")


def fail(msg):
    print(f"\n❌ {msg}")
    sys.exit(1)


# ---------------------------------------------------------------- 1. 依赖
step(1, "检查 Python 依赖")
missing = []
for mod, pkg in [("torch", "torch"), ("transformers", "transformers"),
                 ("faiss", "faiss-cpu"), ("docx", "python-docx"),
                 ("numpy", "numpy"), ("tqdm", "tqdm"), ("requests", "requests"),
                 ("peft", "peft")]:
    try:
        __import__(mod)
        print(f"  ✅ {pkg}")
    except ImportError:
        missing.append(pkg)
        print(f"  ❌ {pkg} 未安装")

if missing:
    fail("依赖不完整，请先执行：\n\n"
         "    pip install -r requirements.txt\n\n"
         "（jieba 已从依赖中移除，旧环境装不上 jieba 不影响本项目）")

import torch  # noqa: E402
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"  🎮 检测到 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  ⚠️ 未检测到 GPU，将使用 CPU（建索引约 15~40 分钟，查询时每次编码约 1~3 秒）")

# ---------------------------------------------------------------- 2. config
step(2, "检查 config/config.py")
if CONFIG.exists():
    print(f"  ✅ 已存在: {CONFIG}")
else:
    text = CONFIG_EXAMPLE.read_text(encoding="utf-8")

    api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if api_key:
        text = text.replace('"your-api-key-here"', f'"{api_key}"')
        key_note = "已从环境变量 SILICONFLOW_API_KEY 写入"
    else:
        key_note = ("未配置（不影响建索引；查询时无法生成答案。"
                    "到 https://siliconflow.cn 申请后填入 config/config.py 的 api_key）")

    text = text.replace('r"您的文档路径"', 'str(PROJECT_ROOT / "docs")')
    text = re.sub(r'"device":\s*"cuda"', f'"device": "{device}"', text)

    CONFIG.write_text(text, encoding="utf-8")
    print(f"  ✅ 已生成 {CONFIG}")
    print(f"     - raw_documents → docs/（31 份报告已在仓库中）")
    print(f"     - device → {device}")
    print(f"     - api_key → {key_note}")

n_docx = len(list(DOCS_DIR.glob("*.docx")))
if n_docx < 31:
    fail(f"docs/ 下只找到 {n_docx} 份 .docx（应为 31 份）。"
         f"请解压 docs/31省区市政府工作报告.zip（注意 GBK 文件名，"
         f"可用 mvp/setup_and_run.py 的解压逻辑）")
print(f"  ✅ 语料就绪: docs/ 下 {n_docx} 份报告")

# ---------------------------------------------------------------- 3. 模型权重
step(3, "检查 Jina v4 模型权重")
if not (MODEL_DIR / "config.json").exists():
    fail(f"模型权重缺失: {MODEL_DIR}\n\n"
         "请先下载（约 7.5GB，只需一次）：\n\n"
         "    pip install -U huggingface_hub\n"
         "    hf download jinaai/jina-embeddings-v4 --local-dir models/jina-embeddings-v4\n\n"
         "国内网络建议走镜像：\n\n"
         "    # Linux/Mac:  export HF_ENDPOINT=https://hf-mirror.com\n"
         "    # Windows:    set HF_ENDPOINT=https://hf-mirror.com\n\n"
         "下载完成后重新运行 python setup_full.py")
print(f"  ✅ 权重就绪: {MODEL_DIR}")

# ---------------------------------------------------------------- 4. 建索引
step(4, "构建向量索引（855 块切块 → 编码 → FAISS）")
vector_index = ROOT / "data" / "vectors" / "faiss_index.bin"
if vector_index.exists():
    print("  ✅ 索引已存在，跳过构建（需要重建请运行 python rebuild_index.py）")
else:
    print(f"  开始构建（{'GPU 约 2~5 分钟' if device == 'cuda' else 'CPU 约 15~40 分钟'}）…\n")
    from rebuild_index import rebuild_index
    if not rebuild_index():
        fail("索引构建失败，请查看上方日志")

print(f"""
{'=' * 60}
🎉 完整模式部署完成！启动服务：

    python API_KIT/web_server.py

浏览器打开 http://127.0.0.1:8000
顶栏显示「完整系统 · Jina v4」即为完整模式生效。
{'=' * 60}""")
