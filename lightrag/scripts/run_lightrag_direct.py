#!/usr/bin/env python3
"""
用 LightRAG 库直接处理 project_cleaned 中的项目任务书。

流程：
  1. 初始化 LightRAG（Qwen3-32B LLM + BGE Embedding）
  2. 读取 5 个项目文件
  3. 逐个 insert → 自动分块/实体抽取/关系抽取/图构建/向量索引
  4. 查询测试

输出目录：output/lightrag_kg/
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import requests
import numpy as np

# 修复 Windows GBK 编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

# =============================================================================
# 配置
# =============================================================================
API_BASE = "  "
API_KEY = " "
LLM_MODEL = " "
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "output" / "project_cleaned"
OUTPUT_DIR = PROJECT_ROOT / "output" / "lightrag_kg"
MAX_PROJECTS = 5


# =============================================================================
# Embedding 函数（BGE 中文模型）
# =============================================================================
class BgeEmbedder:
    """基于 sentence-transformers 的 BGE 中文嵌入"""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"  加载嵌入模型 {model_name}...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.model_name = model_name
        print(f"    嵌入维度: {self.embedding_dim}")

    async def __call__(self, texts: List[str]) -> "np.ndarray":
        """返回 numpy 数组（LightRAG 需要 .size 属性）"""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        emb = self.model.encode(texts, normalize_embeddings=True)
        return emb.astype(np.float32)


# =============================================================================
# LLM 函数（本地 Qwen3-32B）
# =============================================================================
async def qwen_llm(*args, **kwargs) -> str:
    """兼容 LightRAG 各种调用模式的 LLM 函数"""
    # 解析 model 和 messages（兼容位置参数和关键字参数）
    model = kwargs.get("model") or LLM_MODEL
    messages = kwargs.get("messages")
    if not messages and len(args) >= 1:
        # 可能是第一个位置参数是 messages
        messages = args[0]
    if not messages and len(args) >= 2:
        messages = args[1]
    if not messages:
        # 从 kwargs 的其他可能位置查找
        messages = kwargs.get("input") or kwargs.get("prompt") or []
    if not isinstance(messages, list):
        messages = [{"role": "user", "content": str(messages)}]

    url = f"{API_BASE}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": str(model or LLM_MODEL),
        "messages": messages,
        "temperature": float(kwargs.get("temperature", 0.3)),
        "max_tokens": int(kwargs.get("max_tokens", 4096)),
    }
    # 转发 response_format（JSON 模式需要）
    # 升级 json_object → json_schema 以强制包含 description 字段
    if "response_format" in kwargs:
        rf = kwargs["response_format"]
        if isinstance(rf, dict) and rf.get("type") == "json_object":
            # 使用 json_schema 强制要求 description 字段
            data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "entities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "type": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["name", "type", "description"],
                                    "additionalProperties": False,
                                },
                            },
                            "relationships": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source": {"type": "string"},
                                        "target": {"type": "string"},
                                        "keywords": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["source", "target", "keywords", "description"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["entities", "relationships"],
                        "additionalProperties": False,
                    },
                },
            }
        else:
            data["response_format"] = rf
    # 转发 stop 参数
    if "stop" in kwargs:
        data["stop"] = kwargs["stop"]

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=300)
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"].strip()
            # 去除 Qwen 的 <think> 思考标签
            if "<think>" in result:
                # 提取 </think> 之后的内容
                parts = result.split("</think>", 1)
                if len(parts) > 1:
                    result = parts[1].strip()
                else:
                    # 有 <think> 但没 </think>，去掉 <think> 之后的内容
                    result = result.split("<think>")[0].strip()
            return result
        except Exception as e:
            if attempt < 2:
                print(f"    [LLM重试] {e}")
                time.sleep((attempt + 1) * 5)
                continue
            raise


# =============================================================================
# 数据读取
# =============================================================================
def parse_project(filepath: Path) -> Optional[dict]:
    try:
        text = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = filepath.read_text(encoding="gbk")
        except:
            return None
    lines = text.split("\n")
    name, pid = "", ""
    for line in lines[:10]:
        if line.startswith("项目名称:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("项目ID:"):
            pid = line.split(":", 1)[1].strip()
        elif line.startswith("项目编号:"):
            pid = pid or line.split(":", 1)[1].strip()
    return {"id": pid or filepath.stem, "name": name or filepath.stem,
            "text": text, "file": filepath.name}


# =============================================================================
# 主流程
# =============================================================================
async def main():
    print("=" * 60)
    print("LightRAG 库直接抽取 — project_cleaned")
    print("=" * 60)

    # 1. 读取项目文件
    if not INPUT_DIR.exists():
        print(f"[错误] 输入目录不存在: {INPUT_DIR}")
        sys.exit(1)

    files = sorted(INPUT_DIR.glob("*.txt"))[:MAX_PROJECTS]
    print(f"\n读取 {len(files)} 个项目文件...")

    projects = []
    for fp in files:
        p = parse_project(fp)
        if p:
            projects.append(p)
            print(f"  {p['id']}: {p['name']} ({len(p['text'])}字)")

    if not projects:
        print("[错误] 未读取到任何项目")
        sys.exit(1)

    # 2. 初始化 Embedding（BGE 中文模型）
    print("\n--- 初始化 Embedding ---")
    embedder = BgeEmbedder()

    from lightrag.utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(
        embedding_dim=embedder.embedding_dim,
    )
    async def embedding_func(texts: List[str]) -> np.ndarray:
        return await embedder(texts)

    # 3. 初始化 LightRAG
    print("\n--- 初始化 LightRAG ---")
    import os
    os.environ["ENTITY_EXTRACTION_USE_JSON"] = "true"  # 强制 JSON 模式
    from lightrag import LightRAG

    rag = LightRAG(
        working_dir=str(OUTPUT_DIR / "_rag_storage"),
        llm_model_func=qwen_llm,
        llm_model_name=LLM_MODEL,
        embedding_func=embedding_func,
        chunk_token_size=800,
        chunk_overlap_token_size=100,
        top_k=20,
        cosine_threshold=0.2,
        entity_extraction_use_json=True,
        addon_params={
            "language": "Chinese",  # 中文项目任务书，指定中文提取
        },
    )
    print(f"  LLM: {LLM_MODEL}")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={embedder.embedding_dim})")
    print(f"  JSON模式: {rag.entity_extraction_use_json}")
    # 验证 global_config
    gc = rag._build_global_config()
    print(f"  language: {gc.get('_resolved_summary_language')}")
    print(f"  global_config entity_extraction_use_json: {gc.get('entity_extraction_use_json')}")
    print(f"  输出: {OUTPUT_DIR}")

    # 4. 初始化存储
    print("\n--- 初始化存储 ---")
    try:
        await rag.initialize_storages()
        print("  存储初始化完成")
    except Exception as e:
        print(f"  存储初始化警告: {e}")

    # 5. 逐个插入项目（LightRAG 自动建图）
    print("\n--- 插入项目 ---")
    for proj in projects:
        print(f"\n  处理: {proj['name']}")
        try:
            await rag.ainsert(proj["text"])
            print(f"  [完成]")
        except Exception as e:
            print(f"  [失败] {e}")

    # 5. 查询测试
    print("\n--- 查询测试 ---")
    test_queries = ["柔性铜铟镓硒太阳能电池", "生物纤维塑料注射成型", "知识图谱"]
    for q in test_queries:
        print(f"\n  查询: '{q}'")
        try:
            result = await rag.aquery(q)
            print(f"  结果: {str(result)[:300]}")
        except Exception as e:
            print(f"  查询失败: {e}")

    # 6. 查看 LightRAG 存储的内容
    print("\n--- 存储内容 ---")
    storage = OUTPUT_DIR / "_rag_storage"
    if storage.exists():
        for item in sorted(storage.iterdir()):
            if item.is_dir():
                print(f"  [DIR] {item.name}/")
                for f in sorted(item.iterdir())[:5]:
                    if f.is_file() and f.stat().st_size < 100000:
                        print(f"    {f.name} ({f.stat().st_size} bytes)")
            elif item.is_file():
                print(f"  [FILE] {item.name} ({item.stat().st_size} bytes)")

    print(f"\n{'='*60}")
    print("完成。")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
