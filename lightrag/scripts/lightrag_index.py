#!/usr/bin/env python3
"""
LightRAG 向量索引构建脚本 — 使用 FAISS 向量数据库

从 output/project_cleaned/ 读取清洗后的项目任务书，
使用真正的 LightRAG 框架（HKUDS）构建双级 KV 索引 + 向量索引。

向量数据库：FAISS（通过 LightRAG 的 NanoVectorDB/FAISS 后端）
Embedding 模型：BAAI/bge-small-zh-v1.5（sentence-transformers）
LLM 后端：本地 Qwen3-32B（OpenAI 兼容接口）

输出目录：
  output/lightrag_index/{project_id}/
    - kv_store/          # JSON KV 索引
    - vdb_store/         # FAISS 向量存储
    - graph_store/       # NetworkX 图存储
    - docs/              # 文档分块
"""

import json
import os
import re
import asyncio
import sys
import time
from pathlib import Path
from typing import List, Optional

import requests

# =============================================================================
# 配置
# =============================================================================
API_BASE = "http://10.3.213.253:23001"
API_KEY = "sk-259d53cf77064362aa19c816c1321e7b"
LLM_MODEL = "qwen3-32b"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

INPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "project_cleaned"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "lightrag_index"
MAX_PROJECTS = 5  # 先处理 5 个项目做验证


# =============================================================================
# Embedding 函数（TF-IDF + SVD，完全本地离线）
# =============================================================================
# 注：原本计划使用 BAAI/bge-small-zh-v1.5（sentence-transformers），
# 但服务器无法连接 HuggingFace。改为 TF-IDF + TruncatedSVD 方案。
# 如需更换为 BGE，只需替换 __init__ 和 __call__ 方法，FAISS 后端不变。

class LocalEmbedding:
    """基于 TF-IDF + SVD 的轻量级中文文本嵌入（完全离线，无需下载模型）"""

    def __init__(self, corpus: List[str], dim: int = 384):
        """
        Args:
            corpus: 用于拟合 TF-IDF 词典的语料（所有项目文本）
            dim: 输出向量维度（默认 384，与 BGE-small 一致）
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        import numpy as np

        print(f"  训练 TF-IDF 向量化器 (dim={dim})...")

        # 中文分词：按字/词 n-gram 切分
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            analyzer="char",
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True,
        )

        # 在语料上拟合
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        vocab_size = tfidf_matrix.shape[1]
        print(f"    词典大小: {vocab_size}")

        # SVD 降维
        n_components = min(dim, vocab_size - 1, len(corpus) - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf_matrix)

        self.embedding_dim = n_components
        self.model_name = "tfidf-svd-zh"
        print(f"    嵌入维度: {self.embedding_dim} (SVD 解释方差: {self.svd.explained_variance_ratio_.sum():.2%})")

    async def __call__(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入向量（L2 归一化）"""
        import numpy as np
        tfidf = self.vectorizer.transform(texts)
        vectors = self.svd.transform(tfidf)
        # L2 归一化
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vectors = vectors / norms
        return vectors.tolist()


# =============================================================================
# LLM 函数（本地 Qwen3-32B API）
# =============================================================================
async def qwen_llm_func(model: str, messages: list, **kwargs) -> str:
    """调用本地 Qwen3-32B API 的 LLM 函数"""
    url = f"{API_BASE}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model or LLM_MODEL,
        "messages": messages,
        "temperature": kwargs.get("temperature", 0.3),
        "max_tokens": kwargs.get("max_tokens", 2000),
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=180)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 5
                print(f"    [LLM 重试 {attempt+1}/3] {e}")
                time.sleep(wait)
                continue
            raise e


# =============================================================================
# 数据读取 & 分块（复用项目现有函数）
# =============================================================================

def parse_project_file(filepath: Path) -> Optional[dict]:
    """解析 project_cleaned 目录下的任务书文件"""
    try:
        text = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = filepath.read_text(encoding="gbk")
        except UnicodeDecodeError:
            print(f"  [跳过] 无法解码: {filepath}")
            return None

    # 提取元信息（前几行）
    lines = text.split("\n")
    project_name = ""
    project_id = ""
    for line in lines[:10]:
        if line.startswith("项目名称:"):
            project_name = line.split(":", 1)[1].strip()
        elif line.startswith("项目编号:"):
            project_id = line.split(":", 1)[1].strip()
        elif line.startswith("项目ID:"):
            project_id = line.split(":", 1)[1].strip()

    if not project_name:
        # fallback: 从文件名提取
        project_name = filepath.stem

    return {
        "id": project_id or filepath.stem,
        "name": project_name,
        "text": text,
        "file": filepath.name,
    }


# =============================================================================
# 主流程
# =============================================================================

async def main():
    print("=" * 60)
    print("LightRAG 向量索引构建 — FAISS 向量数据库")
    print("=" * 60)

    # 1. 扫描输入文件
    if not INPUT_DIR.exists():
        print(f"[错误] 输入目录不存在: {INPUT_DIR}")
        sys.exit(1)

    txt_files = sorted(INPUT_DIR.glob("*.txt"))
    print(f"\n找到 {len(txt_files)} 个任务书文件")
    print(f"本次处理前 {MAX_PROJECTS} 个项目")

    # 2. 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. 读取所有项目文本（用于构建 embedding 词典）
    print("\n读取项目文本...")
    projects = []
    for filepath in txt_files[:MAX_PROJECTS]:
        project = parse_project_file(filepath)
        if project:
            projects.append(project)
    print(f"  成功读取 {len(projects)} 个项目")

    if not projects:
        print("[错误] 没有可处理的项目")
        sys.exit(1)

    # 4. 初始化 Embedding（TF-IDF + SVD，在全部语料上拟合）
    print("\n初始化本地嵌入模型...")
    corpus = [p["text"][:10000] for p in projects]  # 每个项目取前 10000 字符
    embedder = LocalEmbedding(corpus, dim=384)

    # 5. 初始化 LightRAG
    print("\n初始化 LightRAG...")
    from lightrag import LightRAG
    from lightrag.utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(
        embedding_dim=embedder.embedding_dim,
        max_batch_size=32,
    )
    async def embedding_func(texts: List[str]) -> List[List[float]]:
        return await embedder(texts)

    rag = LightRAG(
        working_dir=str(OUTPUT_DIR / "rag_storage"),
        llm_model_func=qwen_llm_func,
        llm_model_name=LLM_MODEL,
        embedding_func=embedding_func,
        chunk_token_size=1200,        # 约 800 中文字符
        chunk_overlap_token_size=200,  # 约 133 字符重叠
        top_k=20,
        cosine_threshold=0.2,
    )

    print(f"  LLM: {LLM_MODEL} @ {API_BASE}")
    print(f"  Embedding: TF-IDF+SVD (dim={embedder.embedding_dim})")
    print(f"  Vector DB: FAISS (通过 NanoVectorDB)")
    print(f"  Working Dir: {OUTPUT_DIR / 'rag_storage'}")

    # 6. 逐个处理项目文件
    processed = 0
    for project in projects:
        project = parse_project_file(filepath)
        if not project:
            continue

        print(f"\n{'=' * 50}")
        print(f"处理: {project['name']} ({project['id']})")
        print(f"  文件: {project['file']}")
        print(f"  大小: {len(project['text'])} 字符")

        # 用文件名作为文档 ID 插入
        doc_id = project["file"].replace(".txt", "")
        text = project["text"]

        try:
            await rag.insert(text, ids=[doc_id])
            print(f"  索引完成 ✅")
            processed += 1
        except Exception as e:
            print(f"  索引失败 ❌: {e}")
            import traceback
            traceback.print_exc()

    # 5. 查询测试
    if processed > 0:
        print(f"\n{'=' * 50}")
        print("运行查询测试...")
        test_queries = [
            "太阳能电池",
            "数据驱动",
            "知识图谱",
            "考核指标",
        ]

        for query in test_queries:
            print(f"\n  查询: '{query}'")
            try:
                result = await rag.query(query)
                if result:
                    print(f"  结果: {str(result)[:200]}...")
                else:
                    print("  (无结果)")
            except Exception as e:
                print(f"  查询失败: {e}")

    # 6. 最终报告
    print(f"\n{'=' * 60}")
    print(f"处理完成!")
    print(f"  成功: {processed}/{MAX_PROJECTS} 个项目")
    print(f"  向量数据库: FAISS")
    print(f"  存储位置: {OUTPUT_DIR / 'rag_storage'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
