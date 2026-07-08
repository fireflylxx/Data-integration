"""
KG Vector Store — 向量索引模块
================================
构建 Chunk → 向量索引，支持向量检索（NumPy 实现，无需 FAISS）。

支持多种嵌入策略：
  1. SentenceEmbedding: 使用 sentence-transformers（如 bge-small-zh），需联网下载模型
  2. ImprovedTFIDFEmbedding: jieba 分词 + TF-IDF + SVD（推荐，本地运行）
  3. TFIDFEmbedding: 字符级 n-gram TF-IDF + SVD（旧版，保留兼容）

架构定位（图9 阶段1）：
  Research Content → Chunks → Chunk Embedding → FAISS Vector Lib

用法:
    from kg_vector_store import create_embedder, build_vector_index, load_vector_index, vector_search
    embedder = create_embedder(chunks)
    build_vector_index(chunks, index_path, embedder)
    results = vector_search(vector_path, embedder, query, chunks, top_k=5)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional


# =============================================================================
# 改进版 TF-IDF 嵌入（jieba 分词 + TF-IDF + SVD）
# =============================================================================

class ImprovedTFIDFEmbedding:
    """改进版 TF-IDF 嵌入：使用 jieba 分词替代字符 n-gram，适合中文语义

    与 TFIDFEmbedding 接口兼容（encode / encode_single），可直接替换。
    """

    def __init__(self, corpus: Optional[List[str]] = None, dim: int = 384,
                 max_features: int = 8000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, analyzer="word",
            tokenizer=self._tokenize, token_pattern=None,
            min_df=1, sublinear_tf=True)
        self.svd = None
        self.dim = dim

        if corpus:
            self.fit(corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """jieba 分词"""
        import jieba
        return jieba.lcut(text)

    def fit(self, corpus: List[str]):
        """拟合 TF-IDF + SVD"""
        from sklearn.decomposition import TruncatedSVD
        tfidf = self.vectorizer.fit_transform(corpus)
        n_comp = min(self.dim, tfidf.shape[1] - 1, len(corpus) - 1)
        if n_comp < 1:
            n_comp = 1
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf)
        self.dim = n_comp
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        """将文本编码为向量"""
        if self.svd is None:
            raise ValueError("Embedder not fitted. Call fit() or provide corpus to constructor.")
        vec = self.svd.transform(self.vectorizer.transform(texts))
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.where(norms == 0, 1.0, norms)

    def encode_single(self, text: str) -> np.ndarray:
        """将单段文本编码为向量"""
        return self.encode([text])[0]


# =============================================================================
# Sentence Embedding（可选，需 sentence-transformers）
# =============================================================================

class SentenceEmbedding:
    """sentence-transformers 嵌入（需联网下载模型）

    接口兼容 ImprovedTFIDFEmbedding（encode / encode_single）。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        """将文本编码为向量（L2 归一化）"""
        return self.model.encode(texts, normalize_embeddings=True)

    def encode_single(self, text: str) -> np.ndarray:
        """将单段文本编码为向量"""
        return self.encode([text])[0]


def create_embedder(corpus: Optional[List[str]] = None, dim: int = 384,
                    prefer: str = "improved_tfidf") -> object:
    """创建嵌入器实例（工厂函数）

    Args:
        corpus: 用于拟合的语料（改进 TF-IDF 需要）
        dim: 向量维度
        prefer: 首选嵌入类型
            - "improved_tfidf": jieba 分词 TF-IDF（默认，推荐）
            - "sentence": sentence-transformers（如可用）
            - "legacy": 旧版 char n-gram TF-IDF

    Returns:
        嵌入器实例（具有 encode / encode_single 方法）
    """
    if prefer == "sentence":
        try:
            print("  [嵌入] 尝试 sentence-transformers...", end=" ", flush=True)
            emb = SentenceEmbedding()
            print(f"模型加载成功, dim={emb.dim}")
            return emb
        except Exception as e:
            print(f"失败 ({e})")
            print("  [嵌入] 回退到 ImprovedTFIDFEmbedding")
            prefer = "improved_tfidf"

    if prefer == "legacy":
        print(f"  [嵌入] 使用旧版 TFIDFEmbedding (char n-gram)")
        return TFIDFEmbedding(corpus=corpus, dim=dim)

    print(f"  [嵌入] 使用 ImprovedTFIDFEmbedding (jieba 分词)")
    return ImprovedTFIDFEmbedding(corpus=corpus, dim=dim)


# =============================================================================
# 旧版 TFIDFEmbedding（保留兼容）
# =============================================================================

class TFIDFEmbedding:
    """本地 TF-IDF + SVD 嵌入（字符级 n-gram，保留兼容）

    请使用 ImprovedTFIDFEmbedding（jieba 分词效果更好）。
    """

    def __init__(self, corpus: Optional[List[str]] = None, dim: int = 384):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vectorizer = TfidfVectorizer(
            max_features=5000, analyzer="char", ngram_range=(2, 4),
            min_df=1, sublinear_tf=True)
        self.svd = None
        self.dim = dim

        if corpus:
            self.fit(corpus)

    def fit(self, corpus: List[str]):
        """拟合 TF-IDF + SVD"""
        from sklearn.decomposition import TruncatedSVD
        tfidf = self.vectorizer.fit_transform(corpus)
        n_comp = min(self.dim, tfidf.shape[1] - 1, len(corpus) - 1)
        if n_comp < 1:
            n_comp = 1
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf)
        self.dim = n_comp
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        """将文本编码为向量"""
        if self.svd is None:
            raise ValueError("Embedder not fitted. Call fit() or provide corpus to constructor.")
        vec = self.svd.transform(self.vectorizer.transform(texts))
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.where(norms == 0, 1.0, norms)

    def encode_single(self, text: str) -> np.ndarray:
        """将单段文本编码为向量"""
        return self.encode([text])[0]


class NumpyVectorIndex:
    """NumPy 实现的向量索引（FAISS 替代）"""

    def __init__(self):
        self.vectors = None
        self.dimension = 0

    def build(self, vectors: np.ndarray):
        """从向量矩阵构建索引"""
        self.vectors = vectors.astype(np.float32)
        self.dimension = vectors.shape[1]

    def add(self, vectors: np.ndarray):
        """添加向量"""
        if self.vectors is None:
            self.vectors = vectors.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        self.dimension = self.vectors.shape[1]

    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        """搜索 top-k 最相似向量（内积相似度）

        Args:
            query: (1, dim) 查询向量
            k: 返回条数

        Returns:
            (distances, indices) 每个 shape (1, k)
        """
        if self.vectors is None or len(self.vectors) == 0:
            return np.array([[0.0]]), np.array([[-1]])

        # 内积 = cosine for normalized vectors
        sims = np.dot(self.vectors, query.T).flatten()
        top_k = min(k, len(sims))
        idx = np.argpartition(sims, -top_k)[-top_k:]
        idx = idx[np.argsort(-sims[idx])]
        return sims[idx].reshape(1, -1), idx.reshape(1, -1)

    @property
    def ntotal(self) -> int:
        return len(self.vectors) if self.vectors is not None else 0

    def save(self, path: Path):
        """保存到 .npz 文件"""
        np.savez_compressed(str(path), vectors=self.vectors)

    @classmethod
    def load(cls, path: Path) -> "NumpyVectorIndex":
        """从 .npz 文件加载"""
        data = np.load(str(path))
        idx = cls()
        idx.vectors = data["vectors"]
        idx.dimension = idx.vectors.shape[1]
        return idx


def build_vector_index(chunks: List[str], index_path: Path,
                       embedder=None) -> "NumpyVectorIndex":
    """构建向量索引并保存到磁盘

    Args:
        chunks: 文本块列表
        index_path: 索引文件保存路径（将使用 .npz 格式）
        embedder: 嵌入器实例（需有 encode / encode_single 方法，未提供则自动创建）

    Returns:
        NumpyVectorIndex 实例
    """
    if embedder is None:
        embedder = create_embedder(corpus=chunks)

    print(f"    构建向量索引 ({len(chunks)} 块, dim={embedder.dim})...")
    vectors = embedder.encode(chunks).astype(np.float32)

    index = NumpyVectorIndex()
    index.build(vectors)

    # 保存
    save_path = index_path.with_suffix(".npz") if not str(index_path).endswith(".npz") else index_path
    index.save(save_path)
    print(f"    向量索引已保存: {save_path} ({index.ntotal} 向量)")

    return index


def load_vector_index(index_path: Path) -> Optional["NumpyVectorIndex"]:
    """从磁盘加载向量索引"""
    # 尝试 .npz
    npz_path = index_path.with_suffix(".npz") if not str(index_path).endswith(".npz") else index_path
    if npz_path.exists():
        index = NumpyVectorIndex.load(npz_path)
        print(f"    向量索引已加载: {npz_path} ({index.ntotal} 向量)")
        return index

    # 尝试 .index (FAISS 兼容)
    if index_path.exists():
        index = NumpyVectorIndex.load(index_path)
        return index

    return None


def vector_search(index: "NumpyVectorIndex", embedder: TFIDFEmbedding,
                  query: str, chunks: List[str], top_k: int = 5) -> List[dict]:
    """向量检索：查询 → 返回 top-k 块

    Args:
        index: NumpyVectorIndex 实例
        embedder: TFIDFEmbedding 实例
        query: 查询文本
        chunks: 原始文本块列表（与建索引时顺序一致）
        top_k: 返回条数

    Returns:
        [{"chunk": str, "score": float, "rank": int}, ...]
    """
    query_vec = embedder.encode_single(query).reshape(1, -1).astype(np.float32)

    k = min(top_k, len(chunks))
    distances, indices = index.search(query_vec, k)

    results = []
    for i in range(k):
        idx = indices[0][i]
        if idx < 0 or idx >= len(chunks):
            continue
        results.append({
            "chunk": chunks[idx],
            "score": float(distances[0][i]),
            "rank": i + 1,
            "chunk_index": int(idx),
        })

    return results


def save_chunks(chunks: List[str], chunks_dir: Path):
    """保存文本块到磁盘"""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(chunks):
        (chunks_dir / f"chunk_{i:03d}.txt").write_text(c, encoding="utf-8")


def load_chunks(chunks_dir: Path) -> List[str]:
    """从磁盘加载文本块"""
    chunks = []
    for fp in sorted(chunks_dir.glob("chunk_*.txt")):
        chunks.append(fp.read_text(encoding="utf-8"))
    return chunks
