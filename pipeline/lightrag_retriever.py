"""
LightRAG Retriever — 双级 KV + 向量混合检索
=============================================
模仿 LightRAG 论文的双级检索策略：
  1. Low-level: 实体名精确匹配 → 返回实体 profile
  2. High-level: 关键词组合匹配 → 返回关系描述
  3. Vector: FAISS 语义检索 → 返回原文 chunk

与 HybridRetriever 接口兼容（.retrieve() 返回 merged_context）。

用法:
    retriever = LightRAGRetriever(pid, base_dir="output/lightrag_extract_cleaned")
    context = retriever.retrieve("电池效率≥22%")
"""

import json, re
from pathlib import Path
from typing import List, Optional
import numpy as np


class LightRAGRetriever:
    """LightRAG 双级 KV + 向量检索器"""

    def __init__(self, pid: str, base_dir: str = None):
        base = Path(base_dir) if base_dir else \
            Path(__file__).resolve().parent.parent / "output" / "lightrag_extract_cleaned"
        self.pid = pid
        self.data_dir = base / pid

        # 加载 KV 索引
        self.low_level_kv = self._load_json("low_level_kv.json") or {}
        self.high_level_kv = self._load_json("high_level_kv.json") or {}
        self.entities = self._load_json("entities.json") or []
        self.relations = self._load_json("relations.json") or []
        self.chunks = self._load_chunks()

        # 实体名 → type 索引
        self._entity_types = {e["name"]: e.get("type", "") for e in self.entities}

        # 加载 FAISS
        self.faiss_index = None
        self.embedder = None
        self._load_faiss()

    def _load_json(self, filename: str) -> Optional[dict]:
        path = self.data_dir / filename
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _load_chunks(self) -> List[str]:
        chunk_dir = self.data_dir / "chunks"
        chunks = []
        if chunk_dir.exists():
            for f in sorted(chunk_dir.glob("*.txt")):
                chunks.append(f.read_text(encoding="utf-8", errors="ignore"))
        return chunks

    def _load_faiss(self):
        import faiss, joblib
        faiss_path = self.data_dir / "faiss.index"
        pkl_path = self.data_dir / "faiss.pkl"
        if faiss_path.exists() and pkl_path.exists():
            try:
                self.faiss_index = faiss.read_index(str(faiss_path))
                self.embedder = joblib.load(str(pkl_path))
            except Exception as e:
                print(f"  [FAISS加载失败] {self.pid}: {e}")

    # =========================================================================
    # Low-level 检索: 实体名精确/模糊匹配
    # =========================================================================

    def low_level_query(self, keywords: List[str]) -> List[str]:
        """low_level_kv 检索：实体名匹配 → 返回 profile"""
        results = []
        for kw in keywords:
            if not kw:
                continue
            # 精确匹配
            if kw in self.low_level_kv:
                results.append(f"[实体] {kw}: {self.low_level_kv[kw]}")
                continue
            # 模糊匹配（包含关系）
            for entity, profile in self.low_level_kv.items():
                if kw in entity or entity in kw:
                    results.append(f"[实体] {entity}: {profile}")
        return results

    # =========================================================================
    # High-level 检索: 关键词组合匹配
    # =========================================================================

    def high_level_query(self, keywords: List[str]) -> List[str]:
        """high_level_kv 检索：关键词组合匹配 → 返回关系描述"""
        results = []
        for key, desc in self.high_level_kv.items():
            score = 0
            for kw in keywords:
                if not kw:
                    continue
                if kw in key:
                    score += 1
                if kw in desc:
                    score += 0.5
            if score >= 1.5:  # 至少匹配 2 个关键词
                results.append(f"[关系] {key}: {desc}")
        return results[:10]  # 最多返回 10 条

    # =========================================================================
    # 向量检索
    # =========================================================================

    def vector_query(self, query: str, top_k: int = 3) -> List[str]:
        """FAISS 向量检索 → 返回原文 chunk"""
        if self.faiss_index is None or self.embedder is None or not self.chunks:
            return []
        try:
            vec = self.embedder.encode([query])
            D, I = self.faiss_index.search(vec.astype(np.float32), top_k)
            results = []
            for idx in I[0]:
                if 0 <= idx < len(self.chunks):
                    chunk_text = self.chunks[idx][:500]
                    results.append(f"[原文] {chunk_text}")
            return results
        except Exception as e:
            return [f"[向量检索失败: {e}]"]

    # =========================================================================
    # 实体关系路径检索（替代 HybridRetriever 的 kg_path_query）
    # =========================================================================

    def entity_relation_query(self, keywords: List[str]) -> List[str]:
        """基于实体-关系的检索：找到匹配实体 → 取出其关系路径"""
        results = []
        matched_entities = set()

        # 找到匹配的关键实体
        for kw in keywords:
            if not kw:
                continue
            for ent in self.entities:
                name = ent.get("name", "")
                if kw in name or name in kw:
                    matched_entities.add(name)
                    # 加入邻居信息
                    neighbors = ent.get("neighbors", [])
                    for nb in neighbors[:5]:  # 最多 5 个邻居
                        direction = nb.get("direction", "")
                        nb_name = nb.get("name", "")
                        rel = nb.get("relation", "")
                        arrow = "→" if direction == "out" else "←"
                        results.append(f"[路径] {name} {arrow}[{rel}]→ {nb_name}")

        return results[:10]

    # =========================================================================
    # 主检索入口
    # =========================================================================

    def retrieve(self, kpi_text: str, top_k: int = 3) -> str:
        """LightRAG 风格多级检索 → 合并上下文

        Args:
            kpi_text: KPI 描述文本（如"电池效率≥22%"）

        Returns:
            merged_context: 合并后的上下文字符串
        """
        # 1. 从 KPI 文本提取关键词
        keywords = self._extract_keywords(kpi_text)

        # 2. Low-level 检索
        low_results = self.low_level_query(keywords)

        # 3. High-level 检索
        high_results = self.high_level_query(keywords)

        # 4. 实体关系路径检索
        path_results = self.entity_relation_query(keywords)

        # 5. 向量检索
        vector_results = self.vector_query(kpi_text, top_k=top_k)

        # 6. 合并
        context_parts = []
        context_parts.extend(low_results[:8])
        context_parts.extend(high_results[:8])
        context_parts.extend(path_results[:8])
        context_parts.extend(vector_results)

        merged = "\n".join(context_parts) if context_parts else "(无检索结果)"
        return merged

    def _extract_keywords(self, text: str) -> List[str]:
        """从 KPI 文本中提取检索关键词"""
        # 去除数值和单位
        cleaned = re.sub(r'[≤≥<>=]\s*\d+\.?\d*\s*[%°℃ΩWmµnML]?', '', text)
        cleaned = re.sub(r'\d+\.?\d*', '', cleaned)
        # 按常见分隔符分词
        parts = re.split(r'[,，;；、：:\s()（）]', cleaned)
        keywords = [p.strip() for p in parts if p.strip() and len(p.strip()) >= 2]
        return keywords[:10]  # 最多 10 个关键词
