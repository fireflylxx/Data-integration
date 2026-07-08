"""
Hybrid Retriever — KG + Vector 混合检索模块
=============================================
架构定位（图9 阶段3）：
  KPI → KG索引查询(Low-level Key匹配 + 路径遍历2-hop) + Chunk索引查询(向量检索)
  → 融合: KG路径 + Chunk原文

用法:
    retriever = HybridRetriever(entities, relations, index, embedder, chunks)
    result = retriever.retrieve(kpi_object="光纤陀螺仪", kpi_parameter="零偏稳定性")
    # → {kg_paths: [...], chunks: [...], merged_context: "..."}
"""

import re
from typing import List, Optional, Tuple


class HybridRetriever:
    """KG + Vector 混合检索器"""

    def __init__(self, entities: List[dict], relations: List[dict],
                 faiss_index=None, embedder=None, chunks: Optional[List[str]] = None):
        """
        Args:
            entities: entities.json (list of {name, type, profile, chunk_ids, neighbors})
            relations: relations.json (list of {head, relation, tail, context, chunk_id})
            faiss_index: FAISS 索引 (可选)
            embedder: TFIDFEmbedding 实例 (可选)
            chunks: 原始文本块列表 (可选)
        """
        self.entities = entities
        self.relations = relations
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.chunks = chunks or []

        # 构建索引
        self._entity_by_name = {}   # name → entity dict
        self._entity_by_type = {}   # type → [entity dict]
        self._out_edges = {}        # head → [(relation, tail, context)]
        self._in_edges = {}         # tail → [(relation, head, context)]

        for e in entities:
            self._entity_by_name[e["name"]] = e
            self._entity_by_type.setdefault(e["type"], []).append(e)

        for r in relations:
            h, rel, t = r["head"], r["relation"], r["tail"]
            ctx = r.get("context", "")
            self._out_edges.setdefault(h, []).append((rel, t, ctx))
            self._in_edges.setdefault(t, []).append((rel, h, ctx))

    # =========================================================================
    # KG Path Query
    # =========================================================================

    def find_entity(self, keyword: str, entity_type: Optional[str] = None) -> List[dict]:
        """模糊匹配实体名称"""
        results = []
        keyword_lower = keyword.lower()
        for e in self.entities:
            if keyword_lower in e["name"].lower() or e["name"].lower() in keyword_lower:
                if entity_type is None or e.get("type") == entity_type:
                    results.append(e)
        return results

    def traverse_2hop(self, entity_name: str) -> List[dict]:
        """2-hop 路径遍历: OBJECT → VIA → METHOD → VERIFIES → PARAMETER

        Args:
            entity_name: 起始实体名称

        Returns:
            [{"path": [实体, 关系, 实体, 关系, 实体], "context": str, "hops": int}, ...]
        """
        paths = []

        # 1-hop: entity → relation → neighbor
        one_hop = self._out_edges.get(entity_name, [])
        for rel, neighbor, ctx in one_hop:
            # 记录 1-hop 路径
            paths.append({
                "path": [entity_name, rel, neighbor],
                "context": ctx,
                "hops": 1,
            })

            # 2-hop: neighbor → relation → neighbor2
            two_hop = self._out_edges.get(neighbor, [])
            for rel2, neighbor2, ctx2 in two_hop:
                merged_ctx = f"{ctx}；{ctx2}" if ctx and ctx2 else (ctx or ctx2 or "")
                paths.append({
                    "path": [entity_name, rel, neighbor, rel2, neighbor2],
                    "context": merged_ctx,
                    "hops": 2,
                })

        return paths

    def kg_path_query(self, kpi_object: str, kpi_parameter: str,
                      top_k: int = 10) -> List[dict]:
        """KG 路径查询

        步骤：
        1. 用 KPI 中的对象名在 KG 中匹配实体
        2. 从匹配实体出发，遍历 2-hop 路径
        3. 过滤出与参数相关的路径
        4. 按路径完整度排序

        Returns:
            [{"path": [...], "context": str, "hops": int, "score": float}, ...]
        """
        # 匹配实体
        matched_objects = self.find_entity(kpi_object, entity_type="OBJECT")
        matched_topics = self.find_entity(kpi_object, entity_type="TOPIC")
        matched_all = matched_objects + matched_topics

        if not matched_all:
            # 尝试更宽泛的匹配
            for e in self.entities:
                if e.get("type") in ("OBJECT", "TOPIC", "METHOD", "EQUIPMENT"):
                    matched_all.append(e)
            matched_all = matched_all[:5]  # 限制数量

        all_paths = []
        seen_paths = set()

        for ent in matched_all:
            paths = self.traverse_2hop(ent["name"])
            for p in paths:
                path_key = "→".join(p["path"])
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)

                # 计算分数
                score = 0.0
                if p["hops"] == 2:
                    score += 0.6  # 完整路径基础分
                else:
                    score += 0.3  # 半路径基础分

                # 如果参数名出现在路径中，加分
                if kpi_parameter and any(kpi_parameter in step for step in p["path"]):
                    score += 0.3

                # 如果有上下文，加分
                if p["context"]:
                    score += 0.1

                all_paths.append({**p, "score": round(score, 2)})

        # 按分数排序
        all_paths.sort(key=lambda x: x["score"], reverse=True)
        return all_paths[:top_k]

    # =========================================================================
    # Vector Chunk Search
    # =========================================================================

    def vector_chunk_query(self, query: str, top_k: int = 5) -> List[dict]:
        """向量检索：用查询文本搜索相似块"""
        if self.faiss_index is None or self.embedder is None or not self.chunks:
            return []

        from pipeline.kg_vector_store import vector_search
        return vector_search(self.faiss_index, self.embedder, query,
                             self.chunks, top_k=top_k)

    # =========================================================================
    # Fusion
    # =========================================================================

    def retrieve(self, kpi_object: str = "", kpi_parameter: str = "",
                 kpi_full_text: str = "", top_kg: int = 10,
                 top_chunks: int = 5) -> dict:
        """混合检索主入口

        Args:
            kpi_object: KPI 中的研究对象
            kpi_parameter: KPI 中的性能参数
            kpi_full_text: KPI 完整原文（用于向量检索）
            top_kg: KG 路径返回条数
            top_chunks: Chunk 返回条数

        Returns:
            {
                "kg_paths": [...],
                "chunks": [...],
                "merged_context": str,   # 融合后的文本（用于 LLM 规划）
                "kg_found": bool,
                "chunks_found": bool,
            }
        """
        # KG 路径查询
        kg_paths = self.kg_path_query(kpi_object, kpi_parameter, top_k=top_kg)

        # 向量检索
        query_text = kpi_full_text or f"{kpi_object} {kpi_parameter}"
        chunk_results = self.vector_chunk_query(query_text, top_k=top_chunks)

        # 融合上下文
        context_parts = []

        # KG 路径 → 文本
        for p in kg_paths[:5]:
            path_str = " → ".join(p["path"])
            ctx = p.get("context", "").strip()
            if ctx:
                context_parts.append(f"[KG路径] {path_str} | 原文: {ctx}")
            else:
                context_parts.append(f"[KG路径] {path_str}")

        # Chunk 原文
        for c in chunk_results[:3]:
            chunk_text = c["chunk"][:300]  # 截断避免过长
            context_parts.append(f"[原文] {chunk_text}")

        merged = "\n\n".join(context_parts)

        return {
            "kg_paths": kg_paths,
            "chunks": chunk_results,
            "merged_context": merged,
            "kg_found": len(kg_paths) > 0,
            "chunks_found": len(chunk_results) > 0,
        }
