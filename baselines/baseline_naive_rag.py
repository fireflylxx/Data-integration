"""
Naive RAG 基线：FAISS 向量检索
=================================
简单 chunk 切分 + FAISS 向量检索：将 KPI 文本向量化后检索最相似的 top-k chunks，
将检索结果作为上下文喂给 LLM 生成数据集名称。

用法:
  python baselines/baseline_naive_rag.py --chunks 00-05
  python baselines/baseline_naive_rag.py --chunks 00 --max-projects 3
  python baselines/baseline_naive_rag.py --chunks 00 --resume

输出:
  output/comparison_results/naive_rag_names_chunk_{tag}.json
"""

import json, sys, time, argparse, os, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from common_kpi import (
    resolve_chunk_dirs, discover_projects, atomic_write_json,
    llm_chat, _try_extract_name,
)

OUTPUT_DIR = BASE_DIR / "output" / "comparison_results"
KG_BASE = BASE_DIR / "output" / "kg_ontology"

NAIVE_RAG_NAMING_PROMPT = """你是一个科研数据集命名专家。请根据考核指标(KPI)和项目原文参考，生成数据集名称。

## 规则
- 名称必须包含KPI原文中的核心对象和参数
- 以"数据集"或"数据"结尾
- 名称长度15-35字
- 只输出名称，不要解释

## 输出格式
{"name_cn": "..."}"""


def load_chunks(pid: str) -> list:
    """加载项目的原文 chunks"""
    chunk_dir = KG_BASE / pid / "chunks"
    chunks = []
    if chunk_dir.exists():
        for f in sorted(chunk_dir.glob("chunk_*.txt")):
            chunks.append(f.read_text(encoding="utf-8", errors="ignore"))
    return chunks


def load_faiss_assets(pid: str):
    """加载 FAISS 向量索引和 embedder"""
    import faiss, joblib
    faiss_path = KG_BASE / pid / "faiss.index"
    pkl_path = KG_BASE / pid / "faiss.pkl"
    if not faiss_path.exists() or not pkl_path.exists():
        return None, None
    try:
        index = faiss.read_index(str(faiss_path))
        embedder = joblib.load(str(pkl_path))
        return index, embedder
    except Exception:
        return None, None


def naive_rag_name(kpi_description: str, faiss_index, embedder, chunks: list) -> str:
    """Naive RAG 版本: 向量检索 → LLM 命名"""
    # 1. 向量检索
    query_vec = embedder.encode([kpi_description]).astype("float32")
    distances, indices = faiss_index.search(query_vec, min(5, faiss_index.ntotal))

    context_parts = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk_text = chunks[idx][:500]
        context_parts.append(f"[原文参考 {i+1}] {chunk_text}")

    context = "\n\n".join(context_parts) if context_parts else ""

    # 2. 构造 prompt
    if context.strip():
        user = f"## 项目原文参考（向量检索结果）\n{context}\n\n## 考核指标\n{kpi_description}"
    else:
        user = f"## 考核指标\n{kpi_description}"

    # 3. LLM 命名
    resp = llm_chat(NAIVE_RAG_NAMING_PROMPT, user, max_tokens=800, temperature=0.3)
    name = _try_extract_name(resp)
    if name:
        return name

    # 重试
    resp2 = llm_chat(NAIVE_RAG_NAMING_PROMPT, f"考核指标: {kpi_description}",
                     max_tokens=800, temperature=0.5)
    name = _try_extract_name(resp2)
    if name:
        return name

    return "[生成失败]"


def main():
    parser = argparse.ArgumentParser(description="Naive RAG: FAISS 向量检索+LLM 命名")
    parser.add_argument("--chunks", type=str, default="00-05",
                        help="chunk 范围（默认 00-05）")
    parser.add_argument("--max-projects", type=int, default=0,
                        help="最多处理 N 个项目（调试用）")
    parser.add_argument("--resume", action="store_true",
                        help="从上次断点继续")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="并行处理 KPI 数")
    args = parser.parse_args()

    chunk_labels, chunk_dirs = resolve_chunk_dirs(args.chunks)
    chunks_tag = "_".join(chunk_labels)
    output_path = OUTPUT_DIR / f"naive_rag_names_chunk_{chunks_tag}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Chunk 范围: {chunks_tag} ({len(chunk_dirs)} 个目录)")

    # 发现项目（需要 KG 数据以读取 chunks 和 FAISS）
    all_projects = discover_projects(chunk_dirs, require_kg=True)
    print(f"发现 {len(all_projects)} 个项目（均有 KG 数据）")

    if args.max_projects:
        all_projects = all_projects[:args.max_projects]
        print(f"  (调试模式: 只处理前 {args.max_projects} 个项目)")

    # 恢复进度
    completed_pids = set()
    all_results = []
    if args.resume and output_path.exists():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        all_results = saved.get("results", [])
        completed_pids = set(saved.get("completed_pids", []))
        print(f"恢复进度: 已完成 {len(completed_pids)} 个项目, {len(all_results)} 条 KPI")

    total_projects = len(all_projects)
    t_start = time.time()

    remaining = [p for p in all_projects if p["pid"] not in completed_pids]
    if completed_pids:
        print(f"跳过 {len(completed_pids)} 个已完成的项目，剩余 {len(remaining)} 个")

    pbar = tqdm(remaining, desc="Naive RAG 生成", unit="项目", ncols=100)
    for project in pbar:
        pid = project["pid"]
        pname = project["name"]
        kpis = project["kpis"]

        if not kpis:
            completed_pids.add(pid)
            pbar.set_postfix_str(f"{pname[:20]} 无KPI")
            continue

        # 加载 FAISS 和 chunks
        chunks = load_chunks(pid)
        faiss_index, embedder = load_faiss_assets(pid)

        if not faiss_index or not embedder:
            pbar.set_postfix_str(f"{pname[:20]} 无FAISS")
            for kpi in kpis:
                all_results.append({
                    "project_id": pid, "project_name": pname, "kpi": kpi,
                    "name": "[无FAISS索引]", "success": False,
                    "n_chunks": len(chunks), "n_retrieved": 0, "time": 0,
                })
            completed_pids.add(pid)
            continue

        pbar.set_postfix_str(f"{pname[:20]} {len(chunks)}chunks — {len(kpis)}条KPI")

        # 并行处理 KPI
        def process_one(kpi_data):
            kidx, kpi = kpi_data
            t0 = time.time()
            try:
                name = naive_rag_name(kpi, faiss_index, embedder, chunks)
            except Exception as e:
                name = f"[错误:{e}]"
            elapsed = time.time() - t0
            return {
                "project_id": pid, "project_name": pname, "kpi": kpi,
                "name": name, "success": not name.startswith("["),
                "n_chunks": len(chunks), "n_retrieved": 5, "time": round(elapsed, 1),
            }

        project_results = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(process_one, (kidx, kpi)): kidx
                       for kidx, kpi in enumerate(kpis, 1)}
            kpi_pbar = tqdm(total=len(futures), desc=f"  KPI", unit="条",
                            leave=False, ncols=80)
            for future in as_completed(futures):
                try:
                    project_results.append(future.result())
                except Exception:
                    pass
                kpi_pbar.update(1)
            kpi_pbar.close()
            project_results.sort(key=lambda x: kpis.index(x["kpi"]))

        all_results.extend(project_results)
        completed_pids.add(pid)

        # 保存进度
        done = len(completed_pids)
        elapsed = time.time() - t_start
        success_count = sum(1 for r in all_results if r["success"])
        save_data = {
            "completed_pids": list(completed_pids),
            "results": all_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stats": {
                "total_projects": total_projects,
                "completed": done,
                "total_kpis": len(all_results),
                "success_count": success_count,
                "success_rate": round(success_count / max(len(all_results), 1) * 100, 1),
            },
        }
        atomic_write_json(save_data, output_path)
        pbar.set_postfix_str(f"{done}/{total_projects} {elapsed/60:.0f}分")

    # 最终保存
    success_count = sum(1 for r in all_results if r["success"])
    stats = {
        "total_projects": len(completed_pids),
        "total_kpis": len(all_results),
        "success_count": success_count,
        "success_rate": round(success_count / max(len(all_results), 1) * 100, 1),
    }
    atomic_write_json({
        "completed_pids": list(completed_pids),
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": stats,
    }, output_path)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Naive RAG 方案完成!")
    print(f"  项目数: {stats['total_projects']}")
    print(f"  KPI数: {stats['total_kpis']}")
    print(f"  成功率: {stats['success_rate']}% ({stats['success_count']}/{stats['total_kpis']})")
    print(f"  耗时: {elapsed/60:.1f} 分")
    print(f"  输出: {output_path}")


if __name__ == "__main__":
    main()
