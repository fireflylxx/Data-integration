"""
Phase 3: 3D 评估对比（多方案版）
=================================
读取多个方案的命名结果（如 simple、kg、rule、bm25、direct），
对每个 KPI 调用 evaluate_name() 做三维度评估
（语义保真度 50% + 原文可溯性 35% + 命名规范性 15%），
输出对比结果和汇总统计。

用法:
  # 默认：Simple vs KG
  python scripts/evaluate/batch_evaluate_3d.py

  # 多方案对比
  python scripts/evaluate/batch_evaluate_3d.py --methods rule bm25 direct simple kg

  # 指定 chunk
  python scripts/evaluate/batch_evaluate_3d.py --methods simple kg --chunks 00

  # 调试
  python scripts/evaluate/batch_evaluate_3d.py --methods rule bm25 --max-pairs 10
"""
import json, sys, time, argparse, os, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm

# Windows GBK console workaround
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "pipeline"))
from evaluator_3d_refined import evaluate_name

INPUT_DIR = BASE_DIR / "output" / "comparison_results"
KG_BASE = BASE_DIR / "output" / "kg_ontology"


# =============================================================================
# 原子化写入
# =============================================================================
def atomic_write_json(data, path: Path):
    tmp = path.with_suffix(".tmp." + path.name)
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# =============================================================================
# 加载项目的 chunks 和向量索引
# =============================================================================
def load_project_assets(pid: str):
    """加载项目的原文块和向量检索器"""
    kg_dir = KG_BASE / pid
    chunks = []
    index = None
    embedder = None

    chunk_dir = kg_dir / "chunks"
    if chunk_dir.exists():
        for f in sorted(chunk_dir.glob("chunk_*.txt")):
            chunks.append(f.read_text(encoding="utf-8", errors="ignore"))

    faiss_path = kg_dir / "faiss.index"
    pkl_path = kg_dir / "faiss.pkl"
    if faiss_path.exists() and pkl_path.exists() and chunks:
        try:
            import faiss, joblib
            index = faiss.read_index(str(faiss_path))
            embedder = joblib.load(str(pkl_path))
        except Exception as e:
            print(f"    向量索引加载失败: {e}")

    return chunks, index, embedder


# =============================================================================
# 加载多方案结果文件
# =============================================================================
def load_methods_results(methods: list, chunks_tag: str):
    """加载多个方案的 results，返回 {method: [entries]}"""
    method_data = {}
    for method in methods:
        path = INPUT_DIR / f"{method}_names_chunk_{chunks_tag}.json"
        if not path.exists():
            # 尝试无 chunk 后缀
            path = INPUT_DIR / f"{method}_names.json"
        if not path.exists():
            print(f"[警告] 找不到 {method} 结果: {path}")
            method_data[method] = []
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        results = data.get("results", [])
        method_data[method] = results
        print(f"  {method}: {len(results)} 条 ({path.name})")

    return method_data


def build_kpi_index(method_data: dict) -> list:
    """跨方案按 (project_id, kpi) 对齐，返回 KPI 对齐列表

    Returns:
        [{"project_id": str, "project_name": str, "kpi": str,
          "methods": {method: entry}}, ...]
    """
    # 以第一个有数据的方法为基准
    active_methods = [m for m, data in method_data.items() if data]
    if not active_methods:
        return []

    # 建立索引: (pid, kpi) → {method: entry}
    index = defaultdict(dict)
    project_names = {}

    for method in active_methods:
        for entry in method_data[method]:
            key = (entry["project_id"], entry["kpi"].strip())
            index[key][method] = entry
            project_names[entry["project_id"]] = entry.get("project_name", "")

    # 转为列表（只保留所有方案都有数据的 KPI）
    aligned = []
    for (pid, kpi), method_entries in index.items():
        # 要求所有 active_methods 都有数据
        if len(method_entries) >= 1:  # 至少一个方案有数据
            aligned.append({
                "project_id": pid,
                "project_name": project_names.get(pid, ""),
                "kpi": kpi,
                "methods": method_entries,
            })

    return aligned


# =============================================================================
# 评估单个 KPI 的所有方案
# =============================================================================
def evaluate_method(method: str, entry: dict, chunks: list,
                    index, embedder, all_names: list = None) -> dict:
    """对单个方案的单个 KPI 做 3D 评估"""
    name = entry.get("name", "")

    if name.startswith("["):
        return {
            "name": name, "total": 0.0,
            "semantic": 0.0, "traceability": 0.0, "regularity": 0.0,
            "object_score": 0.0, "parameter_score": 0.0, "method_score": 0.0,
            "diagnosis": "生成失败",
        }

    try:
        r = evaluate_name(name, entry.get("kpi", ""), chunks=chunks,
                          index=index, embedder=embedder, all_names=all_names)
        return {
            "name": name,
            "total": round(r.total_score, 4),
            "semantic": round(r.semantic_total, 4),
            "traceability": round(r.traceability_total, 4),
            "regularity": round(r.regularity_total, 4),
            "object_score": round(r.object_score, 4),
            "parameter_score": round(r.parameter_score, 4),
            "method_score": round(r.method_score, 4),
            "diagnosis": r.diagnosis,
        }
    except Exception as e:
        print(f"    评估 {method} 失败: {e}")
        return {"name": name, "total": 0.0, "diagnosis": f"评估异常: {e}"}


def evaluate_kpi(aligned_kpi: dict, chunks: list,
                 index, embedder, all_names: list = None) -> dict:
    """对一个 KPI 的所有方案做 3D 评估"""
    methods_eval = {}
    for method, entry in aligned_kpi["methods"].items():
        methods_eval[method] = evaluate_method(method, entry, chunks,
                                               index, embedder, all_names)

    # 找出最佳方案
    best_method = max(methods_eval, key=lambda m: methods_eval[m].get("total", 0))

    return {
        "project_id": aligned_kpi["project_id"],
        "project_name": aligned_kpi["project_name"],
        "kpi": aligned_kpi["kpi"],
        "methods": methods_eval,
        "best_method": best_method,
    }


# =============================================================================
# 汇聚统计
# =============================================================================
def build_summary(all_pairs: list, methods: list):
    """从评估结果列表生成汇总统计"""
    n = len(all_pairs)
    if n == 0:
        return {"total_pairs": 0}

    active_methods = [m for m in methods if any(m in p["methods"] for p in all_pairs)]

    # 各方案分维度统计
    method_stats = {}
    for method in active_methods:
        totals = [p["methods"][method]["total"] for p in all_pairs
                  if method in p["methods"]]
        if not totals:
            continue
        semantics = [p["methods"][method].get("semantic", 0) for p in all_pairs
                     if method in p["methods"]]
        traces = [p["methods"][method].get("traceability", 0) for p in all_pairs
                  if method in p["methods"]]
        regs = [p["methods"][method].get("regularity", 0) for p in all_pairs
                if method in p["methods"]]
        fails = sum(1 for p in all_pairs if method in p["methods"]
                    and p["methods"][method].get("diagnosis") == "生成失败")

        method_stats[method] = {
            "avg_total": round(sum(totals) / len(totals), 4),
            "avg_semantic": round(sum(semantics) / len(semantics), 4),
            "avg_traceability": round(sum(traces) / len(traces), 4),
            "avg_regularity": round(sum(regs) / len(regs), 4),
            "fail_count": fails,
            "fail_rate": round(fails / len(totals) * 100, 1),
        }

    # 胜出统计
    win_counts = defaultdict(int)
    tie_count = 0
    for p in all_pairs:
        best = p.get("best_method", "")
        if best:
            win_counts[best] += 1

    # 诊断分布
    diag_dist = defaultdict(lambda: defaultdict(int))
    for p in all_pairs:
        for method in active_methods:
            if method in p["methods"]:
                diag = p["methods"][method].get("diagnosis", "")
                diag_dist[method][diag] += 1

    return {
        "total_pairs": n,
        "total_projects": len(set(p["project_id"] for p in all_pairs)),
        "methods": dict(method_stats),
        "win_counts": dict(win_counts),
        "best_method": max(win_counts, key=win_counts.get) if win_counts else "",
        "diagnosis_dist": {m: dict(d) for m, d in diag_dist.items()},
        "examples": _build_examples(all_pairs, active_methods),
    }


def _build_examples(all_pairs, methods):
    """构造示例"""
    examples = {}
    for method in methods:
        # 找该方案最好和最差的
        scored = [(p, p["methods"][method]["total"])
                  for p in all_pairs if method in p["methods"]]
        scored.sort(key=lambda x: -x[1])
        examples[f"{method}最佳"] = [
            {"kpi": p["kpi"][:60], "name": p["methods"][method]["name"],
             "total": s} for p, s in scored[:2]
        ]
        scored_asc = sorted(scored, key=lambda x: x[1])
        examples[f"{method}最差"] = [
            {"kpi": p["kpi"][:60], "name": p["methods"][method]["name"],
             "total": s} for p, s in scored_asc[:2]
        ]

    # 方案间显著差异
    if len(methods) >= 2:
        deltas = []
        for p in all_pairs:
            m_totals = {m: p["methods"][m]["total"] for m in methods if m in p["methods"]}
            if len(m_totals) >= 2:
                max_m = max(m_totals, key=m_totals.get)
                min_m = min(m_totals, key=m_totals.get)
                delta = m_totals[max_m] - m_totals[min_m]
                deltas.append((p, delta, max_m, min_m))
        deltas.sort(key=lambda x: -x[1])
        examples["最大差异"] = [
            {"kpi": p["kpi"][:60], "best": best, "worst": worst, "delta": round(d, 4)}
            for p, d, best, worst in deltas[:3]
        ]

    return examples


# =============================================================================
# 格式化文本报告
# =============================================================================
def save_text_report(summary: dict, path: Path, chunks_tag: str, methods: list):
    """生成可读的 Markdown 评估报告"""
    lines = []
    lines.append("# 多方案 3D 评估对比报告")
    lines.append("")
    lines.append(f"**Chunk 范围**: {chunks_tag}")
    lines.append(f"**评估 KPI 数**: {summary.get('total_pairs', 0)}")
    lines.append(f"**项目数**: {summary.get('total_projects', 0)}")
    lines.append(f"**方案数**: {len(methods)} ({', '.join(methods)})")
    lines.append(f"**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 综合对比表
    lines.append("## 综合对比")
    lines.append("")
    header = "| 方案 | 综合总分 | 语义保真度(50%) | 原文可溯性(35%) | 命名规范性(15%) | 生成失败率 |"
    sep = "|------|---------|----------------|----------------|----------------|----------|"
    lines.append(header)
    lines.append(sep)
    ms = summary.get("methods", {})
    for method in methods:
        if method not in ms:
            continue
        s = ms[method]
        lines.append(
            f"| {method} | {s['avg_total']:.4f} | {s['avg_semantic']:.4f} | "
            f"{s['avg_traceability']:.4f} | {s['avg_regularity']:.4f} | "
            f"{s['fail_rate']}%({s['fail_count']}) |"
        )
    lines.append("")

    # 胜出统计
    lines.append("## 胜出统计")
    lines.append("")
    wc = summary.get("win_counts", {})
    total = summary.get("total_pairs", 1)
    for method, count in sorted(wc.items(), key=lambda x: -x[1]):
        lines.append(f"- **{method}**: {count}/{total} ({count/total*100:.1f}%)")
    lines.append("")

    # 诊断分布
    lines.append("## 诊断分布")
    lines.append("")
    for method in methods:
        diag_d = summary.get("diagnosis_dist", {}).get(method, {})
        if diag_d:
            lines.append(f"### {method}")
            for diag, count in sorted(diag_d.items(), key=lambda x: -x[1]):
                lines.append(f"- {diag}: {count}")
            lines.append("")

    # 示例
    lines.append("## 示例")
    for category, examples in summary.get("examples", {}).items():
        if examples:
            lines.append(f"")
            lines.append(f"### {category}")
            for ex in examples[:3]:
                lines.append(f"- KPI: {ex.get('kpi', '')}")
                if "name" in ex:
                    lines.append(f"  名称: {ex['name']}")
                if "total" in ex:
                    lines.append(f"  总分: {ex['total']}")
                if "best" in ex:
                    lines.append(f"  最佳: {ex['best']} vs 最差: {ex['worst']}")
                if "delta" in ex:
                    lines.append(f"  Δ: {ex['delta']}")

    report = "\n".join(lines)
    path.write_text(report, encoding="utf-8")
    print(f"  文本报告: {path}")


# =============================================================================
# 主流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 3: 多方案 3D 评估对比")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["simple", "kg"],
                        help="要评估的方案列表（默认 simple kg）")
    parser.add_argument("--chunks", type=str, default=None,
                        help="chunk 标签，如 '00' / '00_01_02_03_04_05'")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="最多评估 N 条 KPI（调试用）")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="并行评估数")
    parser.add_argument("--resume", action="store_true",
                        help="从上次断点继续")
    args = parser.parse_args()

    methods = args.methods
    chunks_tag = args.chunks or "00_01_02_03_04_05"

    results_path = INPUT_DIR / f"eval_3d_results_chunk_{chunks_tag}.json"
    summary_path = INPUT_DIR / f"eval_3d_summary_chunk_{chunks_tag}.json"

    print("=" * 60)
    print("Phase 3: 三维度评估对比（多方案版）")
    print(f"公式: Score = 0.50×语义保真度 + 0.35×原文可溯性 + 0.15×命名规范性")
    print(f"方案: {', '.join(methods)}")
    print(f"Chunk: {chunks_tag}")
    print("=" * 60)

    # 加载各方案结果
    print("\n加载结果文件...")
    method_data = load_methods_results(methods, chunks_tag)

    # 对齐 KPI
    aligned = build_kpi_index(method_data)
    print(f"\n对齐 KPI: {len(aligned)} 条")

    if args.max_pairs:
        aligned = aligned[:args.max_pairs]
        print(f"  (调试模式: 只评估前 {args.max_pairs} 条)")

    if not aligned:
        print("[错误] 没有可评估的 KPI")
        return

    # 按项目分组
    project_kpis = defaultdict(list)
    for item in aligned:
        project_kpis[item["project_id"]].append(item)

    print(f"\n开始评估 {len(aligned)} 条 KPI ({len(project_kpis)} 个项目)...")

    # 断点续跑
    all_results = []
    completed_projects = 0
    total_projects = len(project_kpis)
    if args.resume and results_path.exists():
        saved = json.loads(results_path.read_text(encoding="utf-8"))
        saved_results = saved.get("results", [])
        completed_ids = set(r["project_id"] for r in saved_results)
        for cid in list(project_kpis.keys()):
            if cid in completed_ids:
                del project_kpis[cid]
        all_results = saved_results
        completed_projects = saved.get("completed_projects", 0)
        print(f"恢复进度: 跳过 {len(completed_ids)} 个已完成项目，剩余 {len(project_kpis)} 个")

    t_start = time.time()
    project_items = list(project_kpis.items())
    pbar = tqdm(project_items, desc="3D 评估", unit="项目", ncols=100)
    for pid, kpi_list in pbar:
        chunks, index, embedder = load_project_assets(pid)

        # 收集该项目的所有名称（用于规范性唯一性检查）
        all_names = []
        for item in kpi_list:
            for m in methods:
                entry = item["methods"].get(m)
                if entry:
                    all_names.append(entry.get("name", ""))

        pbar.set_postfix_str(f"{pid[-6:]} {len(kpi_list)}条KPI")

        # 并行评估
        def eval_one(item):
            return evaluate_kpi(item, chunks, index, embedder, all_names)

        kpi_results = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(eval_one, item): idx
                       for idx, item in enumerate(kpi_list)}
            kpi_pbar = tqdm(total=len(futures), desc=f"  KPI", unit="条",
                            leave=False, ncols=80)
            for future in as_completed(futures):
                try:
                    kpi_results.append(future.result())
                except Exception as e:
                    pass
                kpi_pbar.update(1)
            kpi_pbar.close()

        kpi_results.sort(key=lambda r: [i["kpi"] for i in kpi_list].index(r["kpi"]))
        all_results.extend(kpi_results)

        completed_projects += 1

        # 每完成一个项目保存
        elapsed = time.time() - t_start
        summary = build_summary(all_results, methods)
        save_data = {
            "completed_projects": completed_projects,
            "total_projects": total_projects,
            "results": all_results,
            "summary": summary,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_write_json(save_data, results_path)
        save_text_report(summary, results_path.with_suffix(".md"), chunks_tag, methods)

    # 最终保存
    summary = build_summary(all_results, methods)
    save_data = {
        "completed_projects": completed_projects,
        "total_projects": total_projects,
        "results": all_results,
        "summary": summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(save_data, results_path)
    atomic_write_json(summary, summary_path)
    save_text_report(summary, results_path.with_suffix(".md"), chunks_tag, methods)

    # 打印汇总
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"3D 评估完成! 耗时: {elapsed/60:.1f} 分")
    print(f"{'=' * 60}")
    print(f"  评估 KPI 数:   {summary['total_pairs']}")
    print(f"  项目数:        {summary['total_projects']}")
    print(f"  最佳方案:      {summary.get('best_method', 'N/A')}")
    print(f"")
    ms = summary.get("methods", {})
    for method in methods:
        if method not in ms:
            continue
        s = ms[method]
        print(f"  {method}:")
        print(f"    总分:      {s['avg_total']:.4f}")
        print(f"    语义:      {s['avg_semantic']:.4f}")
        print(f"    可溯性:    {s['avg_traceability']:.4f}")
        print(f"    规范性:    {s['avg_regularity']:.4f}")
        print(f"    失败率:    {s['fail_rate']}% ({s['fail_count']})")
        print(f"")
    print(f"  胜出统计:")
    for method, count in sorted(summary.get("win_counts", {}).items(),
                                key=lambda x: -x[1]):
        total_kpis = summary["total_pairs"]
        print(f"    {method}: {count}/{total_kpis} ({count/max(total_kpis,1)*100:.1f}%)")
    print(f"")
    print(f"输出文件:")
    print(f"  {results_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
