"""
单套方案 3D 评估
==================
对单个结果文件（如 Simple+全文）进行三维度评估，输出评分。

用法:
  python scripts/evaluate/batch_evaluate_single.py
  python scripts/evaluate/batch_evaluate_single.py --input simple_fulltext_names_chunk_00_01_02_03_04_05.json
  python scripts/evaluate/batch_evaluate_single.py --concurrency 10

输出:
  output/comparison_results/eval_single_{tag}.json   # 每条 KPI 详细评分
  output/comparison_results/eval_single_{tag}.md     # 汇总报告
"""
import json, sys, time, argparse, os, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "pipeline"))
from evaluator_3d_refined import evaluate_name

INPUT_DIR = BASE_DIR / "output" / "comparison_results"
KG_BASE = BASE_DIR / "output" / "kg_ontology"


def atomic_write_json(data, path: Path):
    tmp = path.with_suffix(".tmp." + path.name)
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_project_assets(pid: str):
    """加载项目的 chunks 和向量索引"""
    chunks_dir = KG_BASE / pid / "chunks"
    chunks = []
    if chunks_dir.exists():
        for f in sorted(chunks_dir.glob("chunk_*.txt")):
            chunks.append(f.read_text(encoding="utf-8", errors="ignore"))

    index = None
    embedder = None
    faiss_path = KG_BASE / pid / "faiss.index"
    pkl_path = KG_BASE / pid / "embedder.pkl"
    if faiss_path.exists() and pkl_path.exists():
        try:
            import faiss, joblib
            index = faiss.read_index(str(faiss_path))
            embedder = joblib.load(str(pkl_path))
        except Exception:
            pass
    return chunks, index, embedder


def evaluate_one(entry: dict, chunks: list, index, embedder, all_names: list) -> dict:
    """评估单条 KPI 的 3D 分数"""
    kpi = entry.get("kpi", "")
    name_str = entry.get("name", "")

    if name_str.startswith("["):
        return {
            "project_id": entry["project_id"],
            "project_name": entry.get("project_name", ""),
            "kpi": kpi,
            "name": name_str,
            "total": 0.0,
            "semantic": 0.0,
            "traceability": 0.0,
            "regularity": 0.0,
            "object_score": 0.0,
            "parameter_score": 0.0,
            "method_score": 0.0,
            "diagnosis": "生成失败",
        }

    try:
        sr = evaluate_name(name_str, kpi, chunks=chunks, index=index,
                           embedder=embedder, all_names=all_names)
        return {
            "project_id": entry["project_id"],
            "project_name": entry.get("project_name", ""),
            "kpi": kpi,
            "name": name_str,
            "total": round(sr.total_score, 4),
            "semantic": round(sr.semantic_total, 4),
            "traceability": round(sr.traceability_total, 4),
            "regularity": round(sr.regularity_total, 4),
            "object_score": round(sr.object_score, 4),
            "parameter_score": round(sr.parameter_score, 4),
            "method_score": round(sr.method_score, 4),
            "diagnosis": sr.diagnosis,
        }
    except Exception as e:
        return {
            "project_id": entry["project_id"],
            "project_name": entry.get("project_name", ""),
            "kpi": kpi,
            "name": name_str,
            "total": 0.0,
            "diagnosis": f"评估异常: {e}",
        }


def build_summary(results: list, label: str) -> dict:
    """汇总统计"""
    n = len(results)
    if n == 0:
        return {"total": 0}

    totals = [r["total"] for r in results]
    semantics = [r.get("semantic", 0) for r in results]
    traces = [r.get("traceability", 0) for r in results]
    regs = [r.get("regularity", 0) for r in results]
    fails = sum(1 for r in results if r.get("diagnosis") == "生成失败")

    diag_dist = defaultdict(int)
    for r in results:
        diag_dist[r.get("diagnosis", "")] += 1

    return {
        "label": label,
        "total_pairs": n,
        "total_projects": len(set(r["project_id"] for r in results)),
        "avg_total": round(sum(totals) / n, 4),
        "avg_semantic": round(sum(semantics) / n, 4),
        "avg_traceability": round(sum(traces) / n, 4),
        "avg_regularity": round(sum(regs) / n, 4),
        "fail_count": fails,
        "fail_rate": round(fails / n * 100, 1),
        "diagnosis_dist": dict(diag_dist),
    }


def save_report(summary: dict, results: list, path: Path):
    """保存 Markdown 格式报告"""
    lines = []
    lines.append("# 3D 评估报告")
    lines.append("")
    lines.append(f"**方案**: {summary.get('label', '')}")
    lines.append(f"**评估 KPI 数**: {summary['total_pairs']}")
    lines.append(f"**项目数**: {summary['total_projects']}")
    lines.append(f"**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 综合评分")
    lines.append("")
    lines.append(f"| 维度 | 分数 |")
    lines.append(f"|------|------|")
    lines.append(f"| 综合总分 | {summary['avg_total']:.4f} |")
    lines.append(f"| 语义保真度 (50%) | {summary['avg_semantic']:.4f} |")
    lines.append(f"| 原文可溯性 (35%) | {summary['avg_traceability']:.4f} |")
    lines.append(f"| 命名规范性 (15%) | {summary['avg_regularity']:.4f} |")
    lines.append("")
    lines.append("## 生成情况")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 生成失败数 | {summary['fail_count']} |")
    lines.append(f"| 生成失败率 | {summary['fail_rate']}% |")
    lines.append("")
    lines.append("## 诊断分布")
    for diag, count in summary.get("diagnosis_dist", {}).items():
        lines.append(f"- {diag}: {count}")
    lines.append("")
    lines.append("## 高分示例（Top 5）")
    for r in sorted(results, key=lambda x: -x["total"])[:5]:
        lines.append(f"- **{r['total']:.3f}** | {r.get('name', '')[:40]}")
        lines.append(f"  KPI: {r.get('kpi', '')[:60]}")
    lines.append("")
    lines.append("## 低分示例（Bottom 5）")
    for r in sorted(results, key=lambda x: x["total"])[:5]:
        lines.append(f"- **{r['total']:.3f}** | {r.get('name', '')[:40]}")
        lines.append(f"  KPI: {r.get('kpi', '')[:60]}")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  文本报告: {path}")


def main():
    parser = argparse.ArgumentParser(description="单套方案 3D 评估")
    parser.add_argument("--method", type=str, default=None,
                        help="方案名（rule/bm25/direct/simple/kg），自动匹配 *_names_chunk_*.json")
    parser.add_argument("--input", type=str, default=None,
                        help="输入结果文件（与 --method 二选一）")
    parser.add_argument("--label", type=str, default=None,
                        help="方案显示名称（默认与 method 相同）")
    parser.add_argument("--chunks", type=str, default="00_01_02_03_04_05",
                        help="chunk 标签")
    parser.add_argument("--max-entries", type=int, default=0, help="最多评估 N 条（调试用）")
    parser.add_argument("--concurrency", type=int, default=10, help="并行评估数")
    parser.add_argument("--resume", action="store_true", help="从上次断点继续")
    args = parser.parse_args()

    # 确定输入文件
    if args.method:
        method = args.method
        input_path = INPUT_DIR / f"{method}_names_chunk_{args.chunks}.json"
        label = args.label or method
        tag = f"{method}_chunk_{args.chunks}"
    elif args.input:
        input_path = Path(args.input)
        label = args.label or "custom"
        import re as _re
        m = _re.search(r'(?:.*?)_names_chunk_(.+)\.json', input_path.name)
        tag = m.group(1) if m else "unknown"
    else:
        sys.exit("[错误] 请指定 --method 或 --input")

    if not input_path.exists():
        sys.exit(f"[错误] 找不到 {input_path}")

    results_path = INPUT_DIR / f"eval_3d_{tag}.json"
    report_path = INPUT_DIR / f"eval_3d_{tag}.md"

    print(f"输入文件: {input_path.name}")
    print(f"方案名称: {label}")

    # 加载数据
    data = json.loads(input_path.read_text(encoding="utf-8"))
    entries = data.get("results", [])
    print(f"共 {len(entries)} 条 KPI")

    if args.max_entries:
        entries = entries[:args.max_entries]
        print(f"  (调试模式: 只评估前 {args.max_entries} 条)")

    # 按项目分组
    project_entries = defaultdict(list)
    for e in entries:
        project_entries[e["project_id"]].append(e)

    print(f"评估 {len(entries)} 条 KPI（{len(project_entries)} 个项目）...")

    # 恢复进度
    all_results = []
    completed_projects = 0
    total_projects = len(project_entries)
    if args.resume and results_path.exists():
        saved = json.loads(results_path.read_text(encoding="utf-8"))
        all_results = saved.get("results", [])
        completed_ids = set(r["project_id"] for r in all_results)
        for pid in list(project_entries.keys()):
            if pid in completed_ids:
                del project_entries[pid]
        completed_projects = saved.get("completed_projects", 0)
        print(f"恢复进度: 跳过 {len(completed_ids)} 个已完成项目，剩余 {len(project_entries)} 个")

    t_start = time.time()
    pbar = tqdm(project_entries.items(), desc="3D 评估", unit="项目", ncols=100)
    for pid, entry_list in pbar:
        chunks, index, embedder = load_project_assets(pid)
        all_names_list = [e.get("name", "") for e in entry_list]
        pbar.set_postfix_str(f"{pid[-6:]} chunks={len(chunks)} {len(entry_list)}条KPI")

        def eval_one(e):
            return evaluate_one(e, chunks, index, embedder, all_names_list)

        pair_results = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(eval_one, e): i for i, e in enumerate(entry_list)}
            kpi_pbar = tqdm(total=len(futures), desc=f"  KPI", unit="条", leave=False, ncols=80)
            for f in as_completed(futures):
                try:
                    pair_results.append(f.result())
                except Exception:
                    pass
                kpi_pbar.update(1)
            kpi_pbar.close()

        all_results.extend(pair_results)
        completed_projects += 1

        # 每完成一个项目保存一次
        summary = build_summary(all_results, args.label)
        atomic_write_json({
            "completed_projects": completed_projects,
            "total_projects": total_projects,
            "results": all_results,
            "summary": summary,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, results_path)

    # 最终保存
    summary = build_summary(all_results, args.label)
    save_data = {
        "completed_projects": completed_projects,
        "total_projects": total_projects,
        "results": all_results,
        "summary": summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(save_data, results_path)
    save_report(summary, all_results, report_path)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"3D 评估完成! 耗时: {elapsed/60:.1f} 分")
    print(f"{'=' * 60}")
    print(f"方案: {args.label}")
    print(f"评估 KPI 数: {summary['total_pairs']}")
    print(f"综合总分: {summary['avg_total']:.4f}")
    print(f"  语义保真度: {summary['avg_semantic']:.4f}")
    print(f"  原文可溯性: {summary['avg_traceability']:.4f}")
    print(f"  命名规范性: {summary['avg_regularity']:.4f}")
    print(f"  生成失败: {summary['fail_count']} ({summary['fail_rate']}%)")
    print(f"输出: {results_path.name}")


if __name__ == "__main__":
    main()
