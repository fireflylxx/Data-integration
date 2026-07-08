"""
Phase 1: Simple 方案批量生成
==============================
调用 LLM 直接为 chunk_00 所有项目的 KPI 生成数据集名称（无 KG 检索）。

用法:
  python baselines/batch_generate_simple.py
  python baselines/batch_generate_simple.py --resume
  python baselines/batch_generate_simple.py --max-projects 3

输出:
  output/comparison_results/simple_names.json
"""
import json, sys, time, re, argparse, os, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Windows GBK console workaround
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from common_kpi import (
    resolve_chunk_dirs, discover_projects, atomic_write_json,
    simple_name, _try_extract_name,
)

OUTPUT_DIR = BASE_DIR / "output" / "comparison_results"


# =============================================================================
# 主流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 1: Simple 方案批量生成")
    parser.add_argument("--resume", action="store_true", help="从上次断点继续")
    parser.add_argument("--max-projects", type=int, default=0, help="最多处理N个项目（调试用）")
    parser.add_argument("--chunks", type=str, default="00-05",
                        help="chunk 范围，如 '00-05' / '00,01,03' / '00'（默认 00-05）")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunk_labels, chunk_dirs = resolve_chunk_dirs(args.chunks)
    chunks_tag = "_".join(chunk_labels)
    output_path = OUTPUT_DIR / f"simple_names_chunk_{chunks_tag}.json"
    print(f"Chunk 范围: {chunks_tag} ({len(chunk_dirs)} 个目录)")

    # 发现项目
    all_projects = discover_projects(chunk_dirs)
    print(f"发现 {len(all_projects)} 个项目")

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

    # 过滤已完成的
    remaining_projects = [p for p in all_projects if p["pid"] not in completed_pids]
    if completed_pids:
        print(f"跳过 {len(completed_pids)} 个已完成的项目，剩余 {len(remaining_projects)} 个")

    pbar = tqdm(remaining_projects, desc="Simple 生成", unit="项目", ncols=100)
    for project in pbar:
        pid = project["pid"]
        pname = project["name"]
        kpis = project["kpis"]

        if not kpis:
            completed_pids.add(pid)
            pbar.set_postfix_str(f"{pname[:20]} 无KPI")
            continue

        pbar.set_postfix_str(f"{pname[:20]} — {len(kpis)}条KPI")

        project_results = []
        kpi_bar = tqdm(kpis, desc=f"  KPI", unit="条", leave=False, ncols=80)
        for kpi in kpi_bar:
            t0 = time.time()
            try:
                name = simple_name(kpi)
            except Exception as e:
                name = f"[错误:{e}]"
            elapsed = time.time() - t0

            entry = {
                "project_id": pid,
                "project_name": pname,
                "kpi": kpi,
                "name": name,
                "success": not name.startswith("["),
                "time": round(elapsed, 1),
            }
            project_results.append(entry)
            status = "OK" if entry["success"] else "FAIL"
            kpi_bar.set_postfix_str(f"{status} {name[:25]} {elapsed:.0f}s")

        all_results.extend(project_results)
        completed_pids.add(pid)

        # 每完成一个项目保存一次
        done = len(completed_pids)
        elapsed = time.time() - t_start
        rate = done / max(elapsed, 1)
        remaining = (total_projects - done) / max(rate, 0.001)

        save_data = {
            "completed_pids": list(completed_pids),
            "results": all_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stats": {
                "total_projects": total_projects,
                "completed": done,
                "total_kpis": len(all_results),
                "success_count": sum(1 for r in all_results if r["success"]),
                "fail_count": sum(1 for r in all_results if not r["success"]),
                "success_rate": round(sum(1 for r in all_results if r["success"]) / max(len(all_results), 1) * 100, 1),
            },
        }
        atomic_write_json(save_data, output_path)

        pbar.set_postfix_str(f"{done}/{total_projects} {elapsed/60:.0f}分")

    # 最终保存
    stats = {
        "total_projects": len(completed_pids),
        "total_kpis": len(all_results),
        "success_count": sum(1 for r in all_results if r["success"]),
        "fail_count": sum(1 for r in all_results if not r["success"]),
        "success_rate": round(sum(1 for r in all_results if r["success"]) / max(len(all_results), 1) * 100, 1),
    }
    save_data = {
        "completed_pids": list(completed_pids),
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": stats,
    }
    atomic_write_json(save_data, output_path)

    print(f"\n{'=' * 60}")
    print(f"Simple 方案生成完成!")
    print(f"  项目数: {stats['total_projects']}")
    print(f"  KPI数: {stats['total_kpis']}")
    print(f"  成功率: {stats['success_rate']}% ({stats['success_count']}/{stats['total_kpis']})")
    print(f"  输出: {output_path}")


if __name__ == "__main__":
    main()
