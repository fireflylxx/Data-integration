#!/usr/bin/env python3
"""
LightRAG 数据集名称生成 — chunk_00 ~ chunk_05
=============================================
使用 LightRAG 抽取结果（low_level_kv + high_level_kv + entities/relations）
作为知识来源，生成数据集名称。

流程:
  1. 读取 chunk_00~chunk_05 的项目文件
  2. 加载 LightRAG 抽取结果 (entities, relations, KV)
  3. LightRAGRetriever 多级检索 (Low-level + High-level + 实体路径)
  4. KPI Planner → 命名 → 回译验证
  5. 保存结果

用法:
    python scripts/run_lightrag_planner.py                           # 全部 00-05
    python scripts/run_lightrag_planner.py --chunks 00               # 仅 chunk_00
    python scripts/run_lightrag_planner.py --max-projects 3          # 测试 3 个项目
    python scripts/run_lightrag_planner.py --concurrency 2           # 2 项目并行

输出: output/lightrag_planner_output/{project_id}.json
"""
import json, sys, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from pipeline.kpi_planner import kpi_to_dataset, parse_kpi_structured
from pipeline.lightrag_retriever import LightRAGRetriever
from pipeline.full_pipeline import extract_kpi_list, parse_project_sections

CHUNK_BASE = BASE_DIR / "output" / "project_chunks_cleaned"
LIGHTRAG_DIR = BASE_DIR / "output" / "lightrag_extract_cleaned"
OUTPUT_DIR = BASE_DIR / "output" / "lightrag_planner_output"

MAX_KPI_PER_PROJECT = 30


def load_project(fp: Path) -> dict:
    """从 chunk 目录加载项目"""
    text = fp.read_text(encoding="utf-8", errors="ignore")
    pid = pno = pname = ""
    for line in text.split("\n")[:10]:
        if line.startswith("项目名称:"):
            pname = line.split(":", 1)[1].strip()
        elif line.startswith("项目编号:"):
            pno = line.split(":", 1)[1].strip()
        elif line.startswith("项目ID:"):
            pid = line.split(":", 1)[1].strip()
    return {"pid": pid, "pno": pno, "name": pname, "text": text, "file": fp}


def has_lightrag_data(pid: str) -> bool:
    """检查是否有 LightRAG 抽取数据"""
    lr_dir = LIGHTRAG_DIR / pid
    return (lr_dir / "entities.json").exists() and (lr_dir / "low_level_kv.json").exists()


def process_project(proj: dict) -> list:
    """处理单个项目：提取 KPI → LightRAG 检索 → Planner → 命名

    Returns:
        [{"kpi": str, "dataset_name": str, "validated": bool, ...}, ...]
    """
    pid = proj["pid"]
    pname = proj["name"]

    # 1. 提取 KPI
    try:
        sections = parse_project_sections(proj["text"])
        kpi_list = extract_kpi_list(
            sections.get("kpi_section", ""),
            full_text=proj["text"],
        )
        kpi_list = [k for k in kpi_list if len(k) >= 15][:MAX_KPI_PER_PROJECT]
    except Exception as e:
        print(f"  [KPI提取异常] {pid}: {e}")
        return []

    if not kpi_list:
        print(f"  [无KPI] {pid}: 未找到考核指标 (章节长度={len(sections.get('kpi_section',''))})")
        return []

    # 2. 初始化 LightRAG Retriever
    try:
        retriever = LightRAGRetriever(pid, base_dir=str(LIGHTRAG_DIR))
    except Exception as e:
        print(f"  [Retriever失败] {pid}: {e}")
        # 兜底: 用空上下文
        retriever = None

    # 3. 逐 KPI 处理
    results = []
    for ki, kpi_text in enumerate(kpi_list):
        t0 = time.time()

        # 检索上下文
        if retriever:
            try:
                merged_context = retriever.retrieve(kpi_text, top_k=3)
            except Exception as e:
                merged_context = f"(检索失败: {e})"
        else:
            merged_context = "(无LightRAG数据)"

        # 规划 → 命名 → 验证
        try:
            kpi_results = kpi_to_dataset(
                kpi_description=kpi_text,
                retriever_context=merged_context,
                max_retries=2,
            )
        except Exception as e:
            print(f"    [规划失败] {str(e)[:60]}")
            kpi_results = []

        # 整理
        for r in kpi_results:
            if isinstance(r, dict):
                r["project_id"] = pid
                r["project_name"] = pname
                r["kg_paths_found"] = 0  # LightRAG 不返回路径数
                r["chunks_found"] = 0
                results.append(r)

        elapsed = time.time() - t0
        names = [r.get("dataset_name", "")[:30] for r in kpi_results]
        print(f"    [{ki+1}/{len(kpi_list)}] {kpi_text[:40]:40s} → {names} ({elapsed:.1f}s)")

        time.sleep(0.2)  # API 节流

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LightRAG 数据集名称生成")
    parser.add_argument("--chunks", default="00,01,02,03,04,05",
                        help="chunk 列表（逗号分隔，默认 00-05）")
    parser.add_argument("--max-projects", type=int, default=0,
                        help="最多处理 N 个项目")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="并行项目数（默认 1，API 限流时保持 1）")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="跳过已处理项目")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunk_ids = [c.strip().zfill(2) for c in args.chunks.split(",")]
    print("=" * 60)
    print("LightRAG 数据集名称生成")
    print(f"Chunks: {chunk_ids}")
    print(f"并发数: {args.concurrency}")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 60)

    # 收集项目
    all_projects = []
    for cid in chunk_ids:
        chunk_dir = CHUNK_BASE / f"project_chunk_{cid}"
        if not chunk_dir.exists():
            print(f"  [跳过] chunk_{cid} 目录不存在")
            continue
        files = sorted(chunk_dir.glob("*.txt"))
        count = 0
        for fp in files:
            proj = load_project(fp)
            if not proj["pid"]:
                continue
            if not has_lightrag_data(proj["pid"]):
                continue
            all_projects.append(proj)
            count += 1
        print(f"  chunk_{cid}: {count} 个项目（有 LightRAG 数据）")

    if args.max_projects > 0:
        all_projects = all_projects[:args.max_projects]
        print(f"  限制: 只处理前 {args.max_projects} 个项目")

    if not all_projects:
        print("没有需要处理的项目。")
        return

    # 跳过已处理的
    if args.resume:
        remaining = []
        skipped = 0
        for proj in all_projects:
            out_path = OUTPUT_DIR / f"{proj['pid']}.json"
            if out_path.exists():
                skipped += 1
                continue
            remaining.append(proj)
        print(f"\n跳过 {skipped} 个已处理项目，剩余 {len(remaining)} 个")
        all_projects = remaining

    print(f"\n开始处理 {len(all_projects)} 个项目...")
    if args.concurrency > 1:
        print(f"预计: 每个项目 ~2-5 分钟, 总计 ~{len(all_projects)*3//args.concurrency//60:.0f}-{len(all_projects)*5//args.concurrency//60:.0f} 分钟")

    total_start = time.time()
    all_results = []
    completed = 0
    failed = 0

    if args.concurrency > 1:
        # 并行模式
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(process_project, p): p for p in all_projects}
            for i, future in enumerate(as_completed(futures), 1):
                proj = futures[future]
                try:
                    results = future.result()

                    # 保存
                    output = {
                        "project_id": proj["pid"],
                        "project_name": proj["name"],
                        "pipeline": "lightrag",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_kpis": len(results),
                        "results": results,
                    }
                    json.dump(output, open(OUTPUT_DIR / f"{proj['pid']}.json", "w", encoding="utf-8"),
                              ensure_ascii=False, indent=2)

                    validated = sum(1 for r in results if r.get("validated"))
                    completed += 1
                    elapsed = time.time() - total_start
                    print(f"[{i}/{len(all_projects)}] [OK] {proj['name'][:30]:30s} {len(results):2d}数据集 通过={validated} {elapsed/60:.0f}分")
                except Exception as e:
                    failed += 1
                    print(f"[{i}/{len(all_projects)}] [FAIL] {proj['name'][:30]:30s} {str(e)[:40]}")
    else:
        # 串行模式（默认）
        for i, proj in enumerate(all_projects, 1):
            print(f"\n[{i}/{len(all_projects)}] {proj['name']} ({proj['pid']})")
            t0 = time.time()

            try:
                results = process_project(proj)

                # 保存
                output = {
                    "project_id": proj["pid"],
                    "project_name": proj["name"],
                    "pipeline": "lightrag",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_kpis": len(results),
                    "results": results,
                }
                json.dump(output, open(OUTPUT_DIR / f"{proj['pid']}.json", "w", encoding="utf-8"),
                          ensure_ascii=False, indent=2)

                validated = sum(1 for r in results if r.get("validated"))
                avg_score = sum(r.get("validation_score", 0) for r in results) / max(len(results), 1)
                completed += 1
                elapsed = time.time() - t0
                print(f"  >>> {len(results)}数据集, 通过={validated}, 平均分={avg_score:.2f}, 耗时={elapsed:.0f}s")
            except Exception as e:
                failed += 1
                print(f"  [FAIL] 失败: {str(e)[:60]}")
                import traceback
                traceback.print_exc()

            # 节流
            time.sleep(1)

    # 汇总
    total_elapsed = (time.time() - total_start) / 60
    total_datasets = sum(len(json.loads(open(OUTPUT_DIR / f"{r['pid']}.json", encoding="utf-8").read()).get("results", []))
                         for r in all_projects if (OUTPUT_DIR / f"{r['pid']}.json").exists())

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunks": chunk_ids,
        "stats": {
            "total": len(all_projects),
            "completed": completed,
            "failed": failed,
            "total_datasets": total_datasets,
            "elapsed_min": round(total_elapsed, 1),
        },
    }
    json.dump(manifest, open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"完成! 耗时 {total_elapsed:.1f} 分钟")
    print(f"  项目: {completed}/{len(all_projects)} 成功")
    print(f"  数据集: {total_datasets}")
    print(f"  输出: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
