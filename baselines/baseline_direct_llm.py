"""
Baseline C: Direct LLM 命名方案
================================
将项目完整原文作为上下文喂给 LLM 生成数据集名称（无检索，但有完整信息）。

与 Simple 方案的区别:
- Simple: 只有 KPI 文本，无项目上下文
- Direct: 项目全文 + KPI，LLM 能从全文自主定位相关信息

用法:
  python baselines/baseline_direct_llm.py --chunks 00
  python baselines/baseline_direct_llm.py --chunks 00 --resume
  python baselines/baseline_direct_llm.py --chunks 00 --max-projects 5

输出:
  output/comparison_results/direct_names_chunk_00.json
"""

import json, sys, time, re, argparse, os, io
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

MAX_CONTEXT_CHARS = 5000  # 全文最多取 5000 字（qwen3-32b 支持 128K，但足够）


def load_full_text(pid: str) -> str:
    """加载项目的全部原文（拼接 chunks，截断至 MAX_CONTEXT_CHARS）"""
    chunk_dir = KG_BASE / pid / "chunks"
    parts = []
    total = 0
    if chunk_dir.exists():
        for f in sorted(chunk_dir.glob("chunk_*.txt")):
            text = f.read_text(encoding="utf-8", errors="ignore")
            if total + len(text) > MAX_CONTEXT_CHARS:
                remain = MAX_CONTEXT_CHARS - total
                if remain > 200:
                    parts.append(text[:remain])
                break
            parts.append(text)
            total += len(text)
    return "\n".join(parts)


# =============================================================================
# Direct LLM 命名
# =============================================================================

DIRECT_NAMING_PROMPT = """你是一个科研数据集命名专家。请根据项目任务书原文和考核指标(KPI)，生成数据集名称。

## 规则
- 名称必须包含KPI原文中的核心对象和参数
- 从项目全文中提取相关的技术术语来辅助命名
- 以"数据集"或"数据"结尾
- 名称长度15-35字
- 只输出名称，不要解释

## 输出格式
{"name_cn": "..."}"""


def direct_name(kpi_description: str, full_text: str) -> str:
    """Direct 版本: 全文 + KPI → LLM 命名"""
    # 截断全文避免超出 token 限制
    truncated = full_text[:MAX_CONTEXT_CHARS]

    user = f"## 项目任务书原文\n{truncated}\n\n## 考核指标\n{kpi_description}\n\n请根据以上项目原文和考核指标生成数据集名称。"
    resp = llm_chat(DIRECT_NAMING_PROMPT, user, max_tokens=800, temperature=0.3)
    name = _try_extract_name(resp)
    if name:
        return name

    # 重试
    resp2 = llm_chat(DIRECT_NAMING_PROMPT, f"考核指标: {kpi_description}", max_tokens=800, temperature=0.5)
    name = _try_extract_name(resp2)
    if name:
        return name

    return "[生成失败]"


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline C: Direct LLM 命名")
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
    output_path = OUTPUT_DIR / f"direct_names_chunk_{chunks_tag}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Chunk 范围: {chunks_tag} ({len(chunk_dirs)} 个目录)")

    # 发现项目（需要 KG 数据以读取 chunks）
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

    pbar = tqdm(remaining, desc="Direct 生成", unit="项目", ncols=100)
    for project in pbar:
        pid = project["pid"]
        pname = project["name"]
        kpis = project["kpis"]

        if not kpis:
            completed_pids.add(pid)
            pbar.set_postfix_str(f"{pname[:20]} 无KPI")
            continue

        # 加载全文
        full_text = load_full_text(pid)
        if not full_text:
            pbar.set_postfix_str(f"{pname[:20]} 无原文")
            for kpi in kpis:
                all_results.append({
                    "project_id": pid, "project_name": pname, "kpi": kpi,
                    "name": "[无原文]", "success": False, "time": 0,
                })
            completed_pids.add(pid)
            continue

        pbar.set_postfix_str(f"{pname[:20]} {len(full_text)}字 — {len(kpis)}条KPI")

        # 并行处理 KPI
        def process_one(kpi_data):
            kidx, kpi = kpi_data
            t0 = time.time()
            try:
                name = direct_name(kpi, full_text)
            except Exception as e:
                name = f"[错误:{e}]"
            elapsed = time.time() - t0
            return {
                "project_id": pid, "project_name": pname, "kpi": kpi,
                "name": name, "success": not name.startswith("["),
                "time": round(elapsed, 1),
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
                except Exception as e:
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
    print(f"Direct LLM 方案完成!")
    print(f"  项目数: {stats['total_projects']}")
    print(f"  KPI数: {stats['total_kpis']}")
    print(f"  成功率: {stats['success_rate']}% ({stats['success_count']}/{stats['total_kpis']})")
    print(f"  耗时: {elapsed/60:.1f} 分")
    print(f"  输出: {output_path}")


if __name__ == "__main__":
    main()
