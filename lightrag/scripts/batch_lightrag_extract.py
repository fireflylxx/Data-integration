"""
LightRAG 知识图谱批量构建 — chunk_00 ~ chunk_N
=================================================
从 project_chunks_cleaned 读取项目文件，使用 LightRAG 方式逐块抽取实体+关系，
构建 low_level_kv + high_level_kv 索引。

用法:
  python scripts/batch_lightrag_extract.py --chunks 00
  python scripts/batch_lightrag_extract.py --chunks 00 --resume
  python scripts/batch_lightrag_extract.py --chunks 00 --max-projects 5

输出:
  output/lightrag_extract_cleaned/{project_id}/
    - entities.json      # 去重归一化后的实体列表
    - relations.json     # 去重归一化后的关系列表
    - low_level_kv.json  # Key=实体名, Value=profile
    - high_level_kv.json # Key=主题词, Value=关系描述
    - chunks/            # 原文分块
    - summary.json       # 统计摘要
"""

import json, sys, time, re, argparse, os, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Windows GBK console workaround
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts" / "complex"))

# Import LightRAG extraction functions
from lightrag_extract_cleaned import (
    parse_project_file, chunk_text, extract_chunk,
    dedup_entities, dedup_relations,
    normalize_extraction, build_profile, save_results,
    llm_chat, extract_json,
)

CHUNK_BASE = BASE_DIR / "output" / "project_chunks_cleaned"
OUTPUT_BASE = BASE_DIR / "output" / "lightrag_extract_cleaned"
KG_BASE = BASE_DIR / "output" / "kg_ontology"


def resolve_chunk_dirs(chunks_arg):
    """解析 --chunks 参数，返回 (chunk编号列表, chunk目录列表)"""
    if "-" in chunks_arg:
        start, end = chunks_arg.split("-", 1)
        nums = list(range(int(start), int(end) + 1))
    elif "," in chunks_arg:
        nums = [int(x) for x in chunks_arg.split(",")]
    else:
        nums = [int(chunks_arg)]
    dirs = []
    labels = []
    for n in nums:
        label = f"{n:02d}"
        d = CHUNK_BASE / f"project_chunk_{label}"
        if not d.exists():
            print(f"[警告] chunk 目录不存在: {d}")
            continue
        dirs.append(d)
        labels.append(label)
    return labels, dirs


def atomic_write_json(data, path: Path):
    tmp = path.with_suffix(".tmp." + path.name)
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def discover_projects(chunk_dirs: list, skip_existing: bool = True):
    """扫描 chunk 目录，发现所有项目（仅含 KG 数据存在的项目）"""
    projects = []
    for chunk_dir in chunk_dirs:
        if not chunk_dir.exists():
            print(f"[错误] chunk 目录不存在: {chunk_dir}")
            continue
        for f in sorted(chunk_dir.glob("*.txt")):
            parts = f.stem.split("_")
            pid = parts[-1] if len(parts) > 1 else ""
            if not pid or not pid.isdigit():
                continue

            name_parts = parts[1:-1]
            pname = "_".join(name_parts) if name_parts else f.stem

            # 确保有 KG 数据 (chunks 存在)
            if not (KG_BASE / pid / "chunks").exists():
                continue

            # 如果已抽取完成则跳过
            if skip_existing:
                summary_path = OUTPUT_BASE / pid / "summary.json"
                if summary_path.exists():
                    continue

            projects.append({
                "pid": pid,
                "name": pname,
                "filepath": f,
            })

    return projects


def process_project(project: dict, chunk_concurrency: int = 5) -> dict:
    """处理单个项目：读取 → 分块 → 逐块抽取(并行) → 去重 → 归一化 → 建索引 → 保存"""
    pid = project["pid"]
    pname = project["name"]
    filepath = project["filepath"]

    # 1. 解析项目文件
    proj = parse_project_file(filepath)
    if proj is None:
        return {"project_id": pid, "status": "parse_failed"}

    text = proj["text"]
    if not text or len(text) < 100:
        return {"project_id": pid, "status": "empty_text"}

    # 2. LightRAG 风格分块 (800 chars, paragraph-based)
    chunks = chunk_text(text)
    if not chunks:
        return {"project_id": pid, "status": "no_chunks"}

    # 3. 并行抽取实体+关系
    all_entities = []
    all_relations = []
    success_chunks = 0

    def extract_one(idx_chunk):
        idx, chunk = idx_chunk
        sid = f"{pid}/chunk_{idx+1}"
        result = extract_chunk(chunk, sid)
        if result:
            ents = result.get("entities", [])
            rels = result.get("relations", [])
            for ent in ents:
                ent["chunk_ids"] = [sid]
            for rel in rels:
                rel["chunk_id"] = sid
            return (idx, ents, rels)
        return (idx, [], [])

    with ThreadPoolExecutor(max_workers=chunk_concurrency) as executor:
        futures = {executor.submit(extract_one, (i, c)): i for i, c in enumerate(chunks)}
        for future in as_completed(futures):
            idx, ents, rels = future.result()
            all_entities.extend(ents)
            all_relations.extend(rels)
            if ents or rels:
                success_chunks += 1

    if not all_entities:
        return {
            "project_id": pid, "project_name": pname,
            "status": "no_entities", "chunks": len(chunks),
            "success_chunks": success_chunks,
        }

    # 4. 去重
    de = dedup_entities(all_entities)
    dr = dedup_relations(all_relations)

    # 5. 粒度归一化 (类型归并 + 非技术过滤)
    de, dr = normalize_extraction(de, dr)

    # 6. 构建 profile (low_level_kv + high_level_kv)
    lk, hk = build_profile(de, dr)

    # 7. 保存
    save_results(pid, pname, de, dr, chunks, lk, hk, OUTPUT_BASE)

    return {
        "project_id": pid, "project_name": pname,
        "status": "success", "chunks": len(chunks),
        "success_chunks": success_chunks,
        "entities_raw": len(all_entities),
        "entities_deduped": len(de),
        "relations_raw": len(all_relations),
        "relations_deduped": len(dr),
        "low_level_kv": len(lk),
        "high_level_kv": len(hk),
    }


def main():
    parser = argparse.ArgumentParser(description="LightRAG 知识图谱批量构建")
    parser.add_argument("--chunks", type=str, default="00",
                        help="chunk 范围，如 '00' / '00-05' / '00,01,03'（默认 00）")
    parser.add_argument("--max-projects", type=int, default=0, help="最多处理 N 个项目")
    parser.add_argument("--resume", action="store_true", help="跳过已完成的项目")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="跳过已有 summary.json 的项目（默认关闭，因为存量转换已创建这些文件）")
    parser.add_argument("--force", action="store_true",
                        help="强制重新抽取（忽略已有数据）")
    parser.add_argument("--chunk-concurrency", type=int, default=5,
                        help="项目内并行处理chunk数（默认5）")
    parser.add_argument("--project-concurrency", type=int, default=1,
                        help="并行处理项目数（默认1，API限流时建议保持1）")
    args = parser.parse_args()

    chunk_labels, chunk_dirs = resolve_chunk_dirs(args.chunks)
    chunks_tag = "_".join(chunk_labels)
    print(f"Chunk 范围: {chunks_tag} ({len(chunk_dirs)} 个目录)")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # 发现项目
    skip = args.skip_existing and not args.force
    if args.force:
        print("  (强制模式: 忽略已有数据，重新抽取)")
    all_projects = discover_projects(chunk_dirs, skip_existing=skip)
    print(f"发现 {len(all_projects)} 个待处理项目")

    if args.max_projects:
        all_projects = all_projects[:args.max_projects]
        print(f"  (调试模式: 只处理前 {args.max_projects} 个项目)")

    if not all_projects:
        print("没有需要处理的项目。")
        return

    # 恢复进度
    manifest_path = OUTPUT_BASE / f"manifest_{chunks_tag}.json"
    completed_pids = set()
    all_results = []
    if args.resume and manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        all_results = saved.get("results", [])
        completed_pids = set(saved.get("completed_pids", []))
        print(f"恢复进度: 已完成 {len(completed_pids)} 个项目")

    remaining = [p for p in all_projects if p["pid"] not in completed_pids]
    if completed_pids:
        print(f"跳过 {len(completed_pids)} 个已完成项目，剩余 {len(remaining)} 个")

    t_start = time.time()

    # 串行或并行处理
    if args.project_concurrency > 1 and len(remaining) > 1:
        # 项目级并行
        pbar = tqdm(total=len(remaining), desc="LightRAG 抽取", unit="项目", ncols=100)

        def process_one(proj):
            try:
                return process_project(proj, chunk_concurrency=args.chunk_concurrency)
            except Exception as e:
                return {"project_id": proj["pid"], "status": f"error: {e}"}

        with ThreadPoolExecutor(max_workers=args.project_concurrency) as executor:
            futures = {executor.submit(process_one, p): p for p in remaining}
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                completed_pids.add(result.get("project_id", ""))

                done = len(completed_pids)
                elapsed = time.time() - t_start
                success_count = sum(1 for r in all_results if r.get("status") == "success")
                save_data = {
                    "completed_pids": list(completed_pids),
                    "results": all_results,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "stats": {"total": len(all_projects), "completed": done, "success": success_count},
                }
                atomic_write_json(save_data, manifest_path)
                pbar.update(1)
                pbar.set_postfix_str(f"{done}/{len(all_projects)} 成功{success_count} {elapsed/60:.0f}分")
        pbar.close()
    else:
        # 串行处理（默认）
        pbar = tqdm(remaining, desc="LightRAG 抽取", unit="项目", ncols=100)
        for project in pbar:
            pid = project["pid"]
            pbar.set_postfix_str(f"{project['name'][:20]}")

            try:
                result = process_project(project, chunk_concurrency=args.chunk_concurrency)
            except Exception as e:
                result = {"project_id": pid, "status": f"error: {e}"}

            all_results.append(result)
            completed_pids.add(pid)

            done = len(completed_pids)
            elapsed = time.time() - t_start
            success_count = sum(1 for r in all_results if r.get("status") == "success")
            entity_count = sum(r.get("entities_deduped", 0) for r in all_results if r.get("status") == "success")

            save_data = {
                "completed_pids": list(completed_pids),
                "results": all_results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "stats": {"total": len(all_projects), "completed": done, "success": success_count, "total_entities": entity_count},
            }
            atomic_write_json(save_data, manifest_path)
            pbar.set_postfix_str(f"{done}/{len(all_projects)} 成功{success_count} {elapsed/60:.0f}分")

    # 最终保存
    success_count = sum(1 for r in all_results if r.get("status") == "success")
    entity_count = sum(r.get("entities_deduped", 0) for r in all_results if r.get("status") == "success")
    relation_count = sum(r.get("relations_deduped", 0) for r in all_results if r.get("status") == "success")

    stats = {
        "total": len(all_projects),
        "completed": len(completed_pids),
        "success": success_count,
        "total_entities": entity_count,
        "total_relations": relation_count,
    }
    save_data = {
        "completed_pids": list(completed_pids),
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": stats,
    }
    atomic_write_json(save_data, manifest_path)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"LightRAG 抽取完成!")
    print(f"  项目: {stats['success']}/{stats['total']} 成功")
    print(f"  总计: {stats['total_entities']} 实体, {stats['total_relations']} 关系")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    print(f"  输出: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
