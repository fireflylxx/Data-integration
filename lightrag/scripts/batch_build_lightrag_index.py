"""
LightRAG 索引构建 — 从已有 KG 数据转换
========================================
从 kg_ontology 读取已有 KG 数据（entities.json, relations.json），
转换为 LightRAG 格式（6实体类型 + 7关系谓词 + low/high level KV + FAISS 向量）。

无需 LLM 调用，纯数据格式转换。

用法:
  python scripts/batch_build_lightrag_index.py --chunks 00-05
  python scripts/batch_build_lightrag_index.py --chunks 00-05 --resume

输出:
  output/lightrag_extract_cleaned/{project_id}/
    - entities.json      # LightRAG 格式（含 profile + neighbors）
    - relations.json     # LightRAG 格式（含归一化谓词）
    - low_level_kv.json  # 实体名 → profile
    - high_level_kv.json # 主题词 → 描述
    - chunks/            # 原始文本块（从 kg_ontology 复制）
    - faiss.index        # FAISS 向量索引（从 kg_ontology 复制或重建）
    - faiss.pkl          # Embedder 序列化
    - summary.json       # 统计摘要
"""

import json, sys, time, re, argparse, os, io, shutil
from pathlib import Path
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts" / "complex"))

from lightrag_extract_cleaned import build_profile, save_results

KG_BASE = BASE_DIR / "output" / "kg_ontology"
OUTPUT_BASE = BASE_DIR / "output" / "lightrag_extract_cleaned"
CHUNK_BASE = BASE_DIR / "output" / "project_chunks_cleaned"

# =============================================================================
# KG → LightRAG 类型映射
# =============================================================================

# KG 实体类型 → LightRAG 实体类型
KG_TO_LR_ENTITY = {
    "OBJECT": "OBJECT",
    "METHOD": "METHOD",
    "PARAMETER": "METRIC",
    "ACTIVITY": "METHOD",       # 验证活动 → 方法
    "EQUIPMENT": "EQUIPMENT",
    "MATERIAL": "OBJECT",       # 材料 → 研究对象
    "SOFTWARE": "METHOD",        # 软件算法 → 方法
    "SYSTEM": "OBJECT",         # 系统平台 → 研究对象
    "MODEL": "ACHIEVEMENT",     # 模型 → 成果
    # 额外映射
    "TOPIC": "TOPIC",
    "ACHIEVEMENT": "ACHIEVEMENT",
    "METRIC": "METRIC",
}

# KG 关系谓词 → LightRAG 关系谓词
KG_TO_LR_RELATION = {
    "VIA": "采用",           # 对象通过方法
    "VERIFIES": "测试",      # 方法验证参数
    "EXECUTES": "包含",      # 对象执行活动 → 包含
    "PRODUCES": "产出",      # 对象产出数据集
    "BELONGS_TO": "归属",    # 属于
    "MAPS_TO": "考核",       # 对应指标
}


def atomic_write_json(data, path: Path):
    tmp = path.with_suffix(".tmp." + path.name)
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def resolve_chunk_dirs(chunks_arg):
    """解析 --chunks 参数"""
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


def discover_projects(chunk_dirs: list):
    """扫描 chunk 目录，发现所有有 KG 数据的项目"""
    projects = []
    for chunk_dir in chunk_dirs:
        if not chunk_dir.exists():
            continue
        for f in sorted(chunk_dir.glob("*.txt")):
            parts = f.stem.split("_")
            pid = parts[-1] if len(parts) > 1 else ""
            if not pid or not pid.isdigit():
                continue
            name_parts = parts[1:-1]
            pname = "_".join(name_parts) if name_parts else f.stem

            # 确保有 KG 数据
            if not (KG_BASE / pid / "entities.json").exists():
                continue
            projects.append({"pid": pid, "name": pname, "filepath": f})
    return projects


def convert_project(pid: str, pname: str) -> dict:
    """将单个项目的 KG 数据转换为 LightRAG 格式"""
    kg_dir = KG_BASE / pid
    lr_dir = OUTPUT_BASE / pid

    # 1. 加载 KG 数据
    entities = json.loads((kg_dir / "entities.json").read_text(encoding="utf-8"))
    relations = json.loads((kg_dir / "relations.json").read_text(encoding="utf-8"))

    # 2. 转换实体类型
    lr_entities = []
    entity_name_map = {}  # 原实体名 → 新实体名（用于关系转换）
    for ent in entities:
        orig_name = ent["name"]
        orig_type = ent.get("type", "OBJECT").upper()
        lr_type = KG_TO_LR_ENTITY.get(orig_type, "OBJECT")
        lr_entities.append({
            "name": orig_name,
            "type": lr_type,
            "chunk_ids": ent.get("chunk_ids", []),
        })
        entity_name_map[orig_name] = True

    # 3. 转换关系谓词
    lr_relations = []
    for rel in relations:
        h, r, t = rel.get("head", ""), rel.get("relation", ""), rel.get("tail", "")
        if h not in entity_name_map or t not in entity_name_map:
            continue
        lr_r = KG_TO_LR_RELATION.get(r, "包含")  # 默认映射为"包含"
        lr_relations.append({
            "head": h,
            "relation": lr_r,
            "tail": t,
            "context": rel.get("context", ""),
            "chunk_id": rel.get("chunk_id", ""),
        })

    # 4. 构建 profile (low_level_kv + high_level_kv)
    low_kv, high_kv = build_profile(lr_entities, lr_relations)

    # 5. 读取 chunks
    chunk_dir = kg_dir / "chunks"
    chunks = []
    if chunk_dir.exists():
        for cf in sorted(chunk_dir.glob("*.txt")):
            chunks.append(cf.read_text(encoding="utf-8", errors="ignore"))

    # 6. 构建 neighbors（给 entities 用）
    nb = {e["name"]: [] for e in lr_entities}
    for rel in lr_relations:
        h, r, t = rel["head"], rel["relation"], rel["tail"]
        if h in nb:
            nb[h].append({"name": t, "relation": r, "direction": "out"})
        if t in nb:
            nb[t].append({"name": h, "relation": r, "direction": "in"})

    entities_out = []
    for ent in lr_entities:
        entities_out.append({
            "name": ent["name"],
            "type": ent["type"],
            "profile": low_kv.get(ent["name"], ""),
            "chunk_ids": ent.get("chunk_ids", []),
            "neighbors": nb.get(ent["name"], []),
        })

    # 7. 保存到 LightRAG 目录
    lr_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_json(entities_out, lr_dir / "entities.json")
    atomic_write_json(lr_relations, lr_dir / "relations.json")
    atomic_write_json(low_kv, lr_dir / "low_level_kv.json")
    atomic_write_json(high_kv, lr_dir / "high_level_kv.json")

    # 复制 chunks
    lr_chunk_dir = lr_dir / "chunks"
    lr_chunk_dir.mkdir(exist_ok=True)
    if chunk_dir.exists():
        for cf in sorted(chunk_dir.glob("*.txt")):
            shutil.copy2(cf, lr_chunk_dir / cf.name)

    # 复制或重建 FAISS 索引
    faiss_src = kg_dir / "faiss.index"
    pkl_src = kg_dir / "faiss.pkl"
    if faiss_src.exists() and pkl_src.exists():
        shutil.copy2(faiss_src, lr_dir / "faiss.index")
        shutil.copy2(pkl_src, lr_dir / "faiss.pkl")

    # 9. 摘要
    summary = {
        "project_id": pid,
        "project_name": pname,
        "entity_count": len(entities_out),
        "relation_count": len(lr_relations),
        "chunk_count": len(chunks),
        "entity_types": {},
        "relation_types": {},
        "source": "converted_from_kg_ontology",
    }
    for e in entities_out:
        t = e.get("type", "UNKNOWN")
        summary["entity_types"][t] = summary["entity_types"].get(t, 0) + 1
    for rel in lr_relations:
        r = rel.get("relation", "UNKNOWN")
        summary["relation_types"][r] = summary["relation_types"].get(r, 0) + 1

    atomic_write_json(summary, lr_dir / "summary.json")

    return {
        "project_id": pid,
        "project_name": pname,
        "status": "success",
        "entities": len(entities_out),
        "relations": len(lr_relations),
        "chunks": len(chunks),
        "has_faiss": faiss_src.exists(),
    }


def main():
    parser = argparse.ArgumentParser(description="LightRAG 索引构建（从 KG 数据转换）")
    parser.add_argument("--chunks", type=str, default="00-05",
                        help="chunk 范围，如 '00-05'（默认 00-05）")
    parser.add_argument("--resume", action="store_true", help="跳过已转换的项目")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="并行数（默认1，转换是纯IO密集型，可适当提高）")
    args = parser.parse_args()

    chunk_labels, chunk_dirs = resolve_chunk_dirs(args.chunks)
    chunks_tag = "_".join(chunk_labels)
    print(f"Chunk 范围: {chunks_tag} ({len(chunk_dirs)} 个目录)")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # 发现项目
    all_projects = discover_projects(chunk_dirs)
    print(f"发现 {len(all_projects)} 个有 KG 数据的项目")

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

    pbar = tqdm(remaining, desc="转换 LightRAG 索引", unit="项目", ncols=100)
    for project in pbar:
        pid = project["pid"]
        try:
            result = convert_project(pid, project["name"])
        except Exception as e:
            result = {"project_id": pid, "status": f"error: {e}"}

        all_results.append(result)
        completed_pids.add(pid)

        # 每完成一批保存 manifest
        done = len(completed_pids)
        elapsed = time.time() - t_start
        success = sum(1 for r in all_results if r.get("status") == "success")
        entity_count = sum(r.get("entities", 0) for r in all_results if r.get("status") == "success")

        save_data = {
            "completed_pids": list(completed_pids),
            "results": all_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stats": {
                "total": len(all_projects),
                "completed": done,
                "success": success,
                "total_entities": entity_count,
            },
        }
        atomic_write_json(save_data, manifest_path)
        pbar.set_postfix_str(f"{done}/{len(all_projects)} {entity_count}实体")

    # 最终统计
    success = [r for r in all_results if r.get("status") == "success"]
    total_entities = sum(r.get("entities", 0) for r in success)
    total_relations = sum(r.get("relations", 0) for r in success)
    with_faiss = sum(1 for r in success if r.get("has_faiss"))

    stats = {
        "total": len(all_projects),
        "completed": len(completed_pids),
        "success": len(success),
        "total_entities": total_entities,
        "total_relations": total_relations,
        "with_faiss": with_faiss,
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
    print(f"LightRAG 索引构建完成!")
    print(f"  项目: {stats['success']}/{stats['total']} 成功")
    print(f"  实体: {stats['total_entities']}, 关系: {stats['total_relations']}")
    print(f"  FAISS: {with_faiss} 个项目有向量索引")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  输出: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
