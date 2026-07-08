#!/usr/bin/env python3
"""
================================================================================
可交付系统流水线（端到端）
================================================================================
架构: Project_Cleaned → KG_Extraction → Dataset_Inference → CSV_Export → 3D_Evaluation

实体类型 (6种): TOPIC | OBJECT | METRIC | METHOD | EQUIPMENT | ACHIEVEMENT
关系谓词 (7种): 归属 | 考核 | 产出 | 采用 | 测试 | 改进 | 包含

用法:
    python pipeline/run_pipeline.py --input output/project_cleaned
    python pipeline/run_pipeline.py --input output/project_cleaned --skip-kg
    python pipeline/run_pipeline.py --project 1646724823658876928
================================================================================
"""

import json, sys, time
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
KG_OUTPUT_DIR = BASE_DIR / "output" / "lightrag_extract_cleaned"
PIPELINE_OUTPUT = BASE_DIR / "output" / "pipeline_output"


# =============================================================================
# Phase 1: KG 抽取（调用 lightrag_extract_cleaned.py）
# =============================================================================

def run_kg_extraction(project_files: List[Path], force: bool = False) -> List[dict]:
    """运行 KG 抽取，返回各项目的 KG 路径信息"""
    results = []
    for fp in project_files:
        # 从文件名解析 project_id
        stem = fp.stem  # e.g. "01_xxx"
        # 读取文件头获取 project_id
        content = fp.read_text(encoding="utf-8")
        pid_match = __import__("re").search(r'项目ID:\s*(\d+)', content)
        if not pid_match:
            print(f"  [跳过] 无法解析项目ID: {fp.name}")
            continue
        pid = pid_match.group(1)

        kg_dir = KG_OUTPUT_DIR / pid
        if kg_dir.exists() and not force:
            print(f"  [跳过] KG已存在: {pid}")
            results.append({"project_id": pid, "file": fp.name, "status": "exists"})
            continue

        print(f"  [抽取] {fp.name} → {pid}")
        # 调用子进程执行抽取
        import subprocess
        script = BASE_DIR / "scripts" / "complex" / "lightrag_extract_cleaned.py"
        cmd = [sys.executable, str(script), "--input", str(fp.parent),
               "--output", str(KG_OUTPUT_DIR)]
        try:
            subprocess.run(cmd, check=True, timeout=1800, capture_output=True)
            results.append({"project_id": pid, "file": fp.name, "status": "extracted"})
        except subprocess.TimeoutExpired:
            print(f"    [超时] {fp.name}")
            results.append({"project_id": pid, "file": fp.name, "status": "timeout"})
        except subprocess.CalledProcessError as e:
            print(f"    [失败] {e.stderr.decode()[-200:]}")
            results.append({"project_id": pid, "file": fp.name, "status": "failed"})

    return results


# =============================================================================
# Phase 2: 数据集推理（从 KG 路径生成数据集候选）
# =============================================================================

def infer_datasets(entities: List[dict], relations: List[dict],
                   project_id: str) -> List[dict]:
    """
    从 KG 推理数据集候选。

    推理规则:
      Rule 1: 课题直接产出 (TOPIC --产出→ ACHIEVEMENT)
              生成: "{课题名}相关{成果类型}"

      Rule 2: 课题+指标 (TOPIC --考核→ METRIC)
              生成: "{研究对象}{指标}测试数据集"

      Rule 3: 完整验证链 (OBJECT --采用→ METHOD --测试→ METRIC)
              生成: "{对象}{指标}{方法}测试数据集"

      Rule 4: 半链 (OBJECT --采用→ METHOD)
              生成: "{对象}{方法}测试数据集"
    """
    # 构建索引
    enames = {e["name"]: e["type"] for e in entities}
    datasets = []
    seq = 1
    seen_names = set()

    # ---- Rule 1: TOPIC --产出→ ACHIEVEMENT ----
    for rel in relations:
        if rel["relation"] != "产出":
            continue
        topic = rel["head"]
        achievement = rel["tail"]
        if enames.get(topic) != "TOPIC" or enames.get(achievement) != "ACHIEVEMENT":
            continue
        name = f"{topic}{achievement}"
        if name in seen_names:
            continue
        seen_names.add(name)
        datasets.append({
            "id": f"{project_id}-{seq:03d}",
            "name_cn": name,
            "evidence_type": "TOPIC_ACHIEVEMENT",
            "confidence": "HIGH",
        })
        seq += 1

    # ---- Rule 2: TOPIC --考核→ METRIC + TOPIC --归属→ OBJECT ----
    topic_metrics = {}  # topic → [metrics]
    topic_objects = {}  # topic → [objects]
    for rel in relations:
        r = rel["relation"]
        if r == "考核" and enames.get(rel["head"]) == "TOPIC" and enames.get(rel["tail"]) == "METRIC":
            topic_metrics.setdefault(rel["head"], []).append(rel["tail"])
        elif r == "归属" and enames.get(rel["head"]) == "TOPIC" and enames.get(rel["tail"]) == "OBJECT":
            topic_objects.setdefault(rel["head"], []).append(rel["tail"])

    for topic, metrics in topic_metrics.items():
        objects = topic_objects.get(topic, ["相关"])
        for obj in objects:
            for metric in metrics:
                name = f"{obj}{metric}测试数据集"
                if name in seen_names:
                    continue
                seen_names.add(name)
                datasets.append({
                    "id": f"{project_id}-{seq:03d}",
                    "name_cn": name,
                    "evidence_type": "TOPIC_METRIC",
                    "confidence": "HIGH",
                })
                seq += 1

    # ---- Rule 3 & 4: 研究内容路径 ----
    # 构建邻接索引
    method_to_metrics = {}  # METHOD → [METRIC]
    obj_to_methods = {}     # OBJECT → [METHOD]
    obj_to_equip = {}       # OBJECT → [EQUIPMENT]
    equip_to_metrics = {}   # EQUIPMENT → [METRIC]

    for rel in relations:
        r, h, t = rel["relation"], rel["head"], rel["tail"]
        ht, tt = enames.get(h), enames.get(t)
        if r == "采用":
            if ht == "OBJECT" and tt == "METHOD":
                obj_to_methods.setdefault(h, []).append(t)
            elif ht == "OBJECT" and tt == "EQUIPMENT":
                obj_to_equip.setdefault(h, []).append(t)
        elif r == "测试":
            if ht == "METHOD" and tt == "METRIC":
                method_to_metrics.setdefault(h, []).append(t)
            elif ht == "EQUIPMENT" and tt == "METRIC":
                equip_to_metrics.setdefault(h, []).append(t)

    # Rule 3: OBJECT --采用→ METHOD --测试→ METRIC
    for obj, methods in obj_to_methods.items():
        for method in methods:
            metrics = method_to_metrics.get(method, [])
            for metric in metrics:
                name = f"{obj}{metric}{method}测试数据集"
                if name in seen_names:
                    continue
                seen_names.add(name)
                datasets.append({
                    "id": f"{project_id}-{seq:03d}",
                    "name_cn": name,
                    "evidence_type": "KG_FULL_PATH",
                    "evidence_path": [obj, "采用", method, "测试", metric],
                    "confidence": "HIGH",
                })
                seq += 1

    # OBJECT --采用→ EQUIPMENT --测试→ METRIC
    for obj, equip_list in obj_to_equip.items():
        for equip in equip_list:
            metrics = equip_to_metrics.get(equip, [])
            for metric in metrics:
                name = f"{obj}{metric}{equip}测试数据集"
                if name in seen_names:
                    continue
                seen_names.add(name)
                datasets.append({
                    "id": f"{project_id}-{seq:03d}",
                    "name_cn": name,
                    "evidence_type": "KG_FULL_PATH",
                    "evidence_path": [obj, "采用", equip, "测试", metric],
                    "confidence": "HIGH",
                })
                seq += 1

    # Rule 4: 半链 — OBJECT --采用→ METHOD（无对应METRIC）
    for obj, methods in obj_to_methods.items():
        for method in methods:
            if method in method_to_metrics:
                continue  # 已有完整路径
            name = f"{obj}{method}测试数据集"
            if name in seen_names:
                continue
            seen_names.add(name)
            datasets.append({
                "id": f"{project_id}-{seq:03d}",
                "name_cn": name,
                "evidence_type": "KG_HALF_PATH",
                "evidence_path": [obj, "采用", method],
                "confidence": "MEDIUM",
            })
            seq += 1

    return datasets


# =============================================================================
# Phase 3: CSV 导出
# =============================================================================

def export_to_csv(datasets: List[dict], project_id: str, output_dir: Path) -> Path:
    """将推理出的数据集导出为 CSV"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{project_id}.csv"

    import csv
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["项目ID", "数据集编号", "中文数据集名称", "证据类型", "置信度", "证据路径"])
        for ds in datasets:
            writer.writerow([
                project_id,
                ds["id"],
                ds["name_cn"],
                ds.get("evidence_type", ""),
                ds.get("confidence", ""),
                json.dumps(ds.get("evidence_path", []), ensure_ascii=False),
            ])

    print(f"  CSV导出: {csv_path} ({len(datasets)} 条)")
    return csv_path


# =============================================================================
# Phase 3b: 3D 评测摘要
# =============================================================================

def print_evaluation_summary(datasets: List[dict], entities: List[dict],
                              relations: List[dict], project_id: str):
    """打印 3D 评测摘要"""
    print(f"\n  {'='*40}")
    print(f"  3D 评测摘要 — {project_id}")
    print(f"  {'='*40}")

    # 实体类型分布
    etypes = {}
    for e in entities:
        etypes[e["type"]] = etypes.get(e["type"], 0) + 1
    print(f"  实体 ({len(entities)}): {dict(sorted(etypes.items(), key=lambda x:-x[1]))}")

    # 关系类型分布
    rtypes = {}
    for r in relations:
        rtypes[r["relation"]] = rtypes.get(r["relation"], 0) + 1
    print(f"  关系 ({len(relations)}): {dict(sorted(rtypes.items(), key=lambda x:-x[1]))}")

    # 数据集分布
    conf_counts = {}
    for ds in datasets:
        conf_counts[ds.get("confidence", "UNKNOWN")] = conf_counts.get(ds.get("confidence", "UNKNOWN"), 0) + 1
    print(f"  数据集 ({len(datasets)}): {dict(sorted(conf_counts.items(), key=lambda x:-x[1]))}")

    # 证据类型分布
    ev_counts = {}
    for ds in datasets:
        ev_counts[ds.get("evidence_type", "UNKNOWN")] = ev_counts.get(ds.get("evidence_type", "UNKNOWN"), 0) + 1
    print(f"  证据类型: {dict(sorted(ev_counts.items(), key=lambda x:-x[1]))}")


# =============================================================================
# 主流程
# =============================================================================

def run_pipeline(project_ids: Optional[List[str]] = None,
                 input_dir: Optional[Path] = None,
                 skip_kg: bool = False,
                 force_kg: bool = False):
    """运行端到端 pipeline"""
    print("=" * 60)
    print("  数据汇交 Pipeline — KG驱动")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Schema: 6实体(TOPIC/OBJECT/METRIC/METHOD/EQUIPMENT/ACHIEVEMENT)")
    print(f"          7关系(归属/考核/产出/采用/测试/改进/包含)")
    print("=" * 60)

    # ---- 收集项目文件 ----
    if input_dir is None:
        input_dir = BASE_DIR / "output" / "project_cleaned"

    if project_ids:
        # 按指定ID匹配文件
        project_files = []
        for fp in sorted(input_dir.glob("*.txt")):
            content = fp.read_text(encoding="utf-8")
            m = __import__("re").search(r'项目ID:\s*(\d+)', content)
            if m and m.group(1) in project_ids:
                project_files.append(fp)
    else:
        project_files = sorted(input_dir.glob("*.txt"))

    if not project_files:
        print("[错误] 未找到项目文件")
        return

    print(f"\n项目数: {len(project_files)}")
    PIPELINE_OUTPUT.mkdir(parents=True, exist_ok=True)
    all_results = []

    for fp in project_files:
        content = fp.read_text(encoding="utf-8")
        pid_match = __import__("re").search(r'项目ID:\s*(\d+)', content)
        name_match = __import__("re").search(r'项目名称:\s*(.+)', content)
        if not pid_match:
            print(f"\n[跳过] 无法解析: {fp.name}")
            continue
        pid = pid_match.group(1)
        pname = name_match.group(1).strip() if name_match else fp.stem

        print(f"\n{'='*50}")
        print(f"项目: {pname}")
        print(f"  ID: {pid}")

        # ---- Phase 1: KG 抽取 ----
        kg_dir = KG_OUTPUT_DIR / pid
        if not kg_dir.exists() or force_kg:
            if skip_kg:
                print("  [跳过KG]")
                continue
            print("  Phase 1: KG抽取...")
            run_kg_extraction([fp], force=True)
            # 重新检查
            if not kg_dir.exists():
                print("  [失败] KG抽取未完成")
                continue
        else:
            print(f"  Phase 1: KG已存在 ({kg_dir.name})")

        # 加载 KG
        entities = json.load(open(kg_dir / "entities.json", 'r', encoding='utf-8'))
        relations = json.load(open(kg_dir / "relations.json", 'r', encoding='utf-8'))
        print(f"  KG: {len(entities)}实体, {len(relations)}关系")

        # ---- Phase 2: 数据集推理 ----
        print("  Phase 2: 数据集推理...")
        datasets = infer_datasets(entities, relations, pid)
        print(f"  推理出 {len(datasets)} 个数据集候选")

        # 保存推理结果
        inference_dir = PIPELINE_OUTPUT / pid
        inference_dir.mkdir(parents=True, exist_ok=True)
        json.dump(datasets, open(inference_dir / "datasets.json", 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)

        # ---- Phase 3: CSV 导出 ----
        print("  Phase 3: CSV导出...")
        csv_path = export_to_csv(datasets, pid, PIPELINE_OUTPUT)

        # ---- 3D 评测摘要 ----
        print_evaluation_summary(datasets, entities, relations, pid)

        all_results.append({
            "project_id": pid,
            "project_name": pname,
            "entity_count": len(entities),
            "relation_count": len(relations),
            "dataset_count": len(datasets),
            "csv": str(csv_path),
        })

    # 全局摘要
    print(f"\n{'='*60}")
    print("Pipeline 完成!")
    for r in all_results:
        print(f"  {r['project_name']}: {r['entity_count']}实体 "
              f"→ {r['dataset_count']}数据集 [{r['csv']}]")

    # 保存 manifest
    manifest = PIPELINE_OUTPUT / "manifest.json"
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "total": len(all_results),
        "results": all_results,
    }, open(manifest, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"摘要: {manifest}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="数据汇交 Pipeline")
    parser.add_argument("--input", default=None, help="项目文件目录")
    parser.add_argument("--project", nargs="+", default=None, help="指定项目ID")
    parser.add_argument("--skip-kg", action="store_true", help="跳过KG抽取")
    parser.add_argument("--force-kg", action="store_true", help="强制重新抽取KG")
    args = parser.parse_args()

    input_dir = Path(args.input) if args.input else None
    run_pipeline(
        project_ids=args.project,
        input_dir=input_dir,
        skip_kg=args.skip_kg,
        force_kg=args.force_kg,
    )
