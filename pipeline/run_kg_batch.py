#!/usr/bin/env python3
"""
KG 批处理构建 — 断点续跑包装器

调用 kg_builder.py 的核心函数，逐个处理项目。
支持断点续跑、分批处理、进度估算。

用法:
  python pipeline/run_kg_batch.py --input-dir output/project_chunks_cleaned/project_chunk_00
  python pipeline/run_kg_batch.py --input-dir output/project_chunks_cleaned/project_chunk_00 --max-projects 10
  python pipeline/run_kg_batch.py --input-dir output/project_chunks_cleaned/project_chunk_00 --resume

输出: output/kg_ontology/{project_id}/...
"""
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# 从 kg_builder 导入核心函数
sys.path.insert(0, str(BASE_DIR / "pipeline"))
from kg_builder import (
    process_project, parse_project_file,
    ENTITY_TYPES, RELATION_TYPES,
)

DEFAULT_INPUT_DIR = BASE_DIR / "output" / "project_cleaned"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output" / "kg_ontology"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="KG Batch Builder with Resume")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                        help="输入目录 (默认 output/project_cleaned)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="输出目录 (默认 output/kg_ontology)")
    parser.add_argument("--max-projects", type=int, default=0,
                        help="处理项目数上限 (0=全部)")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已处理的项目")
    parser.add_argument("--start-from", type=int, default=0,
                        help="从第N个项目开始 (0-indexed)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="LLM 请求并发数 (默认8, 设1=串行)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("KG Batch Builder")
    print("=" * 60)
    print(f"实体类型: {len(ENTITY_TYPES)} 种")
    print(f"关系谓词: {len(RELATION_TYPES)} 种")
    print(f"输入: {input_dir}")
    print(f"输出: {output_dir}")
    print(f"断点续跑: {'是' if args.resume else '否'}")
    print(f"并发数: {args.concurrency}")

    if not input_dir.exists():
        print(f"[错误] 输入目录不存在: {input_dir}")
        sys.exit(1)

    all_files = sorted(input_dir.glob("*.txt"))
    if args.start_from > 0:
        all_files = all_files[args.start_from:]
        print(f"跳过前 {args.start_from} 个项目")

    files = all_files
    if args.max_projects > 0 and args.max_projects < len(files):
        files = files[:args.max_projects]

    print(f"\n输入文件: {len(all_files)} 个, 本次处理: {len(files)} 个")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    # 加载已有 manifest
    results = []
    processed_ids = set()
    if args.resume and manifest_path.exists():
        try:
            results = json.loads(manifest_path.read_text(encoding="utf-8"))
            processed_ids = {r["project_id"] for r in results}
            print(f"已加载 {len(results)} 条已有结果")
        except Exception as e:
            print(f"  manifest 加载失败: {e}, 从头开始")

    # 也检查 output_dir 下已有的子目录
    if args.resume:
        for d in output_dir.iterdir():
            if d.is_dir() and (d / "summary.json").exists():
                pid = d.name
                if pid not in processed_ids:
                    try:
                        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
                        results.append(s)
                        processed_ids.add(pid)
                    except:
                        pass
        print(f"扫描目录后: {len(results)} 个已完成项目")

    total_start = time.time()
    completed = 0
    skipped = 0
    failed = 0

    for idx, fp in enumerate(files, 1):
        proj = parse_project_file(fp)
        if not proj:
            print(f"  [{idx}/{len(files)}] {fp.name}: 解析失败, 跳过")
            failed += 1
            continue

        pid = proj["id"]

        # 断点续跑: 跳过已处理
        if args.resume and pid in processed_ids:
            print(f"  [{idx}/{len(files)}] {proj['name'][:35]} 跳过 (已处理)")
            skipped += 1
            continue

        # 检查 output 目录
        summary_path = output_dir / pid / "summary.json"
        if args.resume and summary_path.exists():
            try:
                existing = json.loads(summary_path.read_text(encoding="utf-8"))
                results.append(existing)
                processed_ids.add(pid)
                print(f"  [{idx}/{len(files)}] {proj['name'][:35]} 跳过 (已有输出)")
                skipped += 1
                # 保存进度更新
                manifest_path.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                continue
            except:
                pass

        # 处理
        proj_start = time.time()
        print(f"  [{idx}/{len(files)}] {proj['name'][:35]} ({pid})")
        print(f"      正文: {len(proj['text'])} 字")

        try:
            r = process_project(proj, output_dir, concurrency=args.concurrency)
            elapsed = time.time() - proj_start
            results.append(r)
            processed_ids.add(pid)
            completed += 1

            # 写 manifest
            manifest_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8")

            # 进度
            done = idx
            elapsed_total = time.time() - total_start
            avg_per = elapsed_total / done
            remaining = len(files) - idx
            eta = avg_per * remaining / 60
            print(f"      耗时 {elapsed:.0f}s, 累计 {done}/{len(files)}, "
                  f"剩余约 {eta:.1f}min")

        except Exception as e:
            print(f"      处理失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            # 保存进度以防后续也失败
            manifest_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8")

    # 总结
    total_elapsed = (time.time() - total_start) / 60
    print(f"\n{'=' * 60}")
    print(f"处理完成! 总耗时 {total_elapsed:.1f} min")
    print(f"  成功: {completed}")
    print(f"  跳过: {skipped}")
    print(f"  失败: {failed}")
    print(f"  总计结果: {len(results)} 个项目")
    print(f"  输出: {output_dir}")

    # 最终 manifest
    manifest_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
