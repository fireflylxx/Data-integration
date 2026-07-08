"""
Section-aware RAG 基线：按章节切分检索
=======================================
从原始项目文件按章节标题（一、二、三、四、...）重新切分 chunk，
按 KPI 所属课题（topic_id）筛选对应章节的 chunks，
再用关键词重叠排序取 top-5 作为 LLM 上下文生成数据集名称。

用法:
  python baselines/baseline_section_rag.py --chunks 00-05
  python baselines/baseline_section_rag.py --chunks 00 --max-projects 3
  python baselines/baseline_section_rag.py --chunks 00 --resume

输出:
  output/comparison_results/section_rag_names_chunk_{tag}.json
"""

import json, sys, time, re, argparse, os, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from common_kpi import (
    resolve_chunk_dirs, discover_projects, atomic_write_json,
    llm_chat, _try_extract_name,
)

OUTPUT_DIR = BASE_DIR / "output" / "comparison_results"
KG_BASE = BASE_DIR / "output" / "kg_ontology"
CHUNK_BASE = BASE_DIR / "output" / "project_chunks_cleaned"

SECTION_RAG_NAMING_PROMPT = """你是一个科研数据集命名专家。请根据考核指标(KPI)和项目原文参考，生成数据集名称。

## 规则
- 名称必须包含KPI原文中的核心对象和参数
- 以"数据集"或"数据"结尾
- 名称长度15-35字
- 只输出名称，不要解释

## 输出格式
{"name_cn": "..."}"""

# 章节标题模式（按优先级匹配）
SECTION_PATTERNS = [
    (re.compile(r'^一[、．.]'), "section_1"),
    (re.compile(r'^二[、．.]'), "section_2"),
    (re.compile(r'^三[、．.]'), "section_3"),
    (re.compile(r'^四[、．.]'), "section_4"),
    (re.compile(r'^五[、．.]'), "section_5"),
    (re.compile(r'^六[、．.]'), "section_6"),
]


def find_project_file(pid: str, chunk_dirs: list) -> str:
    """在 chunk 目录中查找项目原始全文"""
    for d in chunk_dirs:
        for f in d.glob(f"*_{pid}.txt"):
            return f.read_text(encoding="utf-8", errors="ignore")
    return ""


def chunk_by_sections(full_text: str):
    """按章节标题切分全文，返回 (chunks, section_labels)

    Args:
        full_text: 项目完整原文

    Returns:
        chunks: [str] — 每个元素是一个章节的文本
        section_labels: [str] — 对应的章节标签
    """
    lines = full_text.split("\n")
    sections = []  # [(section_label, [lines])]
    current_section = "section_0"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        matched = False
        for pattern, label in SECTION_PATTERNS:
            if pattern.match(stripped):
                # 保存上一节
                if current_lines:
                    sections.append((current_section, current_lines))
                # 开始新节
                current_section = label
                current_lines = [line]
                matched = True
                break
        if not matched:
            current_lines.append(line)

    # 最后一节
    if current_lines:
        sections.append((current_section, current_lines))

    # 合并为 chunks，每个 section 一个 chunk
    chunks = []
    labels = []
    for label, sec_lines in sections:
        text = "\n".join(sec_lines).strip()
        if len(text) < 50:  # 太短的节跳过（如仅有标题的页眉）
            continue
        chunks.append(text)
        labels.append(label)

    return chunks, labels


def load_topic_info(pid: str):
    """加载课题信息

    Returns:
        topics: [{id, name}, ...]
        kpi_entries: [{id, description, topic_id}, ...]
    """
    topics = []
    kpi_entries = []
    tpath = KG_BASE / pid / "topics.json"
    if tpath.exists():
        try:
            topics = json.loads(tpath.read_text(encoding="utf-8"))
        except Exception:
            topics = []

    kpath = KG_BASE / pid / "kpis.json"
    if kpath.exists():
        try:
            kpi_entries = json.loads(kpath.read_text(encoding="utf-8"))
        except Exception:
            kpi_entries = []

    return topics, kpi_entries


def match_topics_to_sections(chunks: list, labels: list, topics: list) -> dict:
    """将课题匹配到对应的章节 chunk

    Returns:
        {topic_id: [matching_chunk_indices]}
    """
    topic_chunks = defaultdict(list)
    for topic in topics:
        tid = topic.get("id", 0)
        tname = topic.get("name", "")
        if not tname or len(tname) < 2:
            continue

        # 从课题名提取 4+ 字的关键词
        keywords = set()
        name_clean = re.sub(r'[（(][^）)]*[）)]', '', tname)
        for i in range(len(name_clean)):
            for j in range(i + 4, min(i + 12, len(name_clean) + 1)):
                seg = name_clean[i:j]
                if len(seg) >= 4:
                    keywords.add(seg)

        for idx, chunk in enumerate(chunks):
            for kw in keywords:
                if kw in chunk:
                    topic_chunks[tid].append(idx)
                    break

    return dict(topic_chunks)


def lookup_kpi_topic_id(kpi_description: str, kpi_entries: list) -> int:
    """查找某条 KPI 对应的 topic_id"""
    best_id = 0
    best_score = 0
    kpi_prefix = kpi_description[:30]

    for entry in kpi_entries:
        desc = entry.get("description", "")
        tid = entry.get("topic_id", 0)
        if not desc or not tid:
            continue
        if desc in kpi_description:
            return tid
        common = len(set(kpi_prefix) & set(desc[:30]))
        if common > best_score:
            best_score = common
            best_id = tid

    return best_id


def section_rag_name(kpi_description: str, chunks: list, labels: list,
                     topic_info: tuple, topic_chunks: dict) -> str:
    """Section-aware RAG: 章节过滤 → 关键词排序 → LLM 命名"""
    topics, kpi_entries = topic_info

    # 1. 确定 KPI 所属课题
    topic_id = lookup_kpi_topic_id(kpi_description, kpi_entries)

    # 2. 筛选候选 chunks
    candidate_indices = set()
    if topic_id and topic_id in topic_chunks:
        candidate_indices = set(topic_chunks[topic_id])

    # 如果太少，扩展到同章节的其他 chunks
    if len(candidate_indices) < 2 and topic_id:
        target_sections = set()
        for idx in candidate_indices:
            target_sections.add(labels[idx])
        for idx, sec in enumerate(labels):
            if sec in target_sections:
                candidate_indices.add(idx)

    # 如果还是太少，回退到全部
    if len(candidate_indices) < 2:
        candidate_indices = set(range(len(chunks)))

    # 3. KPI 关键词重叠评分
    kpi_tokens = set()
    for t in re.split(r'[,，;；、：:\s（）()/]', kpi_description):
        t = t.strip()
        if len(t) >= 2:
            kpi_tokens.add(t)

    scored = []
    for idx in candidate_indices:
        if idx >= len(chunks):
            continue
        chunk = chunks[idx]
        overlap = sum(1 for token in kpi_tokens if token in chunk)
        score = overlap / max(1, len(kpi_tokens))
        scored.append((idx, score, chunk))

    scored.sort(key=lambda x: -x[1])
    top_chunks = scored[:5]

    # 4. 构建上下文
    context_parts = []
    for i, (idx, score, chunk_text) in enumerate(top_chunks):
        sec = labels[idx] if idx < len(labels) else ""
        section_tag = f"[{sec}]" if sec else ""
        # 章节 chunk 可能很长，截断至 800 字
        truncated = chunk_text[:800]
        context_parts.append(f"{section_tag} [原文参考 {i+1}] {truncated}")

    context = "\n\n".join(context_parts) if context_parts else ""

    # 5. LLM 命名
    if context.strip():
        user = f"## 项目原文参考（章节过滤结果）\n{context}\n\n## 考核指标\n{kpi_description}"
    else:
        user = f"## 考核指标\n{kpi_description}"

    resp = llm_chat(SECTION_RAG_NAMING_PROMPT, user, max_tokens=800, temperature=0.3)
    name = _try_extract_name(resp)
    if name:
        return name

    resp2 = llm_chat(SECTION_RAG_NAMING_PROMPT, f"考核指标: {kpi_description}",
                     max_tokens=800, temperature=0.5)
    name = _try_extract_name(resp2)
    if name:
        return name

    return "[生成失败]"


def main():
    parser = argparse.ArgumentParser(description="Section-aware RAG: 按章节切分+LLM 命名")
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
    output_path = OUTPUT_DIR / f"section_rag_names_chunk_{chunks_tag}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Chunk 范围: {chunks_tag} ({len(chunk_dirs)} 个目录)")

    # 注意：Section-aware 直接从原始文件按章节切分，不需要 kg_ontology 的 chunks
    # 但需要 kpis.json 和 topics.json 做课题映射
    all_projects = discover_projects(chunk_dirs, require_kg=False)
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

    remaining = [p for p in all_projects if p["pid"] not in completed_pids]
    if completed_pids:
        print(f"跳过 {len(completed_pids)} 个已完成的项目，剩余 {len(remaining)} 个")

    pbar = tqdm(remaining, desc="Section-RAG 生成", unit="项目", ncols=100)
    for project in pbar:
        pid = project["pid"]
        pname = project["name"]
        kpis = project["kpis"]

        if not kpis:
            completed_pids.add(pid)
            pbar.set_postfix_str(f"{pname[:20]} 无KPI")
            continue

        # 从原始全文按章节切分
        full_text = find_project_file(pid, chunk_dirs)
        if not full_text:
            pbar.set_postfix_str(f"{pname[:20]} 无原文")
            for kpi in kpis:
                all_results.append({
                    "project_id": pid, "project_name": pname, "kpi": kpi,
                    "name": "[无原文]", "success": False,
                    "n_chunks": 0, "n_retrieved": 0, "time": 0,
                })
            completed_pids.add(pid)
            continue

        chunks, labels = chunk_by_sections(full_text)
        if not chunks:
            pbar.set_postfix_str(f"{pname[:20]} 切分失败")
            for kpi in kpis:
                all_results.append({
                    "project_id": pid, "project_name": pname, "kpi": kpi,
                    "name": "[章节切分失败]", "success": False,
                    "n_chunks": 0, "n_retrieved": 0, "time": 0,
                })
            completed_pids.add(pid)
            continue

        # 课题映射
        topics, kpi_entries = load_topic_info(pid)
        topic_chunks = match_topics_to_sections(chunks, labels, topics)
        topic_info = (topics, kpi_entries)

        unique_sections = len(set(labels))
        pbar.set_postfix_str(
            f"{pname[:20]} {len(chunks)}节 "
            f"{unique_sections}种 — {len(kpis)}条KPI")

        # 并行处理 KPI
        def process_one(kpi_data):
            kidx, kpi = kpi_data
            t0 = time.time()
            try:
                name = section_rag_name(kpi, chunks, labels, topic_info, topic_chunks)
            except Exception as e:
                name = f"[错误:{e}]"
            elapsed = time.time() - t0
            return {
                "project_id": pid, "project_name": pname, "kpi": kpi,
                "name": name, "success": not name.startswith("["),
                "n_chunks": len(chunks), "n_retrieved": 5,
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
    print(f"Section-aware RAG 方案完成!")
    print(f"  项目数: {stats['total_projects']}")
    print(f"  KPI数: {stats['total_kpis']}")
    print(f"  成功率: {stats['success_rate']}% ({stats['success_count']}/{stats['total_kpis']})")
    print(f"  耗时: {elapsed/60:.1f} 分")
    print(f"  输出: {output_path}")


if __name__ == "__main__":
    main()
