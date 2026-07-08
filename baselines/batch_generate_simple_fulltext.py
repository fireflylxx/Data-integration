"""
Phase 1b: Simple + 全文方案批量生成
=====================================
调用 LLM 直接为项目的 KPI 生成数据集名称，输入包含项目任务书全文作为上下文。

用法:
  python baselines/batch_generate_simple_fulltext.py
  python baselines/batch_generate_simple_fulltext.py --resume
  python baselines/batch_generate_simple_fulltext.py --max-projects 3

输出:
  output/comparison_results/simple_fulltext_names.json
"""
import json, sys, time, re, argparse, os, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Windows GBK console workaround
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "pipeline"))
from kg_builder import llm_chat, extract_json

OUTPUT_DIR = BASE_DIR / "output" / "comparison_results"
CHUNK_BASE = BASE_DIR / "output" / "project_chunks_cleaned"
KG_BASE = BASE_DIR / "output" / "kg_ontology"

MAX_CONTEXT_CHARS = 20000  # 全文截断上限，避免超 token


def resolve_chunk_dirs(chunks_arg: str):
    """解析 --chunks 参数，返回 (chunk编号列表, 目录列表)"""
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


# =============================================================================
# 原子化写入
# =============================================================================
def atomic_write_json(data, path: Path):
    tmp = path.with_suffix(".tmp." + path.name)
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# =============================================================================
# KPI 过滤
# =============================================================================
def filter_technical_kpis(kpis: list) -> list:
    exclude_patterns = [
        r'发表.*篇', r'申请.*项.*专利', r'培养.*研究生',
        r'阶段[：:]', r'^\d{4}年', r'表\s+\d+', r'指标值[/]状态',
        r'成果、考核指标', r'项目[合作目标]', r'合作目标',
        r'评测方式', r'成功条件', r'验收方式',
        r'（[一二三四五六七八九十]）',
        r'^[（(]?\d+[)）]?$',
        r'技术指标', r'序号', r'指标名称',
    ]
    compiled = [re.compile(p) for p in exclude_patterns]
    strong_re = re.compile(
        r'[≤≥<>=]\s*\d+\.?\d*'
        r'|\d+\.?\d*\s*[%°℃ΩWmµnML]'
        r'|\d+\s*ppm'
        r'|\d+\s*[GgMmkK]?[Hh]?[zWVA]'
        r'|\d+[×xX]\d+'
    )
    strong_keywords = [
        '稳定性', '精度', '灵敏度', '分辨率', '重复性', '一致性',
        '均匀性', '可靠性', '转换效率', '透过率', '反射率',
        '速度', '带宽', '功耗', '容量', '功率', '强度',
        '密度', '浓度', '温度', '压力', '波长', '频率', '误差', '噪声',
        'MEMS', 'CMOS', 'LED', 'CIGS', 'BDD',
    ]
    filtered = []
    for kpi in kpis:
        kpi = kpi.strip()
        if len(kpi) < 10 or len(kpi) > 150:
            continue
        if any(p.search(kpi) for p in compiled):
            continue
        if strong_re.search(kpi) or any(kw in kpi for kw in strong_keywords):
            filtered.append(kpi)
    return filtered


# =============================================================================
# 项目发现（包含全文加载）
# =============================================================================
def discover_projects(chunk_dirs: list):
    """扫描多个 chunk 目录，发现所有项目及其 KPI，并加载全文"""
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

            # 加载 KPI
            kpi_path = KG_BASE / pid / "kpis.json"
            kpis = []
            if kpi_path.exists():
                try:
                    kpi_data = json.loads(kpi_path.read_text(encoding="utf-8"))
                    for k in kpi_data:
                        kpi_text = k.get("kpi") or k.get("description") or k.get("text") or ""
                        if kpi_text and len(kpi_text) > 5:
                            kpis.append(kpi_text)
                except Exception as e:
                    print(f"  警告: {pid} 读取kpis.json失败: {e}")

            filtered = filter_technical_kpis(kpis)
            kpi_priority = sorted(filtered, key=lambda x: (
                0 if '考核指标' in x or ('效率' in x and '提高' not in x) else
                1 if '指标' in x else 2
            ))
            kpis = kpi_priority[:8]

            # 加载全文
            full_text = f.read_text(encoding="utf-8", errors="ignore")
            if len(full_text) > MAX_CONTEXT_CHARS:
                full_text = full_text[:MAX_CONTEXT_CHARS] + "\n\n[（下文因长度限制已截断）]"

            projects.append({
                "pid": pid,
                "name": pname,
                "filename": f.name,
                "full_text": full_text,
                "kpis": kpis,
                "kpi_count": len(kpis),
            })

    return projects


# =============================================================================
# Simple + 全文 命名
# =============================================================================
SIMPLE_FULLTEXT_NAMING_PROMPT = """你是一个科研数据集命名专家。下面是一份项目任务书全文和其中一项考核指标(KPI)，请根据项目背景信息为该项KPI生成合适的数据集名称。

## 规则
- 名称必须包含KPI原文中的核心对象和参数
- 结合项目任务书背景，使名称更准确地反映数据集的来源和用途
- 以"数据集"或"数据"结尾
- 名称长度15-35字
- 只输出名称，不要解释

## 输出格式
{"name_cn": "..."}"""

SIMPLE_FULLTEXT_RETRY_PROMPT = """根据项目任务书背景和考核指标，生成数据集名称，以"数据集"结尾，只输出JSON。
{"name_cn": "..."}"""


def _try_extract_name(resp: str):
    """从 LLM 响应中尝试提取数据集名称"""
    cleaned = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()
    if not cleaned:
        return None
    result = extract_json(cleaned)
    if result and "name_cn" in result:
        return result["name_cn"]
    m = re.search(r'"name_cn"\s*:\s*"([^"]+)"', cleaned)
    if m:
        return m.group(1)
    for line in cleaned.split('\n'):
        line = line.strip().strip('"').strip("'")
        if line and not line.startswith('<') and len(line) > 4:
            return line[:35]
    return None


def simple_fulltext_name(kpi_description: str, full_text: str) -> str:
    """Simple + 全文版本: 输入KPI+项目全文，LLM直接命名"""
    truncated_text = full_text if len(full_text) <= MAX_CONTEXT_CHARS else full_text[:MAX_CONTEXT_CHARS]
    user = f"【项目任务书】\n{truncated_text}\n\n【考核指标】\n{kpi_description}"
    resp = llm_chat(SIMPLE_FULLTEXT_NAMING_PROMPT, user, max_tokens=800, temperature=0.3)
    name = _try_extract_name(resp)
    if name:
        return name

    # 重试
    retry_user = f"{truncated_text[-2000:]}\n\n{kpi_description}"
    resp2 = llm_chat(SIMPLE_FULLTEXT_RETRY_PROMPT, retry_user, max_tokens=800, temperature=0.5)
    name = _try_extract_name(resp2)
    if name:
        return name

    return "[生成失败]"


# =============================================================================
# 主流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 1b: Simple + 全文方案批量生成")
    parser.add_argument("--resume", action="store_true", help="从上次断点继续")
    parser.add_argument("--max-projects", type=int, default=0, help="最多处理N个项目（调试用）")
    parser.add_argument("--chunks", type=str, default="00-05",
                        help="chunk 范围，如 '00-05' / '00,01,03' / '00'（默认 00-05）")
    parser.add_argument("--max-context-chars", type=int, default=20000,
                        help="全文截断字符数（默认 20000）")
    args = parser.parse_args()

    global MAX_CONTEXT_CHARS
    MAX_CONTEXT_CHARS = args.max_context_chars

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunk_labels, chunk_dirs = resolve_chunk_dirs(args.chunks)
    chunks_tag = "_".join(chunk_labels)
    output_path = OUTPUT_DIR / f"simple_fulltext_names_chunk_{chunks_tag}.json"
    print(f"Chunk 范围: {chunks_tag} ({len(chunk_dirs)} 个目录)")

    # 发现项目（含全文）
    all_projects = discover_projects(chunk_dirs)
    print(f"发现 {len(all_projects)} 个项目（含全文，截断 {MAX_CONTEXT_CHARS} 字）")

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

    remaining_projects = [p for p in all_projects if p["pid"] not in completed_pids]
    if completed_pids:
        print(f"跳过 {len(completed_pids)} 个已完成的项目，剩余 {len(remaining_projects)} 个")

    pbar = tqdm(remaining_projects, desc="Simple+全文 生成", unit="项目", ncols=100)
    for project in pbar:
        pid = project["pid"]
        pname = project["name"]
        kpis = project["kpis"]
        full_text = project["full_text"]

        if not kpis:
            completed_pids.add(pid)
            pbar.set_postfix_str(f"{pname[:20]} 无KPI")
            continue

        pbar.set_postfix_str(f"{pname[:20]} — {len(kpis)}条KPI, 全文{len(full_text)}字")

        project_results = []
        kpi_bar = tqdm(kpis, desc=f"  KPI", unit="条", leave=False, ncols=80)
        for kpi in kpi_bar:
            t0 = time.time()
            try:
                name = simple_fulltext_name(kpi, full_text)
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
    print(f"Simple + 全文方案生成完成!")
    print(f"  项目数: {stats['total_projects']}")
    print(f"  KPI数: {stats['total_kpis']}")
    print(f"  成功率: {stats['success_rate']}% ({stats['success_count']}/{stats['total_kpis']})")
    print(f"  输出: {output_path}")


if __name__ == "__main__":
    main()
