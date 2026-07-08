"""
Baseline A: Rule-based 命名方案
=================================
纯规则，无 LLM 调用。直接从 KPI 文本中提取对象+参数拼接为数据集名称。

用法:
  python baselines/baseline_rule.py --chunks 00
  python baselines/baseline_rule.py --chunks 00 --max-projects 5

输出:
  output/comparison_results/rule_names_chunk_00.json
"""

import json, sys, time, re, argparse, os, io
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from common_kpi import resolve_chunk_dirs, discover_projects, atomic_write_json

OUTPUT_DIR = BASE_DIR / "output" / "comparison_results"

# =============================================================================
# 纯规则 KPI 解析（无 LLM）
# =============================================================================

# 常见参数词
PARAM_KEYWORDS = [
    "效率", "精度", "稳定性", "灵敏度", "分辨率", "重复性", "一致性",
    "均匀性", "可靠性", "透过率", "反射率", "速度", "带宽", "功耗",
    "容量", "功率", "强度", "密度", "浓度", "温度", "压力", "频率",
    "波长", "误差", "噪声", "失真", "时延", "成品率", "良率",
    "纯度", "转化率", "产率", "收率", "寿命", "衰减率",
    "级数", "通量", "处理量",
]

# 常见对象/领域词
DOMAIN_KEYWORDS = [
    "电池", "薄膜", "光纤", "传感器", "芯片", "电路", "器件",
    "系统", "平台", "装备", "设备", "仪器", "材料", "试剂",
    "溶液", "催化剂", "电极", "激光", "光谱", "波导", "LED",
    "MEMS", "CMOS", "CIGS", "BDD",
]


def rule_extract_kpi(kpi_text: str) -> dict:
    """纯规则从 KPI 文本中提取 object 和 parameter"""
    # 去除数值和单位
    cleaned = re.sub(r'[≤≥<>=]\s*\d+\.?\d*\s*[%°℃ΩWmµnML]?', '', kpi_text)
    cleaned = re.sub(r'\d+\.?\d*', '', cleaned)
    cleaned = re.sub(r'\(.*?\)', '', cleaned)  # 去掉括号内容

    obj, param = "", ""

    # 1. 提取对象：找领域关键词
    for kw in DOMAIN_KEYWORDS:
        if kw in cleaned:
            idx = cleaned.index(kw)
            start = max(0, idx - 8)
            obj = cleaned[start:idx + len(kw)].strip()
            obj = re.sub(r'^[，。；、：\s]+', '', obj)
            if len(obj) > 20:
                obj = obj[-20:]
            break

    if not obj:
        m = re.search(r'[：:]\s*([^，。；≤≥<>\d]+)', cleaned)
        if m:
            obj = m.group(1).strip()[:20]

    # 2. 提取参数
    for kw in PARAM_KEYWORDS:
        if kw in cleaned:
            param = kw
            break

    return {"object": obj, "parameter": param}


def rule_name(kpi_text: str) -> str:
    """纯规则生成数据集名称"""
    parsed = rule_extract_kpi(kpi_text)
    obj = parsed["object"]
    param = parsed["parameter"]

    if obj and param:
        return f"{obj}{param}数据集"
    elif obj:
        return f"{obj}性能数据集"
    elif param:
        return f"{param}测试数据集"
    else:
        prefix = re.sub(r'[≤≥<>=].*', '', kpi_text).strip()[:6]
        return f"{prefix}考核数据集"


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline A: Rule-based 命名")
    parser.add_argument("--chunks", type=str, default="00-05",
                        help="chunk 范围（默认 00-05）")
    parser.add_argument("--max-projects", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="从上次断点继续")
    args = parser.parse_args()

    chunk_labels, chunk_dirs = resolve_chunk_dirs(args.chunks)
    chunks_tag = "_".join(chunk_labels)
    output_path = OUTPUT_DIR / f"rule_names_chunk_{chunks_tag}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 使用共享的 discover_projects（无需 KG 数据）
    all_projects = discover_projects(chunk_dirs, require_kg=False)
    print(f"发现 {len(all_projects)} 个项目")

    if args.max_projects:
        all_projects = all_projects[:args.max_projects]

    # 恢复进度
    completed_pids = set()
    all_results = []
    if args.resume and output_path.exists():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        all_results = saved.get("results", [])
        completed_pids = set(saved.get("completed_pids", []))
        print(f"恢复进度: {len(completed_pids)} 个项目已完成")

    remaining = [p for p in all_projects if p["pid"] not in completed_pids]
    t_start = time.time()

    pbar = tqdm(remaining, desc="Rule 命名", unit="项目", ncols=100)
    for project in pbar:
        pid = project["pid"]
        kpis = project["kpis"]

        for kpi in kpis:
            name = rule_name(kpi)
            all_results.append({
                "project_id": pid,
                "project_name": project["name"],
                "kpi": kpi,
                "name": name,
                "success": True,
            })

        completed_pids.add(pid)
        done = len(completed_pids)
        save_data = {
            "completed_pids": list(completed_pids),
            "results": all_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stats": {"total": len(all_projects), "completed": done, "total_kpis": len(all_results)},
        }
        atomic_write_json(save_data, output_path)
        pbar.set_postfix_str(f"{done}/{len(all_projects)}")

    # 最终保存
    stats = {
        "total": len(all_projects),
        "completed": len(completed_pids),
        "total_kpis": len(all_results),
    }
    atomic_write_json({"completed_pids": list(completed_pids), "results": all_results,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "stats": stats}, output_path)

    elapsed = time.time() - t_start
    print(f"\nRule-based 完成! {stats['total_kpis']} 条KPI, 耗时 {elapsed:.1f} 秒")
    print(f"输出: {output_path}")


if __name__ == "__main__":
    main()
