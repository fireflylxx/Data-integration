"""
三维度评估: Simple vs KG Pipeline 对比示例
==========================================
对精选的6组KPI进行完整的3D评估（语义保真度50% + 原文可溯性35% + 命名规范性15%）
"""
import json, sys, re, io
from pathlib import Path

# Windows GBK console workaround
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "pipeline"))

from evaluator_3d_refined import evaluate_name, compute_regularity

# 选定的6组对比示例（来自完整的435-KPI批量结果）
EXAMPLES = [
    {
        "project_id": "1646707075016945664",
        "kpi": "考核指标：完成生物降解材料注塑过程的数值模拟，求解效率提高 40%以上",
        "simple": "生物降解材料注塑过程数值模拟求解效率提升40%数据集",
        "kg": "生物降解材料注塑过程求解效率数值模拟仿真数据集",
        "issue_type": "目标值残留 → 去除",
    },
    {
        "project_id": "1646724823658876928",
        "kpi": "在柔性衬底上电池光电转换效率 ≥ 22%（面积0.5cm²）；利用阻挡层实现CIGS吸收层",
        "simple": "[生成失败]",
        "kg": "柔性衬底上电池光电转换效率标准太阳模拟器测试数据集",
        "issue_type": "Simple生成失败 → KG成功生成",
    },
    {
        "project_id": "1646724823658876928",
        "kpi": "高效率电池组件工艺关键技术，包括大面积 CIGS 薄膜均匀共蒸发镀膜技术",
        "simple": "大面积CIGS薄膜均匀共蒸发镀膜高效率电池组件工艺数据集",
        "kg": "大面积CIGS薄膜均匀共蒸发镀膜2D-XRF测试数据集",
        "issue_type": "语义漂移C（方法不准确→KG检索到具体方法）",
    },
    {
        "project_id": "1646707075016945664",
        "kpi": "任务 3：搭建针对生物纤维材料注塑过程的实验平台",
        "simple": "生物纤维材料注塑过程实验平台数据集",
        "kg": "生物纤维材料注塑过程实验平台验证数据集",
        "issue_type": "语义漂移D（活动类型混淆→KG明确验证类型）",
    },
    {
        "project_id": "1648590377755713536",
        "kpi": "面向 5G 应用的光传输核心芯片与模块 — 光收发模块测试",
        "simple": "光传输核心芯片与模块5G应用光收发模块测试数据集",
        "kg": "5G光传输核心芯片与模块光收发模块眼图测试数据集",
        "issue_type": "语义漂移C（KG从FAISS检索到具体测试方法'眼图'）",
    },
    {
        "project_id": "1648982109899026432",
        "kpi": "考核指标：MEMS加速度计零偏稳定性≤0.01°/h，全温区范围-40°C~85°C",
        "simple": "MEMS加速度计零偏稳定性≤0.01°/h全温区测试数据集",
        "kg": "MEMS加速度计零偏稳定性全温区高低温箱标定测试数据集",
        "issue_type": "目标值残留+方法缺失→KG补充方法并去除目标值",
    },
]


def load_project_chunks(pid: str):
    """加载项目的原文块"""
    chunk_dir = BASE_DIR / "output" / "kg_ontology" / pid / "chunks"
    chunks = []
    if chunk_dir.exists():
        for f in sorted(chunk_dir.glob("chunk_*.txt")):
            chunks.append(f.read_text(encoding="utf-8", errors="ignore"))
    return chunks


def main():
    print("=" * 100)
    print("三维度精细评估: Simple (旧方案) vs KG Pipeline")
    print("评估公式: Score = 0.50×语义保真度 + 0.35×原文可溯性 + 0.15×命名规范性")
    print("=" * 100)

    for idx, ex in enumerate(EXAMPLES, 1):
        pid = ex["project_id"]
        kpi = ex["kpi"]
        simple_name = ex["simple"]
        kg_name = ex["kg"]
        issue = ex["issue_type"]

        print(f"\n{'─' * 100}")
        print(f"示例 {idx}: {issue}")
        print(f"{'─' * 100}")
        print(f"KPI: {kpi}")

        # 加载原文块
        chunks = load_project_chunks(pid)
        print(f"原文块: {len(chunks)} 个")

        # 评估 Simple
        print(f"\n  Simple: {simple_name}")
        if simple_name.startswith("["):
            print(f"    语义保真度: N/A (生成失败)")
            print(f"    原文可溯性: N/A")
            print(f"    命名规范性: N/A")
            print(f"    综合得分:   0.000")
            print(f"    诊断:       生成失败")
        else:
            sr = evaluate_name(simple_name, kpi, chunks=chunks)
            print(f"    语义保真度: {sr.semantic_total:.3f} (对象={sr.object_score:.2f} 参数={sr.parameter_score:.2f} 方法={sr.method_score:.2f})")
            print(f"    原文可溯性: {sr.traceability_total:.3f}")
            print(f"    命名规范性: {sr.regularity_total:.3f}")
            print(f"    综合得分:   {sr.total_score:.3f}")
            print(f"    诊断:       {sr.diagnosis}")

        # 评估 KG
        print(f"\n  KG:     {kg_name}")
        kr = evaluate_name(kg_name, kpi, chunks=chunks)
        print(f"    语义保真度: {kr.semantic_total:.3f} (对象={kr.object_score:.2f} 参数={kr.parameter_score:.2f} 方法={kr.method_score:.2f})")
        print(f"    原文可溯性: {kr.traceability_total:.3f}")
        print(f"    命名规范性: {kr.regularity_total:.3f}")
        print(f"    综合得分:   {kr.total_score:.3f}")
        print(f"    诊断:       {kr.diagnosis}")

        # 差异
        if not simple_name.startswith("["):
            delta = kr.total_score - sr.total_score
            sym = "+" if delta > 0 else ""
            print(f"\n  ▶ KG改进: {sym}{delta:.3f} 分")
        else:
            print(f"\n  ▶ KG改进: Simple失败 → KG成功生成")

    # 汇总
    print(f"\n\n{'=' * 100}")
    print("评估完成")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
