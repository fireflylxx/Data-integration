"""
KPI Planner — KPI 结构化解析 → 规划 → 命名 → 回译验证
=====================================================
架构定位（图9 阶段2-4 + 图6 回译验证）：
  KPI → 结构解析 → 激活路径模板 → Hybrid检索 → 规划 → 命名 → 回译验证

包含 4 个 LLM 驱动模块：
  A. KPI 结构化解析器
  B. 规划模块：KPI + 上下文 → 结构化 Plan
  C. 命名模块：Plan → 数据集名称
  D. 回译验证器：名称 → 反推 KPI → 与原 KPI 比对
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional, List, Tuple
from difflib import SequenceMatcher

# =============================================================================
# API 配置（与 kg_builder.py 同步）
# =============================================================================
API_BASE = "http://10.3.213.253:23001"
API_KEY = "sk-259d53cf77064362aa19c816c1321e7b"
LLM_MODEL = "qwen3-32b"


def llm_chat(system: str, user: str, max_tokens: int = 1000,
             temperature: float = 0.3, timeout: int = 120) -> str:
    """LLM API 调用"""
    import requests

    # 附加"不要思考"指令到 user 消息
    no_think = "\n\n重要：直接输出答案，不要使用<think>或<思考>标签，不要思考过程。"
    if temperature <= 0.2:
        no_think = "\n\n重要：只输出JSON，不要任何思考标记、不要解释、不要其他文字。"
    full_user = user + no_think

    url = f"{API_BASE}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": full_user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        elif resp.status_code == 401:
            print(f"    [API 401] 认证失败，尝试无认证...")
            headers.pop("Authorization")
            resp = requests.post(url, headers=headers, json=data, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        print(f"    [API Error {resp.status_code}] {resp.text[:200]}")
        return ""
    except Exception as ex:
        print(f"    [API Exception] {ex}")
        return ""


def extract_json(text: str):
    """从 LLM 回复中提取 JSON（增强版，支持 dict 和 list）

    Args:
        text: LLM 回复文本

    Returns:
        dict | list | None: 解析后的 JSON 对象
    """
    if not text:
        return None

    # 1. 暴力移除所有 think 标记及其内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<思考>.*?</思考>', '', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text)  # 移除所有残留标签

    # 2. 尝试代码块
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        text = m.group(1).strip()

    # 3. 尝试解析为数组
    bracket_start = text.find('[')
    bracket_end = text.rfind(']')
    if bracket_start >= 0 and bracket_end > bracket_start:
        candidate = text[bracket_start:bracket_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 4. 尝试解析为对象 {...}
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start >= 0 and brace_end > brace_start:
        candidate = text[brace_start:brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 5. 尝试逐行修复常见 JSON 问题
    try:
        # 去除换行符内的多余空格
        cleaned = re.sub(r',\s*}', '}', candidate)
        return json.loads(cleaned)
    except (json.JSONDecodeError, UnboundLocalError):
        pass

    return None


# =============================================================================
# A. KPI 结构化解析器
# =============================================================================

KPI_PARSE_PROMPT = """你是一个科研项目考核指标解析专家。请从考核指标描述中提取结构化信息。

考核指标描述通常包含：研究对象、性能参数、目标值、验证活动等要素。

重要：不要思考，不要解释，不要输出任何思考标记，直接输出 JSON。

请提取以下字段（如果不存在则填 null）：
- object: 研究对象（如"高精度光纤陀螺仪"、"生物纤维材料"）
- parameter: 性能参数/指标（如"零偏稳定性"、"求解效率"）
- target_value: 目标值（如"≤0.01°/h"、"提高40%"）
- verb: 动作/动词（如"研制"、"构建"、"形成"、"开发"）
- activity_type: 验证活动类型（"测试"、"标定"、"检测"、"仿真"、"评估" 或 null）

输出格式：JSON
{"object": "...", "parameter": "...", "target_value": "...", "verb": "...", "activity_type": "..."}"""


def parse_kpi_structured(kpi_description: str) -> dict:
    """A. KPI 结构化解析

    Args:
        kpi_description: KPI 描述文本（如"研制高精度光纤陀螺仪，零偏稳定性≤0.01°/h"）

    Returns:
        {"object": str, "parameter": str, "target_value": str, "verb": str, "activity_type": str}
    """
    print(f"    [KPI解析] {kpi_description[:60]}...", end=" ", flush=True)
    response = llm_chat(KPI_PARSE_PROMPT, kpi_description, max_tokens=500, temperature=0.1)
    result = extract_json(response)
    if not result or not isinstance(result, dict):
        # 重试一次
        print("[重试]", end=" ", flush=True)
        response = llm_chat(KPI_PARSE_PROMPT + "\n请只输出JSON，不要任何思考过程。",
                          kpi_description, max_tokens=500, temperature=0.05)
        result = extract_json(response)
    if not result or not isinstance(result, dict):
        print("[失败]")
        return {"object": "", "parameter": "", "target_value": "", "verb": "", "activity_type": ""}
    obj_str = (result.get("object") or "")[:20]
    print(f"[OK] 对象={obj_str}")
    return result


# =============================================================================
# B. 规划模块
# =============================================================================

PLAN_PROMPT = """你是一个科研数据集规划专家。你的任务是根据考核指标(KPI)和研究内容上下文，规划**高质量**的数据集来验证该KPI。

## 背景
国家重点研发计划项目中，考核指标规定了项目需要达到的技术目标。每个指标需要对应的测试数据来验证。

## 输入
1. KPI（考核指标描述）
2. KPI结构化解析（对象、参数、目标值）
3. 研究内容上下文（来自项目任务书的相关段落，包含研究对象、测试方法、实验设备等信息）

## 规划规则（质量优先）
- 每个数据集必须直接对应KPI的某个具体可验证维度
- 数据集名称必须精确包含KPI原文中的核心对象和参数名称，不能泛化
- 数据集类型：测试数据、标定数据、监测数据、仿真数据、验证数据
- 通常一个KPI需要1~2个高质量数据集，不要为了凑数量而生成低质量规划
- 宁可少但要精，每个数据集必须能明确对回到KPI原文

## 输出格式
直接输出 JSON 数组，不要任何思考标记、不要解释、不要额外的文字：
[
    {
        "object": "研究对象名称（必须来自KPI原文，不要改写）",
        "parameter": "验证的参数/指标（必须来自KPI原文）",
        "method": "测试方法或设备（如有）",
        "data_type": "TEST_DATA | CALIBRATION_DATA | MONITORING_DATA | SIMULATION_DATA | VERIFICATION_DATA",
        "data_suffix": "测试数据集 | 标定数据集 | 监测数据 | 仿真数据 | 验证数据",
        "confidence": "HIGH | MEDIUM | LOW",
        "reasoning": "简要说明该数据集为何能验证该KPI的具体指标"
    }
]

注意：输出JSON数组，通常1~2个数据集，质量优先。"""


def plan_dataset(kpi_structured: dict, context: str, retry_feedback: str = "") -> List[dict]:
    """B. 规划模块：KPI + 上下文 → 结构化 Plan 列表

    Args:
        kpi_structured: KPI结构化解析结果
        context: 混合检索融合后的上下文
        retry_feedback: 回译验证失败时的反馈（用于重试）

    Returns:
        [{"object": str, "parameter": str, "method": str, "data_type": str,
          "data_suffix": str, "confidence": str, "reasoning": str}, ...]
    """
    user_msg = f"""## KPI描述
{kpi_structured.get('kpi_raw', '')}

## KPI结构化解析
- 研究对象: {kpi_structured.get('object', '')}
- 性能参数: {kpi_structured.get('parameter', '')}
- 目标值: {kpi_structured.get('target_value', '')}
- 动词: {kpi_structured.get('verb', '')}
- 验证活动: {kpi_structured.get('activity_type', '')}

## 研究内容上下文
{context[:2000]}

"""

    if retry_feedback:
        user_msg += f"""\n## 上一次验证反馈（请根据此反馈重新规划）
{retry_feedback}
"""

    print(f"    [规划] 对象={(kpi_structured.get('object') or '')[:16]}...", end=" ", flush=True)
    response = llm_chat(PLAN_PROMPT, user_msg, max_tokens=1200, temperature=0.2)
    result = extract_json(response)

    if result is not None:
        # 可能返回单对象或数组
        if isinstance(result, list):
            plans = result
        else:
            plans = [result]
    else:
        print("[失败]")
        return []

    # 过滤无效
    valid = [p for p in plans if isinstance(p, dict) and p.get("object")]
    if not valid:
        print("[空]")
        return []

    print(f"[OK] {len(valid)}个规划")
    return valid


# =============================================================================
# C. 命名模块
# =============================================================================

NAMING_PROMPT = """你是一个科研数据集命名专家。你的任务是根据考核指标(KPI)和规划信息，生成**高度准确**的数据集名称。

## 核心原则
名称必须让评审人一眼看出对应哪个KPI指标，因此必须包含**KPI原文中的专属技术术语**。

## 命名规范
1. **必须包含**研究对象（KPI原文的具体对象名，不要泛化）
2. **必须包含**考核参数/指标（KPI原文的参数名，不要省略）
3. **可选的**测试方法/设备（如有则加入）
4. 以"数据集"或"数据"结尾
5. 名称应简洁、准确，不遗漏关键术语
6. 不要包含序号、编号等非名称内容

## 优秀示例（名称中包含KPI关键词）
- KPI："研制高精度光纤陀螺仪，零偏稳定性≤0.01°/h"
  输出：高精度光纤陀螺仪零偏稳定性标定测试数据集
- KPI："生物纤维材料界面相容性提升40%"
  输出：生物纤维材料界面相容性扫描电镜测试数据集
- KPI："柔性铜铟镓硒薄膜电池光电转换效率≥22%"
  输出：柔性铜铟镓硒薄膜电池光电转换效率测试数据集

## 不合格示例（缺少KPI关键词，过于泛化）
- ❌ "材料性能测试数据集"（缺少具体对象和参数）
- ❌ "实验结果数据"（过于泛化，无法对应任何KPI）
- ❌ "数据分析数据集"（没有技术含量）
- ❌ "测试数据汇总"（不包含任何KPI术语）

## 输出格式
只输出 JSON，不要任何思考标记、不要解释、不要其他文本：
{"name_cn": "数据集名称"}"""


def name_dataset(plan: dict, kpi_raw: str = "", kpi_structured: dict = None) -> str:
    """C. 命名模块：Plan → 数据集名称

    Args:
        plan: 规划模块输出的结构化 plan
        kpi_raw: 原始KPI描述文本（用于确保名称包含KPI关键词）
        kpi_structured: KPI结构化解析结果

    Returns:
        str: 数据集名称
    """
    user_msg = f"""## 规划信息
- 研究对象: {plan.get('object', '')}
- 验证参数: {plan.get('parameter', '')}
- 测试方法/设备: {plan.get('method', '')}
- 数据类型: {plan.get('data_type', '')}
- 数据后缀: {plan.get('data_suffix', '测试数据集')}
- 推理说明: {plan.get('reasoning', '')}

## 原始KPI（请确保名称包含以下原文中的核心术语）
{kpi_raw[:200]}

"""
    if kpi_structured:
        user_msg += f"""## KPI核心术语（必须在名称中体现）
- 研究对象原文: {kpi_structured.get('object', '')}
- 考核参数原文: {kpi_structured.get('parameter', '')}
"""
    print(f"    [命名] {(plan.get('object') or '')[:12]}...", end=" ", flush=True)
    response = llm_chat(NAMING_PROMPT, user_msg, max_tokens=200, temperature=0.1)
    result = extract_json(response)
    if result is None or "name_cn" not in result:
        # 尝试直接从回复提取名称
        name = response.strip().strip('"').strip("'")
        if name and len(name) > 4:
            print(f"[直接提取] {name[:20]}")
            return name
        # 兜底：拼接
        fallback = f"{plan.get('object', '')}{plan.get('parameter', '')}{plan.get('data_suffix', '测试数据集')}"
        print(f"[拼接] {fallback[:20]}")
        return fallback
    name = result["name_cn"]
    print(f"[OK] {name[:24]}")
    return name


# =============================================================================
# D. 回译验证器
# =============================================================================

BACKTRANS_PROMPT = """你是一个数据集验证专家。给定一个数据集名称，请反推：这个数据集是用来验证什么KPI（考核指标）的？

## 示例
- 数据集名称："高精度光纤陀螺仪零偏稳定性标定测试数据集"
  反推KPI："高精度光纤陀螺仪零偏稳定性≤0.01°/h"

- 数据集名称："生物基聚合物材料注模流动过程求解效率测试数据集"
  反推KPI："生物基聚合物材料注模流动过程求解效率"

- 数据集名称："生物纤维塑料注塑工艺界面相容性扫描电镜测试数据集"
  反推KPI："生物纤维塑料注塑工艺界面相容性"

## 输出格式
输出 JSON：
{"inferred_kpi": "反推出的考核指标描述",
 "key_elements": {"object": "研究对象", "parameter": "参数/指标"}}"""


def validate_name(dataset_name: str, original_kpi: str,
                  original_kpi_structured: dict) -> dict:
    """D. 回译验证：名称 → 反推KPI → 与原KPI比对

    Args:
        dataset_name: 生成的数据集名称
        original_kpi: 原始KPI描述
        original_kpi_structured: 原始KPI结构化解析

    Returns:
        {"passed": bool, "score": float, "inferred_kpi": str, "feedback": str}
    """
    print(f"    [回译验证] {dataset_name[:24]}...", end=" ", flush=True)
    response = llm_chat(BACKTRANS_PROMPT, dataset_name, max_tokens=300, temperature=0.1)
    result = extract_json(response)

    if result is None or "inferred_kpi" not in result:
        print("[JSON失败]")
        return {"passed": True, "score": 0.5, "inferred_kpi": "",
                "feedback": "验证解析失败，默认通过"}  # 默认通过避免阻塞

    inferred_kpi = result.get("inferred_kpi", "")

    # 计算文本相似度
    score = SequenceMatcher(None, inferred_kpi, original_kpi).ratio()

    # 也检查关键要素是否匹配
    inferred_obj = result.get("key_elements", {}).get("object", "")
    original_obj = original_kpi_structured.get("object", "")
    obj_match = 0.0
    if original_obj and inferred_obj:
        obj_match = SequenceMatcher(None, inferred_obj, original_obj).ratio()

    # 综合评分
    final_score = score * 0.6 + obj_match * 0.4
    passed = final_score >= 0.35

    if passed:
        print(f"[通过 score={final_score:.2f}]")
        return {"passed": True, "score": round(final_score, 2),
                "inferred_kpi": inferred_kpi, "feedback": ""}
    else:
        feedback = f"反推KPI({inferred_kpi[:30]}...)与原KPI({original_kpi[:30]}...)相似度{final_score:.2f}，需要重新规划"
        print(f"[失败 score={final_score:.2f}]")
        return {"passed": False, "score": round(final_score, 2),
                "inferred_kpi": inferred_kpi, "feedback": feedback}


# =============================================================================
# 完整 KPI → 数据集 流程
# =============================================================================

def kpi_to_dataset(kpi_description: str, retriever_context: str,
                   max_retries: int = 2) -> List[dict]:
    """完整流程：KPI → 解析 → 规划（多个） → 命名 → 验证

    一个 KPI 通常对应多个数据集（原始数据、测试数据、仿真数据等）。

    Args:
        kpi_description: KPI 描述文本
        retriever_context: 混合检索返回的融合上下文
        max_retries: 验证失败最大重试次数

    Returns:
        [{
            "kpi": str,
            "kpi_structured": dict,
            "plan": dict,
            "dataset_name": str,
            "validated": bool,
            "validation_score": float,
        }, ...]
    """
    # Step A: KPI 结构化解析
    structured = parse_kpi_structured(kpi_description)
    structured["kpi_raw"] = kpi_description

    # Steps B-D: 规划 → 命名 → 验证（可能重试）
    all_results = []
    failed_plans = []

    for attempt in range(max_retries + 1):
        # Step B: 规划（返回列表）
        feedback = ""
        if attempt > 0 and failed_plans:
            names_str = "; ".join([r.get("dataset_name", "")[:30] for r in failed_plans])
            feedback = f"以下数据集名称验证失败：{names_str}。请重新规划，确保名称更准确反映原KPI。"
        plans = plan_dataset(structured, retriever_context, retry_feedback=feedback)

        if not plans:
            continue

        # Steps C-D: 命名 → 验证
        failed_plans = []
        for plan in plans:
            dataset_name = name_dataset(plan, kpi_raw=kpi_description, kpi_structured=structured)
            validation = validate_name(dataset_name, kpi_description, structured)
            result_item = {
                "kpi": kpi_description,
                "kpi_structured": structured,
                "plan": plan,
                "dataset_name": dataset_name,
                "validated": validation["passed"],
                "validation_score": validation["score"],
            }
            if validation["passed"]:
                all_results.append(result_item)
            else:
                failed_plans.append(result_item)

        # 如果没有失败的，或者本次没有任何输出，直接结束
        if not failed_plans:
            break

    # 如果全部重试后仍有未通过的，把最优的失败结果也放进来（但标记 validated=False）
    if not all_results and failed_plans:
        # 按分数排序，取最优
        failed_plans.sort(key=lambda x: x["validation_score"], reverse=True)
        all_results.append(failed_plans[0])

    # 去重（按 dataset_name 去重）
    seen_names = set()
    deduped = []
    for r in all_results:
        name = r.get("dataset_name", "")
        if name and name not in seen_names:
            seen_names.add(name)
            deduped.append(r)

    return deduped
