"""
三维度精细化评估模块 — 重构版
================================
替代 evaluator_3d.py，解决原方案的循环依赖和维度重叠问题。

三个独立维度：
  1. 语义保真度 (50%) — LLM 结构化评分（对象/参数/方法三子项）
  2. 原文可溯性 (35%) — 名称要素在原文 chunks 中的支撑度
  3. 命名规范性 (15%) — 规则检查（长度/格式/唯一性）

Score = 0.50 × 语义保真度 + 0.35 × 原文可溯性 + 0.15 × 命名规范性

用法:
    # 单条评估
    from pipeline.evaluator_3d_refined import RefinedEvaluator
    ev = RefinedEvaluator()
    result = ev.evaluate_name(name="光纤陀螺仪零偏测试数据集",
                              kpi="零偏稳定性≤0.01°/h",
                              chunks=["研制高精度光纤陀螺仪...", ...])

    # 项目评估
    report = ev.evaluate_project(project_id="1646724823658876928")

    # 命令行
    python -m pipeline.evaluator_3d_refined \\
        --name "光纤陀螺仪零偏测试数据集" \\
        --kpi "零偏稳定性≤0.01°/h" \\
        --kg-dir output/kg_ontology/1646724823658876928
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from difflib import SequenceMatcher
from dataclasses import dataclass, field, asdict
from datetime import datetime

import requests
import numpy as np

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
KG_OUTPUT_DIR = BASE_DIR / "output" / "kg_ontology"
OUTPUT_DIR = BASE_DIR / "output" / "eval_reports_refined"

# =============================================================================
# API 配置
# =============================================================================
API_BASE = "http://10.3.213.253:23001"
API_KEY = "sk-259d53cf77064362aa19c816c1321e7b"
LLM_MODEL = "qwen3-32b"


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class EvalResult:
    """单条评估结果"""
    name: str
    kpi: str
    # 维度一：语义保真度
    object_score: float = 0.0
    parameter_score: float = 0.0
    method_score: float = 0.0
    semantic_total: float = 0.0
    semantic_reason: str = ""
    # 维度二：原文可溯性
    object_traceable: float = 0.0
    parameter_traceable: float = 0.0
    method_traceable: float = 0.0
    traceability_detail: str = ""
    traceability_penalty: float = 0.0
    traceability_total: float = 0.0
    # 维度三：命名规范性
    regularity_breakdown: dict = field(default_factory=dict)
    regularity_total: float = 0.0
    # 综合
    total_score: float = 0.0
    diagnosis: str = ""


@dataclass
class ProjectReport:
    """项目评估报告"""
    project_id: str
    project_name: str = ""
    kpi_count: int = 0
    dataset_count: int = 0
    results: List[EvalResult] = field(default_factory=list)
    avg_semantic: float = 0.0
    avg_traceability: float = 0.0
    avg_regularity: float = 0.0
    avg_total: float = 0.0
    distribution: dict = field(default_factory=dict)
    diagnosis_map: dict = field(default_factory=dict)


# =============================================================================
# LLM 调用工具
# =============================================================================

def llm_chat(system: str, user: str, max_tokens: int = 500,
             temperature: float = 0.1, timeout: int = 120) -> str:
    """LLM API 调用（与 kg_builder.py 一致）"""
    url = f"{API_BASE}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=timeout)
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 3
                time.sleep(wait)
                continue
            return ""
    return ""


def extract_json(text: str):
    """从 LLM 回复中提取 JSON"""
    if not text:
        return None
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass
    return None


# =============================================================================
# 工具函数
# =============================================================================

def load_chunks(kg_dir: Path) -> List[str]:
    """从 kg_ontology 目录加载 chunks"""
    chunks_dir = kg_dir / "chunks"
    if not chunks_dir.exists():
        return []
    chunks = []
    for fp in sorted(chunks_dir.glob("chunk_*.txt")):
        try:
            chunks.append(fp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return chunks


def load_vector_index_and_embedder(kg_dir: Path, chunks: List[str]):
    """加载 FAISS 索引和 Embedder，如不存在则构建"""
    from pipeline.kg_vector_store import (
        create_embedder, build_vector_index, load_vector_index, NumpyVectorIndex
    )
    faiss_path = kg_dir / "faiss.index"
    npz_path = kg_dir / "vectors.npz"

    embedder = None
    index = None

    # 尝试加载 NumpyVectorIndex (.npz)
    if npz_path.exists():
        try:
            index = NumpyVectorIndex.load(npz_path)
        except Exception:
            pass

    # 尝试加载 FAISS (.index)
    if index is None and faiss_path.exists():
        try:
            index = load_vector_index(faiss_path)
        except Exception:
            pass

    # 尝试加载 embedder .pkl
    if chunks:
        try:
            import joblib
            pkl_path = kg_dir / "embedder.pkl"
            if pkl_path.exists():
                embedder = joblib.load(pkl_path)
            else:
                embedder = create_embedder(corpus=chunks)
                joblib.dump(embedder, pkl_path)
        except Exception:
            embedder = create_embedder(corpus=chunks)

    return index, embedder


# =============================================================================
# 维度一：语义保真度 (50%)
# =============================================================================

SEMANTIC_SYSTEM_PROMPT = """你是一个数据集命名质量评估专家。你的任务是从三个子维度评估生成的数据集名称是否准确反映了考核指标(KPI)。

## 重要区分：技术参数 vs 性能目标值

KPI中常出现两类数字型指标，它们在数据集命名中的意义完全不同：

**性能目标值（不应算作参数匹配）**：
- 表达形式：提高X%、提升X%、效率≥X%、精度≤X%、误差<X%
- 示例："求解效率提高40%以上"、"光电转换效率≥22%"
- 这些是项目要达成的性能目标/考核要求，不是数据集本身的参数
- 好的数据集名称应当去掉目标值（数据集不包含"提升40%"这样的概念）

**技术参数（应算作参数匹配）**：
- 表达形式：参数名+单位/量词，如"零偏稳定性≤0.01°/h"、"带宽40GHz"
- 示例："零偏稳定性"、"响应时间"、"工作温度范围"
- 这些是数据集所描述对象的技术规格
- 好的数据集名称应当保留这些核心技术参数

### 一、研究对象匹配度 (0~0.4)
0.4: 名称包含KPI的核心研究对象/产品/器件，表述准确
0.2: 名称包含研究对象但过于宽泛或略有偏差
0.0: 名称中对象与KPI无关

### 二、技术参数匹配度 (0~0.4)
0.4: 名称包含KPI中的核心技术参数（零偏稳定性、响应时间等）
0.2: 名称含参数但不完整或表述略有偏差
0.0: 参数明显不匹配或缺失
注意：性能目标值（提高X%、≥X%等）不算参数匹配。名称去除目标值保留真实参数 → 不扣分

### 三、验证方法匹配度 (0~0.2)
0.2: 名称包含合适的验证/测试方法（仿真、测试、标定、眼图测试等）
0.1: 方法存在但不够具体或略有偏差
0.0: 方法缺失或不相关
注意：如果名称中补充了KPI未写明但领域合理的具体测试方法（如"眼图测试"、"高低温箱标定"），应给予0.2满分

### 总分判定
- total = object_score + parameter_score + method_score
- 范围 0~1

## 输出格式
{"object_score": 0.4, "parameter_score": 0.4, "method_score": 0.2, "total": 1.0, "reason": "一句话解释"}

只输出JSON，不要多余文字。"""


def _mask_target_values(text: str) -> str:
    """将KPI中的性能目标值标记为[目标值]，避免干扰语义评分"""
    text = re.sub(r'(?:提高|提升|增强|增加|降低|减少)[\s]*\d+\.?\d*%[\s以上以内以下]*', '[目标值]', text)
    text = re.sub(r'[≥≤<>±][\s]*\d+\.?\d*[\s]*[°/]?[h]?[\s]*[~-]?[\s]*[\d]*\.?\d*[\s]*[°/]?[h]?', '[目标值]', text)
    text = re.sub(r'效率[≥≤<>]\s*\d+%', '[目标值]', text)
    # 清理多余空格
    text = re.sub(r'\s+', '', text)
    return text


def score_semantic_fidelity(name: str, kpi: str) -> dict:
    """语义保真度评分"""
    # 检查是否为非技术 KPI
    admin_keywords = ["合作目标", "填写说明", "应从以下", "指相应成果",
                      "数量指标", "技术指标", "质量指标", "应用指标",
                      "考核方式", "评测方式", "中期指标"]
    is_admin = any(kw in kpi for kw in admin_keywords)
    if is_admin or len(kpi) < 10:
        return {"object_score": 0.0, "parameter_score": 0.0, "method_score": 0.0,
                "total": 0.7, "reason": "非技术性KPI/过短KPI，默认中上分"}

    # 预处理：mask 目标值
    masked_kpi = _mask_target_values(kpi)

    # 检查名称中是否含目标值
    name_contains_target = bool(re.search(r'(?:提高|提升|增强|增加|降低|减少)\s*\d+%', name) or re.search(r'[≥≤<>]', name))

    target_note = ""
    if name_contains_target:
        target_note = "\n⚠️ 注意：该名称包含性能目标值（如提高X%、≥X%等），这些是项目考核目标不是数据集参数，该项应在参数匹配度中得0分"

    user = f"""考核指标(KPI) — 已标注目标值：
{masked_kpi}

生成的数据集名称：
{name}{target_note}

请严格按照评分标准，从三个子维度分别评分。特别注意：性能目标值（提高X%、≥X%、精度≤X%等）是项目考核指标，不是数据集的技术参数。如果名称保留了目标值，参数匹配度应给0分。如果名称去除了目标值但保留了真正的技术参数（如零偏稳定性、响应时间等），参数匹配度正常评分。"""

    resp = llm_chat(SEMANTIC_SYSTEM_PROMPT, user, max_tokens=600, temperature=0.1)
    result = extract_json(resp)

    if result and "total" in result:
        # 校验子项范围
        obj_s = min(max(float(result.get("object_score", 0)), 0), 0.4)
        param_s = min(max(float(result.get("parameter_score", 0)), 0), 0.4)
        meth_s = min(max(float(result.get("method_score", 0)), 0), 0.2)
        total = obj_s + param_s + meth_s
        return {
            "object_score": round(obj_s, 3),
            "parameter_score": round(param_s, 3),
            "method_score": round(meth_s, 3),
            "total": round(total, 3),
            "reason": result.get("reason", ""),
        }

    # Fallback: SequenceMatcher 估算（先去除目标值干扰）
    def _strip_targets(text: str) -> str:
        """去除性能目标值（提高X%、≥X%等），避免它们干扰相似度计算"""
        text = re.sub(r'[提高提升增强增加降低减少]\s*\d+%[\s以上以内以下]*', '', text)
        text = re.sub(r'[≥≤<>±]?\s*\d+\.?\d*\s*%[/°]?[h]?', '', text)
        text = re.sub(r'[≥≤<>]\s*\d+\.?\d*\s*[°/]?[h]?', '', text)
        return text.strip()

    clean_name = _strip_targets(name)
    clean_kpi = _strip_targets(kpi)
    ratio = SequenceMatcher(None, clean_name, clean_kpi).ratio()
    fallback_total = min(0.7, round(ratio, 3))
    return {
        "object_score": round(fallback_total * 0.4, 3),
        "parameter_score": round(fallback_total * 0.4, 3),
        "method_score": round(fallback_total * 0.2, 3),
        "total": fallback_total,
        "reason": f"[fallback] SequenceMatcher={ratio:.3f}",
    }


# =============================================================================
# 维度二：原文可溯性 (35%)
# =============================================================================

# 中文停用词（技术术语提取用）
_STOPWORDS = {
    '研究', '分析', '方法', '技术', '系统', '平台', '装置', '设备',
    '数据', '数据集', '数据库', '模型', '算法', '软件', '程序',
    '实验', '试验', '测试', '检测', '测量', '验证', '评估',
    '制备', '加工', '制造', '构建', '建立', '开发', '研制',
    '基于', '采用', '通过', '实现', '进行', '开展', '提出',
    '问题', '关键', '主要', '基础', '应用', '理论',
    '提升', '优化', '改进', '解决', '利用', '针对', '相关',
    '项目', '课题', '任务', '目标', '指标', '考核',
    '以及', '及其', '等的', '之', '一个', '可以', '能够',
}


def extract_name_elements(name: str) -> dict:
    """从数据集名称中提取三要素（对象/参数/方法）

    使用规则+启发式拆分。对于复杂情况可以 fallback 到 LLM。
    策略：先提取方法词，再提取参数词，剩余做对象。

    参数词边界判断：找到参数词尾后，向前扩展到第一个符合以下条件的位置：
      - 遇到已知的对象结尾字（仪/器/机/计等）→ 停止（该字属于对象不属于参数）
      - 最多扩展 5 个中文字符
    """
    elements = {"object": "", "parameter": "", "method": ""}
    name_clean = name.replace("数据集", "").strip()

    if not name_clean:
        return elements

    # 方法词库（优先匹配长词，所以按长度降序排列）
    method_kw = sorted([
        "无损检测", "全温区测试", "高低温测试",
        "测试", "检测", "测量", "标定", "校准", "验证", "评估",
        "仿真", "模拟", "监测", "识别", "诊断", "预测",
        "实验", "试验", "分析", "计算", "筛查", "成像",
    ], key=lambda x: -len(x))

    # 参数词尾（按长度降序，优先匹配"稳定性"而非"性"）
    param_suffix = sorted([
        "稳定性", "重复性", "准确度", "灵敏度",
        "分辨率", "带宽", "波长", "功率", "电压",
        "电流", "温度", "压力", "速度", "频率",
        "精度", "性", "度", "率", "比", "量", "值",
    ], key=lambda x: -len(x))

    # 已知的对象结尾字（参数名不应包含这些字）
    obj_end_chars = set("仪器计机装置系统器件芯片模块组件电池电池组")
    # 注意: "管"、"池" 等字可能是参数或对象的一部分（二极管、电池），不加入此处

    # Step 1: 提取所有方法词（从原始名称中逐一移除）
    found_methods = []
    remaining = name_clean
    for kw in method_kw:
        while True:
            pos = remaining.find(kw)
            if pos < 0:
                break
            found_methods.append(kw)
            remaining = (remaining[:pos] + remaining[pos + len(kw):]).strip()

    # 取最长的方法词作为主方法（用于展示）
    found_method = max(found_methods, key=len) if found_methods else ""

    # Step 2: 从剩余串中分离参数
    found_param = ""
    param_start = len(remaining)  # default: no parameter found
    param_end = len(remaining)

    for suf in param_suffix:
        idx = remaining.rfind(suf)
        if idx >= 0:
            # 从词尾向前扩展到合理的参数词边界
            start = idx
            max_extend = 5  # 参数前缀最多5字
            extend_count = 0
            while start > 0 and extend_count < max_extend:
                ch = remaining[start - 1]
                if not ('\u4e00' <= ch <= '\u9fff'):
                    break
                # 遇到对象结尾字 → 停止（该字属于对象）
                if ch in obj_end_chars:
                    break
                start -= 1
                extend_count += 1
            found_param = remaining[start:idx + len(suf)]
            param_start = start
            param_end = idx + len(suf)
            break

    # Step 4: 剩余部分作为对象
    rest = (remaining[:param_start] + remaining[param_end:]).strip() if found_param else remaining

    # 清理冗余词
    for noise in ["的", "及", "与", "和", "对", "用", "及其实验", "及其",
                   "进行", "开展", "实现", "研究", "分析", "方法",
                   "高精度", "高灵敏度", "高性能", "大规模"]:
        rest = rest.replace(noise, "").strip()
    rest = re.sub(r'^[与及和、，\s]+', '', rest)
    rest = re.sub(r'[与及和、，\s]+$', '', rest)

    # 如果对象为空或太短，可能整个组合串就是对象
    if (not rest or len(rest) < 2) and found_param:
        # 尝试把参数前的部分作为对象（不含参数本身）
        pre_param = remaining[:param_start].strip()
        rest = pre_param if len(pre_param) >= 2 else ""

    elements["object"] = rest if rest and len(rest) >= 2 else ""
    elements["parameter"] = found_param
    elements["method"] = found_method

    return elements


def extract_name_elements_llm(name: str) -> dict:
    """用 LLM 从名称中提取三要素（精确版，作为规则法的备选）"""
    system = """从数据集名称中提取三要素：研究对象、性能参数、验证方法。
输出格式: {"object": "...", "parameter": "...", "method": "..."}
如果某要素不存在，值为空字符串。只输出JSON。"""
    user = f"数据集名称: {name}"
    resp = llm_chat(system, user, max_tokens=150, temperature=0.1)
    result = extract_json(resp)
    if result and isinstance(result, dict):
        return {
            "object": result.get("object", ""),
            "parameter": result.get("parameter", ""),
            "method": result.get("method", ""),
        }
    return extract_name_elements(name)  # fallback


def count_entity_frequency(chunks: List[str]) -> Dict[str, int]:
    """统计 chunks 中候选实体词频（用于反证检测）"""
    freq = {}
    for c in chunks:
        # 提取 2~6 字中文词
        words = re.findall(r'[\u4e00-\u9fff]{2,6}', c)
        for w in words:
            if w not in _STOPWORDS:
                freq[w] = freq.get(w, 0) + 1
    return freq


def compute_traceability(name: str, chunks: List[str],
                          index=None, embedder=None) -> dict:
    """计算原文可溯性

    Returns:
        {"object_traceable": float, "parameter_traceable": float,
         "method_traceable": float, "total": float,
         "penalty": float, "detail": str}
    """
    if not chunks:
        return {"object_traceable": 0.0, "parameter_traceable": 0.0,
                "method_traceable": 0.0, "total": 0.5,
                "penalty": 0.0, "detail": "无chunks可用，默认中性分"}

    # Step 1: 从名称提取三要素（规则法优先，LLM 补充缺失项）
    elements = extract_name_elements(name)  # 规则法
    llm_elements = extract_name_elements_llm(name)  # LLM 法
    for k in elements:
        if not elements[k] and llm_elements.get(k):
            elements[k] = llm_elements[k]

    # Step 2: 逐个要素检索
    precision_scores = {}  # 精确匹配计数
    semantic_scores = {}  # 语义匹配分数

    for key, value in elements.items():
        if not value or len(value) < 2:
            precision_scores[key] = 0.5  # 无具体值，中性
            semantic_scores[key] = 0.5
            continue

        # 2a: 精确匹配
        exact_count = sum(1 for c in chunks if value in c)
        if exact_count > 0:
            precision_scores[key] = 1.0
            semantic_scores[key] = 1.0
            continue

        # 2b: 精确匹配失败 → 子串匹配
        # 对参数词，尝试匹配部分（如"零偏稳定性" 匹配 "零偏稳定"）
        found = False
        for c in chunks:
            if value[:max(2, len(value) - 1)] in c:
                precision_scores[key] = 0.7
                semantic_scores[key] = 0.7
                found = True
                break
        if found:
            continue

        # 2c: 向量语义检索
        if index is not None and embedder is not None:
            try:
                from pipeline.kg_vector_store import vector_search
                results = vector_search(index, embedder, value, chunks, top_k=3)
                max_sim = max((r["score"] for r in results), default=0)
                if max_sim > 0.8:
                    semantic_scores[key] = 0.7
                elif max_sim > 0.6:
                    semantic_scores[key] = 0.3
                else:
                    semantic_scores[key] = 0.0
            except Exception:
                semantic_scores[key] = 0.0
        else:
            semantic_scores[key] = 0.0
        precision_scores[key] = semantic_scores[key]

    # Step 3: 反证检测（看名称中的主要对象在文档中的相对频次）
    penalty = 0.0
    main_obj = elements.get("object", "")
    if main_obj and len(main_obj) >= 2 and chunks:
        freq = count_entity_frequency(chunks)
        main_freq = freq.get(main_obj, 0)
        if main_freq > 0:
            # 找出现频次最高的其他实体
            sorted_freq = sorted(
                [(w, f) for w, f in freq.items() if w != main_obj and len(w) >= 2],
                key=lambda x: -x[1]
            )
            if sorted_freq:
                top_other, top_freq = sorted_freq[0]
                if top_freq > main_freq * 3 and top_freq > 5:
                    penalty = 0.5
                elif top_freq > main_freq * 2 and top_freq > 3:
                    penalty = 0.3

    # Step 4: 综合
    raw = (precision_scores.get("object", 0) + precision_scores.get("parameter", 0)
           + precision_scores.get("method", 0)) / 3
    total = raw * (1 - penalty)

    # 生成说明
    detail_parts = []
    for key, label in [("object", "对象"), ("parameter", "参数"), ("method", "方法")]:
        val = elements.get(key, "")
        sc = precision_scores.get(key, 0)
        if sc == 1.0:
            detail_parts.append(f"{label}「{val}」精确匹配 [OK]")
        elif sc >= 0.5:
            detail_parts.append(f"{label}「{val}」部分匹配({sc})")
        elif sc > 0:
            detail_parts.append(f"{label}「{val}」语义匹配({sc})")
        else:
            detail_parts.append(f"{label}「{val}」未匹配 [X]")

    if penalty > 0:
        detail_parts.append(f"反证惩罚: -{penalty*100:.0f}%（文档中其他实体频次远高于{main_obj}）")

    return {
        "object_traceable": precision_scores.get("object", 0),
        "parameter_traceable": precision_scores.get("parameter", 0),
        "method_traceable": precision_scores.get("method", 0),
        "total": round(total, 3),
        "penalty": penalty,
        "detail": "；".join(detail_parts),
    }


# =============================================================================
# 维度三：命名规范性 (15%)
# =============================================================================

def compute_regularity(name: str, project_all_names: List[str] = None) -> dict:
    """计算命名规范性

    Returns:
        {"breakdown": {规则名: 通过/未通过}, "total": float}
    """
    rules = [
        ("长度15-35字", 0.20, 15 <= len(name) <= 35),
        ("以'数据集'结尾", 0.15, name.endswith("数据集")),
        ("无特殊符号", 0.10, not bool(re.search(r'[<>:"/\\|?*]', name))),
        ("含具体对象", 0.20, not any(name.startswith(p) for p in ["数据", "测试", "实验", "研究"])),
        ("含动作词", 0.15, any(v in name for v in
                                ["测试", "检测", "分析", "评估", "标定",
                                 "验证", "测量", "仿真", "识别", "监测"])),
        ("无模板占位符", 0.10, not bool(re.search(r'\{.*?\}|XXX|\.{3,}', name))),
    ]

    # 跨 KPI 唯一性（如提供了全量名称列表）
    uniqueness_passed = True
    if project_all_names:
        same_count = sum(1 for n in project_all_names if n == name)
        uniqueness_passed = same_count <= 1
    rules.append(("跨KPI唯一", 0.10, uniqueness_passed))

    breakdown = {label: "通过" if passed else "未通过" for label, w, passed in rules}
    # 折衷：名称短于12字但不含模板符号且有具体对象，仍给基本分
    total = sum(w for label, w, passed in rules if passed)

    # 边界处理：略短但信息完整不严重惩罚
    if 12 <= len(name) < 15 and total >= 0.5:
        total = min(total + 0.10, 1.0)  # 补偿0.1

    return {
        "breakdown": breakdown,
        "total": round(min(total, 1.0), 3),
    }


# =============================================================================
# 诊断
# =============================================================================

def diagnose(semantic_total: float, traceability_total: float,
             regularity_total: float) -> str:
    """给出诊断结论"""
    if semantic_total >= 0.7 and traceability_total >= 0.7 and regularity_total >= 0.8:
        return "[OK] 优秀"
    if semantic_total >= 0.7 and traceability_total >= 0.7 and regularity_total < 0.8:
        return "[WARN] 格式待改进"
    if semantic_total >= 0.7 and traceability_total < 0.5:
        return "[FAIL] 检索召回不足——原文支撑不够"
    if semantic_total < 0.5 and traceability_total >= 0.7:
        return "[FAIL] 推理错误——原文有证据但命名不正确"
    if semantic_total < 0.5 and traceability_total < 0.5:
        return "[FAIL] 双重失败——检索和推理均需优化"
    if traceability_total <= traceability_total and "惩罚" in str(traceability_total):
        return "[WARN] 对象误判——名称实体在文档中非主体"
    return "[INFO] 一般"


# =============================================================================
# 主评估接口
# =============================================================================

def evaluate_name(name: str, kpi: str,
                  chunks: List[str] = None,
                  index=None, embedder=None,
                  all_names: List[str] = None) -> EvalResult:
    """对单个(名称, KPI)对进行三维度评估

    Args:
        name: 数据集名称
        kpi: 考核指标原文
        chunks: 原文文本块列表（可溯性评估用）
        index: 向量索引（可选，语义检索用）
        embedder: 嵌入器（可选）
        all_names: 同一项目所有名称列表（唯一性检查用）

    Returns:
        EvalResult 数据类
    """
    result = EvalResult(name=name, kpi=kpi)

    # 维度一：语义保真度
    sem = score_semantic_fidelity(name, kpi)
    result.object_score = sem["object_score"]
    result.parameter_score = sem["parameter_score"]
    result.method_score = sem["method_score"]
    result.semantic_total = sem["total"]
    result.semantic_reason = sem["reason"]

    # 维度二：原文可溯性
    chunks = chunks or []
    tra = compute_traceability(name, chunks, index, embedder)
    result.object_traceable = tra["object_traceable"]
    result.parameter_traceable = tra["parameter_traceable"]
    result.method_traceable = tra["method_traceable"]
    result.traceability_total = tra["total"]
    result.traceability_penalty = tra["penalty"]
    result.traceability_detail = tra["detail"]

    # 维度三：命名规范性
    reg = compute_regularity(name, all_names)
    result.regularity_breakdown = reg["breakdown"]
    result.regularity_total = reg["total"]

    # 综合
    result.total_score = round(
        0.50 * result.semantic_total
        + 0.35 * result.traceability_total
        + 0.15 * result.regularity_total,
        3,
    )
    result.diagnosis = diagnose(
        result.semantic_total, result.traceability_total, result.regularity_total
    )

    return result


def evaluate_project(project_id: str, kg_dir: Path = None,
                     names_csv: Path = None) -> ProjectReport:
    """对单个项目的所有 KPI 输出进行整体评估

    Args:
        project_id: 项目 ID
        kg_dir: KG 目录（含 chunks、向量索引等）
        names_csv: 数据集名称 CSV（如不指定则在 output/full_pipeline_output/ 下找）

    Returns:
        ProjectReport
    """
    # 1. 定位数据
    if kg_dir is None:
        kg_dir = KG_OUTPUT_DIR / project_id

    # 1a. 加载 chunks + 向量索引
    chunks = load_chunks(kg_dir)
    index, embedder = load_vector_index_and_embedder(kg_dir, chunks)
    print(f"  chunks: {len(chunks)} 块, 向量索引: {'有' if index else '无'}")

    # 1b. 定位并读取名称 CSV
    if names_csv is None:
        csv_dir = BASE_DIR / "output" / "full_pipeline_output"
        candidates = list(csv_dir.glob(f"{project_id}*.csv"))
        if not candidates:
            # 也在 dataset_names2_csv 找
            csv_dir = BASE_DIR / "output" / "dataset_names2_csv"
            candidates = list(csv_dir.glob(f"{project_id}*.csv"))
        names_csv = candidates[0] if candidates else None

    if names_csv is None or not names_csv.exists():
        print(f"  [跳过] 未找到 {project_id} 的 CSV 输出")
        report = ProjectReport(project_id=project_id)
        return report

    # 1c. 读取 CSV 中的名称 + KPI
    import csv
    name_kpi_pairs = []
    with open(names_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("数据集名称") or row.get("建议名称")
                    or row.get("name") or "").strip()
            kpi = (row.get("KPI原文") or row.get("KPI")
                   or row.get("kpi") or "").strip()
            if name and len(name) >= 4:
                name_kpi_pairs.append((name, kpi))

    if not name_kpi_pairs:
        print(f"  [跳过] CSV 中无有效名称: {names_csv.name}")
        report = ProjectReport(project_id=project_id)
        return report

    print(f"  读取 {len(name_kpi_pairs)} 条名称")

    # 2. 逐条评估
    all_names = [p[0] for p in name_kpi_pairs]
    results = []
    for i, (name, kpi) in enumerate(name_kpi_pairs):
        r = evaluate_name(name, kpi, chunks, index, embedder, all_names)
        results.append(r)

    # 3. 聚合报告
    avg_sem = sum(r.semantic_total for r in results) / len(results)
    avg_tra = sum(r.traceability_total for r in results) / len(results)
    avg_reg = sum(r.regularity_total for r in results) / len(results)
    avg_tot = sum(r.total_score for r in results) / len(results)

    # 分布
    dist = {"优秀(>=0.8)": 0, "良好(0.6~0.8)": 0, "一般(0.4~0.6)": 0, "差(<0.4)": 0}
    for r in results:
        if r.total_score >= 0.8:
            dist["优秀(>=0.8)"] += 1
        elif r.total_score >= 0.6:
            dist["良好(0.6~0.8)"] += 1
        elif r.total_score >= 0.4:
            dist["一般(0.4~0.6)"] += 1
        else:
            dist["差(<0.4)"] += 1

    # 诊断分类
    diag_map = {
        "检索召回不足": [],
        "推理错误": [],
        "双重失败": [],
        "格式待改进": [],
        "对象误判": [],
    }
    for r in results:
        diag = r.diagnosis
        for key in diag_map:
            if key in diag:
                diag_map[key].append(r.kpi[:40] + "...")
                break

    # 尝试获取项目名称
    project_name = project_id
    summary_path = kg_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            project_name = summary.get("project_name", project_id)
        except Exception:
            pass

    report = ProjectReport(
        project_id=project_id,
        project_name=project_name,
        kpi_count=len(name_kpi_pairs),
        dataset_count=len(results),
        results=results,
        avg_semantic=round(avg_sem, 3),
        avg_traceability=round(avg_tra, 3),
        avg_regularity=round(avg_reg, 3),
        avg_total=round(avg_tot, 3),
        distribution=dist,
        diagnosis_map={k: v for k, v in diag_map.items() if v},
    )
    return report


# =============================================================================
# 报告输出
# =============================================================================

def print_result(r: EvalResult):
    """打印单条评估结果"""
    print(f"\n  名称: {r.name}")
    print(f"  KPI: {r.kpi[:60]}...")
    print(f"  ─── 语义保真度(50%) ───")
    print(f"    对象: {r.object_score}  参数: {r.parameter_score}  方法: {r.method_score}  →  {r.semantic_total}")
    print(f"    理由: {r.semantic_reason}")
    print(f"  ─── 原文可溯性(35%) ───")
    print(f"    对象: {r.object_traceable}  参数: {r.parameter_traceable}  方法: {r.method_traceable}")
    print(f"    惩罚: {r.traceability_penalty}  →  {r.traceability_total}")
    print(f"    详情: {r.traceability_detail}")
    print(f"  ─── 命名规范性(15%) ───")
    for rule, status in r.regularity_breakdown.items():
        print(f"    {rule}: {status}")
    print(f"  →  规范性: {r.regularity_total}")
    print(f"  ─── 综合: {r.total_score}  |  诊断: {r.diagnosis}")


def print_report(report: ProjectReport):
    """打印项目评估报告"""
    print(f"\n{'='*60}")
    print(f"  项目: {report.project_name} ({report.project_id})")
    print(f"  数据集: {report.dataset_count} 条")
    print(f"{'='*60}")
    print(f"  平均语义保真度:     {report.avg_semantic}")
    print(f"  平均原文可溯性:     {report.avg_traceability}")
    print(f"  平均命名规范性:     {report.avg_regularity}")
    print(f"  ───────────────────────────────")
    print(f"  加权平均总分:       {report.avg_total}")
    print(f"\n  分数分布:")
    for label, cnt in sorted(report.distribution.items(), key=lambda x: -x[1]):
        bar = "█" * cnt if cnt < 50 else "█" * 50
        print(f"    {label}: {cnt:>3d}  {bar}")
    if report.diagnosis_map:
        print(f"\n  诊断分类:")
        for label, items in report.diagnosis_map.items():
            print(f"    {label}: {len(items)} 条")
            for item in items[:5]:
                print(f"      - {item}")
            if len(items) > 5:
                print(f"      ...等 {len(items)} 条")


def save_report(report: ProjectReport, output_dir: Path):
    """保存报告到 JSON + CSV"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / f"{report.project_id}_report.json"
    json.dump({
        "project_id": report.project_id,
        "project_name": report.project_name,
        "dataset_count": report.dataset_count,
        "avg_semantic": report.avg_semantic,
        "avg_traceability": report.avg_traceability,
        "avg_regularity": report.avg_regularity,
        "avg_total": report.avg_total,
        "distribution": report.distribution,
        "diagnosis_map": report.diagnosis_map,
        "timestamp": datetime.now().isoformat(),
    }, open(json_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"  报告保存: {json_path}")

    # CSV 明细
    csv_path = output_dir / f"{report.project_id}_details.csv"
    import csv as csv_module
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv_module.writer(f)
        writer.writerow([
            "名称", "KPI(前60字)",
            "语义_对象", "语义_参数", "语义_方法", "语义总分", "语义理由",
            "可溯_对象", "可溯_参数", "可溯_方法", "可溯总分", "可溯惩罚",
            "规范性", "综合总分", "诊断"
        ])
        for r in report.results:
            writer.writerow([
                r.name, r.kpi[:60],
                r.object_score, r.parameter_score, r.method_score,
                r.semantic_total, r.semantic_reason,
                r.object_traceable, r.parameter_traceable, r.method_traceable,
                r.traceability_total, r.traceability_penalty,
                r.regularity_total,
                r.total_score, r.diagnosis,
            ])
    print(f"  明细保存: {csv_path}")


# =============================================================================
# 批量评估：全部项目
# =============================================================================

def evaluate_all(output_dir: Path = None):
    """评估所有已生成 KG 的项目"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    kg_dirs = sorted(KG_OUTPUT_DIR.glob("*"))
    kg_dirs = [d for d in kg_dirs if d.is_dir() and (d / "summary.json").exists()
               and d.name.isdigit()]

    print(f"找到 {len(kg_dirs)} 个已构建 KG 的项目")

    all_reports = []
    for kg_dir in kg_dirs:
        pid = kg_dir.name
        print(f"\n处理项目: {pid}")
        try:
            report = evaluate_project(pid, kg_dir=kg_dir)
            if report.dataset_count > 0:
                print_report(report)
                save_report(report, output_dir)
                all_reports.append(report)
        except Exception as e:
            print(f"  [错误] {e}")
            import traceback
            traceback.print_exc()

    # 全局汇总
    if all_reports:
        summary_path = output_dir / "summary.json"
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_projects": len(all_reports),
            "total_datasets": sum(r.dataset_count for r in all_reports),
            "global_avg_semantic": round(
                sum(r.avg_semantic * r.dataset_count for r in all_reports)
                / sum(r.dataset_count for r in all_reports), 3),
            "global_avg_traceability": round(
                sum(r.avg_traceability * r.dataset_count for r in all_reports)
                / sum(r.dataset_count for r in all_reports), 3),
            "global_avg_regularity": round(
                sum(r.avg_regularity * r.dataset_count for r in all_reports)
                / sum(r.dataset_count for r in all_reports), 3),
            "global_avg_total": round(
                sum(r.avg_total * r.dataset_count for r in all_reports)
                / sum(r.dataset_count for r in all_reports), 3),
            "projects": [{
                "project_id": r.project_id,
                "project_name": r.project_name,
                "dataset_count": r.dataset_count,
                "avg_semantic": r.avg_semantic,
                "avg_traceability": r.avg_traceability,
                "avg_regularity": r.avg_regularity,
                "avg_total": r.avg_total,
            } for r in all_reports],
        }, open(summary_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        print(f"\n全局汇总: {summary_path}")
        print(f"  项目数: {len(all_reports)}")
        print(f"  总数据集: {sum(r.dataset_count for r in all_reports)}")
        print(f"  全局平均分: {sum(r.avg_total * r.dataset_count for r in all_reports) / sum(r.dataset_count for r in all_reports):.3f}")

    return all_reports


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="三维度精细化评估")
    parser.add_argument("--name", help="数据集名称（单条模式）")
    parser.add_argument("--kpi", help="KPI 文本（单条模式）")
    parser.add_argument("--kg-dir", help="KG 目录（可选，用于可溯性评估）")
    parser.add_argument("--project", help="项目 ID（项目模式）")
    parser.add_argument("--all", action="store_true", help="评估所有项目")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output)

    # 模式1: 单条评估
    if args.name and args.kpi:
        chunks = []
        index = embedder = None
        if args.kg_dir:
            kg_dir = Path(args.kg_dir)
            chunks = load_chunks(kg_dir)
            index, embedder = load_vector_index_and_embedder(kg_dir, chunks)

        result = evaluate_name(args.name, args.kpi, chunks, index, embedder)
        print_result(result)
        return

    # 模式2: 项目评估
    if args.project:
        pid = args.project
        kg_dir = KG_OUTPUT_DIR / pid if not args.kg_dir else Path(args.kg_dir)
        report = evaluate_project(pid, kg_dir=kg_dir)
        if report.dataset_count > 0:
            print_report(report)
            save_report(report, output_dir)
        else:
            print(f"项目 {pid} 无结果（可能未运行 full_pipeline）")
        return

    # 模式3: 全部
    if args.all:
        evaluate_all(output_dir=output_dir)
        return

    # 模式4: 交互
    print("交互模式: 逐条输入名称和 KPI，Ctrl+C 退出")
    try:
        while True:
            name = input("\n数据集名称: ").strip()
            if not name:
                break
            kpi = input("KPI 原文: ").strip()
            if not kpi:
                break
            kg_dir = input("KG 目录（留空跳过）: ").strip()
            chunks = []
            index = embedder = None
            if kg_dir:
                chunks = load_chunks(Path(kg_dir))
                index, embedder = load_vector_index_and_embedder(Path(kg_dir), chunks)
            result = evaluate_name(name, kpi, chunks, index, embedder)
            print_result(result)
    except KeyboardInterrupt:
        print("\n退出")


if __name__ == "__main__":
    main()
