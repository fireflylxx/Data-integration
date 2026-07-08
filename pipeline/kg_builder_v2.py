"""
知识图谱构建引擎 v2 — 本体对齐 + 完备性增强
============================================

基于 kg_builder.py 的增强版本，在原有架构上新增三项改进：
  P1. 两轮异步抽取：先抽实体，再以实体列表为约束抽取关系
  P0. KPI 驱动增强：对考核指标相关文本块进行针对性二次抽取
  P2. 跨块关系发现：基于实体共现发现跨 chunk 的隐性关系

新增特性默认启用，通过 process_project() 的开关参数控制。
原有函数签名保持兼容，可直接替换 kg_builder.py 使用。

注意: Windows 下运行请先设置环境变量 SET PYTHONIOENCODING=utf-8
       或使用 python -u pipeline/run_kg_batch.py ... 配合编码设置。

核心改进：
  1. 实体类型：从自由类型 → 9 种本体约束类型（4核心 + 5辅助）
  2. 关系谓词：从自由谓词 → 6 种本体约束谓词（VIA/VERIFIES/EXECUTES/PRODUCES/BELONGS_TO/MAPS_TO）
  3. KPI 解析模块：从"考核指标"章节提取结构化指标
  4. FAISS 向量索引：文本块检索增强
  5. 课题→KPI层级建模：课题作为OBJECT实体，与KPI通过BELONGS_TO关联
  --- v2 新增 ---
  6. 两轮抽取：实体先于关系，提升关系抽取准确率
  7. KPI 驱动增强：针对考核指标相关块重点抽取
  8. 跨块关系发现：跨越文本块边界的实体关系链接

注：数据集推理已从本模块移除，由 full_pipeline.py 的 kpi_planner.py 负责。

输入：output/project_cleaned/*.txt
输出：output/kg_ontology/{project_id}/
  - entities.json       # 本体实体列表（带 type/profile/neighbors）
  - relations.json      # 本体关系列表（带 head/relation/tail/context）
  - kpis.json           # 解析后的考核指标（含 topic_id）
  - topics.json         # (可选) 课题列表
  - low_level_kv.json   # LightRAG 低层 KV 索引
  - high_level_kv.json  # LightRAG 高层 KV 索引
  - chunks/             # 文本分块
  - summary.json        # 统计摘要
  - faiss.index         # FAISS 向量索引
"""

import json, re, time, sys, asyncio, os
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np

# =============================================================================
# 本体类型（与 submission_ontology.py 同步）
# =============================================================================

# 实体类型 — 4核心 + 5辅助
ENTITY_TYPES = {
    # 核心类型
    "OBJECT": "研究对象（器件/系统/材料/结构）",
    "METHOD": "实验/测试/仿真方法",
    "PARAMETER": "性能参数/技术指标",
    "ACTIVITY": "验证活动（标定/测试/评估等）",
    # 辅助类型
    "EQUIPMENT": "实验设备/仪器",
    "MATERIAL": "材料",
    "SOFTWARE": "软件/算法",
    "SYSTEM": "系统/平台",
    "MODEL": "模型",
}

# 关系谓词 — 6 种本体约束
RELATION_TYPES = {
    "VIA": "[研究对象]—通过→[实验/测试方法]",
    "VERIFIES": "[实验/测试方法]—验证→[性能参数]",
    "EXECUTES": "[研究对象]—执行→[验证活动]",
    "PRODUCES": "[研究对象]—产出→[数据集]",
    "BELONGS_TO": "[数据集/课题]—属于→[课题/项目]",
    "MAPS_TO": "[数据集]—对应→[考核指标]",
}

# 实体类型映射（从 LLM 自由输出 → 本体约束类型）
ENTITY_TYPE_MAP = {
    # → OBJECT
    "OBJECT": "OBJECT", "COMPONENT": "OBJECT", "PART": "OBJECT",
    "MODULE": "OBJECT", "DEVICE": "OBJECT", "UNIT": "OBJECT",
    "PLATFORM": "OBJECT", "STRUCTURE": "OBJECT", "LAYER": "OBJECT",
    "PRODUCT": "OBJECT",
    # → METHOD
    "METHOD": "METHOD", "TECHNOLOGY": "METHOD", "TECHNIQUE": "METHOD",
    "ALGORITHM": "METHOD", "PROCESS": "METHOD", "CRAFT": "METHOD",
    "APPROACH": "METHOD",
    # → PARAMETER
    "PARAMETER": "PARAMETER", "METRIC": "PARAMETER", "INDEX": "PARAMETER",
    "INDICATOR": "PARAMETER",
    # → ACTIVITY
    "ACTIVITY": "ACTIVITY",
    # → EQUIPMENT
    "EQUIPMENT": "EQUIPMENT", "INSTRUMENT": "EQUIPMENT", "APPARATUS": "EQUIPMENT",
    # → MATERIAL
    "MATERIAL": "MATERIAL", "CHEMICAL": "MATERIAL", "COMPOUND": "MATERIAL",
    # → SOFTWARE
    "SOFTWARE": "SOFTWARE",
    # → SYSTEM
    "SYSTEM": "SYSTEM",
    # → MODEL
    "MODEL": "MODEL",
    # 旧系统兼容（从 lightrag_extract_cleaned 继承）
    "TOPIC": "OBJECT",  # 课题 → 研究对象
    "OBJECT": "OBJECT",
}

# 关系映射（从自由谓词 → 本体约束）
RELATION_MAP = {
    # → VIA（对象通过方法）
    "采用": "VIA", "使用": "VIA", "利用": "VIA", "通过": "VIA",
    "应用": "VIA", "基于": "VIA", "运用": "VIA",
    # → VERIFIES（方法验证参数）
    "测试": "VERIFIES", "检测": "VERIFIES", "测量": "VERIFIES",
    "标定": "VERIFIES", "验证": "VERIFIES", "校验": "VERIFIES",
    "试验": "VERIFIES", "实验": "VERIFIES",
    "改进": "VERIFIES", "提升": "VERIFIES", "优化": "VERIFIES",
    "提高": "VERIFIES", "补偿": "VERIFIES",
    # → EXECUTES（对象执行活动）
    "实现": "EXECUTES", "构建": "EXECUTES", "建立": "EXECUTES",
    "开发": "EXECUTES", "研制": "EXECUTES", "制备": "EXECUTES",
    "设计": "EXECUTES", "形成": "EXECUTES", "产生": "EXECUTES",
    # → PRODUCES（对象产出数据集）
    "产出": "PRODUCES", "交付": "PRODUCES", "成果形式": "PRODUCES",
    # → BELONGS_TO（属于关系）
    "归属": "BELONGS_TO", "属于": "BELONGS_TO", "包含": "BELONGS_TO",
    "包括": "BELONGS_TO", "分为": "BELONGS_TO",
    "集成": "BELONGS_TO", "涉及": "BELONGS_TO",
    # → MAPS_TO（对应指标）
    "考核": "MAPS_TO", "指标要求": "MAPS_TO", "要求": "MAPS_TO",
}

# 非技术实体类型 — 直接过滤
NON_ENTITY_TYPES = {
    "PERSON", "ORGANIZATION", "PROJECT", "DOCUMENT", "FIGURE", "TABLE",
    "LOCATION", "EVENT", "FIELD", "DOMAIN", "CONCEPT", "THEORY",
    "PHENOMENON", "PROBLEM", "RESULT", "GOAL", "TASK",
    "INDUSTRY", "STANDARD", "REGULATION",
}

# =============================================================================
# API 配置
# =============================================================================
API_BASE = ""
API_KEY = ""
LLM_MODEL = ""

INPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "project_cleaned"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "kg_ontology"
MAX_PROJECTS = 5

# Runtime overrides (set by main() argparse)
_RUNTIME_INPUT_DIR = None
_RUNTIME_OUTPUT_DIR = None
_RUNTIME_MAX_PROJECTS = None
_RESUME_MODE = False
_START_FROM = 0


# =============================================================================
# LLM 调用
# =============================================================================

def llm_chat(system: str, user: str, max_tokens: int = 2000,
             temperature: float = 0.3, timeout: int = 180) -> str:
    url = f"{API_BASE}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=timeout)
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 5
                print(f"  [LLM重试 {attempt+1}/3] {e}")
                time.sleep(wait)
                continue
            return f"[ERROR] {e}"
    return "[ERROR]"


def extract_json(text: str) -> Optional[dict]:
    text = text.strip()
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


def extract_json_array(text: str) -> Optional[list]:
    """从文本中提取 JSON 数组（用于润色等返回列表的场景）"""
    text = text.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            pass
    arr_start = text.find('[')
    arr_end = text.rfind(']')
    if arr_start >= 0 and arr_end > arr_start:
        try:
            return json.loads(text[arr_start:arr_end + 1])
        except json.JSONDecodeError:
            pass
    return None


# =============================================================================
# Prompt — 本体对齐（原始版本：实体+关系一次抽取）
# =============================================================================

SYSTEM_PROMPT = f"""你是一个科研知识图谱抽取专家。你的任务是从"国家重点研发计划"项目任务书中，抽取符合本体约束的实体和关系。

## 实体类型（共 9 种）

{chr(10).join(f'  {k}：{v}' for k, v in ENTITY_TYPES.items())}

## 关系谓词（共 6 种）

{chr(10).join(f'  {k}：{v}' for k, v in RELATION_TYPES.items())}

## 关系组合规则

完整的验证路径：
  [研究对象] —VIA→ [实验/测试方法] —VERIFIES→ [性能参数]
  [研究对象] —EXECUTES→ [验证活动]

示例：
  光纤陀螺仪 —VIA→ 转台测试 —VERIFIES→ 零偏稳定性
  MEMS加速度计 —EXECUTES→ 温度补偿标定

## 抽取规则

1. 实体名使用原文完整名称
2. 关系必须有原文明确支撑（填入 context 字段）
3. 每对实体之间同一关系只输出一次
4. 不抽取：人名、机构名、项目名称、通用概念
5. 不抽取背景介绍段落

## 输出格式

{{"entities": [{{"name": "...", "type": "..."}}],
 "relations": [{{"head": "...", "relation": "VIA|VERIFIES|EXECUTES|PRODUCES|BELONGS_TO|MAPS_TO", "tail": "...", "context": "原文证据(20-50字)"}}]}}"""

FEW_SHOT = """
=== 示例1：KPI表 ===
段落:
考核指标：
1. 零偏稳定性≤0.01°/h，成果形式：数据集1套
2. 标度因数重复性≤10ppm
采用转台进行标定测试

输出:
{"entities": [
  {"name": "零偏稳定性", "type": "PARAMETER"},
  {"name": "标度因数重复性", "type": "PARAMETER"},
  {"name": "转台", "type": "EQUIPMENT"},
  {"name": "标定测试", "type": "ACTIVITY"}
],"relations": [
  {"head": "零偏稳定性", "relation": "VERIFIES", "tail": "转台", "context": "采用转台进行标定测试"},
  {"head": "转台", "relation": "EXECUTES", "tail": "标定测试", "context": "采用转台进行标定测试"}
]}

=== 示例2：研究内容 ===
段落:
针对 MEMS 加速度计的温度漂移问题，提出了一种基于差分电容检测的温度补偿方法。通过高低温试验箱对加速度计进行全温度范围测试，测试其零偏稳定性。

输出:
{"entities": [
  {"name": "MEMS加速度计", "type": "OBJECT"},
  {"name": "差分电容检测温度补偿方法", "type": "METHOD"},
  {"name": "高低温试验箱", "type": "EQUIPMENT"},
  {"name": "零偏稳定性", "type": "PARAMETER"}
],"relations": [
  {"head": "MEMS加速度计", "relation": "VIA", "tail": "差分电容检测温度补偿方法", "context": "提出了一种基于差分电容检测的温度补偿方法"},
  {"head": "MEMS加速度计", "relation": "VIA", "tail": "高低温试验箱", "context": "通过高低温试验箱对加速度计进行全温度范围测试"},
  {"head": "高低温试验箱", "relation": "VERIFIES", "tail": "零偏稳定性", "context": "测试其零偏稳定性"}
]}

=== 示例3：包含关系 ===
段落:
晶圆级真空封装主要涉及键合区域金属化、封帽晶圆凹槽的刻蚀。

输出:
{"entities": [
  {"name": "晶圆级真空封装", "type": "OBJECT"},
  {"name": "键合区域金属化", "type": "METHOD"},
  {"name": "封帽晶圆凹槽刻蚀", "type": "METHOD"}
],"relations": [
  {"head": "晶圆级真空封装", "relation": "BELONGS_TO", "tail": "键合区域金属化", "context": "晶圆级真空封装主要涉及键合区域金属化"},
  {"head": "晶圆级真空封装", "relation": "BELONGS_TO", "tail": "封帽晶圆凹槽刻蚀", "context": "晶圆级真空封装主要涉及封帽晶圆凹槽的刻蚀"}
]}"""


# =============================================================================
# === IMPROVEMENT P1: 两轮抽取 — Entity-only + Relation-only Prompts ==========
# =============================================================================

ENTITY_ONLY_PROMPT = f"""你是一个科研实体抽取专家。请从以下文本中抽取所有技术实体，只输出实体，不抽取关系。

## 实体类型（共 9 种）

{chr(10).join(f'  {k}：{v}' for k, v in ENTITY_TYPES.items())}

## 抽取规则
1. 实体名使用原文完整名称（保留修饰语）
2. 只抽取与技术内容相关的实体（对象、方法、参数、活动、设备、材料、软件、系统、模型）
3. 不抽取：人名、机构名、项目名称、通用概念、背景介绍
4. 实体必须直接出现在文本中，不要推理或补全

## 输出格式
只输出 JSON 数组，不要思考标记、不要解释：
[{{"name": "实体名称", "type": "OBJECT|METHOD|PARAMETER|ACTIVITY|EQUIPMENT|MATERIAL|SOFTWARE|SYSTEM|MODEL"}}]"""

RELATION_ONLY_PROMPT = f"""你是一个科研关系抽取专家。给定一段文本和已知的技术实体列表，请从文本中找出这些实体之间的关系。

## 关系谓词（共 6 种）

{chr(10).join(f'  {k}：{v}' for k, v in RELATION_TYPES.items())}

## 关系组合规则
完整的验证路径：
  [研究对象] —VIA→ [实验/测试方法] —VERIFIES→ [性能参数]
  [研究对象] —EXECUTES→ [验证活动]

## 抽取规则
1. 关系必须有原文明确支撑（填入 context 字段，摘取20-50字原文）
2. 每对实体之间同一关系只输出一次
3. head 和 tail 必须来自给定的实体列表
4. 仅当原文存在明确的语义关系时才输出
5. 不要推测不存在的关系

## 输出格式
只输出 JSON 数组，不要思考标记、不要解释：
[{{"head": "实体A", "relation": "VIA|VERIFIES|EXECUTES|PRODUCES|BELONGS_TO|MAPS_TO", "tail": "实体B", "context": "原文证据(20-50字)"}}]"""


# =============================================================================
# === IMPROVEMENT P0: KPI 驱动增强 — Prompt ====================================
# =============================================================================

KPI_BOOST_SYSTEM_PROMPT = f"""你是一个科研知识图谱抽取专家，专注于从考核指标(KPI)相关描述中提取实体和关系。

## 背景
考核指标（KPI）是项目任务书中最重要的部分，它们规定了必须达到的技术目标。
你需要从与KPI高度相关的文本中，最大程度地抽取与指标相关的技术实体和关系。

## 实体类型（共 9 种）

{chr(10).join(f'  {k}：{v}' for k, v in ENTITY_TYPES.items())}

## 关系谓词（共 6 种）

{chr(10).join(f'  {k}：{v}' for k, v in RELATION_TYPES.items())}

## 抽取重点
- 优先抽取与考核指标直接相关的实体（指标中提到的对象、参数、方法）
- 明确抽取指标中的目标值（作为 PARAMETER 实体的补充描述）
- 注意抽取验证路径：对象→方法→参数→活动

## 输出格式
{{"entities": [{{"name": "...", "type": "..."}}],
 "relations": [{{"head": "...", "relation": "VIA|VERIFIES|EXECUTES|PRODUCES|BELONGS_TO|MAPS_TO", "tail": "...", "context": "原文证据(20-50字)"}}]}}"""


# =============================================================================
# === IMPROVEMENT P2: 跨块关系发现 — Prompt ====================================
# =============================================================================

CROSS_CHUNK_RELATION_PROMPT = f"""你是一个科研关系推理专家。给定同一项目的两个文本块（chunk A 和 chunk B），以及它们各自包含的实体，请找出跨块的实体关系。

## 关系谓词（共 6 种）

{chr(10).join(f'  {k}：{v}' for k, v in RELATION_TYPES.items())}

## 规则
1. head 必须来自 chunk A 的实体列表，tail 必须来自 chunk B 的实体列表
2. 关系必须有原文支撑（至少一个 chunk 中有相关描述），填入 context 字段
3. 只有存在明确语义关联时再输出，不要强行关联
4. 每对实体之间同一关系只输出一次

## 输出格式
只输出 JSON 数组，不要思考标记、不要解释：
[{{"head": "实体A(chunkA)", "relation": "VIA|VERIFIES|EXECUTES|PRODUCES|BELONGS_TO|MAPS_TO", "tail": "实体B(chunkB)", "context": "推理依据(20-50字)"}}]
如果没有跨块关系，输出空数组 []"""


# =============================================================================
# 数据解析
# =============================================================================

def parse_project_file(filepath: Path) -> Optional[dict]:
    """解析 project_cleaned 文件"""
    try:
        text = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = filepath.read_text(encoding="gbk")
        except:
            return None
    lines = text.split("\n")
    name = ""
    pid = ""
    pno = ""
    for line in lines[:10]:
        if line.startswith("项目名称:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("项目编号:"):
            pno = line.split(":", 1)[1].strip()
        elif line.startswith("项目ID:"):
            pid = line.split(":", 1)[1].strip()
    body_start = text.find("====")
    body = text[body_start:].strip() if body_start > 0 else text
    return {"id": pid or filepath.stem, "name": name or filepath.stem,
            "projectNo": pno, "text": body, "file": filepath.name}


def parse_kpi_section(text: str) -> List[dict]:
    """从任务书中解析考核指标章节"""
    kpis = []
    patterns = [
        r'考核\s*指标[：:\s]*(.*?)(?=\n\n\n|\Z)',
        r'项目目标[及与]*考核指标[：:\s]*(.*?)(?=项目目标、|\Z)',
        r'考核[指标]*表[：:\s]*(.*?)(?=\n\n\n|\Z)',
    ]
    section = ""
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            section = m.group(1)
            break

    if not section:
        return kpis

    for line in section.split("\n"):
        line = line.strip()
        if not line or len(line) < 5:
            continue
        m = re.match(r'^(\d+(?:[\.\d]*))\s*[、.、]\s*(.+)', line)
        if m:
            kpis.append({"id": m.group(1), "description": m.group(2).strip()})
            continue
        m = re.match(r'^[（(](\d+)[)）]\s*(.+)', line)
        if m:
            kpis.append({"id": m.group(1), "description": m.group(2).strip()})
            continue
        if "指标" in line or "考核" in line:
            kpis.append({"id": str(len(kpis) + 1), "description": line})

    # 从ID前缀推断课题号
    for kpi in kpis:
        m_id = re.match(r'(\d+)', kpi["id"])
        kpi["topic_id"] = int(m_id.group(1)) if m_id else None

    return kpis


# =============================================================================
# 课题解析 & KPI-课题匹配
# =============================================================================

_CN_NUM = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
}


def _cn_to_int(s: str) -> Optional[int]:
    """中文数字 → 整数（仅处理单字）。"""
    if s in _CN_NUM:
        return _CN_NUM[s]
    try:
        return int(s)
    except ValueError:
        return None


def parse_topics(text: str) -> List[dict]:
    """从任务书中提取课题列表。

    扫描全文匹配"课题/任务 N：名称"格式的行。
    返回: [{"id": <int>, "name": <str>}, ...]
    """
    topics = []
    seen_ids = set()
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        m = re.search(r'(?:课题|任务)\s*([一二三四五六七八九十\d]+)\s*[：:]\s*(.+)', line)
        if not m:
            continue
        tid = _cn_to_int(m.group(1))
        if tid is None or tid in seen_ids:
            continue
        name = m.group(2).strip()
        if len(name) < 4:
            continue
        seen_ids.add(tid)
        topics.append({"id": tid, "name": name})
    return topics


def _kpi_topic_id(kpi: dict) -> Optional[int]:
    """从 KPI ID 的第一个数字段推断课题号。"""
    m = re.match(r'(\d+)', kpi.get("id", ""))
    return int(m.group(1)) if m else None


def match_kpis_to_topics(kpis: List[dict], topics: List[dict]) -> dict:
    """将 KPI 按 ID 前缀分组到课题。

    为每个 KPI 注入 topic_id 字段（向后兼容）。
    返回: {topic_id: [kpi, ...], '__unmatched__': [kpi, ...]}
    """
    topic_ids = {t["id"] for t in topics}
    result = {tid: [] for tid in topic_ids}
    result["__unmatched__"] = []

    for kpi in kpis:
        tid = _kpi_topic_id(kpi)
        kpi["topic_id"] = tid
        if tid in topic_ids:
            result[tid].append(kpi)
        else:
            result["__unmatched__"].append(kpi)
    return result


def _topic_entity_name(topic: dict) -> str:
    """生成课题实体的标准 KG 名称。"""
    name = topic.get("name", "").strip()
    if not name:
        return f"课题{topic['id']}"
    result = f"课题{topic['id']}:{name}"
    return result[:80] if len(result) > 80 else result


def create_topic_entities(topics: List[dict]) -> List[dict]:
    """创建课题 KG 实体（类型: OBJECT）。"""
    return [{"name": _topic_entity_name(t), "type": "OBJECT"} for t in topics]


def create_topic_relations(topics: List[dict], topic_kpi_map: dict,
                           entities: List[dict], relations: List[dict]) -> None:
    """创建 PARAMETER → 课题 的 BELONGS_TO 关系（追加到 relations）。"""
    param_names = {e["name"] for e in entities if e["type"] == "PARAMETER"}
    existing = {(r["head"], r["relation"], r["tail"]) for r in relations}

    for t in topics:
        ename = _topic_entity_name(t)
        for kpi in topic_kpi_map.get(t["id"], []):
            desc = kpi.get("description", "")
            for pname in param_names:
                if pname in desc:
                    key = (pname, "BELONGS_TO", ename)
                    if key not in existing:
                        existing.add(key)
                        relations.append({
                            "head": pname,
                            "relation": "BELONGS_TO",
                            "tail": ename,
                            "context": f"{pname}属于{ename}"
                        })


def chunk_text(text: str, max_chars: int = 800, min_chars: int = 200) -> List[str]:
    """按行切分后合并为段落块，兼容单换行和双换行格式"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    chunks = []
    cur = ""
    for line in lines:
        if len(line) > max_chars:
            if cur and len(cur) >= min_chars:
                chunks.append(cur)
            cur = ""
            continue
        sep = "\n\n" if (cur and cur[-1] != '\n') else ""
        if len(cur) + len(line) + len(sep) < max_chars:
            cur = (cur + sep + line).strip()
        else:
            if cur and len(cur) >= min_chars:
                chunks.append(cur)
            cur = line
    if cur and len(cur) >= min_chars:
        chunks.append(cur)
    return chunks


# =============================================================================
# 实体 & 关系归一化
# =============================================================================

def normalize_type(raw_type: str) -> Optional[str]:
    t = raw_type.strip().upper()
    if t in ENTITY_TYPE_MAP:
        return ENTITY_TYPE_MAP[t]
    if t in NON_ENTITY_TYPES:
        return None
    return t


def normalize_relation(rel: str) -> Optional[str]:
    r = rel.strip()
    if r in RELATION_MAP:
        return RELATION_MAP[r]
    return r


def normalize(entities: List[dict], relations: List[dict]) -> Tuple[List[dict], List[dict]]:
    """归一化实体类型和关系谓词"""
    emap = {}
    for e in entities:
        nt = normalize_type(e.get("type", ""))
        if nt is None:
            continue
        name = e["name"]
        if name not in emap:
            emap[name] = {"name": name, "type": nt}
        elif emap[name]["type"] != nt:
            if nt in ("OBJECT", "METHOD", "PARAMETER"):
                emap[name]["type"] = nt

    valid = set(emap.keys())
    nrels = []
    for rel in relations:
        h, r, t = rel.get("head", ""), rel.get("relation", ""), rel.get("tail", "")
        if h not in valid or t not in valid:
            continue
        nr = normalize_relation(r)
        if nr is None:
            continue
        nrels.append({"head": h, "relation": nr, "tail": t,
                       "context": rel.get("context", "")})
    return list(emap.values()), nrels


# =============================================================================
# 向量索引构建（FAISS）
# =============================================================================

class TFIDFEmbedding:
    """本地 TF-IDF + SVD 嵌入（无需下载模型）"""
    def __init__(self, corpus: List[str], dim: int = 384):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.vectorizer = TfidfVectorizer(
            max_features=5000, analyzer="char", ngram_range=(2, 4),
            min_df=1, sublinear_tf=True)
        tfidf = self.vectorizer.fit_transform(corpus)
        n_comp = min(dim, tfidf.shape[1] - 1, len(corpus) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf)
        self.dim = n_comp
        print(f"    嵌入维度: {n_comp}")

    def encode(self, texts: List[str]) -> np.ndarray:
        vec = self.svd.transform(self.vectorizer.transform(texts))
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.where(norms == 0, 1.0, norms)


def build_vector_index(chunks: List[str], index_path: Path):
    """构建 FAISS 向量索引，同时保存 Embedder 模型"""
    import faiss, joblib
    print(f"    构建向量索引 ({len(chunks)} 块)...")
    embedder = TFIDFEmbedding(chunks, dim=384)
    vectors = embedder.encode(chunks).astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_path))
    embedder_path = index_path.with_suffix(".pkl")
    joblib.dump(embedder, embedder_path)
    print(f"    FAISS 索引已保存: {index_path} ({index.ntotal} 向量)")
    print(f"    Embedder 模型已保存: {embedder_path}")
    return index


# =============================================================================
# === IMPROVEMENT P1: 两轮抽取 — Entity-only 抽取函数 ===========================
# =============================================================================

def _process_chunk_entities(chunk: str, idx: int, total: int) -> Tuple[int, List[dict]]:
    """第一轮：从 chunk 中只抽取实体"""
    user = f"=== 待抽取 ===\n{chunk}"
    resp = llm_chat(ENTITY_ONLY_PROMPT, user, max_tokens=1500, temperature=0.1)
    result = extract_json_array(resp)
    if result is None:
        # 尝试解析为单个对象
        obj = extract_json(resp)
        if obj and "name" in obj and "type" in obj:
            result = [obj]
        else:
            result = []
    return idx, result


def _process_chunk_relations(chunk: str, known_entities: List[str], idx: int, total: int,
                              chunk_label: str = "") -> Tuple[int, List[dict]]:
    """第二轮：给定 chunk + 已知实体列表，只抽取关系"""
    entity_list_str = "\n".join(f"  - {name}" for name in sorted(known_entities))
    user = f"""## 已知实体列表
{entity_list_str}

## 待抽取文本（{chunk_label}）

{chunk}"""
    resp = llm_chat(RELATION_ONLY_PROMPT, user, max_tokens=1500, temperature=0.2)
    result = extract_json_array(resp)
    if result is None:
        # 尝试解析为单个对象
        obj = extract_json(resp)
        if obj and "head" in obj and "relation" in obj and "tail" in obj:
            result = [obj]
        else:
            result = []
    return idx, result


# =============================================================================
# === IMPROVEMENT P0: KPI 驱动增强 — 相关块检测 + 二次抽取 ======================
# =============================================================================

def _kpi_relevant_chunks(chunks: List[str], kpis: List[dict], top_k: int = 5) -> List[int]:
    """根据 KPI 关键词密度找出最相关的文本块索引

    Args:
        chunks: 所有文本块
        kpis: 解析后的 KPI 列表
        top_k: 返回最相关的前 top_k 个块索引

    Returns:
        排序后的块索引列表（按相关性降序）
    """
    if not kpis:
        return []

    # 收集所有 KPI 关键词
    kpi_keywords = set()
    for kpi in kpis:
        desc = kpi.get("description", "")
        # 提取重要词汇：中文词（至少2字）
        words = re.findall(r'[\u4e00-\u9fff]{2,}', desc)
        # 也保留英文缩写
        eng = re.findall(r'[A-Za-z0-9./°%]+', desc)
        kpi_keywords.update(words)
        kpi_keywords.update(eng)

    # 过滤通用词
    stopwords = {"指标", "考核", "以下", "要求", "目标", "上述", "如下",
                 "方法", "技术", "系统", "研究", "实验", "测试", "数据",
                 "设计", "开发", "制备", "实现", "采用", "使用", "通过",
                 "进行", "达到", "提高", "提升", "降低", "减少", "增加",
                 "完成", "交付", "产出", "形成", "建立", "构建", "开发",
                 "基于", "利用", "相关", "主要", "关键", "重要", "典型"}
    keywords = kpi_keywords - stopwords

    if not keywords:
        return []

    # 计算每个 chunk 的关键词密度
    scores = []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        # 去重计数（一个词出现多次只算一次）
        match_count = sum(1 for kw in keywords if kw in chunk)
        density = match_count / max(len(chunk), 1) * 10000  # 每万字的匹配数
        scores.append((i, density, match_count))

    # 按 (density, match_count) 综合排序
    scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top_indices = [s[0] for s in scores[:top_k] if s[1] > 0]
    return top_indices


def _kpi_boost_extraction(chunk: str, kpi_descriptions: List[str], idx: int, total: int) -> Tuple[int, Optional[dict]]:
    """KPI 驱动二次抽取：针对 KPI 相关块的深度抽取

    Args:
        chunk: 文本块
        kpi_descriptions: KPI 描述列表
        idx: chunk 索引
        total: chunk 总数

    Returns:
        (idx, {"entities": [...], "relations": [...]} or None)
    """
    kpi_context = "\n".join(f"  KPI{i+1}: {desc}" for i, desc in enumerate(kpi_descriptions[:5]))
    user = f"""## 相关考核指标
{kpi_context}

## 待抽取文本

{chunk}

请从上述文本中抽取与考核指标相关的技术实体和关系。"""
    resp = llm_chat(KPI_BOOST_SYSTEM_PROMPT, user, max_tokens=2000, temperature=0.15)
    result = extract_json(resp)
    if result and "entities" in result:
        return idx, result
    return idx, None


# =============================================================================
# === IMPROVEMENT P2: 跨块关系发现 ==============================================
# =============================================================================

def _discover_cross_chunk_relations(chunks: List[str],
                                     chunk_entity_map: dict,
                                     all_entity_names: set,
                                     concurrency: int = 1) -> List[dict]:
    """发现跨越文本块边界的实体关系

    Args:
        chunks: 所有文本块
        chunk_entity_map: {chunk_index: [entity_name, ...]} 每个块包含的实体
        all_entity_names: 全部实体名集合
        concurrency: 并发数

    Returns:
        [{head, relation, tail, context}, ...]
    """
    if not chunk_entity_map:
        return []

    # 找"实体对跨块"的组合：实体A在块i，实体B在块j (i < j)
    chunk_indices = sorted(chunk_entity_map.keys())
    candidate_pairs = []
    seen_pairs = set()

    for i in chunk_indices:
        for j in chunk_indices:
            if j <= i:
                continue
            ents_i = chunk_entity_map[i]
            ents_j = chunk_entity_map[j]
            # 找所有跨块实体对
            for ei in ents_i:
                for ej in ents_j:
                    if ei == ej:
                        continue
                    pair = (ei, ej)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        candidate_pairs.append({
                            "head_chunk": i, "tail_chunk": j,
                            "head": ei, "tail": ej,
                            "head_text": chunks[i][:300],
                            "tail_text": chunks[j][:300],
                        })

    if not candidate_pairs:
        return []

    # 用 LLM 发现跨块关系
    cross_relations = []
    # 限制候选数量（太多会超时）
    max_candidates = 30
    if len(candidate_pairs) > max_candidates:
        # 优先保留实体在多个块出现的候选
        multi_chunk_entities = set()
        for idx, ents in chunk_entity_map.items():
            for e in ents:
                count = sum(1 for e2_list in chunk_entity_map.values() if e in e2_list)
                if count > 1:
                    multi_chunk_entities.add(e)
        candidate_pairs.sort(
            key=lambda x: (x["head"] in multi_chunk_entities, x["tail"] in multi_chunk_entities),
            reverse=True)
        candidate_pairs = candidate_pairs[:max_candidates]

    # 按 (head_chunk, tail_chunk) 分组，合并同一对块的多对实体
    chunk_group = {}
    for pair in candidate_pairs:
        key = (pair["head_chunk"], pair["tail_chunk"])
        if key not in chunk_group:
            chunk_group[key] = {
                "head_chunk": key[0],
                "tail_chunk": key[1],
                "head_text": pair["head_text"],
                "tail_text": pair["tail_text"],
                "pairs": []
            }
        chunk_group[key]["pairs"].append({"head": pair["head"], "tail": pair["tail"]})

    groups = list(chunk_group.values())

    def _process_cross_chunk_group(group: dict) -> list:
        """处理单个跨块组"""
        pairs_str = "\n".join(f"  - {p['head']} (chunk{group['head_chunk']}) → {p['tail']} (chunk{group['tail_chunk']})"
                              for p in group["pairs"][:15])
        user = f"""## Chunk A (index={group['head_chunk']})
{group['head_text']}

## Chunk B (index={group['tail_chunk']})
{group['tail_text']}

## 候选跨块实体对
{pairs_str}

请判断这些候选对中哪些存在语义关系，输出关系结果。"""
        resp = llm_chat(CROSS_CHUNK_RELATION_PROMPT, user, max_tokens=1200, temperature=0.2)
        result = extract_json_array(resp)
        if result:
            return result
        return []

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=min(concurrency, 4)) as executor:
            futures = [executor.submit(_process_cross_chunk_group, g) for g in groups]
            for future in as_completed(futures):
                try:
                    rels = future.result()
                    cross_relations.extend(rels)
                except Exception as e:
                    print(f"    [跨块关系错误] {e}")
    else:
        for g in groups:
            try:
                rels = _process_cross_chunk_group(g)
                cross_relations.extend(rels)
            except Exception as e:
                print(f"    [跨块关系错误] {e}")

    return cross_relations


def _build_entity_chunk_index(all_entities: List[dict],
                                chunk_results: List[tuple]) -> dict:
    """构建实体 → 所在块索引 的映射

    Args:
        all_entities: 全部实体列表
        chunk_results: [(idx, entities_or_result), ...] 每个块的抽取结果

    Returns:
        {chunk_index: [entity_name, ...]}
    """
    entity_chunk_map = {}
    for idx, result in chunk_results:
        if result is None:
            continue
        entities = result if isinstance(result, list) else result.get("entities", [])
        names = [e["name"] for e in entities if isinstance(e, dict) and "name" in e]
        if names:
            entity_chunk_map[idx] = names
    return entity_chunk_map


# =============================================================================
# 主流程
# =============================================================================

def _process_chunk(chunk: str, idx: int, total: int):
    """单个 chunk 的 LLM 抽取（用于并行调用，原始单轮模式）"""
    user = FEW_SHOT + "\n\n" + f"=== 待抽取 ===\n{chunk}"
    resp = llm_chat(SYSTEM_PROMPT, user)
    result = extract_json(resp)
    return idx, result


def process_project(proj: dict, base_dir: Path, concurrency: int = 1,
                    use_two_round: bool = True,
                    use_kpi_boost: bool = True,
                    use_cross_chunk: bool = True) -> dict:
    """处理单个项目，构建知识图谱

    v2 新增参数:
        use_two_round:   启用两轮抽取（P1，实体→关系分离）
        use_kpi_boost:   启用 KPI 驱动增强（P0，对 KPI 相关块二次抽取）
        use_cross_chunk: 启用跨块关系发现（P2）
    """
    pid, pname = proj['id'], proj['name']
    text = proj["text"]
    print(f"\n{'='*50}")
    print(f"处理: {pname} ({pid})")
    print(f"  正文: {len(text)} 字")
    print(f"  v2模式: two_round={use_two_round}, kpi_boost={use_kpi_boost}, cross_chunk={use_cross_chunk}")

    # 1. 解析 KPI
    kpis = parse_kpi_section(text)
    print(f"  考核指标: {len(kpis)} 条")

    # 1b. 解析课题
    topics = parse_topics(text)
    print(f"  课题: {len(topics)} 个" if topics else "  课题: 无")

    # 2. 分块
    chunks = chunk_text(text)
    print(f"  文本分块: {len(chunks)} 块")

    t0 = time.time()

    if use_two_round and concurrency > 0:
        # ================================================================
        # v2 两轮抽取模式
        # ================================================================
        t_entity_start = time.time()

        # --- Round 1: 实体抽取 ---
        all_chunk_entities = []  # [(idx, [entity_dict, ...])]
        print(f"  [P1] 第一轮: 实体抽取 ({len(chunks)} 块)...")
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(_process_chunk_entities, chunk, i, len(chunks))
                           for i, chunk in enumerate(chunks)]
                for future in as_completed(futures):
                    idx, entities = future.result()
                    print(f"  块 {idx+1}/{len(chunks)} 实体 [{len(entities)}个]")
                    all_chunk_entities.append((idx, entities))
        else:
            for i, chunk in enumerate(chunks):
                idx, entities = _process_chunk_entities(chunk, i, len(chunks))
                print(f"  块 {idx+1}/{len(chunks)} 实体 [{len(entities)}个]")
                all_chunk_entities.append((idx, entities))

        # 收集全部实体名
        all_entity_names = set()
        for idx, entities in all_chunk_entities:
            for e in entities:
                if isinstance(e, dict) and "name" in e:
                    all_entity_names.add(e["name"])
        print(f"  [P1] 第一轮完成: 共 {len(all_entity_names)} 个唯一实体, 耗时 {time.time()-t_entity_start:.0f}s")

        # --- Round 2: 关系抽取（带实体约束） ---
        t_rel_start = time.time()
        all_chunk_relations = []  # [(idx, [relation_dict, ...])]
        print(f"  [P1] 第二轮: 关系抽取 ({len(chunks)} 块)...")
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(
                    _process_chunk_relations, chunks[i], list(all_entity_names), i, len(chunks),
                    f"chunk_{i}") for i in range(len(chunks))]
                for future in as_completed(futures):
                    idx, relations = future.result()
                    print(f"  块 {idx+1}/{len(chunks)} 关系 [{len(relations)}个]")
                    all_chunk_relations.append((idx, relations))
        else:
            for i, chunk in enumerate(chunks):
                idx, relations = _process_chunk_relations(chunk, list(all_entity_names), i, len(chunks))
                print(f"  块 {idx+1}/{len(chunks)} 关系 [{len(relations)}个]")
                all_chunk_relations.append((idx, relations))

        print(f"  [P1] 第二轮完成: 共 {sum(len(r) for _, r in all_chunk_relations)} 条关系, 耗时 {time.time()-t_rel_start:.0f}s")

        # 合并结果
        all_entities = []
        for idx, entities in all_chunk_entities:
            all_entities.extend(entities)
        all_relations = []
        for idx, relations in all_chunk_relations:
            all_relations.extend(relations)

        # ================================================================
        # P0: KPI 驱动增强（对 KPI 相关块进行二次抽取）
        # ================================================================
        if use_kpi_boost and kpis:
            t_kpi_start = time.time()
            kpi_descs = [k["description"] for k in kpis if k.get("description")]
            relevant_indices = _kpi_relevant_chunks(chunks, kpis, top_k=max(3, len(chunks) // 4))
            if relevant_indices:
                print(f"  [P0] KPI 驱动增强: {len(relevant_indices)} 个相关块")
                kpi_entities = []
                kpi_relations = []
                for idx in relevant_indices:
                    i, result = _kpi_boost_extraction(chunks[idx], kpi_descs, idx, len(chunks))
                    if result:
                        kpi_entities.extend(result.get("entities", []))
                        kpi_relations.extend(result.get("relations", []))
                        print(f"  块 {idx+1}/{len(chunks)} KPI增强 [{len(result.get('entities', []))}实体 {len(result.get('relations', []))}关系]")
                # 合并 KPI 增强结果
                all_entities.extend(kpi_entities)
                all_relations.extend(kpi_relations)
                print(f"  [P0] KPI增强完成: +{len(kpi_entities)}实体 +{len(kpi_relations)}关系, 耗时 {time.time()-t_kpi_start:.0f}s")
            else:
                print(f"  [P0] 未发现 KPI 相关块，跳过")

        # ================================================================
        # P2: 跨块关系发现
        # ================================================================
        if use_cross_chunk:
            t_cross_start = time.time()
            entity_chunk_index = _build_entity_chunk_index(all_entities, all_chunk_entities)
            cross_relations = _discover_cross_chunk_relations(
                chunks, entity_chunk_index, all_entity_names,
                concurrency=concurrency)
            if cross_relations:
                print(f"  [P2] 跨块关系: 发现 {len(cross_relations)} 条")
                all_relations.extend(cross_relations)
            else:
                print(f"  [P2] 未发现跨块关系")
            print(f"  [P2] 耗时 {time.time()-t_cross_start:.0f}s")

    else:
        # ================================================================
        # 原始单轮抽取模式（向后兼容）
        # ================================================================
        print(f"  原始模式: 单轮抽取 ({len(chunks)} 块)...")
        all_entities, all_relations = [], []
        if concurrency > 1:
            print(f"  并发数: {concurrency}")
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(_process_chunk, chunk, i, len(chunks))
                           for i, chunk in enumerate(chunks)]
                for future in as_completed(futures):
                    idx, result = future.result()
                    if result and "entities" in result and "relations" in result:
                        print(f"  块 {idx+1}/{len(chunks)} OK [{len(result['entities'])}实体 {len(result['relations'])}关系]")
                        all_entities.extend(result["entities"])
                        all_relations.extend(result["relations"])
                    else:
                        print(f"  块 {idx+1}/{len(chunks)} FAIL [失败]")
        else:
            for i, chunk in enumerate(chunks):
                idx, result = _process_chunk(chunk, i, len(chunks))
                if result and "entities" in result and "relations" in result:
                    print(f"  块 {idx+1}/{len(chunks)} OK [{len(result['entities'])}实体 {len(result['relations'])}关系]")
                    all_entities.extend(result["entities"])
                    all_relations.extend(result["relations"])
                else:
                    print(f"  块 {idx+1}/{len(chunks)} FAIL [失败]")

    print(f"  LLM 抽取耗时: {time.time() - t0:.0f}s")

    # 4. 去重
    dedup_entities = []
    seen_names = set()
    for e in all_entities:
        if e["name"] not in seen_names:
            seen_names.add(e["name"])
            dedup_entities.append(e)
    dedup_relations = []
    seen_rel = set()
    for r in all_relations:
        key = (r["head"], r["relation"], r["tail"])
        if key not in seen_rel:
            seen_rel.add(key)
            dedup_relations.append(r)

    print(f"  原始: {len(all_entities)}实体 {len(all_relations)}关系")
    print(f"  去重: {len(dedup_entities)}实体 {len(dedup_relations)}关系")

    # 5. 本体归一化
    entities, relations = normalize(dedup_entities, dedup_relations)
    print(f"  本体对齐: {len(entities)}实体 {len(relations)}关系")
    for t in sorted(set(e["type"] for e in entities)):
        cnt = sum(1 for e in entities if e["type"] == t)
        print(f"    {t}: {cnt}")

    # 5b. 课题实体 & 关系集成
    topic_kpi_map = None
    if topics:
        topic_kpi_map = match_kpis_to_topics(kpis, topics)
        topic_entities = create_topic_entities(topics)
        create_topic_relations(topics, topic_kpi_map, entities, relations)
        entities.extend(topic_entities)
        bel_cnt = sum(1 for r in relations if r['relation'] == 'BELONGS_TO' and r['tail'].startswith('课题'))
        print(f"  课题集成: +{len(topic_entities)}实体, +{bel_cnt}关系")

    # 6. 构建 KV 索引
    entity_ctx = {e["name"]: [] for e in entities}
    for rel in relations:
        ctx = rel.get("context", "")
        if ctx:
            if rel["head"] in entity_ctx:
                entity_ctx[rel["head"]].append(ctx)
            if rel["tail"] in entity_ctx:
                entity_ctx[rel["tail"]].append(ctx)
    low_kv = {e["name"]: "；".join(dict.fromkeys(entity_ctx.get(e["name"], [])))
              or f"{e['name']}（{e['type']}）" for e in entities}
    high_kv = {}
    for rel in relations:
        h, r, t, ctx = rel["head"], rel["relation"], rel["tail"], rel.get("context", "")
        high_kv[f"{h} {r} {t}"] = ctx or ""

    # 7. 保存结果
    out_dir = base_dir / pid
    out_dir.mkdir(parents=True, exist_ok=True)

    json.dump(entities, open(out_dir / "entities.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(relations, open(out_dir / "relations.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(kpis, open(out_dir / "kpis.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    if topics:
        json.dump(topics, open(out_dir / "topics.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    json.dump(low_kv, open(out_dir / "low_level_kv.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(high_kv, open(out_dir / "high_level_kv.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    for i, c in enumerate(chunks):
        (chunk_dir / f"chunk_{i:03d}.txt").write_text(c, encoding="utf-8")

    # 9. FAISS 向量索引
    try:
        build_vector_index(chunks, out_dir / "faiss.index")
        has_vector = True
    except Exception as e:
        print(f"  向量索引构建失败: {e}")
        has_vector = False

    # 10. 摘要
    etypes = {}
    for e in entities:
        etypes[e["type"]] = etypes.get(e["type"], 0) + 1
    rtypes = {}
    for rel in relations:
        rtypes[rel["relation"]] = rtypes.get(rel["relation"], 0) + 1

    # v2 版本标记
    summary = {
        "project_id": pid, "project_name": pname,
        "kpi_count": len(kpis),
        "topic_count": len(topics) if topics else 0,
        "entity_count": len(entities),
        "relation_count": len(relations),
        "chunk_count": len(chunks),
        "has_vector_index": has_vector,
        "entity_types": etypes,
        "relation_types": rtypes,
        "version": "v2",
        "v2_flags": {
            "use_two_round": use_two_round,
            "use_kpi_boost": use_kpi_boost,
            "use_cross_chunk": use_cross_chunk,
        },
    }
    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"  输出: {out_dir}")
    return summary


def main():
    """Standalone mode — process first MAX_PROJECTS files from INPUT_DIR"""
    print("=" * 60)
    print("KG Builder v2 — Ontology-Aligned + Completeness Boost")
    print("=" * 60)
    print("Entity types: %d" % len(ENTITY_TYPES))
    print("Relation types: %d" % len(RELATION_TYPES))
    print("v2 features: two_round=True, kpi_boost=True, cross_chunk=True")

    if not INPUT_DIR.exists():
        print("[Error] Input dir not found: %s" % INPUT_DIR)
        sys.exit(1)

    files = sorted(INPUT_DIR.glob("*.txt"))
    print("Input files: %d (processing first %d)" % (len(files), MAX_PROJECTS))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for fp in files[:MAX_PROJECTS]:
        proj = parse_project_file(fp)
        if proj:
            r = process_project(proj, OUTPUT_DIR,
                                use_two_round=True,
                                use_kpi_boost=True,
                                use_cross_chunk=True)
            results.append(r)

    print("")
    print("=" * 60)
    print("All done!")
    for r in results:
        faiss_flag = "Y" if r.get("has_vector_index") else "N"
        v2_tag = "v2" if r.get("version") == "v2" else "v1"
        print("  %s: %d entities %d relations [FAISS=%s] [%s]" % (
            r["project_name"], r["entity_count"], r["relation_count"], faiss_flag, v2_tag))

    json.dump(results, open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("Manifest: %s" % (OUTPUT_DIR / "manifest.json"))


if __name__ == "__main__":
    main()
