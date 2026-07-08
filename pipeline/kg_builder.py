"""
知识图谱构建引擎 — 本体对齐版
================================
严格按照 submission_ontology.py 定义的模本体构建知识图谱。

注意: Windows 下运行请先设置环境变量 SET PYTHONIOENCODING=utf-8
       或使用 python -u pipeline/run_kg_batch.py ... 配合编码设置。

核心改进：
  1. 实体类型：从自由类型 → 9 种本体约束类型（4核心 + 5辅助）
  2. 关系谓词：从自由谓词 → 6 种本体约束谓词（VIA/VERIFIES/EXECUTES/PRODUCES/BELONGS_TO/MAPS_TO）
  3. KPI 解析模块：从"考核指标"章节提取结构化指标
  4. FAISS 向量索引：文本块检索增强
  5. 课题→KPI层级建模：课题作为OBJECT实体，与KPI通过BELONGS_TO关联

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
API_BASE = "http://10.3.213.253:23001"
API_KEY = "sk-259d53cf77064362aa19c816c1321e7b"
LLM_MODEL = "qwen3-32b"

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
# Prompt — 本体对齐
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
# 主流程
# =============================================================================

def _process_chunk(chunk: str, idx: int, total: int):
    """单个 chunk 的 LLM 抽取（用于并行调用）"""
    user = FEW_SHOT + "\n\n" + f"=== 待抽取 ===\n{chunk}"
    resp = llm_chat(SYSTEM_PROMPT, user)
    result = extract_json(resp)
    return idx, result


def process_project(proj: dict, base_dir: Path, concurrency: int = 1) -> dict:
    pid, pname = proj['id'], proj['name']
    text = proj["text"]
    print(f"\n{'='*50}")
    print(f"处理: {pname} ({pid})")
    print(f"  正文: {len(text)} 字")

    # 1. 解析 KPI
    kpis = parse_kpi_section(text)
    print(f"  考核指标: {len(kpis)} 条")

    # 1b. 解析课题
    topics = parse_topics(text)
    print(f"  课题: {len(topics)} 个" if topics else "  课题: 无")

    # 2. 分块
    chunks = chunk_text(text)
    print(f"  文本分块: {len(chunks)} 块")

    # 3. 并行 LLM 抽取
    all_entities, all_relations = [], []
    t0 = time.time()
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
    }
    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"  输出: {out_dir}")
    return summary


def main():
    """Standalone mode — process first MAX_PROJECTS files from INPUT_DIR"""
    print("=" * 60)
    print("KG Builder — Ontology-Aligned")
    print("=" * 60)
    print("Entity types: %d" % len(ENTITY_TYPES))
    print("Relation types: %d" % len(RELATION_TYPES))

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
            r = process_project(proj, OUTPUT_DIR)
            results.append(r)

    print("")
    print("=" * 60)
    print("All done!")
    for r in results:
        faiss_flag = "Y" if r.get("has_vector_index") else "N"
        print("  %s: %d entities %d relations [FAISS=%s]" % (
            r["project_name"], r["entity_count"], r["relation_count"], faiss_flag))

    json.dump(results, open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("Manifest: %s" % (OUTPUT_DIR / "manifest.json"))


if __name__ == "__main__":
    main()
