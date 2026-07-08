#!/usr/bin/env python3
"""
LightRAG 知识图谱抽取 — 清洗后任务书版本

从 output/project_cleaned/ 读取 5 份清洗后的项目任务书，
按 LightRAG 3 阶段流水线处理：
  阶段1 (Recog): 分块 → LLM 逐块抽取实体+关系
  阶段2 (Profiling): 生成节点 profile + 关系 key
  阶段3 (Dedup): 跨块语义去重

输出：output/lightrag_extract_cleaned/{project_id}/
"""

import json
import re
import time
import sys
from pathlib import Path
from typing import List, Optional
import requests

# =============================================================================
# API 配置
# =============================================================================
API_BASE = "http://10.3.213.253:23001"
API_KEY = "sk-259d53cf77064362aa19c816c1321e7b"
MODEL = "qwen3-32b"

# =============================================================================
# LightRAG 抽取 Prompt
# =============================================================================
SYSTEM_PROMPT_KG = """你是一个科研文本知识图谱抽取专家。你的任务是从"国家重点研发计划"项目任务书中抽取实体和关系，构建知识图谱。

## 实体类型（严格使用以下6种）

| 类型 | 含义 | 示例 |
|------|------|------|
| TOPIC | 课题名称 | 课题一：高精度光纤陀螺仪研制、课题二 |
| OBJECT | 研究对象（器件/系统/材料） | 光纤陀螺仪、MEMS加速度计、铜铟镓硒薄膜电池、真空腔体 |
| METRIC | 指标/参数 | 零偏稳定性、真空度、组件平均效率、成品合格率 |
| METHOD | 方法/工艺/技术 | 温度补偿方法、晶圆级真空封装、磁控溅射、标定测试 |
| EQUIPMENT | 测试/制造设备 | 氦质谱仪、高低温试验箱、共蒸发设备 |
| ACHIEVEMENT | 成果形式 | 数据集、样机、系统、软件 |

## 关系类型（严格使用以下7种）

| 谓词 | 头→尾 | 语义 |
|------|-------|------|
| 归属 | TOPIC→OBJECT | 课题的研究对象是什么 |
| 考核 | TOPIC→METRIC | 课题考核什么指标 |
| 产出 | TOPIC→ACHIEVEMENT | 课题产出的成果类型 |
| 采用 | OBJECT→METHOD, OBJECT→EQUIPMENT | 对象使用什么方法/设备 |
| 测试 | EQUIPMENT→METRIC, METHOD→METRIC | 测试什么指标 |
| 改进 | METHOD→METRIC | 方法提升了什么指标 |
| 包含 | OBJECT→OBJECT | 组成/包含关系 |

## 抽取规则

1. **课题信息优先**：遇到"课题X"章节，优先抽取TOPIC实体和其考核指标、成果
2. 实体名使用原文完整名称
3. **不抽取**：人名、机构名、项目名称、单纯的现象/问题/理论概念
4. 关系必须有原文明确支撑
5. 同一段落中同一对实体之间的重复关系只输出一次

## 输出格式

```json
{
  "entities": [
    {"name": "实体名称", "type": "实体类型"},
    ...
  ],
  "relations": [
    {"head": "头实体", "relation": "关系谓词", "tail": "尾实体", "context": "包含该关系的原文片段(20-50字)"},
    ...
  ]
}
```

如果某段没有符合条件的实体，输出 {"entities": [], "relations": []}"""

FEW_SHOT_EXAMPLES = """
=== 示例1：考核指标表 ===
段落:
课题一：高精度光纤陀螺仪关键技术研究
考核指标：
1. 零偏稳定性≤0.01°/h，成果形式：数据集1套
2. 标度因数重复性≤10ppm，成果形式：数据集1套
课题二：MEMS加速度计关键技术研究
考核指标：
1. 零偏稳定性≤0.1mg，成果形式：样机1台

输出:
{
  "entities": [
    {"name": "课题一：高精度光纤陀螺仪关键技术研究", "type": "TOPIC"},
    {"name": "光纤陀螺仪", "type": "OBJECT"},
    {"name": "零偏稳定性", "type": "METRIC"},
    {"name": "标度因数重复性", "type": "METRIC"},
    {"name": "数据集", "type": "ACHIEVEMENT"},
    {"name": "课题二：MEMS加速度计关键技术研究", "type": "TOPIC"},
    {"name": "MEMS加速度计", "type": "OBJECT"},
    {"name": "零偏稳定性", "type": "METRIC"}
  ],
  "relations": [
    {"head": "课题一：高精度光纤陀螺仪关键技术研究", "relation": "归属", "tail": "光纤陀螺仪", "context": "课题一：高精度光纤陀螺仪关键技术研究"},
    {"head": "课题一：高精度光纤陀螺仪关键技术研究", "relation": "考核", "tail": "零偏稳定性", "context": "零偏稳定性≤0.01°/h"},
    {"head": "课题一：高精度光纤陀螺仪关键技术研究", "relation": "考核", "tail": "标度因数重复性", "context": "标度因数重复性≤10ppm"},
    {"head": "课题一：高精度光纤陀螺仪关键技术研究", "relation": "产出", "tail": "数据集", "context": "成果形式：数据集1套"},
    {"head": "课题二：MEMS加速度计关键技术研究", "relation": "归属", "tail": "MEMS加速度计", "context": "课题二：MEMS加速度计关键技术研究"},
    {"head": "课题二：MEMS加速度计关键技术研究", "relation": "考核", "tail": "零偏稳定性", "context": "零偏稳定性≤0.1mg"}
  ]
}

=== 示例2：研究内容段落 ===
段落:
针对 MEMS 加速度计的温度漂移问题，提出了一种基于差分电容检测的温度补偿方法。通过高低温试验箱对加速度计进行全温度范围测试，测试其零偏稳定性和标度因数重复性。实验结果表明，该补偿方法将零偏稳定性从 0.5mg 提升到 0.1mg。

输出:
{
  "entities": [
    {"name": "MEMS加速度计", "type": "OBJECT"},
    {"name": "差分电容检测温度补偿方法", "type": "METHOD"},
    {"name": "高低温试验箱", "type": "EQUIPMENT"},
    {"name": "零偏稳定性", "type": "METRIC"},
    {"name": "标度因数重复性", "type": "METRIC"}
  ],
  "relations": [
    {"head": "MEMS加速度计", "relation": "采用", "tail": "差分电容检测温度补偿方法", "context": "提出了一种基于差分电容检测的温度补偿方法"},
    {"head": "MEMS加速度计", "relation": "采用", "tail": "高低温试验箱", "context": "通过高低温试验箱对加速度计进行全温度范围测试"},
    {"head": "高低温试验箱", "relation": "测试", "tail": "零偏稳定性", "context": "测试其零偏稳定性和标度因数重复性"},
    {"head": "高低温试验箱", "relation": "测试", "tail": "标度因数重复性", "context": "测试其零偏稳定性和标度因数重复性"},
    {"head": "差分电容检测温度补偿方法", "relation": "改进", "tail": "零偏稳定性", "context": "该补偿方法将零偏稳定性从0.5mg提升到0.1mg"}
  ]
}

=== 示例3：包含关系 ===
段落:
晶圆级真空封装主要涉及键合区域金属化、封帽晶圆凹槽的刻蚀、封帽晶圆上吸气剂的制作、器件晶圆上 MEMS 结构制备等部分。真空腔体的气密性采用常规氦质谱仪进行检测。

输出:
{
  "entities": [
    {"name": "键合区域金属化", "type": "METHOD"},
    {"name": "封帽晶圆凹槽刻蚀", "type": "METHOD"},
    {"name": "真空腔体", "type": "OBJECT"},
    {"name": "氦质谱仪", "type": "EQUIPMENT"},
    {"name": "气密性", "type": "METRIC"}
  ],
  "relations": [
    {"head": "真空腔体", "relation": "采用", "tail": "氦质谱仪", "context": "真空腔体的气密性采用常规氦质谱仪进行检测"},
    {"head": "氦质谱仪", "relation": "测试", "tail": "气密性", "context": "真空腔体的气密性采用常规氦质谱仪进行检测"}
  ]
}

=== 示例4（反例）===
段落:
本项目由上海交通大学联合东风汽车集团股份有限公司、北京工业大学等十家单位联合申报，在铝合金制备加工方面形成上百项专利技术。

输出:
{"entities": [], "relations": []}
（解释：项目背景介绍，无技术实体和关系）"""


def llm_chat(system: str, user: str, max_tokens: int = 2000,
             temperature: float = 0.3, timeout: int = 180) -> str:
    """通用 LLM Chat 调用封装"""
    url = f"{API_BASE}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": MODEL,
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
                print(f"  [重试 {attempt+1}/3] {e}")
                time.sleep(wait)
                continue
            return f"[ERROR] {e}"
    return "[ERROR] max retries exceeded"


def extract_json(text: str) -> Optional[dict]:
    """从 LLM 回复中健壮地提取 JSON（含 think 标签处理）"""
    text = text.strip()
    # 移除 <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 处理未闭合的 <think> 标签（常见于 max_tokens 截断）
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = text.strip()

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

    # 最外层花括号
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


def parse_project_file(filepath: Path) -> Optional[dict]:
    """解析清洗后的任务书文件，提取元信息和正文"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取头部信息
    name_match = re.search(r'项目名称:\s*(.+)', content)
    id_match = re.search(r'项目ID:\s*(\d+)', content)
    pno_match = re.search(r'项目编号:\s*(\S+)', content)

    if not name_match or not id_match:
        print(f"  [跳过] 无法解析文件头: {filepath.name}")
        return None

    project_name = name_match.group(1).strip()
    project_id = id_match.group(1).strip()
    project_no = pno_match.group(1).strip() if pno_match else ""

    # 正文从 === 分隔线之后开始
    body_start = content.find('====')
    if body_start > 0:
        # 跳过分隔行
        body_start = content.find('\n', body_start)
        body_text = content[body_start:].strip()
    else:
        body_text = content

    return {
        "id": project_id,
        "name": project_name,
        "projectNo": project_no,
        "text": body_text
    }


def chunk_text(text: str, max_chars: int = 800, min_chars: int = 200) -> List[str]:
    """将文本按段落切分为块"""
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 20:
            continue

        if len(current) + len(para) < max_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current and len(current) >= min_chars:
                chunks.append(current)
            if len(para) > max_chars:
                sentences = re.split(r'(?<=[。；])', para)
                current = ""
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if len(current) + len(sent) < max_chars:
                        current = (current + sent).strip()
                    else:
                        if current and len(current) >= min_chars:
                            chunks.append(current)
                        current = sent
            else:
                current = para

    if current and len(current) >= min_chars:
        chunks.append(current)

    return chunks


def extract_chunk(chunk_text: str, section_id: str) -> Optional[dict]:
    """对单块调用 LLM 抽取实体和关系"""
    user_msg = f"""=== 待抽取段落 ===
{chunk_text}

=== 章节信息 ===
{section_id}

请抽取该段落的实体和关系，输出 JSON："""

    full_user = FEW_SHOT_EXAMPLES + "\n\n" + user_msg

    print(f"    LLM 调用 ({len(chunk_text)} 字)...", end=" ", flush=True)
    response = llm_chat(SYSTEM_PROMPT_KG, full_user, max_tokens=2000, temperature=0.3)

    result = extract_json(response)
    if result is None:
        print(f"[JSON失败，重试]")
        retry_user = f"""请从以下段落抽取实体和关系。

段落：
{chunk_text}

请直接输出 JSON（不要思考标记，不要解释）：
{{"entities": [{{"name": "...", "type": "..."}}], "relations": [{{"head": "...", "relation": "...", "tail": "...", "context": "..."}}]}}"""
        response2 = llm_chat(SYSTEM_PROMPT_KG, retry_user, max_tokens=2000, temperature=0.1)
        result = extract_json(response2)
        if result is None:
            print(f"[失败] {response[:80]}...")
            return None

    if not isinstance(result, dict) or 'entities' not in result or 'relations' not in result:
        print(f"[结构异常]")
        return None

    ents = result.get('entities', [])
    rels = result.get('relations', [])
    print(f"[OK {len(ents)}实体 {len(rels)}关系]")
    return result


def build_profile(entities: List[dict], relations: List[dict]) -> tuple:
    """构建节点 profile + 关系 key"""
    entity_ctx = {e['name']: [] for e in entities}
    for rel in relations:
        ctx = rel.get('context', '')
        h, t = rel['head'], rel['tail']
        if ctx:
            if h in entity_ctx:
                entity_ctx[h].append(ctx)
            if t in entity_ctx:
                entity_ctx[t].append(ctx)

    low_kv = {}
    for ent in entities:
        ctxs = list(dict.fromkeys(entity_ctx.get(ent['name'], [])))
        low_kv[ent['name']] = "；".join(ctxs) if ctxs else f"{ent['name']}（{ent['type']}）"

    high_kv = {}
    for rel in relations:
        h, r, t, ctx = rel['head'], rel['relation'], rel['tail'], rel.get('context', '')
        for key in [f"{h} {r} {t}", f"{h} {t} {r}"]:
            if key and len(key) > 2:
                high_kv[key] = ctx or f"{h}{r}{t}"
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', f"{h} {t} {ctx}")
        if len(words) >= 3:
            high_kv[" ".join(words[:4])] = ctx or ""

    return low_kv, high_kv


def edit_distance(s1: str, s2: str) -> int:
    if abs(len(s1) - len(s2)) > 5:
        return 999
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def dedup_entities(entities: List[dict]) -> List[dict]:
    merged, names = [], set()
    for ent in entities:
        name = ent['name']
        if name in names:
            continue
        found = False
        for ex in merged:
            en = ex['name']
            if edit_distance(name, en) < 3 or ((name in en or en in name) and ex.get('type') == ent.get('type')):
                found = True
                break
        if not found:
            merged.append(ent)
            names.add(name)
    return merged


def dedup_relations(relations: List[dict]) -> List[dict]:
    seen, out = set(), []
    for rel in relations:
        key = (rel['head'], rel['tail'])
        if key not in seen:
            seen.add(key)
            out.append(rel)
    return out


# =============================================================================
# 粒度归一化 — 实体类型 & 关系谓词归并映射
# =============================================================================
# 目标：仅保留与数据集名称生成相关的实体类型
# 数据集名称要素：研究对象(COMPONENT/SYSTEM/MATERIAL) + 指标(METRIC) + 方法(METHOD/PROCESS) + 设备(EQUIPMENT)

# 实体类型归并映射：细粒度类型 → 规范类型（6种）
ENTITY_TYPE_MERGE = {
    # TOPIC
    "TOPIC": "TOPIC", "SUBJECT": "TOPIC",

    # OBJECT — 研究对象（器件/系统/材料）
    "OBJECT": "OBJECT",
    "COMPONENT": "OBJECT", "PART": "OBJECT", "MODULE": "OBJECT", "DEVICE": "OBJECT",
    "SYSTEM": "OBJECT", "PLATFORM": "OBJECT",
    "MATERIAL": "OBJECT", "CHEMICAL": "OBJECT", "COMPOUND": "OBJECT",
    "LAYER": "OBJECT", "STRUCTURE": "OBJECT",
    "PRODUCT": "OBJECT",

    # METRIC
    "METRIC": "METRIC",
    "PARAMETER": "METRIC", "INDEX": "METRIC", "INDICATOR": "METRIC",
    "PROCESS_PARAMETER": "METRIC",

    # METHOD — 方法/工艺/技术
    "METHOD": "METHOD",
    "ALGORITHM": "METHOD", "TECHNOLOGY": "METHOD",
    "PROCESS": "METHOD", "CRAFT": "METHOD",
    "SOFTWARE": "METHOD",

    # EQUIPMENT
    "EQUIPMENT": "EQUIPMENT",
    "INSTRUMENT": "EQUIPMENT", "APPARATUS": "EQUIPMENT",

    # ACHIEVEMENT
    "ACHIEVEMENT": "ACHIEVEMENT",
    "DATASET": "ACHIEVEMENT", "DATA": "ACHIEVEMENT",
    "MODEL": "ACHIEVEMENT",
}

# 非数据集相关实体类型 — 直接过滤
NON_TECH_ENTITY_TYPES = {
    "PERSON", "ORGANIZATION", "PROJECT", "DOCUMENT",
    "FIGURE", "TABLE", "POSITION", "TIME", "LOCATION",
    "EVENT", "STANDARD", "REGULATION",
    "FIELD", "DOMAIN", "DIRECTION",
    "CONCEPT", "THEORY", "KNOWLEDGE",
    "RESEARCH_TOPIC",
    "PHENOMENON", "PROBLEM", "SCIENCE_PROBLEM",
    "DEFECT", "STATE", "PROPERTY",
    "MECHANISM", "PRINCIPLE", "PHYSICAL_FIELD",
    "ELECTRONIC_STRUCTURE",
    "RESULT", "BENEFIT", "ADVANTAGE",
    "GOAL", "OBJECTIVE", "TASK",
    "INDUSTRY", "SECTION",
}

# 关系谓词归并映射 — 7种标准谓词
RELATION_MERGE = {
    # 归属（课题→对象）
    "归属": "归属", "属于": "归属", "隶属于": "归属",

    # 考核（课题→指标）
    "考核": "考核", "指标要求": "考核", "要求": "考核",

    # 产出（课题→成果）
    "产出": "产出", "交付": "产出", "成果形式": "产出",

    # 采用（对象→方法/设备）
    "采用": "采用", "使用": "采用", "应用": "采用", "利用": "采用",
    "运用": "采用", "通过": "采用", "基于": "采用",

    # 测试（设备/方法→指标）
    "测试": "测试", "检测": "测试", "测量": "测试", "标定": "测试",
    "校验": "测试", "试验": "测试", "实验": "测试",
    "验证": "测试", "证实": "测试", "确认": "测试",
    "测试方法": "测试", "检测方法": "测试", "分析方法": "测试",

    # 改进（方法→指标）
    "改进": "改进", "提升": "改进", "提高": "改进", "优化": "改进",
    "改善": "改进", "增强": "改进", "促进": "改进", "加速": "改进",
    "抑制": "改进", "降低": "改进", "减少": "改进", "补偿": "改进",
    "消除": "改进", "解决": "改进", "突破": "改进",

    # 包含（对象→对象，或泛指构成关系）
    "包含": "包含", "包括": "包含", "涉及": "包含", "分为": "包含",
    "集成": "包含", "融合": "包含", "整合": "包含", "结合": "包含",
    "实现": "包含", "构建": "包含", "建立": "包含", "开发": "包含",
    "研制": "包含", "制备": "包含", "形成": "包含", "产生": "包含",
    "设计": "包含", "规划": "包含", "制定": "包含",
    "研究": "包含", "分析": "包含", "探索": "包含",
    "提出": "包含", "给出": "包含", "引入": "包含",
}

# 非技术关系谓词 — 直接过滤
NON_TECH_RELATIONS = {
    "负责", "主持", "研究方向", "从事", "参与", "承担", "协助",
    "申报", "依托", "组织", "牵头", "联合",
    "发表", "撰写", "出版", "提交", "申请", "授权",
    "获得国际认可",
    "培养", "引进", "交流", "合作",
}


def normalize_entity_type(raw_type: str) -> Optional[str]:
    """归一化实体类型，返回规范类型；非技术类型返回 None（过滤）"""
    t = raw_type.strip().upper()
    if t in ENTITY_TYPE_MERGE:
        return ENTITY_TYPE_MERGE[t]
    if t in NON_TECH_ENTITY_TYPES:
        return None
    # 未知类型尝试包含匹配
    for k, v in ENTITY_TYPE_MERGE.items():
        if k in t or t in k:
            return v
    return t  # 默认保留，避免过度过滤


def normalize_relation(relation: str) -> Optional[str]:
    """归一化关系谓词；非技术关系返回 None（过滤）"""
    r = relation.strip()
    if r in RELATION_MERGE:
        return RELATION_MERGE[r]
    if r in NON_TECH_RELATIONS:
        return None
    return r


def normalize_extraction(entities: List[dict], relations: List[dict]) -> tuple:
    """对抽取结果做粒度归一化：类型归并 + 非技术过滤"""
    entity_map = {}
    for ent in entities:
        norm_type = normalize_entity_type(ent.get('type', ''))
        if norm_type is None:
            continue
        name = ent['name']
        if name not in entity_map:
            entity_map[name] = {
                "name": name, "type": norm_type,
                "chunk_ids": ent.get('chunk_ids', [])
            }
        else:
            # 合并跨块的 chunk_ids
            cids = ent.get('chunk_ids', [])
            for cid in cids:
                if cid not in entity_map[name]['chunk_ids']:
                    entity_map[name]['chunk_ids'].append(cid)

    valid_entity_names = set(entity_map.keys())

    # 仅做谓词归并 + 过滤，不做二次去重（原始已去重）
    norm_relations = []
    for rel in relations:
        h, t, r = rel.get('head', ''), rel.get('tail', ''), rel.get('relation', '')
        if h not in valid_entity_names or t not in valid_entity_names:
            continue
        norm_r = normalize_relation(r)
        if norm_r is None:
            continue
        norm_relations.append({
            "head": h, "relation": norm_r, "tail": t,
            "context": rel.get('context', ''),
            "chunk_id": rel.get('chunk_id', '')
        })

    return list(entity_map.values()), norm_relations


def save_results(proj_id: str, proj_name: str,
                 entities: List[dict], relations: List[dict],
                 chunks: List[str], low_kv: dict, high_kv: dict,
                 base_dir: Path) -> Path:
    out_dir = base_dir / proj_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # neighbors
    nb = {e['name']: [] for e in entities}
    for rel in relations:
        h, t, r = rel['head'], rel['tail'], rel['relation']
        if h in nb:
            nb[h].append({"name": t, "relation": r, "direction": "out"})
        if t in nb:
            nb[t].append({"name": h, "relation": r, "direction": "in"})

    entities_out = []
    for ent in entities:
        entities_out.append({
            "name": ent['name'], "type": ent['type'],
            "profile": low_kv.get(ent['name'], ""),
            "chunk_ids": ent.get('chunk_ids', []),
            "neighbors": nb.get(ent['name'], [])
        })

    with open(out_dir / "entities.json", 'w', encoding='utf-8') as f:
        json.dump(entities_out, f, ensure_ascii=False, indent=2)
    with open(out_dir / "relations.json", 'w', encoding='utf-8') as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)
    with open(out_dir / "low_level_kv.json", 'w', encoding='utf-8') as f:
        json.dump(low_kv, f, ensure_ascii=False, indent=2)
    with open(out_dir / "high_level_kv.json", 'w', encoding='utf-8') as f:
        json.dump(high_kv, f, ensure_ascii=False, indent=2)

    cdir = out_dir / "chunks"
    cdir.mkdir(exist_ok=True)
    for i, c in enumerate(chunks):
        with open(cdir / f"chunk_{i:03d}.txt", 'w', encoding='utf-8') as f:
            f.write(c)

    summary = {
        "project_id": proj_id, "project_name": proj_name,
        "entity_count": len(entities), "relation_count": len(relations),
        "chunk_count": len(chunks),
        "entity_types": {}, "relation_types": {}
    }
    for e in entities:
        t = e.get('type', 'UNKNOWN')
        summary['entity_types'][t] = summary['entity_types'].get(t, 0) + 1
    for rel in relations:
        r = rel.get('relation', 'UNKNOWN')
        summary['relation_types'][r] = summary['relation_types'].get(r, 0) + 1

    with open(out_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return out_dir


def process_project(proj: dict, base_dir: Path, filepath: Path) -> dict:
    """处理单个项目"""
    pid, pname = proj['id'], proj['name']
    text = proj['text']

    print(f"\n{'='*60}")
    print(f"处理: {pname}")
    print(f"  文件: {filepath.name}")
    print(f"  正文长度: {len(text)} 字")

    # 分块
    chunks = chunk_text(text)
    print(f"  切分: {len(chunks)} 块")
    # 打印前几块预览
    for i, c in enumerate(chunks[:3]):
        print(f"    块{i+1}: {len(c)}字 -> {c[:50]}...")
    if len(chunks) > 3:
        print(f"    ... 共{len(chunks)}块")

    # 逐块抽取
    all_entities, all_relations = [], []
    success_chunks = 0
    for i, chunk in enumerate(chunks):
        sid = f"{pid}/chunk_{i+1}"
        result = extract_chunk(chunk, sid)
        if result:
            for ent in result.get('entities', []):
                ent['chunk_ids'] = [sid]
                all_entities.append(ent)
            for rel in result.get('relations', []):
                rel['chunk_id'] = sid
                all_relations.append(rel)
            success_chunks += 1
        # 节流：避免请求过快
        time.sleep(0.5)

    print(f"\n  抽取结果: 成功{success_chunks}/{len(chunks)}块")
    print(f"  原始: {len(all_entities)}实体, {len(all_relations)}关系")

    if not all_entities:
        print("  [跳过] 无实体抽取到")
        return {"project_id": pid, "status": "no_entities"}

    # 去重
    de = dedup_entities(all_entities)
    dr = dedup_relations(all_relations)
    print(f"  去重: {len(de)}实体, {len(dr)}关系")

    # 粒度归一化：类型归并 + 非技术实体/关系过滤
    de, dr = normalize_extraction(de, dr)
    print(f"  归一化: {len(de)}实体, {len(dr)}关系")

    # Profiling
    lk, hk = build_profile(de, dr)
    print(f"  索引: {len(lk)} Low-Level, {len(hk)} High-Level")

    # 保存
    out = save_results(pid, pname, de, dr, chunks, lk, hk, base_dir)
    print(f"  保存: {out}")

    return {
        "project_id": pid, "project_name": pname,
        "status": "success", "chunks": len(chunks),
        "success_chunks": success_chunks,
        "entities_raw": len(all_entities), "entities_deduped": len(de),
        "relations_raw": len(all_relations), "relations_deduped": len(dr),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LightRAG 抽取（清洗后任务书版）")
    parser.add_argument("--input", default="output/project_cleaned",
                        help="清洗后任务书目录")
    parser.add_argument("--output", default="output/lightrag_extract_cleaned",
                        help="输出目录")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集 txt 文件
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        print(f"[错误] 未找到 .txt 文件: {input_dir}")
        sys.exit(1)

    print(f"找到 {len(files)} 个任务书文件")
    print(f"输出目录: {output_dir}")

    results = []
    for fp in files:
        proj = parse_project_file(fp)
        if proj is None:
            continue
        result = process_project(proj, output_dir, fp)
        results.append(result)
        # 项目间停顿
        time.sleep(1)

    # 总摘要
    print(f"\n{'='*60}")
    success = [r for r in results if r.get('status') == 'success']
    print(f"完成: {len(success)}/{len(results)} 项目成功")
    te = sum(r.get('entities_deduped', 0) for r in success)
    tr = sum(r.get('relations_deduped', 0) for r in success)
    print(f"总计: {te} 实体, {tr} 关系")

    with open(output_dir / "manifest.json", 'w', encoding='utf-8') as f:
        json.dump({
            "total": len(results), "success": len(success),
            "total_entities": te, "total_relations": tr,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"摘要: {output_dir / 'manifest.json'}")


if __name__ == '__main__':
    main()
