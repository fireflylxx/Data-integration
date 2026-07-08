#!/usr/bin/env python3
"""
LightRAG 知识图谱抽取脚本

对 project.jsonl 分块文件的前 N 条记录，按 LightRAG 方式：
  阶段1 (Recog): 研究内容分块 → LLM 逐块抽取实体+关系
  阶段2 (Profiling): 聚合生成节点 profile + 关系 key
  阶段3 (Dedup): 跨块语义去重

输出：output/lightrag_extract/{project_id}/ 目录
  - entities.json      # 去重后的实体列表
  - relations.json     # 去重后的关系列表
  - low_level_kv.json  # Key=实体名, Value=profile
  - high_level_kv.json # Key=主题词, Value=关系描述
  - chunks/            # 原文分块
"""

import json
import re
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional
import requests

# =============================================================================
# API 配置
# =============================================================================
API_BASE = "http://10.3.213.253:23001"
API_KEY = "sk-259d53cf77064362aa19c816c1321e7b"
MODEL = "qwen3-32b"

# =============================================================================
# LightRAG 抽取 Prompt（来自 KG抽取设计.md）
# =============================================================================

SYSTEM_PROMPT_KG = """你是一个科研文本知识图谱抽取专家。你的任务是从"国家重点研发计划"项目任务书的研究内容段落中，抽取实体和关系，构建知识图谱。

## 抽取方式

按标准三元组格式抽取：<头实体, 关系谓词, 尾实体>

## 实体抽取规则

1. 实体类型**不由预定义列表限制**，请根据文本内容动态判断最合适的类型
2. 可能的实体类型包括但不限于：TECHNOLOGY（技术）、EQUIPMENT（设备）、MATERIAL（材料）、SYSTEM（系统）、METHOD（方法/算法）、METRIC（指标/参数）、COMPONENT（组件）、DATASET（数据集）、SOFTWARE（软件）、PROCESS（工艺）、MODEL（模型）等
3. 实体名使用原文中的完整名称，保持原文表述
4. 不要抽取通用概念（如"研究"、"分析"、"问题"、"技术"单独出现时），只抽取具体、可指认的实体
5. 尽量覆盖段落中所有核心实体，不要遗漏

## 关系抽取规则

1. 关系谓词使用简洁的动词或介词短语，如：采用、通过、验证、检测、测试、实现、构建、提升、优化、集成、基于、包含、产生等
2. 关系必须有原文明确支撑，不要推断不存在的关系
3. 同一段落中同一对实体之间的重复关系只输出一次

## 输出格式

严格输出 JSON 对象，包含 entities 和 relations 两个数组：

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

如果某段没有符合条件的实体，输出 {"entities": [], "relations": []}"""

FEW_SHOT_EXAMPLES = """
=== 示例1 ===
段落:
晶圆级真空封装主要涉及键合区域金属化、封帽晶圆凹槽的刻蚀、封帽晶圆上吸气剂的制作、器件晶圆上 MEMS 结构制备等部分。真空腔体的气密性采用常规氦质谱仪进行检测，腔体内部的真空度采用基于真空与电阻相关的内置皮拉尼计进行测量，最后通过工艺条件的优化和测试验证，获得高可靠、高真空度的晶圆级封装工艺。

输出:
{
  "entities": [
    {"name": "晶圆级真空封装", "type": "PROCESS"},
    {"name": "真空腔体", "type": "COMPONENT"},
    {"name": "氦质谱仪", "type": "EQUIPMENT"},
    {"name": "皮拉尼计", "type": "EQUIPMENT"},
    {"name": "气密性", "type": "METRIC"},
    {"name": "真空度", "type": "METRIC"}
  ],
  "relations": [
    {"head": "真空腔体", "relation": "采用", "tail": "氦质谱仪", "context": "真空腔体的气密性采用常规氦质谱仪进行检测"},
    {"head": "气密性", "relation": "测试方法", "tail": "氦质谱仪", "context": "真空腔体的气密性采用常规氦质谱仪进行检测"},
    {"head": "真空腔体", "relation": "采用", "tail": "皮拉尼计", "context": "腔体内部的真空度采用内置皮拉尼计进行测量"},
    {"head": "真空度", "relation": "测试方法", "tail": "皮拉尼计", "context": "腔体内部的真空度采用内置皮拉尼计进行测量"},
    {"head": "晶圆级真空封装", "relation": "涉及", "tail": "键合区域金属化", "context": "晶圆级真空封装主要涉及键合区域金属化"},
    {"head": "晶圆级真空封装", "relation": "涉及", "tail": "封帽晶圆凹槽刻蚀", "context": "晶圆级真空封装主要涉及封帽晶圆凹槽的刻蚀"}
  ]
}

=== 示例2 ===
段落:
针对 MEMS 加速度计的温度漂移问题，提出了一种基于差分电容检测的温度补偿方法。通过高低温试验箱对加速度计进行全温度范围测试，测试其零偏稳定性和标度因数重复性。实验结果表明，该补偿方法将零偏稳定性从 0.5mg 提升到 0.1mg。

输出:
{
  "entities": [
    {"name": "MEMS加速度计", "type": "COMPONENT"},
    {"name": "差分电容检测温度补偿方法", "type": "METHOD"},
    {"name": "高低温试验箱", "type": "EQUIPMENT"},
    {"name": "零偏稳定性", "type": "METRIC"},
    {"name": "标度因数重复性", "type": "METRIC"}
  ],
  "relations": [
    {"head": "MEMS加速度计", "relation": "采用", "tail": "差分电容检测温度补偿方法", "context": "提出了一种基于差分电容检测的温度补偿方法"},
    {"head": "MEMS加速度计", "relation": "通过", "tail": "高低温试验箱", "context": "通过高低温试验箱对加速度计进行全温度范围测试"},
    {"head": "高低温试验箱", "relation": "测试", "tail": "零偏稳定性", "context": "测试其零偏稳定性和标度因数重复性"},
    {"head": "高低温试验箱", "relation": "测试", "tail": "标度因数重复性", "context": "测试其零偏稳定性和标度因数重复性"},
    {"head": "差分电容检测温度补偿方法", "relation": "提升", "tail": "零偏稳定性", "context": "该补偿方法将零偏稳定性从0.5mg提升到0.1mg"}
  ]
}

=== 示例3（反例）===
段落:
本项目由上海交通大学联合东风汽车集团股份有限公司、北京工业大学等十家单位联合申报，在铝合金制备加工方面形成上百项专利技术，承担过10余项国家重点研发计划。

输出:
{"entities": [], "relations": []}
（解释：背景介绍段落，无技术实体和关系，不抽取）
"""


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

    # 移除 <think>...</think> 思考标签
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()

    # 尝试直接解析
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # 尝试从 ```json 块中提取
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试从最外层花括号提取（逐层缩小范围）
    while text:
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            candidate = text[brace_start:brace_end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # 尝试移除后半部分不完整内容
                pass
        break

    return None


# =============================================================================
# 文本预处理
# =============================================================================

WATERMARK_PATTERN = re.compile(
    r'国家\s*基础\s*学科\s*公共\s*科学\s*数据\s*中心\s*内部\s*文件\s*请勿\s*使用\s*、?\s*外\s*传',
    re.DOTALL
)
PAGE_HEADER_PATTERN = re.compile(r'第\d+页/共\d+页')


def clean_text(text: str) -> str:
    """去除水印和页眉页脚"""
    text = WATERMARK_PATTERN.sub('', text)
    text = PAGE_HEADER_PATTERN.sub('', text)
    # 合并多余空白
    text = re.sub(r' +\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def find_research_content(all_text: str) -> str:
    """从 all 字段中定位研究内容章节并提取"""
    # 尝试多种可能的章节标题
    markers = [
        '项目的主要研究内容',
        '主要研究内容',
        '二、研究内容',
        '研究内容',
        '研究目标',
    ]
    for marker in markers:
        idx = all_text.find(marker)
        if idx > 0:
            # 从 marker 开始往后取
            start = idx
            # 找下一个主要章节或结束
            # 取最多 8000 字符作为研究内容
            end = min(start + 8000, len(all_text))
            return clean_text(all_text[start:end])
    return ""


def chunk_research_content(text: str, max_chars: int = 800, min_chars: int = 200) -> List[str]:
    """将研究内容按段落切分为块"""
    # 按双换行分割段落
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # 去掉过短的段落（可能是无内容段）
        if len(para) < 20:
            continue

        if len(current) + len(para) < max_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current and len(current) >= min_chars:
                chunks.append(current)
            # 如果段落本身超过 max_chars，按句切分
            if len(para) > max_chars:
                sentences = re.split(r'(?<=[。；])', para)
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


# =============================================================================
# 阶段1: Recog - 逐块抽取
# =============================================================================

def extract_chunk(chunk_text: str, section_id: str) -> Optional[dict]:
    """对单块调用 LLM 抽取实体和关系"""
    user_msg = f"""=== 待抽取段落 ===
{chunk_text}

=== 章节信息 ===
{section_id}

请抽取该段落的实体和关系，输出 JSON："""

    full_user = FEW_SHOT_EXAMPLES + "\n\n" + user_msg

    print(f"    调用 LLM (段落长度 {len(chunk_text)} 字)...")
    response = llm_chat(SYSTEM_PROMPT_KG, full_user, max_tokens=2000, temperature=0.3)

    result = extract_json(response)
    if result is None:
        print(f"    [警告] JSON 解析失败，尝试精简重试...")
        # 重试：用更简短的提示，强制只输出 JSON
        retry_user = f"""请从以下段落抽取实体和关系。

段落：
{chunk_text}

请直接输出 JSON（不要思考标记，不要解释）：
{{"entities": [{{"name": "...", "type": "..."}}], "relations": [{{"head": "...", "relation": "...", "tail": "...", "context": "..."}}]}}"""
        response2 = llm_chat(SYSTEM_PROMPT_KG, retry_user, max_tokens=2000, temperature=0.1)
        result = extract_json(response2)
        if result is None:
            print(f"    [重试失败] 原始回复: {response[:100]}...")
            return None

    # 验证结构
    if not isinstance(result, dict) or 'entities' not in result or 'relations' not in result:
        print(f"    [警告] 结构异常: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        return None

    return result


# =============================================================================
# 阶段2: Profiling - 生成 KV 索引
# =============================================================================

def build_profile(entities: List[dict], relations: List[dict]) -> tuple:
    """
    构建节点 profile 和关系 key。
    返回 (low_level_kv, high_level_kv)
    """
    # 节点 profile：聚合该实体在所有关系中的 context
    entity_contexts = {}
    for ent in entities:
        entity_contexts[ent['name']] = []

    for rel in relations:
        ctx = rel.get('context', '')
        head, tail = rel['head'], rel['tail']
        if ctx:
            if head in entity_contexts:
                entity_contexts[head].append(ctx)
            if tail in entity_contexts:
                entity_contexts[tail].append(ctx)

    low_level_kv = {}
    for ent in entities:
        contexts = entity_contexts.get(ent['name'], [])
        # 去重拼接
        unique_ctx = list(dict.fromkeys(contexts))
        profile = "；".join(unique_ctx) if unique_ctx else f"{ent['name']}（{ent['type']}）"
        low_level_kv[ent['name']] = profile

    # 关系 key：从 head + relation + tail + context 提取关键词组合
    high_level_kv = {}
    for rel in relations:
        head, relation, tail, context = rel['head'], rel['relation'], rel['tail'], rel.get('context', '')
        # 生成多组主题词
        key1 = f"{head} {relation} {tail}"
        key2 = f"{head} {tail} {relation}"
        # 从中文字符中提取关键词（去掉停用词）
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', f"{head} {tail} {context}")
        key3 = " ".join(words[:4]) if len(words) >= 4 else " ".join(words)

        desc = context if context else f"{head}{relation}{tail}"
        for key in [key1, key2, key3]:
            if key and len(key) > 2:
                high_level_kv[key] = desc

    return low_level_kv, high_level_kv


# =============================================================================
# 阶段3: Dedup - 去重融合
# =============================================================================

def edit_distance(s1: str, s2: str) -> int:
    """计算编辑距离"""
    if abs(len(s1) - len(s2)) > 5:
        return 999
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def dedup_entities(entities: List[dict]) -> List[dict]:
    """跨块实体去重"""
    merged = []
    merged_names = set()

    for ent in entities:
        name = ent['name']
        if name in merged_names:
            continue

        # 检查是否需要与已有实体合并
        found = False
        for i, existing in enumerate(merged):
            ename = existing['name']
            # 规则1: 编辑距离 < 3
            if edit_distance(name, ename) < 3:
                found = True
                break
            # 规则2: 包含关系且同类型
            if (name in ename or ename in name) and existing.get('type') == ent.get('type'):
                found = True
                break

        if not found:
            merged.append(ent)
            merged_names.add(name)

    return merged


def dedup_relations(relations: List[dict]) -> List[dict]:
    """关系去重：按 (head, tail) 对去重，保留第一个出现的关系"""
    seen_pairs = set()
    deduped = []
    for rel in relations:
        key = (rel['head'], rel['tail'])
        if key not in seen_pairs:
            seen_pairs.add(key)
            deduped.append(rel)
    return deduped


# =============================================================================
# 输出保存
# =============================================================================

def save_results(project_id: str, project_name: str,
                 entities: List[dict], relations: List[dict],
                 chunks: List[str],
                 low_level_kv: dict, high_level_kv: dict,
                 base_dir: Path):
    """保存抽取结果到文件"""
    out_dir = base_dir / project_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 为实体添加 neighbors 信息
    entity_neighbors = {e['name']: [] for e in entities}
    for rel in relations:
        h, t, r = rel['head'], rel['tail'], rel['relation']
        if h in entity_neighbors:
            entity_neighbors[h].append({"name": t, "relation": r, "direction": "out"})
        if t in entity_neighbors:
            entity_neighbors[t].append({"name": h, "relation": r, "direction": "in"})

    entities_out = []
    for ent in entities:
        entities_out.append({
            "name": ent['name'],
            "type": ent['type'],
            "profile": low_level_kv.get(ent['name'], ""),
            "neighbors": entity_neighbors.get(ent['name'], [])
        })

    # entities.json
    with open(out_dir / "entities.json", 'w', encoding='utf-8') as f:
        json.dump(entities_out, f, ensure_ascii=False, indent=2)

    # relations.json
    with open(out_dir / "relations.json", 'w', encoding='utf-8') as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)

    # low_level_kv.json
    with open(out_dir / "low_level_kv.json", 'w', encoding='utf-8') as f:
        json.dump(low_level_kv, f, ensure_ascii=False, indent=2)

    # high_level_kv.json
    with open(out_dir / "high_level_kv.json", 'w', encoding='utf-8') as f:
        json.dump(high_level_kv, f, ensure_ascii=False, indent=2)

    # chunks
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    for i, chunk in enumerate(chunks):
        with open(chunks_dir / f"chunk_{i:03d}.txt", 'w', encoding='utf-8') as f:
            f.write(chunk)

    # 摘要信息
    summary = {
        "project_id": project_id,
        "project_name": project_name,
        "entity_count": len(entities),
        "relation_count": len(relations),
        "chunk_count": len(chunks),
        "entity_types": {},
        "relation_types": {}
    }
    for ent in entities:
        t = ent.get('type', 'UNKNOWN')
        summary['entity_types'][t] = summary['entity_types'].get(t, 0) + 1
    for rel in relations:
        r = rel.get('relation', 'UNKNOWN')
        summary['relation_types'][r] = summary['relation_types'].get(r, 0) + 1

    with open(out_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return out_dir


# =============================================================================
# 主流程
# =============================================================================

def process_project(data: dict, base_dir: Path) -> dict:
    """处理单个项目"""
    project_id = data['id']
    project_name = data['name']
    all_text = data['all']

    print(f"\n{'='*60}")
    print(f"处理项目: {project_name} ({project_id})")

    # 提取研究内容
    rc_text = find_research_content(all_text)
    if not rc_text:
        print("  [跳过] 未找到研究内容章节")
        return {"project_id": project_id, "status": "skipped", "reason": "no research content found"}

    # 分块
    chunks = chunk_research_content(rc_text)
    print(f"  研究内容长度: {len(rc_text)} 字, 切分为 {len(chunks)} 块")
    for i, chunk in enumerate(chunks):
        print(f"    块 {i+1}: {len(chunk)} 字 - {chunk[:60]}...")

    # 逐块抽取
    all_entities = []
    all_relations = []
    for i, chunk in enumerate(chunks):
        section_id = f"{project_id}/chunk_{i+1}"
        result = extract_chunk(chunk, section_id)
        if result:
            ents = result.get('entities', [])
            rels = result.get('relations', [])
            print(f"    块 {i+1}: 抽取到 {len(ents)} 实体, {len(rels)} 关系")
            all_entities.extend(ents)
            all_relations.extend(rels)
        else:
            print(f"    块 {i+1}: 抽取失败")

    print(f"\n  原始结果: {len(all_entities)} 实体, {len(all_relations)} 关系")

    # 去重
    deduped_entities = dedup_entities(all_entities)
    deduped_relations = dedup_relations(all_relations)
    print(f"  去重后: {len(deduped_entities)} 实体, {len(deduped_relations)} 关系")

    # Profiling
    low_level_kv, high_level_kv = build_profile(deduped_entities, deduped_relations)
    print(f"  索引: {len(low_level_kv)} Low-level Key, {len(high_level_kv)} High-level Key")

    # 保存
    out_dir = save_results(project_id, project_name,
                           deduped_entities, deduped_relations,
                           chunks, low_level_kv, high_level_kv, base_dir)
    print(f"  结果保存至: {out_dir}")

    return {
        "project_id": project_id,
        "project_name": project_name,
        "status": "success",
        "chunks": len(chunks),
        "entities_raw": len(all_entities),
        "entities_deduped": len(deduped_entities),
        "relations_raw": len(all_relations),
        "relations_deduped": len(deduped_relations),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LightRAG 知识图谱抽取")
    parser.add_argument("--input", default="assets/project_chunk_00",
                        help="输入文件路径")
    parser.add_argument("--limit", type=int, default=10,
                        help="处理前 N 条记录")
    parser.add_argument("--output", default="output/lightrag_extract",
                        help="输出目录")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[错误] 输入文件不存在: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取记录
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[警告] 第 {i+1} 行 JSON 解析失败: {e}")

    print(f"读取 {len(records)} 条记录，开始 LightRAG 抽取...")

    # 顺序处理
    results = []
    for data in records:
        result = process_project(data, output_dir)
        results.append(result)

    # 总摘要
    print(f"\n{'='*60}")
    print(f"处理完成! 共 {len(results)} 个项目")
    success = [r for r in results if r.get('status') == 'success']
    print(f"成功: {len(success)}, 跳过: {len(results) - len(success)}")
    total_entities = sum(r.get('entities_deduped', 0) for r in success)
    total_relations = sum(r.get('relations_deduped', 0) for r in success)
    print(f"总计: {total_entities} 实体, {total_relations} 关系")

    # 保存全局摘要
    with open(output_dir / "manifest.json", 'w', encoding='utf-8') as f:
        json.dump({
            "total_projects": len(results),
            "success": len(success),
            "total_entities": total_entities,
            "total_relations": total_relations,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n全局摘要: {output_dir / 'manifest.json'}")


if __name__ == '__main__':
    main()
