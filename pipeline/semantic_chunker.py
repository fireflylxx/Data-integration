"""
semantic_chunker.py — 语义边界滑窗分块

替代原机械 sliding_window_chunks()（每3000字硬切），核心改进：

  1. LLM 识别文档的 4 大模块边界（考核指标、研究内容、创新点、任务分解）
  2. 按自然语义边界切分（段落/小节标题），而非固定字符数
  3. 模块内超长时按段落边界滑窗，保持语义完整性

用法:
  from pipeline.semantic_chunker import SemanticChunker

  chunker = SemanticChunker(llm_func=my_llm_func)
  chunks = chunker.chunk(text)

  # 也可以跳过 LLM，只用规则 + 段落滑窗:
  chunks = chunker.chunk(text, use_llm=False)
"""

import re
import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class Module:
    """文档中的一个模块（考核指标/研究内容/创新点/任务分解）"""
    name: str          # 模块名称
    start_pos: int     # 起始字符位置（原文中的）
    end_pos: int       # 结束字符位置
    source: str = "llm"  # 识别方式: "llm" / "rule" / "full_text"


@dataclass
class Chunk:
    """一个语义完整的分片"""
    text: str          # 分片文本
    module_name: str   # 所属模块名（如 "考核指标"）
    start_pos: int     # 在原文中的起始位置
    end_pos: int       # 在原文中的结束位置
    chunk_index: int   # 模块内的分片序号（0-based）
    total_chunks: int = 1  # 模块总分片数


# =============================================================================
# 模块边界识别 — LLM 方案
# =============================================================================

MODULE_IDENTIFY_SYSTEM_PROMPT = """你是一个科研任务书结构化分析专家。

请分析以下国家重点研发计划任务书的正文，识别出 4 个关键模块在原文中的起止位置：

1. 考核指标（项目目标及考核指标、评测方式/方法）
2. 研究内容（项目研究内容、研究方法及技术路线）
3. 任务分解（项目任务分解、课题设置）
4. 创新点（主要创新点）

注意：
- 任务书正文从"项目名称"或"项目编号"行开始
- 模块之间按"一、" "二、" "三、" "四、" 等序号分隔
- 每个模块的终止位置在下一个大节标题之前

输出 JSON 格式（只返回 JSON，不要解释）：
{
  "modules": [
    {"name": "考核指标", "start_line": 10, "end_line": 245},
    {"name": "研究内容", "start_line": 246, "end_line": 400},
    {"name": "任务分解", "start_line": 401, "end_line": 520},
    {"name": "创新点", "start_line": 521, "end_line": 580}
  ]
}

- start_line / end_line 是 1-indexed 行号
- 找不到某个模块时，设 start_line 和 end_line 为 0
- 严格按行号标记，不添加原文不存在的内容
"""


def llm_identify_modules(text: str, llm_func: Callable) -> List[Module]:
    """
    用 LLM 识别文档中的 4 大模块边界。

    Args:
        text: 完整任务书文本
        llm_func: LLM 调用函数，签名 llm_func(system_prompt, user_prompt) -> str

    Returns:
        识别到的模块列表，按 start_pos 排序
    """
    if not text or len(text.strip()) < 200:
        return []

    # 截断 text 防止超 token（模块信息通常在文档前 2/3 部分）
    input_text = text[:15000] if len(text) > 15000 else text

    lines = input_text.split('\n')
    # 给 LLM 的行号文本（每行前加行号）
    numbered = '\n'.join(f"{i+1}:{line}" for i, line in enumerate(lines))

    user_prompt = f"任务书正文（行号已标注）：\n\n{numbered}"

    try:
        response = llm_func(MODULE_IDENTIFY_SYSTEM_PROMPT, user_prompt)
        result = _extract_json(response)
        if not result or 'modules' not in result:
            return []

        modules = []
        for m in result['modules']:
            name = m.get('name', '').strip()
            start = m.get('start_line', 0)
            end = m.get('end_line', 0)
            if name and start > 0 and end > 0 and end > start:
                # 行号转字符位置
                if start <= len(lines) and end <= len(lines):
                    start_pos = sum(len(l) + 1 for l in lines[:start - 1])
                    end_pos = sum(len(l) + 1 for l in lines[:end])
                    modules.append(Module(
                        name=name,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        source="llm"
                    ))

        modules.sort(key=lambda m: m.start_pos)
        return modules

    except Exception as e:
        return []


def _extract_json(text: str) -> Optional[dict]:
    """从 LLM 回复中健壮地提取 JSON。"""
    text = text.strip()
    # 移除 markdown 代码块
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 查找 {}
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


# =============================================================================
# 模块边界识别 — 规则方案（LLM 失败的兜底）
# =============================================================================

# 任务书标准序号章节模式（按顺序匹配，一一对应）
# 序号越大越靠后，内容越精确
SECTION_PATTERNS = [
    # (模式, 模块名) — 模式按特异性降序排列
    # 一级标题：一/二/三/四 + 关键词
    (r'一[、.\s]*项目目标及考核指标', '考核指标'),
    (r'一[、.\s]*项目合作目标及考核指标', '考核指标'),
    (r'一[、.\s]*项目目标、成果与考核指标', '考核指标'),
    (r'一[、.\s]*合作目标、成果与考核指标', '考核指标'),
    (r'一[、.\s]*成果、考核指标及评测', '考核指标'),
    (r'二[、.\s]*项目研究内容', '研究内容'),
    (r'二[、.\s]*研究方法及技术路线', '研究内容'),
    (r'三[、.\s]*项目任务', '任务分解'),
    (r'三[、.\s]*项目[\s\S]{0,20}课题[\s\S]{0,10}分解', '任务分解'),
    (r'四[、.\s]*主要创新点', '创新点'),
    # 兜底：无序号但含关键词（适配政府间项目模板）
    (r'项目目标及考核指标、评测方式/方法', '考核指标'),
    (r'项目合作目标及考核指标、评测方式/方法', '考核指标'),
    (r'项目目标、成果与考核指标表', '考核指标'),
    (r'合作目标、成果与考核指标表', '考核指标'),
    (r'成果、考核指标及评测方式/方法', '考核指标'),
    (r'项目研究内容、研究方法及技术路线', '研究内容'),
    (r'项目任务\s*\(\s*课题\s*\)\s*分解', '任务分解'),
    (r'主要创新点', '创新点'),
]

# 大节结束标记（下一大节的开头）
SECTION_END_MARKERS = [
    r'\n一、', r'\n二、', r'\n三、', r'\n四、', r'\n五、', r'\n六、',
    r'\n（一）', r'\n（二）', r'\n（三）',
    r'第\d+页/共\d+页',  # 页码标记
]


def rule_identify_modules(text: str) -> List[Module]:
    """
    基于规则识别 4 大模块边界。

    策略：
      1. 在正文中找一/二/三/四的序号章节标题
      2. 按序号顺序映射到模块名（一→考核指标、二→研究内容、三→任务分解、四→创新点）
      3. 每章从标题行到下一大节标题前
      4. 如果序号法失败，回退到关键词匹配
    """
    if not text:
        return []

    # 找 "====" 分割线后的正文起始
    body_start = text.find("====")
    if body_start >= 0:
        body = text[body_start + 4:]
    else:
        body = text

    modules = []

    # ── 策略 A：按序号顺序识别 ──
    # 找所有 "\nX、" 到 "\n" 或行末 的一级标题
    numbered_heads = list(re.finditer(
        r'\n([一二三四五六])[、\s]+\s*(.{2,60}?)(?:\n|$)',
        body
    ))

    # 序号到模块名的映射
    NUM_MAP = {'一': '考核指标', '二': '研究内容', '三': '任务分解',
               '四': '创新点', '五': None, '六': None}

    # 取前 4 个有效序号（一→二→三→四的顺序）
    found_sections = []  # [(start_pos, end_pos, module_name)]
    for m in numbered_heads:
        num = m.group(1)
        title = m.group(2).strip()
        module_name = NUM_MAP.get(num)
        if module_name is None:
            continue

        # 章节起始位置（\n 后面一个字符）
        sec_start = m.start() + 1

        # 如果有已识别的上一节，更新上一节的 end
        if found_sections:
            # 当前章节的开始就是上一节的结束
            found_sections[-1] = (found_sections[-1][0], sec_start, found_sections[-1][2])

        found_sections.append((sec_start, len(body), module_name))

    # 验证：检查是否覆盖了所有核心模块（至少要有考核指标+研究内容）
    detected_modules = set(m for _, _, m in found_sections)
    core_modules = {'考核指标', '研究内容'}
    has_core = core_modules.issubset(detected_modules)

    if found_sections and has_core:
        for sec_start, sec_end, module_name in found_sections:
            if sec_end - sec_start >= 50:
                modules.append(Module(
                    name=module_name,
                    start_pos=sec_start,
                    end_pos=sec_end,
                    source="rule_seq"
                ))

    # ── 策略 B：如果序号法没找到核心模块，回退到关键词匹配 ──
    if not modules:
        keyword_modules = []
        for marker_pat, module_name in SECTION_PATTERNS:
            m = re.search(marker_pat, body)
            if not m:
                continue
            sec_start = body.rfind('\n', 0, m.start()) + 1
            tail = body[sec_start + 1:]
            sec_end = len(body)
            for end_pat in SECTION_END_MARKERS:
                pos = re.search(end_pat, tail)
                if pos:
                    candidate = sec_start + 1 + pos.start()
                    if candidate < sec_end:
                        sec_end = candidate
            if sec_end - sec_start >= 50:
                keyword_modules.append((sec_start, sec_end, module_name))

        # 去重：按位置合并重叠项
        keyword_modules.sort(key=lambda x: x[0])
        merged = []
        for s, e, n in keyword_modules:
            if merged:
                prev_s, prev_e, prev_n = merged[-1]
                # 如果与上一项重叠（小于50字间隔视为重叠）
                if s < prev_e + 50 and n == prev_n:
                    merged[-1] = (prev_s, max(prev_e, e), n)
                    continue
                # 被上一项完全包含
                if s >= prev_s and e <= prev_e and n == prev_n:
                    continue
            merged.append((s, e, n))

        for s, e, n in merged:
            modules.append(Module(name=n, start_pos=s, end_pos=e, source="rule_kw"))

    # ── 策略 C：完全没有任何模块 ──
    if not modules:
        return [Module(name="全文", start_pos=0, end_pos=len(body), source="full_text")]

    # 补齐正文偏移量
    if body_start >= 0:
        for m in modules:
            m.start_pos += body_start + 4
            m.end_pos += body_start + 4

    return modules


# =============================================================================
# 语义边界滑窗 — 按段落/小节切分
# =============================================================================

def _find_paragraph_boundaries(text: str, min_size: int = 1500, max_size: int = 4000) -> List[int]:
    """
    在文本中找到自然的段落/小节边界位置。

    按优先级：
      1. 小节标题行（如 "1."、"（一）"、"步骤1" 等）
      2. 空行（\n\n）
      3. 行尾（\n）

    返回字符位置列表（包含 0 和 len(text)）
    """
    boundaries = {0, len(text)}

    # 优先级 1: 小节标题（"数字." 或 "（数字）" 开头的行）
    for m in re.finditer(r'\n\s*(?:\d+[.、．]|（\d+）|[一二三四五六七八九十]+[.、．])\s*\S', text):
        pos = m.start() + 1  # 跳过 \n
        if min_size <= pos <= len(text) - min_size // 2:
            boundaries.add(pos)

    # 优先级 2: 二级标题（"数字.数字" 或 "数字.数字.数字"）
    for m in re.finditer(r'\n\s*\d+\.\d+(?:\.\d+)?\s+', text):
        pos = m.start() + 1
        if min_size <= pos <= len(text) - min_size // 2:
            boundaries.add(pos)

    # 优先级 3: 空行
    for m in re.finditer(r'\n\s*\n', text):
        pos = m.start() + 1
        if min_size <= pos <= len(text) - min_size // 2:
            boundaries.add(pos)

    return sorted(boundaries)


def paragraph_sliding_window(text: str, max_chunk_size: int = 3000,
                             overlap: int = 200) -> List[Chunk]:
    """
    按段落/小节边界滑窗切分文本。

    与机械滑窗（每 3000 字硬切）的区别：
      - 找到最近的段落边界（\n\n 或小节标题行）作为切分点
      - 每片保持语义完整的段落
      - 重叠部分以段落为单位（而非字符数）

    Args:
        text: 输入文本
        max_chunk_size: 每片目标最大字符数（默认 3000）
        overlap: 重叠区域字符数（默认 200）

    Returns:
        分块列表
    """
    if not text:
        return []

    if len(text) <= max_chunk_size:
        return [Chunk(text=text, module_name="", start_pos=0,
                      end_pos=len(text), chunk_index=0)]

    boundaries = _find_paragraph_boundaries(text, min_size=max_chunk_size // 3)

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        # 目标结束位置
        target_end = start + max_chunk_size

        if target_end >= len(text):
            # 最后一片
            chunks.append(Chunk(
                text=text[start:],
                module_name="",
                start_pos=start,
                end_pos=len(text),
                chunk_index=chunk_idx
            ))
            break

        # 找到离 target_end 最近的段落边界
        best_end = target_end
        min_dist = float('inf')

        for b in boundaries:
            if b > start:  # 必须在起始位置之后
                dist = abs(b - target_end)
                # 优先取不超过 target_end 且最近的边界
                if b <= target_end and dist < min_dist:
                    min_dist = dist
                    best_end = b
                # 如果找不到不超过的，取最近的超过的边界
                elif b > target_end and best_end == target_end:
                    best_end = b

        # 如果边界离目标太远（超过 500 字），回退到段落换行
        if abs(best_end - target_end) > 500:
            # 找最近的 \n\n
            nl_pos = text.rfind('\n\n', start, target_end + 500)
            if nl_pos > start + max_chunk_size // 2:
                best_end = nl_pos

        # 确保不是原地踏步
        if best_end <= start:
            best_end = min(start + max_chunk_size, len(text))

        chunk_text = text[start:best_end]
        chunks.append(Chunk(
            text=chunk_text,
            module_name="",
            start_pos=start,
            end_pos=best_end,
            chunk_index=chunk_idx
        ))

        chunk_idx += 1

        # 重叠：从 best_end - overlap 开始找最近的段落边界
        overlap_start = best_end - overlap
        if overlap_start <= start:
            overlap_start = start + 1  # 至少前进 1 字，防止死循环

        next_start = overlap_start
        # 找到离 overlap_start 最近的段落边界（在 overlap_start 之后）
        for b in boundaries:
            if start < b < best_end:
                continue  # 跳过在当前片内的边界
            if b >= overlap_start and b < best_end + overlap:
                next_start = b
                break

        if next_start <= start:
            next_start = start + max(1, min(overlap, len(text) - start - 1))

        start = next_start

    return chunks


# =============================================================================
# 语义分块主流程
# =============================================================================

class SemanticChunker:
    """
    语义边界滑窗分块器。

    用法:
        chunker = SemanticChunker(llm_func=my_llm_func)
        chunks = chunker.chunk(raw_text, use_llm=True)

        # 每片自带 module_name，可用于后续选择对应 prompt
        for chunk in chunks:
            if chunk.module_name == "考核指标":
                # 用 KPI prompt
            elif chunk.module_name == "研究内容":
                # 用 Research prompt
    """

    def __init__(self, llm_func: Optional[Callable] = None,
                 max_chunk_size: int = 3000,
                 overlap: int = 200):
        """
        Args:
            llm_func: LLM 调用函数，用于模块边界识别。
                      签名: llm_func(system_prompt, user_prompt) -> str
                      为 None 时仅用规则方案
            max_chunk_size: 每片最大字符数
            overlap: 重叠区域字符数
        """
        self.llm_func = llm_func
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk(self, text: str, use_llm: bool = True) -> List[Chunk]:
        """
        主入口：语义分块。

        流程：
          1. 识别模块边界（LLM → 规则兜底）
          2. 对每个模块，模块内按段落边界滑窗
          3. 返回带 module_name 的分块列表

        Args:
            text: 完整任务书文本
            use_llm: 是否先用 LLM 识别模块边界

        Returns:
            语义分块列表
        """
        if not text or len(text.strip()) < 50:
            return []

        # Step 1: 识别模块边界
        modules: List[Module] = []
        if use_llm and self.llm_func:
            modules = llm_identify_modules(text, self.llm_func)

        if not modules:
            modules = rule_identify_modules(text)

        # Step 2: 按模块切分
        if not modules:
            # 彻底兜底：滑窗
            return self._chunk_with_module(text, "全文", 0, len(text))

        chunks = []
        for mod in modules:
            mod_text = text[mod.start_pos:mod.end_pos]
            mod_chunks = self._chunk_with_module(
                mod_text, mod.name, mod.start_pos, mod.end_pos
            )
            chunks.extend(mod_chunks)

        return chunks

    def _chunk_with_module(self, text: str, module_name: str,
                            global_start: int, global_end: int) -> List[Chunk]:
        """对单个模块的文本做段落边界滑窗。"""
        if len(text) <= self.max_chunk_size:
            return [Chunk(
                text=text,
                module_name=module_name,
                start_pos=global_start,
                end_pos=global_end,
                chunk_index=0
            )]

        para_chunks = paragraph_sliding_window(
            text, max_chunk_size=self.max_chunk_size, overlap=self.overlap
        )

        # 填充 module_name 和全局位置
        result = []
        for i, c in enumerate(para_chunks):
            result.append(Chunk(
                text=c.text,
                module_name=module_name,
                start_pos=global_start + c.start_pos,
                end_pos=global_start + c.end_pos,
                chunk_index=i,
                total_chunks=len(para_chunks),
            ))
        return result


# =============================================================================
# 便捷函数（兼容旧接口）
# =============================================================================

def sliding_window_chunks(text: str, window_size: int = 3000,
                           overlap: int = 800) -> list:
    """
    兼容旧接口的语义滑窗。

    与原 sliding_window_chunks 返回格式一致：
      [{'text': ..., 'start_pos': ..., 'end_pos': ...}]

    区别：按段落边界切分，而非固定字符数。
    """
    chunks = paragraph_sliding_window(text, max_chunk_size=window_size, overlap=overlap)
    return [
        {'text': c.text, 'start_pos': c.start_pos, 'end_pos': c.end_pos}
        for c in chunks
    ]


def identify_modules(text: str, llm_func: Optional[Callable] = None) -> List[Module]:
    """便捷函数：识别文档模块边界（LLM → 规则兜底）。"""
    modules = []
    if llm_func:
        modules = llm_identify_modules(text, llm_func)
    if not modules:
        modules = rule_identify_modules(text)
    return modules


# =============================================================================
# 演示 / 自测
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 测试用 LLM 函数（从环境变量或配置获取 API）
    def demo_llm(system: str, user: str) -> str:
        """仅用于演示的 dummy LLM——实际使用时会替换为真实 API 调用"""
        return '{"modules": []}'

    chunker = SemanticChunker(llm_func=demo_llm, max_chunk_size=3000, overlap=200)

    # 读入测试文件
    test_dir = Path("output/project_chunks_cleaned/project_chunk_00")
    test_files = sorted(test_dir.glob("*.txt"))[:3]

    for fp in test_files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        print(f"\n{'='*60}")
        print(f"文件: {fp.name} ({len(text)} 字)")

        # 先用规则识别模块
        modules = identify_modules(text)
        print(f"模块识别: {len(modules)} 个")
        for m in modules:
            mtext = text[m.start_pos:m.end_pos]
            print(f"  {m.name}: 位置 {m.start_pos}-{m.end_pos} ({len(mtext)} 字, 来源: {m.source})")

        # 语义分块
        chunks = chunker.chunk(text, use_llm=False)
        print(f"语义分块: {len(chunks)} 片")
        for c in chunks:
            print(f"  [{c.module_name}] 片{c.chunk_index+1}/{c.total_chunks}: "
                  f"位置 {c.start_pos}-{c.end_pos} ({len(c.text)} 字)")
