"""
Full Pipeline — KG驱动数据集汇交完整流程
=========================================
架构（图9 KPI-Structured Graph RAG）：

  项目任务书
    │
    ├──→ 考核指标 → KPI解析(LLM) → 结构化KPI
    │                                    │
    │    ┌──研究内容──→ Chunks ──→ FAISS索引    │
    │    │                │                    │
    │    └──→ LLM抽取 ──→ KG (实体+关系)       │
    │                       │                  │
    └────── Hybrid Retriever ←┘                  │
               │                                │
               ▼                                │
        规划模块(LLM) → 命名(LLM) → 回译验证(LLM)
               │
               ▼
          CSV导出 + 3D评测

用法:
    python pipeline/full_pipeline.py --project 1646707075016945664
    python pipeline/full_pipeline.py --all
    python pipeline/full_pipeline.py --skip-kg
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from difflib import SequenceMatcher

# 确保项目根目录在路径中
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 项目模块
from pipeline.kg_vector_store import (
    TFIDFEmbedding, build_vector_index, load_vector_index,
    vector_search, save_chunks, load_chunks
)
from pipeline.hybrid_retriever import HybridRetriever
from pipeline.kpi_planner import (
    llm_chat, parse_kpi_structured, plan_dataset, name_dataset,
    validate_name, kpi_to_dataset
)

# =============================================================================
# 路径配置
# =============================================================================
KG_OUTPUT_DIR = BASE_DIR / "output" / "lightrag_extract_cleaned"
PROJECT_DIR = BASE_DIR / "output" / "project_cleaned"
PIPELINE_OUTPUT = BASE_DIR / "output" / "full_pipeline_output"


# =============================================================================
# Phase 0: 项目解析 — 分离考核指标 + 研究内容
# =============================================================================

def parse_project_sections(text: str) -> dict:
    """将项目文本分离为考核指标和研究内容

    Returns:
        {"kpi_section": str, "research_section": str, "task_sections": [...]}
    """
    result = {"kpi_section": "", "research_section": "", "task_sections": []}

    # 找考核指标章节
    kpi_markers = [
        r'考核指标[：:\s,、]',
        r'项目合作目标及考核指标',
        r'合作目标、成果与考核指标表',
    ]
    for marker in kpi_markers:
        m = re.search(marker, text)
        if m:
            start = m.start()
            # 往后找直到下一个大段或结束
            end = len(text)
            next_markers = [r'项目目标、', r'主要研究内容', r'其他目标与考核指标']
            for nm in next_markers:
                nm_pos = text.find(nm, start + len(marker))
                if nm_pos > start:
                    # 不要切太短
                    if nm_pos - start > 100:
                        end = nm_pos
                        break
            result["kpi_section"] = text[start:end].strip()
            break

    # 找研究内容章节
    research_markers = [
        r'主要研究内容[：:\s]',
        r'项目的主要研究内容',
    ]
    for marker in research_markers:
        m = re.search(marker, text)
        if m:
            start = m.start()
            # 往后找直到下一个大段
            end = len(text)
            next_markers = ['合作方的主要研究内容', '逐项分段说明', '主要参考文献']
            for nm in next_markers:
                nm_pos = text.find(nm, start + len(marker))
                if nm_pos > start:
                    end = nm_pos
                    break
            result["research_section"] = text[start:end].strip()
            break

    return result


def extract_kpi_list(text: str, full_text: str = "") -> List[str]:
    """从考核指标章节提取逐条KPI — 多策略增强版

    策略：
      1. 预处理移除模板说明区块（填写说明/备注等）
      2. 从全文扫描（4）考核指标及评测手段/方法 节（课题级KPI）
      3. 从表格格式提取（制表符+编号+：）
      4. 从编号行提取（改进正则）
      5. 后过滤移除管理性/行政性内容

    Args:
        text: 考核指标章节文本（可含模板说明）
        full_text: 项目全文（用于查找各课题下的考核指标节）

    Returns:
        [kpi_description, ...]
    """

    # ═══════════════════════════════════════════════════
    # 辅助判断函数
    # ═══════════════════════════════════════════════════

    def _is_template_instruction(s: str) -> bool:
        """是否为表单填写说明行（X."..." 或含填写说明关键词）"""
        # 1."项目目标"...  2."对应的课题（任务）"... 等编号+引号模式
        # 兼容 ASCII引号 ""、中文引号 “”、直角引号「」
        quote_chars = '""""""\u201c\u201d\u300c\u300d'
        if re.match(
            rf'^\d+\s*[.、]?\s*[{quote_chars}][^"」\u201c\u201d\u300c\u300d]{{2,}}[{quote_chars}]',
            s,
        ):
            return True
        tmpl_kw = [
            '应从以下方面明确描述', '指标指相应成果的',
            '中期指标各专项', '考核方式方法应提出',
            '科技报告类型包括', '公开类别及时限',
            '对应的课题（任务）指', '对应的课题(任务)指',
            '指标值/状态', '立项时已有指标',
        ]
        for kw in tmpl_kw:
            if kw in s:
                return True
        return False

    def _is_form_header(s: str) -> bool:
        """是否为表单章节标题/表结构文本（非KPI）"""
        stripped = s.strip()
        # 参加单位任务分工
        if re.match(r'^[（(]?\d*[）)]?\s*参加单位任务分工', stripped):
            return True
        # XXXX表（短表名）
        if re.search(r'[表图][：:\s]*$', stripped) and len(stripped) < 30:
            return True
        form_headers = [
            '项目目标、成果与考核指标表', '参加单位分工与预期成果表',
            '项目目标（限', '成果类型', '对应的课题',
            '考核方式（方法）', '考核方式(方法)',
            '指标名称', '立项时已有指标', '中期指标值', '完成时指标',
            '评价手段', '课题\d+参加单位分工',
        ]
        for h in form_headers:
            if re.search(h, stripped):
                return True
        return False

    def _has_tech_content(s: str) -> bool:
        """判断文本是否包含技术性内容"""
        if len(s) < 8:
            return False
        # 含数值技术参数: ≥, ≤, %, 数字+单位
        if re.search(r'[≥≤<>]\s*\d+', s):
            return True
        if re.search(r'\d+\s*[%°℃KkMmgWwVhHzFf]', s):
            return True
        if re.search(r'[≥≤%]', s):
            return True
        # 含技术活动动词
        tech_verbs = ['研制', '开发', '构建', '实现', '突破', '建立',
                       '形成', '制备', '设计', '研发', '搭建', '创建',
                       '研建', '制造', '开发出',
                       '制备出', '研制出', '突破面积']
        for v in tech_verbs:
            if v in s:
                return True
        # 含技术名词组合（≥2个且长度≥12，短技术名称依然有效）
        tech_nouns = ['系统', '装置', '设备', '平台', '算法', '模型',
                       '技术', '工艺', '器件', '材料', '软件', '装备',
                       '方法', '机器人', '电路', '芯片', '器件',
                       '原型', '模块', '组件', '电池', '薄膜',
                       '传感器', '探测器', '光源', '光纤',
                       '处理器', '指令集', '编译器', '编程',
                       '模拟器', '量子']
        noun_count = sum(1 for t in tech_nouns if t in s)
        if noun_count >= 2 and len(s) > 12:
            return True
        if noun_count >= 1 and len(s) >= 20:
            return True
        # 含论文/专利/软著等产出物指标
        if re.search(r'(论文|专利|软件著作权)\s*\d+', s):
            return True

    def _is_admin_only(s: str) -> bool:
        """是否为纯管理性内容（仅有人员培养/分工/知识产权数量/报告提交）且无技术内容"""
        # 纯管理：只有人员培养，没有技术参数
        if re.search(r'培养(?:研究|博|硕)(?:生|人员)?\s*\d+', s):
            # 有具体技术参数的不算纯管理
            if not re.search(r'[≥≤%°℃]|研制|开发|构建|实现|突破|设计|制备', s):
                return True
        # 只有发表论文/申请专利数量（无具体技术内容）
        if re.search(r'(?:发表|申请).*(?:论文|专利|著作权)\s*\d+', s):
            if len(s) < 30 and not re.search(r'[≥≤%°℃]|研制|开发|系统|装置|技术|工艺|算法|模型', s):
                return True
        # 纯交付物提交（报告/总结/验收），无技术参数
        if re.match(r'提交.*(?:年度技术进展|中期检查|最终验收|专题科技|总结)报告', s):
            return True
        # 单行仅含管理性指标（如 "发表论文 5 篇" 作为单一行）
        if re.match(r'^(?:发表|申请|提交|培养|出版)\s', s) and len(s) < 25:
            if not re.search(r'[≥≤%°℃]|研制|系统|装置|设备|平台', s):
                return True
        return False

    def _is_project_header(s: str) -> bool:
        """检测是否为项目信息头（非KPI内容）"""
        # 含单位信息
        if '推荐单位' in s or '专业机构' in s or '项目牵头承担单位' in s:
            return True
        # 含密级/经费信息
        if '密级' in s or '经费预算' in s or '项目周期节点' in s:
            return True
        # 含长时间跨度的项目执行信息
        if re.search(r'执行期限.*\d{4}\s*年\s*\d{2}\s*月\s*至', s):
            return True
        return False

    def _is_research_content(s: str) -> bool:
        """检测文本是否为研究描述（非KPI），避免研究内容被错误提取为考核指标"""
        stripped = s.strip()
        # 研究目标/研究内容/科学问题等章节标题
        research_headers = [
            '研究目标', '主要研究内容', '拟解决的重大科学问题',
            '关键技术问题', '技术路线', '研究内容', '研究基础',
            '研究背景及目标', '预期成果和效益', '创新点',
            '具体包含以下部分',
        ]
        for h in research_headers:
            if stripped.startswith(h):
                return True
            # "1 研究目标" 但排在前面的数字
            if re.search(r'研究目标[：:]', stripped[:20]):
                return True
        # 以"研究"开头且不含具体数值参数（如 ≥ ≤ %）的描述
        if re.match(r'研究\S{1,10}[。，]', stripped) and not re.search(r'[≥≤%]', stripped):
            return True
        # "探究/探索/揭示/阐明/分析/讨论XX机制/规律/原理/方法"
        if re.search(r'(?:探究|探索|揭示|阐明)\s*(?:其|了)?', stripped[:30]):
            if re.search(r'(?:机制|规律|原理|方法|模式|机理|途径)', stripped[:60]):
                return True

        # 课题标题（非KPI）
        if re.match(r'课题\s*\d+\s*[：:]', stripped):
            return True
        if re.match(r'任务[（(]\s*课题\s*[）)]\s*\d+', stripped):
            return True

        # "既研究/即研究/既揭示" — 研究内容说明句式
        if re.search(r'[既即](?:研究|揭示|阐明|探索)', stripped[:40]):
            return True

        # "针对XX问题" — 研究问题描述
        if re.match(r'针对', stripped) and any(kw in stripped for kw in
            ['问题', '需求', '挑战', '情况', '目标', '现状']):
            return True

        # "围绕XX开展/进行" — 研究内容句式
        if re.match(r'围绕', stripped) and any(kw in stripped for kw in
            ['研究', '开展', '进行', '探索', '构建']):
            return True

        # "具体包含以下部分" — 研究内容引导
        if '具体包含以下部分' in stripped:
            return True

        # "主要研究以下" — 研究内容引导
        if '主要研究以下' in stripped:
            return True

        # 纯方法描述行（仅含实验/测试方法，无KPI指标值）
        methods_only = ['采用', '使用', '利用', '通过']
        if re.match(r'(?:采用|使用|利用|通过)\s', stripped):
            # 如果有具体指标值(≥≤%)，可能是KPI
            if not re.search(r'[≥≤%]', stripped):
                return True

        return False

    # ═══════════════════════════════════════════════════
    # Step 1: 预处理 — 移除模板说明区块
    # ═══════════════════════════════════════════════════

    def _remove_template_blocks(txt: str) -> str:
        lines = txt.split('\n')
        out = []
        in_block = False
        for line in lines:
            s = line.strip()
            if _is_template_instruction(s):
                in_block = True
                continue
            if in_block:
                # 遇到空行或非编号行，退出模板块
                if not s or not re.match(r'^\d+\s*[.、]', s):
                    in_block = False
                    # 当前行也检查是否为模板行
                    if _is_template_instruction(s):
                        in_block = True
                        continue
            if not in_block:
                out.append(line)
        return '\n'.join(out)

    cleaned = _remove_template_blocks(text)

    # ═══════════════════════════════════════════════════
    # Step 2: 多策略提取
    # ═══════════════════════════════════════════════════

    kpis = []
    seen = set()

    def _add(s: str):
        s = s.strip()
        if not s or len(s) < 8:
            return
        if s in seen:
            return
        if _is_form_header(s):
            return
        seen.add(s)
        kpis.append(s)

    full = full_text if full_text else text

    # ── Strategy A: 从全文中找各课题下的考核指标节 ──
    # Pattern A1: （4）考核指标及评测手段/方法  (project 05)
    # Pattern A2: X.Y 考核指标及评测手段/方法  (project 02, e.g. 1.4)
    # Pattern A3: 考核指标及评测手段/方法：无编号 (project 04)
    for pat in [
        r'[（(]\s*4\s*[）)]\s*考核指标.{0,12}(?:评测手段|评测方法|方法)',
        r'(?<!\d)(\d+\.\d+)\s*考核指标.{0,12}(?:评测手段|评测方法|方法)',
        r'^考核指标及评测手段/方法[：:\s]',
    ]:
        for m in re.finditer(pat, full, re.MULTILINE):
            block_end = len(full)
            after = full[m.end():]
            # 找节结束：下一个课题标题或 (5)/(X)参加单位任务分工
            next_sec = re.search(
                r'(?:'
                r'[（(]\s*\d\s*[）)]\s*(?:参加单位任务分工|拟解决的重大科学问题|研究目标|主要研究内容)'
                r'|'
                r'\d+\.\d+\s*(?:参加单位任务分工|拟解决的重大科学问题|研究目标|主要研究内容)'
                r'|'
                r'(?:二、项目研究内容)'
                r')',
                after,
            )
            if next_sec:
                block_end = m.end() + next_sec.start()
            block = full[m.start():block_end]

            # 子策略A1: 编号条目 （1） （2） （3）
            for item in re.finditer(
                r'^[（(]\s*(\d+)\s*[）)]\s*(.+)', block, re.MULTILINE
            ):
                _add(item.group(2).strip())

            # 子策略A2: 编号条目 1） 2） 3）
            for item in re.finditer(r'^(\d+\s*[）)])\s*(.+)', block, re.MULTILINE):
                _add(item.group(2).strip())

            # 子策略A4: 考核指标后跟的""条目列表（课题5表格式）
            for kpi_line in re.finditer(
                r'^考核指标[：:]\s*(.+)', block, re.MULTILINE
            ):
                content = kpi_line.group(1).strip()
                # 跳过子条目汇总，只提取单个项目
                if re.match(r'\s*[（(]\s*\d+\s*[）)]', content):
                    continue
                if len(content) >= 5:
                    # 若内容是多个""条目拼接，拆开保存
                    if '' in content:
                        for item in re.split(r'[]', content):
                            item = item.strip()
                            if len(item) >= 8:
                                _add(item)
                    else:
                        _add(content)

    # ── Strategy B: 表格KPI  X：技术描述 （来自考核指标节） ──
    # 匹配行首为 数字+：+较长技术描述 的模式（仅在kpi_section内）
    for m in re.finditer(r'^\s*(\d+)\s*[：:]\s*(.+)', cleaned, re.MULTILINE):
        content = m.group(2).strip()
        if len(content) >= 15 and not _is_template_instruction(content):
            # 跳过 "考核指标：（1）..." 这种块级包裹行（子条目已被 Strategy A 提取）
            if re.search(r'考核指标[：:]\s*[（(]', content):
                continue
            _add(content)

    # ── Strategy C: 编号行 + 续行累积（处理OCR碎片，如项目03） ──
    # 注意：仅在考核指标节内提取，研究内容节中的编号行被跳过
    c_lines = cleaned.split('\n')
    current_kpi = ""

    # 研究类章节标题 — 遇到这些时结束当前KPI且不开始新KPI
    # 同时也是判断"是否在研究内容区"的依据
    research_section_headers = [
        '研究目标', '主要研究内容', '拟解决的重大科学问题',
        '关键技术问题', '技术路线', '研究内容', '研究基础',
        '研究背景', '预期成果', '创新点',
    ]
    section_header_pat = re.compile(
        r'(?:研究目标|主要研究内容|拟解决的重大科学问题|'
        r'考核指标及评测|评测手段/方法|参加单位任务分工|'
        r'课题\s*[参共]|关键技术问题|技术路线|研究内容|'
        r'研究基础|研究背景|预期成果)[：:\s]'
    )

    # 当前是否在研究内容区（研发目标/研究内容内，非考核指标节）
    in_research_zone = False
    # 最近遇到的章节标签（用于决定是否开始新KPI）
    last_section_label = ""

    for line in c_lines:
        line = line.strip()
        if not line:
            if current_kpi:
                _add(current_kpi)
                current_kpi = ""
            continue

        # 检测研究/考核章节标签切换
        line_upper = line[:40]  # 只看开头判断节

        # 进入考核指标节 → 退出研究区
        if re.search(r'考核指标[：:\s]', line_upper):
            in_research_zone = False
            last_section_label = "kpi"

        # 进入研究类节 → 进入研究区
        is_research_header = False
        for h in research_section_headers:
            if line_upper.startswith(h) or re.search(rf'^\d+\s*{h}', line_upper):
                in_research_zone = True
                last_section_label = h
                is_research_header = True
                break
        # "课题 X：" 也进入研究区
        if re.match(r'课题\s*\d+\s*[：:]', line):
            in_research_zone = True
            last_section_label = "课题"
            is_research_header = True

        if is_research_header:
            if current_kpi:
                _add(current_kpi)
                current_kpi = ""
            continue

        # 章节标题匹配（无编号行版本）→ 也可能标记研究区
        if section_header_pat.search(line):
            if current_kpi:
                _add(current_kpi)
                current_kpi = ""
            # 如果匹配到的是研究类标题，标记研究区
            for h in research_section_headers:
                if h in line[:30]:
                    in_research_zone = True
                    break
            continue

        # 匹配编号行: X.X、 X.X. X.X：  X. X： X、等
        m = re.match(
            r'^(\d+(?:[\.\d]*))\s*[、.、：:]\s*(.+)',
            line,
        )
        if m:
            # 结束当前KPI
            if current_kpi:
                _add(current_kpi)
            content = m.group(2).strip()

            # 跳过考核指标节标题行（如 "1.4.1 考核指标："）
            if re.match(r'^考核指标[：:\s]', content):
                current_kpi = ""
                continue

            # 在研究区内 → 不开始新KPI
            if in_research_zone:
                current_kpi = ""
                continue

            # 检查内容是否为研究描述（是则跳过）
            if _is_research_content(content):
                current_kpi = ""
                continue

            # 检查内容是否为研究描述（是则跳过）
            if _is_research_content(content):
                current_kpi = ""
                continue

            current_kpi = content
            continue

        # 行首为研究类动词且不在考核区 → 跳过
        if not current_kpi and not in_research_zone:
            if re.match(r'^(研制|开发|实现|构建|研究|探索|分析|揭示|阐明)\s', line):
                if not re.search(r'[≥≤%]', line):
                    continue

        # 续行：当前KPI累积（处理OCR跨行碎片）
        if current_kpi:
            current_kpi += line
            continue

    if current_kpi:
        _add(current_kpi)

    # ── Strategy D: 考核指标： 直接行（仅在不是子条目汇总时） ──
    for m in re.finditer(r'^考核指标[：:]\s*(.+)', cleaned, re.MULTILINE):
        content = m.group(1).strip()
        if len(content) >= 8:
            # 跳过包含子条目的考核指标行（如 "考核指标：（1）在柔性衬底上..."）
            if re.match(r'\s*[（(]\d+[）)]', content):
                continue
            # 拆分""条目
            if '' in content:
                for item in re.split(r'[]', content):
                    item = item.strip()
                    if len(item) >= 8:
                        _add(item)
            else:
                _add(content)

    # ═══════════════════════════════════════════════════
    # Step 3: 后处理过滤
    # ═══════════════════════════════════════════════════

    # 去重（保留排序）
    seen_kpis = set()
    deduped = []
    for kpi in kpis:
        # 用前80字符去重
        key = kpi[:80]
        if key in seen_kpis:
            continue
        seen_kpis.add(key)
        deduped.append(kpi)
    kpis = deduped

    # 技术内容 + 研究描述 + 项目头 过滤
    filtered = []
    for kpi in kpis:
        if len(kpi) < 10:
            continue
        if not _has_tech_content(kpi):
            continue
        if _is_admin_only(kpi):
            continue
        if _is_research_content(kpi):
            continue
        if _is_project_header(kpi):
            continue
        filtered.append(kpi)

    # 回退保护：如过滤过严（全没了），返回原始列表
    if not filtered:
        return kpis

    return filtered


# =============================================================================
# Phase 1-2: 加载/构建 KG + 向量索引
# =============================================================================

def ensure_kg_index(project_id: str, force_rebuild: bool = False) -> Optional[dict]:
    """加载 KG 数据，如果向量索引不存在则构建

    Returns:
        {
            "entities": [...],
            "relations": [...],
            "kpis": [...],
            "chunks": [...],
            "faiss_index": faiss.Index,
            "embedder": TFIDFEmbedding,
        }
        or None if KG not found
    """
    kg_dir = KG_OUTPUT_DIR / project_id
    if not kg_dir.exists() or not (kg_dir / "entities.json").exists():
        print(f"  [跳过] KG不存在: {project_id}")
        return None

    # 加载实体和关系
    entities = json.load(open(kg_dir / "entities.json", 'r', encoding='utf-8'))
    relations = json.load(open(kg_dir / "relations.json", 'r', encoding='utf-8'))
    print(f"  KG: {len(entities)}实体, {len(relations)}关系")

    # 加载 kpis.json（从 KG 构建阶段产出的结构化 KPI）
    kpis = []
    kpis_path = kg_dir / "kpis.json"
    if kpis_path.exists():
        kpis = json.load(open(kpis_path, 'r', encoding='utf-8'))
        print(f"  KPI: {len(kpis)} 条")

    # 加载/构建 chunks
    chunks_dir = kg_dir / "chunks"
    if chunks_dir.exists():
        chunks = []
        for fp in sorted(chunks_dir.glob("*.txt")):
            chunks.append(fp.read_text(encoding="utf-8"))
        print(f"  文本块: {len(chunks)} 块")
    else:
        chunks = []

    # 加载/构建 FAISS 索引
    faiss_path = kg_dir / "faiss.index"
    embedder = None
    faiss_index = None

    if faiss_path.exists() and not force_rebuild:
        try:
            embedder = TFIDFEmbedding()
            # 需要先 fit 才能 encode
            if chunks:
                embedder.fit(chunks)
            faiss_index = load_vector_index(faiss_path)
            print(f"  向量索引已加载")
        except Exception as ex:
            print(f"  向量索引加载失败: {ex}，将重建")
            faiss_index = None

    if faiss_index is None and chunks:
        print(f"  构建向量索引...")
        embedder = TFIDFEmbedding(corpus=chunks)
        faiss_index = build_vector_index(chunks, faiss_path, embedder)
        print(f"  向量索引已保存")

    return {
        "entities": entities,
        "relations": relations,
        "kpis": kpis,
        "chunks": chunks,
        "faiss_index": faiss_index,
        "embedder": embedder,
    }


# =============================================================================
# Phase 3: 逐KPI处理
# =============================================================================

def process_kpis(project_id: str, kg_data: dict) -> List[dict]:
    """逐KPI处理：混合检索 → 规划 → 命名 → 验证

    从 kg_data["kpis"] 读取结构化 KPI 列表（由 KG 构建阶段产出），
    避免重复解析项目书全文。

    Args:
        project_id: 项目ID
        kg_data: KG + 向量索引数据（含 kpis, entities, relations 等）

    Returns:
        [dataset_result, ...]
    """
    # 从 KG 产物读取 KPI description 列表
    raw_kpis = kg_data.get("kpis", [])
    kpi_texts = [kpi["description"] for kpi in raw_kpis if kpi.get("description")]
    if not kpi_texts:
        print(f"  [警告] KG中未找到KPI (kpis.json为空)")
        return []

    print(f"  考核指标: {len(kpi_texts)} 条 (来自KG)")
    for i, kt in enumerate(kpi_texts):
        print(f"    KPI {i+1}: {kt[:80]}...")

    # 构建混合检索器
    retriever = HybridRetriever(
        entities=kg_data["entities"],
        relations=kg_data["relations"],
        faiss_index=kg_data.get("faiss_index"),
        embedder=kg_data.get("embedder"),
        chunks=kg_data.get("chunks", []),
    )

    # 过滤非技术性 KPI
    admin_keywords = ["合作目标", "考核指标", "中期指标", "考核方式", "评测方式",
                      "填写说明", "应从以下", "指相应成果",
                      "数量指标", "技术指标", "质量指标", "应用指标"]
    technical_kpis = []
    for kt in kpi_texts:
        if any(kw in kt for kw in admin_keywords):
            continue
        if len(kt) < 10:
            continue
        technical_kpis.append(kt)

    skipped = len(kpi_texts) - len(technical_kpis)
    if skipped > 0:
        print(f"  (跳过 {skipped} 条非技术性KPI)")

    # 逐KPI处理
    results = []
    for i, kpi_text in enumerate(technical_kpis):
        print(f"\n  [{i+1}/{len(technical_kpis)}] 处理KPI: {kpi_text[:60]}...")

        # Step 1: 混合检索（先检索，为规划提供上下文）
        print(f"    [混合检索] KPI={kpi_text[:40]}...")
        retrieval = retriever.retrieve(
            kpi_full_text=kpi_text,
            top_kg=10,
            top_chunks=5,
        )
        print(f"      KG路径: {len(retrieval['kg_paths'])}条, Chunks: {len(retrieval['chunks'])}条")

        # Step 2-5: KPI解析 → 规划 → 命名 → 验证（kpi_to_dataset 内部完成）
        dataset_result = kpi_to_dataset(
            kpi_description=kpi_text,
            retriever_context=retrieval["merged_context"],
            max_retries=2,
        )

        # 附加检索信息
        dataset_result["evidence_paths"] = [p["path"] for p in retrieval["kg_paths"][:3]]
        dataset_result["evidence_type"] = infer_evidence_type(retrieval)
        dataset_result["project_id"] = project_id

        results.append(dataset_result)
        time.sleep(0.5)  # API 节流

    return results


def infer_evidence_type(retrieval: dict) -> str:
    """推断证据类型"""
    kg_paths = retrieval.get("kg_paths", [])
    if not kg_paths:
        return "NO_KG_EVIDENCE"
    max_hops = max(p.get("hops", 0) for p in kg_paths)
    if max_hops >= 2:
        return "KG_FULL_PATH"
    elif max_hops >= 1:
        return "KG_HALF_PATH"
    else:
        return "KG_MATCH"


# =============================================================================
# Phase 4: 导出
# =============================================================================

def export_results(results: List[dict], project_id: str, output_dir: Path) -> Path:
    """导出 CSV + JSON"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / f"{project_id}.csv"
    import csv
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["项目ID", "KPI原文", "数据集名称", "证据类型",
                         "置信度", "验证状态", "验证分数", "证据路径"])
        for r in results:
            writer.writerow([
                project_id,
                r.get("kpi", "")[:100],
                r.get("dataset_name", ""),
                r.get("evidence_type", ""),
                r.get("plan", {}).get("confidence", ""),
                "通过" if r.get("validated") else "未通过",
                r.get("validation_score", 0),
                json.dumps(r.get("evidence_paths", []), ensure_ascii=False),
            ])

    print(f"  CSV导出: {csv_path} ({len(results)} 条)")

    # JSON 详情
    detail_path = output_dir / f"{project_id}_detail.json"
    json.dump({
        "project_id": project_id,
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "results": results,
    }, open(detail_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"  详情导出: {detail_path}")

    return csv_path


def print_evaluation_summary(results: List[dict], project_id: str):
    """打印 3D 评测摘要"""
    print(f"\n  {'='*40}")
    print(f"  3D 评测摘要 — {project_id}")
    print(f"  {'='*40}")

    total = len(results)
    validated = sum(1 for r in results if r.get("validated"))
    avg_score = sum(r.get("validation_score", 0) for r in results) / max(total, 1)

    conf_counts = {}
    ev_counts = {}
    for r in results:
        conf = r.get("plan", {}).get("confidence", "UNKNOWN")
        conf_counts[conf] = conf_counts.get(conf, 0) + 1
        ev = r.get("evidence_type", "UNKNOWN")
        ev_counts[ev] = ev_counts.get(ev, 0) + 1

    print(f"  数据集: {total} (验证通过: {validated}, 平均分: {avg_score:.2f})")
    print(f"  置信度: {dict(sorted(conf_counts.items(), key=lambda x:-x[1]))}")
    print(f"  证据类型: {dict(sorted(ev_counts.items(), key=lambda x:-x[1]))}")


# =============================================================================
# 主流程
# =============================================================================

def run_pipeline(project_ids: Optional[List[str]] = None,
                 skip_kg: bool = False,
                 force_rebuild: bool = False):
    """运行完整 pipeline"""
    print("=" * 60)
    print("  数据汇交 Full Pipeline — KPI-Structured Graph RAG")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 收集项目文件
    if project_ids:
        project_files = []
        for fp in sorted(PROJECT_DIR.glob("*.txt")):
            content = fp.read_text(encoding="utf-8")
            m = re.search(r'项目ID:\s*(\d+)', content)
            if m and m.group(1) in project_ids:
                project_files.append((m.group(1), fp))
    else:
        project_files = []
        for fp in sorted(PROJECT_DIR.glob("*.txt")):
            content = fp.read_text(encoding="utf-8")
            m = re.search(r'项目ID:\s*(\d+)', content)
            if m:
                project_files.append((m.group(1), fp))

    if not project_files:
        print("[错误] 未找到项目文件")
        return

    print(f"\n项目数: {len(project_files)}")
    PIPELINE_OUTPUT.mkdir(parents=True, exist_ok=True)
    all_summaries = []

    for pid, fp in project_files:
        text = fp.read_text(encoding="utf-8")
        name_match = re.search(r'项目名称:\s*(.+)', text)
        pname = name_match.group(1).strip() if name_match else fp.stem

        print(f"\n{'='*50}")
        print(f"项目: {pname}")
        print(f"  ID: {pid}")

        # Phase 1+2: KG + 向量索引
        kg_data = ensure_kg_index(pid, force_rebuild=force_rebuild)
        if kg_data is None:
            print("  [跳过] 请先运行KG抽取")
            continue

        # Phase 3: 逐KPI处理
        print("  Phase 3: KPI处理...")
        results = process_kpis(pid, kg_data)

        if not results:
            print("  [跳过] 无结果")
            continue

        # Phase 4: 导出
        print("  Phase 4: 导出...")
        csv_path = export_results(results, pid, PIPELINE_OUTPUT)
        print_evaluation_summary(results, pid)

        all_summaries.append({
            "project_id": pid,
            "project_name": pname,
            "dataset_count": len(results),
            "validated_count": sum(1 for r in results if r.get("validated")),
            "csv": str(csv_path),
        })

    # 全局摘要
    print(f"\n{'='*60}")
    print("Full Pipeline 完成!")
    for s in all_summaries:
        print(f"  {s['project_name']}: {s['dataset_count']}数据集 "
              f"(验证通过: {s['validated_count']}) [{s['csv']}]")

    manifest = PIPELINE_OUTPUT / "manifest.json"
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "total": len(all_summaries),
        "summaries": all_summaries,
    }, open(manifest, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"摘要: {manifest}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="数据汇交 Full Pipeline")
    parser.add_argument("--project", nargs="+", default=None, help="指定项目ID")
    parser.add_argument("--all", action="store_true", help="处理所有项目")
    parser.add_argument("--skip-kg", action="store_true", help="跳过KG检查")
    parser.add_argument("--force-rebuild", action="store_true", help="强制重建向量索引")
    args = parser.parse_args()

    if args.all:
        run_pipeline(force_rebuild=args.force_rebuild)
    else:
        run_pipeline(project_ids=args.project, skip_kg=args.skip_kg,
                     force_rebuild=args.force_rebuild)
