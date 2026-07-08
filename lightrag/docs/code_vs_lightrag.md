# 代码逻辑 vs LightRAG 逐模块对比

> 本文件以**代码逻辑粒度**梳理你的系统，并逐模块与 LightRAG 对比。
> 图中 ✅ = 你的系统有此功能，❌ = LightRAG 无此功能，⚠️ = 部分有但方式不同。

---

## 一、全流程概览（代码文件映射）

```mermaid
graph TB
    %% ===== 样式 =====
    classDef file fill:#1a1a2e,color:#fff,stroke:#e94560,stroke-width:2px
    classDef data fill:#16213e,color:#fff,stroke:#0f3460,stroke-dasharray:4
    classDef lightrag fill:#2a1a2e,color:#fff,stroke:#6a2d4a
    classDef compare fill:#1a2a1e,color:#fff,stroke:#2d6a4f

    %% ===== 你的系统 =====
    subgraph YOURS["你的系统 (kg_builder.py / kg_builder_v2.py)"]
        direction TB
        Y1["输入: project_cleaned/*.txt<br/>项目任务书"]:::data
        Y2["[parse_project_file]<br/>提取ID/名称/正文"]:::file
        Y3["[parse_kpi_section]<br/>正则匹配考核指标章节"]:::file
        Y4["[parse_topics]<br/>正则匹配 课题N: 行"]:::file
        Y5["[chunk_text]<br/>800字固定分块"]:::file
        Y6["[两轮抽取 v2]<br/>entity-only → relation-only<br/>带全局实体约束"]:::file
        Y7["[KPI驱动增强 v2]<br/>关键词密度→TopK→二次抽取"]:::file
        Y8["[跨块关系 v2]<br/>实体块索引→跨块对LLM推理"]:::file
        Y9["[normalize]<br/>类型映射+关系映射+过滤"]:::file
        Y10["[课题集成]<br/>create_topic_entities/relations"]:::file
        Y11["[构建KV索引]<br/>low_level + high_level"]:::file
        Y12["[build_vector_index]<br/>TFIDF+SVD→FAISS"]:::file
        Y13["输出: entities.json<br/>relations.json kpis.json<br/>faiss.index chunks/"]:::data

        Y1 --> Y2 --> Y3 --> Y5
        Y2 --> Y4 --> Y5
        Y5 --> Y6 --> Y7 --> Y8
        Y3 -.-> Y7
        Y8 --> Y9 --> Y10 --> Y11 --> Y12 --> Y13
    end

    %% ===== LightRAG =====
    subgraph LR["LightRAG (官方案例)"]
        direction TB
        L1["输入: 任意文档"]:::data
        L2["[insert]<br/>分块(chunk)"]:::file
        L3["[LLM抽取]<br/>实体+关系 同时抽取<br/>无类型约束"]:::file
        L4["[LLM Profiling]<br/>低维描述(entity→context)<br/>高维主题词(relation→theme)"]:::file
        L5["[去重]<br/>余弦相似度>0.9"]:::file
        L6["[构建双向量索引]<br/>实体向量索引(entity→emb)<br/>关系向量索引(theme→emb)"]:::file
        L7["[社区检测]<br/>Leiden聚类→社区摘要"]:::file
        L8["输出: 图结构+<br/>双向量索引+社区摘要"]:::data

        L1 --> L2 --> L3 --> L4 --> L5 --> L6 --> L7 --> L8
    end

    %% ===== 对比标注 =====
    CMP1["✅ 分块"]:::compare
    CMP2["⚠️ 抽取：你=两轮, LR=单轮<br/>你=有本体, LR=自由"]:::compare
    CMP3["⚠️ 你=KV+FAISS, LR=双向量<br/>你的FAISS索引chunk文本<br/>LR的向量索引实体+关系"]:::compare
    CMP4["❌ LR无KPI模块<br/>你的KPI解析+课题是独有"]:::compare
    CMP5["❌ LR无本体归一化<br/>你的normalize是独有"]:::compare
    CMP6["❌ LR无跨块关系<br/>你的P2是独有"]:::compare
```


---

## 二、Phase 1: KG 构建 — 逐函数对比

### 2.1 预处理模块

```mermaid
graph LR
    subgraph Y["你的系统"]
        Y1["parse_project_file()<br/>解析 txt 头信息"]
        Y2["parse_kpi_section()<br/>正则匹配『考核指标』章节<br/>→ kpis.json<br/>提取每条标号+描述"]
        Y3["parse_topics()<br/>正则匹配『课题N:』行<br/>→ topics.json<br/>中文数字→int"]
        Y4["match_kpis_to_topics()<br/>KPI标号前缀→课题ID<br/>→ topic_id 注入"]
        Y5["chunk_text()<br/>800字固定窗口<br/>行级合并<br/>不足200字丢弃"]
    end
    subgraph L["LightRAG"]
        L1["无元数据解析<br/>直接全文输入"]
        L2["无KPI模块"]
        L3["无课题模块"]
        L4["无课题-KPI匹配"]
        L5["chunk_text()<br/>段落级切分<br/>长段按句分割<br/>无固定窗口"]
    end
    Y1 --> Y5
    Y2 --> Y4
    Y3 --> Y4
    Y4 -.-> Y5
```

**关键代码差异**：

| 函数 | 你的代码 | LightRAG |
|------|---------|----------|
| `parse_project_file` | 提取 name/ID/projectNo 元数据 | 无对应，直接 insert 全文 |
| `parse_kpi_section` | 三段正则匹配考核指标章节 | ❌ 无 |
| `parse_topics` | `课题/任务 N：名称` 正则 | ❌ 无 |
| `_cn_to_int` | 中文数字→阿拉伯数字 | ❌ 无 |
| `match_kpis_to_topics` | ID点号前缀→课题分组 | ❌ 无 |
| `chunk_text` | 800字窗口，行级融合 | 段落级，句级分割长段 |

### 2.2 抽取模块

```mermaid
graph TB
    subgraph Y["你的系统 - 抽取"]
        Y1["chunks[ ]"]
        
        %% v1 模式（单轮）
        Y_V1["_process_chunk() ← v1兼容<br/>SYSTEM_PROMPT + FEW_SHOT<br/>一次调用出实体+关系<br/>temperature=0.3"]
        
        %% v2 模式（两轮）
        subgraph V2["v2 两轮模式 (默认)"]
            Y2_1["Round 1: _process_chunk_entities()<br/>ENTITY_ONLY_PROMPT<br/>只抽实体<br/>temperature=0.1<br/>→ 全量唯一实体集 S_e"]
            Y2_2["Round 2: _process_chunk_relations()<br/>RELATION_ONLY_PROMPT<br/>S_e 作为已知实体约束<br/>每块带全局实体列表<br/>temperature=0.2<br/>→ 关系集 R"]
        end
        Y_V2_KPI["P0: _kpi_boost_extraction()<br/>KPI_BOOST_SYSTEM_PROMPT<br/>关键词密度分→TopK块<br/>二次抽取KPI相关实体关系<br/>→ 增强集 E+"]
        Y_V2_CR["P2: _discover_cross_chunk_relations()<br/>CROSS_CHUNK_RELATION_PROMPT<br/>实体块索引→跨块对→LLM推理<br/>→ 跨块关系集 C+"]

        Y1 -->|v1模式| Y_V1
        Y1 -->|v2模式| Y2_1 --> Y2_2
        Y2_2 -.-> Y_V2_KPI
        Y2_2 -.-> Y_V2_CR
    end

    subgraph L["LightRAG - 抽取"]
        L1["chunks[ ]"]
        L2["extract_chunk()<br/>SYSTEM_PROMPT_KG + FEW_SHOT<br/>一次调用出实体+关系<br/>temperature=0.3<br/>→ 无类型约束<br/>→ 任何文本都是实体"]
        L1 --> L2
    end
```

**Prompt 对比**：

```
你的 ENTITY_ONLY_PROMPT：
  "你是一个科研实体抽取专家...只输出实体，不抽取关系...
   实体类型：OBJECT/METHOD/PARAMETER/ACTIVITY/EQUIPMENT..."
  → 有类型枚举，有约束，有方向

你的 RELATION_ONLY_PROMPT：
  "给定已知实体列表：
    光纤陀螺仪 (OBJECT)
    零偏稳定性 (PARAMETER)
    转台 (EQUIPMENT)
    ...
   请从文本中找出关系，head/tail 必须来自此列表"
  → 有约束，有过滤，防幻觉

LightRAG 的 SYSTEM_PROMPT_KG：
  "实体类型不由预定义列表限制，请根据文本内容动态判断最合适的类型"
  → 无约束，完全自由
```

### 2.3 后处理模块

```mermaid
graph LR
    subgraph Y["你的系统 - 后处理"]
        Y1["normalize()"]
        Y1a["normalize_type()<br/>ENTITY_TYPE_MAP<br/>COMPONENT→OBJECT<br/>METRIC→PARAMETER<br/>NON_ENTITY_TYPES→过滤"]
        Y1b["normalize_relation()<br/>RELATION_MAP<br/>『采用』→ VIA<br/>『测试』→ VERIFIES<br/>『研制』→ EXECUTES<br/>无法映射→保留原值"]
        Y1c["(head,relation,tail)三元组去重<br/>精确匹配"]
        Y1d["实体类型冲突: OBJECT优先"]
        
        Y2["create_topic_entities()<br/>课题→OBJECT实体<br/>命名: 『课题{id}:{name}』"]
        Y3["create_topic_relations()<br/>PARAMETER→课题 BELONGS_TO<br/>基于KPI描述中的参数名匹配"]
    end
    subgraph L["LightRAG - 后处理"]
        L1["dedup_entities()<br/>编辑距离<3模糊去重<br/>包含关系去重<br/>无类型过滤"]
        L2["dedup_relations()<br/>(head,tail)对去重<br/>不保留relation类型"]
        L3["build_profile()<br/>low_level_kv: entity→context<br/>high_level_kv: 关键词组合→desc<br/>关键词从head+tail+context提取<br/>无LLM参与"]
    end
```

**对比**：

| 步骤 | 你的系统 | LightRAG |
|------|---------|----------|
| 类型归一化 | 9种+过滤，精确 | 无，直接保留 |
| 关系归一化 | 6种谓词 + 规则映射 | 无，自由保留 |
| 去重策略 | 精确三元组去重 | 模糊编辑距离 (损失精度) |
| 课题集成 | 独有 | ❌ 无 |
| BELONGS_TO 关系 | 独有 | ❌ 无 |
| 过滤非技术实体 | `NON_ENTITY_TYPES` | ❌ 无 |

### 2.4 索引构建

```mermaid
graph TB
    subgraph Y["你的系统 - 索引"]
        Y1["low_level_kv.json<br/>实体名 → 关系context拼接<br/>『零偏稳定性』→ 『采用转台进行标定测试...』"]
        Y2["high_level_kv.json<br/>三元组 → context<br/>『光纤陀螺仪 VIA 转台测试』→context"]
        Y3["faiss.index (chunk级)<br/>TFIDFEmbedding:<br/>TfidfVectorizer(ngram 2-4)<br/>+ TruncatedSVD(384d)<br/>→ 查询chunk文本用"]
        Y4["课题层级 (KB)<br/>topics.json + topic_id<br/>KPI参数→课题的BELONGS_TO"]
    end
    subgraph L["LightRAG - 索引"]
        L1["low_level_kv<br/>实体名 → LLM描述+context<br/>LLM参与生成描述文本"]
        L2["high_level_kv<br/>LLM生成的高维主题词 → context<br/>『惯性测量·标定方法·陀螺精度』→ desc"]
        L3["实体向量索引 (entity→emb)<br/>用于local检索<br/>精确匹配实体"]
        L4["关系向量索引 (theme→emb)<br/>用于global检索<br/>主题匹配"]
    end
```

**本质区别**：

```
你的 KV 索引构建：
  low_level_kv[entity] = "；".join(rel_context for rel in relations if entity in rel)
  → 纯拼接，无 LLM 参与，无主题提取

LightRAG 的 KV 索引构建：
  low_level_kv[entity] = LLM(entity + relation_contexts → 生成描述)
  high_level_kv[theme] = LLM(head + relation + tail + context → 提取高维主题词)
  → LLM 参与理解语义，提取抽象主题词
```

---

## 三、Phase 2: 检索 + 推理 — 逐模块对比

```mermaid
graph TB
    %% ===== 你的系统 Phase 2 =====
    subgraph Y2["你的系统 - 数据集推理 (kpi_planner.py + hybrid_retriever.py)"]
        direction TB
        Y2_IN["KPI文本: 『零偏稳定性≤0.01°/h』"]:::data

        %% 混合检索
        Y2_R1["[hybrid_retriever.kg_path_query]<br/>① find_entity('零偏稳定性', type=PARAMETER)<br/>② traverse_2hop('零偏稳定性')<br/>   零偏稳定性 ←VERIFIES— 转台测试 ←VIA— 光纤陀螺仪<br/>③ 按路径完整度打分<br/>   hops=2 → 0.6, hops=1 → 0.3, 参数匹配 → +0.3"]:::file
        Y2_R2["[hybrid_retriever.vector_chunk_query]<br/>① KPI全文 → FAISS搜索<br/>② top-5 相关chunk文本<br/>③ 返回原文片段"]:::file
        Y2_R3["[merge_context]<br/>KG路径 + 向量chunk → 融合文本"]:::file

        %% KPI 结构解析
        Y2_P["[kpi_planner.parse_kpi_structured]<br/>LLM提取:<br/>object='光纤陀螺仪'<br/>parameter='零偏稳定性'<br/>target_value='≤0.01°/h'<br/>verb='研制'<br/>activity_type='测试'"]:::file

        %% 规划→命名→验证
        Y2_PLAN["[plan_dataset]<br/>LLM: 结构化KPI+上下文→方案<br/>输出: [<br/>  {object, parameter, method,<br/>   data_type, data_suffix, confidence}<br/>]"]:::file
        Y2_NAME["[name_dataset]<br/>LLM: 方案→中文名称<br/>『高精度光纤陀螺仪零偏稳定性<br/>标定测试数据集』"]:::file
        Y2_VAL["[validate_name]<br/>① 名称→LLM反推KPI<br/>② SequenceMatcher 比对原KPI<br/>③ score≥0.35? → 通过<br/>④ 失败→重试(≤2次)"]:::file
        Y2_OUT["数据集名称.csv"]:::data

        Y2_IN --> Y2_R1
        Y2_IN --> Y2_R2
        Y2_R1 --> Y2_R3
        Y2_R2 --> Y2_R3
        Y2_IN --> Y2_P
        Y2_R3 --> Y2_PLAN
        Y2_P --> Y2_PLAN
        Y2_PLAN --> Y2_NAME --> Y2_VAL
        Y2_VAL -->|失败≤2次| Y2_PLAN
        Y2_VAL -->|通过| Y2_OUT
    end

    %% ===== LightRAG Phase 2 =====
    subgraph L2["LightRAG - 检索 + 生成 (query 接口)"]
        direction TB
        L2_IN["用户query: 『零偏稳定性<br/>对光纤陀螺仪性能的影响？』"]:::data

        %% query 处理
        L2_Q["[query() 入口]<br/>① 判断模式: local/global/hybrid<br/>② LLM提取关键词<br/>  低维: ['零偏稳定性','光纤陀螺仪','性能']<br/>  高维: ['陀螺精度','惯性导航','误差分析']"]:::file

        %% Local 路径
        L2_LOCAL["local模式:<br/>① 低维关键词→实体向量索引<br/>② 匹配实体+1-hop邻居<br/>③ 按节点度排序<br/>④ 展开邻居关系<br/>⑤ 获取chunk文本"]:::file

        %% Global 路径
        L2_GLOBAL["global模式:<br/>① 高维关键词→关系向量索引<br/>② 匹配关系+头尾实体<br/>③ 按边权重+节点度排序<br/>④ 关联社区摘要<br/>⑤ 获取chunk文本"]:::file

        %% 合并
        L2_MERGE["[context 融合]<br/>结构化拼接成 3 段:<br/>1. entities (name,type,desc,rank)<br/>2. relationships (src,tgt,desc,weight)<br/>3. text chunks (原文片段)"]:::file
        L2_OUT["LLM → 自然语言回答"]:::data

        L2_IN --> L2_Q
        L2_Q -->|低维| L2_LOCAL
        L2_Q -->|高维| L2_GLOBAL
        L2_LOCAL --> L2_MERGE
        L2_GLOBAL --> L2_MERGE
        L2_MERGE --> L2_OUT
    end
```

### 检索机制逐行对比（代码逻辑级）

| 步骤 | 你的系统 | LightRAG |
|------|---------|----------|
| **查询输入** | `KPI描述文本`（结构化指标） | `任意用户query`（自然语言） |
| **关键词提取** | 无（直接用 KPI 文本） | LLM 拆解为低维具体词 + 高维抽象词 |
| **实体匹配** | `find_entity(name, type)` KG 精确查找 | embedding 向量相似度匹配 |
| **图遍历** | `traverse_2hop()` 沿 VIA→VERIFIES 路径 | 1-hop 邻居展开（不限谓词类型） |
| **排序依据** | 路径完整度（hops 数） | 节点度 / 边权重 |
| **文本检索** | FAISS 向量搜 chunk | 从 KV 索引取 entity/relation 的 context |
| **上下文注入** | 融合文本 → LLM 规划 | 结构化表格（entity+relation+chunk）→ LLM 回答 |
| **验证/约束** | 回译验证（名称→反推 KPI） | 无 |

---

## 四、Phase 3: 评估模块

```mermaid
graph TB
    subgraph Y3["你的系统 - 3D 评估 (evaluator_3d_refined.py)"]
        Y3_IN["数据集名称 + KPI + 证据"]:::data
        
        Y3_D1["维度1: 语义忠实度 (50%)<br/>score_semantic_fidelity()<br/>LLM结构化评分:<br/>- object_score (0/0.5/1)<br/>- parameter_score (0/0.5/1)<br/>- method_score (0/0.5/1)<br/>→ 加权平均"]:::file
        
        Y3_D2["维度2: 可追溯性 (35%)<br/>compute_traceability()<br/>① extract_name_elements()<br/>   规则优先→LLM fallback<br/>   对象/参数/方法 要素提取<br/>② each element → chunk 检索<br/>③ 对抗性检查(常识排除)<br/>④ → traceability_score"]:::file
        
        Y3_D3["维度3: 规范性 (15%)<br/>compute_regularity()<br/>7条规则:<br/>以『数据集』/『数据』结尾<br/>不包含标点符号<br/>长度范围15-35字<br/>..."]:::file
        
        Y3_DIAG["diagnose()<br/>评分模式→诊断标签:<br/>优秀 / 格式待改进 / 检索召回不足<br/>推理错误 / 双重失败 / 对象误判 / 一般"]:::file
        
        Y3_FB["error_kb.json<br/>FP/FN → 知识库修正<br/>→ 反馈到抽取/检索"]:::file

        Y3_IN --> Y3_D1 --> Y3_DIAG
        Y3_IN --> Y3_D2 --> Y3_DIAG
        Y3_IN --> Y3_D3 --> Y3_DIAG
        Y3_DIAG --> Y3_FB
    end

    subgraph L3["LightRAG - 评估"]
        L3_IN["生成回答 + Ground Truth"]:::data
        L3_M["人工评估 4 维度:<br/>- Comprehensiveness(完整性)<br/>- Diversity(多样性)<br/>- Empowerment(赋能性)<br/>- Overall(整体质量)<br/>每个 1-5 分"]:::file
        L3_A["自动指标:<br/>- Hit Rate<br/>- MRR (Mean Reciprocal Rank)<br/>- 消融实验<br/>- 效率对比(时间/成本)"]:::file
    end
```

### 评估机制对比

| 维度 | 你的系统 | LightRAG |
|------|---------|----------|
| 评估方式 | **全自动**（3D 评分公式 + 诊断） | **人工打分**（1-5 量表） |
| 评估粒度 | 每个数据集名称逐条评估 | 整体回答质量评估 |
| 维度数 | 3 维度 × 子项 | 4 维度 |
| 客观性 | 高（规则+LLM双重+对抗检查） | 低（依赖人工主观判断） |
| 反馈闭环 | error_kb.json → FP/FN 修正 | 无（一次性评估） |
| 可复现性 | 高（固定公式） | 低（人工评分波动） |

---

## 五、数据流对比（文件级）

```
你的系统文件结构                          LightRAG 文件结构
─────────────────                       ──────────────────
output/kg_ontology/{pid}/                working_dir/{doc_id}/
├── entities.json                        ├── kv_store_ll.json        # low_level
├── relations.json                       ├── kv_store_lm.json        # 文档全文
├── kpis.json          ← LightRAG 无      ├── kv_store_full_docs.json
├── topics.json        ← LightRAG 无      ├── graph_chunk_entity_relation.graphml  # 图结构
├── low_level_kv.json  ← 类似              ├── text_chunks.json
├── high_level_kv.json ← 类似              ├── community_report.json  # LightRAG 独有
├── chunks/                               ├── ...
├── summary.json                          ├── ...
├── faiss.index        → LightRAG 无      ├── ..._vector_index.pkl   # 双向量索引
                                          ├── ..._full_docs.vec
```

---

## 六、核心差异总结（一句话对应）

```
 你的系统                              LightRAG
 ──────────                           ──────────
 本体约束(9类+6谓词)                   自由类型
 两轮抽取(实体→关系)                   单轮同时
 KPI解析+课题匹配                      无
 600字固定分块                         段落自适应
 单向量索引(TFIDF on chunks)           双向量索引(entity+relation)
 KG路径遍历(2-hop确定性)                双级语义匹配(向量局部+全局)
 回译验证(防幻觉)                       无
 3D自动评估                            人工打分
 输出:结构化数据集名称                    输出:自然语言回答
 图=推理引擎                           图=检索索引
```

---

## 七、核心代码文件映射表

| 你的文件 | 功能 | LightRAG 对应 |
|---------|------|--------------|
| `pipeline/kg_builder.py` | KG 构建（v1 单轮） | `lightrag.py` 的 `insert()` |
| `pipeline/kg_builder_v2.py` | KG 构建（v2 多轮增强） | 无对应（更强） |
| `pipeline/hybrid_retriever.py` | KG路径+FAISS 混合检索 | `retriever.py` 的 `query()` |
| `pipeline/kg_vector_store.py` | TFIDF+SVD 向量索引 | `base.py` 向量存储基类 |
| `pipeline/kpi_planner.py` | KPI→数据集名称生成 | 无对应 |
| `pipeline/evaluator_3d_refined.py` | 3D 自动评估 | 无对应（你独有） |
| `pipeline/submission_ontology.py` | 本体定义 | 无对应 |
| `pipeline/run_kg_batch.py` | 批处理+断点续跑 | 无对应 |
| `scripts/complex/lightrag_extract.py` | LightRAG 风格的抽取实现 | 参考实现 |

| LightRAG 文件 | 功能 | 你的对应 |
|--------------|------|---------|
| `lightrag.py` | 主入口, insert/query | kg_builder.py + kpi_planner.py |
| `retriever.py` | 双级检索实现 | hybrid_retriever.py（但逻辑不同） |
| `kg_extractor.py` | 实体关系抽取 | _process_chunk 系列 |
| `kg_query.py` | 查询分解+路径选择 | 无（你无 query 分解） |
| `kg_storage.py` | 图存储后端 | entities.json + relations.json |
| `base.py` | 向量索引基类 | kg_vector_store.py |
| `community_report.py` | 社区检测+摘要 | 无对应 |
