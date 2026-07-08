"""
国家重点研发计划 数据集汇交本体 (Submission Ontology)
=====================================================

本模块定义了从"项目任务书"到"数据汇交清单"的核心数据结构，
包括 JSON Schema、JSON-LD 上下文、以及 KG→Dataset 的映射规则。

目录：
  1. JSON Schema 定义（核心类）
  2. JSON-LD 上下文（语义化关联）
  3. 数据库 DDL（SQLite 存储）
  4. KG→Dataset 映射规则
  5. 使用示例
"""

# =============================================================================
# 1. JSON Schema 定义
# =============================================================================

PROJECT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Project",
    "description": "国家重点研发计划项目",
    "type": "object",
    "required": ["id", "name"],
    "properties": {
        "id": {"type": "string", "pattern": "^20[0-9]{2}Y[0-9A-Z]{8,12}$",
                "description": "项目编号，如 2018YFE0193900"},
        "name": {"type": "string", "description": "项目全称"},
        "sub_projects": {
            "type": "array",
            "items": {"$ref": "#/definitions/SubProject"}
        }
    },
    "definitions": {
        "SubProject": {
            "type": "object",
            "required": ["id", "name"],
            "properties": {
                "id": {"type": "string", "description": "课题编号"},
                "name": {"type": "string", "description": "课题名称"},
                "order": {"type": "integer", "description": "课题序号"},
                "kpis": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/KPI"}
                }
            }
        },
        "KPI": {
            "type": "object",
            "required": ["id", "description"],
            "properties": {
                "id": {"type": "string", "description": "指标编号，如 2.1, 3.2.1"},
                "description": {"type": "string",
                                "description": "指标原文，如'零偏稳定性≤0.01°/h'"},
                "type": {
                    "type": "string",
                    "enum": ["TECHNICAL", "OUTPUT", "DEMONSTRATION", "OTHER"],
                    "description": "指标类型：技术/成果/示范/其他"
                },
                "target_value": {"type": "string",
                                 "description": "目标值，从描述中抽取如'≤0.01°/h'"},
                "entities": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/KPIEntity"},
                    "description": "从KPI中解析出的研究实体（对象+参数+方法）"
                }
            }
        },
        "KPIEntity": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string",
                         "enum": ["OBJECT", "PARAMETER", "METHOD", "ACTIVITY"]},
                "role": {"type": "string",
                         "description": "在KPI中的角色：subject/object/verb/measure"}
            }
        },
        "Entity": {
            "type": "object",
            "required": ["name", "type"],
            "properties": {
                "name": {"type": "string", "description": "实体名称"},
                "type": {
                    "type": "string",
                    "enum": [
                        "OBJECT",          # 研究对象（光纤陀螺仪）
                        "METHOD",          # 实验/测试方法（转台测试）
                        "PARAMETER",       # 性能参数（零偏稳定性）
                        "ACTIVITY",        # 验证活动（标定）
                        "EQUIPMENT",       # 设备
                        "MATERIAL",        # 材料
                        "SOFTWARE",        # 软件
                        "SYSTEM",          # 系统/平台
                        "MODEL",           # 模型/算法
                    ],
                    "description": "实体语义类型（4类核心 + 5类辅助）"
                },
                "aliases": {
                    "type": "array", "items": {"type": "string"},
                    "description": "别名列表"
                },
                "context": {"type": "string", "description": "上下文中定义"},
            }
        },
        "Relation": {
            "type": "object",
            "required": ["head", "relation", "tail"],
            "properties": {
                "head": {"type": "string", "description": "头实体名称"},
                "relation": {
                    "type": "string",
                    "enum": [
                        "VIA",             # [研究对象]—通过→[实验方法]
                        "VERIFIES",        # [实验方法]—验证→[性能参数]
                        "EXECUTES",        # [研究对象]—执行→[验证活动]
                        "PRODUCES",        # [研究对象]—产出→[数据集]
                        "BELONGS_TO",      # [数据集]—属于→[课题]
                        "MAPS_TO",         # [数据集]—对应→[考核指标]
                    ]
                },
                "tail": {"type": "string", "description": "尾实体名称"},
                "context": {"type": "string", "description": "原文证据"}
            }
        },
        "Dataset": {
            "type": "object",
            "required": ["id", "name_cn", "belongs_to", "maps_to"],
            "properties": {
                "id": {
                    "type": "string",
                    "pattern": "^20[0-9]{2}[A-Z0-9]+-\\d{3}$",
                    "description": "数据集编号：项目编号-序号"
                },
                "name_cn": {
                    "type": "string",
                    "description": "数据集中文名称，以'数据'或'数据集'结尾",
                    "examples": [
                        "高精度光纤陀螺仪零偏稳定性标定测试数据集",
                        "典型污染场地水土样品测试数据"
                    ]
                },
                "name_en": {"type": "string", "description": "英文名称（可选）"},
                "data_type": {
                    "type": "string",
                    "enum": [
                        "TEST_DATA",         # 测试数据
                        "MONITORING_DATA",   # 监测数据
                        "SIMULATION_DATA",   # 仿真数据
                        "MODEL_DATA",        # 模型数据
                        "ALGORITHM_DATA",    # 算法数据
                        "SAMPLE_DATA",       # 样本数据
                        "IMAGE_DATA",        # 图像数据
                        "PROCESS_DATA",      # 工艺数据
                        "SURVEY_DATA",       # 勘测数据
                        "OTHER",             # 其他
                    ],
                    "description": "数据类型"
                },
                "format": {
                    "type": "string",
                    "description": "数据格式，如 CSV, JSON, NetCDF, HDF5"
                },
                "volume": {"type": "string", "description": "数据量级"},
                "keywords": {
                    "type": "array", "items": {"type": "string"},
                    "description": "关键词"
                },
                "description": {"type": "string", "description": "数据集描述"},
                "belongs_to": {
                    "type": "string",
                    "description": "所属课题编号"
                },
                "maps_to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "对应的考核指标编号列表"
                },
                "evidence": {
                    "type": "object",
                    "properties": {
                        "source_text": {"type": "string", "description": "原文证据段落"},
                        "kg_path": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "KG验证路径，如 ['光纤陀螺仪', 'VIA', '转台测试', 'VERIFIES', '零偏稳定性']"
                        },
                        "page": {"type": "string", "description": "任务书页码"}
                    },
                    "description": "证据链（用于汇交审计）"
                }
            }
        }
    }
}

# =============================================================================
# 2. JSON-LD 上下文（语义网关联）
# =============================================================================

JSONLD_CONTEXT = {
    "@context": {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "onto": "https://data-science-center.cn/ontology/nrpd#",

        # 类
        "Project": "onto:Project",
        "SubProject": "onto:SubProject",
        "KPI": "onto:KPI",
        "Entity": "onto:ResearchEntity",
        "Relation": "onto:ResearchRelation",
        "Dataset": "onto:Dataset",

        # 属性
        "id": {"@id": "onto:id", "@type": "xsd:string"},
        "name_cn": {"@id": "onto:nameCN", "@type": "xsd:string"},
        "name_en": {"@id": "onto:nameEN", "@type": "xsd:string"},
        "description": {"@id": "rdfs:comment", "@type": "xsd:string"},

        # 关系谓词
        "via": {"@id": "onto:VIA"},
        "verifies": {"@id": "onto:VERIFIES"},
        "executes": {"@id": "onto:EXECUTES"},
        "produces": {"@id": "onto:PRODUCES"},
        "belongs_to": {"@id": "onto:BELONGS_TO"},
        "maps_to": {"@id": "onto:MAPS_TO"},

        # 实体类型
        "OBJECT": "onto:ResearchObject",
        "METHOD": "onto:TestMethod",
        "PARAMETER": "onto:PerformanceParameter",
        "ACTIVITY": "onto:VerificationActivity",

        # 数据类型
        "TEST_DATA": "onto:TestData",
        "MONITORING_DATA": "onto:MonitoringData",
        "SIMULATION_DATA": "onto:SimulationData",
    }
}

# =============================================================================
# 3. 数据库 DDL（SQLite）
# =============================================================================

SQLITE_DDL = """
-- 国家重点研发计划 数据集汇交本体存储
-- 适用场景：离线评测 / 实验管理 / 结果回溯

CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,           -- 项目编号
    name        TEXT NOT NULL,              -- 项目名称
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sub_projects (
    id          TEXT PRIMARY KEY,           -- 课题编号
    project_id  TEXT NOT NULL REFERENCES projects(id),
    name        TEXT NOT NULL,              -- 课题名称
    seq_order   INTEGER                    -- 课题序号
);

CREATE TABLE IF NOT EXISTS kpis (
    id          TEXT NOT NULL,              -- 指标编号
    project_id  TEXT NOT NULL REFERENCES projects(id),
    sub_project_id TEXT REFERENCES sub_projects(id),
    description TEXT NOT NULL,              -- 指标原文
    kpi_type    TEXT CHECK(kpi_type IN ('TECHNICAL','OUTPUT','DEMONSTRATION','OTHER')),
    PRIMARY KEY (project_id, id)
);

CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT NOT NULL REFERENCES projects(id),
    name        TEXT NOT NULL,              -- 实体名称
    entity_type TEXT NOT NULL CHECK(entity_type IN (
        'OBJECT','METHOD','PARAMETER','ACTIVITY',
        'EQUIPMENT','MATERIAL','SOFTWARE','SYSTEM','MODEL'
    )),
    context     TEXT,                       -- 上下文中定义
    UNIQUE(project_id, name)
);

CREATE TABLE IF NOT EXISTS relations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT NOT NULL REFERENCES projects(id),
    head_id     INTEGER NOT NULL REFERENCES entities(id),
    relation    TEXT NOT NULL CHECK(relation IN (
        'VIA','VERIFIES','EXECUTES','PRODUCES','BELONGS_TO','MAPS_TO'
    )),
    tail_id     INTEGER NOT NULL REFERENCES entities(id),
    context     TEXT                        -- 原文证据
);

CREATE TABLE IF NOT EXISTS datasets (         -- ← 汇交核心产出
    id          TEXT PRIMARY KEY,              -- 数据集编号
    project_id  TEXT NOT NULL REFERENCES projects(id),
    name_cn     TEXT NOT NULL,                 -- 中文名称
    name_en     TEXT,
    data_type   TEXT CHECK(data_type IN (
        'TEST_DATA','MONITORING_DATA','SIMULATION_DATA',
        'MODEL_DATA','ALGORITHM_DATA','SAMPLE_DATA',
        'IMAGE_DATA','PROCESS_DATA','SURVEY_DATA','OTHER'
    )),
    format      TEXT,
    volume      TEXT,
    description TEXT,
    belongs_to  TEXT REFERENCES sub_projects(id),
    source_pipeline TEXT,                     -- 生成管道版本
    f1_score    REAL,                         -- 与GT的匹配得分
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 多对多：数据集 ↔ 考核指标
CREATE TABLE IF NOT EXISTS dataset_kpi_map (
    dataset_id  TEXT NOT NULL REFERENCES datasets(id),
    kpi_id      TEXT NOT NULL,
    project_id  TEXT NOT NULL,
    PRIMARY KEY (dataset_id, kpi_id, project_id),
    FOREIGN KEY (project_id, kpi_id) REFERENCES kpis(project_id, id)
);

-- 向量存储元数据（记录哪些实体/文档有 embedding）
CREATE TABLE IF NOT EXISTS vector_index (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id   INTEGER REFERENCES entities(id),
    chunk_id    TEXT,                       -- 文本块ID
    model_name  TEXT,                       -- embedding模型名
    dimension   INTEGER,                    -- 向量维度
    index_path  TEXT,                       -- FAISS索引文件路径
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_kpis_project ON kpis(project_id);
CREATE INDEX idx_entities_project ON entities(project_id);
CREATE INDEX idx_datasets_project ON datasets(project_id);
CREATE INDEX idx_relations_head ON relations(head_id);
CREATE INDEX idx_relations_tail ON relations(tail_id);
"""

# =============================================================================
# 4. KG → Dataset 映射规则
# =============================================================================

"""
将已有的 Knowledge Graph (entities.json + relations.json) 映射到 Dataset 对象的规则。

现有KG的关系谓词（VIA/VERIFIES/EXECUTES）与 Dataset 本体的映射：

  研究对象 —VIA→ 实验方法 —VERIFIES→ 性能参数
    │                                          │
    └────────── 路径推理 → 生成 Dataset ────────┘

路径推理规则：

  Rule 1: OBJECT → VIA → METHOD
    → 推断：存在测试方法
    → 产出："{OBJECT}{METHOD}测试数据[集]"

  Rule 2: METHOD → VERIFIES → PARAMETER
    → 推断：方法验证了参数
    → 产出："{METHOD}{PARAMETER}验证数据[集]"

  Rule 3: OBJECT → VIA → METHOD → VERIFIES → PARAMETER
    → 推断：完整验证链路
    → 产出："{OBJECT}{PARAMETER}{METHOD}测试数据集"

  Rule 4: KPI中的实体 → KG匹配
    → 推断：KPI描述的成果对应可验证的数据
    → 产出："{KPI对象}{KPI参数}测试数据集"


Dataset 命名公式：

  template = f"{prefix}{core}{suffix}"

  prefix (可选):
    - 应用场景: "基于X的", "面向X的"

  core (必选):
    - 模式1: "{研究对象}{参数}"
    - 模式2: "{研究对象}{方法}"
    - 模式3: "{}{参数}{方法}"

  suffix (必选):
    - 测试数据[集]
    - 监测数据[集]
    - 仿真数据[集]
    - 模型数据[集]
    - 标定数据[集]
    - 识别数据[集]
"""

# =============================================================================
# 5. Python 使用示例
# =============================================================================

USAGE_EXAMPLE = """
# 从KG生成数据集的伪代码流程：

def kpi_to_datasets(project_id: str, entities: list, relations: list, kpis: list) -> list:
    datasets = []

    for kpi in kpis:
        # Step 1: 从KPI描述中抽取实体角色
        kpi_entities = parse_kpi_entities(kpi["description"])

        # Step 2: 在KG中匹配实体
        matched = match_in_kg(kpi_entities, entities, relations)

        if matched["has_full_path"]:
            # 有完整验证路径 → 生成数据集
            ds = {
                "id": f"{project_id}-{next_seq():03d}",
                "name_cn": assemble_dataset_name(matched),
                "data_type": infer_data_type(matched),
                "belongs_to": kpi["sub_project_id"],
                "maps_to": [kpi["id"]],
                "evidence": {
                    "kg_path": matched["path"],
                    "source_text": matched["context"]
                }
            }
            datasets.append(ds)
        elif matched["has_partial_path"]:
            # 部分路径 → 降低置信度，仍生成
            ds = {**same_above, "confidence": "LOW"}
            datasets.append(ds)

    return datasets


# Step 3: 向量检索增强（FAISS）
# 如果FAISS索引中有文本chunk的embedding，
# 可以在KG路径不完整时，用相似chunk补充上下文：

def augment_with_vector_search(kpi: dict, faiss_index, top_k: int = 3) -> list:
    query = kpi["description"]
    query_vec = embed(query)               # 生成查询向量
    distances, indices = faiss_index.search(query_vec, top_k)
    chunks = [load_chunk(i) for i in indices[0]]
    return chunks  # 补充到evidence中
"""

if __name__ == "__main__":
    import json
    print("数据汇交本体 已加载")
    print(f"  JSON Schema: Project → SubProject → KPI → Dataset")
    print(f"  SQLite 表数: 7 (含 project/sub_project/kpi/entity/relation/dataset/vector)")
    print(f"  关系类型: 6 (VIA/VERIFIES/EXECUTES/PRODUCES/BELONGS_TO/MAPS_TO)")
    print(f"  实体类型: 9 (4核心+5辅助)")
    print(f"  数据类型: 10 (测试/监测/仿真/模型/算法/样本/图像/工艺/勘测/其他)")
