# OpsPilot

面向大模型应用部署场景的、证据可追溯的智能故障诊断 RAG 系统。

OpsPilot 围绕 Docker、GPU、CUDA、PyTorch、vLLM、FastAPI 和 Qdrant 等技术文档，
完成文档治理、解析分块、混合检索、Cross-Encoder 重排、引用校验、拒答与追问、
FastAPI 服务、SSE 流式交互以及结构化可观测性。

> 当前进度：已完成 20 天计划中的 Day 1–13。下一阶段为缓存与压测、真实 LLM
> 接入、Docker/GPU 部署和最终端到端评测。

## 项目价值

- 将“只会回答”的 RAG 扩展为能回答、追问和拒答的安全诊断系统；
- 同时利用 Dense、BM25 和 RRF，兼顾语义表达与命令、错误码、版本号；
- 使用 bge-reranker-v2-m3 对候选证据精排，并在 RTX 5090 上完成延迟实验；
- 为答案返回句子级引用、原文片段和来源地址，便于人工核验；
- 提供 FastAPI、OpenAPI、Streamlit、SSE、请求 ID、超时和健康检查；
- 记录脱敏结构化日志及检索、重排、生成、缓存等阶段指标。

## 系统架构

~~~mermaid
flowchart LR
    D["公开技术文档"] --> P["解析、清洗、去重"]
    P --> C["章节感知 / 固定长度 Chunk"]
    C --> B["BM25 索引"]
    C --> V["Embedding + Qdrant"]
    Q["用户问题"] --> B
    Q --> V
    B --> R["RRF 融合 Top-20"]
    V --> R
    R --> X["bge-reranker-v2-m3"]
    X --> G{"可回答性安全门"}
    G -->|证据充分| L["LLM / 结构化生成"]
    G -->|信息不完整| A["澄清问题"]
    G -->|证据不足| F["拒答"]
    L --> E["句子级引用校验"]
    E --> API["FastAPI + SSE"]
    API --> UI["Streamlit 页面"]
    API --> O["脱敏日志与指标"]
~~~

## 已实现能力

### 数据与知识库

- 收集并治理公开部署技术文档，记录 URL、产品、许可证和内容哈希；
- 支持 Markdown、HTML 和 PDF 解析；
- 完成 Unicode NFKC、控制字符、零宽字符和空白清洗；
- 提供章节感知与固定长度两种 Chunk 策略；
- 为 Chunk 保存来源、章节、页码、内容哈希和稳定 ID；
- 当前成功解析 43 份文档，并记录失败与部分修复案例。

### 检索与重排

- HashEmbedding + Qdrant Dense 检索基线；
- rank-bm25 关键词检索；
- Reciprocal Rank Fusion（RRF）混合召回；
- bge-reranker-v2-m3 Cross-Encoder 重排；
- Recall@k、Hit Rate@k、MRR@k、P50 和 P95 评测；
- 支持 CPU 与 GPU Reranker 对照实验。

### 安全回答与应用服务

- answer / clarify / refuse 三态可回答性策略；
- 证据不足拒答和缺失上下文追问；
- 检查未知来源 ID 和未引用事实句；
- Pydantic 请求、响应和错误契约；
- FastAPI 查询、文档注册、健康检查和用户反馈接口；
- SSE 流式事件及客户端断开检测；
- Streamlit 聊天、引用展开和点赞/点踩入口；
- 请求 ID、超时、缓存命中、Token 和阶段耗时日志；
- API Key、Authorization、邮箱和原始问题脱敏。

## 关键实验结果

### Dense、BM25 与 Hybrid RRF

在 42 条具有 gold evidence 的可回答问题上，使用相同 section Chunk、候选数
20 和 rrf_k=60：

| 方法 | Recall@5 | Hit Rate@5 | MRR@5 | P50 | P95 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.1865 | 0.2143 | 0.1437 | 2.919 ms | 3.350 ms |
| BM25 | 0.1984 | 0.2143 | 0.1397 | 3.795 ms | 6.136 ms |
| Hybrid RRF | **0.2341** | **0.2619** | **0.1754** | 6.937 ms | 9.814 ms |

相较 Dense，Hybrid 的 Recall@5、Hit Rate@5 和 MRR@5 分别相对提升约
25.5%、22.2% 和 22.1%。当前 Hybrid 延迟为顺序调用结果；理论并行延迟只是
估计，尚不能作为真实并行报告。

### Reranker 与 RTX 5090

| 方法 | Candidate k | Final k | Hit@5 | MRR@5 | P50 |
|---|---:|---:|---:|---:|---:|
| Hybrid | 20 | 5 | 0.2619 | 0.1754 | 不含重排 |
| Hybrid + bge-reranker-v2-m3 | 20 | 5 | **0.4286** | **0.3591** | 118.198 ms |

RTX 5090、batch_size=4 时，Reranker P95 为 144.090 ms。Candidate Recall@20
仍为 0.3889，说明 Reranker 改善的是已有候选的排序，不能找回未进入候选集的
证据。下一步应优先提升 Embedding 与召回，再继续优化重排。

### 安全门、API 与可观测性

| 实验 | 结果 | 说明 |
|---|---|---|
| Day 10 安全门 | 危险回答率 90% → 0% | 18 条受控校准题，不等同线上效果 |
| Day 11 FastAPI | 30/30 成功，P95 3.776 ms | TestClient + CPU Demo，不含真实模型 |
| Day 12 SSE | 20/20 完整，首事件 P95 6.371 ms | 验证传输协议，不是真实 LLM TTFT |
| Day 13 日志 | 36/36 成功，原始问题未入日志 | 受控重复请求，缓存命中率 91.67% |

## 当前真实模型接入状态

项目已经实现 OpenAI-compatible 模型客户端、结构化 Prompt 和基础 RAG Pipeline，
但当前 FastAPI 默认使用可重复、CPU-only 的 DemoQueryService，以便稳定测试
API、流式输出和可观测性。它不是 ChatGPT 或 DeepSeek，也不能代表真实答案质量。

下一阶段会实现 ProductionRAGQueryService，将以下组件接成端到端链路：

~~~text
Hybrid Top-20
  -> bge-reranker-v2-m3 Top-5
  -> 可回答性安全门
  -> OpenAI / DeepSeek API 或 AutoDL vLLM
  -> 引用校验
  -> FastAPI / Streamlit
~~~

云端 API 不需要本地 GPU；在 AutoDL 自托管开源模型时才需要 GPU。

## 技术栈

- Python 3.11
- FastAPI、Uvicorn、Pydantic、Streamlit、httpx
- Qdrant、rank-bm25、RRF
- HashEmbedding 基线、sentence-transformers
- BAAI/bge-reranker-v2-m3
- pytest、Ruff、uv
- 本地 Windows 开发，AutoDL Ubuntu + RTX 5090 GPU 实验

## 项目结构

~~~text
opspilot/
├─ app/             # FastAPI 路由、Schema、错误和业务服务
├─ core/            # 脱敏日志、指标和公共能力
├─ ingestion/       # 文档解析、清洗和分块
├─ retrieval/       # Dense、BM25、RRF 和 Reranker
├─ generation/      # Prompt、模型客户端、RAG、安全门和引用
├─ evaluation/      # Recall、Hit Rate、MRR 等指标
├─ frontend/        # Streamlit 页面与 SSE 客户端
├─ scripts/         # 建库、查询、评测和指标汇总脚本
├─ tests/           # 单元测试与 API 集成测试
├─ data/            # 数据清单、Chunk、索引元数据和实验结果
└─ docs/            # 每日设计说明、需求和 ADR
~~~

## 快速开始

要求 Python 3.11 或 3.12。

使用 uv：

~~~powershell
git clone https://github.com/YEYEPrincess/OpsPilot-.git
cd OpsPilot-
uv sync --extra dev
Copy-Item .env.example .env
~~~

没有 uv 时：

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
~~~

不要把真实 API Key 提交到 Git。

### 启动 FastAPI

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
~~~

访问：

- OpenAPI 页面：http://127.0.0.1:8000/docs
- 存活检查：http://127.0.0.1:8000/health/live
- 就绪检查：http://127.0.0.1:8000/health/ready

### 启动 Streamlit

另开一个 PowerShell：

~~~powershell
.\.venv\Scripts\python.exe -m streamlit run frontend\streamlit_app.py
~~~

浏览器默认打开 http://localhost:8501。

### 运行检查

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
~~~

## API 概览

| 方法与路径 | 作用 |
|---|---|
| GET /health/live | 检查进程是否存活 |
| GET /health/ready | 检查模型与索引是否就绪 |
| POST /v1/query | 返回完整问答结果 |
| POST /v1/query/stream | 返回 SSE 流式事件 |
| POST /v1/documents | 注册文档元数据 |
| GET /v1/documents | 列出已注册文档 |
| DELETE /v1/documents/{id} | 删除文档注册记录 |
| POST /v1/feedback | 记录回答反馈 |

当前文档接口使用内存注册表，不会自动完成解析、Embedding 和索引更新。生产版本
应将入库作为异步离线任务。

## 设计与实验文档

- [Day 1：需求与工程决策](docs/day1-design-rationale.md)
- [Day 2：数据收集与治理](docs/data-governance.md)
- [Day 3：解析、清洗与分块](docs/day3-design-rationale.md)
- [Day 4–5：向量检索与基础 RAG](docs/day4-day5-design-rationale.md)
- [Day 6：评测集与证据标注](docs/day6-评测.md)
- [Day 7：检索基线](docs/day7-design-rationale.md)
- [Day 8–9：混合检索与 Reranker](docs/day8-day9-design-rationale.md)
- [Day 10：引用、拒答与追问](docs/day10-design-rationale.md)
- [Day 11：FastAPI 服务化](docs/day11-design-rationale.md)
- [Day 12：Streamlit 与流式输出](docs/day12-design-rationale.md)
- [Day 13：日志与可观测性](docs/day13-design-rationale.md)
- [需求说明](docs/requirements.md)
- [开发与部署环境 ADR](docs/adr/ADR-001-development-and-deployment-environment.md)

## 开发进度

- [x] Day 1：需求、范围和环境决策
- [x] Day 2：文档收集、清单和数据治理
- [x] Day 3：解析、清洗、去重和分块
- [x] Day 4：Embedding 与 Qdrant 向量检索
- [x] Day 5：基础 RAG 闭环
- [x] Day 6：60 条评测数据与证据标注
- [x] Day 7：Recall@k、Hit Rate 和 MRR 基线
- [x] Day 8：BM25 与 Hybrid RRF
- [x] Day 9：Cross-Encoder Reranker 与 GPU 实验
- [x] Day 10：引用校验、拒答和追问
- [x] Day 11：FastAPI、Pydantic、错误码和健康检查
- [x] Day 12：Streamlit、SSE 和反馈
- [x] Day 13：脱敏日志和关键指标
- [ ] Day 14：缓存、批处理和并发压测
- [ ] 真实 LLM 端到端接入与评测
- [ ] Docker Compose 与 GPU 部署
- [ ] 稳定性、成本和最终验收

## 安全边界

OpsPilot 只提供基于文档证据的诊断建议，不会自动连接用户服务器或执行运维命令。
涉及删除、覆盖、权限修改、驱动升级和生产配置变更的操作，必须由用户确认并准备
回滚方案。日志默认不保存原始问题、API Key 或 Authorization。

## 已知限制

- 当前 Dense 基线是 hash-v1，不具备充分语义和跨语言能力；
- Candidate Recall@20 仍偏低，应优先升级 Embedding 和召回；
- FastAPI 默认连接 DemoQueryService，尚未形成真实 LLM 生产闭环；
- SSE 实验验证的是协议，尚未报告真实模型逐 Token TTFT；
- 本地 TestClient 延迟不包含网络、网关和 GPU 排队；
- 文档管理目前是内存注册，不等同于在线增量建库。

## License

项目代码按 MIT License 方向维护；正式发布前需补充 LICENSE 文件，并持续确认
所使用技术文档、Embedding、Reranker 和生成模型的许可要求。
