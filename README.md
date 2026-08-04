# OpsPilot

OpsPilot是一个面向大模型应用部署场景的智能故障诊断RAG系统。系统根据Docker、GPU、CUDA、PyTorch、vLLM、FastAPI和Qdrant等可信技术文档，为部署故障生成有引用、可追溯的结构化排查建议；证据不足时拒绝武断判断并提出澄清问题。

> 当前状态：第1天——需求定义与项目初始化。

## 项目目标

- 构建文档解析、清洗、分块和索引链路；
- 实现BM25与向量混合检索；
- 使用Reranker提高证据排序质量；
- 输出可能原因、排查步骤、修复建议、风险和引用；
- 建立拒答、评测、日志和可观测能力；
- 通过FastAPI、Web界面和Docker Compose完成服务化部署；
- 对比云端API模型与本地GPU推理模型的质量、延迟和成本。

## 目标用户

- 大模型应用和RAG开发工程师；
- 部署本地推理模型的算法工程师；
- AI后端、平台和初级运维工程师。

## 首版故障范围

1. Docker镜像、容器、端口、挂载和持久化；
2. NVIDIA驱动、GPU识别、CUDA与PyTorch兼容；
3. vLLM模型加载、显存、上下文和吞吐；
4. FastAPI服务超时、并发和异常；
5. Qdrant连接、索引和数据持久化；
6. Linux磁盘、权限、端口和进程问题。

## 计划架构

```text
技术文档
  -> 解析、清洗、去重、分块
  -> BM25索引 + Embedding向量索引
  -> 混合召回
  -> Reranker重排序
  -> 可回答性判断
  -> LLM结构化生成
  -> 引用校验、展示、反馈和评测
```

## 技术栈

- Python 3.11
- FastAPI、Uvicorn、Pydantic
- BM25、BGE Embedding、BGE Reranker
- Qdrant
- Streamlit（计划）
- Docker Compose（计划）
- pytest、Ruff
- 本地开发：Windows IDE
- 模型运行与压测：Linux GPU服务器

## 项目结构

```text
opspilot/
├─ app/             # FastAPI应用和路由
├─ core/            # 配置、日志、异常和公共组件
├─ ingestion/       # 文档解析、清洗、分块和索引
├─ retrieval/       # BM25、向量检索、融合和重排序
├─ generation/      # Prompt、模型客户端、引用和拒答
├─ evaluation/      # 评测指标和实验脚本
├─ frontend/        # 演示界面
├─ tests/           # 自动化测试
├─ configs/         # 模型、检索和实验配置
├─ scripts/         # 建库、评测和压测脚本
├─ data/            # 原始数据、处理数据和评测集
└─ docs/            # 需求、架构、ADR和实验报告
```

## 本地初始化

要求Python 3.11或3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

`.env`仅保存在本地，不得提交真实API密钥。

## 开发检查

```powershell
python -m ruff check .
python -m pytest
```

## 开发与部署方式

- 在本地IDE编写代码、运行单元测试和轻量调试；
- 使用Git同步代码到Linux GPU服务器；
- 在GPU服务器运行Embedding、Reranker、本地大模型和压测；
- 后续使用Docker Compose统一核心服务环境；
- 模型权重、缓存、真实密钥和向量库数据不进入Git。

完整方案见[需求说明](docs/requirements.md)和[ADR-001](docs/adr/ADR-001-development-and-deployment-environment.md)。

## 开发进度

- [x] 项目需求与范围
- [x] 依赖和环境变量模板
- [x] 本地开发与GPU部署决策
- [ ] 文档收集与数据清单
- [ ] 文档解析和分块
- [ ] 基础向量检索
- [ ] 基础RAG
- [ ] 评测集和检索基线
- [ ] 混合检索与Reranker
- [ ] 引用与拒答
- [ ] API和Web界面
- [ ] Docker部署和可观测性
- [ ] 性能、稳定性和最终评测

## 安全边界

OpsPilot只提供基于文档证据的诊断建议，不会自动连接用户服务器或执行运维命令。涉及删除、覆盖、权限修改和生产配置变更的操作，必须由用户评估、确认并准备回滚方案。

## License

本项目计划使用MIT License；正式发布前应补充`LICENSE`文件并确认所用文档和模型的许可要求。

