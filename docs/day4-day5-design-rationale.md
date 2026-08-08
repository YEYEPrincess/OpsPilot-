# 第4、5天设计说明：向量检索与基础 RAG

## 第4天：Embedding 与 Qdrant

### 1. Embedding 后端

默认使用 `HashEmbeddingProvider`。它把词和字符三元组经过哈希映射到 384 维向量，再做 L2 归一化。这是一个离线、确定性、无需模型下载的召回基线，便于 Windows 本地复现。

生产部署可选择 `SentenceTransformerProvider`，例如在 GPU 服务器加载 BGE/M3 等真实语义模型；也可以增加远程 Embedding API。当前不把大模型权重强行放进 Git 或本地初始化流程，因为下载体积、显存和网络可用性会让第4天无法稳定复现。

### 2. Qdrant 集合

`retrieval/qdrant_store.py` 封装集合创建、批量 upsert、Top-k 查询和计数。默认使用 `data/index/qdrant` 本地磁盘模式，集合名为 `opspilot_chunks_v1`，距离函数是 Cosine。

`v1` 是索引版本：只要 Embedding 模型、维度、分块策略或 Payload 结构发生不兼容变化，就创建 `v2`，而不是覆盖旧索引。部署到 Linux 时只需把 `path` 换成 Qdrant Server URL 或独立 Qdrant 服务。

### 3. 批量建库

`scripts/build_vector_index.py` 逐行读取 `chunks_section.jsonl`，按 `--batch-size` 编码后写入 Qdrant，并写 `data/manifest/vector_index.json`。

批量越大，吞吐通常越高，但 CPU 内存/GPU 显存占用也越大；批量越小，失败重试粒度更细但请求次数更多。默认 32 是保守基线，GPU 部署再通过实验调大。每个批次成功后才继续，失败时可从批次边界重跑。

### 4. 检索

`scripts/query_vector_index.py` 和 `QdrantVectorStore.search()` 对问题生成向量，执行 Cosine Top-k，并原样返回 Chunk 的 URL、章节、页码、哈希和正文。

纯向量检索实现简单、适合语义改写；BM25 对错误码、命令和版本号更敏感。第4天先建立向量基线，第9天再加入 BM25 混合检索和 Reranker。

## 第5天：基础 RAG

### 1. 结构化 Prompt

`generation/prompt.py` 固定 `PROMPT_VERSION=rag-v1`，要求模型只引用 `[S1]` 等证据，并输出 `answer`、`possible_causes`、`steps`、`risks`、`citations`、`clarification` 六个字段。

结构化输出比自由文本更容易被 API、前端和评测程序消费；严格 JSON 也更容易发现模型返回格式错误。自由文本的优点是模型约束少，但字段解析、引用校验和自动评测更困难。

### 2. 生成模型适配

`generation/model_client.py` 提供两个后端：

- `MockGenerationClient`：抽取证据生成可重复 Demo，不依赖模型服务；
- `OpenAICompatibleClient`：调用 vLLM、Ollama 或托管的 `/v1/chat/completions`。

远程客户端设置 `temperature=0`，并对 HTTP 错误重试。真实模型部署时只替换客户端，不改 RAG 编排。

### 3. RAG 编排和降级

`generation/rag_pipeline.py` 完成：问题向量化 → Top-k 检索 → 上下文组装 → 模型生成 → JSON 校验 → 返回答案和来源原文。

如果模型超时、HTTP 失败或 JSON 缺少字段，返回 `status=degraded`，保留检索到的来源和原文，并提示用户检查模型服务或人工判断。这样错误不会伪装成可靠答案。

### 4. 冒烟测试

`scripts/run_rag_smoke.py` 使用前 10 个种子问题，默认 Mock 模式运行，不需要 API Key；结果写入 `data/eval/rag_smoke_results.jsonl`。切换真实模型时传 `--provider openai-compatible --llm-base-url ... --llm-model ...`。

## 部署决策

Embedding：本地 CPU 适合开发和小规模索引；GPU 适合批量重建；远程 API 适合不维护模型权重但要管理成本、网络和隐私的场景。

生成模型：用户请求链路可以调用远程 API 或 GPU 上的 vLLM；Qdrant 和索引构建属于独立服务/离线任务。网络超时、模型服务 5xx 和非法 JSON 都要区分处理：传输层重试，结构层校验，最终返回证据降级结果。
