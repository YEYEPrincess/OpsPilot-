# 第4、5天教学设计：从机器学习到可运行的 RAG

本文面向已经学过基础机器学习算法、但刚开始学习大模型应用开发的同学。重点不是只记住“调用向量数据库和大模型”，而是理解每一个对象、每一条数据和每一个失败分支为什么存在。

## 0. 先建立全局认识：RAG 到底解决什么问题

大模型本身只根据参数和当前上下文生成文本。它不一定知道你的内部文档，也不能保证记住最新的 Docker、CUDA、vLLM 文档。RAG（Retrieval-Augmented Generation，检索增强生成）把“查资料”和“生成回答”连接起来：

```text
用户问题
  ↓
问题 Embedding
  ↓
向量检索相关 Chunk
  ↓
把 Chunk 组织成上下文
  ↓
Prompt 告诉大模型只能依据证据回答
  ↓
生成答案、排查步骤和引用
```

可以把它类比成机器学习中的两阶段系统：

- 检索器是一个候选样本召回模块，目标是尽量找全相关证据；
- 生成器是一个条件生成模型，目标是根据问题和证据组织答案；
- 引用和结构化校验是输出约束与可观测性，而不是“模型自然会做到”的事情。

第4天主要完成检索器，第5天把检索器和生成器接成最小闭环。

## 1. 第4天的输入、输出和数据流

第3天已经生成了章节感知 Chunk：

```text
data/processed/chunks_section.jsonl
```

每一行是一个 JSON 对象，包含：

```json
{
  "chunk_id": "docker-001:section:0006",
  "text": "When you start a container...",
  "source_url": "https://docs.docker.com/...",
  "section_path": ["Running containers", "Foreground and background"],
  "page": null,
  "source_sha256": "...",
  "text_sha256": "..."
}
```

第4天的目标是为每个 `text` 计算一个向量，并把向量和这些元数据一起存储：

```text
Chunk 文本
  ↓ Embedding 模型
384 维向量
  ↓ Qdrant upsert
向量 + Chunk 元数据
```

查询时流程相反：

```text
用户问题
  ↓ 同一个 Embedding 后端
问题向量
  ↓ Cosine 相似度
Top-k Chunk
```

必须使用“同一个” Embedding 空间编码文档和问题。不能用模型 A 编码文档、模型 B 编码问题，因为两套向量的坐标含义不同，距离没有可比性。

## 2. Embedding：从词袋到向量空间

### 2.1 Embedding 是什么

Embedding 是把文本映射到连续向量空间的函数：

```text
f("CUDA out of memory") → [0.12, -0.03, ..., 0.47]
```

在机器学习课程中，你可能见过：

- one-hot 向量：维度很高且稀疏；
- TF-IDF：强调文档中有区分度的词；
- PCA/Word2Vec：把对象映射到低维连续空间。

文本 Embedding 的目标类似，但通常由 Transformer 编码器学习得到：语义相近的句子在向量空间中更接近。例如，真实语义模型希望让“显存不足”和“GPU memory is insufficient”距离较近，即使两者没有完全相同的词。

### 2.2 本项目的两个后端

文件：`retrieval/embeddings.py`

#### HashEmbeddingProvider：本地基线

默认实现把词和字符三元组哈希到 384 个桶，再做 L2 归一化。它不是训练出来的语义模型，而是一个确定性的稀疏特征投影基线。

它的优点：

- 不下载模型权重；
- 不要求 GPU；
- 不需要 API Key；
- 同样输入永远得到同样向量；
- 适合先验证 Qdrant、批处理和 RAG 编排。

它的缺点：

- 不理解真正的上下文语义；
- 中文和英文同义表达不一定接近；
- 对复杂自然语言问题的召回效果有限。

因此 `hash-v1` 只能叫“Embedding 基线”，不能在简历中表述成 BGE 语义模型。

#### SentenceTransformerProvider：真实语义模型入口

同一个文件提供了可选的 Sentence-Transformers 适配器。部署到 GPU 服务器时，可以加载 BGE、E5 等模型，让编码结果真正包含语义信息。

代价是：

- 需要下载权重；
- 需要管理 PyTorch、CUDA 和显存；
- 模型名称、维度和版本必须写入索引清单；
- 更换模型后必须重建向量索引。

### 2.3 为什么要 L2 归一化

两个向量的点积会同时受到方向和长度影响。归一化后：

```text
||v||₂ = 1
```

Cosine 相似度就主要比较方向，也就是文本特征组合是否相似，而不是文本长度。

## 3. Cosine 相似度和 Top-k

给定问题向量 `q` 和文档向量 `d`：

```text
cos(q, d) = (q · d) / (||q|| ||d||)
```

结果通常在 `[-1, 1]`。越接近 1，方向越相似。

Top-k 的意思是返回相似度最高的 k 个 Chunk。例如 `top_k=5`：

```text
候选 Chunk 1000 个
  ↓ 计算相似度
排序
  ↓
返回前 5 个
```

`k` 太小可能漏掉重要证据，`k` 太大则会给 Prompt 塞入大量无关内容。第4天使用 5 作为开发基线，后续应通过评测比较 `k=3/5/10/20`。

## 4. Qdrant：为什么不把向量直接存在 JSONL

文件：`retrieval/qdrant_store.py`

JSONL 适合保存原始记录，但每次检索都遍历 1042 个向量并手动计算距离，会越来越慢，也缺少成熟的索引、Payload 过滤和服务化能力。

Qdrant 是专门的向量数据库。它把一条数据拆成三部分：

```text
Point ID：稳定的数字 ID
Vector：384 维浮点数组
Payload：Chunk 原文和来源元数据
```

本项目 Schema：

```text
collection = opspilot_chunks_v1
dimension = 384
distance = Cosine
```

### 为什么叫 `v1`

索引不是永远兼容的。如果发生以下变化，就应该创建 `v2`：

- 更换 Embedding 模型；
- 向量维度变化；
- Chunk 策略变化；
- Payload 字段变化；
- 相似度或分数融合规则变化。

如果直接覆盖旧索引，线上很难知道当前结果来自哪套配置。索引清单写入：

```text
data/manifest/vector_index.json
```

其中记录模型/Provider、维度、集合名、Chunk 来源、批量大小和点数。

### 本地 Qdrant 和服务器 Qdrant

当前使用：

```python
QdrantClient(path="data/index/qdrant")
```

它是本地磁盘模式，适合开发和小规模测试。

生产环境可以运行独立 Qdrant Server，让 API 服务通过网络访问。这样更适合多实例服务，但需要考虑 Docker、端口、磁盘、备份、鉴权和高可用。

## 5. 批量生成向量和失败重试

文件：`scripts/build_vector_index.py`

脚本执行：

1. 逐行读取 `chunks_section.jsonl`；
2. 按 `batch_size` 取出一批文本；
3. 生成这一批 Embedding；
4. 把向量和 Payload 一起 upsert 到 Qdrant；
5. 继续下一个批次；
6. 最后写索引清单。

默认：

```text
batch_size = 32
```

批量越大，GPU 吞吐通常越高，但显存峰值、失败重试成本也更高。批量越小，内存更安全、失败影响更小，但 Python 循环和模型调用开销更多。

当前使用稳定 Point ID，因此中断后可以不带 `--recreate` 重跑，已写入的 Point 会被幂等覆盖。它还没有保存细粒度 checkpoint；更成熟的生产版本可以记录最后成功的 batch、失败原因和重试次数。

## 6. 第4天检索结果和基线评测

文件：`scripts/query_vector_index.py`

命令：

```powershell
.\.venv\Scripts\python.exe scripts\query_vector_index.py `
  "Docker容器启动后立即退出，应如何排查？" `
  --top-k 5
```

输出不仅包含分数，还包含 URL、章节、页码和原文。检索结果必须保留这些字段，否则第5天模型即使回答正确，也无法向用户解释证据来自哪里。

本项目还加入了轻量词法重排，作为 Hash Embedding 的补救：错误码、命令、版本号和产品名往往需要精确匹配。真正的生产方案可以使用：

```text
BM25 关键词召回
  + Dense 向量召回
  ↓
分数融合
  ↓
Reranker 重排
```

文件：`scripts/evaluate_retrieval_smoke.py`

当前 10 题的 `category_product_hit@5` 为 0.8。它只是一个代理指标：判断 Top-5 是否出现期望产品，不代表答案已经正确，也不代表引用内容与问题完全相关。下一步应人工标注每题的相关 Chunk，计算真正的 Recall@k、MRR 和 nDCG。

## 7. 第5天的 RAG 输入输出

第5天输入：

```text
用户问题 + Top-k Chunk
```

第5天输出：

```json
{
  "status": "ok",
  "answer": "...",
  "possible_causes": ["..."],
  "steps": ["..."],
  "risks": ["..."],
  "citations": ["S1", "S2"],
  "clarification": "",
  "sources": []
}
```

`answer` 是给用户看的说明；`possible_causes` 和 `steps` 便于前端分组展示；`risks` 防止模型把危险变更说得过于轻率；`clarification` 用于证据不足时追问；`sources` 保存原始证据。

## 8. Prompt：为什么要明确告诉模型“只能依据证据”

文件：`generation/prompt.py`

大模型不是数据库查询器，而是概率生成模型。如果只给它一句问题，它可能根据训练记忆或语言模式补全答案。RAG Prompt 的核心约束是：

1. 给模型用户问题；
2. 给模型带编号的证据；
3. 要求只能根据证据回答；
4. 证据不足时明确说不足；
5. 只输出指定 JSON 字段；
6. 引用必须使用 `[S1]` 这样的证据编号。

Prompt 版本写成 `rag-v1`，因为 Prompt 变化也会影响输出质量。后续修改角色、格式、拒答规则或引用规范时，应升级到 `rag-v2`，并重新跑评测。

## 9. 上下文组装和上下文窗口

把 Top-k Chunk 直接拼接不是越多越好。需要考虑：

- 模型上下文窗口有限；
- 每个 Chunk 可能包含重复内容；
- 低分 Chunk 会稀释高分证据；
- 总 Token 数影响延迟和费用。

当前实现按 `[S1]`、`[S2]` 编号，加入 URL、章节、页码和正文。这种格式牺牲了一些 Token，但换来了可追溯性。

后续可以加入：

- 最大上下文 Token 限制；
- 相邻 Chunk 合并；
- 重复 Chunk 去除；
- 按分数截断；
- Reranker 后只保留最相关的 3~5 条。

## 10. 生成模型适配

文件：`generation/model_client.py`

### Mock 客户端

`MockGenerationClient` 是离线测试替身。它不会真正进行复杂推理，只把第一条证据的一部分组织成合法 JSON。

它的价值是验证：

```text
检索是否成功
Prompt 是否构造成功
JSON 是否能解析
来源是否能传到最终响应
```

不能用 Mock 结果证明大模型回答质量。

### OpenAI-compatible 客户端

`OpenAICompatibleClient` 请求：

```text
POST /v1/chat/completions
```

因此可以接入 vLLM、Ollama 或其他兼容服务。当前设置：

```text
temperature = 0
response_format = json_object
```

低温度是为了让评测结果更稳定；JSON response format 是对模型输出格式的额外约束，但它不能替代服务端和客户端的 JSON 校验。

## 11. 非法 JSON、超时和降级

文件：`generation/rag_pipeline.py`

生成阶段至少有三类失败：

### 传输失败

例如连接超时、HTTP 500、服务不可用。客户端可以有限次数重试，但不能无限重试，否则会拖垮请求线程。

### 结构失败

模型返回普通 Markdown，或者 JSON 缺少 `citations` 字段。客户端必须先 `json.loads`，再检查必需字段。

### 内容失败

JSON 格式正确，但引用了不存在的 `[S9]`，或者答案与证据无关。这需要更高层的引用校验和人工/自动评测，第5天只先保留来源并完成结构检查。

当前的降级策略：

```text
生成成功且字段齐全 → status=ok
生成失败或 JSON 非法 → status=degraded
```

降级响应不伪造答案，而是返回检索到的原文和人工确认提示。这是生产系统的重要安全边界。

## 12. 第5天冒烟测试

文件：`scripts/run_rag_smoke.py`

默认使用 Mock：

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_smoke.py
```

结果写入：

```text
data/eval/rag_smoke_results.jsonl
```

切换真实模型：

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_smoke.py `
  --provider openai-compatible `
  --llm-base-url http://127.0.0.1:8000 `
  --llm-model your-model-name
```

真实模型评测不能只看 `status=ok`，还应检查：

- 回答是否真正解决问题；
- 每个引用是否支持对应结论；
- 是否出现知识库外的编造；
- 排查步骤是否安全；
- 证据不足时是否拒答或追问。

## 13. CPU、GPU、远程 API 的部署选择

### Embedding

CPU 适合本地开发、小型知识库和低频更新；GPU 适合批量重建和高并发 Embedding 服务；远程 API 适合不想维护权重的团队，但要管理网络、费用和隐私。

### 生成模型

用户请求链路可以调用远程 API，也可以调用 GPU 服务器上的 vLLM。模型权重不应跟代码一起提交；应通过模型仓库、挂载盘或部署镜像管理。

### Qdrant

本地磁盘模式适合开发，独立 Qdrant Server 适合多实例服务。生产环境还需要备份、鉴权、磁盘容量监控和集合版本切换。

## 14. 用一句话总结第4、5天

第4天把“文本”变成“可计算相似度的向量和可追溯的证据”；第5天把“问题 + 证据”交给生成模型，并通过结构化输出、引用和降级机制把模型能力包装成一个可测试、可解释、可部署的应用闭环。