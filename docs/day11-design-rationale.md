# Day 11：FastAPI 服务化

## 1. 为什么 RAG 需要服务化

前面的 Python 脚本只能由开发者在命令行调用。服务化把检索、Reranker、
安全门和生成模型包装成稳定的 HTTP 契约，使 Web 页面、自动化系统或其他
语言的客户端都能调用，而不需要了解内部模型代码。

## 2. 宏观架构

~~~mermaid
flowchart TB
    Client["Web/CLI/其他服务"] --> Middleware["请求 ID + 耗时中间件"]
    Middleware --> Validation["Pydantic 输入校验"]
    Validation --> Route["FastAPI 路由"]
    Route --> Service["QueryService"]
    Service --> Guard["Day 10 安全门"]
    Service --> Pipeline["检索 → 重排 → 生成"]
    Route --> Schema["统一响应模型"]
    Route --> Errors["错误码与 HTTP 状态"]
    Service --> Logs["Day 13 结构化日志"]
~~~

当前 DemoQueryService 是 CPU、确定性的参考实现，用于验证 API 工程。生产
部署时保持 QueryService 契约不变，把内部替换为 Day 8 Hybrid、Day 9
bge-reranker-v2-m3 和真实 LLM。

## 3. 新增文件及用途

- app/schemas.py：请求和响应的 Pydantic 模型。
- app/errors.py：稳定业务错误码和 APIError。
- app/services.py：文档注册表、可替换查询服务、缓存及阶段耗时。
- app/main.py：应用工厂、中间件、异常处理和所有路由。
- tests/test_api.py：接口集成测试。
- scripts/benchmark_day11_api.py：30 次请求的合约和延迟实验。
- data/eval/day11_api_results.json：实验结果。

## 4. 前端、后端与 API

前端负责用户交互和展示，不应直接连接 Qdrant 或加载模型。后端负责验证
请求、访问模型/索引、执行权限和安全策略，并返回 JSON。API 是双方约定的
“合同”：字段、类型、状态码一旦公开，就不应随内部重构随意变化。

本项目采用 REST/JSON，因为调试方便，浏览器和 Python 都能直接调用。
OpenAPI 由 FastAPI 根据类型自动生成，启动后访问：

~~~text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/openapi.json
~~~

## 5. Pydantic 校验为什么重要

QueryRequest 约束 question 长度、top_k 范围和 include_sources 类型。
校验发生在业务逻辑之前，因此 top_k=100 不会直接放大检索、Reranker 和
显存开销。错误返回稳定的 INVALID_REQUEST，而不是一段 Python traceback。

替代方案是手写 if 判断，但容易遗漏嵌套字段，无法自动生成 OpenAPI。
JSON Schema 也能校验，不过 Pydantic 与 Python 类型、FastAPI 集成更自然。

## 6. 接口设计

| 方法与路径 | 作用 | 成功响应 | 常见错误 |
|---|---|---|---|
| GET /health/live | 进程是否活着 | 200 alive | 进程不可达 |
| GET /health/ready | 模型/索引是否就绪 | 200 ready | 503 SERVICE_NOT_READY |
| POST /v1/query | 完整问答 | QueryResponse | 422/503/504 |
| POST /v1/query/stream | SSE 流式回答 | text/event-stream | 422/503 |
| POST /v1/documents | 注册文档 | DocumentResponse | 422 |
| GET /v1/documents | 列出文档 | 文档数组 | 500 |
| DELETE /v1/documents/{id} | 删除注册记录 | deleted=true | 404 |
| POST /v1/feedback | 记录评价 | feedback_id | 422 |

文档仓库目前是内存实现，进程重启会丢失，且不会自动重建向量索引。这是为了
把接口设计与离线入库流程分开。生产应改为数据库中的任务记录，并把解析、
Embedding、索引更新交给异步任务队列。

## 7. 请求 ID、异常与超时

每个请求都有 req_xxx。客户端也可通过 X-Request-ID 传入自己的追踪 ID，
响应头和响应体会原样返回。出现投诉时可用一个 ID 串联网关、检索和模型日志，
而不是依赖可能包含隐私的原始问题文本。

asyncio.timeout 给整条查询设置上限。超时返回 504 QUERY_TIMEOUT；模型未加载
返回 503；文档不存在返回 404。HTTP 状态方便负载均衡器处理，业务错误码让
客户端稳定展示具体原因。

当前 OpenAICompatibleClient 是同步 httpx 调用。如果直接放进 async 路由，
它会阻塞事件循环。生产可改用 httpx.AsyncClient、线程池，或者把模型服务
独立部署并通过异步网络请求调用。

## 8. 存活检查与就绪检查

存活检查只问“进程是否还能运行”。失败时容器平台可以重启进程。就绪检查问
“模型、Qdrant 集合和依赖是否已加载”。未就绪时负载均衡器停止发送用户请求，
但不应重启正在加载大模型的进程。

因此模型加载几分钟时，/health/live 应返回 200，/health/ready 返回 503。
把二者合并会产生启动抖动：探针误以为服务死亡，持续重启，模型永远加载不完。

## 9. 如何运行

启动后端：

~~~powershell
cd D:\Documents\大模型项目\opspilot
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
~~~

新开 PowerShell 测试：

~~~powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
$body = @{question="Docker 容器退出后如何查看日志？"; top_k=5} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/query -ContentType "application/json" -Body $body
~~~

运行自动实验：

~~~powershell
.\.venv\Scripts\python.exe scripts\benchmark_day11_api.py --requests 30
.\.venv\Scripts\python.exe -m pytest tests\test_api.py -q
~~~

## 10. 实验结果与分析

| 项目 | 结果 |
|---|---:|
| 请求数 | 30 |
| 成功率 | 100% |
| 平均延迟 | 3.297 ms |
| P50 | 2.976 ms |
| P95 | 3.776 ms |
| Max | 12.046 ms |
| OpenAPI/存活/就绪/422 校验 | 全部通过 |

最大值远高于 P95，说明首次导入、客户端初始化或操作系统调度形成冷启动离群点。
看平均值会被单个慢请求影响，所以线上必须同时看 P50、P95、P99。

这些数字不能代表真实大模型服务：TestClient 在同一进程，不包含网络、TLS、
网关和 GPU 推理；查询服务还是确定性 CPU Demo。它证明 API 层开销小且契约
可用，不证明 RAG 答案质量。模型质量仍应看 Day 7–10 的 Recall、MRR、引用
和安全指标；真实端到端延迟需接入 bge Reranker 与 LLM 后重测。

## 11. 替代方案

| 方案 | 优点 | 为什么本项目未选 |
|---|---|---|
| Flask | 简单、生态成熟 | 类型校验和 OpenAPI 需额外组合 |
| Django/DRF | ORM、后台管理完整 | 对单一模型微服务偏重 |
| gRPC | 二进制协议、高效、强类型 | 浏览器和人工调试不如 REST 直接 |
| FastAPI | 类型、异步、OpenAPI 一体 | 本项目选择，适合 Python 模型服务 |

## 12. 部署思考

模型未加载完成时不能接收真实查询。启动顺序应是：创建进程 → 存活成功 →
加载配置和索引 → 模型预热 → 设置 ready=true → 接收流量。关闭时反向执行：
先从就绪池摘除，等待正在处理的请求结束，再释放模型和连接。

多 worker 需要谨慎：每个 Uvicorn worker 都可能各自加载一份 GPU 模型，
造成显存翻倍。常见做法是 API 多副本、独立单副本模型服务，或每张 GPU
绑定一个推理进程，并在网关处限流。
