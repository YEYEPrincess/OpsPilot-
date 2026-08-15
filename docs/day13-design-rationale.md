# Day 13：结构化日志与可观测性

## 1. 可观测性解决什么问题

“服务很慢”不能指导修复。RAG 请求至少经过网络、检索、Reranker、生成模型
和流式传输。只有分阶段记录延迟、错误和资源信号，才能判断该扩 GPU、优化
索引、检查外部 API，还是修复缓存。

日志是离散事件记录；指标是可聚合的数值；Trace 是一次请求跨组件的因果链。
本日实现 JSONL 日志和轻量指标汇总，为后续 Prometheus/OpenTelemetry 留出
字段基础。

## 2. 整体架构

~~~mermaid
flowchart LR
    R["HTTP 请求"] --> ID["request_id"]
    ID --> Q["QueryService"]
    Q --> T1["retrieval_ms"]
    Q --> T2["rerank_ms"]
    Q --> T3["generation_ms"]
    Q --> C["cache_hit / token_total"]
    T1 --> J["JSONL query_completed"]
    T2 --> J
    T3 --> J
    C --> J
    J --> S["summarize_metrics.py"]
    S --> M["P50/P95、错误率、阶段均值、缓存率"]
~~~

## 3. 新增文件及作用

- core/observability.py：递归脱敏、问题指纹、JSONL 写入、分位数和聚合。
- app/main.py：HTTP 与 query_completed 事件埋点。
- app/services.py：retrieval/rerank/generation/cache_lookup 分阶段计时。
- scripts/summarize_metrics.py：把任意日志汇总为 JSON。
- scripts/run_day13_observability_demo.py：生成 36 次请求并定位最大平均阶段。
- tests/test_observability.py：验证密钥/邮箱脱敏和指标公式。
- data/eval/day13_sample_logs.jsonl：脱敏样例日志。
- data/eval/day13_metrics_summary.json：实验汇总。

## 4. 为什么使用结构化 JSONL

普通文本“query took 120 ms”需要正则解析；JSON 能直接按字段过滤：

~~~json
{
  "event": "query_completed",
  "request_id": "req_xxx",
  "latency_ms": 120.0,
  "stage_ms": {"retrieval": 5.0, "rerank": 95.0, "generation": 20.0},
  "cache_hit": false,
  "token_total": 430,
  "error_type": ""
}
~~~

JSONL 每行一个对象，追加写入简单，一行损坏不会影响整个文件。缺点是本地文件
不适合多副本集中查询、轮转和长期保存；生产应输出到 stdout，由日志代理收集
到 Loki、ELK 或云日志平台。

## 5. 应记录哪些字段

- timestamp、event、request_id：事件时间和关联键；
- path、status_code、latency_ms：HTTP 层健康；
- retrieval/rerank/generation/cache_lookup：阶段耗时；
- token_total：生成成本和长度趋势；
- cache_hit：判断低延迟是否来自缓存；
- error_type：超时、上游 429、非法 JSON 等稳定分类；
- question_sha256、question_chars：不保存原文仍能识别重复请求。

不要记录 API key、Authorization、密码、完整 Cookie、未经处理的用户问题、
整段私有文档或模型 Prompt。代码会按敏感键递归替换，并把邮箱替换为 [EMAIL]。
哈希不是加密：短文本可能被字典猜测，因此仍要做访问控制和保留期管理。

## 6. P50、P95 与平均值

把请求延迟排序：

- P50 是中位用户体验；
- P95 表示 95% 请求不超过该值，能反映长尾；
- Max 容易受单次冷启动影响；
- 平均值适合容量趋势，但会被离群点拉高。

本项目使用 nearest-rank 分位数，算法简单可复现。大规模监控通常使用直方图
或 t-digest，避免把所有原始延迟长期保存在内存。

## 7. 分阶段计时的正确口径

计时点应紧贴真实边界：调用检索前后、Reranker 前后、模型请求前后。总耗时
还包含验证、序列化和调度，因此不一定严格等于阶段之和。

首轮实验发现缓存命中仍复用了冷请求的 retrieval/rerank 数值，导致阶段耗时
可能大于本次总耗时。修复后缓存命中只记录 cache_lookup。这个案例说明：
“有指标”不等于“指标可信”，每个字段必须说明在哪开始、在哪结束、是否包含
排队、批处理和网络。

并行阶段也不能简单相加。例如 Dense 和 BM25 并行时，总检索时间接近二者的
较大值，而不是两者之和。批量 Reranker 的单请求耗时和整批 GPU 时间也需要
分别定义。

## 8. 如何运行

生成样例日志并汇总：

~~~powershell
cd D:\Documents\大模型项目\opspilot
.\.venv\Scripts\python.exe scripts\run_day13_observability_demo.py --requests 36
~~~

单独汇总已有日志：

~~~powershell
.\.venv\Scripts\python.exe scripts\summarize_metrics.py data\eval\day13_sample_logs.jsonl --output data\eval\day13_metrics_manual.json
~~~

运行测试：

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests\test_observability.py -q
~~~

## 9. 实验结果与分析

| 指标 | 结果 |
|---|---:|
| 成功请求 | 36/36 |
| query_completed 数 | 36 |
| 总延迟 mean / P50 / P95 | 0.083 / 0.068 / 0.207 ms |
| Max | 0.217 ms |
| 缓存命中率 | 91.67% |
| Token 总计 | 1608 |
| 错误率 | 0% |
| 最大平均冷路径阶段 | retrieval，0.087 ms |
| 原始问题出现在日志 | 否 |
| question_sha256 存在 | 是 |

缓存率高是因为 36 次请求只循环 3 个问题，第一次后都命中内存缓存，所以本次
总延迟极低。它证明缓存指标和脱敏字段有效，不代表真实用户流量也有 91.67%
命中率，更不代表模型推理只需 0.083 ms。

冷路径阶段均值中 retrieval 最大（0.087 ms）；cache_lookup 平均约 0.045 ms，generation
和 rerank 只是 CPU Demo 占位。Day 9 的真实 GPU Reranker P50 约 118 ms，
真实 LLM 还会更慢。接入真实模型后，预计瓶颈会从演示检索转移到 Reranker、
模型排队或生成阶段，必须重新采样。

## 10. 如何根据指标定位问题

| 现象 | 可能原因 | 下一步 |
|---|---|---|
| retrieval P95 上升 | Qdrant IO、网络、索引过大 | 看 Qdrant 延迟和磁盘 |
| rerank P95 上升 | GPU 排队、batch 太大 | 看显存、队列和批大小 |
| generation TTFT 上升 | 模型排队或冷启动 | 看队列长度、预热状态 |
| token 间隔上升 | 解码吞吐下降 | 看每秒 Token 和并发 |
| HTTP 高、阶段低 | 网关、序列化、连接 | 看反向代理和网络 Trace |
| 错误率升高且 429 多 | 上游限流 | 退避、限流和容量扩展 |
| 拒答率升高 | 数据漂移或检索退化 | 分域看召回和知识缺口 |

只看 GPU 利用率不够：GPU 低可能是检索慢、CPU Tokenizer 慢或请求没有形成
批次；GPU 高也可能只是错误请求在浪费算力。

## 11. 替代方案及演进

| 方案 | 优点 | 当前取舍 |
|---|---|---|
| Python JSONL | 零额外服务、便于学习 | 当前采用 |
| Prometheus | 指标告警和时序查询成熟 | 部署阶段加入 |
| OpenTelemetry | 跨 API/Qdrant/模型 Trace | 多服务后加入 |
| ELK/Loki | 日志检索和集中保存 | 多副本后加入 |
| Sentry | 异常聚合、版本关联 | 生产错误治理可加入 |

可观测性库应低开销且失败不影响主请求。日志磁盘写满时不能让 RAG 服务整体
崩溃；生产通常异步输出到 stdout，并限制字段大小和采样率。

## 12. 部署思考

现有字段可以区分检索、重排、生成和 HTTP，但还不能完整判断网络与 GPU 排队。
下一阶段应增加 upstream_connect_ms、queue_ms、ttft_ms、tokens_per_second、
Qdrant 请求耗时、GPU 显存/利用率以及模型 batch size。

告警不能只设一个总延迟阈值。建议至少有成功率、P95、拒答率、上游 429、
GPU OOM 和就绪状态；并按模型版本、索引版本、数据版本分组。否则发布新索引
后性能下降，也无法判断变化来自代码、模型还是数据。
