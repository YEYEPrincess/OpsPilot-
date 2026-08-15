# Day 12：Streamlit 交互界面与流式输出

## 1. 今天的系统位置

Day 11 已有后端 API，Day 12 增加面向用户的前端，但不把模型代码搬到前端。

~~~mermaid
sequenceDiagram
    participant U as 用户
    participant S as Streamlit
    participant A as FastAPI SSE
    participant P as RAG Pipeline
    U->>S: 输入故障问题
    S->>A: POST /v1/query/stream
    A->>P: 检索、重排、安全门、生成
    P-->>A: 答案、来源、耗时
    loop 文本片段
        A-->>S: event: token
        S-->>U: 更新占位区域
    end
    A-->>S: event: sources / done
    U->>S: 点赞或点踩
    S->>A: POST /v1/feedback
~~~

## 2. 新增文件及用途

- frontend/api_client.py：普通查询、SSE 解析和反馈调用。
- frontend/streamlit_app.py：聊天状态、流式显示、来源展开和反馈按钮。
- app/main.py 中 /v1/query/stream：SSE 服务端。
- app/schemas.py 中 FeedbackRequest：限制反馈字段和长度。
- tests/test_api.py：验证 meta/token/sources/done 事件及反馈接口。
- scripts/benchmark_day12_streaming.py：测量首事件和完整响应延迟。
- data/eval/day12_streaming_results.json：本次实验结果。

## 3. 什么是流式输出

非流式请求要等整段答案生成完才显示。流式请求把答案拆为多个增量片段，用户
更早看到内容。两个重要延迟：

- TTFT（Time To First Token/Event）：从发送请求到看到第一个片段；
- total latency：从发送请求到最后一个片段结束。

流式输出通常不减少模型总计算量，但显著改善“感知延迟”。如果 TTFT 很短、
总耗时很长，用户仍会觉得系统在工作；如果 TTFT 接近总耗时，流式价值有限。

## 4. 为什么选择 SSE

SSE 是服务器到浏览器的单向事件流，基于普通 HTTP。事件格式如下：

~~~text
event: token
data: {"delta": "建议先查看"}

event: sources
data: [{"citation_id": "S1", ...}]
~~~

| 技术 | 通信方向 | 优点 | 适用场景 |
|---|---|---|---|
| 轮询 | 客户端重复请求 | 最简单 | 长任务状态查询 |
| SSE | 服务端单向推送 | HTTP 友好、自动分事件 | LLM 文本流 |
| WebSocket | 双向长连接 | 交互能力最强 | 语音、协同、实时控制 |

聊天问答主要是服务端持续输出，SSE 足够；WebSocket 会增加连接状态、心跳和
网关配置复杂度，因此当前不选。

## 5. 当前实现与“真正模型流式”的区别

当前 DemoQueryService 先得到完整答案，再由 FastAPI 每 12 个字符发送一个
token 事件。这验证了传输协议、前端渲染、取消检查和来源事件，但不是模型
逐 Token 解码。

真实流式需让 GenerationClient 暴露 async generator，并将 vLLM/OpenAI
compatible 返回的 delta 直接向外转发。不能先缓冲完整响应，否则 TTFT 仍由
完整模型生成时间决定。文档明确区分二者，避免把“字符串切片”包装成模型
推理性能成果。

## 6. 引用和原文如何展示

主答案只保留 [S1] 等简洁标记；下方 expander 显示标题、score、原文片段和
原始链接。这样既保持聊天页面清晰，又允许用户审计模型答案。

前端只能展示后端返回的安全 URL 和文本。生产中要防止恶意 HTML、超长片段、
javascript URL 和 Markdown 注入；还应检查用户是否有权查看该来源。

## 7. 客户端断开、超时与取消

用户关闭页面后，如果服务器继续做 Reranker 和 LLM 推理，会白白占用 GPU。
事件生成器在每次发送前调用 request.is_disconnected；检测到断开后停止发送
并记录 stream_cancelled。

但这只取消 Python 生成器，不保证上游模型已停止。真正部署还需要：

1. 给模型请求传递取消信号或关闭 HTTP 流；
2. 推理服务支持 request_id 取消；
3. 为排队、首 Token、流间隔和总时间分别设置超时；
4. 在 finally 中释放连接、信号量和临时缓存。

反向代理还可能缓冲 SSE，所以响应设置 X-Accel-Buffering=no。Nginx、CDN
和云负载均衡器仍需单独验证空闲超时和缓冲配置。

## 8. 用户反馈和隐私

反馈只保存 request_id、up/down、问题类别和可选短评论，不重复上传原始问题。
request_id 可与受控日志关联。线上应设置保留周期、访问权限和删除流程，并在
训练前再次脱敏。不能默认把所有点踩文本直接加入训练集，因为其中可能有密钥、
客户数据或错误标注。

反馈的作用不是只算点赞率，还可以分解：

- incorrect：生成或证据理解错误；
- missing：召回缺失或知识库缺文档；
- unsafe：给出危险操作；
- other：交互或格式问题。

这些类别决定下一步应该改检索、Prompt、安全门还是前端。

## 9. 如何运行

PowerShell 窗口 1：

~~~powershell
cd D:\Documents\大模型项目\opspilot
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
~~~

PowerShell 窗口 2：

~~~powershell
cd D:\Documents\大模型项目\opspilot
.\.venv\Scripts\python.exe -m streamlit run frontend\streamlit_app.py
~~~

浏览器通常会自动打开 http://localhost:8501。若后端地址不同：

~~~powershell
$env:OPSPILOT_API_URL="http://服务器地址:8000"
.\.venv\Scripts\python.exe -m streamlit run frontend\streamlit_app.py
~~~

运行实验：

~~~powershell
.\.venv\Scripts\python.exe scripts\benchmark_day12_streaming.py --requests 20
~~~

## 10. 实验结果与分析

| 指标 | 结果 |
|---|---:|
| 请求数 | 20 |
| 完整收到 done 的比例 | 100% |
| 首事件平均 | 5.573 ms |
| 首事件 P50 / P95 | 4.935 / 6.371 ms |
| 完整响应平均 | 5.619 ms |
| 完整响应 P50 / P95 | 4.976 / 6.416 ms |
| 反馈接口 | 通过 |

首事件几乎等于完整响应，原因有两个：Demo 答案很短且先完整生成；TestClient
可能在同一进程缓冲数据。因此这次实验验证协议完整性，不应宣传为真实 LLM
TTFT。上线前必须通过 Uvicorn + 网络客户端 + 真模型重新测试。

Day 9 的 GPU Reranker 约 118 ms P50，而这里的 5 ms 不包含该模型，二者不可
直接比较或相加后当作生产结果。真实端到端路径还包括 Dense/BM25、Reranker、
LLM 首 Token、网络和前端渲染。

## 11. 替代界面

| 方案 | 优点 | 当前取舍 |
|---|---|---|
| Streamlit | Python 即可开发、适合简历 Demo | 采用，开发最快 |
| Gradio | 模型 Demo 组件丰富 | 引用和业务布局定制稍弱 |
| React/Vue | 生产体验和控制最强 | 需要额外前端工程时间 |
| 纯 HTML/JS | 依赖少 | 状态管理和组件需手写 |

Streamlit 适合 20 天项目展示，不代表最终生产前端。若面试岗位强调全栈部署，
可保留 FastAPI 契约，再用 React 替换界面而无需改模型服务。

## 12. 部署思考

流式连接会长时间占用 worker 和连接数。应限制并发流数量、每用户速率和最大
生成长度；断开后及时取消。多个用户同时生成时，GPU 推理服务最好进行连续
批处理，而不是在 FastAPI 中手工把不同用户文本拼成批次。前端还应明确显示
“生成中、已取消、已超时、证据不足”四种状态，避免用户误以为页面卡死。
