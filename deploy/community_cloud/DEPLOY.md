# OpsPilot 免费托管体验版

目标：访客通过固定公网网址体验问答，你的电脑关机后仍可访问。
采用 Streamlit Community Cloud 托管页面、内部 FastAPI 和轻量 Qdrant 本地索引，
生成调用你配置的 DeepSeek 账户。托管免费，模型调用按实际用量计费。

## 配置步骤

1. 打开 https://share.streamlit.io/ ，使用现有 GitHub 账号登录，连接该项目仓库。
2. 选择 **Create app**，填写：

   | 字段 | 值 |
   |---|---|
   | Repository | `YEYEPrincess/OpsPilot-` |
   | Branch | `codex/community-cloud` |
   | Main file path | `deploy/community_cloud/app.py` |
   | Python | `3.11` |
   | App URL | 可选一个未占用的名称；实际域名以平台确认为准 |

3. 在 **Advanced settings → Secrets** 中填写以下 TOML。实际 Key 只粘贴到平台私密设置，
   不要写进 GitHub、README 或聊天。模型名应与自己账户可用模型一致。

   ```toml
   LLM_API_KEY = "在平台私密设置中填入你的 DeepSeek Key"
   LLM_MODEL = "deepseek-flash"
   ```

4. 点击 **Deploy**，等待安装依赖、构建索引。依赖文件与入口同目录，不安装 PyTorch，
   不下载 BGE，不需要额外的 Qdrant 服务器。
5. 打开生成的网址，验证 `docker logs` 的真实回答和引用，以及
   `CUDA 有问题怎么处理？` 的追问；同时检查 SSE 完成、错误提示和公开可访问性。
6. 公网验收通过后，再将实际网址加入主分支 README 的“在线体验”入口。

## 具体配置

```text
访客浏览器 → Streamlit 页面
                ↓ 同一托管实例，127.0.0.1 内部 HTTP
             FastAPI → 基础 hash-v1 检索 / Qdrant 本地索引
                ↓ 安全门与上下文追问
             DeepSeek API → 引用结构校验 → 流式页面
```

- 知识包是原项目 43 份文档的 1,042 个片段，含原文地址；启动校验 SHA256 并重建索引。
  `knowledge.jsonl.gz` 是文本片段，不是模型权重。
- 基础 hash-v1 + 词项重合检索，Top-k=3，不加载 BGE；不套用重排实验的质量指标。
- 仅页面公开；FastAPI 只监听回环地址，管理接口不直接向访客开放。
- 密钥由平台 Secrets 提供；问题和检索片段会发送给 DeepSeek，页面已提示勿提交敏感信息。
- 输出上限 512 tokens，无自动重试；回答缓存最多 64 条，保留 300 秒。
- 调用模型前实施全站限额：UTC 每天 50 次、每自然分钟 4 次、同时最多 2 次。
  失败尝试也计数；缓存命中、直接追问或拒答不计生成次数。
- 计数保存在实例 SQLite，同一文件系统内重启进程不清空；平台重建、迁移或清理临时磁盘
  可能重置。这不是供应商账户的金额上限，也不能跨多个副本共享；应定期检查账单与余额。
- 每个浏览器会话仅保留最近 12 条消息；不向模型发送历史对话，仍是单轮检索问答。

## 免费方案的限制

该方案适合简历演示与小规模试用。连续 12 小时无人访问会休眠，访客可点击唤醒；
不依赖你的电脑，但不能保证全天候立即响应。公网网络可达性和启动时间须部署后实测。
若需要持续运行、更多并发或 GPU 重排，再迁移到付费服务。

当前已准备部署入口；创建公网实例需要仓库所有者在自己的平台账户中执行部署。
尚未完成公网验收时，不应把本机地址或猜测的网址写成“在线体验”。

## 发布前验证（2026-11-12）

- 当前部署分支 pytest：35 passed；新部署文件的 Ruff 检查通过。
- 4 项新增测试覆盖全站限额、并发竞争、同一磁盘重启后保留计数、中断释放资源，
  以及 1,042 片段真实索引上的本机 HTTP / SSE、缓存和追问。
- Streamlit 无密钥启动检查通过：显示配置提示，不抛出页面异常。
- 使用既有 DeepSeek 配置完成 1 次真实生成：`docker logs` 返回 `answer`，
  引用结构通过，消耗 1,291 tokens；含糊 CUDA 问题返回 `clarify`。
  这些是本地发布前验证，不等同于托管平台已经部署成功。

官方资料（2026-11-12 核对）：

- [免费 Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud)
- [部署、Python 与 Secrets 设置](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [休眠与资源限制](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- [入口同目录的依赖文件](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
