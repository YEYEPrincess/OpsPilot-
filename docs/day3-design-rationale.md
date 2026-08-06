# 第3天设计说明：文档解析与分块

## 1. 今天完成什么

流水线位于 `scripts/parse_docs.py`，输入是第2天下载报告中没有 `error` 的本地快照，输出为：

- `data/processed/documents.jsonl`：标准化文档和章节；
- `data/processed/chunks_fixed.jsonl`：固定长度基线分块；
- `data/processed/chunks_section.jsonl`：章节感知分块；
- `data/manifest/parse_stats.json`：文档、章节、分块和失败统计。

本次 44 条目录中，43 份成功解析；6 份 vLLM 页面因外站 HTTP 429 复用了此前已经下载且标题有效的本地快照，`vllm-003` 的旧快照是 `Redirecting...` 占位页，因此没有复用。这个失败案例保留在下载报告中，避免把重定向页当成知识。

## 2. 为什么选择这些解析方式

### 格式解析

- HTML 使用 BeautifulSoup，删除 script/style/nav/footer 等非正文节点，并识别 h1-h6；
- Markdown 使用标题正则和代码围栏状态机；
- PDF 使用 pypdf 按页提取文本，并把页码写进章节元数据。

替代方案可以使用 Unstructured、Apache Tika、LlamaIndex 或 LangChain loaders。它们适合快速覆盖更多格式，但会引入更大的依赖树、隐式切分规则和版本兼容成本。第3天先做可解释、可测试的最小解析器；后续若遇到复杂扫描 PDF，再增加 OCR/Unstructured 适配器，而不是一开始把所有能力耦合进核心流水线。

## 3. 清洗、去重和章节识别

清洗先做 Unicode NFKC、去控制字符、统一空白和删除零宽字符，以减少同一内容因编码或网页排版产生的差异。章节识别保留 heading path，例如 `["GPU", "CUDA graphs"]`，这样检索结果能解释“答案来自哪一节”。

章节去重使用清洗后文本的 SHA-256；相同正文只保留第一次出现的章节。整份文档仍保存 `source_sha256`，清洗后的全文保存 `content_sha256`，Chunk 保存 `text_sha256`，便于缓存失效、重复检测和审计。

## 4. 为什么同时保留两种分块

- 固定长度：1600 字符、240 字符重叠。它是稳定的召回基线，便于比较不同参数和估算索引规模；
- 章节感知：先按章节切分，再在超长章节内使用相同窗口。它尽量不跨越无关章节，适合故障排查语义。

也可以按 tokenizer token 数分块，或只按段落分块。token 分块更接近模型上下文限制，但需要绑定具体模型 tokenizer；只按段落又容易产生过短或超长片段。当前先用字符基线保持模型无关，后续第5/6天用验收集比较召回率、上下文长度和延迟后再调整。

每个 Chunk 都保存来源 URL、来源哈希、章节路径、页码（PDF）、策略、序号、字符数和文本哈希。这样回答出现问题时可以追溯到原始文档和具体章节，而不是只看到一段无来源文本。

## 5. 在线还是离线

解析属于离线 ETL，不放在用户请求链路中。在线请求只读取已经审核过的 Chunk 索引，避免网页波动、PDF 解析耗时和冷启动影响 P95 延迟。采集/解析失败后，按文档粒度重试；报告记录 HTTP 状态和错误；JSONL 采用临时文件写入后原子替换，避免半文件。解析器预留 `--resume`，当 `source_sha256` 没变化时复用已有文档结果。

## 6. 抽查和失败记录

抽查重点是：标题是否变成正文、HTML 是否只剩导航、Markdown 代码围栏是否破坏章节、PDF 页码是否存在、Chunk 是否带来源字段。第3天实际发现并修复了 MIME 为 `text/markdown` 却保存成 `.txt`、PyTorch/VLLM 重定向占位页和 vLLM 429 限流三类问题；这些问题保留在 `data/manifest/download_report.csv`，后续可作为回归测试样例。
