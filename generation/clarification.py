"""Query-only diagnostic context checks; no gold labels or retrieval scores."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticContext:
    issue: str
    missing_slots: tuple[str, ...]
    questions: tuple[str, ...]

    @property
    def clarification(self) -> str:
        return "为了可靠诊断，请补充：" + "；".join(self.questions) + "。"


SYMPTOM = re.compile(
    r"失败|访问不了|连不上|不能用|不可用|有问题|报错|异常|很慢|太慢|不准|不相关|超时|崩溃|"
    r"启动不了|无法启动|没有响应|没响应|加载不了",
    re.I,
)
OPS_TOPIC = re.compile(
    r"docker|cuda|qdrant|fastapi|gpu|pytorch|vllm|ollama|模型|推理|服务|容器|驱动", re.I
)
PROCEDURE = re.compile(
    r"(?:如何|怎么|怎样|应如何)(?:区分|结合|从|同时检查)|分层排查|两层定位|"
    r"(?:什么|哪些).{0,8}(?:日志|指标|信息)|(?:日志|指标|信息).{0,8}(?:什么|哪些)|"
    r"含义|作用|什么意思|有什么区别",
    re.I,
)
DEFINITE_DIAGNOSIS = re.compile(r"确定|根本原因|根因|是不是|(?:改|调整)哪个参数", re.I)
ERROR_DETAIL = re.compile(
    r"traceback|(?:runtimeerror|valueerror|oserror)\s*[:：]|cuda out of memory|"
    r"unsupported toolchain|connection refused|(?:exit code|退出码|错误码)\s*[:：=]?\s*\d+|"
    r"(?:错误信息|错误日志|报错|日志)\s*[:：=]\s*\S.{5,}",
    re.I,
)
ENVIRONMENT = re.compile(
    r"(?:cuda|pytorch|torch|docker|ubuntu|windows|驱动|版本)\s*[:：=]?\s*\d+(?:\.\d+)*|"
    r"rtx\s*\d{4}|\b(?:a100|h100|v100)\b|docker\s+(?:run|compose)",
    re.I,
)
METRICS = re.compile(
    r"(?:p50|p95|ttft|tpot|qps|并发|吞吐)\s*[:：=]?\s*\d+|\d+(?:\.\d+)?\s*(?:ms|毫秒|秒|qps|rps)",
    re.I,
)
WORKLOAD = re.compile(
    r"(?:workers?|batch(?:_size)?|num_ctx|并发|模型|精度|上下文)\s*[:：=]\s*\S+|"
    r"\b(?:fp16|bf16|int8|q4|7b|1\.5b)\b",
    re.I,
)
RETRIEVAL_EXAMPLE = re.compile(r"(?:query|查询样例|问题样例|检索词)\s*[:：=]\s*\S+", re.I)
RETRIEVAL_CONFIG = re.compile(
    r"(?:embedding|向量维度|distance|距离指标|top[-_ ]?k)\s*[:：=]\s*\S+|\b(?:bge|e5|cosine|euclid)\b",
    re.I,
)
NETWORK_CONFIG = re.compile(
    r"docker\s+(?:run|compose)|(?:-p|--publish)\s*\d+|\d+\.\d+\.\d+\.\d+:\d+", re.I
)


def missing_diagnostic_context(question: str) -> DiagnosticContext | None:
    """Check targeted diagnosis, while allowing explanations and scoped procedures.

    This is a first rule-based implementation, not a semantic completeness proof.
    Mentioning a version/log keyword alone does not establish a populated slot.
    """
    clean = question.strip()
    if not OPS_TOPIC.search(clean) or not SYMPTOM.search(clean):
        return None
    if PROCEDURE.search(clean) and not DEFINITE_DIAGNOSIS.search(clean):
        return None
    if re.search(r"检索|召回|排序", clean):
        issue = "retrieval_quality"
        slots = [
            (
                "retrieval_example",
                RETRIEVAL_EXAMPLE.search(clean),
                "具体查询、当前Top-k结果和期望命中的文档",
            ),
            (
                "retrieval_config",
                RETRIEVAL_CONFIG.search(clean),
                "Embedding模型、向量维度和距离指标",
            ),
        ]
    elif re.search(r"很慢|太慢|超时|吞吐|延迟", clean):
        issue = "performance"
        slots = [
            ("measurements", METRICS.search(clean), "并发量、P50/P95或分阶段耗时的实测值"),
            (
                "workload",
                WORKLOAD.search(clean),
                "模型/精度、batch或worker配置，以及CPU/GPU和网络指标",
            ),
        ]
    elif re.search(r"访问不了|连不上|无法访问", clean):
        issue = "network_access"
        slots = [
            (
                "network_config",
                NETWORK_CONFIG.search(clean),
                "启动命令、端口映射和进程实际监听地址",
            ),
            ("error_detail", ERROR_DETAIL.search(clean), "具体访问命令与完整错误信息"),
        ]
    else:
        issue = "runtime_failure"
        slots = [
            ("error_detail", ERROR_DETAIL.search(clean), "完整错误信息或错误日志"),
            ("environment", ENVIRONMENT.search(clean), "软件版本、GPU/驱动信息与实际运行命令"),
        ]
    missing = [(key, text) for key, present, text in slots if not present]
    if not missing:
        return None
    return DiagnosticContext(issue, tuple(x[0] for x in missing), tuple(x[1] for x in missing))
