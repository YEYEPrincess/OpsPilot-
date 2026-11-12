"""Community Cloud entrypoint: deploy this file, not the local-only frontend."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from deploy.community_cloud.runtime import CloudRuntime  # noqa: E402
from frontend.api_client import OpsPilotAPIClient  # noqa: E402

st.set_page_config(page_title="OpsPilot 在线体验", page_icon="🛠️", layout="wide")
st.title("OpsPilot · 大模型部署故障问答")
st.caption("查阅官方技术文档，给出有引用的排查建议；信息不足时追问，证据不足时拒答。")
st.info("公开体验版，请勿提交密码、密钥或含个人信息的日志。"
        "问题和检索片段会发送给 DeepSeek。免费托管休眠后，首次打开需等待唤醒。")


@st.cache_resource(max_entries=1)
def get_runtime(api_key: str, model: str):
    return CloudRuntime(api_key, model)


try:
    api_key = str(st.secrets.get("LLM_API_KEY", ""))
    model = str(st.secrets.get("LLM_MODEL", "deepseek-flash"))
except FileNotFoundError:
    api_key, model = "", "deepseek-flash"
if not api_key:
    st.warning("应用尚未完成模型服务配置，请稍后再试。")
    st.stop()
try:
    with st.spinner("正在准备文档索引……"):
        runtime = get_runtime(api_key, model)
except Exception:
    st.error("应用暂时无法启动，请稍后重试或联系维护者。")
    st.stop()
client = OpsPilotAPIClient(runtime.url, timeout_seconds=65)

with st.expander("示例问题与体验范围"):
    st.write("可从 `docker logs`、`torch.cuda.is_available()` 开始，或提供具体报错与环境。")
    st.write("该轻量版本使用 43 份文档的 1,042 个片段和基础检索，未启用 BGE 重排。"
             "引用格式通过不保证诊断一定正确，请先核对原文。")
    st.write("全站每天最多尝试 50 次模型生成，每分钟 4 次，同时最多 2 次。"
             "命中答案缓存、直接追问或拒答不消耗生成次数；额度不足时暂时停止生成。")

messages = st.session_state.setdefault("messages", [])


def show_sources(sources):
    for source in sources:
        with st.expander(f"{source['citation_id']} · {source.get('title') or '原始文档'}"):
            st.write(source["text"])
            if source.get("source_url", "").startswith(("https://", "http://")):
                st.link_button("核对原文", source["source_url"])


for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))

question = st.chat_input("输入问题、报错和环境信息（请移除敏感内容）", max_chars=2000)
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        notice = st.empty()
        notice.caption("正在检索与生成，流式内容尚未完成引用检查。")
        partial, result, done = "", None, False
        try:
            for event in client.stream_query(question, top_k=3):
                if event["event"] == "token":
                    partial += event["data"]["delta"]
                    placeholder.markdown(partial + "▌")
                elif event["event"] == "result":
                    result = event["data"]
                elif event["event"] == "done":
                    done = True
                elif event["event"] == "error":
                    raise RuntimeError("Incomplete response")
            if not done or result is None:
                raise RuntimeError("Incomplete response")
            quota_messages = {
                "PUBLIC_DAILY_LIMIT": "今天的全站体验额度已用完，请明天再试。",
                "PUBLIC_MINUTE_LIMIT": "当前体验请求较多，请一分钟后再试。",
                "PUBLIC_BUSY": "当前有其他问题正在生成，请稍后再试。",
            }
            answer = quota_messages.get(result.get("error_code"), result["answer"])
            if result.get("clarification"):
                answer += "\n\n" + result["clarification"]
            placeholder.markdown(answer)
            notice.empty()
            show_sources(result["sources"])
            st.caption(f"请求编号：{result['request_id']}")
            messages.append({"role": "assistant", "content": answer,
                             "sources": result["sources"]})
        except Exception:
            placeholder.empty()
            notice.empty()
            st.error("请求未完成，已丢弃未校验的文本，请稍后重试。")
    # Bound per-browser memory; conversation history is not sent to the model.
    st.session_state.messages = messages[-12:]
