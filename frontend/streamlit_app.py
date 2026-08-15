"""Streamlit chat UI for the OpsPilot FastAPI service.

Run the API first:
    uvicorn app.main:app --reload
Then run this page:
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from frontend.api_client import OpsPilotAPIClient

st.set_page_config(page_title="OpsPilot", page_icon="🛠️", layout="wide")
st.title("OpsPilot 大模型部署故障诊断")
st.caption("回答仅依据检索证据；证据不足时会拒答或要求补充信息。")

client = OpsPilotAPIClient(os.getenv("OPSPILOT_API_URL", "http://127.0.0.1:8000"))
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for source in message.get("sources", []):
            with st.expander(
                f"{source['citation_id']} · {source['title']} · "
                f"score={source['score']:.3f}"
            ):
                st.write(source["text"])
                st.link_button("打开原始文档", source["source_url"])

question = st.chat_input("描述错误日志、版本、运行命令和期望结果")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    answer = ""
    sources: list[dict[str, Any]] = []
    request_id = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            for event in client.stream_query(question):
                if event["event"] == "meta":
                    request_id = event["data"]["request_id"]
                elif event["event"] == "token":
                    answer += event["data"]["delta"]
                    placeholder.markdown(answer + "▌")
                elif event["event"] == "sources":
                    sources = event["data"]
            placeholder.markdown(answer)
            for source in sources:
                with st.expander(
                    f"{source['citation_id']} · {source['title']} · "
                    f"score={source['score']:.3f}"
                ):
                    st.write(source["text"])
                    st.link_button("打开原始文档", source["source_url"])
        except Exception as exc:
            st.error(f"请求失败：{exc}")

    if answer:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "request_id": request_id,
            }
        )

if st.session_state.messages:
    latest = st.session_state.messages[-1]
    if latest["role"] == "assistant" and latest.get("request_id"):
        st.divider()
        st.write("这个回答有帮助吗？")
        left, right = st.columns(2)
        if left.button("👍 有帮助", use_container_width=True):
            client.send_feedback(latest["request_id"], "up", "correct")
            st.success("感谢反馈。")
        if right.button("👎 需要改进", use_container_width=True):
            client.send_feedback(latest["request_id"], "down", "incorrect")
            st.info("反馈已记录；系统不会保存你的原始问题文本。")
