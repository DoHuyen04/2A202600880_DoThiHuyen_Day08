"""
RAG Chatbot — Pháp luật & Tin tức về ma túy (Streamlit).

Tính năng:
    - Chat đa lượt với conversation memory.
    - Conversation memory qua "query condensation": câu hỏi follow-up (vd "hình
      phạt của nó?") được viết lại thành câu hỏi ĐỘC LẬP dựa trên lịch sử hội
      thoại trước khi đưa vào retrieval → retrieval hiểu đúng ngữ cảnh.
    - Trả lời có CITATION, hiển thị nguồn (chunk) + điểm + nhánh retrieval.

Chạy:
    streamlit run group_project/app.py
"""

import sys
from pathlib import Path

# cho phép import package src/ khi chạy bằng streamlit
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from openai import OpenAI

from src.task10_generation import generate_with_citation

st.set_page_config(page_title="RAG Chatbot — Ma túy & Pháp luật", page_icon="⚖️", layout="centered")

CONDENSE_MODEL = "gpt-4o-mini"


@st.cache_resource
def _openai_client() -> OpenAI:
    return OpenAI()


def condense_question(history: list[dict], question: str) -> str:
    """Viết lại câu hỏi follow-up thành câu độc lập dựa trên lịch sử hội thoại."""
    if not history:
        return question
    convo = "\n".join(
        f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content']}"
        for m in history[-6:]  # 3 lượt gần nhất
    )
    try:
        resp = _openai_client().chat.completions.create(
            model=CONDENSE_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn viết lại câu hỏi cuối của người dùng thành một câu hỏi "
                        "ĐỘC LẬP, đầy đủ ngữ cảnh dựa trên lịch sử hội thoại (thay các "
                        "đại từ như 'nó', 'việc đó' bằng đối tượng cụ thể). Chỉ trả về "
                        "đúng câu hỏi tiếng Việt, không giải thích."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Hội thoại:\n{convo}\n\nCâu hỏi mới: {question}\n\nCâu hỏi độc lập:",
                },
            ],
        )
        return resp.choices[0].message.content.strip() or question
    except Exception:  # noqa: BLE001 - lỗi condense thì dùng câu gốc
        return question


def render_technique(tech: dict):
    """Hiển thị kỹ thuật đã dùng để tạo câu trả lời."""
    if not tech:
        return
    st.caption(
        f"🔧 **Kỹ thuật tìm kiếm:** {tech.get('retrieval', '—')}  ·  "
        f"**Rerank:** {tech.get('rerank', 'none')}"
    )
    with st.expander("🔧 Chi tiết kỹ thuật"):
        st.markdown(
            f"- **Retrieval:** {tech.get('retrieval', '—')}\n"
            f"- **Reranking:** {tech.get('rerank', 'none')}\n"
            f"- **Embedding:** {tech.get('embedding', '—')}\n"
            f"- **Vector store:** {tech.get('vector_store', '—')}\n"
            f"- **LLM sinh câu trả lời:** {tech.get('llm', '—')}"
        )


def render_sources(sources: list, retrieval_source: str):
    """Hiển thị danh sách nguồn (chunk) đã dùng."""
    with st.expander(f"📚 {len(sources)} nguồn · via {retrieval_source}"):
        for i, s in enumerate(sources, 1):
            meta = s.get("metadata", {})
            label = meta.get("source") or meta.get("title") or f"Nguồn {i}"
            heading = meta.get("heading", "")
            method = s.get("method", "")
            tag = f" · `{method}`" if method else ""
            st.markdown(f"**{i}. {label}** · score `{s.get('score',0):.3f}`{tag}")
            if heading:
                st.caption(heading)
            st.write(s["content"][:400] + ("…" if len(s["content"]) > 400 else ""))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Cấu hình")
    top_k = st.slider("Số chunk đưa vào context (top_k)", 3, 8, 5)
    score_threshold = st.slider(
        "Ngưỡng fallback PageIndex", 0.0, 1.0, 0.3, 0.05,
        help="Nếu điểm hybrid cao nhất < ngưỡng → chuyển sang PageIndex vectorless.",
    )
    show_sources = st.checkbox("Hiển thị nguồn (sources)", value=True)
    use_memory = st.checkbox("Bật conversation memory", value=True)
    st.divider()
    if st.button("🗑️ Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(
        "Corpus: Luật PCMT 2021, NĐ 105/2021, NĐ 73/2018 (danh mục chất), "
        "BLHS Chương XX, và các bài báo về nghệ sĩ liên quan ma túy."
    )


# --------------------------------------------------------------------------- #
# Header + lịch sử
# --------------------------------------------------------------------------- #
st.title("⚖️ RAG Chatbot — Ma túy & Pháp luật Việt Nam")
st.caption("Hỏi đáp pháp luật và tin tức về ma túy, có trích dẫn nguồn.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# render lịch sử
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("technique"):
            render_technique(msg["technique"])
        if msg.get("sources") and show_sources:
            render_sources(msg["sources"], msg.get("retrieval_source", "?"))


# --------------------------------------------------------------------------- #
# Input + xử lý
# --------------------------------------------------------------------------- #
if prompt := st.chat_input("Nhập câu hỏi của bạn…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy hồi và soạn câu trả lời…"):
            history = st.session_state.messages[:-1]
            standalone = (
                condense_question(history, prompt) if use_memory else prompt
            )
            if use_memory and standalone.strip() != prompt.strip():
                st.caption(f"🔎 Truy vấn độc lập: *{standalone}*")

            result = generate_with_citation(standalone, top_k=top_k)
            answer = result["answer"]
            st.markdown(answer)
            render_technique(result.get("technique"))
            if show_sources and result["sources"]:
                render_sources(result["sources"], result["retrieval_source"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
            "technique": result.get("technique"),
        }
    )
