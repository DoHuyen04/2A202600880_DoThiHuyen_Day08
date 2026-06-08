"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

# Phạm vi kiến thức của hệ thống — dùng để trả lời khéo khi câu hỏi ngoài phạm vi.
KB_SCOPE = (
    "- Luật Phòng, chống ma túy 2021; Nghị định 105/2021; Nghị định 73/2018 "
    "(danh mục chất ma túy & tiền chất)\n"
    "- Bộ luật Hình sự — Chương XX: các tội phạm về ma túy (Điều 247–256)\n"
    "- Tin tức về các nghệ sĩ Việt Nam liên quan đến ma túy"
)
EXAMPLE_QUESTIONS = [
    "Tội mua bán trái phép chất ma túy bị phạt bao nhiêu năm tù?",
    "Luật Phòng, chống ma túy 2021 quy định thế nào về cai nghiện bắt buộc?",
    "Những nghệ sĩ nào gần đây bị bắt vì liên quan đến ma túy?",
]

SYSTEM_PROMPT = f"""Bạn là trợ lý hỏi đáp về PHÁP LUẬT và TIN TỨC liên quan đến
ma túy ở Việt Nam. Chỉ trả lời dựa trên phần CONTEXT được cung cấp, trả lời bằng
tiếng Việt, trình bày rõ ràng, thân thiện.

QUY TẮC TRÍCH DẪN:
- Mỗi khẳng định/sự kiện phải kèm citation trong ngoặc vuông, ví dụ
  [Bo-luat-hinh-su-2015, Điều 251] hoặc [VnExpress, 2024].
- Tuyệt đối KHÔNG bịa thông tin hoặc bịa citation.

KHI CONTEXT KHÔNG LIÊN QUAN hoặc KHÔNG ĐỦ để trả lời câu hỏi:
- KHÔNG cố trả lời bừa, KHÔNG dùng kiến thức ngoài context.
- Trả lời theo HƯỚNG GIÚP ĐỠ, gồm 3 ý:
  1) Nói rõ, lịch sự rằng thông tin cụ thể này chưa có trong nguồn tài liệu hiện có.
  2) Tóm tắt ngắn gọn phạm vi mà bạn CÓ THỂ hỗ trợ:
{KB_SCOPE}
  3) Gợi ý 2–3 câu hỏi liên quan mà bạn trả lời được, hoặc mời người dùng diễn đạt lại.
- Nếu context chỉ liên quan MỘT PHẦN: hãy trả lời phần có căn cứ (kèm citation),
  rồi nói rõ phần nào còn thiếu trong nguồn.

Tránh câu trả lời cụt lủn kiểu 'không tìm thấy'; luôn hướng người dùng tới điều
họ có thể hỏi tiếp."""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return list(chunks)

    # Giả định chunks đã sort theo score giảm dần (từ retrieval/rerank).
    # Lấy các vị trí chẵn (0,2,4,...) cho nửa đầu, vị trí lẻ (1,3,...) đảo
    # ngược cho nửa cuối → quan trọng nhất ở ĐẦU và CUỐI, kém nhất vào GIỮA.
    # Vd: [0,1,2,3,4] -> [0,2,4] + reversed([1,3]) = [0,2,4,3,1]
    left = chunks[0::2]
    right = chunks[1::2][::-1]
    return left + right


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {}) or {}
        # nguồn: legal/news dùng 'source'; PageIndex dùng 'title'
        source = meta.get("source") or meta.get("title") or f"Source {i}"
        doc_type = meta.get("doc_type") or meta.get("type") or "unknown"
        heading = meta.get("heading", "")
        label = f"[Document {i} | Nguồn: {source}"
        if doc_type and doc_type != "unknown":
            label += f" | Loại: {doc_type}"
        if heading:
            label += f" | Mục: {heading}"
        label += "]"
        context_parts.append(f"{label}\n{chunk['content']}\n")
    return "\n---\n".join(context_parts)


_METHOD_LABELS = {
    "structural": "Structured retrieval — lọc metadata theo Điều/Danh mục",
    "hybrid": "Hybrid — Semantic (vector) + BM25, hợp nhất RRF",
    "pageindex": "PageIndex — vectorless (duyệt cây tài liệu)",
}


def _build_technique(chunks: list[dict]) -> dict:
    """Tổng hợp các kỹ thuật đã dùng để tạo câu trả lời (hiển thị lên UI)."""
    methods = [c.get("method") for c in chunks if c.get("method")]
    reranks = [c.get("rerank_method") for c in chunks if c.get("rerank_method")]
    used = [_METHOD_LABELS[m] for m in ("structural", "hybrid", "pageindex") if m in methods]
    return {
        "retrieval": " + ".join(used) if used else "—",
        "rerank": reranks[0] if reranks else "none",
        "embedding": "OpenAI text-embedding-3-small (1536d)",
        "vector_store": "Weaviate (collection DrugDocs)",
        "llm": "gpt-4o-mini (temp=0.3, top_p=0.9)",
    }


def _out_of_scope_answer() -> str:
    """Câu trả lời thông minh khi retrieval không trả về nguồn nào."""
    examples = "\n".join(f"- {q}" for q in EXAMPLE_QUESTIONS)
    return (
        "Mình chưa tìm thấy thông tin phù hợp cho câu hỏi này trong nguồn tài liệu "
        "hiện có, nên xin phép không suy đoán để tránh đưa thông tin sai.\n\n"
        f"**Phạm vi mình có thể hỗ trợ:**\n{KB_SCOPE}\n\n"
        f"**Bạn có thể thử hỏi:**\n{examples}\n\n"
        "Hoặc bạn diễn đạt lại câu hỏi theo hướng pháp luật/tin tức về ma túy nhé."
    )


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    from openai import OpenAI

    # Step 1: Retrieve (hybrid + rerank + fallback PageIndex từ Task 9).
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        # Không có nguồn nào → trả lời thông minh, hướng dẫn người dùng.
        return {
            "answer": _out_of_scope_answer(),
            "sources": [],
            "retrieval_source": "none",
            "grounded": False,
            "technique": {
                "retrieval": "Không tìm thấy nguồn phù hợp",
                "rerank": "none",
                "embedding": "OpenAI text-embedding-3-small (1536d)",
                "vector_store": "Weaviate (collection DrugDocs)",
                "llm": "gpt-4o-mini (temp=0.3, top_p=0.9)",
            },
        }

    # Step 2: Reorder chống lost-in-the-middle.
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context kèm nhãn nguồn để LLM trích dẫn.
    context = format_context(reordered)

    # Step 4: Prompt = system + (context + câu hỏi).
    user_message = f"Context:\n{context}\n\n---\n\nCâu hỏi: {query}"

    # Step 5: Gọi LLM.
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,  # 0.3: ưu tiên factual, ít bịa
        top_p=TOP_P,              # 0.9: nucleus sampling vừa đủ tự nhiên
    )
    answer = response.choices[0].message.content

    # Step 6: Trả answer + nguồn + kỹ thuật đã dùng (để hiển thị lên UI).
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "grounded": True,
        "technique": _build_technique(chunks),
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
