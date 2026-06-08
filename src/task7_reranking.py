"""
Task 7 — Reranking Module.

Phương pháp chính: Cross-encoder reranker qua Jina Reranker v2 API
    (model jina-reranker-v2-base-multilingual) — tốt cho tiếng Việt, không cần
    tải model nặng. Cross-encoder đọc ĐỒNG THỜI (query, document) trong một lần
    forward nên đo độ liên quan chính xác hơn nhiều so với bi-encoder (embedding
    riêng rồi tính cosine ở Task 5) → dùng để "chấm lại" top ứng viên retrieval.

Fallback offline: nếu không gọi được API (mất mạng / hết quota), tự động chuyển
    sang re-score bằng độ trùng lặp từ khóa (lexical overlap) để pipeline/tests
    không bị gãy. Fallback là phương án dự phòng, không thay thế cross-encoder.

Cấu hình:
    JINA_API_KEY trong .env (lấy tại https://jina.ai/reranker/).
"""

import os
import re

from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_URL = "https://api.jina.ai/v1/rerank"
JINA_MODEL = "jina-reranker-v2-base-multilingual"

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Rerank bằng Jina Reranker v2 (cross-encoder, multilingual).

    Returns:
        top_k candidates, mỗi item được gán lại 'score' = relevance_score của
        cross-encoder, sorted giảm dần. Giữ nguyên content/metadata.

    Raises:
        Lỗi mạng/HTTP để caller (rerank) quyết định fallback.
    """
    import requests

    documents = [c["content"] for c in candidates]
    resp = requests.post(
        JINA_URL,
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": JINA_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()["results"]  # [{index, relevance_score, document}, ...]

    reranked = []
    for item in data:
        cand = candidates[item["index"]]
        reranked.append(
            {
                "content": cand["content"],
                "score": float(item["relevance_score"]),
                "metadata": cand.get("metadata", {}),
            }
        )
    # API đã sort theo relevance_score, nhưng sort lại cho chắc.
    reranked.sort(key=lambda r: r["score"], reverse=True)
    return reranked[:top_k]


def rerank_lexical_overlap(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Fallback offline: chấm lại theo tỉ lệ token query trùng trong document.

    score = |tokens(query) ∩ tokens(doc)| / |tokens(query)|  ∈ [0, 1]
    """
    q_tokens = _tokens(query)
    scored = []
    for c in candidates:
        overlap = len(q_tokens & _tokens(c["content"]))
        score = overlap / len(q_tokens) if q_tokens else 0.0
        scored.append(
            {
                "content": c["content"],
                "score": float(score),
                "metadata": c.get("metadata", {}),
            }
        )
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k]


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Re-score & re-order candidates theo độ liên quan với query.

    Dùng Jina cross-encoder; nếu lỗi (no key / mạng / quota) → fallback lexical.

    Args:
        query: Câu truy vấn.
        candidates: List of {'content', 'score', 'metadata'} từ retrieval.
        top_k: Số kết quả sau rerank.

    Returns:
        List[dict] {'content', 'score', 'metadata'} sorted giảm dần, tối đa top_k.
    """
    if not candidates:
        return []

    if JINA_API_KEY and "xxx" not in JINA_API_KEY:
        try:
            results = rerank_cross_encoder(query, candidates, top_k)
            for r in results:
                r["rerank_method"] = "Jina cross-encoder v2 (multilingual)"
            return results
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ Jina rerank lỗi ({exc}); fallback lexical-overlap")

    results = rerank_lexical_overlap(query, candidates, top_k)
    for r in results:
        r["rerank_method"] = "lexical-overlap (fallback)"
    return results


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Demo: lấy ứng viên từ hybrid search rồi rerank lại.
    try:
        from src.hybrid_search import hybrid_search

        query = "Hình phạt tội mua bán trái phép chất ma túy"
        candidates = hybrid_search(query, top_k=10)
        print("Query:", query)
        print("\n--- Trước rerank (RRF) ---")
        for r in candidates[:5]:
            print(f"  [{r['score']:.4f}] {r['metadata'].get('heading','')[:55]}")
        print("\n--- Sau rerank (Jina cross-encoder) ---")
        for r in rerank(query, candidates, top_k=5):
            print(f"  [{r['score']:.4f}] {r['metadata'].get('heading','')[:55]}")
    except Exception as exc:  # noqa: BLE001
        print("Demo cần Weaviate/online. Lỗi:", exc)
