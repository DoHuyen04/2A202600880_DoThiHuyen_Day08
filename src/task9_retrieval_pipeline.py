"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

import re
import sys

from src.task5_semantic_search import fetch_by_heading, semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank
from src.task8_pageindex_vectorless import pageindex_search

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RRF_K = 60  # hằng số làm mượt Reciprocal Rank Fusion

# Phát hiện tham chiếu đơn vị cấu trúc trong câu hỏi → lấy chunk theo metadata.
_RE_DANHMUC_REF = re.compile(r"(?:danh\s*mục|nhóm)\s+(IV|III|II|I)\b", re.IGNORECASE)
_RE_DIEU_REF = re.compile(r"điều\s+(\d{1,3})\b", re.IGNORECASE)


def structural_fetch(query: str, limit: int = 12) -> list[dict]:
    """Nếu câu hỏi nhắc 'Danh mục/nhóm X' hoặc 'Điều N' → lấy thẳng chunk đó.

    Khắc phục điểm yếu của semantic/BM25 với câu hỏi liệt kê trên bảng (vd danh
    mục chất ma túy): khớp chính xác heading thay vì dựa vào độ tương đồng.
    """
    out: list[dict] = []
    try:
        m = _RE_DANHMUC_REF.search(query)
        if m:
            out += fetch_by_heading(f"*DANH MỤC {m.group(1).upper()} —*", limit=limit)
        for num in set(_RE_DIEU_REF.findall(query)):
            out += fetch_by_heading(f"*Điều {num}.*", limit=limit)
    except Exception as exc:  # noqa: BLE001 - Weaviate có thể lỗi
        print(f"  ⚠ structural_fetch lỗi: {exc}")
    return out


def hybrid_merge(query: str, pool: int) -> list[dict]:
    """Hybrid = semantic (Task 5) + lexical/BM25 (Task 6), hợp nhất bằng RRF.

    Vì cosine (~0..1) và BM25 (~0..25) khác thang đo, ta gộp theo THỨ HẠNG:
        RRF(d) = Σ_r 1/(RRF_K + rank_r(d))
    Tài liệu xuất hiện ở cả hai danh sách được cộng dồn → ưu tiên. Khóa gộp là
    nội dung chunk (ổn định & duy nhất).
    """
    try:
        sem = semantic_search(query, top_k=pool)
    except Exception as exc:  # noqa: BLE001 - Weaviate/embedding có thể lỗi
        print(f"  ⚠ semantic_search lỗi: {exc}")
        sem = []
    try:
        lex = lexical_search(query, top_k=pool)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ lexical_search lỗi: {exc}")
        lex = []

    fused: dict[str, dict] = {}
    for results in (sem, lex):
        for rank, r in enumerate(results, start=1):
            key = r["content"]
            entry = fused.get(key)
            if entry is None:
                entry = {"content": r["content"], "metadata": r["metadata"], "score": 0.0}
                fused[key] = entry
            entry["score"] += 1.0 / (RRF_K + rank)

    return sorted(fused.values(), key=lambda e: e["score"], reverse=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    pool = max(top_k * 4, 20)  # số ứng viên lấy từ mỗi ranker trước khi rerank

    # Step 0: Truy hồi theo CẤU TRÚC nếu câu hỏi nhắc Danh mục/Điều cụ thể.
    # Đây là khớp chính xác heading → luôn đáng tin, đặt lên ĐẦU kết quả.
    structural = structural_fetch(query)
    for s in structural:
        s["source"] = "hybrid"
        s["method"] = "structural"  # lọc theo metadata heading (Điều/Danh mục)

    # Step 1 + 2: semantic + lexical, merge bằng RRF.
    merged = hybrid_merge(query, pool)

    # Step 3: Rerank bằng cross-encoder (Task 7). Gắn nguồn = 'hybrid'.
    if merged:
        final_results = (
            rerank(query, merged, top_k=top_k) if use_reranking else merged[:top_k]
        )
        for item in final_results:
            item["source"] = "hybrid"
            item["method"] = "hybrid"  # semantic + BM25, hợp nhất RRF (+ rerank)
    else:
        final_results = []

    # Có chunk khớp cấu trúc → ưu tiên, gộp với hybrid (dedup), bỏ qua fallback.
    if structural:
        seen, combined = set(), []
        for r in structural + final_results:
            if r["content"] not in seen:
                seen.add(r["content"])
                combined.append(r)
        return combined[:top_k]

    best_score = final_results[0]["score"] if final_results else 0.0

    # Step 4: Hybrid đủ tốt → trả luôn.
    if final_results and best_score >= score_threshold:
        return final_results[:top_k]

    # Ngược lại (yếu/không có) → fallback PageIndex vectorless.
    print(
        f"  ⚠ Hybrid score ({best_score:.3f}) < threshold ({score_threshold}). "
        "Fallback → PageIndex"
    )
    try:
        fallback = pageindex_search(query, top_k=top_k)  # đã gắn source='pageindex'
        for r in fallback:
            r["method"] = "pageindex"  # vectorless (duyệt cây tài liệu)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ pageindex_search lỗi: {exc}")
        fallback = []

    # Fallback có kết quả → dùng; nếu rỗng/lỗi → trả hybrid (dù dưới ngưỡng).
    return (fallback or final_results)[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
