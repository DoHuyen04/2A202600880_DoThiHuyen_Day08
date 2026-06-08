"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25 (rank-bm25). Nếu dùng phương pháp khác (TF-IDF,
Elasticsearch, Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi
demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Lưu ý corpus: dùng CHÍNH các chunk đã tạo ở Task 4 (chunk_documents) để khớp
với vector store → semantic và lexical search cùng chấm điểm trên một tập chunk,
thuận tiện cho hybrid/RRF ở Task 9.

Tokenization tiếng Việt: ở đây dùng tách theo khoảng trắng + chuẩn hóa (lower,
bỏ dấu câu/ký hiệu markdown). BM25 so khớp theo âm tiết vẫn hiệu quả vì query và
document chia sẻ cùng âm tiết. Có thể nâng cấp bằng pyvi/underthesea (word
segmentation) nếu cài thêm.
"""

import re

# Cache corpus + bm25 index (xây 1 lần, tái dùng cho nhiều lần gọi).
_corpus: list[dict] | None = None
_bm25 = None

# bỏ ký hiệu markdown/dấu câu, giữ chữ-số (kể cả tiếng Việt có dấu).
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Chuẩn hóa + tách token: lower, giữ \\w (chữ/số), bỏ dấu câu & markdown."""
    return _TOKEN_RE.findall(text.lower())


def _load_corpus() -> list[dict]:
    """Lấy corpus chunk từ Task 4 (load_documents → chunk_documents)."""
    from src.task4_chunking_indexing import chunk_documents, load_documents

    return chunk_documents(load_documents())


def build_bm25_index(corpus: list[dict]):
    """Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        Đối tượng BM25Okapi đã fit trên corpus đã tokenize.
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)  # k1=1.5, b=0.75 mặc định


def _get_index():
    """Khởi tạo (lười) corpus + BM25 index và cache lại."""
    global _corpus, _bm25
    if _bm25 is None:
        _corpus = _load_corpus()
        _bm25 = build_bm25_index(_corpus)
    return _corpus, _bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Tìm kiếm từ khóa bằng BM25.

    Args:
        query: Câu truy vấn.
        top_k: Số lượng kết quả tối đa.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict},
        sorted theo BM25 score giảm dần (chỉ trả các chunk có score > 0).
    """
    corpus, bm25 = _get_index()

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    # lấy chỉ số sắp xếp giảm dần theo score
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for i in ranked[:top_k]:
        if scores[i] <= 0:
            break  # đã sort giảm dần → còn lại đều = 0, dừng sớm
        m = corpus[i]["metadata"]
        results.append(
            {
                "content": corpus[i]["content"],
                "score": float(scores[i]),
                # chuẩn hóa metadata trùng schema với semantic_search (Task 5)
                # để Task 9 có thể merge/RRF đồng nhất.
                "metadata": {
                    "source": m.get("source", ""),
                    "doc_type": m.get("type", ""),
                    "heading": m.get("heading", ""),
                    "chunk_index": m.get("chunk_index"),
                },
            }
        )
    return results


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for q in [
        "Điều 249 tàng trữ trái phép chất ma túy",
        "ca sĩ Long Nhật bị bắt",
    ]:
        print("=" * 70)
        print("Query:", q)
        for r in lexical_search(q, top_k=3):
            print(f"  [bm25={r['score']:.3f}] ({r['metadata']['doc_type']}) "
                  f"{r['metadata']['heading'][:55]}")
            print("     ", r["content"][:110].replace("\n", " "), "...")
