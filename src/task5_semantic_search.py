"""
Task 5 — Semantic Search Module (dense retrieval).

Tìm kiếm ngữ nghĩa trên Weaviate đã index ở Task 4:
    1. Embed query bằng CÙNG model với Task 4 (OpenAI text-embedding-3-small).
    2. Truy vấn near_vector trên collection 'DrugDocs' (cosine).
    3. Trả về top_k chunk kèm score (similarity = 1 - distance), sorted desc.

Yêu cầu:
    - Input: query string + top_k
    - Output: list[{'content': str, 'score': float, 'metadata': dict}]
"""

import atexit

from src.task4_chunking_indexing import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    get_weaviate_client,
)

# Cache client (Weaviate + OpenAI) để gọi nhiều lần không phải kết nối lại.
_wv_client = None
_oai_client = None


def _get_clients():
    global _wv_client, _oai_client
    if _wv_client is None:
        from openai import OpenAI

        _wv_client = get_weaviate_client()
        _oai_client = OpenAI()
        atexit.register(_close_clients)
    return _wv_client, _oai_client


def _close_clients():
    global _wv_client
    if _wv_client is not None:
        try:
            _wv_client.close()
        except Exception:  # noqa: BLE001
            pass
        _wv_client = None


def _embed_query(oai, query: str) -> list[float]:
    return oai.embeddings.create(model=EMBEDDING_MODEL, input=[query]).data[0].embedding


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Dense retrieval bằng vector similarity trên Weaviate.

    Args:
        query: Câu truy vấn.
        top_k: Số lượng kết quả tối đa.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict},
        sorted theo score (cosine similarity) giảm dần.
    """
    from weaviate.classes.query import MetadataQuery

    wv, oai = _get_clients()
    collection = wv.collections.get(COLLECTION_NAME)

    query_vec = _embed_query(oai, query)
    response = collection.query.near_vector(
        near_vector=query_vec,
        limit=top_k,
        return_metadata=MetadataQuery(distance=True),
    )

    results = []
    for obj in response.objects:
        props = obj.properties
        distance = obj.metadata.distance
        # Weaviate trả cosine distance ∈ [0, 2]; similarity = 1 - distance.
        score = 1.0 - distance if distance is not None else 0.0
        results.append(
            {
                "content": props.get("content", ""),
                "score": score,
                "metadata": {
                    "source": props.get("source", ""),
                    "doc_type": props.get("doc_type", ""),
                    "heading": props.get("heading", ""),
                    "chunk_index": props.get("chunk_index"),
                },
            }
        )

    # near_vector vốn đã sort theo distance tăng dần (similarity giảm dần),
    # nhưng sort lại cho chắc chắn đúng hợp đồng API.
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def fetch_by_heading(like_pattern: str, limit: int = 12) -> list[dict]:
    """Lấy chunk theo FILTER trên metadata 'heading' (không dùng vector).

    Dùng cho truy vấn tham chiếu đơn vị cấu trúc cụ thể (vd "Danh mục I",
    "Điều 249") — nơi semantic/BM25 thường trượt vì chunk bảng toàn tên chất.
    `like_pattern` dùng wildcard '*' của Weaviate, vd "*DANH MỤC I —*".

    Returns: list[{'content','score','metadata'}] (score=1.0, sort theo chunk_index).
    """
    from weaviate.classes.query import Filter

    wv, _ = _get_clients()
    collection = wv.collections.get(COLLECTION_NAME)
    resp = collection.query.fetch_objects(
        filters=Filter.by_property("heading").like(like_pattern),
        limit=limit,
    )
    results = []
    for obj in resp.objects:
        p = obj.properties
        results.append(
            {
                "content": p.get("content", ""),
                "score": 1.0,  # khớp cấu trúc chính xác → coi như rất liên quan
                "metadata": {
                    "source": p.get("source", ""),
                    "doc_type": p.get("doc_type", ""),
                    "heading": p.get("heading", ""),
                    "chunk_index": p.get("chunk_index"),
                },
            }
        )
    results.sort(key=lambda r: r["metadata"].get("chunk_index") or 0)
    return results


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for q in [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy",
        "Nghệ sĩ nào bị bắt vì liên quan ma túy",
    ]:
        print("=" * 70)
        print("Query:", q)
        for r in semantic_search(q, top_k=3):
            print(f"  [sim={r['score']:.3f}] ({r['metadata']['doc_type']}) "
                  f"{r['metadata']['heading'][:55]}")
            print("     ", r["content"][:110].replace("\n", " "), "...")
