"""
Task 4 — Chunking & Indexing vào Vector Store.

Lựa chọn & lý do
================
1) Chunking strategy: langchain-text-splitters, KẾT HỢP 2 bước
   - MarkdownHeaderTextSplitter: tách theo heading (#, ##, ###). Tài liệu pháp
     luật có cấu trúc Chương/Điều rất rõ; tách theo heading giúp mỗi đoạn nằm
     trọn trong NGỮ CẢNH của một điều/khoản → chunk có ngữ nghĩa mạch lạc và
     giữ được "đường dẫn heading" (h1>h2>h3) làm metadata.
   - RecursiveCharacterTextSplitter: sau đó cắt nhỏ từng section quá dài về
     đúng CHUNK_SIZE (ranh giới đoạn → câu → từ) để không vượt giới hạn token
     của embedding và để retrieval trả về đoạn đủ nhỏ, đúng trọng tâm.
   → Bước 1 cho ngữ nghĩa, bước 2 cho kích thước. Đây là pattern chuẩn cho RAG
     trên văn bản có cấu trúc.

2) Embedding model: OpenAI text-embedding-3-small (1536 dim)
   - Chất lượng cao, hỗ trợ đa ngôn ngữ tốt (kể cả tiếng Việt), giá rẻ
     (~$0.02 / 1M token), không cần GPU. Phù hợp môi trường không có máy mạnh.

3) Vector store: Weaviate
   - Hỗ trợ hybrid search (dense + BM25) built-in → tái dùng cho Task 5/6/9.
   - Ở đây ta tự sinh vector (vectorizer = none) và đẩy vào Weaviate.
   - Kết nối: Weaviate Cloud (WEAVIATE_URL/WEAVIATE_API_KEY trong .env) hoặc
     Docker local (localhost:8080) nếu không cấu hình cloud.

Cài đặt:
    pip install langchain-text-splitters "weaviate-client>=4.5.0" openai python-dotenv tiktoken
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# Console Windows (cp1258) không in được ✓ và tiếng Việt → ép UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# =============================================================================
# CONFIGURATION
# =============================================================================
# 800 ký tự (~250-350 token tiếng Việt): đủ chứa trọn 1 điều/khoản hoặc vài
# đoạn báo, nhưng vẫn đủ nhỏ để retrieval chính xác và tiết kiệm context.
CHUNK_SIZE = 800
# 120 ký tự (~15%): bắc cầu ngữ cảnh giữa 2 chunk liền kề, tránh cắt đứt ý ở
# ranh giới chunk (đặc biệt với câu/điều khoản dài).
CHUNK_OVERLAP = 120
# Hybrid theo loại tài liệu:
#   - legal: structure-aware (MarkdownHeader theo Chương/Điều/Danh mục) — luật có
#            ranh giới ngữ nghĩa hiển thị sẵn nên tách theo cấu trúc chính xác hơn.
#   - news : SemanticChunker (embedding) — bài báo không có heading, tách theo chỗ
#            "đổi chủ đề" giúp mỗi chunk trọn một ý.
CHUNKING_METHOD = "hybrid: legal=markdown_header+recursive, news=semantic"
MIN_CHUNK_CHARS = 30  # bỏ mẩu chunk quá ngắn (rác, không đủ ngữ nghĩa)
# Ngưỡng tách của SemanticChunker: cắt khi khoảng cách embedding giữa các câu
# vượt percentile này → cao = ít điểm cắt = chunk to hơn.
SEMANTIC_BREAKPOINT_TYPE = "percentile"
SEMANTIC_BREAKPOINT_AMOUNT = 90

# OpenAI text-embedding-3-small: 1536 chiều.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBED_BATCH_SIZE = 100  # số chunk gửi mỗi request embedding

VECTOR_STORE = "weaviate"
COLLECTION_NAME = "DrugDocs"

# heading dùng để tách ở bước MarkdownHeaderTextSplitter
_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]


# =============================================================================
# 1) LOAD
# =============================================================================
def load_documents() -> list[dict]:
    """Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue
        doc_type = "legal" if md_file.parent.name == "legal" else "news"
        documents.append(
            {
                "content": content,
                "metadata": {"source": md_file.name, "type": doc_type},
            }
        )
    return documents


# =============================================================================
# 2) CHUNK
# =============================================================================
# Trong văn bản luật, "Chương"/"Điều" được in đậm chứ không phải heading
# markdown. Nâng chúng thành heading để splitter tách đúng theo từng điều khoản.
_RE_CHUONG = re.compile(r"^\*\*(Chương\s+[^\*]+?)\*\*\s*$", re.MULTILINE)
_RE_DIEU = re.compile(r"^\*\*(Điều\s+\d+[^\*]*?)\*\*\s*$", re.MULTILINE)
# DANH MỤC / PHỤ LỤC (vd nghị định danh mục chất ma túy): các bảng chất nằm
# dưới những header này, nếu không nâng thành heading thì chunk bảng sẽ bị gán
# nhầm vào Điều ngay trước đó → mất ngữ cảnh "thuộc Danh mục nào".
# DANH MỤC/PHỤ LỤC: gộp LUÔN dòng mô tả ý nghĩa ngay sau header vào heading,
# để mọi chunk con (bảng chất) đều mang ngữ cảnh "Danh mục I — các chất tuyệt
# đối cấm..." chứ không chỉ trơ tên hóa chất → truy vấn theo ý nghĩa khớp tốt hơn.
_RE_DANHMUC = re.compile(
    r"^\*\*((?:DANH MỤC|PHỤ LỤC)[^\*\n]*)\*\*[ \t]*\n+[ \t]*([^\n|*][^\n]*)",
    re.MULTILINE,
)

# Ngân sách ký tự dành cho dòng ngữ cảnh chèn đầu mỗi chunk (xem chunk_documents).
HEADING_BUDGET = 200


def _danhmuc_repl(m: "re.Match") -> str:
    title = m.group(2).strip().rstrip(".")
    return f"## {m.group(1).strip()} — {title}"


def _promote_legal_headings(text: str) -> str:
    """Nâng Chương/DANH MỤC -> '## ...' và Điều N. -> '### ...' (heading markdown)."""
    text = _RE_CHUONG.sub(r"## \1", text)
    text = _RE_DANHMUC.sub(_danhmuc_repl, text)
    text = _RE_DIEU.sub(r"### \1", text)
    return text


def _context_prefix(heading: str) -> str:
    """Dòng ngữ cảnh chèn đầu mỗi chunk (contextual chunking)."""
    return f"[Bối cảnh: {heading[:HEADING_BUDGET - 14]}]\n" if heading else ""


def _chunk_legal(doc, header_splitter, char_splitter) -> list[tuple[str, str]]:
    """Structure-aware: tách theo Chương/Điều/Danh mục rồi cắt nhỏ về kích thước.

    Returns list[(content, heading)].
    """
    content = _promote_legal_headings(doc["content"])
    out = []
    for section in header_splitter.split_text(content):
        heading = " > ".join(
            section.metadata[k] for k in ("h1", "h2", "h3") if section.metadata.get(k)
        )
        prefix = _context_prefix(heading)
        for piece in char_splitter.split_text(section.page_content):
            piece = piece.strip()
            if len(piece) >= MIN_CHUNK_CHARS:
                out.append((prefix + piece, heading))
    return out


def _news_title(content: str) -> str:
    """Lấy tiêu đề bài báo (dòng '# ...' đầu tiên) làm ngữ cảnh chunk."""
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _chunk_news(doc, semantic_chunker, char_splitter) -> list[tuple[str, str]]:
    """Semantic: tách bài báo theo chỗ 'đổi chủ đề' (embedding), rồi chặn kích thước.

    Returns list[(content, heading)].
    """
    content = doc["content"]
    heading = _news_title(content) or doc["metadata"].get("source", "")
    prefix = _context_prefix(heading)
    max_body = CHUNK_SIZE - HEADING_BUDGET

    out = []
    for seg in semantic_chunker.split_text(content):
        seg = seg.strip()
        if not seg:
            continue
        # SemanticChunker không chặn độ dài → segment quá to thì cắt thêm.
        pieces = char_splitter.split_text(seg) if len(seg) > max_body else [seg]
        for piece in pieces:
            piece = piece.strip()
            if len(piece) >= MIN_CHUNK_CHARS:
                out.append((prefix + piece, heading))
    return out


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk theo HYBRID: legal=structure-aware, news=semantic.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk.
        metadata gồm: source, type, chunk_index, heading.
    """
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # giữ dòng heading trong nội dung để chunk tự mô tả
    )
    # Chừa chỗ cho dòng ngữ cảnh → tổng vẫn ≤ CHUNK_SIZE.
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE - HEADING_BUDGET,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        length_function=len,
    )

    # SemanticChunker khởi tạo lười (chỉ khi có news) vì nó gọi embedding API.
    semantic_chunker = None

    def _get_semantic_chunker():
        nonlocal semantic_chunker
        if semantic_chunker is None:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_openai import OpenAIEmbeddings

            semantic_chunker = SemanticChunker(
                OpenAIEmbeddings(model=EMBEDDING_MODEL),
                breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
                breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_AMOUNT,
            )
        return semantic_chunker

    chunks: list[dict] = []
    for doc in documents:
        if doc["metadata"].get("type") == "legal":
            pairs = _chunk_legal(doc, header_splitter, char_splitter)
        else:
            pairs = _chunk_news(doc, _get_semantic_chunker(), char_splitter)

        for content, heading in pairs:
            chunks.append(
                {
                    "content": content,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": len(chunks),
                        "heading": heading,
                    },
                }
            )
    return chunks


# =============================================================================
# 3) EMBED — OpenAI text-embedding-3-small
# =============================================================================
def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed toàn bộ chunks bằng OpenAI text-embedding-3-small.

    Mỗi chunk được thêm key 'embedding': list[float] (1536 chiều).
    """
    from openai import OpenAI

    client = OpenAI()  # đọc OPENAI_API_KEY từ môi trường
    texts = [c["content"] for c in chunks]

    embeddings: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in resp.data)
        print(f"  embedded {min(start + EMBED_BATCH_SIZE, len(texts))}/{len(texts)}")

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


# =============================================================================
# 4) INDEX — Weaviate
# =============================================================================
def get_weaviate_client():
    """Kết nối Weaviate: ưu tiên Cloud (.env), nếu chưa cấu hình thì thử local."""
    import weaviate
    from weaviate.classes.init import AdditionalConfig, Auth, Timeout

    url = os.getenv("WEAVIATE_URL", "")
    api_key = os.getenv("WEAVIATE_API_KEY", "")
    # tăng timeout + bỏ qua health-check lúc khởi tạo (gRPC cloud hay chậm/latency)
    cfg = AdditionalConfig(timeout=Timeout(init=60, query=60, insert=120))
    # URL hợp lệ và không phải placeholder "xxx" → Weaviate Cloud.
    if url and "xxx" not in url and api_key and "xxx" not in api_key:
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key),
            additional_config=cfg,
            skip_init_checks=True,
        )
    # Fallback: Weaviate chạy bằng Docker ở localhost:8080.
    return weaviate.connect_to_local(additional_config=cfg, skip_init_checks=True)


def index_to_vectorstore(chunks: list[dict]):
    """Tạo collection và đẩy chunks (kèm vector tự sinh) vào Weaviate."""
    from weaviate.classes.config import Configure, DataType, Property

    client = get_weaviate_client()
    try:
        # Tạo lại collection cho idempotent (chạy lại không bị trùng).
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)
        collection = client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),  # ta tự cấp vector
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="heading", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )

        with collection.batch.dynamic() as batch:
            for c in chunks:
                m = c["metadata"]
                batch.add_object(
                    properties={
                        "content": c["content"],
                        "source": m.get("source", ""),
                        "doc_type": m.get("type", ""),
                        "heading": m.get("heading", ""),
                        "chunk_index": m.get("chunk_index", 0),
                    },
                    vector=c["embedding"],
                )
        failed = collection.batch.failed_objects
        if failed:
            print(f"  ⚠ {len(failed)} object insert lỗi (vd: {failed[0].message})")
        print(f"  ✓ Tổng object trong '{COLLECTION_NAME}': {len(collection)}")
    finally:
        client.close()


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking : {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Store    : {VECTOR_STORE} / collection '{COLLECTION_NAME}'")
    print("=" * 60)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to Weaviate")


if __name__ == "__main__":
    run_pipeline()
