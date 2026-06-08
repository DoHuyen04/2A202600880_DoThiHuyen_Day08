"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản: https://pageindex.ai/  | SDK: VectifyAI/PageIndex

PageIndex KHÔNG dùng vector/embedding. Nó xây một "cây cấu trúc" (tree) của
tài liệu (giống mục lục: Chương > Điều > Khoản) rồi dùng LLM duyệt cây để chọn
node liên quan với câu hỏi — gọi là *reasoning-based / vectorless retrieval*.
Rất hợp với văn bản pháp luật có cấu trúc rõ ràng.

Vì PageIndex chỉ nhận PDF, ta gộp toàn bộ markdown pháp luật ở data/standardized
thành 1 file PDF rồi upload (1 doc_id). doc_id được cache lại để không phải
upload + xử lý (OCR + tree) mỗi lần.

Luồng API (bất đồng bộ):
    submit_document(pdf) -> doc_id
    poll is_retrieval_ready(doc_id)            # đợi OCR + tree xong
    submit_query(doc_id, query) -> retrieval_id
    poll get_retrieval(retrieval_id)           # đợi LLM duyệt cây xong

Cài đặt:
    pip install pageindex fpdf2
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_FILE = Path(__file__).parent.parent / "data" / "pageindex_cache.json"
PDF_PATH = Path(__file__).parent.parent / "data" / "drug_legal_corpus.pdf"
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"  # font Unicode hỗ trợ tiếng Việt

READY_TIMEOUT = 600   # giây chờ xử lý tài liệu
QUERY_TIMEOUT = 120   # giây chờ kết quả truy vấn
POLL_INTERVAL = 5


def _client():
    from pageindex import PageIndexClient

    if not PAGEINDEX_API_KEY or "xxx" in PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY chưa cấu hình trong .env")
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


# --------------------------------------------------------------------------- #
# Tạo PDF gộp tài liệu pháp luật
# --------------------------------------------------------------------------- #
def _clean_for_pdf(line: str) -> str:
    """Biến 1 dòng markdown thành text thường, an toàn cho fpdf (không tràn khổ)."""
    line = "".join(ch for ch in line if ch >= " ")          # bỏ ký tự điều khiển
    line = line.replace("|", " ").replace("**", "")          # bảng & in đậm
    line = line.replace("\\_", "_").replace("#", "")         # escaped underscore, heading
    line = re.sub(r"\s+", " ", line).strip()
    # bẻ token dài (URL, ____) mỗi 60 ký tự để multi_cell luôn xuống dòng được
    line = re.sub(r"(\S{60})(?=\S)", r"\1 ", line)
    return line


def _build_corpus_pdf(out_path: Path = PDF_PATH) -> Path:
    """Gộp các markdown trong data/standardized/legal/ thành 1 PDF tiếng Việt."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("arial", "", FONT_PATH)
    pdf.set_font("arial", size=10)

    # new_x=LMARGIN, new_y=NEXT: trả con trỏ về lề trái & xuống dòng sau mỗi cell
    # (mặc định new_x=RIGHT khiến cell kế tiếp hết bề rộng → FPDFException).
    legal_dir = STANDARDIZED_DIR / "legal"
    for md_file in sorted(legal_dir.glob("*.md")):
        pdf.add_page()
        # tiêu đề tài liệu (tên file) để PageIndex tách cây theo từng văn bản
        pdf.set_font("arial", size=13)
        pdf.multi_cell(0, 7, text=md_file.stem, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("arial", size=10)
        for raw in md_file.read_text(encoding="utf-8").splitlines():
            # bỏ dòng phân cách bảng dạng | --- | --- |
            if raw.strip() and set(raw.strip()) <= set("|-: "):
                continue
            line = _clean_for_pdf(raw)
            if not line:
                continue
            try:
                pdf.multi_cell(0, 5, text=line, new_x="LMARGIN", new_y="NEXT")
            except Exception:  # noqa: BLE001 - bỏ qua dòng cá biệt không render được
                continue
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    print(f"  ✓ Tạo PDF: {out_path} ({out_path.stat().st_size // 1024} KB)")
    return out_path


# --------------------------------------------------------------------------- #
# Upload + cache doc_id
# --------------------------------------------------------------------------- #
def upload_documents(force: bool = False) -> str:
    """Tạo PDF, upload lên PageIndex, đợi xử lý xong. Trả về doc_id (đã cache)."""
    pi = _client()

    # đã có cache và doc vẫn sẵn sàng → tái dùng
    if CACHE_FILE.exists() and not force:
        doc_id = json.loads(CACHE_FILE.read_text()).get("doc_id")
        if doc_id and pi.is_retrieval_ready(doc_id):
            print(f"  ✓ Dùng lại doc_id đã cache: {doc_id}")
            return doc_id

    _build_corpus_pdf()
    print("  ↑ Uploading PDF lên PageIndex...")
    doc_id = pi.submit_document(str(PDF_PATH))["doc_id"]
    print(f"  ✓ doc_id = {doc_id}; đợi xử lý (OCR + tree)...")

    deadline = time.time() + READY_TIMEOUT
    while time.time() < deadline:
        if pi.is_retrieval_ready(doc_id):
            print("  ✓ Tài liệu sẵn sàng cho retrieval")
            CACHE_FILE.write_text(json.dumps({"doc_id": doc_id}))
            return doc_id
        time.sleep(POLL_INTERVAL)
    raise TimeoutError("Quá thời gian chờ PageIndex xử lý tài liệu")


def _get_doc_id() -> str:
    if CACHE_FILE.exists():
        doc_id = json.loads(CACHE_FILE.read_text()).get("doc_id")
        if doc_id:
            return doc_id
    return upload_documents()


# --------------------------------------------------------------------------- #
# Parser kết quả retrieval (cấu trúc API có thể khác nhau → parse linh hoạt)
# --------------------------------------------------------------------------- #
def _iter_passages(nodes: list[dict]):
    """Duyệt các đoạn liên quan trong retrieved_nodes.

    Cấu trúc: node['relevant_contents'] là list-of-lists các dict có key
    'relevant_content' (text) và 'section_title'. Hàm này flatten lại.
    """
    for node in nodes:
        node_title = node.get("title", "")
        node_id = node.get("id", "")
        rc = node.get("relevant_contents")
        if isinstance(rc, str):  # phòng trường hợp API trả thẳng text
            yield {"content": rc, "title": node_title, "node_id": node_id}
            continue
        for group in rc or []:
            items = group if isinstance(group, list) else [group]
            for item in items:
                if isinstance(item, dict):
                    text = item.get("relevant_content") or item.get("content") or ""
                    title = item.get("section_title") or node_title
                else:
                    text, title = str(item), node_title
                if str(text).strip():
                    yield {"content": text, "title": title, "node_id": node_id}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless retrieval qua PageIndex. Dùng làm fallback ở Task 9.

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'pageindex'}.
    """
    pi = _client()
    doc_id = _get_doc_id()

    retrieval_id = pi.submit_query(doc_id, query, thinking=False)["retrieval_id"]

    deadline = time.time() + QUERY_TIMEOUT
    result = {}
    while time.time() < deadline:
        result = pi.get_retrieval(retrieval_id)
        status = str(result.get("status", "")).lower()
        if status in ("completed", "success", "done", "ready"):
            break
        if status in ("failed", "error"):
            raise RuntimeError(f"PageIndex retrieval lỗi: {result}")
        time.sleep(POLL_INTERVAL)

    nodes = result.get("retrieved_nodes", [])
    out = []
    for i, p in enumerate(_iter_passages(nodes)):
        if i >= top_k:
            break
        out.append(
            {
                "content": p["content"],
                # PageIndex không trả điểm số → dùng thứ hạng làm score giảm dần.
                "score": round(1.0 - i * 0.01, 4),
                "metadata": {"title": p["title"], "node_id": p["node_id"], "doc_id": doc_id},
                "source": "pageindex",
            }
        )
    return out


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY or "xxx" in PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong .env (đăng ký tại pageindex.ai)")
    else:
        print("=== Upload (chỉ chạy lần đầu, sau đó dùng cache) ===")
        upload_documents()
        print("\n=== Test query ===")
        for r in pageindex_search("Hình phạt tội mua bán trái phép chất ma túy", top_k=3):
            print(f"[{r['score']:.3f}] ({r['metadata'].get('title','')[:50]})")
            print("   ", r["content"][:160].replace("\n", " "), "...")
