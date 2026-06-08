"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown SẠCH.

Sử dụng MarkItDown của Microsoft (https://github.com/microsoft/markitdown)
làm bộ convert chính. Riêng các bài báo lưu dưới dạng HTML ("Save page as")
chứa rất nhiều boilerplate (menu, quảng cáo, script, bài liên quan...), nên
ta dùng BeautifulSoup để BÓC TÁCH phần nội dung chính trước khi đưa qua
MarkItDown. Mục tiêu: file .md chỉ còn nội dung đúng chủ đề, để bước chunking
(Task 4) tách được các đoạn có ngữ nghĩa tốt, không lẫn rác.

Pipeline:
    legal/*.docx  --MarkItDown-->            *.md  (+ dọn khoảng trắng)
    news/*.html   --bs4 (bóc nội dung)--MarkItDown--> *.md  (+ metadata header)
    news/*.json   --đọc content_markdown-->  *.md  (tương thích Crawl4AI)

Cài đặt:
    pip install "markitdown[docx]" beautifulsoup4
"""

import io
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup
from markitdown import MarkItDown

# Console Windows (cp1258/cp1252) không in được ký tự ✓/✗ và tiếng Việt → ép UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

DOC_EXTENSIONS = (".pdf", ".docx", ".doc", ".pptx", ".xlsx")
NEWS_HTML_EXTENSIONS = (".html", ".htm")

# Thẻ chắc chắn là rác / không phải nội dung bài viết.
JUNK_TAGS = [
    "script", "style", "noscript", "nav", "header", "footer", "aside",
    "form", "iframe", "svg", "button", "input", "select", "textarea",
    "link", "meta", "picture", "source", "video", "audio", "figure",
]
# class/id của khối phụ trợ (quảng cáo, chia sẻ, bài liên quan, bình luận...).
# CHỈ áp dụng BÊN TRONG node nội dung đã chọn, để không xóa nhầm container cha
# (vd theme zce-* bọc bài viết trong div class chứa "zone"). Vì vậy pattern này
# cố tình KHÔNG chứa các từ quá rộng như "zone"/"header"/"footer"/"content".
JUNK_PATTERN = re.compile(
    r"(advert|adsbygoogle|\bads?\b|banner|social|share|related|comment|"
    r"sidebar|breadcrumb|newsletter|popup|modal|recommend|subscribe|"
    r"tag-list|author-box|box-category|outbrain|taboola|trending|"
    r"most-?read|hot-?news)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Tiện ích chung
# --------------------------------------------------------------------------- #
def _slugify(name: str) -> str:
    """Tên file an toàn (giữ chữ tiếng Việt, thay ký tự lạ bằng '-')."""
    out = "".join(ch if (ch.isalnum() or ch in " -_") else "-" for ch in name)
    out = re.sub(r"\s+", "-", out.strip())
    out = re.sub(r"-{2,}", "-", out)
    return out.strip("-") or "untitled"


# dòng "rác" hay gặp ở trang báo: marker quảng cáo, player audio/video.
_AD_MARKER = re.compile(r"^(q\.?c|quảng cáo|advertisement|ad)$", re.IGNORECASE)
_PLAYER_TS = re.compile(r"^\d{1,2}:\d{2}(\s*/\s*\d{1,2}:\d{2})?$")
# điểm CẮT phần đuôi (mọi thứ sau đó là rác: tags, bài liên quan, tác giả...).
_CUT_TAG = re.compile(r"^#*\s*(từ\s*kho[áa]|tags?)\s*[:\b]", re.IGNORECASE)
# heading mà bản thân nó là một link → đây là box "bài liên quan", không phải
# tiêu đề mục thật (tiêu đề thật không bọc trong link http).
_CUT_RELATED = re.compile(r"^#{1,6}\s.*\]\(https?://", re.IGNORECASE)
# link đứng một mình kiểu "Theo dõi ... trên", "Follow us" → rác điều hướng.
_FOLLOW_LINK = re.compile(
    r"^\[[^\]]*(theo dõi|follow|tải app)[^\]]*\]\([^)]*\)$", re.IGNORECASE
)
# dòng cụt mở box "bài liên quan": [##### ...
_CUT_LINKHEAD = re.compile(r"^\[#{1,6}")
# widget video player / nhãn quảng cáo ở cuối bài → cắt cả khối từ đây.
_CUT_PLAYER = re.compile(
    r"^(current time|duration\b|nội dung quảng cáo|sponsored|"
    r"bài viết liên quan|tự động phát|autoplay)",
    re.IGNORECASE,
)
# link quảng cáo / tracking (mgid, taboola, outbrain, doubleclick...).
_AD_LINK = re.compile(
    r"^\[[^\]]*\]\([^)]*"
    r"(mgid|clck\.|taboola|outbrain|doubleclick|googlesyndication|"
    r"googleadservices|adservice|/ads?/)[^)]*\)$",
    re.IGNORECASE,
)
# gỡ link inline, giữ lại anchor text: [text](url) -> text
_INLINE_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def _clean_markdown(text: str) -> str:
    """Dọn markdown sau convert: bỏ ảnh, bảng rỗng, rác báo, gộp dòng trống."""
    # bỏ ảnh kể cả khi nằm trong link: [![alt](img)](url) / ![alt](img)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # cắt toàn bộ phần đuôi: tags / bài liên quan và mọi thứ sau đó
        if (
            _CUT_TAG.match(stripped)
            or _CUT_RELATED.match(stripped)
            or _CUT_LINKHEAD.match(stripped)
            or _CUT_PLAYER.match(stripped)
        ):
            break
        # bỏ hàng bảng rỗng / hàng phân cách dạng | --- | --- |
        if stripped and set(stripped) <= set("|-: "):
            continue
        # bỏ marker quảng cáo & timestamp player còn sót
        if _AD_MARKER.match(stripped) or _PLAYER_TS.match(stripped):
            continue
        # bỏ link rỗng [](url), link điều hướng "Theo dõi/Follow", link quảng cáo
        if (
            re.fullmatch(r"\[\]\([^)]*\)", stripped)
            or _FOLLOW_LINK.match(stripped)
            or _AD_LINK.match(stripped)
        ):
            continue
        lines.append(line.rstrip())

    text = "\n".join(lines)
    # gỡ mọi link inline còn lại, chỉ giữ anchor text (URL không có ý nghĩa cho RAG)
    text = _INLINE_LINK.sub(r"\1", text)
    # gộp >2 dòng trống liên tiếp thành 1 dòng trống
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


_MD = MarkItDown()


def _html_to_markdown(html_fragment: str) -> str:
    """Convert một đoạn HTML (đã bóc sạch) sang markdown bằng MarkItDown."""
    stream = io.BytesIO(html_fragment.encode("utf-8"))
    result = _MD.convert_stream(stream, file_extension=".html")
    return result.text_content


# --------------------------------------------------------------------------- #
# Trích xuất nội dung chính từ HTML bài báo
# --------------------------------------------------------------------------- #
def _meta(soup: BeautifulSoup, key: str) -> str | None:
    tag = soup.find("meta", attrs={"property": key}) or soup.find(
        "meta", attrs={"name": key}
    )
    if tag and tag.get("content"):
        return tag["content"].strip()
    return None


def _extract_article(html: str) -> tuple[str, dict]:
    """Bóc nội dung chính + metadata từ HTML bài báo.

    Returns:
        (html_nội_dung_chính, metadata dict gồm title/url/date)
    """
    soup = BeautifulSoup(html, "html.parser")

    meta = {
        "title": _meta(soup, "og:title")
        or (soup.title.get_text(strip=True) if soup.title else None),
        "url": _meta(soup, "og:url"),
        "date": _meta(soup, "article:published_time")
        or _meta(soup, "pubdate")
        or _meta(soup, "publishdate"),
    }
    if not meta["title"]:
        h1 = soup.find("h1")
        meta["title"] = h1.get_text(strip=True) if h1 else "Unknown"

    # 1) Vứt các thẻ rác toàn cục (an toàn: header/footer/nav/aside/script...).
    for tag in soup(JUNK_TAGS):
        tag.decompose()

    # 2) Chọn node chứa nhiều text trong <p> nhất, nhưng "chật" nhất.
    candidates = soup.find_all(["article", "main", "div", "section"])
    best, best_len = None, 0
    scores = []
    for node in candidates:
        p_len = sum(len(p.get_text(strip=True)) for p in node.find_all("p"))
        if p_len > 0:
            scores.append((node, p_len))
            if p_len > best_len:
                best, best_len = node, p_len

    if best is None:
        # Fallback: dùng cả body nếu không tìm được <p>.
        return str(soup.body or soup), meta

    # Trong các node giữ được >=90% lượng text, chọn node ít con nhất
    # (container sát nội dung nhất, tránh ôm thêm khối lân cận).
    node = min(
        (n for n, l in scores if l >= 0.9 * best_len),
        key=lambda n: len(n.find_all(True)),
    )

    # 3) Dọn các khối phụ trợ CÒN SÓT bên trong node nội dung (share/related...).
    for el in node.find_all(attrs={"class": JUNK_PATTERN}):
        el.decompose()
    for el in node.find_all(attrs={"id": JUNK_PATTERN}):
        el.decompose()

    return str(node), meta


# --------------------------------------------------------------------------- #
# Convert legal
# --------------------------------------------------------------------------- #
def convert_legal_docs() -> int:
    """Convert PDF/DOCX trong data/landing/legal/ sang markdown sạch."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print(f"  ! Bỏ qua: {legal_dir} chưa tồn tại")
        return 0

    count = 0
    for filepath in sorted(legal_dir.iterdir()):
        if not (filepath.is_file() and filepath.suffix.lower() in DOC_EXTENSIONS):
            continue
        print(f"Converting: {filepath.name}")
        try:
            result = _MD.convert(str(filepath))
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ Lỗi convert {filepath.name}: {exc}")
            continue
        content = f"# {filepath.stem}\n\n" + _clean_markdown(result.text_content)
        out = output_dir / f"{_slugify(filepath.stem)}.md"
        out.write_text(content, encoding="utf-8")
        print(f"  ✓ {out.name} ({len(content)} chars)")
        count += 1
    return count


# --------------------------------------------------------------------------- #
# Convert news
# --------------------------------------------------------------------------- #
def _save_news_md(output_dir: Path, stem: str, meta: dict, body_md: str) -> None:
    header = (
        f"# {meta.get('title', stem)}\n\n"
        f"**Source:** {meta.get('url') or 'N/A'}\n"
        f"**Published:** {meta.get('date') or 'N/A'}\n"
        f"**Crawled:** {date.today().isoformat()}\n\n"
        "---\n\n"
    )
    content = header + _clean_markdown(body_md)
    out = output_dir / f"{_slugify(stem)}.md"
    out.write_text(content, encoding="utf-8")
    print(f"  ✓ {out.name} ({len(content)} chars)")


def convert_news_articles() -> int:
    """Convert bài báo (.html bóc nội dung, hoặc .json từ Crawl4AI) sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print(f"  ! Bỏ qua: {news_dir} chưa tồn tại")
        return 0

    count = 0
    for filepath in sorted(news_dir.iterdir()):
        if not filepath.is_file():
            continue  # bỏ qua thư mục *_files/ đi kèm trang HTML
        suffix = filepath.suffix.lower()

        if suffix in NEWS_HTML_EXTENSIONS:
            print(f"Converting: {filepath.name}")
            try:
                html = filepath.read_text(encoding="utf-8", errors="ignore")
                body_html, meta = _extract_article(html)
                body_md = _html_to_markdown(body_html)
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ Lỗi convert {filepath.name}: {exc}")
                continue
            _save_news_md(output_dir, filepath.stem, meta, body_md)
            count += 1

        elif suffix == ".json":
            print(f"Converting: {filepath.name}")
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ Lỗi đọc JSON {filepath.name}: {exc}")
                continue
            meta = {
                "title": data.get("title", filepath.stem),
                "url": data.get("url"),
                "date": data.get("date_published") or data.get("date_crawled"),
            }
            body_md = (
                data.get("content_markdown")
                or data.get("markdown")
                or data.get("content", "")
            )
            _save_news_md(output_dir, filepath.stem, meta, body_md)
            count += 1
    return count


def convert_all():
    """Convert toàn bộ files. Dọn sạch output cũ trước để không lẫn file thừa."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown + bs4)")
    print("=" * 50)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Legal Documents ---")
    n_legal = convert_legal_docs()

    print("\n--- News Articles ---")
    n_news = convert_news_articles()

    print(f"\n✓ Done! Legal: {n_legal} file, News: {n_news} file")
    print("✓ Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
