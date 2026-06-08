# RAG Evaluation Results

## Framework sử dụng

**LLM-as-judge** (gpt-4o-mini) chấm theo đúng định nghĩa 4 metric của **RAGAS**.
Tự implement thay vì gọi RAGAS/DeepEval do xung đột dependency trong môi trường (RAGAS kéo openai 2.x làm vỡ pipeline). Mỗi metric ∈ [0,1], judge chạy ở `temperature=0` để tái lập. Chi tiết: `eval_pipeline.py`.

- **Số câu hỏi (golden dataset):** 18
- **Generation model:** gpt-4o-mini (temp=0.3, top_p=0.9)
- **Judge model:** gpt-4o-mini (temp=0)

---

## Overall Scores

| Metric | A_hybrid_rerank | B_hybrid_norerank | Δ (A−B) |
|--------|-------------|-------------|---|
| Faithfulness | 0.794 | 0.844 | -0.050 |
| Answer Relevance | 0.844 | 0.900 | -0.056 |
| Context Recall | 0.822 | 0.883 | -0.061 |
| Context Precision | 0.722 | 0.728 | -0.006 |
| **Average** | **0.796** | **0.839** | **-0.043** |

---

## A/B Comparison Analysis

**Config A — A_hybrid_rerank:**
> Hybrid (semantic+BM25, RRF) + cross-encoder rerank

**Config B — B_hybrid_norerank:**
> Hybrid (semantic+BM25, RRF), KHÔNG rerank

**Kết luận:**
> Config **B_hybrid_norerank** tốt hơn (average 0.839 so với 0.796). Config KHÔNG rerank lại thắng vì reranker đang bị suy giảm: Jina API trả 403 (key hết hạn) nên hệ thống rơi về rerank lexical-overlap yếu, đôi khi đẩy đoạn liên quan ra khỏi top-k khiến câu trả lời thành 'ngoài phạm vi'. → Reranker chỉ có ích khi mô hình rerank đủ tốt; reranker hỏng còn hại hơn không rerank.

---

## Worst Performers (Bottom 3)

_(theo điểm trung bình mỗi câu của config B_hybrid_norerank)_

| # | Question | Faith. | Relev. | Recall | Precis. | Root Cause |
|---|----------|--------|--------|--------|---------|------------|
| 1 | Người trồng cây có chứa chất ma tuý có thể được miễn tr… | 0.000 | 0.000 | 0.000 | 0.200 | Câu trả lời không cung cấp thông tin chính xác về miễn trách nhiệm hình sự và kh… |
| 2 | Ca sĩ Chi Dân bị cơ quan chức năng xử lý vì liên quan đ… | 0.600 | 0.800 | 0.900 | 0.400 | Câu trả lời chứa thông tin không hoàn toàn chính xác và có phần bịa đặt, nhưng v… |
| 3 | Danh mục các chất ma tuý thuộc Danh mục I theo quy định… | 0.800 | 0.900 | 0.600 | 0.700 | Câu trả lời chứa thông tin chính xác nhưng không đầy đủ về danh mục các chất ma … |

---

## Recommendations

### Cải tiến 1 — Bật reranker chất lượng cao
**Action:** Jina API đang trả 403 (key hết hạn) nên rerank rơi về lexical-overlap. Cấp lại `JINA_API_KEY` hợp lệ hoặc dùng cross-encoder local (`jinaai/jina-reranker-v2-base-multilingual`).  
**Expected impact:** tăng context_precision & faithfulness, nhất là câu nhiều đoạn nhiễu.

### Cải tiến 2 — Cải thiện context_recall cho câu hỏi dạng liệt kê/bảng
**Action:** mở rộng `structural_fetch` (Task 9) để bắt thêm tham chiếu Chương/Mục, tăng `top_k` cho câu hỏi liệt kê (vd danh mục chất ma túy).  
**Expected impact:** giảm trường hợp thiếu evidence ở các câu recall thấp.

### Cải tiến 3 — Bổ sung dữ liệu tin tức
**Action:** crawl thêm bài báo có nội dung sạch (một số file HTML bị lỗi encoding), chuẩn hóa lại để chunk tin tức giàu thông tin hơn.  
**Expected impact:** tăng faithfulness/recall cho nhóm câu hỏi về nghệ sĩ.
