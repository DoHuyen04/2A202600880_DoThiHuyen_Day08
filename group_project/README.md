# Bài Tập Nhóm — Search Engine / RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1:  Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="DrugLaw_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
                        ┌─────────────────────────────┐
                        │   Streamlit UI (app.py)     │
                        │  - chat đa lượt             │
                        │  - conversation memory      │
                        │  - hiển thị nguồn + score   │
                        └──────────────┬──────────────┘
                                       │ câu hỏi
                    ┌──────────────────▼───────────────────┐
                    │  Query condensation (gpt-4o-mini)    │
                    │  follow-up → câu hỏi độc lập         │
                    └──────────────────┬───────────────────┘
                                       │ standalone query
              ┌────────────────────────▼────────────────────────┐
              │       retrieve()  — Task 9 pipeline             │
              │                                                  │
              │   ┌─ semantic_search (Task 5) ─┐                 │
              │   │   OpenAI embed + Weaviate   │                │
              │   │                              ├─ RRF merge ─┐  │
              │   └─ lexical_search (Task 6) ───┘             │  │
              │       BM25 (rank-bm25)                        │  │
              │                                    rerank (Task 7)│
              │                                    Jina x-encoder │
              │                                    (fallback:    │
              │                                     lexical)     │
              │            top_score < threshold ? ──► PageIndex │
              │                                       (Task 8)   │
              └────────────────────────┬────────────────────────┘
                                       │ top_k chunks (+source)
              ┌────────────────────────▼────────────────────────┐
              │   generate_with_citation()  — Task 10            │
              │   reorder (chống lost-in-middle)                 │
              │   → format_context (nhãn nguồn)                  │
              │   → GPT-4o-mini  → câu trả lời có [Nguồn, Năm]   │
              └─────────────────────────────────────────────────┘

Data layer: data/standardized/  →  Weaviate Cloud (598 chunks, dim 1536)
            +  data/drug_legal_corpus.pdf  →  PageIndex (vectorless tree)
```

**Luồng xử lý 1 câu hỏi:**
1. UI nhận câu hỏi → nếu là follow-up, *condense* thành câu hỏi độc lập (memory).
2. `retrieve()`: hybrid (semantic+BM25, RRF) → rerank → fallback PageIndex nếu yếu.
3. `generate_with_citation()`: reorder context → GPT-4o-mini sinh câu trả lời có citation.
4. UI hiển thị câu trả lời + danh sách nguồn (chunk, score, nhánh retrieval).

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Đỗ Thị Huyền | 2A202600880 | Task 1–10 (toàn bộ pipeline cá nhân: thu thập dữ liệu, chunking/indexing, semantic/lexical search, rerank, PageIndex, retrieval pipeline, generation) + Evaluation pipeline (golden dataset, eval_pipeline.py, results.md) + Chatbot Streamlit | ✅ Hoàn thành |
| _(thành viên 2)_ | | | |
| _(thành viên 3)_ | | | |
| _(thành viên 4)_ | | | |

> Cập nhật tên/MSSV các thành viên còn lại của nhóm và phần việc tương ứng trước buổi nộp.

---

## Hướng Dẫn Chạy

**Yêu cầu trước khi chạy:**
- `.env` đã có `OPENAI_API_KEY`, `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `PAGEINDEX_API_KEY`.
- Đã chạy Task 4 để index dữ liệu lên Weaviate (`python src/task4_chunking_indexing.py`).
- (Tùy chọn) Đã upload PageIndex (`python src/task8_pageindex_vectorless.py`) cho nhánh fallback.

```bash
# Cài đặt dependencies (từ thư mục gốc dự án)
pip install -r requirements.txt

# Chạy chatbot (chạy từ thư mục GỐC dự án để import được package src/)
streamlit run group_project/app.py
```

App mở tại http://localhost:8501. Trong sidebar có thể chỉnh `top_k`, ngưỡng
fallback PageIndex, bật/tắt hiển thị nguồn và conversation memory.

> Lưu ý: nếu `JINA_API_KEY` hết hạn/quota, reranking tự fallback sang
> lexical-overlap (không sập). Weaviate sandbox sống 14 ngày → hết hạn cần tạo
> lại cluster và chạy lại Task 4.

---

## Lưu ý: Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
