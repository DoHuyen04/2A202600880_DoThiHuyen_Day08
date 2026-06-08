"""
RAG Evaluation Pipeline — LLM-as-judge theo định nghĩa metric của RAGAS.

Vì sao tự implement LLM-judge thay vì gọi thẳng RAGAS/DeepEval?
    Môi trường hiện tại có xung đột dependency nặng (RAGAS kéo theo
    openai 2.x làm vỡ pipeline đang chạy, và langchain_community lệch bản
    gây lỗi import). Để eval CHẮC CHẮN CHẠY ĐƯỢC và tái lập, ta tự chấm điểm
    bằng một LLM-judge (gpt-4o-mini) theo ĐÚNG định nghĩa 4 metric của RAGAS:

        - Faithfulness     : câu trả lời có bám đúng context không (không bịa)?
        - Answer Relevancy : câu trả lời có trúng câu hỏi không?
        - Context Recall   : context lấy về có đủ evidence cho đáp án chuẩn không?
        - Context Precision : trong context lấy về, bao nhiêu % thật sự hữu ích?

    Mỗi metric ∈ [0,1]. Judge trả JSON kèm lý do ngắn để có thể audit.

A/B comparison (theo gợi ý README — "có reranking vs không reranking"):
    - Config A: hybrid (semantic + BM25, RRF) + cross-encoder rerank
    - Config B: hybrid KHÔNG rerank (lấy thẳng theo điểm RRF)

Chạy:
    python -m group_project.evaluation.eval_pipeline           # full (18 Q × 2 config)
    python -m group_project.evaluation.eval_pipeline --limit 4 # chạy nhanh để thử
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# cho phép import package src/ khi chạy trực tiếp
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from openai import OpenAI

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import (
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    format_context,
    reorder_for_llm,
)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

GEN_MODEL = "gpt-4o-mini"     # model sinh câu trả lời (giống app)
JUDGE_MODEL = "gpt-4o-mini"   # model chấm điểm (LLM-as-judge)

METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

# Hai cấu hình so sánh A/B.
CONFIGS = {
    "A_hybrid_rerank": {"use_reranking": True, "desc": "Hybrid (semantic+BM25, RRF) + cross-encoder rerank"},
    "B_hybrid_norerank": {"use_reranking": False, "desc": "Hybrid (semantic+BM25, RRF), KHÔNG rerank"},
}

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =============================================================================
# Load dữ liệu
# =============================================================================

def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Chạy RAG pipeline cho 1 câu hỏi với 1 config
# =============================================================================

def run_pipeline(question: str, use_reranking: bool, top_k: int = 5) -> dict:
    """Chạy retrieve (theo config) + generate, trả về answer và contexts.

    Tách riêng (không gọi generate_with_citation) để KIỂM SOÁT được tham số
    use_reranking phục vụ A/B; phần generation tái dùng đúng prompt/cấu hình
    của Task 10 để công bằng giữa 2 config (chỉ khác bước retrieval).
    """
    chunks = retrieve(question, top_k=top_k, use_reranking=use_reranking)
    contexts = [c["content"] for c in chunks]

    if not chunks:
        return {"answer": "I cannot verify this information", "contexts": [], "sources": []}

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nCâu hỏi: {question}"

    resp = _client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return {"answer": resp.choices[0].message.content, "contexts": contexts, "sources": chunks}


# =============================================================================
# LLM-as-judge — chấm 4 metric (định nghĩa RAGAS)
# =============================================================================

_JUDGE_SYSTEM = """Bạn là giám khảo đánh giá hệ thống RAG (Retrieval-Augmented
Generation) tiếng Việt về pháp luật & tin tức ma túy. Chấm KHÁCH QUAN, NGHIÊM KHẮC.
Trả về DUY NHẤT một JSON hợp lệ, không kèm giải thích ngoài JSON."""

_JUDGE_TEMPLATE = """Đánh giá một mẫu RAG theo 4 chỉ số, mỗi chỉ số là số thực trong [0,1].

ĐỊNH NGHĨA (theo RAGAS):
- faithfulness: tỉ lệ các khẳng định trong CÂU TRẢ LỜI được suy ra trực tiếp từ
  CONTEXT. Nếu câu trả lời chứa thông tin không có trong context (bịa) → giảm mạnh.
- answer_relevancy: câu trả lời có trực tiếp & đầy đủ giải quyết CÂU HỎI không
  (không lan man, không lạc đề).
- context_recall: các ý trong ĐÁP ÁN CHUẨN (ground truth) có được CONTEXT bao phủ
  không. 1.0 = context chứa đủ mọi ý của đáp án chuẩn.
- context_precision: tỉ lệ các đoạn trong CONTEXT thật sự LIÊN QUAN/hữu ích để trả
  lời câu hỏi (ít đoạn nhiễu → điểm cao).

CÂU HỎI:
{question}

ĐÁP ÁN CHUẨN (ground truth):
{ground_truth}

CÂU TRẢ LỜI CỦA HỆ THỐNG:
{answer}

CONTEXT ĐÃ TRUY HỒI ({n_ctx} đoạn):
{contexts}

Trả JSON đúng schema:
{{"faithfulness": <float>, "answer_relevancy": <float>, "context_recall": <float>,
"context_precision": <float>, "reason": "<1-2 câu lý do ngắn gọn>"}}"""


def judge_sample(question: str, ground_truth: str, answer: str, contexts: list[str]) -> dict:
    """Gọi LLM-judge chấm 1 mẫu, trả dict 4 metric + reason."""
    if contexts:
        ctx_str = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    else:
        ctx_str = "(không có context nào được truy hồi)"

    prompt = _JUDGE_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        n_ctx=len(contexts),
        contexts=ctx_str,
    )
    resp = _client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,  # chấm điểm cần ổn định, tái lập
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    out = {}
    for k in METRIC_KEYS:
        try:
            out[k] = max(0.0, min(1.0, float(data.get(k, 0.0))))
        except (TypeError, ValueError):
            out[k] = 0.0
    out["reason"] = str(data.get("reason", ""))
    return out


# =============================================================================
# Evaluate 1 config trên toàn dataset
# =============================================================================

def evaluate_config(config_name: str, params: dict, dataset: list[dict]) -> dict:
    """Chạy + chấm toàn bộ dataset cho 1 config. Trả về per-item và trung bình."""
    print(f"\n{'='*70}\nCONFIG {config_name}: {params['desc']}\n{'='*70}")
    items = []
    for i, row in enumerate(dataset, 1):
        q = row["question"]
        gt = row["expected_answer"]
        print(f"  [{i}/{len(dataset)}] {q[:60]}...")
        try:
            out = run_pipeline(q, use_reranking=params["use_reranking"])
            scores = judge_sample(q, gt, out["answer"], out["contexts"])
        except Exception as exc:  # noqa: BLE001 - 1 câu lỗi không nên chặn cả batch
            print(f"      ⚠ lỗi: {exc}")
            scores = {k: 0.0 for k in METRIC_KEYS} | {"reason": f"error: {exc}"}
            out = {"answer": "", "contexts": []}
        items.append({
            "question": q,
            "expected_answer": gt,
            "answer": out["answer"],
            "n_contexts": len(out["contexts"]),
            **{k: scores[k] for k in METRIC_KEYS},
            "reason": scores["reason"],
        })

    averages = {
        k: round(sum(it[k] for it in items) / len(items), 4) if items else 0.0
        for k in METRIC_KEYS
    }
    averages["overall"] = round(sum(averages[k] for k in METRIC_KEYS) / len(METRIC_KEYS), 4)
    return {"config": config_name, "desc": params["desc"], "items": items, "averages": averages}


# =============================================================================
# A/B comparison
# =============================================================================

def compare_configs(dataset: list[dict]) -> dict:
    """So sánh tất cả config trong CONFIGS trên cùng dataset."""
    return {name: evaluate_config(name, params, dataset) for name, params in CONFIGS.items()}


# =============================================================================
# Export results.md
# =============================================================================

def _fmt(x: float) -> str:
    return f"{x:.3f}"


def export_results(results: dict) -> None:
    """Ghi báo cáo Markdown: bảng điểm, A/B, worst performers, đề xuất."""
    names = list(results.keys())
    a, b = names[0], names[1]
    avg_a, avg_b = results[a]["averages"], results[b]["averages"]

    lines = ["# RAG Evaluation Results", ""]
    lines += [
        "## Framework sử dụng",
        "",
        "**LLM-as-judge** (gpt-4o-mini) chấm theo đúng định nghĩa 4 metric của **RAGAS**.",
        "Tự implement thay vì gọi RAGAS/DeepEval do xung đột dependency trong môi trường "
        "(RAGAS kéo openai 2.x làm vỡ pipeline). Mỗi metric ∈ [0,1], judge chạy ở "
        "`temperature=0` để tái lập. Chi tiết: `eval_pipeline.py`.",
        "",
        f"- **Số câu hỏi (golden dataset):** {len(results[a]['items'])}",
        f"- **Generation model:** {GEN_MODEL} (temp={TEMPERATURE}, top_p={TOP_P})",
        f"- **Judge model:** {JUDGE_MODEL} (temp=0)",
        "",
        "---",
        "",
        "## Overall Scores",
        "",
        f"| Metric | {a} | {b} | Δ (A−B) |",
        "|--------|-------------|-------------|---|",
    ]
    label = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }
    for k in METRIC_KEYS:
        d = avg_a[k] - avg_b[k]
        lines.append(f"| {label[k]} | {_fmt(avg_a[k])} | {_fmt(avg_b[k])} | {d:+.3f} |")
    d_overall = avg_a["overall"] - avg_b["overall"]
    lines.append(f"| **Average** | **{_fmt(avg_a['overall'])}** | **{_fmt(avg_b['overall'])}** | **{d_overall:+.3f}** |")

    lines += [
        "",
        "---",
        "",
        "## A/B Comparison Analysis",
        "",
        f"**Config A — {a}:**",
        f"> {results[a]['desc']}",
        "",
        f"**Config B — {b}:**",
        f"> {results[b]['desc']}",
        "",
        "**Kết luận:**",
    ]
    winner = a if avg_a["overall"] >= avg_b["overall"] else b
    loser = b if winner == a else a
    win_avg = results[winner]["averages"]["overall"]
    lose_avg = results[loser]["averages"]["overall"]
    winner_reranks = CONFIGS[winner]["use_reranking"]
    if winner_reranks:
        why = (
            "Bước rerank đẩy các đoạn liên quan lên đầu top-k, cải thiện "
            "context_precision và faithfulness so với chỉ xếp theo điểm RRF."
        )
    else:
        why = (
            "Config KHÔNG rerank lại thắng vì reranker đang bị suy giảm: Jina API trả "
            "403 (key hết hạn) nên hệ thống rơi về rerank lexical-overlap yếu, đôi khi "
            "đẩy đoạn liên quan ra khỏi top-k khiến câu trả lời thành 'ngoài phạm vi'. "
            "→ Reranker chỉ có ích khi mô hình rerank đủ tốt; reranker hỏng còn hại hơn "
            "không rerank."
        )
    lines.append(
        f"> Config **{winner}** tốt hơn (average {_fmt(win_avg)} so với {_fmt(lose_avg)}). {why}"
    )

    # Worst performers: theo overall của config thắng.
    lines += ["", "---", "", "## Worst Performers (Bottom 3)", "",
              f"_(theo điểm trung bình mỗi câu của config {winner})_", "",
              "| # | Question | Faith. | Relev. | Recall | Precis. | Root Cause |",
              "|---|----------|--------|--------|--------|---------|------------|"]
    items = results[winner]["items"]
    ranked = sorted(
        items, key=lambda it: sum(it[k] for k in METRIC_KEYS) / len(METRIC_KEYS)
    )[:3]
    for i, it in enumerate(ranked, 1):
        q = it["question"].replace("|", "/")
        q = (q[:55] + "…") if len(q) > 55 else q
        reason = it["reason"].replace("|", "/").replace("\n", " ")
        reason = (reason[:80] + "…") if len(reason) > 80 else reason
        lines.append(
            f"| {i} | {q} | {_fmt(it['faithfulness'])} | {_fmt(it['answer_relevancy'])} "
            f"| {_fmt(it['context_recall'])} | {_fmt(it['context_precision'])} | {reason} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Recommendations",
        "",
        "### Cải tiến 1 — Bật reranker chất lượng cao",
        "**Action:** Jina API đang trả 403 (key hết hạn) nên rerank rơi về lexical-overlap. "
        "Cấp lại `JINA_API_KEY` hợp lệ hoặc dùng cross-encoder local "
        "(`jinaai/jina-reranker-v2-base-multilingual`).  ",
        "**Expected impact:** tăng context_precision & faithfulness, nhất là câu nhiều đoạn nhiễu.",
        "",
        "### Cải tiến 2 — Cải thiện context_recall cho câu hỏi dạng liệt kê/bảng",
        "**Action:** mở rộng `structural_fetch` (Task 9) để bắt thêm tham chiếu Chương/Mục, "
        "tăng `top_k` cho câu hỏi liệt kê (vd danh mục chất ma túy).  ",
        "**Expected impact:** giảm trường hợp thiếu evidence ở các câu recall thấp.",
        "",
        "### Cải tiến 3 — Bổ sung dữ liệu tin tức",
        "**Action:** crawl thêm bài báo có nội dung sạch (một số file HTML bị lỗi encoding), "
        "chuẩn hóa lại để chunk tin tức giàu thông tin hơn.  ",
        "**Expected impact:** tăng faithfulness/recall cho nhóm câu hỏi về nghệ sĩ.",
        "",
    ]

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Đã ghi báo cáo: {RESULTS_PATH}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation pipeline")
    parser.add_argument("--limit", type=int, default=0, help="chỉ chạy N câu đầu (để thử nhanh)")
    args = parser.parse_args()

    dataset = load_golden_dataset()
    if args.limit:
        dataset = dataset[: args.limit]
    print(f"Loaded {len(dataset)} test cases")

    results = compare_configs(dataset)

    # In tóm tắt ra console
    print(f"\n{'='*70}\nKẾT QUẢ TRUNG BÌNH\n{'='*70}")
    for name, res in results.items():
        avg = res["averages"]
        print(f"{name:22s} | " + " | ".join(f"{k}={avg[k]:.3f}" for k in METRIC_KEYS)
              + f" | overall={avg['overall']:.3f}")

    export_results(results)

    # Lưu raw per-item để truy vết (không bắt buộc nhưng hữu ích)
    raw_path = Path(__file__).parent / "eval_raw_results.json"
    raw_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"📄 Chi tiết per-item: {raw_path}")


if __name__ == "__main__":
    main()
