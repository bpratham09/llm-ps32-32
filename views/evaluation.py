"""Evaluation dashboard: run the 10-question test set and see the metrics."""
import json

import pandas as pd
import streamlit as st

from rulebook_qa.config import settings
from rulebook_qa.evaluation import evaluate, load_questions, save_results
from rulebook_qa.ui import get_engine, hero

hero("📊 Evaluation",
     "How well does the assistant answer real student questions? Ten questions are checked for "
     "correct retrieval, correct page citation, correct facts, and refusing out-of-scope questions.")

out_dir = settings.eval_output_dir
summary_file, results_file = out_dir / "summary.json", out_dir / "results.csv"

with st.expander("📐 How the metrics are measured"):
    st.markdown("""
| Metric | Meaning |
|---|---|
| **Retrieval Hit@k** | The page holding the answer is among the top-k chunks FAISS returned |
| **Citation accuracy** | The answer cites the correct page, e.g. *(Source: Page 2)* |
| **Answer accuracy** | The answer contains the key facts from the handbook (e.g. *75%*, *Rs. 700*) |
| **Refusal accuracy** | Questions the handbook does not cover get the *not found* reply |
| **Avg latency** | Seconds from question to answer |

A question **passes** when all its checks pass. Edit `data/eval_questions.json` to add your own questions.
""")

col_run, col_note = st.columns([1, 3])
run = col_run.button("▶️ Run evaluation", type="primary", width="stretch")
col_note.caption("Runs all questions through the live system (about 30 seconds). "
                 "You can also run `python evaluate.py` in the terminal.")

if run:
    engine = get_engine()
    if engine is None:
        st.stop()
    qa, _ = engine
    questions = load_questions(settings.eval_questions)
    bar = st.progress(0.0, text="Starting...")
    rows, summary = evaluate(qa, questions, pause_s=1.0,
                             progress=lambda i, n: bar.progress(i / n, text=f"Question {i} of {n}"))
    save_results(rows, summary, out_dir)
    bar.empty()
    st.success(f"Done: {summary['passed']}/{summary['questions']} questions passed.")

if not summary_file.exists():
    st.info("No results yet. Click **Run evaluation** to test the system.")
    st.stop()

summary = json.loads(summary_file.read_text())
df = pd.read_csv(results_file)

st.markdown("#### Results")
m = st.columns(5)
m[0].metric("Passed", f"{summary['passed']}/{summary['questions']}")
m[1].metric("Retrieval Hit@k", f"{summary['retrieval_hit_rate']}%")
m[2].metric("Citation accuracy", f"{summary['citation_accuracy']}%")
m[3].metric("Answer accuracy", f"{summary['answer_accuracy']}%")
m[4].metric("Refusal accuracy", f"{summary['refusal_accuracy']}%")
st.caption(f"Average latency: {summary['avg_latency_s']} s per question")

show = df[["id", "category", "question", "expected_page", "cited_pages", "passed", "latency_s", "answer"]].copy()
show["expected_page"] = show["expected_page"].apply(lambda v: "Not found" if pd.isna(v) else f"Page {int(v)}")
st.dataframe(
    show, hide_index=True, width="stretch",
    column_config={
        "id": st.column_config.NumberColumn("#", width="small"),
        "category": "Category",
        "question": st.column_config.TextColumn("Question", width="large"),
        "expected_page": "Expected",
        "cited_pages": "Cited",
        "passed": st.column_config.CheckboxColumn("Pass"),
        "latency_s": st.column_config.NumberColumn("Latency (s)", format="%.2f"),
        "answer": st.column_config.TextColumn("Bot answer", width="large"),
    })

by_cat = df.groupby("category")["passed"].mean().mul(100).round(0).rename("Pass rate (%)")
st.markdown("#### Pass rate by category")
st.bar_chart(by_cat, horizontal=True, height=220)

st.download_button("⬇️ Download results (CSV)", results_file.read_bytes(), "evaluation_results.csv", "text/csv")
