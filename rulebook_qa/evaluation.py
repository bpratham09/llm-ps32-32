"""Evaluation of the RAG system on a fixed set of student questions.

Metrics
- Retrieval Hit@k : the expected page is among the top-k retrieved chunks
- Citation accuracy: the answer cites the expected page
- Answer accuracy  : the answer contains the key facts (keywords) from the handbook
- Refusal accuracy : out-of-scope questions get the "not found" reply
- Avg latency      : seconds per answer
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .qa import RulebookQA


@dataclass
class EvalRow:
    id: int
    category: str
    question: str
    expected_page: int | None
    answer: str
    retrieved_pages: str
    cited_pages: str
    retrieval_hit: bool | None
    citation_ok: bool | None
    answer_ok: bool | None
    refusal_ok: bool | None
    passed: bool
    latency_s: float


def load_questions(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate(qa: RulebookQA, questions: list[dict], pause_s: float = 0.0,
             progress: Callable[[int, int], None] | None = None) -> tuple[list[EvalRow], dict]:
    rows: list[EvalRow] = []
    for i, q in enumerate(questions, 1):
        res = qa.ask(q["question"])
        retrieved = res.retrieved_pages
        exp = q.get("expected_page")
        if exp is None:  # out-of-scope question
            hit = cite = ans = None
            refusal = not res.found
            passed = refusal
        else:
            refusal = None
            hit = exp in retrieved
            cite = exp in res.cited_pages
            ans = res.found and all(k.lower() in res.answer.lower() for k in q.get("keywords", []))
            passed = hit and cite and ans
        rows.append(EvalRow(i, q.get("category", ""), q["question"], exp, res.answer,
                            " ".join(map(str, retrieved)), " ".join(map(str, res.cited_pages)),
                            hit, cite, ans, refusal, passed, round(res.latency_s, 2)))
        if progress:
            progress(i, len(questions))
        if pause_s:
            time.sleep(pause_s)  # respect free-tier rate limits
    return rows, summarise(rows)


def _rate(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(100 * sum(vals) / len(vals), 1) if vals else None


def summarise(rows: list[EvalRow]) -> dict:
    return {
        "questions": len(rows),
        "passed": sum(r.passed for r in rows),
        "retrieval_hit_rate": _rate(r.retrieval_hit for r in rows),
        "citation_accuracy": _rate(r.citation_ok for r in rows),
        "answer_accuracy": _rate(r.answer_ok for r in rows),
        "refusal_accuracy": _rate(r.refusal_ok for r in rows),
        "avg_latency_s": round(sum(r.latency_s for r in rows) / max(len(rows), 1), 2),
    }


def save_results(rows: list[EvalRow], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        w.writerows(asdict(r) for r in rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    mark = lambda v: "-" if v is None else ("✅" if v else "❌")
    lines = [
        "# Evaluation results", "",
        f"**Passed {summary['passed']}/{summary['questions']}**", "",
        "| Metric | Value |", "|---|---|",
        f"| Retrieval Hit@k | {summary['retrieval_hit_rate']}% |",
        f"| Citation accuracy | {summary['citation_accuracy']}% |",
        f"| Answer accuracy | {summary['answer_accuracy']}% |",
        f"| Refusal accuracy | {summary['refusal_accuracy']}% |",
        f"| Avg latency | {summary['avg_latency_s']} s |", "",
        "| # | Question | Expected | Cited | Hit | Cite | Answer | Refusal | Pass |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        exp = f"Page {r.expected_page}" if r.expected_page else "Not found"
        lines.append(f"| {r.id} | {r.question} | {exp} | {r.cited_pages or '-'} | {mark(r.retrieval_hit)} | "
                     f"{mark(r.citation_ok)} | {mark(r.answer_ok)} | {mark(r.refusal_ok)} | {mark(r.passed)} |")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
