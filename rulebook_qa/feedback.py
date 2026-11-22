"""Stores thumbs-up / thumbs-down feedback from the UI in a CSV file for later review."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .config import settings

FIELDS = ["timestamp", "question", "answer", "rating", "cited_pages"]


def log_feedback(question: str, answer: str, rating: str, cited_pages: list[int],
                 path: Path | None = None) -> None:
    path = path or settings.feedback_file
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"), "question": question,
                    "answer": answer, "rating": rating, "cited_pages": " ".join(map(str, cited_pages))})
