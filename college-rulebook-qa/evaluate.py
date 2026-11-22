"""
Evaluate the assistant on the test questions in data/eval_questions.json.

Usage:
    python evaluate.py                          # sample handbook
    python evaluate.py --pdf my_handbook.pdf --questions my_questions.json

Writes evaluation/results.csv, evaluation/results.md and evaluation/summary.json
(also shown on the Evaluation page of the app).
"""
import argparse
from pathlib import Path

from rulebook_qa.config import has_api_key, key_env_name, settings
from rulebook_qa.evaluation import evaluate, load_questions, save_results
from rulebook_qa.qa import RulebookQA
from rulebook_qa.vectorstore import build_or_load


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", type=Path, default=settings.sample_pdf)
    ap.add_argument("--questions", type=Path, default=settings.eval_questions)
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between questions (free-tier limits)")
    args = ap.parse_args()

    if not has_api_key():
        raise SystemExit(f"{key_env_name()} is not set. Copy .env.example to .env and add your key.")

    info = build_or_load(args.pdf)
    print(f"Indexed {info.file_name}: {info.n_pages} pages, {info.n_chunks} chunks"
          f"{' (from cache)' if info.from_cache else ''}\n")
    qa = RulebookQA(info.store)
    rows, summary = evaluate(qa, load_questions(args.questions), pause_s=args.pause,
                             progress=lambda i, n: print(f"  [{i}/{n}] done"))

    print()
    for r in rows:
        print(f"Q{r.id} {'PASS' if r.passed else 'FAIL'} | {r.question}\n     -> {r.answer}\n")
    save_results(rows, summary, settings.eval_output_dir)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k:20s} {v}")
    print(f"\nSaved to {settings.eval_output_dir}/results.md")


if __name__ == "__main__":
    main()
