from rulebook_qa.evaluation import evaluate, load_questions, save_results
from rulebook_qa.config import settings
from rulebook_qa.prompts import NOT_FOUND
from rulebook_qa.qa import RulebookQA, is_not_found, parse_cited_pages


def test_parse_cited_pages():
    assert parse_cited_pages("Rule text. (Source: Page 2)") == [2]
    assert parse_cited_pages("Rule text. (Source: Pages 3, 4)") == [3, 4]
    assert parse_cited_pages("No citation here") == []


def test_not_found_detection():
    assert is_not_found(NOT_FOUND)
    assert is_not_found("sorry, i could NOT find this in the college handbook")
    assert not is_not_found("You need 75% attendance. (Source: Page 2)")


def test_answer_has_sources_and_citation(index, fake_llm):
    qa = RulebookQA(index.store, llm=fake_llm)
    res = qa.ask("What is the fine for returning a library book late?")
    assert res.found
    assert res.cited_pages == [6]
    assert res.sources[0].cited and res.sources[0].page == 6  # cited source listed first
    assert "Answer ONLY using the handbook context" in fake_llm.prompts[-1]


def test_out_of_scope_returns_not_found(index, fake_llm):
    res = RulebookQA(index.store, llm=fake_llm).ask("Who is the placement officer?")
    assert not res.found
    assert res.sources == [] and res.cited_pages == []


def test_follow_up_question_is_rewritten(index, fake_llm):
    qa = RulebookQA(index.store, llm=fake_llm)
    history = [{"role": "user", "content": "What is the attendance rule?"},
               {"role": "assistant", "content": "75% per course. (Source: Page 2)"}]
    res = qa.ask("and for practicals?", history=history)
    assert res.standalone_question == "and for practicals? (rewritten)"
    assert "Standalone question:" in fake_llm.prompts[0]


def test_evaluation_pipeline(index, fake_llm, tmp_path):
    qa = RulebookQA(index.store, llm=fake_llm)
    rows, summary = evaluate(qa, load_questions(settings.eval_questions))
    assert summary["questions"] == 10
    assert summary["retrieval_hit_rate"] == 100.0
    assert summary["refusal_accuracy"] == 100.0
    save_results(rows, summary, tmp_path)
    assert (tmp_path / "results.md").exists() and (tmp_path / "summary.json").exists()
