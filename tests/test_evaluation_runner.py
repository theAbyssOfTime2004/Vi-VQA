"""Evaluation runner: prompt replay, scoring and failure handling.

These tests exercise `evaluate()`/`messages_from_sample()` directly against
hand-built sample files — not through `prepare_fvqa()` — since what is
under test here is the evaluation runner's own behavior (prompt replay,
failure handling, result provenance), independent of which dataset loader
produced the file.
"""

import json

import pytest

from fvqa.config import Config
from fvqa.data.grounding import apply_grounding
from fvqa.data.samples import IMAGE_TOKEN, write_split
from fvqa.evaluation.runner import evaluate, load_split, messages_from_sample, reference_of


class FakeModel:
    """Records the prompts it is given and returns scripted answers."""

    def __init__(self, answers=None, fail_on=None):
        self.answers = answers or {}
        self.fail_on = fail_on or set()
        self.seen = []

    def generate(self, messages, **kwargs):
        question = messages[-1]["content"][-1]["text"]
        self.seen.append(messages)
        if question in self.fail_on:
            raise RuntimeError("boom")
        return self.answers.get(question, question)


def sample(with_system=False):
    conversations = []
    if with_system:
        conversations.append({"from": "system", "value": "Context: Bến Thành market."})
    conversations += [
        {"from": "human", "value": f"{IMAGE_TOKEN}\nWhat is this?"},
        {"from": "gpt", "value": "It is a market."},
    ]
    return {"id": "1_0", "image": "image_1.jpg", "image_id": "1", "conversations": conversations}


def write_samples(config, records):
    """Write already-built (question, answer, description) records to
    `train.json`, applying grounding the same way a real loader would.

    A stand-in for a dataset-specific prepare step: these tests are about
    `evaluate()`'s own behavior, not about how any particular loader turns
    raw records into samples.
    """
    written = []
    for record in records:
        prompt = apply_grounding(record["question"], record.get("description"), config.data.grounding)
        turns: list[dict[str, str]] = []
        if prompt.system:
            turns.append({"from": "system", "value": prompt.system})
        turns.append({"from": "human", "value": f"{IMAGE_TOKEN}\n{prompt.question}"})
        turns.append({"from": "gpt", "value": record["answer"]})
        written.append(
            {
                "id": f"{record['id']}_0",
                "image": f"image_{record['id']}.jpg",
                "image_id": str(record["id"]),
                "conversations": turns,
            }
        )
    write_split(written, config.data.split_file("train"))
    return written


class TestMessagesFromSample:
    def test_image_token_is_stripped_from_the_question(self):
        messages = messages_from_sample(sample(), "/data/images")
        assert messages[-1]["content"][-1]["text"] == "What is this?"

    def test_image_path_is_joined_to_the_folder(self):
        messages = messages_from_sample(sample(), "/data/images")
        assert messages[-1]["content"][0]["image"] == "/data/images/image_1.jpg"

    def test_system_turn_is_carried_through(self):
        messages = messages_from_sample(sample(with_system=True), "/img")
        assert messages[0]["role"] == "system"
        assert "Bến Thành" in messages[0]["content"][0]["text"]

    def test_reference_answer_is_not_part_of_the_prompt(self):
        messages = messages_from_sample(sample(), "/img")
        assert all(m["role"] != "assistant" for m in messages)

    def test_reference_of_reads_the_gpt_turn(self):
        assert reference_of(sample()) == "It is a market."


class TestEvaluate:
    @pytest.fixture
    def config(self, tmp_path):
        config = Config()
        config.data.data_dir = str(tmp_path)
        config.data.image_folder = str(tmp_path / "images")
        config.evaluation.num_samples = -1
        return config

    @pytest.fixture
    def prepared(self, config):
        records = [
            {"id": i, "description": "Bến Thành market is in district 1.",
             "question": f"Question {i}?", "answer": f"Answer {i}."}
            for i in range(20)
        ]
        write_samples(config, records)
        return config

    def test_perfect_model_scores_full_marks(self, prepared):
        samples = load_split(prepared.data.split_file("train"))
        answers = {
            s["conversations"][-2]["value"].replace(f"{IMAGE_TOKEN}\n", ""): s["conversations"][-1][
                "value"
            ]
            for s in samples
        }
        result = evaluate(FakeModel(answers), prepared, split="train")
        assert result["metrics"]["exact_match"] == pytest.approx(100.0)

    def test_grounding_is_not_applied_twice(self, config):
        # The split file already contains the grounded prompt. Re-applying
        # grounding would score the model on a prompt it never saw.
        config.data.grounding.enabled = True
        write_samples(config, [
            {"id": 0, "description": "Bến Thành market is in district 1.",
             "question": "What is this?", "answer": "A market."}
        ])
        model = FakeModel()
        evaluate(model, config, split="train")

        prompt = model.seen[0][-1]["content"][-1]["text"]
        assert prompt.count("Bến Thành market is in district 1.") == 1
        assert prompt.count("What is this?") == 1

    def test_num_samples_limits_the_run(self, prepared):
        result = evaluate(FakeModel(), prepared, split="train", num_samples=5)
        assert result["num_samples"] == 5

    def test_a_failing_sample_does_not_end_the_run(self, prepared):
        samples = load_split(prepared.data.split_file("train"))
        doomed = samples[0]["conversations"][-2]["value"].replace(f"{IMAGE_TOKEN}\n", "")
        result = evaluate(FakeModel(fail_on={doomed}), prepared, split="train")
        assert result["num_failed"] == 1
        assert result["num_samples"] == len(samples) - 1

    def test_results_are_written_to_disk(self, prepared, tmp_path):
        output = tmp_path / "results" / "eval.json"
        evaluate(FakeModel(), prepared, split="train", output_path=str(output))
        written = json.loads(output.read_text(encoding="utf-8"))
        assert written["metrics"]
        assert len(written["predictions"]) == written["num_samples"]

    def test_grounding_state_is_recorded_in_the_results(self, prepared):
        # Scores are only comparable between runs with the same setting.
        result = evaluate(FakeModel(), prepared, split="train")
        assert result["grounding_enabled"] is False

    def test_missing_split_file_is_reported(self, config):
        with pytest.raises(FileNotFoundError, match="fvqa prepare"):
            evaluate(FakeModel(), config, split="val")


class TestStyleSystemPrompt:
    def test_no_system_turn_when_unset(self):
        assert messages_from_sample(sample(), "/img")[0]["role"] == "user"

    def test_style_prompt_becomes_the_first_turn(self):
        messages = messages_from_sample(sample(), "/img", system_prompt="Answer briefly.")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"][0]["text"] == "Answer briefly."

    def test_style_prompt_precedes_a_grounding_turn(self):
        messages = messages_from_sample(
            sample(with_system=True), "/img", system_prompt="Answer briefly."
        )
        assert [m["role"] for m in messages] == ["system", "system", "user"]
        # Style first, then the grounding context it applies to.
        assert messages[0]["content"][0]["text"] == "Answer briefly."
        assert "Bến Thành" in messages[1]["content"][0]["text"]

    def test_question_is_untouched_by_the_style_prompt(self):
        messages = messages_from_sample(sample(), "/img", system_prompt="Answer briefly.")
        assert messages[-1]["content"][-1]["text"] == "What is this?"


class TestResultProvenance:
    def test_prompt_settings_are_recorded_in_the_results(self, tmp_path):
        # Two runs differing only by prompt must be distinguishable later.
        config = Config()
        config.data.data_dir = str(tmp_path)
        config.data.image_folder = str(tmp_path / "images")
        config.inference.system_prompt = "Answer briefly."
        config.inference.max_new_tokens = 128

        write_samples(config, [{"id": 0, "description": "x", "question": "Q", "answer": "A"}])

        result = evaluate(FakeModel(), config, split="train")
        assert result["system_prompt"] == "Answer briefly."
        assert result["max_new_tokens"] == 128

    def test_style_prompt_reaches_the_model(self, tmp_path):
        config = Config()
        config.data.data_dir = str(tmp_path)
        config.data.image_folder = str(tmp_path / "images")
        config.inference.system_prompt = "One sentence only."

        write_samples(config, [{"id": 0, "description": "x", "question": "Q", "answer": "A"}])

        model = FakeModel()
        evaluate(model, config, split="train")
        assert model.seen[0][0]["role"] == "system"


class TestRetrievalSummary:
    """The run-level bound on what grounding could have contributed."""

    def test_counts_how_often_the_supporting_fact_was_retrieved(self):
        from fvqa.evaluation.runner import summarize_retrieval

        records = [
            {"retrieval": {"status": "ok", "oracle_fact_retrieved": True}},
            {"retrieval": {"status": "ok", "oracle_fact_retrieved": False}},
            {"retrieval": {"status": "ok", "oracle_fact_retrieved": True}},
        ]
        summary = summarize_retrieval(records)
        assert summary["oracle_fact_retrieved"] == 2
        assert summary["num_with_provenance"] == 3
        assert summary["recall"] == pytest.approx(2 / 3)

    def test_counts_retrieval_failures_separately(self):
        # A seed that resolved to nothing is a different failure from a
        # seed that resolved and simply missed the fact.
        from fvqa.evaluation.runner import summarize_retrieval

        records = [
            {"retrieval": {"status": "no_seed_match", "oracle_fact_retrieved": False}},
            {"retrieval": {"status": "ok", "oracle_fact_retrieved": True}},
        ]
        summary = summarize_retrieval(records)
        assert summary["failed_retrievals"] == 1
        assert summary["oracle_fact_retrieved"] == 1

    def test_ignores_records_without_provenance(self):
        from fvqa.evaluation.runner import summarize_retrieval

        summary = summarize_retrieval([{"id": "x"}, {"id": "y"}])
        assert summary["num_with_provenance"] == 0
        assert summary["recall"] == 0.0

    def test_format_scores_reports_the_condition(self):
        from fvqa.evaluation.runner import format_scores

        rendered = format_scores(
            {
                "split": "val",
                "condition": "oracle-seed-graph",
                "num_samples": 10,
                "num_failed": 0,
                "metrics": {"exact_match": 42.0},
                "retrieval": {"max_hops": 2, "top_k_facts": 5, "ranking_method": "lexical"},
                "retrieval_summary": {
                    "num_with_provenance": 10,
                    "oracle_fact_retrieved": 6,
                    "recall": 0.6,
                    "failed_retrievals": 1,
                },
            }
        )
        assert "oracle-seed-graph" in rendered
        assert "60.0%" in rendered
        assert "2 hop(s)" in rendered

    def test_format_scores_omits_retrieval_for_conditions_that_do_not_retrieve(self):
        from fvqa.evaluation.runner import format_scores

        rendered = format_scores(
            {
                "split": "val",
                "condition": "no-context",
                "num_samples": 10,
                "num_failed": 0,
                "metrics": {"exact_match": 12.0},
            }
        )
        assert "Retrieval:" not in rendered
        assert "no-context" in rendered
