"""Evaluation runner: prompt replay, scoring and failure handling."""

import json

import pytest

from vivqa.config import Config
from vivqa.data.prepare import IMAGE_TOKEN, prepare
from vivqa.evaluation.runner import evaluate, load_split, messages_from_sample, reference_of


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
        conversations.append({"from": "system", "value": "Ngữ cảnh: Chợ Bến Thành."})
    conversations += [
        {"from": "human", "value": f"{IMAGE_TOKEN}\nĐây là gì?"},
        {"from": "gpt", "value": "Đây là khu chợ."},
    ]
    return {"id": "1_0", "image": "image_1.jpg", "image_id": "1", "conversations": conversations}


class TestMessagesFromSample:
    def test_image_token_is_stripped_from_the_question(self):
        messages = messages_from_sample(sample(), "/data/images")
        assert messages[-1]["content"][-1]["text"] == "Đây là gì?"

    def test_image_path_is_joined_to_the_folder(self):
        messages = messages_from_sample(sample(), "/data/images")
        assert messages[-1]["content"][0]["image"] == "/data/images/image_1.jpg"

    def test_system_turn_is_carried_through(self):
        messages = messages_from_sample(sample(with_system=True), "/img")
        assert messages[0]["role"] == "system"
        assert "Chợ Bến Thành" in messages[0]["content"][0]["text"]

    def test_reference_answer_is_not_part_of_the_prompt(self):
        messages = messages_from_sample(sample(), "/img")
        assert all(m["role"] != "assistant" for m in messages)

    def test_reference_of_reads_the_gpt_turn(self):
        assert reference_of(sample()) == "Đây là khu chợ."


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
            {
                "id": i,
                "description": "Chợ Bến Thành nằm ở quận 1.",
                "conversations": [
                    {"role": "user", "content": f"Câu hỏi {i}?"},
                    {"role": "assistant", "content": f"Trả lời {i}."},
                ],
            }
            for i in range(20)
        ]
        prepare(config, records=records)
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
        prepare(
            config,
            records=[
                {
                    "id": 0,
                    "description": "Chợ Bến Thành nằm ở quận 1.",
                    "conversations": [
                        {"role": "user", "content": "Đây là gì?"},
                        {"role": "assistant", "content": "Khu chợ."},
                    ],
                }
            ],
        )
        model = FakeModel()
        evaluate(model, config, split="train")

        prompt = model.seen[0][-1]["content"][-1]["text"]
        assert prompt.count("Chợ Bến Thành nằm ở quận 1.") == 1
        assert prompt.count("Đây là gì?") == 1

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
        with pytest.raises(FileNotFoundError, match="vivqa prepare"):
            evaluate(FakeModel(), config, split="val")


class TestStyleSystemPrompt:
    def test_no_system_turn_when_unset(self):
        assert messages_from_sample(sample(), "/img")[0]["role"] == "user"

    def test_style_prompt_becomes_the_first_turn(self):
        messages = messages_from_sample(sample(), "/img", system_prompt="Trả lời ngắn gọn.")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"][0]["text"] == "Trả lời ngắn gọn."

    def test_style_prompt_precedes_a_grounding_turn(self):
        messages = messages_from_sample(
            sample(with_system=True), "/img", system_prompt="Trả lời ngắn gọn."
        )
        assert [m["role"] for m in messages] == ["system", "system", "user"]
        # Style first, then the grounding context it applies to.
        assert messages[0]["content"][0]["text"] == "Trả lời ngắn gọn."
        assert "Chợ Bến Thành" in messages[1]["content"][0]["text"]

    def test_question_is_untouched_by_the_style_prompt(self):
        messages = messages_from_sample(sample(), "/img", system_prompt="Trả lời ngắn gọn.")
        assert messages[-1]["content"][-1]["text"] == "Đây là gì?"


class TestResultProvenance:
    def test_prompt_settings_are_recorded_in_the_results(self, tmp_path):
        # Two runs differing only by prompt must be distinguishable later.
        config = Config()
        config.data.data_dir = str(tmp_path)
        config.data.image_folder = str(tmp_path / "images")
        config.inference.system_prompt = "Trả lời ngắn gọn."
        config.inference.max_new_tokens = 128

        from vivqa.data.prepare import prepare

        prepare(config, records=[{"id": 0, "description": "x", "conversations": [
            {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}])

        result = evaluate(FakeModel(), config, split="train")
        assert result["system_prompt"] == "Trả lời ngắn gọn."
        assert result["max_new_tokens"] == 128

    def test_style_prompt_reaches_the_model(self, tmp_path):
        config = Config()
        config.data.data_dir = str(tmp_path)
        config.data.image_folder = str(tmp_path / "images")
        config.inference.system_prompt = "Một câu duy nhất."

        from vivqa.data.prepare import prepare

        prepare(config, records=[{"id": 0, "description": "x", "conversations": [
            {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}])

        model = FakeModel()
        evaluate(model, config, split="train")
        assert model.seen[0][0]["role"] == "system"
