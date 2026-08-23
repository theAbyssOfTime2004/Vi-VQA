"""Dataset preparation: QA extraction, splitting and file output."""

import json

import pytest

from vivqa.config import Config
from vivqa.data.prepare import (
    IMAGE_TOKEN,
    assign_splits,
    build_samples,
    extract_qa_pairs,
    image_filename,
    prepare,
)


def make_record(record_id, num_pairs=2, description="Mô tả ảnh."):
    conversations = []
    for i in range(num_pairs):
        conversations.append({"role": "user", "content": f"Câu hỏi {i} về ảnh {record_id}?"})
        conversations.append({"role": "assistant", "content": f"Trả lời {i}."})
    return {"id": record_id, "description": description, "conversations": conversations}


class TestExtractQAPairs:
    def test_extracts_multi_turn_pairs(self):
        pairs = extract_qa_pairs(
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ]
        )
        assert pairs == [("Q1", "A1"), ("Q2", "A2")]

    def test_accepts_the_from_value_spelling(self):
        # The dataset uses both key spellings depending on the split.
        pairs = extract_qa_pairs(
            [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]
        )
        assert pairs == [("Q", "A")]

    def test_unanswered_question_is_dropped(self):
        assert extract_qa_pairs([{"role": "user", "content": "Q"}]) == []

    def test_orphan_answer_is_dropped(self):
        assert extract_qa_pairs([{"role": "assistant", "content": "A"}]) == []

    def test_consecutive_questions_keep_the_last(self):
        pairs = extract_qa_pairs(
            [
                {"role": "user", "content": "Q1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A"},
            ]
        )
        assert pairs == [("Q2", "A")]

    def test_blank_sides_are_dropped(self):
        pairs = extract_qa_pairs(
            [
                {"role": "user", "content": "   "},
                {"role": "assistant", "content": "A"},
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": ""},
            ]
        )
        assert pairs == []

    def test_empty_conversation(self):
        assert extract_qa_pairs([]) == []


class TestBuildSamples:
    def test_produces_one_sample_per_pair(self):
        samples = build_samples(7, make_record(7, 3)["conversations"], None, Config().data)
        assert len(samples) == 3
        assert [s["id"] for s in samples] == ["7_0", "7_1", "7_2"]
        assert all(s["image"] == image_filename(7) for s in samples)

    def test_image_token_prefixes_the_user_turn(self):
        samples = build_samples(1, make_record(1, 1)["conversations"], None, Config().data)
        assert samples[0]["conversations"][0]["value"].startswith(f"{IMAGE_TOKEN}\n")
        assert samples[0]["conversations"][1]["from"] == "gpt"

    def test_grounding_off_leaves_the_question_alone(self):
        data = Config().data
        data.grounding.enabled = False
        samples = build_samples(1, make_record(1, 1)["conversations"], "Mô tả.", data)
        assert "Mô tả." not in samples[0]["conversations"][0]["value"]

    def test_prefix_grounding_folds_context_into_the_user_turn(self):
        data = Config().data
        data.grounding.enabled = True
        samples = build_samples(1, make_record(1, 1)["conversations"], "Chợ Bến Thành.", data)
        turns = samples[0]["conversations"]
        assert len(turns) == 2
        assert "Chợ Bến Thành." in turns[0]["value"]

    def test_system_grounding_adds_a_system_turn(self):
        data = Config().data
        data.grounding.enabled = True
        data.grounding.mode = "system"
        samples = build_samples(1, make_record(1, 1)["conversations"], "Chợ Bến Thành.", data)
        turns = samples[0]["conversations"]
        assert [t["from"] for t in turns] == ["system", "human", "gpt"]
        assert "Chợ Bến Thành." in turns[0]["value"]
        # The question must not be duplicated into the system turn.
        assert "Câu hỏi" not in turns[0]["value"]

    def test_missing_description_falls_back_to_the_bare_question(self):
        data = Config().data
        data.grounding.enabled = True
        samples = build_samples(1, make_record(1, 1)["conversations"], None, data)
        assert samples[0]["conversations"][0]["value"] == f"{IMAGE_TOKEN}\nCâu hỏi 0 về ảnh 1?"


class TestAssignSplits:
    def test_every_image_lands_in_exactly_one_split(self):
        config = Config()
        assignment = assign_splits(range(200), config.data.splits, seed=42)
        assert len(assignment) == 200
        assert set(assignment.values()) <= {"train", "val", "test"}

    def test_proportions_are_respected(self):
        config = Config()
        assignment = assign_splits(range(1000), config.data.splits, seed=42)
        counts = {name: list(assignment.values()).count(name) for name in ("train", "val", "test")}
        assert counts["val"] == 50
        assert counts["test"] == 50
        assert counts["train"] == 900

    def test_assignment_is_stable_across_input_order(self):
        config = Config()
        forward = assign_splits(range(100), config.data.splits, seed=7)
        backward = assign_splits(reversed(range(100)), config.data.splits, seed=7)
        assert forward == backward

    def test_different_seeds_give_different_assignments(self):
        config = Config()
        assert assign_splits(range(100), config.data.splits, 1) != assign_splits(
            range(100), config.data.splits, 2
        )

    def test_train_absorbs_the_rounding_remainder(self):
        # 7 images against 90/5/5 cannot divide evenly; nothing may be lost.
        config = Config()
        assignment = assign_splits(range(7), config.data.splits, seed=3)
        assert len(assignment) == 7
        assert list(assignment.values()).count("train") == 7


class TestPrepare:
    @pytest.fixture
    def config(self, tmp_path):
        config = Config()
        config.data.data_dir = str(tmp_path)
        config.data.image_folder = str(tmp_path / "images")
        return config

    def test_writes_every_split(self, config):
        counts = prepare(config, records=[make_record(i) for i in range(100)])
        assert counts["train"] + counts["val"] + counts["test"] == 200
        assert counts["records"] == 100

    def test_no_image_appears_in_two_splits(self, config):
        # The previous pipeline shuffled QA pairs before splitting, so
        # questions about one photograph landed on both sides of the
        # train/val boundary and validation loss read better than it was.
        prepare(config, records=[make_record(i) for i in range(100)])

        seen = {}
        for split in ("train", "val", "test"):
            with open(config.data.split_file(split), encoding="utf-8") as handle:
                for sample in json.load(handle):
                    seen.setdefault(sample["image_id"], set()).add(split)

        assert not [image for image, splits in seen.items() if len(splits) > 1]

    def test_limit_stops_early(self, config):
        counts = prepare(config, limit=10, records=[make_record(i) for i in range(100)])
        assert counts["records"] == 10

    def test_records_without_conversations_are_skipped(self, config):
        records = [make_record(0), {"id": 1, "description": "x", "conversations": []}]
        counts = prepare(config, records=records)
        assert counts["train"] == 2

    def test_output_is_byte_stable_across_runs(self, config, tmp_path):
        records = [make_record(i) for i in range(50)]
        prepare(config, records=records)
        first = (tmp_path / "train.json").read_text(encoding="utf-8")
        prepare(config, records=list(reversed(records)))
        assert (tmp_path / "train.json").read_text(encoding="utf-8") == first

    def test_zero_weighted_split_is_not_written(self, config):
        config.data.splits.train = 1.0
        config.data.splits.val = 0.0
        config.data.splits.test = 0.0
        counts = prepare(config, records=[make_record(i) for i in range(10)])
        assert "val" not in counts and "test" not in counts
