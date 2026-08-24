"""assign_splits and write_split: the loader-agnostic pieces every dataset
loader (currently just FVQA) builds samples on top of."""

import json

import pytest

from fvqa.config import Config
from fvqa.data.samples import assign_splits, write_split


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


class TestWriteSplit:
    def test_writes_valid_json(self, tmp_path):
        path = str(tmp_path / "train.json")
        write_split([{"id": "a"}, {"id": "b"}], path)
        assert json.loads(open(path, encoding="utf-8").read()) == [{"id": "a"}, {"id": "b"}]

    def test_creates_missing_parent_directories(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "train.json")
        write_split([], path)
        assert json.loads(open(path, encoding="utf-8").read()) == []

    def test_preserves_non_ascii_text(self, tmp_path):
        path = str(tmp_path / "train.json")
        write_split([{"answer": "Bến Thành"}], path)
        assert "Bến Thành" in open(path, encoding="utf-8").read()
