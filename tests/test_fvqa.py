"""FVQA loading: facts, questions, official splits, and prepare_fvqa.

Fixtures reproduce the on-disk layout of the real release (verified by
downloading and extracting it directly, 2026-08-24):

    Name_Lists/{train,test}_list_{fold}.txt
    new_dataset_release/all_fact_triples_release.json
    new_dataset_release/all_qs_dict_release.json
    new_dataset_release/images/*.{jpg,JPEG}

Real counts differ from FVQA's own README (225,434 facts vs. the
documented 193,449; 5,826 questions vs. 5,286) and `fact` is a list, not
a single ID — this file's fixtures follow what was actually downloaded.
"""

import json
import os

import pytest

from fvqa.config import Config
from fvqa.data.fvqa import (
    build_sample,
    image_path,
    load_facts,
    load_questions,
    load_split_images,
    prepare_fvqa,
)


def write_release(root: str, facts: dict, questions: dict, folds: dict) -> None:
    """Write a minimal FVQA release to `root`.

    `folds` maps fold number -> {"train": [...], "test": [...]} of image
    filenames.
    """
    release_dir = os.path.join(root, "new_dataset_release")
    name_lists_dir = os.path.join(root, "Name_Lists")
    os.makedirs(release_dir, exist_ok=True)
    os.makedirs(name_lists_dir, exist_ok=True)

    with open(os.path.join(release_dir, "all_fact_triples_release.json"), "w") as f:
        json.dump(facts, f)
    with open(os.path.join(release_dir, "all_qs_dict_release.json"), "w") as f:
        json.dump(questions, f)

    for fold, images in folds.items():
        for split, names in images.items():
            path = os.path.join(name_lists_dir, f"{split}_list_{fold}.txt")
            with open(path, "w") as f:
                f.write("\n".join(names) + "\n")


FACTS = {
    "conceptnet/e/1": {
        "KB": "conceptnet",
        "e1": "/c/en/trumpet",
        "e2": "/c/en/jazz_club",
        "e1_label": "a trumpet",
        "e2_label": "a jazz club",
        "r": "/r/AtLocation",
        "surface": "You are likely to find [[a trumpet]] in [[a jazz club]]",
    },
    "webchild/e/2": {
        "KB": "webchild",
        "e1": "gove truck",
        "e2": "car",
        "e1_label": "gove truck",
        "e2_label": "car",
        "r": "long#f",
        "surface": "[[gove truck]] are much longer than [[car]]",
    },
}

QUESTIONS = {
    "270": {
        "fact_surface": "You are likely to find [[a trumpet]] in [[a jazz club]]",
        "ans_source": "image",
        "answer": "trumpet",
        "question": "Which object can be found in a jazz club",
        "img_file": "ILSVRC2012_test_00050748.JPEG",
        "visual_concept": "obj",
        "kb_source": "conceptnet",
        "fact": ["conceptnet/e/1"],
        "question_id": "270",
    },
    "999": {
        "fact_surface": "[[gove truck]] are much longer than [[car]]",
        "ans_source": "kb",
        "answer": "gove truck",
        "question": "Which vehicle is longer than a car",
        "img_file": "COCO_val2014_000000000136.jpg",
        "visual_concept": "obj",
        "kb_source": "webchild",
        "fact": ["webchild/e/2"],
        "question_id": "999",
    },
}


@pytest.fixture
def fvqa_root(tmp_path):
    root = str(tmp_path / "fvqa")
    write_release(
        root,
        FACTS,
        QUESTIONS,
        folds={0: {"train": ["ILSVRC2012_test_00050748.JPEG"], "test": ["COCO_val2014_000000000136.jpg"]}},
    )
    return root


class TestLoaders:
    def test_load_facts(self, fvqa_root):
        facts = load_facts(fvqa_root)
        assert len(facts) == 2
        assert facts["conceptnet/e/1"]["e1_label"] == "a trumpet"

    def test_load_questions(self, fvqa_root):
        questions = load_questions(fvqa_root)
        assert len(questions) == 2
        assert questions["270"]["answer"] == "trumpet"

    def test_missing_facts_file_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Download"):
            load_facts(str(tmp_path / "absent"))

    def test_load_split_images(self, fvqa_root):
        train = load_split_images(fvqa_root, fold=0, split="train")
        test = load_split_images(fvqa_root, fold=0, split="test")
        assert train == {"ILSVRC2012_test_00050748.JPEG"}
        assert test == {"COCO_val2014_000000000136.jpg"}

    def test_invalid_split_name_is_rejected(self, fvqa_root):
        with pytest.raises(ValueError, match="train.*test"):
            load_split_images(fvqa_root, fold=0, split="val")

    def test_missing_fold_is_reported(self, fvqa_root):
        with pytest.raises(FileNotFoundError):
            load_split_images(fvqa_root, fold=3, split="train")

    def test_image_path_preserves_extension_case(self, fvqa_root):
        # COCO ships lowercase .jpg, ImageNet ships uppercase .JPEG — the
        # path builder must not normalize either away.
        p = image_path(fvqa_root, "ILSVRC2012_test_00050748.JPEG")
        assert p.endswith("ILSVRC2012_test_00050748.JPEG")


class TestBuildSample:
    @pytest.fixture
    def config(self):
        return Config()

    def test_produces_the_shared_fvqa_sample_shape(self, config):
        config.data.grounding.enabled = False
        sample = build_sample("270", QUESTIONS["270"], FACTS, config.data.grounding)
        assert sample["id"] == "fvqa_270"
        assert sample["image"] == "ILSVRC2012_test_00050748.JPEG"
        assert sample["conversations"][0]["value"] == "<image>\nWhich object can be found in a jazz club"
        assert sample["conversations"][1] == {"from": "gpt", "value": "trumpet"}

    def test_carries_the_oracle_fact_id_and_visual_concept(self, config):
        sample = build_sample("270", QUESTIONS["270"], FACTS, config.data.grounding)
        assert sample["fvqa_fact_ids"] == ["conceptnet/e/1"]
        assert sample["fvqa_visual_concept"] == "obj"

    def test_grounding_injects_the_oracle_facts_surface_text(self, config):
        config.data.grounding.enabled = True
        sample = build_sample("270", QUESTIONS["270"], FACTS, config.data.grounding)
        assert "a trumpet" in sample["conversations"][0]["value"]
        assert "jazz club" in sample["conversations"][0]["value"]

    def test_grounding_off_leaves_the_question_bare(self, config):
        config.data.grounding.enabled = False
        sample = build_sample("270", QUESTIONS["270"], FACTS, config.data.grounding)
        assert "trumpet" not in sample["conversations"][0]["value"]

    def test_falls_back_to_the_questions_own_fact_surface_if_the_id_is_dangling(self, config):
        config.data.grounding.enabled = True
        question = dict(QUESTIONS["270"], fact=["missing/id"])
        sample = build_sample("270", question, FACTS, config.data.grounding)
        assert "jazz club" in sample["conversations"][0]["value"]


class TestPrepareFvqa:
    @pytest.fixture
    def config(self, fvqa_root):
        config = Config()
        config.data.root = fvqa_root
        config.data.data_dir = fvqa_root  # write outputs alongside the fixture
        return config

    def test_writes_the_official_test_fold_untouched(self, config):
        counts = prepare_fvqa(config)
        test_samples = json.load(open(config.data.split_file("test")))
        assert len(test_samples) == 1
        assert test_samples[0]["image"] == "COCO_val2014_000000000136.jpg"
        assert counts["test"] == 1

    def test_train_fold_is_split_into_train_and_val(self, config):
        # Only one train-fold image in this fixture; with a nonzero val
        # weight it must land in exactly one of train/val, not both.
        counts = prepare_fvqa(config)
        assert counts["train"] + counts["val"] == 1

    def test_question_count_is_reported(self, config):
        counts = prepare_fvqa(config)
        assert counts["questions"] == 2

    def test_limit_stops_early(self, config):
        counts = prepare_fvqa(config, limit=1)
        assert counts["questions"] == 1

    def test_unknown_split_image_is_skipped_not_fatal(self, config, fvqa_root, caplog):
        questions = dict(QUESTIONS)
        questions["orphan"] = dict(QUESTIONS["270"], img_file="not_in_any_split.jpg")
        with open(
            os.path.join(fvqa_root, "new_dataset_release", "all_qs_dict_release.json"), "w"
        ) as f:
            json.dump(questions, f)

        counts = prepare_fvqa(config)
        assert counts["questions"] == 3
        assert counts["test"] + counts["train"] + counts["val"] == 2
        assert "unknown-split" in caplog.text

    def test_test_fold_never_receives_val_samples(self, fvqa_root):
        # A larger fixture: 4 train-fold images, 1 test-fold image. The
        # test split must stay exactly 1 regardless of the configured
        # train/val ratio — assign_splits must never touch it.
        facts = FACTS
        questions = {
            str(i): {
                "fact_surface": "x", "ans_source": "image", "answer": "x",
                "question": f"q{i}", "img_file": f"train_{i}.jpg",
                "visual_concept": "obj", "kb_source": "conceptnet",
                "fact": ["conceptnet/e/1"], "question_id": str(i),
            }
            for i in range(4)
        }
        questions["test_q"] = dict(QUESTIONS["270"], img_file="test_0.jpg")

        write_release(
            fvqa_root, facts, questions,
            folds={0: {"train": [f"train_{i}.jpg" for i in range(4)], "test": ["test_0.jpg"]}},
        )
        config = Config()
        config.data.root = fvqa_root
        config.data.data_dir = fvqa_root

        counts = prepare_fvqa(config)
        assert counts["test"] == 1
        assert counts["train"] + counts["val"] == 4
