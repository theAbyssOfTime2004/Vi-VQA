"""The import-graph walk behind scripts/check_trainer_imports.py.

Two packages the trainer needs (`trl`, `ujson`) were missing from the
Modal image, and each surfaced separately three minutes into a GPU
container. Neither was reachable by reading the SFT code: both come in
because `src/dataset/__init__.py` imports the DPO dataset alongside the
SFT one. A flat grep of the SFT path finds neither; following the graph
finds both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_trainer_imports import third_party_imports  # noqa: E402


@pytest.fixture
def trainer_tree(tmp_path):
    """A miniature trainer laid out like the real one.

    `dataset/__init__.py` pulling in a sibling it does not need is the
    shape that hid the real bug, so it is the shape reproduced here.
    """
    src = tmp_path / "src"
    (src / "train").mkdir(parents=True)
    (src / "dataset").mkdir()

    (src / "train" / "__init__.py").write_text("")
    (src / "train" / "train_sft.py").write_text(
        "import torch\n"
        "from dataset import make_supervised_data_module\n"
        "from params import TrainingArguments\n"
    )
    (src / "params.py").write_text("from trl import DPOConfig\n")
    (src / "dataset" / "__init__.py").write_text(
        "from .sft_dataset import make_supervised_data_module\n"
        "from .dpo_dataset import make_dpo_data_module\n"
    )
    (src / "dataset" / "sft_dataset.py").write_text("import torch\n")
    (src / "dataset" / "dpo_dataset.py").write_text("import ujson as json\n")
    return src


def imports_of(src):
    return third_party_imports(src, Path("train") / "train_sft.py")


class TestThirdPartyImports:
    def test_finds_direct_imports(self, trainer_tree):
        assert "torch" in imports_of(trainer_tree)

    def test_follows_local_packages_into_siblings(self, trainer_tree):
        # ujson is imported by dpo_dataset, which the SFT run reaches only
        # because dataset/__init__ imports it. This is the real case.
        assert "ujson" in imports_of(trainer_tree)

    def test_follows_local_modules(self, trainer_tree):
        # trl arrives via params.py, two hops from the entry point.
        assert "trl" in imports_of(trainer_tree)

    def test_records_the_chain_that_reached_each_package(self, trainer_tree):
        chain = imports_of(trainer_tree)["ujson"]
        assert chain[0] == "train_sft"
        assert "dataset.dpo_dataset" in chain

    def test_excludes_the_standard_library(self, trainer_tree):
        (trainer_tree / "train" / "train_sft.py").write_text(
            "import os\nimport json\nimport torch\n"
        )
        found = imports_of(trainer_tree)
        assert "os" not in found
        assert "json" not in found
        assert "torch" in found

    def test_excludes_the_trainers_own_modules(self, trainer_tree):
        found = imports_of(trainer_tree)
        assert "dataset" not in found
        assert "params" not in found

    def test_handles_relative_imports(self, trainer_tree):
        # `from .dpo_dataset import ...` must resolve inside the package,
        # not be mistaken for a third-party name.
        assert "dpo_dataset" not in imports_of(trainer_tree)

    def test_survives_an_import_cycle(self, trainer_tree):
        (trainer_tree / "dataset" / "sft_dataset.py").write_text(
            "import params\nimport torch\n"
        )
        (trainer_tree / "params.py").write_text(
            "from trl import DPOConfig\nfrom dataset import x\n"
        )
        found = imports_of(trainer_tree)
        assert {"torch", "trl", "ujson"} <= set(found)


class TestAgainstTheRealTrainer:
    """Runs only when the pinned trainer happens to be checked out."""

    @pytest.fixture
    def real_trainer(self):
        src = Path(__file__).resolve().parent.parent / "Qwen-VL-Series-Finetune" / "src"
        if not (src / "train" / "train_sft.py").is_file():
            pytest.skip("trainer repo not cloned here")
        return src

    def test_trl_and_ujson_are_reachable_from_the_sft_entrypoint(self, real_trainer):
        # The regression itself: these two are what the image was missing.
        found = imports_of(real_trainer)
        assert "trl" in found
        assert "ujson" in found

    def test_the_known_dependency_set_is_stable(self, real_trainer):
        # If the pinned trainer starts importing something new, that is a
        # decision to make deliberately — not a surprise on a GPU.
        assert set(imports_of(real_trainer)) == {
            "accelerate",
            "deepspeed",
            "numpy",
            "peft",
            "qwen_vl_utils",
            "torch",
            "transformers",
            "trl",
            "ujson",
        }
