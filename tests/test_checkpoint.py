"""Checkpoint detection and the LoRA load path.

No model is downloaded here. Detection is pure filesystem inspection, and
the loading tests substitute fakes for `from_pretrained` / `PeftModel` —
the thing under test is the *order and completeness* of the load, which
is exactly what an 8B download would not tell you anything more about.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from fvqa.checkpoint import (
    CheckpointInfo,
    detect_checkpoint_type,
    inspect_checkpoint,
    normalize_non_lora_keys,
)
from fvqa.config import Config


def write_adapter(directory, base_model="Qwen/Qwen3-VL-8B-Instruct", weights=True, non_lora=False):
    """A LoRA checkpoint shaped the way the trainer actually writes one."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model, "r": 128}), encoding="utf-8"
    )
    if weights:
        (directory / "adapter_model.safetensors").write_bytes(b"")
    # The trainer writes config.json into an adapter directory too, via
    # model.config.save_pretrained — the whole reason detection cannot key
    # off config.json.
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}), encoding="utf-8")
    if non_lora:
        (directory / "non_lora_state_dict.bin").write_bytes(b"")
    return directory


def write_full(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}), encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"")
    return directory


class TestDetectCheckpointType:
    def test_a_non_directory_is_treated_as_a_model_id(self, tmp_path):
        assert detect_checkpoint_type("Qwen/Qwen3-VL-8B-Instruct") == "hf_id"
        assert detect_checkpoint_type(str(tmp_path / "does-not-exist")) == "hf_id"

    def test_full_weights_directory(self, tmp_path):
        assert detect_checkpoint_type(str(write_full(tmp_path / "full"))) == "full"

    def test_adapter_directory(self, tmp_path):
        assert detect_checkpoint_type(str(write_adapter(tmp_path / "lora"))) == "peft"

    def test_adapter_wins_over_the_config_json_sitting_next_to_it(self, tmp_path):
        # The trainer saves both. Keying off config.json would send an
        # adapter directory down the full-weights path, where it loads a
        # model with no trained weights in it at all.
        directory = write_adapter(tmp_path / "lora")
        assert (directory / "config.json").exists()
        assert detect_checkpoint_type(str(directory)) == "peft"

    def test_adapter_config_without_weights_is_an_error(self, tmp_path):
        directory = write_adapter(tmp_path / "broken", weights=False)
        with pytest.raises(FileNotFoundError, match="adapter weights are missing"):
            detect_checkpoint_type(str(directory))

    def test_accepts_the_bin_flavour_of_adapter_weights(self, tmp_path):
        directory = write_adapter(tmp_path / "lora", weights=False)
        (directory / "adapter_model.bin").write_bytes(b"")
        assert detect_checkpoint_type(str(directory)) == "peft"


class TestInspectCheckpoint:
    def test_reads_the_base_model_from_the_adapter_config(self, tmp_path):
        directory = write_adapter(tmp_path / "lora", base_model="Qwen/Qwen3-VL-8B-Instruct")
        info = inspect_checkpoint(str(directory))
        assert info.kind == "peft"
        assert info.base_model_id == "Qwen/Qwen3-VL-8B-Instruct"
        assert info.is_adapter

    def test_finds_non_lora_state_dict_when_present(self, tmp_path):
        directory = write_adapter(tmp_path / "lora", non_lora=True)
        info = inspect_checkpoint(str(directory))
        assert info.non_lora_path is not None
        assert info.non_lora_path.endswith("non_lora_state_dict.bin")

    def test_absent_non_lora_state_dict_is_reported_as_none(self, tmp_path):
        info = inspect_checkpoint(str(write_adapter(tmp_path / "lora")))
        assert info.non_lora_path is None

    def test_adapter_config_naming_no_base_model_is_an_error(self, tmp_path):
        directory = tmp_path / "lora"
        directory.mkdir()
        (directory / "adapter_config.json").write_text(json.dumps({"r": 128}), encoding="utf-8")
        (directory / "adapter_model.safetensors").write_bytes(b"")
        with pytest.raises(ValueError, match="does not name a base model"):
            inspect_checkpoint(str(directory))

    def test_unreadable_adapter_config_is_an_error(self, tmp_path):
        directory = tmp_path / "lora"
        directory.mkdir()
        (directory / "adapter_config.json").write_text("{not json", encoding="utf-8")
        (directory / "adapter_model.safetensors").write_bytes(b"")
        with pytest.raises(ValueError, match="could not read"):
            inspect_checkpoint(str(directory))

    def test_a_model_id_carries_no_base_model_or_processor(self, tmp_path):
        info = inspect_checkpoint("Qwen/Qwen3-VL-8B-Instruct")
        assert info.kind == "hf_id"
        assert info.base_model_id is None
        assert not info.has_processor


class TestProcessorSource:
    def test_prefers_the_checkpoints_own_processor(self, tmp_path):
        directory = write_adapter(tmp_path / "lora")
        (directory / "preprocessor_config.json").write_text("{}", encoding="utf-8")
        info = inspect_checkpoint(str(directory))
        assert info.has_processor
        assert info.processor_source(fallback="Qwen/base") == str(directory)

    def test_falls_back_to_the_base_model_when_there_is_none(self, tmp_path):
        info = inspect_checkpoint(str(write_adapter(tmp_path / "lora")))
        assert not info.has_processor
        assert info.processor_source() == "Qwen/Qwen3-VL-8B-Instruct"

    def test_explicit_fallback_wins_over_the_adapter_config(self, tmp_path):
        info = inspect_checkpoint(str(write_adapter(tmp_path / "lora")))
        assert info.processor_source(fallback="Qwen/other") == "Qwen/other"

    def test_no_processor_and_no_fallback_is_an_error(self):
        info = CheckpointInfo(kind="full", path="/somewhere", has_processor=False)
        with pytest.raises(ValueError, match="no processor config"):
            info.processor_source()


class TestNormalizeNonLoraKeys:
    def test_strips_the_peft_base_model_prefix(self):
        assert normalize_non_lora_keys({"base_model.visual.merger.weight": 1}) == {
            "visual.merger.weight": 1
        }

    def test_strips_the_inner_model_prefix_only_when_doubled(self):
        # `base_model.model.model.x` -> `model.model.x` -> `model.x`
        result = normalize_non_lora_keys({"base_model.model.model.layers.0.weight": 1})
        assert result == {"model.layers.0.weight": 1}

    def test_leaves_a_single_model_prefix_alone(self):
        # Nothing starts with `model.model.`, so the second strip must not
        # fire — doing it anyway would mangle keys that were already right.
        result = normalize_non_lora_keys({"base_model.model.visual.weight": 1})
        assert result == {"model.visual.weight": 1}

    def test_passes_through_keys_with_no_prefix(self):
        assert normalize_non_lora_keys({"visual.weight": 1}) == {"visual.weight": 1}


class FakeLoadResult:
    def __init__(self, unexpected_keys=()):
        self.unexpected_keys = list(unexpected_keys)
        self.missing_keys = []


class FakeModel:
    def __init__(self, unexpected_keys=()):
        self.loaded_state_dicts = []
        self._unexpected = unexpected_keys

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dicts.append((state_dict, strict))
        return FakeLoadResult(self._unexpected)


@pytest.fixture
def fake_torch(monkeypatch):
    """A stand-in `torch` whose `load` returns a fixed state dict."""
    module = types.ModuleType("torch")
    module.loaded_from = []

    def fake_load(path, map_location=None, weights_only=None):
        module.loaded_from.append(path)
        return {"base_model.model.visual.merger.weight": "tensor"}

    module.load = fake_load
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


class TestLoadNonLoraWeights:
    def test_loads_normalized_keys_non_strictly(self, fake_torch):
        from fvqa.checkpoint import load_non_lora_weights

        model = FakeModel()
        count = load_non_lora_weights(model, "/checkpoints/non_lora_state_dict.bin")

        assert count == 1
        state_dict, strict = model.loaded_state_dicts[0]
        # strict=False is required (LoRA tensors live in a separate file).
        assert strict is False
        assert list(state_dict) == ["model.visual.merger.weight"]

    def test_keys_matching_nothing_raise_instead_of_being_dropped(self, fake_torch):
        # These are trained weights. Silently discarding them yields a
        # model that runs with the untrained module and never says so.
        from fvqa.checkpoint import load_non_lora_weights

        model = FakeModel(unexpected_keys=["visual.merger.weight"])
        with pytest.raises(RuntimeError, match="discarded silently"):
            load_non_lora_weights(model, "/checkpoints/non_lora_state_dict.bin")


# --------------------------------------------------------------------------
# load_model dispatch
# --------------------------------------------------------------------------


class RecordingModelClass:
    """Stands in for Qwen3VLForConditionalGeneration."""

    def __init__(self, log):
        self.log = log

    def from_pretrained(self, path, **kwargs):
        self.log.append(("base_from_pretrained", path))
        return FakeModel()


@pytest.fixture
def fake_stack(monkeypatch, fake_torch):
    """Fake transformers + peft, recording the order of every load call."""
    log: list[tuple] = []

    fake_torch.bfloat16 = "bfloat16-dtype"

    def recording_load(path, map_location=None, weights_only=None):
        log.append(("torch_load", path))
        return {"base_model.model.visual.merger.weight": "tensor"}

    fake_torch.load = recording_load

    transformers = types.ModuleType("transformers")
    transformers.__version__ = "4.57.0"
    transformers.Qwen3VLForConditionalGeneration = RecordingModelClass(log)

    class AutoProcessor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            log.append(("processor_from_pretrained", path))
            return f"processor:{path}"

    transformers.AutoProcessor = AutoProcessor

    peft = types.ModuleType("peft")

    class PeftModel:
        @staticmethod
        def from_pretrained(model, adapter_path, **kwargs):
            log.append(("peft_from_pretrained", adapter_path))
            return f"peft-wrapped:{adapter_path}"

    peft.PeftModel = PeftModel

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    return log


class TestLoadModelDispatch:
    def test_full_checkpoint_never_goes_through_peft(self, tmp_path, fake_stack):
        from fvqa.model import load_model

        directory = write_full(tmp_path / "full")
        load_model(str(directory), Config())

        kinds = [entry[0] for entry in fake_stack]
        assert "peft_from_pretrained" not in kinds
        assert ("base_from_pretrained", str(directory)) in fake_stack

    def test_model_id_never_goes_through_peft(self, fake_stack):
        from fvqa.model import load_model

        load_model("Qwen/Qwen3-VL-8B-Instruct", Config())

        kinds = [entry[0] for entry in fake_stack]
        assert "peft_from_pretrained" not in kinds

    def test_adapter_loads_the_base_model_first_then_the_adapter(self, tmp_path, fake_stack):
        from fvqa.model import load_model

        directory = write_adapter(tmp_path / "lora", base_model="Qwen/Qwen3-VL-8B-Instruct")
        load_model(str(directory), Config())

        kinds = [entry[0] for entry in fake_stack]
        assert kinds.index("base_from_pretrained") < kinds.index("peft_from_pretrained")
        # The base model comes from adapter_config.json, not from the
        # adapter directory — loading the adapter path as a base model is
        # the mistake this whole module exists to prevent.
        assert ("base_from_pretrained", "Qwen/Qwen3-VL-8B-Instruct") in fake_stack
        assert ("peft_from_pretrained", str(directory)) in fake_stack

    def test_non_lora_weights_load_between_the_base_and_the_adapter(self, tmp_path, fake_stack):
        # Order is not cosmetic: the tensors are keyed against the
        # unwrapped base model, so loading them after PEFT wraps it would
        # match nothing at all, and match nothing *quietly*.
        from fvqa.model import load_model

        directory = write_adapter(tmp_path / "lora", non_lora=True)
        load_model(str(directory), Config())

        kinds = [entry[0] for entry in fake_stack]
        assert kinds.index("base_from_pretrained") < kinds.index("torch_load")
        assert kinds.index("torch_load") < kinds.index("peft_from_pretrained")

    def test_an_adapter_without_non_lora_weights_skips_that_step(self, tmp_path, fake_stack):
        from fvqa.model import load_model

        load_model(str(write_adapter(tmp_path / "lora")), Config())

        assert "torch_load" not in [entry[0] for entry in fake_stack]

    def test_base_model_override_wins_over_the_adapter_config(self, tmp_path, fake_stack):
        from fvqa.model import load_model

        directory = write_adapter(tmp_path / "lora", base_model="Qwen/stale-local-path")
        load_model(str(directory), Config(), base_model_id="Qwen/Qwen3-VL-8B-Instruct")

        assert ("base_from_pretrained", "Qwen/Qwen3-VL-8B-Instruct") in fake_stack
        assert ("base_from_pretrained", "Qwen/stale-local-path") not in fake_stack

    def test_processor_falls_back_to_the_base_model_for_a_bare_adapter(
        self, tmp_path, fake_stack
    ):
        from fvqa.model import load_model

        directory = write_adapter(tmp_path / "lora", base_model="Qwen/Qwen3-VL-8B-Instruct")
        load_model(str(directory), Config())

        assert ("processor_from_pretrained", "Qwen/Qwen3-VL-8B-Instruct") in fake_stack

    def test_processor_comes_from_the_checkpoint_when_it_has_one(self, tmp_path, fake_stack):
        from fvqa.model import load_model

        directory = write_adapter(tmp_path / "lora")
        (directory / "preprocessor_config.json").write_text("{}", encoding="utf-8")
        load_model(str(directory), Config())

        assert ("processor_from_pretrained", str(directory)) in fake_stack
