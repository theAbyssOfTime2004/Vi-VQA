"""Configuration loading, overrides and validation."""

import pytest
import yaml

from fvqa.config import Config, ConfigError, GroundingConfig, load_config


def write_config(tmp_path, mapping):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(mapping, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_loads_repository_config(repo_config):
    config = load_config(repo_config)
    assert config.model.model_id.startswith("Qwen/Qwen3-VL")
    assert config.training.effective_batch_size == (
        config.training.per_device_train_batch_size
        * config.training.gradient_accumulation_steps
    )


def test_defaults_apply_to_missing_sections(tmp_path):
    config = load_config(write_config(tmp_path, {"project_name": "X"}))
    assert config.project_name == "X"
    assert config.seed == Config().seed
    assert config.model.lora.rank == 128


def test_trainer_revision_defaults_to_a_real_pinned_commit():
    # A run's reproducibility depends on this never silently tracking
    # HEAD — the default must be an actual commit SHA, not a branch name.
    config = Config()
    assert len(config.trainer.revision) == 40
    assert all(c in "0123456789abcdef" for c in config.trainer.revision)


def test_unknown_key_is_rejected(tmp_path):
    # A typo must fail loudly rather than be silently ignored, which is how
    # a run finishes with a hyperparameter the config never applied.
    path = write_config(tmp_path, {"training": {"learnin_rate": 1e-5}})
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "absent.yaml"))


@pytest.mark.parametrize(
    "mapping, message",
    [
        ({"data": {"splits": {"train": 0.8, "val": 0.3, "test": 0.1}}}, "sum to 1.0"),
        ({"training": {"bf16": True, "fp16": True}}, "mutually exclusive"),
        ({"training": {"num_train_epochs": 0}}, "must be positive"),
        ({"model": {"torch_dtype": "int8"}}, "torch_dtype"),
        ({"model": {"image_min_pixels": 999999999}}, "image_min_pixels"),
        ({"inference": {"top_p": 1.5}}, "top_p"),
        ({"evaluation": {"metrics": ["meteor"]}}, "unknown metric"),
        (
            {"training": {"freeze_llm": True, "freeze_vision_tower": True, "freeze_merger": True}},
            "nothing would train",
        ),
        ({"model": {"lora": {"enabled": True}, "qlora": {"enabled": True}}}, "not both"),
        ({"trainer": {"revision": ""}}, "revision must not be empty"),
        ({"trainer": {"repo_url": ""}}, "repo_url must not be empty"),
    ],
)
def test_invalid_values_are_rejected(tmp_path, mapping, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write_config(tmp_path, mapping))


def test_type_errors_are_reported_with_path(tmp_path):
    path = write_config(tmp_path, {"training": {"num_train_epochs": "three"}})
    with pytest.raises(ConfigError, match="training.num_train_epochs"):
        load_config(path)


def test_overrides_are_parsed_as_yaml(repo_config):
    config = load_config(
        repo_config,
        overrides=[
            "training.num_train_epochs=5",
            "training.bf16=false",
            "training.learning_rate=1e-4",
            "data.grounding.enabled=true",
        ],
    )
    assert config.training.num_train_epochs == 5
    assert config.training.bf16 is False
    assert config.training.learning_rate == pytest.approx(1e-4)
    assert config.data.grounding.enabled is True


def test_override_without_equals_is_rejected(repo_config):
    with pytest.raises(ConfigError, match="key=value"):
        load_config(repo_config, overrides=["training.bf16"])


def test_optional_field_accepts_null(tmp_path):
    config = load_config(write_config(tmp_path, {"training": {"deepspeed": None}}))
    assert config.training.deepspeed is None


def test_split_file_paths_follow_data_dir(tmp_path):
    config = load_config(write_config(tmp_path, {"data": {"data_dir": "/somewhere"}}))
    assert config.data.split_file("val") == "/somewhere/val.json"


def test_system_template_must_not_repeat_the_question():
    # In system mode the question already lives in the user turn.
    grounding = GroundingConfig(
        enabled=True, mode="system", system_template="{description} {question}"
    )
    with pytest.raises(ConfigError, match="must not contain"):
        grounding.validate("data.grounding")


@pytest.mark.parametrize("literal", ["2e-5", "2.0e-5", "0.00002"])
def test_scientific_notation_spellings_all_load(tmp_path, literal):
    # YAML 1.1 parses `2e-5` as a string, not a float. Both spellings are
    # natural for a learning rate, so both must work.
    path = tmp_path / "config.yaml"
    path.write_text(f"training:\n  learning_rate: {literal}\n", encoding="utf-8")
    config = load_config(str(path))
    assert config.training.learning_rate == pytest.approx(2e-5)


def test_non_numeric_string_is_still_rejected(tmp_path):
    path = write_config(tmp_path, {"training": {"learning_rate": "fast"}})
    with pytest.raises(ConfigError, match="must be a number"):
        load_config(path)
