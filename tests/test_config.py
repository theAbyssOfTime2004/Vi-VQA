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


@pytest.mark.parametrize(
    "method, uses_lora, uses_quantization",
    [("full", False, False), ("lora", True, False), ("qlora", True, True)],
)
def test_tuning_method_decides_what_is_used(method, uses_lora, uses_quantization):
    # QLoRA implies LoRA — the enum exists so that "quantized but with no
    # adapters attached" cannot be expressed at all.
    config = Config()
    config.model.tuning_method = method
    assert config.model.uses_lora is uses_lora
    assert config.model.uses_quantization is uses_quantization


def test_max_steps_defaults_to_the_full_schedule(tmp_path):
    # null, not the trainer's -1 sentinel: absent means the flag is never
    # emitted at all.
    assert Config().training.max_steps is None
    config = load_config(write_config(tmp_path, {"training": {"max_steps": None}}))
    assert config.training.max_steps is None


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
            {
                "model": {"tuning_method": "full"},
                "training": {
                    "freeze_llm": True,
                    "freeze_vision_tower": True,
                    "freeze_merger": True,
                },
            },
            "nothing would train",
        ),
        ({"model": {"tuning_method": "adapter"}}, "tuning_method"),
        ({"model": {"quantization": {"quant_type": "int8"}}}, "quant_type"),
        ({"model": {"lora": {"rank": 0}}}, "rank must be positive"),
        ({"training": {"max_steps": 0}}, "max_steps must be positive"),
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


class TestYamlBooleanAliases:
    """YAML 1.1 resolves a bare `no` to False. That is wrong here.

    HuggingFace's `eval_strategy` and `save_strategy` take the *string*
    "no", so `--set training.eval_strategy=no` used to yield False and
    die in validation with "must be a string, got False" — after the
    container had started and the model had begun loading.
    """

    @pytest.mark.parametrize("field", ["eval_strategy", "save_strategy"])
    def test_no_stays_a_string_in_an_override(self, repo_config, field):
        config = load_config(repo_config, overrides=[f"training.{field}=no"])
        assert getattr(config.training, field) == "no"

    @pytest.mark.parametrize("field", ["eval_strategy", "save_strategy"])
    def test_no_stays_a_string_in_the_config_file(self, tmp_path, field):
        path = tmp_path / "config.yaml"
        path.write_text(f"training:\n  {field}: no\n", encoding="utf-8")
        assert getattr(load_config(str(path)).training, field) == "no"

    @pytest.mark.parametrize("word", ["yes", "on", "off"])
    def test_the_other_yaml_1_1_aliases_stay_strings_too(self, tmp_path, word):
        # Same trap, same fix — checked so a future loader change cannot
        # quietly reinstate only some of them.
        path = tmp_path / "config.yaml"
        path.write_text(f"training:\n  report_to: {word}\n", encoding="utf-8")
        assert load_config(str(path)).training.report_to == word

    @pytest.mark.parametrize(
        "literal, expected",
        [("true", True), ("false", False), ("True", True), ("FALSE", False)],
    )
    def test_real_booleans_still_parse(self, repo_config, literal, expected):
        config = load_config(repo_config, overrides=[f"training.bf16={literal}"])
        assert config.training.bf16 is expected

    def test_a_boolean_field_given_no_now_fails_loudly(self, tmp_path):
        # The cost of the fix: `bf16: no` is a string now. That is a clear
        # type error rather than a silently accepted value, which is the
        # trade this loader is making.
        path = tmp_path / "config.yaml"
        path.write_text("training:\n  bf16: no\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="must be a boolean"):
            load_config(str(path))

    def test_the_smoke_overrides_load(self, repo_config):
        # The exact override list scripts/smoke_gpu.sh and the Modal
        # smoke_train function pass. This is the case that failed on a
        # real GPU run.
        config = load_config(
            repo_config,
            overrides=[
                "training.max_steps=2",
                "training.save_steps=1",
                "training.eval_strategy=no",
                "training.load_best_model_at_end=false",
                "training.logging_steps=1",
            ],
        )
        assert config.training.max_steps == 2
        assert config.training.eval_strategy == "no"
        assert config.training.load_best_model_at_end is False


class TestTunerConstraints:
    """Rules the trainer enforces, checked before a GPU is involved.

    `train_sft.py` asserts these right after parsing its arguments and
    raises minutes into a run, once deepspeed has started. Each is a real
    contradiction, so there is nothing lost by rejecting it at load time —
    and a config error at `fvqa config` costs seconds instead of an A100
    container.
    """

    def test_lora_requires_a_frozen_llm(self):
        # The exact combination that died on a real GPU run.
        config = Config()
        config.model.tuning_method = "lora"
        config.training.freeze_llm = False
        with pytest.raises(ConfigError, match="freeze_llm must be true"):
            config.validate()

    def test_qlora_requires_a_frozen_llm_too(self):
        config = Config()
        config.model.tuning_method = "qlora"
        config.training.freeze_llm = False
        with pytest.raises(ConfigError, match="freeze_llm must be true"):
            config.validate()

    def test_full_tuning_may_train_the_llm(self):
        config = Config()
        config.model.tuning_method = "full"
        config.training.freeze_llm = False
        config.validate()

    def test_vision_lora_needs_an_adapter_to_attach_to(self):
        config = Config()
        config.model.tuning_method = "full"
        config.model.lora.vision_lora = True
        with pytest.raises(ConfigError, match="vision_lora requires an adapter"):
            config.validate()

    def test_vision_lora_requires_a_frozen_vision_tower(self):
        config = Config()
        config.model.tuning_method = "lora"
        config.model.lora.vision_lora = True
        config.training.freeze_vision_tower = False
        with pytest.raises(ConfigError, match="freeze_vision_tower must be true"):
            config.validate()

    def test_everything_frozen_is_fine_under_lora(self):
        # The adapters still train, so this is not a contradiction — the
        # "nothing would train" rule applies to full fine-tuning only.
        config = Config()
        config.model.tuning_method = "lora"
        config.training.freeze_llm = True
        config.training.freeze_vision_tower = True
        config.training.freeze_merger = True
        config.validate()

    def test_the_shipped_config_satisfies_all_of_them(self, repo_config):
        # config.yaml had freeze_llm: false with tuning_method: lora, which
        # is what reached the GPU.
        load_config(repo_config).validate()
