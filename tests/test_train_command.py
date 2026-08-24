"""Translation from config to the trainer's command line."""

import os

import pytest

from fvqa.config import Config
from fvqa.train.command import (
    TRAIN_ENTRYPOINT,
    build_train_command,
    latest_checkpoint,
    resolve_model_source,
)


def value_after(command, flag):
    return command[command.index(flag) + 1]


@pytest.fixture
def command():
    return build_train_command(
        Config(),
        train_path="/data/train.json",
        val_path="/data/val.json",
        image_folder="/data/images",
        output_dir="/ckpt",
    )


class TestEntrypoint:
    def test_uses_the_sft_entrypoint(self, command):
        # The old scripts invoked `train.py` at the repo root, which does
        # not exist in Qwen-VL-Series-Finetune.
        assert TRAIN_ENTRYPOINT in command
        assert "train.py" not in command

    def test_deepspeed_launcher_when_configured(self, command):
        assert command[0] == "deepspeed"
        assert command[1] == "--num_gpus=1"

    def test_plain_python_without_deepspeed(self):
        config = Config()
        config.training.deepspeed = None
        command = build_train_command(
            config, train_path="/t.json", image_folder="/i", output_dir="/o"
        )
        assert command[0] == "python"
        assert "--deepspeed" not in command

    def test_num_gpus_is_forwarded(self):
        command = build_train_command(
            Config(), train_path="/t.json", image_folder="/i", output_dir="/o", num_gpus=4
        )
        assert command[1] == "--num_gpus=4"


class TestConfigIsHonoured:
    def test_hyperparameters_come_from_the_config(self):
        config = Config()
        config.training.num_train_epochs = 7
        config.training.learning_rate = 3e-5
        config.model.lora.rank = 64
        command = build_train_command(
            config, train_path="/t.json", image_folder="/i", output_dir="/o"
        )
        assert value_after(command, "--num_train_epochs") == "7"
        assert value_after(command, "--learning_rate") == "3e-05"
        assert value_after(command, "--lora_rank") == "64"

    def test_booleans_are_rendered_as_capitalised_strings(self, command):
        assert value_after(command, "--bf16") == "True"
        assert value_after(command, "--fp16") == "False"

    def test_flash_attention_flag_is_negated(self):
        config = Config()
        config.model.use_flash_attn = False
        command = build_train_command(
            config, train_path="/t.json", image_folder="/i", output_dir="/o"
        )
        assert value_after(command, "--disable_flash_attn2") == "True"

    def test_lora_disabled_omits_lora_hyperparameters(self):
        config = Config()
        config.model.tuning_method = "full"
        command = build_train_command(
            config, train_path="/t.json", image_folder="/i", output_dir="/o"
        )
        assert value_after(command, "--lora_enable") == "False"
        assert "--lora_rank" not in command
        assert "--bits" not in command

    def test_qlora_enables_adapters_as_well_as_four_bit_weights(self):
        # The bug this replaced: `lora.enabled=False, qlora.enabled=True`
        # produced `--lora_enable False --bits 4`, which quantizes the
        # base model and attaches nothing trainable to it. QLoRA is LoRA
        # on a quantized base, so both flags have to be set.
        config = Config()
        config.model.tuning_method = "qlora"
        command = build_train_command(
            config, train_path="/t.json", image_folder="/i", output_dir="/o"
        )
        assert value_after(command, "--lora_enable") == "True"
        assert value_after(command, "--bits") == "4"
        assert value_after(command, "--lora_rank") == str(config.model.lora.rank)

    def test_qlora_forwards_the_quantization_settings(self):
        config = Config()
        config.model.tuning_method = "qlora"
        config.model.quantization.quant_type = "fp4"
        config.model.quantization.double_quant = False
        command = build_train_command(
            config, train_path="/t.json", image_folder="/i", output_dir="/o"
        )
        # The trainer's own flag names — it builds the BitsAndBytesConfig
        # itself and takes no bnb_4bit_* arguments.
        assert value_after(command, "--quant_type") == "fp4"
        assert value_after(command, "--double_quant") == "False"

    def test_plain_lora_does_not_quantize(self):
        config = Config()
        config.model.tuning_method = "lora"
        command = build_train_command(
            config, train_path="/t.json", image_folder="/i", output_dir="/o"
        )
        assert value_after(command, "--lora_enable") == "True"
        assert "--bits" not in command
        assert "--quant_type" not in command


class TestMaxSteps:
    def test_absent_by_default(self):
        # The trainer's own default (-1) already means "use the epoch
        # schedule"; passing it explicitly would say nothing.
        command = build_train_command(
            Config(), train_path="/t.json", image_folder="/i", output_dir="/o"
        )
        assert "--max_steps" not in command

    def test_forwarded_when_set(self):
        config = Config()
        config.training.max_steps = 2
        command = build_train_command(
            config, train_path="/t.json", image_folder="/i", output_dir="/o"
        )
        assert value_after(command, "--max_steps") == "2"


class TestEvaluationWiring:
    def test_validation_path_enables_evaluation(self, command):
        assert value_after(command, "--eval_path") == "/data/val.json"
        assert value_after(command, "--eval_strategy") == "steps"

    def test_evaluation_is_off_without_a_validation_file(self):
        # Asking for eval with no val.json fails minutes into the run.
        command = build_train_command(
            Config(), train_path="/t.json", image_folder="/i", output_dir="/o"
        )
        assert value_after(command, "--eval_strategy") == "no"
        assert "--eval_path" not in command
        assert "--load_best_model_at_end" not in command


class TestLatestCheckpoint:
    def test_returns_none_for_a_missing_directory(self, tmp_path):
        assert latest_checkpoint(str(tmp_path / "absent")) is None

    def test_returns_none_when_there_are_no_checkpoints(self, tmp_path):
        assert latest_checkpoint(str(tmp_path)) is None

    def test_orders_by_step_number_not_lexicographically(self, tmp_path):
        for step in (500, 1000, 1500):
            (tmp_path / f"checkpoint-{step}").mkdir()
        # Sorting as strings would pick checkpoint-1000 and resume from the
        # wrong place.
        assert latest_checkpoint(str(tmp_path)).endswith("checkpoint-1500")

    def test_ignores_unrelated_entries(self, tmp_path):
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "runs").mkdir()
        (tmp_path / "checkpoint-abc").mkdir()
        assert latest_checkpoint(str(tmp_path)).endswith("checkpoint-100")

    def test_resume_appends_the_checkpoint_path(self, tmp_path):
        (tmp_path / "checkpoint-750").mkdir()
        command = build_train_command(
            Config(),
            train_path="/t.json",
            image_folder="/i",
            output_dir=str(tmp_path),
            resume=True,
        )
        assert value_after(command, "--resume_from_checkpoint") == os.path.join(
            str(tmp_path), "checkpoint-750"
        )

    def test_resume_is_silently_skipped_with_no_checkpoints(self, tmp_path):
        command = build_train_command(
            Config(),
            train_path="/t.json",
            image_folder="/i",
            output_dir=str(tmp_path),
            resume=True,
        )
        assert "--resume_from_checkpoint" not in command


class TestResolveModelSource:
    def test_model_path_is_passed_through_untouched(self):
        # A HuggingFace id must survive verbatim — this is what makes it
        # possible to score the un-finetuned base model.
        assert (
            resolve_model_source(model_path="Qwen/Qwen3-VL-8B-Instruct")
            == "Qwen/Qwen3-VL-8B-Instruct"
        )

    def test_model_path_wins_over_a_checkpoint(self, tmp_path):
        (tmp_path / "checkpoint-500").mkdir()
        assert (
            resolve_model_source(model_path="Qwen/Qwen3-VL-8B-Instruct", output_dir=str(tmp_path))
            == "Qwen/Qwen3-VL-8B-Instruct"
        )

    def test_named_checkpoint_is_joined_to_output_dir(self, tmp_path):
        (tmp_path / "checkpoint-500").mkdir()
        resolved = resolve_model_source(checkpoint="checkpoint-500", output_dir=str(tmp_path))
        assert resolved == os.path.join(str(tmp_path), "checkpoint-500")

    def test_missing_named_checkpoint_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            resolve_model_source(checkpoint="checkpoint-999", output_dir=str(tmp_path))

    def test_falls_back_to_the_newest_checkpoint(self, tmp_path):
        for step in (500, 1500):
            (tmp_path / f"checkpoint-{step}").mkdir()
        assert resolve_model_source(output_dir=str(tmp_path)).endswith("checkpoint-1500")

    def test_no_checkpoint_suggests_scoring_the_base_model(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Qwen/Qwen3-VL-8B-Instruct"):
            resolve_model_source(output_dir=str(tmp_path))

    def test_nothing_to_resolve_from_is_rejected(self):
        with pytest.raises(ValueError, match="pass model_path"):
            resolve_model_source()
