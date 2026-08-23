"""Translation from config to the trainer's command line."""

import os

import pytest

from vivqa.config import Config
from vivqa.train.command import TRAIN_ENTRYPOINT, build_train_command, latest_checkpoint


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
        config.model.lora.enabled = False
        command = build_train_command(
            config, train_path="/t.json", image_folder="/i", output_dir="/o"
        )
        assert value_after(command, "--lora_enable") == "False"
        assert "--lora_rank" not in command

    def test_qlora_requests_four_bit_weights(self):
        config = Config()
        config.model.lora.enabled = False
        config.model.qlora.enabled = True
        command = build_train_command(
            config, train_path="/t.json", image_folder="/i", output_dir="/o"
        )
        assert value_after(command, "--bits") == "4"


class TestEvaluationWiring:
    def test_validation_path_enables_evaluation(self, command):
        assert value_after(command, "--eval_data_path") == "/data/val.json"
        assert value_after(command, "--eval_strategy") == "steps"

    def test_evaluation_is_off_without_a_validation_file(self):
        # Asking for eval with no val.json fails minutes into the run.
        command = build_train_command(
            Config(), train_path="/t.json", image_folder="/i", output_dir="/o"
        )
        assert value_after(command, "--eval_strategy") == "no"
        assert "--eval_data_path" not in command
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
